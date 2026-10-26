"""Jina reranking stage.

Security note: this runs AFTER the authorization re-check, so only already
authorized passages are ever sent to the reranker. The reranker orders and
filters by relevance; it can never grant access to a passage, and a reranker
failure degrades relevance quality without weakening authorization.

Availability note: falling back to fusion order is a real quality loss, not a
neutral default - an unranked context is what the document-aware selection
stage is protecting against. A rate limit is transient, so it is retried with
backoff rather than treated as a permanent failure. Only a genuinely
unrecoverable response degrades to fusion order.

Phase 7 note: the reranker can additionally consume a QueryPlan to tell an
answer-bearing passage from a merely related one. That influence is bounded and
ordering-only - the relevance threshold is applied to the raw Jina score before
any plan term exists - so a plan can reorder the surviving passages but can
never change which passages survive, and a plan-aware run degrades to exactly
the pre-Phase-7 behaviour when the plan is absent or the feature is disabled.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from . import config, query_aware_rerank

logger = logging.getLogger(__name__)

# The reranker must see the whole passage it is scoring. Ingestion emits
# 800-token chunks (avg ~2.4k chars, max ~4.4k), so a 2k window hid the back
# half of roughly two thirds of the corpus: evidence past the cut could not
# influence a chunk's score, and the document holding it was dropped despite
# having been retrieved correctly. Keep this >= the largest indexed chunk.
MAX_DOC_CHARS = config.RERANK_MAX_DOC_CHARS
MAX_QUERY_CHARS = 2000

# Retried: the provider is busy or briefly unreachable, and the request is
# unchanged on a second attempt. 5xx is included because it is not a statement
# about the request itself.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class _Retryable(RuntimeError):
    """A transient rerank failure worth another attempt."""

    def __init__(self, message: str, wait: float | None = None) -> None:
        super().__init__(message)
        self.wait = wait


@dataclass
class RerankOutcome:
    items: list[tuple[object, float | None]]
    applied: bool
    error: str | None = None
    attempts: int = 0
    # Phase 7: whether the bounded QueryPlan compatibility term was added to
    # the Jina scores, and how much it moved them. Diagnostics only.
    plan_applied: bool = False
    mean_adjustment: float = 0.0
    max_adjustment: float = 0.0


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.JINA_API_KEY}",
        "Content-Type": "application/json",
    }


def _retry_after(response) -> float | None:
    """Honour an explicit Retry-After before falling back to backoff."""
    try:
        value = float((response.headers or {}).get("Retry-After", ""))
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= config.RERANK_MAX_BACKOFF_S else None


def _request(query: str, documents: list[str]) -> list[dict]:
    response = requests.post(
        config.JINA_RERANK_URL,
        headers=_headers(),
        json={
            "model": config.JINA_RERANK_MODEL,
            "query": query[:MAX_QUERY_CHARS],
            "documents": documents,
            "top_n": len(documents),
        },
        timeout=config.RERANK_TIMEOUT_S,
    )
    if response.status_code in RETRYABLE_STATUS:
        raise _Retryable(
            f"Jina rerank error {response.status_code}: {response.text[:200]}",
            _retry_after(response))
    if response.status_code != 200:
        raise RuntimeError(
            f"Jina rerank error {response.status_code}: {response.text[:200]}")
    results = response.json().get("results")
    if not results:
        raise RuntimeError("Jina rerank returned no results")
    return results


def rerank(query: str, chunks: list, threshold: float | None = None,
           plan=None, weights=None) -> RerankOutcome:
    """Order authorized chunks by relevance and drop those below threshold.

    Returns the original order with no scores if reranking is unavailable.

    Phase 7: when `plan` is supplied and plan-aware reranking is enabled, the
    Jina query gains a compact statement of the information need and each score
    gains a bounded QueryPlan-compatibility term. The relevance THRESHOLD is
    deliberately applied to the raw Jina score, so the set of passages that
    survive is identical to the pre-Phase-7 pipeline and the plan can only ever
    change their order. Phase 7 reorders; it never filters.
    """
    if not chunks:
        return RerankOutcome(items=[], applied=False)
    if not config.RERANK_ENABLED or not config.JINA_API_KEY:
        return RerankOutcome([(c, None) for c in chunks], False, "disabled")

    plan_aware = plan is not None and config.RERANK_PLAN_AWARE
    effective_query = (query_aware_rerank.plan_aware_query(query, plan)
                       if plan_aware else query)

    documents = [(c.text or "")[:MAX_DOC_CHARS] for c in chunks]
    attempts = 0
    results = None
    last_error: Exception | None = None
    for attempt in range(config.RERANK_MAX_RETRIES + 1):
        attempts = attempt + 1
        try:
            results = _request(effective_query, documents)
            last_error = None
            break
        except _Retryable as exc:
            last_error = exc
            if attempt >= config.RERANK_MAX_RETRIES:
                break
            wait = getattr(exc, "wait", None)
            if wait is None:
                wait = min(config.RERANK_BACKOFF_S * (2 ** attempt),
                           config.RERANK_MAX_BACKOFF_S)
            logger.warning("Rerank rate limited, retrying in %.1fs (%d/%d): %s",
                           wait, attempt + 1, config.RERANK_MAX_RETRIES, exc)
            time.sleep(wait)
        except (requests.RequestException, RuntimeError, ValueError) as exc:
            last_error = exc
            break

    if results is None:
        logger.warning("Reranking unavailable after %d attempt(s), falling back "
                       "to fusion order: %s", attempts, last_error)
        return RerankOutcome([(c, None) for c in chunks], False,
                             str(last_error)[:200], attempts)

    scored: list[tuple[object, float]] = []
    for item in results:
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < len(chunks):
            continue
        scored.append((chunks[index], float(item.get("relevance_score", 0.0))))
    if not scored:
        return RerankOutcome([(c, None) for c in chunks], False,
                             "no usable scores", attempts)

    # Relevance gating stays a pure Jina decision, on the raw score, BEFORE any
    # plan term is added. This is what makes Phase 7 provably recall-neutral at
    # this stage: the surviving set cannot differ from baseline.
    if threshold is not None:
        scored = [pair for pair in scored if pair[1] >= threshold]

    if not plan_aware:
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return RerankOutcome(items=scored, applied=True, attempts=attempts)

    adjusted: list[tuple[object, float]] = []
    deltas: list[float] = []
    for chunk, raw in scored:
        value = query_aware_rerank.adjust(raw, chunk, plan, weights)
        deltas.append(value - raw)
        adjusted.append((chunk, value))
    # Stable sort: an equal adjusted score keeps Jina's own ordering.
    adjusted.sort(key=lambda pair: pair[1], reverse=True)
    return RerankOutcome(
        items=adjusted, applied=True, attempts=attempts, plan_applied=True,
        mean_adjustment=(sum(deltas) / len(deltas)) if deltas else 0.0,
        max_adjustment=max(deltas) if deltas else 0.0)
