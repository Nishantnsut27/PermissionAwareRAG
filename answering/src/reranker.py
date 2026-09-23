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
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import requests

from . import config

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


def rerank(query: str, chunks: list, threshold: float | None = None) -> RerankOutcome:
    """Order authorized chunks by relevance and drop those below threshold.

    Returns the original order with no scores if reranking is unavailable.
    """
    if not chunks:
        return RerankOutcome(items=[], applied=False)
    if not config.RERANK_ENABLED or not config.JINA_API_KEY:
        return RerankOutcome([(c, None) for c in chunks], False, "disabled")

    documents = [(c.text or "")[:MAX_DOC_CHARS] for c in chunks]
    attempts = 0
    results = None
    last_error: Exception | None = None
    for attempt in range(config.RERANK_MAX_RETRIES + 1):
        attempts = attempt + 1
        try:
            results = _request(query, documents)
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

    scored: list[tuple[object, float | None]] = []
    for item in results:
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < len(chunks):
            continue
        scored.append((chunks[index], float(item.get("relevance_score", 0.0))))
    if not scored:
        return RerankOutcome([(c, None) for c in chunks], False,
                             "no usable scores", attempts)

    scored.sort(key=lambda pair: pair[1], reverse=True)
    if threshold is not None:
        scored = [pair for pair in scored if pair[1] >= threshold]
    return RerankOutcome(items=scored, applied=True, attempts=attempts)
