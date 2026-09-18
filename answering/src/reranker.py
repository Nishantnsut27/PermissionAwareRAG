"""Jina reranking stage.

Security note: this runs AFTER the authorization re-check, so only already
authorized passages are ever sent to the reranker. The reranker orders and
filters by relevance; it can never grant access to a passage, and a reranker
failure degrades relevance quality without weakening authorization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

from . import config

logger = logging.getLogger(__name__)

MAX_DOC_CHARS = 2000


@dataclass
class RerankOutcome:
    items: list[tuple[object, float | None]]
    applied: bool
    error: str | None = None


def _headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {config.JINA_API_KEY}",
        "Content-Type": "application/json",
    }


def rerank(query: str, chunks: list, threshold: float | None = None) -> RerankOutcome:
    """Order authorized chunks by relevance and drop those below threshold.

    Returns the original order with no scores if reranking is unavailable.
    """
    if not chunks:
        return RerankOutcome(items=[], applied=False)
    if not config.RERANK_ENABLED or not config.JINA_API_KEY:
        return RerankOutcome([(c, None) for c in chunks], False, "disabled")

    documents = [(c.text or "")[:MAX_DOC_CHARS] for c in chunks]
    try:
        response = requests.post(
            config.JINA_RERANK_URL,
            headers=_headers(),
            json={
                "model": config.JINA_RERANK_MODEL,
                "query": query[:MAX_DOC_CHARS],
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=config.RERANK_TIMEOUT_S,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Jina rerank error {response.status_code}: "
                f"{response.text[:200]}")
        results = response.json().get("results")
        if not results:
            raise RuntimeError("Jina rerank returned no results")
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        logger.warning("Reranking unavailable, falling back to fusion order: %s",
                       exc)
        return RerankOutcome([(c, None) for c in chunks], False, str(exc)[:200])

    scored: list[tuple[object, float | None]] = []
    for item in results:
        index = item.get("index")
        if not isinstance(index, int) or not 0 <= index < len(chunks):
            continue
        scored.append((chunks[index], float(item.get("relevance_score", 0.0))))
    if not scored:
        return RerankOutcome([(c, None) for c in chunks], False, "no usable scores")

    scored.sort(key=lambda pair: pair[1], reverse=True)
    if threshold is not None:
        scored = [pair for pair in scored if pair[1] >= threshold]
    return RerankOutcome(items=scored, applied=True)
