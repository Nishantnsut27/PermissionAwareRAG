"""Jina Embeddings API client with batching, validation and retries."""
from __future__ import annotations

import numbers
import time

import requests

from . import config
from answering.src.deadline import record_retry, remaining_timeout
from answering.src.http_deadline import provider_deadline, session

HTTP = session()


class EmbeddingError(RuntimeError):
    pass


class EmbeddingHTTPError(EmbeddingError):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.retryable = status_code == 429 or status_code == 408 or status_code >= 500


def _headers() -> dict[str, str]:
    if not config.JINA_API_KEY:
        raise EmbeddingError(
            "JINA_API_KEY is not set; refusing to call the embedding API.")
    return {
        "Authorization": f"Bearer {config.JINA_API_KEY}",
        "Content-Type": "application/json",
    }


def _post_batch(texts: list[str], task: str, timeout_s: int) -> list[list[float]]:
    with provider_deadline(timeout_s):
        return _post_batch_body(texts, task, timeout_s)


def _post_batch_body(texts: list[str], task: str, timeout_s: int) -> list[list[float]]:
    response = HTTP.post(
        config.JINA_API_URL,
        headers=_headers(),
        json={
            "model": config.JINA_EMBEDDING_MODEL,
            "task": task,
            "input": texts,
        },
        timeout=remaining_timeout(timeout_s),
    )
    try:
        if response.status_code != 200:
            raise EmbeddingHTTPError(response.status_code,
                f"Jina API error {response.status_code}: {response.text[:300]}")
        body = response.json()
    finally:
        close = getattr(response, "close", None)
        if close:
            close()
    items = body.get("data") if isinstance(body, dict) else None
    if not items:
        raise EmbeddingError(f"Jina response missing 'data': {str(body)[:300]}")
    ordered = sorted(items, key=lambda d: d.get("index", 0))
    vectors = [d.get("embedding") for d in ordered]
    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"Jina returned {len(vectors)} embeddings for {len(texts)} inputs")
    return vectors


def validate_embeddings(vectors: list[list[float]]) -> int:
    if not vectors:
        raise EmbeddingError("No embeddings returned")
    dim = None
    for i, vec in enumerate(vectors):
        if not isinstance(vec, list) or not vec:
            raise EmbeddingError(f"Embedding {i} is missing or empty")
        if not all(isinstance(v, numbers.Real) for v in vec):
            raise EmbeddingError(f"Embedding {i} contains non-numeric values")
        if dim is None:
            dim = len(vec)
        elif len(vec) != dim:
            raise EmbeddingError(
                f"Inconsistent embedding dimension: {len(vec)} != {dim}")
    assert dim is not None
    return dim


def embed_texts(
    texts: list[str],
    task: str = "retrieval.passage",
    batch_size: int | None = None,
    timeout_s: int | None = None,
    max_retries: int | None = None,
) -> tuple[list[list[float] | None], list[int]]:
    """Embed texts in batches. Returns (vectors aligned with input, failed_indexes).

    A batch that never succeeds leaves `None` in place rather than shifting the
    alignment, so the caller can map failures back to specific chunks. The
    dimension is validated across every batch, not only within one, so a model
    change mid-run cannot produce a mixed-width index.
    """
    if not texts:
        raise EmbeddingError("embed_texts called with no input")
    cfg = config.DEFAULT_RETRIEVAL
    batch_size = batch_size or cfg.embed_batch_size
    timeout_s = timeout_s or cfg.embed_timeout_s
    max_retries = cfg.embed_max_retries if max_retries is None else max_retries

    vectors: list[list[float] | None] = [None] * len(texts)
    failed: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                result = _post_batch(batch, task, timeout_s)
                validate_embeddings(result)
                for offset, vec in enumerate(result):
                    vectors[start + offset] = vec
                last_error = None
                break
            except (EmbeddingError, requests.RequestException) as exc:
                last_error = exc
                if attempt < max_retries and (
                        isinstance(exc, requests.RequestException)
                        or getattr(exc, "retryable", False)):
                    record_retry("embedding")
                    time.sleep(remaining_timeout(2 ** attempt))
                else:
                    break
        if last_error is not None:
            failed.extend(range(start, start + len(batch)))
    ok = [v for v in vectors if v is not None]
    if ok:
        validate_embeddings(ok)
    assert len(ok) + len(failed) == len(texts)
    return vectors, failed


def embed_query(text: str) -> list[float]:
    vectors, failed = embed_texts([text], task="retrieval.query")
    if failed or not vectors or vectors[0] is None:
        raise EmbeddingError("Failed to embed query")
    return vectors[0]
