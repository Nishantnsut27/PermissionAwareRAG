"""Retrieval interface: dense, sparse and hybrid search over Qdrant."""
from __future__ import annotations

from qdrant_client.models import SparseVector

from . import config
from .embeddings import embed_query
from .hybrid_search import reciprocal_rank_fusion
from .models import RetrievalResult
from .qdrant_store import build_filter, connect
from .sparse_search import SparseEncoder

METADATA_KEYS = (
    "seller_id", "department", "classification", "document_type",
    "scenario_id", "source_file", "page_numbers",
)

_encoder: SparseEncoder | None = None


def _sparse_encoder() -> SparseEncoder:
    global _encoder
    if _encoder is None:
        _encoder = SparseEncoder.load()
    return _encoder


def _to_result(point, method: str) -> RetrievalResult:
    payload = point.payload or {}
    metadata = {k: payload.get(k) for k in METADATA_KEYS}
    metadata.update({k: v for k, v in payload.items()
                     if k not in ("text",) and k not in metadata})
    return RetrievalResult(
        chunk_id=payload.get("chunk_id", str(point.id)),
        document_id=payload.get("document_id"),
        text=payload.get("text", ""),
        score=float(point.score or 0.0),
        retrieval_method=method,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def dense_search(query: str, top_k: int,
                 filters: dict | None = None) -> list[RetrievalResult]:
    client = connect()
    vector = embed_query(query)
    points = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=vector,
        using=config.DEFAULT_RETRIEVAL.dense_vector_name,
        query_filter=build_filter(filters),
        limit=top_k,
        with_payload=True,
    ).points
    return [_to_result(p, "dense") for p in points]


def sparse_search(query: str, top_k: int,
                  filters: dict | None = None) -> list[RetrievalResult]:
    client = connect()
    sparse = _sparse_encoder().encode(query)
    points = client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=SparseVector(indices=sparse.indices, values=sparse.values),
        using=config.DEFAULT_RETRIEVAL.sparse_vector_name,
        query_filter=build_filter(filters),
        limit=top_k,
        with_payload=True,
    ).points
    return [_to_result(p, "sparse") for p in points]


def retrieve(query: str, top_k: int = 10,
             filters: dict | None = None,
             method: str = "hybrid") -> list[RetrievalResult]:
    cfg = config.DEFAULT_RETRIEVAL
    if method == "dense":
        return dense_search(query, top_k, filters)[:top_k]
    if method == "sparse":
        return sparse_search(query, top_k, filters)[:top_k]
    dense = dense_search(query, cfg.dense_top_k, filters)
    sparse = sparse_search(query, cfg.sparse_top_k, filters)
    return reciprocal_rank_fusion(dense, sparse, cfg.rrf_k)[:top_k]
