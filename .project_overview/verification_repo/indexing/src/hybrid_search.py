"""Reciprocal Rank Fusion over dense and sparse rankings."""
from __future__ import annotations

from .models import RetrievalResult


def reciprocal_rank_fusion(
    dense: list[RetrievalResult],
    sparse: list[RetrievalResult],
    rrf_k: int = 60,
) -> list[RetrievalResult]:
    fused: dict[str, RetrievalResult] = {}
    rrf_scores: dict[str, float] = {}
    for rank, item in enumerate(dense, start=1):
        fused.setdefault(item.chunk_id, item)
        rrf_scores[item.chunk_id] = (
            rrf_scores.get(item.chunk_id, 0.0) + 1.0 / (rrf_k + rank))
    for rank, item in enumerate(sparse, start=1):
        if item.chunk_id in fused:
            existing = fused[item.chunk_id]
            existing.retrieval_method = "hybrid"
        else:
            item.retrieval_method = "hybrid"
            fused[item.chunk_id] = item
        rrf_scores[item.chunk_id] = (
            rrf_scores.get(item.chunk_id, 0.0) + 1.0 / (rrf_k + rank))
    ranked = sorted(fused.values(),
                    key=lambda r: rrf_scores[r.chunk_id], reverse=True)
    for item in ranked:
        item.score = rrf_scores[item.chunk_id]
    return ranked
