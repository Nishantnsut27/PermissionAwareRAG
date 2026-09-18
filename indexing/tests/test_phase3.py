"""Offline unit checks: RRF fusion, sparse encoder, filter builder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.hybrid_search import reciprocal_rank_fusion  # noqa: E402
from src.models import RetrievalResult  # noqa: E402
from src.qdrant_store import build_filter  # noqa: E402
from src.sparse_search import SparseEncoder, tokenize  # noqa: E402


def make_result(chunk_id: str, score: float, method: str) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, document_id="D1", text="t",
                           score=score, retrieval_method=method, metadata={})


def test_tokenizer_keeps_identifiers() -> None:
    tokens = tokenize("Incident INC-008 for seller S001 ticket ST-1042")
    assert "inc-008" in tokens, tokens
    assert "st-1042" in tokens, tokens
    assert "s001" in tokens, tokens


def test_sparse_encoder_exact_match() -> None:
    enc = SparseEncoder().fit(["incident INC-008 database failover",
                               "unrelated policy text here"])
    vec = enc.encode("INC-008")
    assert vec.indices, "query with a known identifier must encode non-empty"


def test_rrf_merges_duplicates() -> None:
    dense = [make_result("A", 0.9, "dense"), make_result("B", 0.8, "dense")]
    sparse = [make_result("B", 2.0, "sparse"), make_result("C", 1.0, "sparse")]
    fused = reciprocal_rank_fusion(dense, sparse)
    ids = [r.chunk_id for r in fused]
    assert sorted(ids) == ["A", "B", "C"], ids
    assert next(r for r in fused if r.chunk_id == "B").retrieval_method == "hybrid"
    assert fused[0].chunk_id == "B"


def test_build_filter_none() -> None:
    assert build_filter(None) is None
    assert build_filter({}) is None
    filt = build_filter({"seller_id": "S001",
                         "classification": ["CONFIDENTIAL", "INTERNAL"]})
    assert filt is not None and len(filt.must) == 2


if __name__ == "__main__":
    test_tokenizer_keeps_identifiers()
    test_sparse_encoder_exact_match()
    test_rrf_merges_duplicates()
    test_build_filter_none()
    print("All Phase 3 unit checks passed.")
