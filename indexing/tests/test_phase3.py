"""Offline unit checks: fusion, sparse encoding, filter safety, failure handling."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import embeddings
from src.embeddings import EmbeddingError, embed_texts, validate_embeddings
from src.hybrid_search import reciprocal_rank_fusion
from src.models import Chunk, RetrievalResult
from src.qdrant_store import (
    FilterError,
    build_filter,
    point_id_for,
    upsert_points,
)
from src.sparse_search import SparseEncoder, tokenize


def make_result(chunk_id: str, score: float, method: str) -> RetrievalResult:
    return RetrievalResult(chunk_id=chunk_id, document_id="D1", text="t",
                           score=score, retrieval_method=method, metadata={})


def test_tokenizer_keeps_identifiers() -> None:
    tokens = tokenize("Incident INC-2026-0618 for seller S001 ticket "
                      "TKT-2026-0805-301 runbook OPS-RB-004")
    for identifier in ("inc-2026-0618", "s001", "tkt-2026-0805-301",
                       "ops-rb-004"):
        assert identifier in tokens, (identifier, tokens)


def test_sparse_encoder_exact_match() -> None:
    enc = SparseEncoder().fit(["incident INC-2026-0618 shipping label failover",
                               "unrelated policy text here"])
    vec = enc.encode("INC-2026-0618")
    assert vec.indices, "query with a known identifier must encode non-empty"
    assert enc.encode("zzz-unknown-token").indices == []


def test_rrf_merges_duplicates() -> None:
    dense = [make_result("A", 0.9, "dense"), make_result("B", 0.8, "dense")]
    sparse = [make_result("B", 2.0, "sparse"), make_result("C", 1.0, "sparse")]
    fused = reciprocal_rank_fusion(dense, sparse)
    ids = [r.chunk_id for r in fused]
    assert sorted(ids) == ["A", "B", "C"], ids
    assert next(r for r in fused if r.chunk_id == "B").retrieval_method == "hybrid"
    assert fused[0].chunk_id == "B"


def test_build_filter_shapes() -> None:
    assert build_filter(None) is None
    assert build_filter({}) is None
    filt = build_filter({"seller_id": "S001",
                         "classification": ["CONFIDENTIAL", "INTERNAL"],
                         "document_type": {"any": ["Policy"]}})
    assert filt is not None and len(filt.must) == 3
    assert {c.key for c in filt.must} == {"seller_id", "classification",
                                          "document_type"}


def test_build_filter_never_widens() -> None:
    """A condition that cannot be built must raise, not silently disappear:
    Phase 4 derives its authorization pre-filter from this function."""
    for spec in ({"seller_id": {"any": []}},
                 {"seller_id": {"any": [None]}},
                 {"classification": []},
                 {"seller_id": None},
                 {"seller_id": {"all": ["S001"]}},
                 "not-a-dict"):
        try:
            build_filter(spec)
        except FilterError:
            continue
        raise AssertionError(f"build_filter accepted {spec!r}")


def test_point_id_is_deterministic() -> None:
    assert point_id_for("abc_chunk_001") == point_id_for("abc_chunk_001")
    assert point_id_for("abc_chunk_001") != point_id_for("abc_chunk_002")


def test_chunk_payload_preserves_phase2_metadata() -> None:
    raw = {"chunk_id": "D_chunk_001", "text": "body", "document_id": "D",
           "document_type": "Policy", "seller_id": "Organization-wide",
           "department": "IT", "classification": "RESTRICTED",
           "scenario_id": "SC-004", "source_file": "a/b.pdf",
           "page_numbers": [1, 2], "token_count": 733}
    payload = Chunk.from_dict(raw).payload()
    for key, value in raw.items():
        assert payload[key] == value, key


def test_validate_embeddings_rejects_bad_vectors() -> None:
    for bad in ([], [[]], [[1.0, 2.0], [1.0]], [[1.0, "x"]], [None]):
        try:
            validate_embeddings(bad)
        except EmbeddingError:
            continue
        raise AssertionError(f"validate_embeddings accepted {bad!r}")
    assert validate_embeddings([[1.0, 2.0], [3.0, 4.0]]) == 2


def test_embed_texts_keeps_alignment_when_a_batch_fails() -> None:
    original = embeddings._post_batch

    def flaky(texts, task, timeout_s):
        if "boom" in texts[0]:
            raise EmbeddingError("simulated API failure")
        return [[0.1, 0.2] for _ in texts]

    embeddings._post_batch = flaky
    try:
        vectors, failed = embed_texts(["ok-a", "boom-b", "ok-c"],
                                      batch_size=1, max_retries=0)
    finally:
        embeddings._post_batch = original
    assert failed == [1], failed
    assert vectors[0] is not None and vectors[2] is not None
    assert vectors[1] is None, "failed chunk must stay None, not shift alignment"


def test_embed_texts_rejects_mixed_dimensions_across_batches() -> None:
    original = embeddings._post_batch
    sizes = iter([2, 3])

    def drifting(texts, task, timeout_s):
        width = next(sizes)
        return [[0.1] * width for _ in texts]

    embeddings._post_batch = drifting
    try:
        embed_texts(["a", "b"], batch_size=1, max_retries=0)
    except EmbeddingError:
        return
    finally:
        embeddings._post_batch = original
    raise AssertionError("a mid-run dimension change was accepted")


def test_upsert_points_reports_errors_instead_of_hiding_them() -> None:
    class BrokenClient:
        def upsert(self, **kwargs):
            raise RuntimeError("qdrant unavailable")

    chunk = Chunk.from_dict({"chunk_id": "c1", "text": "t", "document_id": "D"})
    ok, failed, errors = upsert_points(
        BrokenClient(), [(chunk, [0.1, 0.2], {"indices": [], "values": []})])
    assert ok == 0 and failed == 1, (ok, failed)
    assert errors and "qdrant unavailable" in errors[0], errors


def main() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except Exception as exc:
            failed += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{passed} passed, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
