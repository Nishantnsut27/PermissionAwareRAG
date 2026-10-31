"""Retrieval interface: dense, sparse and hybrid search over Qdrant."""
from __future__ import annotations

import json
import contextvars
import hashlib
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from qdrant_client.models import SparseVector

from . import config
from answering.src.deadline import (DeadlineExceeded, check_deadline,
                                    remaining_timeout, request_deadline)
from .embeddings import embed_query
from .hybrid_search import reciprocal_rank_fusion
from .models import RetrievalResult
from .qdrant_store import build_filter, connect
from .sparse_search import SparseEncoder

METADATA_KEYS = (
    "seller_id", "department", "classification", "document_type",
    "scenario_id", "source_file", "page_numbers",
)

_encoders: dict[str, SparseEncoder] = {}
_pinned_revision = contextvars.ContextVar("pinned_index_revision", default=None)


def _file_hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "<missing>"


def revision_snapshot(client=None) -> dict:
    collection, manifest = active_revision(client)
    folder = config.REVISIONS_DIR / collection
    snapshot = {"collection_name": collection,
                "manifest_sha256": _file_hash(folder / "manifest.json"),
                "vocabulary_sha256": _file_hash(folder / "sparse_vocab.json"),
                "revocations_sha256": _file_hash(config.REVOCATIONS_FILE)}
    if "<missing>" in (snapshot["manifest_sha256"], snapshot["vocabulary_sha256"]):
        raise RuntimeError("Active revision artifacts are missing")
    return snapshot


@contextmanager
def pin_revision(snapshot: dict):
    token = _pinned_revision.set(dict(snapshot))
    try:
        yield
    finally:
        _pinned_revision.reset(token)


def _resolve_for_retrieval(client):
    snapshot = _pinned_revision.get()
    if snapshot is None:
        return active_revision(client)
    collection = snapshot["collection_name"]
    folder = config.REVISIONS_DIR / collection
    current = {"collection_name": collection,
               "manifest_sha256": _file_hash(folder / "manifest.json"),
               "vocabulary_sha256": _file_hash(folder / "sparse_vocab.json"),
               "revocations_sha256": _file_hash(config.REVOCATIONS_FILE)}
    if current != snapshot:
        raise RuntimeError("Pinned index artifacts or emergency revocations changed")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (not manifest.get("complete") or manifest.get("collection_name") != collection
            or manifest.get("manifest_schema") != config.INDEX_MANIFEST_SCHEMA):
        raise RuntimeError("Pinned index manifest is invalid")
    return collection, manifest


def _qdrant_timeout(seconds: int) -> int:
    remaining = remaining_timeout(float(seconds))
    if remaining < 1:
        raise DeadlineExceeded("request deadline is too close for Qdrant")
    return min(seconds, int(remaining))


def _effective_filter(filters: dict | None) -> dict | None:
    """Add trusted local emergency revocations without trusting caller filters."""
    if not config.REVOCATIONS_FILE.exists():
        return filters
    try:
        raw = json.loads(config.REVOCATIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError("Emergency revocation file is malformed; retrieval denied") from exc
    revoked = raw.get("document_ids") if isinstance(raw, dict) else None
    if (not isinstance(revoked, list) or any(not isinstance(x, str) or not x
                                             for x in revoked)):
        raise RuntimeError("Emergency revocation file is invalid; retrieval denied")
    if not revoked:
        return filters
    secured = {"must_not": [{"document_id": {"any": sorted(set(revoked))}}]}
    if filters:
        secured["must"] = [filters]
    return secured


def active_revision(client=None) -> tuple[str, dict]:
    """Resolve the Qdrant active alias and its immutable local manifest."""
    client = client or connect()
    alias = f"{config.QDRANT_COLLECTION_NAME}_active"
    aliases = client.get_aliases().aliases
    collection = next((row.collection_name for row in aliases
                       if row.alias_name == alias), None)
    if not collection:
        from .revisions import RevisionError
        raise RevisionError(
            f"No published index revision is attached to alias '{alias}'")
    manifest_path = config.REVISIONS_DIR / collection / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        from .revisions import RevisionError
        raise RevisionError("Active Qdrant alias has no valid local manifest") from exc
    if (not manifest.get("complete")
            or manifest.get("collection_name") != collection
            or manifest.get("manifest_schema") != config.INDEX_MANIFEST_SCHEMA):
        from .revisions import RevisionError
        raise RevisionError("Active index manifest is incomplete or incompatible")
    return collection, manifest


def _sparse_encoder(collection: str | None = None,
                    manifest: dict | None = None) -> SparseEncoder:
    if not collection or manifest is None:
        from .sparse_search import SparseCompatibilityError
        raise SparseCompatibilityError(
            "Sparse retrieval requires a pinned active revision manifest")
    if collection not in _encoders:
        vocab_path = config.REVISIONS_DIR / collection / "sparse_vocab.json"
        encoder = SparseEncoder.load(vocab_path)
        report = manifest
        if not isinstance(report, dict):
            from .sparse_search import SparseCompatibilityError
            raise SparseCompatibilityError("Malformed sparse index report")
        sparse_report = report.get("sparse_config", {})
        if (not report.get("complete")
                or report.get("collection_name") != collection
                or sparse_report.get("vocabulary_fingerprint") != encoder.fingerprint
                or sparse_report.get("variant") != "bm25-rsj-positive-idf-linear-query-tf-v2"):
            from .sparse_search import SparseCompatibilityError
            raise SparseCompatibilityError(
                "Sparse vocabulary and active index report do not match; publish a "
                "matching index revision before sparse retrieval")
        _encoders[collection] = encoder
    return _encoders[collection]


def _to_result(point, method: str, index_revision: str | None = None) -> RetrievalResult:
    payload = point.payload or {}
    metadata = {k: payload.get(k) for k in METADATA_KEYS}
    metadata.update({k: v for k, v in payload.items()
                     if k not in ("text",) and k not in metadata})
    if index_revision:
        metadata["_index_revision"] = index_revision
    return RetrievalResult(
        chunk_id=payload.get("chunk_id", str(point.id)),
        document_id=payload.get("document_id"),
        text=payload.get("text", ""),
        score=float(point.score or 0.0),
        retrieval_method=method,
        metadata={k: v for k, v in metadata.items() if v is not None},
    )


def dense_search(query: str, top_k: int,
                 filters: dict | None = None,
                 collection_name: str | None = None,
                 _presecured: bool = False,
                 _query_vector=None) -> list[RetrievalResult]:
    client = connect()
    vector = embed_query(query) if _query_vector is None else _query_vector
    points = client.query_points(
        collection_name=collection_name or config.QDRANT_COLLECTION_NAME,
        query=vector,
        using=config.DEFAULT_RETRIEVAL.dense_vector_name,
        query_filter=build_filter(filters if _presecured else _effective_filter(filters)),
        limit=top_k,
        with_payload=True,
        timeout=_qdrant_timeout(30),
    ).points
    return [_to_result(p, "dense", collection_name) for p in points]


def _sparse_revision_filter(filters: dict | None, fingerprint: str) -> dict:
    revision = {"_sparse_revision": {"value": fingerprint}}
    if not filters:
        return revision
    if set(filters) & {"must", "should", "must_not"}:
        return {"must": [filters, revision]}
    return {"must": [{key: value} for key, value in filters.items()] + [revision]}


def sparse_search(query: str, top_k: int,
                  filters: dict | None = None,
                  collection_name: str | None = None,
                  manifest: dict | None = None,
                  _presecured: bool = False) -> list[RetrievalResult]:
    client = connect()
    collection = collection_name or config.QDRANT_COLLECTION_NAME
    if manifest is None or collection_name is None:
        from .sparse_search import SparseCompatibilityError
        raise SparseCompatibilityError(
            "Sparse retrieval requires a pinned active revision manifest")
    encoder = _sparse_encoder(collection, manifest)
    sparse = encoder.encode_query(query)
    secured_filters = filters if _presecured else _effective_filter(filters)
    points = client.query_points(
        collection_name=collection,
        query=SparseVector(indices=sparse.indices, values=sparse.values),
        using=config.DEFAULT_RETRIEVAL.sparse_vector_name,
        query_filter=build_filter(_sparse_revision_filter(secured_filters, encoder.fingerprint)),
        limit=top_k,
        with_payload=True,
        timeout=_qdrant_timeout(30),
    ).points
    return [_to_result(p, "sparse", collection) for p in points]


def retrieve(query: str, top_k: int = 10,
             filters: dict | None = None,
             method: str = "hybrid") -> list[RetrievalResult]:
    cfg = config.DEFAULT_RETRIEVAL
    client = connect()
    collection, manifest = _resolve_for_retrieval(client)
    secured_filters = _effective_filter(filters)
    if method == "dense":
        return dense_search(query, top_k, secured_filters, collection,
                            _presecured=True)[:top_k]
    if method == "sparse":
        return sparse_search(query, top_k, secured_filters, collection, manifest,
                             _presecured=True)[:top_k]
    # Both calls receive the exact same server-compiled authorization filter
    # and pinned collection revision. Fusion still runs in deterministic order.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="retrieval") as pool:
        dense_context, sparse_context = contextvars.copy_context(), contextvars.copy_context()
        dense_future = pool.submit(dense_context.run, dense_search, query,
                                   cfg.dense_top_k, secured_filters, collection, True)
        sparse_future = pool.submit(sparse_context.run, sparse_search, query,
                                    cfg.sparse_top_k, secured_filters, collection, manifest, True)
        dense = dense_future.result()
        sparse = sparse_future.result()
    return reciprocal_rank_fusion(dense, sparse, cfg.rrf_k)[:top_k]


def retrieve_partitioned(query: str, partitions: dict[str, tuple[dict, int]]
                         ) -> tuple[dict[str, list[RetrievalResult]], str]:
    """Bounded hybrid partitions sharing one embedding and resolved revision.

    The caller compiles authorization filters. Dense and sparse allocations
    across ALL partitions stay within their existing global branch limits.
    An evaluation's pinned revision is honored by the normal resolver.
    """
    # A single embedding timeout plus the existing 30-second search budget,
    # shared across every partition even for trusted CLI calls without an API
    # deadline. An enclosing request deadline always takes precedence.
    with request_deadline(config.DEFAULT_RETRIEVAL.embed_timeout_s + 30):
        result = _retrieve_partitioned(query, partitions)
        check_deadline()
        return result


def _retrieve_partitioned(query: str, partitions: dict[str, tuple[dict, int]]
                          ) -> tuple[dict[str, list[RetrievalResult]], str]:
    cfg = config.DEFAULT_RETRIEVAL
    collection, manifest = _resolve_for_retrieval(connect())
    revocations = _effective_filter(None)
    secured = {key: ({"must": [scope, revocations]} if revocations else scope)
               for key, (scope, _) in partitions.items()}

    def allocations(branch_limit):
        quotas = {key: 0 for key in partitions}
        remaining = min(branch_limit, sum(limit for _, limit in partitions.values()))
        while remaining:
            for key, (_, limit) in partitions.items():
                if remaining and quotas[key] < limit:
                    quotas[key] += 1
                    remaining -= 1
        return quotas

    dense_quotas = allocations(cfg.dense_top_k)
    sparse_quotas = allocations(cfg.sparse_top_k)
    vector = embed_query(query)

    def dense_partitions():
        return {key: dense_search(query, limit, secured[key], collection, True,
                                  _query_vector=vector) if limit else []
                for key, limit in dense_quotas.items()}

    def sparse_partitions():
        return {key: sparse_search(query, limit, secured[key], collection,
                                   manifest, True) if limit else []
                for key, limit in sparse_quotas.items()}

    # Same two-worker concurrency ceiling as global hybrid retrieval. Request
    # deadlines propagate into each provider call, including across partitions.
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="seller-retrieval") as pool:
        dense_future = pool.submit(contextvars.copy_context().run, dense_partitions)
        sparse_future = pool.submit(contextvars.copy_context().run, sparse_partitions)
        dense, sparse = dense_future.result(), sparse_future.result()
    return ({key: reciprocal_rank_fusion(dense[key], sparse[key], cfg.rrf_k)[:limit]
             for key, (_, limit) in partitions.items()}, collection)
