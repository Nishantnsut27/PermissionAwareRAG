"""Indexing orchestration: Phase 2 chunks -> Jina -> Qdrant."""
from __future__ import annotations

import json
import time

from . import config
from .embeddings import embed_texts
from .models import Chunk
from .qdrant_store import (
    connect,
    ensure_collection,
    ensure_payload_indexes,
    upsert_points,
)
from .revisions import (expected_payload_hashes, expected_point_ids,
                        publish_alias, revision_id)
from .sparse_search import SparseEncoder


class IndexingError(RuntimeError):
    pass


def finalize_revision(client, collection: str, stats: dict,
                      expected_ids: set[str],
                      payload_hashes: dict[str, str] | None = None) -> None:
    """Publish only a complete uploaded set; failed attempts remain inactive."""
    if not stats.get("complete"):
        raise IndexingError("refusing to activate an incomplete index revision")
    publish_alias(client, collection, stats, expected_ids, payload_hashes)


def _corpus_snapshot():
    from ingestion.src import config as ingestion_config
    active_path = config.PROJECT_ROOT / "ingestion" / "processed" / "active.json"
    pointer = (json.loads(active_path.read_text(encoding="utf-8"))
               if active_path.is_file() else None)
    revision = pointer.get("revision") if pointer else None
    revision_dir = ingestion_config.REVISIONS_DIR / revision if revision else None
    return active_path, pointer, (
        revision_dir / "chunks.jsonl" if revision_dir else config.PHASE2_CHUNKS_FILE), (
        revision_dir / "documents.jsonl" if revision_dir else ingestion_config.DOCUMENTS_FILE)


def _read_active_pointer(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _assert_active_corpus_unchanged(active_path, corpus_pointer):
    try:
        current_pointer = _read_active_pointer(active_path)
    except (OSError, ValueError) as exc:
        raise IndexingError("Active corpus changed during index build") from exc
    if current_pointer != corpus_pointer:
        raise IndexingError("Active corpus changed during index build")


def load_chunks(chunk_path=None) -> list[Chunk]:
    if chunk_path is None:
        _, _, chunk_path, _ = _corpus_snapshot()
    chunks: list[Chunk] = []
    seen: set[str] = set()
    with open(chunk_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"Malformed chunk at line {line_no}: {exc}") from exc
            text = (raw.get("text") or "").strip()
            chunk_id = raw.get("chunk_id")
            if not chunk_id or not text:
                raise RuntimeError(
                    f"Chunk at line {line_no} missing chunk_id or text")
            if chunk_id in seen:
                raise RuntimeError(f"Duplicate chunk_id: {chunk_id}")
            seen.add(chunk_id)
            chunks.append(Chunk.from_dict(raw))
    if not chunks:
        raise RuntimeError("Phase 2 corpus contains no chunks")
    return chunks


def run_indexing() -> dict:
    config.validate_env()
    started = time.time()
    active_path, corpus_pointer, chunk_path, docs_path = _corpus_snapshot()
    chunks = load_chunks(chunk_path)
    received = len(chunks)

    texts = [c.text for c in chunks]
    vectors, failed_indexes = embed_texts(texts)
    failed_chunk_ids = [chunks[i].chunk_id for i in failed_indexes]
    embedded = [v for v in vectors if v is not None]
    if not embedded:
        raise IndexingError("All embedding requests failed; aborting indexing")
    dim = len(embedded[0])

    encoder = SparseEncoder().fit(texts)
    client = connect()
    collection = revision_id(chunks, config.JINA_EMBEDDING_MODEL, dim,
                             encoder.fingerprint)
    revision_dir = config.REVISIONS_DIR / collection
    # An existing immutable revision is never overwritten. It may be selected
    # again only after its exact point set is revalidated below.
    if client.collection_exists(collection):
        raise IndexingError(f"Immutable revision already exists: {collection}")
    ensure_collection(client, dim, collection)
    ensure_payload_indexes(client, collection)
    encoder.save(revision_dir / "sparse_vocab.json")

    items = []
    for chunk, vector in zip(chunks, vectors):
        if vector is None:
            continue
        sparse = encoder.encode(chunk.text)
        items.append((chunk, vector,
                      {"indices": sparse.indices, "values": sparse.values}))
    indexed, failed_points, upsert_errors = upsert_points(
        client, items, sparse_revision=encoder.fingerprint,
        collection_name=collection)

    stats = {
        "chunks_received": received,
        "chunks_embedded": len(embedded),
        "embedding_failures": len(failed_indexes),
        "failed_chunk_ids": failed_chunk_ids,
        "qdrant_points_indexed": indexed,
        "qdrant_points_failed": failed_points,
        "qdrant_errors": upsert_errors,
        "collection_name": collection,
        "embedding_model": config.JINA_EMBEDDING_MODEL,
        "vector_dimensionality": dim,
        "dense_config": {"vector_name": config.DEFAULT_RETRIEVAL.dense_vector_name,
                         "distance": "Cosine"},
        "sparse_config": {"vector_name": config.DEFAULT_RETRIEVAL.sparse_vector_name,
                          "method": "BM25", "variant": "bm25-rsj-positive-idf-linear-query-tf-v2",
                          "k1": 1.2, "b": 0.75,
                          "vocabulary_fingerprint": encoder.fingerprint},
        "hybrid_config": {"fusion": "RRF",
                          "rrf_k": config.DEFAULT_RETRIEVAL.rrf_k,
                          "dense_top_k": config.DEFAULT_RETRIEVAL.dense_top_k,
                          "sparse_top_k": config.DEFAULT_RETRIEVAL.sparse_top_k,
                          "final_top_k": config.DEFAULT_RETRIEVAL.final_top_k},
        "complete": not failed_indexes and not failed_points and indexed == received,
        "elapsed_s": round(time.time() - started, 1),
    }
    # Do not publish a partial embedding result or a point set that differs
    # from the corpus. In particular, no old points can survive in a new
    # collection after shortening, deletion, or ACL restriction.
    if stats["complete"]:
        ids = expected_point_ids(chunks)
        if len(ids) != received:
            raise IndexingError("Chunk IDs do not map to unique Qdrant point IDs")
        try:
            docs = [json.loads(line) for line in docs_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()]
        except (OSError, ValueError) as exc:
            raise IndexingError("Cannot validate the source document manifest") from exc
        permission_fixture_path = (config.PROJECT_ROOT / "access_control"
                                   / "fixtures" / "expected_access_matrix.json")
        try:
            permission_fixture = json.loads(
                permission_fixture_path.read_text(encoding="utf-8"))
            fixture_rows = {row["document_id"]: row
                            for row in permission_fixture["documents"]}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise IndexingError("Reviewed permission fixture is unavailable or malformed") from exc
        doc_rows = {row.get("document_id"): row for row in docs}
        fields = ("document_id", "document_type", "seller_id", "department",
                  "classification", "scenario_id")
        if set(doc_rows) != set(fixture_rows):
            raise IndexingError("Corpus document set differs from reviewed permission fixture")
        chunks_by_document: dict[str, int] = {}
        for chunk in chunks:
            chunks_by_document[chunk.document_id] = chunks_by_document.get(chunk.document_id, 0) + 1
        if sum(int(row.get("chunk_count", 0)) for row in docs) != received:
            raise IndexingError("Document and chunk manifest counts do not agree")
        if any(chunks_by_document.get(document_id, 0) != int(row.get("chunk_count", 0))
               for document_id, row in doc_rows.items()):
            raise IndexingError("Per-document chunk counts do not agree with the corpus manifest")
        for document_id, row in doc_rows.items():
            expected = fixture_rows[document_id]
            if any(row.get(field) != expected.get(field) for field in fields):
                raise IndexingError(
                    f"ACL metadata for {document_id} differs from reviewed fixture")
        if any(not isinstance(c.seller_id, str) or not c.seller_id.strip()
               or c.classification not in {"PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED"}
               or not c.document_type for c in chunks):
            raise IndexingError("ACL metadata validation failed; revision not published")
        stats["document_count"] = len(docs)
        stats["corpus_revision"] = corpus_pointer.get("revision") if corpus_pointer else None
        if len({row.get("document_id") for row in docs}) != len(docs):
            raise IndexingError("Source corpus has duplicate document IDs")
        stats["chunk_count"] = received
        stats["point_ids"] = sorted(ids)
        payload_hashes = expected_payload_hashes(chunks, encoder.fingerprint)
        stats["point_payload_hashes"] = payload_hashes
        stats["manifest_schema"] = config.INDEX_MANIFEST_SCHEMA
        stats["corpus_fingerprint"] = __import__("hashlib").sha256(
            json.dumps([c.to_dict() for c in chunks], sort_keys=True,
                       separators=(",", ":")).encode()).hexdigest()
        stats["revision"] = collection
        # Exact point-count/ID validation precedes the atomic alias change.
        _assert_active_corpus_unchanged(active_path, corpus_pointer)
        finalize_revision(client, collection, stats, ids, payload_hashes)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.INDEXING_REPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    if not stats["complete"]:
        raise IndexingError(
            f"Incomplete index: {received} chunks received, "
            f"{len(failed_indexes)} embedding failures, "
            f"{failed_points} upsert failures, {indexed} points indexed. "
            f"Report: {config.INDEXING_REPORT_FILE}. "
            + ("; ".join(upsert_errors[:3]) if upsert_errors else "")
        )
    return stats
