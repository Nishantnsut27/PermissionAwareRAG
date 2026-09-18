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
from .sparse_search import SparseEncoder


def load_chunks() -> list[Chunk]:
    chunks: list[Chunk] = []
    seen: set[str] = set()
    with open(config.PHASE2_CHUNKS_FILE, encoding="utf-8") as fh:
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
    chunks = load_chunks()
    received = len(chunks)

    texts = [c.text for c in chunks]
    vectors, failed_indexes = embed_texts(texts)
    failed_chunk_ids = [chunks[i].chunk_id for i in failed_indexes]
    embedded = [v for v in vectors if v is not None]
    if not embedded:
        raise RuntimeError("All embedding requests failed; aborting indexing")
    dim = len(embedded[0])

    encoder = SparseEncoder().fit(texts)
    encoder.save()

    client = connect()
    collection = ensure_collection(client, dim)
    ensure_payload_indexes(client)

    items = []
    for chunk, vector in zip(chunks, vectors):
        if vector is None:
            continue
        sparse = encoder.encode(chunk.text)
        items.append((chunk, vector,
                      {"indices": sparse.indices, "values": sparse.values}))
    indexed, failed_points = upsert_points(client, items)

    stats = {
        "chunks_received": received,
        "chunks_embedded": len(embedded),
        "embedding_failures": len(failed_indexes),
        "failed_chunk_ids": failed_chunk_ids,
        "qdrant_points_indexed": indexed,
        "qdrant_points_failed": failed_points,
        "collection_name": collection,
        "embedding_model": config.JINA_EMBEDDING_MODEL,
        "vector_dimensionality": dim,
        "dense_config": {"vector_name": config.DEFAULT_RETRIEVAL.dense_vector_name,
                         "distance": "Cosine"},
        "sparse_config": {"vector_name": config.DEFAULT_RETRIEVAL.sparse_vector_name,
                          "method": "BM25", "k1": 1.2, "b": 0.75},
        "hybrid_config": {"fusion": "RRF",
                          "rrf_k": config.DEFAULT_RETRIEVAL.rrf_k,
                          "dense_top_k": config.DEFAULT_RETRIEVAL.dense_top_k,
                          "sparse_top_k": config.DEFAULT_RETRIEVAL.sparse_top_k,
                          "final_top_k": config.DEFAULT_RETRIEVAL.final_top_k},
        "elapsed_s": round(time.time() - started, 1),
    }
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.INDEXING_REPORT_FILE, "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)
    return stats
