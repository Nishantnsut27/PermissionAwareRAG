"""Qdrant collection management, upserts and filtered vector search."""
from __future__ import annotations

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    KeywordIndexParams,
    MatchAny,
    MatchValue,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

from . import config
from .models import Chunk


class CollectionMismatchError(RuntimeError):
    pass


def connect() -> QdrantClient:
    try:
        client = QdrantClient(
            url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)
        client.get_collections()
        return client
    except Exception as exc:
        raise RuntimeError(f"Cannot reach Qdrant Cloud: {exc}") from exc


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def ensure_collection(client: QdrantClient, vector_size: int) -> str:
    name = config.QDRANT_COLLECTION_NAME
    if client.collection_exists(name):
        info = client.get_collection(name)
        params = info.config.params
        dense = (params.vectors or {}).get(config.DEFAULT_RETRIEVAL.dense_vector_name)
        size = getattr(dense, "size", None)
        has_sparse = (
            config.DEFAULT_RETRIEVAL.sparse_vector_name
            in (params.sparse_vectors or {})
        )
        problems = []
        if size != vector_size:
            problems.append(
                f"existing dense size {size} != embedding size {vector_size}")
        if not has_sparse:
            problems.append("existing collection has no sparse slot "
                            f"'{config.DEFAULT_RETRIEVAL.sparse_vector_name}'")
        if problems:
            raise CollectionMismatchError(
                f"Qdrant collection '{name}' is incompatible: "
                + "; ".join(problems)
                + ". Delete it explicitly if a rebuild is intended.")
        return name
    client.create_collection(
        collection_name=name,
        vectors_config={
            config.DEFAULT_RETRIEVAL.dense_vector_name: VectorParams(
                size=vector_size, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            config.DEFAULT_RETRIEVAL.sparse_vector_name: SparseVectorParams()
        },
    )
    return name


def ensure_payload_indexes(client: QdrantClient) -> None:
    for field_name in config.PAYLOAD_KEYWORD_FIELDS:
        client.create_payload_index(
            collection_name=config.QDRANT_COLLECTION_NAME,
            field_name=field_name,
            field_schema=KeywordIndexParams(type="keyword"),
            wait=True,
        )


def build_filter(filters: dict | None) -> Filter | None:
    # Keep authorization outside Phase 3; filters are only retrieval capabilities.
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, list):
            conditions.append(FieldCondition(
                key=key, match=MatchAny(any=[str(v) for v in value])))
        else:
            conditions.append(FieldCondition(
                key=key, match=MatchValue(value=str(value))))
    return Filter(must=conditions) if conditions else None


def upsert_points(
    client: QdrantClient,
    items: list[tuple[Chunk, list[float], dict]],
) -> tuple[int, int]:
    ok, failed = 0, 0
    batch: list[PointStruct] = []
    for chunk, dense, sparse in items:
        try:
            batch.append(PointStruct(
                id=point_id_for(chunk.chunk_id),
                vector={
                    config.DEFAULT_RETRIEVAL.dense_vector_name: dense,
                    config.DEFAULT_RETRIEVAL.sparse_vector_name: sparse,
                },
                payload=chunk.payload(),
            ))
        except Exception:
            failed += 1
    for i in range(0, len(batch), 100):
        try:
            client.upsert(
                collection_name=config.QDRANT_COLLECTION_NAME,
                points=batch[i:i + 100],
                wait=True,
            )
            ok += len(batch[i:i + 100])
        except Exception:
            failed += len(batch[i:i + 100])
    return ok, failed
