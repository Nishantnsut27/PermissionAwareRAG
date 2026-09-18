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


class FilterError(ValueError):
    pass


_client: QdrantClient | None = None


def connect() -> QdrantClient:
    global _client
    if _client is not None:
        return _client
    try:
        client = QdrantClient(
            url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY, timeout=60)
        client.get_collections()
    except Exception as exc:
        raise RuntimeError(f"Cannot reach Qdrant Cloud: {exc}") from exc
    _client = client
    return _client


def reset_client() -> None:
    global _client
    _client = None


def point_id_for(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))


def ensure_collection(client: QdrantClient, vector_size: int) -> str:
    name = config.QDRANT_COLLECTION_NAME
    if client.collection_exists(name):
        info = client.get_collection(name)
        params = info.config.params
        problems = []
        vectors = params.vectors
        if not isinstance(vectors, dict):
            problems.append(
                "existing collection uses a single unnamed dense vector; "
                f"'{config.DEFAULT_RETRIEVAL.dense_vector_name}' is required")
            size = None
        else:
            dense = vectors.get(config.DEFAULT_RETRIEVAL.dense_vector_name)
            size = getattr(dense, "size", None)
        has_sparse = (
            config.DEFAULT_RETRIEVAL.sparse_vector_name
            in (params.sparse_vectors or {})
        )
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
        try:
            client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION_NAME,
                field_name=field_name,
                field_schema=KeywordIndexParams(type="keyword"),
                wait=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Cannot create payload index '{field_name}' on "
                f"'{config.QDRANT_COLLECTION_NAME}': {exc}") from exc


def _match_any_condition(key: str, values) -> FieldCondition:
    options = [str(v) for v in values if v is not None]
    if not options:
        raise FilterError(
            f"Filter '{key}' has an empty allow-list. Dropping it would widen "
            "the query, so it is rejected instead.")
    return FieldCondition(key=key, match=MatchAny(any=options))


def build_filter(filters: dict | None) -> Filter | None:
    """Translate a filter spec into a Qdrant filter.

    Phase 3 owns no authorization, but Phase 4 derives its pre-filter from this
    function, so it must never silently drop a condition: a key that cannot be
    turned into a constraint raises instead of widening the search. `None` or
    `{}` means "deliberately unfiltered" and is the only way to get no filter.
    """
    if filters is None:
        return None
    if not isinstance(filters, dict):
        raise FilterError(
            f"filters must be a dict, got {type(filters).__name__}")
    if not filters:
        return None
    conditions = []
    for key, value in filters.items():
        if value is None:
            raise FilterError(f"Filter '{key}' is None; omit the key instead.")
        if isinstance(value, dict):
            if "any" not in value:
                raise FilterError(
                    f"Filter '{key}' dict must contain an 'any' allow-list.")
            conditions.append(_match_any_condition(key, value["any"]))
        elif isinstance(value, (list, tuple, set, frozenset)):
            conditions.append(_match_any_condition(key, value))
        else:
            conditions.append(FieldCondition(
                key=key, match=MatchValue(value=str(value))))
    return Filter(must=conditions)


def upsert_points(
    client: QdrantClient,
    items: list[tuple[Chunk, list[float], dict]],
    batch_size: int = 100,
) -> tuple[int, int, list[str]]:
    """Upsert points; returns (indexed, failed, errors).

    Point IDs are uuid5(chunk_id), so re-running overwrites instead of
    duplicating. Upsert errors are reported, never swallowed.
    """
    points = [
        PointStruct(
            id=point_id_for(chunk.chunk_id),
            vector={
                config.DEFAULT_RETRIEVAL.dense_vector_name: dense,
                config.DEFAULT_RETRIEVAL.sparse_vector_name: sparse,
            },
            payload=chunk.payload(),
        )
        for chunk, dense, sparse in items
    ]
    ok = 0
    errors: list[str] = []
    for i in range(0, len(points), batch_size):
        window = points[i:i + batch_size]
        try:
            client.upsert(
                collection_name=config.QDRANT_COLLECTION_NAME,
                points=window,
                wait=True,
            )
            ok += len(window)
        except Exception as exc:
            errors.append(
                f"upsert failed for points {i}..{i + len(window) - 1}: "
                f"{type(exc).__name__}: {exc}")
    return ok, len(points) - ok, errors
