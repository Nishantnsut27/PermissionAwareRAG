"""Immutable Qdrant index revisions and atomic active-alias publication.

Revision collections are never overwritten or deleted by this module. The
logical active alias is changed only after the collection and its local
manifest have been validated. Rollback is another alias switch to a complete
revision; deletion/restriction updates require a newly built exact point set.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from qdrant_client.models import (
    CreateAlias, CreateAliasOperation, DeleteAlias, DeleteAliasOperation,
)

from . import config


class RevisionError(RuntimeError):
    pass


def logical_alias() -> str:
    return f"{config.QDRANT_COLLECTION_NAME}_active"


def revision_id(chunks: list, embedding_model: str, dimension: int,
                sparse_fingerprint: str) -> str:
    material = {
        "chunks": sorted((c.chunk_id, hashlib.sha256(json.dumps(
            c.to_dict(), sort_keys=True, ensure_ascii=False).encode()).hexdigest())
                          for c in chunks),
        "embedding_model": embedding_model,
        "dimension": dimension,
        "sparse_fingerprint": sparse_fingerprint,
        "schema": config.INDEX_MANIFEST_SCHEMA,
    }
    digest = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()[:20]
    # A fresh immutable name per build keeps an abandoned partial upload from
    # blocking a later retry of the same corpus/configuration.
    return f"{config.QDRANT_COLLECTION_NAME}__r_{digest}_{uuid.uuid4().hex[:8]}"


def expected_point_ids(chunks: list) -> set[str]:
    from .qdrant_store import point_id_for
    return {point_id_for(chunk.chunk_id) for chunk in chunks}


def expected_payload_hashes(chunks: list, sparse_fingerprint: str) -> dict[str, str]:
    from .qdrant_store import point_id_for
    result = {}
    for chunk in chunks:
        payload = {**chunk.payload(), "_sparse_revision": sparse_fingerprint}
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False)
        result[point_id_for(chunk.chunk_id)] = hashlib.sha256(
            canonical.encode()).hexdigest()
    return result


def _stored_point_records(client, collection: str) -> dict[str, str]:
    records: dict[str, str] = {}
    offset = None
    while True:
        rows, offset = client.scroll(collection_name=collection, limit=256,
                                     offset=offset, with_payload=True,
                                     with_vectors=False)
        for row in rows:
            canonical = json.dumps(row.payload or {}, sort_keys=True,
                                   separators=(",", ":"), ensure_ascii=False)
            records[str(row.id)] = hashlib.sha256(canonical.encode()).hexdigest()
        if offset is None:
            break
    return records


def validate_collection(client, collection: str, expected_ids: set[str],
                        expected_payload_hashes: dict[str, str] | None = None) -> None:
    count = client.count(collection_name=collection, exact=True).count
    actual_records = _stored_point_records(client, collection)
    actual_ids = set(actual_records)
    if count != len(expected_ids) or actual_ids != expected_ids:
        raise RevisionError(
            f"revision point set mismatch: expected {len(expected_ids)}, "
            f"count={count}, unique_ids={len(actual_ids)}")
    if (expected_payload_hashes is not None
            and actual_records != expected_payload_hashes):
        raise RevisionError("revision payload/ACL metadata differs from its manifest")


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(value, fh, sort_keys=True, indent=2, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def publish_alias(client, collection: str, manifest: dict,
                  expected_ids: set[str],
                  expected_payload_hashes: dict[str, str] | None = None) -> None:
    """Validate, persist the manifest, then atomically point the alias."""
    if (not manifest.get("complete")
            or manifest.get("collection_name") != collection
            or manifest.get("manifest_schema") != config.INDEX_MANIFEST_SCHEMA
            or set(manifest.get("point_ids", [])) != expected_ids
            or expected_payload_hashes is None
            or manifest.get("point_payload_hashes") != expected_payload_hashes):
        raise RevisionError("refusing to publish an incomplete/mismatched manifest")
    validate_collection(client, collection, expected_ids,
                        expected_payload_hashes)
    manifest_path = config.REVISIONS_DIR / collection / "manifest.json"
    _atomic_json(manifest_path, manifest)
    aliases = client.get_aliases().aliases
    current = next((row.collection_name for row in aliases
                    if row.alias_name == logical_alias()), None)
    operations = []
    if current:
        operations.append(DeleteAliasOperation(delete_alias=DeleteAlias(
            alias_name=logical_alias())))
    operations.append(CreateAliasOperation(create_alias=CreateAlias(
        collection_name=collection, alias_name=logical_alias())))
    # Qdrant applies operations in this request atomically. No destructive
    # collection operation is issued here.
    client.update_collection_aliases(change_aliases_operations=operations)
    _atomic_json(config.REVISIONS_DIR / "active.json", {
        "alias": logical_alias(), "collection_name": collection,
        "manifest_schema": config.INDEX_MANIFEST_SCHEMA,
    })


def rollback(client, collection: str) -> None:
    """Select a previously validated revision without deleting any revision."""
    path = config.REVISIONS_DIR / collection / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RevisionError("rollback target has no valid local manifest") from exc
    if (not manifest.get("complete")
            or manifest.get("collection_name") != collection
            or manifest.get("manifest_schema") != config.INDEX_MANIFEST_SCHEMA
            or not isinstance(manifest.get("point_payload_hashes"), dict)
            or not manifest.get("corpus_fingerprint")):
        raise RevisionError("rollback target is incomplete or mismatched")
    # A rollback must not resurrect deleted, shortened, or newly restricted
    # content. It is allowed only among index builds for the identical corpus
    # and ACL fingerprint. Rebuild the target revision otherwise.
    active = config.REVISIONS_DIR / "active.json"
    try:
        pointer = json.loads(active.read_text(encoding="utf-8"))
        current_manifest = json.loads((config.REVISIONS_DIR
            / pointer["collection_name"] / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RevisionError("cannot verify current corpus for safe rollback") from exc
    if manifest.get("corpus_fingerprint") != current_manifest.get("corpus_fingerprint"):
        raise RevisionError(
            "rollback would restore stale corpus/ACL data; rebuild from the active corpus")
    expected = set(manifest.get("point_ids", []))
    validate_collection(client, collection, expected,
                        manifest.get("point_payload_hashes"))
    aliases = client.get_aliases().aliases
    current = next((row.collection_name for row in aliases
                    if row.alias_name == logical_alias()), None)
    ops = ([DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=logical_alias()))]
           if current else [])
    ops.append(CreateAliasOperation(create_alias=CreateAlias(
        collection_name=collection, alias_name=logical_alias())))
    client.update_collection_aliases(change_aliases_operations=ops)
    _atomic_json(config.REVISIONS_DIR / "active.json", {
        "alias": logical_alias(), "collection_name": collection,
        "manifest_schema": config.INDEX_MANIFEST_SCHEMA,
    })
