"""Internal representations shared across the ingestion stages."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SourceFile:
    path: Path
    rel_path: str
    extension: str
    format: str


@dataclass
class ExtractedMetadata:
    values: dict[str, object]
    provenance: dict[str, str] = field(default_factory=dict)
    conflicts: list[str] = field(default_factory=list)


@dataclass
class ValidationIssue:
    severity: str
    code: str
    message: str
    source_file: str
    document_id: str | None = None
    field: str | None = None

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "document_id": self.document_id,
            "source_file": self.source_file,
            "field": self.field,
        }


@dataclass
class CleanedDocument:
    text: str
    page_spans: list[tuple[int, int, int]]
    page_count: int


@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_id: str
    document_type: str
    seller_id: str
    department: str | None
    classification: str
    created_date: str | None
    owner: str | None
    scenario_id: str | None
    scenario_ids: list[str]
    source_file: str
    source_format: str
    page_numbers: list[int]
    chunk_index: int
    token_count: int
    char_count: int

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "seller_id": self.seller_id,
            "department": self.department,
            "classification": self.classification,
            "created_date": self.created_date,
            "owner": self.owner,
            "scenario_id": self.scenario_id,
            "scenario_ids": self.scenario_ids,
            "source_file": self.source_file,
            "source_format": self.source_format,
            "page_numbers": self.page_numbers,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "char_count": self.char_count,
        }


@dataclass
class ProcessedDocument:
    document_id: str
    document_type: str
    seller_id: str
    department: str | None
    classification: str
    created_date: str | None
    owner: str | None
    scenario_id: str | None
    scenario_ids: list[str]
    source_file: str
    source_format: str
    page_count: int
    char_count: int
    token_count: int
    chunk_count: int
    metadata_provenance: dict[str, str]
    extra_metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "seller_id": self.seller_id,
            "department": self.department,
            "classification": self.classification,
            "created_date": self.created_date,
            "owner": self.owner,
            "scenario_id": self.scenario_id,
            "scenario_ids": self.scenario_ids,
            "source_file": self.source_file,
            "source_format": self.source_format,
            "page_count": self.page_count,
            "char_count": self.char_count,
            "token_count": self.token_count,
            "chunk_count": self.chunk_count,
            "metadata_provenance": self.metadata_provenance,
            "extra_metadata": self.extra_metadata,
        }
