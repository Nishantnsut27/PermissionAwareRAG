"""Shared data shapes for Phase 3."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    text: str
    document_id: str
    document_type: str | None = None
    seller_id: str | None = None
    department: str | None = None
    classification: str | None = None
    scenario_id: str | None = None
    source_file: str | None = None
    page_numbers: list[int] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict) -> "Chunk":
        known = set(cls.__dataclass_fields__) - {"extra"}
        init = {k: raw.get(k) for k in known if k != "page_numbers"}
        init["page_numbers"] = list(raw.get("page_numbers") or [])
        extra = {k: v for k, v in raw.items() if k not in known}
        return cls(extra=extra, **init)  # type: ignore[arg-type]

    def payload(self) -> dict:
        data: dict = {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_type": self.document_type,
            "seller_id": self.seller_id,
            "department": self.department,
            "classification": self.classification,
            "scenario_id": self.scenario_id,
            "source_file": self.source_file,
            "page_numbers": self.page_numbers,
            "text": self.text,
        }
        data.update(self.extra)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class RetrievalResult:
    chunk_id: str
    document_id: str | None
    text: str
    score: float
    retrieval_method: str
    metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)
