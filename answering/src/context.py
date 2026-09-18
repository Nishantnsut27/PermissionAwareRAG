"""Context construction and citations from authorized chunks only."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config

SEPARATOR = "\n\n----------------------------\n\n"


@dataclass
class Source:
    document_id: str
    document_type: str | None
    label: str
    seller_id: str | None
    department: str | None
    pages: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    relevance: float | None = None

    def citation(self) -> str:
        page = ""
        if self.pages:
            page = (f" — Page {self.pages[0]}" if len(self.pages) == 1
                    else f" — Pages {', '.join(str(p) for p in self.pages)}")
        return f"{self.label}{page}"

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_type": self.document_type,
            "label": self.label,
            "seller_id": self.seller_id,
            "department": self.department,
            "pages": self.pages,
            "chunk_ids": self.chunk_ids,
            "relevance": (None if self.relevance is None
                          else round(self.relevance, 4)),
            "citation": self.citation(),
        }


def _label(document_id: str, document_type: str | None) -> str:
    return f"{document_type} {document_id}" if document_type else document_id


def build(scored_chunks: list[tuple[object, float | None]],
          cfg: config.AnsweringConfig | None = None) -> tuple[str, list[Source]]:
    """Group authorized chunks by document into a bounded context block.

    Returns (context_text, sources). Only chunks that actually fit inside the
    character budget are cited, so the source list never overstates what the
    model was shown.
    """
    cfg = cfg or config.DEFAULT_ANSWERING
    grouped: dict[str, dict] = {}
    for chunk, score in scored_chunks:
        metadata = chunk.metadata or {}
        document_id = chunk.document_id or metadata.get("document_id") or "unknown"
        entry = grouped.setdefault(document_id, {
            "document_type": metadata.get("document_type"),
            "seller_id": metadata.get("seller_id"),
            "department": metadata.get("department"),
            "pages": [],
            "chunks": [],
            "best": score,
        })
        for page in metadata.get("page_numbers") or []:
            if isinstance(page, int) and page not in entry["pages"]:
                entry["pages"].append(page)
        entry["chunks"].append((chunk, score))
        if score is not None and (entry["best"] is None or score > entry["best"]):
            entry["best"] = score

    order = sorted(
        grouped.items(),
        key=lambda kv: (kv[1]["best"] is None, -(kv[1]["best"] or 0.0)),
    )

    blocks: list[str] = []
    sources: list[Source] = []
    used = 0
    for document_id, entry in order:
        label = _label(document_id, entry["document_type"])
        header = [f"Document: {label}"]
        if entry["seller_id"]:
            header.append(f"Seller: {entry['seller_id']}")
        if entry["department"]:
            header.append(f"Department: {entry['department']}")
        if entry["pages"]:
            header.append("Page: " + ", ".join(str(p) for p in sorted(entry["pages"])))

        included_chunks: list[str] = []
        included_ids: list[str] = []
        header_text = "\n".join(header)
        block_len = len(header_text)
        for chunk, _ in entry["chunks"]:
            text = (chunk.text or "").strip()
            if not text:
                continue
            cost = len(text) + 2
            over = (used + block_len + cost + len(SEPARATOR)
                    > cfg.max_context_chars)
            if over and (blocks or included_chunks):
                break
            included_chunks.append(text)
            included_ids.append(chunk.chunk_id)
            block_len += cost
        if not included_chunks:
            continue

        blocks.append(header_text + "\n\n" + "\n\n".join(included_chunks))
        sources.append(Source(
            document_id=document_id,
            document_type=entry["document_type"],
            label=label,
            seller_id=entry["seller_id"],
            department=entry["department"],
            pages=sorted(entry["pages"]),
            chunk_ids=included_ids,
            relevance=entry["best"],
        ))
        used += block_len + len(SEPARATOR)
        if used >= cfg.max_context_chars:
            break

    return SEPARATOR.join(blocks), sources
