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
        # ASCII-only separator: the model output and the Streamlit UI must
        # never contain an em/en dash (they mis-render as boxes/mojibake on
        # some Windows fonts and latin-1 hops). See text_norm.py.
        page = ""
        if self.pages:
            page = (f" - Page {self.pages[0]}" if len(self.pages) == 1
                    else f" - Pages {', '.join(str(p) for p in self.pages)}")
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


def _header(document_id: str, entry: dict) -> str:
    lines = [f"Document: {_label(document_id, entry['document_type'])}"]
    if entry["seller_id"]:
        lines.append(f"Seller: {entry['seller_id']}")
    if entry["department"]:
        lines.append(f"Department: {entry['department']}")
    return "\n".join(lines)


def _reading_key(pair: tuple) -> tuple:
    """Render a document's passages in the order the document reads."""
    chunk, _ = pair
    metadata = getattr(chunk, "metadata", None) or {}
    index = metadata.get("chunk_index")
    if not isinstance(index, int):
        index = 0
    return (index, str(getattr(chunk, "chunk_id", "")))


def build(scored_chunks: list[tuple[object, float | None]],
          cfg: config.AnsweringConfig | None = None) -> tuple[str, list[Source]]:
    """Group authorized chunks by document into a bounded context block.

    Returns (context_text, sources). Only chunks that actually fit inside the
    character budget are cited, so the source list never overstates what the
    model was shown.

    Document order and passage order are taken as given. `selection.select`
    already ranked documents by aggregate relevance and restored each
    document's passages to reading order; re-sorting here by best-chunk score
    would discard that and split a document's prose into score-ordered
    fragments.

    The budget is spent breadth-first: every selected document gets its
    strongest passage before any document gets a second one. Packing
    depth-first instead let the leading documents consume the whole budget and
    silently dropped the tail, which discarded the breadth `select` had just
    decided on and cost recall on multi-document questions.
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
            "chunks": [],
            "best": score,
        })
        entry["chunks"].append((chunk, score))
        if score is not None and (entry["best"] is None or score > entry["best"]):
            entry["best"] = score

    order = list(grouped.items())
    headers = {document_id: _header(document_id, entry)
               for document_id, entry in order}

    # Strongest passage first *within* a document decides what is offered in
    # each round; reading order is restored at render time.
    pending = {
        document_id: sorted(entry["chunks"],
                            key=lambda pair: (pair[1] is None, -(pair[1] or 0.0)))
        for document_id, entry in order
    }
    allocated: dict[str, list] = {document_id: [] for document_id, _ in order}
    used = 0
    rounds = max((len(v) for v in pending.values()), default=0)
    for depth in range(rounds):
        for document_id, _ in order:
            queue = pending[document_id]
            if depth >= len(queue):
                continue
            chunk, score = queue[depth]
            text = (chunk.text or "").strip()
            if not text:
                continue
            cost = len(text) + 2
            if not allocated[document_id]:
                # First passage also pays for this document's header block.
                cost += len(headers[document_id]) + 2 + len(SEPARATOR)
            if used + cost > cfg.max_context_chars and (used or allocated[document_id]):
                continue
            allocated[document_id].append((chunk, score))
            used += cost

    blocks: list[str] = []
    sources: list[Source] = []
    for document_id, entry in order:
        chosen = allocated[document_id]
        if not chosen:
            continue
        pages: list[int] = []
        for chunk, _ in chosen:
            for page in (chunk.metadata or {}).get("page_numbers") or []:
                if isinstance(page, int) and page not in pages:
                    pages.append(page)
        chosen = sorted(chosen, key=_reading_key)
        header_text = headers[document_id]
        if pages:
            header_text += "\nPage: " + ", ".join(str(p) for p in sorted(pages))
        blocks.append(header_text + "\n\n"
                      + "\n\n".join((c.text or "").strip() for c, _ in chosen))
        sources.append(Source(
            document_id=document_id,
            document_type=entry["document_type"],
            label=_label(document_id, entry["document_type"]),
            seller_id=entry["seller_id"],
            department=entry["department"],
            pages=sorted(pages),
            chunk_ids=[c.chunk_id for c, _ in chosen],
            relevance=entry["best"],
        ))

    return SEPARATOR.join(blocks), sources
