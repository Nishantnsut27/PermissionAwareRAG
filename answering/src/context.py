"""Token-bounded context packing with provenance derived from included spans."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from functools import lru_cache

from . import config

SEPARATOR = "\n\n----------------------------\n\n"
OMISSION = "\n[... middle omitted ...]\n"


@lru_cache(maxsize=1)
def _encoding():
    # The chosen hosted model has no tiktoken mapping in the checked-in
    # runtime. A UTF-8 byte is a conservative upper bound on model tokens and
    # works fully offline; never let counting trigger an implicit download.
    return _Utf8ByteBound()


class _Utf8ByteBound:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="ignore")


def count_tokens(text: str) -> int:
    """Return a conservative token upper bound (UTF-8 bytes, offline)."""
    return len(_encoding().encode(text))


@dataclass
class Source:
    document_id: str
    document_type: str | None
    label: str
    seller_id: str | None
    department: str | None
    pages: list[int] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    relevance: float | None = None
    spans: list[dict] = field(default_factory=list, repr=False)

    def citation(self) -> str:
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
            "evidence_ids": self.evidence_ids,
            "relevance": (None if self.relevance is None
                          else round(self.relevance, 4)),
            "citation": self.citation(),
        }


def _label(document_id: str, document_type: str | None) -> str:
    return f"{document_type} {document_id}" if document_type else document_id


def _pages(chunk) -> list[int]:
    values = getattr(chunk, "metadata", {}).get("page_numbers") or []
    return sorted({page for page in values
                   if isinstance(page, int) and not isinstance(page, bool) and page > 0})


def _included_pages(chunk, ranges: list[list[int]], trimmed: int = 0) -> list[int]:
    """Map included character ranges to chunk-relative page spans.

    Legacy multi-page chunks have no reliable mapping for an excerpt. In that
    case omit page attribution rather than citing text the model did not see.
    """
    metadata = getattr(chunk, "metadata", {}) or {}
    spans = metadata.get("page_spans")
    if isinstance(spans, list) and spans:
        valid = [span for span in spans if isinstance(span, dict)
                 and type(span.get("page")) is int and span["page"] > 0
                 and type(span.get("start")) is int
                 and type(span.get("end")) is int
                 and 0 <= span["start"] < span["end"] <= len(chunk.text)]
        if len(valid) != len(spans):
            return []
        return sorted({span["page"] for span in valid
                       if any(span["start"] < b + trimmed
                              and span["end"] > a + trimmed for a, b in ranges)})
    pages = _pages(chunk)
    full = ranges == [[0, len((chunk.text or "").strip())]]
    return pages if full or len(pages) == 1 else []


def _overlap_trim(text: str, previous: str) -> tuple[str, int]:
    if not text or not previous:
        return text, 0
    maximum = min(len(text), len(previous), 3000)
    for width in range(maximum, 39, -1):
        if previous.endswith(text[:width]):
            return text[width:], width
    return text, 0


def _excerpt(text: str, token_limit: int) -> tuple[str, list[list[int]]]:
    """Return a bounded head/tail excerpt and original character ranges."""
    if token_limit <= count_tokens(OMISSION) + 2:
        return "", []
    if count_tokens(text) <= token_limit:
        return text, [[0, len(text)]]
    low, high = 1, len(text)
    best = ("", [])
    while low <= high:
        keep = (low + high) // 2
        left_count = (keep + 1) // 2
        right_count = keep - left_count
        left = text[:left_count]
        right = text[-right_count:] if right_count else ""
        candidate = left + OMISSION + right
        if count_tokens(candidate) <= token_limit:
            ranges = [[0, len(left)]]
            if right:
                ranges.append([len(text) - len(right), len(text)])
            best = (candidate, ranges)
            low = keep + 1
        else:
            high = keep - 1
    return best


def _evidence_id(chunk_id: str, ranges: list[list[int]], text: str) -> str:
    raw = f"{chunk_id}|{ranges}|{text}".encode("utf-8")
    return "EV-" + hashlib.sha256(raw).hexdigest()[:12]


def build(scored_chunks: list[tuple[object, float | None]],
          cfg: config.AnsweringConfig | None = None) -> tuple[str, list[Source]]:
    """Pack evidence under hard token/character limits, then derive citations.

    An oversized first passage is represented by a head/tail excerpt if the
    excerpt and its complete provenance header fit. Otherwise it is omitted;
    an empty result lets the service return its insufficient-evidence outcome.
    """
    cfg = cfg or config.DEFAULT_ANSWERING
    grouped: dict[str, list[tuple[object, float | None]]] = {}
    metadata_by_doc: dict[str, dict] = {}
    best_by_doc: dict[str, float | None] = {}
    for chunk, score in scored_chunks:
        metadata = getattr(chunk, "metadata", {}) or {}
        document_id = getattr(chunk, "document_id", None) or metadata.get("document_id") or "unknown"
        grouped.setdefault(document_id, []).append((chunk, score))
        metadata_by_doc.setdefault(document_id, metadata)
        old = best_by_doc.get(document_id)
        if score is not None and (old is None or score > old):
            best_by_doc[document_id] = score

    order = sorted(grouped, key=lambda doc_id: (
        best_by_doc.get(doc_id) is None, -(best_by_doc.get(doc_id) or 0.0)))
    blocks: list[str] = []
    selected: list[dict] = []
    previous_by_doc: dict[str, str] = {}

    def within_budget(candidate_blocks: list[str]) -> bool:
        rendered = SEPARATOR.join(candidate_blocks)
        return (count_tokens(rendered) <= cfg.max_context_tokens
                and len(rendered) <= cfg.max_context_chars)

    for document_id in order:
        metadata = metadata_by_doc[document_id]
        for chunk, score in grouped[document_id]:
            raw_text = getattr(chunk, "text", "") or ""
            leading_trim = len(raw_text) - len(raw_text.lstrip())
            original = raw_text.strip()
            if not original:
                continue
            duplicate_of = next((row for row in selected
                                 if row["document_id"] == document_id
                                 and row["original_text"] == original), None)
            pages = _pages(chunk)
            chunk_id = str(getattr(chunk, "chunk_id", ""))
            if duplicate_of:
                if chunk_id and chunk_id not in duplicate_of["chunk_ids"]:
                    duplicate_of["chunk_ids"].append(chunk_id)
                continue

            prior = previous_by_doc.get(document_id, "")
            start_trim = 0
            body, start_trim = _overlap_trim(original, prior)
            if not body.strip():
                # Fully overlapping chunks still add provenance to the span.
                match = next((row for row in reversed(selected)
                              if row["document_id"] == document_id), None)
                if match:
                    if chunk_id and chunk_id not in match["chunk_ids"]:
                        match["chunk_ids"].append(chunk_id)
                continue

            ranges = [[start_trim, len(original)]]
            pages = _included_pages(chunk, ranges, leading_trim)
            evidence_text = body
            evidence_id = _evidence_id(chunk_id, ranges, evidence_text)
            page_text = ", ".join(map(str, pages)) if pages else "unspecified"
            label = _label(document_id, metadata.get("document_type"))
            header = (f"Evidence [{evidence_id}] | {label} | Pages {page_text}"
                      f"\n")
            block = header + evidence_text
            candidate = blocks + [block]
            if not within_budget(candidate) and not selected:
                reserve_header = (f"Evidence [{evidence_id}] | {label} | Pages "
                                  f"{page_text if len(page_text) > 11 else 'unspecified'}\n")
                remaining = cfg.max_context_tokens - count_tokens(
                    SEPARATOR.join(blocks) + reserve_header)
                remaining = min(remaining, cfg.max_context_chars - len(reserve_header))
                excerpt, excerpt_ranges = _excerpt(original, remaining)
                if excerpt:
                    ranges = [[a, b] for a, b in excerpt_ranges]
                    pages = _included_pages(chunk, ranges, leading_trim)
                    page_text = ", ".join(map(str, pages)) if pages else "unspecified"
                    evidence_text = excerpt
                    evidence_id = _evidence_id(chunk_id, ranges, evidence_text)
                    header = (f"Evidence [{evidence_id}] | {label} | Pages {page_text}"
                              f"\n")
                    block = header + evidence_text
                    candidate = blocks + [block]
            if not within_budget(candidate):
                continue

            blocks.append(block)
            selected.append({
                "document_id": document_id, "document_type": metadata.get("document_type"),
                "seller_id": metadata.get("seller_id"),
                "department": metadata.get("department"),
                "chunk_ids": [chunk_id] if chunk_id else [],
                "evidence_id": evidence_id, "pages": pages,
                "score": score, "text": evidence_text,
                "original_text": original, "ranges": ranges,
                "start_trim": start_trim,
            })
            previous_by_doc[document_id] = (prior + " " + body).strip()

    sources_by_doc: dict[str, Source] = {}
    for row in selected:
        source = sources_by_doc.get(row["document_id"])
        if source is None:
            label = _label(row["document_id"], row["document_type"])
            source = Source(row["document_id"], row["document_type"], label,
                            row["seller_id"], row["department"], relevance=row["score"])
            sources_by_doc[row["document_id"]] = source
        source.pages = sorted(set(source.pages + row["pages"]))
        source.chunk_ids = list(dict.fromkeys(source.chunk_ids + row["chunk_ids"]))
        source.evidence_ids.append(row["evidence_id"])
        source.spans.append({
            "evidence_id": row["evidence_id"],
            "document_id": row["document_id"],
            "chunk_ids": row["chunk_ids"],
            "pages": row["pages"],
            "text": row["text"],
            "offsets": row["ranges"],
        })

    return SEPARATOR.join(blocks), list(sources_by_doc.values())


def build_balanced(scored_chunks: list[tuple[object, float | None]],
                   sellers: list[str], cfg: config.AnsweringConfig) -> tuple[str, list[Source]]:
    """Partition the existing context budget so one seller cannot exhaust it.

    Each included seller uses the existing span/citation packer. Separators
    count against BOTH limits; tiny budgets may omit sellers, never exceed caps.
    """
    groups = {seller: [pair for pair in scored_chunks
                       if pair[0].metadata.get("seller_id") == seller]
              for seller in sellers}
    active = [seller for seller in sellers if groups[seller]]
    if not active:
        return "", []
    separators = len(active) - 1
    token_pool = max(0, cfg.max_context_tokens - separators * count_tokens(SEPARATOR))
    char_pool = max(0, cfg.max_context_chars - separators * len(SEPARATOR))
    blocks, sources = [], []
    for index, seller in enumerate(active):
        tokens = token_pool // len(active) + (index < token_pool % len(active))
        chars = char_pool // len(active) + (index < char_pool % len(active))
        header = f"Seller account: {seller}\n"
        block, included = build(groups[seller], replace(
            cfg, max_context_tokens=max(0, tokens - count_tokens(header)),
            max_context_chars=max(0, chars - len(header))))
        if block:
            blocks.append(header + block)
            sources.extend(included)
    return SEPARATOR.join(blocks), sources
