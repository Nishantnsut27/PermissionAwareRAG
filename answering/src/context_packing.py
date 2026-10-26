"""Phase 9: evidence-aware context packing.

Phase 8 decides WHICH authorized passages are worth showing. This stage decides
HOW that fixed, already-authorized evidence is laid out in the final LLM context
under an unchanged budget. It is the last stop before the prompt.

Why it exists (measured, see docs/phase9-context-packing.md): the character
budget binds in 28 of 45 golden cases and drops ~24% of the passages Phase 8
selected. The previous builder (`context.build`) fills the budget breadth-first
by depth ROUND and, within a document, by raw score - so when the budget is
tight it can keep a document's redundant background passage and drop the
aspect-complementary one Phase 8 deliberately paired with it (the remediation
half of a "cause + remediation" answer). That is exactly the "drop a low-value
block before cutting a high-value one" rule of spec section 21, unmet.

What Phase 9 changes, and nothing else:

- Document BREADTH is packed first, identically to the old builder: every
  selected document still gets its single strongest passage before any document
  gets a second. The set of documents that reach the context is therefore
  byte-identical to before, so Recall / Precision / MRR (all document-level)
  cannot move and gold documents cannot be dropped here.
- The REMAINING budget is then filled by marginal evidence value: a depth
  passage that carries a required query aspect none of its document's already
  placed passages carry is preferred over one that does not, and ties break by
  score then document/reading order. This raises how much answer-bearing
  evidence survives a tight budget without spending a single extra character.

Safety properties, each a hard invariant and each tested:

- Subtractive only. Phase 9 can drop and reorder passages it was handed; it can
  never introduce one, so `Phase 9 output subset of Phase 8 output` and the
  final `can_access` gate upstream still bounds everything shown.
- Budget never grows (spec section 20): the operative limit is the same
  `cfg.max_context_chars` the old builder used.
- Whole passages only (spec section 19/21): a passage is kept or dropped
  entirely; its text is never summarised, paraphrased or cut mid-way.
- Deterministic (spec section 30): every ordering has an explicit stable
  tie-break; no randomness.
- Purely local (spec section 29): only text, scores and metadata already
  computed upstream; no OpenJEV, Jina, Qdrant or LLM call.

When Phase 9 is disabled, or no plan is available, callers fall back to
`context.build` unchanged, so a packing decision can never become a recall or
availability failure.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config, context, evidence_selection, query_aware

# Reuse the existing rendering vocabulary so the on-the-wire context format is
# unchanged unless a feature explicitly changes it.
SEPARATOR = context.SEPARATOR
Source = context.Source
_label = context._label
_reading_key = context._reading_key


@dataclass
class PackStats:
    """What the packer did, for evaluation and debugging (never sent to the LLM)."""
    documents: int = 0
    placed_chunks: int = 0
    dropped_chunks: int = 0
    aspect_chunks_placed: int = 0
    aspect_chunks_dropped: int = 0
    budget_bound: bool = False

    def to_dict(self) -> dict:
        return {
            "documents": self.documents,
            "placed_chunks": self.placed_chunks,
            "dropped_chunks": self.dropped_chunks,
            "aspect_chunks_placed": self.aspect_chunks_placed,
            "aspect_chunks_dropped": self.aspect_chunks_dropped,
            "budget_bound": self.budget_bound,
        }


@dataclass
class _Doc:
    document_id: str
    document_type: str | None = None
    seller_id: str | None = None
    department: str | None = None
    order: int = 0
    best: float | None = None
    chunks: list = field(default_factory=list)  # [(chunk, score)]


def _group(pairs: list) -> list[_Doc]:
    """Group scored passages by document, preserving first-seen (Phase 8) order.

    Identical grouping to `context.build`, so the document set and order are the
    same before any Phase 9 feature acts.
    """
    docs: dict[str, _Doc] = {}
    for chunk, score in pairs:
        metadata = getattr(chunk, "metadata", None) or {}
        document_id = (getattr(chunk, "document_id", None)
                       or metadata.get("document_id") or "unknown")
        entry = docs.get(document_id)
        if entry is None:
            entry = _Doc(document_id=document_id,
                         document_type=metadata.get("document_type"),
                         seller_id=metadata.get("seller_id"),
                         department=metadata.get("department"),
                         order=len(docs))
            docs[document_id] = entry
        entry.chunks.append((chunk, score))
        if score is not None and (entry.best is None or score > entry.best):
            entry.best = score
    return list(docs.values())


def _header(entry: _Doc, compact: bool) -> str:
    """Per-document header. `compact` merges the seller/department onto one line;
    identifiers are always preserved (spec section 15)."""
    label = _label(entry.document_id, entry.document_type)
    if not compact:
        lines = [f"Document: {label}"]
        if entry.seller_id:
            lines.append(f"Seller: {entry.seller_id}")
        if entry.department:
            lines.append(f"Department: {entry.department}")
        return "\n".join(lines)
    tail = " | ".join(
        part for part in (
            f"Seller: {entry.seller_id}" if entry.seller_id else "",
            f"Dept: {entry.department}" if entry.department else "",
        ) if part)
    return f"Document: {label}" + (f"\n{tail}" if tail else "")


def _passage_cost(text: str) -> int:
    return len(text) + 2


def _text_of(chunk) -> str:
    return (getattr(chunk, "text", "") or "").strip()


def _pack(pairs: list, cfg, plan, *, compact_headers: bool, query_order: bool,
          dedup: bool, value_depth: bool) -> tuple[str, list, PackStats]:
    """Feature-flagged packer. With every flag off and `value_depth` off it
    reproduces `context.build`'s breadth-first-by-round allocation exactly."""
    stats = PackStats()
    docs = _group(pairs)
    stats.documents = len(docs)
    if not docs:
        return "", [], stats

    required = frozenset(plan.required_aspects) if plan else frozenset()

    # Document order. Phase 8 already hands documents over best-relevance-first;
    # `query_order` only promotes exact identifier/entity matches ahead of ties,
    # and never changes which documents get a breadth slot (every document is
    # offered its best passage regardless of order), so recall is unaffected.
    if query_order and plan is not None:
        def _identifier_doc(entry: _Doc) -> bool:
            text = " ".join(_text_of(c) for c, _ in entry.chunks)
            return any(query_aware.identifier_in_text(i, text)
                       for i in query_aware.structured_identifiers(plan))
        docs = sorted(docs, key=lambda d: (0 if _identifier_doc(d) else 1,
                                           d.order))

    headers = {d.document_id: _header(d, compact_headers) for d in docs}
    # Passages of a document, strongest first; reading order restored at render.
    ranked = {
        d.document_id: sorted(
            d.chunks, key=lambda pair: (pair[1] is None, -(pair[1] or 0.0),
                                        str(getattr(pair[0], "chunk_id", ""))))
        for d in docs
    }
    allocated: dict[str, list] = {d.document_id: [] for d in docs}
    placed_tokens: list[frozenset] = []  # for dedup
    used = 0

    def _fits(cost: int, document_id: str) -> bool:
        return not (used + cost > cfg.max_context_chars
                    and (used or allocated[document_id]))

    # Phase A - breadth. Every document's strongest passage, in document order.
    # Identical to the old builder's round 0, so the document set is preserved.
    for entry in docs:
        queue = ranked[entry.document_id]
        if not queue:
            continue
        chunk, score = queue[0]
        text = _text_of(chunk)
        if not text:
            continue
        cost = _passage_cost(text) + len(headers[entry.document_id]) + 2 + len(SEPARATOR)
        if not _fits(cost, entry.document_id):
            continue
        allocated[entry.document_id].append((chunk, score))
        placed_tokens.append(evidence_selection._tokens(text))
        used += cost

    # Phase B - depth. Fill the rest of the budget with the remaining passages.
    if value_depth:
        depth = _value_ordered_depth(docs, ranked, required, plan)
    else:
        depth = _round_ordered_depth(docs, ranked)

    for entry, chunk, score in depth:
        text = _text_of(chunk)
        if not text:
            continue
        tokens = evidence_selection._tokens(text)
        if dedup and _is_redundant(tokens, text, entry, allocated, plan,
                                   required, placed_tokens, cfg):
            continue
        cost = _passage_cost(text)
        if not allocated[entry.document_id]:
            cost += len(headers[entry.document_id]) + 2 + len(SEPARATOR)
        if not _fits(cost, entry.document_id):
            continue
        allocated[entry.document_id].append((chunk, score))
        placed_tokens.append(tokens)
        used += cost

    context_text, sources = _render(docs, allocated, headers, plan, stats,
                                    required, pairs)
    return context_text, sources, stats


def _round_ordered_depth(docs: list[_Doc], ranked: dict) -> list:
    """Depth passages in the old builder's order: round 1 across all documents,
    then round 2, ... (breadth-first by depth round). Used by the control arms
    so they differ from the baseline in exactly one dimension."""
    out = []
    rounds = max((len(ranked[d.document_id]) for d in docs), default=0)
    for depth_index in range(1, rounds):
        for entry in docs:
            queue = ranked[entry.document_id]
            if depth_index < len(queue):
                chunk, score = queue[depth_index]
                out.append((entry, chunk, score))
    return out


def _value_ordered_depth(docs: list[_Doc], ranked: dict,
                         required: frozenset, plan) -> list:
    """Depth passages ordered by marginal evidence value (spec sections 17/21).

    Tier 1: a passage that carries a required aspect none of the ALREADY-PLACED
    passages of its own document carry (the complementary half of a multi-part
    answer). Tier 2: everything else. Within a tier, stronger score first, then
    document order and reading order for a fully deterministic result.

    Tier 1 membership is computed against Phase A's breadth passage (the
    document's strongest), which is what makes "keep the remediation passage the
    cause passage did not cover" win over "keep another cause passage".
    """
    covered: dict[str, set] = {}
    for entry in docs:
        queue = ranked[entry.document_id]
        first_text = _text_of(queue[0][0]) if queue else ""
        covered[entry.document_id] = set(query_aware.aspect_hits(plan, first_text))

    tier1, tier2 = [], []
    for entry in docs:
        queue = ranked[entry.document_id]
        for depth_index in range(1, len(queue)):
            chunk, score = queue[depth_index]
            hits = set(query_aware.aspect_hits(plan, _text_of(chunk)))
            complementary = bool((hits & required) - covered[entry.document_id])
            key = (-(score or 0.0), entry.order, depth_index)
            (tier1 if complementary else tier2).append((key, entry, chunk, score))
    tier1.sort(key=lambda t: t[0])
    tier2.sort(key=lambda t: t[0])
    return [(entry, chunk, score) for _key, entry, chunk, score in tier1 + tier2]


def _is_redundant(tokens: frozenset, text: str, entry: _Doc, allocated: dict,
                  plan, required: frozenset, placed_tokens: list, cfg) -> bool:
    """A passage is redundant only if it is a near-duplicate of one already
    placed AND it carries no required aspect that its document's placed passages
    lack. The aspect guard means dedup can never drop answer-bearing evidence."""
    tau = config.CONTEXT_PACKING_REDUNDANCY_TAU
    hits = set(query_aware.aspect_hits(plan, text))
    placed_here = [c for c, _ in allocated[entry.document_id]]
    covered_here: set = set()
    for c in placed_here:
        covered_here |= set(query_aware.aspect_hits(plan, _text_of(c)))
    if (hits & required) - covered_here:
        return False  # complementary - never drop
    return any(evidence_selection._similarity(tokens, prev) >= tau
               for prev in placed_tokens)


def _render(docs: list[_Doc], allocated: dict, headers: dict, plan,
            stats: PackStats, required: frozenset,
            original_pairs: list) -> tuple[str, list]:
    blocks: list[str] = []
    sources: list = []
    placed_ids: set = set()
    for entry in docs:
        chosen = allocated[entry.document_id]
        if not chosen:
            continue
        pages: list[int] = []
        for chunk, _ in chosen:
            for page in (getattr(chunk, "metadata", None) or {}).get("page_numbers") or []:
                if isinstance(page, int) and page not in pages:
                    pages.append(page)
        chosen_reading = sorted(chosen, key=_reading_key)
        header_text = headers[entry.document_id]
        if pages:
            header_text += "\nPage: " + ", ".join(str(p) for p in sorted(pages))
        blocks.append(header_text + "\n\n"
                      + "\n\n".join(_text_of(c) for c, _ in chosen_reading))
        sources.append(Source(
            document_id=entry.document_id,
            document_type=entry.document_type,
            label=_label(entry.document_id, entry.document_type),
            seller_id=entry.seller_id,
            department=entry.department,
            pages=sorted(pages),
            chunk_ids=[getattr(c, "chunk_id", None) for c, _ in chosen_reading],
            relevance=entry.best,
        ))
        for chunk, _ in chosen_reading:
            placed_ids.add(getattr(chunk, "chunk_id", None))
            stats.placed_chunks += 1
            if query_aware.aspect_hits(plan, _text_of(chunk)):
                stats.aspect_chunks_placed += 1

    # Diagnostics on everything Phase 8 handed us but the budget could not fit.
    for chunk, _ in original_pairs:
        if getattr(chunk, "chunk_id", None) in placed_ids:
            continue
        stats.dropped_chunks += 1
        if query_aware.aspect_hits(plan, _text_of(chunk)):
            stats.aspect_chunks_dropped += 1
    stats.budget_bound = stats.dropped_chunks > 0
    return SEPARATOR.join(blocks), sources


# --- Public API ------------------------------------------------------------

def build_packed(pairs: list, cfg: config.AnsweringConfig | None = None,
                 plan: object | None = None) -> tuple[str, list]:
    """Production entry point. Returns (context_text, sources), same contract as
    `context.build`. Falls back to `context.build` when Phase 9 is disabled or
    no plan is available, so behaviour is never worse than the frozen builder."""
    cfg = cfg or config.DEFAULT_ANSWERING
    if not config.CONTEXT_PACKING_ENABLED or plan is None:
        return context.build(pairs, cfg)
    context_text, sources, _stats = _pack(
        pairs, cfg, plan, compact_headers=False, query_order=False,
        dedup=True, value_depth=True)
    return context_text, sources


def build_packed_with_stats(pairs: list, cfg, plan) -> tuple[str, list, PackStats]:
    """Same as `build_packed` but also returns the packing diagnostics."""
    return _pack(pairs, cfg, plan, compact_headers=False, query_order=False,
                 dedup=True, value_depth=True)


# Ablation arms (spec section 31), cumulative. `grouped` delegates to the frozen
# builder so it is a true control; each later arm adds exactly one feature.
_ARM_FLAGS = {
    "grouped": None,
    "metadata_dedup": dict(compact_headers=True, query_order=False,
                           dedup=False, value_depth=False),
    "ordering": dict(compact_headers=True, query_order=True,
                     dedup=False, value_depth=False),
    "redundancy": dict(compact_headers=True, query_order=True,
                       dedup=True, value_depth=False),
    "final": dict(compact_headers=False, query_order=False,
                  dedup=True, value_depth=True),
}


def pack(pairs: list, cfg, plan, arm: str) -> tuple[str, list]:
    """Evaluation entry: run one ablation arm and return (context_text, sources)."""
    if arm == "grouped" or arm not in _ARM_FLAGS or _ARM_FLAGS[arm] is None:
        return context.build(pairs, cfg)
    context_text, sources, _stats = _pack(pairs, cfg, plan, **_ARM_FLAGS[arm])
    return context_text, sources
