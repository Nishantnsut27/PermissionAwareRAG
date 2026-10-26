"""Phase 8 evidence-aware selection.

The stages before this one answer different questions. Phase 6 asks "which
authorized chunks could possibly matter", Phase 7 asks "which of them is most
relevant to the query". Neither asks the question that actually determines what
the reader sees:

    Given what I have already decided to show, is this next document still
    worth a slot?

The existing selector cannot ask it. It ranks documents by score and keeps the
top few, so five documents that all answer the same half of a two-part question
score identically well and crowd out the document answering the other half.
That is the mechanism behind the two largest false-positive categories Phase 6
measured ("wrong document type", "generic organization document"): they are not
badly retrieved, they are redundantly selected.

This module replaces "top K by score" with bounded greedy marginal utility. A
document's value is its relevance PLUS what it adds that nothing already
selected provides, MINUS how much it repeats. The selector stops when the best
remaining candidate is no longer worth a slot, which is what lets the context
shrink on simple queries without capping it on hard ones.

Safety properties, each of them a lesson from an earlier phase:

- Subtractive only. Selection can drop and reorder candidates it was handed;
  it can never introduce one. Authorization therefore cannot be widened here,
  and the final `can_access` gate still runs afterwards regardless.
- Relevance stays primary. Coverage is a bounded bonus on top of the Jina
  score, never a replacement for it (spec section 18). An irrelevant document
  cannot buy its way in by covering an aspect - the relative floor is a hard
  gate applied before any coverage reasoning (spec section 22).
- The first document is always kept. No threshold can empty the context, which
  is how E053 collapsed when the candidate pool was cut in Phase 6.
- Deterministic. Every ordering has an explicit stable tie-break, so the same
  inputs always produce the same evidence set (spec section 39).
- Purely local. No network calls, no model: only scores and metadata that
  Phases 5-7 already computed (spec sections 41-42).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import config, query_aware

# Words that carry no evidence signal, so two documents sharing only these are
# not similar in any useful sense.
_STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for with
from by as is are was were be been being it its has have had not no do does did
we you they he she i our your their his her them us me my
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{2,}")


@dataclass(frozen=True)
class EvidenceWeights:
    """Bounded coverage bonuses, relative to a relevance scale of 0..1.

    Relevance is normalised against the best chunk in the query and enters the
    utility with an implicit coefficient of 1.0, so these weights are literally
    "how much of the best document's relevance is one new piece of coverage
    worth". Keeping them well under 1.0 is what stops coverage from overturning
    a decisive relevance judgment.

    `min_marginal_utility` is the stopping rule and the real precision lever:
    it is the value a candidate must still be worth, given everything already
    selected, to earn a slot at all.
    """
    aspect: float = 0.30
    evidence_type: float = 0.15
    entity: float = 0.20
    redundancy: float = 0.25
    min_marginal_utility: float = 0.0

    @classmethod
    def from_config(cls) -> "EvidenceWeights":
        return cls(
            aspect=config.EVIDENCE_ASPECT_WEIGHT,
            evidence_type=config.EVIDENCE_TYPE_WEIGHT,
            entity=config.EVIDENCE_ENTITY_WEIGHT,
            redundancy=config.EVIDENCE_REDUNDANCY_PENALTY,
            min_marginal_utility=config.EVIDENCE_MIN_MARGINAL_UTILITY,
        )

    def only(self, *names: str) -> "EvidenceWeights":
        """Ablation view keeping only the named signals.

        The stopping rule is preserved so arms differ in WHICH evidence they
        value, not in how many documents they are allowed to keep.
        """
        kept = set(names)
        return EvidenceWeights(
            aspect=self.aspect if "aspect" in kept else 0.0,
            evidence_type=self.evidence_type if "evidence_type" in kept else 0.0,
            entity=self.entity if "entity" in kept else 0.0,
            redundancy=self.redundancy if "redundancy" in kept else 0.0,
            min_marginal_utility=self.min_marginal_utility,
        )


@dataclass
class EvidenceStats:
    """Why the selector chose what it chose (spec sections 30 and 40).

    Diagnostics only: this never reaches the LLM context, it exists so a
    selection decision can be explained and measured after the fact.
    """
    applied: bool = False
    required_aspects: tuple[str, ...] = ()
    covered_aspects: tuple[str, ...] = ()
    evidence_types: tuple[str, ...] = ()
    selected_documents: int = 0
    selected_chunks: int = 0
    considered_documents: int = 0
    stopped_early: bool = False
    redundancy_rate: float = 0.0
    decisions: list[dict] = field(default_factory=list)

    @property
    def aspect_coverage(self) -> float | None:
        if not self.required_aspects:
            return None
        return len(self.covered_aspects) / len(self.required_aspects)

    def to_dict(self) -> dict:
        return {
            "applied": self.applied,
            "required_aspects": list(self.required_aspects),
            "covered_aspects": list(self.covered_aspects),
            "aspect_coverage": (None if self.aspect_coverage is None
                                else round(self.aspect_coverage, 4)),
            "evidence_types": list(self.evidence_types),
            "selected_documents": self.selected_documents,
            "selected_chunks": self.selected_chunks,
            "considered_documents": self.considered_documents,
            "stopped_early": self.stopped_early,
            "redundancy_rate": round(self.redundancy_rate, 4),
            "decisions": self.decisions,
        }


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_RE.findall((text or "").lower())
                     if token not in _STOPWORDS)


def _similarity(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap. Local, deterministic, and good enough to notice that
    two passages are restating each other without pretending to be semantic."""
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


@dataclass
class _Document:
    """A candidate document and the evidence it carries."""
    document_id: str
    order: int
    chunks: list = field(default_factory=list)      # [(chunk, score)]
    best: float | None = None
    aggregate: float = 0.0
    aspects: frozenset[str] = frozenset()
    evidence_type: str | None = None
    entities: frozenset[str] = frozenset()
    tokens: frozenset[str] = frozenset()


def _document_id(chunk) -> str:
    metadata = getattr(chunk, "metadata", None) or {}
    return getattr(chunk, "document_id", None) or metadata.get(
        "document_id") or "unknown"


def _entity_keys(plan, text: str) -> frozenset[str]:
    """Which specific event this passage is about.

    Structured identifiers only, plus explicit month tokens. Bare seller codes
    are excluded by `structured_identifiers`, because every document of a
    seller carries the code and treating it as an event key would make every
    one of them look like the same event - the Phase 6 flood, rebuilt.
    """
    if plan is None:
        return frozenset()
    keys = {identifier for identifier in query_aware.structured_identifiers(plan)
            if query_aware.identifier_in_text(identifier, text)}
    keys |= set(query_aware.date_identity(plan, text))
    return frozenset(keys)


def _group(items: list, plan, support_weight: float) -> list[_Document]:
    """Group scored chunks into documents, first-seen order preserved.

    `items` arrives best-first from the reranker, so first-seen order is the
    score-free tie-break and matches what the existing selector uses.

    The relevance figure kept here is the SAME aggregate the existing selector
    ranks by - best passage plus a damped sum of the rest - not merely the best
    passage. Corroboration is a real signal in this corpus: a document whose
    evidence is spread over several passages can outrank a single stronger
    passage, and dropping that term silently changed which documents survived
    the capacity cap (measured: E021 lost the gold INC-2026-0312 to a
    single-chunk ticket). Phase 8 must add coverage on top of Phase 7's
    ranking, never replace part of it.
    """
    documents: dict[str, _Document] = {}
    for chunk, score in items:
        document_id = _document_id(chunk)
        entry = documents.get(document_id)
        if entry is None:
            entry = _Document(document_id=document_id, order=len(documents))
            documents[document_id] = entry
        entry.chunks.append((chunk, score))
        if score is not None and (entry.best is None or score > entry.best):
            entry.best = score
    for entry in documents.values():
        values = sorted((s for _, s in entry.chunks if s is not None),
                        reverse=True)
        entry.aggregate = (values[0] + support_weight * sum(values[1:])
                           if values else 0.0)
        text = " ".join((getattr(c, "text", "") or "") for c, _ in entry.chunks)
        entry.aspects = frozenset(query_aware.aspect_hits(plan, text))
        entry.entities = _entity_keys(plan, text)
        entry.tokens = _tokens(text)
        metadata = getattr(entry.chunks[0][0], "metadata", None) or {}
        entry.evidence_type = query_aware.DOCUMENT_TYPE_TO_EVIDENCE.get(
            metadata.get("document_type"))
    return list(documents.values())


def _marginal_utility(document: _Document, selected: list[_Document],
                      plan, weights: EvidenceWeights, best_overall: float,
                      ) -> tuple[float, dict]:
    """Value of adding this document to what is already selected.

        utility = relevance
                + aspect   * (new required aspects / all required aspects)
                + evidence * (this record type is new and the plan wants it)
                + entity   * (covers an event nothing selected covers)
                - redundancy * (closest textual overlap with a selected doc)

    Relevance is normalised to 0..1 against the best chunk in this query, which
    keeps the weights comparable across queries: Jina scores are only
    meaningful relative to their own query.
    """
    relevance = (document.aggregate / best_overall) if best_overall else 0.0
    reasons: dict = {"relevance": round(relevance, 4)}
    utility = relevance

    required = frozenset(plan.required_aspects) if plan else frozenset()
    if weights.aspect > 0 and required:
        covered = frozenset().union(*(d.aspects for d in selected)) \
            if selected else frozenset()
        new_aspects = (document.aspects & required) - covered
        if new_aspects:
            gain = weights.aspect * (len(new_aspects) / len(required))
            utility += gain
            reasons["new_aspects"] = sorted(new_aspects)
            reasons["aspect_gain"] = round(gain, 4)

    if weights.evidence_type > 0 and document.evidence_type:
        wanted = set(plan.evidence_types) if plan else set()
        already = {d.evidence_type for d in selected}
        if document.evidence_type in wanted and document.evidence_type not in already:
            utility += weights.evidence_type
            reasons["new_evidence_type"] = document.evidence_type

    if weights.entity > 0 and document.entities:
        covered = frozenset().union(*(d.entities for d in selected)) \
            if selected else frozenset()
        new_entities = document.entities - covered
        if new_entities:
            utility += weights.entity
            reasons["new_entities"] = sorted(new_entities)

    if weights.redundancy > 0 and selected:
        overlap = max(_similarity(document.tokens, d.tokens) for d in selected)
        if overlap > 0:
            penalty = weights.redundancy * overlap
            utility -= penalty
            reasons["redundancy"] = round(overlap, 4)
            reasons["redundancy_penalty"] = round(penalty, 4)

    reasons["utility"] = round(utility, 4)
    return utility, reasons


def _pick_chunks(document: _Document, plan, floor: float | None,
                 max_chunks: int) -> list:
    """Chunks inside a kept document, complementary evidence first.

    The best passage always stays - it is why the document earned a slot. After
    that a passage is preferred when it carries a required aspect none of the
    already-kept passages of THIS document carry, which is what lets a document
    whose cause is in chunk 1 and whose remediation is in chunk 2 contribute
    both. When nothing complementary remains it falls back to score order, so
    behaviour matches the existing selector on single-topic documents.
    """
    ordered = sorted(document.chunks,
                     key=lambda pair: (pair[1] is None, -(pair[1] or 0.0),
                                       str(getattr(pair[0], "chunk_id", ""))))
    if max_chunks <= 0:
        return list(ordered)
    keep = [ordered[0]]
    covered = set(query_aware.aspect_hits(plan, getattr(ordered[0][0], "text", "")))
    required = set(plan.required_aspects) if plan else set()

    remaining = ordered[1:]
    while len(keep) < max_chunks and remaining:
        complementary = None
        for candidate in remaining:
            chunk, score = candidate
            # The floor applies to every passage after the first, exactly as in
            # the existing selector: a document earning a slot on one strong
            # passage must not drag in its weak tail.
            if floor is not None and score is not None and score < floor:
                continue
            hits = set(query_aware.aspect_hits(plan, getattr(chunk, "text", "")))
            if required and (hits & required) - covered:
                complementary = candidate
                break
        if complementary is None:
            for candidate in remaining:
                _chunk, score = candidate
                if floor is not None and score is not None and score < floor:
                    continue
                complementary = candidate
                break
        if complementary is None:
            break
        keep.append(complementary)
        remaining.remove(complementary)
        covered |= set(query_aware.aspect_hits(
            plan, getattr(complementary[0], "text", "")))
    return keep


def select_evidence(items: list, plan, cfg, floor: float | None,
                    weights: EvidenceWeights | None = None,
                    ) -> tuple[list, EvidenceStats]:
    """Choose the smallest complementary evidence set from ranked candidates.

    `items` must be the reranked (chunk, score) pairs, best first, and `floor`
    the relative relevance floor the existing selector would apply. Returns
    (selected_pairs, stats) with the pairs grouped by document and ordered
    document-by-document, ready for the caller to restore reading order.
    """
    weights = weights or EvidenceWeights.from_config()
    stats = EvidenceStats(applied=True)
    if not items:
        return [], stats

    documents = _group(items, plan, cfg.document_support_weight)
    stats.considered_documents = len(documents)
    stats.required_aspects = tuple(plan.required_aspects) if plan else ()

    # Relevance is normalised against the strongest DOCUMENT in this query, on
    # the same aggregate scale, so the weights mean the same thing regardless
    # of how Jina happened to scale this particular query's scores.
    best_overall = max((d.aggregate for d in documents), default=0.0)

    # Hard gate, before any coverage reasoning: evidence coverage may rescue a
    # moderately relevant document, never an irrelevant one (spec section 22).
    eligible = [d for d in documents
                if floor is None or d.best is None or d.best >= floor]
    if not eligible:
        eligible = documents

    max_documents = cfg.max_documents if cfg.max_documents > 0 else len(eligible)
    selected: list[_Document] = []
    remaining = list(eligible)

    while remaining and len(selected) < max_documents:
        scored = []
        for document in remaining:
            utility, reasons = _marginal_utility(
                document, selected, plan, weights, best_overall)
            scored.append((utility, document.order, document, reasons))
        # Stable: highest utility, then original fusion order.
        scored.sort(key=lambda entry: (-entry[0], entry[1]))
        utility, _order, document, reasons = scored[0]

        # The first document is always kept: a stopping rule must never be able
        # to empty the context.
        if selected and utility < weights.min_marginal_utility:
            stats.stopped_early = True
            for _u, _o, rejected, rejected_reasons in scored:
                stats.decisions.append({
                    "document_id": rejected.document_id, "selected": False,
                    "reason": "marginal utility below threshold",
                    **rejected_reasons})
            break

        selected.append(document)
        remaining.remove(document)
        stats.decisions.append({"document_id": document.document_id,
                                "selected": True, **reasons})

    # Output ordering is a packing concern (Phase 9, spec section 50), not a
    # selection one. The greedy loop above decides WHICH documents earn a slot;
    # emitting them in that selection order would let a coverage bonus that only
    # won a slot also reorder the context, demoting a higher-relevance document
    # below a lower-relevance one it merely edged out on novelty. Measured: on
    # E020 the greedy order pushed the gold INC-2026-0812 (relevance 1.00) to
    # rank 2 behind a non-gold same-aspect incident (relevance 0.77), costing
    # MRR for zero recall gain - exactly the "coverage overturning strong
    # relevance" the spec forbids (sections 18-19). So the chosen SET is emitted
    # in Phase 7 relevance order - the same (-aggregate, fusion-order) key the
    # score-ordered selector uses - which keeps Phase 8 to changing the set and
    # leaves ordering to Phase 9. The greedy order is still recorded in
    # `decisions` for explainability.
    chosen: list = []
    for document in sorted(selected, key=lambda d: (-d.aggregate, d.order)):
        chosen.extend(_pick_chunks(document, plan, floor,
                                   cfg.max_chunks_per_document))

    covered = frozenset().union(*(d.aspects for d in selected)) \
        if selected else frozenset()
    required = frozenset(plan.required_aspects) if plan else frozenset()
    stats.covered_aspects = tuple(sorted(covered & required))
    stats.evidence_types = tuple(sorted(
        {d.evidence_type for d in selected if d.evidence_type}))
    stats.selected_documents = len(selected)
    stats.selected_chunks = len(chosen)
    stats.redundancy_rate = _redundancy_rate(selected)
    return chosen, stats


def _redundancy_rate(selected: list[_Document]) -> float:
    """Mean pairwise textual overlap across the selected documents.

    A measure of how much of the context restates itself: 0 means every
    document contributes distinct material, higher means the reader is being
    shown the same thing repeatedly.
    """
    if len(selected) < 2:
        return 0.0
    overlaps = [
        _similarity(selected[i].tokens, selected[j].tokens)
        for i in range(len(selected))
        for j in range(i + 1, len(selected))
    ]
    return sum(overlaps) / len(overlaps) if overlaps else 0.0
