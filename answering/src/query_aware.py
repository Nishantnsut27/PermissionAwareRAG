"""Phase 6 query-aware retrieval signals.

Consumes the Phase 5 QueryPlan and turns it into retrieval inputs. Four
mechanisms, in increasing order of invasiveness:

1. Exact identifier candidates. Chunks whose text contains a query identifier
   (boundary-aware) are added to the fused candidate pool BEFORE the
   authorization re-check. They gain nothing by this: every added candidate
   passes through the same `authorize_chunks` gate as dense/BM25 candidates,
   and an identifier that matches an unauthorized document is denied exactly
   like any other candidate. Dense and BM25 stay authoritative; this only
   rescues documents whose identifier evidence sits outside the fused top-K.

2. Deterministic ordering boost. The authorized candidate list is stably
   reordered so identifier-matched chunks enter the truncated pool first.
   Only the identifier signal drives ordering: stage-level evaluation showed
   that promoting seller-matched chunks as a block (every chunk from a
   mentioned seller matches) evicted org-wide policy documents that
   multi-document queries need - the seller constraint is already enforced
   by the Qdrant scope filter, so boosting it again only distorts ranking.
   Entity/seller/evidence signals are recorded as diagnostics for later
   phases, which consume them at the reranking/selection stages.

3. Bounded evidence-aware promotion. The rework's precision mechanism: a
   small, fixed number of chunks OUTSIDE the candidate pool that carry
   evidence for a required aspect or the query's explicit time reference are
   promoted into the pool ahead of the cut. The pool keeps its size; weaker
   tail chunks give way to evidence-bearing ones. Promotion is bounded
   (config-capped, one chunk per new document, new documents first) because
   stage measurements show no gold document starts in the pool's last six
   slots, while the previous all-block seller boost evicted exactly such
   tail documents (the E014 regression). Seller/entity matches stay
   diagnostics-only for the same measured reason.

4. Per-candidate diagnostics for evaluation and debugging.

Non-goals (later phases): no evidence coverage logic, no adaptive breadth, no
selection or threshold changes, no reranker changes.

Entity/seller/evidence signals are deliberately NOT candidate sources: seller
names appear in a large share of the corpus, so text-matching them would
flood the pool. They only contribute to ordering and diagnostics.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_FILE = (PROJECT_ROOT / "ingestion" / "processed" / "chunks"
               / "chunks.jsonl")

DOCUMENT_TYPE_TO_EVIDENCE = {
    "Support Ticket": "support_ticket",
    "Incident Report": "incident_report",
    "Security Incident Report": "incident_report",
    "Project Document": "project_status",
    "Project Status Report": "project_status",
    "Implementation Plan": "project_status",
    "Requirements Document": "project_status",
    "Technical Notes": "project_status",
    "Runbook": "runbook",
    "Operational Procedure": "runbook",
    "Operational Checklist": "runbook",
    "Policy": "policy",
    "Call Transcript": "call_transcript",
    "Account Overview": "account_overview",
    "Decision Record": "decision_record",
    "Organization Reference": "reference_data",
    "Scenario Context": "reference_data",
}

_SELLER_DISPLAY_NAMES = {
    "S001": "aurelia home decor",
    "S002": "bluepeak electronics",
    "S003": "greencart organics",
    "S004": "strideone footwear",
}

ASPECT_LEXICON = {
    "cause": ("root cause", "caused by", "cause of", "because", "due to",
              "triggered", "defect", "failure", "misconfig", "bug"),
    "containment_actions": ("containment", "mitigation", "rolled back",
                            "rollback", "revoked", "blocked", "rotated",
                            "remediation", "corrective", "hotfix", "patched"),
    "financial_impact": ("inr", "refund", "credit note", "settlement",
                         "shortfall", "amount", "cost", "fine", "penalty",
                         "absorbed", "revenue", "payout", "reversal"),
    "timeline": ("on ", "dated", "duration", "resolved by", "restored",
                 "timeline", "detected at", "window"),
    "status": ("status", "currently", "pending", "approved", "completed",
               "in progress", "closed", "open"),
    "policy_requirement": ("must", "shall", "required", "policy states",
                           "requires", "procedure requires", "threshold"),
    "comparison": ("whereas", "in contrast", "compared", "similar", "differ"),
    "quantities": ("sku", "orders", "count", "number of", "volume", "records",
                   "affected"),
    "responsibility": ("approved by", "signed off", "decision", "owner",
                       "accountable", "agreed"),
}

_MONTH_RE = re.compile(
    r"(?<![a-z])(?:january|february|march|april|june|july|august|september"
    r"|october|november|december)(?![a-z])"
    r"|may(?=\s+(?:\d{1,2}|20\d\d)(?![a-z0-9]))")


def aspect_hits(plan, text: str) -> tuple[str, ...]:
    """Required plan aspects for which the text carries lexical evidence."""
    if plan is None or not plan.required_aspects or not text:
        return ()
    lowered = text.lower()
    hits = []
    for aspect in plan.required_aspects:
        if any(marker in lowered
               for marker in ASPECT_LEXICON.get(aspect, ())):
            hits.append(aspect)
    return tuple(dict.fromkeys(hits))


def date_tokens(text: str) -> frozenset[str]:
    """Explicit month references in text. Bare years are excluded: nearly
    every corpus document shares the same year, so a year token cannot
    identify a specific event. "may" counts only before a day or year so the
    modal verb does not fire the signal."""
    if not text:
        return frozenset()
    return frozenset(m.group(0).lower()
                     for m in _MONTH_RE.finditer(text.lower()))


def date_identity(plan, text: str) -> frozenset[str]:
    """Month tokens shared by the query and a candidate's text."""
    if plan is None or not plan.original_query:
        return frozenset()
    query_tokens = date_tokens(plan.original_query)
    if not query_tokens:
        return frozenset()
    return query_tokens & date_tokens(text)


def identifier_in_text(identifier: str, text: str) -> bool:
    """Boundary-aware exact match.

    The character before and after the match must not be alphanumeric or a
    hyphen, so INC-2026-0519 matches INC-2026-0519 but neither
    INC-2026-05190 nor INC-2026-051, and S001 does not match inside
    S001-ACC-001.
    """
    if not identifier or not text:
        return False
    pattern = (r"(?<![A-Za-z0-9-])" + re.escape(identifier)
               + r"(?![A-Za-z0-9-])")
    return re.search(pattern, text, re.IGNORECASE) is not None


@dataclass
class ChunkSignals:
    """QueryPlan signals observed on one candidate. Diagnostics only."""
    chunk_id: str
    document_id: str | None
    retrieval_method: str
    identifier_match: bool = False
    entity_match: bool = False
    seller_match: bool = False
    evidence_type_match: bool = False
    aspect_hits: tuple[str, ...] = ()
    date_match: frozenset[str] = field(default_factory=frozenset)

    @property
    def matched(self) -> bool:
        return (self.identifier_match or self.entity_match
                or self.seller_match or self.evidence_type_match
                or bool(self.aspect_hits) or bool(self.date_match))

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "retrieval_method": self.retrieval_method,
            "identifier_match": self.identifier_match,
            "entity_match": self.entity_match,
            "seller_match": self.seller_match,
            "evidence_type_match": self.evidence_type_match,
            "aspect_hits": list(self.aspect_hits),
            "date_match": sorted(self.date_match),
        }


@dataclass
class PoolDiagnostics:
    """What the retrieval stage did, for stats and evaluation."""
    fused_candidates: int = 0
    dense_candidates: int = 0
    sparse_candidates: int = 0
    identifier_candidates: int = 0
    authorized_candidates: int = 0
    withheld_candidates: int = 0
    boost_applied: bool = False
    promoted_chunks: int = 0
    promoted_documents: int = 0
    signals: list[ChunkSignals] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fused_candidates": self.fused_candidates,
            "dense_candidates": self.dense_candidates,
            "sparse_candidates": self.sparse_candidates,
            "identifier_candidates": self.identifier_candidates,
            "authorized_candidates": self.authorized_candidates,
            "withheld_candidates": self.withheld_candidates,
            "boost_applied": self.boost_applied,
            "promoted_chunks": self.promoted_chunks,
            "promoted_documents": self.promoted_documents,
            "signals": [signal.to_dict() for signal in self.signals],
        }


_corpus: list | None = None


def _load_corpus(path: Path | None = None) -> list[dict]:
    """The indexed chunk corpus, for local exact-identifier lookup.

    Deliberately tiny memory footprint: only the fields needed to build a
    RetrievalResult-compatible candidate are kept.
    """
    global _corpus
    if _corpus is not None and path is None:
        return _corpus
    path = path or CHUNKS_FILE
    corpus: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            corpus.append({
                "chunk_id": raw.get("chunk_id"),
                "document_id": raw.get("document_id"),
                "text": raw.get("text") or "",
                "document_type": raw.get("document_type"),
                "seller_id": raw.get("seller_id"),
                "department": raw.get("department"),
                "classification": raw.get("classification"),
                "scenario_id": raw.get("scenario_id"),
                "source_file": raw.get("source_file"),
                "page_numbers": raw.get("page_numbers") or [],
            })
    if path is None or path == CHUNKS_FILE:
        _corpus = corpus
    return corpus


def reset_corpus_cache() -> None:
    global _corpus
    _corpus = None


BARE_SELLER_CODE = re.compile(r"^S\d{3}$")


def structured_identifiers(plan) -> list[str]:
    """Query identifiers that may drive candidate generation and ordering.

    Bare seller codes are excluded everywhere: every document of that seller
    contains the code in its text, so both text-matching candidates and the
    ordering boost would promote the seller's whole corpus as a block - the
    measured E014 regression - while the seller constraint is already
    enforced by the Qdrant scope filter and recorded as a metadata signal.
    """
    return [
        i for i in (plan.identifiers if plan else ())
        if i and not i.startswith("SC-") and not BARE_SELLER_CODE.fullmatch(i)
    ]


def identifier_candidates(plan, exclude_chunk_ids: set[str] | None = None,
                          corpus: list[dict] | None = None) -> list:
    """Chunks containing an exact query identifier, best evidence first.

    Bare seller codes (S001..S004) are deliberately NOT identifier candidates:
    every document of that seller contains the code in its text, so text
    matching would flood the pool with the seller's whole corpus. Seller codes
    act through the seller metadata signal and the Qdrant scope filter instead.
    Only structured document ids (INC-, TKT-, PRJ-, POL-, OPS-RB-, CALL-,
    -ACC-) spawn candidates.

    Returns model objects shaped like RetrievalResult (chunk_id, document_id,
    text, score, retrieval_method, metadata) with retrieval_method
    "identifier" and score 0.0: the score carries no ranking meaning, the
    boost is applied later, and authorization is applied by the caller.
    """
    from indexing.src.models import RetrievalResult

    identifiers = structured_identifiers(plan)
    if not identifiers:
        return []
    exclude_chunk_ids = exclude_chunk_ids or set()
    corpus = corpus if corpus is not None else _load_corpus()
    matches: list[RetrievalResult] = []
    for chunk in corpus:
        if chunk["chunk_id"] in exclude_chunk_ids:
            continue
        text = chunk["text"]
        if not any(identifier_in_text(identifier, text)
                   for identifier in identifiers):
            continue
        matched = sum(1 for identifier in identifiers
                      if identifier_in_text(identifier, text))
        matches.append(RetrievalResult(
            chunk_id=chunk["chunk_id"],
            document_id=chunk["document_id"],
            text=text,
            score=float(matched),
            retrieval_method="identifier",
            metadata={key: value for key, value in (
                ("document_type", chunk["document_type"]),
                ("seller_id", chunk["seller_id"]),
                ("department", chunk["department"]),
                ("classification", chunk["classification"]),
                ("scenario_id", chunk["scenario_id"]),
                ("source_file", chunk["source_file"]),
                ("page_numbers", chunk["page_numbers"]),
            ) if value is not None},
        ))
    matches.sort(key=lambda candidate: (-candidate.score, candidate.chunk_id))
    return matches


def _chunk_text(chunk) -> str:
    return getattr(chunk, "text", "") or ""


def _chunk_metadata(chunk) -> dict:
    return getattr(chunk, "metadata", None) or {}


def evaluate_signals(chunk, plan) -> ChunkSignals:
    """Match one candidate against the plan's retrieval signals."""
    metadata = _chunk_metadata(chunk)
    text = _chunk_text(chunk)
    document_id = (getattr(chunk, "document_id", None)
                   or metadata.get("document_id"))
    signals = ChunkSignals(
        chunk_id=getattr(chunk, "chunk_id", ""),
        document_id=document_id,
        retrieval_method=getattr(chunk, "retrieval_method", "unknown"),
    )
    if plan is None:
        return signals

    for identifier in plan.identifiers:
        if identifier_in_text(identifier, text):
            signals.identifier_match = True
            break

    signals.aspect_hits = aspect_hits(plan, text)
    signals.date_match = date_identity(plan, text)

    seller_scope = set(plan.seller_scope or ())
    seller_id = metadata.get("seller_id")
    if seller_id and seller_id != "Organization-wide" and seller_id in seller_scope:
        signals.seller_match = True

    named_sellers = {_SELLER_DISPLAY_NAMES.get(s) for s in seller_scope
                     if _SELLER_DISPLAY_NAMES.get(s)}
    lowered = text.lower()
    for entity in plan.entities or ():
        if entity in plan.identifiers or entity in seller_scope:
            continue
        if re.search(r"(?<![A-Za-z0-9-])" + re.escape(entity.lower())
                     + r"(?![A-Za-z0-9-])", lowered):
            signals.entity_match = True
            break
    if not signals.entity_match and not signals.seller_match and named_sellers:
        if any(re.search(r"(?<![A-Za-z0-9-])" + re.escape(name)
                         + r"(?![A-Za-z0-9-])", lowered)
               for name in named_sellers):
            signals.entity_match = True

    evidence_types = set(plan.evidence_types or ())
    document_type = metadata.get("document_type")
    if (document_type and evidence_types
            and DOCUMENT_TYPE_TO_EVIDENCE.get(document_type) in evidence_types):
        signals.evidence_type_match = True
    return signals


def promote_evidence(chunks: list, plan,
                     pool_size: int | None = None) -> tuple[list, list]:
    """Bounded promotion of evidence-bearing chunks into the pool.

    The pool cut (allowed[:top_k]) is the only place ordering changes pool
    composition, so that is where precision lives. Chunks beyond the cut that
    carry evidence for a required aspect or the query's explicit month
    reference are promoted into the pool, displacing the weakest tail chunks
    instead of shrinking the pool. Guards, each measured during the Phase 6
    rework:

    - No demotion: tail chunks are displaced by push-out only, so no authorized
      candidate can be dropped for lacking a signal (E020's gold document has
      no lexical aspect markers in any of its chunks yet is the strongest
      fusion hit - a demotion would evict it).
    - Cap: at most `QUERY_AWARE_MAX_PROMOTED` chunks enter per query.
    - New documents only: a document already in the pool never gets a second
      chunk through promotion. Swapping chunks inside a pooled document is
      pure churn - it can replace that document's best-scoring passage with a
      weaker one (measured: E001 lost PRJ-S001-003 at selection exactly this
      way) - and adds no evidence diversity. Unfilled cap stays unfilled.
    - Within one document, the most evidence-bearing chunk wins.
    - Boundary insertion: promoted chunks are placed at the pool cut, not the
      front. The pool SET is identical either way, but front insertion
      reorders gold documents inside the pool and drags the pool-order MRR
      down; the reranker scores every chunk independently, so the front order
      carries no information and must not be disturbed.

    `pool_size` (the production pool cut) is required: "new document" means
    absent from the pool slice, and only chunks beyond the cut can belong to
    one. Without a cut there is no pool to defend and promotion is a no-op.

    Returns (new_order, promoted_chunk_ids). With no plan, no criteria, no
    eligible new document, or no cut, the input order is returned unchanged.
    """
    if plan is None or not config.QUERY_AWARE_PROMOTION:
        return chunks, []
    criteria = tuple(config.QUERY_AWARE_PROMOTE_CRITERIA)
    if not criteria:
        return chunks, []

    def _eligible(chunk) -> bool:
        if "aspect" in criteria and aspect_hits(plan, _chunk_text(chunk)):
            return True
        if "date" in criteria and date_identity(plan, _chunk_text(chunk)):
            return True
        return False

    eligible = [chunk for chunk in chunks if _eligible(chunk)]
    if not eligible:
        return chunks, []

    cap = max(0, config.QUERY_AWARE_MAX_PROMOTED)
    if not cap:
        return chunks, []

    eligible.sort(key=_chunk_id)
    eligible.sort(key=lambda chunk: -_evidence_weight(plan, chunk))

    if pool_size is None or pool_size <= 0:
        return chunks, []
    pool_docs = {chunk.document_id for chunk in chunks[:pool_size]}
    seen: set[str] = set()
    new_docs: list = []
    for chunk in eligible:
        if len(new_docs) >= cap:
            break
        if chunk.document_id in pool_docs or chunk.document_id in seen:
            continue
        seen.add(chunk.document_id)
        new_docs.append(chunk)
    promoted = new_docs[:cap]
    if not promoted:
        return chunks, []

    promoted_ids = {getattr(chunk, "chunk_id", None) for chunk in promoted}
    remaining = [chunk for chunk in chunks
                 if getattr(chunk, "chunk_id", None) not in promoted_ids]
    cut = pool_size - len(promoted)
    ordered = [*remaining[:cut], *promoted, *remaining[cut:]]
    return ordered, [getattr(chunk, "chunk_id", "") for chunk in promoted]


def _chunk_id(chunk):
    return getattr(chunk, "chunk_id", "")


def _evidence_weight(plan, chunk) -> int:
    text = _chunk_text(chunk)
    return len(aspect_hits(plan, text)) + (2 if date_identity(plan, text) else 0)


def signal_score(signals: ChunkSignals) -> float:
    """Full additive signal value. Diagnostics only.

    The ordering boost deliberately does NOT use this sum: identifier matches
    are few and unambiguous, while seller/evidence matches apply to large
    blocks of candidates and caused a measured recall regression when used
    for ordering (E014: the org-wide escalation policy was evicted by the
    mentioned seller's own documents). Weights live in config.
    """
    return (
        (config.QUERY_AWARE_IDENTIFIER_BOOST if signals.identifier_match else 0.0)
        + (config.QUERY_AWARE_ENTITY_BOOST if signals.entity_match else 0.0)
        + (config.QUERY_AWARE_SELLER_BOOST if signals.seller_match else 0.0)
        + (config.QUERY_AWARE_EVIDENCE_TYPE_BOOST
           if signals.evidence_type_match else 0.0))


def boosted_order(chunks: list, plan) -> tuple[list, list[ChunkSignals]]:
    """Stable reorder of authorized candidates: identifier matches first.

    Only structured document identifiers drive ordering (see
    `structured_identifiers`). Ties keep fusion order, and the relative order
    of unmatched candidates is preserved exactly, so without a structured
    identifier match the pool is identical to the baseline. Returns
    (ordered_chunks, signals) so the caller can log what matched.
    """
    signals = [evaluate_signals(chunk, plan) for chunk in chunks]
    structured = structured_identifiers(plan)
    if plan is None or not structured:
        return chunks, signals

    # A chunk is boosted only when its text matches a STRUCTURED identifier.
    # ChunkSignals.identifier_match cannot be used here: it also fires on a
    # bare seller code (S001) that co-occurs with a structured id in the same
    # query, which would re-promote the seller's whole block ahead of org-wide
    # documents - exactly the E014 regression this ordering exists to avoid.
    boosted = [
        any(identifier_in_text(identifier, _chunk_text(chunk))
            for identifier in structured)
        for chunk in chunks
    ]
    if not any(boosted):
        return chunks, signals
    ordered = sorted(
        enumerate(chunks),
        key=lambda pair: (0 if boosted[pair[0]] else 1, pair[0]))
    return [chunk for _, chunk in ordered], signals


def augment_fused(fused: list, plan, corpus: list[dict] | None = None
                  ) -> tuple[list, int]:
    """Add exact-identifier candidates missing from the fused pool.

    Returns (combined_candidates, added_count). The caller MUST pass the
    combined list through the normal authorization re-check; nothing here is
    pre-authorized.
    """
    if plan is None or not plan.identifiers:
        return fused, 0
    existing = {getattr(chunk, "chunk_id", None) for chunk in fused}
    extra = identifier_candidates(plan, exclude_chunk_ids=existing,
                                  corpus=corpus)
    cap = max(0, config.QUERY_AWARE_IDENTIFIER_CANDIDATES)
    extra = extra[:cap]
    if not extra:
        return fused, 0
    return [*fused, *extra], len(extra)
