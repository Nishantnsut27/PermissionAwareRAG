"""Document-aware selection of reranked chunks.

Why this stage exists
---------------------
Reranking returns a flat, score-ordered list of chunks. Taking the top N of
that list spends a *chunk* budget, but everything downstream counts
*documents*: the citation list, the reader's attention, and Recall/Precision@K
in Phase 7 all operate on distinct document ids. Because the corpus has no
single-chunk documents (every document is 2-8 chunks), a flat top-N almost
always resolved to N different documents contributing one passage each. That is
the worst of both trades - it maximises the number of documents the model must
reconcile while minimising the evidence it gets from any one of them.

This module converts the flat list into an explicit breadth/depth decision:

  1. Relative pruning. An absolute floor cannot distinguish "weak but the best
     thing available" from "weak noise sitting next to a clear winner". Both
     chunks and documents are therefore also cut relative to the strongest
     score for the current query, so the context narrows when the evidence is
     decisive and stays wide when it genuinely is not.
  2. Document scoring. A document is ranked by its best passage plus a damped
     contribution from its other surviving passages, so corroboration counts
     without letting several mediocre passages outrank one decisive passage.
  3. Bounded depth. Within a kept document several passages are allowed, and
     they are restored to reading order so the model sees the document as prose
     rather than as score-ordered fragments.

Security note: this stage only ever removes chunks and only ever reorders
chunks it was given. It cannot introduce a passage, and it runs before the
final authorization gate in `service._prepare`, so it can never widen access.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from . import config

Scored = tuple[object, "float | None"]


@dataclass
class SelectionStats:
    """What the selector did, for logging and evaluation diagnostics."""
    considered_chunks: int = 0
    considered_documents: int = 0
    selected_chunks: int = 0
    selected_documents: int = 0
    dropped_by_relative_cut: int = 0
    dropped_by_document_cap: int = 0
    dropped_by_depth_cap: int = 0
    best_score: float | None = None
    weakest_selected_score: float | None = None
    scored: bool = True
    # Provenance of the scores this stage acted on. Without these a run whose
    # reranker was rate limited is indistinguishable from a healthy one, and
    # every relative cut below was skipped.
    rerank_applied: bool = True
    rerank_attempts: int = 0
    rerank_error: str | None = None

    def to_dict(self) -> dict:
        return {
            "considered_chunks": self.considered_chunks,
            "considered_documents": self.considered_documents,
            "selected_chunks": self.selected_chunks,
            "selected_documents": self.selected_documents,
            "dropped_by_relative_cut": self.dropped_by_relative_cut,
            "dropped_by_document_cap": self.dropped_by_document_cap,
            "dropped_by_depth_cap": self.dropped_by_depth_cap,
            "best_score": (None if self.best_score is None
                           else round(self.best_score, 4)),
            "weakest_selected_score": (
                None if self.weakest_selected_score is None
                else round(self.weakest_selected_score, 4)),
            "scored": self.scored,
            "rerank_applied": self.rerank_applied,
            "rerank_attempts": self.rerank_attempts,
            "rerank_error": self.rerank_error,
        }


@dataclass
class _Document:
    document_id: str
    chunks: list[Scored] = field(default_factory=list)
    best: float | None = None
    order: int = 0

    def score(self, support_weight: float) -> float | None:
        """Best passage plus a damped sum of the remaining passages.

        Returns None when the reranker did not run, in which case fusion order
        is the only available signal and must be preserved.
        """
        values = sorted((s for _, s in self.chunks if s is not None),
                        reverse=True)
        if not values:
            return None
        return values[0] + support_weight * sum(values[1:])


def _document_id(chunk) -> str:
    metadata = getattr(chunk, "metadata", None) or {}
    return getattr(chunk, "document_id", None) or metadata.get(
        "document_id") or "unknown"


def _reading_key(pair: Scored) -> tuple:
    """Order passages within a document as the document reads."""
    chunk, _ = pair
    metadata = getattr(chunk, "metadata", None) or {}
    index = getattr(chunk, "chunk_index", None)
    if index is None:
        index = metadata.get("chunk_index")
    if not isinstance(index, int):
        index = 0
    return (index, str(getattr(chunk, "chunk_id", "")))


def select(items: list[Scored],
           cfg: config.AnsweringConfig | None = None,
           outcome: object | None = None,
           ) -> tuple[list[Scored], SelectionStats]:
    """Pick the passages that go into the context.

    `items` must be ordered best-first, as `reranker.rerank` returns them.
    `outcome` is the originating `RerankOutcome`, recorded so a context built
    from unranked fusion order is never mistaken for a ranked one.

    Returns (selected, stats) where `selected` is grouped by document, the
    documents ordered by aggregate relevance and the passages inside each
    document ordered for reading.
    """
    cfg = cfg or config.DEFAULT_ANSWERING
    stats = SelectionStats(considered_chunks=len(items))
    if outcome is not None:
        stats.rerank_applied = bool(getattr(outcome, "applied", True))
        stats.rerank_attempts = int(getattr(outcome, "attempts", 0) or 0)
        stats.rerank_error = getattr(outcome, "error", None)
    if not items:
        return [], stats

    scores = [s for _, s in items if s is not None]
    stats.scored = bool(scores)
    stats.best_score = max(scores) if scores else None
    floor = (stats.best_score * cfg.rerank_relative_threshold
             if stats.scored and cfg.rerank_relative_threshold > 0 else None)

    # 1. Group every candidate into its document, tracking the best passage.
    #    First-seen order (already best-first) is the score-free tie-break.
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
    stats.considered_documents = len(documents)

    ranked = sorted(
        documents.values(),
        key=lambda d: (d.score(cfg.document_support_weight) is None,
                       -(d.score(cfg.document_support_weight) or 0.0),
                       d.order),
    )

    # 2. Breadth. A document enters on the strength of its BEST passage relative
    #    to the best passage overall. The support weight orders documents but is
    #    kept out of this test: otherwise the top document's extra passages
    #    inflate the floor and evict a genuinely relevant single-passage
    #    document scoring just under the cluster. Skipped without scores, so a
    #    reranker outage degrades relevance, never recall.
    if floor is not None:
        surviving = [d for d in ranked if d.best is None or d.best >= floor]
        if surviving:
            stats.dropped_by_relative_cut = len(ranked) - len(surviving)
            ranked = surviving

    # 3. Hard breadth cap as a backstop on a flat score distribution.
    if cfg.max_documents > 0 and len(ranked) > cfg.max_documents:
        stats.dropped_by_document_cap = len(ranked) - cfg.max_documents
        ranked = ranked[:cfg.max_documents]

    # 4. Depth. Within a kept document, pack its strongest passages up to the
    #    cap, but never pad with a passage weaker than the same relative floor:
    #    a document earning a slot on one strong passage should not drag in its
    #    unrelated tail. Passages are then restored to reading order.
    selected: list[Scored] = []
    for document in ranked:
        passages = sorted(
            document.chunks,
            key=lambda pair: (pair[1] is None, -(pair[1] or 0.0)),
        )
        keep: list[Scored] = []
        for chunk, score in passages:
            if len(keep) >= cfg.max_chunks_per_document > 0:
                stats.dropped_by_depth_cap += 1
                continue
            # The first passage always stays; it is why the document was kept.
            if keep and floor is not None and score is not None and score < floor:
                stats.dropped_by_depth_cap += 1
                continue
            keep.append((chunk, score))
        selected.extend(sorted(keep, key=_reading_key))

    stats.selected_chunks = len(selected)
    stats.selected_documents = len(ranked)
    chosen = [s for _, s in selected if s is not None]
    stats.weakest_selected_score = min(chosen) if chosen else None
    return selected, stats
