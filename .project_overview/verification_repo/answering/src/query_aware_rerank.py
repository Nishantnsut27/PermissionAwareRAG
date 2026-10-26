"""Phase 7 query-aware reranking signals.

Phase 6 established that the authorized candidate pool is already at ~98.9% of
its mathematical precision ceiling, so the remaining precision problem is not
"which candidates are retrieved" but "which of them actually ANSWER the query".
This module supplies that judgment as a small, bounded adjustment on top of
Jina's relevance score.

Design constraints, each taken from a measured Phase 6 failure:

1. Bounded. The total adjustment can never exceed `weights.total_weight`
   (config `RERANK_PLAN_WEIGHT`, default 0.06) no matter how many signals fire.
   Jina separates on-topic (>0.5) from off-topic (<0.05) passages by an order
   of magnitude more than that, so a metadata match can nudge ties but can
   never overturn a decisive relevance judgment. This is the structural answer
   to the prohibition on blind fixed boosts.

2. Answer-bearing dominates membership. Required-aspect coverage carries the
   largest weight because it is the only signal that measures whether a passage
   contains the requested information, rather than whether it is merely about
   the right subject. Seller match carries the smallest weight: the Phase 6
   E014 regression happened because every document of a mentioned seller
   matched as a block and evicted the org-wide policy the query actually
   needed.

3. Non-negative. A passage is never penalised, only promoted, so no candidate
   can be pushed out of the pool by lacking a signal.

4. Applicability-normalised. A signal only enters the denominator when the plan
   actually expresses that requirement (no required_aspects -> the aspect term
   is not scored at all, rather than scoring zero for everyone). Without this,
   an inapplicable signal silently rescales every candidate in the query.

5. Structured identifiers only for event identity. A bare seller code (S001)
   is not an event id; treating it as one is exactly what flooded the pool in
   Phase 6.

The reranker applies the *relevance* threshold to the raw Jina score, so the
SET of passages surviving the gate is identical to baseline and this module can
only ever change their ORDER. Filtering stays a Jina decision; Phase 7 reorders.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import config, query_aware

# Which record types plausibly carry the answer for a given information need.
# This is a semantic mapping between an intent and a document genre, not a set
# of per-query rules: it generalises to any query of that intent and contains
# no evaluation-specific knowledge. The two deliberately empty entries matter:
# `factual_lookup` is too generic to justify a genre preference, and
# `unauthorized_request` must never express one, so in both cases the intent
# term drops out of the score entirely instead of inventing a preference.
INTENT_EVIDENCE_AFFINITY: dict[str, frozenset[str]] = {
    "root_cause": frozenset({"incident_report", "support_ticket",
                             "call_transcript"}),
    "impact": frozenset({"incident_report", "project_status",
                         "support_ticket", "account_overview"}),
    "procedure": frozenset({"runbook", "policy"}),
    "policy": frozenset({"policy", "runbook", "decision_record"}),
    "status": frozenset({"project_status", "support_ticket",
                         "account_overview"}),
    "comparison": frozenset({"account_overview", "project_status",
                             "reference_data"}),
    "explanation": frozenset({"reference_data", "policy", "project_status"}),
    "factual_lookup": frozenset(),
    "unauthorized_request": frozenset(),
}

# A date/month agreement is weaker evidence of "same event" than an exact
# structured identifier, so it earns partial credit on the same axis.
DATE_IDENTITY_CREDIT = 0.5

MAX_HINT_CHARS = 180


@dataclass(frozen=True)
class CompatibilityWeights:
    """Relative importance of each signal, and the absolute bound on the sum.

    The four relative weights are normalised per query over the signals that
    actually apply, so `total_weight` alone bounds the adjustment.
    """
    aspect: float = 0.40
    event: float = 0.25
    intent: float = 0.15
    evidence: float = 0.12
    seller: float = 0.08
    total_weight: float = 0.06

    @classmethod
    def from_config(cls) -> "CompatibilityWeights":
        return cls(
            aspect=config.RERANK_PLAN_ASPECT_WEIGHT,
            event=config.RERANK_PLAN_EVENT_WEIGHT,
            intent=config.RERANK_PLAN_INTENT_WEIGHT,
            evidence=config.RERANK_PLAN_EVIDENCE_WEIGHT,
            seller=config.RERANK_PLAN_SELLER_WEIGHT,
            total_weight=config.RERANK_PLAN_WEIGHT,
        )

    def only(self, *names: str) -> "CompatibilityWeights":
        """An ablation view keeping only the named signals.

        Used by the evaluation harness to isolate one signal at a time; the
        absolute bound is deliberately preserved so arms differ in WHICH
        signal fires, not in how much a match is worth.
        """
        kept = set(names)
        return CompatibilityWeights(
            aspect=self.aspect if "aspect" in kept else 0.0,
            event=self.event if "event" in kept else 0.0,
            intent=self.intent if "intent" in kept else 0.0,
            evidence=self.evidence if "evidence" in kept else 0.0,
            seller=self.seller if "seller" in kept else 0.0,
            total_weight=self.total_weight,
        )


# The shipped tuning lives in config so it can be changed without a code edit;
# the class defaults above are the neutral "all signals equal" starting point
# used by the ablation harness and by tests that exercise the mechanism itself.
DEFAULT_WEIGHTS = CompatibilityWeights()


def _evidence_type_of(chunk) -> str | None:
    metadata = getattr(chunk, "metadata", None) or {}
    return query_aware.DOCUMENT_TYPE_TO_EVIDENCE.get(
        metadata.get("document_type"))


def _structured_event_credit(plan, text: str) -> tuple[float, bool]:
    """Event-identity credit and whether the signal applies at all.

    Exact structured identifiers are decisive; a shared explicit month is
    partial credit. Bare seller codes are excluded by `structured_identifiers`,
    which is what keeps this from re-creating the Phase 6 seller flood.
    """
    structured = query_aware.structured_identifiers(plan)
    query_dates = query_aware.date_tokens(plan.original_query)
    applicable = bool(structured or query_dates)
    if not applicable:
        return 0.0, False
    if any(query_aware.identifier_in_text(identifier, text)
           for identifier in structured):
        return 1.0, True
    if query_dates and (query_dates & query_aware.date_tokens(text)):
        return DATE_IDENTITY_CREDIT, True
    return 0.0, True


def compatibility(chunk, plan,
                  weights: CompatibilityWeights | None = None) -> float:
    """How well one candidate matches what the query actually asked for.

    Returns a value in [0, 1]. Only signals the plan actually expresses are
    scored, so the result is a coverage fraction over the applicable
    requirements rather than a count of coincidental metadata matches.
    """
    if plan is None:
        return 0.0
    weights = weights or CompatibilityWeights.from_config()
    text = getattr(chunk, "text", "") or ""

    earned = 0.0
    available = 0.0

    # Answer-bearing evidence: what fraction of the requested aspects does this
    # passage actually carry lexical evidence for. This is the signal that
    # separates "about the incident" from "states the cause of the incident".
    if weights.aspect > 0 and plan.required_aspects:
        hits = query_aware.aspect_hits(plan, text)
        earned += weights.aspect * (len(hits) / len(plan.required_aspects))
        available += weights.aspect

    if weights.event > 0:
        credit, applicable = _structured_event_credit(plan, text)
        if applicable:
            earned += weights.event * credit
            available += weights.event

    evidence_type = _evidence_type_of(chunk)

    if weights.intent > 0:
        affinity = INTENT_EVIDENCE_AFFINITY.get(plan.intent, frozenset())
        if affinity:
            available += weights.intent
            if evidence_type in affinity:
                earned += weights.intent

    if weights.evidence > 0 and plan.evidence_types:
        available += weights.evidence
        if evidence_type and evidence_type in set(plan.evidence_types):
            earned += weights.evidence

    if weights.seller > 0 and plan.seller_scope:
        available += weights.seller
        metadata = getattr(chunk, "metadata", None) or {}
        seller_id = metadata.get("seller_id")
        if (seller_id and seller_id != "Organization-wide"
                and seller_id in set(plan.seller_scope)):
            earned += weights.seller

    if available <= 0:
        return 0.0
    return earned / available


def adjust(score: float, chunk, plan,
           weights: CompatibilityWeights | None = None) -> float:
    """Jina relevance plus the bounded plan-compatibility term.

    Monotone and non-negative: the adjustment can promote a passage but never
    demote one, so no candidate is lost by lacking a signal.
    """
    weights = weights or CompatibilityWeights.from_config()
    if plan is None or weights.total_weight <= 0:
        return score
    return score + weights.total_weight * compatibility(chunk, plan, weights)


def _aspect_phrase(plan) -> str:
    return ", ".join(aspect.replace("_", " ") for aspect in plan.required_aspects)


def _evidence_phrase(plan) -> str:
    return ", ".join(evidence.replace("_", " ")
                     for evidence in plan.evidence_types[:2])


def plan_aware_query(query: str, plan) -> str:
    """The original query plus a compact statement of the information need.

    The user's own wording is never replaced or reordered - it stays the whole
    prefix - because it is the only text guaranteed to describe the ask. The
    suffix names the aspects the answer must cover and the record type most
    likely to carry it, which is what turns "documents about this incident"
    into "documents stating the cause of this incident".

    Nothing derived from identity, scope or clearance is included: every term
    here comes from the query text itself, so the external reranker learns
    nothing about who is asking or what they may see.
    """
    if plan is None or not config.RERANK_PLAN_QUERY_HINTS:
        return query
    hints: list[str] = []
    if plan.required_aspects:
        hints.append(f"needs: {_aspect_phrase(plan)}")
    if plan.evidence_types:
        hints.append(f"prefer: {_evidence_phrase(plan)}")
    if not hints:
        return query
    suffix = f" ({'; '.join(hints)})"[:MAX_HINT_CHARS]
    return f"{query}{suffix}"
