"""Phase 5 query understanding: OpenJEV judgments + deterministic extraction.

Given the raw query text, produce a validated QueryPlan. The plan is a
retrieval/evaluation hint only: it is never an authorization object, and a
failed or low-quality understanding call degrades to a safe deterministic plan
rather than fabricated semantics.

Deterministic extraction is preferred wherever it is more reliable than a
model judgment. Exact identifiers (INC-2026-0519, TKT-2026-0805-301,
PRJ-S001-003, POL-OPS-006, S001) are regex/token matches, never JEV output.
Seller names (Aurelia Home Decor, ...) come from the organization reference
corpus and are mapped to seller ids deterministically. OpenJEV contributes the
semantic judgments: intent, evidence types, aspects, multi-part and complexity.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

from . import config, openjev_client
from .openjev_client import OpenJEVError, Question

logger = logging.getLogger(__name__)

INTENTS = (
    "factual_lookup",
    "root_cause",
    "procedure",
    "comparison",
    "status",
    "policy",
    "impact",
    "explanation",
    "unauthorized_request",
)

EVIDENCE_TYPES = (
    "support_ticket",
    "incident_report",
    "project_status",
    "runbook",
    "policy",
    "call_transcript",
    "account_overview",
    "decision_record",
    "reference_data",
)

COMPLEXITY_LEVELS = ("simple", "moderate", "complex")

FALLBACK_INTENT = "factual_lookup"

ASPECT_OPTIONS = (
    ("cause", "why the event happened, its root cause"),
    ("containment_actions", "containment, mitigation or remediation steps taken"),
    ("financial_impact", "monetary amounts, settlements, refunds, shortfalls"),
    ("timeline", "dates, durations, when events occurred"),
    ("status", "current state or progress of a request, incident or project"),
    ("policy_requirement", "what a policy or procedure requires"),
    ("comparison", "similarities or differences between records"),
    ("quantities", "counts, volumes, sizes, numbers of items"),
    ("responsibility", "who approved, decided, or is accountable"),
)

SELLER_NAMES = {
    "s001": "S001",
    "s002": "S002",
    "s003": "S003",
    "s004": "S004",
    "aurelia home decor": "S001",
    "aurelia": "S001",
    "bluepeak electronics": "S002",
    "bluepeak": "S002",
    "greencart organics": "S003",
    "greencart": "S003",
    "strideone footwear": "S004",
    "strideone": "S004",
    "nexora commerce": "ORG",
    "nexora": "ORG",
}

IDENTIFIER_PATTERNS = (
    re.compile(r"\bPOL-[A-Z]{2,3}-\d{2,3}\b", re.IGNORECASE),
    re.compile(r"\bOPS-RB-\d{3}\b", re.IGNORECASE),
    re.compile(r"\bPRJ-S\d{3}-\d{3}\b", re.IGNORECASE),
    re.compile(r"\bTKT-\d{4}-\d{4}-\d{3}\b", re.IGNORECASE),
    re.compile(r"\bCALL-\d{4}-\d{4}-\d{3}\b", re.IGNORECASE),
    re.compile(r"\bINC-\d{4}-\d{4}\b", re.IGNORECASE),
    re.compile(r"\bS\d{3}-ACC-\d{3}\b", re.IGNORECASE),
    re.compile(r"\bSC-\d{3}\b", re.IGNORECASE),
)

BARE_SELLER_PATTERN = re.compile(r"\bS\d{3}\b", re.IGNORECASE)

MIN_CONFIDENCE = 0.55

_INJECTION_PATTERN = re.compile(
    r"(ignore|forget|override|bypass|escalate|elevate|administrator mode|"
    r"system override|previous assistant|update my permissions|"
    r"forget my normal permissions|developer debug mode|safety filters)",
    re.IGNORECASE)


def injection_shaped(text: str) -> bool:
    """Deterministic authority-override pattern detection.

    Used in two places: as the intent label when JEV is unavailable, and as a
    deterministic override on top of the JEV intent. It is a retrieval and
    evaluation hint only; enforcement never depends on it.
    """
    return bool(_INJECTION_PATTERN.search(text or ""))


class QueryPlanError(ValueError):
    """A plan could not be built from valid-looking inputs."""


@dataclass(frozen=True)
class QueryPlan:
    """Validated, typed representation of what a query is asking.

    Phase 6 query-aware retrieval (`query_aware.py`) consumes identifiers,
    entities, seller_scope, evidence_types, required_aspects and
    original_query. intent, is_multi_part, complexity and confidence are
    currently diagnostics only: they are produced, logged and evaluated, but no
    production retrieval path branches on them today (they are reserved for
    later phases, e.g. Phase 9 adaptive behaviour). The plan carries no
    authorization semantics: seller_scope is what the query mentions, not what
    the user may access.
    """
    original_query: str
    intent: str
    entities: tuple[str, ...] = ()
    seller_scope: tuple[str, ...] = ()
    identifiers: tuple[str, ...] = ()
    required_aspects: tuple[str, ...] = ()
    evidence_types: tuple[str, ...] = ()
    is_multi_part: bool = False
    complexity: str = "simple"
    confidence: float | None = None
    fallback_used: bool = False
    jev_status: str = "not_called"
    jev_latency_ms: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.original_query, str) or not self.original_query.strip():
            raise QueryPlanError("original_query must be a non-empty string")
        if self.intent not in INTENTS:
            raise QueryPlanError(f"invalid intent: {self.intent!r}")
        if self.complexity not in COMPLEXITY_LEVELS:
            raise QueryPlanError(f"invalid complexity: {self.complexity!r}")
        if self.jev_status not in ("not_called", "ok", "unavailable", "disabled"):
            raise QueryPlanError(f"invalid jev_status: {self.jev_status!r}")
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise QueryPlanError("confidence must be within 0..1")
        for name in ("entities", "seller_scope", "identifiers",
                     "required_aspects", "evidence_types"):
            value = getattr(self, name)
            if not isinstance(value, tuple) or not all(
                    isinstance(item, str) and item for item in value):
                raise QueryPlanError(f"{name} must be a tuple of non-empty strings")

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "intent": self.intent,
            "entities": list(self.entities),
            "seller_scope": list(self.seller_scope),
            "identifiers": list(self.identifiers),
            "required_aspects": list(self.required_aspects),
            "evidence_types": list(self.evidence_types),
            "is_multi_part": self.is_multi_part,
            "complexity": self.complexity,
            "confidence": self.confidence,
            "fallback_used": self.fallback_used,
            "jev_status": self.jev_status,
            "jev_latency_ms": self.jev_latency_ms,
        }


def _ordered_unique(values) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            ordered.append(value)
    return tuple(ordered)


def extract_identifiers(text: str) -> tuple[str, ...]:
    """Exact, boundary-aware identifier extraction. Never invents ids.

    Longer patterns run first and claim their spans, so PRJ-S001-003 yields
    one identifier and not PRJ plus a stray S001. A bare seller code (S001)
    is accepted only when it is not hyphen-glued into a longer code, and word
    boundaries on both sides stop INC-2026-0519 matching inside
    INC-2026-05190. Results are ordered by position in the text.
    """
    if not isinstance(text, str) or not text.strip():
        return ()
    spans: list[tuple[int, int, str]] = []
    for pattern in IDENTIFIER_PATTERNS:
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(),
                          match.group(0).upper()))
    for match in BARE_SELLER_PATTERN.finditer(text):
        start, end = match.start(), match.end()
        before = text[start - 1:start]
        after = text[end:end + 1]
        if before == "-" or after == "-":
            continue
        if any(start < s_end and end > s_start
               for s_start, s_end, _ in spans):
            continue
        spans.append((start, end, match.group(0).upper()))
    spans.sort(key=lambda span: span[0])
    return _ordered_unique(value for _, _, value in spans)


def extract_sellers(text: str) -> tuple[str, ...]:
    """Deterministic seller id / seller name / organization mentions."""
    if not isinstance(text, str) or not text.strip():
        return ()
    lowered = text.lower()
    found: list[str] = []
    for name, seller_id in SELLER_NAMES.items():
        if re.search(r"\b" + re.escape(name) + r"\b", lowered):
            found.append(seller_id)
    return _ordered_unique(found)


def extract_entities(text: str) -> tuple[str, ...]:
    """Names and ids a query explicitly refers to, in order of appearance.

    Identifier-shaped tokens and seller names/ids only. The planner does not
    guess at people, tickets or incidents that were not named.
    """
    if not isinstance(text, str) or not text.strip():
        return ()
    lowered = text.lower()
    mentions: list[tuple[int, str]] = []
    for match in BARE_SELLER_PATTERN.finditer(text):
        mentions.append((match.start(), match.group(0).upper()))
    for name, seller_id in SELLER_NAMES.items():
        for match in re.finditer(r"\b" + re.escape(name) + r"\b", lowered):
            mentions.append((match.start(), seller_id))
    for pattern in (*IDENTIFIER_PATTERNS, BARE_SELLER_PATTERN):
        for match in pattern.finditer(text):
            mentions.append((match.start(), match.group(0).upper()))
    mentions.sort(key=lambda mention: mention[0])
    return _ordered_unique(value for _, value in mentions)


def _seller_label(seller_ids) -> str:
    labels = {
        "S001": "Aurelia Home Decor (S001)",
        "S002": "BluePeak Electronics (S002)",
        "S003": "GreenCart Organics (S003)",
        "S004": "StrideOne Footwear (S004)",
        "ORG": "the Nexora Commerce platform organization itself",
    }
    return ", ".join(labels.get(s, s) for s in seller_ids) or "none"


def _question(kind: str, instructions: str, criteria=None) -> Question:
    return Question(kind=kind, instructions=instructions, criteria=criteria)


def _questions(seller_ids: tuple[str, ...]) -> dict[str, Question]:
    seller_names = _seller_label(seller_ids)
    return {
        "intent": _question(
            "choice",
            "What is the primary information need of this enterprise query? "
            "Pick exactly one intent.",
            criteria=[
                ("factual_lookup",
                 "asks for a specific fact, figure, date or record detail"),
                ("root_cause", "asks why something happened"),
                ("procedure", "asks for steps to follow or a process to run"),
                ("comparison",
                 "asks to compare two or more records, sellers or outcomes"),
                ("status",
                 "asks for the current state or progress of a request, "
                 "incident, project or account"),
                ("policy",
                 "asks what a policy, runbook or procedure requires or says"),
                ("impact",
                 "asks about the effect, damage or cost of an event"),
                ("explanation",
                 "asks to explain how something works or relates to something"),
                ("unauthorized_request",
                 "asks to ignore or override previous instructions or "
                 "permissions, claims a special authority or mode such as "
                 "administrator or developer access, or demands restricted "
                 "records be dumped or printed verbatim"),
            ]),
        "evidence": _question(
            "choice",
            "Which single evidence source is most likely to contain the "
            "primary answer to this enterprise query?",
            criteria=[
                ("support_ticket",
                 "a customer support ticket, its resolution or correspondence"),
                ("incident_report",
                 "an incident or security incident report"),
                ("project_status",
                 "a project document, status report or implementation plan"),
                ("runbook",
                 "a runbook, operational procedure or checklist"),
                ("policy",
                 "a company policy document"),
                ("call_transcript",
                 "a recorded customer call transcript"),
                ("account_overview",
                 "a seller account overview or account profile"),
                ("decision_record",
                 "a decision record or approval note"),
                ("reference_data",
                 "an organization reference, catalogue or general reference "
                 "material, when no specific record type fits"),
            ]),
        "aspects": _question(
            "score",
            "How many distinct aspects does the query explicitly ask about? "
            "A query that asks both why something happened AND what actions "
            "were taken asks about two aspects; a query that asks only why it "
            "happened asks about one.",
            criteria=[
                "exactly one distinct aspect",
                "two distinct aspects",
                "three distinct aspects",
                "four or more distinct aspects",
            ]),
        "multi_part": _question(
            "noul",
            "Does this enterprise query contain two or more independently "
            "answerable requirements, where each requirement could be answered "
            "from a different document on its own? Simple follow-ups and "
            "single-fact questions are not multi-part."),
        "complexity": _question(
            "score",
            "How complex is the retrieval and reasoning task this enterprise "
            "query implies? Consider whether one document suffices, several "
            "records must be reconciled, or a comparison across records or "
            "sellers is demanded.",
            criteria=[
                "simple: one specific fact or one record answers it",
                "moderate: several related details or records are needed",
                "complex: cross-record or cross-seller comparison or "
                "broad synthesis is needed",
            ]),
        "seller_referenced": _question(
            "noul",
            f"The query mentions: {seller_names}. Does the query's information "
            "need depend on these specific sellers or organization, or are "
            "they incidental?"),
    }


def _validate_jev_answers(answers: dict) -> dict:
    """Check every judgment is the type we asked for, in range, and known."""
    checked: dict = {}
    for name in ("intent", "evidence", "aspects", "multi_part",
                 "complexity", "seller_referenced"):
        answer = answers.get(name)
        if answer is None:
            raise OpenJEVError(f"missing JEV answer for {name!r}")
    if answers["intent"].kind != "choice" or not answers["intent"].choice:
        raise OpenJEVError("intent answer is not a usable choice")
    if answers["intent"].choice not in INTENTS:
        raise OpenJEVError(f"unknown intent {answers['intent'].choice!r}")
    if answers["evidence"].kind != "choice" or not answers["evidence"].choice:
        raise OpenJEVError("evidence answer is not a usable choice")
    if answers["evidence"].choice not in EVIDENCE_TYPES:
        raise OpenJEVError(f"unknown evidence type {answers['evidence'].choice!r}")
    if answers["aspects"].kind != "score" or not (
            0 <= (answers["aspects"].score or 0) <= 3):
        raise OpenJEVError("aspects answer is not a score on the 1..4 rubric")
    if answers["multi_part"].kind != "noul" or answers["multi_part"].noul is None:
        raise OpenJEVError("multi_part answer is not a noul value")
    if answers["complexity"].kind != "score" or not (
            0 <= (answers["complexity"].score or 0) <= 2):
        raise OpenJEVError("complexity answer is not a score on the simple/moderate/complex rubric")
    if (answers["seller_referenced"].kind != "noul"
            or answers["seller_referenced"].noul is None):
        raise OpenJEVError("seller_referenced answer is not a noul value")
    for name, answer in answers.items():
        checked[name] = answer
    return checked


def _aspects_from_score(score: float, query_text: str) -> tuple[str, ...]:
    # Score positions are 0-indexed: level 0 is "one aspect".
    count = int(round(score)) + 1
    lowered = query_text.lower()
    detected = [name for name, _ in ASPECT_OPTIONS
                if re.search(rf"\b{re.escape(name.replace('_', ' '))}\b", lowered)]
    if not detected:
        if "why" in lowered or "cause" in lowered:
            detected.append("cause")
        if "impact" in lowered or "how much" in lowered or "how large" in lowered:
            detected.append("financial_impact")
        if "status" in lowered or "progress" in lowered:
            detected.append("status")
    if not detected:
        return ()
    detected = detected[:max(count, 1)]
    return tuple(detected)


def plan_query(query: str) -> QueryPlan:
    """Build a QueryPlan from the raw query text.

    Combines deterministic extraction with OpenJEV semantic judgments. Any JEV
    failure (timeout, rate limit, auth, malformed response) falls back to a
    deterministic-only plan that preserves the identifiers, entities and
    simple keyword intent, and reports jev_status="unavailable".
    """
    if not isinstance(query, str) or not query.strip():
        raise QueryPlanError("query must be a non-empty string")
    started = time.monotonic()

    identifiers = extract_identifiers(query)
    entities = extract_entities(query)
    sellers = extract_sellers(query)
    lowered = query.strip().lower()
    if re.search(r"\b(organization|organisation|platform|company|nexora)\b", lowered):
        sellers = _ordered_unique([*sellers, "ORG"])

    if not config.OPENJEV_ENABLED:
        return _fallback_plan(query, identifiers, entities, sellers,
                              "disabled", started)

    try:
        result = openjev_client.system_one(
            state={"query": query.strip()},
            questions=_questions(sellers))
        answers = _validate_jev_answers(result.answers)
    except OpenJEVError as exc:
        logger.warning("OpenJEV unavailable, using deterministic fallback: %s", exc)
        return _fallback_plan(query, identifiers, entities, sellers,
                              "unavailable", started, error=str(exc))

    confidence_values = [
        a.confidence for a in answers.values() if a.confidence is not None
    ]
    confidence = (round(sum(confidence_values) / len(confidence_values), 4)
                  if confidence_values else None)

    intent = answers["intent"].choice
    evidence_types: list[str] = [answers["evidence"].choice]
    for name, probability in answers["evidence"].probabilities.items():
        if name in EVIDENCE_TYPES and name != evidence_types[0] \
                and probability >= 0.35:
            evidence_types.append(name)
    evidence_types = _ordered_unique(evidence_types)

    # Score 0.0 is a real value on this rubric ("exactly one aspect"), so it
    # must not be coalesced away by `or`: `0.0 or 1` -> 1 would silently turn a
    # one-aspect judgment into a two-aspect one. The value is already validated
    # to be a non-None float in _validate_jev_answers.
    aspects_score = answers["aspects"].score
    aspects = _aspects_from_score(
        aspects_score if aspects_score is not None else 1, query)
    if not aspects:
        aspects = ("cause",) if intent == "root_cause" else \
            ("policy_requirement",) if intent == "policy" else \
            ("status",) if intent == "status" else ()
    multi_part = bool(answers["multi_part"].noul >= 0.5)
    # Score positions are 0-indexed: 0 -> simple, 1 -> moderate, 2 -> complex.
    complexity_index = int(round(answers["complexity"].score or 0))
    complexity = COMPLEXITY_LEVELS[max(0, min(2, complexity_index))]

    if intent == "unauthorized_request":
        multi_part = False
        complexity = "simple"
    elif injection_shaped(query):
        intent = "unauthorized_request"
        multi_part = False

    return QueryPlan(
        original_query=query,
        intent=intent,
        entities=entities,
        seller_scope=sellers,
        identifiers=identifiers,
        required_aspects=aspects,
        evidence_types=evidence_types,
        is_multi_part=multi_part,
        complexity=complexity,
        confidence=confidence,
        fallback_used=False,
        jev_status="ok",
        jev_latency_ms=result.latency_ms,
    )


def _fallback_intent(query: str) -> str:
    """Conservative keyword intent: label only unambiguous signals."""
    lowered = query.lower()
    if injection_shaped(query):
        return "unauthorized_request"
    if any(word in lowered for word in ("why", "root cause", "caused")):
        return "root_cause"
    if any(word in lowered for word in ("policy", "runbook", "procedure",
                                        "must happen", "require")):
        return "policy"
    if any(word in lowered for word in ("compare", "versus", " vs ", "difference")):
        return "comparison"
    if any(word in lowered for word in ("status", "current state", "progress")):
        return "status"
    if any(word in lowered for word in ("steps", "how do i", "rollback")):
        return "procedure"
    if any(word in lowered for word in ("impact", "cost", "how much")):
        return "impact"
    return FALLBACK_INTENT


def _fallback_plan(query: str, identifiers, entities, sellers, status: str,
                   started: float, error: str | None = None) -> QueryPlan:
    """Deterministic-only plan. Never fabricates JEV semantics.

    The keyword intent mapping is deliberately conservative: it only labels
    the intent when the signal is unambiguous, otherwise it stays at the safe
    generic default. required_aspects stays empty rather than guessed.
    """
    return QueryPlan(
        original_query=query,
        intent=_fallback_intent(query),
        entities=entities,
        seller_scope=sellers,
        identifiers=identifiers,
        required_aspects=(),
        evidence_types=(),
        is_multi_part=False,
        complexity="simple",
        confidence=None,
        fallback_used=True,
        jev_status=status,
        jev_latency_ms=int((time.monotonic() - started) * 1000),
    )


def log_plan(request_id: str, plan: QueryPlan) -> None:
    """Safe structured log: plan fields only, never keys or scope secrets."""
    logger.info(
        "request=%s query_plan=%s jev_status=%s jev_confidence=%s "
        "jev_fallback=%s jev_latency_ms=%d",
        request_id,
        plan.to_dict(),
        plan.jev_status,
        plan.confidence,
        plan.fallback_used,
        plan.jev_latency_ms,
    )
