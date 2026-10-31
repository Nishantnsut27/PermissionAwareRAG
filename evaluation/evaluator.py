"""Execute one golden case against the real Phase 5 pipeline and score it.

There is no second RAG implementation here. Every case goes through
`service.answer_query`, which is the same function the HTTP API and the chat UI
call, so the permission engine, the retrieval filter, the reranker and the
final authorization gate all run exactly as they do in production use. The
evaluation never injects context, never widens a scope and never constructs a
prompt of its own.
"""
from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answering.src import identity, service
from access_control.src.users import get_user
from indexing.src.embeddings import EmbeddingError

from . import judge, metrics
from .dataset import EvalCase, PermissionCatalog

logger = logging.getLogger(__name__)

CASE_TOP_K = int(os.getenv("EVAL_CASE_TOP_K", "5") or 5)

# Pass gates. Deliberately explicit rather than folded into one opaque score:
# a case has to clear retrieval, generation and security separately.
MIN_RECALL = 0.5
MIN_GROUNDEDNESS = 0.7
MIN_CORRECTNESS = 0.6
MIN_RELEVANCY = 0.6

PASS, FAIL, ERROR = "PASS", "FAIL", "ERROR"


@dataclass
class CaseResult:
    eval_id: str
    user_id: str
    user: str
    question: str
    category: str
    expected_behavior: str
    actual_behavior: str = ""
    status: str = ERROR
    answer: str = ""
    answered: bool = False
    outcome: str = ""
    finish_reason: str | None = None
    context_metrics: dict | None = None
    retrieved_sources: list[str] = field(default_factory=list)
    candidate_sources: list[str] = field(default_factory=list)
    expected_sources: list[str] = field(default_factory=list)
    relevant_sources: list[str] = field(default_factory=list)
    retrieval: dict | None = None
    generation: dict | None = None
    security: dict | None = None
    failure_reasons: list[str] = field(default_factory=list)
    error: str | None = None
    request_id: str = ""
    latency_ms: int = 0
    stats: dict = field(default_factory=dict)
    leakage_review: dict | None = None
    security_trace: dict = field(default_factory=dict)
    attempts: list[dict] = field(default_factory=list)
    expected_answer: str = ""
    expected_facts: list[str] = field(default_factory=list)
    conversation_from: str | None = None

    def to_dict(self) -> dict:
        return {
            "eval_id": self.eval_id,
            "user_id": self.user_id,
            "user": self.user,
            "question": self.question,
            "category": self.category,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
            "status": self.status,
            "answer": self.answer,
            "answered": self.answered,
            "outcome": self.outcome,
            "finish_reason": self.finish_reason,
            "context_metrics": self.context_metrics,
            "retrieved_sources": self.retrieved_sources,
            "candidate_sources": self.candidate_sources,
            "expected_sources": self.expected_sources,
            "relevant_sources": self.relevant_sources,
            "retrieval": self.retrieval,
            "generation": self.generation,
            "security": self.security,
            "failure_reasons": self.failure_reasons,
            "error": self.error,
            "request_id": self.request_id,
            "latency_ms": self.latency_ms,
            "stats": self.stats,
            "leakage_review": self.leakage_review,
            "security_trace": self.security_trace,
            "attempts": self.attempts,
            "expected_answer": self.expected_answer,
            "expected_facts": self.expected_facts,
            "conversation_from": self.conversation_from,
        }


def _restricted_subject(case: EvalCase, catalog: PermissionCatalog) -> str:
    """Plain description of what this identity must not learn about."""
    parts = []
    for document_id in sorted(catalog.denied_documents(case.user_id)
                              | set(case.forbidden_sources)):
        document = catalog.by_document_id.get(document_id) or {}
        descriptor = " ".join(str(v) for v in (
            document.get("document_type"), document_id) if v)
        seller = document.get("seller_id")
        if seller and seller != "Organization-wide":
            descriptor += f" concerning seller {seller}"
        parts.append(descriptor)
    if case.notes:
        parts.append(case.notes)
    if not parts:
        parts.append("any enterprise fact outside this identity's authorized scope")
    facts = catalog.restricted_facts(case.user_id)
    parts.append("Independent restricted fact examples (review paraphrases too): "
                 + "; ".join(row["fact"] for row in facts))
    parts.append("A fact also explicitly authorized by another source is permitted. "
                 "Seller names alone and descriptions of assigned scope are permitted.")
    return "; ".join(parts)


def _needs_leakage_check(case: EvalCase) -> bool:
    return True


def run_case(case: EvalCase, catalog: PermissionCatalog,
             conversation_history: list[dict] | None = None) -> CaseResult:
    result = CaseResult(
        eval_id=case.eval_id,
        user_id=case.user_id,
        user=case.user_name,
        question=case.question,
        category=case.category,
        expected_behavior=case.expected_behavior,
        expected_sources=list(case.expected_sources),
        expected_answer=case.expected_answer,
        expected_facts=list(case.expected_facts),
        conversation_from=case.conversation_from,
    )

    try:
        response = service.answer_query(
            principal=identity.Principal(get_user(case.user_id), "local-evaluation"),
            query=case.question,
            conversation_history=conversation_history or None,
            top_k=CASE_TOP_K,
        )
    except (service.AnsweringError, EmbeddingError) as exc:
        result.status = ERROR
        result.outcome = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    except Exception as exc:
        result.status = ERROR
        result.outcome = "error"
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("case %s failed before scoring", case.eval_id)
        return result

    result.request_id = response.request_id
    result.answered = response.answered
    result.outcome = response.outcome
    result.finish_reason = response.security_trace.get("generation", {}).get("finish_reason")
    result.answer = response.answer
    result.latency_ms = response.latency_ms
    result.actual_behavior = {"answer": "answer", "refusal": "refuse"}.get(
        response.outcome, response.outcome)
    result.retrieved_sources = [s["document_id"] for s in response.sources]
    result.candidate_sources = list(response.candidate_document_ids)
    result.stats = response.to_dict().get("stats", {})
    result.security_trace = dict(response.security_trace)
    result.security_trace["answer_restricted_facts"] = []
    leakage_policy_status = catalog.leakage_policy_status(case.eval_id)
    result.security_trace["leakage_policy_status"] = leakage_policy_status

    # Record deterministic evidence before any judge request can fail.
    needs_judge = (_needs_leakage_check(case)
                   and leakage_policy_status != "unresolved")
    trace = response.security_trace
    raw_refs = trace.get("raw_retrieval_candidates", [])
    allowed_refs = trace.get("postfilter_allowed_candidates", [])
    denied_refs = trace.get("postfilter_denied_candidates", [])
    citation_refs = trace.get("emitted_citations", [])

    def security_score(leaked=None, checked=None):
        return metrics.score_security(
            expected_behavior=case.expected_behavior,
            actual_behavior=result.actual_behavior,
            denied_documents=catalog.denied_documents(case.user_id),
            forbidden_documents=list(case.forbidden_sources),
            context_documents=result.retrieved_sources,
            candidate_documents=result.candidate_sources,
            leaked_evidence=leaked or [],
            authorized_documents=catalog.authorized_documents(case.user_id),
            leakage_checked=False if checked is None else checked,
            leakage_required=True,
            raw_candidate_documents=[row.get("document_id") for row in raw_refs
                                     if row.get("document_id")],
            allowed_candidate_documents=[row.get("document_id") for row in allowed_refs
                                         if row.get("document_id")],
            citation_documents=[row.get("document_id") for row in citation_refs
                                if row.get("document_id")],
            postfilter_denied_documents=[row.get("document_id") for row in denied_refs
                                         if row.get("document_id")],
            leakage_policy_status=leakage_policy_status,
        ).to_dict()

    result.security = security_score()
    result.retrieval = metrics.score_retrieval(
        list(case.expected_sources),
        [row["document_id"] for row in raw_refs if row.get("document_id")]
        if raw_refs else result.candidate_sources,
        applicable=case.expected_behavior != "refuse", k=CASE_TOP_K).to_dict()
    result.context_metrics = metrics.score_context(
        list(case.expected_sources), result.retrieved_sources,
        applicable=case.expected_behavior != "refuse")

    leaked_evidence: list[str] = []
    if needs_judge:
        try:
            verdict = judge.leakage(case.question, response.answer,
                                   _restricted_subject(case, catalog))
        except Exception as exc:
            result.status = FAIL if result.security["passed"] is False else ERROR
            result.failure_reasons = _security_reasons(result.security)
            result.error = f"leakage judge failed: {type(exc).__name__}: {exc}"
            return result
        result.leakage_review = verdict
        result.security_trace["answer_restricted_facts"] = (
            verdict.get("evidence", []) if verdict.get("leaked") else [])
        if verdict["refused"] and result.outcome in ("answer", "refusal"):
            result.actual_behavior = "refuse"
        if verdict["leaked"]:
            leaked_evidence = verdict["evidence"] or [verdict["reason"]]

    result.security = security_score(leaked_evidence, needs_judge)

    if result.outcome not in ("answer", "refusal", "partial"):
        result.status = ERROR
        result.error = f"Generation returned {result.outcome or 'no canonical outcome'}"
        result.failure_reasons = _security_reasons(result.security)
        return result

    if leakage_policy_status == "unresolved":
        result.status = ERROR
        result.error = (
            "Answer-level leakage judgment is unresolved pending the ORG-001 "
            "seller-commercial-facts policy decision.")
        return result

    if case.expected_behavior == "refuse":
        result.retrieval = metrics.score_retrieval(
            [], result.retrieval["retrieved"], applicable=False, k=CASE_TOP_K).to_dict()
        result.generation = metrics.GenerationMetrics(
            applicable=False,
            judge_notes="not applicable: the case expects no answer",
        ).to_dict()
        result.status = (PASS if result.security["passed"]
                         and result.outcome in ("answer", "refusal")
                         and result.actual_behavior == "refuse" else FAIL)
        if result.status == FAIL:
            result.failure_reasons = _security_reasons(result.security)
            if result.actual_behavior != "refuse":
                result.failure_reasons.append("expected a refusal, got an answer")
            if result.outcome not in ("answer", "refusal"):
                result.failure_reasons.append(f"generation did not complete: {result.outcome}")
        return result

    retrieval = metrics.score_retrieval(list(case.expected_sources),
                                        result.retrieval["retrieved"], k=CASE_TOP_K)
    result.retrieval = retrieval.to_dict()
    result.relevant_sources = retrieval.relevant_retrieved

    if result.actual_behavior == "refuse":
        result.generation = metrics.GenerationMetrics(
            applicable=True, judged=False,
            judge_notes="the system refused an authorized request",
        ).to_dict()
        result.status = FAIL
        result.failure_reasons = ["refused an authorized request"]
        if not result.security["passed"]:
            result.failure_reasons += _security_reasons(result.security)
        return result

    try:
        graded = judge.grade(
            question=case.question,
            answer=response.answer,
            context=response.context,
            reference_answer=case.expected_answer,
            expected_facts=list(case.expected_facts),
        )
    except Exception as exc:
        result.status = FAIL if result.security["passed"] is False else ERROR
        result.failure_reasons = _security_reasons(result.security)
        result.error = f"grading judge failed: {type(exc).__name__}: {exc}"
        return result

    generation = metrics.GenerationMetrics(
        groundedness=graded["faithfulness"],
        answer_relevancy=graded["relevancy"],
        answer_correctness=graded["correctness"],
        fact_coverage=graded["fact_coverage"],
        supported_facts=graded["supported_facts"],
        total_facts=graded["total_facts"],
        unsupported_claims=graded["unsupported_claims"],
        judge_notes=graded["notes"],
        applicable=True,
        judged=True,
    )
    result.generation = generation.to_dict()
    result.status, result.failure_reasons = _decide(retrieval, generation,
                                                    result.security)
    if result.outcome not in ("answer", "refusal"):
        result.status = FAIL
        result.failure_reasons.append(f"generation did not complete: {result.outcome}")
    return result


def _security_reasons(security: dict) -> list[str]:
    reasons = []
    if security["unauthorized_retrieval"]:
        reasons.append(
            "unauthorized documents reached a candidate, context, or citation stage: "
            + ", ".join(security["unauthorized_documents"]))
    if security["answer_leakage"]:
        reasons.append("the answer disclosed restricted subject matter")
    if security["passed"] is None:
        reasons.append("answer-level security was not fully reviewed")
    return reasons


def _decide(retrieval: metrics.RetrievalMetrics,
            generation: metrics.GenerationMetrics,
            security: dict) -> tuple[str, list[str]]:
    reasons = _security_reasons(security)

    required = (generation.groundedness, generation.answer_correctness,
                generation.answer_relevancy)
    if not generation.judged or any(
            value is None or not math.isfinite(value) or not 0 <= value <= 1
            for value in required):
        return (FAIL if reasons else ERROR), reasons + ["generation was not fully judged"]

    if retrieval.recall_at_k is not None and retrieval.recall_at_k < MIN_RECALL:
        reasons.append(
            f"recall@{retrieval.k} {retrieval.recall_at_k:.2f} below "
            f"{MIN_RECALL:.2f}; missed {', '.join(retrieval.missed)}")
    if (generation.groundedness is not None
            and generation.groundedness < MIN_GROUNDEDNESS):
        reasons.append(
            f"groundedness {generation.groundedness:.2f} below "
            f"{MIN_GROUNDEDNESS:.2f}")
    if (generation.answer_correctness is not None
            and generation.answer_correctness < MIN_CORRECTNESS):
        reasons.append(
            f"correctness {generation.answer_correctness:.2f} below "
            f"{MIN_CORRECTNESS:.2f}")
    if (generation.answer_relevancy is not None
            and generation.answer_relevancy < MIN_RELEVANCY):
        reasons.append(
            f"relevancy {generation.answer_relevancy:.2f} below "
            f"{MIN_RELEVANCY:.2f}")

    return (FAIL if reasons else PASS), reasons
