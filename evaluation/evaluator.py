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

from answering.src import service
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
            "expected_answer": self.expected_answer,
            "expected_facts": self.expected_facts,
            "conversation_from": self.conversation_from,
        }


def _restricted_subject(case: EvalCase, catalog: PermissionCatalog) -> str:
    """Plain description of what this identity must not learn about."""
    parts = []
    for document_id in case.forbidden_sources:
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
        parts.append("any document outside this identity's authorized scope")
    return "; ".join(parts)


def _needs_leakage_check(case: EvalCase) -> bool:
    return bool(case.forbidden_sources) or case.expected_behavior == "refuse"


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
            user_id=case.user_id,
            query=case.question,
            conversation_history=conversation_history or None,
            top_k=CASE_TOP_K,
        )
    except (service.AnsweringError, EmbeddingError) as exc:
        result.status = ERROR
        result.error = f"{type(exc).__name__}: {exc}"
        return result
    except Exception as exc:
        result.status = ERROR
        result.error = f"{type(exc).__name__}: {exc}"
        logger.exception("case %s failed before scoring", case.eval_id)
        return result

    result.request_id = response.request_id
    result.answered = response.answered
    result.answer = response.answer
    result.latency_ms = response.latency_ms
    result.actual_behavior = "answer" if response.answered else "refuse"
    result.retrieved_sources = [s["document_id"] for s in response.sources]
    result.candidate_sources = list(response.candidate_document_ids)
    result.stats = response.to_dict().get("stats", {})

    # Record deterministic evidence before any judge request can fail.
    needs_judge = response.answered and _needs_leakage_check(case)
    result.security = metrics.score_security(
        expected_behavior=case.expected_behavior,
        actual_behavior=result.actual_behavior,
        denied_documents=catalog.denied_documents(case.user_id),
        forbidden_documents=list(case.forbidden_sources),
        context_documents=result.retrieved_sources,
        candidate_documents=result.candidate_sources,
        authorized_documents=catalog.authorized_documents(case.user_id),
        leakage_checked=not response.answered,
        leakage_required=needs_judge,
    ).to_dict()
    result.retrieval = metrics.score_retrieval(
        list(case.expected_sources), result.retrieved_sources,
        applicable=case.expected_behavior != "refuse").to_dict()

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
        if verdict["refused"]:
            result.actual_behavior = "refuse"
        if verdict["leaked"]:
            leaked_evidence = verdict["evidence"] or [verdict["reason"]]

    result.security = metrics.score_security(
        expected_behavior=case.expected_behavior,
        actual_behavior=result.actual_behavior,
        denied_documents=catalog.denied_documents(case.user_id),
        forbidden_documents=list(case.forbidden_sources),
        context_documents=result.retrieved_sources,
        candidate_documents=result.candidate_sources,
        leaked_evidence=leaked_evidence,
        authorized_documents=catalog.authorized_documents(case.user_id),
        leakage_checked=needs_judge or not response.answered,
        leakage_required=needs_judge,
    ).to_dict()

    if case.expected_behavior == "refuse":
        result.retrieval = metrics.score_retrieval(
            [], result.retrieved_sources, applicable=False).to_dict()
        result.generation = metrics.GenerationMetrics(
            applicable=False,
            judge_notes="not applicable: the case expects no answer",
        ).to_dict()
        result.status = (PASS if result.security["passed"]
                         and result.actual_behavior == "refuse" else FAIL)
        if result.status == FAIL:
            result.failure_reasons = _security_reasons(result.security)
            if result.actual_behavior != "refuse":
                result.failure_reasons.append("expected a refusal, got an answer")
        return result

    retrieval = metrics.score_retrieval(list(case.expected_sources),
                                        result.retrieved_sources)
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
    return result


def _security_reasons(security: dict) -> list[str]:
    reasons = []
    if security["unauthorized_retrieval"]:
        reasons.append(
            "unauthorized documents reached retrieval or context: "
            + ", ".join(security["unauthorized_documents"]))
    if security["answer_leakage"]:
        reasons.append("the answer disclosed restricted subject matter")
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
