"""Phase 5 configuration: Groq answering, Jina reranking, retrieval budgets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from . import groq_keys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, "") or default)
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


GROQ_KEY_COOLDOWN_S = _float("GROQ_KEY_COOLDOWN_S", 60.0)
# The pool is built here, after load_dotenv, so it always sees the repository
# .env. Key material lives only inside the pool: it is never a module constant.
groq_keys.set_pool(groq_keys.GroqKeyPool(cooldown_s=GROQ_KEY_COOLDOWN_S))
GROQ_KEY_COUNT = groq_keys.pool().size

GROQ_API_URL = os.getenv(
    "GROQ_API_URL", "https://api.groq.com/openai/v1/chat/completions")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = _float("GROQ_TEMPERATURE", 0.0)
GROQ_MAX_TOKENS = _int("GROQ_MAX_TOKENS", 1200)
# gpt-oss models spend part of max_tokens on hidden reasoning; "low" keeps the
# visible answer within budget.
GROQ_REASONING_EFFORT = os.getenv("GROQ_REASONING_EFFORT", "low")
GROQ_TIMEOUT_S = _int("GROQ_TIMEOUT_S", 60)
GROQ_MAX_RETRIES = _int("GROQ_MAX_RETRIES", 2)

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_RERANK_URL = os.getenv("JINA_RERANK_URL", "https://api.jina.ai/v1/rerank")
JINA_RERANK_MODEL = os.getenv(
    "JINA_RERANK_MODEL", "jina-reranker-v2-base-multilingual")
RERANK_TIMEOUT_S = _int("RERANK_TIMEOUT_S", 30)
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").strip().lower() != "false"
# Must be >= the largest indexed chunk or the reranker scores a truncated
# passage. Ingestion caps chunks at 800 tokens (~4.4k chars observed).
RERANK_MAX_DOC_CHARS = _int("RERANK_MAX_DOC_CHARS", 4800)
# A full candidate pool at that width is ~24k tokens per call, and Jina meters
# tokens per minute. Falling back to fusion order on a rate limit would quietly
# ship an unranked context, so transient failures are retried instead.
RERANK_MAX_RETRIES = _int("RERANK_MAX_RETRIES", 4)
RERANK_BACKOFF_S = _float("RERANK_BACKOFF_S", 5.0)
RERANK_MAX_BACKOFF_S = _float("RERANK_MAX_BACKOFF_S", 60.0)

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = _int("API_PORT", 8000)

OPENJEV_API_KEY = os.getenv("OPENJEV_API_KEY", "").strip()
OPENJEV_BASE_URL = os.getenv(
    "OPENJEV_BASE_URL", "https://api.openjev.sh/v1/systemone")
OPENJEV_MODEL = os.getenv("OPENJEV_MODEL", "openjev")
# Query understanding sits before retrieval on the request path, so the budget
# is small and a failure falls back rather than retrying for long.
OPENJEV_TIMEOUT_S = _int("OPENJEV_TIMEOUT_S", 8)
OPENJEV_MAX_RETRIES = _int("OPENJEV_MAX_RETRIES", 1)
OPENJEV_BACKOFF_S = _float("OPENJEV_BACKOFF_S", 1.0)
OPENJEV_MAX_BACKOFF_S = _float("OPENJEV_MAX_BACKOFF_S", 6.0)
OPENJEV_ENABLED = (OPENJEV_API_KEY
                   and os.getenv("OPENJEV_ENABLED", "true").strip().lower()
                   != "false")

QUERY_AWARE_ENABLED = os.getenv("QUERY_AWARE_ENABLED", "true").strip().lower() \
    != "false"
QUERY_AWARE_IDENTIFIER_CANDIDATES = _int("QUERY_AWARE_IDENTIFIER_CANDIDATES", 10)
QUERY_AWARE_IDENTIFIER_BOOST = _float("QUERY_AWARE_IDENTIFIER_BOOST", 1.0)
QUERY_AWARE_ENTITY_BOOST = _float("QUERY_AWARE_ENTITY_BOOST", 0.2)
QUERY_AWARE_SELLER_BOOST = _float("QUERY_AWARE_SELLER_BOOST", 0.15)
QUERY_AWARE_EVIDENCE_TYPE_BOOST = _float("QUERY_AWARE_EVIDENCE_TYPE_BOOST", 0.1)

# Phase 7 query-aware reranking. The plan signal is a bounded ADDITIVE term on
# top of Jina relevance, never a replacement: with the default weight the total
# adjustment is capped at 0.06 while Jina separates on-topic (>0.5) from
# off-topic (<0.05) passages, so a metadata match can break a tie but cannot
# overturn a relevance judgment. Set RERANK_PLAN_AWARE=false for exact
# pre-Phase-7 ordering.
RERANK_PLAN_AWARE = (
    os.getenv("RERANK_PLAN_AWARE", "true").strip().lower() != "false")
RERANK_PLAN_WEIGHT = _float("RERANK_PLAN_WEIGHT", 0.06)
# Relative importance inside that bound, set by the Phase 7 ablation over the
# 45 scored golden cases (evaluation/runs/phase7-eval.json). Only the
# evidence-type signal earned its place: it improved 9 cases and regressed 0
# (recall 0.9556->0.9741, precision 0.4156->0.4285, MRR 0.8674->0.9081). That
# is the signal that directly attacks the two largest Phase 6 false-positive
# categories - "wrong document type" (33%) and "generic organization document"
# (28%) - which is why it generalises rather than fitting this benchmark.
#
# The others are implemented and configurable but ship at zero because they
# measured worse: intent cost recall on E001 (1.0->0.5), aspect and event cost
# precision, seller was inert, and combining them diluted the evidence gain
# back to baseline precision. Raise one only with a re-run of the ablation.
RERANK_PLAN_ASPECT_WEIGHT = _float("RERANK_PLAN_ASPECT_WEIGHT", 0.0)
RERANK_PLAN_EVENT_WEIGHT = _float("RERANK_PLAN_EVENT_WEIGHT", 0.0)
RERANK_PLAN_INTENT_WEIGHT = _float("RERANK_PLAN_INTENT_WEIGHT", 0.0)
RERANK_PLAN_EVIDENCE_WEIGHT = _float("RERANK_PLAN_EVIDENCE_WEIGHT", 1.0)
RERANK_PLAN_SELLER_WEIGHT = _float("RERANK_PLAN_SELLER_WEIGHT", 0.0)
# Appending "needs: ...; prefer: ..." to the reranker query measured WORSE than
# leaving the query alone (recall 0.9556->0.9333, two cases regressed, E054 to
# zero): the hint competes with the user's own wording for Jina's attention.
# Off by default; the code path is kept for future experiments.
RERANK_PLAN_QUERY_HINTS = (
    os.getenv("RERANK_PLAN_QUERY_HINTS", "false").strip().lower() == "true")

QUERY_AWARE_PROMOTION = (
    os.getenv("QUERY_AWARE_PROMOTION", "true").strip().lower() != "false")
QUERY_AWARE_PROMOTE_CRITERIA = tuple(
    part.strip() for part in
    os.getenv("QUERY_AWARE_PROMOTE_CRITERIA", "aspect,date").split(",")
    if part.strip())
QUERY_AWARE_MAX_PROMOTED = _int("QUERY_AWARE_MAX_PROMOTED", 4)

# Full evaluation answers span all demo identities. Expose them only in an
# explicitly enabled, trusted local operator session until real auth exists.
EVALUATION_REPORTS_ENABLED = os.getenv(
    "EVALUATION_REPORTS_ENABLED", "false").strip().lower() == "true"


# Phase 8 evidence-aware selection. The selector replaces "top K by score" with
# bounded greedy marginal utility: a document is worth a slot for its relevance
# PLUS what it adds that nothing already selected covers, MINUS how much it
# repeats. Relevance is normalised to 0..1 per query, so these weights read as
# "how much of the best document's relevance one new piece of coverage is
# worth" - all well under 1.0, because coverage must never overturn a decisive
# relevance judgment (spec section 18).
EVIDENCE_SELECTION_ENABLED = (
    os.getenv("EVIDENCE_SELECTION_ENABLED", "true").strip().lower() != "false")
EVIDENCE_ASPECT_WEIGHT = _float("EVIDENCE_ASPECT_WEIGHT", 0.30)
EVIDENCE_TYPE_WEIGHT = _float("EVIDENCE_TYPE_WEIGHT", 0.15)
EVIDENCE_ENTITY_WEIGHT = _float("EVIDENCE_ENTITY_WEIGHT", 0.20)
EVIDENCE_REDUNDANCY_PENALTY = _float("EVIDENCE_REDUNDANCY_PENALTY", 0.25)
# The stopping rule, and the actual precision lever: the value a candidate must
# still be worth, given everything already selected, to earn a slot. 0.0 keeps
# every document the capacity allows; raising it trades breadth for precision.
# Tuned on the Phase 8 dev split only - see docs/phase8-evidence-aware-selection.md.
EVIDENCE_MIN_MARGINAL_UTILITY = _float("EVIDENCE_MIN_MARGINAL_UTILITY", 0.0)


# Phase 9 evidence-aware context packing. The context budget is unchanged
# (max_context_chars); Phase 9 only changes HOW that fixed budget is spent.
# Measured motivation (docs/phase9-context-packing.md): the budget binds in 28
# of 45 cases and drops ~24% of Phase 8's selected chunks, and the old
# depth-round x score fill can drop an aspect-complementary passage (e.g. the
# remediation half of a two-part answer) while keeping a redundant background
# one. Phase 9 keeps document breadth first (so recall is untouched) and then
# fills the remaining budget with aspect-complementary passages before
# redundant depth, which is the "drop low-value before high-value" rule of
# spec section 21. Set CONTEXT_PACKING_ENABLED=false for the exact pre-Phase-9
# breadth-first-by-depth-round builder.
CONTEXT_PACKING_ENABLED = (
    os.getenv("CONTEXT_PACKING_ENABLED", "true").strip().lower() != "false")
# Two selected passages of one document count as redundant only above this
# Jaccard overlap. Deliberately high: on this corpus the maximum observed
# same-document passage overlap is well below it (measured 0 pairs >= 0.5), so
# this is a conservative guard that never fires today and exists only to stay
# safe if the corpus changes. It never drops a passage that carries a required
# aspect none of the already-kept passages carry.
CONTEXT_PACKING_REDUNDANCY_TAU = _float("CONTEXT_PACKING_REDUNDANCY_TAU", 0.85)


@dataclass(frozen=True)
class AnsweringConfig:
    top_k: int = _int("ANSWER_TOP_K", 5)
    candidate_pool: int = _int("ANSWER_CANDIDATE_POOL", 20)
    # Jina rerank relevance is 0..1; clearly irrelevant passages score <0.05
    # while on-topic ones score >0.5, so 0.20 separates them with margin.
    rerank_threshold: float = _float("RERANK_SCORE_THRESHOLD", 0.20)
    # Absolute floors cannot tell "weak but best available" from "weak noise
    # next to a clear winner". This keeps a passage only if it scores within a
    # fraction of the strongest passage for THIS query, so the context narrows
    # when the evidence is decisive and stays wide when it genuinely is not.
    rerank_relative_threshold: float = _float("RERANK_RELATIVE_THRESHOLD", 0.40)
    # Retrieval metrics and the reader both count DOCUMENTS, not chunks. These
    # two bound breadth and depth separately, so the budget can buy several
    # passages from one document instead of one passage from several.
    #
    # Tuned on the golden set against cached reranker scores: with the reranker
    # query and truncation fixed, this recall-first point reaches recall 0.96 /
    # MRR 0.86 at efficiency 0.95. Precision is intentionally not maximised - in
    # a permission-aware RAG the reader can ignore an extra authorized document
    # but can never recover one that was never shown, and groundedness is
    # already ~0.96. Narrow max_documents toward 3 (with rel ~0.55) to trade
    # recall for a higher precision number.
    max_documents: int = _int("ANSWER_MAX_DOCUMENTS", 5)
    max_chunks_per_document: int = _int("ANSWER_MAX_CHUNKS_PER_DOCUMENT", 3)
    # Additional chunks from a document corroborate it without being allowed to
    # outweigh a single decisively better passage elsewhere.
    document_support_weight: float = _float("ANSWER_DOC_SUPPORT_WEIGHT", 0.25)
    max_history_turns: int = _int("MAX_HISTORY_TURNS", 6)
    max_history_chars: int = _int("MAX_HISTORY_CHARS", 1500)
    max_context_chars: int = _int("MAX_CONTEXT_CHARS", 16000)
    max_query_chars: int = _int("MAX_QUERY_CHARS", 1000)


DEFAULT_ANSWERING = AnsweringConfig()

REFUSAL_ANSWER = (
    "I couldn't find sufficient authorized information to answer this request."
)

REQUIRED_ENV_VARS = ("JINA_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")


def validate_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if not groq_keys.pool().size:
        missing.append("GROQ_API_KEY_1")
    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
            + ". Check .env at the repository root.")
