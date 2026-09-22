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

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = _int("API_PORT", 8000)

# Full evaluation answers span all demo identities. Expose them only in an
# explicitly enabled, trusted local operator session until real auth exists.
EVALUATION_REPORTS_ENABLED = os.getenv(
    "EVALUATION_REPORTS_ENABLED", "false").strip().lower() == "true"


@dataclass(frozen=True)
class AnsweringConfig:
    top_k: int = _int("ANSWER_TOP_K", 5)
    candidate_pool: int = _int("ANSWER_CANDIDATE_POOL", 20)
    # Jina rerank relevance is 0..1; clearly irrelevant passages score <0.05
    # while on-topic ones score >0.5, so 0.20 separates them with margin.
    rerank_threshold: float = _float("RERANK_SCORE_THRESHOLD", 0.20)
    max_history_turns: int = _int("MAX_HISTORY_TURNS", 6)
    max_history_chars: int = _int("MAX_HISTORY_CHARS", 1500)
    max_context_chars: int = _int("MAX_CONTEXT_CHARS", 12000)
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
