"""Phase 5 pipeline: identity -> scope -> retrieval -> authorization -> rerank
-> context -> Groq -> grounded answer with sources.

Authorization is enforced entirely before this module builds any context. The
LLM is an answer generator; it is never consulted about access.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from access_control.src.engine import PermissionEngine
from access_control.src.policy import IdentityError, ScopeDeniedError
from access_control.src.retrieval import authorized_retrieve
from access_control.src.users import USERS, User, get_user
from indexing.src.embeddings import EmbeddingError

from . import config, context, conversation, llm, prompts, reranker

logger = logging.getLogger(__name__)

_engine = PermissionEngine()


class AnsweringError(RuntimeError):
    """Base class for failures that map to a safe client-facing message."""
    status = 500
    client_message = "The request could not be completed."


class InvalidUserError(AnsweringError):
    status = 400
    client_message = "Unknown or invalid user."


class InvalidQueryError(AnsweringError):
    status = 400
    client_message = "A non-empty query is required."


class RetrievalUnavailableError(AnsweringError):
    status = 503
    client_message = "The knowledge base is temporarily unavailable."


class AnswerUnavailableError(AnsweringError):
    status = 503
    client_message = "The answering service is temporarily unavailable."


@dataclass
class AnswerResponse:
    request_id: str
    user_id: str
    query: str
    answer: str
    answered: bool
    sources: list[dict] = field(default_factory=list)
    candidates: int = 0
    authorized: int = 0
    selected: int = 0
    withheld: int = 0
    reranked: bool = False
    effective_sellers: list[str] = field(default_factory=list)
    latency_ms: int = 0
    model: str | None = None

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "query": self.query,
            "answer": self.answer,
            "answered": self.answered,
            "sources": self.sources,
            "effective_sellers": self.effective_sellers,
            "stats": {
                "candidates": self.candidates,
                "authorized": self.authorized,
                "selected": self.selected,
                "withheld": self.withheld,
                "reranked": self.reranked,
                "latency_ms": self.latency_ms,
                "model": self.model,
            },
        }


def list_users() -> list[dict]:
    """Prototype identity directory.

    `seller_scope` is read back from the permission engine rather than restated
    here, so the UI can only ever display the scope the server will actually
    enforce. It is display-only: the server never accepts a scope from a client.
    """
    return [
        {"user_id": u.user_id, "name": u.name, "role": u.role,
         "department": u.department,
         "seller_scope": _engine.effective_seller_scope(u)}
        for u in USERS.values()
    ]


def _resolve_identity(user_id) -> User:
    if isinstance(user_id, User):
        return user_id
    if not isinstance(user_id, str) or not user_id.strip():
        raise InvalidUserError("user_id must be a non-empty string")
    try:
        return get_user(user_id.strip())
    except IdentityError as exc:
        raise InvalidUserError(str(exc)) from exc


def _clean_query(query, cfg: config.AnsweringConfig) -> str:
    if not isinstance(query, str) or not query.strip():
        raise InvalidQueryError("query must be a non-empty string")
    return query.strip()[:cfg.max_query_chars]


def _refusal(request_id: str, user: User, query: str, started: float,
             sellers: list[str], **stats) -> AnswerResponse:
    return AnswerResponse(
        request_id=request_id,
        user_id=user.user_id,
        query=query,
        answer=config.REFUSAL_ANSWER,
        answered=False,
        effective_sellers=sellers,
        latency_ms=int((time.time() - started) * 1000),
        **stats,
    )


def _with_sources(answer: str, sources: list) -> str:
    return answer + _sources_block(sources)


def _sources_block(sources: list) -> str:
    if not sources:
        return ""
    lines = "\n".join(f"- {s.citation()}" for s in sources)
    return f"\n\nSources:\n{lines}"


@dataclass
class _Prepared:
    """Everything the LLM step needs, after authorization is fully settled."""
    request_id: str
    user: User
    question: str
    messages: list[dict]
    sources: list
    sellers: list[str]
    stats: dict
    reranked: bool
    started: float


def _prepare(user_id, query, conversation_history, seller_filter, top_k,
             request_id, cfg) -> tuple[_Prepared | None, AnswerResponse | None]:
    """Run the whole permission-aware pipeline up to prompt assembly.

    Returns (prepared, None) when there is authorized context to answer from,
    or (None, refusal) otherwise. Both the buffered and the streaming entry
    points go through here, so authorization exists in exactly one place.
    """
    started = time.time()
    user = _resolve_identity(user_id)
    question = _clean_query(query, cfg)
    turns = conversation.parse_history(conversation_history, cfg)
    top_k = max(1, min(int(top_k or cfg.top_k), cfg.candidate_pool))

    # Scope comes from the identity only. seller_filter can narrow it, never
    # widen it; a request with no authorized overlap refuses immediately.
    try:
        sellers = _engine.effective_seller_scope(user, seller_filter)
    except IdentityError as exc:
        raise InvalidUserError(str(exc)) from exc
    if sellers is None:
        logger.info("request=%s user=%s status=scope_denied", request_id,
                    user.user_id)
        return None, _refusal(request_id, user, question, started, [])

    search_text = conversation.retrieval_query(turns, question, cfg)

    try:
        access = authorized_retrieve(
            user, search_text,
            top_k=cfg.candidate_pool,
            request_id=request_id,
            requested_sellers=seller_filter,
        )
    except ScopeDeniedError:
        return None, _refusal(request_id, user, question, started, [])
    except IdentityError as exc:
        raise InvalidUserError(str(exc)) from exc
    except EmbeddingError as exc:
        logger.error("request=%s embedding failure: %s", request_id, exc)
        raise RetrievalUnavailableError(str(exc)) from exc
    except Exception as exc:
        logger.error("request=%s retrieval failure: %s", request_id, exc)
        raise RetrievalUnavailableError(str(exc)) from exc

    authorized_chunks = access.chunks
    stats = {"candidates": len(authorized_chunks) + access.withheld_count,
             "authorized": len(authorized_chunks),
             "withheld": access.withheld_count}

    if not authorized_chunks:
        logger.info("request=%s user=%s status=no_authorized_context",
                    request_id, user.user_id)
        return None, _refusal(request_id, user, question, started, sellers,
                              **stats)

    outcome = reranker.rerank(question, authorized_chunks,
                              threshold=cfg.rerank_threshold)
    selected = [(chunk, score) for chunk, score in outcome.items][:top_k]

    # Final gate. Reranking reorders authorized material and must never be able
    # to reintroduce anything unauthorized.
    verified = []
    for chunk, score in selected:
        decision = _engine.can_access(user, chunk.metadata)
        if decision.allow:
            verified.append((chunk, score))
        else:
            logger.error("request=%s chunk=%s dropped at final gate: %s",
                         request_id, chunk.chunk_id, decision.reason)

    if not verified:
        logger.info("request=%s user=%s status=below_threshold", request_id,
                    user.user_id)
        return None, _refusal(request_id, user, question, started, sellers,
                              reranked=outcome.applied, **stats)

    context_block, sources = context.build(verified, cfg)
    if not context_block.strip():
        return None, _refusal(request_id, user, question, started, sellers,
                              reranked=outcome.applied, **stats)

    messages = prompts.build_messages(
        prompts.SYSTEM_PROMPT,
        conversation.render(turns),
        context_block,
        question,
    )
    return _Prepared(request_id, user, question, messages, sources, sellers,
                     stats, outcome.applied, started), None


def _finalize(prepared: _Prepared, answer_text: str,
              model: str | None) -> AnswerResponse:
    response = AnswerResponse(
        request_id=prepared.request_id,
        user_id=prepared.user.user_id,
        query=prepared.question,
        answer=_with_sources(answer_text, prepared.sources),
        answered=True,
        sources=[s.to_dict() for s in prepared.sources],
        selected=len(prepared.sources),
        reranked=prepared.reranked,
        effective_sellers=prepared.sellers,
        latency_ms=int((time.time() - prepared.started) * 1000),
        model=model,
        **prepared.stats,
    )
    logger.info(
        "request=%s user=%s status=answered candidates=%d authorized=%d "
        "withheld=%d selected=%d reranked=%s docs=%s latency_ms=%d",
        response.request_id, response.user_id, response.candidates,
        response.authorized, response.withheld, response.selected,
        response.reranked,
        [s["document_id"] for s in response.sources], response.latency_ms,
    )
    return response


def answer_query(user_id, query, conversation_history=None,
                 seller_filter=None, top_k: int | None = None,
                 request_id: str | None = None,
                 cfg: config.AnsweringConfig | None = None) -> AnswerResponse:
    cfg = cfg or config.DEFAULT_ANSWERING
    request_id = request_id or uuid.uuid4().hex[:12]

    prepared, refusal = _prepare(user_id, query, conversation_history,
                                 seller_filter, top_k, request_id, cfg)
    if refusal is not None:
        return refusal

    try:
        answer_text, usage = llm.generate(prepared.messages)
    except llm.LLMError as exc:
        logger.error("request=%s groq failure: %s", request_id, exc)
        raise AnswerUnavailableError(str(exc)) from exc

    # Citations are appended deterministically from the chunks that were
    # actually authorized and shown, so they cannot be hallucinated.
    return _finalize(prepared, answer_text, usage.get("model"))


def answer_query_stream(user_id, query, conversation_history=None,
                        seller_filter=None, top_k: int | None = None,
                        request_id: str | None = None,
                        cfg: config.AnsweringConfig | None = None):
    """Yield pipeline events. Authorization completes before the first token.

    Events: meta -> delta* -> done, or meta -> delta -> done for a refusal,
    or an error event if generation fails mid-answer.
    """
    cfg = cfg or config.DEFAULT_ANSWERING
    request_id = request_id or uuid.uuid4().hex[:12]

    prepared, refusal = _prepare(user_id, query, conversation_history,
                                 seller_filter, top_k, request_id, cfg)
    if refusal is not None:
        payload = refusal.to_dict()
        yield {"type": "meta", **{k: payload[k] for k in
                                  ("request_id", "user_id", "answered",
                                   "sources", "effective_sellers")},
               "stats": payload["stats"]}
        yield {"type": "delta", "text": refusal.answer}
        yield {"type": "done", **payload}
        return

    yield {
        "type": "meta",
        "request_id": prepared.request_id,
        "user_id": prepared.user.user_id,
        "answered": True,
        "sources": [s.to_dict() for s in prepared.sources],
        "effective_sellers": prepared.sellers,
        "stats": {**prepared.stats, "selected": len(prepared.sources),
                  "reranked": prepared.reranked},
    }

    pieces: list[str] = []
    try:
        for delta in llm.stream(prepared.messages):
            pieces.append(delta)
            yield {"type": "delta", "text": delta}
    except llm.LLMError as exc:
        logger.error("request=%s groq streaming failure: %s", request_id, exc)
        yield {"type": "error", "error": "AnswerUnavailableError",
               "message": AnswerUnavailableError.client_message}
        return

    # The deterministic citation block is streamed too, so the text the client
    # accumulated equals the final answer exactly.
    block = _sources_block(prepared.sources)
    if block:
        yield {"type": "delta", "text": block}

    response = _finalize(prepared, "".join(pieces), config.GROQ_MODEL)
    yield {"type": "done", **response.to_dict()}
