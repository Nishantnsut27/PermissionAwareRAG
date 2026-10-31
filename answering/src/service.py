"""Phase 5 pipeline: identity -> scope -> retrieval -> authorization -> rerank
-> context -> Groq -> grounded answer with sources.

Authorization is enforced entirely before this module builds any context. The
LLM is an answer generator; it is never consulted about access.
"""
from __future__ import annotations

import logging
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from access_control.src.engine import PermissionEngine
from access_control.src.policy import IdentityError, ScopeDeniedError
from access_control.src.retrieval import authorized_retrieve, authorized_retrieve_balanced
from access_control.src.users import USERS, User
from indexing.src.embeddings import EmbeddingError

from . import answer_scope, config, context, conversation, identity, llm, prompts, reranker, seller_balanced
from .deadline import check_deadline, retry_snapshot

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
    outcome: str = "answer"
    sources: list[dict] = field(default_factory=list)
    candidates: int = 0
    authorized: int = 0
    selected: int = 0
    withheld: int = 0
    reranked: bool = False
    effective_sellers: list[str] = field(default_factory=list)
    latency_ms: int = 0
    model: str | None = None
    # Phase 7 introspection. Deliberately absent from to_dict(): the HTTP
    # payload and the chat UI are unchanged, and full passage text is not put
    # on the wire for an answer that already carries its citations. In-process
    # callers (the evaluation runner) read these to measure groundedness and
    # to check authorization at the retrieval level, not only in the answer.
    context: str = ""
    candidate_document_ids: list[str] = field(default_factory=list)
    # Operator-only evaluation trace. Never serialized in user API responses.
    security_trace: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "query": self.query,
            "answer": self.answer,
            "outcome": self.outcome,
            "completed": self.outcome in ("answer", "refusal"),
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


def get_demo_user(user_id) -> User:
    """Resolve an identity only for the explicitly isolated local demo route."""
    if not config.API_DEMO_MODE:
        raise InvalidUserError("Demo identity selection is disabled")
    from access_control.src.users import get_user
    if not isinstance(user_id, str) or not user_id.strip():
        raise InvalidUserError("A demo user ID is required")
    try:
        return get_user(user_id.strip())
    except IdentityError as exc:
        raise InvalidUserError(str(exc)) from exc


def _resolve_identity(principal) -> User:
    if isinstance(principal, identity.Principal):
        return principal.user
    # User IDs and bare User records are not proof of an authenticated caller.
    raise InvalidUserError("A verified server identity is required")


def _clean_query(query, cfg: config.AnsweringConfig) -> str:
    if not isinstance(query, str) or not query.strip():
        raise InvalidQueryError("query must be a non-empty string")
    if len(query) > cfg.max_query_chars:
        raise InvalidQueryError("query exceeds the configured character limit")
    return query.strip()


def _refusal(request_id: str, user: User, query: str, started: float,
             sellers: list[str], candidate_document_ids=None,
             security_trace=None, missing_sellers=None,
             **stats) -> AnswerResponse:
    check_deadline()
    trace = dict(security_trace or {})
    if "intent_usage" in trace:
        rows = trace.get("intent_usage") or []
        trace["model_usage"] = (_usage_summary(*rows) if rows else
                                {"reported": True, "total_tokens": 0})
    return AnswerResponse(
        request_id=request_id,
        user_id=user.user_id,
        query=query,
        answer=config.REFUSAL_ANSWER + _missing_note(missing_sellers or [], []),
        outcome="refusal",
        answered=False,
        effective_sellers=sellers,
        latency_ms=int((time.time() - started) * 1000),
        candidate_document_ids=list(candidate_document_ids or []),
        security_trace=trace,
        **stats,
    )


def _with_sources(answer: str, sources: list) -> str:
    return answer + _sources_block(sources)


def _sources_block(sources: list) -> str:
    if not sources:
        return ""
    lines = "\n".join(
        f"- [{evidence_id}] {source.citation()}"
        for source in sources for evidence_id in source.evidence_ids)
    return f"\n\nEvidence available in context:\n{lines}"


def _strip_model_sources(text: str) -> str:
    for title in ("Sources", "Evidence available in context"):
        match = re.search(rf"(?im)^\s*{re.escape(title)}:\s*$", text)
        if match:
            after = text[match.end():]
            lines = [line for line in after.splitlines() if line.strip()]
            if not lines or all(line.lstrip().startswith(("- ", "* "))
                                for line in lines):
                text = text[:match.start()].rstrip()
    return text


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
    context: str = ""
    candidate_document_ids: list[str] = field(default_factory=list)
    security_trace: dict = field(default_factory=dict)
    coverage_sellers: list[str] = field(default_factory=list)
    portfolio_only: bool = False


def _chunk_ref(chunk, score=None) -> dict:
    metadata = getattr(chunk, "metadata", None) or {}
    return {"chunk_id": getattr(chunk, "chunk_id", None),
            "document_id": (getattr(chunk, "document_id", None)
                            or metadata.get("document_id")),
            **({"score": score} if score is not None else {})}


def _context_spans(context_block: str, sources: list, chunks: list) -> list[dict]:
    spans = []
    cursor = 0
    for source in sources:
        for span in source.spans:
            start = context_block.find(span["text"], cursor)
            if start < 0:
                continue
            end = start + len(span["text"])
            spans.append({**span, "start": start, "end": end})
            cursor = end
    return spans


def _prepare(principal, query, conversation_history, seller_filter, top_k,
             request_id, cfg) -> tuple[_Prepared | None, AnswerResponse | None]:
    """Run the whole permission-aware pipeline up to prompt assembly.

    Returns (prepared, None) when there is authorized context to answer from,
    or (None, refusal) otherwise. Both the buffered and the streaming entry
    points go through here, so authorization exists in exactly one place.
    """
    started = time.time()
    user = _resolve_identity(principal)
    question = _clean_query(query, cfg)
    if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int)
                              or not 1 <= top_k <= cfg.candidate_pool):
        raise ValueError("top_k must be an integer in the configured range")
    turns = conversation.parse_history(conversation_history, cfg)
    intent_usage = []
    top_k = cfg.top_k if top_k is None else top_k
    if seller_filter is not None:
        if (not isinstance(seller_filter, list) or len(seller_filter) > 4
                or any(not isinstance(value, str) for value in seller_filter)
                or len(set(seller_filter)) != len(seller_filter)
                or any(value not in {"S001", "S002", "S003", "S004"}
                       for value in seller_filter)):
            raise ValueError("seller_filter must be a unique list of at most four seller IDs")

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

    intent_started = time.monotonic()
    try:
        portfolio_only = answer_scope.is_portfolio(question, turns, intent_usage)
    except answer_scope.IntentUnavailableError as exc:
        raise AnswerUnavailableError(str(exc)) from exc
    if portfolio_only and not sellers:
        return None, _refusal(request_id, user, question, started, sellers,
                              security_trace={"intent_usage": intent_usage})
    # Conversation can resolve the topic, never the account inventory. Remove
    # assistant assertions and prior questions naming excluded sellers before
    # retrieval/generation; use the current principal on every follow-up.
    if portfolio_only:
        turns = [turn for turn in turns if turn.role == "user"
                 and answer_scope.mentioned_sellers(turn.content) <= set(sellers)]
    scoped_question = (answer_scope.safe_question(question, sellers)
                       if portfolio_only else question)
    search_text = conversation.retrieval_query(turns, scoped_question, cfg)
    coverage_sellers = (list(sellers) if portfolio_only else
                        seller_balanced.relevant_sellers(question, sellers)
                        if cfg.multi_seller_retrieval == "seller_balanced" else [])

    retrieval_started = time.monotonic()
    try:
        if coverage_sellers:
            access = authorized_retrieve_balanced(
                user, search_text, coverage_sellers, cfg.candidate_pool, request_id)
        else:
            access = authorized_retrieve(
                user, search_text,
                top_k=cfg.candidate_pool,
                request_id=request_id,
                requested_sellers=seller_filter,
            )
    except ScopeDeniedError:
        return None, _refusal(request_id, user, question, started, [],
                              security_trace={"intent_usage": intent_usage})
    except IdentityError as exc:
        raise InvalidUserError(str(exc)) from exc
    except EmbeddingError as exc:
        logger.error("request=%s embedding failure: %s", request_id, exc)
        raise RetrievalUnavailableError(str(exc)) from exc
    except TimeoutError:
        raise
    except Exception as exc:
        logger.error("request=%s retrieval failure: %s", request_id, exc)
        raise RetrievalUnavailableError(str(exc)) from exc

    authorized_chunks = access.chunks
    stage_trace = dict(access.stage_trace)
    stage_trace["intent_usage"] = intent_usage
    stage_trace["answer_scope"] = "portfolio" if portfolio_only else "document_policy"
    if portfolio_only:
        stage_trace["portfolio_passage_dropped"] = [
            _chunk_ref(chunk) for chunk in authorized_chunks
            if not answer_scope.safe_passage(chunk, sellers)]
        authorized_chunks = [chunk for chunk in authorized_chunks
                             if answer_scope.safe_passage(chunk, sellers)]
    stage_trace.setdefault("retrieval_path", "global")
    if coverage_sellers:
        stage_trace["seller_coverage"] = {seller: False for seller in coverage_sellers}
    stage_trace["stage_latency_ms"] = {
        "answer_intent": round((retrieval_started - intent_started) * 1000),
        "retrieval_and_authorization": round((time.monotonic() - retrieval_started) * 1000)
    }
    stage_trace["request_id"] = request_id
    stage_trace["reranked_items"] = []
    stage_trace["final_gate_denied"] = []
    stage_trace["context_spans"] = []
    stage_trace["emitted_citations"] = []
    stats = {"candidates": len(access.chunks) + access.withheld_count,
             "authorized": len(access.chunks),
             "withheld": access.withheld_count}
    # Ranked document ids of every chunk that survived authorization, before
    # reranking and the context budget. Phase 7 uses this to tell an
    # unauthorized *retrieval* apart from an unauthorized *answer*.
    candidate_ids: list[str] = []
    for chunk in authorized_chunks:
        document_id = chunk.document_id or (chunk.metadata or {}).get(
            "document_id")
        if document_id and document_id not in candidate_ids:
            candidate_ids.append(document_id)

    if not authorized_chunks:
        logger.info("request=%s user=%s status=no_authorized_context",
                    request_id, user.user_id)
        return None, _refusal(request_id, user, question, started, sellers,
                              missing_sellers=coverage_sellers if portfolio_only else [],
                              security_trace=stage_trace,
                              **stats)

    check_deadline()
    rerank_started = time.monotonic()
    if coverage_sellers:
        outcome = seller_balanced.rerank_sellers(
            search_text, authorized_chunks, coverage_sellers, cfg.rerank_threshold)
    else:
        outcome = reranker.rerank(search_text, authorized_chunks,
                                  threshold=cfg.rerank_threshold)
    stage_trace["stage_latency_ms"]["reranking"] = round(
        (time.monotonic() - rerank_started) * 1000)
    if not outcome.applied:
        stage_trace["degraded_mode"] = ("seller_partition_fallback" if coverage_sellers
                                         else "authorized_fusion_order")
    stage_trace["reranked_items"] = [
        _chunk_ref(chunk, score) for chunk, score in outcome.items]
    selected = [(chunk, score) for chunk, score in outcome.items][:top_k]

    # Final gate. Reranking reorders authorized material and must never be able
    # to reintroduce anything unauthorized.
    verified = []
    for chunk, score in selected:
        decision = _engine.can_access(user, chunk.metadata)
        if decision.allow and (not coverage_sellers
                               or chunk.metadata.get("seller_id") in coverage_sellers) and (
                not portfolio_only or answer_scope.safe_passage(chunk, sellers)):
            verified.append((chunk, score))
        else:
            stage_trace["final_gate_denied"].append(_chunk_ref(chunk, score))
            logger.error("request=%s chunk=%s dropped at final gate: %s",
                         request_id, chunk.chunk_id,
                         decision.reason if not decision.allow else "answer_scope_mismatch")

    if not verified:
        logger.info("request=%s user=%s status=below_threshold", request_id,
                    user.user_id)
        return None, _refusal(request_id, user, question, started, sellers,
                              missing_sellers=coverage_sellers if portfolio_only else [],
                              candidate_document_ids=candidate_ids,
                              security_trace=stage_trace,
                              reranked=outcome.applied, **stats)

    check_deadline()
    context_started = time.monotonic()
    if coverage_sellers:
        context_block, sources = context.build_balanced(verified, coverage_sellers, cfg)
        included = {source.seller_id for source in sources}
        stage_trace["seller_coverage"] = {
            seller: seller in included for seller in coverage_sellers}
    else:
        context_block, sources = context.build(verified, cfg)
    stage_trace["stage_latency_ms"]["context"] = round(
        (time.monotonic() - context_started) * 1000)
    if not context_block.strip():
        return None, _refusal(request_id, user, question, started, sellers,
                              missing_sellers=coverage_sellers if portfolio_only else [],
                              candidate_document_ids=candidate_ids,
                              security_trace=stage_trace,
                              reranked=outcome.applied, **stats)

    stage_trace["context_spans"] = _context_spans(context_block, sources, verified)
    stage_trace["emitted_citations"] = [source.to_dict() for source in sources]

    messages = prompts.build_messages(
        prompts.SYSTEM_PROMPT + (prompts.seller_coverage_instruction(coverage_sellers, sources)
                                 if coverage_sellers else "")
        + (prompts.portfolio_instruction(sellers, sources) if portfolio_only else ""),
        conversation.render(turns),
        context_block,
        scoped_question,
    )
    return _Prepared(request_id, user, question, messages, sources, sellers,
                     stats, outcome.applied, started, context_block,
                     candidate_ids, stage_trace, coverage_sellers, portfolio_only), None


_EVIDENCE_REF = re.compile(r"\[(EV-[0-9a-f]{12})\]")


def _is_refusal(text: str) -> bool:
    lowered = " ".join(text.casefold().split())
    return (lowered == prompts.REFUSAL_TEXT.casefold()
            or lowered == config.REFUSAL_ANSWER.casefold()
            or lowered.startswith(("i cannot answer", "i can't answer",
                                   "i do not have enough information",
                                   "i don't have enough information",
                                   "i couldn't find sufficient information")))


def _usage_summary(*usage_rows: dict | None) -> dict:
    total = 0
    reported = False
    for row in usage_rows:
        if not isinstance(row, dict):
            continue
        value = row.get("total_tokens")
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            total += value
            reported = True
            continue
        for key in ("prompt_tokens", "completion_tokens"):
            value = row.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                total += value
                reported = True
    return {"reported": reported, "total_tokens": total}


def _coverage_note(prepared: _Prepared) -> str:
    return _missing_note(prepared.coverage_sellers, prepared.sources)


def _missing_note(sellers: list[str], sources: list) -> str:
    included = {source.seller_id for source in sources}
    missing = [seller for seller in sellers if seller not in included]
    return ("\n\n" + "\n".join(
        f"{seller}: Information unavailable in the sources included for this request."
        for seller in missing)) if missing else ""


def _finalize(prepared: _Prepared, answer_text: str,
              model: str | None, finish_reason: str = "stop",
              provider_refusal: bool = False,
              usage: dict | None = None) -> AnswerResponse:
    check_deadline()
    if getattr(prepared, "portfolio_only", False):
        included = {source.seller_id for source in prepared.sources}
        if not answer_scope.mentioned_sellers(answer_text) <= included:
            raise AnswerUnavailableError("The model returned an unsupported seller reference.")
    answer_text = _strip_model_sources(answer_text)
    if not answer_text.strip():
        raise AnswerUnavailableError("The model returned no substantive answer.")
    allowed_ids = {evidence_id for source in prepared.sources
                   for evidence_id in source.evidence_ids}
    supplied_ids = _EVIDENCE_REF.findall(answer_text)
    if any(evidence_id not in allowed_ids for evidence_id in supplied_ids):
        raise AnswerUnavailableError("The model returned an unknown evidence reference.")
    if any(not _EVIDENCE_REF.fullmatch(token) for token in
           re.findall(r"\[EV-[^\]\n]*(?:\]|(?=\n)|$)", answer_text)):
        raise AnswerUnavailableError("The model returned a malformed evidence reference.")
    # IDs identify the supplied evidence inventory, not a validated
    # claim-to-span entailment. Remove model-authored inline attribution.
    answer_text = _EVIDENCE_REF.sub("", answer_text).strip()
    if not answer_text:
        raise AnswerUnavailableError("The model returned no substantive answer.")
    if getattr(prepared, "portfolio_only", False):
        # This includes model-authored headings, tables and source labels. The
        # application alone appends missing-account notes and source inventory.
        if any(not answer_scope.mentioned_sellers(str(source.to_dict())) <= set(prepared.sellers)
               for source in prepared.sources):
            raise AnswerUnavailableError("The source inventory exceeded the portfolio scope.")
    is_refusal = _is_refusal(answer_text)
    prepared.security_trace["generation"] = {
        "model": model, "finish_reason": finish_reason,
        "usage": {key: value for key, value in (usage or {}).items()
                  if key in {"prompt_tokens", "completion_tokens", "total_tokens"}},
    }
    prepared.security_trace["model_usage"] = _usage_summary(
        *(prepared.security_trace.get("intent_usage") or []), usage or {})
    prepared.security_trace["retry_counts"] = retry_snapshot()
    prepared.security_trace.setdefault("stage_latency_ms", {})["total"] = int(
        (time.time() - prepared.started) * 1000)
    outcome = ("partial" if finish_reason == "length" else
               "refusal" if is_refusal or provider_refusal else "answer")
    response = AnswerResponse(
        request_id=prepared.request_id,
        user_id=prepared.user.user_id,
        query=prepared.question,
        answer=_with_sources(answer_text + _coverage_note(prepared), prepared.sources),
        outcome=outcome,
        answered=outcome in ("answer", "partial"),
        sources=[s.to_dict() for s in prepared.sources],
        selected=len(prepared.sources),
        reranked=prepared.reranked,
        effective_sellers=prepared.sellers,
        latency_ms=int((time.time() - prepared.started) * 1000),
        model=model,
        context=prepared.context,
        candidate_document_ids=prepared.candidate_document_ids,
        security_trace=prepared.security_trace,
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


def answer_query(principal, query, conversation_history=None,
                 seller_filter=None, top_k: int | None = None,
                 request_id: str | None = None,
                 cfg: config.AnsweringConfig | None = None) -> AnswerResponse:
    cfg = cfg or config.DEFAULT_ANSWERING
    request_id = request_id or uuid.uuid4().hex[:12]

    prepared, refusal = _prepare(principal, query, conversation_history,
                                 seller_filter, top_k, request_id, cfg)
    if refusal is not None:
        return refusal

    try:
        check_deadline()
        generation_started = time.monotonic()
        answer_text, usage = llm.generate(prepared.messages)
    except llm.LLMError as exc:
        logger.error("request=%s groq failure: %s", request_id, exc)
        raise AnswerUnavailableError(str(exc)) from exc
    prepared.security_trace.setdefault("stage_latency_ms", {})["generation"] = round(
        (time.monotonic() - generation_started) * 1000)

    # Citations are appended deterministically from the chunks that were
    # actually authorized and shown, so they cannot be hallucinated.
    return _finalize(prepared, answer_text, usage.get("model"),
                     usage.get("finish_reason", "stop"),
                     usage.get("provider_refusal", False), usage)


def answer_query_stream(principal, query, conversation_history=None,
                        seller_filter=None, top_k: int | None = None,
                        request_id: str | None = None,
                        cfg: config.AnsweringConfig | None = None):
    """Yield pipeline events. Authorization completes before the first token.

    Events: provisional meta -> delta* -> canonical done, or error when the
    provider stream is malformed, truncated, disconnected, or invalid.
    """
    cfg = cfg or config.DEFAULT_ANSWERING
    request_id = request_id or uuid.uuid4().hex[:12]

    prepared, refusal = _prepare(principal, query, conversation_history,
                                 seller_filter, top_k, request_id, cfg)
    if refusal is not None:
        payload = refusal.to_dict()
        yield {"type": "meta", **{k: payload[k] for k in
                                  ("request_id", "user_id", "answered", "outcome",
                                   "completed",
                                   "sources", "effective_sellers")},
               "stats": payload["stats"]}
        yield {"type": "delta", "text": refusal.answer}
        yield {"type": "done", **payload}
        return

    yield {
        "type": "meta",
        "request_id": prepared.request_id,
        "user_id": prepared.user.user_id,
        # This is provisional metadata. Only the terminal done event contains
        # a validated outcome, so clients cannot treat streamed deltas as final.
        "answered": False,
        "outcome": "pending",
        "completed": False,
        "sources": [s.to_dict() for s in prepared.sources],
        "effective_sellers": prepared.sellers,
        "stats": {**prepared.stats, "selected": len(prepared.sources),
                  "reranked": prepared.reranked},
    }

    pieces: list[str] = []
    suppressed = False
    stream = None
    generation_started = time.monotonic()
    try:
        check_deadline()
        stream = iter(llm.stream(prepared.messages))
        while True:
            try:
                delta = next(stream)
                check_deadline()
            except StopIteration as stopped:
                check_deadline()
                usage = stopped.value or {}
                break
            pieces.append(delta)
            joined = "".join(pieces)
            if "Sources:" in joined and _strip_model_sources(joined) != joined:
                suppressed = True
            if not suppressed and not getattr(prepared, "portfolio_only", False):
                yield {"type": "delta", "text": delta}
    except llm.LLMError as exc:
        logger.error("request=%s groq streaming failure: %s", request_id, exc)
        yield {"type": "error", "outcome": "error", "completed": False,
               "answered": False, "error": "AnswerUnavailableError",
               "message": AnswerUnavailableError.client_message}
        return
    except TimeoutError:
        yield {"type": "error", "outcome": "error", "completed": False,
               "answered": False, "error": "request_timeout",
               "message": "The request deadline was exceeded."}
        return
    finally:
        close = getattr(stream, "close", None)
        if close:
            close()

    prepared.security_trace.setdefault("stage_latency_ms", {})["generation"] = round(
        (time.monotonic() - generation_started) * 1000)

    try:
        response = _finalize(prepared, "".join(pieces),
                             usage.get("model", config.GROQ_MODEL),
                             usage.get("finish_reason", "stop"),
                             usage.get("provider_refusal", False), usage)
    except TimeoutError:
        yield {"type": "error", "outcome": "error", "completed": False,
               "answered": False, "error": "request_timeout",
               "message": "The request deadline was exceeded."}
        return
    except AnswerUnavailableError:
        yield {"type": "error", "outcome": "error", "completed": False,
               "answered": False, "error": "AnswerUnavailableError",
               "message": AnswerUnavailableError.client_message}
        return
    if prepared.portfolio_only:
        # Raw provider frames can contain a split forbidden label; no portfolio
        # text leaves the server until the full answer passes the same finalizer.
        yield {"type": "delta", "text": response.answer}
    else:
        note = _coverage_note(prepared)
        if note:
            yield {"type": "delta", "text": note}
        if prepared.sources:
            yield {"type": "delta", "text": _sources_block(prepared.sources)}
    yield {"type": "done", **response.to_dict(),
           "_token_usage": response.security_trace.get("model_usage")}
