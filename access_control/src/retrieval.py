"""Permission-aware retrieval: scope filter -> Qdrant -> re-check -> gate."""
from __future__ import annotations

import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from indexing.src.retrieval import (
    RetrievalResult,
    retrieve as phase3_retrieve,
)

from .audit import log_request
from .engine import PermissionEngine
from .policy import IdentityError
from .users import User, get_user

REFUSAL_MESSAGE = (
    "I couldn't find information available to you "
    "that can answer this request."
)

MAX_TOP_K = 20
CANDIDATE_MULTIPLIER = 4
MAX_CANDIDATES = 40

_engine = PermissionEngine()


class AuthorizationInvariantError(RuntimeError):
    """Raised if an unauthorized chunk ever reaches the LLM boundary."""


@dataclass
class AccessResponse:
    user_id: str
    query: str
    request_id: str
    chunks: list[RetrievalResult] = field(default_factory=list)
    withheld_count: int = 0
    refused: bool = False

    def build_llm_context(self) -> str:
        """The only sanctioned way to hand retrieved text to an LLM."""
        if self.refused or not self.chunks:
            return REFUSAL_MESSAGE
        return "\n\n".join(
            f"[{chunk.document_id} / {chunk.chunk_id}]\n{chunk.text}"
            for chunk in self.chunks
        )


def _resolve(user: User | str) -> User:
    if isinstance(user, str):
        return get_user(user)
    if isinstance(user, User):
        return user
    raise IdentityError(
        f"user must be a User or a user_id, got {type(user).__name__}")


def authorized_retrieve(user: User | str, query: str, top_k: int = 5,
                        request_id: str | None = None) -> AccessResponse:
    identity = _resolve(user)
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    top_k = max(1, min(int(top_k), MAX_TOP_K))
    request_id = request_id or uuid.uuid4().hex[:12]

    # Raises for a malformed identity rather than issuing an unfiltered query.
    scope = _engine.get_authorized_scope(identity)
    candidates = phase3_retrieve(
        query,
        top_k=min(max(top_k * CANDIDATE_MULTIPLIER, 10), MAX_CANDIDATES),
        filters=scope,
    )
    allowed, denied = _engine.authorize_chunks(identity, candidates, request_id)
    selected = allowed[:top_k]

    for chunk in selected:
        if not _engine.can_access(identity, chunk.metadata).allow:
            raise AuthorizationInvariantError(
                f"chunk {chunk.chunk_id} passed the filter but is not "
                f"authorized for {identity.user_id}")

    response = AccessResponse(
        user_id=identity.user_id,
        query=query,
        request_id=request_id,
        chunks=selected,
        withheld_count=len(denied),
        refused=not selected,
    )
    log_request(identity.user_id, request_id, len(candidates), len(selected),
                len(denied), response.refused)
    return response
