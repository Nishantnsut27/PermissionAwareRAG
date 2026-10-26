"""OpenJEV System One client for structured query-understanding judgments.

OpenJEV is used ONLY for query understanding. It receives the raw query text
and returns typed judgments; it never receives identity, scope or permission
information and never makes an authorization decision. Authorization stays
entirely inside the server-side PermissionEngine.

One endpoint, one request: the query is the shared `state` and every question
is a typed judgment (choice / score / noul) evaluated in isolation. A question
asks about the query itself, so the state carries only the query text.

Error semantics follow the OpenJEV API documentation:
  401  authentication  -> permanent for this call, no retry
  422  invalid request -> permanent for this call, no retry
  429  rate limited    -> retryable, honours Retry-After
  503  unavailable     -> retryable with backoff
Other non-2xx statuses and network failures are treated as retryable once and
then permanent. Every failure surfaces as OpenJEVError so the planner can fall
back to a safe, deterministic plan instead of fabricating semantics.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from . import config

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
MAX_QUESTIONS = 255
MAX_SCORE_LEVELS = 10

CHOICE = "choice"
SCORE = "score"
NOUL = "noul"


class OpenJEVError(RuntimeError):
    """OpenJEV call failed after its retry budget. Safe to log, no secrets."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Question:
    """One typed OpenJEV question.

    `kind` selects the primitive: choice picks one option, score picks a
    position on an ordered rubric, noul answers a yes/no judgment with a
    probability. Criteria are the named options or ordered levels.
    """
    kind: str
    instructions: str
    criteria: "list | dict[str, str] | None" = None

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {
            "type": self.kind,
            "instructions": self.instructions,
        }
        if self.criteria is not None:
            if self.kind == SCORE:
                payload["criteria"] = [
                    str(level) for level in self.criteria]
            elif isinstance(self.criteria, dict):
                payload["criteria"] = self.criteria
            else:
                named: dict[str, str] = {}
                for option in self.criteria:
                    if isinstance(option, (tuple, list)) and len(option) == 2:
                        named[str(option[0])] = str(option[1])
                    else:
                        named[str(option)] = str(option)
                payload["criteria"] = named
        return payload


@dataclass
class Answer:
    """One typed answer from the OpenJEV response."""
    kind: str = ""
    choice: str | None = None
    score: float | None = None
    noul: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    def value(self) -> str | float | None:
        if self.kind == CHOICE:
            return self.choice
        if self.kind == SCORE:
            return self.score
        if self.kind == NOUL:
            return self.noul
        return None


@dataclass
class SystemOneResult:
    answers: dict[str, Answer] = field(default_factory=dict)
    latency_ms: int = 0


def _validate_question(name: str, question: Question) -> None:
    if question.kind not in (CHOICE, SCORE, NOUL):
        raise OpenJEVError(f"question {name!r}: unknown type {question.kind!r}")
    if not question.instructions.strip():
        raise OpenJEVError(f"question {name!r}: empty instructions")
    if question.kind in (CHOICE, SCORE):
        if not question.criteria:
            raise OpenJEVError(f"question {name!r}: {question.kind} needs criteria")
        options = len(question.to_dict()["criteria"] or {})
        if question.kind == CHOICE and options > MAX_QUESTIONS:
            raise OpenJEVError(f"question {name!r}: more than {MAX_QUESTIONS} options")
        if question.kind == SCORE and options > MAX_SCORE_LEVELS:
            raise OpenJEVError(f"question {name!r}: more than {MAX_SCORE_LEVELS} levels")
    if not question.instructions.strip():
        raise OpenJEVError(f"question {name!r}: empty instructions")


def _parse_answer(name: str, raw: object) -> Answer:
    if not isinstance(raw, dict):
        raise OpenJEVError(f"answer {name!r} is not an object")
    kind = raw.get("type")
    answer = Answer(kind=str(kind or ""))
    if answer.kind not in (CHOICE, SCORE, NOUL):
        raise OpenJEVError(f"answer {name!r}: unknown type {kind!r}")
    probabilities = raw.get("probabilities")
    if isinstance(probabilities, dict):
        for key, value in probabilities.items():
            try:
                answer.probabilities[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
    confidence = raw.get("confidence")
    if confidence is not None:
        try:
            answer.confidence = float(confidence)
        except (TypeError, ValueError):
            answer.confidence = None
    if answer.kind == CHOICE:
        choice = raw.get("choice")
        if not isinstance(choice, str) or not choice:
            raise OpenJEVError(f"answer {name!r}: missing choice")
        answer.choice = choice
    elif answer.kind == SCORE:
        score = raw.get("score")
        if score is None:
            raise OpenJEVError(f"answer {name!r}: missing score")
        try:
            answer.score = float(score)
        except (TypeError, ValueError):
            raise OpenJEVError(f"answer {name!r}: score is not numeric") from None
    else:
        noul = raw.get("noul")
        if noul is None:
            raise OpenJEVError(f"answer {name!r}: missing noul")
        try:
            answer.noul = float(noul)
        except (TypeError, ValueError):
            raise OpenJEVError(f"answer {name!r}: noul is not numeric") from None
    return answer


def _retry_after(response: requests.Response) -> float | None:
    try:
        value = float((response.headers or {}).get("Retry-After", ""))
    except (TypeError, ValueError):
        return None
    return value if 0 <= value <= config.OPENJEV_MAX_BACKOFF_S else None


def _request_once(state: Any, questions: dict[str, Question]) -> dict:
    if not config.OPENJEV_API_KEY:
        raise OpenJEVError("OPENJEV_API_KEY is not configured")
    if not questions:
        raise OpenJEVError("questions map must be non-empty")
    for name, question in questions.items():
        _validate_question(name, question)

    body = {
        "model": config.OPENJEV_MODEL,
        "state": state,
        "questions": {name: q.to_dict() for name, q in questions.items()},
    }
    try:
        response = requests.post(
            config.OPENJEV_BASE_URL,
            headers={
                "Authorization": f"Bearer {config.OPENJEV_API_KEY}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=config.OPENJEV_TIMEOUT_S,
        )
    except requests.Timeout as exc:
        raise OpenJEVError(f"OpenJEV timed out after {config.OPENJEV_TIMEOUT_S}s") from exc
    except requests.RequestException as exc:
        raise OpenJEVError(f"OpenJEV request failed: {type(exc).__name__}") from exc

    if response.status_code in RETRYABLE_STATUS:
        error = OpenJEVError(
            f"OpenJEV error {response.status_code}: {response.text[:200]}",
            response.status_code)
        error.wait = _retry_after(response)
        raise error
    if response.status_code == 401:
        raise OpenJEVError("OpenJEV rejected the API key (401)", 401)
    if response.status_code == 422:
        raise OpenJEVError(f"OpenJEV rejected the request (422): {response.text[:200]}", 422)
    if response.status_code != 200:
        raise OpenJEVError(
            f"OpenJEV error {response.status_code}: {response.text[:200]}",
            response.status_code)
    try:
        payload = response.json()
    except ValueError as exc:
        raise OpenJEVError("OpenJEV returned malformed JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), dict):
        raise OpenJEVError("OpenJEV response is missing the answers map")
    return payload


def system_one(state: Any, questions: dict[str, Question]) -> SystemOneResult:
    """Send one System One request and return parsed typed answers.

    Retries only retryable failures (429 / 503 / network), honouring
    Retry-After when present. Permanent failures (401 / 422 / other 4xx)
    surface immediately. A missing answer for a question that was asked is an
    error: the caller must not guess which judgments succeeded.
    """
    started = time.monotonic()
    payload: dict | None = None
    last_error: OpenJEVError | None = None
    for attempt in range(config.OPENJEV_MAX_RETRIES + 1):
        try:
            payload = _request_once(state, questions)
            break
        except OpenJEVError as exc:
            if exc.status is not None and exc.status not in RETRYABLE_STATUS:
                raise
            last_error = exc
            if attempt >= config.OPENJEV_MAX_RETRIES:
                break
            wait = getattr(exc, "wait", None)
            if wait is None:
                wait = min(config.OPENJEV_BACKOFF_S * (2 ** attempt),
                           config.OPENJEV_MAX_BACKOFF_S)
            logger.warning("OpenJEV retry %d/%d in %.1fs: %s", attempt + 1,
                           config.OPENJEV_MAX_RETRIES, wait, exc)
            time.sleep(wait)
    if payload is None:
        raise last_error or OpenJEVError("OpenJEV call failed")

    answers: dict[str, Answer] = {}
    for name in questions:
        if name not in payload["answers"]:
            raise OpenJEVError(f"OpenJEV returned no answer for {name!r}")
        answers[name] = _parse_answer(name, payload["answers"][name])
    return SystemOneResult(answers=answers,
                           latency_ms=int((time.monotonic() - started) * 1000))



