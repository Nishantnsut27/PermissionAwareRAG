"""Groq chat completion client (buffered and streaming).

A single call may be attempted on several keys: `groq_keys` decides which
failures justify moving to the next key and which must not. The number of
attempts is bounded by the pool size plus the configured retry budget, so a
persistent outage fails fast instead of cycling.
"""
from __future__ import annotations

import json
import logging
import time

import requests

from . import config, groq_keys
from .deadline import check_deadline, record_retry, remaining_timeout
from .http_deadline import provider_deadline, session

logger = logging.getLogger(__name__)
HTTP = session()


class LLMError(RuntimeError):
    """Answer generation failed. The message is safe to log, not to return."""


def _headers(key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _payload(messages: list[dict], max_tokens: int | None = None,
             temperature: float | None = None,
             response_format: dict | None = None) -> dict:
    payload = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "temperature": (config.GROQ_TEMPERATURE if temperature is None
                        else temperature),
        "max_tokens": max_tokens or config.GROQ_MAX_TOKENS,
    }
    if config.GROQ_REASONING_EFFORT:
        payload["reasoning_effort"] = config.GROQ_REASONING_EFFORT
    if response_format:
        payload["response_format"] = response_format
    return payload


def _fail(message: object) -> LLMError:
    return LLMError(groq_keys.pool().redact(message))


def _post(payload: dict, stream: bool):
    """POST to Groq, rotating keys only for key-specific failures.

    Returns the successful response together with the key label that served
    it. Raises LLMError once the attempt budget is spent.
    """
    keys = groq_keys.pool()
    exhausted: set[str] = set()
    transient_retries = 0
    max_key_attempts = max(1, keys.size)

    for _ in range(max_key_attempts + config.GROQ_MAX_RETRIES + 1):
        try:
            label, key = keys.acquire(exclude=exhausted)
        except groq_keys.NoAvailableKeyError as exc:
            raise LLMError(str(exc)) from exc

        try:
            response = HTTP.post(
                config.GROQ_API_URL,
                headers=_headers(key),
                json=payload,
                timeout=remaining_timeout(config.GROQ_TIMEOUT_S),
                stream=stream,
            )
        except requests.RequestException as exc:
            check_deadline()
            if transient_retries < config.GROQ_MAX_RETRIES:
                record_retry("generation")
                time.sleep(remaining_timeout(2 ** transient_retries))
                transient_retries += 1
                continue
            raise _fail(f"Groq request failed: {exc}") from exc

        try:
            check_deadline()
        except TimeoutError:
            response.close()
            raise
        if response.status_code == 200:
            keys.report_success(label)
            return response, label

        body = response.text[:300]
        kind = groq_keys.classify_status(response.status_code, body)
        response.close()

        if kind in (groq_keys.RATE_LIMIT, groq_keys.INVALID_KEY):
            keys.report_failure(label, kind)
            exhausted.add(label)
            if len(exhausted) < keys.size:
                record_retry("generation")
                continue
            raise _fail(
                f"Groq API error {response.status_code} on every key: {body}")
        if kind == groq_keys.TRANSIENT:
            if transient_retries < config.GROQ_MAX_RETRIES:
                record_retry("generation")
                time.sleep(remaining_timeout(2 ** transient_retries))
                transient_retries += 1
                continue
            raise _fail(f"Groq API error {response.status_code}: {body}")
        raise _fail(f"Groq API error {response.status_code}: {body}")

    raise LLMError("Groq request exceeded the attempt budget.")


def generate(messages: list[dict], max_tokens: int | None = None,
             temperature: float | None = None,
             response_format: dict | None = None) -> tuple[str, dict]:
    with provider_deadline(config.GROQ_TIMEOUT_S):
        return _generate(messages, max_tokens, temperature, response_format)


def _generate(messages: list[dict], max_tokens: int | None = None,
              temperature: float | None = None,
              response_format: dict | None = None) -> tuple[str, dict]:
    """Return (answer_text, usage). Raises LLMError on failure."""
    payload = _payload(messages, max_tokens, temperature, response_format)
    last_error: Exception | None = None

    for attempt in range(config.GROQ_MAX_RETRIES + 1):
        response, _ = _post(payload, stream=False)
        try:
            body = response.json()
            check_deadline()
            if not isinstance(body, dict):
                raise LLMError("Groq returned a malformed response")
            choices = body.get("choices") or []
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise LLMError("Groq response must contain exactly one valid choice")
            choice = choices[0]
            message = choice.get("message")
            if not isinstance(message, dict):
                raise LLMError("Groq response is missing its message")
            text = message.get("content")
            provider_refusal = message.get("refusal")
            if (not isinstance(text, str) or not text.strip()) and isinstance(provider_refusal, str):
                text = provider_refusal
            if not isinstance(text, str) or not text.strip():
                raise LLMError("Groq returned an empty answer")
            finish_reason = choice.get("finish_reason")
            if finish_reason not in ("stop", "length"):
                raise LLMError("Groq response has no valid terminal finish reason")
            usage = body.get("usage") or {}
            check_deadline()
            return text.strip(), {
                "model": body.get("model", config.GROQ_MODEL),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": finish_reason,
                "outcome": "partial" if finish_reason == "length" else "complete",
                "provider_refusal": bool(provider_refusal),
            }
        except (LLMError, ValueError, TypeError, AttributeError) as exc:
            last_error = exc
            # A malformed provider payload is not a transient failure. Do not
            # burn quota by repeating a permanent response error.
            break
        finally:
            response.close()

    logger.error("Groq generation failed after retries: %s",
                 groq_keys.pool().redact(last_error))
    raise _fail(last_error)


def stream(messages: list[dict]):
    with provider_deadline(config.GROQ_TIMEOUT_S):
        return (yield from _stream(messages))


def _stream(messages: list[dict]):
    """Yield answer text deltas as Groq produces them.

    Retries only while nothing has been emitted yet; once the caller has seen
    part of an answer, restarting would duplicate text, so the error surfaces.
    """
    payload = _payload(messages)
    payload["stream"] = True
    last_error: Exception | None = None

    for attempt in range(config.GROQ_MAX_RETRIES + 1):
        emitted = False
        try:
            response, _ = _post(payload, stream=True)
            finish_reason = None
            saw_done = False
            provider_refusal = False
            with response:
                # requests falls back to ISO-8859-1 for text/* without an
                # explicit charset, which mangles non-ASCII answer text.
                response.encoding = "utf-8"
                for raw in response.iter_lines(decode_unicode=True):
                    check_deadline()
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    try:
                        event = json.loads(data)
                    except ValueError:
                        raise LLMError("Groq stream contained malformed event data")
                    if not isinstance(event, dict):
                        raise LLMError("Groq stream event must be an object")
                    choices = event.get("choices") or []
                    if not choices:
                        # Provider usage-only frames are valid and carry no text.
                        if "usage" not in event:
                            raise LLMError("Groq stream event is missing choices")
                        continue
                    if (not isinstance(choices, list) or len(choices) != 1
                            or not isinstance(choices[0], dict)):
                        raise LLMError("Groq stream choice is malformed")
                    choice = choices[0]
                    already_finished = finish_reason is not None
                    current_finish = choice.get("finish_reason")
                    if current_finish is not None:
                        if current_finish not in ("stop", "length") or finish_reason is not None:
                            raise LLMError("Groq stream has an invalid finish reason")
                        finish_reason = current_finish
                    delta_obj = choice.get("delta") or {}
                    if not isinstance(delta_obj, dict):
                        raise LLMError("Groq stream delta is malformed")
                    delta = delta_obj.get("content")
                    refusal = delta_obj.get("refusal")
                    if refusal is not None and not isinstance(refusal, str):
                        raise LLMError("Groq stream refusal must be text")
                    provider_refusal = provider_refusal or bool(refusal)
                    if not delta and refusal:
                        delta = refusal
                    if delta is not None and not isinstance(delta, str):
                        raise LLMError("Groq stream content must be text")
                    if already_finished and delta:
                        raise LLMError("Groq stream emitted content after completion")
                    if delta:
                        emitted = True
                        yield delta
                check_deadline()
                if not saw_done or finish_reason is None:
                    raise LLMError("Groq stream ended before a valid terminal state")
            return {"model": config.GROQ_MODEL,
                    "finish_reason": finish_reason,
                    "outcome": "partial" if finish_reason == "length" else "complete",
                    "provider_refusal": provider_refusal}
        except (LLMError, requests.RequestException, ValueError) as exc:
            check_deadline()
            last_error = exc
            if (emitted or not isinstance(exc, requests.RequestException)
                    or attempt >= config.GROQ_MAX_RETRIES):
                break
            record_retry("generation")
            time.sleep(remaining_timeout(2 ** attempt))

    logger.error("Groq streaming failed: %s",
                 groq_keys.pool().redact(last_error))
    raise _fail(last_error)
