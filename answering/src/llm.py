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

logger = logging.getLogger(__name__)


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
            response = requests.post(
                config.GROQ_API_URL,
                headers=_headers(key),
                json=payload,
                timeout=config.GROQ_TIMEOUT_S,
                stream=stream,
            )
        except requests.RequestException as exc:
            if transient_retries < config.GROQ_MAX_RETRIES:
                time.sleep(2 ** transient_retries)
                transient_retries += 1
                continue
            raise _fail(f"Groq request failed: {exc}") from exc

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
                continue
            raise _fail(
                f"Groq API error {response.status_code} on every key: {body}")
        if kind == groq_keys.TRANSIENT:
            if transient_retries < config.GROQ_MAX_RETRIES:
                time.sleep(2 ** transient_retries)
                transient_retries += 1
                continue
            raise _fail(f"Groq API error {response.status_code}: {body}")
        raise _fail(f"Groq API error {response.status_code}: {body}")

    raise LLMError("Groq request exceeded the attempt budget.")


def generate(messages: list[dict], max_tokens: int | None = None,
             temperature: float | None = None,
             response_format: dict | None = None) -> tuple[str, dict]:
    """Return (answer_text, usage). Raises LLMError on failure."""
    payload = _payload(messages, max_tokens, temperature, response_format)
    last_error: Exception | None = None

    for attempt in range(config.GROQ_MAX_RETRIES + 1):
        response, _ = _post(payload, stream=False)
        try:
            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise _fail(f"Groq response has no choices: {str(body)[:200]}")
            text = (choices[0].get("message") or {}).get("content") or ""
            if not text.strip():
                raise LLMError("Groq returned an empty answer")
            usage = body.get("usage") or {}
            return text.strip(), {
                "model": body.get("model", config.GROQ_MODEL),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "finish_reason": choices[0].get("finish_reason"),
            }
        except (LLMError, ValueError) as exc:
            last_error = exc
            if attempt < config.GROQ_MAX_RETRIES:
                time.sleep(2 ** attempt)
        finally:
            response.close()

    logger.error("Groq generation failed after retries: %s",
                 groq_keys.pool().redact(last_error))
    raise _fail(last_error)


def stream(messages: list[dict]):
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
            with response:
                # requests falls back to ISO-8859-1 for text/* without an
                # explicit charset, which mangles non-ASCII answer text.
                response.encoding = "utf-8"
                for raw in response.iter_lines(decode_unicode=True):
                    if not raw or not raw.startswith("data:"):
                        continue
                    data = raw[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        event = json.loads(data)
                    except ValueError:
                        continue
                    choices = event.get("choices") or []
                    if not choices:
                        continue
                    delta = (choices[0].get("delta") or {}).get("content")
                    if delta:
                        emitted = True
                        yield delta
                return
        except (LLMError, requests.RequestException, ValueError) as exc:
            last_error = exc
            if emitted:
                break
            if attempt < config.GROQ_MAX_RETRIES:
                time.sleep(2 ** attempt)

    logger.error("Groq streaming failed: %s",
                 groq_keys.pool().redact(last_error))
    raise _fail(last_error)
