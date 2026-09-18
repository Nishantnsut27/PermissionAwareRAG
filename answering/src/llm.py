"""Groq chat completion client (buffered and streaming)."""
from __future__ import annotations

import json
import logging
import time

import requests

from . import config

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Answer generation failed. The message is safe to log, not to return."""


def _headers() -> dict[str, str]:
    if not config.GROQ_API_KEY:
        raise LLMError("GROQ_API_KEY is not set; refusing to call the LLM.")
    return {
        "Authorization": f"Bearer {config.GROQ_API_KEY}",
        "Content-Type": "application/json",
    }


def _payload(messages: list[dict]) -> dict:
    payload = {
        "model": config.GROQ_MODEL,
        "messages": messages,
        "temperature": config.GROQ_TEMPERATURE,
        "max_tokens": config.GROQ_MAX_TOKENS,
    }
    if config.GROQ_REASONING_EFFORT:
        payload["reasoning_effort"] = config.GROQ_REASONING_EFFORT
    return payload


def generate(messages: list[dict]) -> tuple[str, dict]:
    """Return (answer_text, usage). Raises LLMError on failure."""
    last_error: Exception | None = None
    for attempt in range(config.GROQ_MAX_RETRIES + 1):
        try:
            response = requests.post(
                config.GROQ_API_URL,
                headers=_headers(),
                json=_payload(messages),
                timeout=config.GROQ_TIMEOUT_S,
            )
            if response.status_code != 200:
                raise LLMError(
                    f"Groq API error {response.status_code}: "
                    f"{response.text[:300]}")
            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise LLMError(f"Groq response has no choices: {str(body)[:200]}")
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
        except (LLMError, requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt < config.GROQ_MAX_RETRIES:
                time.sleep(2 ** attempt)
    logger.error("Groq generation failed after retries: %s", last_error)
    raise LLMError(str(last_error))


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
            with requests.post(
                config.GROQ_API_URL,
                headers=_headers(),
                json=payload,
                timeout=config.GROQ_TIMEOUT_S,
                stream=True,
            ) as response:
                if response.status_code != 200:
                    raise LLMError(
                        f"Groq API error {response.status_code}: "
                        f"{response.text[:300]}")
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

    logger.error("Groq streaming failed: %s", last_error)
    raise LLMError(str(last_error))
