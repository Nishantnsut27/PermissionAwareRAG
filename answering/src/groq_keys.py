"""Groq API key pool with rate-limit-aware fallback.

One pool is shared by every Groq caller in the process (the answering pipeline
and the Phase 7 evaluation judge), so a key exhausted by one path is also
skipped by the other.

Rotation policy, which is the security- and quota-relevant part:

  * Only key-specific failures rotate: 429 / quota exhaustion (cooldown, key
    may recover) and 401 / 403 (disabled for the process, the key is bad).
  * 5xx and network errors are transient and retried on the SAME key, because
    rotating would burn the pool on a server-side outage.
  * Every other 4xx is an application error (bad payload, bad model name) and
    rotates nothing, because no other key would behave differently.

Key values are never logged, never returned and never rendered. Callers refer
to a key by its pool label ("key 2 of 5"), and `redact()` scrubs any key value
plus anything matching the Groq secret shape out of text that is about to be
logged or raised.
"""
from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import timezone
from email.utils import parsedate_to_datetime

logger = logging.getLogger(__name__)

MAX_NUMBERED_KEYS = 20
DEFAULT_COOLDOWN_S = 60.0

_SECRET_SHAPE = re.compile(r"gsk_[A-Za-z0-9]{10,}")
_REDACTED = "[redacted]"

RATE_LIMIT = "rate_limit"
INVALID_KEY = "invalid_key"
TRANSIENT = "transient"
APPLICATION = "application"

_RATE_LIMIT_MARKERS = (
    "rate limit", "rate_limit", "ratelimit", "quota", "too many requests",
    "insufficient_quota", "tokens per", "requests per", "tpm", "rpm",
    "capacity exceeded",
)


class NoAvailableKeyError(RuntimeError):
    """Every key in the pool is disabled or cooling down."""


def load_keys(env: dict | None = None) -> list[str]:
    """Collect keys from GROQ_API_KEY_1..N, then the legacy GROQ_API_KEY.

    Order is the fallback order. Duplicates are dropped so a key repeated in
    .env cannot make the pool look larger than it is.
    """
    source = os.environ if env is None else env
    names = [f"GROQ_API_KEY_{i}" for i in range(1, MAX_NUMBERED_KEYS + 1)]
    names.append("GROQ_API_KEY")
    keys: list[str] = []
    for name in names:
        value = (source.get(name) or "").strip()
        if value and value not in keys:
            keys.append(value)
    return keys


def redact(text: object, keys: list[str] | None = None) -> str:
    """Remove key values and Groq-shaped secrets from text."""
    result = str(text)
    for key in keys or []:
        if key:
            result = result.replace(key, _REDACTED)
    return _SECRET_SHAPE.sub(_REDACTED, result)


def classify_status(status: int, body: str = "") -> str:
    """Map an HTTP outcome onto a rotation decision."""
    lowered = (body or "").lower()
    if status == 429 or status == 402:
        return RATE_LIMIT
    if status in (401, 403):
        return INVALID_KEY
    if status >= 500 or status == 408:
        return TRANSIENT
    if any(marker in lowered for marker in _RATE_LIMIT_MARKERS):
        return RATE_LIMIT
    return APPLICATION


def _duration_seconds(value: str) -> float | None:
    """Read numeric seconds or provider durations such as 1h2m3.5s."""
    value = re.sub(r"\s+", "", value.lower())
    try:
        result = float(value)
    except ValueError:
        parts = re.findall(r"(\d+(?:\.\d+)?)(ms|[dhms])", value)
        if not parts or "".join(number + unit for number, unit in parts) != value:
            return None
        units = {"d": 86400, "h": 3600, "m": 60, "s": 1, "ms": .001}
        result = sum(float(number) * units[unit] for number, unit in parts)
    return result if math.isfinite(result) and result >= 0 else None


def cooldown_seconds(headers, body: str = "") -> float | None:
    """Honor provider retry hints without logging headers or credentials."""
    values = {str(key).lower(): str(value) for key, value in (headers or {}).items()}
    retry_after = values.get("retry-after", "")
    if retry_after:
        delay = _duration_seconds(retry_after)
        if delay is not None:
            return delay
        try:
            when = parsedate_to_datetime(retry_after)
            when = when if when.tzinfo else when.replace(tzinfo=timezone.utc)
            return max(0, when.timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            pass
    message = (body or "").lower()
    hint = re.search(r"try again in\s+((?:\d+(?:\.\d+)?\s*(?:ms|[dhms])\s*)+)", message)
    if hint:
        delay = _duration_seconds(hint.group(1))
        if delay is not None:
            return delay
    resources = [name for name in ("tokens", "requests") if name in message]
    delays = [_duration_seconds(values.get(f"x-ratelimit-reset-{name}", ""))
              for name in resources or ("tokens", "requests")]
    return max((delay for delay in delays if delay is not None), default=None)


@dataclass
class _KeyState:
    label: str
    value: str
    disabled: bool = False
    cooldown_until: float = 0.0
    failures: int = 0
    successes: int = 0


@dataclass
class PoolStatus:
    total: int
    available: int
    disabled: int
    cooling_down: int
    rotations: int
    per_key: list[dict] = field(default_factory=list)


class GroqKeyPool:
    def __init__(self, keys: list[str] | None = None,
                 cooldown_s: float = DEFAULT_COOLDOWN_S) -> None:
        values = keys if keys is not None else load_keys()
        self._states = [
            _KeyState(label=f"key-{index + 1}", value=value)
            for index, value in enumerate(values)
        ]
        self._cooldown_s = cooldown_s
        self._cursor = 0
        self._rotations = 0
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._states)

    @property
    def size(self) -> int:
        return len(self._states)

    @property
    def values(self) -> list[str]:
        return [state.value for state in self._states]

    def redact(self, text: object) -> str:
        return redact(text, self.values)

    def acquire(self, exclude: set[str] | None = None) -> tuple[str, str]:
        """Return (label, key) for the next usable key.

        Round-robins from the last used position so concurrent requests spread
        across the pool instead of hammering key 1.
        """
        exclude = exclude or set()
        with self._lock:
            if not self._states:
                raise NoAvailableKeyError(
                    "No Groq API key is configured. Set GROQ_API_KEY_1 "
                    "(optionally GROQ_API_KEY_2..N) in .env.")
            now = time.monotonic()
            count = len(self._states)
            for offset in range(count):
                index = (self._cursor + offset) % count
                state = self._states[index]
                if state.disabled or state.label in exclude:
                    continue
                if state.cooldown_until > now:
                    continue
                self._cursor = (index + 1) % count
                return state.label, state.value
            waits = [state.cooldown_until - now for state in self._states
                     if not state.disabled and state.cooldown_until > now]
            wait = f" Earliest retry in {math.ceil(min(waits))}s." if waits else ""
            raise NoAvailableKeyError(
                f"All {count} Groq API keys are rate-limited or unavailable.{wait}")

    def report_success(self, label: str) -> None:
        with self._lock:
            state = self._find(label)
            if state is None:
                return
            state.successes += 1
            state.failures = 0
            # An overlapping successful request must not cancel a newer 429.

    def report_failure(self, label: str, kind: str,
                       retry_after_s: float | None = None) -> None:
        """Apply the rotation policy for a key-specific failure."""
        with self._lock:
            state = self._find(label)
            if state is None:
                return
            state.failures += 1
            if kind == INVALID_KEY:
                state.disabled = True
                logger.warning("Groq %s rejected as invalid; disabled for "
                               "this process", state.label)
            elif kind == RATE_LIMIT:
                delay = self._cooldown_s
                if retry_after_s is not None and math.isfinite(retry_after_s):
                    delay = max(delay, retry_after_s)
                state.cooldown_until = max(state.cooldown_until, time.monotonic() + delay)
                logger.warning("Groq %s rate-limited; cooling down %.0fs",
                               state.label, delay)
            else:
                return
            self._rotations += 1

    def status(self) -> PoolStatus:
        """Pool health for diagnostics. Contains no key material."""
        with self._lock:
            now = time.monotonic()
            per_key = [{
                "label": state.label,
                "disabled": state.disabled,
                "cooling_down": state.cooldown_until > now,
                "cooldown_remaining_s": max(
                    0, round(state.cooldown_until - now)),
                "successes": state.successes,
                "consecutive_failures": state.failures,
            } for state in self._states]
            return PoolStatus(
                total=len(self._states),
                available=sum(1 for k in per_key
                              if not k["disabled"] and not k["cooling_down"]),
                disabled=sum(1 for k in per_key if k["disabled"]),
                cooling_down=sum(1 for k in per_key if k["cooling_down"]),
                rotations=self._rotations,
                per_key=per_key,
            )

    def reset(self) -> None:
        with self._lock:
            for state in self._states:
                state.disabled = False
                state.cooldown_until = 0.0
                state.failures = 0
            self._cursor = 0

    def _find(self, label: str) -> _KeyState | None:
        return next((s for s in self._states if s.label == label), None)


_pool: GroqKeyPool | None = None
_pool_lock = threading.Lock()


def pool() -> GroqKeyPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                try:
                    cooldown = float(os.getenv("GROQ_KEY_COOLDOWN_S", DEFAULT_COOLDOWN_S))
                except (TypeError, ValueError):
                    cooldown = DEFAULT_COOLDOWN_S
                _pool = GroqKeyPool(cooldown_s=cooldown)
                logger.info("Groq key pool initialised with %d key(s)",
                            _pool.size)
    return _pool


def set_pool(new_pool: GroqKeyPool | None) -> None:
    """Replace the process pool. Used by tests; not part of the request path."""
    global _pool
    with _pool_lock:
        _pool = new_pool
