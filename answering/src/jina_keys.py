"""Shared Jina key rotation and local, secret-free request diagnostics."""
from __future__ import annotations

import os
import re
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone

import requests


def load_keys(env=None):
    source = os.environ if env is None else env
    numbered = [(name, (source.get(name) or '').strip())
                for name in ('JINA_API_KEY_1', 'JINA_API_KEY_2')]
    candidates = numbered if any(value for _, value in numbered) else [
        ('JINA_API_KEY', (source.get('JINA_API_KEY') or '').strip())]
    seen, keys = set(), []
    for name, value in candidates:
        if value and value not in seen:
            seen.add(value)
            keys.append((name, value))
    return keys


class JinaKeyPool:
    def __init__(self, keys=None):
        self._keys = tuple(load_keys() if keys is None else keys)
        self._lock = threading.Lock()
        self._sequence = 0

    @property
    def size(self):
        return len(self._keys)

    def acquire(self):
        with self._lock:
            if not self._keys:
                raise RuntimeError('Configure JINA_API_KEY_1 and JINA_API_KEY_2 (or legacy JINA_API_KEY).')
            name, value = self._keys[self._sequence % len(self._keys)]
            self._sequence += 1
            return name, value, self._sequence

    def redact(self, value):
        text = str(value)
        for _, secret in sorted(self._keys, key=lambda item: len(item[1]), reverse=True):
            text = text.replace(secret, '[redacted]')
        return re.sub(r'(?i)Bearer\s+\S+', 'Bearer [redacted]', text)


_pool = None
_pool_lock = threading.Lock()


def pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            _pool = JinaKeyPool()
        return _pool


class JinaRequestError(RuntimeError):
    def __init__(self, message, *, status_code=None, retryable=False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class EvaluationRequests:
    calls: list[dict] = field(default_factory=list)
    failure: JinaRequestError | None = None
    lock: object = field(default_factory=threading.Lock, repr=False)

    def check(self):
        if self.failure is not None:
            raise self.failure from None


_evaluation = ContextVar('jina_evaluation_requests', default=None)


def strict_evaluation():
    return _evaluation.get() is not None


@contextmanager
def evaluation_requests():
    """The state is shared by worker threads through their copied context."""
    state = EvaluationRequests()
    token = _evaluation.set(state)
    try:
        yield state
    finally:
        _evaluation.reset(token)


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


@dataclass
class Request:
    headers: dict = field(repr=False)
    record: dict

    def response(self, status_code):
        self.record['http_status'] = status_code


@contextmanager
def request(operation):
    """Allocate once per outgoing request, including ordinary app retries.

    Evaluation disables retries and records the first failure for all sibling
    workers. The failure gate prevents queued work from issuing more calls.
    """
    keys = pool()
    state = _evaluation.get()
    gate = state.lock if state is not None else threading.Lock()
    with gate:
        if state is not None:
            state.check()
        label, value, sequence = keys.acquire()
        row = {'provider': 'jina', 'operation': operation, 'key_label': label,
               'sequence': sequence, 'started_at': _now(), 'http_status': None,
               'outcome': 'in_progress'}
        if state is not None:
            state.calls.append(row)
    call = Request({'Authorization': f'Bearer {value}', 'Content-Type': 'application/json'}, row)
    try:
        yield call
    except KeyboardInterrupt:
        row.update(outcome='interrupted')
        raise
    except Exception as exc:
        status = row['http_status']
        detail = keys.redact(f'{type(exc).__name__}: {exc}')[:600]
        message = (f'Jina {operation} failed using {label}: {detail}. '
                   f'Check/change {label} in .env, then rerun the same evaluation command.')
        error = JinaRequestError(message, status_code=status, retryable=(
            status in (408, 429) or (isinstance(status, int) and status >= 500)
            or (isinstance(exc, requests.RequestException) and not isinstance(exc, ValueError))))
        row.update(outcome='error', error=message)
        if state is not None:
            with state.lock:
                if state.failure is None:
                    state.failure = error
            raise error from None
        if isinstance(exc, TimeoutError):
            raise
        raise error from None
    else:
        row.update(outcome='success')
    finally:
        row['finished_at'] = _now()
