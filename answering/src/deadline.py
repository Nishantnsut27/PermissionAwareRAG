"""Per-request monotonic deadline propagated to provider calls."""
from __future__ import annotations

from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import dataclass, field
import threading
import time
import socket

_deadline: ContextVar[float | None] = ContextVar("answering_deadline", default=None)


@dataclass
class _RetryLedger:
    counts: dict[str, int] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)


_retry_ledger: ContextVar[_RetryLedger | None] = ContextVar(
    "answering_retry_ledger", default=None)


class DeadlineExceeded(TimeoutError):
    pass


@contextmanager
def request_deadline(seconds: float):
    parent = _deadline.get()
    token = _deadline.set(min(parent, time.monotonic() + seconds)
                          if parent is not None else time.monotonic() + seconds)
    ledger_token = _retry_ledger.set(_retry_ledger.get() or _RetryLedger())
    try:
        yield
    finally:
        _retry_ledger.reset(ledger_token)
        _deadline.reset(token)


def remaining_timeout(configured: float) -> float:
    deadline = _deadline.get()
    if deadline is None:
        return configured
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise DeadlineExceeded("request deadline exceeded")
    return min(configured, remaining)


def check_deadline() -> None:
    remaining_timeout(float("inf"))


class SocketDeadline:
    """Interrupt blocking socket I/O at an absolute deadline, even on trickles."""

    def __init__(self, seconds: float, read_only: bool = False):
        self.expired = False
        self._read_only = read_only
        self._cancelled = False
        self._sockets = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(seconds, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def register(self, sock) -> None:
        if sock is None:
            return
        with self._lock:
            if not self.expired:
                if sock not in self._sockets:
                    self._sockets.append(sock)
                return
        self._shutdown(sock)
        raise DeadlineExceeded("socket deadline exceeded")

    def _shutdown(self, sock):
        try:
            sock.shutdown(socket.SHUT_RD if self._read_only else socket.SHUT_RDWR)
        except OSError:
            pass

    def unregister(self, sock):
        with self._lock:
            if sock in self._sockets:
                self._sockets.remove(sock)

    def _expire(self):
        with self._lock:
            if self._cancelled:
                return
            self.expired = True
            for sock in self._sockets:
                self._shutdown(sock)

    def cancel(self):
        with self._lock:
            self._cancelled = True
            self._sockets.clear()
        self._timer.cancel()


def record_retry(provider: str) -> None:
    ledger = _retry_ledger.get()
    if ledger is None:
        return
    with ledger.lock:
        ledger.counts[provider] = ledger.counts.get(provider, 0) + 1


def retry_snapshot() -> dict[str, int]:
    ledger = _retry_ledger.get()
    if ledger is None:
        return {}
    with ledger.lock:
        return dict(ledger.counts)
