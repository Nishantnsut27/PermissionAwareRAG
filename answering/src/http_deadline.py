"""Requests transport that interrupts active I/O at the request deadline.

Socket inactivity timeouts alone do not bound response headers or a steadily
arriving body. Register both newly connected and reused sockets before sending
HTTP data, then shut them down when the monotonic request budget expires.
"""
from contextlib import contextmanager
from contextvars import ContextVar

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool

from .deadline import DeadlineExceeded, SocketDeadline, check_deadline, remaining_timeout

_guard = ContextVar("provider_socket_deadline", default=None)


class _GuardedConnection:
    def send(self, data):
        if self.sock is None:
            self.connect()
        guard = _guard.get()
        if guard is not None:
            self._deadline_guard = guard
            guard.register(self.sock)
        return super().send(data)


class _HTTPConnection(_GuardedConnection, HTTPConnection):
    pass


class _HTTPSConnection(_GuardedConnection, HTTPSConnection):
    pass


class _ReleaseGuard:
    def _put_conn(self, conn):
        # A buffered response returns its connection before JSON validation.
        # Remove the timer's ownership before the pool lends it to another
        # caller, so a late deadline cannot cancel somebody else's request.
        guard = getattr(conn, "_deadline_guard", None)
        if guard is not None:
            guard.unregister(conn.sock)
            conn._deadline_guard = None
        return super()._put_conn(conn)


class _HTTPPool(_ReleaseGuard, HTTPConnectionPool):
    ConnectionCls = _HTTPConnection


class _HTTPSPool(_ReleaseGuard, HTTPSConnectionPool):
    ConnectionCls = _HTTPSConnection


class DeadlineAdapter(HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {"http": _HTTPPool, "https": _HTTPSPool}

    def proxy_manager_for(self, *args, **kwargs):
        manager = super().proxy_manager_for(*args, **kwargs)
        manager.pool_classes_by_scheme = {"http": _HTTPPool, "https": _HTTPSPool}
        return manager


def session() -> requests.Session:
    result = requests.Session()
    result.mount("http://", DeadlineAdapter())
    result.mount("https://", DeadlineAdapter())
    return result


@contextmanager
def provider_deadline(configured: float):
    guard = SocketDeadline(remaining_timeout(configured))
    token = _guard.set(guard)
    try:
        yield
        if guard.expired:
            raise DeadlineExceeded("provider wall-clock deadline exceeded")
        check_deadline()
    finally:
        guard.cancel()
        _guard.reset(token)
        if guard.expired:
            raise DeadlineExceeded("provider wall-clock deadline exceeded")
