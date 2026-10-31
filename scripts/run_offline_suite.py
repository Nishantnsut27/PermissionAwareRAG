"""Run an offline test module, rejecting external network requests by default.

Loopback remains available for the existing HTTP/deadline regressions. Provider
test doubles can still be patched normally. This guard is a test harness only.
"""
from __future__ import annotations

import ipaddress
import runpy
import socket
import sys
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlsplit

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _loopback(host):
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main():
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    original_request = requests.sessions.Session.request
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex

    def request(session, method, url, *args, **kwargs):
        if not _loopback(urlsplit(url).hostname):
            raise AssertionError("External network forbidden by offline runner")
        return original_request(session, method, url, *args, **kwargs)

    def connect(sock, address, *args, **kwargs):
        if not isinstance(address, tuple) or not _loopback(address[0]):
            raise AssertionError("External network forbidden by offline runner")
        return original_connect(sock, address, *args, **kwargs)

    def connect_ex(sock, address, *args, **kwargs):
        if not isinstance(address, tuple) or not _loopback(address[0]):
            raise AssertionError("External network forbidden by offline runner")
        return original_connect_ex(sock, address, *args, **kwargs)

    with patch.object(requests.sessions.Session, "request", request), \
         patch.object(socket.socket, "connect", connect), \
         patch.object(socket.socket, "connect_ex", connect_ex):
        runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
