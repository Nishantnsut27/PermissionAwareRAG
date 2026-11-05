"""Offline API liveness checks independent of the React frontend."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import requests

from answering.src import api
from security_tests.phase1_identity_boundary import api_server


class PingTests(unittest.TestCase):
    def test_probe_is_tiny_and_independent_of_readiness_and_authentication(self):
        with patch.object(api.Handler, "_readiness", side_effect=AssertionError("index access")), \
             patch.object(api.Handler, "_authenticate", side_effect=AssertionError("authentication")), \
             patch.object(api, "_evaluation", side_effect=AssertionError("report access")), \
             api_server() as base:
            for path in ("/ping", "/ping/", "/ping?monitor=cron"):
                with self.subTest(path=path):
                    response = requests.get(base + path, timeout=3)
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.content, b"ok")
                    self.assertEqual(response.headers["Content-Length"], "2")
                    self.assertEqual(response.headers["Content-Type"], "text/plain; charset=utf-8")
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
            response = requests.head(base + "/ping", timeout=3)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, b"")
            self.assertEqual(response.headers["Content-Length"], "2")
            self.assertEqual(requests.head(base + "/unknown", timeout=3).status_code, 404)
            self.assertEqual(requests.get(base + "/health", timeout=3).json(), {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
