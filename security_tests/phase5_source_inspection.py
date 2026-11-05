"""Offline source-route checks; React rendering is covered in frontend/tests."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answering.src import api


class NoDirectSourceLookupTests(unittest.TestCase):
    def test_arbitrary_source_id_route_returns_not_found_without_lookup(self):
        handler = object.__new__(api.Handler)
        handler.path = "/sources/%3Cscript%3Ehostile%3C%2Fscript%3E"
        with patch.object(handler, "_send") as send:
            handler.do_GET()
        send.assert_called_once()
        self.assertEqual(send.call_args.args[0], 404)
        self.assertEqual(send.call_args.args[1], {"error": "not_found"})


if __name__ == "__main__":
    unittest.main()
