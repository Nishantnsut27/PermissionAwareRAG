"""Offline checks for source rendering and canonical UI completion states."""
from __future__ import annotations

import sys
import html
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app
from answering.src import api


class SourceRenderingTests(unittest.TestCase):
    def test_hostile_source_metadata_is_rendered_only_as_plain_text(self):
        expander = MagicMock()
        expander.__enter__.return_value = None
        fake_st = SimpleNamespace(
            expander=MagicMock(return_value=expander),
            text=MagicMock(), markdown=MagicMock())
        hostile = {
            "document_id": "<img src=x onerror=alert(1)>",
            "label": "[click](javascript:alert(1)) <script>alert(2)</script>",
            "seller_id": "S001</code><script>alert(3)</script>",
            "pages": ["<svg onload=alert(4)>", 3],
            "relevance": float("nan"),
        }
        with patch.object(app, "st", fake_st):
            app._render_sources([hostile])
        rendered = " ".join(str(call.args[0]) for call in fake_st.markdown.call_args_list)
        for value in (hostile["document_id"], hostile["label"], hostile["seller_id"], hostile["pages"][0]):
            self.assertIn(html.escape(value), rendered)
        self.assertNotIn("<script>", rendered)
        self.assertNotIn("<img ", rendered)
        self.assertNotIn("<svg ", rendered)
        self.assertNotIn("Relevance: nan", rendered)
        fake_st.text.assert_not_called()

    def test_malformed_source_rows_are_ignored(self):
        expander = MagicMock()
        expander.__enter__.return_value = None
        fake_st = SimpleNamespace(expander=MagicMock(return_value=expander),
                                  markdown=MagicMock())
        with patch.object(app, "st", fake_st):
            app._render_sources([None, "not a source row", {"pages": "bad"}])
        self.assertEqual(fake_st.markdown.call_count, 1)
        self.assertIn("Untitled source", fake_st.markdown.call_args.args[0])
        self.assertTrue(all("bad" not in str(call.args[0])
                            for call in fake_st.markdown.call_args_list))


class CanonicalCompletionTests(unittest.TestCase):
    def _stream(self, events):
        with patch.object(app.api_client, "ask_demo_stream",
                          return_value=iter(events)), \
             patch.object(app.st, "error"):
            return app._stream_answer({"user_id": "U-001"}, "http://local",
                                     "question", None, [])

    def test_stream_terminal_outcomes_are_distinct(self):
        cases = [
            ([{"type": "done", "outcome": "answer", "answer": "yes"}], "answer"),
            ([{"type": "done", "outcome": "partial", "answer": "part"}], "partial"),
            ([{"type": "done", "outcome": "refusal", "answer": ""}], "refusal"),
            ([{"type": "error", "message": "provider failed"}], "error"),
            ([{"type": "delta", "text": "not final"}], "interrupted"),
            ([{"type": "done", "outcome": "complete", "answer": "bad"}], "error"),
        ]
        for events, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(self._stream(events)["outcome"], expected)

    def test_completion_labels_name_the_canonical_state(self):
        self.assertIn("interrupted", app._completion_label("interrupted"))
        self.assertIn("Partial answer", app._completion_label("partial"))
        self.assertIn("refused", app._completion_label("refusal"))
        self.assertIn("failed", app._completion_label("error"))


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
