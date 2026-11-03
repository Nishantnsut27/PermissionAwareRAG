"""Offline UI startup and tiny uptime-probe regressions."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import requests
from streamlit.testing.v1 import AppTest

import app
from answering.src import api
from security_tests.phase1_identity_boundary import api_server


USER = {"user_id": "U-001", "name": "Aditya Verma", "role": "Account Manager",
        "department": "Business", "seller_scope": ["S001", "S003"]}
APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class StartupTests(unittest.TestCase):
    def setUp(self):
        app.st.cache_data.clear()
        self.addCleanup(app.st.cache_data.clear)
        for name in ("health", "readiness", "evaluation_summary"):
            mock = self.enterContext(patch.object(
                app.api_client, name,
                side_effect=AssertionError(f"Unexpected startup call: {name}")))
            setattr(self, name, mock)

    def ui(self):
        return AppTest.from_file(str(APP_PATH), default_timeout=10)

    def test_login_and_theme_reruns_reuse_the_public_directory(self):
        with patch.object(app.api_client, "list_demo_users", return_value=[USER]) as users:
            at = self.ui().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.selectbox[0].value, "U-001")
            at.toggle[0].set_value(True).run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["theme"], "dark")
            users.assert_called_once()
        self.readiness.assert_not_called()
        self.health.assert_not_called()
        self.evaluation_summary.assert_not_called()

    def test_sleeping_backend_shows_login_shell_and_can_retry(self):
        with patch.object(app.api_client, "list_demo_users", side_effect=[
                app.api_client.ApiError("Service waking up"), [USER]]) as users:
            at = self.ui().run()
            self.assertFalse(at.exception)
            self.assertTrue(at.error)
            self.assertTrue(any("Explore a sample account" in row.value for row in at.markdown))
            at.button(key="retry_demo_users").click().run()
            self.assertFalse(at.exception)
            self.assertFalse(at.error)
            self.assertEqual(at.selectbox[0].value, "U-001")
            self.assertEqual(users.call_count, 2)

    def test_chat_controls_do_not_wait_for_backend_diagnostics(self):
        with patch.object(app.api_client, "list_demo_users", return_value=[USER]), \
             patch.object(app.api_client, "demo_login", return_value={
                 "user": USER, "access_token": "test-session"}) as login:
            at = self.ui().run()
            at.text_input[0].input("aditya@login")
            next(button for button in at.button if button.label == "Continue").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(at.session_state["demo_access_token"], "test-session")
            self.assertTrue(at.chat_input)
            at.selectbox(key="seller_select").select("S001 - Aurelia Home Decor").run()
            at.toggle[0].set_value(True).run()
            self.assertFalse(at.exception)
            login.assert_called_once()
        self.readiness.assert_not_called()
        self.health.assert_not_called()
        self.evaluation_summary.assert_not_called()

    def test_directory_cache_is_separate_for_each_backend(self):
        with patch.object(app.api_client, "list_demo_users", return_value=[USER]) as users:
            app._demo_users("http://one")
            app._demo_users("http://one")
            app._demo_users("http://two")
        self.assertEqual(users.call_count, 2)

    def test_service_diagnostic_runs_only_when_requested(self):
        self.readiness.side_effect = None
        self.readiness.return_value = {"reachable": True, "ready": True, "problems": []}
        with patch.object(app.api_client, "list_demo_users", return_value=[USER]):
            at = self.ui().run()
            self.readiness.assert_not_called()
            at.button(key="check_service").click().run()
            self.assertFalse(at.exception)
            self.readiness.assert_called_once()
            self.assertTrue(any("Backend and knowledge index ready" in row.value for row in at.markdown))
            at.run()
            self.readiness.assert_called_once()
            at.button(key="check_service").click().run()
            self.assertFalse(at.exception)
            self.assertEqual(self.readiness.call_count, 2)

    def test_evaluation_is_loaded_on_demand_and_refresh_invalidates_cache(self):
        self.evaluation_summary.side_effect = None
        self.evaluation_summary.return_value = {
            "has_results": True, "total_cases": 4, "run_id": "saved"}
        at = self.ui()
        at.session_state["user"] = USER
        at.session_state["demo_login_version"] = app.DEMO_LOGIN_VERSION
        at.session_state["view"] = "chat"
        at.run()
        self.evaluation_summary.assert_not_called()
        at.button(key="eval_toggle").click().run()
        self.assertFalse(at.exception)
        self.evaluation_summary.assert_called_once()
        at.run()
        self.evaluation_summary.assert_called_once()
        next(button for button in at.button if button.label == "Refresh saved results").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(self.evaluation_summary.call_count, 2)


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
