from __future__ import annotations

import json
import os
import threading
import unittest
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from unittest.mock import patch

import requests

from answering.src import api, config, identity, service
from access_control.src.users import get_user


class FakeResponse:
    def __init__(self, claims):
        self.claims = claims

    def raise_for_status(self):
        return None

    def json(self):
        return self.claims


def claims(**updates):
    value = {"active": True, "exp": 5000, "iss": "https://id.example",
             "aud": "rag-api", "sub": "person-1"}
    value.update(updates)
    return value


@contextmanager
def api_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), api.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class IdentityProviderTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {
            "OIDC_INTROSPECTION_URL": "https://id.example/introspect",
            "OIDC_CLIENT_ID": "rag", "OIDC_CLIENT_SECRET": "test-only",
            "OIDC_ISSUER": "https://id.example", "OIDC_AUDIENCE": "rag-api",
            "OIDC_SUBJECT_MAP": json.dumps({"person-1": "U-001"}),
        })
        self.env.start()

    def tearDown(self):
        self.env.stop()

    def authenticate_claims(self, value):
        return identity.authenticate(
            "Bearer test-token", post=lambda *a, **k: FakeResponse(value), now=100)

    def test_missing_credential_is_rejected(self):
        with self.assertRaises(identity.AuthenticationError):
            identity.authenticate(None)

    def test_invalid_credential_is_rejected(self):
        with self.assertRaises(identity.AuthenticationError):
            self.authenticate_claims(claims(active=False))

    def test_expired_credential_is_rejected(self):
        with self.assertRaises(identity.AuthenticationError):
            self.authenticate_claims(claims(exp=99))

    def test_wrong_audience_is_rejected(self):
        with self.assertRaises(identity.AuthenticationError):
            self.authenticate_claims(claims(aud="other-api"))


class ApiBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.demo = patch.object(config, "API_DEMO_MODE", False)
        self.demo.start()

    def tearDown(self):
        self.demo.stop()

    def auth(self, token):
        if token == "Bearer normal":
            return identity.Principal(get_user("U-001"), "normal-sub")
        if token == "Bearer admin":
            return identity.Principal(get_user("U-005"), "admin-sub")
        raise identity.AuthenticationError("invalid credential")

    def test_anonymous_protected_query_fails_closed_without_provider(self):
        with patch.dict(os.environ, {k: "" for k in (
                "OIDC_INTROSPECTION_URL", "OIDC_CLIENT_ID", "OIDC_CLIENT_SECRET",
                "OIDC_ISSUER", "OIDC_AUDIENCE", "OIDC_SUBJECT_MAP")}):
            with api_server() as base:
                response = requests.post(base + "/query", json={"query": "hello"})
                users = requests.get(base + "/users")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"], "identity_provider_unavailable")
        self.assertEqual(users.status_code, 503)

    def test_normal_principal_wins_over_body_admin_id(self):
        result = service.AnswerResponse("id", "U-001", "q", "a", True)
        with patch.object(api.Handler, "_authenticate", lambda h: self.auth("Bearer normal")), \
             patch.object(service, "answer_query", return_value=result) as answer:
            with api_server() as base:
                response = requests.post(base + "/query", json={
                    "user_id": "U-005", "query": "hello"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(answer.call_args.kwargs["principal"].user.user_id, "U-001")

    def test_json_and_sse_use_same_verified_principal(self):
        result = service.AnswerResponse("id", "U-001", "q", "a", True)
        events = [{"type": "meta", "user_id": "U-001"},
                  {"type": "done", "user_id": "U-001"}]
        with patch.object(api.Handler, "_authenticate", lambda h: self.auth("Bearer normal")), \
             patch.object(service, "answer_query", return_value=result) as answer, \
             patch.object(service, "answer_query_stream", return_value=iter(events)) as stream:
            with api_server() as base:
                json_response = requests.post(base + "/query", json={"user_id": "U-005", "query": "q"},
                                              headers={"Authorization": "Bearer normal"})
                sse_response = requests.post(base + "/query/stream", json={"user_id": "U-005", "query": "q"},
                                             headers={"Authorization": "Bearer normal", "Accept": "text/event-stream"})
        self.assertEqual(json_response.status_code, 200)
        self.assertEqual(sse_response.status_code, 200)
        self.assertEqual(answer.call_args.kwargs["principal"].user.user_id, "U-001")
        self.assertEqual(stream.call_args.kwargs["principal"].user.user_id, "U-001")
        self.assertIn('"user_id": "U-001"', sse_response.text)

    def test_invalid_credential_is_rejected_for_sse_and_users(self):
        with patch.object(identity, "authenticate", side_effect=identity.AuthenticationError(
                "invalid credential")):
            with api_server() as base:
                query = requests.post(base + "/query/stream", json={"query": "q"},
                                      headers={"Authorization": "Bearer invalid"})
                users = requests.get(base + "/users", headers={"Authorization": "Bearer invalid"})
        self.assertEqual(query.status_code, 401)
        self.assertEqual(users.status_code, 401)

    def test_report_matrix_requires_operator_even_when_feature_enabled(self):
        document = {"results": [{"security_trace": {
            "raw_retrieval_candidates": [{"document_id": "S004-SECRET"}]}}]}
        reports = type("Reports", (), {
            "load": staticmethod(lambda _: document),
            "summary": staticmethod(lambda _: {"run_id": "r"}),
        })
        with patch.object(config, "EVALUATION_REPORTS_ENABLED", True), \
             patch.object(api.Handler, "_authenticate", lambda h: self.auth("Bearer normal")), \
             patch.object(api, "_evaluation", return_value=(None, reports)):
            with api_server() as base:
                response = requests.get(base + "/evaluation/matrix")
        self.assertEqual(response.status_code, 403)
        self.assertNotIn("S004-SECRET", response.text)

    def test_operator_can_read_detail_only_when_feature_enabled(self):
        document = {"summary": {}, "matrix": [{"answer": "sensitive"}],
                    "results": [{"answer": "sensitive", "evidence": ["private"],
                                 "security_trace": {"raw_retrieval_candidates": [
                                     {"document_id": "S004-SECRET"}]}}]}
        reports = type("Reports", (), {"load": staticmethod(lambda _: document),
                                       "summary": staticmethod(lambda _: {"run_id": "r"})})
        with patch.object(config, "EVALUATION_REPORTS_ENABLED", True), \
             patch.object(api.Handler, "_authenticate", lambda h: self.auth("Bearer admin")), \
             patch.object(api, "_evaluation", return_value=(None, reports)):
            with api_server() as base:
                response = requests.get(base + "/evaluation/matrix")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"], document["results"])

    def test_report_flag_alone_never_authorizes(self):
        document = {"results": [{"security_trace": {
            "raw_retrieval_candidates": [{"document_id": "S004-SECRET"}]}}]}
        reports = type("Reports", (), {
            "load": staticmethod(lambda _: document),
            "summary": staticmethod(lambda _: {"run_id": "r"}),
        })
        with patch.object(config, "EVALUATION_REPORTS_ENABLED", True), \
             patch.object(identity, "authenticate", side_effect=identity.AuthenticationError(
                 "missing")), \
             patch.object(api, "_evaluation", return_value=(None, reports)):
            with api_server() as base:
                response = requests.get(base + "/evaluation/matrix")
        self.assertEqual(response.status_code, 401)
        self.assertNotIn("S004-SECRET", response.text)

    def test_public_report_summary_contains_no_case_details(self):
        document = {"run_id": "run-1", "generated_at": "2026-10-31",
                    "summary": {"total_cases": 7},
                    "results": [{"answer": "private answer", "evidence": ["secret"]}]}
        reports = type("Reports", (), {
            "load": staticmethod(lambda _: document),
            "list_reports": staticmethod(lambda: [
                {"report_id": "runs/run-1.json", "generated_at": "2026-10-31",
                 "results": document["results"]}]),
        })
        with patch.object(api, "_evaluation", return_value=(None, reports)):
            with api_server() as base:
                response = requests.get(base + "/evaluation")
        self.assertEqual(response.status_code, 200)
        self.assertNotIn("results", response.json())
        self.assertNotIn("answer", response.text)
        self.assertNotIn("evidence", response.text)

    def test_api_refuses_remote_bind_without_provider(self):
        with patch.object(identity, "configured", return_value=False):
            with self.assertRaises(RuntimeError):
                api.serve("0.0.0.0", 8000)

    def test_demo_mode_always_refuses_remote_bind(self):
        with patch.object(config, "API_DEMO_MODE", True), \
             patch.object(identity, "configured", return_value=True):
            with self.assertRaises(RuntimeError):
                api.serve("0.0.0.0", 8000)


class DirectServiceBoundaryTests(unittest.TestCase):
    def test_direct_service_string_identity_is_rejected_before_pipeline(self):
        with self.assertRaises(service.InvalidUserError):
            service.answer_query("U-005", "hello")

    def test_direct_service_bare_user_is_rejected_before_pipeline(self):
        with self.assertRaises(service.InvalidUserError):
            service.answer_query(get_user("U-005"), "hello")

    def test_direct_stream_string_identity_is_rejected_before_pipeline(self):
        events = service.answer_query_stream("U-005", "hello")
        with self.assertRaises(service.InvalidUserError):
            next(events)


if __name__ == "__main__":
    unittest.main()
