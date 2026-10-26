"""Phase 7 tests.

Covers the parts that must be correct without spending API quota: the Groq key
fallback policy (mocked transport), the metric definitions, the golden dataset
consistency with the permission engine, and the security scoring.

Run with:  .venv\\Scripts\\python.exe -m evaluation.tests.test_phase7
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

from answering.src import groq_keys, llm
from evaluation import dataset as ds
from evaluation import metrics


class FakeResponse:
    def __init__(self, status_code: int, body: str = "",
                 payload: dict | None = None) -> None:
        self.status_code = status_code
        self.text = body
        self._payload = payload or {}
        self.encoding = None
        self.closed = False

    def json(self) -> dict:
        return self._payload

    def close(self) -> None:
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _ok_payload(text: str = "hello") -> dict:
    return {
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "model": "test-model",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
    }


class KeyLoadingTest(unittest.TestCase):
    def test_numbered_keys_in_order_then_legacy(self):
        env = {"GROQ_API_KEY_2": "b", "GROQ_API_KEY_1": "a",
               "GROQ_API_KEY_3": "c", "GROQ_API_KEY": "legacy"}
        self.assertEqual(groq_keys.load_keys(env), ["a", "b", "c", "legacy"])

    def test_duplicates_and_blanks_are_dropped(self):
        env = {"GROQ_API_KEY_1": "a", "GROQ_API_KEY_2": "  ",
               "GROQ_API_KEY_3": "a", "GROQ_API_KEY": "a"}
        self.assertEqual(groq_keys.load_keys(env), ["a"])

    def test_no_keys_yields_empty_pool(self):
        self.assertEqual(groq_keys.load_keys({}), [])


class ClassificationTest(unittest.TestCase):
    def test_429_is_rate_limit(self):
        self.assertEqual(groq_keys.classify_status(429), groq_keys.RATE_LIMIT)

    def test_401_and_403_are_invalid_key(self):
        self.assertEqual(groq_keys.classify_status(401), groq_keys.INVALID_KEY)
        self.assertEqual(groq_keys.classify_status(403), groq_keys.INVALID_KEY)

    def test_5xx_is_transient_not_a_key_problem(self):
        self.assertEqual(groq_keys.classify_status(500), groq_keys.TRANSIENT)
        self.assertEqual(groq_keys.classify_status(503), groq_keys.TRANSIENT)

    def test_ordinary_4xx_is_an_application_error(self):
        self.assertEqual(groq_keys.classify_status(400, "bad model"),
                         groq_keys.APPLICATION)
        self.assertEqual(groq_keys.classify_status(404), groq_keys.APPLICATION)

    def test_quota_wording_without_429_is_still_rate_limit(self):
        self.assertEqual(
            groq_keys.classify_status(400, "Organization quota exceeded"),
            groq_keys.RATE_LIMIT)


class RedactionTest(unittest.TestCase):
    def test_key_values_are_removed(self):
        pool = groq_keys.GroqKeyPool(["gsk_supersecretvalue123456"])
        self.assertNotIn("supersecret",
                         pool.redact("failed with gsk_supersecretvalue123456"))

    def test_groq_shaped_secrets_are_removed_even_if_unknown(self):
        text = groq_keys.redact("token gsk_abcdefghijklmnop0123 leaked", [])
        self.assertNotIn("gsk_abcdefghijklmnop0123", text)
        self.assertIn("[redacted]", text)

    def test_pool_status_contains_no_key_material(self):
        pool = groq_keys.GroqKeyPool(["gsk_aaaaaaaaaaaaaaaa", "gsk_bbbbbbbbbbbbbbbb"])
        rendered = repr(pool.status())
        self.assertNotIn("gsk_", rendered)


class PoolBehaviourTest(unittest.TestCase):
    def test_rate_limited_key_is_skipped_until_cooldown_expires(self):
        pool = groq_keys.GroqKeyPool(["k1", "k2"], cooldown_s=60)
        label, key = pool.acquire()
        self.assertEqual(key, "k1")
        pool.report_failure(label, groq_keys.RATE_LIMIT)
        self.assertEqual(pool.acquire()[1], "k2")

    def test_invalid_key_is_disabled_permanently(self):
        pool = groq_keys.GroqKeyPool(["k1", "k2"], cooldown_s=0)
        pool.report_failure("key-1", groq_keys.INVALID_KEY)
        pool.reset()
        status = pool.status()
        self.assertEqual(status.disabled, 0)

    def test_transient_failure_does_not_rotate(self):
        pool = groq_keys.GroqKeyPool(["k1", "k2"], cooldown_s=60)
        pool.report_failure("key-1", groq_keys.TRANSIENT)
        self.assertEqual(pool.status().available, 2)
        self.assertEqual(pool.status().rotations, 0)

    def test_all_keys_unavailable_raises(self):
        pool = groq_keys.GroqKeyPool(["k1", "k2"], cooldown_s=600)
        pool.report_failure("key-1", groq_keys.RATE_LIMIT)
        pool.report_failure("key-2", groq_keys.RATE_LIMIT)
        with self.assertRaises(groq_keys.NoAvailableKeyError):
            pool.acquire()

    def test_empty_pool_raises_with_configuration_guidance(self):
        pool = groq_keys.GroqKeyPool([])
        with self.assertRaises(groq_keys.NoAvailableKeyError) as ctx:
            pool.acquire()
        self.assertIn("GROQ_API_KEY_1", str(ctx.exception))

    def test_success_clears_the_failure_streak(self):
        pool = groq_keys.GroqKeyPool(["k1"], cooldown_s=60)
        pool.report_failure("key-1", groq_keys.RATE_LIMIT)
        pool.report_success("key-1")
        self.assertEqual(pool.acquire()[1], "k1")


class KeyFallbackIntegrationTest(unittest.TestCase):
    """The rotation policy as seen through llm.generate, with a fake transport."""

    def setUp(self):
        self._saved = groq_keys._pool
        self.pool = groq_keys.GroqKeyPool(["k1", "k2", "k3"], cooldown_s=60)
        groq_keys.set_pool(self.pool)
        self.calls: list[str] = []

    def tearDown(self):
        groq_keys.set_pool(self._saved)

    def _transport(self, responses):
        queue = list(responses)

        def post(url, headers=None, json=None, timeout=None, stream=False):
            self.calls.append(headers["Authorization"].split()[-1])
            return queue.pop(0) if queue else FakeResponse(200, payload=_ok_payload())
        return post

    def test_primary_key_is_used_when_it_works(self):
        with mock.patch.object(requests, "post",
                               self._transport([FakeResponse(200, payload=_ok_payload("fine"))])):
            text, _ = llm.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(text, "fine")
        self.assertEqual(self.calls, ["k1"])

    def test_rate_limit_falls_back_to_the_next_key(self):
        responses = [FakeResponse(429, "rate limit reached"),
                     FakeResponse(200, payload=_ok_payload("recovered"))]
        with mock.patch.object(requests, "post", self._transport(responses)):
            text, _ = llm.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(text, "recovered")
        self.assertEqual(self.calls, ["k1", "k2"])

    def test_application_error_does_not_burn_any_key(self):
        responses = [FakeResponse(400, "model_not_found")]
        with mock.patch.object(requests, "post", self._transport(responses)):
            with self.assertRaises(llm.LLMError):
                llm.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(self.calls, ["k1"])
        self.assertEqual(self.pool.status().available, 3)

    def test_server_error_retries_the_same_key_then_gives_up(self):
        responses = [FakeResponse(500, "upstream"), FakeResponse(500, "upstream"),
                     FakeResponse(500, "upstream"), FakeResponse(500, "upstream")]
        with mock.patch.object(requests, "post", self._transport(responses)), \
                mock.patch("answering.src.llm.time.sleep"):
            with self.assertRaises(llm.LLMError):
                llm.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(set(self.calls), {"k1"})
        self.assertEqual(self.pool.status().available, 3)

    def test_keys_are_not_exhausted_beyond_the_pool_size(self):
        responses = [FakeResponse(429, "rate limit")] * 10
        with mock.patch.object(requests, "post", self._transport(responses)), \
                mock.patch("answering.src.llm.time.sleep"):
            with self.assertRaises(llm.LLMError):
                llm.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(sorted(set(self.calls)), ["k1", "k2", "k3"])

    def test_error_message_never_contains_a_key(self):
        pool = groq_keys.GroqKeyPool(["gsk_topsecretkeyvalue00"], cooldown_s=60)
        groq_keys.set_pool(pool)
        responses = [FakeResponse(400, "bad request for gsk_topsecretkeyvalue00")]
        with mock.patch.object(requests, "post", self._transport(responses)):
            with self.assertRaises(llm.LLMError) as ctx:
                llm.generate([{"role": "user", "content": "hi"}])
        self.assertNotIn("topsecret", str(ctx.exception))

    def test_invalid_key_is_disabled_and_the_next_key_serves(self):
        responses = [FakeResponse(401, "invalid api key"),
                     FakeResponse(200, payload=_ok_payload("ok"))]
        with mock.patch.object(requests, "post", self._transport(responses)):
            llm.generate([{"role": "user", "content": "hi"}])
        self.assertEqual(self.calls, ["k1", "k2"])
        self.assertEqual(self.pool.status().disabled, 1)


class RetrievalMetricTest(unittest.TestCase):
    def test_recall_counts_expected_documents_found(self):
        self.assertEqual(metrics.recall_at_k(["a", "b"], ["a", "c"]), 0.5)
        self.assertEqual(metrics.recall_at_k(["a"], ["a"]), 1.0)

    def test_precision_counts_retrieved_documents_that_were_wanted(self):
        self.assertAlmostEqual(
            metrics.precision_at_k(["a"], ["a", "b", "c"]), 1 / 3)

    def test_mrr_uses_the_rank_of_the_first_relevant_document(self):
        self.assertEqual(metrics.mean_reciprocal_rank(["b"], ["a", "b"]), 0.5)
        self.assertEqual(metrics.mean_reciprocal_rank(["a"], ["a", "b"]), 1.0)
        self.assertEqual(metrics.mean_reciprocal_rank(["z"], ["a", "b"]), 0.0)

    def test_empty_inputs_are_not_applicable_rather_than_zero(self):
        self.assertIsNone(metrics.recall_at_k([], ["a"]))
        self.assertIsNone(metrics.precision_at_k(["a"], []))
        self.assertIsNone(metrics.mean_reciprocal_rank([], []))

    def test_scoring_reports_missed_documents(self):
        scored = metrics.score_retrieval(["a", "b"], ["b", "c"])
        self.assertEqual(scored.missed, ["a"])
        self.assertEqual(scored.relevant_retrieved, ["b"])
        self.assertEqual(scored.k, 2)

    def test_not_applicable_leaves_every_metric_none(self):
        scored = metrics.score_retrieval([], ["a"], applicable=False)
        self.assertIsNone(scored.recall_at_k)
        self.assertIsNone(scored.precision_at_k)
        self.assertIsNone(scored.mrr)
        self.assertFalse(scored.applicable)


class SecurityMetricTest(unittest.TestCase):
    def test_clean_run_passes(self):
        scored = metrics.score_security(
            "answer", "answer", denied_documents={"X"}, forbidden_documents=[],
            context_documents=["A"], candidate_documents=["A", "B"])
        self.assertTrue(scored.passed)
        self.assertFalse(scored.unauthorized_retrieval)

    def test_denied_document_in_context_is_an_unauthorized_retrieval(self):
        scored = metrics.score_security(
            "refuse", "answer", denied_documents={"X"}, forbidden_documents=[],
            context_documents=["X"], candidate_documents=["X"])
        self.assertTrue(scored.unauthorized_retrieval)
        self.assertFalse(scored.passed)
        self.assertIn("X", scored.forbidden_documents_in_context)

    def test_denied_document_retrieved_but_not_shown_is_still_detected(self):
        scored = metrics.score_security(
            "refuse", "refuse", denied_documents={"X"}, forbidden_documents=[],
            context_documents=[], candidate_documents=["X"])
        self.assertTrue(scored.unauthorized_retrieval)
        self.assertEqual(scored.forbidden_documents_in_context, [])
        self.assertEqual(scored.forbidden_documents_retrieved, ["X"])

    def test_leakage_fails_even_with_a_clean_context(self):
        scored = metrics.score_security(
            "refuse", "answer", denied_documents={"X"}, forbidden_documents=[],
            context_documents=["A"], candidate_documents=["A"],
            leaked_evidence=["the S002 incident was caused by ..."])
        self.assertTrue(scored.answer_leakage)
        self.assertFalse(scored.passed)

    def test_refusing_instead_of_partially_answering_is_conservative(self):
        scored = metrics.score_security(
            "partial_answer", "refuse", denied_documents=set(),
            forbidden_documents=[], context_documents=[],
            candidate_documents=[])
        self.assertTrue(scored.behavior_match)


class AggregateTest(unittest.TestCase):
    def test_not_applicable_values_do_not_drag_the_mean_down(self):
        results = [
            {"status": "PASS", "category": "authorized_factual",
             "retrieval": {"recall_at_k": 1.0},
             "generation": {"groundedness": 1.0},
             "security": {"passed": True, "unauthorized_retrieval": False,
                          "answer_leakage": False, "behavior_match": True}},
            {"status": "PASS", "category": "unauthorized_access",
             "retrieval": {"recall_at_k": None},
             "generation": {"groundedness": None},
             "security": {"passed": True, "unauthorized_retrieval": False,
                          "answer_leakage": False, "behavior_match": True}},
        ]
        summary = metrics.aggregate(results)
        self.assertEqual(summary["retrieval"]["recall_at_k"], 1.0)
        self.assertEqual(summary["retrieval"]["scored_cases"], 1)
        self.assertEqual(summary["security"]["authorization_accuracy"], 1.0)
        self.assertEqual(summary["passed"], 2)

    def test_errors_are_counted_separately_from_failures(self):
        results = [
            {"status": "ERROR", "category": "authorized_factual",
             "error": "Groq unavailable"},
            {"status": "FAIL", "category": "authorized_factual",
             "security": {"passed": False, "unauthorized_retrieval": True,
                          "answer_leakage": False, "behavior_match": False}},
        ]
        summary = metrics.aggregate(results)
        self.assertEqual(summary["errors"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["passed"], 0)


class GoldenDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = ds.load()
        cls.catalog = ds.PermissionCatalog()

    def test_dataset_loads_sixty_cases(self):
        self.assertEqual(len(self.dataset), 60)

    def test_dataset_is_consistent_with_the_permission_engine(self):
        self.assertEqual(ds.validate(self.dataset, self.catalog), [])

    def test_every_required_category_is_represented(self):
        counts = self.dataset.category_counts()
        for category in ds.CATEGORIES:
            self.assertIn(category, counts, f"missing category {category}")

    def test_all_five_identities_are_exercised(self):
        self.assertEqual(len(self.dataset.user_counts()), 5)

    def test_eval_ids_are_unique(self):
        ids = [c.eval_id for c in self.dataset.cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_expected_sources_are_authorized_for_their_identity(self):
        for case in self.dataset.cases:
            authorized = self.catalog.authorized_documents(case.user_id)
            for document_id in case.expected_sources:
                self.assertIn(document_id, authorized,
                              f"{case.eval_id} expects unauthorized {document_id}")

    def test_forbidden_sources_are_genuinely_denied(self):
        for case in self.dataset.cases:
            denied = self.catalog.denied_documents(case.user_id)
            for document_id in case.forbidden_sources:
                self.assertIn(document_id, denied,
                              f"{case.eval_id} forbids readable {document_id}")

    def test_refusal_cases_name_the_documents_they_protect(self):
        for case in self.dataset.cases:
            if case.expected_behavior == "refuse":
                self.assertTrue(case.forbidden_sources,
                                f"{case.eval_id} has no forbidden_sources")

    def test_follow_up_cases_reference_an_earlier_case_of_the_same_user(self):
        for case in self.dataset.cases:
            if case.conversation_from:
                parent = self.dataset.by_id(case.conversation_from)
                self.assertIsNotNone(parent)
                self.assertEqual(parent.user_id, case.user_id)

    def test_injection_cases_target_documents_outside_the_scope(self):
        injections = [c for c in self.dataset.cases
                      if c.category == "prompt_injection"]
        self.assertGreaterEqual(len(injections), 5)
        for case in injections:
            self.assertEqual(case.expected_behavior, "refuse")
            self.assertTrue(case.forbidden_sources)


class PermissionCatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = ds.PermissionCatalog()

    def test_account_manager_cannot_read_incident_reports(self):
        self.assertNotIn("INC-2026-0312",
                         self.catalog.authorized_documents("U-001"))

    def test_seller_scope_blocks_documents_outside_the_portfolio(self):
        # Scope is checked before the role matrix, so the denial reason names
        # the portfolio rather than the document type.
        self.assertNotIn("INC-2026-0519",
                         self.catalog.authorized_documents("U-001"))
        self.assertEqual(
            self.catalog.denial_reason("U-001", "INC-2026-0519"),
            "seller_scope_mismatch")

    def test_role_matrix_blocks_a_type_inside_an_authorized_portfolio(self):
        self.assertNotIn("INC-2026-0312",
                         self.catalog.authorized_documents("U-001"))
        self.assertEqual(
            self.catalog.denial_reason("U-001", "INC-2026-0312"),
            "role_document_type_denied")

    def test_restricted_clearance_alone_does_not_grant_a_role_denied_type(self):
        # Neha holds RESTRICTED and S002 is in scope, but a Security Incident
        # Report is not a document type her role reads.
        self.assertNotIn("INC-2026-0421",
                         self.catalog.authorized_documents("U-004"))
        self.assertEqual(
            self.catalog.denial_reason("U-004", "INC-2026-0421"),
            "role_document_type_denied")

    def test_org_wide_restricted_requires_department_fit(self):
        self.assertNotIn("POL-IT-001",
                         self.catalog.authorized_documents("U-004"))
        self.assertEqual(
            self.catalog.denial_reason("U-004", "POL-IT-001"),
            "org_restricted_department_mismatch")
        self.assertIn("POL-OPS-006",
                      self.catalog.authorized_documents("U-004"))

    def test_administrator_reads_everything(self):
        self.assertEqual(len(self.catalog.denied_documents("U-005")), 0)

    def test_confidential_clearance_cannot_reach_restricted(self):
        self.assertEqual(
            self.catalog.denial_reason("U-001", "POL-OPS-006"),
            "insufficient_clearance")


class ResultsIsolationTest(unittest.TestCase):
    def test_runner_refuses_to_write_over_the_golden_dataset(self):
        from evaluation import runner
        with self.assertRaises(ValueError):
            runner.save({"x": 1}, ds.GOLDEN_DATASET_FILE)

    def test_results_and_dataset_are_different_files(self):
        self.assertNotEqual(ds.RESULTS_FILE.resolve(),
                            ds.GOLDEN_DATASET_FILE.resolve())


if __name__ == "__main__":
    unittest.main(verbosity=2)
