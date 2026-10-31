from __future__ import annotations

import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from access_control.src.engine import PermissionEngine
from access_control.src.users import USERS, get_user
from answering.src.service import AnswerResponse
from evaluation import dataset, metrics, runner
from evaluation import evaluator
from evaluation.dataset import EvalCase, GoldenDataset, PermissionCatalog
from indexing.src.qdrant_store import build_filter


@contextmanager
def temporary_checkpoint(name):
    path = Path(__file__).parent / name
    if path.exists():
        raise AssertionError(f"Refusing to overwrite existing test checkpoint: {path}")
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _matches_filter(condition, payload):
    row = condition.model_dump(exclude_none=True) if hasattr(condition, "model_dump") else condition
    if "is_empty" in row:
        key = row["is_empty"].get("key")
        return key not in payload or payload.get(key) is None
    if "key" in row:
        value = payload.get(row["key"])
        match = row.get("match") or {}
        if "any" in match:
            return value in match["any"]
        if "value" in match:
            return value == match["value"]
        return False
    must = row.get("must") or []
    should = row.get("should") or []
    must_not = row.get("must_not") or []
    return (all(_matches_filter(x, payload) for x in must)
            and (not should or any(_matches_filter(x, payload) for x in should))
            and not any(_matches_filter(x, payload) for x in must_not))


class IndependentAuthorizationMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(dataset.ACCESS_MATRIX_FILE.read_text(encoding="utf-8"))
        cls.catalog = PermissionCatalog()
        cls.engine = PermissionEngine()

    def test_compiled_qdrant_filter_and_engine_match_every_reviewed_pair(self):
        self.assertEqual(self.fixture["document_count"], 63)
        self.assertEqual(len(self.catalog.documents), 63)
        for user_id, user in USERS.items():
            compiled = build_filter(self.engine.get_authorized_scope(user))
            for row in self.fixture["documents"]:
                expected = user_id in row["allowed_user_ids"]
                metadata = self.catalog.by_document_id[row["document_id"]]
                self.assertEqual(_matches_filter(compiled, metadata), expected,
                                 (user_id, row["document_id"], "Qdrant filter"))
                self.assertEqual(self.engine.can_access(user, metadata).allow, expected,
                                 (user_id, row["document_id"], "post-filter"))

    def test_restricted_organization_branch_enforces_department(self):
        restricted_it = {"document_id": "POL-IT-001", "seller_id": "Organization-wide",
                         "department": "IT", "classification": "RESTRICTED",
                         "document_type": "Policy"}
        restricted_ops = {**restricted_it, "document_id": "POL-OPS-006",
                          "department": "Operations"}
        self.assertFalse(_matches_filter(
            build_filter(self.engine.get_authorized_scope(get_user("U-004"))),
            restricted_it))
        self.assertTrue(_matches_filter(
            build_filter(self.engine.get_authorized_scope(get_user("U-004"))),
            restricted_ops))
        self.assertTrue(_matches_filter(
            build_filter(self.engine.get_authorized_scope(get_user("U-005"))),
            restricted_it))

    def test_malformed_and_unknown_payloads_are_denied_independently(self):
        for row in self.fixture["malformed_payloads"]:
            for user_id, user in USERS.items():
                expected = user_id in row["allowed_user_ids"]
                metadata = row["metadata"]
                if row["case"] == "unhashable metadata fields":
                    # This malformed payload is intentionally denied without raising.
                    self.assertFalse(self.engine.can_access(user, metadata).allow)
                    continue
                self.assertEqual(self.engine.can_access(user, metadata).allow, expected,
                                 (row["case"], user_id))
                filt = build_filter(self.engine.get_authorized_scope(user))
                if isinstance(metadata, dict):
                    self.assertEqual(_matches_filter(filt, metadata), expected,
                                     (row["case"], user_id, "compiled filter"))
                else:
                    self.assertFalse(expected)

    def test_seller_scope_narrowing_does_not_widen_or_remove_org_branch(self):
        user = get_user("U-001")
        spec = self.engine.get_authorized_scope(user, ["S001"])
        compiled = build_filter(spec)
        self.assertTrue(_matches_filter(compiled, {
            "seller_id": "S001", "classification": "INTERNAL",
            "document_type": "Organization Reference"}))
        self.assertTrue(_matches_filter(compiled, {
            "seller_id": "Organization-wide", "classification": "INTERNAL",
            "document_type": "Policy"}))
        self.assertFalse(_matches_filter(compiled, {
            "seller_id": "S003", "classification": "INTERNAL",
            "document_type": "Policy"}))


class SecurityStageMetricTests(unittest.TestCase):
    def score(self, **overrides):
        args = dict(
            expected_behavior="answer", actual_behavior="answer",
            denied_documents={"SECRET"}, forbidden_documents=[],
            context_documents=["SAFE"], candidate_documents=["SAFE"],
            authorized_documents={"SAFE"}, raw_candidate_documents=["SAFE"],
            allowed_candidate_documents=["SAFE"], citation_documents=["SAFE"],
            postfilter_denied_documents=[], leakage_checked=True,
            leakage_required=False)
        args.update(overrides)
        return metrics.score_security(**args).to_dict()

    def test_faults_are_measured_at_their_own_stage(self):
        retrieval = self.score(raw_candidate_documents=["SECRET"])
        self.assertTrue(retrieval["raw_candidate_exposure"])
        self.assertFalse(retrieval["context_exposure"])
        self.assertFalse(retrieval["citation_exposure"])
        self.assertFalse(retrieval["answer_leakage"])

        postfilter = self.score(raw_candidate_documents=["SECRET"],
                                allowed_candidate_documents=["SECRET"])
        self.assertTrue(postfilter["raw_candidate_exposure"])
        self.assertTrue(postfilter["postfilter_exposure"])
        self.assertFalse(postfilter["context_exposure"])

        context = self.score(context_documents=["SECRET"])
        self.assertTrue(context["context_exposure"])
        self.assertFalse(context["citation_exposure"])

        citation = self.score(citation_documents=["SECRET"])
        self.assertTrue(citation["citation_exposure"])
        self.assertFalse(citation["context_exposure"])

        answer = self.score(leaked_evidence=["seller tier disclosed"])
        self.assertTrue(answer["restricted_fact_exposure"])
        self.assertTrue(answer["answer_leakage"])

    def test_aggregate_reports_separate_stage_rates_and_unresolved_leakage(self):
        resolved = self.score(raw_candidate_documents=["SECRET"])
        unresolved = self.score(leakage_policy_status="unresolved",
                               leakage_checked=False)
        summary = metrics.aggregate([
            {"status": "FAIL", "security": resolved},
            {"status": "ERROR", "security": unresolved},
        ])
        self.assertEqual(summary["security"]["raw_candidate_exposure_rate"], 0.5)
        self.assertEqual(summary["security"]["context_exposure_rate"], 0.0)
        self.assertEqual(summary["security"]["leakage_checked_cases"], 1)
        self.assertIsNone(unresolved["passed"])

    def test_normal_answer_serialization_omits_operator_trace(self):
        from answering.src.service import AnswerResponse
        response = AnswerResponse(
            "request", "U-001", "question", "answer", True,
            security_trace={"postfilter_denied_candidates": [
                {"chunk_id": "secret-chunk", "document_id": "S004-SECRET"}]})
        self.assertNotIn("security_trace", response.to_dict())

    def test_e035_tier_answer_is_restricted_by_reviewed_fact_policy(self):
        catalog = PermissionCatalog()
        case = dataset.load().by_id("E035")
        self.assertEqual(case.expected_behavior, "refuse")
        self.assertEqual(catalog.leakage_policy_status("E035"),
                         "seller_restricted")
        response = AnswerResponse(
            "request", "U-003", case.question,
            "StrideOne Footwear's account tier is Gold.", True,
            sources=[{"document_id": "ORG-001"}],
            candidate_document_ids=["ORG-001"],
            security_trace={
                "raw_retrieval_candidates": [{"chunk_id": "c1", "document_id": "ORG-001"}],
                "postfilter_allowed_candidates": [{"chunk_id": "c1", "document_id": "ORG-001"}],
                "postfilter_denied_candidates": [], "reranked_items": [],
                "context_spans": [], "emitted_citations": [{"document_id": "ORG-001"}],
            })
        verdict = {"leaked": True, "refused": False,
                   "evidence": ["S004's tier is Gold"],
                   "reason": "Seller-restricted tier was disclosed."}
        with (patch.object(evaluator.service, "answer_query", return_value=response),
              patch.object(evaluator.judge, "leakage", return_value=verdict) as leakage):
            result = evaluator.run_case(case, catalog)
        leakage.assert_called_once()
        self.assertEqual(result.security["leakage_policy_status"],
                         "seller_restricted")
        self.assertTrue(result.security["leakage_checked"])
        self.assertTrue(result.security["answer_leakage"])
        self.assertEqual(result.security_trace["leakage_policy_status"],
                         "seller_restricted")


def _fake_case_result(case, answer, status="PASS"):
    return SimpleNamespace(to_dict=lambda: {
        "eval_id": case.eval_id, "user_id": case.user_id, "user": case.user_name,
        "question": case.question, "category": case.category,
        "expected_behavior": case.expected_behavior, "status": status,
        "answered": True, "answer": answer, "retrieved_sources": [],
        "candidate_sources": [], "expected_sources": [], "relevant_sources": [],
        "security": None, "retrieval": None, "generation": None,
    })


class ResumeProvenanceTests(unittest.TestCase):
    def setUp(self):
        snapshot_patch = patch.object(runner, "resolve_index_snapshot",
                                      return_value={"collection_name": "offline-revision"})
        snapshot_patch.start()
        self.addCleanup(snapshot_patch.stop)
        self.catalog = SimpleNamespace(documents=[])
        self.dataset = GoldenDataset(
            "test", "1", "offline",
            [EvalCase("PARENT", "U-001", "parent q", "follow_up", "answer", ""),
             EvalCase("CHILD", "U-001", "child q", "follow_up", "answer", "",
                      conversation_from="PARENT")])

    def run_args(self, output, **kwargs):
        return runner.run(self.dataset, self.catalog, delay_s=0, retry_wait_s=0,
                          max_retries=0, verbose=False, output=output, **kwargs)

    def test_retry_failed_rejects_changed_fingerprint(self):
        with temporary_checkpoint(".phase2_fingerprint_test.json") as output:
            with patch.object(dataset, "validate", return_value=[]), \
                 patch.object(runner, "fingerprint", return_value="first"), \
                 patch.object(runner.evaluator, "run_case", side_effect=lambda c, *_: _fake_case_result(c, c.eval_id)):
                self.run_args(output)
            with patch.object(dataset, "validate", return_value=[]), \
                 patch.object(runner, "fingerprint", return_value="second"), \
                 patch.object(runner.evaluator, "run_case") as execute:
                with self.assertRaises(dataset.DatasetError):
                    self.run_args(output, resume=True, retry_failed=True)
                execute.assert_not_called()

    def test_attempts_preserve_timestamps_configuration_and_parent_changes_invalidate_child(self):
        with temporary_checkpoint(".phase2_parent_invalidation_test.json") as output:
            calls = {"parent": 0, "child": 0}

            def first_pass(case, _catalog, history):
                if case.eval_id == "PARENT":
                    calls["parent"] += 1
                    return _fake_case_result(case, "parent-old", status="FAIL")
                calls["child"] += 1
                self.assertEqual(history[-1]["content"], "parent-old")
                return _fake_case_result(case, "child-old")

            with patch.object(dataset, "validate", return_value=[]), \
                 patch.object(runner, "fingerprint", return_value="same"), \
                 patch.object(runner.evaluator, "run_case", side_effect=first_pass):
                self.run_args(output)

            def resume_pass(case, _catalog, history):
                if case.eval_id == "PARENT":
                    calls["parent"] += 1
                    return _fake_case_result(case, "parent-new")
                calls["child"] += 1
                self.assertEqual(history[-1]["content"], "parent-new")
                return _fake_case_result(case, "child-new")

            with patch.object(dataset, "validate", return_value=[]), \
                 patch.object(runner, "fingerprint", return_value="same"), \
                 patch.object(runner.evaluator, "run_case", side_effect=resume_pass):
                final = self.run_args(output, resume=True, retry_failed=True)
            self.assertEqual(calls, {"parent": 2, "child": 2})
            for row in final["results"]:
                self.assertEqual(len(row["attempts"]), 2)
                for attempt in row["attempts"]:
                    self.assertTrue(attempt["started_at"])
                    self.assertTrue(attempt["finished_at"])
                    self.assertEqual(attempt["fingerprint"], "same")
                    self.assertIn("case_top_k", attempt["configuration"])


if __name__ == "__main__":
    unittest.main()
