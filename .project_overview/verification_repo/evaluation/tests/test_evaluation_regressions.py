r"""Offline regressions: every external transport is blocked or mocked.

Run with: .venv\Scripts\python.exe -m unittest discover -s evaluation/tests -v
These tests do not call Groq, Jina, Qdrant, or a running application.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from access_control.src import retrieval as access
from access_control.src.engine import PermissionEngine
from access_control.src.users import get_user
from answering.src import api, config, context, reranker, selection, service
from evaluation import dataset as ds, evaluator, judge, metrics, runner
from indexing.src.models import RetrievalResult


DOCUMENT = {"document_id": "A", "seller_id": "S001", "classification": "INTERNAL",
            "document_type": "Support Ticket", "department": "Support"}


def case(eval_id="E001", parent=None, behavior="answer"):
    return ds.EvalCase(eval_id, "U-001", "What happened?", "authorized_factual",
                       behavior, "Supported answer", expected_sources=("A",),
                       conversation_from=parent)


def response(document_id="A"):
    return service.AnswerResponse("test", "U-001", "What happened?", "Supported answer",
                                  True, sources=[{"document_id": document_id}],
                                  context="Supported answer",
                                  candidate_document_ids=[document_id])


class OfflineTest(unittest.TestCase):
    def setUp(self):
        self.network = mock.patch("socket.socket.connect", side_effect=AssertionError(
            "A regression test attempted a network connection"))
        self.network.start()
        self.addCleanup(self.network.stop)


class JudgeValidationTest(OfflineTest):
    def grade(self, verdict):
        with mock.patch.object(judge.llm, "generate", return_value=(json.dumps(verdict), {})):
            return judge.grade("question", "answer", "context", "reference", ["fact"])

    def test_missing_nonfinite_or_out_of_range_scores_are_errors(self):
        for invalid in (None, "1", True, float("nan"), float("inf"), -0.1, 1.1):
            with self.subTest(invalid=invalid), self.assertRaises(judge.JudgeError):
                self.grade({"faithfulness": invalid, "relevancy": 1,
                            "correctness": 1, "fact_support": [1]})

    def test_missing_fact_judgments_are_not_padded_into_fake_scores(self):
        with self.assertRaises(judge.JudgeError):
            self.grade({"faithfulness": 1, "relevancy": 1,
                        "correctness": 1, "fact_support": []})

    def test_leakage_missing_boolean_or_invented_evidence_is_an_error(self):
        for verdict in ({}, {"leaked": "false", "refused": True, "evidence": []},
                        {"leaked": True, "refused": False, "evidence": ["invented"]}):
            with mock.patch.object(judge.llm, "generate", return_value=(json.dumps(verdict), {})):
                with self.assertRaises(judge.JudgeError):
                    judge.leakage("question", "answer", "restricted subject")

    def test_answer_tail_is_reviewed_instead_of_truncated(self):
        answer = "a" * 5000 + " tail disclosure"
        verdict = {"leaked": True, "refused": False, "evidence": ["tail disclosure"]}
        with mock.patch.object(judge.llm, "generate", return_value=(json.dumps(verdict), {})) as generate:
            self.assertTrue(judge.leakage("q", answer, "subject")["leaked"])
            self.assertIn("tail disclosure", generate.call_args.args[0][1]["content"])

    def test_oversize_input_is_unjudged_instead_of_silently_cut(self):
        with mock.patch.object(judge.llm, "generate") as generate:
            with self.assertRaises(judge.JudgeError):
                judge.leakage("q", "x" * (judge.MAX_ANSWER_CHARS + 1), "subject")
            generate.assert_not_called()


class ScoringRegressionTest(OfflineTest):
    def test_unknown_document_is_not_treated_as_authorized(self):
        result = metrics.score_security("answer", "answer", set(), [], ["NEW"], ["NEW"],
                                        authorized_documents={"A"})
        self.assertFalse(result.passed)
        self.assertEqual(result.unauthorized_documents, ["NEW"])

    def test_unjudged_generation_cannot_pass(self):
        outcome, _ = evaluator._decide(metrics.score_retrieval(["A"], ["A"]),
                                       metrics.GenerationMetrics(),
                                       metrics.SecurityMetrics().to_dict())
        self.assertEqual(outcome, evaluator.ERROR)

    def test_empty_retrieval_counts_as_a_miss_for_mrr(self):
        self.assertEqual(metrics.mean_reciprocal_rank(["A"], []), 0)

    def test_deterministic_security_failure_survives_judge_outage(self):
        with mock.patch.object(service, "answer_query", return_value=response("UNKNOWN")), \
                mock.patch.object(judge, "grade", side_effect=judge.JudgeError("unavailable")):
            result = evaluator.run_case(case(), ds.PermissionCatalog([DOCUMENT]))
        self.assertEqual(result.status, evaluator.FAIL)
        self.assertTrue(result.security["unauthorized_retrieval"])

    def test_failed_leakage_judge_preserves_retrieval_and_unknown_security(self):
        restricted = ds.EvalCase("R", "U-001", "Show restricted info", "unauthorized_access",
                                 "refuse", "Decline", forbidden_sources=("X",))
        with mock.patch.object(service, "answer_query", return_value=response()), \
                mock.patch.object(judge, "leakage", side_effect=judge.JudgeError("quota")):
            result = evaluator.run_case(restricted, ds.PermissionCatalog([DOCUMENT]))
        self.assertEqual(result.status, evaluator.ERROR)
        self.assertIsNotNone(result.retrieval)
        self.assertIsNone(result.security["passed"])
        self.assertFalse(result.security["leakage_checked"])
        summary = metrics.aggregate([result.to_dict()])
        self.assertEqual(summary["security"]["retrieval_checked_cases"], 1)
        self.assertIsNone(summary["security"]["answer_leakage_rate"])

    def test_safe_but_substantive_answer_fails_explicit_refusal_expectation(self):
        restricted = ds.EvalCase("R", "U-001", "question", "unauthorized_access", "refuse", "")
        with mock.patch.object(service, "answer_query", return_value=response()), \
                mock.patch.object(judge, "leakage", return_value={
                    "leaked": False, "refused": False, "evidence": [], "reason": "safe answer"}):
            result = evaluator.run_case(restricted, ds.PermissionCatalog([DOCUMENT]))
        self.assertEqual(result.status, evaluator.FAIL)
        self.assertIn("expected a refusal, got an answer", result.failure_reasons)

    def test_model_written_refusal_can_pass_even_when_application_answered_is_true(self):
        restricted = ds.EvalCase("R", "U-001", "question", "unauthorized_access", "refuse", "")
        with mock.patch.object(service, "answer_query", return_value=response()), \
                mock.patch.object(judge, "leakage", return_value={
                    "leaked": False, "refused": True, "evidence": [], "reason": "declined"}):
            result = evaluator.run_case(restricted, ds.PermissionCatalog([DOCUMENT]))
        self.assertEqual(result.status, evaluator.PASS)
        self.assertEqual(result.actual_behavior, "refuse")


class RunnerRegressionTest(OfflineTest):
    def setUp(self):
        super().setUp()
        self.dataset = ds.GoldenDataset("test", "1", "offline", [case(), case("E002", "E001")])
        self.catalog = ds.PermissionCatalog([DOCUMENT])
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "results.json"
        signature = mock.patch.object(runner, "fingerprint", return_value="test-signature")
        signature.start()
        self.addCleanup(signature.stop)

    def result(self, target, catalog, history):
        return evaluator.CaseResult(target.eval_id, target.user_id, target.user_name,
                                     target.question, target.category, target.expected_behavior,
                                     status=evaluator.PASS, answered=True, answer="Actual prior answer")

    def run_batch(self, **kwargs):
        return runner.run(self.dataset, self.catalog, delay_s=0, retry_wait_s=0,
                          max_retries=0, output=self.path, verbose=False, **kwargs)

    def test_selecting_follow_up_includes_its_parent(self):
        self.assertEqual([c.eval_id for c in runner.select_cases(self.dataset, only=["E002"])],
                         ["E001", "E002"])

    def test_unknown_selection_and_invalid_limits_are_rejected(self):
        for options in ({"only": ["UNKNOWN"]}, {"only": []}, {"limit": 0}, {"limit": -1}):
            with self.subTest(options=options), self.assertRaises(ds.DatasetError):
                runner.select_cases(self.dataset, **options)

    def test_follow_up_receives_real_prior_answer(self):
        with mock.patch.object(evaluator, "run_case", side_effect=self.result) as run_case:
            result = self.run_batch()
        self.assertEqual(run_case.call_args_list[1].args[2][1]["content"], "Actual prior answer")
        self.assertFalse(result["partial"])

    def test_quota_stops_then_resume_retries_only_pending_cases(self):
        quota = self.result(self.dataset.cases[0], None, None)
        quota.status, quota.error = evaluator.ERROR, "429 quota exceeded"
        with mock.patch.object(evaluator, "run_case", return_value=quota) as run_case:
            partial = self.run_batch()
        self.assertEqual(run_case.call_count, 1)
        self.assertTrue(partial["partial"])
        self.assertEqual(partial["stop_reason"], "quota_exhausted")
        with mock.patch.object(evaluator, "run_case", side_effect=self.result):
            completed = self.run_batch(resume=True)
        self.assertEqual(completed["run_id"], partial["run_id"])
        self.assertEqual(completed["summary"]["passed"], 2)
        self.assertFalse(completed["partial"])

    def test_completed_resume_does_not_call_the_pipeline(self):
        with mock.patch.object(evaluator, "run_case", side_effect=self.result):
            self.run_batch()
        with mock.patch.object(evaluator, "run_case") as run_case:
            self.run_batch(resume=True)
        run_case.assert_not_called()

    def test_mismatched_resume_or_existing_output_cannot_overwrite(self):
        with mock.patch.object(evaluator, "run_case", side_effect=self.result):
            self.run_batch()
        before = self.path.read_bytes()
        with mock.patch.object(evaluator, "run_case") as run_case:
            with self.assertRaises(ds.DatasetError):
                self.run_batch()
            with self.assertRaises(ds.DatasetError):
                self.run_batch(resume=True, limit=1)
            with mock.patch.object(runner, "fingerprint", return_value="changed"):
                with self.assertRaises(ds.DatasetError):
                    self.run_batch(resume=True)
        run_case.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_interruption_leaves_a_readable_checkpoint(self):
        with mock.patch.object(evaluator, "run_case", side_effect=KeyboardInterrupt):
            result = self.run_batch()
        self.assertEqual(result["stop_reason"], "interrupted")
        self.assertTrue(runner.load_results(self.path)["partial"])

    def test_missing_parent_answer_blocks_follow_up_without_a_call(self):
        refused = self.result(self.dataset.cases[0], None, None)
        refused.status, refused.answered, refused.answer = evaluator.FAIL, False, ""
        with mock.patch.object(evaluator, "run_case", return_value=refused) as run_case:
            result = self.run_batch()
        self.assertEqual(run_case.call_count, 1)
        self.assertEqual(result["results"][1]["status"], evaluator.ERROR)

    def test_forward_or_self_history_references_are_invalid(self):
        for parent in ("E001", "E002"):
            dataset = ds.GoldenDataset("test", "1", "", [case(parent=parent), case("E002")])
            self.assertTrue(ds.validate(dataset, self.catalog))

    def test_failed_atomic_replace_keeps_previous_checkpoint(self):
        runner.save({"results": [], "marker": "original"}, self.path)
        with mock.patch.object(runner.os, "replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                runner.save({"results": [], "marker": "new"}, self.path)
        self.assertEqual(runner.load_results(self.path)["marker"], "original")
        self.assertEqual(list(self.path.parent.glob("*.tmp")), [])

    def test_cli_returns_nonzero_for_a_completed_failed_run(self):
        failed = {"run_id": "test", "partial": False, "selection_complete": True,
                  "stop_reason": None, "summary": {"failed": 1, "errors": 0}}
        with mock.patch.object(runner.sys, "argv", ["evaluation", "--no-save"]), \
                mock.patch.object(ds, "load", return_value=self.dataset), \
                mock.patch.object(ds, "PermissionCatalog", return_value=self.catalog), \
                mock.patch.object(config, "validate_env"), \
                mock.patch.object(runner, "run", return_value=failed), \
                mock.patch("builtins.print"):
            self.assertEqual(runner.main(), 1)

    def test_plan_never_runs_queries_or_requires_provider_credentials(self):
        with mock.patch.object(runner.sys, "argv", ["evaluation", "--plan"]), \
                mock.patch.object(ds, "load", return_value=self.dataset), \
                mock.patch.object(ds, "PermissionCatalog", return_value=self.catalog), \
                mock.patch.object(config, "validate_env") as validate_env, \
                mock.patch.object(runner, "run") as run, \
                mock.patch("builtins.print"):
            self.assertEqual(runner.main(), 0)
        run.assert_not_called()
        validate_env.assert_not_called()


class PermissionBoundaryTest(OfflineTest):
    def chunk(self, seller, marker):
        return RetrievalResult(marker, marker, marker, 1, "mock",
                               {**DOCUMENT, "document_id": marker, "seller_id": seller})

    def test_denied_text_never_reaches_reranker_or_model(self):
        allowed, denied = self.chunk("S001", "ALLOWED"), self.chunk("S002", "DENIED_CANARY")
        with mock.patch.object(access, "phase3_retrieve", return_value=[denied, allowed]) as retrieve, \
                mock.patch("access_control.src.audit._write"), \
                mock.patch.object(reranker, "rerank", return_value=reranker.RerankOutcome([(allowed, 1)], True)) as rerank, \
                mock.patch.object(service.llm, "generate", return_value=("Supported", {})) as generate:
            result = service.answer_query("U-001", "question")
        self.assertEqual(rerank.call_args.args[1], [allowed])
        self.assertNotIn("DENIED_CANARY", str(generate.call_args))
        self.assertNotIn("S002", retrieve.call_args.kwargs["filters"]["seller_id"]["any"])
        self.assertEqual(result.sources[0]["document_id"], "ALLOWED")

    def test_final_gate_blocks_reranker_injection_for_buffered_and_streamed_answers(self):
        allowed, denied = self.chunk("S001", "ALLOWED"), self.chunk("S002", "DENIED_CANARY")
        for streaming in (False, True):
            with self.subTest(streaming=streaming), \
                    mock.patch.object(access, "phase3_retrieve", return_value=[allowed]), \
                    mock.patch("access_control.src.audit._write"), \
                    mock.patch.object(reranker, "rerank", return_value=reranker.RerankOutcome([(denied, 1)], True)), \
                    mock.patch.object(service.llm, "generate") as generate, \
                    mock.patch.object(service.llm, "stream") as stream:
                if streaming:
                    result = list(service.answer_query_stream("U-001", "question"))[-1]
                    self.assertFalse(result["answered"])
                else:
                    self.assertFalse(service.answer_query("U-001", "question").answered)
                generate.assert_not_called()
                stream.assert_not_called()

    def test_scope_narrowing_cannot_grant_another_seller(self):
        with mock.patch.object(service, "authorized_retrieve") as retrieve:
            self.assertFalse(service.answer_query("U-001", "question", seller_filter=["S002"]).answered)
        retrieve.assert_not_called()

    def test_known_policy_decisions_are_not_derived_from_catalog_expectations(self):
        engine = PermissionEngine()
        self.assertFalse(engine.can_access(get_user("U-001"), {**DOCUMENT, "seller_id": "S002"}).allow)
        self.assertFalse(engine.can_access(get_user("U-001"), {**DOCUMENT, "classification": "RESTRICTED"}).allow)
        self.assertFalse(engine.can_access(get_user("U-001"), {**DOCUMENT, "classification": None}).allow)
        self.assertTrue(engine.can_access(get_user("U-001"), DOCUMENT).allow)

    def test_full_evaluation_report_is_disabled_by_default(self):
        handler = object.__new__(api.Handler)
        handler._send = mock.Mock()
        with mock.patch.object(config, "EVALUATION_REPORTS_ENABLED", False), \
                mock.patch.object(api, "_evaluation") as load:
            handler._evaluation_matrix()
        self.assertEqual(handler._send.call_args.args[0], 403)
        load.assert_not_called()

    def test_http_cannot_start_a_live_evaluation(self):
        handler = object.__new__(api.Handler)
        handler.path = "/evaluation/run"
        handler._send = mock.Mock()
        handler.do_POST()
        self.assertEqual(handler._send.call_args.args[0], 405)


class RetrievalSelectionTest(OfflineTest):
    """Regressions for the document-aware selection stage.

    Each test pins a defect that measurably cost recall or precision in the
    golden run, so a future tuning pass cannot silently reintroduce it.
    """

    def chunk(self, document_id, index=0, text="body"):
        return RetrievalResult(
            f"{document_id}_chunk_{index:03d}", document_id, text, 1.0, "mock",
            {**DOCUMENT, "document_id": document_id, "chunk_index": index})

    def test_reranker_scores_the_conversation_resolved_query(self):
        """A follow-up must be reranked with its referent resolved.

        Retrieval already expands "how much of that" using prior turns. Scoring
        the bare pronoun against the candidates dropped every passage below the
        threshold and returned an empty context (golden case E027).
        """
        allowed = self.chunk("A")
        history = [{"role": "user", "content": "What caused the shortfall?"},
                   {"role": "assistant", "content": "A ledger defect."}]
        with mock.patch.object(access, "phase3_retrieve", return_value=[allowed]), \
                mock.patch("access_control.src.audit._write"), \
                mock.patch.object(reranker, "rerank",
                                  return_value=reranker.RerankOutcome([(allowed, 0.9)], True)) as rerank, \
                mock.patch.object(service.llm, "generate", return_value=("Supported", {})):
            service.answer_query("U-001", "How much of that was credited back?",
                                 conversation_history=history)
        self.assertIn("What caused the shortfall?", rerank.call_args.args[0])

    def test_reranker_is_shown_the_whole_chunk(self):
        """The rerank window must cover the largest indexed chunk.

        Ingestion emits 800-token chunks; a 2000 char window hid the back half
        of most of them, so evidence past the cut could not raise a document's
        score and the document was dropped despite being retrieved.
        """
        self.assertGreaterEqual(reranker.MAX_DOC_CHARS, 4400)
        marker = "EVIDENCE_PAST_THE_OLD_CUT"
        chunk = self.chunk("A", text="x" * 3000 + marker)
        with mock.patch.object(config, "JINA_API_KEY", "test-key"), \
                mock.patch.object(config, "RERANK_ENABLED", True), \
                mock.patch.object(reranker.requests, "post") as post:
            post.return_value = mock.Mock(
                status_code=200,
                json=lambda: {"results": [{"index": 0, "relevance_score": 0.9}]})
            reranker.rerank("question", [chunk])
        self.assertIn(marker, post.call_args.kwargs["json"]["documents"][0])

    def test_one_document_can_contribute_several_passages(self):
        """Budget is spent on documents, not on a flat chunk list.

        Every corpus document is multi-chunk, so a flat top-N resolved to N
        documents of one passage each: maximum breadth, minimum evidence.
        """
        items = [(self.chunk("A", 0), 0.9), (self.chunk("A", 1), 0.8),
                 (self.chunk("B", 0), 0.75)]
        selected, stats = selection.select(items, config.DEFAULT_ANSWERING)
        self.assertEqual(stats.selected_documents, 2)
        self.assertEqual([c.chunk_id for c, _ in selected],
                         ["A_chunk_000", "A_chunk_001", "B_chunk_000"])

    def test_passages_are_restored_to_reading_order_within_a_document(self):
        items = [(self.chunk("A", 2), 0.9), (self.chunk("A", 0), 0.5)]
        selected, _ = selection.select(items, config.DEFAULT_ANSWERING)
        self.assertEqual([c.metadata["chunk_index"] for c, _ in selected], [0, 2])

    def test_a_decisive_winner_narrows_the_context(self):
        """Relative pruning is what actually lifts precision.

        An absolute floor keeps a 0.21 document beside a 0.95 one, spending a
        document slot on noise.
        """
        items = [(self.chunk("A"), 0.95), (self.chunk("B"), 0.22),
                 (self.chunk("C"), 0.21)]
        selected, stats = selection.select(items, config.DEFAULT_ANSWERING)
        self.assertEqual([c.document_id for c, _ in selected], ["A"])
        self.assertGreater(stats.dropped_by_relative_cut, 0)

    def test_a_flat_distribution_keeps_breadth(self):
        """The cut is relative, so weak-but-comparable evidence is retained."""
        items = [(self.chunk("A"), 0.44), (self.chunk("B"), 0.41),
                 (self.chunk("C"), 0.39)]
        selected, _ = selection.select(items, config.DEFAULT_ANSWERING)
        self.assertEqual({c.document_id for c, _ in selected}, {"A", "B", "C"})

    def test_unscored_candidates_are_never_pruned_by_a_relative_cut(self):
        """A reranker outage must degrade relevance, never recall."""
        items = [(self.chunk("A"), None), (self.chunk("B"), None)]
        selected, stats = selection.select(items, config.DEFAULT_ANSWERING)
        self.assertFalse(stats.scored)
        self.assertEqual(stats.dropped_by_relative_cut, 0)
        self.assertEqual({c.document_id for c, _ in selected}, {"A", "B"})

    def test_selection_never_invents_a_passage(self):
        items = [(self.chunk("A"), 0.9)]
        selected, _ = selection.select(items, config.DEFAULT_ANSWERING)
        self.assertTrue({id(c) for c, _ in selected} <= {id(c) for c, _ in items})

    def test_top_k_bounds_documents_not_chunks(self):
        items = [(self.chunk(d), 0.9) for d in ("A", "B", "C", "D", "E")]
        with mock.patch.object(access, "phase3_retrieve", return_value=[c for c, _ in items]), \
                mock.patch("access_control.src.audit._write"), \
                mock.patch.object(reranker, "rerank",
                                  return_value=reranker.RerankOutcome(items, True)), \
                mock.patch.object(service.llm, "generate", return_value=("Supported", {})):
            result = service.answer_query("U-001", "question", top_k=2)
        self.assertEqual(len(result.sources), 2)

    def test_citations_only_name_pages_that_survived_the_budget(self):
        """A dropped passage must not contribute a page to the citation."""
        small = self.chunk("A", 0, text="short")
        small.metadata["page_numbers"] = [1]
        large = self.chunk("A", 1, text="y" * 5000)
        large.metadata["page_numbers"] = [9]
        cfg = dataclasses.replace(config.DEFAULT_ANSWERING, max_context_chars=200)
        _, sources = context.build([(small, 0.9), (large, 0.8)], cfg)
        self.assertEqual(sources[0].pages, [1])

    def test_rate_limited_rerank_is_retried_not_silently_degraded(self):
        """A 429 must not become an unranked context.

        The full candidate pool at full document width is ~24k tokens per call
        against a 100k/minute provider limit, so bursts rate limit. Treating
        that as permanent silently returned fusion order: during a sweep it
        degraded 13 of 45 cases without any failure being reported.
        """
        ok = mock.Mock(status_code=200,
                       json=lambda: {"results": [{"index": 0, "relevance_score": 0.9}]})
        limited = mock.Mock(status_code=429, text="rate limit", headers={})
        with mock.patch.object(config, "JINA_API_KEY", "test-key"), \
                mock.patch.object(config, "RERANK_ENABLED", True), \
                mock.patch.object(reranker.time, "sleep") as sleep, \
                mock.patch.object(reranker.requests, "post",
                                  side_effect=[limited, limited, ok]):
            outcome = reranker.rerank("question", [self.chunk("A")])
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.attempts, 3)
        self.assertEqual(outcome.items[0][1], 0.9)
        self.assertEqual(sleep.call_count, 2)

    def test_rerank_backoff_honours_retry_after(self):
        limited = mock.Mock(status_code=429, text="slow down",
                            headers={"Retry-After": "12"})
        with mock.patch.object(config, "JINA_API_KEY", "test-key"), \
                mock.patch.object(config, "RERANK_ENABLED", True), \
                mock.patch.object(reranker.time, "sleep") as sleep, \
                mock.patch.object(reranker.requests, "post", return_value=limited):
            outcome = reranker.rerank("question", [self.chunk("A")])
        self.assertFalse(outcome.applied)
        self.assertEqual(sleep.call_args_list[0].args[0], 12.0)

    def test_unauthorized_rerank_response_is_not_retried(self):
        """A rejected key is permanent; retrying it just wastes the request."""
        denied = mock.Mock(status_code=401, text="bad key", headers={})
        with mock.patch.object(config, "JINA_API_KEY", "test-key"), \
                mock.patch.object(config, "RERANK_ENABLED", True), \
                mock.patch.object(reranker.requests, "post",
                                  return_value=denied) as post:
            outcome = reranker.rerank("question", [self.chunk("A")])
        self.assertFalse(outcome.applied)
        self.assertEqual(post.call_count, 1)

    def test_degraded_ranking_is_reported_in_request_stats(self):
        """A run must never look representative when ranking was unavailable."""
        allowed = self.chunk("A")
        with mock.patch.object(access, "phase3_retrieve", return_value=[allowed]), \
                mock.patch("access_control.src.audit._write"), \
                mock.patch.object(reranker, "rerank",
                                  return_value=reranker.RerankOutcome(
                                      [(allowed, None)], False, "rate limit", 4)), \
                mock.patch.object(service.llm, "generate", return_value=("Supported", {})):
            result = service.answer_query("U-001", "question")
        self.assertFalse(result.reranked)
        self.assertEqual(result.selection["rerank_attempts"], 4)
        self.assertEqual(result.selection["rerank_error"], "rate limit")

    def test_rate_limited_rerank_is_retried_not_silently_degraded(self):
        """A 429 must not ship an unranked context.

        Widening the rerank window to cover a whole chunk raised each call to
        roughly 24k tokens, and Jina meters tokens per minute. Treating the
        resulting 429 as permanent replaced ranked results with raw fusion
        order for 13 of 45 golden cases before this was fixed.
        """
        chunk = self.chunk("A")
        limited = mock.Mock(status_code=429, text="rate limit", headers={})
        ok = mock.Mock(status_code=200, headers={},
                       json=lambda: {"results": [{"index": 0, "relevance_score": 0.9}]})
        with mock.patch.object(config, "JINA_API_KEY", "test-key"), \
                mock.patch.object(config, "RERANK_ENABLED", True), \
                mock.patch.object(reranker.time, "sleep") as sleep, \
                mock.patch.object(reranker.requests, "post",
                                  side_effect=[limited, limited, ok]):
            outcome = reranker.rerank("question", [chunk])
        self.assertTrue(outcome.applied)
        self.assertEqual(outcome.attempts, 3)
        self.assertEqual(outcome.items[0][1], 0.9)
        self.assertEqual(sleep.call_count, 2)

    def test_rerank_honours_retry_after_before_backoff(self):
        chunk = self.chunk("A")
        limited = mock.Mock(status_code=429, text="slow down",
                            headers={"Retry-After": "7"})
        ok = mock.Mock(status_code=200, headers={},
                       json=lambda: {"results": [{"index": 0, "relevance_score": 0.5}]})
        with mock.patch.object(config, "JINA_API_KEY", "test-key"), \
                mock.patch.object(config, "RERANK_ENABLED", True), \
                mock.patch.object(reranker.time, "sleep") as sleep, \
                mock.patch.object(reranker.requests, "post", side_effect=[limited, ok]):
            reranker.rerank("question", [chunk])
        self.assertEqual(sleep.call_args.args[0], 7.0)

    def test_rerank_does_not_retry_a_rejected_request(self):
        """401/400 will fail identically on a second attempt; do not burn time."""
        chunk = self.chunk("A")
        rejected = mock.Mock(status_code=401, text="bad key", headers={})
        with mock.patch.object(config, "JINA_API_KEY", "test-key"), \
                mock.patch.object(config, "RERANK_ENABLED", True), \
                mock.patch.object(reranker.time, "sleep") as sleep, \
                mock.patch.object(reranker.requests, "post", return_value=rejected) as post:
            outcome = reranker.rerank("question", [chunk])
        self.assertFalse(outcome.applied)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    def test_exhausted_retries_still_return_every_authorized_chunk(self):
        """Degrading relevance must never degrade recall or authorization."""
        chunks = [self.chunk("A"), self.chunk("B")]
        limited = mock.Mock(status_code=429, text="rate limit", headers={})
        with mock.patch.object(config, "JINA_API_KEY", "test-key"), \
                mock.patch.object(config, "RERANK_ENABLED", True), \
                mock.patch.object(config, "RERANK_MAX_RETRIES", 1), \
                mock.patch.object(reranker.time, "sleep"), \
                mock.patch.object(reranker.requests, "post", return_value=limited):
            outcome = reranker.rerank("question", chunks)
        self.assertFalse(outcome.applied)
        self.assertEqual([c for c, _ in outcome.items], chunks)
        self.assertTrue(all(s is None for _, s in outcome.items))

    def test_context_budget_cuts_depth_before_breadth(self):
        """Every selected document must get a passage before any gets a second.

        Packing depth-first let the leading documents consume the whole budget,
        silently dropping the tail and discarding the breadth `select` had just
        decided on. Golden cases E019 and E026 regressed from PASS to FAIL that
        way: K fell from 4 to 3 and took an expected source with it.
        """
        big = 1000
        pairs = []
        for doc in ("A", "B", "C"):
            for i in range(3):
                pairs.append((self.chunk(doc, i, text=f"{doc}{i} " + "x" * big),
                              1.0 - 0.01 * len(pairs)))
        # Room for ~4 passages: depth-first would spend all of it on A.
        cfg = dataclasses.replace(config.DEFAULT_ANSWERING, max_context_chars=4400)
        _, sources = context.build(pairs, cfg)
        self.assertEqual([s.document_id for s in sources], ["A", "B", "C"])
        self.assertTrue(all(len(s.chunk_ids) >= 1 for s in sources))

    def test_context_still_packs_depth_when_budget_allows(self):
        pairs = [(self.chunk("A", 0), 0.9), (self.chunk("A", 1), 0.8),
                 (self.chunk("B", 0), 0.7)]
        _, sources = context.build(pairs, config.DEFAULT_ANSWERING)
        by_id = {s.document_id: s for s in sources}
        self.assertEqual(len(by_id["A"].chunk_ids), 2)
        self.assertEqual(len(by_id["B"].chunk_ids), 1)

    def test_precision_ceiling_separates_breadth_from_error(self):
        """Perfect retrieval into a wider context is not a retrieval defect."""
        perfect = metrics.score_retrieval(["A"], ["A", "B"])
        self.assertEqual(perfect.recall_at_k, 1.0)
        self.assertEqual(perfect.precision_at_k, 0.5)
        self.assertEqual(perfect.precision_ceiling, 0.5)
        self.assertEqual(perfect.precision_efficiency, 1.0)

        wrong = metrics.score_retrieval(["A"], ["B", "C"])
        self.assertEqual(wrong.precision_at_k, 0.0)
        self.assertEqual(wrong.precision_efficiency, 0.0)

    def test_headline_retrieval_formulas_are_unchanged(self):
        """The diagnostics must not quietly redefine the reported scores."""
        self.assertEqual(metrics.precision_at_k(["A", "B"], ["A", "X", "Y"]), 1 / 3)
        self.assertEqual(metrics.recall_at_k(["A", "B"], ["A", "X", "Y"]), 0.5)
        self.assertEqual(metrics.mean_reciprocal_rank(["B"], ["A", "B"]), 0.5)
        self.assertIsNone(metrics.precision_at_k(["A"], []))
        self.assertIsNone(metrics.recall_at_k([], ["A"]))


if __name__ == "__main__":
    unittest.main()
