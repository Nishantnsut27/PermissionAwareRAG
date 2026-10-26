"""Phase 7 tests: query-aware Jina reranking.

Run with:  .venv\\Scripts\\python.exe -m unittest discover -s evaluation/tests -v

No test calls Jina, Qdrant, Groq or OpenJEV. The Jina transport is mocked at
`reranker._request` (or at `requests.post` where the HTTP contract itself is
under test), so every assertion here is about our own ranking logic.

The invariant that matters most is tested first: Phase 7 may reorder the
passages Jina returns, but it may never change WHICH passages survive the
relevance gate. That is what keeps a ranking feature from becoming a recall
regression, which is the failure mode Phase 6 was audited for.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import requests

from answering.src import config, query_aware_rerank, reranker
from answering.src.query_aware_rerank import CompatibilityWeights
from answering.src.query_planner import QueryPlan
from indexing.src.models import RetrievalResult


def chunk(document_id, text="body", seller="S001", doc_type="Support Ticket",
          index=0, method="hybrid"):
    return RetrievalResult(
        f"{document_id}_chunk_{index:03d}", document_id, text, 0.0, method,
        {"document_id": document_id, "seller_id": seller,
         "classification": "INTERNAL", "document_type": doc_type,
         "department": "Support", "chunk_index": index})


def plan(query="what happened", intent="factual_lookup", aspects=(),
         evidence=(), identifiers=(), sellers=(), entities=()):
    return QueryPlan(
        original_query=query, intent=intent,
        required_aspects=tuple(aspects), evidence_types=tuple(evidence),
        identifiers=tuple(identifiers), seller_scope=tuple(sellers),
        entities=tuple(entities))


ALL_SIGNALS = CompatibilityWeights()


class RerankerTestCase(unittest.TestCase):
    """Base for tests that exercise the reranker itself.

    `rerank()` returns early when reranking is switched off or no Jina key is
    present, so a test of ranking behaviour has to assert those preconditions
    rather than inherit them. Without this the suite passes or fails according
    to the developer's .env instead of according to the code.
    """

    PLAN_AWARE = True

    def setUp(self):
        patches = [
            mock.patch.object(config, "RERANK_ENABLED", True),
            mock.patch.object(config, "JINA_API_KEY", "test-key"),
            mock.patch.object(config, "RERANK_PLAN_AWARE", self.PLAN_AWARE),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)


def plan_aware_enabled():
    """Force the feature on for tests that exercise plan-aware ranking.

    Without this a developer whose environment sets RERANK_PLAN_AWARE=false
    would see spurious failures: the test would be measuring their .env rather
    than the behaviour it claims to pin. Tests that assert the DISABLED path
    patch it to False explicitly for the same reason.
    """
    return mock.patch.object(config, "RERANK_PLAN_AWARE", True)


def jina(scores):
    """Fake Jina results: index -> relevance_score, best first."""
    return [{"index": i, "relevance_score": s} for i, s in scores]


def order(outcome):
    return [c.document_id for c, _ in outcome.items]


class FakeResponse:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is _MALFORMED:
            raise ValueError("no json")
        return self._payload


_MALFORMED = object()


class RecallNeutralityTest(RerankerTestCase):
    """The relevance gate must stay a pure Jina decision."""

    def test_threshold_is_applied_to_the_raw_score_not_the_adjusted_one(self):
        # A passage scoring just under the threshold carries every plan signal
        # there is. If the plan term were added before the gate it would be
        # lifted over the line and change the surviving set.
        chunks = [chunk("STRONG", "unrelated prose", doc_type="Policy"),
                  chunk("WEAK", "root cause was a database timeout in May 2026",
                        doc_type="Incident Report")]
        p = plan("why did it fail in May 2026", intent="root_cause",
                 aspects=("cause",), evidence=("incident_report",))
        with mock.patch.object(reranker, "_request",
                               return_value=jina([(0, 0.90), (1, 0.19)])):
            outcome = reranker.rerank("q", chunks, threshold=0.20, plan=p)
        self.assertEqual(order(outcome), ["STRONG"])

    def test_surviving_set_is_identical_with_and_without_a_plan(self):
        chunks = [chunk("A", "root cause analysis", doc_type="Incident Report"),
                  chunk("B", "background", doc_type="Account Overview"),
                  chunk("C", "cause of failure", doc_type="Incident Report")]
        p = plan("why", intent="root_cause", aspects=("cause",),
                 evidence=("incident_report",))
        results = jina([(0, 0.80), (1, 0.21), (2, 0.19)])
        with mock.patch.object(reranker, "_request", return_value=results):
            base = reranker.rerank("q", chunks, threshold=0.20)
        with mock.patch.object(reranker, "_request", return_value=results):
            aware = reranker.rerank("q", chunks, threshold=0.20, plan=p)
        self.assertEqual(sorted(order(base)), sorted(order(aware)))

    def test_adjustment_is_never_negative(self):
        c = chunk("A", "nothing matching here", doc_type="Account Overview")
        p = plan("why", intent="root_cause", aspects=("cause",),
                 evidence=("incident_report",), sellers=("S002",))
        self.assertGreaterEqual(query_aware_rerank.adjust(0.5, c, p), 0.5)


class BoundednessTest(RerankerTestCase):
    """Plan signals may break ties; they may not overturn relevance (§32)."""

    def test_total_adjustment_cannot_exceed_the_configured_bound(self):
        # Every signal fires at once on this chunk.
        c = chunk("EVERYTHING",
                  "root cause INC-2026-0519 in May 2026 refund settlement",
                  seller="S001", doc_type="Incident Report")
        p = plan("why did INC-2026-0519 happen in May 2026",
                 intent="root_cause",
                 aspects=("cause", "financial_impact"),
                 evidence=("incident_report",), sellers=("S001",),
                 identifiers=("INC-2026-0519",))
        weights = CompatibilityWeights()
        adjusted = query_aware_rerank.adjust(0.50, c, p, weights)
        self.assertLessEqual(adjusted - 0.50, weights.total_weight + 1e-9)
        self.assertLessEqual(query_aware_rerank.compatibility(c, p, weights),
                             1.0 + 1e-9)

    def test_signals_cannot_overwhelm_a_clear_jina_preference(self):
        # Jina strongly prefers the passage with no plan signals at all.
        weak_but_matching = chunk(
            "MATCHES", "root cause INC-2026-0519 May 2026 refund",
            seller="S001", doc_type="Incident Report")
        strong_but_plain = chunk("PLAIN", "prose", doc_type="Account Overview")
        p = plan("why did INC-2026-0519 happen in May 2026",
                 intent="root_cause", aspects=("cause",),
                 evidence=("incident_report",), sellers=("S001",),
                 identifiers=("INC-2026-0519",))
        with plan_aware_enabled(), \
             mock.patch.object(reranker, "_request",
                               return_value=jina([(1, 0.70), (0, 0.55)])):
            outcome = reranker.rerank("q", [weak_but_matching,
                                            strong_but_plain], plan=p,
                                      weights=ALL_SIGNALS)
        self.assertEqual(order(outcome)[0], "PLAIN")


class AnswerBearingTest(RerankerTestCase):
    """The core Phase 7 objective: answer-bearing above merely related."""

    def test_aspect_coverage_breaks_a_near_tie(self):
        related = chunk("ACC", "Incident INC-2026-0519 affected seller S001.",
                        doc_type="Account Overview")
        answering = chunk(
            "INC", "INC-2026-0519 was caused by a database timeout.",
            doc_type="Incident Report")
        p = plan("what caused INC-2026-0519", intent="root_cause",
                 aspects=("cause",), evidence=("incident_report",),
                 identifiers=("INC-2026-0519",))
        with plan_aware_enabled(), \
             mock.patch.object(reranker, "_request",
                               return_value=jina([(0, 0.71), (1, 0.70)])):
            outcome = reranker.rerank("q", [related, answering], plan=p,
                                      weights=ALL_SIGNALS)
        self.assertEqual(order(outcome)[0], "INC")

    def test_multi_part_query_prefers_the_passage_covering_both_aspects(self):
        one = chunk("ONE", "The root cause was a failed migration.",
                    doc_type="Incident Report")
        both = chunk("BOTH",
                     "The root cause was a failed migration; the refund "
                     "settlement absorbed INR 40,000.",
                     doc_type="Incident Report")
        p = plan("what caused it and what was the financial impact",
                 intent="root_cause",
                 aspects=("cause", "financial_impact"))
        self.assertGreater(
            query_aware_rerank.compatibility(both, p, ALL_SIGNALS),
            query_aware_rerank.compatibility(one, p, ALL_SIGNALS))

    def test_exact_event_outranks_a_different_event_of_the_same_seller(self):
        wrong = chunk("INC-2026-0602",
                      "Incident INC-2026-0602 summary for S001.",
                      seller="S001", doc_type="Incident Report")
        right = chunk("INC-2026-0519",
                      "Incident INC-2026-0519 summary for S001.",
                      seller="S001", doc_type="Incident Report")
        p = plan("what happened in INC-2026-0519", intent="factual_lookup",
                 identifiers=("INC-2026-0519",), sellers=("S001",))
        with plan_aware_enabled(), \
             mock.patch.object(reranker, "_request",
                               return_value=jina([(0, 0.66), (1, 0.65)])):
            outcome = reranker.rerank("q", [wrong, right], plan=p,
                                      weights=ALL_SIGNALS)
        self.assertEqual(order(outcome)[0], "INC-2026-0519")


class NoFloodingTest(RerankerTestCase):
    """The Phase 6 lessons, re-tested at the reranking stage."""

    def test_bare_seller_code_is_not_treated_as_an_event_identifier(self):
        # S001 appears in the plan identifiers exactly as extract_identifiers
        # produces it, but it must not earn event-identity credit.
        c = chunk("ANY", "Seller S001 routine correspondence.", seller="S001")
        p = plan("tell me about S001", identifiers=("S001",), sellers=("S001",))
        credit, applicable = query_aware_rerank._structured_event_credit(
            p, c.text)
        self.assertFalse(applicable)
        self.assertEqual(credit, 0.0)

    def test_seller_match_alone_cannot_outrank_the_actual_answer(self):
        # The §18 example: an org-wide policy is what the query asks for, a
        # seller account overview merely matches the mentioned seller.
        seller_doc = chunk("S001-ACC-001", "Account overview for S001.",
                           seller="S001", doc_type="Account Overview")
        org_policy = chunk("POL-OPS-006",
                           "The escalation policy requires a response within "
                           "four hours.",
                           seller="Organization-wide", doc_type="Policy")
        p = plan("what is the organization-wide escalation policy",
                 intent="policy", aspects=("policy_requirement",),
                 evidence=("policy",), sellers=("S001",))
        with plan_aware_enabled(), \
             mock.patch.object(reranker, "_request",
                               return_value=jina([(0, 0.62), (1, 0.61)])):
            outcome = reranker.rerank("q", [seller_doc, org_policy], plan=p,
                                      weights=ALL_SIGNALS)
        self.assertEqual(order(outcome)[0], "POL-OPS-006")

    def test_seller_query_does_not_auto_promote_a_generic_org_document(self):
        # A seller-specific ask: the org-wide reference should not be lifted
        # above the seller's own record just for being organization-wide.
        seller_doc = chunk("S001-ACC-001", "Account tier for S001 is Gold.",
                           seller="S001", doc_type="Account Overview")
        org_doc = chunk("ORG-001", "Organization reference material.",
                        seller="Organization-wide",
                        doc_type="Organization Reference")
        p = plan("what is the account tier for S001", intent="factual_lookup",
                 sellers=("S001",), evidence=("account_overview",))
        with plan_aware_enabled(), \
             mock.patch.object(reranker, "_request",
                               return_value=jina([(0, 0.60), (1, 0.59)])):
            outcome = reranker.rerank("q", [seller_doc, org_doc], plan=p)
        self.assertEqual(order(outcome)[0], "S001-ACC-001")

    def test_wrong_evidence_type_does_not_dominate_on_a_seller_match(self):
        # Same seller, wrong record type, versus the right record type from
        # another seller's scope. Evidence/intent fit must outweigh seller fit.
        same_seller_wrong_type = chunk(
            "CALL-1", "Routine call transcript with the seller.",
            seller="S001", doc_type="Call Transcript")
        right_type = chunk(
            "INC-1", "Incident report: the cause was a timeout.",
            seller="Organization-wide", doc_type="Incident Report")
        p = plan("what caused the incident", intent="root_cause",
                 aspects=("cause",), evidence=("incident_report",),
                 sellers=("S001",))
        # Explicit weights: this pins the RELATIVE ordering of the signals
        # (evidence must outrank seller), which is a property of the design.
        # Reading it from config would instead test the current tuning.
        self.assertGreater(
            query_aware_rerank.compatibility(right_type, p, ALL_SIGNALS),
            query_aware_rerank.compatibility(same_seller_wrong_type, p,
                                             ALL_SIGNALS))


class DisabledAndAbsentPlanTest(RerankerTestCase):
    def test_no_plan_reproduces_baseline_ordering_exactly(self):
        chunks = [chunk("A", "x"), chunk("B", "y"), chunk("C", "z")]
        results = jina([(2, 0.9), (0, 0.5), (1, 0.4)])
        with mock.patch.object(reranker, "_request", return_value=results):
            outcome = reranker.rerank("q", chunks)
        self.assertEqual(order(outcome), ["C", "A", "B"])
        self.assertFalse(outcome.plan_applied)

    def test_disabling_the_feature_reproduces_baseline_ordering(self):
        chunks = [chunk("ACC", "Incident affected S001.",
                        doc_type="Account Overview"),
                  chunk("INC", "The cause was a timeout.",
                        doc_type="Incident Report")]
        p = plan("what caused it", intent="root_cause", aspects=("cause",))
        results = jina([(0, 0.71), (1, 0.70)])
        with mock.patch.object(config, "RERANK_PLAN_AWARE", False):
            with mock.patch.object(reranker, "_request", return_value=results):
                outcome = reranker.rerank("q", chunks, plan=p)
        self.assertEqual(order(outcome), ["ACC", "INC"])
        self.assertFalse(outcome.plan_applied)

    def test_zero_weight_is_ordering_neutral(self):
        chunks = [chunk("ACC", "Incident affected S001.",
                        doc_type="Account Overview"),
                  chunk("INC", "The cause was a timeout.",
                        doc_type="Incident Report")]
        p = plan("what caused it", intent="root_cause", aspects=("cause",))
        weights = CompatibilityWeights(total_weight=0.0)
        with mock.patch.object(reranker, "_request",
                               return_value=jina([(0, 0.71), (1, 0.70)])):
            outcome = reranker.rerank("q", chunks, plan=p, weights=weights)
        self.assertEqual(order(outcome), ["ACC", "INC"])


class QueryRepresentationTest(unittest.TestCase):
    def test_original_query_is_preserved_as_the_prefix(self):
        p = plan("what caused the settlement issue", intent="root_cause",
                 aspects=("cause", "financial_impact"),
                 evidence=("incident_report",))
        built = query_aware_rerank.plan_aware_query(
            "what caused the settlement issue", p)
        self.assertTrue(built.startswith("what caused the settlement issue"))
        self.assertIn("cause", built)

    def test_hint_is_compact(self):
        p = plan("q", intent="root_cause",
                 aspects=("cause", "financial_impact", "timeline", "status"),
                 evidence=("incident_report", "support_ticket",
                           "project_status"))
        built = query_aware_rerank.plan_aware_query("q", p)
        self.assertLessEqual(len(built) - len("q"),
                             query_aware_rerank.MAX_HINT_CHARS)

    def test_no_hint_when_the_plan_expresses_no_requirement(self):
        p = plan("q")
        self.assertEqual(query_aware_rerank.plan_aware_query("q", p), "q")

    def test_hints_can_be_disabled_independently(self):
        p = plan("q", intent="root_cause", aspects=("cause",))
        with mock.patch.object(config, "RERANK_PLAN_QUERY_HINTS", False):
            self.assertEqual(query_aware_rerank.plan_aware_query("q", p), "q")


class ExternalPrivacyTest(RerankerTestCase):
    """Nothing about the caller may reach the external reranker (§29)."""

    def test_request_body_carries_no_identity_or_permission_data(self):
        captured = {}

        def fake_post(url, headers=None, json=None, timeout=None):
            captured["json"] = json
            return FakeResponse(200, {"results": [
                {"index": 0, "relevance_score": 0.5}]})

        c = chunk("A", "text", seller="S001")
        p = plan("what caused it", intent="root_cause", aspects=("cause",),
                 sellers=("S001",))
        with mock.patch.object(requests, "post", side_effect=fake_post):
            reranker.rerank("what caused it", [c], plan=p)
        body = repr(captured["json"]).lower()
        for forbidden in ("user_id", "clearance", "seller_scope", "role",
                          "department", "authorized", "permission", "u-00"):
            self.assertNotIn(forbidden, body)

    def test_query_hint_contains_no_seller_scope(self):
        p = plan("q", intent="root_cause", aspects=("cause",),
                 sellers=("S001", "S002"))
        built = query_aware_rerank.plan_aware_query("q", p)
        self.assertNotIn("S001", built)
        self.assertNotIn("S002", built)


class FailureHandlingTest(RerankerTestCase):
    """§27: every documented failure degrades to the safe fusion order."""

    def _degrades_safely(self, response_factory, chunks, plan_obj):
        with mock.patch.object(config, "RERANK_MAX_RETRIES", 0):
            with mock.patch.object(requests, "post",
                                   side_effect=response_factory):
                outcome = reranker.rerank("q", chunks, threshold=0.20,
                                          plan=plan_obj)
        # Fusion order, every candidate kept, nothing scored, and the caller
        # can see that ranking did not happen.
        self.assertFalse(outcome.applied)
        self.assertFalse(outcome.plan_applied)
        self.assertEqual(order(outcome), [c.document_id for c in chunks])
        self.assertTrue(all(score is None for _, score in outcome.items))
        return outcome

    def setUp(self):
        super().setUp()
        self.chunks = [chunk("A", "a"), chunk("B", "b")]
        self.plan = plan("why", intent="root_cause", aspects=("cause",))

    def test_rate_limit_degrades_safely(self):
        self._degrades_safely(
            lambda *a, **k: FakeResponse(429, text="slow down"),
            self.chunks, self.plan)

    def test_server_error_degrades_safely(self):
        self._degrades_safely(lambda *a, **k: FakeResponse(500, text="boom"),
                              self.chunks, self.plan)

    def test_unavailable_degrades_safely(self):
        self._degrades_safely(lambda *a, **k: FakeResponse(503, text="down"),
                              self.chunks, self.plan)

    def test_timeout_degrades_safely(self):
        def boom(*a, **k):
            raise requests.Timeout("timed out")
        self._degrades_safely(boom, self.chunks, self.plan)

    def test_malformed_json_degrades_safely(self):
        self._degrades_safely(
            lambda *a, **k: FakeResponse(200, _MALFORMED), self.chunks,
            self.plan)

    def test_empty_results_degrade_safely(self):
        self._degrades_safely(
            lambda *a, **k: FakeResponse(200, {"results": []}), self.chunks,
            self.plan)

    def test_recall_is_preserved_on_failure_even_with_a_threshold(self):
        outcome = self._degrades_safely(
            lambda *a, **k: FakeResponse(503, text="down"), self.chunks,
            self.plan)
        # A threshold must not be applied to unscored candidates, or a Jina
        # outage would empty the context.
        self.assertEqual(len(outcome.items), len(self.chunks))


class AuthorizationBoundaryTest(unittest.TestCase):
    """§28: the reranker only ever sees already-authorized candidates."""

    def test_service_reranks_exactly_the_authorized_chunks(self):
        from answering.src import service
        from access_control.src.retrieval import AccessResponse

        authorized = [chunk("A", "a"), chunk("B", "b")]
        denied = chunk("SECRET", "restricted material", seller="S004")
        captured = {}

        def fake_rerank(query, chunks, threshold=None, plan=None, weights=None):
            captured["chunks"] = list(chunks)
            captured["plan"] = plan
            return reranker.RerankOutcome([(c, 0.9) for c in chunks], True)

        response = AccessResponse(
            user_id="U-001", query="q", request_id="r1",
            chunks=authorized, withheld_count=1, refused=False)

        with mock.patch.object(service, "authorized_retrieve",
                               return_value=response), \
             mock.patch.object(service.reranker, "rerank",
                               side_effect=fake_rerank), \
             mock.patch.object(service, "build_query_plan",
                               return_value=plan("q")):
            service._prepare("U-001", "q", None, None, 5, "r1",
                             config.DEFAULT_ANSWERING)

        sent = [c.document_id for c in captured["chunks"]]
        self.assertEqual(sent, ["A", "B"])
        self.assertNotIn(denied.document_id, sent)

    def test_plan_reaches_the_reranker_when_query_aware_is_enabled(self):
        from answering.src import service
        from access_control.src.retrieval import AccessResponse

        captured = {}

        def fake_rerank(query, chunks, threshold=None, plan=None, weights=None):
            captured["plan"] = plan
            return reranker.RerankOutcome([(c, 0.9) for c in chunks], True)

        response = AccessResponse(
            user_id="U-001", query="q", request_id="r1",
            chunks=[chunk("A", "a")], withheld_count=0, refused=False)
        built = plan("why did it fail", intent="root_cause", aspects=("cause",))

        with mock.patch.object(config, "QUERY_AWARE_ENABLED", True), \
             mock.patch.object(service, "authorized_retrieve",
                               return_value=response), \
             mock.patch.object(service.reranker, "rerank",
                               side_effect=fake_rerank), \
             mock.patch.object(service, "build_query_plan", return_value=built):
            service._prepare("U-001", "why did it fail", None, None, 5, "r1",
                             config.DEFAULT_ANSWERING)

        self.assertIsNotNone(captured["plan"])
        self.assertEqual(captured["plan"].intent, "root_cause")


class CompatibilitySemanticsTest(RerankerTestCase):
    def test_inapplicable_signals_are_excluded_from_the_denominator(self):
        # With no aspects/evidence/seller in the plan, only intent applies, so
        # a genre match must score the full 1.0 rather than a diluted fraction.
        c = chunk("INC", "text", doc_type="Incident Report")
        p = plan("why", intent="root_cause")
        self.assertAlmostEqual(
            query_aware_rerank.compatibility(c, p, ALL_SIGNALS), 1.0)

    def test_plan_with_no_expressed_requirement_scores_zero(self):
        c = chunk("A", "text")
        p = plan("q", intent="factual_lookup")
        self.assertEqual(query_aware_rerank.compatibility(c, p), 0.0)

    def test_generic_intent_expresses_no_genre_preference(self):
        self.assertEqual(
            query_aware_rerank.INTENT_EVIDENCE_AFFINITY["factual_lookup"],
            frozenset())

    def test_unauthorized_request_intent_expresses_no_preference(self):
        self.assertEqual(
            query_aware_rerank.INTENT_EVIDENCE_AFFINITY["unauthorized_request"],
            frozenset())

    def test_ablation_view_keeps_the_bound_and_drops_other_signals(self):
        full = CompatibilityWeights()
        only_aspect = full.only("aspect")
        self.assertEqual(only_aspect.total_weight, full.total_weight)
        self.assertEqual(only_aspect.seller, 0.0)
        self.assertEqual(only_aspect.aspect, full.aspect)

    def test_date_identity_earns_partial_event_credit(self):
        c = chunk("A", "The May 2026 reconciliation run failed.")
        p = plan("what happened in May 2026")
        credit, applicable = query_aware_rerank._structured_event_credit(
            p, c.text)
        self.assertTrue(applicable)
        self.assertEqual(credit, query_aware_rerank.DATE_IDENTITY_CREDIT)

    def test_ties_keep_jina_ordering(self):
        a = chunk("A", "plain")
        b = chunk("B", "plain")
        p = plan("q", intent="root_cause")
        with mock.patch.object(reranker, "_request",
                               return_value=jina([(0, 0.5), (1, 0.5)])):
            outcome = reranker.rerank("q", [a, b], plan=p)
        self.assertEqual(order(outcome), ["A", "B"])


class ShippedTuningTest(RerankerTestCase):
    """Pins the configuration the Phase 7 ablation actually selected.

    These are not arbitrary constants: every one of them was measured on the 45
    scored golden cases. If a future change moves them, the ablation has to be
    re-run, and this test is the reminder.

    Each assertion is skipped when the corresponding variable is explicitly
    overridden in the environment. Overriding is a supported operation, so it
    must not be reported as a failure - the test pins the SHIPPED DEFAULT, not
    whatever the current shell happens to say.
    """

    def _require_default(self, name: str):
        if os.getenv(name) is not None:
            self.skipTest(f"{name} is overridden in this environment")

    def test_production_weights_come_from_config_not_class_defaults(self):
        weights = CompatibilityWeights.from_config()
        self.assertEqual(weights.total_weight, config.RERANK_PLAN_WEIGHT)
        self.assertEqual(weights.evidence, config.RERANK_PLAN_EVIDENCE_WEIGHT)

    def test_only_the_evidence_signal_ships_enabled(self):
        for name in ("RERANK_PLAN_EVIDENCE_WEIGHT", "RERANK_PLAN_ASPECT_WEIGHT",
                     "RERANK_PLAN_EVENT_WEIGHT", "RERANK_PLAN_INTENT_WEIGHT",
                     "RERANK_PLAN_SELLER_WEIGHT"):
            self._require_default(name)
        weights = CompatibilityWeights.from_config()
        self.assertGreater(weights.evidence, 0.0)
        self.assertEqual(weights.aspect, 0.0)
        self.assertEqual(weights.event, 0.0)
        self.assertEqual(weights.intent, 0.0)
        self.assertEqual(weights.seller, 0.0)

    def test_query_hints_ship_disabled(self):
        # Measured worse than leaving the user's query alone.
        self._require_default("RERANK_PLAN_QUERY_HINTS")
        self.assertFalse(config.RERANK_PLAN_QUERY_HINTS)

    def test_shipped_bound_keeps_jina_dominant(self):
        # The entire plan term must stay far below the gap Jina puts between an
        # on-topic (>0.5) and an off-topic (<0.05) passage. The bound is also
        # enforced structurally by BoundednessTest regardless of configuration;
        # this assertion pins the value we actually ship.
        self._require_default("RERANK_PLAN_WEIGHT")
        self.assertLessEqual(config.RERANK_PLAN_WEIGHT, 0.10)

    def test_default_path_promotes_the_predicted_evidence_type(self):
        # End to end on the shipped configuration, with no explicit weights.
        self._require_default("RERANK_PLAN_WEIGHT")
        self._require_default("RERANK_PLAN_EVIDENCE_WEIGHT")
        wrong_genre = chunk("ACC", "Background on the incident.",
                            doc_type="Account Overview")
        right_genre = chunk("INC", "Incident report detail.",
                            doc_type="Incident Report")
        p = plan("what caused the incident", intent="root_cause",
                 evidence=("incident_report",))
        with plan_aware_enabled(), \
             mock.patch.object(reranker, "_request",
                               return_value=jina([(0, 0.71), (1, 0.70)])):
            outcome = reranker.rerank("q", [wrong_genre, right_genre], plan=p)
        self.assertEqual(order(outcome)[0], "INC")


if __name__ == "__main__":
    unittest.main()
