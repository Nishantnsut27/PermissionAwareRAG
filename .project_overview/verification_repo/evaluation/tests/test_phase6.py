"""Phase 6 tests: query-aware retrieval.

Run with:  .venv\\Scripts\\python.exe -m unittest discover -s evaluation/tests -v
No test calls Qdrant, Jina, Groq or OpenJEV: retrieval, rerank and generation
are mocked; identifier candidates come from a small synthetic corpus.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from access_control.src import retrieval as access
from access_control.src.engine import PermissionEngine
from answering.src import config, query_aware, service
from answering.src.query_planner import QueryPlan
from indexing.src.models import RetrievalResult


def chunk(document_id, seller="S001", text="body", doc_type="Support Ticket",
          index=0, method="hybrid"):
    return RetrievalResult(
        f"{document_id}_chunk_{index:03d}", document_id, text, 0.0, method,
        {"document_id": document_id, "seller_id": seller,
         "classification": "INTERNAL", "document_type": doc_type,
         "department": "Support", "chunk_index": index})


def plan(query="q", identifiers=(), entities=(), sellers=(), evidence=(),
         intent="factual_lookup"):
    return QueryPlan(
        original_query=query, intent=intent, entities=tuple(entities),
        seller_scope=tuple(sellers), identifiers=tuple(identifiers),
        evidence_types=tuple(evidence))


SYNTHETIC_CORPUS = [
    {"chunk_id": "INC-2026-0519_chunk_001", "document_id": "INC-2026-0519",
     "text": "Incident INC-2026-0519 report. Root cause was a feed failure.",
     "document_type": "Incident Report", "seller_id": "S002",
     "department": "Support", "classification": "CONFIDENTIAL",
     "scenario_id": "SC-002", "source_file": "x.pdf", "page_numbers": [1]},
    {"chunk_id": "INC-2026-05190_chunk_001", "document_id": "INC-2026-05190",
     "text": "Unrelated incident INC-2026-05190 about something else.",
     "document_type": "Incident Report", "seller_id": "S002",
     "department": "Support", "classification": "CONFIDENTIAL",
     "scenario_id": "SC-002", "source_file": "y.pdf", "page_numbers": [1]},
    {"chunk_id": "TKT-2026-0805-301_chunk_001",
     "document_id": "TKT-2026-0805-301",
     "text": "Support ticket TKT-2026-0805-301 about the StrideOne outage.",
     "document_type": "Support Ticket", "seller_id": "S004",
     "department": "Support", "classification": "INTERNAL",
     "scenario_id": "SC-005", "source_file": "t.pdf", "page_numbers": [1]},
    {"chunk_id": "POL-OPS-006_chunk_001", "document_id": "POL-OPS-006",
     "text": "Account suspension policy POL-OPS-006 trigger thresholds.",
     "document_type": "Policy", "seller_id": "Organization-wide",
     "department": "Operations", "classification": "RESTRICTED",
     "scenario_id": "SC-001", "source_file": "p.pdf", "page_numbers": [2]},
    {"chunk_id": "S001-ACC-001_chunk_001", "document_id": "S001-ACC-001",
     "text": "Account overview for seller S001 - Aurelia Home Decor.",
     "document_type": "Account Overview", "seller_id": "S001",
     "department": "Business", "classification": "INTERNAL",
     "scenario_id": "SC-004", "source_file": "a.pdf", "page_numbers": [1]},
]


def corpus_of(entries):
    """Accept RetrievalResult chunks or raw corpus dicts."""
    rows = []
    for entry in entries:
        if isinstance(entry, dict):
            rows.append(dict(entry))
        else:
            rows.append({
                "chunk_id": entry.chunk_id, "document_id": entry.document_id,
                "text": entry.text,
                "document_type": entry.metadata.get("document_type"),
                "seller_id": entry.metadata.get("seller_id"),
                "department": entry.metadata.get("department"),
                "classification": entry.metadata.get("classification"),
                "scenario_id": entry.metadata.get("scenario_id"),
                "source_file": None, "page_numbers": []})
    return rows


class IdentifierMatchTest(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(
            query_aware.identifier_in_text("INC-2026-0519",
                                           "see incident INC-2026-0519 now"))

    def test_rejects_longer_supersets(self):
        self.assertFalse(
            query_aware.identifier_in_text("INC-2026-0519",
                                           "records INC-2026-05190 and more"))
        self.assertFalse(
            query_aware.identifier_in_text("TKT-2026-0805-301",
                                           "ticket TKT-2026-0805-3011 open"))
        self.assertFalse(
            query_aware.identifier_in_text("PRJ-S001-003",
                                           "project PRJ-S001-0033 done"))

    def test_rejects_shorter_substrings(self):
        self.assertFalse(
            query_aware.identifier_in_text("INC-2026-0519",
                                           "id INC-2026-051 was partial"))
        self.assertFalse(
            query_aware.identifier_in_text("S001", "account S001-ACC-001"))

    def test_rejects_year_and_prefix_variants(self):
        self.assertFalse(
            query_aware.identifier_in_text("INC-2026-0519",
                                           "older INC-2025-0519 case"))
        self.assertFalse(
            query_aware.identifier_in_text("S001", "code XS001 embedded"))

    def test_multiple_identifiers_all_matchable(self):
        text = "compare INC-2026-0519 with INC-2026-0520 today"
        self.assertTrue(query_aware.identifier_in_text("INC-2026-0519", text))
        self.assertTrue(query_aware.identifier_in_text("INC-2026-0520", text))

    def test_case_insensitive(self):
        self.assertTrue(
            query_aware.identifier_in_text("inc-2026-0519",
                                           "INC-2026-0519 in caps"))


class IdentifierCandidateTest(unittest.TestCase):
    def test_finds_exact_identifier_chunks(self):
        matches = query_aware.identifier_candidates(
            plan(identifiers=("INC-2026-0519",)),
            corpus=corpus_of(SYNTHETIC_CORPUS))
        self.assertEqual([m.document_id for m in matches], ["INC-2026-0519"])
        self.assertEqual(matches[0].retrieval_method, "identifier")

    def test_multiple_identifiers_rank_by_match_count(self):
        matches = query_aware.identifier_candidates(
            plan(identifiers=("INC-2026-0519", "TKT-2026-0805-301")),
            corpus=corpus_of(SYNTHETIC_CORPUS))
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].score, 1.0)

    def test_no_invention_of_near_matches(self):
        matches = query_aware.identifier_candidates(
            plan(identifiers=("INC-2026-0519",)),
            corpus=corpus_of([c for c in SYNTHETIC_CORPUS
                              if c["document_id"] == "INC-2026-05190"]))
        self.assertEqual(matches, [])

    def test_existing_candidates_are_not_duplicated(self):
        existing = [chunk("INC-2026-0519", seller="S002",
                          doc_type="Incident Report", index=1)]
        corpus = corpus_of(SYNTHETIC_CORPUS)
        combined, added = query_aware.augment_fused(
            existing, plan(identifiers=("INC-2026-0519",)), corpus=corpus)
        self.assertEqual(added, 0)
        self.assertEqual(len(combined), 1)

    def test_augment_adds_missing_identifier_documents(self):
        fused = [chunk("S001-ACC-001", doc_type="Account Overview")]
        combined, added = query_aware.augment_fused(
            fused, plan(identifiers=("TKT-2026-0805-301",)),
            corpus=corpus_of(SYNTHETIC_CORPUS))
        self.assertEqual(added, 1)
        ids = [c.document_id for c in combined]
        self.assertIn("TKT-2026-0805-301", ids)
        self.assertEqual(combined[-1].retrieval_method, "identifier")

    def test_no_plan_or_no_identifiers_is_a_no_op(self):
        fused = [chunk("A")]
        combined, added = query_aware.augment_fused(fused, None)
        self.assertEqual((combined, added), (fused, 0))
        combined, added = query_aware.augment_fused(fused, plan())
        self.assertEqual((combined, added), (fused, 0))

    def test_sc_codes_do_not_spawn_candidates(self):
        combined, added = query_aware.augment_fused(
            [], plan(identifiers=("SC-002",)), corpus=corpus_of(SYNTHETIC_CORPUS))
        self.assertEqual(added, 0)

    def test_bare_seller_code_spawns_no_candidates(self):
        # S001..S004 are excluded from identifier candidates even though the
        # corpus text literally contains the code: matching them would flood
        # the pool with the seller's whole corpus (the E014 flood). The
        # SYNTHETIC_CORPUS S001-ACC-001 chunk contains "seller S001".
        matches = query_aware.identifier_candidates(
            plan(identifiers=("S001",)), corpus=corpus_of(SYNTHETIC_CORPUS))
        self.assertEqual(matches, [])
        combined, added = query_aware.augment_fused(
            [chunk("X")], plan(identifiers=("S001",)),
            corpus=corpus_of(SYNTHETIC_CORPUS))
        self.assertEqual(added, 0)
        self.assertEqual([c.document_id for c in combined], ["X"])


class SignalEvaluationTest(unittest.TestCase):
    def test_identifier_signal(self):
        candidate = chunk("INC-2026-0519", seller="S002", method="identifier",
                          text="incident INC-2026-0519 details")
        signals = query_aware.evaluate_signals(
            candidate, plan(identifiers=("INC-2026-0519",),
                            sellers=("S002",)))
        self.assertTrue(signals.identifier_match)
        self.assertTrue(signals.seller_match)

    def test_seller_match_uses_metadata_not_text_flooding(self):
        candidate = chunk("A", seller="S002")
        signals = query_aware.evaluate_signals(candidate, plan(sellers=("S002",)))
        self.assertTrue(signals.seller_match)
        self.assertFalse(signals.entity_match)

    def test_org_wide_documents_never_seller_match(self):
        candidate = chunk("POL-OPS-006", seller="Organization-wide",
                          doc_type="Policy")
        signals = query_aware.evaluate_signals(candidate, plan(sellers=("S002",)))
        self.assertFalse(signals.seller_match)

    def test_entity_name_match_in_text(self):
        candidate = chunk("A", seller="Organization-wide",
                          text="Aurelia Home Decor reported the issue")
        signals = query_aware.evaluate_signals(candidate, plan(sellers=("S001",)))
        self.assertFalse(signals.seller_match)
        self.assertTrue(signals.entity_match)

    def test_evidence_type_hint_matches_document_type(self):
        candidate = chunk("A", doc_type="Incident Report")
        signals = query_aware.evaluate_signals(
            candidate, plan(evidence=("incident_report",)))
        self.assertTrue(signals.evidence_type_match)

    def test_no_plan_yields_all_false(self):
        signals = query_aware.evaluate_signals(chunk("A"), None)
        self.assertFalse(signals.matched)

    def test_signal_score_uses_configured_weights(self):
        with mock.patch.object(config, "QUERY_AWARE_IDENTIFIER_BOOST", 1.0), \
                mock.patch.object(config, "QUERY_AWARE_ENTITY_BOOST", 0.2), \
                mock.patch.object(config, "QUERY_AWARE_SELLER_BOOST", 0.15), \
                mock.patch.object(config, "QUERY_AWARE_EVIDENCE_TYPE_BOOST", 0.1):
            signals = query_aware.ChunkSignals(
                chunk_id="x", document_id="A", retrieval_method="hybrid",
                identifier_match=True, evidence_type_match=True)
            self.assertAlmostEqual(query_aware.signal_score(signals), 1.1)


class BoostedOrderTest(unittest.TestCase):
    def test_identifier_candidates_sort_first(self):
        fused = [chunk("A"), chunk("B", seller="S002",
                                   text="mentions INC-2026-0519 inside")]
        ordered, signals = query_aware.boosted_order(
            fused, plan(identifiers=("INC-2026-0519",)))
        self.assertEqual(ordered[0].document_id, "B")
        self.assertTrue(signals[1].identifier_match)

    def test_ties_preserve_fusion_order(self):
        fused = [chunk("A"), chunk("B"), chunk("C")]
        ordered, _ = query_aware.boosted_order(fused, plan(sellers=("S001",)))
        self.assertEqual([c.document_id for c in ordered], ["A", "B", "C"])

    def test_no_matches_returns_input_order(self):
        fused = [chunk("A"), chunk("B")]
        ordered, signals = query_aware.boosted_order(fused, plan())
        self.assertEqual([c.document_id for c in ordered], ["A", "B"])
        self.assertFalse(any(s.matched for s in signals))

    def test_none_plan_is_safe(self):
        fused = [chunk("A")]
        ordered, signals = query_aware.boosted_order(fused, None)
        self.assertEqual(ordered, fused)

    def test_bare_seller_code_does_not_drive_ordering_in_mixed_query(self):
        # A query can contain BOTH a structured id and a bare seller code;
        # query_planner.extract_identifiers puts both in plan.identifiers. Only
        # the structured id may reorder. If the bare seller code (S001) drove
        # ordering it would re-promote the seller's whole block and evict the
        # org-wide policy below it - the measured E014 regression this ordering
        # exists to prevent.
        fused = [
            chunk("INC-2026-0519", seller="S002", doc_type="Incident Report",
                  text="Incident INC-2026-0519 root cause feed defect"),
            chunk("ORG-POL", seller="Organization-wide", doc_type="Policy",
                  text="Escalation policy the case needs"),
            chunk("S001-ACC-001", seller="S001", doc_type="Account Overview",
                  text="Account overview for seller S001"),
            chunk("S001-DOC-2", seller="S001", text="Seller S001 notes",
                  index=1),
        ]
        ordered, _ = query_aware.boosted_order(
            fused, plan(identifiers=("INC-2026-0519", "S001"),
                        sellers=("S001",)))
        # Only the structured-id chunk moves; the org-wide policy keeps its
        # place ahead of the seller chunks.
        self.assertEqual([c.document_id for c in ordered],
                         ["INC-2026-0519", "ORG-POL", "S001-ACC-001",
                          "S001-DOC-2"])


class PermissionSafetyTest(unittest.TestCase):
    """The critical invariant: query-aware signals never bypass authorization."""

    def setUp(self):
        self.engine = PermissionEngine()

    def test_identifier_candidate_for_unauthorized_document_is_denied(self):
        # U-001 (Business, S001/S003) asks for a BluePeak incident by exact id.
        user = access._resolve("U-001")
        unauthorized = RetrievalResult(
            "INC-2026-0519_chunk_001", "INC-2026-0519",
            "Incident INC-2026-0519 report", 0.0, "identifier",
            {"document_id": "INC-2026-0519", "seller_id": "S002",
             "classification": "CONFIDENTIAL", "document_type": "Incident Report",
             "department": "Support"})
        allowed, denied = self.engine.authorize_chunks(user, [unauthorized], "test")
        self.assertEqual(allowed, [])
        self.assertEqual(len(denied), 1)

    def test_augmented_candidates_pass_through_authorize_chunks(self):
        # The augmenter output must flow into authorize_chunks, not around it.
        captured = {}

        def fake_phase3_retrieve(query, top_k=10, filters=None):
            captured["filters"] = filters
            return [chunk("S001-ACC-001", doc_type="Account Overview")]

        def augmenter(candidates):
            combined, added = query_aware.augment_fused(
                candidates, plan(identifiers=("INC-2026-0519",)),
                corpus=corpus_of(SYNTHETIC_CORPUS))
            self.assertEqual(added, 1)
            return combined

        with mock.patch.object(access, "phase3_retrieve",
                               side_effect=fake_phase3_retrieve), \
                mock.patch.object(access._engine, "authorize_chunks",
                                  wraps=access._engine.authorize_chunks) as wrap:
            response = access.authorized_retrieve(
                "U-001", "What happened in INC-2026-0519?", top_k=5,
                candidate_augmenter=augmenter)
        self.assertIsNotNone(captured["filters"])
        self.assertEqual(wrap.call_count, 1)
        submitted = wrap.call_args.args[1]
        self.assertIn("INC-2026-0519_chunk_001",
                      [c.chunk_id for c in submitted])
        self.assertNotIn("INC-2026-0519",
                         [c.document_id for c in response.chunks])

    def test_unauthorized_seller_entity_match_is_not_retrieved(self):
        user = access._resolve("U-001")
        matching = RetrievalResult(
            "S002-ACC_chunk_001", "S002-ACC", "BluePeak Electronics account", 0.0,
            "hybrid", {"document_id": "S002-ACC", "seller_id": "S002",
                       "classification": "INTERNAL", "document_type": "Account Overview",
                       "department": "Business"})
        allowed, denied = self.engine.authorize_chunks(user, [matching], "test")
        self.assertEqual(allowed, [])
        signals = query_aware.evaluate_signals(matching, plan(sellers=("S002",)))
        self.assertTrue(signals.seller_match)
        self.assertEqual(allowed, [])

    def test_boost_cannot_reintroduce_a_denied_chunk(self):
        # The reorderer only receives already-authorized candidates.
        allowed_in = [chunk("A")]
        def reorderer(allowed):
            self.assertEqual([c.chunk_id for c in allowed],
                             [c.chunk_id for c in allowed_in])
            return query_aware.boosted_order(allowed, plan(sellers=("S001",)))[0]
        with mock.patch.object(access, "phase3_retrieve",
                               return_value=allowed_in):
            response = access.authorized_retrieve(
                "U-001", "question", top_k=5, candidate_reorderer=reorderer)
        self.assertEqual([c.document_id for c in response.chunks], ["A"])


class BackwardCompatibilityTest(unittest.TestCase):
    def test_no_callables_behaves_exactly_as_before(self):
        with mock.patch.object(access, "phase3_retrieve",
                               return_value=[chunk("A")]) as retrieve, \
                mock.patch.object(access._engine, "authorize_chunks",
                                  wraps=access._engine.authorize_chunks) as wrap:
            response = access.authorized_retrieve("U-001", "question", top_k=5)
        self.assertEqual([c.document_id for c in response.chunks], ["A"])
        self.assertEqual(wrap.call_count, 1)

    def test_service_skips_query_aware_when_disabled(self):
        with mock.patch.object(config, "QUERY_AWARE_ENABLED", False), \
                mock.patch.object(service.planner, "plan_query") as plan_call, \
                mock.patch.object(query_aware, "augment_fused") as augment, \
                mock.patch.object(query_aware, "boosted_order") as boost, \
                mock.patch.object(access, "phase3_retrieve",
                                  return_value=[chunk("A")]), \
                mock.patch("access_control.src.audit._write"), \
                mock.patch.object(reranker_module(), "rerank",
                                  return_value=reranker_module().RerankOutcome(
                                      [(chunk("A"), 0.9)], True)), \
                mock.patch.object(service.llm, "generate",
                                  return_value=("Supported", {})):
            result = service.answer_query("U-001", "What about the account?")
        plan_call.assert_not_called()
        augment.assert_not_called()
        boost.assert_not_called()
        self.assertEqual(result.sources[0]["document_id"], "A")

    def test_service_with_query_plan_but_no_signals_keeps_fusion_order(self):
        plan = QueryPlan(original_query="question", intent="factual_lookup")
        fused = [chunk("A"), chunk("B")]
        with mock.patch.object(service, "build_query_plan", return_value=plan), \
                mock.patch.object(access, "phase3_retrieve", return_value=fused), \
                mock.patch("access_control.src.audit._write"), \
                mock.patch.object(reranker_module(), "rerank",
                                  return_value=reranker_module().RerankOutcome(
                                      [(fused[0], 0.9), (fused[1], 0.8)], True)) as rerank, \
                mock.patch.object(service.llm, "generate",
                                  return_value=("Supported", {})):
            result = service.answer_query("U-001", "question")
        # Fusion order preserved into the reranker.
        self.assertEqual([c.document_id for c in rerank.call_args.args[1]],
                         ["A", "B"])
        self.assertFalse(result.pool.get("boost_applied", False))

    def test_effective_plan_resolves_follow_up_identifiers(self):
        turns = [type("T", (), {"role": "user",
                                "content": "Tell me about INC-2026-0519"})()]
        base = plan(query="how much was the impact?")
        effective = service._effective_plan(base, turns, config.DEFAULT_ANSWERING)
        self.assertEqual(effective.identifiers, ("INC-2026-0519",))

    def test_effective_plan_without_history_is_identity(self):
        base = plan(identifiers=("S001",), sellers=("S001",))
        self.assertIs(service._effective_plan(base, [], config.DEFAULT_ANSWERING),
                      base)


def reranker_module():
    from answering.src import reranker
    return reranker


class DiagnosticsTest(unittest.TestCase):
    def test_pool_diagnostics_shape(self):
        pool = query_aware.PoolDiagnostics(
            fused_candidates=20, dense_candidates=20, sparse_candidates=20,
            identifier_candidates=2, authorized_candidates=21,
            withheld_candidates=1, boost_applied=True,
            signals=[query_aware.ChunkSignals("c1", "A", "identifier",
                                              identifier_match=True)])
        data = pool.to_dict()
        self.assertEqual(data["fused_candidates"], 20)
        self.assertEqual(data["identifier_candidates"], 2)
        self.assertTrue(data["boost_applied"])
        self.assertEqual(data["signals"][0]["identifier_match"], True)

    def test_stats_pool_round_trips_through_response(self):
        plan_obj = QueryPlan(original_query="What is INC-2026-0519?",
                             intent="factual_lookup",
                             identifiers=("INC-2026-0519",))
        inc_chunk = RetrievalResult(
            "INC-2026-0519_chunk_001", "INC-2026-0519",
            "Incident INC-2026-0519 report", 0.0, "hybrid",
            {"document_id": "INC-2026-0519", "seller_id": "S002",
             "classification": "CONFIDENTIAL",
             "document_type": "Incident Report", "department": "Support"})
        fused = [chunk("S001-ACC-001", doc_type="Account Overview"), inc_chunk]
        # The pool diagnostics only exist when query-aware retrieval is on, so
        # the test must assert that configuration rather than inherit it from
        # the developer's .env.
        with mock.patch.object(config, "QUERY_AWARE_ENABLED", True), \
                mock.patch.object(service, "build_query_plan",
                                  return_value=plan_obj), \
                mock.patch.object(query_aware, "_load_corpus",
                                  return_value=corpus_of(SYNTHETIC_CORPUS)), \
                mock.patch.object(access, "phase3_retrieve", return_value=fused), \
                mock.patch("access_control.src.audit._write"), \
                mock.patch.object(reranker_module(), "rerank",
                                  return_value=reranker_module().RerankOutcome(
                                      [(fused[1], 0.95), (fused[0], 0.5)], True)), \
                mock.patch.object(service.llm, "generate",
                                  return_value=("Supported", {})):
            result = service.answer_query("U-005", "What is INC-2026-0519?")
        self.assertTrue(result.pool["boost_applied"])
        self.assertTrue(any(s["identifier_match"] for s in result.pool["signals"]))


class EvidencePromotionTest(unittest.TestCase):
    """Phase 6 rework: bounded, boundary-inserted evidence promotion."""

    POOL = 6

    def _pool(self, entries):
        return [chunk(e[0], seller=e[1], text=e[2], doc_type=e[3], index=i)
                for i, e in enumerate(entries)]

    def _plan(self, query, aspects=(), months_in_query=True):
        return QueryPlan(original_query=query, intent="root_cause",
                         required_aspects=tuple(aspects))

    def test_aspect_promotion_pulls_eligible_chunk_into_pool(self):
        query = "What caused the incident and what was the financial impact?"
        plan_obj = self._plan(query, aspects=("cause", "financial_impact"))
        pool = self._pool([
            ("D1", "S001", "account overview for the seller", "Account Overview"),
            ("D2", "S001", "another overview", "Account Overview"),
            ("D3", "S001", "more overview text", "Account Overview"),
            ("D4", "S001", "even more overview", "Account Overview"),
            ("D5", "S001", "still overview", "Account Overview"),
            ("D6", "S001", "overview again", "Account Overview"),
            ("INC", "S002", "root cause was a feed defect", "Incident Report"),
        ])
        with mock.patch.object(config, "QUERY_AWARE_PROMOTE_CRITERIA",
                               ("aspect", "date")), \
                mock.patch.object(config, "QUERY_AWARE_MAX_PROMOTED", 4):
            ordered, promoted = query_aware.promote_evidence(
                pool, plan_obj, pool_size=self.POOL)
        self.assertEqual(promoted, ["INC_chunk_006"])
        self.assertEqual(len(ordered), len(pool))
        self.assertEqual([c.chunk_id for c in ordered[:self.POOL]],
                         ["D1_chunk_000", "D2_chunk_001", "D3_chunk_002",
                          "D4_chunk_003", "D5_chunk_004", "INC_chunk_006"])

    def test_promotion_is_capped_and_prefers_new_documents(self):
        query = "What caused it?"
        plan_obj = self._plan(query, aspects=("cause",))
        pool = self._pool([
            ("D1", "S001", "plain text", "Account Overview"),
            ("D2", "S001", "plain text", "Account Overview"),
            ("D3", "S001", "plain text", "Account Overview"),
            ("D4", "S001", "plain text", "Account Overview"),
            ("D5", "S001", "plain text", "Account Overview"),
            ("D6", "S001", "plain text", "Account Overview"),
            ("INC", "S002", "root cause: defect", "Incident Report", 0),
            ("TKT", "S002", "root cause: defect", "Support Ticket", 0),
            ("INC2", "S002", "root cause: defect", "Incident Report", 0),
        ])
        with mock.patch.object(config, "QUERY_AWARE_PROMOTE_CRITERIA",
                               ("aspect", "date")), \
                mock.patch.object(config, "QUERY_AWARE_MAX_PROMOTED", 2):
            ordered, promoted = query_aware.promote_evidence(
                pool, plan_obj, pool_size=self.POOL)
        self.assertEqual(len(promoted), 2)
        new_docs = {ordered[self.POOL - 2].document_id,
                    ordered[self.POOL - 1].document_id}
        self.assertTrue(new_docs.isdisjoint(
            {c.document_id for c in pool[:self.POOL]}))

    def test_promotion_respects_evidence_weight_order(self):
        query = "Incident in June 2026, what caused it?"
        plan_obj = self._plan(query, aspects=("cause",))
        pool = self._pool([
            ("D1", "S001", "filler", "Account Overview"),
            ("D2", "S001", "filler", "Account Overview"),
            ("D3", "S001", "filler", "Account Overview"),
            ("D4", "S001", "filler", "Account Overview"),
            ("D5", "S001", "filler", "Account Overview"),
            ("D6", "S001", "filler", "Account Overview"),
            ("A", "S002", "root cause noted", "Incident Report", 0),
            ("B", "S002", "june 2026: root cause", "Incident Report", 0),
        ])
        with mock.patch.object(config, "QUERY_AWARE_PROMOTE_CRITERIA",
                               ("aspect", "date")), \
                mock.patch.object(config, "QUERY_AWARE_MAX_PROMOTED", 1):
            ordered, promoted = query_aware.promote_evidence(
                pool, plan_obj, pool_size=self.POOL)
        self.assertEqual(promoted, ["B_chunk_007"])

    def test_promotion_never_reduces_pool_size(self):
        query = "What caused it?"
        plan_obj = self._plan(query, aspects=("cause",))
        pool = self._pool([
            ("D1", "S001", "plain", "Account Overview"),
            ("D2", "S001", "plain", "Account Overview"),
            ("INC", "S002", "root cause: defect", "Incident Report", 0),
        ])
        with mock.patch.object(config, "QUERY_AWARE_PROMOTE_CRITERIA",
                               ("aspect",)), \
                mock.patch.object(config, "QUERY_AWARE_MAX_PROMOTED", 4):
            ordered, promoted = query_aware.promote_evidence(
                pool, plan_obj, pool_size=2)
        self.assertEqual(len(ordered), 3)
        self.assertEqual(len(ordered[:2]), 2)

    def test_promotion_never_swaps_chunks_within_a_pooled_document(self):
        plan_obj = self._plan("What caused the settlement issue?",
                              aspects=("cause",))
        pool = self._pool([
            ("PRJ", "S001", "neutral narrative", "Project Document"),
            ("PRJ", "S001", "more neutral narrative", "Project Document",
             1),
            ("ACC", "S001", "plain overview", "Account Overview"),
            ("ACC2", "S001", "plain overview two", "Account Overview",
             0),
        ])
        prj_extra = chunk("PRJ", seller="S001", doc_type="Project Document",
                          text="root cause was a defect", index=2)
        with mock.patch.object(config, "QUERY_AWARE_PROMOTE_CRITERIA",
                               ("aspect", "date")), \
                mock.patch.object(config, "QUERY_AWARE_MAX_PROMOTED", 4):
            ordered, promoted = query_aware.promote_evidence(
                [*pool, prj_extra], plan_obj, pool_size=3)
        self.assertEqual(promoted, [])
        self.assertEqual(
            [c.chunk_id for c in ordered],
            [c.chunk_id for c in [*pool, prj_extra]])

    def test_disabled_or_planless_promotion_is_identity(self):
        pool = self._pool([
            ("A", "S001", "root cause here", "Incident Report"),
            ("B", "S001", "plain", "Account Overview"),
        ])
        plan_obj = self._plan("What caused it?", aspects=("cause",))
        with mock.patch.object(config, "QUERY_AWARE_PROMOTION", False):
            ordered, promoted = query_aware.promote_evidence(
                pool, plan_obj, pool_size=1)
        self.assertEqual(ordered, pool)
        self.assertEqual(promoted, [])
        ordered, promoted = query_aware.promote_evidence(pool, None,
                                                         pool_size=1)
        self.assertEqual(ordered, pool)
        self.assertEqual(promoted, [])

    def test_boundary_insertion_preserves_pool_front_order(self):
        query = "What caused it?"
        plan_obj = self._plan(query, aspects=("cause",))
        pool = self._pool([
            ("D1", "S001", "plain", "Account Overview"),
            ("D2", "S001", "plain", "Account Overview"),
            ("D3", "S001", "plain", "Account Overview"),
            ("INC", "S002", "root cause: defect", "Incident Report", 0),
        ])
        with mock.patch.object(config, "QUERY_AWARE_PROMOTE_CRITERIA",
                               ("aspect",)), \
                mock.patch.object(config, "QUERY_AWARE_MAX_PROMOTED", 1):
            ordered, _ = query_aware.promote_evidence(pool, plan_obj,
                                                      pool_size=3)
        self.assertEqual([c.chunk_id for c in ordered[:3]],
                         ["D1_chunk_000", "D2_chunk_001", "INC_chunk_003"])


class AspectAndDateSignalTest(unittest.TestCase):
    def test_aspect_hits_match_plan_aspects(self):
        plan_obj = plan(query="cause and impact?",
                        evidence=())
        plan_obj = QueryPlan(
            original_query="What caused the issue and what was the impact?",
            intent="root_cause", required_aspects=("cause", "financial_impact"))
        text = "The root cause was a defect; INR 40,000 was refunded."
        self.assertEqual(query_aware.aspect_hits(plan_obj, text),
                         ("cause", "financial_impact"))

    def test_aspect_hits_empty_without_plan_or_text(self):
        self.assertEqual(query_aware.aspect_hits(None, "text"), ())
        plan_obj = QueryPlan(original_query="q", intent="root_cause",
                             required_aspects=("cause",))
        self.assertEqual(query_aware.aspect_hits(plan_obj, ""), ())

    def test_date_identity_requires_month_in_both_query_and_text(self):
        plan_obj = QueryPlan(
            original_query="What happened in September 2026?",
            intent="factual_lookup")
        self.assertEqual(query_aware.date_identity(
            plan_obj, "The September 2026 outage was mitigated."),
            frozenset({"september"}))
        self.assertEqual(query_aware.date_identity(
            plan_obj, "The October 2026 outage was mitigated."),
            frozenset())

    def test_bare_year_and_modal_may_do_not_signal(self):
        plan_obj = QueryPlan(original_query="What happened in 2026?",
                             intent="factual_lookup")
        self.assertEqual(query_aware.date_identity(
            plan_obj, "Restored in 2026."), frozenset())
        plan_obj = QueryPlan(original_query="May we know the status?",
                             intent="status")
        self.assertEqual(query_aware.date_identity(
            plan_obj, "Status remains open."), frozenset())

    def test_signals_expose_aspect_and_date_diagnostics(self):
        plan_obj = QueryPlan(
            original_query="September incident root cause",
            intent="root_cause", required_aspects=("cause",))
        target = chunk("INC", seller="S002",
                       text="September root cause: defect", doc_type="Incident Report")
        signals = query_aware.evaluate_signals(target, plan_obj)
        self.assertEqual(signals.aspect_hits, ("cause",))
        self.assertEqual(signals.date_match, frozenset({"september"}))
        data = signals.to_dict()
        self.assertEqual(data["aspect_hits"], ["cause"])
        self.assertEqual(data["date_match"], ["september"])


class EventIdentityRankingTest(unittest.TestCase):
    def test_identifier_chunk_outranks_wrong_event_in_order(self):
        plan_obj = plan(identifiers=("INC-2026-0519",),
                        query="What happened in INC-2026-0519?")
        target = chunk("INC-2026-0519", seller="S002",
                       text="Incident INC-2026-0519 report",
                       doc_type="Incident Report")
        wrong_event = chunk("INC-2026-0602", seller="S002",
                            text="Different seller incident report",
                            doc_type="Incident Report")
        other = chunk("ACC", seller="S002", text="account overview",
                      doc_type="Account Overview")
        ordered, signals = query_aware.boosted_order(
            [other, wrong_event, target], plan_obj)
        self.assertEqual(ordered[0].document_id, "INC-2026-0519")
        self.assertTrue(signals[2].identifier_match)
        self.assertFalse(signals[1].identifier_match)

    def test_root_cause_query_does_not_prefer_account_overview(self):
        plan_obj = QueryPlan(
            original_query="What caused the settlement issue?",
            intent="root_cause", evidence_types=("incident_report",
                                                 "support_ticket"))
        incident = chunk("INC", seller="S002", text="incident narrative",
                         doc_type="Incident Report")
        overview = chunk("ACC", seller="S002", text="overview of the seller",
                         doc_type="Account Overview")
        inc_signals = query_aware.evaluate_signals(incident, plan_obj)
        acc_signals = query_aware.evaluate_signals(overview, plan_obj)
        self.assertTrue(inc_signals.evidence_type_match)
        self.assertFalse(acc_signals.evidence_type_match)
        self.assertFalse(acc_signals.matched)

    def test_multi_part_aspects_both_detected(self):
        plan_obj = QueryPlan(
            original_query=("What caused the incident and what containment "
                            "actions were taken?"),
            intent="root_cause", required_aspects=("cause",
                                                   "containment_actions"))
        cause_only = query_aware.aspect_hits(
            plan_obj, "root cause was a misconfig")
        both = query_aware.aspect_hits(
            plan_obj, "root cause was a misconfig, mitigated by rollback")
        self.assertEqual(cause_only, ("cause",))
        self.assertEqual(both, ("cause", "containment_actions"))


class PromotionPermissionSafetyTest(unittest.TestCase):
    """No promotion/reorder path may bypass the PermissionEngine."""

    INC_TEXT = "Incident INC-2026-0519 root cause: feed defect"

    def _inc_result(self):
        return RetrievalResult(
            "INC-2026-0519_chunk_001", "INC-2026-0519", self.INC_TEXT, 0.0,
            "hybrid", {"document_id": "INC-2026-0519", "seller_id": "S002",
                       "classification": "CONFIDENTIAL",
                       "document_type": "Incident Report",
                       "department": "Support"})

    def test_promoted_unauthorized_chunk_is_denied_by_the_gate(self):
        with mock.patch.object(config, "QUERY_AWARE_PROMOTE_CRITERIA",
                               ("aspect", "date")), \
                mock.patch.object(config, "QUERY_AWARE_MAX_PROMOTED", 4):
            fused = [chunk("A", seller="S001", text="plain"),
                     chunk("B", seller="S001", text="plain", index=1),
                     chunk("C", seller="S001", text="plain", index=2),
                     self._inc_result()]
            plan_obj = QueryPlan(
                original_query="What caused it?", intent="root_cause",
                required_aspects=("cause",))
            ordered, promoted = query_aware.promote_evidence(
                fused, plan_obj, pool_size=2)
            self.assertEqual(promoted, ["INC-2026-0519_chunk_001"])
            engine = PermissionEngine()
            allowed, denied = engine.authorize_chunks(
                access._resolve("U-001"), ordered, "test")
        self.assertNotIn("INC-2026-0519", [c.document_id for c in allowed])
        self.assertEqual(len(denied), 1)

    def test_authorized_user_still_sees_promoted_document(self):
        with mock.patch.object(config, "QUERY_AWARE_PROMOTE_CRITERIA",
                               ("aspect", "date")), \
                mock.patch.object(config, "QUERY_AWARE_MAX_PROMOTED", 4):
            fused = [chunk("A", seller="S003", text="plain"),
                     chunk("B", seller="S003", text="plain", index=1),
                     chunk("C", seller="S003", text="plain", index=2),
                     chunk("D", seller="S003", text="plain", index=3),
                     self._inc_result()]
            plan_obj = QueryPlan(
                original_query="What caused it?", intent="root_cause",
                required_aspects=("cause",))
            ordered, _ = query_aware.promote_evidence(fused, plan_obj,
                                                      pool_size=4)
            engine = PermissionEngine()
            allowed, _denied = engine.authorize_chunks(
                access._resolve("U-005"), ordered, "test")
        self.assertIn("INC-2026-0519",
                      [c.document_id for c in allowed[:4]])

    def test_reorderer_output_cannot_contain_denied_chunks(self):
        fused = [chunk("A", seller="S001", text="plain"),
                 chunk("B", seller="S001", text="plain", index=1),
                 self._inc_result()]
        plan_obj = QueryPlan(
            original_query="What caused it?", intent="root_cause",
            required_aspects=("cause",))
        with mock.patch.object(access, "phase3_retrieve", return_value=fused):
            response = access.authorized_retrieve(
                "U-001", "What caused it?", top_k=3)
        self.assertNotIn("INC-2026-0519",
                         [c.document_id for c in response.chunks])


if __name__ == "__main__":
    unittest.main(verbosity=2)
