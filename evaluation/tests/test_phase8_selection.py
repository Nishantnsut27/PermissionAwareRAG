"""Phase 8 tests: evidence-aware selection.

Run with:  .venv\\Scripts\\python.exe -m unittest discover -s evaluation/tests -v

No network calls: candidates are synthetic and scores are supplied directly.

These tests pin the selector's BEHAVIOUR (what it guarantees) rather than the
shipped tuning, because the shipped tuning is deliberately recall-neutral - see
docs/phase8-evidence-aware-selection.md. Tests that need a signal to actually
bite supply explicit weights, so they keep working if the defaults change.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from answering.src import config, evidence_selection, selection
from answering.src.evidence_selection import EvidenceWeights
from answering.src.query_planner import QueryPlan
from indexing.src.models import RetrievalResult

CFG = config.DEFAULT_ANSWERING
# Weights that make every coverage signal actually fire, for behaviour tests.
ACTIVE = EvidenceWeights(aspect=0.30, evidence_type=0.15, entity=0.20,
                         redundancy=0.25, min_marginal_utility=0.0)


class EvidenceTestCase(unittest.TestCase):
    """Base for tests that exercise the evidence selector.

    `selection.select` falls through to the score-ordered path when Phase 8 is
    switched off, so a test of evidence behaviour has to assert that
    precondition rather than inherit it from the developer's .env - otherwise
    the suite reports on the environment instead of on the code.
    """

    def setUp(self):
        patch = mock.patch.object(config, "EVIDENCE_SELECTION_ENABLED", True)
        patch.start()
        self.addCleanup(patch.stop)


def chunk(document_id, text="body", index=0, seller="S001",
          doc_type="Support Ticket"):
    return RetrievalResult(
        f"{document_id}_chunk_{index:03d}", document_id, text, 0.0, "hybrid",
        {"document_id": document_id, "seller_id": seller,
         "classification": "INTERNAL", "document_type": doc_type,
         "department": "Support", "chunk_index": index})


def plan(query="q", aspects=(), evidence=(), identifiers=(), sellers=(),
         intent="factual_lookup", multi_part=False):
    return QueryPlan(
        original_query=query, intent=intent,
        required_aspects=tuple(aspects), evidence_types=tuple(evidence),
        identifiers=tuple(identifiers), seller_scope=tuple(sellers),
        is_multi_part=multi_part)


def docs_of(selected):
    return list(dict.fromkeys(c.document_id for c, _ in selected))


class SupersetOfPhase7Test(EvidenceTestCase):
    """Phase 8 must add to Phase 7's ranking, never discard part of it."""

    def test_corroboration_signal_is_preserved(self):
        # A multi-chunk document whose passages are individually weaker can
        # still outrank a single stronger passage - that is the existing
        # `best + support_weight * rest` rule. Dropping it silently changed
        # which documents survived the cap and lost gold (measured on E021).
        spread = [(chunk("SPREAD", "alpha", 0), 0.50),
                  (chunk("SPREAD", "beta", 1), 0.50),
                  (chunk("SPREAD", "gamma", 2), 0.50)]
        single = [(chunk("SINGLE", "delta", 0), 0.60)]
        items = sorted(single + spread, key=lambda p: -p[1])
        cfg = replace(CFG, max_documents=1)
        chosen, _ = selection.select(items, cfg, plan=plan(), weights=ACTIVE)
        self.assertEqual(docs_of(chosen), ["SPREAD"])

    def test_disabled_falls_back_to_the_existing_selector(self):
        items = [(chunk("A", "x"), 0.9), (chunk("B", "y"), 0.5)]
        with mock.patch.object(config, "EVIDENCE_SELECTION_ENABLED", False):
            chosen, stats = selection.select(items, CFG, plan=plan())
        self.assertIsNone(stats.evidence)
        self.assertEqual(docs_of(chosen), ["A", "B"])

    def test_no_plan_falls_back_to_the_existing_selector(self):
        items = [(chunk("A", "x"), 0.9), (chunk("B", "y"), 0.5)]
        chosen, stats = selection.select(items, CFG, plan=None)
        self.assertIsNone(stats.evidence)
        self.assertEqual(docs_of(chosen), ["A", "B"])

    def test_unscored_rerank_outage_falls_back(self):
        # With no scores there is no relevance to trade coverage against; the
        # safe behaviour is the existing one (keep breadth, skip every cut).
        items = [(chunk("A", "x"), None), (chunk("B", "y"), None)]
        chosen, stats = selection.select(items, CFG, plan=plan(aspects=("cause",)))
        self.assertIsNone(stats.evidence)
        self.assertEqual(len(docs_of(chosen)), 2)


class MarginalUtilityTest(EvidenceTestCase):
    """Spec section 19: the second copy of an aspect is worth less than the first."""

    def test_complementary_document_beats_a_redundant_one(self):
        # A covers cause. B repeats cause. C covers remediation slightly less
        # relevantly. After A, C must outrank B.
        a = chunk("A", "the root cause was a database timeout", 0)
        b = chunk("B", "the root cause was a database timeout", 0)
        c = chunk("C", "containment: the change was rolled back", 0)
        items = [(a, 0.90), (b, 0.80), (c, 0.70)]
        cfg = replace(CFG, max_documents=2)
        chosen, _ = selection.select(
            items, cfg, plan=plan(aspects=("cause", "containment_actions")),
            weights=ACTIVE)
        self.assertEqual(docs_of(chosen), ["A", "C"])

    def test_without_coverage_signals_pure_relevance_wins(self):
        a = chunk("A", "the root cause was a database timeout", 0)
        b = chunk("B", "the root cause was a database timeout", 0)
        c = chunk("C", "containment: the change was rolled back", 0)
        items = [(a, 0.90), (b, 0.80), (c, 0.70)]
        cfg = replace(CFG, max_documents=2)
        chosen, _ = selection.select(
            items, cfg, plan=plan(aspects=("cause", "containment_actions")),
            weights=EvidenceWeights(0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(docs_of(chosen), ["A", "B"])

    def test_new_evidence_type_is_rewarded_once(self):
        ticket = chunk("TKT", "support correspondence", 0,
                       doc_type="Support Ticket")
        incident = chunk("INC", "incident detail", 0, doc_type="Incident Report")
        items = [(ticket, 0.80), (incident, 0.75)]
        cfg = replace(CFG, max_documents=1)
        chosen, _ = selection.select(
            items, cfg, plan=plan(evidence=("incident_report",)),
            weights=replace(ACTIVE, evidence_type=0.50))
        self.assertEqual(docs_of(chosen), ["INC"])

    def test_redundancy_penalty_demotes_a_near_duplicate(self):
        text = "settlement shortfall reconciliation failure ledger mismatch"
        a = chunk("A", text, 0)
        duplicate = chunk("DUP", text, 0)
        distinct = chunk("NEW", "unrelated catalogue onboarding checklist", 0)
        items = [(a, 0.90), (duplicate, 0.80), (distinct, 0.78)]
        cfg = replace(CFG, max_documents=2)
        chosen, _ = selection.select(items, cfg, plan=plan(),
                                     weights=replace(ACTIVE, redundancy=0.60))
        self.assertEqual(docs_of(chosen), ["A", "NEW"])


class OutputOrderingTest(EvidenceTestCase):
    """Selection decides the SET; the emitted ORDER stays Phase 7 relevance.

    Ordering the context is a packing concern (Phase 9, spec section 50). The
    greedy loop may pick a lower-relevance document first because it covers a
    new aspect, but emitting it ahead of a higher-relevance document would demote
    that document for no recall gain - the E020 regression. So the chosen set is
    emitted best-relevance-first regardless of the order it was selected in.
    """

    def test_selected_set_is_emitted_in_relevance_order_not_greedy_order(self):
        # COVER has lower relevance (0.70) but covers the required aspect, so
        # the greedy loop selects it FIRST. GOLD has higher relevance (0.90) and
        # no aspect marker. Both are kept; the output must still lead with GOLD.
        gold = chunk("GOLD", "quarterly revenue summary and outlook", 0)
        cover = chunk("COVER", "the root cause was a database timeout", 0)
        items = [(gold, 0.90), (cover, 0.70)]
        cfg = replace(CFG, max_documents=2)
        chosen, stats = selection.select(
            items, cfg, plan=plan(aspects=("cause",)),
            weights=replace(ACTIVE, aspect=0.30))
        # Emitted best-relevance-first...
        self.assertEqual(docs_of(chosen), ["GOLD", "COVER"])
        # ...even though the greedy selector reached COVER first (preserved in
        # the diagnostics, which record selection order).
        selected = [d["document_id"] for d in stats.evidence["decisions"]
                    if d.get("selected")]
        self.assertEqual(selected, ["COVER", "GOLD"])


class SameDocumentChunksTest(EvidenceTestCase):
    """Spec section 15: complementary chunks of one document must both survive."""

    def test_two_complementary_chunks_of_one_document_are_kept(self):
        cause = chunk("DOC", "the root cause was a failed migration", 0)
        remediation = chunk("DOC", "containment: the change was rolled back", 1)
        background = chunk("DOC", "the seller joined the platform in 2024", 2)
        items = [(cause, 0.90), (background, 0.85), (remediation, 0.80)]
        cfg = replace(CFG, max_documents=1, max_chunks_per_document=2)
        chosen, _ = selection.select(
            items, cfg, plan=plan(aspects=("cause", "containment_actions")),
            weights=ACTIVE)
        kept = [c.chunk_id for c, _ in chosen]
        self.assertIn("DOC_chunk_000", kept)
        self.assertIn("DOC_chunk_001", kept)
        self.assertNotIn("DOC_chunk_002", kept)

    def test_chunks_are_returned_in_reading_order(self):
        first = chunk("DOC", "the root cause was a failed migration", 0)
        second = chunk("DOC", "containment: the change was rolled back", 1)
        items = [(second, 0.90), (first, 0.85)]
        cfg = replace(CFG, max_documents=1, max_chunks_per_document=2)
        chosen, _ = selection.select(
            items, cfg, plan=plan(aspects=("cause", "containment_actions")),
            weights=ACTIVE)
        self.assertEqual([c.metadata["chunk_index"] for c, _ in chosen], [0, 1])


class CapacityAndFloorTest(EvidenceTestCase):
    def test_document_cap_is_respected(self):
        items = [(chunk(f"D{i}", f"text {i}", 0), 0.9 - i * 0.05)
                 for i in range(10)]
        cfg = replace(CFG, max_documents=3)
        chosen, _ = selection.select(items, cfg, plan=plan(), weights=ACTIVE)
        self.assertEqual(len(docs_of(chosen)), 3)

    def test_chunk_cap_is_respected(self):
        items = [(chunk("DOC", f"passage {i}", i), 0.9 - i * 0.01)
                 for i in range(6)]
        cfg = replace(CFG, max_documents=1, max_chunks_per_document=2)
        chosen, _ = selection.select(items, cfg, plan=plan(), weights=ACTIVE)
        self.assertEqual(len(chosen), 2)

    def test_coverage_cannot_rescue_an_irrelevant_document(self):
        # Spec section 22: coverage rescues a moderately relevant document,
        # never one below the relative floor.
        strong = chunk("STRONG", "unrelated prose", 0)
        irrelevant = chunk("WEAK", "the root cause was a database timeout", 0)
        items = [(strong, 0.90), (irrelevant, 0.05)]
        cfg = replace(CFG, max_documents=5)
        chosen, _ = selection.select(
            items, cfg, plan=plan(aspects=("cause",)),
            weights=replace(ACTIVE, aspect=0.90))
        self.assertEqual(docs_of(chosen), ["STRONG"])

    def test_stopping_rule_never_empties_the_context(self):
        items = [(chunk("A", "x"), 0.9), (chunk("B", "y"), 0.8)]
        chosen, stats = selection.select(
            items, CFG, plan=plan(),
            weights=replace(ACTIVE, min_marginal_utility=99.0))
        self.assertEqual(docs_of(chosen), ["A"])
        self.assertTrue(stats.evidence["stopped_early"])


class QueryShapeTest(EvidenceTestCase):
    def test_query_without_required_aspects_falls_back_to_relevance(self):
        # Spec section 37: no aspects must not mean fabricated aspects.
        items = [(chunk("A", "x"), 0.9), (chunk("B", "y"), 0.8),
                 (chunk("C", "z"), 0.7)]
        chosen, stats = selection.select(items, CFG, plan=plan(aspects=()),
                                         weights=ACTIVE)
        self.assertEqual(docs_of(chosen), ["A", "B", "C"])
        self.assertIsNone(stats.evidence["aspect_coverage"])

    def test_identifier_evidence_stays_dominant(self):
        # Spec section 35: diversity must not push the exact incident below
        # generic supporting material.
        incident = chunk("INC-2026-0519", "incident INC-2026-0519 detail", 0,
                         doc_type="Incident Report")
        generic = chunk("ORG-001", "organization reference material", 0,
                        seller="Organization-wide",
                        doc_type="Organization Reference")
        items = [(incident, 0.85), (generic, 0.80)]
        cfg = replace(CFG, max_documents=1)
        chosen, _ = selection.select(
            items, cfg, plan=plan(query="what happened in INC-2026-0519",
                                  identifiers=("INC-2026-0519",)),
            weights=ACTIVE)
        self.assertEqual(docs_of(chosen), ["INC-2026-0519"])

    def test_organization_evidence_is_not_penalised_for_being_org_wide(self):
        # Spec section 36: an org-wide policy answer must be allowed to win.
        policy = chunk("POL-OPS-006", "the escalation policy requires...", 0,
                       seller="Organization-wide", doc_type="Policy")
        seller_doc = chunk("S001-ACC-001", "account overview", 0,
                           doc_type="Account Overview")
        items = [(policy, 0.85), (seller_doc, 0.80)]
        cfg = replace(CFG, max_documents=1)
        chosen, _ = selection.select(
            items, cfg,
            plan=plan(query="what is the escalation policy", intent="policy",
                      evidence=("policy",), sellers=("S001",)),
            weights=ACTIVE)
        self.assertEqual(docs_of(chosen), ["POL-OPS-006"])

    def test_bare_seller_code_is_not_an_event_key(self):
        # The Phase 6 flood lesson, re-tested at the selection stage.
        p = plan(query="tell me about S001", identifiers=("S001",),
                 sellers=("S001",))
        keys = evidence_selection._entity_keys(p, "seller S001 correspondence")
        self.assertEqual(keys, frozenset())


class DeterminismTest(EvidenceTestCase):
    """Spec section 39."""

    def test_repeated_selection_is_identical(self):
        items = [(chunk(f"D{i}", f"shared settlement text {i}", 0), 0.9 - i * 0.03)
                 for i in range(8)]
        p = plan(aspects=("cause",), evidence=("support_ticket",))
        runs = [docs_of(selection.select(items, CFG, plan=p, weights=ACTIVE)[0])
                for _ in range(5)]
        self.assertEqual(len(set(map(tuple, runs))), 1)

    def test_equal_utility_keeps_input_order(self):
        items = [(chunk("A", "same text", 0), 0.5),
                 (chunk("B", "same text", 0), 0.5)]
        cfg = replace(CFG, max_documents=2)
        chosen, _ = selection.select(items, cfg, plan=plan(), weights=ACTIVE)
        self.assertEqual(docs_of(chosen), ["A", "B"])


class AuthorizationTest(EvidenceTestCase):
    """Spec sections 24-25: selection is subtractive and cannot widen access."""

    def test_selector_cannot_introduce_a_candidate(self):
        items = [(chunk("A", "x"), 0.9), (chunk("B", "y"), 0.8)]
        chosen, _ = selection.select(items, CFG, plan=plan(aspects=("cause",)),
                                     weights=ACTIVE)
        given = {c.chunk_id for c, _ in items}
        returned = {c.chunk_id for c, _ in chosen}
        self.assertTrue(returned.issubset(given))

    def test_high_coverage_cannot_resurrect_an_absent_document(self):
        # Anything not handed to the selector simply cannot appear.
        items = [(chunk("PRESENT", "the root cause was a timeout"), 0.9)]
        chosen, _ = selection.select(
            items, CFG, plan=plan(aspects=("cause",)), weights=ACTIVE)
        self.assertEqual(docs_of(chosen), ["PRESENT"])

    def test_final_gate_still_runs_after_selection(self):
        # service._prepare re-checks every selected chunk with can_access.
        import inspect
        from answering.src import service
        source = inspect.getsource(service._prepare)
        gate = source.index("_engine.can_access")
        select_call = source.index("selection.select")
        self.assertLess(select_call, gate,
                        "the final authorization gate must follow selection")


class DiagnosticsTest(EvidenceTestCase):
    """Spec sections 30 and 40."""

    def test_stats_report_coverage_and_reasons(self):
        a = chunk("A", "the root cause was a database timeout", 0,
                  doc_type="Incident Report")
        b = chunk("B", "containment: the change was rolled back", 0)
        items = [(a, 0.90), (b, 0.70)]
        _chosen, stats = selection.select(
            items, CFG,
            plan=plan(aspects=("cause", "containment_actions"),
                      evidence=("incident_report",)),
            weights=ACTIVE)
        evidence = stats.evidence
        self.assertTrue(evidence["applied"])
        self.assertEqual(evidence["aspect_coverage"], 1.0)
        self.assertIn("incident_report", evidence["evidence_types"])
        self.assertTrue(any(d.get("selected") for d in evidence["decisions"]))
        self.assertTrue(any("new_aspects" in d for d in evidence["decisions"]))

    def test_redundancy_rate_is_zero_for_one_document(self):
        items = [(chunk("A", "x"), 0.9)]
        _chosen, stats = selection.select(items, CFG, plan=plan(),
                                          weights=ACTIVE)
        self.assertEqual(stats.evidence["redundancy_rate"], 0.0)

    def test_similarity_is_symmetric_and_bounded(self):
        left = evidence_selection._tokens("settlement shortfall ledger")
        right = evidence_selection._tokens("settlement ledger mismatch")
        forward = evidence_selection._similarity(left, right)
        backward = evidence_selection._similarity(right, left)
        self.assertEqual(forward, backward)
        self.assertGreaterEqual(forward, 0.0)
        self.assertLessEqual(forward, 1.0)

    def test_stopwords_do_not_create_similarity(self):
        left = evidence_selection._tokens("the and of for with that this")
        self.assertEqual(left, frozenset())


class ShippedConfigurationTest(unittest.TestCase):
    """The default ships recall-neutral; the precision lever is off.

    Measured on the Phase 8 dev split: every positive stopping threshold bought
    precision by losing more recall, so the guard in spec section 48 forbids
    enabling it. This test records that decision.
    """

    def test_stopping_rule_ships_disabled(self):
        import os
        if os.getenv("EVIDENCE_MIN_MARGINAL_UTILITY") is not None:
            self.skipTest("overridden in this environment")
        self.assertEqual(config.EVIDENCE_MIN_MARGINAL_UTILITY, 0.0)

    def test_default_configuration_is_recall_neutral_vs_phase7(self):
        items = [(chunk(f"D{i}", f"settlement text {i}", 0), 0.9 - i * 0.05)
                 for i in range(8)]
        p = plan(aspects=("cause",), evidence=("support_ticket",))
        phase7, _ = selection.select(items, CFG, plan=None)
        phase8, _ = selection.select(items, CFG, plan=p)
        self.assertEqual(set(docs_of(phase7)), set(docs_of(phase8)))


if __name__ == "__main__":
    unittest.main()
