"""Phase 9 tests: evidence-aware context packing.

Run with:  .venv\\Scripts\\python.exe -m unittest discover -s evaluation/tests -v

No network calls: passages are synthetic and scores are supplied directly.

These pin the packer's BEHAVIOUR (its safety guarantees and the value-aware
depth rule) rather than the shipped-neutral aggregate numbers - see
docs/phase9-context-packing.md for why the benchmark cannot show a gain.
"""
from __future__ import annotations

import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from answering.src import config, context, context_packing as cp
from answering.src.query_planner import QueryPlan
from indexing.src.models import RetrievalResult

CFG = config.DEFAULT_ANSWERING


def chunk(document_id, text, index=0, seller="S001", doc_type="Support Ticket",
          department="Support", pages=None):
    return RetrievalResult(
        f"{document_id}_chunk_{index:03d}", document_id, text, 0.0, "hybrid",
        {"document_id": document_id, "seller_id": seller,
         "classification": "INTERNAL", "document_type": doc_type,
         "department": department, "chunk_index": index,
         "page_numbers": pages or []})


def plan(query="q", aspects=(), evidence=(), identifiers=(), multi_part=False):
    return QueryPlan(
        original_query=query, intent="factual_lookup",
        required_aspects=tuple(aspects), evidence_types=tuple(evidence),
        identifiers=tuple(identifiers), is_multi_part=multi_part)


def docs_of(sources):
    return [s.document_id for s in sources]


def chunk_ids(sources):
    return [cid for s in sources for cid in s.chunk_ids]


def big_budget():
    return replace(CFG, max_context_chars=10_000_000)


class SubsetAndRecallTest(unittest.TestCase):
    """Spec section 28: Phase 9 output is a subset of Phase 8 input, and the
    document set (which drives recall) is exactly the frozen builder's."""

    def test_output_chunks_are_a_subset_of_input(self):
        items = [(chunk("A", "alpha text here", 0), 0.9),
                 (chunk("B", "beta text here", 0), 0.8),
                 (chunk("A", "alpha second passage", 1), 0.7)]
        _txt, sources, _stats = cp.build_packed_with_stats(items, big_budget(),
                                                           plan())
        given = {c.chunk_id for c, _ in items}
        self.assertTrue(set(chunk_ids(sources)).issubset(given))

    def test_document_set_matches_context_build_on_every_budget(self):
        items = [(chunk(f"D{i}", f"passage {i} " * 40, 0), 0.9 - i * 0.05)
                 for i in range(5)]
        for budget in (200, 800, 2000, 6000, 10_000_000):
            cfg = replace(CFG, max_context_chars=budget)
            _b, b_src = context.build(items, cfg)
            _p, p_src, _s = cp.build_packed_with_stats(items, cfg, plan())
            self.assertEqual(set(docs_of(b_src)), set(docs_of(p_src)),
                             f"document set diverged at budget={budget}")

    def test_gold_documents_are_never_dropped_by_packing(self):
        # Every document context.build keeps, the packer keeps too (superset of
        # documents), so a gold document selected upstream cannot vanish here.
        items = [(chunk("GOLD", "the settlement shortfall was 12000 INR", 0), 0.9),
                 (chunk("OTHER", "unrelated onboarding checklist", 0), 0.4)]
        cfg = replace(CFG, max_context_chars=180)
        _b, b_src = context.build(items, cfg)
        _p, p_src, _s = cp.build_packed_with_stats(items, cfg, plan())
        self.assertEqual(set(docs_of(b_src)), set(docs_of(p_src)))


class SupersetOfBuilderTest(unittest.TestCase):
    """Phase 9 must not change behaviour when the budget has slack."""

    def test_byte_identical_to_context_build_without_budget_pressure(self):
        items = [(chunk("A", "alpha " * 30, 0), 0.9),
                 (chunk("A", "alpha two " * 30, 1), 0.6),
                 (chunk("B", "beta " * 30, 0), 0.8)]
        cfg = big_budget()
        baseline, _ = context.build(items, cfg)
        packed, _s, _st = cp.build_packed_with_stats(items, cfg, plan())
        self.assertEqual(baseline, packed)

    def test_disabled_falls_back_to_context_build(self):
        items = [(chunk("A", "alpha text", 0), 0.9)]
        with mock.patch.object(config, "CONTEXT_PACKING_ENABLED", False):
            packed, _ = cp.build_packed(items, big_budget(), plan())
        baseline, _ = context.build(items, big_budget())
        self.assertEqual(baseline, packed)

    def test_no_plan_falls_back_to_context_build(self):
        items = [(chunk("A", "alpha text", 0), 0.9)]
        packed, _ = cp.build_packed(items, big_budget(), None)
        baseline, _ = context.build(items, big_budget())
        self.assertEqual(baseline, packed)


class ValueAwareDepthTest(unittest.TestCase):
    """Spec sections 17/21: under budget pressure keep the complementary
    (new-aspect) passage over a redundant one, even when it scores lower."""

    def _budget_for_two_passages(self, header_len, text_len):
        # header + separator + first passage + one more, but not a third.
        first = text_len + 2 + header_len + 2 + len(cp.SEPARATOR)
        return first + (text_len + 2)

    def test_complementary_passage_beats_higher_scored_redundant_one(self):
        cause1 = "the root cause was a database timeout during settlement"
        cause2 = "the root cause was again a database timeout at settlement"
        remediation = "containment the change was rolled back and patched now"
        # Equal length so the budget maths is about which passage, not size.
        L = max(len(cause1), len(cause2), len(remediation))
        cause1 = cause1.ljust(L, ".")
        cause2 = cause2.ljust(L, ".")
        remediation = remediation.ljust(L, ".")
        items = [(chunk("DOC", cause1, 0), 0.90),
                 (chunk("DOC", cause2, 1), 0.80),   # redundant aspect, higher score
                 (chunk("DOC", remediation, 2), 0.70)]  # complementary, lower score
        header = cp._header(cp._group(items)[0], compact=False)
        cfg = replace(CFG,
                      max_context_chars=self._budget_for_two_passages(len(header), L))
        p = plan(aspects=("cause", "containment_actions"))
        packed, _s, _st = cp.build_packed_with_stats(items, cfg, p)
        self.assertIn(remediation, packed)      # complementary kept
        self.assertNotIn(cause2, packed)        # redundant dropped
        # The frozen builder does the opposite (keeps the higher score) - this
        # is exactly the section 21 behaviour Phase 9 adds.
        baseline, _ = context.build(items, cfg)
        self.assertIn(cause2, baseline)
        self.assertNotIn(remediation, baseline)


class MultiPartTest(unittest.TestCase):
    """Spec section 38: both halves of a two-part answer survive when they fit."""

    def test_cause_and_remediation_both_present(self):
        items = [(chunk("INC", "the root cause was a failed migration", 0), 0.9),
                 (chunk("TKT", "containment the change was rolled back", 0), 0.7)]
        packed, _s, _st = cp.build_packed_with_stats(
            items, big_budget(),
            plan(aspects=("cause", "containment_actions"), multi_part=True))
        self.assertIn("root cause", packed)
        self.assertIn("rolled back", packed)


class SameDocumentChunksTest(unittest.TestCase):
    """Spec section 13: multiple passages of one document render under a single
    header, in reading order."""

    def test_same_document_chunks_grouped_under_one_header(self):
        items = [(chunk("DOC", "second passage body", 1), 0.7),
                 (chunk("DOC", "first passage body", 0), 0.9)]
        packed, sources, _st = cp.build_packed_with_stats(items, big_budget(),
                                                          plan())
        self.assertEqual(len(sources), 1)
        self.assertEqual(packed.count("Document:"), 1)
        # Reading order: chunk_index 0 before 1.
        self.assertLess(packed.index("first passage body"),
                        packed.index("second passage body"))


class IdentifierTest(unittest.TestCase):
    """Spec section 39: exact identifier evidence stays present and prominent."""

    def test_identifier_document_survives_and_leads(self):
        items = [(chunk("INC-2026-0519", "incident INC-2026-0519 root cause", 0,
                        doc_type="Incident Report"), 0.85),
                 (chunk("ORG-001", "general reference material", 0,
                        seller="Organization-wide"), 0.80)]
        packed, sources, _st = cp.build_packed_with_stats(
            items, big_budget(),
            plan(query="what happened in INC-2026-0519",
                 identifiers=("INC-2026-0519",)))
        self.assertIn("INC-2026-0519", docs_of(sources))


class WholeChunkTest(unittest.TestCase):
    """Spec section 19/21: passages are kept whole or dropped whole, never cut."""

    def test_placed_passages_appear_verbatim(self):
        text_a = "alpha body that must appear entirely and never be truncated"
        text_b = "beta body that must appear entirely and never be truncated"
        items = [(chunk("A", text_a, 0), 0.9), (chunk("B", text_b, 0), 0.8)]
        packed, _s, _st = cp.build_packed_with_stats(items, big_budget(), plan())
        self.assertIn(text_a, packed)
        self.assertIn(text_b, packed)

    def test_dropped_passage_is_absent_entirely_not_partially(self):
        keep = "keep this whole passage intact in the context"
        drop = "drop this whole passage do not include any fragment of it"
        items = [(chunk("A", keep, 0), 0.9), (chunk("A", drop, 1), 0.5)]
        header = cp._header(cp._group(items)[0], compact=False)
        cfg = replace(CFG,
                      max_context_chars=len(keep) + 2 + len(header) + 2
                      + len(cp.SEPARATOR))
        packed, _s, _st = cp.build_packed_with_stats(items, cfg, plan())
        self.assertIn(keep, packed)
        # No fragment of the dropped passage leaks in.
        for word in drop.split():
            if word not in keep:
                self.assertNotIn(word, packed)


class BudgetTest(unittest.TestCase):
    def test_never_exceeds_budget_meaningfully(self):
        items = [(chunk(f"D{i}", "x" * 500, 0), 0.9 - i * 0.02) for i in range(8)]
        cfg = replace(CFG, max_context_chars=1500)
        packed, _s, _st = cp.build_packed_with_stats(items, cfg, plan())
        # Never far past budget: the only allowed overshoot is the first block,
        # which is always admitted so the context can never be empty.
        self.assertLessEqual(len(packed), 1500 + 700)

    def test_very_small_budget_still_returns_one_block(self):
        items = [(chunk("A", "the only passage", 0), 0.9),
                 (chunk("B", "another passage", 0), 0.8)]
        cfg = replace(CFG, max_context_chars=1)
        packed, sources, _st = cp.build_packed_with_stats(items, cfg, plan())
        self.assertTrue(packed.strip())
        self.assertGreaterEqual(len(sources), 1)

    def test_large_budget_places_everything(self):
        items = [(chunk(f"D{i}", f"passage {i}", 0), 0.9 - i * 0.05)
                 for i in range(6)]
        packed, sources, stats = cp.build_packed_with_stats(items, big_budget(),
                                                            plan())
        self.assertEqual(stats.dropped_chunks, 0)
        self.assertEqual(len(chunk_ids(sources)), len(items))


class MetadataTest(unittest.TestCase):
    """Spec sections 14/15: metadata may be compacted but identity is preserved."""

    def test_compact_header_preserves_identifiers(self):
        items = [(chunk("INC-2026-0519", "body text", 0,
                        doc_type="Incident Report", seller="S002",
                        department="Engineering"), 0.9)]
        packed, _sources = cp.pack(items, big_budget(),
                                   plan(), "metadata_dedup")
        self.assertIn("INC-2026-0519", packed)
        self.assertIn("S002", packed)
        self.assertIn("Engineering", packed)

    def test_grouped_arm_is_identical_to_context_build(self):
        items = [(chunk("A", "alpha", 0), 0.9), (chunk("B", "beta", 0), 0.8)]
        arm, _ = cp.pack(items, big_budget(), plan(), "grouped")
        baseline, _ = context.build(items, big_budget())
        self.assertEqual(arm, baseline)


class DeterminismTest(unittest.TestCase):
    """Spec section 30."""

    def test_repeated_packing_is_identical(self):
        items = [(chunk(f"D{i}", f"settlement text passage {i}", i % 3), 0.9 - i * 0.03)
                 for i in range(8)]
        p = plan(aspects=("cause",), evidence=("support_ticket",))
        runs = [cp.build_packed_with_stats(items, replace(CFG, max_context_chars=900),
                                           p)[0] for _ in range(5)]
        self.assertEqual(len(set(runs)), 1)


class EdgeCaseTest(unittest.TestCase):
    def test_empty_input_returns_empty_context(self):
        packed, sources, stats = cp.build_packed_with_stats([], big_budget(),
                                                            plan())
        self.assertEqual(packed, "")
        self.assertEqual(sources, [])
        self.assertEqual(stats.documents, 0)

    def test_blank_passages_are_skipped(self):
        items = [(chunk("A", "   ", 0), 0.9), (chunk("B", "real body", 0), 0.8)]
        packed, sources, _st = cp.build_packed_with_stats(items, big_budget(),
                                                          plan())
        self.assertEqual(docs_of(sources), ["B"])


if __name__ == "__main__":
    unittest.main()
