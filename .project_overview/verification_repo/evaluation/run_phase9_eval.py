"""Phase 9 evaluation: evidence-aware context packing.

Every arm starts from the SAME Phase 8 selected evidence (reconstructed from the
frozen Jina score cache, exactly as `run_phase8_eval` does), so retrieval, Jina
and selection are byte-identical across arms and the whole sweep costs zero
external calls (spec sections 29/31). The only thing that varies between arms is
how that fixed evidence is packed into the final LLM context.

Arms (spec section 31):
    A baseline        current context.build
    B grouped         group by document (already what build does - control)
    C metadata_dedup  + compress repeated per-document metadata
    D ordering        + query-aware document ordering
    E redundancy      + drop near-duplicate passages under budget pressure
    F final           the shipped Phase 9 packer

Recall / Precision / MRR are measured on the DOCUMENTS that actually reach the
context (i.e. get at least one passage placed), which is what the LLM cites -
not on the pre-context selection. Context size is measured in tiktoken o200k
tokens (the closest public encoding to the Groq gpt-oss model) and in
characters.

Usage:
    python -m evaluation.run_phase9_eval --split holdout
    python -m evaluation.run_phase9_eval --split all --arm baseline
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tiktoken

from access_control.src import audit
from access_control.src.engine import PermissionEngine
from access_control.src.retrieval import MAX_CANDIDATES
from access_control.src.users import get_user
from answering.src import (config, context, evidence_selection,
                           query_aware, query_aware_rerank, selection)
from evaluation import (dataset as ds, metrics, phase8_split,
                        run_stage_eval as rse)
from evaluation.run_phase8_eval import load_cache
from evaluation.run_final_metrics_eval import build_pool, plan_from_record

ENGINE = PermissionEngine()
_ENC = tiktoken.get_encoding("o200k_base")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def n_tokens(text: str) -> int:
    return len(_ENC.encode(text or ""))


def context_redundancy(placed: list) -> float:
    """Mean pairwise Jaccard over the passages actually placed in context.

    Measures how much the context restates itself at passage granularity -
    higher means the reader is shown the same material twice.
    """
    toks = [evidence_selection._tokens(getattr(c, "text", "") or "")
            for c, _ in placed]
    toks = [t for t in toks if t]
    if len(toks) < 2:
        return 0.0
    pairs = [evidence_selection._similarity(toks[i], toks[j])
             for i in range(len(toks)) for j in range(i + 1, len(toks))]
    return sum(pairs) / len(pairs) if pairs else 0.0


def aspect_coverage(plan, placed: list) -> float | None:
    """Passage-level aspect coverage (spec sections 23/41): an aspect counts
    only if some PLACED passage carries a marker for it, not merely if the
    marker appears anywhere across the selected documents."""
    required = set(plan.required_aspects) if plan else set()
    if not required:
        return None
    covered: set[str] = set()
    for chunk, _ in placed:
        covered |= set(query_aware.aspect_hits(plan, getattr(chunk, "text", "")))
    return len(covered & required) / len(required)


def measure(context_text: str, placed: list, sources: list, plan,
            expected: list[str], selected_docs: list[str]) -> dict:
    context_docs = [s.document_id for s in sources]
    scored = metrics.score_retrieval(expected, context_docs)
    dropped_docs = [d for d in selected_docs if d not in set(context_docs)]
    return {
        "recall": scored.recall_at_k,
        "precision": scored.precision_at_k,
        "mrr": scored.mrr,
        "context_docs": context_docs,
        "n_context_docs": len(context_docs),
        "n_placed_chunks": len(placed),
        "chars": len(context_text),
        "tokens": n_tokens(context_text),
        "redundancy": context_redundancy(placed),
        "aspect_coverage": aspect_coverage(plan, placed),
        "dropped_documents": dropped_docs,
    }


def placed_from_sources(selected_pairs: list, sources: list) -> list:
    """The (chunk, score) pairs that survived the budget, by chunk_id."""
    kept_ids = {cid for s in sources for cid in s.chunk_ids}
    return [(c, sc) for c, sc in selected_pairs
            if getattr(c, "chunk_id", None) in kept_ids]


def run_arm(selected_pairs: list, plan, cfg, arm: str, expected: list[str],
            packer) -> dict:
    selected_docs = list(dict.fromkeys(
        getattr(c, "document_id", None) for c, _ in selected_pairs
        if getattr(c, "document_id", None)))
    if arm == "baseline":
        context_text, sources = context.build(selected_pairs, cfg)
    else:
        context_text, sources = packer(selected_pairs, cfg, plan, arm)
    placed = placed_from_sources(selected_pairs, sources)
    return measure(context_text, placed, sources, plan, expected, selected_docs)


def reconstruct(case, dataset, cache, saved, cfg):
    """The exact Phase 8 selected evidence for one case, or None to skip."""
    record = saved.get(case.eval_id)
    if record is None or case.eval_id not in cache:
        return None
    plan = plan_from_record(record, case.question)
    user = get_user(case.user_id)
    scope = ENGINE.get_authorized_scope(user, None)
    text = rse.search_text_for(case, dataset)
    top_k = min(max(cfg.candidate_pool * 4, 10), MAX_CANDIDATES)
    fused = rse.phase3_retrieve(text, top_k=top_k, filters=scope)
    pool = build_pool(fused, plan, user, f"phase9-{case.eval_id}", rework=True)
    raw = cache[case.eval_id]["plain"]
    surviving = [(c, raw[c.chunk_id]) for c in pool
                 if c.chunk_id in raw and raw[c.chunk_id] >= cfg.rerank_threshold]
    items = [(c, query_aware_rerank.adjust(s, c, plan)) for c, s in surviving]
    items.sort(key=lambda pair: pair[1], reverse=True)
    if not items:
        return None
    # Shipped Phase 8 selection (plan supplied, default weights).
    selected_pairs, _stats = selection.select(items, cfg, plan=plan)
    return plan, selected_pairs


def run(case_ids: set[str], arms: list[str], packer) -> list[dict]:
    dataset = ds.load()
    cache = load_cache()
    saved = rse.load_saved_plans(
        ds.PACKAGE_ROOT / "runs" / "query-understanding-eval.json")
    cfg = config.DEFAULT_ANSWERING
    rows: list[dict] = []
    for case in dataset.cases:
        if case.eval_id not in case_ids or case.expected_behavior == "refuse":
            continue
        built = reconstruct(case, dataset, cache, saved, cfg)
        if built is None:
            continue
        plan, selected_pairs = built
        expected = list(dict.fromkeys(case.expected_sources))
        row: dict = {"eval_id": case.eval_id, "category": case.category,
                     "expected": expected,
                     "plan": {"intent": plan.intent,
                              "required_aspects": list(plan.required_aspects),
                              "evidence_types": list(plan.evidence_types),
                              "identifiers": list(plan.identifiers),
                              "is_multi_part": plan.is_multi_part}}
        for arm in arms:
            row[arm] = run_arm(selected_pairs, plan, cfg, arm, expected, packer)
        rows.append(row)
        a = row[arms[0]]
        last = row[arms[-1]]
        print(f"{case.eval_id}: docs {a['n_context_docs']} "
              f"tokens {a['tokens']}->{last['tokens']} "
              f"chunks {a['n_placed_chunks']}->{last['n_placed_chunks']} "
              f"redund {a['redundancy']:.3f}->{last['redundancy']:.3f}",
              flush=True)
    return rows


def aggregate(rows: list[dict], arms: list[str]) -> dict:
    def mean(key, arm):
        values = [r[arm][key] for r in rows if r[arm].get(key) is not None]
        return round(statistics.mean(values), 4) if values else None

    summary: dict = {"scored_cases": len(rows)}
    for arm in arms:
        summary[arm] = {
            "recall": mean("recall", arm),
            "precision": mean("precision", arm),
            "mrr": mean("mrr", arm),
            "mean_tokens": mean("tokens", arm),
            "mean_chars": mean("chars", arm),
            "mean_context_docs": mean("n_context_docs", arm),
            "mean_placed_chunks": mean("n_placed_chunks", arm),
            "redundancy": mean("redundancy", arm),
            "aspect_coverage": mean("aspect_coverage", arm),
        }
    # Gold-preservation and determinism checks against the baseline arm.
    base = arms[0]
    dropped = []
    for r in rows:
        for arm in arms:
            gold_dropped = [d for d in r[arm]["dropped_documents"]
                            if d in set(r["expected"])]
            if gold_dropped:
                dropped.append({"eval_id": r["eval_id"], "arm": arm,
                                "gold_dropped": gold_dropped})
    summary["gold_documents_dropped"] = dropped
    return summary


def _packer(arm_module):
    def packer(pairs, cfg, plan, arm):
        return arm_module.pack(pairs, cfg, plan, arm)
    return packer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout", "all"),
                        default="all")
    parser.add_argument("--arms", default="baseline",
                        help="comma-separated arm list")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config.validate_env()
    audit.AUDIT_FILE = ds.PACKAGE_ROOT / "runs" / "phase9-audit.jsonl"
    dev, holdout = phase8_split.split()
    case_ids = {"dev": set(dev), "holdout": set(holdout),
                "all": set(dev) | set(holdout)}[args.split]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    # Only the packer arms need the module; baseline-only measurement does not,
    # so it can run before the packer exists.
    if any(arm != "baseline" for arm in arms):
        from answering.src import context_packing
        packer = _packer(context_packing)
    else:
        packer = None

    rows = run(case_ids, arms, packer)
    output = args.output or (ds.PACKAGE_ROOT / "runs"
                             / f"phase9-eval-{args.split}.json")
    document = {
        "schema_version": 1,
        "generated_at": _now(),
        "split": args.split,
        "arms": arms,
        "tokenizer": "tiktoken/o200k_base (approx of Groq gpt-oss)",
        "summary": aggregate(rows, arms),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(json.dumps({arm: document["summary"][arm] for arm in arms}, indent=2))
    print(f"Results: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
