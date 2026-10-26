"""Phase 8 evaluation: evidence-aware selection, ablated on a held-out split.

Every arm reuses ONE cached Jina ranking per case, so the candidate pool and
the Phase 7 scores are byte-identical across arms (spec section 28) and the
whole sweep costs zero API calls (spec section 41). Phase 7 is frozen: this
harness never varies a Phase 7 setting.

Arms (spec section 28):
    A phase7          existing score-ordered selector (frozen baseline)
    B aspect          + required-aspect coverage
    C evidence_type   + evidence-type coverage
    D redundancy      + redundancy penalty
    E diversity       + redundancy penalty and a stopping rule
    F entity          + entity/event coverage
    G combined        all signals

Splits (spec section 44): the configuration is chosen on `dev` and reported on
`holdout`, which played no part in choosing it.

Usage:
    python -m evaluation.run_phase8_eval --split dev
    python -m evaluation.run_phase8_eval --split holdout
    python -m evaluation.run_phase8_eval --split dev --stop 0.55
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_control.src import audit
from access_control.src.engine import PermissionEngine
from access_control.src.retrieval import MAX_CANDIDATES
from access_control.src.users import get_user
from answering.src import (config, evidence_selection, query_aware,
                           query_aware_rerank, reranker, selection)
from answering.src.evidence_selection import EvidenceWeights
from evaluation import (dataset as ds, metrics, phase8_split,
                        run_stage_eval as rse)
from evaluation.run_final_metrics_eval import build_pool, plan_from_record

ENGINE = PermissionEngine()
CACHE = ds.PACKAGE_ROOT / "runs" / "phase7-jina-scores.json"

ARMS: dict[str, tuple[str, ...]] = {
    "phase7": (),
    "aspect": ("aspect",),
    "evidence_type": ("evidence_type",),
    "redundancy": ("redundancy",),
    "diversity": ("redundancy",),
    "entity": ("entity",),
    "combined": ("aspect", "evidence_type", "entity", "redundancy"),
}
# `diversity` is `redundancy` plus an active stopping rule: the penalty decides
# what is redundant, the stopping rule decides whether redundant material still
# earns a slot. Isolating them shows which half does the work.
STOPPING_ARMS = {"diversity", "combined"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_cache() -> dict:
    if not CACHE.is_file():
        raise SystemExit(
            f"missing Jina score cache: {CACHE}\n"
            "Run `python -m evaluation.run_phase7_eval` first.")
    return json.loads(CACHE.read_text(encoding="utf-8"))


def aspect_coverage(plan, documents: list[str], by_document: dict) -> float | None:
    """Fraction of the plan's required aspects present in the selected text."""
    required = set(plan.required_aspects) if plan else set()
    if not required:
        return None
    covered: set[str] = set()
    for document_id in documents:
        covered |= set(query_aware.aspect_hits(
            plan, by_document.get(document_id, "")))
    return len(covered & required) / len(required)


def redundancy_rate(documents: list[str], by_document: dict) -> float:
    texts = [evidence_selection._tokens(by_document.get(d, ""))
             for d in documents]
    if len(texts) < 2:
        return 0.0
    pairs = [evidence_selection._similarity(texts[i], texts[j])
             for i in range(len(texts)) for j in range(i + 1, len(texts))]
    return sum(pairs) / len(pairs) if pairs else 0.0


def score_arm(items, plan, cfg, arm: str, base: EvidenceWeights,
              expected: list[str], by_document: dict) -> dict:
    """One selection arm over an identical Phase 7 ranking."""
    if arm == "phase7":
        chosen, stats = selection.select(items, cfg, plan=None)
    else:
        weights = base.only(*ARMS[arm])
        if arm not in STOPPING_ARMS:
            weights = replace(weights, min_marginal_utility=0.0)
        chosen, stats = selection.select(items, cfg, plan=plan,
                                         weights=weights)
    documents = list(dict.fromkeys(
        getattr(chunk, "document_id", None) for chunk, _ in chosen
        if getattr(chunk, "document_id", None)))
    scored = metrics.score_retrieval(expected, documents)
    return {
        "recall": scored.recall_at_k,
        "precision": scored.precision_at_k,
        "mrr": scored.mrr,
        "precision_efficiency": scored.precision_efficiency,
        "documents": documents,
        "n_documents": len(documents),
        "n_chunks": len(chosen),
        "aspect_coverage": aspect_coverage(plan, documents, by_document),
        "redundancy": redundancy_rate(documents, by_document),
        "evidence": stats.evidence,
    }


def run(case_ids: set[str], stop: float | None) -> list[dict]:
    dataset = ds.load()
    cache = load_cache()
    saved = rse.load_saved_plans(
        ds.PACKAGE_ROOT / "runs" / "query-understanding-eval.json")
    cfg = config.DEFAULT_ANSWERING
    base = EvidenceWeights.from_config()
    if stop is not None:
        base = replace(base, min_marginal_utility=stop)

    rows: list[dict] = []
    for case in dataset.cases:
        if case.eval_id not in case_ids or case.expected_behavior == "refuse":
            continue
        record = saved.get(case.eval_id)
        if record is None or case.eval_id not in cache:
            continue
        plan = plan_from_record(record, case.question)
        user = get_user(case.user_id)
        scope = ENGINE.get_authorized_scope(user, None)
        text = rse.search_text_for(case, dataset)
        top_k = min(max(cfg.candidate_pool * 4, 10), MAX_CANDIDATES)
        fused = rse.phase3_retrieve(text, top_k=top_k, filters=scope)
        pool = build_pool(fused, plan, user, f"phase8-{case.eval_id}",
                          rework=True)

        # Rebuild the exact Phase 7 ranking from cache: raw gate, then the
        # frozen plan-aware adjustment.
        raw = cache[case.eval_id]["plain"]
        surviving = [(c, raw[c.chunk_id]) for c in pool
                     if c.chunk_id in raw
                     and raw[c.chunk_id] >= cfg.rerank_threshold]
        items = [(c, query_aware_rerank.adjust(s, c, plan))
                 for c, s in surviving]
        items.sort(key=lambda pair: pair[1], reverse=True)
        if not items:
            continue
        by_document: dict[str, str] = {}
        for chunk, _score in items:
            by_document.setdefault(chunk.document_id, "")
            by_document[chunk.document_id] += " " + (chunk.text or "")

        expected = list(dict.fromkeys(case.expected_sources))
        row: dict = {
            "eval_id": case.eval_id, "category": case.category,
            "expected": expected,
            "plan": {"intent": plan.intent,
                     "required_aspects": list(plan.required_aspects),
                     "evidence_types": list(plan.evidence_types),
                     "identifiers": list(plan.identifiers),
                     "is_multi_part": plan.is_multi_part},
        }
        for arm in ARMS:
            row[arm] = score_arm(items, plan, cfg, arm, base, expected,
                                 by_document)
        rows.append(row)
        b, g = row["phase7"], row["combined"]
        print(f"{case.eval_id}: docs {b['n_documents']}->{g['n_documents']} "
              f"R {b['recall']:.2f}->{g['recall']:.2f} "
              f"P {b['precision']:.3f}->{g['precision']:.3f} "
              f"M {b['mrr']:.3f}->{g['mrr']:.3f}", flush=True)
    return rows


def aggregate(rows: list[dict]) -> dict:
    def mean(key, arm):
        values = [r[arm][key] for r in rows if r[arm].get(key) is not None]
        return round(statistics.mean(values), 4) if values else None

    summary: dict = {"scored_cases": len(rows)}
    for arm in ARMS:
        summary[arm] = {
            "recall": mean("recall", arm),
            "precision": mean("precision", arm),
            "mrr": mean("mrr", arm),
            "precision_efficiency": mean("precision_efficiency", arm),
            "mean_documents": mean("n_documents", arm),
            "mean_chunks": mean("n_chunks", arm),
            "aspect_coverage": mean("aspect_coverage", arm),
            "redundancy": mean("redundancy", arm),
        }
    comparisons = {}
    for arm in ARMS:
        if arm == "phase7":
            continue
        improved, regressed, unchanged = [], [], []
        for r in rows:
            base, cur = r["phase7"], r[arm]
            dr = (cur["recall"] or 0) - (base["recall"] or 0)
            dp = (cur["precision"] or 0) - (base["precision"] or 0)
            dm = (cur["mrr"] or 0) - (base["mrr"] or 0)
            entry = {"eval_id": r["eval_id"], "category": r["category"],
                     "recall": [base["recall"], cur["recall"]],
                     "precision": [round(base["precision"] or 0, 4),
                                   round(cur["precision"] or 0, 4)],
                     "mrr": [round(base["mrr"] or 0, 4),
                             round(cur["mrr"] or 0, 4)],
                     "documents": [base["documents"], cur["documents"]],
                     "plan": r["plan"]}
            if dr < -1e-9:
                regressed.append({**entry, "reason": "recall"})
            elif dp > 1e-9 or dm > 1e-9:
                improved.append(entry)
            elif dp < -1e-9 or dm < -1e-9:
                regressed.append({**entry, "reason": "precision/mrr"})
            else:
                unchanged.append(r["eval_id"])
        comparisons[arm] = {"improved": improved, "regressed": regressed,
                            "unchanged_count": len(unchanged)}
    summary["per_query"] = comparisons
    summary["recall_regressions"] = {
        arm: [c["eval_id"] for c in comparisons[arm]["regressed"]
              if c.get("reason") == "recall"] for arm in comparisons}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "holdout", "all"),
                        default="dev")
    parser.add_argument("--stop", type=float,
                        help="override EVIDENCE_MIN_MARGINAL_UTILITY")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    config.validate_env()
    audit.AUDIT_FILE = ds.PACKAGE_ROOT / "runs" / "phase8-audit.jsonl"
    dev, holdout = phase8_split.split()
    case_ids = {"dev": set(dev), "holdout": set(holdout),
                "all": set(dev) | set(holdout)}[args.split]

    rows = run(case_ids, args.stop)
    output = args.output or (ds.PACKAGE_ROOT / "runs"
                             / f"phase8-eval-{args.split}.json")
    document = {
        "schema_version": 1,
        "generated_at": _now(),
        "split": args.split,
        "split_case_ids": sorted(case_ids),
        "min_marginal_utility": (args.stop if args.stop is not None
                                 else config.EVIDENCE_MIN_MARGINAL_UTILITY),
        "weights": {"aspect": config.EVIDENCE_ASPECT_WEIGHT,
                    "evidence_type": config.EVIDENCE_TYPE_WEIGHT,
                    "entity": config.EVIDENCE_ENTITY_WEIGHT,
                    "redundancy": config.EVIDENCE_REDUNDANCY_PENALTY},
        "summary": aggregate(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(json.dumps({arm: document["summary"][arm] for arm in ARMS}, indent=2))
    print(f"Results: {output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
