"""Final-metrics evaluation for the Phase 6 rework (offline of generation).

Runs the golden scored cases through the two candidate-generation arms and
then the REAL reranker (Jina, live) and the REAL selection stage, so Final
Recall / Precision / MRR (document-level, after selection) are measured for:

    baseline  dense+BM25 -> RRF -> authorization -> pool cut
    plan      + identifier augmentation + identifier-first reorder
              + bounded evidence promotion (aspect/date)

LLM answering, the judge, and JEV are NOT run: this measures retrieval +
rerank + selection, which are exactly the stages Phase 6 is allowed to touch.
Jina is live, so this harness consumes rerank quota and paces itself.

Usage:
    python -m evaluation.run_final_metrics_eval
    python -m evaluation.run_final_metrics_eval --limit 5
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_control.src import audit
from access_control.src.engine import PermissionEngine
from access_control.src.retrieval import MAX_CANDIDATES
from access_control.src.users import get_user
from answering.src import config, query_aware, reranker, selection
from answering.src.query_planner import QueryPlan
from evaluation import dataset as ds, metrics, run_stage_eval as rse

ENGINE = PermissionEngine()
INTER_CASE_DELAY_S = 1.0
POOL_OVERRIDE = [None]


def load_saved_plans(path: Path) -> dict[str, dict]:
    document = json.loads(path.read_text(encoding="utf-8"))
    return {record["eval_id"]: record for record in document["records"]}


def plan_from_record(record: dict, question: str) -> QueryPlan:
    return QueryPlan(
        original_query=question,
        intent=record.get("intent", "factual_lookup"),
        entities=tuple(record.get("entities") or ()),
        seller_scope=tuple(record.get("seller_scope") or ()),
        identifiers=tuple(record.get("identifiers") or ()),
        required_aspects=tuple(record.get("required_aspects") or ()),
        evidence_types=tuple(record.get("evidence_types") or ()),
        is_multi_part=bool(record.get("is_multi_part")),
        complexity=record.get("complexity", "simple"),
        confidence=record.get("confidence"),
        fallback_used=False,
        jev_status="ok",
    )


def build_pool(fused, plan_obj, user, request_id: str, rework: bool,
               pool_size: int | None = None) -> list:
    pool_size = pool_size or config.DEFAULT_ANSWERING.candidate_pool
    if not rework:
        allowed, _ = ENGINE.authorize_chunks(user, list(fused), request_id)
        return allowed[:pool_size]
    candidates, _added = query_aware.augment_fused(fused, plan_obj)
    allowed, _denied = ENGINE.authorize_chunks(user, candidates, request_id)
    allowed, _signals = query_aware.boosted_order(allowed, plan_obj)
    allowed, _promoted = query_aware.promote_evidence(
        allowed, plan_obj, pool_size=pool_size)
    return allowed[:pool_size]


def score_arm(text: str, pool, request_id: str, expected: list[str]) -> dict:
    reranked = reranker.rerank(text, pool)
    cfg = config.DEFAULT_ANSWERING
    selected, _stats = selection.select(
        reranked.items, cfg, outcome=reranked)
    documents = list(dict.fromkeys(
        getattr(chunk, "document_id", None) for chunk, _ in selected
        if getattr(chunk, "document_id", None)))
    scored = metrics.score_retrieval(expected, documents)
    return {"recall": scored.recall_at_k, "precision": scored.precision_at_k,
            "mrr": scored.mrr, "documents": documents,
            "n_selected": len(selected)}


def search_text(case: dict) -> str:
    dataset = case["_dataset"]
    return rse.search_text_for(case["_case"], dataset)


def run(output: Path, limit: int | None = None, plans_path: Path | None = None):
    dataset = ds.load()
    saved_plans = load_saved_plans(plans_path) if plans_path else {}
    cases = dataset.cases if limit is None else dataset.cases[:limit]
    rows = []
    for index, raw_case in enumerate(cases):
        if raw_case.expected_behavior == "refuse":
            continue
        case = {"_case": raw_case, "_dataset": dataset}
        user = get_user(raw_case.user_id)
        scope = ENGINE.get_authorized_scope(user, None)
        text = search_text(case)
        top_k = min(max(config.DEFAULT_ANSWERING.candidate_pool * 4, 10),
                    MAX_CANDIDATES)
        fused = rse.phase3_retrieve(text, top_k=top_k, filters=scope)
        saved = saved_plans.get(raw_case.eval_id)
        plan_obj = plan_from_record(saved, raw_case.question) if saved else None
        if plan_obj is None:
            print(f"{raw_case.eval_id}: skipped (no saved plan)", flush=True)
            continue
        request_id = f"final-{raw_case.eval_id.lower()}"
        row: dict = {"eval_id": raw_case.eval_id,
                     "category": raw_case.category}
        for arm, rework in (("baseline", False), ("plan", True)):
            pool = build_pool(fused, plan_obj, user, request_id, rework,
                              pool_size=POOL_OVERRIDE[0])
            row[arm] = score_arm(text, pool, request_id,
                                 list(dict.fromkeys(raw_case.expected_sources)))
            time.sleep(INTER_CASE_DELAY_S)
        rows.append(row)
        b, p = row["baseline"], row["plan"]
        print(f"{raw_case.eval_id}: final recall {b['recall']} -> "
              f"{p['recall']} | precision {b['precision']} -> "
              f"{p['precision']} | mrr {b['mrr']} -> {p['mrr']}", flush=True)
    return rows


def aggregate(rows: list[dict]) -> dict:
    def mean(key, arm):
        values = [r[arm][key] for r in rows if r[arm][key] is not None]
        return round(statistics.mean(values), 4) if values else None

    summary: dict = {"scored_cases": len(rows)}
    for arm in ("baseline", "plan"):
        summary[arm] = {"recall": mean("recall", arm),
                        "precision": mean("precision", arm),
                        "mrr": mean("mrr", arm),
                        "mean_selected_chunks": mean("n_selected", arm)}
    summary["recall_regressions"] = [
        {"eval_id": r["eval_id"], "baseline": r["baseline"]["recall"],
         "plan": r["plan"]["recall"]}
        for r in rows
        if r["baseline"]["recall"] is not None
        and r["plan"]["recall"] is not None
        and r["plan"]["recall"] < r["baseline"]["recall"]]
    summary["recall_improvements"] = sum(
        1 for r in rows
        if r["baseline"]["recall"] is not None
        and r["plan"]["recall"] is not None
        and r["plan"]["recall"] > r["baseline"]["recall"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ds.PACKAGE_ROOT / "runs"
                        / "final-metrics-eval.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plans", type=Path,
                        default=ds.PACKAGE_ROOT / "runs"
                        / "query-understanding-eval.json")
    parser.add_argument("--pool", type=int,
                        help="override the candidate pool cut for this run")
    args = parser.parse_args()
    if args.pool:
        POOL_OVERRIDE[0] = args.pool
    config.validate_env()
    audit.AUDIT_FILE = ds.PACKAGE_ROOT / "runs" / "final-metrics-audit.jsonl"
    rows = run(args.output, args.limit, args.plans)
    document = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_pool": POOL_OVERRIDE[0]
        or config.DEFAULT_ANSWERING.candidate_pool,
        "summary": aggregate(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(document["summary"], indent=2))
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
