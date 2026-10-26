"""Phase 6 stage-level evaluation: baseline vs query-aware candidate pools.

Measures the retrieval stage in isolation, exactly as production composes it:

    scope -> dense+BM25 -> RRF (phase3_retrieve)
          -> [arm: exact-identifier augmentation]
          -> authorization re-check (authorize_chunks)
          -> [arm: QueryPlan boost reorder]
          -> top candidate_pool chunks -> documents entering Jina

Jina reranking, selection, context and generation are NOT run, so no final
answer metric is claimed here. The question answered is whether the correct
documents enter the reranker candidate pool more reliably.

Arms:
    baseline   RRF candidates only (pre-Phase-6 behaviour)
    plan       QueryPlan from the saved Phase 5 evaluation run (JEV signals)
    fallback   deterministic-only plan (identifiers/entities/sellers), the
               safe degradation path with JEV unavailable

Usage:
    python -m evaluation.run_stage_eval --output evaluation/runs/stage-eval.json
    python -m evaluation.run_stage_eval --limit 5
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
from access_control.src.retrieval import MAX_CANDIDATES, phase3_retrieve
from access_control.src.users import get_user
from answering.src import config, query_aware
from answering.src.query_planner import QueryPlan
from evaluation import dataset as ds, metrics

ENGINE = PermissionEngine()
INTER_CASE_DELAY_S = 0.3
EVAL_POOL = config.DEFAULT_ANSWERING.candidate_pool

ABLATION_ARMS = {
    "aspect_only": ("aspect",),
    "date_only": ("date",),
    "aspect_date": ("aspect", "date"),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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


def fallback_plan(question: str) -> QueryPlan:
    """Deterministic-only plan: what Phase 6 still gets when JEV is down."""
    identifiers = query_aware_qp().extract_identifiers(question)
    sellers = query_aware_qp().extract_sellers(question)
    entities = query_aware_qp().extract_entities(question)
    lowered = question.lower()
    if "organization" in lowered or "organisation" in lowered or "company" in lowered:
        sellers = tuple(dict.fromkeys([*sellers, "ORG"]))
    return QueryPlan(
        original_query=question,
        intent="factual_lookup",
        entities=entities,
        seller_scope=sellers,
        identifiers=identifiers,
    )


def query_aware_qp():
    from answering.src import query_planner
    return query_planner


def search_text_for(case, dataset: ds.GoldenDataset) -> str:
    """Mirror service/conversation retrieval-query expansion for follow-ups."""
    parts = []
    parent_id = case.conversation_from
    while parent_id:
        parent = dataset.by_id(parent_id)
        if parent is None:
            break
        parts.insert(0, parent.question)
        parent_id = parent.conversation_from
    parts = parts[-2:]
    parts.append(case.question)
    return " ".join(parts)


def run_arm(fused, plan_obj, user, request_id, augment: bool = False,
            boost: bool = False, promote_criteria: tuple | None = None):
    """One candidate-generation arm through the production authorization path."""
    if augment:
        candidates, _ = query_aware.augment_fused(fused, plan_obj)
    else:
        candidates = list(fused)
    allowed, denied = ENGINE.authorize_chunks(user, candidates, request_id)
    promoted: list[str] = []
    if boost:
        allowed, _signals = query_aware.boosted_order(allowed, plan_obj)
    if promote_criteria is not None and plan_obj is not None:
        saved_criteria = config.QUERY_AWARE_PROMOTE_CRITERIA
        config.QUERY_AWARE_PROMOTE_CRITERIA = promote_criteria
        try:
            allowed, promoted = query_aware.promote_evidence(
                allowed, plan_obj, pool_size=EVAL_POOL)
        finally:
            config.QUERY_AWARE_PROMOTE_CRITERIA = saved_criteria
    pool = allowed[:EVAL_POOL]
    documents = list(dict.fromkeys(
        chunk.document_id for chunk in pool if chunk.document_id))
    methods = {}
    for chunk in pool:
        methods[chunk.retrieval_method] = methods.get(chunk.retrieval_method, 0) + 1
    return {
        "documents": documents,
        "pool_size": len(pool),
        "authorized": len(allowed),
        "withheld": len(denied),
        "methods": methods,
        "promoted_chunks": promoted,
    }


def evaluate_case(case, dataset, fused, user, request_id, saved_plan) -> dict:
    expected = list(dict.fromkeys(case.expected_sources))
    row: dict = {
        "eval_id": case.eval_id,
        "category": case.category,
        "expected_behavior": case.expected_behavior,
    }
    if case.expected_behavior == "refuse":
        row["applicable"] = False
        return row
    row["applicable"] = True
    row["expected"] = expected

    arms = {
        "baseline": run_arm(fused, None, user, request_id),
    }
    plan_obj = None
    if saved_plan is not None:
        plan_obj = plan_from_record(saved_plan, case.question)
        arms["plan"] = run_arm(fused, plan_obj, user, request_id,
                               augment=True, boost=True,
                               promote_criteria=("aspect", "date"))
    det = fallback_plan(case.question)
    # Always run the fallback arm so aggregate means cover the same rows as
    # the other arms: skipping signal-free cases would bias the comparison.
    arms["fallback"] = run_arm(fused, det, user, request_id,
                               augment=True, boost=True,
                               promote_criteria=("aspect", "date"))
    if saved_plan is not None:
        # Ablation arms isolate ONE variable: the promotion criteria. They
        # therefore share identical augment/boost settings (both off). The
        # previous code augmented for `aspect_date` but not for the single
        # criteria arms, so an arm difference could in principle have come from
        # augmentation rather than from the criterion under test. On this
        # dataset augmentation is provably a no-op (no scored query contains a
        # structured identifier - every arm reports identifier_pool_chunks=0),
        # so the recorded numbers are unaffected; the inconsistency is fixed
        # here because Phase 7 reuses these arms, where it would not be a no-op.
        for name, criteria in ABLATION_ARMS.items():
            arms[name] = run_arm(fused, plan_obj, user, request_id,
                                 augment=False,
                                 boost=False,
                                 promote_criteria=criteria)

    for name, result in arms.items():
        scored = metrics.score_retrieval(expected, result["documents"])
        row[name] = {
            **result,
            "recall": scored.recall_at_k,
            "precision": scored.precision_at_k,
            "mrr": scored.mrr,
        }
    return row


def aggregate(rows: list[dict]) -> dict:
    applicable = [r for r in rows if r.get("applicable")]
    summary: dict = {"scored_cases": len(applicable)}

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 4) if values else None

    for arm in ("baseline", "plan", "fallback", "aspect_only", "date_only",
                "aspect_date"):
        rows_with_arm = [r for r in applicable if arm in r]
        if not rows_with_arm:
            continue
        summary[arm] = {
            "recall": mean([r[arm]["recall"] for r in rows_with_arm]),
            "precision": mean([r[arm]["precision"] for r in rows_with_arm]),
            "mrr": mean([r[arm]["mrr"] for r in rows_with_arm]),
            "mean_pool_documents": mean(
                [len(r[arm]["documents"]) for r in rows_with_arm]),
            "mean_pool_chunks": mean([r[arm]["pool_size"] for r in rows_with_arm]),
            "mean_withheld": mean([r[arm]["withheld"] for r in rows_with_arm]),
            "identifier_pool_chunks": sum(
                r[arm]["methods"].get("identifier", 0) for r in rows_with_arm),
        }
    if "plan" in summary and "baseline" in summary:
        deltas = []
        for r in applicable:
            if "plan" not in r:
                continue
            base, plan = r["baseline"]["recall"], r["plan"]["recall"]
            if base is not None and plan is not None and plan < base:
                deltas.append((r["eval_id"], base, plan,
                               r["baseline"]["documents"], r["plan"]["documents"]))
        summary["plan_recall_regressions"] = deltas
        improved = sum(
            1 for r in applicable if "plan" in r
            and (r["plan"]["recall"] or 0) > (r["baseline"]["recall"] or 0)
            and r["baseline"]["recall"] is not None)
        summary["plan_recall_improvements"] = improved

    ablations = {}
    for name in ABLATION_ARMS:
        if name not in summary:
            continue
        ablations[name] = summary[name]
    summary["ablation"] = ablations
    if ablations and "baseline" in summary:
        summary["ablation_recall_regressions"] = {
            name: [
                (r["eval_id"],
                 r["baseline"]["recall"], r[name]["recall"])
                for r in applicable if name in r
                and r["baseline"]["recall"] is not None
                and r[name]["recall"] is not None
                and r[name]["recall"] < r["baseline"]["recall"]
            ]
            for name in ablations
        }
    return summary


def run(output: Path | None = None, limit: int | None = None,
        plans_path: Path | None = None) -> dict:
    dataset = ds.load()
    catalog = ds.PermissionCatalog()
    saved_plans = load_saved_plans(plans_path) if plans_path and plans_path.is_file() else {}
    cases = dataset.cases if limit is None else dataset.cases[:limit]
    rows = []
    for index, case in enumerate(cases):
        user = get_user(case.user_id)
        scope = ENGINE.get_authorized_scope(user, None)
        text = search_text_for(case, dataset)
        top_k = min(max(EVAL_POOL * 4, 10), MAX_CANDIDATES)
        fused = phase3_retrieve(text, top_k=top_k, filters=scope)
        row = evaluate_case(case, dataset, fused, user,
                            f"stageeval-{case.eval_id.lower()}",
                            saved_plans.get(case.eval_id))
        rows.append(row)
        if row.get("applicable"):
            base = row["baseline"]
            plan_entry = row.get("plan")
            print(f"{case.eval_id} {case.category}: pool_docs="
                  f"{len(base['documents'])} recall={base['recall']}"
                  + (f" -> plan recall={plan_entry['recall']}"
                     if plan_entry else ""), flush=True)
        else:
            print(f"{case.eval_id} {case.category}: not applicable", flush=True)
        if index < len(cases) - 1:
            time.sleep(INTER_CASE_DELAY_S)
    document = {
        "schema_version": 1,
        "generated_at": _now(),
        "candidate_pool": EVAL_POOL,
        "phase5_plans_source": (str(plans_path) if saved_plans
                                else "none (deterministic fallback arms only)"),
        "summary": aggregate(rows),
        "rows": rows,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ds.PACKAGE_ROOT / "runs" / "stage-eval.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plans", type=Path,
                        default=ds.PACKAGE_ROOT / "runs"
                        / "query-understanding-eval.json")
    args = parser.parse_args()
    config.validate_env()
    audit.AUDIT_FILE = (ds.PACKAGE_ROOT / "runs" / "stage-eval-audit.jsonl")
    document = run(args.output, args.limit, args.plans)
    print(json.dumps(document["summary"], indent=2))
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
