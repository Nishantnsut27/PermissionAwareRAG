"""Phase 7 evaluation: query-aware Jina reranking, with ablation.

Measures Final Recall / Precision / MRR (document level, after the REAL
selection stage) for the Phase 6 candidate pool reranked in several ways.

Cost design
-----------
The plan-compatibility term is applied to Jina's scores AFTER Jina returns
them, so every score-side arm can reuse one set of scores. Jina is therefore
called at most twice per case - once with the plain query, once with the
plan-hinted query - and all seven ablation arms are then swept offline from the
cache. Re-running with the cache present costs zero API calls.

Arms (spec section 21):
    A baseline        Phase 6 ordering, untouched Jina scores
    B +intent         intent -> record-genre affinity only
    C +aspects        required-aspect coverage only
    D +event          structured identifier / explicit month only
    E +evidence       predicted evidence-type fit only
    F +seller         mentioned-seller fit only
    G combined        all five, sharing the same total bound
    H combined+hints  G, plus the plan-hinted Jina query

Every arm shares one candidate pool and one relevance threshold, so a
difference between arms can only come from the signal under test.

Usage:
    python -m evaluation.run_phase7_eval --limit 3      # smoke
    python -m evaluation.run_phase7_eval                # full
    python -m evaluation.run_phase7_eval --weight 0.10  # bound sweep
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
from answering.src import config, query_aware_rerank, reranker, selection
from answering.src.query_aware_rerank import CompatibilityWeights
from evaluation import dataset as ds, metrics, run_stage_eval as rse
from evaluation.run_final_metrics_eval import build_pool, plan_from_record

ENGINE = PermissionEngine()
INTER_CALL_DELAY_S = 1.0

PLAIN = "plain"
HINTED = "hinted"

# name -> (signals kept, which Jina query variant the arm scores against)
ARMS: dict[str, tuple[tuple[str, ...], str]] = {
    "baseline": ((), PLAIN),
    "intent": (("intent",), PLAIN),
    "aspects": (("aspect",), PLAIN),
    "event": (("event",), PLAIN),
    "evidence": (("evidence",), PLAIN),
    "seller": (("seller",), PLAIN),
    "combined": (("aspect", "event", "intent", "evidence", "seller"), PLAIN),
    "combined_hints": (("aspect", "event", "intent", "evidence", "seller"),
                       HINTED),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ScoreCache:
    """Jina scores keyed by case and query variant, persisted between runs."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = {}
        if path.is_file():
            self.data = json.loads(path.read_text(encoding="utf-8"))
        self.calls = 0
        self.latencies: list[float] = []
        self.query_chars: dict[str, int] = {}

    def get(self, eval_id: str, variant: str) -> dict[str, float] | None:
        return (self.data.get(eval_id) or {}).get(variant)

    def put(self, eval_id: str, variant: str, scores: dict[str, float]) -> None:
        self.data.setdefault(eval_id, {})[variant] = scores

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1), encoding="utf-8")


def jina_scores(cache: ScoreCache, eval_id: str, variant: str, query: str,
                pool: list) -> dict[str, float]:
    """Raw Jina relevance per chunk id, from cache when available.

    No threshold and no plan term here: this is the unmodified provider signal
    that every arm then interprets differently.
    """
    cached = cache.get(eval_id, variant)
    if cached is not None:
        return cached
    started = time.monotonic()
    outcome = reranker.rerank(query, pool, threshold=None, plan=None)
    cache.latencies.append(time.monotonic() - started)
    cache.calls += 1
    cache.query_chars[variant] = max(cache.query_chars.get(variant, 0),
                                     len(query))
    if not outcome.applied:
        raise RuntimeError(f"{eval_id}/{variant}: rerank unavailable "
                           f"({outcome.error}) - refusing to record a "
                           f"degraded run as a measurement")
    scores = {chunk.chunk_id: float(score) for chunk, score in outcome.items}
    cache.put(eval_id, variant, scores)
    time.sleep(INTER_CALL_DELAY_S)
    return scores


def score_arm(pool: list, raw: dict[str, float], plan_obj,
              signals: tuple[str, ...], weights: CompatibilityWeights,
              expected: list[str]) -> dict:
    """Reproduce the production rerank+selection path for one arm.

    Mirrors `reranker.rerank` exactly: the relevance threshold is applied to
    the RAW Jina score, then the bounded plan term reorders what survived.
    """
    cfg = config.DEFAULT_ANSWERING
    surviving = [(chunk, raw[chunk.chunk_id]) for chunk in pool
                 if chunk.chunk_id in raw
                 and raw[chunk.chunk_id] >= cfg.rerank_threshold]
    if signals:
        arm_weights = weights.only(*signals)
        items = [(chunk, query_aware_rerank.adjust(score, chunk, plan_obj,
                                                   arm_weights))
                 for chunk, score in surviving]
    else:
        items = list(surviving)
    items.sort(key=lambda pair: pair[1], reverse=True)

    outcome = reranker.RerankOutcome(items=items, applied=True)
    selected, _stats = selection.select(items, cfg, outcome=outcome)
    documents = list(dict.fromkeys(
        getattr(chunk, "document_id", None) for chunk, _ in selected
        if getattr(chunk, "document_id", None)))
    scored = metrics.score_retrieval(expected, documents)
    ranks = {document: index + 1 for index, document in enumerate(documents)}
    return {"recall": scored.recall_at_k, "precision": scored.precision_at_k,
            "mrr": scored.mrr, "documents": documents,
            "n_selected": len(selected),
            "gold_ranks": {g: ranks.get(g) for g in expected}}


def run(cache: ScoreCache, limit: int | None, plans_path: Path,
        weights: CompatibilityWeights) -> list[dict]:
    dataset = ds.load()
    saved_plans = rse.load_saved_plans(plans_path)
    cases = dataset.cases if limit is None else dataset.cases[:limit]
    rows: list[dict] = []
    for raw_case in cases:
        if raw_case.expected_behavior == "refuse":
            continue
        saved = saved_plans.get(raw_case.eval_id)
        if saved is None:
            print(f"{raw_case.eval_id}: skipped (no saved plan)", flush=True)
            continue
        plan_obj = plan_from_record(saved, raw_case.question)
        user = get_user(raw_case.user_id)
        scope = ENGINE.get_authorized_scope(user, None)
        text = rse.search_text_for(raw_case, dataset)
        top_k = min(max(config.DEFAULT_ANSWERING.candidate_pool * 4, 10),
                    MAX_CANDIDATES)
        fused = rse.phase3_retrieve(text, top_k=top_k, filters=scope)
        request_id = f"phase7-{raw_case.eval_id.lower()}"
        # One pool for every arm: Phase 7 changes ranking, never retrieval.
        pool = build_pool(fused, plan_obj, user, request_id, rework=True)
        expected = list(dict.fromkeys(raw_case.expected_sources))

        variants = {PLAIN: text,
                    HINTED: query_aware_rerank.plan_aware_query(text, plan_obj)}
        scores = {name: jina_scores(cache, raw_case.eval_id, name, query, pool)
                  for name, query in variants.items()}

        row: dict = {"eval_id": raw_case.eval_id,
                     "category": raw_case.category,
                     "expected": expected,
                     "pool_documents": len(dict.fromkeys(
                         c.document_id for c in pool if c.document_id)),
                     "hint_added_chars": len(variants[HINTED]) - len(text),
                     "plan": {"intent": plan_obj.intent,
                              "required_aspects": list(plan_obj.required_aspects),
                              "evidence_types": list(plan_obj.evidence_types),
                              "identifiers": list(plan_obj.identifiers),
                              "seller_scope": list(plan_obj.seller_scope)}}
        for arm, (signals, variant) in ARMS.items():
            row[arm] = score_arm(pool, scores[variant], plan_obj, signals,
                                 weights, expected)
        rows.append(row)
        base, comb = row["baseline"], row["combined"]
        print(f"{raw_case.eval_id}: recall {base['recall']}->{comb['recall']} "
              f"precision {base['precision']:.3f}->{comb['precision']:.3f} "
              f"mrr {base['mrr']:.3f}->{comb['mrr']:.3f}", flush=True)
        cache.save()
    return rows


def aggregate(rows: list[dict], cache: ScoreCache,
              weights: CompatibilityWeights) -> dict:
    def mean(key, arm):
        values = [r[arm][key] for r in rows if r[arm][key] is not None]
        return round(statistics.mean(values), 4) if values else None

    summary: dict = {"scored_cases": len(rows),
                     "weights": {"total": weights.total_weight,
                                 "aspect": weights.aspect,
                                 "event": weights.event,
                                 "intent": weights.intent,
                                 "evidence": weights.evidence,
                                 "seller": weights.seller}}
    for arm in ARMS:
        summary[arm] = {"recall": mean("recall", arm),
                        "precision": mean("precision", arm),
                        "mrr": mean("mrr", arm),
                        "mean_selected_chunks": mean("n_selected", arm)}

    # Per-query movement against the Phase 6 baseline (spec section 25).
    comparisons = {}
    for arm in ARMS:
        if arm == "baseline":
            continue
        improved, regressed, unchanged = [], [], []
        for r in rows:
            base, cur = r["baseline"], r[arm]
            delta_p = (cur["precision"] or 0) - (base["precision"] or 0)
            delta_r = (cur["recall"] or 0) - (base["recall"] or 0)
            delta_m = (cur["mrr"] or 0) - (base["mrr"] or 0)
            entry = {"eval_id": r["eval_id"], "category": r["category"],
                     "recall": [base["recall"], cur["recall"]],
                     "precision": [round(base["precision"] or 0, 4),
                                   round(cur["precision"] or 0, 4)],
                     "mrr": [round(base["mrr"] or 0, 4),
                             round(cur["mrr"] or 0, 4)],
                     "gold_ranks": [base["gold_ranks"], cur["gold_ranks"]],
                     "plan": r["plan"]}
            if delta_r < 0:
                regressed.append({**entry, "reason": "recall"})
            elif delta_p > 1e-9 or delta_m > 1e-9:
                improved.append(entry)
            elif delta_p < -1e-9 or delta_m < -1e-9:
                regressed.append({**entry, "reason": "precision/mrr"})
            else:
                unchanged.append(r["eval_id"])
        comparisons[arm] = {"improved": improved, "regressed": regressed,
                            "unchanged_count": len(unchanged)}
    summary["per_query"] = comparisons
    summary["recall_regressions"] = {
        arm: [c["eval_id"] for c in comparisons[arm]["regressed"]
              if c.get("reason") == "recall"]
        for arm in comparisons}
    summary["cost"] = {
        "jina_calls_this_run": cache.calls,
        "jina_calls_per_case_in_production": 1,
        "mean_jina_latency_s": (round(statistics.mean(cache.latencies), 3)
                                if cache.latencies else None),
        "max_query_chars": cache.query_chars,
        "mean_hint_added_chars": (
            round(statistics.mean([r["hint_added_chars"] for r in rows]), 1)
            if rows else None)}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ds.PACKAGE_ROOT / "runs" / "phase7-eval.json")
    parser.add_argument("--cache", type=Path,
                        default=ds.PACKAGE_ROOT / "runs"
                        / "phase7-jina-scores.json")
    parser.add_argument("--plans", type=Path,
                        default=ds.PACKAGE_ROOT / "runs"
                        / "query-understanding-eval.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--weight", type=float,
                        help="override the total plan bound for a sweep")
    args = parser.parse_args()

    config.validate_env()
    audit.AUDIT_FILE = ds.PACKAGE_ROOT / "runs" / "phase7-audit.jsonl"
    weights = CompatibilityWeights.from_config()
    if args.weight is not None:
        weights = CompatibilityWeights(
            aspect=weights.aspect, event=weights.event, intent=weights.intent,
            evidence=weights.evidence, seller=weights.seller,
            total_weight=args.weight)

    cache = ScoreCache(args.cache)
    rows = run(cache, args.limit, args.plans, weights)
    cache.save()
    document = {
        "schema_version": 1,
        "generated_at": _now(),
        "candidate_pool": config.DEFAULT_ANSWERING.candidate_pool,
        "rerank_threshold": config.DEFAULT_ANSWERING.rerank_threshold,
        "summary": aggregate(rows, cache, weights),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    table = {arm: document["summary"][arm] for arm in ARMS}
    print(json.dumps(table, indent=2))
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
