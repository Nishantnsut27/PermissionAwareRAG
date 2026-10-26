"""Phase 5 evaluation: run the golden dataset through Query Understanding.

This evaluates the QueryPlan layer in isolation. Retrieval, reranking and
selection are NOT called, so no retrieval metric here is comparable to the
frozen baseline and none is claimed. The question answered is whether the
QueryPlan is reliable enough for Phase 6 to consume.

Usage:
    python -m evaluation.run_query_plan_eval --output evaluation/runs/plan-eval.json
    python -m evaluation.run_query_plan_eval --plan
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

from answering.src import query_planner as qp
from evaluation import dataset as ds

INTER_CALL_DELAY_S = 0.2

EVIDENCE_TYPE_BY_DOCUMENT_TYPE = {
    "Support Ticket": "support_ticket",
    "Incident Report": "incident_report",
    "Security Incident Report": "incident_report",
    "Project Document": "project_status",
    "Project Status Report": "project_status",
    "Implementation Plan": "project_status",
    "Requirements Document": "project_status",
    "Technical Notes": "project_status",
    "Runbook": "runbook",
    "Operational Procedure": "runbook",
    "Operational Checklist": "runbook",
    "Policy": "policy",
    "Call Transcript": "call_transcript",
    "Account Overview": "account_overview",
    "Decision Record": "decision_record",
    "Organization Reference": "reference_data",
    "Scenario Context": "reference_data",
}

INTENT_BY_CATEGORY = {
    # Only prompt_injection queries carry intent-level ground truth: their
    # information need IS an override attempt. unauthorized_access queries are
    # ordinary factual/policy asks by a user who happens to lack clearance;
    # labelling those as an intent would conflate what is asked with who is
    # asking, which the PermissionEngine - not the QueryPlan - decides.
    "prompt_injection": "unauthorized_request",
    "policy_governance": "policy",
}

KEYWORD_MULTI_PART = (" and ", " then ", "each ", "both ", "compare", "; ")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def document_type_vocabulary(catalog: ds.PermissionCatalog) -> set[str]:
    return set(EVIDENCE_TYPE_BY_DOCUMENT_TYPE)


def weak_intent_label(case) -> str | None:
    return INTENT_BY_CATEGORY.get(case.category)


def weak_evidence_label(case, catalog) -> set[str]:
    labels = set()
    for document_id in case.expected_sources:
        document = catalog.by_document_id.get(document_id) or {}
        mapped = EVIDENCE_TYPE_BY_DOCUMENT_TYPE.get(document.get("document_type"))
        if mapped:
            labels.add(mapped)
    return labels


def identifier_expectations(case) -> tuple[set[str], set[str]]:
    """Expected identifiers and whether the query itself contains any."""
    expected = {d for d in case.expected_sources}
    return expected, set()


def keyword_multi_part(case) -> bool:
    lowered = case.question.lower()
    return any(marker in lowered for marker in KEYWORD_MULTI_PART)


def seller_ids_of(case, catalog) -> set[str]:
    sellers = set()
    for document_id in case.expected_sources:
        document = catalog.by_document_id.get(document_id) or {}
        seller = document.get("seller_id")
        if seller and seller != "Organization-wide":
            sellers.add(seller)
    return sellers


def identifiers_in_text(text: str, ids: set[str]) -> set[str]:
    import re
    return {i for i in ids
            if re.search(rf"\b{re.escape(i)}\b", text or "", re.IGNORECASE)}


def sellers_in_text(text: str, sellers: set[str]) -> set[str]:
    import re
    lowered = (text or "").lower()
    mentioned = set()
    for seller in sellers:
        if re.search(rf"\b{re.escape(seller)}\b", lowered):
            mentioned.add(seller)
            continue
        for name, seller_id in qp.SELLER_NAMES.items():
            if seller_id == seller and re.search(rf"\b{re.escape(name)}\b", lowered):
                mentioned.add(seller)
                break
    return mentioned


def evaluate_case(case, catalog) -> dict:
    plan = qp.plan_query(case.question)
    record = {
        "eval_id": case.eval_id,
        "category": case.category,
        "expected_behavior": case.expected_behavior,
        "jev_status": plan.jev_status,
        "fallback_used": plan.fallback_used,
        "jev_latency_ms": plan.jev_latency_ms,
        "confidence": plan.confidence,
        "intent": plan.intent,
        "evidence_types": list(plan.evidence_types),
        "required_aspects": list(plan.required_aspects),
        "identifiers": list(plan.identifiers),
        "seller_scope": list(plan.seller_scope),
        "entities": list(plan.entities),
        "is_multi_part": plan.is_multi_part,
        "complexity": plan.complexity,
    }

    expected_ids = set(case.expected_sources) | set(case.forbidden_sources)
    in_query = identifiers_in_text(case.question, expected_ids)
    if in_query:
        hits = in_query & set(plan.identifiers)
        record["identifier_recall"] = round(len(hits) / len(in_query), 4)
        record["identifier_overreach"] = [
            i for i in plan.identifiers if i not in expected_ids]
        record["identifier_applicable"] = True
    else:
        record["identifier_applicable"] = False
        record["identifier_recall"] = None
        record["identifier_overreach"] = []

    expected_sellers = seller_ids_of(case, catalog)
    for document_id in case.forbidden_sources:
        document = catalog.by_document_id.get(document_id) or {}
        seller = document.get("seller_id")
        if seller and seller != "Organization-wide":
            expected_sellers.add(seller)
    # NOTE (metric honesty): this is recall-by-construction, not an accuracy.
    # `sellers_in_text` derives `mentioned` with the same qp.SELLER_NAMES map
    # and word-boundary regex that `extract_sellers` uses to build
    # plan.seller_scope, then intersects with the gold sellers. `mentioned` is
    # therefore always a subset of plan.seller_scope, so seller_recall is
    # guaranteed to be 1.0 and measures no error mode (and no precision). It is
    # a smoke test that the shared extractor runs, not a quality measurement.
    mentioned = sellers_in_text(case.question, expected_sellers)
    if mentioned:
        hits = mentioned & set(plan.seller_scope)
        record["seller_recall"] = round(len(hits) / len(mentioned), 4)
        record["seller_applicable"] = True
    else:
        record["seller_recall"] = None
        record["seller_applicable"] = False

    intent_label = weak_intent_label(case)
    record["intent_label"] = intent_label
    record["intent_match"] = (plan.intent == intent_label
                              if intent_label else None)

    evidence_labels = weak_evidence_label(case, catalog)
    record["evidence_label"] = sorted(evidence_labels)
    if evidence_labels and plan.evidence_types:
        record["evidence_match"] = plan.evidence_types[0] in evidence_labels
    else:
        record["evidence_match"] = None

    record["multipart_keyword_label"] = keyword_multi_part(case)

    if case.expected_behavior == "refuse":
        record["unauthorized_flagged"] = plan.intent == "unauthorized_request"
    else:
        record["unauthorized_flagged"] = None
    return record


def summarise(records: list[dict]) -> dict:
    total = len(records)
    fallback = sum(1 for r in records if r["fallback_used"])
    disabled = sum(1 for r in records if r["jev_status"] == "disabled")
    ok = sum(1 for r in records if r["jev_status"] == "ok")
    latencies = [r["jev_latency_ms"] for r in records if r["jev_status"] == "ok"]
    confidences = [r["confidence"] for r in records if r["confidence"] is not None]

    def rate(key, applicable_key=None):
        values = [r[key] for r in records if r.get(key) is not None]
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    intent_rows = [r for r in records if r["intent_match"] is not None]
    evidence_rows = [r for r in records if r["evidence_match"] is not None]
    identifier_rows = [r for r in records if r["identifier_applicable"]]
    seller_rows = [r for r in records if r["seller_applicable"]]
    multipart_rows = [r for r in records
                      if r["jev_status"] == "ok"]
    unauthorized_rows = [r for r in records
                         if r.get("unauthorized_flagged") is not None]

    complexity_counts: dict[str, int] = {}
    for r in records:
        complexity_counts[r["complexity"]] = complexity_counts.get(
            r["complexity"], 0) + 1

    return {
        "total_queries": total,
        "jev_ok": ok,
        "jev_fallback": fallback - disabled,
        "jev_disabled": disabled,
        "avg_jev_latency_ms": round(statistics.mean(latencies), 1) if latencies else None,
        "p95_jev_latency_ms": (round(sorted(latencies)[int(0.95 * (len(latencies) - 1))], 1)
                               if latencies else None),
        "avg_confidence": round(statistics.mean(confidences), 4) if confidences else None,
        "identifier_recall": rate("identifier_recall"),
        "identifier_queries": len(identifier_rows),
        "identifier_overreach_queries": sum(
            1 for r in identifier_rows if r.get("identifier_overreach")),
        "seller_recall": rate("seller_recall"),
        "seller_queries": len(seller_rows),
        "intent_agreement": (round(sum(1 for r in intent_rows if r["intent_match"])
                                   / len(intent_rows), 4) if intent_rows else None),
        "intent_labelled_queries": len(intent_rows),
        "evidence_type_agreement": (round(sum(1 for r in evidence_rows if r["evidence_match"])
                                          / len(evidence_rows), 4)
                                    if evidence_rows else None),
        "evidence_labelled_queries": len(evidence_rows),
        "multipart_jev_rate": (round(sum(1 for r in multipart_rows if r["is_multi_part"])
                                     / len(multipart_rows), 4) if multipart_rows else None),
        "multipart_keyword_rate": (round(sum(1 for r in records
                                             if r["multipart_keyword_label"])
                                         / total, 4) if total else None),
        "multipart_jev_keyword_agreement": (
            round(sum(1 for r in multipart_rows
                      if r["is_multi_part"] == r["multipart_keyword_label"])
                  / len(multipart_rows), 4) if multipart_rows else None),
        "unauthorized_flagged_rate": (
            round(sum(1 for r in unauthorized_rows if r["unauthorized_flagged"])
                  / len(unauthorized_rows), 4) if unauthorized_rows else None),
        "unauthorized_labelled_queries": len(unauthorized_rows),
        "complexity_distribution": complexity_counts,
    }


def run(output: Path | None = None, limit: int | None = None) -> dict:
    dataset = ds.load()
    catalog = ds.PermissionCatalog()
    cases = dataset.cases if limit is None else dataset.cases[:limit]
    records = []
    for index, case in enumerate(cases):
        record = evaluate_case(case, catalog)
        records.append(record)
        status = record["jev_status"]
        print(f"{case.eval_id} {case.category}: intent={record['intent']} "
              f"jev={status} {record['jev_latency_ms']}ms", flush=True)
        if index < len(cases) - 1:
            time.sleep(INTER_CALL_DELAY_S)
    document = {
        "schema_version": 1,
        "generated_at": _now(),
        "dataset_version": dataset.version,
        "summary": summarise(records),
        "records": records,
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ds.PACKAGE_ROOT / "runs"
                        / "query-understanding-eval.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    if args.plan:
        dataset = ds.load()
        print(json.dumps({"queries": len(dataset.cases),
                          "network_calls": len(dataset.cases),
                          "output": str(args.output)}, indent=2))
        return 0
    if not qp.config.OPENJEV_API_KEY:
        print("OPENJEV_API_KEY is not configured; JEV judgments would all "
              "fall back.", file=sys.stderr)
        return 2
    document = run(args.output, args.limit)
    print(json.dumps(document["summary"], indent=2))
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
