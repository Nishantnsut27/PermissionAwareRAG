"""Phase 6 rework investigation: why do false positives enter the candidate pool?

For every scored golden case this captures the authorized candidate pool in
fusion order (exactly what production would truncate to the reranker pool),
the signals each candidate carries, and a taxonomy classification of every
non-gold document, so the precision problem is described by data instead of
assumption. Read-only: no pipeline code is modified.

Usage:
    python -m evaluation.run_fp_investigation --output evaluation/runs/fp-investigation.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_control.src.engine import PermissionEngine
from access_control.src.retrieval import MAX_CANDIDATES, phase3_retrieve
from access_control.src.users import get_user
from answering.src import config, query_aware
from answering.src.query_planner import QueryPlan
from evaluation import dataset as ds

ENGINE = PermissionEngine()
INTER_CASE_DELAY_S = 0.3
EVAL_POOL = config.DEFAULT_ANSWERING.candidate_pool

MONTHS = ("january|february|march|april|may|june|july|august|"
          "september|october|november|december")
DATE_TOKEN = re.compile(
    rf"\b({MONTHS})\s+(\d{{4}})\b|\b(\d{{1,2}})\s+({MONTHS})\s+(\d{{4}})\b"
    rf"|\b({MONTHS})\b|\b(20\d{{2}})\b", re.IGNORECASE)


def query_dates(text: str) -> set[str]:
    """Date and year tokens a query explicitly mentions, lower-cased."""
    found = set()
    for match in DATE_TOKEN.finditer(text or ""):
        found.add(re.sub(r"\s+", " ", match.group(0).strip().lower()))
    return found


def chunk_date_tokens(text: str) -> set[str]:
    found = set()
    lowered = (text or "").lower()
    for match in DATE_TOKEN.finditer(lowered):
        found.add(re.sub(r"\s+", " ", match.group(0).strip().lower()))
    return found


def date_overlap(query_dates_: set[str], chunk_dates: set[str]) -> bool:
    """True when the chunk repeats a specific month/year the query named."""
    for q in query_dates_:
        parts = q.split()
        for c in chunk_dates:
            if q == c:
                return True
            # "march 2026" matches "5 march 2026"; a bare year matches any
            # occurrence of that year in the chunk.
            if len(parts) >= 2 and all(p in c for p in parts):
                return True
            if len(parts) == 1 and parts[0].isdigit() and parts[0] in c:
                return True
    return False


def classify_fp(candidate_doc: str, gold_docs: set[str], pool_docs: list[str],
                text: str, metadata: dict, plan, q_dates: set[str]) -> str:
    """Assign one FP category from the phase taxonomy using observable data."""
    if candidate_doc in gold_docs:
        return "gold"
    lowered = (text or "").lower()
    seller = metadata.get("seller_id")
    gold_sellers = set()
    # Seller of the gold docs is approximated by what the query mentions,
    # which is what the candidate shares.
    shared_seller = bool(seller and seller != "Organization-wide"
                         and seller in set(plan.seller_scope or ()))
    org_wide = seller == "Organization-wide"
    doc_type = metadata.get("document_type") or ""
    evidence = query_aware.DOCUMENT_TYPE_TO_EVIDENCE.get(doc_type)
    planned_evidence = set(plan.evidence_types or ())
    q_lower = (plan.original_query or "").lower()

    if shared_seller and not date_overlap(q_dates, chunk_date_tokens(text)):
        # Same seller, but the query's explicit time frame is absent.
        if plan.required_aspects and not _aspect_hits(plan, text):
            return "A. same seller, wrong event/time"
        return "A. same seller, wrong evidence"
    if not shared_seller and org_wide:
        return "D. generic organization document"
    if planned_evidence and evidence not in planned_evidence:
        return "F. wrong document type"
    if re.search(rf"(?<![a-z0-9-]){'|'.join(re.escape(w) for w in q_lower.split() if len(w) > 4)}(?![a-z0-9-])",
                 lowered) and not plan.required_aspects:
        return "C. same keywords, wrong event"
    if evidence and planned_evidence and evidence in planned_evidence:
        return "G. related evidence type, not answer-bearing"
    return "I. other"


ASPECT_LEXICON = {
    "cause": ("root cause", "caused by", "cause of", "because", "due to",
              "triggered", "defect", "failure", "misconfig", "bug"),
    "containment_actions": ("containment", "mitigation", "rolled back",
                            "rollback", "revoked", "blocked", "rotated",
                            "remediation", "corrective", "hotfix", "patched"),
    "financial_impact": ("inr", "refund", "credit note", "settlement",
                         "shortfall", "amount", "cost", "fine", "penalty",
                         "absorbed", "revenue", "payout", "reversal"),
    "timeline": ("on ", "dated", "duration", "resolved by", "restored",
                 "timeline", "detected at", "window"),
    "status": ("status", "currently", "pending", "approved", "completed",
               "in progress", "closed", "open"),
    "policy_requirement": ("must", "shall", "required", "policy states",
                           "requires", "procedure requires", "threshold"),
    "comparison": ("whereas", "in contrast", "compared", "similar", "differ"),
    "quantities": ("sku", "orders", "count", "number of", "volume", "records",
                   "affected"),
    "responsibility": ("approved by", "signed off", "decision", "owner",
                       "accountable", "agreed"),
}


def _aspect_hits(plan, text: str) -> list[str]:
    lowered = (text or "").lower()
    hits = []
    for aspect in plan.required_aspects or ():
        for marker in ASPECT_LEXICON.get(aspect, ()):
            if marker in lowered:
                hits.append(aspect)
                break
    return hits


def run(output: Path, limit: int | None = None) -> dict:
    dataset = ds.load()
    saved = json.loads((ds.PACKAGE_ROOT / "runs"
                        / "query-understanding-eval.json").read_text(encoding="utf-8"))
    plans = {r["eval_id"]: r for r in saved["records"]}
    cases = dataset.cases if limit is None else dataset.cases[:limit]
    rows = []
    for index, case in enumerate(cases):
        if case.expected_behavior == "refuse":
            continue
        user = get_user(case.user_id)
        scope = ENGINE.get_authorized_scope(user, None)
        fused = phase3_retrieve(case.question, top_k=min(max(EVAL_POOL * 4, 10),
                                                         MAX_CANDIDATES),
                                filters=scope)
        allowed, _denied = ENGINE.authorize_chunks(user, fused,
                                                   f"fpinv-{case.eval_id.lower()}")
        record = plans.get(case.eval_id) or {}
        plan = QueryPlan(
            original_query=case.question,
            intent=record.get("intent", "factual_lookup"),
            entities=tuple(record.get("entities") or ()),
            seller_scope=tuple(record.get("seller_scope") or ()),
            identifiers=tuple(record.get("identifiers") or ()),
            required_aspects=tuple(record.get("required_aspects") or ()),
            evidence_types=tuple(record.get("evidence_types") or ()),
        )
        gold = set(case.expected_sources)
        pool = allowed[:EVAL_POOL]
        pool_docs = list(dict.fromkeys(
            c.document_id for c in pool if c.document_id))
        q_dates = query_dates(case.question)
        per_doc: dict[str, dict] = {}
        for position, chunk in enumerate(pool, start=1):
            doc = chunk.document_id
            entry = per_doc.setdefault(doc, {
                "rrf_best": position, "chunks": 0, "text": "",
                "metadata": chunk.metadata or {},
                "identifier": False})
            entry["chunks"] += 1
            entry["rrf_best"] = min(entry["rrf_best"], position)
            entry["text"] += " " + (chunk.text or "")[:1500]
            signals = query_aware.evaluate_signals(chunk, plan)
            entry["identifier"] = entry["identifier"] or signals.identifier_match
        docs = []
        for doc, entry in per_doc.items():
            category = classify_fp(doc, gold, pool_docs, entry["text"],
                                   entry["metadata"], plan, q_dates)
            docs.append({
                "document_id": doc,
                "gold": doc in gold,
                "rrf_best": entry["rrf_best"],
                "pool_chunks": entry["chunks"],
                "identifier_match": entry["identifier"],
                "aspect_hits": _aspect_hits(plan, entry["text"]),
                "date_overlap": date_overlap(q_dates,
                                             chunk_date_tokens(entry["text"])),
                "document_type": entry["metadata"].get("document_type"),
                "seller_id": entry["metadata"].get("seller_id"),
                "fp_category": None if doc in gold else category,
            })
        rows.append({
            "eval_id": case.eval_id,
            "question": case.question,
            "category": case.category,
            "expected": sorted(gold),
            "plan": {"intent": plan.intent,
                     "aspects": list(plan.required_aspects),
                     "evidence": list(plan.evidence_types),
                     "sellers": list(plan.seller_scope),
                     "identifiers": list(plan.identifiers)},
            "query_dates": sorted(q_dates),
            "pool_docs": pool_docs,
            "gold_in_pool": sorted(gold & set(pool_docs)),
            "gold_missed": sorted(gold - set(pool_docs)),
            "documents": docs,
        })
        print(f"{case.eval_id}: pool_docs={len(pool_docs)} "
              f"gold={len(gold & set(pool_docs))}/{len(gold)}", flush=True)
        if index < len(cases) - 1:
            time.sleep(INTER_CASE_DELAY_S)

    category_counts: Counter = Counter()
    type_counts: Counter = Counter()
    for row in rows:
        for doc in row["documents"]:
            if not doc["gold"]:
                category_counts[doc["fp_category"]] += 1
                if doc["fp_category"] in ("A. same seller, wrong evidence",
                                          "A. same seller, wrong event/time"):
                    type_counts[doc["document_type"]] += 1
    total_fp = sum(category_counts.values())
    total_gold_slots = sum(len(r["gold_in_pool"]) for r in rows)
    total_docs = sum(len(r["pool_docs"]) for r in rows)
    summary = {
        "scored_cases": len(rows),
        "pool_doc_slots": total_docs,
        "gold_slots": total_gold_slots,
        "implied_pool_precision": round(total_gold_slots / total_docs, 4)
        if total_docs else None,
        "false_positives": total_fp,
        "fp_categories": dict(category_counts.most_common()),
        "same_seller_fp_document_types": dict(type_counts.most_common()),
        "cases_with_date_in_query": sum(1 for r in rows if r["query_dates"]),
        "gold_missed_cases": [r["eval_id"] for r in rows if r["gold_missed"]],
    }
    document = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_pool": EVAL_POOL,
        "summary": summary,
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ds.PACKAGE_ROOT / "runs" / "fp-investigation.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config.validate_env()
    document = run(args.output, args.limit)
    print(json.dumps(document["summary"], indent=2))
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
