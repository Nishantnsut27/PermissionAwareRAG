"""Targeted identifier-aware retrieval evaluation (Phase 6 rework).

The golden scored queries contain no structured document identifiers, so
identifier mechanics cannot show gains there (documented limitation). This
harness is the separate, targeted evaluation required for the identifier
feature: for every structured document id in the corpus it synthesizes a
query that names the id ("What happened in INC-2026-0519?"), then measures
whether the named document enters the 20-chunk candidate pool.

Arms per case:
    baseline  dense+BM25 -> RRF -> authorization -> cut
    plan      + exact-identifier augmentation + identifier-first reorder
              + bounded evidence promotion

Controls:
    positive  a user authorized for the named document must see it in the pool
    negative  a user NOT authorized for it must never see it (any arm)

Usage:
    python -m evaluation.run_targeted_identifier_eval
    python -m evaluation.run_targeted_identifier_eval --limit 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_control.src import audit
from access_control.src.engine import PermissionEngine
from access_control.src.retrieval import MAX_CANDIDATES
from access_control.src.users import USERS, get_user
from answering.src import config, query_aware
from answering.src.query_planner import (
    extract_entities,
    extract_identifiers,
    extract_sellers,
)
from evaluation import dataset as ds

ENGINE = PermissionEngine()
INTER_CASE_DELAY_S = 0.15

_QUERY_TEMPLATES = {
    "Incident Report": "What happened in {doc_id}?",
    "Security Incident Report": "What happened in {doc_id}?",
    "Support Ticket": "What is the resolution status of {doc_id}?",
    "Project Document": "What is the current status of {doc_id}?",
    "Project Status Report": "What is the current status of {doc_id}?",
    "Implementation Plan": "What is the delivery status of {doc_id}?",
    "Requirements Document": "What is the scope of {doc_id}?",
    "Decision Record": "What was decided in {doc_id}?",
    "Technical Notes": "What do {doc_id} technical notes cover?",
    "Runbook": "What does the runbook {doc_id} say?",
    "Operational Procedure": "What steps does {doc_id} define?",
    "Operational Checklist": "What items does the checklist {doc_id} cover?",
    "Policy": "What does policy {doc_id} require?",
}

_DOCUMENT_ID_RE = query_aware.DOCUMENT_ID_RE if hasattr(
    query_aware, "DOCUMENT_ID_RE") else None

_FALLBACK_ID_RE = __import__("re").compile(
    r"^(INC-\d{4}-\d{3,4}|TKT-\d{4}-\d{3,4}-\d+|PRJ-S\d{3}-\d+"
    r"|POL-[A-Z]+-\d+|OPS-RB-\d+|CALL-\d{4}-\d{3,4}-\d+|S\d-ACC-\d+)$")


def structured_documents(corpus: list[dict]) -> dict[str, dict]:
    """First chunk record per structured document id, with metadata."""
    docs: dict[str, dict] = {}
    for chunk in corpus:
        doc_id = chunk.get("document_id") or ""
        if not _FALLBACK_ID_RE.match(doc_id):
            continue
        if doc_id not in docs:
            docs[doc_id] = chunk
    return docs


def access_map(user_ids: list[str], doc_meta: dict) -> dict[str, bool]:
    return {user_id: ENGINE.can_access(get_user(user_id), doc_meta).allow
            for user_id in user_ids}


def plan_for(query: str):
    identifiers = extract_identifiers(query)
    return query_aware_qp().build_deterministic_plan(query) if hasattr(
        query_aware_qp(), "build_deterministic_plan") else None


def query_aware_qp():
    from answering.src import query_planner
    return query_planner


def deterministic_plan(query: str):
    qp = query_aware_qp()
    return qp.QueryPlan(
        original_query=query,
        intent="factual_lookup",
        entities=qp.extract_entities(query),
        seller_scope=qp.extract_sellers(query),
        identifiers=qp.extract_identifiers(query),
        fallback_used=True,
        jev_status="not_called",
    )


def run_case(doc_id: str, chunk: dict, user_ids: list[str]) -> dict | None:
    template = _QUERY_TEMPLATES.get(chunk.get("document_type"),
                                    "What information does {doc_id} contain?")
    query = template.format(doc_id=doc_id)
    access = access_map(user_ids, chunk)
    authorized_users = [u for u in user_ids if access[u]]
    unauthorized_users = [u for u in user_ids if not access[u]]
    if not authorized_users:
        return None
    user_id = authorized_users[0]
    user = get_user(user_id)
    scope = ENGINE.get_authorized_scope(user, None)
    plan = deterministic_plan(query)

    top_k = min(max(config.DEFAULT_ANSWERING.candidate_pool * 4, 10),
                MAX_CANDIDATES)
    arms: dict[str, list] = {}
    fused = query_aware.phase3_retrieve_fn(query, top_k=top_k, filters=scope)

    candidates, _added = query_aware.augment_fused(fused, plan)
    allowed, denied = ENGINE.authorize_chunks(user, candidates,
                                              f"tid-{doc_id.lower()}")
    allowed, _signals = query_aware.boosted_order(allowed, plan)
    allowed, promoted = query_aware.promote_evidence(
        allowed, plan, pool_size=config.DEFAULT_ANSWERING.candidate_pool)
    arms["plan"] = allowed[:config.DEFAULT_ANSWERING.candidate_pool]
    arms["baseline"] = fused[:config.DEFAULT_ANSWERING.candidate_pool]
    baseline_gate, _denied_base = ENGINE.authorize_chunks(
        user, list(fused), f"tid-base-{doc_id.lower()}")
    arms["baseline"] = baseline_gate[:config.DEFAULT_ANSWERING.candidate_pool]

    row: dict = {
        "doc_id": doc_id,
        "document_type": chunk.get("document_type"),
        "seller_id": chunk.get("seller_id"),
        "query": query,
        "user": user_id,
        "n_authorized_users": len(authorized_users),
        "n_unauthorized_users": len(unauthorized_users),
    }
    for name, pool in arms.items():
        chunk_ids = [c.chunk_id for c in pool]
        first_rank = next(
            (i + 1 for i, c in enumerate(pool)
             if c.document_id == doc_id), None)
        row[name] = {
            "in_pool": first_rank is not None,
            "first_chunk_rank": first_rank,
            "chunks_of_doc": sum(1 for c in pool if c.document_id == doc_id),
        }
    if unauthorized_users:
        intruder = get_user(unauthorized_users[0])
        intruder_scope = ENGINE.get_authorized_scope(intruder, None)
        intruder_candidates, _ = query_aware.augment_fused(fused, plan)
        intruder_allowed, _denied = ENGINE.authorize_chunks(
            intruder, intruder_candidates, f"tid-neg-{doc_id.lower()}")
        intruder_pool = intruder_allowed[
            :config.DEFAULT_ANSWERING.candidate_pool]
        row["negative_control"] = {
            "user": intruder.user_id,
            "leaked": any(c.document_id == doc_id for c in intruder_pool),
        }
    else:
        row["negative_control"] = {"user": None, "leaked": None,
                                   "skipped": "no unauthorized user available"}
    return row


def aggregate(rows: list[dict]) -> dict:
    def pct(values):
        return round(100 * sum(values) / len(values), 2) if values else None

    def mean(values):
        values = [v for v in values if v is not None]
        return round(statistics.mean(values), 3) if values else None

    summary: dict = {"cases": len(rows)}
    for arm in ("baseline", "plan"):
        summary[arm] = {
            "in_pool_rate": pct([r[arm]["in_pool"] for r in rows]),
            "mean_first_chunk_rank": mean(
                [r[arm]["first_chunk_rank"] for r in rows]),
            "mean_chunks_of_doc": mean(
                [r[arm]["chunks_of_doc"] for r in rows]),
        }
    summary["gains"] = {
        "rescued": sum(
            1 for r in rows
            if r["plan"]["in_pool"] and not r["baseline"]["in_pool"]),
        "lost": sum(
            1 for r in rows
            if r["baseline"]["in_pool"] and not r["plan"]["in_pool"]),
        "rank_improved": sum(
            1 for r in rows
            if r["plan"]["first_chunk_rank"] is not None
            and r["baseline"]["first_chunk_rank"] is not None
            and r["plan"]["first_chunk_rank"]
            < r["baseline"]["first_chunk_rank"]),
    }
    summary["negative_controls"] = {
        "checked": sum(1 for r in rows if r["negative_control"]["leaked"]
                       is not None),
        "leaks": sum(1 for r in rows if r["negative_control"]["leaked"]),
    }
    return summary


def run(output: Path, limit: int | None = None) -> dict:
    global query_aware
    from indexing.src.retrieval import retrieve as _retrieve
    query_aware.phase3_retrieve_fn = _retrieve

    corpus = query_aware._load_corpus()
    docs = structured_documents(corpus)
    user_ids = sorted(USERS)
    cases = sorted(docs)
    if limit:
        cases = cases[:limit]
    rows = []
    for index, doc_id in enumerate(cases):
        row = run_case(doc_id, docs[doc_id], user_ids)
        if row is not None:
            rows.append(row)
            print(f"{doc_id}: baseline_in_pool="
                  f"{row['baseline']['in_pool']} plan_in_pool="
                  f"{row['plan']['in_pool']} leak="
                  f"{row['negative_control']['leaked']}", flush=True)
        else:
            print(f"{doc_id}: skipped (no authorized demo user)", flush=True)
        if index < len(cases) - 1:
            import time
            time.sleep(INTER_CASE_DELAY_S)
    document = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidate_pool": config.DEFAULT_ANSWERING.candidate_pool,
        "summary": aggregate(rows),
        "rows": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ds.PACKAGE_ROOT / "runs"
                        / "targeted-identifier-eval.json")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    config.validate_env()
    audit.AUDIT_FILE = ds.PACKAGE_ROOT / "runs" / "targeted-identifier-audit.jsonl"
    document = run(args.output, args.limit)
    print(json.dumps(document["summary"], indent=2))
    print(f"Results: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
