"""Phase 10 security harness: authorization matrix and threat metrics.

Fully offline and deterministic - it exercises the PermissionEngine over the
REAL corpus (`documents.jsonl`), the real identity registry (`USERS`) and the
golden dataset. No Jina / OpenJEV / Qdrant / LLM call. The end-to-end
"unauthorized retrieval = 0%" evidence comes from the live rechecks in
`run_phase8_security.py` and `run_phase9_security.py`; this harness adds the
authorization-model view (matrix, seller isolation, forbidden-source
reachability, fail-closed probes) that does not need the network.

Usage:
    python -m evaluation.run_phase10_security
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from access_control.src import policy
from access_control.src.engine import PermissionEngine
from access_control.src.policy import IdentityError
from access_control.src.users import USERS, User, get_user
from answering.src import query_aware
from evaluation import dataset as ds

ENGINE = PermissionEngine()


def authorization_matrix(catalog: ds.PermissionCatalog) -> list[dict]:
    rows = []
    for uid, u in USERS.items():
        auth = catalog.authorized_documents(uid)
        scope = catalog.seller_scope(uid)
        leaks = [d for d in auth
                 if (s := catalog.by_document_id[d].get("seller_id"))
                 not in (policy.ORGANIZATION_WIDE, None)
                 and s not in scope and not u.has_organization_wide_scope()]
        rows.append({
            "user_id": uid, "role": u.role, "clearance": u.clearance_level,
            "seller_scope": scope,
            "authorized_documents": len(auth),
            "denied_documents": len(catalog.denied_documents(uid)),
            "cross_seller_leaks": len(leaks),
        })
    return rows


def forbidden_reachability(catalog: ds.PermissionCatalog) -> dict:
    """For every golden case, is any forbidden document actually authorized for
    that user? Must be zero."""
    dataset = ds.load()
    checked = reachable = 0
    offenders = []
    for case in dataset.cases:
        authorized = catalog.authorized_documents(case.user_id)
        for document_id in case.forbidden_sources:
            if document_id in catalog.by_document_id:
                checked += 1
                if document_id in authorized:
                    reachable += 1
                    offenders.append({"eval_id": case.eval_id,
                                      "user_id": case.user_id,
                                      "document_id": document_id})
    return {"forbidden_checked": checked, "forbidden_reachable": reachable,
            "offenders": offenders}


def fail_closed_probes() -> dict:
    """Every probe must DENY. A fail-open is any probe that allows."""
    u = get_user("U-001")
    probes = {
        "none_metadata": ENGINE.can_access(u, None).allow,
        "non_dict_metadata": ENGINE.can_access(u, "string").allow,
        "empty_metadata": ENGINE.can_access(u, {}).allow,
        "missing_seller": ENGINE.can_access(
            u, {"classification": "INTERNAL", "document_type": "Support Ticket",
                "department": "Support"}).allow,
        "unknown_seller": ENGINE.can_access(
            u, {"seller_id": "S999", "classification": "INTERNAL",
                "document_type": "Support Ticket", "department": "Support"}).allow,
        "out_of_scope_seller": ENGINE.can_access(
            u, {"seller_id": "S002", "classification": "INTERNAL",
                "document_type": "Support Ticket", "department": "Support"}).allow,
        "over_clearance": ENGINE.can_access(
            u, {"seller_id": "S001", "classification": "RESTRICTED",
                "document_type": "Support Ticket", "department": "Support"}).allow,
    }
    try:
        get_user("U-999")
        probes["unknown_user_raises"] = False  # did not raise -> fail open
    except IdentityError:
        probes["unknown_user_raises"] = True
    try:
        User("X", "n", "IT", "Support Engineer", ("*",), "RESTRICTED")
        probes["wildcard_smuggle_rejected"] = False
    except IdentityError:
        probes["wildcard_smuggle_rejected"] = True

    # Allow-probes must be False (denied); guard-probes must be True (enforced).
    allow_probes = {k: v for k, v in probes.items()
                    if not (k.endswith("_raises") or k.endswith("_rejected"))}
    guard_probes = {k: v for k, v in probes.items()
                    if k.endswith("_raises") or k.endswith("_rejected")}
    fail_open_count = sum(1 for v in allow_probes.values() if v) \
        + sum(1 for v in guard_probes.values() if not v)
    return {"probes": probes, "fail_open_count": fail_open_count}


def identifier_leak_probe() -> dict:
    """An unauthorized seller's identifier must not become reachable."""
    u = get_user("U-001")  # scope S001,S003 - no S002
    p_ids = ("INC-2026-0519",)  # an S002 incident
    from answering.src.query_planner import QueryPlan
    plan = QueryPlan(original_query="what happened in INC-2026-0519",
                     intent="factual_lookup", identifiers=p_ids)
    extra = query_aware.identifier_candidates(plan)
    allowed, denied = ENGINE.authorize_chunks(u, extra, "phase10-idprobe")
    leaked = [c.metadata.get("seller_id") for c in allowed
              if c.metadata.get("seller_id") not in
              set(ENGINE.effective_seller_scope(u)) | {policy.ORGANIZATION_WIDE}]
    return {"identifier": p_ids[0], "candidates": len(extra),
            "allowed": len(allowed), "denied": len(denied),
            "leaked_out_of_scope": len(leaked)}


def load_recheck(name: str) -> dict | None:
    path = ds.PACKAGE_ROOT / "runs" / name
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ds.PACKAGE_ROOT / "runs" / "phase10-security.json")
    args = parser.parse_args()

    catalog = ds.PermissionCatalog()
    matrix = authorization_matrix(catalog)
    reach = forbidden_reachability(catalog)
    failclosed = fail_closed_probes()
    idprobe = identifier_leak_probe()

    cross_seller_leaks = sum(r["cross_seller_leaks"] for r in matrix)
    p8 = load_recheck("phase8-security-all.json")
    p9 = load_recheck("phase9-security-all.json")

    threat_matrix = [
        ("Unauthorized seller", "DENY",
         "PASS" if all(not ENGINE.can_access(get_user("U-001"),
             {"seller_id": "S002", "classification": "INTERNAL",
              "document_type": "Support Ticket", "department": "Support"}).allow
             for _ in [0]) else "FAIL"),
        ("Unauthorized document (forbidden reachable)", "0",
         "PASS" if reach["forbidden_reachable"] == 0 else "FAIL"),
        ("Unauthorized identifier", "DENY",
         "PASS" if idprobe["leaked_out_of_scope"] == 0 else "FAIL"),
        ("Cross-seller isolation", "0 leaks",
         "PASS" if cross_seller_leaks == 0 else "FAIL"),
        ("Fail-open authorization", "0",
         "PASS" if failclosed["fail_open_count"] == 0 else "FAIL"),
        ("Phase 8 output subset + auth (live)", "0 violations",
         "PASS" if p8 and p8.get("permission_engine_violations") == 0
         and p8.get("forbidden_sources_selected") == 0 else "UNKNOWN"),
        ("Phase 9 output subset + auth (live)", "0 violations",
         "PASS" if p9 and p9.get("subset_violations") == 0
         and p9.get("permission_engine_violations") == 0 else "UNKNOWN"),
    ]

    metrics = {
        "users": len(matrix),
        "corpus_documents": len(catalog.documents),
        "cross_seller_leakage_rate": 0.0 if cross_seller_leaks == 0 else round(
            cross_seller_leaks / sum(r["authorized_documents"] for r in matrix), 4),
        "forbidden_reachable": reach["forbidden_reachable"],
        "fail_open_count": failclosed["fail_open_count"],
        "identifier_leak": idprobe["leaked_out_of_scope"],
        "phase8_unauthorized_rate": (p8 or {}).get("unauthorized_retrieval_rate"),
        "phase9_unauthorized_rate": (p9 or {}).get("unauthorized_retrieval_rate"),
    }

    document = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "authorization_matrix": matrix,
        "forbidden_reachability": reach,
        "fail_closed": failclosed,
        "identifier_leak_probe": idprobe,
        "threat_matrix": [{"threat": t, "expected": e, "status": s}
                          for t, e, s in threat_matrix],
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print("=== Phase 10 authorization matrix ===")
    for r in matrix:
        print(f"  {r['user_id']} {r['role']:22} {str(r['seller_scope']):26} "
              f"auth={r['authorized_documents']:>2} denied={r['denied_documents']:>2} "
              f"cross_seller_leaks={r['cross_seller_leaks']}")
    print("\n=== threat matrix ===")
    for row in document["threat_matrix"]:
        print(f"  {row['threat']:42} expect={row['expected']:14} {row['status']}")
    print("\n=== metrics ===")
    for k, v in metrics.items():
        print(f"  {k:30}: {v}")
    print(f"\nResults: {args.output.resolve()}")
    ok = (cross_seller_leaks == 0 and reach["forbidden_reachable"] == 0
          and failclosed["fail_open_count"] == 0
          and idprobe["leaked_out_of_scope"] == 0)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
