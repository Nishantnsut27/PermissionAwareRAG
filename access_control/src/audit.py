"""Lightweight authorization audit log (no document content, no query text)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

AUDIT_FILE = Path(__file__).resolve().parents[1] / "reports" / "audit.jsonl"


@dataclass
class AuditEvent:
    timestamp: str
    user_id: str
    request_id: str
    document_id: str | None
    chunk_id: str | None
    decision: str
    reason: str


def _write(lines: list[str]) -> None:
    if not lines:
        return
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_FILE, "a", encoding="utf-8") as fh:
        fh.write("".join(lines))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_decision(user_id: str, request_id: str, decision: str, reason: str,
                 document_id: str | None = None,
                 chunk_id: str | None = None) -> AuditEvent:
    event = AuditEvent(_now(), user_id, request_id, document_id, chunk_id,
                       decision, reason)
    _write([json.dumps(asdict(event)) + "\n"])
    return event


def log_decisions(events: list[tuple]) -> list[AuditEvent]:
    """Batch variant of log_decision; one file write per request."""
    timestamp = _now()
    records = [
        AuditEvent(timestamp, user_id, request_id, document_id, chunk_id,
                   decision, reason)
        for user_id, request_id, decision, reason, document_id, chunk_id
        in events
    ]
    _write([json.dumps(asdict(r)) + "\n" for r in records])
    return records


def log_request(user_id: str, request_id: str, candidates: int, allowed: int,
                withheld: int, refused: bool) -> None:
    """Request-level outcome. The query text is deliberately not recorded."""
    _write([json.dumps({
        "timestamp": _now(),
        "user_id": user_id,
        "request_id": request_id,
        "event": "request",
        "candidates": candidates,
        "allowed": allowed,
        "withheld": withheld,
        "refused": refused,
    }) + "\n"])
