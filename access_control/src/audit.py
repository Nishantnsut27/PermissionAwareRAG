"""Lightweight authorization audit log (no document content, no query text)."""
from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

AUDIT_FILE = Path(__file__).resolve().parents[1] / "reports" / "audit.jsonl"
MAX_AUDIT_BYTES = 10 * 1024 * 1024
_lock = threading.Lock()


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
    data = "".join(lines)
    if len(data.encode("utf-8")) > MAX_AUDIT_BYTES:
        import logging
        logging.getLogger(__name__).error("audit batch exceeds configured sink size")
        return
    AUDIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        try:
            existing = AUDIT_FILE.stat().st_size if AUDIT_FILE.exists() else 0
            if existing + len(data.encode("utf-8")) > MAX_AUDIT_BYTES:
                previous = AUDIT_FILE.with_suffix(AUDIT_FILE.suffix + ".1")
                if previous.exists():
                    previous.unlink()
                if AUDIT_FILE.exists():
                    os.replace(AUDIT_FILE, previous)
            with open(AUDIT_FILE, "a", encoding="utf-8") as fh:
                fh.write(data)
        except OSError:
            # Audit availability is reported in application logs by callers;
            # an I/O failure must not turn an authorization denial into allow.
            import logging
            logging.getLogger(__name__).exception("audit sink write failed")


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
