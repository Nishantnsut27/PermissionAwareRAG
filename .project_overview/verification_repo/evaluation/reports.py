"""Read saved checkpoints for the dashboard without executing the pipeline."""
from __future__ import annotations

import re
from pathlib import Path

from . import dataset as ds, runner


class ReportNotFound(ValueError):
    pass


def resolve(report: str | None = None) -> Path:
    """Accept only the default report or one file immediately under runs/."""
    if report is None or report == "results.json":
        path = ds.RESULTS_FILE
        root = ds.PACKAGE_ROOT.resolve()
    elif re.fullmatch(r"runs/[A-Za-z0-9][A-Za-z0-9_.-]*\.json", report):
        root = (ds.PACKAGE_ROOT / "runs").resolve()
        path = root / report.removeprefix("runs/")
    else:
        raise ReportNotFound("Unknown evaluation report.")
    resolved = path.resolve()
    if resolved.parent != root or path.is_symlink():
        raise ReportNotFound("Unknown evaluation report.")
    return resolved


def load(report: str | None = None) -> dict | None:
    document = runner.load_results(resolve(report))
    if document is None and report is not None:
        raise ReportNotFound("The selected report is missing or unreadable. Refresh the run list.")
    return document


def summary(document: dict) -> dict:
    result = {key: document.get(key) for key in (
        "run_id", "generated_at", "started_at", "duration_s", "model", "partial",
        "summary", "schema_version", "selection_complete", "stop_reason", "progress")}
    rows = document.get("results") or []
    selected_ids = document.get("selected_ids") or [r["eval_id"] for r in rows]
    recorded_ids = {r["eval_id"] for r in rows}
    result.update({
        "selected_cases": len(selected_ids),
        "pending_ids": [key for key in selected_ids if key not in recorded_ids],
        "report_dataset": document.get("dataset"),
    })
    return result


def list_reports() -> list[dict]:
    paths = [ds.RESULTS_FILE, *sorted((ds.PACKAGE_ROOT / "runs").glob("*.json"))]
    reports = []
    for path in paths:
        report_id = path.relative_to(ds.PACKAGE_ROOT).as_posix()
        try:
            document = load(report_id)
        except (ReportNotFound, OSError):
            continue
        reports.append({"report_id": report_id, **summary(document)})
    return sorted(reports, key=lambda row: row.get("generated_at") or "", reverse=True)
