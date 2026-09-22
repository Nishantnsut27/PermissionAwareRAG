"""Explicit, paced evaluation runs. --plan and --validate-only are offline.

Checkpoints are atomic and resumable only for matching inputs. Exhausted quota
stops the batch at its checkpoint. Existing results are never replaced implicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answering.src import config
from indexing.src import config as index_config
from . import dataset as ds
from . import evaluator, metrics

logger = logging.getLogger(__name__)
SCHEMA_VERSION = 2
CASE_DELAY_S = float(os.getenv("EVAL_CASE_DELAY_S", "90") or 90)
RETRY_WAIT_S = float(os.getenv("EVAL_RETRY_WAIT_S", "90") or 90)
MAX_RETRIES = int(os.getenv("EVAL_MAX_RETRIES", "0") or 0)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_quota_error(message: str | None) -> bool:
    return any(marker in (message or "").lower() for marker in
               ("rate-limited", "rate limit", "quota", "429", "noavailablekeyerror"))


def select_cases(dataset: ds.GoldenDataset, limit=None, only=None) -> list[ds.EvalCase]:
    if limit is not None and limit <= 0:
        raise ds.DatasetError("--limit must be positive")
    cases = dataset.cases
    if only is not None:
        wanted = {value.strip().upper() for value in only if value.strip()}
        unknown = wanted - {case.eval_id.upper() for case in cases}
        if unknown or not wanted:
            raise ds.DatasetError(f"Unknown or empty case selection: {sorted(unknown)}")
        cases = [case for case in cases if case.eval_id.upper() in wanted]
    if limit is not None:
        cases = cases[:limit]
    wanted = {case.eval_id for case in cases}
    for case in list(cases):
        parent = case.conversation_from
        visited = {case.eval_id}
        while parent:
            if parent in visited or dataset.by_id(parent) is None:
                raise ds.DatasetError("Invalid conversation dependency")
            visited.add(parent)
            wanted.add(parent)
            parent = dataset.by_id(parent).conversation_from
    selected = [case for case in dataset.cases if case.eval_id in wanted]
    if not selected:
        raise ds.DatasetError("No cases selected")
    return selected


def fingerprint(dataset: ds.GoldenDataset, catalog: ds.PermissionCatalog) -> str:
    """Local signature excluding credentials and quota pacing.

    Cannot verify remote index contents: use a versioned collection per corpus.
    """
    settings = {
        "schema": SCHEMA_VERSION, "dataset": asdict(dataset),
        "documents": catalog.documents, "answering": asdict(config.DEFAULT_ANSWERING),
        "retrieval": asdict(index_config.DEFAULT_RETRIEVAL),
        "model": config.GROQ_MODEL, "temperature": config.GROQ_TEMPERATURE,
        "max_tokens": config.GROQ_MAX_TOKENS,
        "reasoning_effort": config.GROQ_REASONING_EFFORT,
        "rerank_model": config.JINA_RERANK_MODEL, "rerank_enabled": config.RERANK_ENABLED,
        "embedding_model": index_config.JINA_EMBEDDING_MODEL,
        "collection": index_config.QDRANT_COLLECTION_NAME,
        "qdrant_endpoint": index_config.QDRANT_URL,
        "embedding_endpoint": index_config.JINA_API_URL,
        "rerank_endpoint": config.JINA_RERANK_URL,
        "generation_endpoint": config.GROQ_API_URL,
        "case_top_k": evaluator.CASE_TOP_K,
        "judge_tokens": evaluator.judge.JUDGE_MAX_TOKENS,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    files = []
    for folder in ("evaluation", "access_control/src", "answering/src", "indexing/src"):
        files.extend((ds.PROJECT_ROOT / folder).glob("*.py"))
    files.extend([index_config.PHASE2_CHUNKS_FILE, index_config.VOCAB_FILE])
    for path in sorted(files):
        digest.update(str(path.relative_to(ds.PROJECT_ROOT)).encode())
        digest.update(path.read_bytes() if path.is_file() else b"<missing>")
    return digest.hexdigest()


def _matrix_row(result: dict) -> dict:
    retrieval = result.get("retrieval") or {}
    generation = result.get("generation") or {}
    security = result.get("security") or {}
    passed = security.get("passed")
    return {
        **{key: result.get(key, "") for key in
           ("eval_id", "user", "category", "expected_behavior", "actual_behavior", "status")},
        "recall_at_k": retrieval.get("recall_at_k"),
        "precision_at_k": retrieval.get("precision_at_k"), "mrr": retrieval.get("mrr"),
        "groundedness": generation.get("groundedness"),
        "relevancy": generation.get("answer_relevancy"),
        "correctness": generation.get("answer_correctness"),
        "security": "PASS" if passed is True else "FAIL" if passed is False else None,
    }


def save(document: dict, path: Path | None = None) -> Path:
    path = Path(path or ds.RESULTS_FILE).resolve()
    if path == ds.GOLDEN_DATASET_FILE.resolve():
        raise ValueError("refusing to write evaluation results over the golden dataset")
    if path.suffix.lower() != ".json":
        raise ValueError("evaluation output must be a .json file")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(document, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def load_results(path: Path | None = None) -> dict | None:
    path = Path(path or ds.RESULTS_FILE)
    if not path.is_file():
        return None
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or not isinstance(document.get("results"), list):
            raise ValueError("invalid results document")
        return document
    except (ValueError, OSError):
        logger.warning("Stored evaluation results could not be read: %s", path)
        return None


def _execute_case(case, catalog, history, retry_wait_s, max_retries):
    result = evaluator.run_case(case, catalog, history).to_dict()
    attempt = 0
    while (result["status"] == evaluator.ERROR and _is_quota_error(result.get("error"))
           and attempt < max_retries):
        attempt += 1
        print(f"Quota wall: waiting {retry_wait_s:.0f}s; retry {attempt}/{max_retries}",
              flush=True)
        time.sleep(retry_wait_s)
        result = evaluator.run_case(case, catalog, history).to_dict()
    return result


def run(dataset=None, catalog=None, limit=None, only=None, delay_s=None,
        retry_wait_s=None, max_retries=None, resume=False, checkpoint=True,
        verbose=True, output: Path | None = None, retry_failed: bool = False) -> dict:
    dataset = ds.load() if dataset is None else dataset
    catalog = catalog or ds.PermissionCatalog()
    delay_s = CASE_DELAY_S if delay_s is None else delay_s
    retry_wait_s = RETRY_WAIT_S if retry_wait_s is None else retry_wait_s
    max_retries = MAX_RETRIES if max_retries is None else max_retries
    if (any(not math.isfinite(v) or v < 0 for v in (delay_s, retry_wait_s))
            or max_retries < 0):
        raise ds.DatasetError("Delays and retry counts must be finite and non-negative")
    problems = ds.validate(dataset, catalog)
    if problems:
        raise ds.DatasetError("\n".join(problems))
    cases = select_cases(dataset, limit, only)
    selected_ids = [case.eval_id for case in cases]
    signature = fingerprint(dataset, catalog)
    output = Path(output or ds.RESULTS_FILE)
    if checkpoint and (output.suffix.lower() != ".json"
                       or output.resolve() == ds.GOLDEN_DATASET_FILE.resolve()):
        raise ds.DatasetError("Choose a .json results path separate from the golden dataset")
    rows: dict[str, dict] = {}
    previous_duration = 0
    run_id = datetime.now(timezone.utc).strftime("run-%Y%m%d-%H%M%S-%f")
    started_at = _now()
    if resume:
        previous = load_results(output)
        fingerprint_mismatch = (previous is not None
                                and previous.get("fingerprint") != signature)
        if (previous is None
                or (fingerprint_mismatch and not retry_failed)
                or previous.get("selected_ids") != selected_ids):
            raise ds.DatasetError("Cannot resume: missing, legacy or mismatched checkpoint. "
                                  "Use a new --output path with the intended selection.")
        rows = {row["eval_id"]: row for row in previous["results"]}
        if len(rows) != len(previous["results"]) or set(rows) - set(selected_ids):
            raise ds.DatasetError("Checkpoint contains duplicate or unexpected cases")
        run_id, started_at = previous["run_id"], previous["started_at"]
        previous_duration = previous.get("duration_s", 0)
        if retry_failed:
            # Give previously scored failures/errors one additional attempt while
            # preserving all PASS rows in the checkpoint.
            rows = {key: row for key, row in rows.items()
                    if row.get("status") == evaluator.PASS}
    elif checkpoint and output.exists():
        raise ds.DatasetError("Output already exists. Archive it, choose a new --output "
                              "path, or use --resume for a matching run.")

    for case in cases:
        parent = rows.get(case.conversation_from)
        if case.conversation_from and (not parent or parent.get("status") == evaluator.ERROR):
            rows.pop(case.eval_id, None)
    started = time.monotonic()
    stop_reason = None
    progress = {"phase": "starting", "case_id": None, "updated_at": _now()}

    def document():
        results = [rows[key] for key in selected_ids if key in rows]
        summary = metrics.aggregate(results)
        complete = len(results) == len(cases) and summary["errors"] == 0
        return {
            "schema_version": SCHEMA_VERSION, "fingerprint": signature,
            "run_id": run_id, "started_at": started_at, "generated_at": _now(),
            "partial": not complete or len(cases) != len(dataset),
            "selection_complete": complete, "selected_ids": selected_ids,
            "stop_reason": stop_reason, "dataset": dataset.summary(),
            "progress": dict(progress),
            "model": config.GROQ_MODEL, "case_delay_s": delay_s,
            "duration_s": round(previous_duration + time.monotonic() - started, 1),
            "summary": summary, "matrix": [_matrix_row(row) for row in results],
            "results": results,
        }

    executed = 0
    try:
        # Publish an empty checkpoint so a new run appears before its first call.
        if checkpoint:
            save(document(), output)
        for case in cases:
            if rows.get(case.eval_id, {}).get("status") in (evaluator.PASS, evaluator.FAIL):
                continue
            history = None
            if case.conversation_from:
                parent = rows.get(case.conversation_from)
                if not parent or not parent.get("answered") or not parent.get("answer"):
                    blocked = evaluator.CaseResult(
                        case.eval_id, case.user_id, case.user_name, case.question,
                        case.category, case.expected_behavior,
                        expected_sources=list(case.expected_sources),
                        expected_answer=case.expected_answer,
                        expected_facts=list(case.expected_facts),
                        conversation_from=case.conversation_from,
                        error=f"Dependency {case.conversation_from} produced no answer")
                    rows[case.eval_id] = blocked.to_dict()
                    if checkpoint:
                        save(document(), output)
                    continue
                history = [{"role": "user", "content": parent["question"]},
                           {"role": "assistant", "content": parent["answer"]}]
            # Pace the first resumed request too: a quota wall may be recent.
            if delay_s and (executed or resume):
                progress.update(phase="waiting", case_id=case.eval_id, updated_at=_now())
                if checkpoint:
                    save(document(), output)
                time.sleep(delay_s)
            progress.update(phase="evaluating", case_id=case.eval_id, updated_at=_now())
            if checkpoint:
                save(document(), output)
            rows[case.eval_id] = _execute_case(case, catalog, history, retry_wait_s, max_retries)
            executed += 1
            row = rows[case.eval_id]
            if verbose:
                print(f"{case.eval_id} {case.category}: {row['status']}", flush=True)
            if _is_quota_error(row.get("error")):
                stop_reason = "quota_exhausted"
            if checkpoint:
                save(document(), output)
            if stop_reason:
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted"
    progress.update(phase="stopped" if stop_reason else "finished", case_id=None,
                    updated_at=_now())
    final = document()
    if checkpoint:
        save(final, output)
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", help="comma-separated ids; prerequisite turns are included")
    parser.add_argument("--delay", type=float, default=CASE_DELAY_S)
    parser.add_argument("--retry-wait", type=float, default=RETRY_WAIT_S)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true",
                        help="on resume, rerun each saved FAIL/ERROR case once")
    parser.add_argument("--output", type=Path, default=ds.RESULTS_FILE)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--plan", action="store_true", help="preview without API calls")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.ERROR))
    try:
        dataset, catalog = ds.load(), ds.PermissionCatalog()
        problems = ds.validate(dataset, catalog)
        if problems:
            raise ds.DatasetError("\n".join(problems))
        cases = select_cases(dataset, args.limit, args.only.split(",") if args.only is not None else None)
        if (not math.isfinite(args.delay) or not math.isfinite(args.retry_wait)
                or min(args.delay, args.retry_wait, args.max_retries) < 0):
            raise ds.DatasetError("Delays and retry counts must be finite and non-negative")
        if args.validate_only or args.plan:
            print(json.dumps({"dataset": dataset.summary(),
                              "selected_ids": [c.eval_id for c in cases],
                              "selected_cases": len(cases), "network_calls": 0,
                              "minimum_pacing_minutes": round(max(0, len(cases) - 1) * args.delay / 60, 1)}, indent=2))
            print("Offline validation complete. No queries or model calls were made.")
            return 0
        config.validate_env()
        result = run(dataset, catalog, args.limit,
                     args.only.split(",") if args.only is not None else None,
                     args.delay, args.retry_wait, args.max_retries, args.resume,
                     not args.no_save, output=args.output,
                     retry_failed=args.retry_failed)
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Evaluation could not run: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"run_id": result["run_id"], "partial": result["partial"],
                      "stop_reason": result["stop_reason"], "summary": result["summary"]}, indent=2))
    if not args.no_save:
        print(f"Results: {args.output.resolve()}")
    if result["stop_reason"] == "interrupted":
        return 130
    if result["summary"]["errors"] or not result["selection_complete"]:
        return 2
    return 1 if result["summary"]["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
