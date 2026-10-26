"""Ingestion reporting.

Every figure in the report is computed from the corpus that was actually
produced in this run. Nothing is estimated or hardcoded.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import config


def _distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def build_summary(
    discovery: dict,
    processed: list[dict],
    failed: list[dict],
    chunks: list[dict],
    issues: list[dict],
    integrity: dict,
    duration_seconds: float,
    input_dir: Path,
    output_dir: Path,
) -> dict:
    token_sizes = [chunk["token_count"] for chunk in chunks] or [0]

    by_document_type = Counter()
    for chunk in chunks:
        by_document_type[chunk["document_type"]] += 1

    projects = sum(1 for chunk in chunks
                   if "/projects/" in chunk["source_file"])
    calls = sum(1 for chunk in chunks if "/calls/" in chunk["source_file"])
    engineering = sum(1 for chunk in chunks
                      if "/engineering/" in chunk["source_file"])
    operations = sum(1 for chunk in chunks
                     if "/operations/" in chunk["source_file"])
    policies = sum(1 for chunk in chunks if "/policies/" in chunk["source_file"])
    support = sum(1 for chunk in chunks if "/support/" in chunk["source_file"])

    error_count = sum(1 for issue in issues if issue["severity"] == "ERROR")
    warning_count = sum(1 for issue in issues if issue["severity"] == "WARNING")

    return {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "discovery": discovery,
        "documents": {
            "processed": len(processed),
            "failed": len(failed),
            "failed_documents": failed,
        },
        "validation": {
            "errors": error_count,
            "warnings": warning_count,
            "issues": issues,
        },
        "chunks": {
            "total": len(chunks),
            "by_document_type": dict(sorted(by_document_type.items())),
            "by_category": {
                "projects": projects,
                "support": support,
                "calls": calls,
                "engineering": engineering,
                "operations": operations,
                "policies": policies,
            },
            "by_seller": _distribution([c["seller_id"] for c in chunks]),
            "by_classification": _distribution(
                [c["classification"] for c in chunks]),
            "by_department": _distribution(
                [c["department"] or "Unknown" for c in chunks]),
            "token_stats": {
                "average": round(sum(token_sizes) / len(token_sizes), 1),
                "minimum": min(token_sizes),
                "maximum": max(token_sizes),
            },
        },
        "chunking": {
            "strategy": "RecursiveCharacterTextSplitter",
            "token_aware": True,
            "tokenizer": config.TOKENIZER_ENCODING,
            "chunk_size_tokens": config.CHUNK_SIZE_TOKENS,
            "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS,
            "separators": config.TEXT_SEPARATORS,
        },
        "integrity": integrity,
        "duration_seconds": round(duration_seconds, 2),
    }


def write_summary(summary: dict) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = config.REPORTS_DIR / "ingestion_summary.json"
    txt_path = config.REPORTS_DIR / "ingestion_summary.txt"

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    discovery = summary["discovery"]
    documents = summary["documents"]
    chunks = summary["chunks"]
    lines = [
        "=" * 62,
        "PHASE 2 INGESTION SUMMARY",
        "=" * 62,
        "",
        "DOCUMENTS DISCOVERED",
        f"  PDF files discovered      : {discovery['pdf']}",
        f"  TXT files discovered      : {discovery['txt']}",
        f"  Total documents           : {discovery['total']}",
        "",
        "PROCESSING",
        f"  Successfully processed    : {documents['processed']}",
        f"  Failed                    : {documents['failed']}",
        f"  Validation errors         : {summary['validation']['errors']}",
        f"  Validation warnings       : {summary['validation']['warnings']}",
        "",
        "CHUNKS",
        f"  Total chunks              : {chunks['total']}",
        f"  Average chunk tokens      : {chunks['token_stats']['average']}",
        f"  Minimum chunk tokens      : {chunks['token_stats']['minimum']}",
        f"  Maximum chunk tokens      : {chunks['token_stats']['maximum']}",
        "",
        "CHUNKS BY DOCUMENT TYPE",
    ]
    for name, count in chunks["by_document_type"].items():
        lines.append(f"  {name:<28}: {count}")
    lines.extend(["", "CHUNKS BY CATEGORY"])
    for name, count in chunks["by_category"].items():
        lines.append(f"  {name:<28}: {count}")
    lines.extend(["", "CHUNKS BY SELLER"])
    for name, count in chunks["by_seller"].items():
        lines.append(f"  {name:<28}: {count}")
    lines.extend(["", "CHUNKS BY CLASSIFICATION"])
    for name, count in chunks["by_classification"].items():
        lines.append(f"  {name:<28}: {count}")
    lines.extend([
        "",
        "CHUNKING CONFIGURATION",
        f"  Splitter                  : {summary['chunking']['strategy']}",
        f"  Tokenizer                 : {summary['chunking']['tokenizer']}",
        f"  Target chunk size         : {summary['chunking']['chunk_size_tokens']} tokens",
        f"  Overlap                   : {summary['chunking']['chunk_overlap_tokens']} tokens",
        "",
        "INTEGRITY CHECKS",
    ])
    for name, result in summary["integrity"].items():
        if isinstance(result, bool):
            lines.append(f"  {name:<34}: {'PASS' if result else 'FAIL'}")
    lines.extend([
        "",
        f"Duration                    : {summary['duration_seconds']} s",
        "=" * 62,
    ])
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_errors(summary: dict, failed: list[dict]) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "errors": [i for i in summary["validation"]["issues"]
                   if i["severity"] == "ERROR"],
        "warnings": [i for i in summary["validation"]["issues"]
                     if i["severity"] == "WARNING"],
        "failed_documents": failed,
    }
    path = config.REPORTS_DIR / "validation_errors.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
