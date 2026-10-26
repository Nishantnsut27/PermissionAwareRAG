"""Entry point for the Phase 2 ingestion pipeline.

Usage:
    python run_ingestion.py
    python run_ingestion.py --input Data
    python run_ingestion.py --no-reset
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src import config
from src.pipeline import run_ingestion


def configure_logging(verbose: bool) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    handlers.append(logging.FileHandler(
        config.LOG_FILE, encoding="utf-8"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 ingestion: raw PDF/TXT -> chunked corpus")
    parser.add_argument("--input", type=Path, default=config.DEFAULT_INPUT_DIR,
                        help="directory containing the Phase 1 dataset")
    parser.add_argument("--no-reset", action="store_true",
                        help="do not wipe processed output before ingestion")
    parser.add_argument("--verbose", action="store_true",
                        help="enable debug logging")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    summary = run_ingestion(input_dir=args.input, reset=not args.no_reset)

    documents = summary["documents"]
    integrity = summary["integrity"]
    print()
    print("Ingestion complete")
    print(f"  discovered       : {summary['discovery']['total']} "
          f"({summary['discovery']['pdf']} PDF, {summary['discovery']['txt']} TXT)")
    print(f"  processed        : {documents['processed']}")
    print(f"  failed           : {documents['failed']}")
    print(f"  validation errors: {summary['validation']['errors']}")
    print(f"  total chunks     : {summary['chunks']['total']}")
    print(f"  integrity        : "
          f"{'PASS' if all(integrity.values()) else 'FAIL'}")
    print(f"  output           : {config.CHUNKS_FILE}")

    return 0 if documents["failed"] == 0 and all(integrity.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
