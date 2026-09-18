"""Ask the permission-aware pipeline a question from the command line."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from answering.src import config, service

# LLM answers contain characters the default Windows console codepage cannot
# encode; force UTF-8 so printing an answer never raises.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 5 permission-aware question answering")
    parser.add_argument("user_id", help="prototype identity, e.g. U-001")
    parser.add_argument("query")
    parser.add_argument("--seller", action="append", dest="sellers",
                        help="narrow to a seller (repeatable)")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.WARNING),
        format="%(levelname)s %(name)s %(message)s")
    try:
        config.validate_env()
    except RuntimeError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    try:
        response = service.answer_query(
            user_id=args.user_id, query=args.query,
            seller_filter=args.sellers, top_k=args.top_k)
    except service.AnsweringError as exc:
        print(f"{exc.client_message}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(response.to_dict(), indent=2, ensure_ascii=False))
        return 0

    print(f"User      : {response.user_id}")
    print(f"Request   : {response.request_id}")
    print(f"Answered  : {response.answered}")
    print(f"Scope     : {', '.join(response.effective_sellers) or '(none)'}")
    print(f"Retrieval : candidates={response.candidates} "
          f"authorized={response.authorized} withheld={response.withheld} "
          f"selected={response.selected} reranked={response.reranked}")
    print(f"Latency   : {response.latency_ms} ms\n")
    print(response.answer)
    if response.sources:
        print("\nSources:")
        for source in response.sources:
            print(f"- {source['citation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
