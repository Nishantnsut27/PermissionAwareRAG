"""Deterministic dev / held-out split for Phase 8 tuning (spec section 44).

Phase 7's evidence-type arm was chosen from eight arms measured on the same 45
cases it was then reported against, which makes its headline number
optimistically biased. Phase 8 must not repeat that, so the selector
configuration is chosen on DEV and the number that gets reported comes from
HELD-OUT cases that played no part in choosing it.

The split is deterministic and content-independent: scored case ids are sorted
and alternately assigned. It therefore cannot be cherry-picked to flatter a
result, it is stable across runs and machines, and it is reproducible by
anyone from the dataset alone. It is written to disk purely for auditability -
`split()` recomputes the same answer without reading that file.
"""
from __future__ import annotations

import json
from pathlib import Path

from evaluation import dataset as ds

SPLIT_FILE = ds.PACKAGE_ROOT / "runs" / "phase8-split.json"


def scored_case_ids(dataset=None) -> list[str]:
    """Golden cases that carry retrieval ground truth, in dataset order.

    `refuse` cases are excluded exactly as every other retrieval harness in
    this project excludes them: there is no expected document set to score.
    """
    dataset = dataset or ds.load()
    return [case.eval_id for case in dataset.cases
            if case.expected_behavior != "refuse"]


def split(dataset=None) -> tuple[list[str], list[str]]:
    """(dev, holdout) - sorted ids assigned alternately, dev first.

    Alternating a sorted list keeps the two halves balanced across the id
    space (and therefore across the categories, which are grouped by id) while
    involving no randomness and no seed to get wrong.
    """
    ordered = sorted(scored_case_ids(dataset))
    dev = [case_id for index, case_id in enumerate(ordered) if index % 2 == 0]
    holdout = [case_id for index, case_id in enumerate(ordered) if index % 2 == 1]
    return dev, holdout


def write(path: Path | None = None, dataset=None) -> dict:
    dataset = dataset or ds.load()
    dev, holdout = split(dataset)
    by_id = {case.eval_id: case for case in dataset.cases}
    document = {
        "method": "sorted scored eval_ids, alternately assigned (even->dev)",
        "purpose": "select the Phase 8 configuration on dev, report on holdout",
        "total_scored": len(dev) + len(holdout),
        "dev_count": len(dev),
        "holdout_count": len(holdout),
        "dev": dev,
        "holdout": holdout,
        "dev_categories": _categories(dev, by_id),
        "holdout_categories": _categories(holdout, by_id),
    }
    path = path or SPLIT_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return document


def _categories(ids: list[str], by_id: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case_id in ids:
        category = getattr(by_id.get(case_id), "category", "unknown")
        counts[category] = counts.get(category, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    document = write()
    print(json.dumps({k: v for k, v in document.items()
                      if k != "dev" and k != "holdout"}, indent=2))
    print(f"dev     ({document['dev_count']}): {', '.join(document['dev'])}")
    print(f"holdout ({document['holdout_count']}): {', '.join(document['holdout'])}")
    print(f"written: {SPLIT_FILE}")
