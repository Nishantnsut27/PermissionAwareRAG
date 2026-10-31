"""Reviewed content-visibility projections applied before chunking."""
from __future__ import annotations

import re


class VisibilityProjectionError(RuntimeError):
    pass


_ORG001_COMMERCIAL_START = re.compile(
    r"The dataset contains four seller accounts\.", re.IGNORECASE)
_ORG001_SAFE_RESUME = re.compile(
    r"8\.\s+Document Metadata Convention", re.IGNORECASE)


def apply_content_visibility(document_id: str, pages: list[str]) -> list[str]:
    """Remove seller-commercial rows from shared ORG-001 before indexing.

    The same account facts are present in seller-scoped Account Overview
    documents. If the expected source boundaries drift, fail ingestion closed
    instead of accidentally publishing the shared copy.
    """
    if document_id != "ORG-001":
        return pages
    if len(pages) < 5:
        raise VisibilityProjectionError(
            "ORG-001 page structure changed; seller facts were not projected")
    shared_page = pages[4]
    start = _ORG001_COMMERCIAL_START.search(shared_page)
    resume = (_ORG001_SAFE_RESUME.search(shared_page, start.end())
              if start else None)
    if start is None or resume is None:
        raise VisibilityProjectionError(
            "ORG-001 seller facts boundary changed; refusing unreviewed content")
    projected = list(pages)
    projected[4] = shared_page[:start.start()] + shared_page[resume.start():]
    return projected
