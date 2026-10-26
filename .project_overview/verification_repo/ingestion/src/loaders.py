"""Document loaders.

Both formats ultimately produce the same structure: a list of pages, where a
TXT file is treated as a single page carrying its whole content. This keeps the
downstream stages format-agnostic.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_pdf_pages(path: Path) -> list[str]:
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        reader.decrypt("")
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Page %d of %s failed to extract: %s",
                           index, path.name, exc)
            pages.append("")
    return pages


def load_txt_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def load_pages(source) -> list[str]:
    if source.format == "pdf":
        return load_pdf_pages(source.path)
    return [load_txt_text(source.path)]
