"""Document loaders.

Both formats ultimately produce the same structure: a list of pages, where a
TXT file is treated as a single page carrying its whole content. This keeps the
downstream stages format-agnostic.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from . import config

logger = logging.getLogger(__name__)


class PartialExtractionError(RuntimeError):
    """Extraction was incomplete; security-sensitive source is quarantined."""


def _check_size(path: Path) -> None:
    if path.stat().st_size > config.MAX_FILE_BYTES:
        raise PartialExtractionError(
            f"FILE_SIZE_LIMIT: {path.name} exceeds {config.MAX_FILE_BYTES} bytes")


def load_pdf_pages(path: Path) -> list[str]:
    _check_size(path)
    reader = PdfReader(str(path))
    if reader.is_encrypted:
        if not reader.decrypt(""):
            raise PartialExtractionError("PDF_ENCRYPTED: empty password did not decrypt file")
    if len(reader.pages) > config.MAX_PDF_PAGES:
        raise PartialExtractionError(
            f"PAGE_LIMIT: PDF has more than {config.MAX_PDF_PAGES} pages")
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
            if len(text) > config.MAX_PAGE_CHARS:
                raise PartialExtractionError(
                    f"PAGE_TEXT_LIMIT: page {index} exceeds configured character limit")
            pages.append(text)
        except PartialExtractionError:
            raise
        except Exception as exc:
            logger.warning("Page %d of %s failed to extract: %s",
                           index, path.name, exc)
            raise PartialExtractionError(
                f"PARTIAL_PDF_EXTRACTION: page {index} could not be extracted") from exc
    if sum(map(len, pages)) > config.MAX_DOCUMENT_CHARS:
        raise PartialExtractionError("DOCUMENT_TEXT_LIMIT: extracted text exceeds limit")
    return pages


def load_txt_text(path: Path) -> str:
    _check_size(path)
    try:
        text = path.read_text(encoding="utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise PartialExtractionError("PARTIAL_TEXT_EXTRACTION: invalid UTF-8") from exc
    if len(text) > config.MAX_DOCUMENT_CHARS:
        raise PartialExtractionError("DOCUMENT_TEXT_LIMIT: extracted text exceeds limit")
    return text


def load_pages(source) -> list[str]:
    if source.format == "pdf":
        return load_pdf_pages(source.path)
    return [load_txt_text(source.path)]
