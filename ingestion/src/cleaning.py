"""Formatting normalisation.

Cleaning never summarises, paraphrases or rewrites. It only fixes the
mechanical artefacts introduced by PDF text extraction:
    * inconsistent horizontal whitespace
    * repeated page headers and footers
    * lines wrapped mid-sentence
    * runs of blank lines

Where a transformation is uncertain, the original content is preserved.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

from . import config
from .models import CleanedDocument

_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_SPACES = re.compile(r"[ \t\u00a0]+")
_BLANK_RUN = re.compile(r"\n{3,}")
_LIST_MARKER = re.compile(r"^(?:[-*\u2022]|\d+[.)])\s")


def _normalize_line(line: str) -> str:
    line = unicodedata.normalize("NFC", line)
    line = _SPACES.sub(" ", line)
    return line.strip()


def clean_page_text(text: str) -> list[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub(" ", text)
    return [_normalize_line(line) for line in text.split("\n")]


def strip_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    if len(pages) < config.CLEAN_REPEATED_LINE_MIN_PAGES:
        return pages
    counter: Counter[str] = Counter()
    for lines in pages:
        for line in set(lines):
            if line and len(line) <= config.CLEAN_REPEATED_LINE_MAX_LEN:
                counter[line] += 1

    threshold = max(2, int(len(pages) * config.CLEAN_REPEATED_LINE_RATIO))
    repeated = {line for line, count in counter.items() if count >= threshold}

    cleaned: list[list[str]] = []
    for lines in pages:
        kept = [line for line in lines if line not in repeated]
        cleaned.append(kept if any(kept) else lines)
    return cleaned


def dewarp_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for line in lines:
        if not line:
            merged.append("")
            continue
        if merged and merged[-1]:
            previous = merged[-1]
            if (len(previous) >= config.CLEAN_DEWRAP_MIN_PREV_LEN
                    and previous[-1].islower()
                    and line[0].islower()
                    and not _LIST_MARKER.match(line)):
                merged[-1] = previous + " " + line
                continue
        merged.append(line)
    return merged


def _render(lines: list[str]) -> str:
    text = "\n".join(lines)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def clean_pdf_pages(pages: list[str], dewarp: bool = True) -> list[str]:
    split_pages = [clean_page_text(page) for page in pages]
    split_pages = strip_repeated_lines(split_pages)
    rendered: list[str] = []
    for lines in split_pages:
        if dewarp:
            lines = dewarp_lines(lines)
        rendered.append(_render(lines))
    return rendered


def clean_txt_text(text: str) -> str:
    lines = clean_page_text(text)
    return _render(lines)


def build_document_text(page_texts: list[str]) -> CleanedDocument:
    """Join pages into one logical document while keeping page offsets."""
    full = ""
    spans: list[tuple[int, int, int]] = []
    for number, page_text in enumerate(page_texts, start=1):
        if not page_text.strip():
            continue
        if full:
            full += "\n\n"
        start = len(full)
        full += page_text
        spans.append((number, start, len(full)))
    return CleanedDocument(text=full, page_spans=spans,
                           page_count=len(page_texts))
