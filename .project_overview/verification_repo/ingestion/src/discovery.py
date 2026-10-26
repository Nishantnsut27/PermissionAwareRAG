"""Recursive discovery of supported source documents."""
from __future__ import annotations

import logging
from pathlib import Path

from . import config
from .models import SourceFile

logger = logging.getLogger(__name__)


def _is_excluded(relative: Path) -> bool:
    parts = set(relative.parts[:-1])
    if parts & config.EXCLUDED_DIR_NAMES:
        return True
    return False


def discover(input_dir: Path) -> tuple[list[SourceFile], list[str]]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    found: list[SourceFile] = []
    skipped: list[str] = []

    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(input_dir)
        name = path.name

        if name.startswith(".") or _is_excluded(relative):
            continue

        extension = path.suffix.lower()
        if extension in config.SUPPORTED_EXTENSIONS:
            found.append(SourceFile(
                path=path,
                rel_path=relative.as_posix(),
                extension=extension,
                format=config.SUPPORTED_EXTENSIONS[extension],
            ))
            continue

        if (name.lower() in config.EXCLUDED_FILE_NAMES
                or extension in config.EXCLUDED_EXTENSIONS):
            skipped.append(relative.as_posix())

    return found, skipped


def summarize(files: list[SourceFile]) -> dict[str, int]:
    pdf_count = sum(1 for f in files if f.format == "pdf")
    txt_count = sum(1 for f in files if f.format == "txt")
    return {"pdf": pdf_count, "txt": txt_count, "total": len(files)}
