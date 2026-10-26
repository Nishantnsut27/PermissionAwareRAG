"""Metadata extraction and merging.

Priority order:
    1. explicit metadata block inside the document (primary)
    2. folder / path information (secondary)
    3. filename-derived information (tertiary)

Path and filename values are used mostly to validate the explicit metadata.
Conflicts are never resolved silently; they are recorded on the result.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from . import config
from .models import ExtractedMetadata, SourceFile

_WS = re.compile(r"[ \t\u00a0]+")

FILENAME_ID_PATTERNS = (
    re.compile(r"^(TKT-\d{4}-\d{4}-\d{3})"),
    re.compile(r"^(INC-\d{4}-\d{4})"),
    re.compile(r"^(CALL-\d{4}-\d{4}-\d{3})"),
    re.compile(r"^(OPS-RB-\d{3})"),
    re.compile(r"^(POL-[A-Z]+-\d{3})"),
)

_CALL_ID = re.compile(r"^Call ID\s*:\s*(.+?)\s*$", re.MULTILINE)
_PARTICIPANT = re.compile(
    r"^\s{2,}([A-Z][A-Za-z.\-' ]+?)\s{2,}.+?Nexora \(([A-Za-z]+)\)\s*$",
    re.MULTILINE,
)
_CLASS_LABEL = re.compile(
    r"Classification\s*:\s*(PUBLIC|INTERNAL|CONFIDENTIAL|RESTRICTED)"
)


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text.replace("\n", " ")).strip()


def _first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text, re.MULTILINE)
    return match.group(1).strip() if match else None


def normalize_date(raw: str | None) -> tuple[str | None, str | None]:
    if not raw or not raw.strip():
        return None, None
    value = raw.strip()
    for fmt in config.DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date().isoformat(), None
        except ValueError:
            continue
    return None, value


def parse_scenarios(raw: str | None) -> list[str]:
    if not raw:
        return []
    if "not scenario" in raw.lower():
        return []
    return sorted(set(re.findall(config.SCENARIO_PATTERN, raw)))


def summarize_classification(text: str) -> str | None:
    ranks = [config.CLASSIFICATION_RANK[value]
             for value in _CLASS_LABEL.findall(text)]
    if not ranks:
        return None
    return config.CLASSIFICATIONS[max(ranks)]


def _match_document_type(raw: str | None) -> str | None:
    if not raw:
        return None
    candidate = raw.strip()
    for known in sorted(config.DOCUMENT_TYPES, key=len, reverse=True):
        if candidate.lower().startswith(known.lower()):
            return known
    return None


def extract_pdf_metadata(page_one_text: str) -> dict[str, str]:
    text = normalize_whitespace(page_one_text)
    positions: list[tuple[str, int, int]] = []
    cursor = 0
    for key in config.PDF_METADATA_KEYS:
        index = text.find(key, cursor)
        if index == -1:
            continue
        positions.append((key, index, index + len(key)))
        cursor = index + len(key)

    values: dict[str, str] = {}
    for order, (key, _, end) in enumerate(positions):
        if key == "Document Type":
            match = re.search(
                r"Document Type\s+(.+?)(?=\s\d+\.\s|\sTicket Header"
                r"|\sAccount Snapshot|$)",
                text,
            )
            values[key] = match.group(1).strip() if match else ""
            continue
        next_start = positions[order + 1][1] if order + 1 < len(positions) else len(text)
        values[key] = text[end:next_start].strip()
    return values


def extract_call_metadata(text: str) -> dict[str, object]:
    seller_raw = _first_match(r"^Seller\s*:\s*(.+?)\s*$", text)
    seller_match = re.search(r"(S00\d)", seller_raw) if seller_raw else None

    owner = None
    department = None
    departments: list[str] = []
    for match in _PARTICIPANT.finditer(text):
        name = match.group(1).strip()
        dept = match.group(2).strip()
        departments.append(dept)
        if department is None:
            owner, department = name, dept

    return {
        "document_id": _first_match(r"^Call ID\s*:\s*(.+?)\s*$", text),
        "created_date": _first_match(r"^Date\s*:\s*(.+?)\s*$", text),
        "seller_id": seller_match.group(1) if seller_match else None,
        "scenario": _first_match(r"^Related scenario\s*:\s*(.+?)\s*$", text),
        "classification": _first_match(
            r"^Classification\s*:\s*(.+?)\s*$", text),
        "owner": owner,
        "department": department,
        "departments": sorted(set(departments)),
    }


def derive_path_metadata(rel_path: str) -> dict[str, object]:
    parts = Path(rel_path).parts
    seller = next((part for part in parts if part in config.SELLER_IDS), None)

    if "documents" in parts:
        index = parts.index("documents")
        category = parts[index + 1] if index + 1 < len(parts) else None
    elif parts:
        category = parts[0]
    else:
        category = None

    return {
        "seller_id": seller or config.ORGANIZATION_WIDE,
        "seller_folder": seller,
        "category": category,
        "document_type": config.CATEGORY_TO_DOCUMENT_TYPE.get(category or ""),
    }


def derive_filename_document_id(filename: str) -> tuple[str, bool]:
    stem = Path(filename).stem
    for pattern in FILENAME_ID_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group(1), True
    return stem, False


def _looks_like_call(text: str) -> bool:
    return bool(_CALL_ID.search(text))


def build_metadata(source: SourceFile, pages: list[str]) -> ExtractedMetadata:
    values: dict[str, object] = {}
    provenance: dict[str, str] = {}
    conflicts: list[str] = []

    path_md = derive_path_metadata(source.rel_path)
    path_seller = path_md["seller_folder"]
    file_id, file_id_is_real = derive_filename_document_id(source.path.name)

    explicit_seller: str | None = None
    explicit_id: str | None = None
    extra: dict[str, object] = {}
    pdf_raw: dict[str, str] = {}

    if source.format == "pdf":
        pdf_raw = extract_pdf_metadata(pages[0] if pages else "")
        raw = pdf_raw
        explicit_id = (raw.get("Document ID") or "").strip() or None
        explicit_seller = (raw.get("Seller ID") or "").strip() or None
        values["document_id"] = explicit_id
        values["document_type"] = _match_document_type(raw.get("Document Type"))
        values["seller_id"] = explicit_seller
        values["department"] = (raw.get("Department") or "").strip() or None
        values["classification"] = (raw.get("Classification") or "").strip() or None
        values["owner"] = (raw.get("Owner") or "").strip() or None
        for field in ("document_id", "document_type", "seller_id",
                      "department", "classification", "owner"):
            if values.get(field):
                provenance[field] = "explicit"
        created_iso, unparsed = normalize_date(raw.get("Created Date"))
        values["created_date"] = created_iso
        if created_iso:
            provenance["created_date"] = "explicit"
        if unparsed:
            extra["unparsed_created_date"] = unparsed
        if raw.get("Version"):
            extra["version"] = raw["Version"].strip()
        if raw.get("Effective Date"):
            eff_iso, _ = normalize_date(raw["Effective Date"])
            extra["effective_date"] = eff_iso or raw["Effective Date"].strip()
    elif _looks_like_call(pages[0] if pages else ""):
        text = pages[0] if pages else ""
        call = extract_call_metadata(text)
        explicit_id = call.get("document_id")
        explicit_seller = call.get("seller_id")
        values["document_id"] = explicit_id
        values["seller_id"] = explicit_seller
        values["classification"] = call.get("classification")
        values["owner"] = call.get("owner")
        values["department"] = call.get("department")
        for field in ("document_id", "seller_id", "classification"):
            if values.get(field):
                provenance[field] = "explicit"
        for field in ("owner", "department"):
            if values.get(field):
                provenance[field] = "participants"
        created_iso, unparsed = normalize_date(call.get("created_date"))
        values["created_date"] = created_iso
        if created_iso:
            provenance["created_date"] = "explicit"
        if unparsed:
            extra["unparsed_created_date"] = unparsed
        if call.get("departments"):
            extra["departments_present"] = call["departments"]
    else:
        text = pages[0] if pages else ""
        values["document_id"] = file_id
        values["document_type"] = path_md.get("document_type")
        values["seller_id"] = path_md["seller_id"]
        values["classification"] = summarize_classification(text)
        values["department"] = None
        values["owner"] = None
        values["created_date"] = None
        provenance["document_id"] = "filename"
        if values.get("document_type"):
            provenance["document_type"] = "path"
        provenance["seller_id"] = "path"
        if values.get("classification"):
            provenance["classification"] = "content"

    scenario_raw = None
    if source.format == "pdf":
        scenario_raw = pdf_raw.get("Related Scenario ID")
    else:
        scenario_raw = _first_match(r"^Related scenario\s*:\s*(.+?)\s*$",
                                    pages[0] if pages else "")
    scenario_ids = parse_scenarios(scenario_raw)
    values["scenario_ids"] = scenario_ids
    values["scenario_id"] = ", ".join(scenario_ids) if scenario_ids else None
    if scenario_ids:
        provenance["scenario_id"] = "explicit"

    if not values.get("document_type"):
        values["document_type"] = path_md.get("document_type")
        if values.get("document_type"):
            provenance["document_type"] = "path"
    if not values.get("seller_id"):
        values["seller_id"] = path_md["seller_id"]
        provenance["seller_id"] = "path"

    if explicit_id and file_id_is_real and explicit_id != file_id:
        conflicts.append(
            f"document_id mismatch: explicit '{explicit_id}' vs "
            f"filename '{file_id}' ({source.rel_path})")
    if explicit_seller and path_seller and explicit_seller != path_seller:
        conflicts.append(
            f"seller_id mismatch: explicit '{explicit_seller}' vs "
            f"path '{path_seller}' ({source.rel_path})")

    values["extra_metadata"] = extra
    return ExtractedMetadata(values=values, provenance=provenance,
                             conflicts=conflicts)
