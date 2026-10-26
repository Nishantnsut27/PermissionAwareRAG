"""Central configuration for the Phase 2 ingestion pipeline.

Everything that a maintainer might reasonably want to change lives here so the
pipeline itself stays free of magic values.
"""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR_CANDIDATES = ("Data", "data", "phase-1-data")
DEFAULT_INPUT_DIR = next(
    (PROJECT_ROOT / name for name in INPUT_DIR_CANDIDATES
     if (PROJECT_ROOT / name).is_dir()),
    PROJECT_ROOT / "Data",
)

OUTPUT_DIR = PACKAGE_ROOT
PROCESSED_DIR = OUTPUT_DIR / "processed"
DOCUMENTS_DIR = PROCESSED_DIR / "documents"
CHUNKS_DIR = PROCESSED_DIR / "chunks"
REPORTS_DIR = OUTPUT_DIR / "reports"

DOCUMENTS_FILE = DOCUMENTS_DIR / "documents.jsonl"
CHUNKS_FILE = CHUNKS_DIR / "chunks.jsonl"
LOG_FILE = REPORTS_DIR / "ingestion.log"

SUPPORTED_EXTENSIONS = {".pdf": "pdf", ".txt": "txt"}
EXCLUDED_DIR_NAMES = {
    "processed", "reports", "__pycache__", ".git", ".github",
    "node_modules", ".ipynb_checkpoints", ".venv", "venv",
}
EXCLUDED_FILE_NAMES = {"readme.md", "readme.txt", "readme"}
EXCLUDED_EXTENSIONS = {
    ".py", ".json", ".jsonl", ".yaml", ".yml", ".csv", ".md", ".db",
    ".sqlite", ".log", ".png", ".jpg", ".jpeg", ".svg", ".html",
}

SELLER_IDS = ("S001", "S002", "S003", "S004")
ORGANIZATION_WIDE = "Organization-wide"
VALID_SELLER_VALUES = set(SELLER_IDS) | {ORGANIZATION_WIDE}

DEPARTMENTS = ("Business", "Support", "Engineering", "Operations", "IT")

CLASSIFICATIONS = ("PUBLIC", "INTERNAL", "CONFIDENTIAL", "RESTRICTED")
CLASSIFICATION_RANK = {name: rank for rank, name in enumerate(CLASSIFICATIONS)}

SCENARIO_IDS = tuple(f"SC-{i:03d}" for i in range(1, 13))
SCENARIO_PATTERN = r"SC-\d{3}"

DOCUMENT_TYPES = (
    "Organization Reference",
    "Account Overview",
    "Project Document",
    "Project Status Report",
    "Decision Record",
    "Requirements Document",
    "Implementation Plan",
    "Technical Notes",
    "Support Ticket",
    "Incident Report",
    "Security Incident Report",
    "Runbook",
    "Operational Procedure",
    "Operational Checklist",
    "Policy",
    "Call Transcript",
    "Scenario Context",
)

CATEGORY_TO_DOCUMENT_TYPE = {
    "calls": "Call Transcript",
    "support": "Support Ticket",
    "policies": "Policy",
    "sellers": "Account Overview",
    "identity": "Organization Reference",
    "scenario-context": "Scenario Context",
}

DOCUMENT_CATEGORIES = (
    "projects", "support", "calls", "engineering", "operations",
    "policies", "sellers", "identity", "scenario-context",
)

TOKENIZER_ENCODING = "cl100k_base"
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 120

TEXT_SEPARATORS = [
    "\n\n\n",
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    "",
]

PDF_METADATA_KEYS = (
    "Document ID",
    "Seller ID",
    "Department",
    "Classification",
    "Owner",
    "Created Date",
    "Version",
    "Effective Date",
    "Related Scenario ID",
    "Document Type",
)

REQUIRED_FIELDS = ("document_id", "document_type", "seller_id", "classification")
CONDITIONAL_FIELDS = ("department", "created_date", "owner", "scenario_id")

CLEAN_REPEATED_LINE_MIN_PAGES = 3
CLEAN_REPEATED_LINE_RATIO = 0.6
CLEAN_REPEATED_LINE_MAX_LEN = 120
CLEAN_DEWRAP_MIN_PREV_LEN = 60

DATE_FORMATS = ("%Y-%m-%d", "%d %B %Y", "%d %b %Y", "%d/%m/%Y", "%B %d, %Y")
