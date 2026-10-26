"""Central configuration for Phase 3 indexing and retrieval."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_EMBEDDING_MODEL = os.getenv("JINA_EMBEDDING_MODEL", "jina-embeddings-v4")
JINA_API_URL = os.getenv("JINA_API_URL", "https://api.jina.ai/v1/embeddings")

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME", "enterprise_knowledge")

PHASE2_CHUNKS_FILE = PROJECT_ROOT / "ingestion" / "processed" / "chunks" / "chunks.jsonl"

REPORTS_DIR = PACKAGE_ROOT / "reports"
INDEXING_REPORT_FILE = REPORTS_DIR / "indexing_report.json"
SANITY_REPORT_FILE = REPORTS_DIR / "sanity_checks.json"
VOCAB_FILE = REPORTS_DIR / "sparse_vocab.json"

REQUIRED_ENV_VARS = ("JINA_API_KEY", "QDRANT_URL", "QDRANT_API_KEY")


def validate_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Check .env at the repository root."
        )
    if not PHASE2_CHUNKS_FILE.is_file():
        raise RuntimeError(
            f"Phase 2 chunk corpus not found: {PHASE2_CHUNKS_FILE}. "
            "Run the Phase 2 ingestion pipeline first."
        )


@dataclass(frozen=True)
class RetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    final_top_k: int = 10
    rrf_k: int = 60
    dense_vector_name: str = "dense"
    sparse_vector_name: str = "text"
    embed_batch_size: int = 32
    embed_timeout_s: int = 60
    embed_max_retries: int = 3


DEFAULT_RETRIEVAL = RetrievalConfig()

PAYLOAD_KEYWORD_FIELDS = (
    "seller_id",
    "department",
    "classification",
    "document_type",
    "scenario_id",
)

SANITY_QUERIES = (
    ("seller information", "What is the account overview for seller S001 Aurelia Home Decor?"),
    ("support issue", "GST rate applied incorrectly on decorative lighting invoices support ticket"),
    ("engineering incident", "engineering incident INC-2026-0618 shipping label failover"),
    ("policy question", "What does policy POL-FIN-002 say about settlement adjustments?"),
    ("operations procedure", "operations runbook procedure OPS-RB-004 deployment rollback steps"),
    ("call transcript", "call transcript discussion about credit notes for incorrect tax invoices"),
    ("cross-document terminology", "settlement adjustment credit note reconciliation"),
    ("exact document identifier", "OPS-RB-004"),
)
