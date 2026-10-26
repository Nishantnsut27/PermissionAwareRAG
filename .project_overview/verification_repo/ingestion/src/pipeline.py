"""Pipeline orchestration.

The pipeline answers only "what does this document belong to?". It never
filters content by user, department or classification. Every security-relevant
field is preserved so a later authorisation layer can make that decision.
"""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

from . import chunking, cleaning, config, discovery, loaders, metadata
from . import reporting, validation
from .models import ProcessedDocument, SourceFile

logger = logging.getLogger(__name__)


def _hash_sources(files: list[SourceFile]) -> dict[str, str]:
    digests: dict[str, str] = {}
    for source in files:
        digest = hashlib.sha256(source.path.read_bytes()).hexdigest()
        digests[source.rel_path] = digest
    return digests


def reset_output() -> None:
    """Clean-rebuild strategy: previous processed output is removed first."""
    for path in (config.PROCESSED_DIR, config.REPORTS_DIR):
        if path.exists():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file():
                    if child == config.LOG_FILE:
                        continue
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            try:
                path.rmdir()
            except OSError:
                pass
    config.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _clean_document(source: SourceFile, pages: list[str]):
    if source.format == "pdf":
        cleaned_pages = cleaning.clean_pdf_pages(pages)
        return cleaning.build_document_text(cleaned_pages)
    text = cleaning.clean_txt_text(pages[0] if pages else "")
    return cleaning.CleanedDocument(text=text, page_spans=[], page_count=1)


def _integrity_checks(
    processed: list[ProcessedDocument],
    chunks: list,
    discovery_summary: dict,
    failed: list[dict],
    sources_unchanged: bool,
) -> dict:
    source_map = {doc.document_id: doc for doc in processed}

    def chunk_doc(chunk) -> ProcessedDocument | None:
        return source_map.get(chunk.document_id)

    all_chunks = chunks
    pdf_chunks = [c for c in all_chunks if c.source_format == "pdf"]

    seen_ids: set[str] = set()
    duplicate_ids = False
    seen_text: set[tuple[str, str]] = set()
    duplicate_text = False
    for chunk in all_chunks:
        if chunk.chunk_id in seen_ids:
            duplicate_ids = True
        seen_ids.add(chunk.chunk_id)
        key = (chunk.document_id, chunk.text)
        if key in seen_text:
            duplicate_text = True
        seen_text.add(key)

    checks = {
        "chunks_generated": len(all_chunks) > 0,
        "expected_documents_match": (
            len(processed) + len(failed) == discovery_summary["total"]),
        "all_chunks_have_text": all(c.text.strip() for c in all_chunks),
        "chunk_ids_unique": not duplicate_ids,
        "all_chunks_have_document_id": all(c.document_id for c in all_chunks),
        "all_chunks_have_source": all(c.source_file for c in all_chunks),
        "all_chunks_have_classification": all(
            c.classification in config.CLASSIFICATIONS for c in all_chunks),
        "seller_metadata_preserved": all(
            c.seller_id in config.VALID_SELLER_VALUES for c in all_chunks),
        "department_metadata_preserved": all(
            (chunk_doc(c) is None) or (chunk_doc(c).department is None)
            or bool(c.department) for c in all_chunks),
        "scenario_metadata_preserved": all(
            (not c.scenario_ids) or (chunk_doc(c) is not None
                                     and c.scenario_ids
                                     == chunk_doc(c).scenario_ids)
            for c in all_chunks),
        "pdf_page_provenance_present": all(c.page_numbers for c in pdf_chunks),
        "no_duplicate_chunks": not duplicate_text,
        "chunk_sizes_within_target": all(
            c.token_count <= config.CHUNK_SIZE_TOKENS for c in all_chunks),
        "overlap_configured": (
            0 < config.CHUNK_OVERLAP_TOKENS < config.CHUNK_SIZE_TOKENS),
        "source_files_unchanged": sources_unchanged,
    }
    return checks


def run_ingestion(
    input_dir: Path | None = None,
    reset: bool = True,
) -> dict:
    started = time.perf_counter()
    input_dir = Path(input_dir) if input_dir else config.DEFAULT_INPUT_DIR

    logger.info("Discovery: scanning %s", input_dir)
    files, skipped = discovery.discover(input_dir)
    discovery_summary = discovery.summarize(files)
    logger.info("Discovery: %d PDF, %d TXT, %d total (%d skipped)",
                discovery_summary["pdf"], discovery_summary["txt"],
                discovery_summary["total"], len(skipped))

    expected_total = discovery_summary["total"]
    if expected_total != 63:
        logger.warning(
            "Discovered %d documents; the Phase 1 dataset is expected to "
            "contain 63 (52 PDF + 11 TXT)", expected_total)

    hashes_before = _hash_sources(files)
    if reset:
        reset_output()

    chunker = chunking.TokenChunker()
    logger.info("Chunking: %s, size=%d tokens, overlap=%d tokens",
                type(chunker.splitter).__name__, chunker.chunk_size,
                chunker.chunk_overlap)

    processed: list[ProcessedDocument] = []
    chunks: list = []
    failed: list[dict] = []
    issues: list[dict] = []

    for position, source in enumerate(files, start=1):
        try:
            pages = loaders.load_pages(source)
            extracted = metadata.build_metadata(source, pages)
            doc_issues = validation.validate(source, extracted)
            issues.extend(issue.to_dict() for issue in doc_issues)

            if validation.has_errors(doc_issues):
                failed.append({
                    "source_file": source.rel_path,
                    "document_id": extracted.values.get("document_id"),
                    "errors": [issue.to_dict() for issue in doc_issues
                               if issue.severity == "ERROR"],
                })
                logger.warning("Failed [%d/%d]: %s", position, len(files),
                               source.rel_path)
                continue

            cleaned = _clean_document(source, pages)
            base_metadata = dict(extracted.values)
            base_metadata.update({
                "source_file": source.rel_path,
                "source_format": source.format,
            })
            document_chunks = chunking.chunk_document(
                base_metadata, cleaned, chunker)

            token_count = sum(chunk.token_count for chunk in document_chunks)
            processed.append(ProcessedDocument(
                document_id=base_metadata["document_id"],
                document_type=base_metadata["document_type"],
                seller_id=base_metadata["seller_id"],
                department=base_metadata.get("department"),
                classification=base_metadata["classification"],
                created_date=base_metadata.get("created_date"),
                owner=base_metadata.get("owner"),
                scenario_id=base_metadata.get("scenario_id"),
                scenario_ids=list(base_metadata.get("scenario_ids") or []),
                source_file=source.rel_path,
                source_format=source.format,
                page_count=cleaned.page_count,
                char_count=len(cleaned.text),
                token_count=token_count,
                chunk_count=len(document_chunks),
                metadata_provenance=extracted.provenance,
                extra_metadata=base_metadata.get("extra_metadata", {}),
            ))
            chunks.extend(document_chunks)
            logger.info("Processed [%d/%d]: %s -> %d chunks",
                        position, len(files), source.rel_path,
                        len(document_chunks))
        except Exception as exc:  # one bad file must not stop the run
            logger.exception("Error processing %s", source.rel_path)
            failed.append({
                "source_file": source.rel_path,
                "document_id": None,
                "errors": [{"severity": "ERROR", "code": "LOAD_FAILURE",
                            "message": str(exc),
                            "source_file": source.rel_path}],
            })

    hashes_after = _hash_sources(files)
    sources_unchanged = hashes_before == hashes_after

    integrity = _integrity_checks(processed, chunks, discovery_summary,
                                  failed, sources_unchanged)
    if not all(integrity.values()):
        failed_checks = [name for name, ok in integrity.items() if not ok]
        logger.warning("Integrity checks failed: %s", ", ".join(failed_checks))

    chunk_dicts = [chunk.to_dict() for chunk in chunks]
    document_dicts = [doc.to_dict() for doc in processed]
    reporting.write_jsonl(config.DOCUMENTS_FILE, document_dicts)
    reporting.write_jsonl(config.CHUNKS_FILE, chunk_dicts)
    logger.info("Output: %d documents, %d chunks written",
                len(document_dicts), len(chunk_dicts))

    duration = time.perf_counter() - started
    summary = reporting.build_summary(
        discovery=discovery_summary,
        processed=document_dicts,
        failed=failed,
        chunks=chunk_dicts,
        issues=issues,
        integrity=integrity,
        duration_seconds=duration,
        input_dir=input_dir,
        output_dir=config.OUTPUT_DIR,
    )
    summary["skipped_files"] = skipped
    reporting.write_summary(summary)
    reporting.write_validation_errors(summary, failed)
    logger.info("Reports written to %s (%.2fs)", config.REPORTS_DIR, duration)

    return summary
