"""Metadata validation.

Validation is split into two severities:
    ERROR   -> the document is rejected and excluded from the processed corpus
    WARNING -> the document is processed, but the gap is recorded

The pipeline never resolves conflicts by silently choosing a value; conflicts
produced during metadata merging are converted into ERROR issues here.
"""
from __future__ import annotations

from . import config
from .models import ExtractedMetadata, SourceFile, ValidationIssue


def _issue(severity, code, message, source, doc_id=None, field=None):
    return ValidationIssue(severity=severity, code=code, message=message,
                           source_file=source.rel_path,
                           document_id=doc_id, field=field)


def validate(source: SourceFile, metadata: ExtractedMetadata
             ) -> list[ValidationIssue]:
    values = metadata.values
    issues: list[ValidationIssue] = []
    doc_id = values.get("document_id")

    for conflict in metadata.conflicts:
        issues.append(_issue("ERROR", "METADATA_CONFLICT", conflict, source,
                             doc_id))

    if not doc_id:
        issues.append(_issue("ERROR", "DOCUMENT_ID_MISSING",
                             "No document_id could be determined", source,
                             doc_id, "document_id"))

    document_type = values.get("document_type")
    if not document_type:
        issues.append(_issue("ERROR", "DOCUMENT_TYPE_MISSING",
                             "No document_type could be determined", source,
                             doc_id, "document_type"))
    elif document_type not in config.DOCUMENT_TYPES:
        issues.append(_issue("WARNING", "DOCUMENT_TYPE_UNKNOWN",
                             f"document_type '{document_type}' is not in the "
                             "known Phase 1 vocabulary", source, doc_id,
                             "document_type"))

    seller_id = values.get("seller_id")
    if not seller_id:
        issues.append(_issue("ERROR", "SELLER_ID_MISSING",
                             "No seller_id could be determined", source,
                             doc_id, "seller_id"))
    elif seller_id not in config.VALID_SELLER_VALUES:
        issues.append(_issue("ERROR", "SELLER_ID_INVALID",
                             f"seller_id '{seller_id}' is not valid", source,
                             doc_id, "seller_id"))

    department = values.get("department")
    if not department:
        issues.append(_issue("WARNING", "DEPARTMENT_MISSING",
                             "department is not present in the source", source,
                             doc_id, "department"))
    elif department not in config.DEPARTMENTS:
        issues.append(_issue("ERROR", "DEPARTMENT_INVALID",
                             f"department '{department}' is not in the "
                             "defined department vocabulary", source, doc_id,
                             "department"))

    classification = values.get("classification")
    if not classification:
        issues.append(_issue("ERROR", "CLASSIFICATION_MISSING",
                             "No classification could be determined", source,
                             doc_id, "classification"))
    elif classification not in config.CLASSIFICATIONS:
        issues.append(_issue("ERROR", "CLASSIFICATION_INVALID",
                             f"classification '{classification}' is not valid",
                             source, doc_id, "classification"))

    created_date = values.get("created_date")
    if not created_date:
        if values.get("extra_metadata", {}).get("unparsed_created_date"):
            issues.append(_issue(
                "ERROR", "DATE_INVALID",
                "created_date could not be parsed: "
                f"'{values['extra_metadata']['unparsed_created_date']}'",
                source, doc_id, "created_date"))
        else:
            issues.append(_issue("WARNING", "CREATED_DATE_MISSING",
                                 "created_date is not present in the source",
                                 source, doc_id, "created_date"))

    scenario_ids = values.get("scenario_ids") or []
    for scenario in scenario_ids:
        if scenario not in config.SCENARIO_IDS:
            issues.append(_issue("ERROR", "SCENARIO_ID_INVALID",
                                 f"scenario_id '{scenario}' is not a valid "
                                 "Phase 1 scenario", source, doc_id,
                                 "scenario_id"))
    if not scenario_ids:
        issues.append(_issue("WARNING", "SCENARIO_ID_ABSENT",
                             "document is not linked to a scenario", source,
                             doc_id, "scenario_id"))

    if not values.get("owner"):
        issues.append(_issue("WARNING", "OWNER_MISSING",
                             "owner is not present in the source", source,
                             doc_id, "owner"))

    return issues


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(issue.severity == "ERROR" for issue in issues)
