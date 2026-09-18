"""RBAC + ABAC authorization policy.

Clearance rank gates classification; seller scope gates seller documents;
the role matrix gates document types; organization-wide RESTRICTED material
additionally requires department fit. Clearance never overrides scope.

Everything an identity is allowed to claim is declared here too, so the engine
can reject a malformed identity instead of interpreting it generously.
"""
from __future__ import annotations

CLEARANCE_RANK = {"PUBLIC": 0, "INTERNAL": 1, "CONFIDENTIAL": 2, "RESTRICTED": 3}

ORGANIZATION_WIDE = "Organization-wide"
WILDCARD_SCOPE = "*"

SELLER_IDS = frozenset({"S001", "S002", "S003", "S004"})
DEPARTMENTS = frozenset({"Business", "Support", "Engineering", "Operations", "IT"})

# Roles whose scope may legitimately be the wildcard. Encoded as policy data so
# breadth of access is granted by the same engine as every other decision.
ORG_WIDE_SCOPE_ROLES = frozenset({"Platform Administrator"})

_COMMERCIAL = {
    "Account Overview", "Decision Record", "Project Document",
    "Project Status Report", "Requirements Document", "Implementation Plan",
}
_SHARED = {"Policy", "Organization Reference", "Scenario Context"}

ROLE_DOCUMENT_TYPES: dict[str, frozenset[str]] = {
    "Account Manager": frozenset(
        _COMMERCIAL | {"Support Ticket", "Call Transcript"} | _SHARED),
    "Support Engineer": frozenset(
        {"Support Ticket", "Call Transcript", "Incident Report",
         "Project Status Report"} | _SHARED),
    "Software Engineer": frozenset(
        {"Incident Report", "Security Incident Report", "Technical Notes",
         "Support Ticket"}
        | (_COMMERCIAL - {"Account Overview", "Decision Record"})
        | _SHARED),
    "System Engineer": frozenset(
        {"Runbook", "Operational Procedure", "Operational Checklist",
         "Incident Report", "Decision Record", "Project Status Report"}
        | _SHARED),
    "Platform Administrator": frozenset({
        "Account Overview", "Project Document", "Project Status Report",
        "Decision Record", "Requirements Document", "Implementation Plan",
        "Technical Notes", "Support Ticket", "Incident Report",
        "Security Incident Report", "Runbook", "Operational Procedure",
        "Operational Checklist", "Policy", "Call Transcript",
        "Organization Reference", "Scenario Context",
    }),
}


class IdentityError(ValueError):
    pass


class ScopeDeniedError(PermissionError):
    """Raised when a requested narrowing has no overlap with the authorized scope."""


def clearance_sufficient(user_clearance: str | None,
                         classification: str | None) -> bool:
    return (CLEARANCE_RANK.get(user_clearance, -1)
            >= CLEARANCE_RANK.get(classification, 99))


def role_permits(role: str | None, document_type: str | None) -> bool:
    allowed = ROLE_DOCUMENT_TYPES.get(role)
    return allowed is not None and document_type in allowed


def identity_problem(role, clearance, seller_scope, department) -> str | None:
    """Return a reason string if the identity is not well-formed, else None.

    An empty seller scope is valid (it simply grants no seller documents); an
    unrecognised value is not, because unrecognised values are how privilege
    escalation is smuggled in.
    """
    if role not in ROLE_DOCUMENT_TYPES:
        return f"unknown_role:{role!r}"
    if clearance not in CLEARANCE_RANK:
        return f"unknown_clearance:{clearance!r}"
    if department not in DEPARTMENTS:
        return f"unknown_department:{department!r}"
    if isinstance(seller_scope, str) or not hasattr(seller_scope, "__iter__"):
        return f"seller_scope_not_a_collection:{type(seller_scope).__name__}"
    unknown = [s for s in seller_scope
               if s != WILDCARD_SCOPE and s not in SELLER_IDS]
    if unknown:
        return f"unknown_seller_ids:{sorted(map(str, unknown))}"
    if WILDCARD_SCOPE in seller_scope and role not in ORG_WIDE_SCOPE_ROLES:
        return f"wildcard_scope_not_permitted_for_role:{role!r}"
    return None
