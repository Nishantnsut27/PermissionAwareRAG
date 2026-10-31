"""Central PermissionEngine: every access decision flows through here."""
from __future__ import annotations

from dataclasses import dataclass

from . import policy
from .audit import log_decisions
from .policy import IdentityError, ScopeDeniedError
from .users import User


@dataclass(frozen=True)
class Decision:
    allow: bool
    reason: str


class PermissionEngine:
    def _identity_problem(self, user: User) -> str | None:
        return policy.identity_problem(
            getattr(user, "role", None),
            getattr(user, "clearance_level", None),
            getattr(user, "seller_scope", None),
            getattr(user, "department", None),
        )

    def can_access(self, user: User, document: dict) -> Decision:
        problem = self._identity_problem(user)
        if problem is not None:
            return Decision(False, f"invalid_identity:{problem}")
        if not isinstance(document, dict):
            return Decision(False, "malformed_document_metadata")

        classification = document.get("classification")
        document_type = document.get("document_type")
        department = document.get("department")
        if (not isinstance(classification, str)
                or classification not in policy.CLEARANCE_RANK
                or not isinstance(document_type, str)
                or (department is not None
                    and (not isinstance(department, str)
                         or department not in policy.DEPARTMENTS))):
            return Decision(False, "malformed_document_metadata")
        if not policy.clearance_sufficient(user.clearance_level, classification):
            return Decision(False, "insufficient_clearance")

        seller_id = document.get("seller_id")
        if seller_id == policy.ORGANIZATION_WIDE:
            return self._organization_wide_decision(user, document)
        # Anything that is not a known seller and not the organization-wide
        # sentinel is unclassifiable scope, so it is denied rather than
        # defaulting into the organization-wide branch.
        if not isinstance(seller_id, str) or seller_id not in policy.SELLER_IDS:
            return Decision(False, "unknown_or_missing_seller_id")
        if (not user.has_organization_wide_scope()
                and seller_id not in user.seller_scope):
            return Decision(False, "seller_scope_mismatch")
        if not policy.role_permits(user.role, document.get("document_type")):
            return Decision(False, "role_document_type_denied")
        return Decision(True, "seller_scope_and_clearance_ok")

    def _organization_wide_decision(self, user: User, document: dict) -> Decision:
        if not policy.role_permits(user.role, document.get("document_type")):
            return Decision(False, "role_document_type_denied")
        if document.get("classification") == "RESTRICTED":
            department = document.get("department")
            if not user.has_organization_wide_scope() and (
                    department is None or user.department != department):
                return Decision(False, "org_restricted_department_mismatch")
        return Decision(True, "organization_wide_policy_ok")

    def effective_seller_scope(self, user: User,
                               requested_sellers=None) -> list[str] | None:
        """Authorized sellers, optionally narrowed by a caller request.

        The request can only ever intersect, never extend: an unauthorized
        seller in `requested_sellers` is dropped, and a request that overlaps
        nothing returns None so the caller can refuse.
        """
        if user.has_organization_wide_scope():
            authorized = set(policy.SELLER_IDS)
        else:
            authorized = {s for s in user.seller_scope if s in policy.SELLER_IDS}
        if requested_sellers is None:
            return sorted(authorized)
        requested = {str(s).strip().upper()
                     for s in requested_sellers if str(s).strip()}
        if not requested:
            return sorted(authorized)
        effective = authorized & requested
        return sorted(effective) if effective else None

    def get_authorized_scope(self, user: User, requested_sellers=None) -> dict:
        """Qdrant pre-filter derived from the identity. Never widens on error.

        Encodes seller-scoped and organization-wide alternatives as explicit
        boolean branches. The organization-wide RESTRICTED branch requires the
        caller's department; the independent post-retrieval check remains the
        final authority if payload metadata is malformed or inconsistent.
        """
        problem = self._identity_problem(user)
        if problem is not None:
            raise IdentityError(
                f"Refusing to build a retrieval scope for an invalid "
                f"identity: {problem}")
        user_rank = policy.CLEARANCE_RANK[user.clearance_level]
        allowed_levels = [level for level, rank in policy.CLEARANCE_RANK.items()
                          if rank <= user_rank]
        scope: dict = {
            "must": [
                {"classification": {"any": allowed_levels}},
                {"document_type": {"any": sorted(
                    policy.ROLE_DOCUMENT_TYPES[user.role])}},
            ],
            "should": [],
        }
        sellers = self.effective_seller_scope(user, requested_sellers)
        if sellers is None:
            raise ScopeDeniedError(
                "the requested sellers are outside the authorized scope")
        if sellers:
            scope["should"].append({"must": [
                {"seller_id": {"any": sellers}},
            ]})

        organization_branch = {
            "must": [{"seller_id": {"value": policy.ORGANIZATION_WIDE}}],
            "should": [],
        }
        non_restricted = [level for level in allowed_levels
                          if level != "RESTRICTED"]
        if non_restricted:
            organization_branch["should"].append({
                "classification": {"any": non_restricted}})
        if "RESTRICTED" in allowed_levels:
            restricted_branch = {
                "must": [{"classification": {"value": "RESTRICTED"}}],
            }
            if not user.has_organization_wide_scope():
                restricted_branch["must"].append(
                    {"department": {"value": user.department}})
            else:
                restricted_branch["should"] = [
                    {"department": {"any": sorted(policy.DEPARTMENTS)}},
                    {"department": {"missing": True}},
                ]
            organization_branch["should"].append(restricted_branch)
        if organization_branch["should"]:
            scope["should"].append(organization_branch)
        if not scope["should"]:
            raise ScopeDeniedError("identity has no retrievable document scope")
        return scope

    def authorize_chunks(self, user: User, chunks: list,
                         request_id: str) -> tuple[list, list]:
        allowed, denied, events = [], [], []
        user_id = getattr(user, "user_id", "<unknown>")
        for chunk in chunks:
            decision = self.can_access(user, getattr(chunk, "metadata", None))
            events.append((user_id, request_id,
                           "ALLOW" if decision.allow else "DENY",
                           decision.reason,
                           getattr(chunk, "document_id", None),
                           getattr(chunk, "chunk_id", None)))
            (allowed if decision.allow else denied).append(chunk)
        log_decisions(events)
        return allowed, denied
