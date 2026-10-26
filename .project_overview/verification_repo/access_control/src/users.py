"""User identities from the Phase 1 organization model (ORG-001)."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import policy
from .policy import IdentityError


@dataclass(frozen=True)
class User:
    user_id: str
    name: str
    department: str
    role: str
    seller_scope: tuple[str, ...] = field(default_factory=tuple)
    clearance_level: str = "INTERNAL"

    def __post_init__(self) -> None:
        problem = policy.identity_problem(
            self.role, self.clearance_level, self.seller_scope, self.department)
        if problem is not None:
            raise IdentityError(
                f"Invalid identity {self.user_id!r}: {problem}")

    def has_organization_wide_scope(self) -> bool:
        return (policy.WILDCARD_SCOPE in self.seller_scope
                and self.role in policy.ORG_WIDE_SCOPE_ROLES)


USERS: dict[str, User] = {
    "U-001": User("U-001", "Aditya Verma", "Business", "Account Manager",
                  ("S001", "S003"), "CONFIDENTIAL"),
    "U-002": User("U-002", "Rahul Sharma", "Support", "Support Engineer",
                  ("S001", "S002"), "CONFIDENTIAL"),
    "U-003": User("U-003", "Vikram Singh", "Engineering", "Software Engineer",
                  ("S001", "S002"), "CONFIDENTIAL"),
    "U-004": User("U-004", "Neha Gupta", "Operations", "System Engineer",
                  ("S002", "S004"), "RESTRICTED"),
    "U-005": User("U-005", "Admin", "IT", "Platform Administrator",
                  ("*",), "RESTRICTED"),
}


def get_user(user_id: str) -> User:
    try:
        return USERS[user_id]
    except (KeyError, TypeError):
        raise IdentityError(f"Unknown user_id: {user_id!r}") from None
