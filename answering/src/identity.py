"""Server-resolved request identity."""
from __future__ import annotations

from dataclasses import dataclass

from access_control.src.users import User


@dataclass(frozen=True)
class Principal:
    user: User
    subject: str

    @property
    def is_operator(self) -> bool:
        return self.user.role in {"Platform Administrator", "Security Administrator"}
