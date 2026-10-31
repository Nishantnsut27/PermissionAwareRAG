"""OIDC introspection boundary and trusted identity mapping.

The API never treats a decoded token or a caller supplied user ID as identity.
An OIDC provider must introspect the bearer token, and its verified subject is
mapped to a server-owned user record before permissions are evaluated.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import requests

from access_control.src.users import User, get_user
from .deadline import remaining_timeout
from .http_deadline import provider_deadline, session

HTTP = session()


class AuthenticationError(RuntimeError):
    status = 401


class IdentityProviderUnavailable(RuntimeError):
    status = 503


@dataclass(frozen=True)
class Principal:
    user: User
    subject: str

    @property
    def is_operator(self) -> bool:
        return self.user.role in {"Platform Administrator", "Security Administrator"}


def _settings():
    return {
        "url": os.getenv("OIDC_INTROSPECTION_URL", "").strip(),
        "client_id": os.getenv("OIDC_CLIENT_ID", "").strip(),
        "client_secret": os.getenv("OIDC_CLIENT_SECRET", ""),
        "issuer": os.getenv("OIDC_ISSUER", "").rstrip("/"),
        "audience": os.getenv("OIDC_AUDIENCE", "").strip(),
        "subject_map": os.getenv("OIDC_SUBJECT_MAP", "").strip(),
    }


def configured() -> bool:
    settings = _settings()
    return (all(settings.values())
            and settings["url"].lower().startswith("https://"))


def authenticate(authorization: str | None, *, post=None, now=None) -> Principal:
    settings = _settings()
    if not configured():
        raise IdentityProviderUnavailable("OIDC identity provider is not configured.")
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthenticationError("Bearer authentication is required.")
    token = authorization[7:].strip()
    if not token or len(token) > 16384:
        raise AuthenticationError("Invalid bearer credential.")
    try:
        with provider_deadline(5):
            response = (post or HTTP.post)(
                settings["url"], data={"token": token},
                auth=(settings["client_id"], settings["client_secret"]),
                headers={"Accept": "application/json"}, timeout=remaining_timeout(5))
            try:
                response.raise_for_status()
                claims = response.json()
            finally:
                close = getattr(response, "close", None)
                if close:
                    close()
        mapping = json.loads(settings["subject_map"])
    except (requests.RequestException, ValueError, TypeError) as exc:
        raise IdentityProviderUnavailable("Identity provider validation failed.") from exc
    if not isinstance(claims, dict) or claims.get("active") is not True:
        raise AuthenticationError("Credential is invalid or expired.")
    current = int(time.time() if now is None else now)
    try:
        exp = int(claims["exp"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AuthenticationError("Credential has no valid expiry.") from exc
    if exp <= current:
        raise AuthenticationError("Credential is expired.")
    issuer = claims.get("iss")
    if not isinstance(issuer, str) or issuer.rstrip("/") != settings["issuer"]:
        raise AuthenticationError("Credential issuer is not trusted.")
    audience = claims.get("aud")
    audiences = audience if isinstance(audience, list) else [audience]
    if settings["audience"] not in audiences:
        raise AuthenticationError("Credential audience is not accepted.")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject:
        raise AuthenticationError("Credential has no subject.")
    if not isinstance(mapping, dict) or not isinstance(mapping.get(subject), str):
        raise AuthenticationError("Credential subject is not provisioned.")
    try:
        user = get_user(mapping[subject])
    except Exception as exc:
        raise AuthenticationError("Credential subject is not provisioned.") from exc
    return Principal(user=user, subject=subject)
