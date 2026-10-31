"""HTTP client for the Phase 5 answering API.

The UI talks to the API only. No retrieval, ranking or permission logic lives
in this process, so the client cannot bypass authorization even if tampered
with: it can ask for a narrower scope, never a wider one.
"""
from __future__ import annotations

import json
import os
from urllib.parse import urlencode

import requests

DEFAULT_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


class ApiError(RuntimeError):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def _url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _message_from(response) -> str:
    try:
        return response.json().get("message", "The request failed.")
    except ValueError:
        return "The request failed."


def _stream_events(response):
    terminal = False
    try:
        response.encoding = "utf-8"
        for raw in response.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            try:
                event = json.loads(raw[5:].strip())
            except ValueError as exc:
                raise ApiError("The answer stream contained malformed data.") from exc
            if not isinstance(event, dict) or event.get("type") not in (
                    "meta", "delta", "done", "error"):
                raise ApiError("The answer stream contained an invalid event.")
            if terminal:
                raise ApiError("The answer stream continued after its terminal event.")
            yield event
            if event["type"] in ("done", "error"):
                terminal = True
                break
    except requests.RequestException as exc:
        raise ApiError("The answer stream disconnected before completion.") from exc
    if not terminal:
        raise ApiError("The answer stream ended before completion.")


def health(base_url: str = DEFAULT_BASE_URL, timeout: int = 5) -> bool:
    try:
        return requests.get(_url(base_url, "/health"),
                            timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def list_users(token: str, base_url: str = DEFAULT_BASE_URL,
               timeout: int = 10) -> list[dict]:
    try:
        response = requests.get(_url(base_url, "/users"),
                                headers={"Authorization": f"Bearer {token}"},
                                timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the assistant service at {base_url}.") from exc
    if response.status_code != 200:
        raise ApiError("The service did not return the user directory.",
                       response.status_code)
    return response.json().get("users", [])


def list_demo_users(base_url: str = DEFAULT_BASE_URL, timeout: int = 10) -> list[dict]:
    try:
        response = requests.get(_url(base_url, "/demo/users"), timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the local demo service at {base_url}.") from exc
    if response.status_code != 200:
        raise ApiError("The local demo identity directory is unavailable.",
                       response.status_code)
    return response.json().get("users", [])


def _body(query: str, conversation, seller_filter, top_k) -> dict:
    payload: dict = {"query": query}
    if conversation:
        payload["conversation"] = conversation
    if seller_filter:
        payload["seller_filter"] = seller_filter
    if top_k:
        payload["top_k"] = top_k
    return payload


def ask(query: str, conversation=None, seller_filter=None, top_k=None,
        token: str | None = None, base_url: str = DEFAULT_BASE_URL,
        timeout: int = 180) -> dict:
    try:
        response = requests.post(
            _url(base_url, "/query"),
            json=_body(query, conversation, seller_filter, top_k),
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=timeout)
    except requests.Timeout as exc:
        raise ApiError("The assistant took too long to respond.") from exc
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the assistant service at {base_url}.") from exc
    if response.status_code != 200:
        raise ApiError(_message_from(response), response.status_code)
    return response.json()


def ask_stream(query: str, conversation=None, seller_filter=None, top_k=None,
               token: str | None = None, base_url: str = DEFAULT_BASE_URL,
               timeout: int = 180):
    """Yield decoded SSE events from /query/stream as the server emits them."""
    try:
        response = requests.post(
            _url(base_url, "/query/stream"),
            json=_body(query, conversation, seller_filter, top_k),
            stream=True,
            headers={"Accept": "text/event-stream",
                     **({"Authorization": f"Bearer {token}"} if token else {})},
            timeout=timeout)
    except requests.Timeout as exc:
        raise ApiError("The assistant took too long to respond.") from exc
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the assistant service at {base_url}.") from exc

    if response.status_code != 200:
        message = _message_from(response)
        response.close()
        raise ApiError(message, response.status_code)

    with response:
        yield from _stream_events(response)


def ask_demo_stream(user_id: str, query: str, conversation=None,
                    seller_filter=None, top_k=None,
                    base_url: str = DEFAULT_BASE_URL, timeout: int = 180):
    """Explicit local demo call; user_id has no meaning on protected routes."""
    try:
        response = requests.post(
            _url(base_url, "/demo/query/stream"),
            json={**_body(query, conversation, seller_filter, top_k),
                  "user_id": user_id}, stream=True,
            headers={"Accept": "text/event-stream"}, timeout=timeout)
    except requests.Timeout as exc:
        raise ApiError("The assistant took too long to respond.") from exc
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the local demo service at {base_url}.") from exc
    if response.status_code != 200:
        message = _message_from(response)
        response.close()
        raise ApiError(message, response.status_code)
    with response:
        yield from _stream_events(response)


def _get(path: str, base_url: str, timeout: int,
         token: str | None = None) -> dict:
    try:
        response = requests.get(
            _url(base_url, path),
            headers={"Authorization": f"Bearer {token}"} if token else {},
            timeout=timeout)
    except requests.RequestException as exc:
        raise ApiError(f"Cannot reach the assistant service at {base_url}."
                       ) from exc
    if response.status_code != 200:
        raise ApiError(_message_from(response), response.status_code)
    return response.json()


def evaluation_summary(base_url: str = DEFAULT_BASE_URL,
                       timeout: int = 15, report: str | None = None) -> dict:
    """Read the dataset and saved run summary without launching a run."""
    query = "?" + urlencode({"report": report}) if report else ""
    return _get("/evaluation" + query, base_url, timeout)


def evaluation_matrix(base_url: str = DEFAULT_BASE_URL,
                     timeout: int = 30, report: str | None = None,
                     token: str | None = None) -> dict:
    query = "?" + urlencode({"report": report}) if report else ""
    return _get("/evaluation/matrix" + query, base_url, timeout, token=token)
