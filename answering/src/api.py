"""JSON + SSE HTTP API over the Phase 5 service.

Deliberately stdlib-only: the pipeline lives in `service.py` and is framework
agnostic, so swapping in FastAPI later is a transport change, not a rewrite.

Protected routes authenticate through configured OIDC token introspection and
derive the user from a server-owned subject map. Separate `/demo/*` routes can
accept a caller-selected `user_id` only when explicit local demo mode is on;
they are always loopback-only and are not authentication.

`POST /query`        buffered JSON answer
`POST /query/stream` Server-Sent Events: meta -> delta* -> done
`GET  /evaluation`        stored evaluation summary
`GET  /evaluation/matrix` stored evaluation matrix and per-case results
`/demo/*`             loopback-only selectable-identity demo

The evaluation endpoints are read-only. A run is a paced, quota-heavy batch
job executed from the command line (`python -m evaluation.runner`); serving it
from an HTTP request would let a page load burn the API budget.
"""
from __future__ import annotations

import json
import ipaddress
import logging
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import config, identity, service
from .deadline import (DeadlineExceeded, SocketDeadline, check_deadline,
                       remaining_timeout, request_deadline)

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 * 1024
_budget_lock = threading.Lock()
_principal_budgets: dict[str, deque[list[float | int]]] = {}
MAX_BUDGET_PRINCIPALS = 10000


def _reserved_tokens(payload: dict) -> int:
    query = payload.get("query") if isinstance(payload.get("query"), str) else ""
    conversation = payload.get("conversation")
    history_chars = (sum(len(row.get("content", "")) for row in conversation
                         if isinstance(row, dict)
                         and isinstance(row.get("content"), str))
                     if isinstance(conversation, list) else 0)
    input_chars = min(len(query), config.DEFAULT_ANSWERING.max_query_chars)
    input_chars += min(history_chars, config.DEFAULT_ANSWERING.max_history_chars)
    # Context is already bounded in tokens by the context builder. Reserve it
    # in full, plus both system prompts, bounded user/history text, framing,
    # and the maximum output of the answer and intent calls.
    answer_prompt = (config.DEFAULT_ANSWERING.max_context_tokens
                     + (len(service.prompts.SYSTEM_PROMPT) + input_chars + 2048 + 2) // 3)
    intent_prompt = (len(service.answer_scope._INTENT_SYSTEM) + input_chars + 512 + 2) // 3
    return answer_prompt + intent_prompt + config.GROQ_MAX_TOKENS + 256


def _charge_budget(principal, payload: dict):
    # subject is issued/verified by the configured identity provider; the JSON
    # user_id is deliberately not consulted here.
    key = principal.subject
    tokens = _reserved_tokens(payload)
    now = time.monotonic()
    with _budget_lock:
        if key not in _principal_budgets and len(_principal_budgets) >= MAX_BUDGET_PRINCIPALS:
            oldest = next(iter(_principal_budgets), None)
            if oldest is not None:
                del _principal_budgets[oldest]
        budget = _principal_budgets.setdefault(key, deque())
        while budget and budget[0][0] <= now - 60:
            budget.popleft()
        if (len(budget) >= config.API_REQUESTS_PER_MINUTE
                or sum(item[1] for item in budget) + tokens > config.API_TOKENS_PER_MINUTE):
            return False
        reservation = [now, tokens]
        budget.append(reservation)
        return reservation


def _reconcile_budget(principal, reservation, usage: dict | None) -> None:
    if not reservation or not isinstance(usage, dict) or not usage.get("reported"):
        return
    actual = usage.get("total_tokens")
    if isinstance(actual, bool) or not isinstance(actual, int) or actual < 0:
        return
    key = principal.subject
    with _budget_lock:
        if any(item is reservation for item in _principal_budgets.get(key, ())):
            reservation[1] = actual


def _validate_payload(payload: dict, demo: bool) -> bool:
    allowed = {"query", "conversation", "seller_filter", "top_k", "user_id"}
    if set(payload) - allowed:
        return False
    if demo and "user_id" in payload and not isinstance(payload["user_id"], str):
        return False
    return True


class BoundedHTTPServer(ThreadingHTTPServer):
    """Bound concurrent handlers and return an explicit overload response."""
    daemon_threads = True

    def __init__(self, address, handler):
        self._admission = threading.BoundedSemaphore(config.API_MAX_CONCURRENT_REQUESTS)
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        if not self._admission.acquire(blocking=False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\n"
                                b"Connection: close\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self._admission.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._admission.release()


def _evaluation():
    """Import Phase 7 lazily so Phase 5 does not depend on it at startup."""
    from evaluation import dataset, reports
    return dataset, reports


class Handler(BaseHTTPRequestHandler):
    server_version = "NexoraKnowledge/6.0"
    protocol_version = "HTTP/1.1"

    def handle_one_request(self):
        # This starts before BaseHTTPRequestHandler reads even the request
        # line. Shutdown interrupts buffered reads and writes, including peers
        # that keep sending bytes below an inactivity timeout.
        with request_deadline(config.API_REQUEST_TIMEOUT_S):
            total_guard = SocketDeadline(config.API_REQUEST_TIMEOUT_S)
            total_guard.register(self.connection)
            self._header_guard = SocketDeadline(
                min(config.API_HEADER_TIMEOUT_S, config.API_REQUEST_TIMEOUT_S),
                read_only=True)
            self._header_guard.register(self.connection)
            self.connection.settimeout(remaining_timeout(config.API_HEADER_TIMEOUT_S))
            try:
                super().handle_one_request()
            except (OSError, DeadlineExceeded):
                self.close_connection = True
            finally:
                self._header_guard.cancel()
                total_guard.cancel()

    def parse_request(self):
        parsed = super().parse_request()
        self._header_guard.cancel()
        if self._header_guard.expired:
            self.close_connection = True
            return False
        check_deadline()
        self.connection.settimeout(remaining_timeout(config.API_REQUEST_TIMEOUT_S))
        return parsed

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _open_event_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # Close-delimited: the body length is unknown up front, and disabling
        # proxy buffering is what makes the delivery genuinely incremental.
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

    def _emit(self, event: dict) -> None:
        frame = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    def do_OPTIONS(self) -> None:
        self._send(405, {"error": "method_not_allowed",
                         "message": "Cross-origin API access is not enabled."})

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        route = parsed.path.rstrip("/")
        report = parse_qs(parsed.query).get("report", [None])[0]
        if route == "/health":
            self._send(200, {"status": "ok"})
        elif route == "/live":
            self._send(200, {"status": "live"})
        elif route == "/ready":
            self._readiness()
        elif route == "/demo/users":
            if self._demo_request():
                self._send(200, {"users": service.list_users()})
        elif route == "/users":
            principal = self._authenticate()
            if principal is not None:
                self._send(200, {"users": service.list_users()})
        elif route == "/evaluation":
            self._evaluation_summary(report)
        elif route == "/evaluation/matrix":
            self._evaluation_matrix(report)
        else:
            self._send(404, {"error": "not_found"})

    def _evaluation_summary(self, report: str | None = None) -> None:
        try:
            _, reports = _evaluation()
            document = reports.load(report)
            payload = {
                "has_results": document is not None,
                "reports_enabled": config.EVALUATION_REPORTS_ENABLED,
                "available_reports": [
                    {"report_id": row["report_id"],
                     "generated_at": row.get("generated_at")}
                    for row in reports.list_reports()],
            }
            if document:
                summary = document.get("summary") or {}
                payload.update({"report_id": report or "results.json",
                                "run_id": document.get("run_id"),
                                "generated_at": document.get("generated_at"),
                                "total_cases": summary.get("total_cases", 0)})
        except ValueError as exc:
            self._send(404, {"error": "report_not_found", "message": str(exc)})
            return
        except Exception:
            logger.exception("evaluation summary failed")
            self._send(500, {"error": "internal_error",
                             "message": "Evaluation data is unavailable."})
            return
        self._send(200, payload)

    def _readiness(self) -> None:
        problems = []
        try:
            config.validate_settings()
        except RuntimeError as exc:
            problems.append(str(exc))
        try:
            config.validate_env()
        except RuntimeError as exc:
            problems.append(str(exc))
        if not (identity.configured() or config.API_DEMO_MODE):
            problems.append("no identity provider or explicit local demo mode")
        try:
            from indexing.src import config as index_config
            from indexing.src.retrieval import active_revision
            collection, _ = active_revision()
            pointer = json.loads((index_config.REVISIONS_DIR / "active.json")
                                 .read_text(encoding="utf-8"))
            if pointer.get("collection_name") != collection:
                problems.append("local active pointer differs from the Qdrant alias")
        except Exception:
            problems.append("no valid local active index manifest")
        if problems:
            self._send(503, {"status": "not_ready", "problems": problems})
        else:
            self._send(200, {"status": "ready"})

    def _evaluation_matrix(self, report: str | None = None) -> None:
        if not config.EVALUATION_REPORTS_ENABLED:
            self._send(403, {"error": "reports_disabled", "message":
                            "Detailed reports are disabled. See guide.md for local operator access."})
            return
        principal = self._authenticate()
        if principal is None:
            return
        if not principal.is_operator:
            self._send(403, {"error": "forbidden",
                             "message": "Operator permission is required."})
            return
        try:
            _, reports = _evaluation()
            document = reports.load(report)
        except ValueError as exc:
            self._send(404, {"error": "report_not_found", "message": str(exc)})
            return
        except Exception:
            logger.exception("evaluation matrix failed")
            self._send(500, {"error": "internal_error",
                             "message": "Evaluation data is unavailable."})
            return
        if document is None:
            self._send(404, {"error": "no_results",
                             "message": "No evaluation has been run yet."})
            return
        self._send(200, {**reports.summary(document),
                         "report_id": report or "results.json",
                         "matrix": document.get("matrix", []),
                         "results": document.get("results", [])})

    def _read_json_body(self) -> dict | None:
        self.connection.settimeout(remaining_timeout(config.API_BODY_TIMEOUT_S))
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._send(400, {"error": "invalid_request",
                             "message": "Invalid Content-Length."})
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            self._send(400, {"error": "invalid_request",
                             "message": "Request body is missing or too large."})
            return None
        guard = SocketDeadline(remaining_timeout(config.API_BODY_TIMEOUT_S), read_only=True)
        guard.register(self.connection)
        try:
            raw = self.rfile.read(length)
            if guard.expired:
                raise socket.timeout("absolute body deadline exceeded")
            check_deadline()
            if len(raw) != length:
                raise ValueError("incomplete body")
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
            return payload
        except (socket.timeout, OSError):
            self._send(408, {"error": "request_timeout",
                             "message": "Request body read deadline exceeded."})
            return None
        except (ValueError, UnicodeDecodeError):
            self._send(400, {"error": "invalid_request",
                             "message": "Request body must be a JSON object."})
            return None
        finally:
            guard.cancel()
            self.connection.settimeout(remaining_timeout(config.API_REQUEST_TIMEOUT_S))

    def _reject(self, exc: Exception) -> None:
        if isinstance(exc, DeadlineExceeded):
            self._send(504, {"outcome": "error", "completed": False,
                             "answered": False, "error": "request_timeout",
                             "message": "The request deadline was exceeded."})
        elif isinstance(exc, service.AnsweringError):
            logger.warning("request rejected: %s: %s", type(exc).__name__, exc)
            self._send(exc.status, {"outcome": "error", "completed": False,
                                    "answered": False,
                                    "error": type(exc).__name__,
                                    "message": exc.client_message})
        elif isinstance(exc, ValueError):
            logger.warning("bad request: %s", exc)
            self._send(400, {"outcome": "error", "completed": False,
                             "answered": False, "error": "invalid_request",
                             "message": "The request could not be processed."})
        else:
            logger.exception("unhandled error while answering")
            self._send(500, {"outcome": "error", "completed": False,
                             "answered": False, "error": "internal_error",
                             "message": "The request could not be completed."})

    def _authenticate(self):
        try:
            return identity.authenticate(self.headers.get("Authorization"))
        except identity.AuthenticationError as exc:
            self._discard_rejected_body()
            self._send(401, {"error": "unauthorized", "message": str(exc)})
        except identity.IdentityProviderUnavailable:
            self._discard_rejected_body()
            self._send(503, {"error": "identity_provider_unavailable",
                             "message": "Authenticated identity is not configured."})
        return None

    def _discard_rejected_body(self) -> None:
        if self.command != "POST":
            return
        self.connection.settimeout(remaining_timeout(config.API_BODY_TIMEOUT_S))
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return
        if 0 < length <= MAX_BODY_BYTES:
            guard = SocketDeadline(remaining_timeout(config.API_BODY_TIMEOUT_S), read_only=True)
            guard.register(self.connection)
            try:
                self.rfile.read(length)
            except (socket.timeout, OSError):
                return
            finally:
                guard.cancel()
                self.connection.settimeout(remaining_timeout(config.API_REQUEST_TIMEOUT_S))

    def _demo_request(self) -> bool:
        if not config.API_DEMO_MODE:
            self._send(404, {"error": "not_found"})
            return False
        try:
            if not ipaddress.ip_address(self.client_address[0]).is_loopback:
                raise ValueError("non-loopback peer")
        except ValueError:
            self._send(403, {"error": "demo_local_only",
                             "message": "Demo routes are available locally only."})
            return False
        return True

    def do_POST(self) -> None:
        route = self.path.rstrip("/")
        demo = route in {"/demo/query", "/demo/query/stream"}
        if route in {"/query", "/demo/query"}:
            principal = None if demo else self._authenticate()
            if (not demo and principal is None) or (demo and not self._demo_request()):
                return
            self._handle_query(principal, demo)
        elif route in {"/query/stream", "/demo/query/stream"}:
            principal = None if demo else self._authenticate()
            if (not demo and principal is None) or (demo and not self._demo_request()):
                return
            self._handle_query_stream(principal, demo)
        elif route == "/evaluation/run":
            # The UI and API never start quota-consuming evaluation jobs.
            self.close_connection = True
            self._send(405, {"error": "method_not_allowed", "message":
                            "Run evaluations explicitly from the CLI; see guide.md."})
        else:
            self._send(404, {"error": "not_found"})

    def _handle_query(self, principal=None, demo: bool = False) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        if not _validate_payload(payload, demo):
            self._send(400, {"error": "invalid_request",
                             "message": "Request fields do not match the supported schema."})
            return
        try:
            budget_principal = (identity.Principal(
                service.get_demo_user(payload.get("user_id")),
                f"demo:{payload.get('user_id')}") if demo else principal)
        except Exception as exc:
            self._reject(exc)
            return
        reservation = _charge_budget(budget_principal, payload)
        if not reservation:
            self._send(429, {"error": "rate_limited", "message":
                             "The per-principal request or token budget is exhausted."})
            return
        try:
            with request_deadline(config.API_REQUEST_TIMEOUT_S):
                response = service.answer_query(
                    principal=budget_principal,
                    query=payload.get("query"),
                    conversation_history=payload.get("conversation"),
                    seller_filter=payload.get("seller_filter"),
                    top_k=payload.get("top_k"),
                )
        except Exception as exc:
            self._reject(exc)
            return
        _reconcile_budget(budget_principal, reservation,
                          response.security_trace.get(
                              "model_usage", {"reported": True, "total_tokens": 0}))
        self._send(200, response.to_dict())

    def _handle_query_stream(self, principal=None, demo: bool = False) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        if not _validate_payload(payload, demo):
            self._send(400, {"error": "invalid_request",
                             "message": "Request fields do not match the supported schema."})
            return
        try:
            resolved = (service.get_demo_user(payload.get("user_id"))
                        if demo else None)
            service_principal = (identity.Principal(
                resolved, f"demo:{payload.get('user_id')}") if demo else principal)
        except Exception as exc:
            self._reject(exc)
            return
        reservation = _charge_budget(service_principal, payload)
        if not reservation:
            self._send(429, {"error": "rate_limited", "message":
                             "The per-principal request or token budget is exhausted."})
            return
        deadline_context = request_deadline(config.API_REQUEST_TIMEOUT_S)
        deadline_context.__enter__()
        events = service.answer_query_stream(
            principal=service_principal,
            query=payload.get("query"),
            conversation_history=payload.get("conversation"),
            seller_filter=payload.get("seller_filter"),
            top_k=payload.get("top_k"),
        )
        # Pull the first event before committing to 200: validation errors are
        # raised on first advance and still deserve a proper status code.
        try:
            first = next(events)
        except StopIteration:
            deadline_context.__exit__(None, None, None)
            self._send(500, {"error": "internal_error",
                             "message": "The request could not be completed."})
            return
        except Exception as exc:
            deadline_context.__exit__(type(exc), exc, exc.__traceback__)
            self._reject(exc)
            return

        self._open_event_stream()
        try:
            if first.get("type") == "done":
                _reconcile_budget(service_principal, reservation,
                                  first.pop("_token_usage",
                                            {"reported": True, "total_tokens": 0}))
            self._emit(first)
            for event in events:
                if event.get("type") == "done":
                    _reconcile_budget(service_principal, reservation,
                                      event.pop("_token_usage",
                                                {"reported": True, "total_tokens": 0}))
                self._emit(event)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.info("client disconnected during stream")
            events.close()
        except Exception:
            logger.exception("error while streaming")
            try:
                self._emit({"type": "error", "outcome": "error",
                            "completed": False, "answered": False,
                            "error": "internal_error",
                            "message": "The request could not be completed."})
            except OSError:
                pass
        finally:
            deadline_context.__exit__(None, None, None)
            self.close_connection = True


def serve(host: str | None = None, port: int | None = None) -> None:
    config.validate_settings()
    host = host or config.API_HOST
    port = port or config.API_PORT
    # Demo routes remain loopback only. With no provider, all routes remain
    # loopback only and protected endpoints fail closed.
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host.lower() == "localhost"
    if config.API_DEMO_MODE and not loopback:
        raise RuntimeError("Demo mode may bind only to a loopback address.")
    if not identity.configured() and not loopback:
        raise RuntimeError(
            "No OIDC identity provider is configured; API must bind to loopback.")
    httpd = BoundedHTTPServer((host, port), Handler)
    logger.info("Phase 5 API listening on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
