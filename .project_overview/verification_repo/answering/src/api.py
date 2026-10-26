"""JSON + SSE HTTP API over the Phase 5 service.

Deliberately stdlib-only: the pipeline lives in `service.py` and is framework
agnostic, so swapping in FastAPI later is a transport change, not a rewrite.

The request body carries `user_id` only. Permissions are always derived
server-side from that identity; any scope-like field a client sends is treated
as a narrowing request and is intersected with the authorized scope.

`POST /query`        buffered JSON answer
`POST /query/stream` Server-Sent Events: meta -> delta* -> done
`GET  /evaluation`        stored evaluation summary
`GET  /evaluation/matrix` stored evaluation matrix and per-case results

The evaluation endpoints are read-only. A run is a paced, quota-heavy batch
job executed from the command line (`python -m evaluation.runner`); serving it
from an HTTP request would let a page load burn the API budget.
"""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import config, service

logger = logging.getLogger(__name__)

MAX_BODY_BYTES = 64 * 1024


def _evaluation():
    """Import Phase 7 lazily so Phase 5 does not depend on it at startup."""
    from evaluation import dataset, reports
    return dataset, reports


class Handler(BaseHTTPRequestHandler):
    server_version = "NexoraKnowledge/6.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _send(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _open_event_stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        # Close-delimited: the body length is unknown up front, and disabling
        # proxy buffering is what makes the delivery genuinely incremental.
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()

    def _emit(self, event: dict) -> None:
        frame = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        self.wfile.write(frame.encode("utf-8"))
        self.wfile.flush()

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        route = parsed.path.rstrip("/")
        report = parse_qs(parsed.query).get("report", [None])[0]
        if route == "/health":
            self._send(200, {"status": "ok"})
        elif route == "/users":
            self._send(200, {"users": service.list_users()})
        elif route == "/evaluation":
            self._evaluation_summary(report)
        elif route == "/evaluation/matrix":
            self._evaluation_matrix(report)
        else:
            self._send(404, {"error": "not_found"})

    def _evaluation_summary(self, report: str | None = None) -> None:
        try:
            dataset, reports = _evaluation()
            document = reports.load(report)
            payload = {
                "dataset": dataset.load().summary(),
                "has_results": document is not None,
                "reports_enabled": config.EVALUATION_REPORTS_ENABLED,
                "report_id": report or "results.json",
                "available_reports": reports.list_reports(),
            }
            if document:
                payload.update(reports.summary(document))
        except ValueError as exc:
            self._send(404, {"error": "report_not_found", "message": str(exc)})
            return
        except Exception:
            logger.exception("evaluation summary failed")
            self._send(500, {"error": "internal_error",
                             "message": "Evaluation data is unavailable."})
            return
        self._send(200, payload)

    def _evaluation_matrix(self, report: str | None = None) -> None:
        if not config.EVALUATION_REPORTS_ENABLED:
            self._send(403, {"error": "reports_disabled", "message":
                            "Detailed reports are disabled. See guide.md for local operator access."})
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
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("body must be a JSON object")
            return payload
        except (ValueError, UnicodeDecodeError):
            self._send(400, {"error": "invalid_request",
                             "message": "Request body must be a JSON object."})
            return None

    def _reject(self, exc: Exception) -> None:
        if isinstance(exc, service.AnsweringError):
            logger.warning("request rejected: %s: %s", type(exc).__name__, exc)
            self._send(exc.status, {"error": type(exc).__name__,
                                    "message": exc.client_message})
        elif isinstance(exc, ValueError):
            logger.warning("bad request: %s", exc)
            self._send(400, {"error": "invalid_request",
                             "message": "The request could not be processed."})
        else:
            logger.exception("unhandled error while answering")
            self._send(500, {"error": "internal_error",
                             "message": "The request could not be completed."})

    def do_POST(self) -> None:
        route = self.path.rstrip("/")
        if route == "/query":
            self._handle_query()
        elif route == "/query/stream":
            self._handle_query_stream()
        elif route == "/evaluation/run":
            # The UI and API never start quota-consuming evaluation jobs.
            self.close_connection = True
            self._send(405, {"error": "method_not_allowed", "message":
                            "Run evaluations explicitly from the CLI; see guide.md."})
        else:
            self._send(404, {"error": "not_found"})

    def _handle_query(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        try:
            response = service.answer_query(
                user_id=payload.get("user_id"),
                query=payload.get("query"),
                conversation_history=payload.get("conversation"),
                seller_filter=payload.get("seller_filter"),
                top_k=payload.get("top_k"),
            )
        except Exception as exc:
            self._reject(exc)
            return
        self._send(200, response.to_dict())

    def _handle_query_stream(self) -> None:
        payload = self._read_json_body()
        if payload is None:
            return
        events = service.answer_query_stream(
            user_id=payload.get("user_id"),
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
            self._send(500, {"error": "internal_error",
                             "message": "The request could not be completed."})
            return
        except Exception as exc:
            self._reject(exc)
            return

        self._open_event_stream()
        try:
            self._emit(first)
            for event in events:
                self._emit(event)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.info("client disconnected during stream")
            events.close()
        except Exception:
            logger.exception("error while streaming")
            try:
                self._emit({"type": "error", "error": "internal_error",
                            "message": "The request could not be completed."})
            except OSError:
                pass
        finally:
            self.close_connection = True


def serve(host: str | None = None, port: int | None = None) -> None:
    host = host or config.API_HOST
    port = port or config.API_PORT
    httpd = ThreadingHTTPServer((host, port), Handler)
    logger.info("Phase 5 API listening on http://%s:%d", host, port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
