"""Start a fresh local API/UI pair without taking over occupied ports."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / ".cache"


def free_port(preferred: int, *, exclude: set[int] | None = None) -> int:
    for port in range(preferred, min(preferred + 100, 65536)):
        if port in (exclude or set()):
            continue
        with socket.socket() as listener:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                listener.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No available loopback port near {preferred}.")


def start_service(arguments: list[str], env: dict[str, str], log: Path) -> subprocess.Popen:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with log.open("w", encoding="utf-8") as output:
        return subprocess.Popen(
            [sys.executable, *arguments], cwd=ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            creationflags=flags,
        )


def wait_for_service(process: subprocess.Popen, url: str, log: Path, *, timeout: float = 60) -> None:
    # Local startup must not be redirected through a configured external proxy.
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Service exited with code {process.returncode}. See {log}")
        try:
            with client.open(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    raise RuntimeError(f"Startup timed out at {url}. See {log}")


def stop_started(processes: list[subprocess.Popen]) -> None:
    # Only clean up children created by this invocation after a startup failure.
    for process in reversed(processes):
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW, check=False,
                )
            else:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-browser", action="store_true", help="Do not open a browser tab")
    parser.add_argument("--api-port", type=int, default=8000, help="Preferred API port (default: 8000)")
    parser.add_argument("--ui-port", type=int, default=8501, help="Preferred UI port (default: 8501)")
    args = parser.parse_args()
    if not all(1 <= port <= 65535 for port in (args.api_port, args.ui_port)):
        parser.error("Ports must be between 1 and 65535.")
    processes = []
    try:
        if not (ROOT / ".env").is_file():
            raise RuntimeError("Copy .env.example to .env and configure it before starting.")
        CACHE.mkdir(exist_ok=True)
        api_port = free_port(args.api_port)
        ui_port = free_port(args.ui_port, exclude={api_port})
        api_url = f"http://127.0.0.1:{api_port}"
        ui_url = f"http://127.0.0.1:{ui_port}"
        env = {**os.environ, "API_HOST": "127.0.0.1", "API_PORT": str(api_port),
               "API_BASE_URL": api_url, "PYTHONUNBUFFERED": "1"}
        api_log = CACHE / f"local-api-{api_port}.log"
        ui_log = CACHE / f"local-ui-{ui_port}.log"
        print(f"[1/2] Starting API at {api_url}", flush=True)
        api_process = start_service(
            [str(ROOT / "answering/serve.py"), "--host", "127.0.0.1", "--port", str(api_port)],
            env, api_log,
        )
        processes.append(api_process)
        wait_for_service(api_process, api_url + "/live", api_log)
        print(f"[2/2] Starting UI at {ui_url}", flush=True)
        ui_process = start_service(
            ["-m", "streamlit", "run", str(ROOT / "app.py"), "--server.address", "127.0.0.1",
             "--server.port", str(ui_port), "--server.headless", "true",
             "--browser.gatherUsageStats", "false"],
            env, ui_log,
        )
        processes.append(ui_process)
        wait_for_service(ui_process, ui_url + "/_stcore/health", ui_log)
        state = {"project": str(ROOT), "api_port": api_port, "ui_port": ui_port,
                 "api_url": api_url, "ui_url": ui_url,
                 "api_pid": api_process.pid, "ui_pid": ui_process.pid,
                 "api_log": str(api_log), "ui_log": str(ui_log)}
        (CACHE / "launcher.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, KeyboardInterrupt) as exc:
        stop_started(processes)
        print(f"[!] {exc or 'Startup cancelled.'}", file=sys.stderr)
        return 1

    print(f"\nApplication: {ui_url}\nAPI: {api_url}\nLogs and process IDs: {CACHE / 'launcher.json'}")
    print("Both services are running in the background. Existing services were left running.")
    if not args.no_browser:
        try:
            webbrowser.open(ui_url)
        except webbrowser.Error:
            print(f"Open {ui_url} in your browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
