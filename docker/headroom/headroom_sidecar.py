from __future__ import annotations

import hmac
import http.client
import os
import re
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = os.environ.get("HEADROOM_PROXY_TOKEN", "")
MAX_BODY_BYTES = 64 * 1024 * 1024
RETRIEVE_PATH = re.compile(r"^/v1/retrieve/(?:[a-f0-9]{12}|[a-f0-9]{24})(?:\?.*)?$")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        if self.path != "/v1/compress":
            self.send_error(404)
            return
        self._forward()

    def do_GET(self) -> None:
        if RETRIEVE_PATH.fullmatch(self.path) is None:
            self.send_error(404)
            return
        self._forward()

    def log_message(self, format: str, *args: object) -> None:
        return

    def _forward(self) -> None:
        supplied = self.headers.get("Authorization", "")
        expected = f"Bearer {TOKEN}"
        if not TOKEN or not hmac.compare_digest(supplied, expected):
            self._reply(401, b'{"error":"unauthorized"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self.send_error(400)
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self.send_error(413)
            return
        body = self.rfile.read(length) if length else None
        connection = http.client.HTTPConnection("127.0.0.1", 8787, timeout=300)
        headers = {"Host": "127.0.0.1:8787"}
        content_type = self.headers.get("Content-Type")
        if content_type:
            headers["Content-Type"] = content_type
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read()
            self._reply(response.status, payload, response.getheader("Content-Type") or "application/json")
        except (OSError, http.client.HTTPException):
            self._reply(502, b'{"error":"headroom_unavailable"}', "application/json")
        finally:
            connection.close()

    def _reply(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(payload)


def wait_until_ready(process: subprocess.Popen[bytes]) -> None:
    for _ in range(120):
        if process.poll() is not None:
            raise RuntimeError("Headroom exited during startup")
        connection = http.client.HTTPConnection("127.0.0.1", 8787, timeout=1)
        try:
            connection.request("GET", "/readyz")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(1)
    raise RuntimeError("Headroom readiness timed out")


def main() -> int:
    if not TOKEN:
        raise RuntimeError("HEADROOM_PROXY_TOKEN is required")
    process = subprocess.Popen(
        [
            "headroom",
            "proxy",
            "--host",
            "127.0.0.1",
            "--port",
            "8787",
            "--mode",
            "token",
            "--workers",
            "1",
            "--no-rate-limit",
            "--no-subscription-tracking",
            "--no-telemetry",
        ]
    )

    def stop(_signum: int, _frame: object) -> None:
        process.terminate()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    wait_until_ready(process)
    server = ThreadingHTTPServer(("0.0.0.0", 8788), Handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        process.terminate()
        process.wait(timeout=30)
    return process.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
