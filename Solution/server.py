"""Serve the exam application and persist browser-generated results locally.

Adapted from the author's week6/server.py: a static file server plus one local-only endpoint,
POST /api/export {"filename", "content", "encoding"}, which writes CSV/JSON/SVG exports (and base64 PNG) to results/browser/.
Usage:  python3 server.py [--port 8000]   then open http://127.0.0.1:8000/
"""

import argparse
import base64
import json
import os
import re
import tempfile
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / "results" / "browser"
NAME_RE = re.compile(r"^[a-z0-9_\-]+\.(csv|json|svg|png)$")
MAX_BODY_BYTES = 20 * 1024 * 1024


class ExamHandler(SimpleHTTPRequestHandler):
    """Static-file handler with one local-only export endpoint."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if self.path != "/api/export":
            self.send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint")
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > MAX_BODY_BYTES:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Invalid export size")
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            filename, content = payload["filename"], payload["content"]
            if not isinstance(filename, str) or not NAME_RE.match(filename) or not isinstance(content, str):
                raise ValueError("Unsupported export payload")
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError, ValueError) as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        EXPORTS.mkdir(parents=True, exist_ok=True)
        binary = payload.get("encoding") == "base64"
        try:
            data = base64.b64decode(content, validate=True) if binary else content.encode("utf-8")
        except ValueError as error:
            self.send_error(HTTPStatus.BAD_REQUEST, str(error))
            return
        with tempfile.NamedTemporaryFile("wb", dir=EXPORTS, delete=False) as temporary:
            temporary.write(data)
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, EXPORTS / filename)
        response = json.dumps({"saved": f"results/browser/{filename}"}).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def main():
    parser = argparse.ArgumentParser(description="Serve the exam app and save browser exports.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), ExamHandler)
    print(f"Exam app: http://127.0.0.1:{args.port}/index.html")
    print("Browser exports are saved to Solution/results/browser/.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
