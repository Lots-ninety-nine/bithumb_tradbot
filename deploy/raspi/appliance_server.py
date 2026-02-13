"""Minimal web control panel for Raspberry Pi appliance mode.

Runs an HTTP server that can:
- show tradbot systemd status
- start/stop/restart the bot service
- show recent runtime/journal logs

Designed for Debian/Raspberry Pi OS with systemd.
"""

from __future__ import annotations

from html import escape
import os
import subprocess
from urllib.parse import parse_qs, urlparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOST = os.getenv("WEB_HOST", "0.0.0.0")
PORT = int(os.getenv("WEB_PORT", "8080"))
TRADBOT_SERVICE = os.getenv("TRADBOT_SERVICE", "bithumb-tradbot")
RUNTIME_LOG_FILE = os.getenv("TRADBOT_LOG_FILE", "/opt/bithumb_tradbot/logs/runtime.log")
UI_TOKEN = os.getenv("UI_TOKEN", "").strip()


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
        output = (result.stdout or result.stderr or "").strip()
        return result.returncode, output
    except Exception as exc:  # pragma: no cover
        return 1, f"{exc.__class__.__name__}: {exc}"


def _service_status() -> dict[str, str]:
    _, active = _run(["systemctl", "is-active", TRADBOT_SERVICE])
    _, enabled = _run(["systemctl", "is-enabled", TRADBOT_SERVICE])
    _, since = _run(
        [
            "systemctl",
            "show",
            TRADBOT_SERVICE,
            "--property=ActiveEnterTimestamp",
            "--value",
        ]
    )
    return {
        "active": active or "unknown",
        "enabled": enabled or "unknown",
        "since": since or "-",
    }


def _tail_runtime_log(lines: int = 120) -> str:
    if not os.path.exists(RUNTIME_LOG_FILE):
        return f"runtime log not found: {RUNTIME_LOG_FILE}"
    code, out = _run(["tail", "-n", str(lines), RUNTIME_LOG_FILE])
    if code != 0:
        return f"tail failed: {out}"
    return out


def _tail_journal(lines: int = 80) -> str:
    code, out = _run(["journalctl", "-u", TRADBOT_SERVICE, "-n", str(lines), "--no-pager"])
    if code != 0:
        return f"journalctl failed: {out}"
    return out


def _apply_action(action: str) -> tuple[bool, str]:
    action = action.strip().lower()
    if action not in {"start", "stop", "restart"}:
        return False, f"unsupported action: {action}"
    code, out = _run(["systemctl", action, TRADBOT_SERVICE])
    if code != 0:
        return False, out or f"systemctl {action} failed"
    return True, f"systemctl {action} {TRADBOT_SERVICE}: ok"


class Handler(BaseHTTPRequestHandler):
    server_version = "TradbotAppliance/1.0"

    def _write(self, code: int, body: str, content_type: str = "text/html; charset=utf-8") -> None:
        encoded = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _render_page(self, flash: str = "") -> str:
        status = _service_status()
        runtime_log = escape(_tail_runtime_log(100))
        journal_log = escape(_tail_journal(60))
        token_input = (
            f'<input type="password" name="token" placeholder="UI token" value="" autocomplete="off" />'
            if UI_TOKEN
            else "<em>token disabled</em>"
        )
        flash_html = f"<p><b>{escape(flash)}</b></p>" if flash else ""
        return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <meta http-equiv="refresh" content="20" />
  <title>Tradbot Appliance</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 18px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; }}
    pre {{ white-space: pre-wrap; word-break: break-word; max-height: 45vh; overflow: auto; background: #111; color: #ddd; padding: 10px; border-radius: 8px; }}
    button {{ margin-right: 6px; }}
    .k {{ color: #666; width: 110px; display: inline-block; }}
  </style>
</head>
<body>
  <h2>Tradbot Appliance</h2>
  {flash_html}
  <div class="card">
    <div><span class="k">service</span>{escape(TRADBOT_SERVICE)}</div>
    <div><span class="k">active</span>{escape(status["active"])}</div>
    <div><span class="k">enabled</span>{escape(status["enabled"])}</div>
    <div><span class="k">active since</span>{escape(status["since"])}</div>
    <div><span class="k">log file</span>{escape(RUNTIME_LOG_FILE)}</div>
    <hr />
    <form method="post" action="/action">
      {token_input}
      <button type="submit" name="action" value="start">Start</button>
      <button type="submit" name="action" value="stop">Stop</button>
      <button type="submit" name="action" value="restart">Restart</button>
    </form>
  </div>
  <div class="grid">
    <div class="card">
      <h3>Runtime Log (tail)</h3>
      <pre>{runtime_log}</pre>
    </div>
    <div class="card">
      <h3>Journal (tail)</h3>
      <pre>{journal_log}</pre>
    </div>
  </div>
</body>
</html>"""

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            status = _service_status()
            code, body = 200, (
                '{'
                f'"service":"{TRADBOT_SERVICE}",'
                f'"active":"{status["active"]}",'
                f'"enabled":"{status["enabled"]}",'
                f'"since":"{status["since"]}"'
                "}"
            )
            self._write(code, body, content_type="application/json; charset=utf-8")
            return

        flash = ""
        if parsed.query:
            q = parse_qs(parsed.query)
            flash = (q.get("msg") or [""])[0][:500]
        self._write(200, self._render_page(flash=flash))

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/action":
            self._write(404, "not found", content_type="text/plain; charset=utf-8")
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = self.rfile.read(length).decode("utf-8", errors="replace")
        form = parse_qs(payload)
        token = (form.get("token") or [""])[0].strip()
        action = (form.get("action") or [""])[0].strip()

        if UI_TOKEN and token != UI_TOKEN:
            self._write(403, self._render_page(flash="invalid token"))
            return

        ok, msg = _apply_action(action)
        code = 200 if ok else 500
        self._write(code, self._render_page(flash=msg))

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"[appliance] listening on http://{HOST}:{PORT} service={TRADBOT_SERVICE}")
    server.serve_forever()


if __name__ == "__main__":
    main()
