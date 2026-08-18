"""Webhook delivery.

These exist because of a live failure: every report the VPS produced was
generated correctly and then silently dropped, because Discord's edge answers
urllib's default `Python-urllib/x.y` User-Agent with a bare 403 before it ever
looks at the payload. The webhook itself was fine — a curl test against the same
URL had succeeded. Only the header differed.

So the header is asserted on the wire rather than by reading the source, and a
refused delivery has to explain itself in the log instead of failing quietly.
"""

from __future__ import annotations

import http.server
import io
import json
import logging
import threading

import pytest

from marketswarm.notify import USER_AGENT, send_webhook


def _serve(handler_cls):
    """One-shot local HTTP server. Returns the URL it is listening on."""
    srv = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
    threading.Thread(target=srv.handle_request, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/hook"


def _capturing_handler(seen, status=204, body=b""):
    class H(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["user_agent"] = self.headers.get("User-Agent")
            seen["content_type"] = self.headers.get("Content-Type")
            length = int(self.headers.get("Content-Length", 0))
            seen["payload"] = json.loads(self.rfile.read(length))
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, *args):  # keep pytest output clean
            pass

    return H


@pytest.fixture
def notify_logs():
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log = logging.getLogger("marketswarm.notify")
    log.addHandler(handler)
    log.setLevel(logging.WARNING)
    yield buf
    log.removeHandler(handler)


def test_delivery_identifies_itself_rather_than_as_urllib():
    seen = {}
    assert send_webhook(_serve(_capturing_handler(seen)), "hello") is True
    assert seen["user_agent"] == USER_AGENT
    # The specific regression: Discord 403s this exact default.
    assert "urllib" not in seen["user_agent"].lower()
    assert seen["user_agent"].startswith("MarketSwarm/")


def test_payload_suits_discord_and_slack_from_one_config():
    seen = {}
    send_webhook(_serve(_capturing_handler(seen)), "the message")
    assert seen["content_type"] == "application/json"
    # `content` is what Discord reads, `text` is what Slack reads.
    assert seen["payload"]["content"] == "the message"
    assert seen["payload"]["text"] == "the message"


def test_a_refusal_reports_why_and_does_not_raise(notify_logs):
    body = b'{"message": "You are being blocked", "code": 40333}'
    url = _serve(_capturing_handler({}, status=403, body=body))

    assert send_webhook(url, "x") is False  # never raises into the run

    logged = notify_logs.getvalue()
    assert "403" in logged
    # Without the response body a 403 is unactionable, which is what made the
    # original failure take a day to spot.
    assert "blocked" in logged


def test_an_unreachable_endpoint_degrades_instead_of_failing_the_run(notify_logs):
    # Port 1 is reserved and refuses immediately — no network access needed.
    assert send_webhook("http://127.0.0.1:1/hook", "x", timeout=2.0) is False
    assert "webhook delivery failed" in notify_logs.getvalue()
