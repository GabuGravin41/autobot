"""
Tests for autobot/browser/extension_bridge.py — the CDP-free DOM bridge.

Fully offline: no browser, no extension, no backend process. These just
prove the queue/event mechanics (take_next / complete / run / timeout /
liveness) behave correctly in isolation, since that's the one piece that
has to be exactly right for a real extension round trip to work at all.
"""
from __future__ import annotations

import threading
import time

from autobot.browser.extension_bridge import ExtensionBridge


def test_take_next_returns_none_when_empty():
    bridge = ExtensionBridge()
    assert bridge.take_next() is None


def test_run_times_out_when_nobody_answers():
    bridge = ExtensionBridge()
    result = bridge.run("read_text", timeout=0.2)
    assert result["ok"] is False
    assert "Timed out" in result["error"]


def test_extension_connected_false_before_any_poll():
    bridge = ExtensionBridge()
    assert bridge.extension_connected is False


def test_take_next_marks_extension_connected():
    bridge = ExtensionBridge()
    bridge.take_next()  # simulates the extension polling, even with nothing queued
    assert bridge.extension_connected is True


def test_full_round_trip_via_take_next_and_complete():
    """Simulates the real flow: run() queues -> take_next() (the 'extension'
    polling) pops it -> complete() (the 'extension' answering) delivers the
    result back to the blocked run() call."""
    bridge = ExtensionBridge()
    result_holder: dict = {}

    def caller():
        result_holder["result"] = bridge.run("list_elements", timeout=5.0)

    t = threading.Thread(target=caller)
    t.start()

    # Give run() a moment to enqueue, then simulate the extension's poll + reply.
    time.sleep(0.05)
    cmd = bridge.take_next()
    assert cmd is not None
    assert cmd["type"] == "list_elements"

    ok = bridge.complete(cmd["id"], True, data={"elements": [{"index": 1, "tag": "button", "text": "OK"}]})
    assert ok is True

    t.join(timeout=5.0)
    assert result_holder["result"]["ok"] is True
    assert result_holder["result"]["data"]["elements"][0]["text"] == "OK"


def test_complete_with_unknown_id_returns_false():
    bridge = ExtensionBridge()
    assert bridge.complete("nonexistent", True, data={}) is False


def test_run_reports_extension_offline_hint_when_never_polled():
    bridge = ExtensionBridge()
    result = bridge.run("read_text", timeout=0.1)
    assert "not connected" in result["error"]


def test_run_reports_stale_hint_when_extension_polled_but_silent():
    """If the extension IS polling (so it's 'connected') but never answers
    this specific command, the timeout hint should be different — pointing
    at the active tab, not at the extension being uninstalled."""
    bridge = ExtensionBridge()
    bridge.take_next()  # marks the extension as recently polling
    result = bridge.run("read_text", timeout=0.1)
    assert result["ok"] is False
    assert "connected but didn't answer" in result["error"]


if __name__ == "__main__":
    tests = [
        test_take_next_returns_none_when_empty,
        test_run_times_out_when_nobody_answers,
        test_extension_connected_false_before_any_poll,
        test_take_next_marks_extension_connected,
        test_full_round_trip_via_take_next_and_complete,
        test_complete_with_unknown_id_returns_false,
        test_run_reports_extension_offline_hint_when_never_polled,
        test_run_reports_stale_hint_when_extension_polled_but_silent,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {t.__name__} — {e}")
    print(f"\n{passed}/{len(tests)} passed")
