from __future__ import annotations

import time
from typing import Any

from .base import ActionSpec, BaseAdapter


class LeetCodeWebAdapter(BaseAdapter):
    name = "leetcode_web"
    description = "Interact with LeetCode: open problems, copy statement, paste code, submit, read result."
    actions = {
        "open_home": ActionSpec("Open LeetCode problems list"),
        "open_problem": ActionSpec("Open a problem by slug or URL (e.g. two-sum)"),
        "copy_problem_statement": ActionSpec("Copy problem description to clipboard"),
        "paste_code": ActionSpec("Paste code into editor (from clipboard or param)"),
        "submit": ActionSpec("Click Submit and run solution"),
        "get_submission_result": ActionSpec("Read pass/fail and status from submission result"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://leetcode.com/problemset/")
        if self._human_mode():
            time.sleep(4)
        return "Opened LeetCode problems list."

    def do_open_problem(self, params: dict[str, Any]) -> str:
        slug = (params.get("slug") or params.get("problem") or "").strip()
        url_param = (params.get("url") or "").strip()
        if url_param:
            self._ensure_url(url_param)
        elif slug:
            # Normalize: two-sum -> two-sum
            if not slug.startswith("https://"):
                slug = slug.strip("/")
                url = f"https://leetcode.com/problems/{slug}/"
            else:
                url = slug
            self._ensure_url(url)
        else:
            raise ValueError("open_problem requires 'slug' or 'url'.")
        if self._human_mode():
            time.sleep(5)
        return "Opened LeetCode problem."

    def do_copy_problem_statement(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("copy_problem_statement")
            return "Copied problem statement (human mode)."
        # Fallback: select common content area and copy
        self.browser.start()
        try:
            self.browser.page.locator("[data-track-load='code_editor']").first.wait_for(state="visible", timeout=5000)
        except Exception:  # noqa: S110
            pass
        self.browser.press("Control+A")
        time.sleep(0.2)
        self.browser.press("Control+C")
        return "Copied problem area to clipboard."

    def do_paste_code(self, params: dict[str, Any]) -> str:
        code = (params.get("code") or "").strip()
        if self._human_mode():
            if code:
                self.run_human_nav("paste_code", {"code": code})
            else:
                self.run_human_nav("paste_code")
            return "Pasted code (human mode)."
        if code:
            self.browser.page.keyboard.press("Control+a")
            time.sleep(0.1)
            self.browser.page.keyboard.type(code, delay=0.01)
        else:
            self.browser.press("Control+V")
        return "Pasted code into editor."

    def do_submit(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("submit")
            return "Clicked Submit (human mode)."
        candidates = [
            "button:has-text('Submit')",
            "[data-e2e-locator='console-submit-button']",
            "button >> text=Submit",
        ]
        clicked = self._click_if_present(candidates, timeout_ms=5000)
        if not clicked:
            self.browser.press("Control+Enter")  # Some editors submit on Ctrl+Enter
        return "Submitted solution."

    def do_get_submission_result(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("get_submission_result")
            return "Read submission result (human mode); check clipboard or state."
        self.browser.start()
        try:
            # Try to read result area text
            loc = self.browser.page.locator("[data-e2e-locator='console-result']").first
            if loc.is_visible(timeout=3000):
                text = loc.inner_text()
                return (text or "").strip() or "Result area empty."
        except Exception:  # noqa: S110
            pass
        return "Submission result not found; check page manually."
