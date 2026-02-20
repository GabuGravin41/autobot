from __future__ import annotations

import time
from typing import Any

from .base import ActionSpec, BaseAdapter


class GoogleDocsWebAdapter(BaseAdapter):
    name = "google_docs_web"
    actions = {
        "open_home": ActionSpec("Open Google Docs home"),
        "open_new_document": ActionSpec("Open a blank Google Doc"),
        "open_document_url": ActionSpec("Open an existing Google Doc URL"),
        "type_text": ActionSpec("Type text in active Google Doc"),
        "copy_all_text": ActionSpec("Copy all text from active Google Doc"),
        "attempt_google_continue_login": ActionSpec("Try Continue with Google login button"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://docs.google.com/document/")
        if self._human_mode():
            wait_s = self._load_wait_seconds("AUTOBOT_GOOGLE_DOCS_LOAD_WAIT", 4.0)
            if wait_s > 0:
                self.logger(f"Waiting {wait_s:.0f}s for Google Docs to load.")
                time.sleep(wait_s)
        return "Opened Google Docs."

    def do_open_new_document(self, _params: dict[str, Any]) -> str:
        self.browser.goto("https://docs.google.com/document/create")
        if self._human_mode():
            wait_s = self._load_wait_seconds("AUTOBOT_GOOGLE_DOCS_LOAD_WAIT", 4.0)
            if wait_s > 0:
                time.sleep(wait_s)
        return "Opened new Google Doc."

    def do_open_document_url(self, params: dict[str, Any]) -> str:
        url = str(params.get("url", "")).strip()
        if not url:
            raise ValueError("Missing required param: url")
        self.browser.goto(url)
        return f"Opened Google Doc URL: {url}"

    def do_type_text(self, params: dict[str, Any]) -> str:
        text = str(params.get("text", ""))
        if not text:
            raise ValueError("Missing required param: text")
        if self._human_mode():
            self.run_human_nav("type_text", {"text": text})
            return "Typed text in Google Doc (human mode)."
        self.browser.page.keyboard.type(text, delay=1)
        return "Typed text in Google Doc."

    def do_copy_all_text(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("copy_all_text")
            return "Copied all text from Google Doc (human mode)."
        self.browser.press("Control+A")
        self.browser.press("Control+C")
        return "Copied all text from Google Doc."

    def do_attempt_google_continue_login(self, _params: dict[str, Any]) -> str:
        clicked = self._click_if_present(self.selector_candidates("login_google_button"))
        if clicked:
            return "Clicked Continue with Google."
        return "Continue with Google button not found. Wait for saved password autofill or sign in manually."
