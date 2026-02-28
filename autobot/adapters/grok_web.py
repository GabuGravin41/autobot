from __future__ import annotations

import time
from typing import Any

from .base import ActionSpec, BaseAdapter


class GrokWebAdapter(BaseAdapter):
    name = "grok_web"
    description = "Handles interactions with xAI Grok web interface."
    actions = {
        "open_home": ActionSpec("Open Grok web"),
        "send_message": ActionSpec("Type and send a message to Grok"),
        "ask_latex_from_clipboard": ActionSpec("Paste prompt from clipboard and submit"),
        "copy_visible_response": ActionSpec("Copy visible response text"),
        "attempt_google_continue_login": ActionSpec("Try Continue with Google login button"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://grok.com")
        if self._human_mode():
            wait_s = self._load_wait_seconds("AUTOBOT_GROK_LOAD_WAIT", 4.0)
            if wait_s > 0:
                self.logger(f"Waiting {wait_s:.0f}s for Grok to load.")
                time.sleep(wait_s)
        return "Opened Grok."
    def do_send_message(self, params: dict[str, Any]) -> str:
        message = str(params.get("message", ""))
        if not message:
            raise ValueError("send_message requires 'message' parameter.")
        if self._human_mode():
            self._human_type(message)
            time.sleep(0.5)
            self._human_press("enter")
            return f"Sent message to Grok in human mode: {message[:50]}..."
        # Devtools: type and enter
        self.browser.page.keyboard.type(message, delay=0.02)
        self.browser.press("Enter")
        return f"Sent message to Grok: {message[:50]}..."
    def do_ask_latex_from_clipboard(self, params: dict[str, Any]) -> str:
        instruction = str(
            params.get(
                "instruction",
                "Convert the pasted document into accurate LaTeX for Overleaf and return only latex source.",
            )
        )
        if self._human_mode():
            self.run_human_nav("ask_latex_from_clipboard", {"instruction": instruction})
            return "Submitted LaTeX request in Grok (human mode)."
        # Devtools: type and enter
        self.browser.page.keyboard.type(instruction, delay=0.02)
        self.browser.press("Enter")
        msg_preview = (instruction[:50] + "...") if len(instruction) > 50 else instruction
        return f"Sent message to Grok: {msg_preview}"

    def do_copy_visible_response(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("copy_visible_response")
            return "Copied visible response in Grok (human mode)."
        self.browser.press("Control+A")
        self.browser.press("Control+C")
        return "Copied visible response in Grok."

    def do_attempt_google_continue_login(self, _params: dict[str, Any]) -> str:
        clicked = self._click_if_present(self.selector_candidates("login_google_button"))
        if clicked:
            return "Clicked Continue with Google."
        return "Continue with Google button not found. Wait for saved password autofill or sign in manually."
