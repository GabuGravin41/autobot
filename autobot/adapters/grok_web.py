from __future__ import annotations

from typing import Any

from .base import ActionSpec, BaseAdapter


class GrokWebAdapter(BaseAdapter):
    name = "grok_web"
    actions = {
        "open_home": ActionSpec("Open Grok web"),
        "ask_latex_from_clipboard": ActionSpec("Paste prompt from clipboard and submit"),
        "copy_visible_response": ActionSpec("Copy visible response text"),
        "attempt_google_continue_login": ActionSpec("Try Continue with Google login button"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://grok.com")
        return "Opened Grok."

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
        # Devtools fallback: just type instruction
        self.browser.page.keyboard.type(instruction, delay=1)
        self.browser.press("Enter")
        return "Submitted LaTeX request in Grok."

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
