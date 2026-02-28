from __future__ import annotations

import time
from typing import Any

from .base import ActionSpec, BaseAdapter


class ChatGPTWebAdapter(BaseAdapter):
    name = "chatgpt_web"
    description = "Handles interactions with OpenAI ChatGPT web interface."
    actions = {
        "open_home": ActionSpec("Open ChatGPT home page"),
        "send_message": ActionSpec("Type and send a message to ChatGPT"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://chatgpt.com")
        if self._human_mode():
            wait_s = self._load_wait_seconds("AUTOBOT_CHATGPT_LOAD_WAIT", 5.0)
            if wait_s > 0:
                self.logger(f"Waiting {wait_s:.0f}s for ChatGPT to load.")
                time.sleep(wait_s)
        return "Opened ChatGPT."

    def do_send_message(self, params: dict[str, Any]) -> str:
        message = str(params.get("message", ""))
        if not message:
            raise ValueError("send_message requires 'message' parameter.")
            
        if self._human_mode():
            # In human mode, we type it out
            self._human_type(message)
            time.sleep(0.5)
            self._human_press("enter")
            msg_preview = (message[:50] + "...") if len(message) > 50 else message
            return f"Sent message to ChatGPT in human mode: {msg_preview}"
            
        # Devtools mode: wait for prompt, fill, and press enter
        prompt_selector = self.selector("prompt_textarea")
        self.browser.fill(prompt_selector, message)
        time.sleep(0.5)
        self.browser.press("Enter")
        msg_preview = (message[:50] + "...") if len(message) > 50 else message
        return f"Sent message to ChatGPT: {msg_preview}"

    def do_attempt_google_continue_login(self, _params: dict[str, Any]) -> str:
        # ChatGPT often has a 'Continue with Google' button
        clicked = self._click_if_present(self.selector_candidates("login_google_button"))
        if clicked:
            return "Clicked Continue with Google on ChatGPT login page."
        return "Continue with Google button not found on ChatGPT."
