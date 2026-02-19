from __future__ import annotations

from typing import Any

from .base import ActionSpec, BaseAdapter


class InstagramWebAdapter(BaseAdapter):
    name = "instagram_web"
    actions = {
        "open_home": ActionSpec("Open Instagram home feed"),
        "open_inbox": ActionSpec("Open direct message inbox"),
        "open_chat": ActionSpec("Open DM thread by username"),
        "type_message": ActionSpec("Type DM in active conversation"),
        "send_typed_message": ActionSpec("Send typed DM", requires_confirmation=True),
        "send_message_to_user": ActionSpec("Open DM and send message", requires_confirmation=True),
        "attempt_google_continue_login": ActionSpec("Try Continue with Google login button"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://www.instagram.com")
        return "Opened Instagram."

    def do_open_inbox(self, _params: dict[str, Any]) -> str:
        self.do_open_home({})
        self.browser.goto("https://www.instagram.com/direct/inbox/")
        return "Opened Instagram inbox."

    def do_open_chat(self, params: dict[str, Any]) -> str:
        username = str(params.get("username", "")).strip()
        if not username:
            raise ValueError("Missing required param: username")
        self.do_open_inbox({})
        self.browser.goto(f"https://www.instagram.com/{username}/")
        try:
            self.browser.click("div[role='button']:has-text('Message')", timeout_ms=8000)
        except Exception:  # noqa: BLE001
            self.browser.goto("https://www.instagram.com/direct/new/")
            self.browser.fill("input[name='queryBox']", username, timeout_ms=10000)
            self.browser.press("Enter")
            self.browser.click("div[role='button']:has-text('Chat')", timeout_ms=10000)
        self.state["active_user"] = username
        return f"Opened Instagram chat with {username}."

    def do_type_message(self, params: dict[str, Any]) -> str:
        text = str(params.get("text", "")).strip()
        if not text:
            raise ValueError("Missing required param: text")
        self.browser.fill("textarea[placeholder='Message...'], div[contenteditable='true'][role='textbox']", text)
        return "Instagram message typed."

    def do_send_typed_message(self, _params: dict[str, Any]) -> str:
        clicked = self._click_if_present(["div[role='button']:has-text('Send')", "button:has-text('Send')"], 4000)
        if not clicked:
            self.browser.press("Enter")
        return "Instagram message sent."

    def do_send_message_to_user(self, params: dict[str, Any]) -> str:
        username = str(params.get("username", "")).strip()
        text = str(params.get("text", "")).strip()
        if not username or not text:
            raise ValueError("send_message_to_user requires 'username' and 'text'.")
        self.do_open_chat({"username": username})
        self.do_type_message({"text": text})
        self.do_send_typed_message({})
        return f"Message sent to Instagram user: {username}"

    def do_attempt_google_continue_login(self, _params: dict[str, Any]) -> str:
        clicked = self._click_if_present(
            [
                "button:has-text('Continue with Google')",
                "a:has-text('Continue with Google')",
                "[aria-label*='Google']",
            ]
        )
        if clicked:
            return "Clicked Continue with Google."
        return "Continue with Google button not found. Wait for saved password autofill or sign in manually."
