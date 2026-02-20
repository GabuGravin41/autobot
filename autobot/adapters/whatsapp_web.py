from __future__ import annotations

import time
from typing import Any

from .base import ActionSpec, BaseAdapter


def _normalize_phone(phone: str) -> str:
    """Digits only, for WhatsApp URL and search. Removes +, spaces, dashes, parens."""
    return "".join(c for c in str(phone).strip() if c.isdigit())


class WhatsAppWebAdapter(BaseAdapter):
    name = "whatsapp_web"
    actions = {
        "open_home": ActionSpec("Open WhatsApp Web home"),
        "open_chat": ActionSpec("Open chat by display name"),
        "type_message": ActionSpec("Type a message in current chat input"),
        "send_typed_message": ActionSpec("Send currently typed message", requires_confirmation=True),
        "send_message_to_chat": ActionSpec("Open chat and send message", requires_confirmation=True),
        "attach_file": ActionSpec("Attach file from path (open file picker then type path)"),
        "read_recent_messages": ActionSpec("Read recent visible messages from active chat"),
        "list_visible_chats": ActionSpec("List visible chat names"),
        "attempt_google_continue_login": ActionSpec("Try Continue with Google login button"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://web.whatsapp.com")
        if self._human_mode():
            wait_s = self._load_wait_seconds("AUTOBOT_WHATSAPP_LOAD_WAIT", 8.0)
            if wait_s > 0:
                self.logger(f"Waiting {wait_s:.0f}s for WhatsApp to load chats.")
                time.sleep(wait_s)
            return "Opened WhatsApp Web in human profile mode."
        return "Opened WhatsApp Web."

    def do_open_chat(self, params: dict[str, Any]) -> str:
        chat = str(params.get("chat", "")).strip()
        phone = str(params.get("phone", "")).strip()
        use_search = bool(params.get("use_search", False))  # True = search by number instead of send URL
        if not chat and not phone:
            raise ValueError("Missing required param: chat or phone")
        if self._human_mode():
            if phone:
                digits = _normalize_phone(phone)
                if not digits:
                    raise ValueError("Phone number has no digits.")
                # Ensure Chrome is focused so the new tab is visible and active
                focus_result = self.focus.ensure_keywords_focused(("chrome", "whatsapp"))
                if not focus_result.ok:
                    self.logger(f"Focus: {focus_result.reason}. Click the Chrome window before continuing.")
                # Open home first so WhatsApp Web is loaded, then go to chat
                self.browser.goto("https://web.whatsapp.com")
                wait_s = self._load_wait_seconds("AUTOBOT_WHATSAPP_LOAD_WAIT", 8.0)
                if wait_s > 0:
                    self.logger(f"Waiting {wait_s:.0f}s for WhatsApp to load.")
                    time.sleep(wait_s)
                if use_search:
                    self.run_human_nav("open_chat_by_phone", {"phone": digits})
                else:
                    self.browser.goto(f"https://web.whatsapp.com/send?phone={digits}")
                chat_wait = self._load_wait_seconds("AUTOBOT_WHATSAPP_CHAT_LOAD_WAIT", 5.0)
                if chat_wait > 0:
                    self.logger(f"Waiting {chat_wait:.0f}s for chat to be ready.")
                    time.sleep(chat_wait)
                self.state["active_chat"] = phone
                return f"Opened WhatsApp chat by phone in human mode: {phone}"
            focus_result = self.focus.ensure_keywords_focused(("chrome", "whatsapp"))
            if not focus_result.ok:
                self.logger(f"Focus: {focus_result.reason}. Click the Chrome window before continuing.")
            self.do_open_home({})
            chat_wait = self._load_wait_seconds("AUTOBOT_WHATSAPP_CHAT_LOAD_WAIT", 5.0)
            if chat_wait > 0:
                time.sleep(chat_wait)
            self.run_human_nav("open_chat_by_name", {"chat": chat})
            chat_wait_after = self._load_wait_seconds("AUTOBOT_WHATSAPP_CHAT_LOAD_WAIT", 5.0)
            if chat_wait_after > 0:
                self.logger(f"Waiting {chat_wait_after:.0f}s for chat to be ready.")
                time.sleep(chat_wait_after)
            self.state["active_chat"] = chat
            return f"Attempted chat open in human mode: {chat}"
        if not chat:
            raise ValueError("Missing required param: chat")
        self.do_open_home({})
        self.fill_any("chat_search_input", chat, timeout_ms=15000)
        self.browser.press("Enter")
        self.state["active_chat"] = chat
        return f"Opened chat: {chat}"

    def do_type_message(self, params: dict[str, Any]) -> str:
        text = str(params.get("text", "")).strip()
        if not text:
            raise ValueError("Missing required param: text")
        if self._human_mode():
            # Short wait so chat input is ready and focused
            time.sleep(0.8)
            self.run_human_nav("type_message", {"text": text})
            return "Message typed in human profile mode."
        self.fill_any("message_input", text, timeout_ms=15000)
        return "Message typed."

    def do_send_typed_message(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            time.sleep(0.3)
            self.run_human_nav("send_message")
            return "Typed message sent in human profile mode."
        self.browser.press("Enter")
        return "Typed message sent."

    def do_send_message_to_chat(self, params: dict[str, Any]) -> str:
        chat = str(params.get("chat", "")).strip()
        text = str(params.get("text", "")).strip()
        if not chat or not text:
            raise ValueError("send_message_to_chat requires 'chat' and 'text'.")
        self.do_open_chat({"chat": chat})
        self.do_type_message({"text": text})
        self.do_send_typed_message({})
        return f"Message sent to chat: {chat}"

    def do_attach_file(self, params: dict[str, Any]) -> str:
        path = str(params.get("path", params.get("file_path", ""))).strip()
        if not path:
            raise ValueError("attach_file requires 'path' or 'file_path'.")
        if self._human_mode():
            time.sleep(0.5)
            self.run_human_nav("attach_file", {"path": path.replace("\\", "/")})
            return f"Attempted to attach file in human mode: {path}"
        self.browser.start()
        try:
            file_input = self.browser.page.locator('input[type="file"]').first
            file_input.set_input_files(path)
            return f"Attached file: {path}"
        except Exception:
            self.logger("DevTools file input not found; falling back to human nav.")
            time.sleep(0.5)
            self.run_human_nav("attach_file", {"path": path.replace("\\", "/")})
            return f"Attempted to attach file via human nav: {path}"

    def do_read_recent_messages(self, params: dict[str, Any]) -> list[str]:
        if self._human_mode():
            raise RuntimeError("Reading DOM messages is unavailable in human profile mode.")
        limit = int(params.get("limit", 5))
        self.browser.start()
        nodes = self.browser.page.locator(self.selector("visible_message_text")).all_inner_texts()
        return [item.strip() for item in nodes if item.strip()][-limit:]

    def do_list_visible_chats(self, params: dict[str, Any]) -> list[str]:
        if self._human_mode():
            raise RuntimeError("Listing visible chats is unavailable in human profile mode.")
        limit = int(params.get("limit", 20))
        self.browser.start()
        names = self.browser.page.locator(self.selector("visible_chat_name")).all_inner_texts()
        cleaned = [name.strip() for name in names if name.strip()]
        deduped = list(dict.fromkeys(cleaned))
        return deduped[:limit]

    def do_attempt_google_continue_login(self, _params: dict[str, Any]) -> str:
        self.browser.start()
        clicked = self._click_if_present(self.selector_candidates("login_google_button"))
        if clicked:
            return "Clicked Continue with Google."
        return "Continue with Google button not found. Wait for saved password autofill or sign in manually."
