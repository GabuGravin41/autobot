from __future__ import annotations

from typing import Any

from .base import ActionSpec, BaseAdapter


class WhatsAppWebAdapter(BaseAdapter):
    name = "whatsapp_web"
    actions = {
        "open_home": ActionSpec("Open WhatsApp Web home"),
        "open_chat": ActionSpec("Open chat by display name"),
        "type_message": ActionSpec("Type a message in current chat input"),
        "send_typed_message": ActionSpec("Send currently typed message", requires_confirmation=True),
        "send_message_to_chat": ActionSpec("Open chat and send message", requires_confirmation=True),
        "read_recent_messages": ActionSpec("Read recent visible messages from active chat"),
        "list_visible_chats": ActionSpec("List visible chat names"),
        "attempt_google_continue_login": ActionSpec("Try Continue with Google login button"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://web.whatsapp.com")
        return "Opened WhatsApp Web."

    def do_open_chat(self, params: dict[str, Any]) -> str:
        chat = str(params.get("chat", "")).strip()
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
        self.fill_any("message_input", text, timeout_ms=15000)
        return "Message typed."

    def do_send_typed_message(self, _params: dict[str, Any]) -> str:
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

    def do_read_recent_messages(self, params: dict[str, Any]) -> list[str]:
        limit = int(params.get("limit", 5))
        self.browser.start()
        nodes = self.browser.page.locator(self.selector("visible_message_text")).all_inner_texts()
        return [item.strip() for item in nodes if item.strip()][-limit:]

    def do_list_visible_chats(self, params: dict[str, Any]) -> list[str]:
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
