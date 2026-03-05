from __future__ import annotations
import time
from typing import Any
from .base import ActionSpec, BaseAdapter

class ClaudeWebAdapter(BaseAdapter):
    name = "claude_web"
    description = "Interact with Claude.ai: send prompts and copy responses."
    actions = {
        "open_home": ActionSpec("Open Claude home page"),
        "send_message": ActionSpec("Type and send a message to Claude"),
        "copy_response": ActionSpec("Copy the latest response from Claude"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://claude.ai/chats")
        if self._human_mode():
            time.sleep(5)
        return "Opened Claude home."

    def do_send_message(self, params: dict[str, Any]) -> str:
        message = str(params.get("message", "")).strip()
        if not message:
            raise ValueError("send_message requires 'message'.")
        
        if self._human_mode():
            self.run_human_nav("send_message", {"message": message})
            return "Message sent to Claude (human mode)."
            
        self.browser.start()
        # Claude textarea usually has contenteditable or is a textarea
        # Selector for Claude's input is often div[contenteditable="true"]
        textarea_selector = "fieldset div[contenteditable='true']"
        self.browser.fill(textarea_selector, message)
        time.sleep(1)
        self.browser.press("Enter")
        return "Message sent to Claude."

    def do_copy_response(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("copy_response")
            return "Copied Claude response (human mode)."
            
        self.browser.start()
        # Claude's latest response usually has a specific class or is the last message-content
        # We can try to find the last message-content div
        try:
            # This is a bit brittle, but often works for a 'copy' button if it exists, 
            # or just grabbing the text of the last message.
            # Usually there's a 'Copy' button in the toolbar of the last message response.
            copy_button = self.browser.page.locator("button:has-text('Copy')").last
            if copy_button.is_visible(timeout=5000):
                copy_button.click()
                return "Latest response copied to clipboard (via Copy button)."
            
            last_msg = self.browser.page.locator(".font-claude-message").last
            if last_msg.is_visible(timeout=2000):
                text = last_msg.inner_text()
                # If no copy button, we can set clipboard via powershell or return text
                return f"Response captured: {text[:100]}..."
        except Exception as e:
            return f"Error copying response: {e}"
        return "Could not find response to copy."
