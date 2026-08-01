"""
WhatsApp Listener — Background observer for WhatsApp Web remote interventions.

Allows the user to send commands from their phone via WhatsApp Web to:
1. Trigger mid-flight goal pivots (e.g. "/override change focus to strategy B")
2. Supply requested passwords or answers (e.g. "/reply mysecretpassword")
"""
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)


class WhatsAppListener:
    """
    Monitors an active WhatsApp Web Playwright page for incoming commands.
    """

    def __init__(
        self,
        page: Any,  # Playwright Page
        target_contact: str = "",
        override_callback: Callable[[str], None] | None = None,
    ):
        self.page = page
        self.target_contact = target_contact
        self.override_callback = override_callback
        self._running = False

    async def check_messages(self) -> list[str]:
        """
        Check WhatsApp Web for unread messages containing /autobot or /override.
        """
        if not self.page or self.page.is_closed():
            return []

        if "web.whatsapp.com" not in self.page.url:
            return []

        new_commands: list[str] = []
        try:
            # Extract last message text from active chat stream
            messages = await self.page.locator("div.message-in span.selectable-text").all_text_contents()
            for msg in messages[-3:]:  # Check last 3 incoming messages
                msg_text = msg.strip()
                if msg_text.startswith("/autobot") or msg_text.startswith("/override"):
                    cmd = msg_text.split(" ", 1)[-1].strip()
                    new_commands.append(cmd)
                    if self.override_callback:
                        self.override_callback(cmd)
                        logger.info(f"📱 WhatsApp Remote Command Received: {cmd}")
        except Exception as e:
            logger.debug(f"WhatsApp check failed: {e}")

        return new_commands

    async def send_status_update(self, text: str) -> bool:
        """
        Send an outbound progress update text message back to active WhatsApp chat.
        """
        if not self.page or self.page.is_closed() or "web.whatsapp.com" not in self.page.url:
            return False

        try:
            input_box = self.page.locator('div[contenteditable="true"][data-tab="10"]')
            if await input_box.count() > 0:
                await input_box.first.click()
                await input_box.first.fill(f"🤖 Autobot Status: {text}")
                await self.page.keyboard.press("Enter")
                logger.info(f"📤 Sent WhatsApp status update: '{text[:50]}'")
                return True
        except Exception as e:
            logger.warning(f"Failed to send WhatsApp status update: {e}")

        return False
