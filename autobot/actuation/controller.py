"""
Actuation Controller — Universal System Action Dispatcher.
Executes DOM Playwright actions, native OS mouse/keyboard actions, and CLI commands.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class ActuationController:
    """Dispatches actions to Browser (Playwright CDP) or Native OS (PyAutoGUI/CLI)."""

    def __init__(self, page: Optional[Any] = None):
        self.page = page

    async def execute(self, action_name: str, args: Dict[str, Any]) -> str:
        """
        Execute an action by name and parameters.

        Returns:
            Human-readable result summary of execution.
        """
        logger.info(f"⚡ Actuation executing: '{action_name}' with args {args}")

        try:
            if action_name == "navigate":
                url = args.get("url", "")
                if self.page and not self.page.is_closed():
                    await self.page.goto(url, wait_until="domcontentloaded", timeout=15000)
                    return f"Navigated to {url}"
                else:
                    return f"Browser page not attached. Failed to navigate to {url}"

            elif action_name == "click":
                element_id = args.get("element_id")
                if element_id is not None and self.page and not self.page.is_closed():
                    selector = f'[data-autobot-id="{element_id}"]'
                    await self.page.click(selector, timeout=5000)
                    return f"Clicked element [{element_id}]"
                else:
                    # Fallback to native pyautogui mouse click if x,y provided
                    x, y = args.get("x"), args.get("y")
                    if x is not None and y is not None:
                        import pyautogui
                        pyautogui.FAILSAFE = False
                        pyautogui.click(x, y)
                        return f"Clicked native screen coordinates ({x}, {y})"
                    return f"Click failed: missing valid element_id or coordinates"

            elif action_name == "type":
                text = args.get("text", "")
                element_id = args.get("element_id")
                if element_id is not None and self.page and not self.page.is_closed():
                    selector = f'[data-autobot-id="{element_id}"]'
                    await self.page.fill(selector, text, timeout=5000)
                    return f"Typed '{text}' into element [{element_id}]"
                else:
                    import pyautogui
                    pyautogui.typewrite(text)
                    return f"Typed '{text}' via native keyboard"

            elif action_name == "run_command":
                cmd = args.get("command", "")
                res = await asyncio.to_thread(
                    subprocess.run, cmd, shell=True, capture_output=True, text=True, timeout=30
                )
                output = (res.stdout or res.stderr or "").strip()
                return f"Executed CLI command: '{cmd}'. Output: {output[:300]}"

            elif action_name == "wait":
                seconds = float(args.get("seconds", 2.0))
                await asyncio.sleep(seconds)
                return f"Waited {seconds} seconds"

            elif action_name == "done":
                return "Task marked complete by agent."

            else:
                return f"Unknown action: '{action_name}'"

        except Exception as e:
            logger.error(f"Actuation error on action '{action_name}': {e}")
            return f"Action '{action_name}' failed: {e}"
