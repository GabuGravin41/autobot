from __future__ import annotations

from typing import Any, Callable

from ..browser_agent import BrowserController
from .base import AdapterConfirmationError, BaseAdapter
from .instagram_web import InstagramWebAdapter
from .overleaf_web import OverleafWebAdapter
from .vscode_desktop import VSCodeDesktopAdapter
from .whatsapp_web import WhatsAppWebAdapter


class AdapterManager:
    def __init__(self, browser: BrowserController, logger: Callable[[str], None] | None = None) -> None:
        self.browser = browser
        self.logger = logger or (lambda _msg: None)
        self._adapters: dict[str, BaseAdapter] = {
            "whatsapp_web": WhatsAppWebAdapter(browser=browser, logger=self.logger),
            "instagram_web": InstagramWebAdapter(browser=browser, logger=self.logger),
            "overleaf_web": OverleafWebAdapter(browser=browser, logger=self.logger),
            "vscode_desktop": VSCodeDesktopAdapter(browser=browser, logger=self.logger),
        }

    def list_adapters(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {name: adapter.action_library() for name, adapter in self._adapters.items()}

    def call(self, adapter_name: str, action: str, params: dict[str, Any] | None, confirmed: bool = False) -> Any:
        adapter = self._adapters.get(adapter_name)
        if adapter is None:
            raise ValueError(f"Unknown adapter '{adapter_name}'.")
        args = params or {}
        try:
            return adapter.execute(action=action, params=args, confirmed=confirmed)
        except AdapterConfirmationError:
            raise
