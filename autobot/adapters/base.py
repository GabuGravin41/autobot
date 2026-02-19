from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ..browser_agent import BrowserController


@dataclass(frozen=True)
class ActionSpec:
    description: str
    requires_confirmation: bool = False


class AdapterConfirmationError(RuntimeError):
    pass


class BaseAdapter:
    name: str = "base"
    actions: dict[str, ActionSpec] = {}

    def __init__(self, browser: BrowserController, logger: Callable[[str], None] | None = None) -> None:
        self.browser = browser
        self.logger = logger or (lambda _msg: None)
        self.state: dict[str, Any] = {}

    def action_library(self) -> dict[str, dict[str, Any]]:
        return {
            action: {
                "description": spec.description,
                "requires_confirmation": spec.requires_confirmation,
            }
            for action, spec in self.actions.items()
        }

    def execute(self, action: str, params: dict[str, Any], confirmed: bool = False) -> Any:
        if action not in self.actions:
            raise ValueError(f"Unknown action '{action}' for adapter '{self.name}'.")

        spec = self.actions[action]
        if spec.requires_confirmation and not confirmed:
            raise AdapterConfirmationError(
                f"Action '{self.name}.{action}' requires explicit confirmation. Re-run with confirmed=true."
            )

        handler_name = f"do_{action}"
        handler = getattr(self, handler_name, None)
        if handler is None:
            raise NotImplementedError(f"Adapter action handler not implemented: {handler_name}")
        return handler(params)

    def _ensure_url(self, url: str) -> None:
        self.browser.start()
        current = self.browser.page.url
        if not current.startswith(url):
            self.browser.goto(url)

    def _click_if_present(self, selectors: list[str], timeout_ms: int = 1500) -> bool:
        self.browser.start()
        for selector in selectors:
            try:
                locator = self.browser.page.locator(selector).first
                if locator.is_visible(timeout=timeout_ms):
                    locator.click()
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False
