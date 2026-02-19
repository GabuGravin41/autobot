from __future__ import annotations

import time
import uuid
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
        self.policy_profile = "balanced"
        self._pending_sensitive_actions: dict[str, dict[str, Any]] = {}
        self._adapters: dict[str, BaseAdapter] = {
            "whatsapp_web": WhatsAppWebAdapter(browser=browser, logger=self.logger),
            "instagram_web": InstagramWebAdapter(browser=browser, logger=self.logger),
            "overleaf_web": OverleafWebAdapter(browser=browser, logger=self.logger),
            "vscode_desktop": VSCodeDesktopAdapter(browser=browser, logger=self.logger),
        }

    def list_adapters(self) -> dict[str, dict[str, dict[str, Any]]]:
        return {name: adapter.action_library() for name, adapter in self._adapters.items()}

    def telemetry(self) -> dict[str, dict[str, Any]]:
        return {name: adapter.telemetry() for name, adapter in self._adapters.items()}

    def set_policy(self, profile: str) -> str:
        normalized = profile.strip().lower()
        if normalized not in {"strict", "balanced", "trusted"}:
            raise ValueError("Policy must be one of: strict, balanced, trusted.")
        self.policy_profile = normalized
        return self.policy_profile

    def prepare_sensitive_action(self, adapter_name: str, action: str, params: dict[str, Any] | None) -> dict[str, Any]:
        adapter = self._adapters.get(adapter_name)
        if adapter is None:
            raise ValueError(f"Unknown adapter '{adapter_name}'.")
        spec = adapter.actions.get(action)
        if spec is None:
            raise ValueError(f"Unknown action '{action}' for adapter '{adapter_name}'.")
        if not spec.requires_confirmation:
            raise ValueError("prepare_sensitive_action only applies to confirmation-gated adapter actions.")
        token = str(uuid.uuid4())
        payload = {
            "adapter": adapter_name,
            "action": action,
            "params": params or {},
            "created_at": time.time(),
            "expires_at": time.time() + 300,
        }
        self._pending_sensitive_actions[token] = payload
        return {"token": token, **payload}

    def confirm_sensitive_action(self, token: str) -> Any:
        pending = self._pending_sensitive_actions.get(token)
        if pending is None:
            raise ValueError("Invalid or expired confirmation token.")
        if time.time() > float(pending["expires_at"]):
            del self._pending_sensitive_actions[token]
            raise ValueError("Confirmation token expired.")
        result = self.call(
            adapter_name=str(pending["adapter"]),
            action=str(pending["action"]),
            params=dict(pending["params"]),
            confirmed=True,
        )
        del self._pending_sensitive_actions[token]
        return result

    def call(self, adapter_name: str, action: str, params: dict[str, Any] | None, confirmed: bool = False) -> Any:
        adapter = self._adapters.get(adapter_name)
        if adapter is None:
            raise ValueError(f"Unknown adapter '{adapter_name}'.")
        args = params or {}
        if action != "attempt_google_continue_login":
            try:
                health = adapter.ensure_session_ready()
                self.logger(f"Adapter session check [{adapter_name}]: {health.get('status')}")
            except Exception as error:  # noqa: BLE001
                self.logger(f"Adapter session check failed [{adapter_name}]: {error}")
        is_sensitive = bool(adapter.actions.get(action).requires_confirmation) if action in adapter.actions else False
        if is_sensitive and self.policy_profile == "strict":
            if confirmed:
                raise AdapterConfirmationError(
                    "Strict policy enabled. Use adapter_prepare_sensitive + adapter_confirm_sensitive token flow."
                )
        if is_sensitive and self.policy_profile == "trusted":
            confirmed = True
        try:
            return adapter.execute(action=action, params=args, confirmed=confirmed)
        except AdapterConfirmationError:
            raise
