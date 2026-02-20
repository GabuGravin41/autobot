from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..browser_agent import BrowserController
from ..focus_manager import FocusManager
try:
    import pyautogui
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None


@dataclass(frozen=True)
class ActionSpec:
    description: str
    requires_confirmation: bool = False


class AdapterConfirmationError(RuntimeError):
    pass


class BaseAdapter:
    name: str = "base"
    actions: dict[str, ActionSpec] = {}
    login_url_fragments: tuple[str, ...] = ("login", "signin", "accounts.google.com")

    def __init__(self, browser: BrowserController, logger: Callable[[str], None] | None = None) -> None:
        self.browser = browser
        self.logger = logger or (lambda _msg: None)
        self.focus = FocusManager(logger=self.logger)
        self.state: dict[str, Any] = {}
        self._selectors = self._load_selectors()
        self._human_nav_maps = self._load_human_nav_maps()
        self._action_metrics: dict[str, dict[str, Any]] = {}
        self._selector_metrics: dict[str, dict[str, int]] = {}

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
        started_at = time.time()
        try:
            result = handler(params)
            self._record_action_metric(action=action, duration_s=time.time() - started_at, success=True)
            return result
        except Exception as error:  # noqa: BLE001
            self._record_action_metric(action=action, duration_s=time.time() - started_at, success=False, error=error)
            self._capture_failure_snapshot(error)
            raise

    def ensure_session_ready(self) -> dict[str, Any]:
        if not self.name.endswith("_web"):
            return {"status": "not_web_adapter"}

        self.browser.start()
        if self.browser.mode == "human_profile":
            self.state["session_health"] = "human_mode"
            return {"status": "human_mode", "url": ""}
        page = self.browser.page
        current_url = page.url
        looks_like_login = any(fragment in current_url.lower() for fragment in self.login_url_fragments)
        has_google_button = self._selector_visible(self.selector_candidates("login_google_button"), timeout_ms=800)
        if looks_like_login or has_google_button:
            handler = getattr(self, "do_attempt_google_continue_login", None)
            if callable(handler):
                message = handler({})
                self.state["session_health"] = "login_intervention_attempted"
                return {"status": "intervention_attempted", "message": message}
            self.state["session_health"] = "needs_login"
            return {"status": "needs_login", "url": current_url}

        self.state["session_health"] = "ready"
        return {"status": "ready", "url": current_url}

    def _load_wait_seconds(self, env_key: str, default: float) -> float:
        """Return configured wait in seconds for slow site loads (e.g. AUTOBOT_WHATSAPP_LOAD_WAIT=8)."""
        try:
            val = os.getenv(env_key, "").strip()
            if val:
                return max(0.0, float(val))
        except (ValueError, TypeError):
            pass
        return default

    def _ensure_url(self, url: str) -> None:
        self.browser.start()
        if self.browser.mode == "human_profile":
            self.browser.goto(url)
            return
        current = self.browser.page.url
        if not current.startswith(url):
            self.browser.goto(url)

    def _click_if_present(self, selectors: list[str], timeout_ms: int = 1500) -> bool:
        self.browser.start()
        if self.browser.mode == "human_profile":
            return False
        for selector in selectors:
            try:
                locator = self.browser.page.locator(selector).first
                if locator.is_visible(timeout=timeout_ms):
                    locator.click()
                    self._record_selector_metric(selector=selector, success=True)
                    return True
            except Exception:  # noqa: BLE001
                self._record_selector_metric(selector=selector, success=False)
                continue
        return False

    def selector_candidates(self, key: str) -> list[str]:
        selectors = self._selectors.get(key, [])
        return [item for item in selectors if isinstance(item, str) and item.strip()]

    def selector(self, key: str) -> str:
        selectors = self.selector_candidates(key)
        if not selectors:
            raise KeyError(f"No selectors configured for key '{key}' in adapter '{self.name}'.")
        return selectors[0]

    def fill_any(self, selector_key: str, text: str, timeout_ms: int = 10000) -> str:
        selectors = self.selector_candidates(selector_key)
        if not selectors:
            raise KeyError(f"No selector key found: {selector_key}")
        self.browser.start()
        for selector in selectors:
            try:
                self.browser.fill(selector=selector, text=text, timeout_ms=timeout_ms)
                self._record_selector_metric(selector=selector, success=True)
                return selector
            except Exception:  # noqa: BLE001
                self._record_selector_metric(selector=selector, success=False)
        raise RuntimeError(f"Could not fill any selector for key '{selector_key}'.")

    def click_any(self, selector_key: str, timeout_ms: int = 5000) -> str:
        selectors = self.selector_candidates(selector_key)
        if not selectors:
            raise KeyError(f"No selector key found: {selector_key}")
        clicked = self._click_if_present(selectors, timeout_ms=timeout_ms)
        if not clicked:
            raise RuntimeError(f"Could not click any selector for key '{selector_key}'.")
        return selectors[0]

    def _selector_visible(self, selectors: list[str], timeout_ms: int = 800) -> bool:
        if not selectors:
            return False
        self.browser.start()
        if self.browser.mode == "human_profile":
            return False
        for selector in selectors:
            try:
                if self.browser.page.locator(selector).first.is_visible(timeout=timeout_ms):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def telemetry(self) -> dict[str, Any]:
        return {
            "actions": self._action_metrics,
            "selectors": self._selector_metrics,
            "last_failure_snapshot": self.state.get("last_failure_snapshot"),
            "session_health": self.state.get("session_health"),
        }

    def _load_selectors(self) -> dict[str, list[str]]:
        selector_file = Path(__file__).resolve().parent / "selectors" / f"{self.name}.json"
        if not selector_file.exists():
            return {}
        try:
            payload = json.loads(selector_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, dict):
            return {}
        normalized: dict[str, list[str]] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                normalized[str(key)] = [str(item) for item in value]
        return normalized

    def _record_action_metric(self, action: str, duration_s: float, success: bool, error: Exception | None = None) -> None:
        entry = self._action_metrics.setdefault(action, {"count": 0, "success": 0, "failure": 0, "total_duration_s": 0.0})
        entry["count"] += 1
        entry["total_duration_s"] += duration_s
        if success:
            entry["success"] += 1
        else:
            entry["failure"] += 1
            entry["last_error"] = str(error) if error else "unknown"
        self.state["telemetry"] = self.telemetry()

    def _record_selector_metric(self, selector: str, success: bool) -> None:
        entry = self._selector_metrics.setdefault(selector, {"success": 0, "failure": 0})
        if success:
            entry["success"] += 1
        else:
            entry["failure"] += 1
        self.state["telemetry"] = self.telemetry()

    def _capture_failure_snapshot(self, error: Exception) -> None:
        snapshot: dict[str, Any] = {"error": str(error)}
        try:
            self.browser.start()
            snapshot["url"] = self.browser.page.url
            html = self.browser.page.content()
            snapshot["html_snippet"] = html[:2000]
        except Exception:  # noqa: BLE001
            snapshot["url"] = ""
            snapshot["html_snippet"] = ""
        self.state["last_failure_snapshot"] = snapshot

    def _human_mode(self) -> bool:
        self.browser.start()
        return self.browser.mode == "human_profile"

    def _human_type(self, text: str, interval: float = 0.01) -> None:
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for human profile mode keyboard automation.")
        self._ensure_human_target_focus()
        pyautogui.write(text, interval=interval)

    def _human_press(self, key: str) -> None:
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for human profile mode keyboard automation.")
        self._ensure_human_target_focus()
        pyautogui.press(key)

    def _human_hotkey(self, *keys: str) -> None:
        if pyautogui is None:
            raise RuntimeError("pyautogui is required for human profile mode keyboard automation.")
        self._ensure_human_target_focus()
        pyautogui.hotkey(*keys)

    def _ensure_human_target_focus(self, expected_keywords: tuple[str, ...] = ("chrome", "whatsapp", "instagram", "overleaf")) -> None:
        if pyautogui is None:
            return
        result = self.focus.ensure_keywords_focused(expected_keywords)
        title = result.title
        if not result.ok:
            raise RuntimeError(result.reason or "Unable to focus target window.")
        lowered = title.lower()
        if "autobot" in lowered or "cursor" in lowered or "command wizard" in lowered:
            raise RuntimeError(
                "Human-mode typing blocked because automation UI/editor appears focused. "
                "Click the target Chrome window first, then retry."
            )
        if expected_keywords and title:
            if not any(keyword in lowered for keyword in expected_keywords):
                raise RuntimeError(
                    f"Human-mode typing blocked; active window is '{title}'. Focus target Chrome tab and retry."
                )

    def human_nav_steps(self, key: str) -> list[dict[str, Any]]:
        value = self._human_nav_maps.get(key, [])
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    def run_human_nav(self, key: str, variables: dict[str, str] | None = None) -> None:
        variables = variables or {}
        steps = self.human_nav_steps(key)
        if not steps:
            raise RuntimeError(f"No human navigation map found for '{self.name}:{key}'.")
        for step in steps:
            action = str(step.get("action", "")).strip()
            value = str(step.get("value", ""))
            for var_key, var_value in variables.items():
                value = value.replace("{" + var_key + "}", var_value)
            delay = float(step.get("delay_s", 0.05))
            if action == "hotkey":
                keys = [item.strip() for item in value.split("+") if item.strip()]
                self._human_hotkey(*keys)
            elif action == "press":
                self._human_press(value)
            elif action == "type":
                self._human_type(value)
            elif action == "sleep":
                time.sleep(float(value or "0.2"))
            else:
                raise RuntimeError(f"Unsupported human nav action: {action}")
            if delay > 0:
                time.sleep(delay)

    def _load_human_nav_maps(self) -> dict[str, list[dict[str, Any]]]:
        nav_file = Path(__file__).resolve().parent / "human_nav" / f"{self.name}.json"
        if not nav_file.exists():
            return {}
        try:
            payload = json.loads(nav_file.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, dict):
            return {}
        result: dict[str, list[dict[str, Any]]] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                items = []
                for item in value:
                    if isinstance(item, dict):
                        items.append(item)
                result[str(key)] = items
        return result
