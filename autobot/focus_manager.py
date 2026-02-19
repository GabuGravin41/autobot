from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable

try:
    import pyautogui
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None


@dataclass
class FocusResult:
    ok: bool
    title: str
    reason: str = ""


class FocusManager:
    def __init__(self, logger=None) -> None:
        self.logger = logger or (lambda _msg: None)

    def ensure_keywords_focused(self, keywords: Iterable[str], tries: int = 3, delay_s: float = 0.2) -> FocusResult:
        if pyautogui is None:
            return FocusResult(ok=False, title="", reason="pyautogui unavailable")

        normalized = [item.lower().strip() for item in keywords if item.strip()]
        if not normalized:
            return FocusResult(ok=True, title=self.active_window_title(), reason="no-keywords")

        for _ in range(max(1, tries)):
            title = self.active_window_title()
            lowered = title.lower()
            if any(key in lowered for key in normalized):
                return FocusResult(ok=True, title=title)
            if self._activate_window_by_keywords(normalized):
                time.sleep(delay_s)
                title = self.active_window_title()
                lowered = title.lower()
                if any(key in lowered for key in normalized):
                    return FocusResult(ok=True, title=title)
            time.sleep(delay_s)

        return FocusResult(
            ok=False,
            title=self.active_window_title(),
            reason=f"Could not focus window with keywords: {', '.join(normalized)}",
        )

    def active_window_title(self) -> str:
        if pyautogui is None:
            return ""
        try:
            win = pyautogui.getActiveWindow()
            return str(getattr(win, "title", "") or "")
        except Exception:  # noqa: BLE001
            return ""

    def _activate_window_by_keywords(self, keywords: list[str]) -> bool:
        if pyautogui is None:
            return False
        try:
            windows = pyautogui.getAllWindows()
        except Exception:  # noqa: BLE001
            return False
        for win in windows:
            title = str(getattr(win, "title", "") or "")
            if not title:
                continue
            lowered = title.lower()
            if any(key in lowered for key in keywords):
                try:
                    win.activate()
                    self.logger(f"Activated window: {title}")
                    return True
                except Exception:  # noqa: BLE001
                    continue
        return False
