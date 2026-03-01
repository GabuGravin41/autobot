from __future__ import annotations

import time
from typing import Any

from .base import ActionSpec, BaseAdapter


class KaggleWebAdapter(BaseAdapter):
    name = "kaggle_web"
    description = "Interact with Kaggle: open competition, run notebook, submit, wait for score."
    actions = {
        "open_home": ActionSpec("Open Kaggle home"),
        "open_competition": ActionSpec("Open a competition by slug or URL"),
        "open_my_notebooks": ActionSpec("Open My Notebooks for the current competition"),
        "run_notebook": ActionSpec("Run all cells in the current notebook"),
        "submit_to_competition": ActionSpec("Submit notebook output to competition leaderboard"),
        "read_leaderboard_status": ActionSpec("Read current rank/score from leaderboard"),
        "copy_notebook_code": ActionSpec("Copy notebook code to clipboard"),
    }

    def do_open_home(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://www.kaggle.com")
        if self._human_mode():
            time.sleep(4)
        return "Opened Kaggle."

    def do_open_competition(self, params: dict[str, Any]) -> str:
        slug = (params.get("slug") or params.get("competition") or "").strip()
        url_param = (params.get("url") or "").strip()
        if url_param:
            self._ensure_url(url_param)
        elif slug:
            if not slug.startswith("http"):
                url = f"https://www.kaggle.com/competitions/{slug}"
            else:
                url = slug
            self._ensure_url(url)
        else:
            raise ValueError("open_competition requires 'slug' or 'url'.")
        if self._human_mode():
            time.sleep(5)
        return "Opened Kaggle competition."

    def do_open_my_notebooks(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("open_my_notebooks")
            return "Opened My Notebooks (human mode)."
        # Common link text
        clicked = self._click_if_present(["a:has-text('Notebooks')", "a:has-text('My Notebooks')"], timeout_ms=5000)
        if not clicked:
            self.logger("My Notebooks link not found; navigate manually.")
        return "Opened My Notebooks."

    def do_run_notebook(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("run_notebook")
            return "Run notebook (human mode)."
        self.browser.press("Control+Enter")  # Run all / run cell
        time.sleep(2)
        return "Triggered notebook run."

    def do_submit_to_competition(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("submit_to_competition")
            return "Submit to competition (human mode)."
        clicked = self._click_if_present(
            ["button:has-text('Submit')", "button:has-text('Submit to Competition')", "[data-testid='submit-button']"],
            timeout_ms=8000,
        )
        if not clicked:
            self.logger("Submit button not found; click Submit manually.")
        return "Submitted to competition."

    def do_read_leaderboard_status(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("read_leaderboard_status")
            return "Read leaderboard (human mode); check clipboard."
        self.browser.start()
        try:
            loc = self.browser.page.locator("text=Leaderboard").first
            if loc.is_visible(timeout=3000):
                # Try to get nearby score/rank text
                return "Leaderboard visible; parse score manually if needed."
        except Exception:  # noqa: S110
            pass
        return "Leaderboard area not found."

    def do_copy_notebook_code(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("copy_notebook_code")
            return "Copied notebook code (human mode)."
        self.browser.press("Control+A")
        time.sleep(0.2)
        self.browser.press("Control+C")
        return "Copied notebook content to clipboard."
