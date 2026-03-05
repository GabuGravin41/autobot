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
        "paste_notebook_code": ActionSpec("Paste code from clipboard into the notebook (replaces all)"),
        "find_joined_competitions": ActionSpec("Find competitions the user has already joined"),
        "read_competition_overview": ActionSpec("Extract competition description and rules"),
        "create_new_notebook": ActionSpec("Create a new notebook in the current competition"),
        "read_notebook_status": ActionSpec("Check if notebook is idle, running, or has errors"),
        "read_last_cell_output": ActionSpec("Read output from the last executed cell"),
        "run_all_cells": ActionSpec("Trigger 'Run All' in the notebook"),
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
            # Look for rank/score in the sidebar or leaderboard page
            score_loc = self.browser.page.locator("[data-testid='competition-leaderboard-score']").last
            rank_loc = self.browser.page.locator("[data-testid='competition-leaderboard-rank']").last
            
            score = score_loc.inner_text(timeout=3000) if score_loc.is_visible() else "Unknown"
            rank = rank_loc.inner_text(timeout=3000) if rank_loc.is_visible() else "Unknown"
            
            return f"Leaderboard Status: Rank={rank}, Score={score}"
        except Exception:
            return "Read leaderboard failed. Page might still be loading or element changed."

    def do_read_notebook_status(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("read_notebook_status")
            return "Notebook status check (human mode)."
        
        self.browser.start()
        try:
            # Kaggle status bar items
            error_indicator = self.browser.page.locator("div:has-text('Error')").first
            if error_indicator.is_visible(timeout=2000):
                return "Status: ERROR - Notebook execution hit an error."
            
            running_indicator = self.browser.page.locator("div:has-text('Running')").first
            if running_indicator.is_visible(timeout=2000):
                return "Status: RUNNING - Notebook is currently executing cells."
            
            return "Status: IDLE - Notebook is ready."
        except Exception:
            return "Status: UNKNOWN - Could not determine notebook state."

    def do_read_last_cell_output(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("read_last_cell_output")
            return "Captured last cell output (human mode)."
            
        self.browser.start()
        try:
            # Kaggle outputs are often in div[data-testid='cell-output']
            outputs = self.browser.page.locator("[data-testid='cell-output']").all()
            if outputs:
                last_output = outputs[-1].inner_text()
                return f"Last Cell Output: {last_output[:1200]}..."
            return "No cell outputs found."
        except Exception as e:
            return f"Error reading output: {e}"

    def do_run_all_cells(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("run_all_cells")
            return "Triggered Run All (human mode)."
            
        self.browser.start()
        # Kaggle shortcut for Run All is Ctrl+Shift+Enter
        self.browser.press("Control+Shift+Enter")
        time.sleep(1)
        return "Triggered Run All cells."

    def do_copy_notebook_code(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("copy_notebook_code")
            return "Copied notebook code (human mode)."
            
        self.browser.start()
        self.browser.press("Control+a")
        self.browser.press("Control+c")
        time.sleep(1)
        return "Copied notebook code to clipboard."

    def do_paste_notebook_code(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("paste_notebook_code")
            return "Pasted code (human mode)."
            
        self.browser.start()
        # Focus editor, Select All, Paste
        # We assume for now that simply pressing Ctrl+A then Ctrl+V works on the active editor context.
        self.browser.click("[data-testid='cell-editor']", timeout_ms=5000)
        self.browser.press("Control+a")
        self.browser.press("Backspace")
        self.browser.press("Control+v")
        time.sleep(1)
        return "Pasted code into notebook."

    def do_find_joined_competitions(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://www.kaggle.com/competitions")
        if self._human_mode():
            time.sleep(5)
            self.run_human_nav("find_joined_competitions")
            return "Find joined competitions (human mode)."
        
        # Click "Entered" filter if it exists, or just look for 'In Progress'
        # Kaggle's UI changes, so we search for common patterns
        self.browser.start()
        try:
            # Try to grab competition names from the list
            titles = self.browser.page.locator("a[href^='/competitions/'] h2, a[href^='/competitions/'] div").all_inner_texts()
            # Clean up and filter
            unique_titles_list = sorted(list(set([str(t).strip() for t in titles if len(str(t).strip()) > 3])))
            subset = unique_titles_list[0:5] if len(unique_titles_list) > 5 else unique_titles_list
            return f"Found competitions: {', '.join(subset)}"
        except Exception as e:
            return f"Could not find competitions: {e}"

    def do_read_competition_overview(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("read_competition_overview")
            return "Read overview (human mode)."
        
        self.browser.start()
        try:
            # Overview is usually in a specific container
            overview = self.browser.page.locator("#competition-overview").inner_text()
            if not overview:
                overview = self.browser.page.locator("div:has-text('Description')").first.inner_text()
            return f"Competition Overview: {overview[:1000]}..."
        except Exception:
            return "Could not read overview text."

    def do_create_new_notebook(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("create_new_notebook")
            return "Create new notebook (human mode)."
            
        self.browser.start()
        # "Code" tab then "New Notebook"
        clicked_code = self._click_if_present(["a:has-text('Code')"], timeout_ms=5000)
        if clicked_code:
            time.sleep(2)
            clicked_new = self._click_if_present(["button:has-text('New Notebook')"], timeout_ms=5000)
            if clicked_new:
                return "Creating new notebook..."
        
        # Alternative: direct URL
        curr_url = self.browser.get_url()
        if "competitions/" in curr_url:
            base = curr_url.split("?")[0]
            if not base.endswith("/"): base += "/"
            self._ensure_url(base + "code?newNotebook=true")
            return "Directly opening new notebook creator."
            
        return "Could not find 'New Notebook' button."
