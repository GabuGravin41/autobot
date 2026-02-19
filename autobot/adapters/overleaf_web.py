from __future__ import annotations

from typing import Any

from .base import ActionSpec, BaseAdapter


class OverleafWebAdapter(BaseAdapter):
    name = "overleaf_web"
    actions = {
        "open_dashboard": ActionSpec("Open Overleaf dashboard"),
        "open_project": ActionSpec("Open project by title from dashboard"),
        "replace_editor_text": ActionSpec("Replace entire editor content"),
        "append_editor_text": ActionSpec("Append text to editor"),
        "compile_project": ActionSpec("Compile current Overleaf project"),
        "download_pdf": ActionSpec("Download PDF", requires_confirmation=True),
        "attempt_google_continue_login": ActionSpec("Try Continue with Google login button"),
    }

    def do_open_dashboard(self, _params: dict[str, Any]) -> str:
        self._ensure_url("https://www.overleaf.com/project")
        if self._human_mode():
            return "Opened Overleaf dashboard in human profile mode."
        return "Opened Overleaf dashboard."

    def do_open_project(self, params: dict[str, Any]) -> str:
        title = str(params.get("title", "")).strip()
        if not title:
            raise ValueError("Missing required param: title")
        if self._human_mode():
            self.do_open_dashboard({})
            self.run_human_nav("focus_project_search", {"title": title})
            self.state["active_project"] = title
            return f"Attempted Overleaf project focus in human mode: {title}"
        self.do_open_dashboard({})
        self.browser.click(f"{self.selector('project_link')}:has-text('{title}')", timeout_ms=12000)
        self.state["active_project"] = title
        return f"Opened Overleaf project: {title}"

    def do_replace_editor_text(self, params: dict[str, Any]) -> str:
        text = str(params.get("text", ""))
        if self._human_mode():
            self.run_human_nav("replace_editor_text", {"text": text})
            return "Replaced Overleaf editor text in human profile mode."
        self.click_any("editor_surface", timeout_ms=15000)
        self.browser.press("Control+A")
        self.browser.press("Backspace")
        self.browser.page.keyboard.type(text, delay=1)
        return "Replaced Overleaf editor text."

    def do_append_editor_text(self, params: dict[str, Any]) -> str:
        text = str(params.get("text", ""))
        if self._human_mode():
            self.run_human_nav("append_editor_text", {"text": text})
            return "Appended Overleaf editor text in human profile mode."
        self.click_any("editor_surface", timeout_ms=15000)
        self.browser.press("End")
        self.browser.page.keyboard.type(text, delay=1)
        return "Appended text in Overleaf editor."

    def do_compile_project(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            self.run_human_nav("compile_project")
            return "Triggered Overleaf compile in human profile mode (Ctrl+Enter)."
        clicked = self._click_if_present(self.selector_candidates("compile_button"), 6000)
        if not clicked:
            raise RuntimeError("Compile button not found.")
        return "Triggered Overleaf compile."

    def do_download_pdf(self, _params: dict[str, Any]) -> str:
        if self._human_mode():
            raise RuntimeError("Download PDF is not yet automated in human profile mode.")
        clicked = self._click_if_present(self.selector_candidates("download_pdf_button"), 6000)
        if not clicked:
            raise RuntimeError("Download PDF action not found.")
        return "Triggered PDF download."

    def do_attempt_google_continue_login(self, _params: dict[str, Any]) -> str:
        clicked = self._click_if_present(self.selector_candidates("login_google_button"))
        if clicked:
            return "Clicked Continue with Google."
        return "Continue with Google button not found. Wait for saved password autofill or sign in manually."
