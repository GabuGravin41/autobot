"""
Overleaf & Complex Web App Domain Helper — Provides resilient UI element locators,
keyboard shortcuts, and clipboard paste workflows for Overleaf, Grok, DeepSeek, and ChatGPT.

Solves the historical Overleaf button stall issue where custom React/Shadow DOM elements
stutter or fail under basic DOM clicks.
"""
import logging
from typing import Any

logger = logging.getLogger(__name__)


class OverleafHelper:
    """
    Resilient automation helpers for Overleaf project creation, CodeMirror editor input,
    and LaTeX document compilation.
    """

    # Primary and fallback selectors for Overleaf UI elements
    SELECTORS = {
        "new_project_button": [
            "button:has-text('New Project')",
            "a:has-text('New Project')",
            "[data-test='new-project-button']",
            "button.btn-primary:has-text('New Project')",
            "a.btn-primary",
        ],
        "blank_project_option": [
            "a:has-text('Blank Project')",
            "span:has-text('Blank Project')",
            "li:has-text('Blank Project')",
            "[data-test='blank-project-option']",
        ],
        "project_name_input": [
            "input[name='name']",
            "input[type='text']",
            "form input.form-control",
            "div.modal-body input",
        ],
        "create_button": [
            "button[type='submit']",
            "button:has-text('Create')",
            "input[type='submit']",
            "button.btn-primary:has-text('Create')",
        ],
        "codemirror_editor": [
            "div.cm-content",
            "div.cm-line",
            "textarea.form-control",
            "div.monaco-editor",
            "div.ace_content",
        ],
        "recompile_button": [
            "button:has-text('Recompile')",
            "button.btn-recompile",
            "[data-test='recompile-button']",
        ],
    }

    @classmethod
    async def find_and_click(cls, page: Any, element_key: str) -> bool:
        """
        Attempts to click an Overleaf UI element by iterating through robust fallback selectors.
        """
        selectors = cls.SELECTORS.get(element_key, [])
        for sel in selectors:
            try:
                locator = page.locator(sel)
                if await locator.count() > 0 and await locator.first.is_visible():
                    await locator.first.click(timeout=3000)
                    logger.info(f"✨ OverleafHelper: Successfully clicked '{element_key}' using selector '{sel}'")
                    return True
            except Exception as e:
                logger.debug(f"Selector '{sel}' attempt failed: {e}")

        logger.warning(f"⚠️ OverleafHelper: Could not click '{element_key}' with primary selectors. Falling back to visual/coordinate click.")
        return False

    @classmethod
    async def inject_latex_into_editor(cls, page: Any, latex_code: str, computer_clipboard: Any = None) -> bool:
        """
        Injects LaTeX code into Overleaf's CodeMirror editor cleanly via select-all + paste or keyboard input.
        """
        # Step 1: Click CodeMirror editor container
        editor_clicked = await cls.find_and_click(page, "codemirror_editor")
        if not editor_clicked:
            try:
                # Force click on active editor area
                await page.mouse.click(500, 400)
            except Exception:
                pass

        # Step 2: Select all existing content (Ctrl+A)
        await page.keyboard.press("Control+a")
        await page.keyboard.press("Backspace")

        # Step 3: Paste LaTeX code via clipboard or fill
        if computer_clipboard:
            computer_clipboard.set(latex_code)
            await page.keyboard.press("Control+v")
            logger.info("📋 Injected LaTeX code via Clipboard Paste (Ctrl+V)")
            return True
        else:
            # Fallback to direct keyboard typing / fill
            try:
                await page.keyboard.insert_text(latex_code)
                logger.info("⌨️ Injected LaTeX code via insert_text")
                return True
            except Exception as e:
                logger.error(f"Failed to insert LaTeX text: {e}")
                return False

    @classmethod
    async def recompile_document(cls, page: Any) -> bool:
        """
        Triggers document recompile in Overleaf via shortcut (Ctrl+Enter) or Recompile button.
        """
        try:
            await page.keyboard.press("Control+Enter")
            logger.info("⚡ Triggered Overleaf Recompile via Ctrl+Enter")
            return True
        except Exception:
            return await cls.find_and_click(page, "recompile_button")
