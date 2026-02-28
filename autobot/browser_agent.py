import json
import os
import shutil
import subprocess
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus, urlparse
from typing import Literal

from .focus_manager import FocusManager
from playwright.sync_api import BrowserContext
from playwright.sync_api import ConsoleMessage
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import Playwright
from playwright.sync_api import sync_playwright

try:
    import pyautogui
except ImportError:  # pragma: no cover - optional dependency
    pyautogui = None

BrowserMode = Literal["human_profile", "devtools"]


@dataclass(frozen=True)
class BrowserSessionConfig:
    user_data_dir: Path
    profile_directory: str
    executable_path: Path | None
    headless: bool = False
    launch_timeout_ms: int = 15000
    source_user_data_dir: Path | None = None
    source_profile_directory: str = "Default"
    browser_mode: str = "human_profile"

    @classmethod
    def from_env(cls) -> "BrowserSessionConfig":
        user_data = os.getenv("AUTOBOT_CHROME_USER_DATA_DIR") or _default_user_data_dir()
        profile_dir = os.getenv("AUTOBOT_CHROME_PROFILE_DIR", "Default")
        executable = os.getenv("AUTOBOT_CHROME_EXECUTABLE") or _detect_chrome_executable()
        source_user_data = os.getenv("AUTOBOT_CHROME_SOURCE_USER_DATA_DIR") or _real_chrome_user_data_dir()
        source_profile = os.getenv("AUTOBOT_CHROME_SOURCE_PROFILE_DIR", profile_dir)
        launch_timeout_ms = int(os.getenv("AUTOBOT_CHROME_LAUNCH_TIMEOUT_MS", "15000"))
        browser_mode = os.getenv("AUTOBOT_BROWSER_MODE", "human_profile").strip().lower()
        if browser_mode not in ("human_profile", "devtools"):
            browser_mode = "human_profile"
        return cls(
            user_data_dir=Path(user_data),
            profile_directory=profile_dir,
            executable_path=Path(executable) if executable else None,
            headless=False,
            launch_timeout_ms=max(5000, launch_timeout_ms),
            source_user_data_dir=Path(source_user_data) if source_user_data else None,
            source_profile_directory=source_profile,
            browser_mode=browser_mode,
        )


def _same_origin(url1: str | None, url2: str | None) -> bool:
    """True if both URLs share scheme, host, and port (same origin)."""
    try:
        if url1 is None or url2 is None:
            return False
        p1 = urlparse(url1 or "")
        p2 = urlparse(url2 or "")
        return (
            (p1.scheme or "https") == (p2.scheme or "https")
            and (p1.hostname or "").lower() == (p2.hostname or "").lower()
            and (p1.port or (443 if (p1.scheme or "").lower() == "https" else 80))
            == (p2.port or (443 if (p2.scheme or "").lower() == "https" else 80))
        )
    except Exception:
        return False


class BrowserController:
    def __init__(self, config: BrowserSessionConfig | None = None) -> None:
        self.config = config or BrowserSessionConfig.from_env()
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._console_errors: list[str] = []
        self.mode: BrowserMode = "human_profile"
        self._last_launch_error: str = ""
        self._last_opened_url: str | None = None

    def start(self) -> None:
        if self.config.browser_mode == "human_profile":
            self.mode = "human_profile"
            return
        if self.mode == "human_profile":
            return
        if self._context is not None:
            return

        _initialize_user_data_dir(self.config)
        self._playwright = sync_playwright().start()
        launch_kwargs = {
            "user_data_dir": str(self.config.user_data_dir),
            "headless": self.config.headless,
            "args": [f"--profile-directory={self.config.profile_directory}"],
            "timeout": self.config.launch_timeout_ms,
        }

        if self.config.executable_path:
            launch_kwargs["executable_path"] = str(self.config.executable_path)
        else:
            launch_kwargs["channel"] = "chrome"

        try:
            self._context = self._playwright.chromium.launch_persistent_context(**launch_kwargs)
            self._page = self._context.pages[0] if self._context.pages else self._context.new_page()
            self._console_errors = []
            self._page.on("console", self._on_console_message)
        except PlaywrightError as error:
            self._last_launch_error = str(error)
            self.close()
            if self.config.browser_mode == "auto" and _is_profile_block_error(error):
                self.mode = "human_profile"
                return
            raise _friendly_browser_error(error) from error

    def close(self) -> None:
        if self._context is not None:
            self._context.close()
            self._context = None
            self._page = None
        if self._playwright is not None:
            self._playwright.stop()
            self._playwright = None

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("Browser session is not started.")
        return self._page

    def goto(self, url: str) -> str:
        self.start()
        normalized = _normalize_url(url)
        if self.mode == "human_profile":
            self._open_in_human_profile(normalized)
            return f"Opened URL in your real Chrome profile (human mode): {normalized}"
        self.page.goto(normalized, wait_until="load")
        return f"Opened URL in your Chrome profile (devtools mode): {normalized}"

    def search(self, query: str) -> str:
        self.start()
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        if self.mode == "human_profile":
            self._open_in_human_profile(search_url)
            return f"Google search opened in your real Chrome profile (human mode): {query}"
        self.page.goto(search_url, wait_until="load")
        return f"Google search completed in your Chrome profile (devtools mode): {query}"

    def fill(self, selector: str, text: str, timeout_ms: int = 10000) -> str:
        self.start()
        if self.mode == "human_profile":
            raise RuntimeError(
                "Selector-based fill is unavailable in human profile mode. "
                "Use desktop typing actions or adapter flows designed for human mode."
            )
        self.page.wait_for_selector(selector, timeout=timeout_ms)
        self.page.fill(selector, text)
        return f"Filled '{selector}'."

    def click(self, selector: str, timeout_ms: int = 10000) -> str:
        self.start()
        if self.mode == "human_profile":
            raise RuntimeError(
                "Selector-based click is unavailable in human profile mode. "
                "Use desktop click actions with coordinates or dedicated human-mode adapter steps."
            )
        self.page.wait_for_selector(selector, timeout=timeout_ms)
        self.page.click(selector)
        return f"Clicked '{selector}'."

    def click_text(self, text: str, timeout_ms: int = 10000) -> str:
        self.start()
        if self.mode == "human_profile":
            raise RuntimeError("Text-based click is unavailable in human profile mode. Switch to devtools mode.")
        # Uses Playwright's locator to find elements by text (case-insensitive substring match)
        selector = f"text='{text}'"
        self.page.wait_for_selector(selector, timeout=timeout_ms)
        self.page.click(selector)
        return f"Clicked element with text '{text}'."

    def press(self, key: str) -> str:
        self.start()
        if self.mode == "human_profile":
            _require_pyautogui()
            pyautogui.press(_normalize_key_for_pyautogui(key))
            return f"Pressed '{key}' in human profile mode."
        
        if not self._page:
            return "Failed to press key: No active page."
            
        self.page.keyboard.press(key)
        return f"Pressed '{key}'."

    def read_text(self, selector: str, timeout_ms: int = 10000) -> str:
        self.start()
        if self.mode == "human_profile":
            return "Selector-based read_text is unavailable in human profile mode. Use visual descriptions or search."
        self.page.wait_for_selector(selector, timeout=timeout_ms)
        return self.page.inner_text(selector)

    def get_url(self) -> str:
        self.start()
        if self.mode == "human_profile":
            return "URL extraction unavailable in pure human profile mode."
        return self.page.url

    def read_console_errors(self) -> list[str]:
        self.start()
        if self.mode == "human_profile":
            return []
        return list(self._console_errors)

    def wait(self, milliseconds: int) -> str:
        self.start()
        if self.mode == "human_profile":
            if pyautogui is not None:
                pyautogui.sleep(milliseconds / 1000.0)
            else:
                import time
                time.sleep(milliseconds / 1000.0)
            return f"Waited {milliseconds} ms in human profile mode."
        self.page.wait_for_timeout(milliseconds)
        return f"Waited {milliseconds} ms."

    def scroll(self, direction: Literal["up", "down"], amount: int = 500) -> str:
        self.start()
        if self.mode == "human_profile":
            _require_pyautogui()
            scroll_amount = amount if direction == "up" else -amount
            pyautogui.scroll(scroll_amount)
            return f"Scrolled {direction} by {amount} units in human profile mode."
        
        if not self.page:
            return "Failed to scroll: No active page."
            
        # In devtools mode, we can use page.mouse.wheel or evaluate JS
        if direction == "up":
            self.page.evaluate(f"window.scrollBy(0, -{amount})")
        else:
            self.page.evaluate(f"window.scrollBy(0, {amount})")
        return f"Scrolled {direction} by {amount} pixels."

    def get_ui_snapshot(self) -> str:
        self.start()
        if self.mode == "human_profile":
            return "UI snapshot unavailable in human profile mode. Use visual descriptions or logs."
        
        # Capture visible interactive elements
        script = """
        () => {
            const elements = [];
            document.querySelectorAll('button, input, a, [role="button"], [role="link"]').forEach(el => {
                const rect = el.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0 && rect.top >= 0 && rect.left >= 0) {
                    elements.push({
                        type: el.tagName.toLowerCase(),
                        text: (el.innerText || el.value || el.placeholder || "").trim().substring(0, 50),
                        role: el.getAttribute('role') || el.type || '',
                        visible: true
                    });
                }
            });
            return JSON.stringify(elements.slice(0, 40));
        }
        """
        try:
            data = self.page.evaluate(script)
            elements = json.loads(data)
            lines = [f"Found {len(elements)} interactive elements:"]
            for el in elements:
                lines.append(f"- {el['type']}[{el['role']}]: '{el['text']}'")
            return "\n".join(lines)
        except Exception as e:
            return f"Error capturing UI snapshot: {str(e)}"

    def get_content(self) -> str:
        """Capture the entire visible text content of the page."""
        self.start()
        if self.mode == "human_profile":
            return "Content extraction via selector is restricted in human mode. Context: Use devtools mode for reading page text."
        try:
            # Simple text extraction. In devtools mode we can be more sophisticated.
            return self.page.inner_text("body")
        except Exception as e:
            return f"Error reading page content: {str(e)}"

    def _on_console_message(self, msg: ConsoleMessage) -> None:
        if msg.type == "error":
            self._console_errors.append(msg.text)

    def reset_last_opened_url(self) -> None:
        """Reset last-opened URL so the next goto can open a new tab. Call at plan start if desired."""
        self._last_opened_url = None

    def _open_in_human_profile(self, url: str) -> None:
        # When open_new_tab is true: same origin -> reuse current tab; different/first -> new tab.
        normalized = _normalize_url(url)
        open_new_tab = os.getenv("AUTOBOT_OPEN_NEW_TAB", "1").strip().lower() in ("1", "true", "yes")
        reuse_tab = (
            open_new_tab
            and self._last_opened_url is not None
            and _same_origin(self._last_opened_url, normalized)
            and pyautogui is not None
        )
        self._last_opened_url = normalized

        if pyautogui is not None:
            focus = FocusManager(logger=lambda _: None)
            result = focus.ensure_keywords_focused(("chrome", "whatsapp", "overleaf", "grok"))
            title = (result.title or "").strip()
            lowered = title.lower()
            # Safety: never type into automation/editor/terminal windows, even if they match keywords.
            if "autobot" in lowered or "cursor" in lowered or "command wizard" in lowered or "powershell" in lowered:
                result = None  # force fallback path below
            if result and result.ok:
                if not reuse_tab and open_new_tab:
                    pyautogui.hotkey("ctrl", "t")
                    time.sleep(0.6)
                pyautogui.hotkey("ctrl", "l")
                time.sleep(0.2)
                pyautogui.write(normalized, interval=0.02)
                time.sleep(0.1)
                pyautogui.press("enter")
                return
        if self.config.executable_path and self.config.executable_path.exists():
            subprocess.Popen([str(self.config.executable_path), normalized], shell=False)
            return
        webbrowser.open(normalized, new=0, autoraise=True)

    def mode_status(self) -> dict[str, str]:
        return {
            "active_mode": self.mode,
            "configured_mode": self.config.browser_mode,
            "last_launch_error": self._last_launch_error or "none",
        }

    def get_status(self) -> str:
        """Returns a JSON string summarizing the current browser health and state."""
        status = {
            "is_active": self.is_active(),
            "mode": self.mode,
            "url": "none",
            "title": "none",
            "captcha_detected": self.detect_recaptcha() if self.is_active() else False,
            "error": self._last_launch_error or "none"
        }
        if self.is_active() and self._page and self.mode != "human_profile":
            try:
                status["url"] = self._page.url
                status["title"] = self._page.title()
            except Exception:
                status["is_active"] = False
        
        return json.dumps(status)

    def is_active(self) -> bool:
        """Check if the browser and page are still responsive."""
        if not self._context or not self._page:
            return False
        try:
            return not self._page.is_closed()
        except Exception:
            return False

    def set_mode(self, mode: Literal["human_profile", "devtools"]) -> str:
        """Dynamically switch between stealth (human) and high-speed (devtools) modes."""
        if mode == self.mode:
            return f"Already in {mode} mode."
        
        self.close() # Must restart to change mode
        self.mode = mode
        self.start()
        return f"Switched to {mode} mode."

    def detect_recaptcha(self) -> bool:
        """Analyze DOM for known RECAPTCHA / Cloudflare / Bot detection markers."""
        self.start()
        if self.mode == "human_profile":
            # Best effort check via pyautogui if needed, but for now we look at browser context if available
            return False
            
        markers = [
            "iframe[src*='recaptcha']",
            "iframe[src*='hcaptcha']",
            "div.g-recaptcha",
            "#cf-turnstile",
            ".cf-browser-verification",
            "text='Verify you are human'",
            "text='Checking if the site connection is secure'"
        ]
        
        for marker in markers:
            try:
                if self.page.locator(marker).count() > 0:
                    return True
            except Exception:
                continue
        return False

    def wait_for_load(self, timeout_ms: int = 15000) -> None:
        """Wait for the page to reach 'load' or 'networkidle' state."""
        self.start()
        if self.mode == "human_profile":
            time.sleep(timeout_ms / 1000.0)
            return
        try:
            self.page.wait_for_load_state("load", timeout=timeout_ms)
            self.page.wait_for_load_state("networkidle", timeout=timeout_ms)
        except PlaywrightError:
            # If load state wait fails / times out, we continue anyway as the UI might be partially ready
            pass

    def screenshot(self, filepath: str) -> str:
        """Capture screenshot: in human_profile or when page closed uses pyautogui; else page.screenshot."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.mode == "human_profile":
            _require_pyautogui()
            pyautogui.screenshot(imageFilename=filepath)
            return filepath
        self.start()
        if self._page is not None:
            try:
                self._page.screenshot(path=filepath)
                return filepath
            except Exception:
                if pyautogui is not None:
                    pyautogui.screenshot(imageFilename=filepath)
                    return filepath
                raise
        if pyautogui is not None:
            pyautogui.screenshot(imageFilename=filepath)
            return filepath
        raise RuntimeError("Cannot take screenshot: no browser page and pyautogui not available.")


def _default_user_data_dir() -> str:
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        return str(Path(local_app_data) / "Autobot" / "ChromeAutomationProfile")
    return str(Path.home() / "AppData" / "Local" / "Autobot" / "ChromeAutomationProfile")


def _real_chrome_user_data_dir() -> str:
    local_app_data = os.getenv("LOCALAPPDATA", "")
    if local_app_data:
        return str(Path(local_app_data) / "Google" / "Chrome" / "User Data")
    return str(Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data")


def _detect_chrome_executable() -> str | None:
    candidates = [
        Path("C:/Program Files/Google/Chrome/Application/chrome.exe"),
        Path("C:/Program Files (x86)/Google/Chrome/Application/chrome.exe"),
        Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def _normalize_url(url: str) -> str:
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return f"https://{url}"


def _friendly_browser_error(error: Exception) -> RuntimeError:
    details = str(error)
    lowered = details.lower()
    if "process singleton" in details.lower() or "another browser is using your profile" in details.lower():
        return RuntimeError(
            "Chrome profile is currently locked. Close all Chrome windows first, then run Autobot again."
        )
    if "requires a non-default data directory" in lowered:
        return RuntimeError(
            "Chrome blocked automation on default profile data dir. "
            "Use an Autobot automation profile directory and optionally bootstrap from your Chrome profile."
        )
    return RuntimeError(f"Failed to start browser session: {details}")


def _is_profile_block_error(error: Exception) -> bool:
    text = str(error).lower()
    markers = [
        "requires a non-default data directory",
        "target page, context or browser has been closed",
        "devtools remote debugging requires a non-default data directory",
    ]
    return any(marker in text for marker in markers)


def _initialize_user_data_dir(config: BrowserSessionConfig) -> None:
    config.user_data_dir.mkdir(parents=True, exist_ok=True)
    marker = config.user_data_dir / ".autobot_bootstrap_done"
    if marker.exists():
        return

    source_root = config.source_user_data_dir
    if source_root is None:
        marker.write_text("no_source", encoding="utf-8")
        return

    if source_root.resolve() == config.user_data_dir.resolve():
        marker.write_text("same_source", encoding="utf-8")
        return

    source_profile = source_root / config.source_profile_directory
    target_profile = config.user_data_dir / config.profile_directory
    try:
        if source_profile.exists() and not target_profile.exists():
            shutil.copytree(source_profile, target_profile)
        local_state_src = source_root / "Local State"
        local_state_dst = config.user_data_dir / "Local State"
        if local_state_src.exists() and not local_state_dst.exists():
            shutil.copy2(local_state_src, local_state_dst)
    except Exception:
        # Bootstrap copy is best-effort. If it fails, user can still authenticate manually once.
        pass
    marker.write_text("done", encoding="utf-8")


def _require_pyautogui() -> None:
    if pyautogui is None:
        raise RuntimeError("Human profile key actions require pyautogui. Install it with: pip install pyautogui")


def _normalize_key_for_pyautogui(key: str) -> str:
    table = {
        "Enter": "enter",
        "Backspace": "backspace",
        "Escape": "esc",
        "ArrowLeft": "left",
        "ArrowRight": "right",
        "ArrowUp": "up",
        "ArrowDown": "down",
    }
    return table.get(key, key.lower())


def run_google_search(query: str) -> str:
    controller = BrowserController()
    try:
        res = controller.search(query)
        return str(res)
    finally:
        controller.close()


def open_url(url: str) -> str:
    controller = BrowserController()
    try:
        res = controller.goto(url)
        return str(res)
    finally:
        controller.close()
