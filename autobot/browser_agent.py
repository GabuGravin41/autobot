import os
import shutil
import subprocess
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus
from typing import Literal

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

BrowserMode = Literal["devtools", "human_profile"]


@dataclass(frozen=True)
class BrowserSessionConfig:
    user_data_dir: Path
    profile_directory: str
    executable_path: Path | None
    headless: bool = False
    launch_timeout_ms: int = 15000
    source_user_data_dir: Path | None = None
    source_profile_directory: str = "Default"
    browser_mode: str = "auto"

    @classmethod
    def from_env(cls) -> "BrowserSessionConfig":
        user_data = os.getenv("AUTOBOT_CHROME_USER_DATA_DIR") or _default_user_data_dir()
        profile_dir = os.getenv("AUTOBOT_CHROME_PROFILE_DIR", "Default")
        executable = os.getenv("AUTOBOT_CHROME_EXECUTABLE") or _detect_chrome_executable()
        source_user_data = os.getenv("AUTOBOT_CHROME_SOURCE_USER_DATA_DIR") or _real_chrome_user_data_dir()
        source_profile = os.getenv("AUTOBOT_CHROME_SOURCE_PROFILE_DIR", profile_dir)
        launch_timeout_ms = int(os.getenv("AUTOBOT_CHROME_LAUNCH_TIMEOUT_MS", "15000"))
        browser_mode = os.getenv("AUTOBOT_BROWSER_MODE", "auto").strip().lower()
        if browser_mode not in {"auto", "human_profile", "devtools"}:
            browser_mode = "auto"
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


class BrowserController:
    def __init__(self, config: BrowserSessionConfig | None = None) -> None:
        self.config = config or BrowserSessionConfig.from_env()
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._console_errors: list[str] = []
        self.mode: BrowserMode = "devtools"
        self._last_launch_error: str = ""

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

    def press(self, key: str) -> str:
        self.start()
        if self.mode == "human_profile":
            _require_pyautogui()
            pyautogui.press(_normalize_key_for_pyautogui(key))
            return f"Pressed '{key}' in human profile mode."
        self.page.keyboard.press(key)
        return f"Pressed '{key}'."

    def read_text(self, selector: str, timeout_ms: int = 10000) -> str:
        self.start()
        if self.mode == "human_profile":
            raise RuntimeError("Selector-based read_text is unavailable in human profile mode.")
        self.page.wait_for_selector(selector, timeout=timeout_ms)
        return self.page.inner_text(selector)

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

    def _on_console_message(self, msg: ConsoleMessage) -> None:
        if msg.type == "error":
            self._console_errors.append(msg.text)

    def _open_in_human_profile(self, url: str) -> None:
        if self.config.executable_path and self.config.executable_path.exists():
            subprocess.Popen([str(self.config.executable_path), url], shell=False)
            return
        webbrowser.open(url, new=0, autoraise=True)

    def mode_status(self) -> dict[str, str]:
        return {
            "active_mode": self.mode,
            "configured_mode": self.config.browser_mode,
            "last_launch_error": self._last_launch_error,
        }


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
        return controller.search(query)
    finally:
        controller.close()


def open_url(url: str) -> str:
    controller = BrowserController()
    try:
        return controller.goto(url)
    finally:
        controller.close()
