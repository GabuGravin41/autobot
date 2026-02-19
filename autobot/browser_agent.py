import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus

from playwright.sync_api import BrowserContext
from playwright.sync_api import ConsoleMessage
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page
from playwright.sync_api import Playwright
from playwright.sync_api import sync_playwright


@dataclass(frozen=True)
class BrowserSessionConfig:
    user_data_dir: Path
    profile_directory: str
    executable_path: Path | None
    headless: bool = False

    @classmethod
    def from_env(cls) -> "BrowserSessionConfig":
        user_data = os.getenv("AUTOBOT_CHROME_USER_DATA_DIR") or _default_user_data_dir()
        profile_dir = os.getenv("AUTOBOT_CHROME_PROFILE_DIR", "Default")
        executable = os.getenv("AUTOBOT_CHROME_EXECUTABLE") or _detect_chrome_executable()
        return cls(
            user_data_dir=Path(user_data),
            profile_directory=profile_dir,
            executable_path=Path(executable) if executable else None,
            headless=False,
        )


class BrowserController:
    def __init__(self, config: BrowserSessionConfig | None = None) -> None:
        self.config = config or BrowserSessionConfig.from_env()
        self._playwright: Playwright | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None
        self._console_errors: list[str] = []

    def start(self) -> None:
        if self._context is not None:
            return

        self._playwright = sync_playwright().start()
        launch_kwargs = {
            "user_data_dir": str(self.config.user_data_dir),
            "headless": self.config.headless,
            "args": [f"--profile-directory={self.config.profile_directory}"],
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
            self.close()
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
        self.page.goto(_normalize_url(url), wait_until="load")
        return f"Opened URL in your Chrome profile: {_normalize_url(url)}"

    def search(self, query: str) -> str:
        self.start()
        search_url = f"https://www.google.com/search?q={quote_plus(query)}"
        self.page.goto(search_url, wait_until="load")
        return f"Google search completed in your Chrome profile: {query}"

    def fill(self, selector: str, text: str, timeout_ms: int = 10000) -> str:
        self.start()
        self.page.wait_for_selector(selector, timeout=timeout_ms)
        self.page.fill(selector, text)
        return f"Filled '{selector}'."

    def click(self, selector: str, timeout_ms: int = 10000) -> str:
        self.start()
        self.page.wait_for_selector(selector, timeout=timeout_ms)
        self.page.click(selector)
        return f"Clicked '{selector}'."

    def press(self, key: str) -> str:
        self.start()
        self.page.keyboard.press(key)
        return f"Pressed '{key}'."

    def read_text(self, selector: str, timeout_ms: int = 10000) -> str:
        self.start()
        self.page.wait_for_selector(selector, timeout=timeout_ms)
        return self.page.inner_text(selector)

    def read_console_errors(self) -> list[str]:
        self.start()
        return list(self._console_errors)

    def wait(self, milliseconds: int) -> str:
        self.start()
        self.page.wait_for_timeout(milliseconds)
        return f"Waited {milliseconds} ms."

    def _on_console_message(self, msg: ConsoleMessage) -> None:
        if msg.type == "error":
            self._console_errors.append(msg.text)


def _default_user_data_dir() -> str:
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
    if "process singleton" in details.lower() or "another browser is using your profile" in details.lower():
        return RuntimeError(
            "Chrome profile is currently locked. Close all Chrome windows first, then run Autobot again."
        )
    return RuntimeError(f"Failed to start browser session: {details}")


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
