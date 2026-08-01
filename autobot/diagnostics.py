"""
Preflight Diagnostics - answer "why isn't it working?" before spending a run.

Every serious bug found in this project so far (a dead import, a missing
action field, an uncalled function, a stubbed endpoint, an unguarded
Windows-only import) failed at a different layer, and every one of them
surfaced to the user as the same thing: the agent just didn't work. There
was no way to tell a missing API key from a missing package from Chrome not
listening on the debug port.

This module checks each layer independently and reports what is broken and
how to fix it - with no LLM calls and no browser automation, so it costs
nothing to run and works even when the agent itself is badly broken.

Run with:  autobot --doctor
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

# Status values, ordered by severity.
OK = "OK"
WARN = "WARN"
FAIL = "FAIL"

_ICON = {OK: "[ OK ]", WARN: "[WARN]", FAIL: "[FAIL]"}


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""
    fix: str = ""

    def render(self) -> str:
        line = f"{_ICON[self.status]} {self.name}"
        if self.detail:
            line += f"\n        {self.detail}"
        if self.fix and self.status != OK:
            line += f"\n        -> {self.fix}"
        return line


# ── Individual checks ─────────────────────────────────────────────────────────

def check_python() -> Check:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 10):
        return Check(
            "Python version", FAIL, f"found {version}",
            "Autobot uses 3.10+ syntax (X | None). Install Python 3.10 or newer.",
        )
    return Check("Python version", OK, version)


def _module_present(name: str) -> bool:
    """True if importable WITHOUT importing it (avoids side effects)."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def check_required_packages() -> list[Check]:
    """Packages the agent cannot run at all without."""
    # Note: playwright's bundled browser download is NOT required. Autobot
    # attaches to the user's real Chrome via connect_over_cdp() and never
    # calls chromium.launch(), so only the Python driver matters here.
    required = {
        "pydantic": "pip install pydantic",
        "openai": "pip install openai",
        "playwright": "pip install playwright  (the browser download is NOT needed)",
        "httpx": "pip install httpx",
        "websockets": "pip install websockets",
        "PIL": "pip install Pillow",
    }
    checks = []
    missing = [n for n in required if not _module_present(n)]
    for name, fix in required.items():
        if name in missing:
            checks.append(Check(f"package: {name}", FAIL, "not installed", fix))
        else:
            checks.append(Check(f"package: {name}", OK))
    if missing:
        checks.append(Check(
            "install all requirements", FAIL,
            f"{len(missing)} required package(s) missing",
            "pip install -r requirements.txt",
        ))
    return checks


def check_optional_packages() -> list[Check]:
    """Packages that disable a capability when missing, but don't break the agent."""
    checks: list[Check] = []

    if platform.system() == "Windows":
        if _module_present("uiautomation"):
            checks.append(Check("native app control (uiautomation)", OK,
                                "can drive Artemis / VESTA / Excel"))
        else:
            checks.append(Check(
                "native app control (uiautomation)", WARN,
                "not installed - Autobot is blind outside the browser",
                "pip install uiautomation",
            ))
    else:
        checks.append(Check(
            "native app control (uiautomation)", WARN,
            f"unavailable on {platform.system()} (Windows-only)",
            "Native desktop automation currently requires Windows.",
        ))

    if _module_present("pyautogui"):
        checks.append(Check("mouse/keyboard control (pyautogui)", OK))
    else:
        checks.append(Check(
            "mouse/keyboard control (pyautogui)", FAIL,
            "not installed - anti_sleep imports it at module load, which "
            "breaks Computer() and therefore the whole agent",
            "pip install pyautogui",
        ))
    return checks


def check_chrome() -> Check:
    """Locate the Chrome executable the launcher will try to start."""
    candidates = [
        os.getenv("AUTOBOT_CHROME_EXECUTABLE"),
        os.getenv("CHROME_EXECUTABLE"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/google-chrome",
        "/usr/bin/chromium-browser",
    ]
    for path in candidates:
        if path and Path(path).exists():
            return Check("Chrome executable", OK, path)
    if shutil.which("google-chrome") or shutil.which("chrome"):
        return Check("Chrome executable", OK, "found on PATH")
    return Check(
        "Chrome executable", FAIL, "not found in any standard location",
        "Install Chrome, or set AUTOBOT_CHROME_EXECUTABLE to its full path in .env",
    )


def check_cdp(port: int | None = None) -> Check:
    """Is a Chrome already listening on the DevTools port we attach to?"""
    port = port or int(os.getenv("AUTOBOT_CDP_PORT", "9222"))
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as r:
            info = json.loads(r.read())
        browser = info.get("Browser", "unknown")
        return Check(f"Chrome DevTools port {port}", OK, f"reachable - {browser}")
    except Exception:
        return Check(
            f"Chrome DevTools port {port}", WARN,
            "nothing listening (Autobot will try to launch Chrome itself)",
            "If launching fails, close ALL Chrome windows first, or start Chrome "
            f'manually with --remote-debugging-port={port}',
        )


def check_llm_config() -> list[Check]:
    """Which provider will be used, and is a key present?

    Only reports whether a key is SET - never its value, and never validates
    it against the network (that would cost a request and could leak the key
    into logs on failure).
    """
    providers = {
        "ANTHROPIC_API_KEY": "anthropic",
        "OPENROUTER_API_KEY": "openrouter",
        "OPENAI_API_KEY": "openai",
        "GEMINI_API_KEY": "gemini",
        "GOOGLE_API_KEY": "gemini",
    }
    present = [(k, v) for k, v in providers.items() if os.getenv(k)]
    checks: list[Check] = []

    if not present:
        checks.append(Check(
            "LLM API key", FAIL, "no provider key found in environment or .env",
            "Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY in .env.",
        ))
    else:
        names = ", ".join(k for k, _ in present)
        checks.append(Check("LLM API key", OK, f"set: {names}"))

    configured = os.getenv("AUTOBOT_LLM_PROVIDER", "(auto-detect)")
    model = os.getenv("AUTOBOT_LLM_MODEL", "(provider default)")
    checks.append(Check("LLM provider / model", OK, f"provider={configured}  model={model}"))
    return checks


def check_llm_connectivity() -> list[Check]:
    """Can we actually REACH the configured LLM provider?

    A missing key and an unreachable host produce very different errors but
    look identical from the agent's side ("LLM call failed"), so we separate
    the layers here: DNS, then TCP, then TLS, then HTTP. No API key is sent
    and no tokens are spent - the models endpoint is public.
    """
    import socket

    provider = (os.getenv("AUTOBOT_LLM_PROVIDER") or "").lower()
    host = {
        "openrouter": "openrouter.ai",
        "openai": "api.openai.com",
        "gemini": "generativelanguage.googleapis.com",
    }.get(provider, "openrouter.ai")

    checks: list[Check] = []

    # Layer 1: DNS
    try:
        socket.getaddrinfo(host, 443)
    except Exception as e:
        return [Check(
            f"reach {host} (DNS)", FAIL, f"cannot resolve: {e}",
            "No DNS for the provider. Check your internet connection, VPN, or DNS settings.",
        )]
    checks.append(Check(f"reach {host} (DNS)", OK))

    # Layer 2: TCP
    try:
        with socket.create_connection((host, 443), timeout=8):
            pass
    except Exception as e:
        checks.append(Check(
            f"reach {host} (TCP 443)", FAIL, f"cannot connect: {e}",
            "A firewall or network policy is blocking outbound HTTPS to this host.",
        ))
        return checks
    checks.append(Check(f"reach {host} (TCP 443)", OK))

    # Layer 3/4: TLS + HTTP. Separating TLS failures matters because they mean
    # something is intercepting HTTPS (corporate/campus proxy, or antivirus
    # with HTTPS scanning) rather than the service being down.
    import urllib.error
    import urllib.request
    url = f"https://{host}/api/v1/models" if host == "openrouter.ai" else f"https://{host}/"
    try:
        with urllib.request.urlopen(url, timeout=12) as r:
            r.read(64)
        checks.append(Check(f"reach {host} (HTTPS)", OK, "TLS verified, endpoint responded"))
    except urllib.error.HTTPError as e:
        # An HTTP status means TLS worked - that's all we needed to prove.
        checks.append(Check(f"reach {host} (HTTPS)", OK, f"TLS verified (HTTP {e.code})"))
    except urllib.error.URLError as e:
        reason = str(getattr(e, "reason", e))
        if "CERTIFICATE_VERIFY_FAILED" in reason or "SSL" in reason.upper():
            checks.append(Check(
                f"reach {host} (HTTPS)", FAIL,
                f"TLS certificate verification failed: {reason[:120]}",
                "Something is intercepting HTTPS on this network (campus/corporate "
                "proxy, or antivirus with HTTPS scanning). Options: switch network "
                "(e.g. phone hotspot) to confirm; or export the interceptor's root CA "
                "and point Python at it with SSL_CERT_FILE=C:\\path\\to\\ca.pem. "
                "Do NOT disable certificate verification - that exposes your API key.",
            ))
        else:
            checks.append(Check(
                f"reach {host} (HTTPS)", FAIL, f"request failed: {reason[:120]}",
                "Check proxy settings (HTTPS_PROXY) or try a different network.",
            ))
    except Exception as e:
        checks.append(Check(f"reach {host} (HTTPS)", WARN, f"unexpected: {e}"))

    return checks


def check_proxy_env() -> list[Check]:
    """Report proxy/CA variables, which silently change where requests go.

    A CA path that doesn't exist is worse than not setting one at all: httpx
    raises FileNotFoundError while merely CONSTRUCTING the client, so the run
    dies with a bare traceback before any task logic executes.
    """
    checks: list[Check] = []
    names = ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE")
    set_vars = {n: os.getenv(n) for n in names if os.getenv(n)}

    if not set_vars:
        checks.append(Check("proxy / CA environment", OK, "none set"))
    else:
        detail = "; ".join(f"{k}={v}" for k, v in set_vars.items())
        checks.append(Check("proxy / CA environment", OK, detail))

    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        path = os.getenv(var)
        if path and not Path(path).is_file():
            checks.append(Check(
                f"{var} points to a real file", FAIL,
                f"{path} does not exist",
                f"Every HTTPS call will crash with FileNotFoundError. Either point "
                f"{var} at a real CA bundle, or clear it: Remove-Item Env:{var}",
            ))

    # truststore is how we cope with TLS-intercepting networks.
    if _module_present("truststore"):
        checks.append(Check("OS certificate store (truststore)", OK,
                            "Python will trust the same CAs as your browser"))
    else:
        checks.append(Check(
            "OS certificate store (truststore)", WARN,
            "not installed - Python uses its own bundled CA list",
            "If HTTPS fails on this network but works in Chrome, run: "
            "pip install truststore",
        ))
    return checks


def check_approval_mode() -> Check:
    mode = os.getenv("AUTOBOT_APPROVAL_MODE", "balanced").lower()
    if mode not in ("strict", "balanced", "trusted"):
        return Check(
            "approval mode", FAIL, f"unrecognized value '{mode}'",
            "Set AUTOBOT_APPROVAL_MODE to strict, balanced, or trusted.",
        )
    note = {
        "strict": "pauses for CAUTION and above",
        "balanced": "pauses for DANGER and above",
        "trusted": "pauses only for IRREVERSIBLE actions",
    }[mode]
    return Check("approval mode", OK, f"{mode} - {note}")


def check_env_file() -> Check:
    """Locate .env and check it for a byte-order mark.

    A BOM makes the FIRST variable in the file unreadable while every other
    line loads normally - which presents as "my API key isn't being picked
    up" even though the key is plainly there in the file. Windows PowerShell
    5.1's `Out-File -Encoding utf8` produces exactly this.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return Check(
            ".env file", WARN, "not found - relying on shell environment only",
            f"Create {env_path} with your API key(s) so settings persist between runs.",
        )

    try:
        head = env_path.read_bytes()[:3]
    except Exception as e:
        return Check(".env file", WARN, f"{env_path} - could not read: {e}")

    if head[:3] == b"\xef\xbb\xbf":
        return Check(
            ".env file", WARN,
            f"{env_path} - starts with a UTF-8 BOM; the FIRST setting in it "
            "may read as unset in older tooling",
            "Autobot handles this, but to clean it up rewrite the file without "
            "a BOM. In PowerShell 5.1 use: "
            "[IO.File]::WriteAllText('.env', (Get-Content .env -Raw))",
        )
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return Check(
            ".env file", FAIL,
            f"{env_path} - is UTF-16 encoded; none of its settings will load",
            "Rewrite it as UTF-8. In PowerShell: "
            "[IO.File]::WriteAllText('.env', (Get-Content .env -Raw))",
        )
    return Check(".env file", OK, str(env_path))


def check_writable_dirs() -> list[Check]:
    """Directories the agent writes to during a run."""
    targets = {
        "learned skills": Path.cwd() / "autobot" / "knowledge" / "skills",
        "run history": Path.cwd() / "runs",
    }
    checks = []
    for label, path in targets.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_probe"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
            checks.append(Check(f"writable: {label}", OK, str(path)))
        except Exception as e:
            checks.append(Check(
                f"writable: {label}", FAIL, f"{path} - {e}",
                "Check folder permissions, or run Autobot from a directory you own.",
            ))
    return checks


def check_learned_skills() -> Check:
    """How much the agent has learned - directly predicts token cost per run."""
    skills_dir = Path.cwd() / "autobot" / "knowledge" / "skills"
    if not skills_dir.exists():
        return Check("learned skills", WARN, "none yet - first runs cost full price")
    files = list(skills_dir.glob("*.json"))
    if not files:
        return Check(
            "learned skills", WARN,
            "none yet - every run re-derives from scratch at full token cost",
            "Skills are saved automatically when a run ends with done(success=true).",
        )
    return Check("learned skills", OK, f"{len(files)} saved - repeat tasks run cheaper")


# ── Runner ────────────────────────────────────────────────────────────────────

def run_all() -> list[Check]:
    """Run every check. Never raises - a broken check must not break the doctor."""
    checks: list[Check] = []
    steps = [
        lambda: [check_python()],
        check_required_packages,
        check_optional_packages,
        lambda: [check_env_file()],
        check_llm_config,
        check_proxy_env,
        check_llm_connectivity,
        lambda: [check_approval_mode()],
        lambda: [check_chrome()],
        lambda: [check_cdp()],
        check_writable_dirs,
        lambda: [check_learned_skills()],
    ]
    for step in steps:
        try:
            checks.extend(step())
        except Exception as e:
            checks.append(Check(f"check failed: {step}", WARN, str(e)))
    return checks


def report(checks: list[Check]) -> int:
    """Print the report. Returns a process exit code (0 = usable)."""
    fails = [c for c in checks if c.status == FAIL]
    warns = [c for c in checks if c.status == WARN]

    print("=" * 68)
    print(" Autobot Preflight Diagnostics")
    print("=" * 68)
    for c in checks:
        print(c.render())
    print("-" * 68)

    if fails:
        print(f"{len(fails)} blocking problem(s), {len(warns)} warning(s).")
        print("\nFix the [FAIL] items above before running a task - the agent")
        print("cannot work until they are resolved. Warnings reduce capability")
        print("but still allow a run.")
        return 1

    if warns:
        print(f"No blocking problems. {len(warns)} warning(s) - reduced capability.")
    else:
        print("All checks passed. Autobot is ready to run.")
    print("\nNext: try the cheapest possible end-to-end test first -")
    print('  autobot "open Notepad and type hello"')
    print("It exercises window focus, native UI extraction, and computer_call")
    print("in ~4 steps, so a failure points at one layer instead of nine.")
    return 0


def main() -> int:
    return report(run_all())
