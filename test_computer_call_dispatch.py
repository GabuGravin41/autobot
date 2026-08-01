"""Behavioral test for autobot/computer/dispatch.py.

Loads the module directly (not via the autobot package) so it runs without
pydantic/playwright/etc. installed. Exercises the happy path, the error
paths, and — most importantly — the security properties.
"""
import asyncio
import importlib.util
import sys
from pathlib import Path

DISPATCH = Path(__file__).resolve().parent / "autobot" / "computer" / "dispatch.py"
spec = importlib.util.spec_from_file_location("dispatch", DISPATCH)
d = importlib.util.module_from_spec(spec)
sys.modules["dispatch"] = d
spec.loader.exec_module(d)


# ── Fake Computer standing in for the real one ────────────────────────────
class FakeMouse:
    def __init__(self): self.calls = []
    def click(self, x, y, button="left", clicks=1):
        self.calls.append((x, y, button, clicks))
        return None
    def position(self): return (7, 9)

class FakeFiles:
    def read(self, path): return "x" * 5000          # tests truncation
    def boom(self, path): raise RuntimeError("disk on fire")

class FakeComputer:
    def __init__(self):
        self.mouse = FakeMouse()
        self.files = FakeFiles()
        self._secret = "should-not-be-reachable"
        self.pwned = False
    def anti_sleep(self): return "awake"


PASS, FAIL = [], []
def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}: {name}" + (f"  — {detail}" if detail and not cond else ""))


async def main():
    c = FakeComputer()
    call = lambda s, **kw: asyncio.run if False else d.dispatch_computer_call(c, s, **kw)

    # ---- happy path: positional + keyword args ----
    ok, res = await d.dispatch_computer_call(c, "computer.mouse.click(x=640, y=400)")
    check("kwargs call succeeds", ok and c.mouse.calls[-1] == (640, 400, "left", 1), res)

    ok, res = await d.dispatch_computer_call(c, "computer.mouse.click(10, 20, 'right')")
    check("positional call succeeds", ok and c.mouse.calls[-1] == (10, 20, "right", 1), res)

    # ---- return value formatting ----
    ok, res = await d.dispatch_computer_call(c, "computer.mouse.position()")
    check("return value surfaced", ok and "7" in res and "9" in res, res)

    ok, res = await d.dispatch_computer_call(c, "computer.mouse.click(x=1, y=2)")
    check("None result renders as OK", ok and res.startswith("OK:"), res)

    # ---- 2-part top-level callable (regression: old regex couldn't parse this) ----
    ok, res = await d.dispatch_computer_call(c, "computer.anti_sleep()")
    check("top-level callable parses", ok and "awake" in res, res)

    # ---- truncation ----
    ok, res = await d.dispatch_computer_call(c, "computer.files.read('/tmp/x')")
    check("long output truncated", ok and len(res) < 2200 and "truncated" in res, f"len={len(res)}")

    # ---- error paths return, never raise ----
    ok, res = await d.dispatch_computer_call(c, "computer.files.boom('/tmp/x')")
    check("method exception -> (False, msg)", (not ok) and "disk on fire" in res, res)

    ok, res = await d.dispatch_computer_call(c, "computer.nosuch.method()")
    check("unknown module -> error", (not ok) and "unknown module" in res.lower(), res)

    ok, res = await d.dispatch_computer_call(c, "computer.mouse.nosuch()")
    check("unknown method -> error", (not ok) and "unknown method" in res.lower(), res)

    ok, res = await d.dispatch_computer_call(c, "os.system('rm -rf /')")
    check("non-computer prefix rejected", (not ok) and "must start with" in res, res)

    ok, res = await d.dispatch_computer_call(c, "computer.mouse.click(")
    check("malformed call rejected", (not ok), res)

    # ---- SECURITY: no arbitrary code execution via arguments ----
    ok, res = await d.dispatch_computer_call(
        c, "computer.mouse.click(x=__import__('os').system('echo pwned'), y=1)")
    check("__import__ in args blocked", (not ok) and "literal" in res.lower(), res)

    ok, res = await d.dispatch_computer_call(c, "computer.mouse.click(x=open('/etc/passwd'), y=1)")
    check("function call in args blocked", (not ok) and "literal" in res.lower(), res)

    # ---- SECURITY: no private/dunder traversal ----
    ok, res = await d.dispatch_computer_call(c, "computer._secret.upper()")
    check("private attr blocked", (not ok) and "private" in res.lower(), res)

    ok, res = await d.dispatch_computer_call(c, "computer.__class__.__init__()")
    check("dunder blocked", (not ok), res)

    # ---- blocked modules (background mode) ----
    ok, res = await d.dispatch_computer_call(
        c, "computer.mouse.click(x=1, y=2)", blocked_modules=d.SCREEN_MODULES)
    check("screen module blocked in bg mode", (not ok) and "BLOCKED" in res, res)

    ok, res = await d.dispatch_computer_call(
        c, "computer.files.read('/tmp/x')", blocked_modules=d.SCREEN_MODULES)
    check("non-screen module allowed in bg mode", ok, res)

    check("no side-effect from injection attempts", c.pwned is False)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", ", ".join(FAIL))
        sys.exit(1)


asyncio.run(main())
