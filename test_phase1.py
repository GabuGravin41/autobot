"""
Phase 1 Test — Launch Chrome with user's real profile and verify Grok access.

Strategy A: Windows `start` command + CDP on port 9222
Strategy B: Launch Chrome normally + visual profile selection via mouse click
Strategy C: Playwright launch_persistent_context with channel="chrome"
"""
import asyncio
import os
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


async def check_cdp(port=9222, timeout=10):
    """Poll CDP endpoint until ready or timeout."""
    import httpx
    for i in range(timeout):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(f"http://127.0.0.1:{port}/json/version")
                if resp.status_code == 200:
                    browser = resp.json().get("Browser", "unknown")
                    print(f"  ✅ CDP ready on attempt {i+1}! Browser: {browser}")
                    return True
        except Exception:
            pass
        await asyncio.sleep(1.0)
    return False


async def strategy_a():
    """Launch Chrome via Windows `start` command with --remote-debugging-port."""
    print("\n" + "=" * 70)
    print("STRATEGY A: Windows `start` command + CDP")
    print("=" * 70)

    # Kill existing Chrome
    print("🧹 Killing existing Chrome processes...")
    subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
    await asyncio.sleep(1.5)

    # Launch via `start` — this runs Chrome as a normal desktop app
    cmd = f'start "" "{CHROME_EXE}" --remote-debugging-port=9222 --profile-directory=Default --no-first-run'
    print(f"🚀 Launching: {cmd}")
    os.system(cmd)

    # Wait for CDP
    print("⏳ Waiting for CDP on port 9222 (max 10s)...")
    if not await check_cdp(port=9222, timeout=10):
        print("❌ Strategy A FAILED: CDP not available after 10 seconds.")
        return None

    # Connect Playwright
    print("🔌 Connecting Playwright via CDP...")
    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp("http://127.0.0.1:9222")
    context = browser.contexts[0]
    page = context.pages[0] if context.pages else await context.new_page()

    print(f"✅ Strategy A SUCCESS! Page URL: {page.url}")
    return pw, browser, page


async def strategy_b():
    """Launch Chrome normally and visually click the correct profile."""
    print("\n" + "=" * 70)
    print("STRATEGY B: Launch Chrome normally + visual profile click")
    print("=" * 70)

    # Kill existing Chrome
    print("🧹 Killing existing Chrome processes...")
    subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
    await asyncio.sleep(1.5)

    # Launch Chrome with no flags — this triggers the "Who's using Chrome?" picker
    print("🚀 Launching Chrome normally (no CDP flags)...")
    os.system(f'start "" "{CHROME_EXE}"')
    await asyncio.sleep(3.0)

    # Take desktop screenshot
    print("📸 Taking desktop screenshot to find profile picker...")
    try:
        import pyautogui
        screenshot = pyautogui.screenshot()
        screenshot_path = os.path.join("tmp", "profile_picker.png")
        os.makedirs("tmp", exist_ok=True)
        screenshot.save(screenshot_path)
        print(f"  Screenshot saved to: {screenshot_path}")
        print("  ⚠️ Strategy B requires LLM vision to identify profile tile — not yet wired.")
        print("  ❌ Strategy B INCOMPLETE (needs LLM vision integration)")
    except ImportError:
        print("  ❌ pyautogui not installed. Run: pip install pyautogui")
    
    return None


async def strategy_c():
    """Use Playwright launch_persistent_context with channel='chrome'."""
    print("\n" + "=" * 70)
    print("STRATEGY C: Playwright launch_persistent_context + channel='chrome'")
    print("=" * 70)

    # Kill existing Chrome
    print("🧹 Killing existing Chrome processes...")
    subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
    await asyncio.sleep(1.5)

    user_data = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data")
    print(f"🌐 Launching with channel='chrome', user_data='{user_data}', profile='Default'...")

    from playwright.async_api import async_playwright
    pw = await async_playwright().start()
    
    try:
        context = await pw.chromium.launch_persistent_context(
            user_data_dir=user_data,
            channel="chrome",
            headless=False,
            args=["--profile-directory=Default", "--no-first-run", "--no-default-browser-check"],
        )
        page = context.pages[0] if context.pages else await context.new_page()
        print(f"✅ Strategy C SUCCESS! Page URL: {page.url}")
        return pw, None, page
    except Exception as e:
        print(f"❌ Strategy C FAILED: {e}")
        await pw.stop()
        return None


async def verify_grok(page):
    """Navigate to grok.com and check if user is logged in."""
    print("\n" + "-" * 70)
    print("VERIFICATION: Navigate to grok.com and check login status")
    print("-" * 70)

    print("🧭 Navigating to https://grok.com ...")
    await page.goto("https://grok.com", wait_until="domcontentloaded")
    await asyncio.sleep(3.0)

    title = await page.title()
    url = page.url
    print(f"  📄 Title: '{title}'")
    print(f"  🔗 URL: {url}")

    # Take screenshot for visual verification
    screenshot_bytes = await page.screenshot(type="png")
    os.makedirs("tmp", exist_ok=True)
    with open("tmp/grok_verification.png", "wb") as f:
        f.write(screenshot_bytes)
    print(f"  📸 Screenshot saved: tmp/grok_verification.png ({len(screenshot_bytes)} bytes)")

    # Check DOM for login indicators
    content = await page.content()
    has_sign_in = "Sign in" in content or "sign-in" in content.lower()
    has_chat_input = "Ask Grok" in content or "textarea" in content.lower()

    if has_chat_input and not has_sign_in:
        print("  🎉 LOGGED IN! Chat input detected.")
        return True
    elif has_sign_in:
        print("  ⚠️ NOT LOGGED IN — 'Sign in' button found.")
        return False
    else:
        print(f"  🤔 UNCLEAR — no definitive indicators. Check screenshot manually.")
        return False


async def main():
    print("=" * 70)
    print("🧪 PHASE 1 TEST: Launch Chrome with Real Profile")
    print("=" * 70)

    # Try Strategy A
    result = await strategy_a()
    
    if result is None:
        # Try Strategy C (skip B for now since it needs pyautogui + LLM vision)
        result = await strategy_c()
    
    if result is None:
        print("\n❌ ALL STRATEGIES FAILED. Cannot open Chrome with user profile.")
        print("   Next step: Install pyautogui and implement Strategy B (visual click).")
        return

    pw, browser, page = result

    # Verify Grok access
    logged_in = await verify_grok(page)

    if logged_in:
        print("\n" + "=" * 70)
        print("✅ PHASE 1 PASSED — Chrome open, user logged in, Grok accessible!")
        print("   Ready for Phase 2: Multi-turn Grok research.")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠️ PHASE 1 PARTIAL — Chrome open but not logged in to Grok.")
        print("   Need to handle sign-in flow (Strategy B visual click or OAuth).")
        print("=" * 70)

    # Cleanup
    await pw.stop()


if __name__ == "__main__":
    asyncio.run(main())
