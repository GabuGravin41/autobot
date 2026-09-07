"""
Test Direct Chrome Launch with --remote-debugging-port=9222 and --user-data-dir
"""
import subprocess
import time
import asyncio
import sys
import os
import httpx
from playwright.async_api import async_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

async def test_cdp():
    print("🧹 Closing background Chrome processes to release profile lock...")
    subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
    await asyncio.sleep(1.5)

    chrome_exe = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    user_data = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data")

    args = [
        chrome_exe,
        "--remote-debugging-port=9222",
        f"--user-data-dir={user_data}",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    print(f"🌐 Launching Chrome: {' '.join(args[:3])}...")
    proc = subprocess.Popen(args)

    print("⏳ Waiting for CDP endpoint on http://127.0.0.1:9222...")
    cdp_ready = False
    for i in range(15):
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get("http://127.0.0.1:9222/json/version")
                if resp.status_code == 200:
                    print(f"  ✅ CDP Ready on attempt {i+1}! Info: {resp.json().get('Browser')}")
                    cdp_ready = True
                    break
        except Exception:
            await asyncio.sleep(1.0)

    if not cdp_ready:
        print("❌ CDP failed to become ready.")
        return False

    print("🔌 Connecting Playwright via CDP...")
    pw = await async_playwright().start()
    browser = await pw.chromium.connect_over_cdp("http://127.0.0.1:9222")
    context = browser.contexts[0]
    page = context.pages[0] if context.pages else await context.new_page()

    print(f"🎉 SUCCESS! Connected to Real Chrome Profile! Active URL: '{page.url}'")
    await pw.stop()
    return True

if __name__ == "__main__":
    asyncio.run(test_cdp())
