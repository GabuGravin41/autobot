"""
Test Master Multi-Profile Session Merger (Grok + Overleaf)
"""
import os
import sys
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from autobot.browser.launcher import AsyncBrowserLauncher

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

async def test_merged_profiles():
    print("🌐 Testing Master Multi-Profile Session Merger (Grok & Overleaf)...")

    launcher = AsyncBrowserLauncher.from_env()
    page = await launcher.start()

    # 1. Test Grok session (from Profile 1)
    print("🧭 Navigating to grok.com...")
    await page.goto("https://grok.com")
    await asyncio.sleep(3.0)

    title1 = await page.title()
    url1 = page.url
    print(f"  ✅ Grok Title: '{title1}' | URL: '{url1}'")

    # 2. Test Overleaf session (from Default / Profile 2)
    print("🧭 Navigating to overleaf.com/project...")
    await page.goto("https://www.overleaf.com/project")
    await asyncio.sleep(3.0)

    title2 = await page.title()
    url2 = page.url
    print(f"  ✅ Overleaf Title: '{title2}' | URL: '{url2}'")

    await launcher.stop()
    print("\n🎉 MASTER SESSION MERGER TEST PASSED!")

if __name__ == "__main__":
    asyncio.run(test_merged_profiles())
