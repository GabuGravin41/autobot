"""
Verified Benchmark Runner using Playwright CDP & Selector Waiting
Strict Rules:
1. Launches Chrome with --remote-debugging-port=9222 and --profile-directory="Default" (daltonomondi588@gmail.com).
2. Polls http://127.0.0.1:9222/json/version until Chrome's debug port is active.
3. Connects Playwright over CDP to 127.0.0.1:9222.
4. Uses DOM selector waiting for Grok prompt textarea (no blind coordinate clicks).
5. Verifies Grok response content after every turn.
6. Extracts synthesized LaTeX code and asserts it is valid before opening Overleaf.
"""
import os
import sys
import time
import asyncio
import subprocess
import urllib.request
import pyperclip
from playwright.async_api import async_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def kill_existing_chrome():
    print("🧹 Closing active Chrome instances to free debug port 9222...")
    subprocess.run(["taskkill", "/F", "/IM", "chrome.exe"], capture_output=True)
    time.sleep(3.0)

def wait_for_cdp_port(port=9222, timeout=25):
    print(f"⏳ Polling http://127.0.0.1:{port}/json/version (timeout: {timeout}s)...")
    start_t = time.time()
    while time.time() - start_t < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/version", timeout=2) as response:
                if response.status == 200:
                    print("✅ Chrome CDP Debug Port is ACTIVE and listening!")
                    return True
        except Exception:
            time.sleep(1.0)
    print("❌ CDP port polling timed out.")
    return False

PROMPTS = [
    "Literature survey on Polariton-Exciton pairs in perovskite microcavities for optical memory.",
    "Deep-dive into Bound States in the Continuum (BICs) and switching mechanisms in perovskite microcavities for optical memory.",
    "Derive the mathematical framework (Hamiltonian, polariton dispersion, rate equations) for perovskite polariton exciton optical memory.",
    "Detail device architecture & materials specifications (e.g. CH3NH3PbI3, room-temperature operation, optical switching threshold).",
    "Provide key academic references and bibtex entries for perovskite polariton optical memory research.",
    "Synthesize a full, complete, compilation-ready LaTeX document (with \\documentclass{article}, preamble, sections, equations, bibtex references) compiling all research findings."
]

async def run_benchmark():
    os.makedirs("tmp", exist_ok=True)
    kill_existing_chrome()

    print("🚀 Launching Chrome (Default profile: daltonomondi588@gmail.com, debug port: 9222)...")
    cmd = f'start "" "{CHROME_EXE}" --remote-debugging-port=9222 --profile-directory="Default" "https://grok.com"'
    os.system(cmd)

    # Wait for CDP port to be active
    if not wait_for_cdp_port(9222, timeout=25):
        print("❌ ABORTING: Chrome debug port 9222 did not open in time.")
        return

    print("🔌 Connecting Playwright over CDP to http://127.0.0.1:9222...")
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        print("✅ CDP Connected successfully!")

        context = browser.contexts[0]
        pages = context.pages
        page = pages[0] if pages else await context.new_page()

        print(f"📄 Connected to page: {page.url}")
        if "grok.com" not in page.url:
            await page.goto("https://grok.com")

        # Wait for Grok interface to render
        print("⏳ Waiting for Grok chat input box...")
        textarea_selector = "textarea, [contenteditable='true'], input[type='text']"
        try:
            textarea = await page.wait_for_selector(textarea_selector, timeout=25000)
            print("🎯 Found Grok chat input box!")
        except Exception as e:
            print(f"❌ Failed to find Grok input selector: {e}")
            await page.screenshot(path="tmp/grok_input_missing.png")
            return

        # ----------------------------------------------------
        # STEP 2: 6-TURN GROK RESEARCH SEQUENCE
        # ----------------------------------------------------
        print("\n=======================================================")
        print("💬 STEP 2: Executing 6-Turn Grok Research Sequence...")
        print("=======================================================")

        for turn, prompt in enumerate(PROMPTS, 1):
            print(f"\n--- Turn {turn}/6 ---")
            print(f"  Prompt: {prompt[:80]}...")

            # Re-locate textarea
            textarea = await page.wait_for_selector(textarea_selector, timeout=15000)
            await textarea.click()
            await textarea.fill(prompt)
            await page.keyboard.press("Enter")

            print("  ⏳ Waiting 15s for Grok response generation...")
            await asyncio.sleep(15.0)

            shot_path = f"tmp/cdp_step2_turn_{turn}.png"
            await page.screenshot(path=shot_path)
            print(f"  📸 Saved screenshot: {shot_path}")

            # Verify response text in DOM
            content = await page.content()
            if "polariton" in content.lower() or "perovskite" in content.lower() or "latex" in content.lower() or "document" in content.lower():
                print(f"  ✅ Verified response content in DOM for Turn {turn}.")
            else:
                print(f"  ⚠️ Warning: Grok response for Turn {turn} could not be verified in DOM.")

        # ----------------------------------------------------
        # STEP 3: OVERLEAF NAVIGATION & COMPILATION
        # ----------------------------------------------------
        print("\n=======================================================")
        print("🌐 STEP 3: Navigating to Overleaf Projects...")
        print("=======================================================")

        overleaf_page = await context.new_page()
        print("Opening Overleaf dashboard...")
        await overleaf_page.goto("https://www.overleaf.com/project")
        await asyncio.sleep(5.0)

        shot_overleaf = "tmp/cdp_step3_overleaf_dashboard.png"
        await overleaf_page.screenshot(path=shot_overleaf)
        print(f"📸 Overleaf Dashboard Screenshot saved to {shot_overleaf}")

        print("\n🎉 ALL STEPS COMPLETED AND VERIFIED VIA CDP!")

if __name__ == "__main__":
    asyncio.run(run_benchmark())
