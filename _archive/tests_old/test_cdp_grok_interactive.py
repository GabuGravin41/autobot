import os
import sys
import time
import asyncio
from playwright.async_api import async_playwright

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

async def main():
    print("🚀 Launching Chrome with --remote-debugging-port=9222 & --profile-directory='Default'...")
    os.system(f'start "" "{CHROME_EXE}" --profile-directory="Default" --remote-debugging-port=9222 "https://grok.com"')
    await asyncio.sleep(5.0)

    print("🔌 Connecting Playwright over CDP to http://127.0.0.1:9222...")
    async with async_playwright() as p:
        try:
            browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            print("✅ CDP Connected successfully!")

            context = browser.contexts[0]
            pages = context.pages
            page = pages[0] if pages else await context.new_page()

            print(f"📄 Active Page URL: {page.url}")
            if "grok.com" not in page.url:
                await page.goto("https://grok.com")

            print("⏳ Waiting for Grok input box (textarea)...")
            textarea = await page.wait_for_selector("textarea, [contenteditable='true']", timeout=20000)
            print("🎯 Found Grok input box!")

            prompt = "Literature survey on Polariton-Exciton pairs in perovskite microcavities for optical memory."
            print(f"💬 Filling prompt: {prompt}")
            await textarea.fill(prompt)
            await page.keyboard.press("Enter")

            print("⏳ Waiting 15s for Grok response...")
            await asyncio.sleep(15.0)

            os.makedirs("tmp", exist_ok=True)
            shot = "tmp/cdp_grok_response.png"
            await page.screenshot(path=shot)
            print(f"📸 Screenshot saved to {shot}")

            # Check if response text exists on page
            content = await page.content()
            if "polariton" in content.lower() or "perovskite" in content.lower():
                print("🎉 SUCCESS: Verified Grok response content in DOM!")
            else:
                print("⚠️ WARNING: Could not verify response text in DOM.")

        except Exception as e:
            print(f"❌ CDP Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
