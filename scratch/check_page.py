import asyncio
from playwright.async_api import async_playwright

async def check():
    async with async_playwright() as p:
        b = await p.chromium.connect_over_cdp('http://127.0.0.1:9222')
        for ctx in b.contexts:
            for page in ctx.pages:
                url = page.url
                title = await page.title()
                print(f"PAGE: '{title}' | {url}")
                if "whatsapp" in url:
                    text = await page.evaluate("document.body.innerText")
                    print("--- WHATSAPP TEXT SNIPPET ---")
                    print(text[:500])

asyncio.run(check())
