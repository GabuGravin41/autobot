import asyncio
from playwright.async_api import async_playwright

async def click_all():
    async with async_playwright() as p:
        b = await p.chromium.connect_over_cdp('http://127.0.0.1:9222')
        for ctx in b.contexts:
            for page in ctx.pages:
                if "whatsapp" in page.url:
                    try:
                        btn = page.get_by_role("button", name="Use here")
                        if await btn.is_visible(timeout=2000):
                            print("Clicking 'Use here' on page:", page.url)
                            await btn.click()
                    except Exception as e:
                        print("Error clicking:", e)

asyncio.run(click_all())
