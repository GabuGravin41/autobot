import asyncio
from playwright.async_api import async_playwright

async def click_use_here():
    async with async_playwright() as p:
        b = await p.chromium.connect_over_cdp('http://127.0.0.1:9222')
        for ctx in b.contexts:
            for page in ctx.pages:
                url = page.url
                if "whatsapp" in url:
                    print("Found WhatsApp page! Checking for 'Use here' button...")
                    use_here = page.get_by_role("button", name="Use here")
                    if await use_here.is_visible():
                        print("Clicking 'Use here' button...")
                        await use_here.click()
                        await asyncio.sleep(5)
                        text = await page.evaluate("document.body.innerText")
                        print("AFTER CLICK SNIPPET:", text[:500])
                    else:
                        print("'Use here' button not visible directly.")

asyncio.run(click_use_here())
