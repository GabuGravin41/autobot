import asyncio
from playwright.async_api import async_playwright

async def send_whatsapp_test():
    async with async_playwright() as p:
        b = await p.chromium.connect_over_cdp('http://127.0.0.1:9222')
        page = None
        for ctx in b.contexts:
            for pg in ctx.pages:
                if "whatsapp" in pg.url:
                    page = pg
                    break
        if not page:
            print("No WhatsApp page found!")
            return

        print("Target page:", page.url)
        print("Waiting up to 30s for WhatsApp chats/search to load...")
        
        # WhatsApp Web search selectors across various WAWeb versions
        search_selector = 'div[contenteditable="true"][data-tab="3"], #side div[contenteditable="true"], div[title="Search input textbox"], p.selectable-text'
        
        for i in range(15):
            loc = page.locator(search_selector)
            count = await loc.count()
            print(f"Check {i+1}/15: Found {count} search locators")
            if count > 0:
                try:
                    target = loc.first
                    if await target.is_visible():
                        print("Clicking search box...")
                        await target.click()
                        await target.fill("Dalton Omondi")
                        await asyncio.sleep(3)
                        
                        # Look for Dalton Omondi contact
                        contact = page.locator('span[title="Dalton Omondi"], span:has-text("Dalton Omondi")')
                        if await contact.count() > 0:
                            print("Found Dalton Omondi contact! Clicking...")
                            await contact.first.click()
                            await asyncio.sleep(2)
                            
                            # Message input
                            msg_box = page.locator('footer div[contenteditable="true"], div[data-tab="10"]')
                            await msg_box.first.click()
                            await msg_box.first.fill("Hello Dalton! Autobot WhatsApp integration test successful.")
                            await page.keyboard.press("Enter")
                            print("🎉 SUCCESS! MESSAGE SENT TO DALTON OMONDI ON WHATSAPP!")
                            return
                        else:
                            print("Contact 'Dalton Omondi' not found in search results yet.")
                except Exception as e:
                    print(f"Attempt {i+1} interaction error:", e)
            await asyncio.sleep(3)

        text = await page.evaluate("document.body.innerText")
        print("FINAL PAGE TEXT:\n", text[:600])

asyncio.run(send_whatsapp_test())
