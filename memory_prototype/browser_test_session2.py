#!/usr/bin/env python3
"""
Playwright script: Switch to "复诊随访：降压效果", auto-play, wait 60s, screenshot.
"""

import asyncio
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright not installed. Run: pip install playwright && playwright install chromium")
    sys.exit(1)


OUTPUT_DIR = Path(__file__).parent / "browser_screenshots"
OUTPUT_DIR.mkdir(exist_ok=True)


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 1200})
        page = await context.new_page()

        try:
            print("Navigating to http://127.0.0.1:7860/ ...")
            await page.goto("http://127.0.0.1:7860/", wait_until="networkidle", timeout=15000)
            await asyncio.sleep(2)

            # Reset system first to clear previous session state
            print("Resetting system...")
            reset_btn = page.get_by_role("button", name="重置系统")
            await reset_btn.click()
            await asyncio.sleep(2)

            # 1. Find dropdown "预设会话场景" and change to "复诊随访：降压效果"
            print("Selecting '复诊随访：降压效果' in dropdown...")
            target = "复诊随访：降压效果"
            # Try multiple selectors for Gradio dropdown
            dd = page.get_by_label("预设会话场景")
            if await dd.count() == 0:
                dd = page.locator('[role="combobox"]').first
            if await dd.count() == 0:
                dd = page.locator("button").filter(has=page.locator("text=首次就诊")).first
            if await dd.count() > 0:
                await dd.click()
                await asyncio.sleep(1)
                opt = page.get_by_role("option", name=target)
                if await opt.count() > 0:
                    await opt.click()
                else:
                    await page.get_by_text(target, exact=True).click()
                await asyncio.sleep(0.8)

            # 2. Click "自动播放该会话"
            print("Clicking '自动播放该会话' button...")
            play_btn = page.get_by_role("button", name="自动播放该会话")
            await play_btn.click()

            # 3. Wait 60 seconds for all messages to process
            print("Waiting 60 seconds for all messages to process...")
            await asyncio.sleep(60)

            # 4 & 5. Take full-page screenshot and save to 03_session2.png
            output_path = OUTPUT_DIR / "03_session2.png"
            await page.screenshot(path=str(output_path), full_page=True)
            print(f"Screenshot saved to {output_path}")

        except Exception as e:
            print(f"Error: {e}")
            await page.screenshot(path=str(OUTPUT_DIR / "error_session2.png"))
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
