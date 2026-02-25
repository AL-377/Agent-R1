#!/usr/bin/env python3
"""
Playwright script to test the Gradio medical dialogue prototype.
1. Navigate to http://127.0.0.1:7860/
2. Take initial screenshot
3. Select "首次就诊：头晕耳鸣" in dropdown, click "自动播放该会话"
4. Wait 30 seconds for LLM processing
5. Take second screenshot
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
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()

        try:
            # 1. Navigate
            print("Navigating to http://127.0.0.1:7860/ ...")
            await page.goto("http://127.0.0.1:7860/", wait_until="networkidle", timeout=15000)
            await asyncio.sleep(2)

            # 2. Initial screenshot
            initial_path = OUTPUT_DIR / "01_initial.png"
            await page.screenshot(path=str(initial_path))
            print(f"Initial screenshot saved to {initial_path}")

            # 3. Select "首次就诊：头晕耳鸣" in dropdown (default is first session)
            # Open dropdown by clicking label area or the dropdown trigger
            print("Selecting '首次就诊：头晕耳鸣' in dropdown...")
            target_label = "首次就诊：头晕耳鸣"
            # Gradio: find dropdown near "预设会话场景" label
            dd = page.locator("text=预设会话场景").locator("..").locator("button, [role=combobox], select").first
            if await dd.count() > 0:
                await dd.click()
                await asyncio.sleep(0.8)
                await page.get_by_text(target_label, exact=True).click()
                await asyncio.sleep(0.5)

            # 4. Click "自动播放该会话"
            print("Clicking '自动播放该会话' button...")
            play_btn = page.get_by_role("button", name="自动播放该会话")
            await play_btn.click()

            # 5. Wait 30 seconds for first message to process
            print("Waiting 30 seconds for LLM to process first message...")
            await asyncio.sleep(30)

            # 6. Second screenshot
            second_path = OUTPUT_DIR / "02_after_autoplay.png"
            await page.screenshot(path=str(second_path))
            print(f"Second screenshot saved to {second_path}")

            # Check for visible errors
            errors = await page.locator(".error, .gr-error, [class*='error']").all_text_contents()
            if errors:
                print("Errors found on page:", errors)

        except Exception as e:
            print(f"Error: {e}")
            await page.screenshot(path=str(OUTPUT_DIR / "error_screenshot.png"))
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
