#!/usr/bin/env python3
"""Take full-page screenshot of clean Gradio interface (no dev controls)."""

import asyncio
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright not installed.")
    sys.exit(1)

OUTPUT_PATH = Path("/Users/bytedance/repo/graduate/Agent-R1/memory_prototype/browser_screenshots/clean_01_initial.png")


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 1200})
        page = await context.new_page()

        try:
            await page.goto("http://127.0.0.1:7860/", wait_until="networkidle", timeout=15000)
            await asyncio.sleep(2)

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(OUTPUT_PATH), full_page=True)
            print(f"Screenshot saved to {OUTPUT_PATH}")

            # Verify no dev controls visible
            dd = page.get_by_label("预设会话场景")
            play_btn = page.get_by_role("button", name="自动播放该会话")
            reset_btn = page.get_by_role("button", name="重置系统")
            dd_count = await dd.count()
            play_count = await play_btn.count()
            reset_count = await reset_btn.count()
            print(f"Dev controls check: dropdown={dd_count}, auto-play={play_count}, reset={reset_count}")
            if dd_count == 0 and play_count == 0 and reset_count == 0:
                print("OK: No developer controls visible")
            else:
                print("WARNING: Developer controls are still visible")
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
