#!/usr/bin/env python3
"""
Simulate session 2 (follow-up visit): send 4 messages, wait for each LLM response,
then take full-page screenshot.
"""

import asyncio
import sys
from pathlib import Path

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Playwright not installed.")
    sys.exit(1)

MESSAGES = [
    "医生您好，我是上次来看头晕的那位患者，今天来复查了。",
    "上次开的降压药我一直在吃，头晕的症状好了很多。但是最近左边太阳穴偶尔会跳痛。",
    "血压最近自己在家量了，基本在130/85左右，比之前好多了。",
    "血糖也在控制，空腹血糖降到6.5了。不过还是容易疲劳。",
]

OUTPUT_PATH = Path("/Users/bytedance/repo/graduate/Agent-R1/memory_prototype/browser_screenshots/clean_03_session2.png")
WAIT_AFTER_SEND = 13


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 1200})
        page = await context.new_page()

        try:
            print("Navigating to http://127.0.0.1:7860/ ...")
            await page.goto("http://127.0.0.1:7860/", wait_until="networkidle", timeout=15000)
            await asyncio.sleep(2)

            input_box = page.get_by_placeholder("输入患者消息...")
            send_btn = page.get_by_role("button", name="发送")

            for i, msg in enumerate(MESSAGES, 1):
                print(f"Sending message {i}/4: {msg[:40]}...")
                await input_box.fill(msg)
                await asyncio.sleep(0.3)
                await send_btn.click()
                print(f"  Waiting {WAIT_AFTER_SEND}s for LLM response...")
                await asyncio.sleep(WAIT_AFTER_SEND)

            await asyncio.sleep(2)

            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.3)

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(OUTPUT_PATH), full_page=True)
            print(f"Screenshot saved to {OUTPUT_PATH}")

            dd = page.get_by_label("预设会话场景")
            play_btn = page.get_by_role("button", name="自动播放该会话")
            reset_btn = page.get_by_role("button", name="重置系统")
            if await dd.count() == 0 and await play_btn.count() == 0 and await reset_btn.count() == 0:
                print("OK: No developer controls visible")
        except Exception as e:
            print(f"Error: {e}")
            await page.screenshot(path=str(OUTPUT_PATH.parent / "error_session2.png"))
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
