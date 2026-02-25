#!/usr/bin/env python3
"""
Simulate patient dialogue: type 5 messages one by one, wait for each LLM response,
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
    "医生您好，我是李明，今年52岁。最近总觉得头晕，有时候还会耳鸣，已经持续两周了。",
    "平时血压偏高，大概145/95左右，吃过一段时间降压药但没坚持。另外我有糖尿病，空腹血糖7.8。",
    "对了，我对青霉素过敏，以前打针的时候出过皮疹。",
    "最近工作压力比较大，经常熬夜，饮食也不太规律。",
    "好的医生，那我需要做哪些检查呢？",
]

OUTPUT_PATH = Path("/Users/bytedance/repo/graduate/Agent-R1/memory_prototype/browser_screenshots/clean_02_session1.png")
WAIT_AFTER_SEND = 13  # seconds for LLM to respond


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
                print(f"Sending message {i}/5: {msg[:40]}...")
                await input_box.fill(msg)
                await asyncio.sleep(0.3)
                await send_btn.click()
                print(f"  Waiting {WAIT_AFTER_SEND}s for LLM response...")
                await asyncio.sleep(WAIT_AFTER_SEND)

            # Extra wait for last response to fully render
            await asyncio.sleep(2)

            # Scroll to ensure full page is captured
            await page.evaluate("window.scrollTo(0, 0)")
            await asyncio.sleep(0.3)

            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(OUTPUT_PATH), full_page=True)
            print(f"Screenshot saved to {OUTPUT_PATH}")

            # Verify no dev controls
            dd = page.get_by_label("预设会话场景")
            play_btn = page.get_by_role("button", name="自动播放该会话")
            reset_btn = page.get_by_role("button", name="重置系统")
            if await dd.count() == 0 and await play_btn.count() == 0 and await reset_btn.count() == 0:
                print("OK: No developer controls visible")
        except Exception as e:
            print(f"Error: {e}")
            await page.screenshot(path=str(OUTPUT_PATH.parent / "error_session1.png"))
            raise
        finally:
            await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
