import os
import sys
import time
import subprocess
import pyperclip
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("🧹 Focus Chrome and dismiss popup...")
    ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Chrome')"
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    time.sleep(0.5)
    pyautogui.hotkey("win", "up")
    time.sleep(0.5)
    pyautogui.press("escape")
    time.sleep(0.5)

    prompt = "Literature survey on Polariton-Exciton pairs in perovskite microcavities for optical memory."
    print(f"💬 Sending Turn 1 prompt: {prompt}")

    pyperclip.copy(prompt)
    time.sleep(0.3)

    # Click chat input bar at bottom center
    pyautogui.click(960, 950)
    time.sleep(0.3)
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    pyautogui.press("enter")

    print("⏳ Waiting 12 seconds for Grok response...")
    time.sleep(12.0)

    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/turn1_result.png"
    pyautogui.screenshot().save(out_path)
    print(f"📸 Saved screenshot: {out_path}")

if __name__ == "__main__":
    main()
