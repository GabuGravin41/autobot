import os
import sys
import time
import subprocess
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def main():
    print("🌐 Navigating to Overleaf projects in default profile (daltonomondi588@gmail.com)...")
    # Focus Chrome window
    ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Chrome')"
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    time.sleep(0.5)

    pyautogui.hotkey("ctrl", "t")
    time.sleep(0.8)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.3)
    pyautogui.write("https://www.overleaf.com/project", interval=0.03)
    pyautogui.press("enter")
    time.sleep(5.0)

    # Maximize window if needed
    pyautogui.hotkey("win", "up")
    time.sleep(1.0)

    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/overleaf_logged_in_check.png"
    img = pyautogui.screenshot()
    img.save(out_path)
    print(f"📸 Saved screenshot: {out_path}")

if __name__ == "__main__":
    main()
