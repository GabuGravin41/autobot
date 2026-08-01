import os
import sys
import time
import subprocess
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def maximize_and_screenshot(filename):
    # Send Win+Up to maximize active window
    pyautogui.hotkey("win", "up")
    time.sleep(1.0)
    os.makedirs("tmp", exist_ok=True)
    out_path = os.path.join("tmp", filename)
    img = pyautogui.screenshot()
    img.save(out_path)
    print(f"📸 Saved screenshot: {out_path}")

def main():
    print("🖥️ Maximizing Chrome window...")
    # Focus Grok window
    ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Grok')"
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    time.sleep(1.0)
    
    maximize_and_screenshot("grok_fully_loaded.png")

    print("🌐 Navigating to Overleaf in new tab...")
    pyautogui.hotkey("ctrl", "t")
    time.sleep(1.0)
    pyautogui.hotkey("ctrl", "l")
    time.sleep(0.5)
    pyautogui.write("https://www.overleaf.com/project", interval=0.05)
    pyautogui.press("enter")
    time.sleep(5.0)

    maximize_and_screenshot("overleaf_loaded.png")

if __name__ == "__main__":
    main()
