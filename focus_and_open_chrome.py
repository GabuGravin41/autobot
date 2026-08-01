import os
import sys
import time
import subprocess
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def focus_chrome():
    print("🖥️ Bringing Chrome to front...")
    ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Google Chrome')"
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    time.sleep(1.0)

def main():
    focus_chrome()
    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/chrome_focused.png"
    img = pyautogui.screenshot()
    img.save(out_path)
    print(f"📸 Saved screenshot: {out_path}")

if __name__ == "__main__":
    main()
