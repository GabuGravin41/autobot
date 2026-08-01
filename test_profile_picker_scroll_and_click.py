import os
import sys
import time
import subprocess
import pyautogui

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def test_direct_launch_default():
    print("🚀 1. Launching Chrome natively with --profile-directory='Default' (daltonomondi588@gmail.com)...")
    cmd = f'start "" "{CHROME_EXE}" --profile-directory="Default" "https://grok.com"'
    os.system(cmd)
    time.sleep(5.0)

    # Bring Chrome window to front using PowerShell
    ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Grok')"
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    time.sleep(1.0)

    # Maximize Chrome window
    pyautogui.hotkey("win", "up")
    time.sleep(1.0)

    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/default_profile_grok.png"
    img = pyautogui.screenshot()
    img.save(out_path)
    print(f"📸 Saved screenshot: {out_path}")

def main():
    test_direct_launch_default()

if __name__ == "__main__":
    main()
