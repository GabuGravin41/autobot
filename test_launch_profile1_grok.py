import os
import sys
import time
import subprocess

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def main():
    import pyautogui
    print("🚀 Launching Chrome natively with --profile-directory='Profile 1' to grok.com...")
    cmd = f'start "" "{CHROME_EXE}" --profile-directory="Profile 1" "https://grok.com"'
    os.system(cmd)
    time.sleep(5.0)

    # Bring Chrome window to front using PowerShell
    ps_cmd = "$wshell = New-Object -ComObject WScript.Shell; $wshell.AppActivate('Grok')"
    subprocess.run(["powershell", "-Command", ps_cmd], capture_output=True)
    time.sleep(1.0)

    os.makedirs("tmp", exist_ok=True)
    out_path = "tmp/grok_profile1_result.png"
    img = pyautogui.screenshot()
    img.save(out_path)
    print(f"📸 Saved screenshot: {out_path}")

if __name__ == "__main__":
    main()
