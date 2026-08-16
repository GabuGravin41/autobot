import subprocess
import time
import urllib.request
import os

print("1. Force killing stale chrome processes...")
subprocess.run("taskkill /F /IM chrome.exe", shell=True, capture_output=True)
time.sleep(2)

user_data = r"C:\Users\User 1\AppData\Local\Autobot\ChromeAutomationProfile"
lock = os.path.join(user_data, "SingletonLock")
if os.path.exists(lock):
    try:
        os.unlink(lock)
        print("Cleared SingletonLock")
    except Exception as e:
        print("SingletonLock error:", e)

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
cmd_line = f'cmd.exe /c start "" "{chrome_path}" --remote-debugging-port=9222 --remote-allow-origins=* --user-data-dir="{user_data}" --no-first-run --no-default-browser-check https://web.whatsapp.com'
print("2. Spawning Chrome on desktop:", cmd_line)
subprocess.run(cmd_line, shell=True)

for i in range(10):
    time.sleep(1)
    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json/version", timeout=2)
        print(f"3. CDP ACTIVE ON PORT 9222! Version: {req.read().decode()}")
        break
    except Exception as e:
        print(f"Attempt {i+1} waiting for CDP...", e)
