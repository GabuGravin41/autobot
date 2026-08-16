import subprocess
import time
import urllib.request
import json
import os

subprocess.run('taskkill /F /IM chrome.exe', shell=True, capture_output=True)
time.sleep(2)

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
user_data = r"C:\Users\User 1\AppData\Local\Google\Chrome\User Data"

lock_file = os.path.join(user_data, "SingletonLock")
if os.path.exists(lock_file):
    try:
        os.unlink(lock_file)
    except Exception as e:
        print("Could not unlink lock file:", e)

cmd_args = [
    chrome_path,
    "--remote-debugging-port=9222",
    "--remote-allow-origins=*",
    f"--user-data-dir={user_data}",
    "--profile-directory=Default",
    "--no-first-run",
    "--no-default-browser-check"
]

print("Launching via cmd start...")
cmd_line = f'cmd.exe /c start "" "{chrome_path}" --remote-debugging-port=9222 --remote-allow-origins=* --user-data-dir="{user_data}" --profile-directory=Default'
subprocess.run(cmd_line, shell=True)

for i in range(10):
    time.sleep(1)
    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json/version", timeout=2)
        print("SUCCESS ON ATTEMPT", i+1, ":", req.read().decode())
        break
    except Exception as e:
        print(f"Attempt {i+1} failed:", e)
