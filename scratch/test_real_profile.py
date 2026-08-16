import subprocess
import time
import urllib.request
import os

print("Killing chrome...")
subprocess.run('taskkill /F /IM chrome.exe', shell=True, capture_output=True)
time.sleep(3)

user_data = r"C:\Users\User 1\AppData\Local\Google\Chrome\User Data"
lock = os.path.join(user_data, "SingletonLock")
if os.path.exists(lock):
    try:
        os.unlink(lock)
        print("Removed SingletonLock")
    except Exception as e:
        print("SingletonLock remove error:", e)

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
cmd = f'cmd.exe /c start "" "{chrome_path}" --remote-debugging-port=9222 --remote-allow-origins=* --user-data-dir="{user_data}" --profile-directory="Default" --no-first-run --no-default-browser-check'
print("Launching Real Chrome Profile:", cmd)
subprocess.run(cmd, shell=True)

for i in range(15):
    time.sleep(1)
    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json/version", timeout=2)
        print(f"CDP SUCCESS ON ATTEMPT {i+1}:", req.read().decode())
        break
    except Exception as e:
        print(f"Attempt {i+1} waiting for CDP...", e)
