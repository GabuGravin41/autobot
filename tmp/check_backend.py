import requests
import time

try:
    print("Checking http://127.0.0.1:8000/api/status...")
    start = time.time()
    resp = requests.get("http://127.0.0.1:8000/api/status", timeout=10)
    print(f"Status {resp.status_code} in {time.time() - start:.2f}s")
    print(resp.json())
except Exception as e:
    print(f"Failed: {e}")
