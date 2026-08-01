import urllib.request
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def check_cdp():
    print("🔍 Checking if CDP port 9222 is currently active...")
    try:
        req = urllib.request.urlopen("http://127.0.0.1:9222/json/version", timeout=3)
        data = json.loads(req.read().decode())
        print("✅ CDP is ACTIVE on port 9222!")
        print(f"   Browser: {data.get('Browser')}")
        print(f"   WebSocket URL: {data.get('webSocketDebuggerUrl')}")
        return True
    except Exception as e:
        print(f"❌ CDP port 9222 is NOT active: {e}")
        return False

if __name__ == "__main__":
    check_cdp()
