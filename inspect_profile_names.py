import os
import sys
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def inspect_profiles():
    user_data = Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data"))
    print("🔍 Reading Chrome Profile Preferences & Signed-In Emails:\n")

    for entry in user_data.iterdir():
        if entry.is_dir() and (entry.name == "Default" or entry.name.startswith("Profile")):
            pref_path = entry / "Preferences"
            if pref_path.exists():
                try:
                    with open(pref_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    
                    profile_name = data.get("profile", {}).get("name", "Unknown")
                    account_info = data.get("account_info", [])
                    email = "None"
                    if account_info and isinstance(account_info, list) and len(account_info) > 0:
                        email = account_info[0].get("email", "Unknown")
                    elif "google" in data and "services" in data["google"]:
                        email = data.get("google", {}).get("services", {}).get("signin", {}).get("user_name", "Unknown")

                    print(f"📁 Directory: '{entry.name}'")
                    print(f"   Profile Name: {profile_name}")
                    print(f"   Email: {email}")
                    print("-" * 50)
                except Exception as e:
                    print(f"📁 Directory: '{entry.name}' (Error reading Preferences: {e})")

if __name__ == "__main__":
    inspect_profiles()
