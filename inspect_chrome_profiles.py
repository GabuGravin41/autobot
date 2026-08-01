"""
Inspect Chrome Profiles in User Data directory
"""
import sys
import os
import json
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def list_profiles():
    user_data = Path(os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data"))
    print(f"🔍 Inspecting Chrome User Data Directory: '{user_data}'\n")

    if not user_data.exists():
        print("❌ User Data directory not found")
        return

    profiles = []
    # Check Default and Profile 1, Profile 2, etc.
    for folder in user_data.iterdir():
        if folder.is_dir() and (folder.name == "Default" or folder.name.startswith("Profile ")):
            pref_file = folder / "Preferences"
            name = folder.name
            profile_label = folder.name
            if pref_file.exists():
                try:
                    data = json.loads(pref_file.read_text(encoding="utf-8"))
                    profile_label = data.get("profile", {}).get("name", folder.name)
                except Exception:
                    pass
            profiles.append((folder.name, profile_label))

    print(f"Found {len(profiles)} Chrome Profiles:")
    for dir_name, label in profiles:
        print(f"  - Directory: '{dir_name}' | Label/Name: '{label}'")

if __name__ == "__main__":
    list_profiles()
