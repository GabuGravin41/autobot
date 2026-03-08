import sys
import os
import asyncio

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from autobot.computer.computer import Computer

async def test_native():
    computer = Computer()
    
    print("--- Testing Window Listing ---")
    windows = computer.window.list_all()
    print(f"Found {len(windows)} windows.")
    for w in windows[:10]:
        print(f" - {w}")
    
    print("\n--- Testing UI Extraction (Foreground Window) ---")
    # Wait a bit so the user can focus something if they want
    import time
    print("Extracting in 2 seconds... focus a window now!")
    time.sleep(2)
    
    ui_tree = computer.window.extract_ui()
    print("Extracted UI Tree (First 1000 chars):")
    print(ui_tree[:1000])
    
    print("\n--- Tool Catalog Verification ---")
    catalog = computer.get_tool_catalog()
    if "computer.window.click" in catalog and "computer.window.extract_ui" in catalog:
        print("[OK] Tool catalog correctly includes native window tools.")
    else:
        print("[FAIL] Tool catalog is missing native window tools.")

if __name__ == "__main__":
    asyncio.run(test_native())
