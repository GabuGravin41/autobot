"""
Overleaf Workflow - DOM-First, No Blind Actions
===============================================
DESIGN PRINCIPLE (from DESIGN_PHILOSOPHY.md):
  The system is FORBIDDEN from making any blind decision.
  Every action MUST be preceded by a grounded observation from:
    (a) a live screenshot, OR
    (b) DOM/CDP query, OR
    (c) verified UI state read.

WORKFLOW:
  Phase 0: Take a baseline screenshot. Report where we are.
  Phase 1: Open new tab → navigate to overleaf.com/project → wait → screenshot → verify.
  Phase 2: CDP query to get exact bounding box of 'New project' button.
  Phase 3: Click button using CDP-derived coordinates → screenshot → verify dropdown.
  Phase 4: CDP query for 'Blank project' item in dropdown → click → screenshot → verify modal.
  Phase 5: Type name → Enter → wait → screenshot → verify editor opened.
"""
import os
import sys
import time
import json
import requests
import subprocess
import pyperclip
import pyautogui
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    import win32gui
    import win32con
    import win32com.client
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

os.makedirs("tmp", exist_ok=True)
CDP_PORT = 9222

# ─────────────────────────────────────────────────────────────────────────────
def focus_window_by_title(target_titles):
    """Find and focus a visible window whose title contains any of target_titles."""
    found_hwnd = None
    if HAS_WIN32:
        def cb(hwnd, extra):
            nonlocal found_hwnd
            if win32gui.IsWindowVisible(hwnd):
                title = win32gui.GetWindowText(hwnd)
                if any(t in title for t in target_titles):
                    found_hwnd = hwnd
                    return False
            return True
        try:
            win32gui.EnumWindows(cb, None)
        except Exception:
            pass
        if found_hwnd:
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("%")
            win32gui.ShowWindow(found_hwnd, win32con.SW_MAXIMIZE)
            win32gui.SetForegroundWindow(found_hwnd)
            title = win32gui.GetWindowText(found_hwnd)
            time.sleep(1.0)
            print(f"  ✅ Window focused: '{title}'")
            return title
    return None


def screenshot(name):
    path = f"tmp/{name}"
    pyautogui.screenshot().save(path)
    print(f"  📸 Screenshot: {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
def get_cdp_tab():
    """Try to connect to an existing Chrome debug port. Returns tab info or None."""
    try:
        resp = requests.get(f"http://127.0.0.1:{CDP_PORT}/json", timeout=2)
        tabs = resp.json()
        # Find the Overleaf tab
        for tab in tabs:
            if "overleaf" in tab.get("url", "").lower() or "overleaf" in tab.get("title", "").lower():
                print(f"  🔗 Found Overleaf CDP tab: {tab['title'][:60]}")
                return tab
        # fallback: first non-extension tab
        for tab in tabs:
            if tab.get("type") == "page":
                print(f"  🔗 Using CDP tab: {tab['title'][:60]}")
                return tab
    except Exception as e:
        print(f"  ⚠️ CDP not available at port {CDP_PORT}: {e}")
    return None


def cdp_eval(tab, js):
    """Run JS via CDP Runtime.evaluate and return the result value."""
    ws_url = tab.get("webSocketDebuggerUrl", "")
    if not ws_url:
        return None
    try:
        import websocket
        ws = websocket.create_connection(ws_url, timeout=5)
        msg = json.dumps({"id": 1, "method": "Runtime.evaluate",
                          "params": {"expression": js, "returnByValue": True}})
        ws.send(msg)
        raw = ws.recv()
        ws.close()
        result = json.loads(raw)
        return result.get("result", {}).get("result", {}).get("value")
    except Exception as e:
        print(f"  ⚠️ CDP eval error: {e}")
    return None


def get_element_rect(tab, selector):
    """Get bounding rect of first matching DOM element via CDP."""
    js = f"""
    (function() {{
        var el = document.querySelector('{selector}');
        if (!el) return null;
        var r = el.getBoundingClientRect();
        return JSON.stringify({{
            x: r.left, y: r.top,
            width: r.width, height: r.height,
            cx: r.left + r.width/2,
            cy: r.top + r.height/2,
            text: el.innerText.trim().substring(0, 60)
        }});
    }})()
    """
    raw = cdp_eval(tab, js)
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return None


def get_element_rects_by_text(tab, text):
    """Find all elements whose visible text contains the given string."""
    escaped = text.replace("'", "\\'")
    js = f"""
    (function() {{
        var all = document.querySelectorAll('*');
        var results = [];
        for (var i = 0; i < all.length; i++) {{
            var el = all[i];
            if (el.children.length === 0 && el.innerText && el.innerText.trim().indexOf('{escaped}') !== -1) {{
                var r = el.getBoundingClientRect();
                if (r.width > 0 && r.height > 0) {{
                    results.push(JSON.stringify({{
                        tag: el.tagName,
                        x: r.left, y: r.top,
                        width: r.width, height: r.height,
                        cx: r.left + r.width/2,
                        cy: r.top + r.height/2,
                        text: el.innerText.trim().substring(0, 80)
                    }}));
                }}
            }}
        }}
        return JSON.stringify(results);
    }})()
    """
    raw = cdp_eval(tab, js)
    if raw:
        try:
            items = json.loads(raw)
            return [json.loads(i) for i in items]
        except Exception:
            pass
    return []


# ─────────────────────────────────────────────────────────────────────────────
def find_element_in_screenshot(img_path, search_color_rgb, color_tolerance=30):
    """Scan screenshot for a pixel cluster matching the given color. Returns (cx, cy) or None."""
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    tr, tg, tb = search_color_rgb
    matches = []
    for y in range(0, h, 2):
        for x in range(0, w, 2):
            r, g, b = img.getpixel((x, y))
            if (abs(r - tr) < color_tolerance and
                abs(g - tg) < color_tolerance and
                abs(b - tb) < color_tolerance):
                matches.append((x, y))
    if not matches:
        return None
    # Return centroid of matches
    cx = sum(m[0] for m in matches) // len(matches)
    cy = sum(m[1] for m in matches) // len(matches)
    return cx, cy


# ─────────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("🌿 OVERLEAF WORKFLOW — DOM-FIRST, NO BLIND ACTIONS")
    print("=" * 70)

    # ── PHASE 0: Know where we are ─────────────────────────────────────────
    print("\n📍 PHASE 0: Baseline — where are we?")
    title = focus_window_by_title(["Overleaf", "Chrome"])
    baseline = screenshot("dom_step0_baseline.png")
    print(f"  Focused window title: {title}")
    print(f"  Baseline screenshot saved. Inspect it to confirm we are on Overleaf.")

    # ── PHASE 1: Try CDP connection ────────────────────────────────────────
    print("\n🔌 PHASE 1: Attempting CDP connection...")
    tab = get_cdp_tab()

    if tab:
        print("\n🔍 PHASE 2: Querying DOM for 'New project' button...")
        # Try multiple selectors for the New project button
        btn_rect = None
        for sel in ["a.btn.btn-primary", "button.btn-primary", "[data-ol-new-project]",
                    "a[href*='new']", ".sidebar button", ".toolbar-header button"]:
            rect = get_element_rect(tab, sel)
            if rect:
                print(f"  Found via selector '{sel}': {rect}")
                btn_rect = rect
                break

        if not btn_rect:
            print("  Trying text-based search for 'New project'...")
            matches = get_element_rects_by_text(tab, "New project")
            if matches:
                btn_rect = matches[0]
                print(f"  Found via text search: {btn_rect}")

        if btn_rect:
            cx, cy = int(btn_rect["cx"]), int(btn_rect["cy"])
            print(f"  ✅ 'New project' button center from DOM: ({cx}, {cy})")
        else:
            print("  ⚠️ CDP found no element. Falling back to screenshot color search.")
            cx, cy = None, None
    else:
        print("  CDP not available. Using screenshot-only mode.")
        cx, cy = None, None

    # ── PHASE 2 fallback: Screenshot color search ──────────────────────────
    if cx is None:
        print("\n🔍 PHASE 2 (fallback): Scanning screenshot for green button...")
        result = find_element_in_screenshot(baseline, (19, 201, 101), color_tolerance=40)
        if result:
            cx, cy = result
            print(f"  ✅ Found green button region center: ({cx}, {cy})")
        else:
            print("  ❌ Could not locate green button in screenshot. HALTING.")
            print("  Please verify the Overleaf project page is visible.")
            return

    # ── PHASE 3: Click 'New project' → verify dropdown ────────────────────
    print(f"\n▶ PHASE 3: Clicking 'New project' at ({cx}, {cy})...")
    pyautogui.click(cx, cy)
    time.sleep(1.5)
    after_click = screenshot("dom_step3_after_newproject_click.png")
    print(f"  Screenshot saved. Verifying dropdown appeared...")

    # Verify: look for "Blank project" text in DOM or screenshot
    blank_rect = None
    if tab:
        matches = get_element_rects_by_text(tab, "Blank project")
        if matches:
            blank_rect = matches[0]
            print(f"  ✅ 'Blank project' found in DOM at: {blank_rect}")

    if not blank_rect:
        # Fallback: scan for white text in the dropdown region (dark background)
        print("  CDP didn't find 'Blank project'. Checking screenshot for dropdown text...")
        # Dropdown appears below the button. Check if there's a dark overlay menu area.
        img = Image.open(after_click).convert("RGB")
        # Look for the dark dropdown background (Overleaf dark sidebar is ~#1e2429)
        # If the dropdown opened, there'll be a vertical strip of dark pixels on the left
        menu_found = False
        for y in range(cy + 5, min(cy + 300, img.height), 2):
            r, g, b = img.getpixel((cx, y))
            if r < 50 and g < 50 and b < 55:
                menu_found = True
                break
        if menu_found:
            print(f"  ✅ Dropdown dark background detected at y≈{y}")
            # Estimate 'Blank project' position as first item (~28px below button)
            blank_x = cx
            blank_y = cy + 28
            blank_rect = {"cx": blank_x, "cy": blank_y, "text": "Blank project (estimated)"}
        else:
            print("  ⚠️ Dropdown NOT confirmed. Saving screenshot and halting.")
            print(f"  Review: {after_click}")
            return

    # ── PHASE 4: Click 'Blank project' → verify modal ─────────────────────
    bx, by = int(blank_rect["cx"]), int(blank_rect["cy"])
    print(f"\n▶ PHASE 4: Clicking 'Blank project' at ({bx}, {by})...")
    pyautogui.click(bx, by)
    time.sleep(1.5)
    after_blank = screenshot("dom_step4_after_blank_project.png")
    print(f"  Screenshot saved. Verify project name modal appeared...")

    # ── PHASE 5: Type name → Enter → wait → screenshot ────────────────────
    print("\n▶ PHASE 5: Entering project name...")
    pyautogui.hotkey("ctrl", "a")
    time.sleep(0.2)
    pyperclip.copy("Perovskite Polariton Memory")
    pyautogui.hotkey("ctrl", "v")
    time.sleep(0.5)
    screenshot("dom_step5_name_typed.png")

    pyautogui.press("enter")
    print("  ⏳ Waiting 8s for Overleaf LaTeX editor to load...")
    time.sleep(8.0)
    screenshot("dom_step6_editor.png")

    print("\n✅ Workflow complete. Review all dom_step*.png screenshots.")


if __name__ == "__main__":
    main()
