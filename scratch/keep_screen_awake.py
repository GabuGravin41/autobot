import ctypes
import time
import sys

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002

VK_F15 = 0x7E
KEYEVENTF_KEYUP = 0x0002
MOUSEEVENTF_MOVE = 0x0001

print("Triple-Lock Keep-Screen-Awake Daemon started.")
print("1. SetThreadExecutionState display + system locked.")
print("2. Harmless F15 hardware event emitted to reset user idle timers.")
print("3. Mouse nudge (0,0) emitted to keep display adapter active.")
sys.stdout.flush()

while True:
    try:
        # 1. Thread execution state (both continuous and reset)
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
        )

        # 2. Virtual harmless keypress (F15) to reset OS lock/screensaver timers without typing characters
        ctypes.windll.user32.keybd_event(VK_F15, 0, 0, 0)
        ctypes.windll.user32.keybd_event(VK_F15, 0, KEYEVENTF_KEYUP, 0)

        # 3. Mouse move event (relative 0,0)
        ctypes.windll.user32.mouse_event(MOUSEEVENTF_MOVE, 0, 0, 0, 0)
    except Exception as e:
        print(f"Keep-awake heartbeat error: {e}")
        sys.stdout.flush()

    time.sleep(30)
