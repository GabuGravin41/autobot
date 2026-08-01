"""
Anti-Sleep Manager — Periodically moves the mouse to prevent system sleep.
"""
import logging
import threading
import time
import random
import pyautogui

logger = logging.getLogger(__name__)

class AntiSleepManager:
    def __init__(self, interval_seconds: int = 60):
        self.interval = interval_seconds
        self.enabled = False
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the background anti-sleep thread."""
        if self.enabled:
            return
        
        self.enabled = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AntiSleep")
        self._thread.start()
        logger.info("Anti-sleep mouse mover started.")

    def stop(self):
        """Stop the background anti-sleep thread."""
        self.enabled = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Anti-sleep mouse mover stopped.")

    def _run(self):
        while not self._stop_event.is_set():
            self.move_mouse()
            # Sleep in small increments to respond quickly to stop event
            for _ in range(self.interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

    def move_mouse(self):
        """Perform a subtle mouse movement."""
        try:
            x, y = pyautogui.position()
            # Move 1-2 pixels in a random direction and back
            dx = random.choice([-1, 1])
            dy = random.choice([-1, 1])
            
            pyautogui.moveRel(dx, dy, duration=0.1)
            pyautogui.moveRel(-dx, -dy, duration=0.1)
            logger.debug(f"Anti-sleep: nudged mouse at ({x}, {y})")
        except Exception as e:
            logger.error(f"Anti-sleep: failed to move mouse: {e}")

# Global instance
anti_sleep = AntiSleepManager()
