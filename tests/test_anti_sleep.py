import asyncio
import sys
import unittest
from unittest.mock import MagicMock, patch
from autobot.computer.anti_sleep import AntiSleepManager

class TestAntiSleep(unittest.TestCase):
    def test_singleton(self):
        from autobot.computer.anti_sleep import anti_sleep
        self.assertIsInstance(anti_sleep, AntiSleepManager)

    def test_toggle(self):
        mgr = AntiSleepManager(interval_seconds=1)
        self.assertFalse(mgr.enabled)

        # anti_sleep.py imports pyautogui lazily (inside move_mouse()) so a
        # broken/missing display doesn't crash the whole app on import.
        # Substituting a fake module in sys.modules — instead of
        # patch('pyautogui.position', ...), which would force a real import
        # of pyautogui and fail on any host without a live X11/Win32 display
        # — keeps this test runnable in headless/CI environments too.
        mock_pyautogui = MagicMock()
        mock_pyautogui.position.return_value = (100, 100)
        with patch.dict(sys.modules, {'pyautogui': mock_pyautogui}):
            mgr.start()
            self.assertTrue(mgr.enabled)
            # Wait for at least one move
            import time
            time.sleep(1.5)
            mgr.stop()
            self.assertFalse(mgr.enabled)
            self.assertGreaterEqual(mock_pyautogui.moveRel.call_count, 2)

if __name__ == '__main__':
    unittest.main()
