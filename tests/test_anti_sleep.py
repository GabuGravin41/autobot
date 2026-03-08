import asyncio
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
        
        with patch('pyautogui.position', return_value=(100, 100)):
            with patch('pyautogui.moveRel') as mock_move:
                mgr.start()
                self.assertTrue(mgr.enabled)
                # Wait for at least one move
                import time
                time.sleep(1.5)
                mgr.stop()
                self.assertFalse(mgr.enabled)
                self.assertGreaterEqual(mock_move.call_count, 2)

if __name__ == '__main__':
    unittest.main()
