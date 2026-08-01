"""
System Liveness Manager — Keeps the laptop awake and prevents sleep during active agent runs.

Inspired by Autonomous Vehicle background execution locks.
On Windows, uses kernel32 SetThreadExecutionState to prevent system sleep and screen turn-off.
"""
import logging
import sys

logger = logging.getLogger(__name__)

# Windows Execution State Flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class SystemLivenessManager:
    """
    Prevents the operating system from going to sleep or locking the screen
    while Autobot is executing tasks.
    """

    def __init__(self) -> None:
        self._active = False

    def enable_liveness(self) -> bool:
        """Keep the system awake."""
        if self._active:
            return True

        if sys.platform == "win32":
            try:
                import ctypes
                result = ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
                )
                if result != 0:
                    self._active = True
                    logger.info("⚡ System Liveness Enabled: Laptop sleep prevention active.")
                    return True
            except Exception as e:
                logger.warning(f"Failed to enable Windows liveness: {e}")
        else:
            # Fallback / non-Windows logging
            self._active = True
            logger.info("⚡ System Liveness Enabled (Non-Windows mode).")
            return True

        return False

    def disable_liveness(self) -> bool:
        """Allow system to sleep normally again."""
        if not self._active:
            return True

        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
                self._active = False
                logger.info("💤 System Liveness Disabled: Normal power management restored.")
                return True
            except Exception as e:
                logger.warning(f"Failed to disable Windows liveness: {e}")
        else:
            self._active = False
            logger.info("💤 System Liveness Disabled.")
            return True

        return False

    @property
    def is_active(self) -> bool:
        return self._active
