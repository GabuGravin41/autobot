"""
Autobot — A sovereign digital agent with full control of your computer.

Core components:
    - Agent: Observe → think → act → verify loop with evaluation, stop conditions,
             perpetual mode, and multi-task scheduling
    - Computer: OS-level control (mouse, keyboard, display, clipboard, anti-sleep)
    - Browser: Chrome CDP via Playwright in human-profile mode
    - Web: FastAPI backend with task queue, run history, and approval gating
"""

__version__ = "0.1.0"
