# Autobot Phase 2: Total Computer Control Roadmap

Following the successful delivery of the distribution-ready browser automation baseline, we are now planning Phase 2. The goal is to transition Autobot from a "Browser Agent" to a "Total Computer Agent" that can flexibly navigate and control any application on your machine.

## 1. Native OS Perception (The "Desktop DOM")
Currently, Autobot is "blind" outside the browser, relying only on screenshots and raw mouse/keyboard commands.
- **Implement `NativeExtractionService`**: Use Windows UI Automation (UIA) via `pywinauto` or `uiautomation` to extract a tree of interactive elements from native applications.
- **Indexed Native Elements**: Elements in the active window (buttons, text fields, menus) will be assigned numeric indexes (e.g., `[Desktop 10] Button "Submit"`) and injected into the LLM's state description.
- **OCR Fallback**: For apps without accessibility support, use local OCR (e.g., EasyOCR or Tesseract) to identify text blocks and interactive regions visually.

## 2. Hierarchical Mission Planning
For "Ambitious" tasks like completing a full Kaggle contest, a single loop is not enough.
- **Strategist Layer**: A high-level planner that breaks a "Mission" into "Objectives" (e.g., Phase 1: Data Gathering, Phase 2: Baseline Modeling).
- **Mission Log & Memory**: A persistent log of progress that survives across browser restarts or app switches.
- **Long-Term Memory**: A vector-based memory system (using `ChromaDB` or simple JSON) to remember successful strategies for specific websites or app workflows.

## 3. Advanced OS Muscles
Expand the `Computer` class with high-level adapters:
- **`WindowAdapter`**: `list_windows()`, `focus_window("Notepad")`, `close_app()`.
- **`FileAdapter`**: High-level filesystem management (safe directory traversal, file content analysis).
- **`SystemAdapter`**: Monitor system health, battery, and network status for long-running autonomous tasks.

## 4. Enhanced Flexibility (Brain vs. Muscle)
Move away from site-specific tools toward site-agnostic reasoning:
- **General Tool Discovery**: Allow the LLM to "query" available OS tools dynamically.
- **Self-Correction**: Improved reasoning loops specifically for fixing UI interaction failures (e.g., "The button I clicked didn't trigger a page change, let me try right-clicking or scrolling").

---
> [!NOTE]
> This roadmap aligns with your vision of a flexible, smart system that "figures it out" rather than relying on site-specific Python scripts.
