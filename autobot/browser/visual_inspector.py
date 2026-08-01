"""
Visual Inspector — Compares browser screenshots before/after actions to verify visual state change.

Component 41 of the Autobot 2.0 Master Roadmap.
Prevents false-positive actions (e.g. clicking an element that does nothing visually).
"""
import hashlib
import logging

logger = logging.getLogger(__name__)


class VisualStateInspector:
    """
    Inspects visual changes across agent steps using base64 screenshot hashes or pixel diffs.
    """

    def __init__(self) -> None:
        self.previous_hash: str | None = None

    def compute_hash(self, screenshot_b64: str | None) -> str | None:
        """Compute SHA256 hash of screenshot b64 string."""
        if not screenshot_b64:
            return None
        return hashlib.sha256(screenshot_b64.encode("utf-8")).hexdigest()

    def has_visually_changed(self, current_screenshot_b64: str | None) -> bool:
        """
        Check if current screenshot differs from previous step.
        """
        if not current_screenshot_b64:
            return True  # Cannot verify, assume changed

        current_hash = self.compute_hash(current_screenshot_b64)
        if self.previous_hash is None:
            self.previous_hash = current_hash
            return True

        changed = current_hash != self.previous_hash
        self.previous_hash = current_hash

        if not changed:
            logger.warning("👁️ Visual Inspector: Screen did NOT visually change after action!")

        return changed
