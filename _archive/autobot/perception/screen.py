"""
Screen Perception — Fast OS screenshot capture and encoding for LLM vision feedback.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import Tuple

try:
    from PIL import Image, ImageGrab
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

logger = logging.getLogger(__name__)


class ScreenPerception:
    """Captures and encodes native OS desktop screen state."""

    @staticmethod
    def capture_screenshot(max_dimension: int = 1280) -> Tuple[bytes, str]:
        """
        Capture current OS screen and resize for optimal token utilization.

        Returns:
            Tuple of (raw_jpeg_bytes, base64_jpeg_string)
        """
        if not HAS_PIL:
            logger.warning("PIL/ImageGrab not available for screen perception")
            return b"", ""

        try:
            img = ImageGrab.grab()
            # Scale down proportionally if image exceeds max_dimension
            width, height = img.size
            if max(width, height) > max_dimension:
                scale = max_dimension / float(max(width, height))
                new_size = (int(width * scale), int(height * scale))
                img = img.resize(new_size, Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.ANTIALIAS)

            buffer = io.BytesIO()
            img.convert("RGB").save(buffer, format="JPEG", quality=80)
            raw_bytes = buffer.getvalue()
            b64_str = base64.b64encode(raw_bytes).decode("utf-8")
            return raw_bytes, b64_str
        except Exception as e:
            logger.error(f"Failed to capture screen screenshot: {e}")
            return b"", ""
