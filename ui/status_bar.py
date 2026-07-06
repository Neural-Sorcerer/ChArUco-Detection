"""
Top Status Bar
==============

The header strip: app identity on the left, live status (camera state,
resolution, FPS) on the right. Updated via simple setters from the main window.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel

# === Local ===
from app.config import APP_NAME, APP_VERSION

__all__ = ["TopBar"]

log = logging.getLogger(__name__)


class TopBar(QFrame):
    """Header widget showing the app name and live camera status."""

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("TopBar")

        title = QLabel(f"  {APP_NAME}")
        title_font = title.font()
        title_font.setPointSize(title_font.pointSize() + 5)
        title_font.setBold(True)
        title.setFont(title_font)

        version = QLabel(f"v{APP_VERSION}")
        version.setStyleSheet("color: #7a8a99;")

        self._status = self._value_label("idle")
        self._resolution = self._value_label("-")
        self._fps = self._value_label("0.0")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 12, 4)
        layout.addWidget(title)
        layout.addWidget(version)
        layout.addStretch(1)
        layout.addWidget(self._chip("Status", self._status))
        layout.addWidget(self._chip("Resolution", self._resolution))
        layout.addWidget(self._chip("FPS", self._fps))

    @staticmethod
    def _value_label(text: str) -> QLabel:
        """Create a bold value label used inside a status chip."""
        label = QLabel(text)
        label.setObjectName("StatusValue")
        return label

    @staticmethod
    def _chip(caption: str, value: QLabel) -> QFrame:
        """Wrap a caption + value pair into a compact status chip."""
        chip = QFrame()
        chip_layout = QHBoxLayout(chip)
        chip_layout.setContentsMargins(10, 2, 10, 2)
        caption_label = QLabel(f"{caption}:")
        caption_label.setStyleSheet("color: #7a8a99;")
        chip_layout.addWidget(caption_label)
        chip_layout.addWidget(value)
        return chip

    # --- Live setters ---

    def set_status(self, text: str) -> None:
        """Set the camera status text (e.g. running / paused / stopped)."""
        self._status.setText(text)

    def set_resolution(self, width: int, height: int) -> None:
        """Set the displayed resolution."""
        self._resolution.setText(f"{width}x{height}")

    def set_fps(self, fps: float) -> None:
        """Set the displayed frame rate."""
        self._fps.setText(f"{fps:.1f}")
