"""
Live Image Viewer
=================

A pyqtgraph-backed widget for fast real-time frame display. Uses a single
:class:`pg.ImageItem` in an aspect-locked view; pan/zoom with the mouse for
inspection. Frames are pushed in as RGB ``HxWx3`` uint8 arrays.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Slot

__all__ = ["ImageViewer"]

log = logging.getLogger(__name__)

# Display images the natural way: data[row, col] with the origin at top-left.
pg.setConfigOptions(imageAxisOrder="row-major")


class ImageViewer(pg.GraphicsLayoutWidget):
    """Aspect-locked live viewer for RGB frames."""

    def __init__(self) -> None:
        super().__init__()
        self.setBackground("#101418")

        self._view = self.addViewBox()
        self._view.setAspectLocked(True)
        self._view.invertY(True)          # image origin at top-left
        self._view.setMenuEnabled(False)

        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._view.addItem(self._image_item)

        self._placeholder = pg.TextItem("No signal - press Start", color="#7a8a99", anchor=(0.5, 0.5))
        self._view.addItem(self._placeholder)
        self._has_frame = False

    @Slot(object)
    def update_frame(self, rgb: np.ndarray) -> None:
        """
        Display a new RGB frame.

        Args:
            rgb: An ``HxWx3`` uint8 RGB array
        """
        if not self._has_frame:
            self._view.removeItem(self._placeholder)
            self._has_frame = True
            first = True
        else:
            first = False

        self._image_item.setImage(rgb, autoLevels=False, levels=(0, 255))
        if first:
            self._view.autoRange(padding=0)

    def clear(self) -> None:
        """Drop the current image and restore the placeholder."""
        if self._has_frame:
            self._image_item.clear()
            self._view.addItem(self._placeholder)
            self._has_frame = False
