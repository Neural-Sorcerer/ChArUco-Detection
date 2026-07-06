"""
Calibration Readiness Panel
===========================

Native Qt rendering of the data-quality judge's guidance dashboard (the same
information the CLI draws with OpenCV in ``DataQualityJudge.render_panel``):
current-view verdict, sample count, live reprojection error + focus, per-axis
coverage bars and a corner-coverage mini-map, plus the next-action hint.

It is a pure view: :meth:`update_snapshot` is fed a :class:`JudgmentSnapshot`
(built off the capture thread) and repaints - it holds no judge references.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
import numpy as np
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QGroupBox, QLabel, QProgressBar, QSizePolicy, QVBoxLayout, QWidget,
)

# === Local ===
from core.quality_collector import JudgmentSnapshot

__all__ = ["JudgmentPanel"]

log = logging.getLogger(__name__)

# Shared verdict/coverage colours (green good, amber marginal, red poor).
_GREEN = QColor("#37d67a")
_AMBER = QColor("#f0a83c")
_RED = QColor("#e35d6a")
_GREY = QColor("#8a97a5")
_TRACK = QColor("#39424d")


def _grade(value: float, good: float = 0.8, ok: float = 0.5) -> QColor:
    """Return the colour for a 0..1 score (green / amber / red by threshold)."""
    if value >= good:
        return _GREEN
    if value >= ok:
        return _AMBER
    return _RED


def _jet(t: float) -> QColor:
    """Approximate the OpenCV JET colormap for a normalized value ``t`` in [0, 1]."""
    t = float(np.clip(t, 0.0, 1.0))
    r = np.clip(min(4 * t - 1.5, -4 * t + 4.5), 0.0, 1.0)
    g = np.clip(min(4 * t - 0.5, -4 * t + 3.5), 0.0, 1.0)
    b = np.clip(min(4 * t + 0.5, -4 * t + 2.5), 0.0, 1.0)
    return QColor(int(r * 255), int(g * 255), int(b * 255))


class _CoverageBar(QWidget):
    """A labelled, colour-graded progress bar for one coverage axis."""

    def __init__(self, name: str) -> None:
        super().__init__()
        self._name = name
        self._value = 0.0
        self.setMinimumHeight(38)

    def set_value(self, value: float) -> None:
        """Set the 0..1 coverage value and repaint."""
        self._value = float(np.clip(value, 0.0, 1.0))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        """Draw the caption + a filled track coloured by the value."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        painter.setPen(QColor("#e6ebf0"))
        painter.drawText(0, 14, f"{self._name}: {self._value:.0%}")

        track = self.rect().adjusted(0, 22, -2, -6)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(_TRACK))
        painter.drawRoundedRect(track, 4, 4)

        fill = track.adjusted(0, 0, 0, 0)
        fill.setWidth(int(track.width() * self._value))
        painter.setBrush(QBrush(_grade(self._value)))
        painter.drawRoundedRect(fill, 4, 4)
        painter.end()


class _MiniMap(QWidget):
    """Corner-coverage heatmap on a camera-aspect grid of square cells (JET; grey = untouched).

    The judge tracks coverage on a coarse grid; this view resamples it onto a grid
    whose columns:rows match the camera aspect (e.g. 16:9 → 16×9) so every cell is
    the same square size and the map reads like the camera frame.
    """

    _BASE_CELLS = 16   # cells along the longer axis; the other axis scales by aspect

    def __init__(self) -> None:
        super().__init__()
        self._counts = np.zeros((6, 9), dtype=np.int32)
        self._cols = 16
        self._rows = 9
        self.setMinimumHeight(160)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_aspect(self, width: int, height: int) -> None:
        """Shape the display grid to the camera aspect ratio, keeping cells square."""
        if width <= 0 or height <= 0:
            return
        if width >= height:
            self._cols = self._BASE_CELLS
            self._rows = max(1, round(self._BASE_CELLS * height / width))
        else:
            self._rows = self._BASE_CELLS
            self._cols = max(1, round(self._BASE_CELLS * width / height))
        self.update()

    def set_counts(self, counts: np.ndarray) -> None:
        """Set the coarse per-cell corner counts and repaint."""
        self._counts = counts
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802 (Qt override)
        """Resample the coverage onto the aspect grid and paint square, colour-graded cells."""
        painter = QPainter(self)
        src_rows, src_cols = self._counts.shape
        if src_rows == 0 or src_cols == 0:
            painter.end()
            return

        # Nearest-neighbour resample of the coarse coverage onto the display grid
        # (keeps empty cells exactly zero so "grey = untouched" stays crisp).
        row_idx = (np.arange(self._rows) * src_rows // self._rows).clip(0, src_rows - 1)
        col_idx = (np.arange(self._cols) * src_cols // self._cols).clip(0, src_cols - 1)
        grid = self._counts[row_idx][:, col_idx]
        peak = max(1, int(grid.max()))

        # One square cell size for both axes; the grid is centred (may letterbox).
        cell = max(1, int(min(self.width() / self._cols, self.height() / self._rows)))
        x0 = (self.width() - cell * self._cols) // 2
        y0 = (self.height() - cell * self._rows) // 2

        for r in range(self._rows):
            for c in range(self._cols):
                count = int(grid[r, c])
                colour = QColor("#37414c") if count == 0 else _jet(count / peak)
                x, y = x0 + c * cell, y0 + r * cell
                painter.fillRect(x, y, cell, cell, colour)
                painter.setPen(QPen(QColor("#232a31"), 1))
                painter.drawRect(x, y, cell, cell)
        painter.end()


class JudgmentPanel(QGroupBox):
    """Native readiness dashboard fed by :class:`JudgmentSnapshot` updates."""

    def __init__(self) -> None:
        super().__init__("Calibration readiness")
        self.setMinimumWidth(280)
        self.setMaximumWidth(360)

        self._status = QLabel("Idle - press Start")
        self._status.setWordWrap(True)
        status_font = self._status.font()
        status_font.setPointSize(status_font.pointSize() + 1)
        status_font.setBold(True)
        self._status.setFont(status_font)

        self._samples = QLabel("Samples: 0 / 0")
        self._samples.setObjectName("StatusValue")
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(10)

        self._rms = QLabel("Live RMS: -")
        self._focus = QLabel("Focus: -")

        self._bars = {
            "position": _CoverageBar("Position (where)"),
            "zoom_in": _CoverageBar("Zoom-in (near)"),
            "zoom_out": _CoverageBar("Zoom-out (far)"),
            "tilt": _CoverageBar("Tilt (angle)"),
        }

        self._minimap = _MiniMap()

        self._hint = QLabel("NEXT: -")
        self._hint.setWordWrap(True)
        self._hint.setStyleSheet("color: #ffcf5c; font-weight: bold;")

        self._footer = QLabel("Keep collecting...")
        self._footer.setStyleSheet("color: #8a97a5;")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)
        layout.addWidget(self._status)
        layout.addWidget(self._samples)
        layout.addWidget(self._progress)
        layout.addWidget(self._rms)
        layout.addWidget(self._focus)
        layout.addSpacing(4)
        for bar in self._bars.values():
            layout.addWidget(bar)
        layout.addSpacing(4)
        layout.addWidget(self._minimap)
        layout.addStretch(1)
        layout.addWidget(self._hint)
        layout.addWidget(self._footer)

    def set_aspect(self, width: int, height: int) -> None:
        """Shape the coverage heatmap grid to the camera aspect ratio (square cells)."""
        self._minimap.set_aspect(width, height)

    @Slot(object)
    def update_snapshot(self, snap: JudgmentSnapshot) -> None:
        """Repaint the whole dashboard from a fresh :class:`JudgmentSnapshot`."""
        if snap.status == "none":
            self._set_status("No board detected", _RED)
        elif snap.status == "good":
            self._set_status("GOOD - press Save to keep", _GREEN)
        else:
            self._set_status(f"SKIP: {snap.reject_reason}", _AMBER)

        self._samples.setText(f"Samples: {snap.accepted} / {snap.target}")
        ratio = snap.accepted / snap.target if snap.target else 0.0
        self._progress.setValue(int(min(ratio, 1.0) * 100))

        if snap.live_rms is None:
            self._rms.setText("Live RMS: warming up...")
            self._rms.setStyleSheet("color: #8a97a5;")
        else:
            colour = _GREEN if snap.live_rms < 0.7 else _AMBER if snap.live_rms < 1.5 else _RED
            self._rms.setText(f"Live RMS: {snap.live_rms:.2f} px")
            self._rms.setStyleSheet(f"color: {colour.name()}; font-weight: bold;")

        if snap.sharpness >= 0:
            blurry = snap.min_sharpness > 0 and snap.sharpness < snap.min_sharpness
            threshold = f" / min {snap.min_sharpness:.0f}" if snap.min_sharpness > 0 else ""
            self._focus.setText(f"Focus: {snap.sharpness:.0f}{threshold}" + (" BLUR" if blurry else ""))
            self._focus.setStyleSheet(f"color: {(_AMBER if blurry else _GREY).name()};")
        else:
            self._focus.setText("Focus: -")
            self._focus.setStyleSheet("color: #8a97a5;")

        for key, bar in self._bars.items():
            bar.set_value(snap.coverage.get(key, 0.0))
        self._minimap.set_counts(snap.cell_counts)

        self._hint.setText(f"NEXT: {snap.next_hint}")
        if snap.is_sufficient:
            self._footer.setText("READY TO CALIBRATE")
            self._footer.setStyleSheet(f"color: {_GREEN.name()}; font-weight: bold;")
        else:
            self._footer.setText("Keep collecting...")
            self._footer.setStyleSheet("color: #8a97a5;")

    def reset(self) -> None:
        """Return the panel to its idle look (no active session)."""
        self._set_status("Idle - press Start", _GREY)
        self._samples.setText("Samples: 0 / 0")
        self._progress.setValue(0)
        self._rms.setText("Live RMS: -")
        self._rms.setStyleSheet("color: #8a97a5;")
        self._focus.setText("Focus: -")
        self._focus.setStyleSheet("color: #8a97a5;")
        for bar in self._bars.values():
            bar.set_value(0.0)
        self._minimap.set_counts(np.zeros((6, 9), dtype=np.int32))
        self._hint.setText("NEXT: -")
        self._footer.setText("Keep collecting...")
        self._footer.setStyleSheet("color: #8a97a5;")

    def _set_status(self, text: str, colour: QColor) -> None:
        """Set the current-view verdict text and colour."""
        self._status.setText(text)
        self._status.setStyleSheet(f"color: {colour.name()};")
