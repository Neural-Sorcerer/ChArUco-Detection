"""
Runtime Control Bar
===================

The bottom action strip: Start / Pause / Stop the camera, Save a single frame,
and toggle timed collection. Emits one signal per action; :meth:`set_running`
keeps the buttons enabled/disabled consistently with the pipeline state.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QPushButton

__all__ = ["ControlBar"]

log = logging.getLogger(__name__)


class ControlBar(QFrame):
    """Start/Pause/Stop/Save/Collect controls with consistent enablement."""

    start_clicked = Signal()
    pause_toggled = Signal(bool)      # True = paused
    stop_clicked = Signal()
    save_clicked = Signal()
    collection_toggled = Signal(bool)  # True = collecting

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("ControlBar")
        self._quality_mode = False

        self._start = QPushButton("▶  Start")
        self._pause = QPushButton("⏸  Pause")
        self._pause.setCheckable(True)
        self._stop = QPushButton("⏹  Stop")
        self._save = QPushButton("\U0001f4f7  Save frame")
        self._collect = QPushButton("⏺  Start collection")
        self._collect.setCheckable(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        for button in (self._start, self._pause, self._stop):
            button.setMinimumHeight(38)
            layout.addWidget(button)
        layout.addStretch(1)
        for button in (self._save, self._collect):
            button.setMinimumHeight(38)
            layout.addWidget(button)

        self._start.clicked.connect(self.start_clicked)
        self._pause.toggled.connect(self._on_pause_toggled)
        self._stop.clicked.connect(self.stop_clicked)
        self._save.clicked.connect(self.save_clicked)
        self._collect.toggled.connect(self._on_collect_toggled)

        self.set_running(False)

    def _on_pause_toggled(self, checked: bool) -> None:
        """Relabel the pause button and forward the paused state."""
        self._pause.setText("▶  Resume" if checked else "⏸  Pause")
        self.pause_toggled.emit(checked)

    def _on_collect_toggled(self, checked: bool) -> None:
        """Relabel the collect button and forward the collecting state."""
        self._collect.setText("⏹  Stop collection" if checked else "⏺  Start collection")
        self.collection_toggled.emit(checked)

    def set_quality_mode(self, quality: bool) -> None:
        """
        Adapt the controls to the collection mode.

        In quality-guided mode auto-save is driven by the collection panel, so the
        timed-collection toggle is hidden to avoid two switches for one behaviour.

        Args:
            quality: Whether quality-guided collection is active
        """
        self._quality_mode = quality
        running = not self._start.isEnabled()
        self._collect.setVisible(not quality)
        self._collect.setEnabled(running and not quality)
        if quality:
            self._save.setText("💾  Save current view")
            self._save.setToolTip(
                "Manually keep the current view if the judge accepts it "
                "(useful when Auto-save is off)"
            )
        else:
            self._save.setText("💾  Save frame")
            self._save.setToolTip("Save the current frame into the session folder")

    def set_running(self, running: bool) -> None:
        """
        Enable/disable buttons to match the pipeline state.

        Args:
            running: Whether the camera pipeline is currently active
        """
        self._start.setEnabled(not running)
        for button in (self._pause, self._stop, self._save):
            button.setEnabled(running)
        self._collect.setEnabled(running and not self._quality_mode)

        if not running:
            for button in (self._pause, self._collect):
                button.blockSignals(True)
                button.setChecked(False)
                button.blockSignals(False)
            self._pause.setText("⏸  Pause")
            self._collect.setText("⏺  Start collection")
