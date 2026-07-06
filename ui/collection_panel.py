"""
Data Collection Panel
=====================

Where frames are saved and how. Shows the resolved session folder and a live
saved-frame counter, and exposes the quality-guided collection settings (judge
on/off, target sample count, minimum sharpness, auto-save, heatmap tint) plus the
plain-mode timed auto-capture.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QVBoxLayout,
)

# === Local ===
from core.config_manager import CollectionSettings
from ui.widgets import block_wheel, make_browse_button

__all__ = ["CollectionPanel"]

log = logging.getLogger(__name__)


class CollectionPanel(QGroupBox):
    """Group box for output location, quality-guided collection and auto-capture."""

    auto_capture_changed = Signal(bool, float)  # timed capture (plain mode)
    quality_mode_changed = Signal(bool)         # quality-guided collection on/off
    auto_save_changed = Signal(bool)            # auto-keep accepted views (quality mode)
    heatmap_changed = Signal(bool)              # live coverage-heatmap tint (quality mode)

    def __init__(self) -> None:
        super().__init__("Data collection")

        self._output_dir = QLineEdit("outputs")
        browse = make_browse_button(self, "Choose output directory", folder=True)
        browse.clicked.connect(self._browse_output)
        dir_row = QHBoxLayout()
        dir_row.addWidget(self._output_dir)
        dir_row.addWidget(browse)

        self._session_name = QLineEdit("calibration_session")

        # --- Quality-guided collection (judge) ---
        self._use_quality = QCheckBox("Quality-guided collection (judge)")
        self._use_quality.setChecked(True)

        self._target_samples = QSpinBox()
        self._target_samples.setRange(1, 10000)
        self._target_samples.setValue(100)

        self._min_sharpness = QDoubleSpinBox()
        self._min_sharpness.setRange(0.0, 100000.0)
        self._min_sharpness.setDecimals(0)
        self._min_sharpness.setSingleStep(50.0)
        self._min_sharpness.setValue(400.0)
        self._min_sharpness.setSpecialValueText("off")   # shown when value == 0

        self._auto_save = QCheckBox("Auto-save accepted views")
        self._show_heatmap = QCheckBox("Show coverage heatmap")

        quality_form = QFormLayout()
        quality_form.setHorizontalSpacing(18)
        quality_form.setVerticalSpacing(8)
        quality_form.addRow("Target samples", self._target_samples)
        quality_form.addRow("Min sharpness", self._min_sharpness)

        # --- Save options ---
        self._auto_timestamp = QCheckBox("Append timestamp to folder")
        self._auto_timestamp.setChecked(True)
        self._save_raw = QCheckBox("Save raw frames")
        self._save_raw.setChecked(True)
        self._save_visualized = QCheckBox("Save visualized frames")
        self._save_metadata = QCheckBox("Save metadata JSON")
        self._save_metadata.setChecked(True)

        # --- Plain-mode timed auto-capture ---
        self._auto_capture = QCheckBox("Timed auto-capture")
        self._interval = QDoubleSpinBox()
        self._interval.setRange(0.05, 60.0)
        self._interval.setDecimals(2)
        self._interval.setSingleStep(0.25)
        self._interval.setSuffix(" s")
        self._interval.setValue(1.0)
        capture_row = QHBoxLayout()
        capture_row.addWidget(self._auto_capture)
        capture_row.addWidget(QLabel("every"))
        capture_row.addWidget(self._interval)

        self._session_dir_label = QLabel("session: -")
        self._session_dir_label.setWordWrap(True)

        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(8)
        form.addRow("Output dir", dir_row)
        form.addRow("Session", self._session_name)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(quality_form)          # Target samples, Min sharpness
        layout.addWidget(self._use_quality)     # judge on/off sits under its settings
        layout.addWidget(self._auto_save)
        layout.addWidget(self._show_heatmap)
        layout.addWidget(self._auto_timestamp)
        layout.addWidget(self._save_raw)
        layout.addWidget(self._save_visualized)
        layout.addWidget(self._save_metadata)
        layout.addLayout(capture_row)
        layout.addWidget(self._session_dir_label)

        block_wheel(self._target_samples, self._min_sharpness, self._interval)

        self._use_quality.toggled.connect(self.quality_mode_changed)
        self._use_quality.toggled.connect(self._sync_enabled)
        self._auto_save.toggled.connect(self.auto_save_changed)
        self._show_heatmap.toggled.connect(self.heatmap_changed)
        self._auto_capture.toggled.connect(self._emit_auto_capture)
        self._interval.valueChanged.connect(self._emit_auto_capture)
        self._sync_enabled(self._use_quality.isChecked())

    def _sync_enabled(self, quality: bool) -> None:
        """Grey out the settings that do not apply to the active collection mode."""
        self._target_samples.setEnabled(quality)
        self._min_sharpness.setEnabled(quality)
        self._auto_save.setEnabled(quality)
        self._show_heatmap.setEnabled(quality)
        self._auto_capture.setEnabled(not quality)
        self._interval.setEnabled(not quality)

    def _emit_auto_capture(self) -> None:
        """Broadcast the current timed auto-capture state to the worker."""
        self.auto_capture_changed.emit(self._auto_capture.isChecked(), self._interval.value())

    def _browse_output(self) -> None:
        """Open a directory picker for the output folder."""
        path = QFileDialog.getExistingDirectory(self, "Select output directory", self._output_dir.text())
        if path:
            self._output_dir.setText(path)

    # --- Live status slots ---

    def set_session_dir(self, path: str) -> None:
        """Show the resolved session directory."""
        self._session_dir_label.setText(f"session: {path}")

    # --- Config round-tripping ---

    def get_settings(self) -> CollectionSettings:
        """Return the panel state as a :class:`CollectionSettings`."""
        return CollectionSettings(
            output_dir=self._output_dir.text().strip() or "outputs",
            session_name=self._session_name.text().strip() or "session",
            auto_timestamp_folder=self._auto_timestamp.isChecked(),
            save_raw_frames=self._save_raw.isChecked(),
            save_visualized_frames=self._save_visualized.isChecked(),
            save_metadata=self._save_metadata.isChecked(),
            auto_capture=self._auto_capture.isChecked(),
            capture_interval=self._interval.value(),
            use_quality_judge=self._use_quality.isChecked(),
            target_samples=self._target_samples.value(),
            min_sharpness=self._min_sharpness.value(),
            auto_save=self._auto_save.isChecked(),
            show_heatmap=self._show_heatmap.isChecked(),
        )

    def apply_settings(self, settings: CollectionSettings) -> None:
        """Apply a :class:`CollectionSettings` to the widgets."""
        self._output_dir.setText(settings.output_dir)
        self._session_name.setText(settings.session_name)
        self._auto_timestamp.setChecked(settings.auto_timestamp_folder)
        self._save_raw.setChecked(settings.save_raw_frames)
        self._save_visualized.setChecked(settings.save_visualized_frames)
        self._save_metadata.setChecked(settings.save_metadata)
        self._interval.setValue(settings.capture_interval)
        self._auto_capture.setChecked(settings.auto_capture)
        self._use_quality.setChecked(settings.use_quality_judge)
        self._target_samples.setValue(settings.target_samples)
        self._min_sharpness.setValue(settings.min_sharpness)
        self._auto_save.setChecked(settings.auto_save)
        self._show_heatmap.setChecked(settings.show_heatmap)
        self._sync_enabled(settings.use_quality_judge)
