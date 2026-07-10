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
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QSpinBox, QVBoxLayout,
)

# === Local ===
from core.config_manager import CollectionSettings
from ui.widgets import block_wheel, make_browse_button, make_check_row, TitledGroupBox

__all__ = ["CollectionPanel"]

log = logging.getLogger(__name__)


class CollectionPanel(TitledGroupBox):
    """Group box for output location, quality-guided collection and auto-capture."""

    auto_capture_changed = Signal(bool, float)  # timed capture (plain mode)
    quality_mode_changed = Signal(bool)         # quality-guided collection on/off
    auto_save_changed = Signal(bool)            # auto-keep accepted views (quality mode)
    heatmap_changed = Signal(bool)              # live coverage-heatmap tint (quality mode)

    def __init__(self) -> None:
        super().__init__("DATA COLLECTION")

        self._output_dir = QLineEdit("outputs")
        self._output_dir.setToolTip("Root folder that collection sessions are written into")
        browse = make_browse_button(self, "Choose output directory", folder=True)
        browse.setFixedHeight(self._output_dir.sizeHint().height())   # same row height as the plain fields
        browse.clicked.connect(self._browse_output)
        dir_row = QHBoxLayout()
        dir_row.setContentsMargins(0, 0, 0, 0)
        dir_row.setSpacing(6)
        dir_row.addWidget(self._output_dir)
        dir_row.addWidget(browse)

        self._session_name = QLineEdit("calibration_session")
        self._session_name.setToolTip("Name of this session's sub-folder under the output directory")

        # --- Quality-guided collection ---
        self._use_quality_row, self._use_quality = make_check_row(
            "Quality-guided collection",
            "Keep only sharp, diverse views and show the readiness dashboard",
        )
        self._use_quality.setChecked(True)

        self._target_samples = QSpinBox()
        self._target_samples.setRange(1, 10000)
        self._target_samples.setValue(100)
        self._target_samples.setToolTip("How many diverse frames to collect before the set is 'ready'")

        self._min_sharpness = QDoubleSpinBox()
        self._min_sharpness.setRange(0.0, 100000.0)
        self._min_sharpness.setDecimals(0)
        self._min_sharpness.setSingleStep(50.0)
        self._min_sharpness.setValue(400.0)
        self._min_sharpness.setSpecialValueText("off")   # shown when value == 0
        self._min_sharpness.setToolTip("Reject frames blurrier than this (variance of Laplacian); 0 = off")

        self._auto_save_row, self._auto_save = make_check_row(
            "Auto-save accepted views",
            "Automatically keep every accepted (sharp + diverse) view",
        )
        self._show_heatmap_row, self._show_heatmap = make_check_row(
            "Show coverage heatmap",
            "Tint the live frame with the coverage heatmap",
        )

        # --- Save options ---
        self._auto_timestamp_row, self._auto_timestamp = make_check_row(
            "Append timestamp to folder",
            "Add a date-time suffix to the session folder so runs never overwrite",
        )
        self._auto_timestamp.setChecked(True)
        self._save_raw_row, self._save_raw = make_check_row("Save raw frames")
        self._save_raw.setChecked(True)
        self._save_visualized_row, self._save_visualized = make_check_row("Save visualized frames")
        self._save_metadata_row, self._save_metadata = make_check_row("Save metadata JSON")
        self._save_metadata.setChecked(True)

        # --- Plain-mode timed auto-capture ---
        self._auto_capture_row, self._auto_capture = make_check_row("Timed auto-capture")
        self._interval = QDoubleSpinBox()
        self._interval.setRange(0.05, 60.0)
        self._interval.setDecimals(2)
        self._interval.setSingleStep(0.25)
        self._interval.setSuffix(" s")
        self._interval.setValue(1.0)
        capture_row = QHBoxLayout()
        capture_row.addWidget(self._auto_capture_row)
        capture_row.addWidget(QLabel("every"))
        capture_row.addWidget(self._interval)
        capture_row.addStretch(1)

        self._session_dir_label = QLabel("session: -")
        self._session_dir_label.setWordWrap(True)
        self._session_dir_label.setToolTip("Folder the current session writes into")
        self._session_dir_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        # One shared form so every field lines up to the same width: Output folder and
        # Session name match Target samples / Min sharpness. AllNonFixedFieldsGrow makes
        # the spin boxes grow to the same width as the line edits (equal right edges).
        form = QFormLayout()
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)   # centre labels against their field
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Output folder", dir_row)
        form.addRow("Session name", self._session_name)
        form.addRow("Target samples", self._target_samples)
        form.addRow("Min sharpness", self._min_sharpness)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._use_quality_row)     # judge on/off sits under its settings
        layout.addWidget(self._auto_save_row)
        layout.addWidget(self._show_heatmap_row)
        layout.addWidget(self._auto_timestamp_row)
        layout.addWidget(self._save_raw_row)
        layout.addWidget(self._save_visualized_row)
        layout.addWidget(self._save_metadata_row)
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
        self._auto_save_row.setEnabled(quality)      # disable the row so its caption greys out too
        self._show_heatmap_row.setEnabled(quality)
        self._auto_capture_row.setEnabled(not quality)
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
