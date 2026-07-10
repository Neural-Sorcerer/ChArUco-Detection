"""
Visualization Panel
===================

Real-time overlay toggles. Any change emits :attr:`visualization_changed` so the
running frame processor applies it immediately - no restart required.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QVBoxLayout

# === Local ===
from core.config_manager import VisualizationSettings
from ui.widgets import make_check_row, TitledGroupBox

__all__ = ["VisualizationPanel"]

log = logging.getLogger(__name__)

# (VisualizationSettings field, checkbox label) in display order. Only the overlays
# that matter for the plain live view are exposed; the rest keep their config defaults.
_TOGGLES: list[tuple[str, str]] = [
    ("show_corners", "Show detected corners (pretty)"),
    ("show_board_pose", "Show board pose axes"),
    ("use_estimate_pose_board", "Pose via estimatePoseCharucoBoard"),
    ("project_points", "Project 3D grid points"),
    ("show_center_point", "Show center point"),
    ("show_grid", "Show guide grid"),
]


class VisualizationPanel(TitledGroupBox):
    """Group box of overlay checkboxes bound to :class:`VisualizationSettings`."""

    visualization_changed = Signal()

    def __init__(self) -> None:
        super().__init__("VISUALIZATION")
        self._checks: dict[str, QCheckBox] = {}

        layout = QVBoxLayout(self)
        defaults = VisualizationSettings()
        for field_name, label in _TOGGLES:
            row, check = make_check_row(label)
            check.setChecked(getattr(defaults, field_name))
            check.toggled.connect(self.visualization_changed)
            layout.addWidget(row)
            self._checks[field_name] = check

    def get_settings(self) -> VisualizationSettings:
        """Return the checkbox states as a :class:`VisualizationSettings`."""
        return VisualizationSettings(**{name: check.isChecked() for name, check in self._checks.items()})

    def apply_settings(self, settings: VisualizationSettings) -> None:
        """Apply a :class:`VisualizationSettings` to the checkboxes."""
        for name, check in self._checks.items():
            check.blockSignals(True)
            check.setChecked(getattr(settings, name))
            check.blockSignals(False)
        self.visualization_changed.emit()
