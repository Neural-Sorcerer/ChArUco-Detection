"""
Calibration Board Panel
=======================

Charuco board geometry, marker dictionary and camera model (plain + fisheye).
Board-affecting changes emit :attr:`board_changed` so the live detector can be
rebuilt without restarting the camera.

Board type is a drop-down today (only Charuco is wired), leaving room to add
Checkerboard / AprilGrid detectors behind the same panel later.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QSpinBox, QVBoxLayout,
)

# === Local ===
from app.constants import ARUCO_DICT_NAMES
from core.config_manager import BoardSettings, CalibrationSettings
from ui.widgets import block_wheel, make_browse_button, make_check_row, TitledGroupBox, TrimmedDoubleSpinBox

__all__ = ["CalibrationPanel"]

log = logging.getLogger(__name__)


class CalibrationPanel(TitledGroupBox):
    """Group box for board geometry and calibration options."""

    board_changed = Signal()

    def __init__(self) -> None:
        super().__init__("CALIBRATION BOARD")

        self._board_type = QComboBox()
        self._board_type.addItems(["ChArUco", "Checkerboard (soon)", "AprilGrid (soon)"])

        self._board_id = QSpinBox()
        self._board_id.setRange(0, 999)

        self._x_squares = QSpinBox()
        self._x_squares.setRange(2, 50)
        self._x_squares.setValue(7)
        self._x_squares.setToolTip("Number of chessboard squares along the board's X axis")

        self._y_squares = QSpinBox()
        self._y_squares.setRange(2, 50)
        self._y_squares.setValue(5)
        self._y_squares.setToolTip("Number of chessboard squares along the board's Y axis")

        self._square_length = TrimmedDoubleSpinBox()
        self._square_length.setRange(0.001, 10.0)
        self._square_length.setDecimals(4)
        self._square_length.setSingleStep(0.005)
        self._square_length.setSuffix(" m")
        self._square_length.setValue(0.03)
        self._square_length.setToolTip("Printed side length of one square, in metres")

        self._marker_length = TrimmedDoubleSpinBox()
        self._marker_length.setRange(0.0, 10.0)
        self._marker_length.setDecimals(4)
        self._marker_length.setSingleStep(0.005)
        self._marker_length.setSuffix(" m")
        self._marker_length.setSpecialValueText("auto (75%)")   # shown when value == 0
        self._marker_length.setValue(0.0)
        self._marker_length.setToolTip("Printed side length of one marker; 0 = auto (75% of the square)")

        self._dictionary = QComboBox()
        self._dictionary.addItems(ARUCO_DICT_NAMES)
        self._select_text(self._dictionary, "DICT_6X6_1000")
        self._dictionary.setToolTip("ArUco marker dictionary the board was generated with")

        fisheye_check_row, self._fisheye = make_check_row(
            "Fisheye camera model",
            "Use the fisheye distortion model instead of the pinhole one",
        )

        # Informational field of view for the fisheye model (typed in directly);
        # not used for calibration, only recorded in the config so we know what was used.
        self._fisheye_fov = QDoubleSpinBox()
        self._fisheye_fov.setRange(0.0, 360.0)
        self._fisheye_fov.setDecimals(1)
        self._fisheye_fov.setSingleStep(5.0)
        self._fisheye_fov.setSuffix(" °")
        self._fisheye_fov.setValue(180.0)
        self._fisheye_fov.setEnabled(False)   # only meaningful with the fisheye model
        self._fisheye_fov.setToolTip("Fisheye field of view (informational; stored in the config)")

        self._camera_params = QLineEdit()
        self._camera_params.setPlaceholderText("optional intrinsics .xml (synthetic if empty)")
        self._camera_params.setToolTip("Path to a pre-computed intrinsics .xml; leave empty for synthetic intrinsics")
        browse = make_browse_button(self, "Choose intrinsics .xml", folder=False)
        browse.clicked.connect(self._browse_params)

        params_row = QHBoxLayout()
        params_row.addWidget(self._camera_params)
        params_row.addWidget(browse)

        form = QFormLayout()
        form.setHorizontalSpacing(18)   # a little more air between labels and fields
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)   # centre labels against their field
        form.addRow("Board type", self._board_type)
        form.addRow("Board ID", self._board_id)
        form.addRow("Squares X", self._x_squares)
        form.addRow("Squares Y", self._y_squares)
        form.addRow("Square length", self._square_length)
        form.addRow("Marker length", self._marker_length)
        form.addRow("Dictionary", self._dictionary)
        form.addRow("Intrinsics", params_row)

        fisheye_row = QHBoxLayout()
        fisheye_row.addWidget(fisheye_check_row)
        fisheye_row.addWidget(QLabel("FoV"))
        fisheye_row.addWidget(self._fisheye_fov)
        fisheye_row.addStretch(1)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addLayout(fisheye_row)

        # Any board-geometry change should rebuild the live detector.
        for spin in (self._board_id, self._x_squares, self._y_squares):
            spin.valueChanged.connect(self._emit_board_changed)
        for dspin in (self._square_length, self._marker_length):
            dspin.valueChanged.connect(self._emit_board_changed)
        self._dictionary.currentIndexChanged.connect(self._emit_board_changed)

        # The FoV field only makes sense for the fisheye model.
        self._fisheye.toggled.connect(self._fisheye_fov.setEnabled)

        # Values change only via keyboard / clicks - never an accidental wheel scroll.
        block_wheel(
            self._board_type, self._board_id, self._x_squares, self._y_squares,
            self._square_length, self._marker_length, self._dictionary,
            self._fisheye_fov,
        )

    def _emit_board_changed(self) -> None:
        """Notify listeners that the board geometry/dictionary changed."""
        self.board_changed.emit()

    def _browse_params(self) -> None:
        """Open a file dialog to pick an intrinsics .xml."""
        path, _ = QFileDialog.getOpenFileName(self, "Select intrinsics file", "", "XML files (*.xml)")
        if path:
            self._camera_params.setText(path)

    # --- Config round-tripping ---

    def get_board_settings(self) -> BoardSettings:
        """Return the board geometry as a :class:`BoardSettings`."""
        marker = self._marker_length.value()
        return BoardSettings(
            board_id=self._board_id.value(),
            x_squares=self._x_squares.value(),
            y_squares=self._y_squares.value(),
            square_length=self._square_length.value(),
            marker_length=marker if marker > 0 else None,
            dictionary=self._dictionary.currentText(),
        )

    def get_calibration_settings(self) -> CalibrationSettings:
        """Return the calibration options as a :class:`CalibrationSettings`."""
        params = self._camera_params.text().strip()
        return CalibrationSettings(
            camera_params=params or None,
            fisheye=self._fisheye.isChecked(),
            fisheye_fov=self._fisheye_fov.value(),
        )

    def apply_board_settings(self, board: BoardSettings) -> None:
        """Apply a :class:`BoardSettings` to the widgets."""
        widgets = (
            self._board_id, self._x_squares, self._y_squares,
            self._square_length, self._marker_length, self._dictionary,
        )
        for widget in widgets:
            widget.blockSignals(True)
        self._board_id.setValue(board.board_id)
        self._x_squares.setValue(board.x_squares)
        self._y_squares.setValue(board.y_squares)
        self._square_length.setValue(board.square_length)
        self._marker_length.setValue(board.marker_length or 0.0)
        self._select_text(self._dictionary, board.dictionary)
        for widget in widgets:
            widget.blockSignals(False)

    def apply_calibration_settings(self, calibration: CalibrationSettings) -> None:
        """Apply a :class:`CalibrationSettings` to the widgets."""
        self._camera_params.setText(calibration.camera_params or "")
        self._fisheye.setChecked(calibration.fisheye)
        self._fisheye_fov.setValue(calibration.fisheye_fov)
        self._fisheye_fov.setEnabled(calibration.fisheye)

    @staticmethod
    def _select_text(combo: QComboBox, text: str) -> None:
        """Select the combo entry whose text matches, if present."""
        index = combo.findText(text, Qt.MatchFixedString)
        if index >= 0:
            combo.setCurrentIndex(index)
