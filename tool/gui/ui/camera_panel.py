"""
Camera Settings Panel
=====================

Device / pixel-format / resolution / frame-rate selection. The drop-downs
cascade: picking a device repopulates its formats, which repopulates the
resolutions, which repopulates the frame rates - all from the capability map
discovered by :mod:`core.camera_manager`.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget,
)

# === Local ===
from core.camera_manager import CameraDevice
from core.config_manager import CameraSettings
from ui.widgets import block_wheel, TitledGroupBox

__all__ = ["CameraPanel"]

log = logging.getLogger(__name__)


class CameraPanel(TitledGroupBox):
    """Group box exposing the camera device and its capture format choices."""

    request_refresh = Signal()
    apply_requested = Signal()

    def __init__(self) -> None:
        super().__init__("CAMERA")
        self._devices: list[CameraDevice] = []

        self._device_combo = QComboBox()
        self._device_combo.setToolTip("Camera to open (/dev/videoN)")
        self._fourcc_combo = QComboBox()
        self._fourcc_combo.setToolTip("Pixel format the camera streams in (e.g. MJPG, YUYV)")
        self._resolution_combo = QComboBox()
        self._resolution_combo.setToolTip("Capture resolution supported by this device + format")
        self._fps_combo = QComboBox()
        self._fps_combo.setToolTip("Frame rate supported at this resolution")

        refresh_button = QPushButton("🔄  Refresh cameras")
        refresh_button.setToolTip("Re-scan for connected cameras and their supported formats")
        refresh_button.clicked.connect(self.request_refresh)

        apply_button = QPushButton("✔  Apply settings")
        apply_button.setToolTip("(Re)open the camera with the selected settings and show it live")
        apply_button.clicked.connect(self.apply_requested)

        form = QFormLayout()
        form.setHorizontalSpacing(18)   # a little more air between labels and fields
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)   # centre labels against their field
        form.addRow("Device", self._device_combo)
        form.addRow("Format", self._fourcc_combo)
        form.addRow("Resolution", self._resolution_combo)
        form.addRow("FPS", self._fps_combo)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        row = QHBoxLayout()
        row.addWidget(refresh_button)
        row.addWidget(apply_button)
        layout.addLayout(row)

        block_wheel(self._device_combo, self._fourcc_combo, self._resolution_combo, self._fps_combo)

        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        self._fourcc_combo.currentIndexChanged.connect(self._on_fourcc_changed)
        self._resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)

    # --- Population ---

    def populate(self, devices: list[CameraDevice]) -> None:
        """
        Fill the device drop-down and cascade the format choices.

        Args:
            devices: Discovered capture devices
        """
        self._devices = devices
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        for device in devices:
            self._device_combo.addItem(device.label, device)
        self._device_combo.blockSignals(False)

        if devices:
            self._device_combo.setCurrentIndex(0)
            self._on_device_changed(0)
        else:
            self._fourcc_combo.clear()
            self._resolution_combo.clear()
            self._fps_combo.clear()
            log.warning("No camera devices to populate")

    def _current_device(self) -> CameraDevice | None:
        """Return the currently selected device, if any."""
        return self._device_combo.currentData()

    def _on_device_changed(self, _index: int) -> None:
        """Repopulate the pixel-format list for the newly selected device."""
        device = self._current_device()
        self._fourcc_combo.blockSignals(True)
        self._fourcc_combo.clear()
        if device is not None:
            for fourcc in device.fourccs() or ["MJPG"]:
                self._fourcc_combo.addItem(fourcc)
        self._fourcc_combo.blockSignals(False)
        self._on_fourcc_changed(0)

    def _on_fourcc_changed(self, _index: int) -> None:
        """Repopulate resolutions for the selected device + format."""
        device = self._current_device()
        fourcc = self._fourcc_combo.currentText() or None
        self._resolution_combo.blockSignals(True)
        self._resolution_combo.clear()
        if device is not None:
            for width, height in device.resolutions(fourcc):
                self._resolution_combo.addItem(f"{width} x {height}", (width, height))
        self._resolution_combo.blockSignals(False)
        self._on_resolution_changed(0)

    def _on_resolution_changed(self, _index: int) -> None:
        """Repopulate frame rates for the selected device + format + size."""
        device = self._current_device()
        fourcc = self._fourcc_combo.currentText() or None
        resolution = self._resolution_combo.currentData()
        self._fps_combo.blockSignals(True)
        self._fps_combo.clear()
        if device is not None and resolution is not None:
            for fps in device.fps_options(resolution, fourcc):
                self._fps_combo.addItem(f"{fps:g}", fps)
        self._fps_combo.blockSignals(False)

    # --- Config round-tripping ---

    def get_settings(self) -> CameraSettings:
        """Return the panel state as a :class:`CameraSettings`."""
        device = self._current_device()
        resolution = self._resolution_combo.currentData() or (1280, 720)
        fps = self._fps_combo.currentData()
        return CameraSettings(
            device=device.index if device is not None else 0,
            resolution=tuple(resolution),
            fps=int(fps) if fps is not None else 30,
            fourcc=self._fourcc_combo.currentText() or "MJPG",
        )

    def apply_settings(self, settings: CameraSettings) -> None:
        """
        Best-effort selection of the drop-downs to match a config.

        Args:
            settings: Desired camera selection
        """
        for i in range(self._device_combo.count()):
            device = self._device_combo.itemData(i)
            if device is not None and device.index == settings.device:
                self._device_combo.setCurrentIndex(i)
                break

        self._select_by_text(self._fourcc_combo, settings.fourcc)
        self._on_fourcc_changed(0)
        self._select_by_data(self._resolution_combo, tuple(settings.resolution))
        self._on_resolution_changed(0)
        self._select_by_data(self._fps_combo, float(settings.fps))

    @staticmethod
    def _select_by_text(combo: QComboBox, text: str) -> None:
        """Select the combo entry whose text matches, if present."""
        index = combo.findText(text, Qt.MatchFixedString)
        if index >= 0:
            combo.setCurrentIndex(index)

    @staticmethod
    def _select_by_data(combo: QComboBox, data: object) -> None:
        """Select the combo entry whose userData matches, if present."""
        index = combo.findData(data)
        if index >= 0:
            combo.setCurrentIndex(index)
