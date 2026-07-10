"""
Main Window
===========

Assembles the whole GUI and owns the camera pipeline lifecycle. Layout:

    +------------------------------------------------------------------+
    | TopBar (name | status | resolution | fps)                        |
    +----------------+----------------------------+--------------------+
    | settings       |                            | calibration        |
    | panels         |     live image viewer      | readiness panel    |
    | (scrollable)   |                            | (quality mode)     |
    +----------------+----------------------------+--------------------+
    | ControlBar (Start / Pause / Stop / Save / Collect)               |
    +------------------------------------------------------------------+

Two collection paths share the same worker:

* **Quality-guided** (default) reuses the CLI's :class:`DataQualityJudge` via
  :class:`QualityCollector` - judge overlay, diversity/sharpness gating and a native
  readiness panel (coverage bars + heatmap); the full heatmap+radar report is written
  to ``final_heatmap.png`` on stop, identical to
  ``run.py calibrate collect --use-quality-judge``.
* **Plain** live view keeps the :class:`FrameProcessor` overlays + timed capture.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import Qt, Slot
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (
    QLabel, QMainWindow, QMessageBox, QScrollArea, QSplitter, QVBoxLayout, QWidget, QFileDialog,
)

# === Local ===
from app.config import APP_NAME, DEFAULT_CONFIG_PATH
from core import camera_manager
from core.camera_worker import CameraWorker
from core.data_collector import DataCollector, build_session_dir
from core.frame_processor import FrameProcessor
from core.quality_collector import QualityCollector
from core.detector_factory import build_detector
from core.config_manager import (
    AppConfig, BoardSettings, CalibrationSettings, CameraSettings,
    CollectionSettings, VisualizationSettings, load_config, save_config,
)
from ui.status_bar import TopBar
from ui.control_bar import ControlBar
from ui.image_viewer import ImageViewer
from ui.camera_panel import CameraPanel
from ui.collection_panel import CollectionPanel
from ui.calibration_panel import CalibrationPanel
from ui.visualization_panel import VisualizationPanel
from ui.judgment_panel import JudgmentPanel

__all__ = ["MainWindow"]

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Top-level window that wires the panels to the camera pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1560, 880)

        self._worker: CameraWorker | None = None
        self._processor: FrameProcessor | None = None
        self._collector: DataCollector | None = None
        self._quality_mode = True
        self._pending_restart = False

        # --- Widgets ---
        self._top_bar = TopBar()
        self._viewer = ImageViewer()
        self._judgment_panel = JudgmentPanel()
        self._control_bar = ControlBar()
        self._camera_panel = CameraPanel()
        self._calibration_panel = CalibrationPanel()
        self._collection_panel = CollectionPanel()
        self._visualization_panel = VisualizationPanel()

        self._build_layout()
        self._build_menu()
        self._connect_signals()
        self._make_text_selectable()

        self.refresh_cameras()
        self._load_initial_config()
        self._on_quality_mode_changed(self._collection_panel.get_settings().use_quality_judge)

    # --- Construction ---

    def _build_layout(self) -> None:
        """Assemble the top bar, split settings/viewer/readiness, report dock and controls."""
        settings_container = QWidget()
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(6, 6, 6, 6)
        for panel in (
            self._camera_panel, self._calibration_panel,
            self._collection_panel, self._visualization_panel,
        ):
            settings_layout.addWidget(panel)
        settings_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(settings_container)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(460)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(scroll)
        splitter.addWidget(self._viewer)
        splitter.addWidget(self._judgment_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([460, 800, 300])   # start the settings sidebar at its max width

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._top_bar)
        layout.addWidget(splitter, stretch=1)
        layout.addWidget(self._control_bar)
        self.setCentralWidget(central)

    def _make_text_selectable(self) -> None:
        """Let the user select and copy the text of every static label (headers + field names)."""
        for label in self.findChildren(QLabel):
            label.setTextInteractionFlags(
                label.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse
            )

    def _build_menu(self) -> None:
        """Build the File menu (config save/load + exit)."""
        file_menu = self.menuBar().addMenu("&File")

        load_action = QAction("Load config...", self)
        load_action.triggered.connect(self._on_load_config)
        save_action = QAction("Save config...", self)
        save_action.triggered.connect(self._on_save_config)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)

        file_menu.addAction(load_action)
        file_menu.addAction(save_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)

    def _connect_signals(self) -> None:
        """Wire panel/control signals to their handlers."""
        self._control_bar.start_clicked.connect(self.start_pipeline)
        self._control_bar.stop_clicked.connect(self.stop_pipeline)
        self._control_bar.pause_toggled.connect(self._on_pause_toggled)
        self._control_bar.save_clicked.connect(self._on_save_frame)
        self._control_bar.collection_toggled.connect(self._on_collection_toggled)

        self._camera_panel.request_refresh.connect(self.refresh_cameras)
        self._camera_panel.apply_requested.connect(self._apply_camera_settings)
        self._calibration_panel.board_changed.connect(self._on_board_changed)
        self._visualization_panel.visualization_changed.connect(self._on_visualization_changed)
        self._collection_panel.auto_capture_changed.connect(self._on_auto_capture_changed)
        self._collection_panel.quality_mode_changed.connect(self._on_quality_mode_changed)
        self._collection_panel.auto_save_changed.connect(self._on_auto_save_changed)
        self._collection_panel.heatmap_changed.connect(self._on_heatmap_changed)

    # --- Camera discovery ---

    @Slot()
    def refresh_cameras(self) -> None:
        """Re-run device discovery and repopulate the camera panel."""
        devices = camera_manager.list_cameras()
        self._camera_panel.populate(devices)
        if not devices:
            self.statusBar().showMessage("No camera devices found", 5000)

    # --- Pipeline lifecycle ---

    @Slot()
    def start_pipeline(self) -> None:
        """Build a fresh worker (quality-guided or plain) and start capturing."""
        if self._worker is not None:
            return

        camera = self._camera_panel.get_settings()
        board = self._calibration_panel.get_board_settings()
        calibration = self._calibration_panel.get_calibration_settings()
        collection = self._collection_panel.get_settings()
        visualization = self._visualization_panel.get_settings()

        self._quality_mode = collection.use_quality_judge
        self._control_bar.set_quality_mode(self._quality_mode)
        self._set_quality_ui_visible(self._quality_mode)

        if self._quality_mode:
            self._worker = self._build_quality_worker(camera, board, calibration, visualization, collection)
        else:
            self._worker = self._build_plain_worker(camera, board, calibration, visualization, collection)

        self._worker.frame_ready.connect(self._viewer.update_frame)
        self._worker.stats_ready.connect(self._top_bar.set_fps)
        self._worker.opened.connect(self._on_camera_opened)
        self._worker.error.connect(self._on_worker_error)
        self._worker.finished_cleanly.connect(self._on_worker_finished)

        self._worker.start()
        self._control_bar.set_running(True)
        self._top_bar.set_status("running")
        self.statusBar().showMessage(f"Started camera /dev/video{camera.device}", 4000)

    def _build_quality_worker(
        self,
        camera: CameraSettings,
        board: BoardSettings,
        calibration: CalibrationSettings,
        visualization: VisualizationSettings,
        collection: CollectionSettings,
    ) -> CameraWorker:
        """Build a worker that runs the reused quality judge (matches the CLI collect)."""
        detector = build_detector(board, visualization, calibration.camera_params, camera.resolution)
        # Resolve the folder now (so we can show it) but create it lazily on the first
        # kept frame, so an "Apply settings" restart that keeps nothing leaves no folder.
        session_dir = build_session_dir(
            collection.output_dir, collection.session_name, collection.auto_timestamp_folder,
            create=False,
        )
        self._collection_panel.set_session_dir(str(session_dir))
        self._judgment_panel.reset()

        def factory(size: tuple[int, int]) -> QualityCollector:
            return QualityCollector(
                detector, size, session_dir, collection.min_sharpness, collection.target_samples
            )

        worker = CameraWorker(
            device_index=camera.device,
            resolution=camera.resolution,
            fps=camera.fps,
            fourcc=camera.fourcc,
            quality_factory=factory,
        )
        worker.set_show_heatmap(collection.show_heatmap)
        worker.set_auto_capture(collection.auto_save, 0.0)   # pre-arm auto-save from the panel
        worker.judgment_ready.connect(self._judgment_panel.update_snapshot)
        worker.capture_skipped.connect(self._on_capture_skipped)
        self._processor = None
        self._collector = None
        return worker

    def _build_plain_worker(
        self,
        camera: CameraSettings,
        board: BoardSettings,
        calibration: CalibrationSettings,
        visualization: VisualizationSettings,
        collection: CollectionSettings,
    ) -> CameraWorker:
        """Build a worker for the plain live view + timed data collection."""
        self._processor = FrameProcessor(
            board=board,
            viz=visualization,
            camera_params=calibration.camera_params,
            resolution=camera.resolution,
        )
        self._collector = DataCollector(collection)
        return CameraWorker(
            device_index=camera.device,
            resolution=camera.resolution,
            fps=camera.fps,
            fourcc=camera.fourcc,
            processor=self._processor,
            collector=self._collector,
        )

    @Slot()
    def stop_pipeline(self) -> None:
        """Stop the worker, release the camera and reset the UI state."""
        if self._worker is None:
            return
        self._worker.stop()
        self._worker.wait(3000)
        # Cleanup happens in _on_worker_finished (also fires on natural exit).

    @Slot()
    def _apply_camera_settings(self) -> None:
        """Apply the selected camera settings by (re)starting the live pipeline."""
        if self._worker is None:
            self.start_pipeline()
            return
        self._pending_restart = True
        self.statusBar().showMessage("Applying settings...", 2000)
        self.stop_pipeline()

    @Slot(int, int, float)
    def _on_camera_opened(self, width: int, height: int, fps: float) -> None:
        """Reflect the camera's actual resolution once it is open."""
        self._top_bar.set_resolution(width, height)
        self._judgment_panel.set_aspect(width, height)   # square-cell coverage grid

    @Slot()
    def _on_worker_finished(self) -> None:
        """Reset UI and drop references after the worker loop exits."""
        self._worker = None
        self._processor = None
        self._collector = None
        self._control_bar.set_running(False)
        self._top_bar.set_status("stopped")
        self._top_bar.set_fps(0.0)
        self._viewer.clear()

        if self._pending_restart:
            self._pending_restart = False
            self.start_pipeline()

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        """Surface a worker error and tear the pipeline down."""
        log.error("Worker error: %s", message)
        self.statusBar().showMessage(message, 6000)
        QMessageBox.warning(self, "Camera error", message)
        self.stop_pipeline()

    # --- Live control handlers ---

    @Slot(bool)
    def _on_pause_toggled(self, paused: bool) -> None:
        """Pause or resume the running worker."""
        if self._worker is None:
            return
        if paused:
            self._worker.pause()
            self._top_bar.set_status("paused")
        else:
            self._worker.resume()
            self._top_bar.set_status("running")

    @Slot()
    def _on_save_frame(self) -> None:
        """Request a single frame capture / keep from the worker (both modes)."""
        if self._worker is not None:
            self._worker.capture_once()

    @Slot(str)
    def _on_capture_skipped(self, reason: str) -> None:
        """Report why a manual keep was rejected (quality mode)."""
        self.statusBar().showMessage(f"Not saved: {reason}", 4000)

    @Slot(bool)
    def _on_collection_toggled(self, collecting: bool) -> None:
        """Start or stop a timed collection session (plain mode only)."""
        if self._worker is None or self._collector is None or self._quality_mode:
            return

        collection = self._collection_panel.get_settings()
        if collecting:
            session_dir = self._collector.start_session()
            self._collection_panel.set_session_dir(str(session_dir))
            self._worker.set_auto_capture(collection.auto_capture, collection.capture_interval)
            self.statusBar().showMessage(f"Collecting into {session_dir}", 4000)
        else:
            self._worker.set_auto_capture(False, collection.capture_interval)
            board = self._calibration_panel.get_board_settings()
            camera = self._camera_panel.get_settings()
            self._collector.write_metadata(camera, board)
            self.statusBar().showMessage("Collection stopped", 4000)

    @Slot()
    def _on_board_changed(self) -> None:
        """Rebuild the live detector for new board geometry (plain mode)."""
        if self._processor is not None:
            self._processor.set_board(self._calibration_panel.get_board_settings())

    @Slot()
    def _on_visualization_changed(self) -> None:
        """Apply overlay toggles to the live processor (plain mode)."""
        if self._processor is not None:
            self._processor.set_visualization(self._visualization_panel.get_settings())

    @Slot(bool, float)
    def _on_auto_capture_changed(self, enabled: bool, interval: float) -> None:
        """Forward timed auto-capture changes to the running worker (plain mode)."""
        if self._worker is not None and not self._quality_mode:
            self._worker.set_auto_capture(enabled, interval)

    @Slot(bool)
    def _on_auto_save_changed(self, enabled: bool) -> None:
        """Forward the auto-save switch to the running worker (quality mode)."""
        if self._worker is not None and self._quality_mode:
            self._worker.set_auto_capture(enabled, 0.0)
            self.statusBar().showMessage(f"Auto-save {'ON' if enabled else 'OFF'}", 2500)

    @Slot(bool)
    def _on_heatmap_changed(self, enabled: bool) -> None:
        """Toggle the live coverage-heatmap tint (quality mode)."""
        if self._worker is not None and self._quality_mode:
            self._worker.set_show_heatmap(enabled)

    @Slot(bool)
    def _on_quality_mode_changed(self, quality: bool) -> None:
        """Show/hide the quality-only UI when the judge is toggled (while idle)."""
        self._set_quality_ui_visible(quality)
        self._control_bar.set_quality_mode(quality)

    def _set_quality_ui_visible(self, quality: bool) -> None:
        """Show the readiness panel in quality mode; the overlay toggles only apply to plain mode."""
        self._judgment_panel.setVisible(quality)
        self._visualization_panel.setVisible(not quality)

    # --- Config persistence ---

    def _gather_config(self) -> AppConfig:
        """Collect the current panel state into an :class:`AppConfig`."""
        return AppConfig(
            camera=self._camera_panel.get_settings(),
            board=self._calibration_panel.get_board_settings(),
            calibration=self._calibration_panel.get_calibration_settings(),
            collection=self._collection_panel.get_settings(),
            visualization=self._visualization_panel.get_settings(),
        )

    def _apply_config(self, config: AppConfig) -> None:
        """Apply an :class:`AppConfig` to every panel."""
        self._camera_panel.apply_settings(config.camera)
        self._calibration_panel.apply_board_settings(config.board)
        self._calibration_panel.apply_calibration_settings(config.calibration)
        self._collection_panel.apply_settings(config.collection)
        self._visualization_panel.apply_settings(config.visualization)

    def _load_initial_config(self) -> None:
        """Load the default config on startup, if present."""
        if DEFAULT_CONFIG_PATH.exists():
            self._apply_config(load_config(DEFAULT_CONFIG_PATH))

    @Slot()
    def _on_load_config(self) -> None:
        """Load a config file chosen by the user and apply it."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load config", str(DEFAULT_CONFIG_PATH.parent), "YAML files (*.yaml *.yml)"
        )
        if path:
            self._apply_config(load_config(path))
            self.statusBar().showMessage(f"Loaded config from {path}", 4000)

    @Slot()
    def _on_save_config(self) -> None:
        """Save the current settings to a config file chosen by the user."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save config", str(DEFAULT_CONFIG_PATH), "YAML files (*.yaml *.yml)"
        )
        if path:
            save_config(self._gather_config(), path)
            self.statusBar().showMessage(f"Saved config to {path}", 4000)

    # --- Qt lifecycle ---

    def closeEvent(self, event: QCloseEvent) -> None:
        """Stop the worker cleanly before the window closes."""
        self._pending_restart = False
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait(3000)
        event.accept()
