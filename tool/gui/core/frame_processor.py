"""
Frame Processor
===============

Owns a :class:`CharucoDetector` and turns a raw BGR frame into an annotated one:
board detection/pose (via the reused CLI pipeline) plus GUI-only HUD overlays
(FPS, frame counter, centre point, guide grid, debug text).

Designed to live inside the camera worker thread. All mutating setters and the
:meth:`process` call are guarded by a lock so the UI thread can retune overlays
or swap the board live without racing the grabber.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging
from threading import Lock

# === Third-Party Libraries ===
import cv2
import numpy as np

# === Local ===
from core.detector_factory import build_detector, make_detector_args
from core.config_manager import BoardSettings, VisualizationSettings

__all__ = ["FrameProcessor"]

log = logging.getLogger(__name__)

# HUD colours (BGR).
_HUD_GREEN = (0, 255, 0)
_HUD_CYAN = (255, 255, 0)
_HUD_GREY = (200, 200, 200)


class FrameProcessor:
    """Detect + annotate frames, with thread-safe live-tunable settings."""

    def __init__(
        self,
        board: BoardSettings,
        viz: VisualizationSettings,
        camera_params: str | None = None,
        resolution: tuple[int, int] = (1280, 720),
    ) -> None:
        """
        Args:
            board: Initial board geometry
            viz: Initial visualization toggles
            camera_params: Path to an intrinsics .xml, or ``None`` for synthetic
            resolution: Initial working resolution for synthetic intrinsics
        """
        self._lock = Lock()
        self._board = board
        self._viz = viz
        self._camera_params = camera_params
        self._resolution = resolution
        self._detector = build_detector(board, viz, camera_params, resolution)

        # Live stats surfaced as overlays (written by the worker each frame).
        self._fps = 0.0
        self._counter = 0

    # --- Live-tunable settings (called from the UI thread) ---

    def set_visualization(self, viz: VisualizationSettings) -> None:
        """Apply new overlay toggles without rebuilding the detector."""
        with self._lock:
            self._viz = viz
            # Push the draw flags onto the detector's namespace in place.
            self._detector.args = make_detector_args(self._board, viz, self._camera_params)

    def set_board(self, board: BoardSettings) -> None:
        """Rebuild the detector for new board geometry / dictionary."""
        with self._lock:
            self._board = board
            self._detector = build_detector(board, self._viz, self._camera_params, self._resolution)

    def set_resolution(self, resolution: tuple[int, int]) -> None:
        """Re-pin synthetic intrinsics to the camera's actual resolution."""
        with self._lock:
            self._resolution = resolution
            if self._camera_params is None:
                self._detector.set_synthetic_camera_params(resolution=resolution, fov_deg=60)

    def set_stats(self, fps: float, counter: int) -> None:
        """Update the FPS / frame-counter values shown in the HUD."""
        self._fps = fps
        self._counter = counter

    # --- Per-frame processing (called from the worker thread) ---

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect the board and draw all enabled overlays on a BGR frame.

        Args:
            frame: A BGR frame owned by the caller (mutated in place)

        Returns:
            The same frame with detection + HUD overlays drawn
        """
        with self._lock:
            frame = self._detector.run_charuco_pipeline(frame)
            self._draw_hud(frame, self._viz)
        return frame

    def _draw_hud(self, frame: np.ndarray, viz: VisualizationSettings) -> None:
        """Draw the GUI-only heads-up overlays according to the toggles."""
        height, width = frame.shape[:2]

        if viz.show_grid:
            for fraction in (1 / 3, 2 / 3):
                x, y = int(width * fraction), int(height * fraction)
                cv2.line(frame, (x, 0), (x, height), _HUD_GREY, 1, cv2.LINE_AA)
                cv2.line(frame, (0, y), (width, y), _HUD_GREY, 1, cv2.LINE_AA)

        if viz.show_center_point:
            cx, cy = width // 2, height // 2
            cv2.drawMarker(frame, (cx, cy), _HUD_CYAN, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)

        # Stacked top-left text block: FPS, frame counter, then debug line.
        y = 30
        if viz.show_fps:
            self._text(frame, f"FPS: {self._fps:5.1f}", (12, y), _HUD_GREEN)
            y += 32
        if viz.show_frame_counter:
            self._text(frame, f"Saved: {self._counter}", (12, y), _HUD_GREEN)
            y += 32
        if viz.show_debug_text:
            self._text(frame, f"{width}x{height}", (12, y), _HUD_GREY, scale=0.6)

    @staticmethod
    def _text(
        frame: np.ndarray,
        text: str,
        org: tuple[int, int],
        color: tuple[int, int, int],
        scale: float = 0.8,
    ) -> None:
        """Draw HUD text with a black outline so it reads over any background."""
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
