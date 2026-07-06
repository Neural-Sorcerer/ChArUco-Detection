"""
Detector Factory
================

Adapter that builds the existing :class:`src.charuco_detector.CharucoDetector`
from GUI settings. The CLI drives that class with an ``argparse.Namespace`` of
flags; here we synthesise an equivalent namespace so the same detection code is
reused unchanged.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging
from types import SimpleNamespace

# === Local ===
from app.constants import aruco_dictionary_id
from src.charuco_detector import CharucoDetector
from core.config_manager import BoardSettings, VisualizationSettings
from configs.config import CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig

__all__ = ["build_detector", "make_detector_args"]

log = logging.getLogger(__name__)


def make_detector_args(
    board: BoardSettings,
    viz: VisualizationSettings,
    camera_params: str | None,
) -> SimpleNamespace:
    """
    Build the flag namespace ``CharucoDetector`` reads at runtime.

    Args:
        board: Board geometry (only ``dictionary`` matters for flags here)
        viz: Visualization toggles mapped onto the detector's draw flags
        camera_params: Path to an intrinsics .xml, or ``None`` for synthetic

    Returns:
        A namespace exposing every attribute the detector's pipeline reads
    """
    return SimpleNamespace(
        draw_charuco_markers_cv2=viz.show_markers,
        draw_charuco_corners_cv2=viz.show_corners_cv2,
        draw_board_pose_cv2=viz.show_board_pose,
        use_estimate_pose_charuco_board=viz.use_estimate_pose_board,
        draw_charuco_corners=viz.show_corners,
        project_points=viz.project_points,
        camera_params=camera_params,
        resolution="HD",  # only used as a synthetic-intrinsics fallback name
    )


def build_detector(
    board: BoardSettings,
    viz: VisualizationSettings,
    camera_params: str | None,
    resolution: tuple[int, int],
) -> CharucoDetector:
    """
    Construct a fully configured :class:`CharucoDetector` for live processing.

    When no intrinsics file is supplied, synthetic intrinsics are set for the
    actual working resolution so pose/projection overlays still render.

    Args:
        board: Charuco board geometry and dictionary
        viz: Visualization toggles
        camera_params: Path to an intrinsics .xml, or ``None`` for synthetic
        resolution: The ``(width, height)`` the camera actually delivers

    Returns:
        A ready-to-use detector instance
    """
    args = make_detector_args(board, viz, camera_params)

    board_config = CharucoBoardConfig(
        board_id=board.board_id,
        x_squares=board.x_squares,
        y_squares=board.y_squares,
        square_length=board.square_length,
        marker_length=board.marker_length,
        dictionary_type=aruco_dictionary_id(board.dictionary),
    )
    detector = CharucoDetector(args, board_config, DetectorConfig(), CharucoDetectorConfig())

    # Re-pin synthetic intrinsics to the real resolution (the detector's own
    # default keys off the "HD" name above, which may not match the camera).
    if camera_params is None:
        detector.set_synthetic_camera_params(resolution=resolution, fov_deg=60)

    log.info(
        "Built detector: %dx%d board '%s', intrinsics=%s",
        board.x_squares, board.y_squares, board.dictionary,
        camera_params or "synthetic",
    )
    return detector
