"""
Configuration Model & Persistence
=================================

Typed dataclasses mirroring ``configs/default_config.yaml`` plus load/save
helpers. The whole GUI reads and writes a single :class:`AppConfig`, and every
settings panel converts to/from one of these sub-structs.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict

# === Third-Party Libraries ===
import yaml

__all__ = [
    "CameraSettings",
    "BoardSettings",
    "CalibrationSettings",
    "CollectionSettings",
    "VisualizationSettings",
    "AppConfig",
    "load_config",
    "save_config",
]

log = logging.getLogger(__name__)


@dataclass
class CameraSettings:
    """Selected camera device, resolution, frame rate and pixel format."""

    device: int = 0
    resolution: tuple[int, int] = (1280, 720)
    fps: int = 30
    fourcc: str = "MJPG"


@dataclass
class BoardSettings:
    """Charuco board geometry and marker dictionary."""

    board_id: int = 0
    x_squares: int = 7
    y_squares: int = 5
    square_length: float = 0.03
    marker_length: float | None = None      # None -> 75% of square_length
    dictionary: str = "DICT_6X6_1000"


@dataclass
class CalibrationSettings:
    """How calibration should be performed and where intrinsics come from."""

    camera_params: str | None = None        # path to an intrinsics .xml
    fisheye: bool = False
    fisheye_fov: float = 180.0              # informational FoV (deg) recorded in the config
    mode: str = "Intrinsic"


@dataclass
class CollectionSettings:
    """Data-collection output location and capture behaviour."""

    output_dir: str = "outputs"
    session_name: str = "calibration_session"
    auto_timestamp_folder: bool = True     # append a date-time suffix to the session folder
    save_raw_frames: bool = True
    save_visualized_frames: bool = False
    save_metadata: bool = True
    auto_capture: bool = False
    capture_interval: float = 1.0

    # Quality-guided collection (mirrors the CLI's `collect --use-quality-judge`).
    use_quality_judge: bool = True
    target_samples: int = 100
    min_sharpness: float = 400.0            # variance of Laplacian; 0 disables the blur gate
    auto_save: bool = False                 # auto-keep every accepted (diverse+sharp) view
    show_heatmap: bool = False              # tint the live frame with the coverage heatmap


@dataclass
class VisualizationSettings:
    """Real-time overlay toggles applied to every processed frame."""

    show_fps: bool = False               # never burn FPS onto the frame (shown in the top bar)
    show_frame_counter: bool = False
    show_markers: bool = False
    show_corners: bool = True
    show_corners_cv2: bool = False
    show_board_pose: bool = True
    use_estimate_pose_board: bool = False
    project_points: bool = True
    show_center_point: bool = False
    show_grid: bool = False
    show_debug_text: bool = False


@dataclass
class AppConfig:
    """Aggregate of every settings group; the single source of truth for the UI."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    board: BoardSettings = field(default_factory=BoardSettings)
    calibration: CalibrationSettings = field(default_factory=CalibrationSettings)
    collection: CollectionSettings = field(default_factory=CollectionSettings)
    visualization: VisualizationSettings = field(default_factory=VisualizationSettings)

    def to_dict(self) -> dict:
        """Return a plain, YAML-serialisable dict (tuples become lists)."""
        data = asdict(self)
        data["camera"]["resolution"] = list(self.camera.resolution)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """
        Build an :class:`AppConfig` from a (possibly partial) dict.

        Unknown keys are ignored and missing keys fall back to defaults, so a
        hand-edited or older config file still loads.

        Args:
            data: Nested mapping matching the ``default_config.yaml`` layout

        Returns:
            A fully populated :class:`AppConfig`
        """
        data = data or {}

        def pick(section: str, allowed: type) -> dict:
            raw = data.get(section, {}) or {}
            valid = allowed.__dataclass_fields__.keys()
            return {k: v for k, v in raw.items() if k in valid}

        camera = CameraSettings(**pick("camera", CameraSettings))
        if isinstance(camera.resolution, list):
            camera.resolution = tuple(camera.resolution)

        return cls(
            camera=camera,
            board=BoardSettings(**pick("board", BoardSettings)),
            calibration=CalibrationSettings(**pick("calibration", CalibrationSettings)),
            collection=CollectionSettings(**pick("collection", CollectionSettings)),
            visualization=VisualizationSettings(**pick("visualization", VisualizationSettings)),
        )


def load_config(path: Path | str) -> AppConfig:
    """
    Load an :class:`AppConfig` from a YAML file, falling back to defaults.

    Args:
        path: Path to the YAML config file

    Returns:
        The parsed config, or an all-defaults config if the file is missing
    """
    path = Path(path)
    if not path.exists():
        log.warning("Config not found at %s - using defaults", path)
        return AppConfig()

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    log.info("Loaded config from %s", path)
    return AppConfig.from_dict(data)


def save_config(config: AppConfig, path: Path | str) -> None:
    """
    Write an :class:`AppConfig` to a YAML file, creating parent dirs as needed.

    Args:
        config: The config to persist
        path: Destination YAML path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, default_flow_style=False)
    log.info("Saved config to %s", path)
