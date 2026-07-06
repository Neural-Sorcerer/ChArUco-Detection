"""
Data Collector
==============

Manages a capture session: resolves the (optionally timestamped) output folder,
writes raw and/or annotated frames, counts them, and dumps a metadata JSON at
the end. Pure file I/O + counting - no camera or UI dependencies.
"""
from __future__ import annotations

# === Standard Libraries ===
import json
import logging
from pathlib import Path
from datetime import datetime

# === Third-Party Libraries ===
import cv2
import numpy as np

# === Local ===
from core.config_manager import CollectionSettings, BoardSettings, CameraSettings

__all__ = ["DataCollector", "build_session_dir"]

log = logging.getLogger(__name__)

# Human-readable, filesystem-safe session timestamp, e.g. 2026-07-01_17-13-11.
_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def build_session_dir(
    output_dir: str,
    session_name: str,
    auto_timestamp: bool,
    create: bool = True,
) -> Path:
    """
    Resolve the folder a collection session writes into.

    Args:
        output_dir: Root output directory
        session_name: Base name for this session's folder
        auto_timestamp: Whether to append a timestamp to the folder name
        create: Whether to create the folder now (``False`` to defer it until the
            first save, so restarts that keep nothing leave no empty folder)

    Returns:
        The resolved session directory
    """
    base = Path(output_dir) / session_name
    if auto_timestamp:
        stamp = datetime.now().strftime(_TIMESTAMP_FORMAT)
        base = base.with_name(f"{base.name}_{stamp}")
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


class DataCollector:
    """Owns one collection session's folder, frame counter and metadata."""

    def __init__(self, settings: CollectionSettings) -> None:
        """
        Args:
            settings: Output location and capture behaviour
        """
        self._settings = settings
        self._session_dir: Path | None = None
        self._count = 0

    @property
    def count(self) -> int:
        """Number of frames saved in the current session."""
        return self._count

    @property
    def session_dir(self) -> Path | None:
        """The active session folder, or ``None`` before it is started."""
        return self._session_dir

    def start_session(self) -> Path:
        """
        Create the session folder and reset the counter.

        Returns:
            The resolved session directory
        """
        base = build_session_dir(
            self._settings.output_dir,
            self._settings.session_name,
            self._settings.auto_timestamp_folder,
        )
        self._session_dir = base
        self._count = 0
        log.info("Collection session started: %s", base)
        return base

    def save(self, raw: np.ndarray, annotated: np.ndarray | None = None) -> Path | None:
        """
        Save the current frame(s) according to the collection settings.

        Args:
            raw: The unannotated BGR frame
            annotated: The overlaid BGR frame (saved only if enabled)

        Returns:
            Path to the primary saved file, or ``None`` if nothing was written
        """
        if self._session_dir is None:
            self.start_session()

        stem = f"frame_{self._count:06d}"
        primary: Path | None = None

        if self._settings.save_raw_frames:
            primary = self._session_dir / f"{stem}.png"
            cv2.imwrite(str(primary), raw)

        if self._settings.save_visualized_frames and annotated is not None:
            vis_path = self._session_dir / f"{stem}_annotated.png"
            cv2.imwrite(str(vis_path), annotated)
            primary = primary or vis_path

        if primary is not None:
            self._count += 1
            log.info("Saved %s (total %d)", primary.name, self._count)
        return primary

    def write_metadata(
        self,
        camera: CameraSettings,
        board: BoardSettings,
        extra: dict | None = None,
    ) -> Path | None:
        """
        Dump a ``metadata.json`` summarising the session.

        Args:
            camera: Camera settings used for the session
            board: Board settings used for the session
            extra: Optional additional key/values to merge in

        Returns:
            Path to the metadata file, or ``None`` if disabled / no session
        """
        if not self._settings.save_metadata or self._session_dir is None:
            return None

        metadata = {
            "session_name": self._settings.session_name,
            "created": datetime.now().isoformat(timespec="seconds"),
            "frame_count": self._count,
            "camera": {
                "device": camera.device,
                "resolution": list(camera.resolution),
                "fps": camera.fps,
                "fourcc": camera.fourcc,
            },
            "board": {
                "board_id": board.board_id,
                "x_squares": board.x_squares,
                "y_squares": board.y_squares,
                "square_length": board.square_length,
                "marker_length": board.marker_length,
                "dictionary": board.dictionary,
            },
        }
        if extra:
            metadata.update(extra)

        path = self._session_dir / "metadata.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=4)
        log.info("Wrote metadata to %s", path)
        return path
