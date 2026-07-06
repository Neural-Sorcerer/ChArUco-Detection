"""
Quality-Guided Collector
========================

Thin GUI-side wrapper around the existing :class:`utils.data_judgment.DataQualityJudge`.
It reproduces the per-frame logic of the CLI's ``collect --use-quality-judge`` loop
(:func:`scripts.calibrate_camera.collect_with_quality_assessment`) exactly - detect,
pretty-overlay, evaluate, render the guidance overlay, save+commit accepted views, and
finalize with a coverage heatmap + summary - so the desktop collection behaves the same.

The judge itself is reused unchanged; this class only adapts its inputs/outputs to the
camera worker and packages a :class:`JudgmentSnapshot` for the native Qt readiness panel.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging
from pathlib import Path
from dataclasses import dataclass, field

# === Third-Party Libraries ===
import cv2
import numpy as np

# === Local ===
from src.charuco_detector import CharucoDetector
from utils.data_judgment import DataQualityJudge, CalibrationSample

__all__ = ["JudgmentSnapshot", "QualityCollector"]

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class JudgmentSnapshot:
    """A per-frame, thread-safe copy of the judge state the readiness panel renders."""

    status: str                     # "none" | "good" | "skip"
    reject_reason: str
    sharpness: float                # focus score of the current view (-1 if unknown)
    min_sharpness: float
    accepted: int                   # committed (kept) samples so far
    target: int
    live_rms: float | None          # live reprojection error in px (None while warming up)
    coverage: dict[str, float]      # position / zoom_in / zoom_out / tilt / overall
    is_sufficient: bool
    next_hint: str
    cell_counts: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), dtype=np.int32))


class QualityCollector:
    """Owns a :class:`DataQualityJudge` and drives one quality-guided session."""

    def __init__(
        self,
        detector: CharucoDetector,
        image_size: tuple[int, int],
        session_dir: Path,
        min_sharpness: float,
        target_samples: int,
    ) -> None:
        """
        Args:
            detector: Shared detector (its ``board_config.board`` powers the live RMS)
            image_size: The camera's ACTUAL ``(width, height)`` (all metrics normalize by it)
            session_dir: Existing folder to write ``calib_XXXX.png`` + reports into
            min_sharpness: Reject views blurrier than this (variance of Laplacian); 0 disables
            target_samples: Target number of diverse samples
        """
        self._detector = detector
        self._session_dir = session_dir
        self._judge = DataQualityJudge(
            image_size=image_size,
            board=detector.board_config.board,
            min_sharpness=min_sharpness,
            target_samples=target_samples,
        )
        self._frame_id = 1   # saved files are 0001.png, 0002.png, ...

    @property
    def count(self) -> int:
        """Number of accepted (saved) samples so far."""
        return len(self._judge.accepted_samples)

    @property
    def session_dir(self) -> Path:
        """The folder this session writes into."""
        return self._session_dir

    def process(
        self,
        frame: np.ndarray,
        show_heatmap: bool,
        fps: float | None,
    ) -> tuple[np.ndarray, CalibrationSample | None]:
        """
        Detect, annotate and evaluate one frame (read-only - does not commit).

        Mirrors the CLI loop body: pretty marker/corner overlay, then the judge's
        guidance overlay (status border, target marker, optional heatmap tint, FPS).

        Args:
            frame: Raw BGR frame from the camera
            show_heatmap: Whether to blend the coverage heatmap onto the frame
            fps: Current loop FPS to show in the corner (or ``None``)

        Returns:
            ``(annotated_bgr, sample)`` where ``sample`` is ``None`` if no board was seen
        """
        display = frame.copy()

        charuco_corners, charuco_ids, marker_corners, marker_ids = self._detector.detect_board(display)

        if marker_corners:
            self._detector.draw_detected_markers_pretty(display, marker_corners, marker_ids)
        if (charuco_corners is not None) and (len(charuco_corners) > 0):
            self._detector.draw_detected_corners(display, charuco_corners, charuco_ids)

        sample: CalibrationSample | None = None
        if (charuco_corners is not None) and (len(charuco_corners) > 0):
            sample = self._judge.evaluate(charuco_corners, charuco_ids, timestamp=self._frame_id, image=frame)

        annotated = self._judge.render_frame(display, sample, show_heatmap=show_heatmap, fps=fps)
        return annotated, sample

    def save(self, frame: np.ndarray, sample: CalibrationSample | None) -> Path | None:
        """
        Persist an accepted view and commit it into the coverage accumulators.

        Args:
            frame: The raw (un-annotated) BGR frame to write
            sample: The sample returned by :meth:`process` for this frame

        Returns:
            The saved path, or ``None`` if the view was not acceptable
        """
        if sample is None or not sample.is_accepted:
            return None
        self._session_dir.mkdir(parents=True, exist_ok=True)   # created lazily on first keep
        path = self._session_dir / f"{self._frame_id:04d}.png"
        cv2.imwrite(str(path), frame)
        self._judge.commit(sample)      # only now is the view recorded into coverage
        self._frame_id += 1
        log.info("Saved %s (accepted %d)", path.name, self.count)
        return path

    def snapshot(self, sample: CalibrationSample | None) -> JudgmentSnapshot:
        """Package the current judge state for the readiness panel (thread-safe copy)."""
        progress = self._judge.get_progress_info()
        if sample is None:
            status, reason, sharpness = "none", "", -1.0
        elif sample.is_accepted:
            status, reason, sharpness = "good", "", sample.sharpness
        else:
            status, reason, sharpness = "skip", sample.reject_reason, sample.sharpness

        return JudgmentSnapshot(
            status=status,
            reject_reason=reason,
            sharpness=sharpness,
            min_sharpness=self._judge.min_sharpness,
            accepted=progress["accepted_samples"],
            target=progress["target_samples"],
            live_rms=self._judge.live_rms,
            coverage=dict(progress["coverage"]),
            is_sufficient=progress["is_sufficient"],
            next_hint=self._judge.next_action_hint()[0],
            cell_counts=self._judge.cell_counts.copy(),
        )

    def finalize(self) -> None:
        """Write the final heatmap + JSON summary and retire the judge's workers.

        A session with no kept views (e.g. the pipeline was restarted by "Apply
        settings" before anything was collected) writes nothing, so those restarts
        never litter the output folder with empty sessions.
        """
        self._judge.close()
        if self.count == 0:
            return
        self._session_dir.mkdir(parents=True, exist_ok=True)
        heatmap_path = self._session_dir / "final_heatmap.png"
        self._judge.generate_heatmap(str(heatmap_path))   # heatmap + radar, saved together
        self._judge.export_summary(str(self._session_dir / "collection_summary.json"))
        progress = self._judge.get_progress_info()
        log.info("Collection complete: %d diverse samples", progress["accepted_samples"])
