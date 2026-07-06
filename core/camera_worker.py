"""
Camera Worker
=============

A :class:`QThread` that owns the OpenCV capture loop so the GUI thread never
blocks. It grabs frames, hands each to a :class:`FrameProcessor`, saves through
a :class:`DataCollector` on request, and emits results as Qt signals.

Control methods (:meth:`pause`, :meth:`resume`, :meth:`stop`,
:meth:`capture_once`, :meth:`set_auto_capture`) are safe to call from the GUI
thread - they only flip flags the loop polls.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging
from typing import Callable
from time import perf_counter

# === Third-Party Libraries ===
import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

# === Local ===
from core.frame_processor import FrameProcessor
from core.data_collector import DataCollector
from core.quality_collector import QualityCollector

__all__ = ["CameraWorker"]

log = logging.getLogger(__name__)

_FPS_SMOOTHING = 0.9  # exponential moving average weight for the FPS readout


class CameraWorker(QThread):
    """Threaded camera grabber that detects, annotates, saves and emits frames."""

    frame_ready = Signal(object)     # annotated RGB frame (np.ndarray, HxWx3)
    stats_ready = Signal(float)      # smoothed FPS
    opened = Signal(int, int, float)  # actual width, height, fps
    saved = Signal(int, str)         # running count, saved path
    error = Signal(str)              # human-readable failure message
    finished_cleanly = Signal()      # camera released, loop exited
    judgment_ready = Signal(object)  # JudgmentSnapshot for the readiness panel (quality mode)
    capture_skipped = Signal(str)    # manual save rejected: human-readable reason

    def __init__(
        self,
        device_index: int,
        resolution: tuple[int, int],
        fps: int,
        fourcc: str,
        processor: FrameProcessor | None = None,
        collector: DataCollector | None = None,
        quality_factory: Callable[[tuple[int, int]], QualityCollector] | None = None,
    ) -> None:
        """
        Args:
            device_index: V4L2 index to open (``/dev/videoN``)
            resolution: Requested ``(width, height)``
            fps: Requested frame rate
            fourcc: Requested pixel format, e.g. ``"MJPG"``
            processor: Plain-mode detection + overlay engine (lives in this thread)
            collector: Plain-mode session file writer used when saving frames
            quality_factory: When given, runs quality-guided collection instead - the
                factory is called with the camera's ACTUAL size once it opens, so the
                judge normalizes its metrics by the real resolution
        """
        super().__init__()
        self._device_index = device_index
        self._resolution = resolution
        self._fps = fps
        self._fourcc = fourcc
        self._processor = processor
        self._collector = collector
        self._quality_factory = quality_factory
        self._quality: QualityCollector | None = None
        self._actual_size = resolution

        # Flags polled by the loop; written from the GUI thread.
        self._running = False
        self._paused = False
        self._capture_once = False
        self._auto_capture = False
        self._capture_interval = 1.0
        self._show_heatmap = False

        self._measured_fps = 0.0

    # --- Control API (GUI thread) ---

    def set_show_heatmap(self, enabled: bool) -> None:
        """Toggle the live coverage-heatmap tint (quality mode only)."""
        self._show_heatmap = enabled

    def pause(self) -> None:
        """Freeze grabbing while keeping the camera open."""
        self._paused = True

    def resume(self) -> None:
        """Resume grabbing after a pause."""
        self._paused = False

    def stop(self) -> None:
        """Ask the loop to exit and release the camera."""
        self._running = False

    def capture_once(self) -> None:
        """Request a single frame be saved on the next iteration."""
        self._capture_once = True

    def set_auto_capture(self, enabled: bool, interval: float) -> None:
        """Enable/disable timed auto-capture and set its interval (seconds)."""
        self._auto_capture = enabled
        self._capture_interval = max(0.05, interval)

    # --- Capture loop (worker thread) ---

    def run(self) -> None:
        """Open the camera, then grab/process/emit until stopped.

        ``finished_cleanly`` is emitted on every exit path (including a failed open)
        so the GUI always drops its worker reference and never gets stuck - which is
        what let a second "Apply settings" leave a black view.
        """
        cap = None
        try:
            cap = self._open_capture()
            if cap is None:
                return

            # Quality mode: build the judge now that the real resolution is known.
            if self._quality_factory is not None:
                self._quality = self._quality_factory(self._actual_size)

            self._running = True
            last_time = perf_counter()
            last_capture = 0.0

            while self._running:
                if self._paused:
                    self.msleep(20)
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    self.error.emit("Camera returned no frame - stopping.")
                    break

                now = perf_counter()
                self._update_fps(now - last_time)
                last_time = now

                if self._quality is not None:
                    self._process_quality(frame)
                else:
                    last_capture = self._process_plain(frame, now, last_capture)

                self.stats_ready.emit(self._measured_fps)
        finally:
            if cap is not None:
                cap.release()
                log.info("Camera %d released", self._device_index)
            if self._quality is not None:
                self._quality.finalize()
            self.finished_cleanly.emit()

    def _process_plain(self, frame: np.ndarray, now: float, last_capture: float) -> float:
        """Plain live-view path: detect + HUD overlays, timed/manual save. Returns last_capture."""
        annotated = self._processor.process(frame.copy())

        if self._should_capture(now, last_capture):
            self._save(frame, annotated)
            last_capture = now

        self._processor.set_stats(self._measured_fps, self._collector.count)
        self.frame_ready.emit(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        return last_capture

    def _process_quality(self, frame: np.ndarray) -> None:
        """Quality-guided path: judge overlay, gated save, readiness signal."""
        # No FPS burned onto the frame - the FPS lives in the top bar instead.
        annotated, sample = self._quality.process(frame, self._show_heatmap, None)

        # Manual save keeps only an acceptable view; auto-save keeps every accepted
        # (diverse + sharp) view - committing it makes near-duplicates get rejected next.
        if self._capture_once:
            self._capture_once = False
            if sample is not None and sample.is_accepted:
                self._save_quality(frame, sample)
            else:
                self.capture_skipped.emit(sample.reject_reason if sample is not None else "no board detected")
        elif self._auto_capture and sample is not None and sample.is_accepted:
            self._save_quality(frame, sample)

        self.frame_ready.emit(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        self.judgment_ready.emit(self._quality.snapshot(sample))

    def _save_quality(self, frame: np.ndarray, sample) -> None:
        """Save an accepted view through the quality collector and notify listeners."""
        path = self._quality.save(frame, sample)
        if path is not None:
            self.saved.emit(self._quality.count, str(path))

    def _open_capture(self) -> cv2.VideoCapture | None:
        """Open and configure the capture device, emitting on failure."""
        cap = cv2.VideoCapture(self._device_index, cv2.CAP_V4L2)
        if not cap.isOpened():
            self.error.emit(f"Cannot open camera /dev/video{self._device_index}")
            return None

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        cap.set(cv2.CAP_PROP_FPS, self._fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # A webcam silently ignores requests it cannot honour; report and use
        # the size it actually delivers so overlays stay geometrically correct.
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        self._actual_size = (actual_w, actual_h)
        if self._processor is not None:
            self._processor.set_resolution((actual_w, actual_h))

        # Warm-up: a freshly (re)opened V4L2 device streams a few black/garbage
        # frames before the sensor settles. Discard them here so pressing "Apply
        # settings" a second time never leaves the viewer showing a black frame.
        warmed = 0
        for _ in range(15):
            ok, _ = cap.read()
            if ok:
                warmed += 1
                if warmed >= 3:
                    break
            self.msleep(20)
        log.info(
            "Camera %d opened at %dx%d @ %.1f fps (requested %dx%d @ %d)",
            self._device_index, actual_w, actual_h, actual_fps,
            self._resolution[0], self._resolution[1], self._fps,
        )
        self.opened.emit(actual_w, actual_h, actual_fps)
        return cap

    def _should_capture(self, now: float, last_capture: float) -> bool:
        """Decide whether this frame should be saved this iteration."""
        if self._capture_once:
            self._capture_once = False
            return True
        return self._auto_capture and (now - last_capture) >= self._capture_interval

    def _save(self, raw: np.ndarray, annotated: np.ndarray) -> None:
        """Persist a frame and notify listeners."""
        path = self._collector.save(raw, annotated)
        if path is not None:
            self.saved.emit(self._collector.count, str(path))

    def _update_fps(self, dt: float) -> None:
        """Fold the latest inter-frame delta into the smoothed FPS estimate."""
        if dt <= 0:
            return
        instant = 1.0 / dt
        if self._measured_fps == 0.0:
            self._measured_fps = instant
        else:
            self._measured_fps = _FPS_SMOOTHING * self._measured_fps + (1 - _FPS_SMOOTHING) * instant
