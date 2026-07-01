"""Data quality assessment module for camera calibration.

This module evaluates and filters Charuco detections so that only a *diverse* and
*well-spread* set of views is kept for camera calibration. Good calibration data
must cover the whole field of view (including edges/corners), span a range of board
sizes (near/far), and include perspective tilt (out-of-plane rotation). This module
quantifies those axes, deduplicates near-identical views, builds a per-corner
coverage heatmap, and renders a live guidance dashboard for the operator.
"""
# === Standard Libraries ===
import json
import logging
import threading
from math import sqrt, pi, acos
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any

# === Third-Party Libraries ===
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


@dataclass
class CalibrationSample:
    """A single Charuco view with the metrics used to judge its usefulness."""
    corners: np.ndarray                 # Detected corners, shape (N, 1, 2)
    ids: Optional[np.ndarray]           # Corner IDs (may be None)
    image_path: str
    timestamp: float
    x: float                            # Board center X, normalized [0, 1]
    y: float                            # Board center Y, normalized [0, 1]
    size: float                         # sqrt(board_area / image_area), [0, 1]
    skew: float                         # Perspective tilt, 0 = frontal, 1 = strong
    rotation: float                     # In-plane rotation, degrees [0, 90)
    sharpness: float = -1.0             # Focus score (variance of Laplacian); -1 = unknown
    feature: np.ndarray = field(default_factory=lambda: np.zeros(4))  # [x, y, size, skew]
    is_accepted: bool = False
    reject_reason: str = ""


class DataQualityJudge:
    """Evaluates Charuco views for diversity/quality and tracks coverage."""

    def __init__(self,
                 image_size: Tuple[int, int],
                 coverage_grid: Tuple[int, int] = (9, 6),
                 min_distance_threshold: float = 0.12,
                 min_size_ratio: float = 0.05,
                 max_size_ratio: float = 0.80,
                 max_skew: float = 0.80,
                 min_corners: int = 8,
                 n_size_bins: int = 5,
                 n_tilt_bins: int = 4,
                 min_per_bin: int = 3,
                 size_cov_max: float = 0.55,
                 min_sharpness: float = 0.0,
                 board: Optional[Any] = None,
                 target_samples: int = 50):
        """Initialize the data quality judge.

        Args:
            image_size: Image dimensions (width, height).
            coverage_grid: Coarse grid (cols, rows) used for position coverage/guidance.
            min_distance_threshold: Minimum Manhattan distance in [x, y, size, skew]
                space for a new view to count as non-redundant.
            min_size_ratio: Minimum board size relative to the image.
            max_size_ratio: Maximum board size relative to the image.
            max_skew: Maximum allowed perspective skew (rejects degenerate views).
            min_corners: Minimum number of detected corners to consider a view.
            n_size_bins: Number of size buckets used for size-coverage scoring.
            n_tilt_bins: Number of skew buckets used for tilt-coverage scoring.
            min_per_bin: Samples needed in a bin before it counts as fully covered.
                Coverage is graduated, so one sample no longer saturates an axis.
            size_cov_max: Upper board-size bound for the distance/zoom coverage bins.
                Set to a realistically reachable size so the near (zoom-in) bins can
                actually fill; views larger than this still count as the nearest bin.
            min_sharpness: Minimum focus score (variance of Laplacian) to accept a
                view; 0 disables the blur gate.
            board: Optional ``cv2.aruco.CharucoBoard`` used to run a live incremental
                calibration (real-time reprojection-error readout).
            target_samples: Target number of diverse samples.
        """
        self.image_size = image_size
        self.coverage_grid = coverage_grid
        self.min_distance_threshold = min_distance_threshold
        self.min_size_ratio = min_size_ratio
        self.max_size_ratio = max_size_ratio
        self.max_skew = max_skew
        self.min_corners = min_corners
        self.n_size_bins = n_size_bins
        self.n_tilt_bins = n_tilt_bins
        self.min_per_bin = max(1, min_per_bin)
        self.size_cov_max = size_cov_max
        self.min_sharpness = min_sharpness
        self.board = board
        self.target_samples = target_samples

        # Accepted (committed) samples
        self.accepted_samples: List[CalibrationSample] = []

        # Incremental calibration state (live reprojection-error readout)
        self.obj_points: List[np.ndarray] = []
        self.img_points: List[np.ndarray] = []
        self.live_rms: Optional[float] = None     # running reprojection error (px)
        self.last_view_error: Optional[float] = None

        # Per-corner heatmap accumulator (low-res; rendered/blurred on demand)
        w, h = image_size
        self.heat_w = 192
        self.heat_h = max(1, int(round(self.heat_w * h / w)))
        self.heatmap = np.zeros((self.heat_h, self.heat_w), dtype=np.float32)

        # Coarse occupancy of detected corners (drives position coverage + guidance)
        cols, rows = coverage_grid
        self.cell_counts = np.zeros((rows, cols), dtype=np.int32)

        # Histograms for size / tilt coverage
        self.size_bins = np.zeros(n_size_bins, dtype=np.int32)
        self.tilt_bins = np.zeros(n_tilt_bins, dtype=np.int32)

        # Background worker for the live reprojection-error readout. cv2.calibrateCamera
        # grows expensive as views accumulate, so it runs off the capture thread to keep
        # the preview smooth; the panel just reads the latest value when it is ready.
        self._cal_lock = threading.Lock()
        self._cal_dirty = threading.Event()   # set when new points await a recompute
        self._cal_stop = threading.Event()    # set by close() to retire the worker
        self._cal_thread: Optional[threading.Thread] = None
        if self.board is not None:
            self._cal_thread = threading.Thread(target=self._calibration_worker, daemon=True)
            self._cal_thread.start()

        # Cache for the full report view (regenerated only when a new view is kept)
        self._report_cache: Optional[np.ndarray] = None
        self._report_cache_n: int = -1

    # ───────────────────────────── Metrics ─────────────────────────────

    def compute_metrics(self, corners: np.ndarray) -> Dict[str, float]:
        """Compute geometric metrics for a detected board.

        Args:
            corners: Detected corner points, shape (N, 1, 2) or (N, 2).

        Returns:
            Dictionary with x, y, size, skew and rotation.
        """
        pts = corners.reshape(-1, 2).astype(np.float32)
        w, h = self.image_size

        # Center from the corner centroid (robust to partial detections)
        center = pts.mean(axis=0)
        x_pos = float(center[0] / w)
        y_pos = float(center[1] / h)

        # Projected board area via the convex hull (handles tilt honestly)
        hull = cv2.convexHull(pts)
        area = float(cv2.contourArea(hull))
        size = float(sqrt(max(area, 0.0) / (w * h)))

        # Outer quadrilateral → perspective skew + in-plane rotation
        quad = self._outer_quad(hull)
        skew = self._quad_skew(quad)
        rotation = self._inplane_rotation(pts)

        return {"x": x_pos, "y": y_pos, "size": size, "skew": skew, "rotation": rotation}

    def _outer_quad(self, hull: np.ndarray) -> np.ndarray:
        """Reduce a convex hull to its 4 dominant corners.

        Args:
            hull: Convex hull points from ``cv2.convexHull``.

        Returns:
            A (4, 2) array of corner points (best-effort quadrilateral).
        """
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        peri = cv2.arcLength(hull_pts, True)

        # Increase epsilon until the polygon collapses to (at most) 4 vertices
        for frac in (0.02, 0.04, 0.06, 0.08, 0.10, 0.15):
            approx = cv2.approxPolyDP(hull_pts, frac * peri, True).reshape(-1, 2)
            if len(approx) <= 4:
                break
        if len(approx) == 4:
            return approx.astype(np.float32)

        # Fallback: minimum-area rotated rectangle
        return cv2.boxPoints(cv2.minAreaRect(hull_pts)).astype(np.float32)

    def _quad_skew(self, quad: np.ndarray) -> float:
        """Perspective skew of a quadrilateral: deviation of its angles from 90°.

        A frontal (un-tilted) board projects to a near-rectangle (all 90°). Tilting
        the board in depth makes the angles diverge from 90°, which is exactly the
        perspective variation calibration needs.

        Args:
            quad: A (4, 2) array of ordered corner points.

        Returns:
            Skew in [0, 1] (0 = frontal rectangle, 1 = strongly skewed).
        """
        if len(quad) < 4:
            return 1.0

        # Order points so consecutive vertices share an edge (CW/CCW around center)
        c = quad.mean(axis=0)
        order = np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))
        q = quad[order]

        deviations = []
        for i in range(4):
            p1, p2, p3 = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
            v1, v2 = p1 - p2, p3 - p2
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-6 or n2 < 1e-6:
                return 1.0
            cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
            deviations.append(abs(acos(cos_a) - pi / 2))

        # Normalize: a 45° deviation already counts as fully skewed
        return float(min(1.0, max(deviations) / (pi / 4)))

    def _inplane_rotation(self, pts: np.ndarray) -> float:
        """In-plane (roll) rotation of the board in degrees [0, 90).

        Args:
            pts: Corner points as an (N, 2) array.

        Returns:
            Rotation angle in degrees, folded to [0, 90).
        """
        if len(pts) < 3:
            return 0.0
        angle = cv2.minAreaRect(pts.astype(np.float32))[2]
        return float(abs(angle) % 90)

    # ──────────────────────────── Evaluation ───────────────────────────

    def _sharpness(self, image: np.ndarray, corners: np.ndarray) -> float:
        """Focus score of the board region (variance of Laplacian).

        Computed on the board ROI resized to a fixed width so the score is roughly
        resolution-independent and comparable across views.

        Args:
            image: BGR or grayscale frame.
            corners: Detected corner points used to crop the board ROI.

        Returns:
            The variance of the Laplacian (higher = sharper).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        x, y, bw, bh = cv2.boundingRect(corners.reshape(-1, 2).astype(np.float32))
        roi = gray[max(y, 0):y + bh, max(x, 0):x + bw]
        if roi.size == 0:
            return 0.0
        if roi.shape[1] > 480:
            scale = 480.0 / roi.shape[1]
            roi = cv2.resize(roi, (480, max(1, int(roi.shape[0] * scale))))
        return float(cv2.Laplacian(roi, cv2.CV_64F).var())

    def evaluate(self, corners: np.ndarray, ids: Optional[np.ndarray] = None,
                 image_path: str = "", timestamp: float = 0.0,
                 image: Optional[np.ndarray] = None) -> CalibrationSample:
        """Evaluate a view *without* mutating state.

        This is safe to call on every live frame: it decides whether the view
        *would* be accepted, but does not record it. Call :meth:`commit` to keep it.

        Args:
            corners: Detected corner points.
            ids: Corner IDs (optional).
            image_path: Source image path (optional).
            timestamp: Sample timestamp (optional).
            image: Source frame; if given, a focus (sharpness) score is computed and,
                when ``min_sharpness`` > 0, blurry views are rejected.

        Returns:
            A populated :class:`CalibrationSample` (``is_accepted`` / ``reject_reason`` set).
        """
        m = self.compute_metrics(corners)
        sharpness = self._sharpness(image, corners) if image is not None else -1.0
        sample = CalibrationSample(
            corners=corners, ids=ids, image_path=image_path, timestamp=timestamp,
            x=m["x"], y=m["y"], size=m["size"], skew=m["skew"], rotation=m["rotation"],
            sharpness=sharpness,
            feature=np.array([m["x"], m["y"], m["size"], m["skew"]]),
        )

        n_corners = len(corners.reshape(-1, 2))
        if n_corners < self.min_corners:
            sample.reject_reason = f"too few corners ({n_corners})"
        elif sample.size < self.min_size_ratio:
            sample.reject_reason = "board too far / small"
        elif sample.size > self.max_size_ratio:
            sample.reject_reason = "board too close / large"
        elif sample.skew > self.max_skew:
            sample.reject_reason = "extreme skew"
        elif self.min_sharpness > 0 and 0 <= sharpness < self.min_sharpness:
            sample.reject_reason = f"too blurry ({sharpness:.0f})"
        else:
            dist = self._nearest_distance(sample.feature)
            if dist < self.min_distance_threshold:
                sample.reject_reason = "duplicate (move/tilt the board)"
            else:
                sample.is_accepted = True

        return sample

    def _nearest_distance(self, feature: np.ndarray) -> float:
        """Manhattan distance to the closest accepted sample (``inf`` if none).

        Args:
            feature: The [x, y, size, skew] vector of the candidate view.

        Returns:
            Distance to the nearest accepted sample in feature space.
        """
        if not self.accepted_samples:
            return float("inf")
        existing = np.array([s.feature for s in self.accepted_samples])
        return float(np.min(np.sum(np.abs(existing - feature), axis=1)))

    def commit(self, sample: CalibrationSample) -> None:
        """Record an accepted sample and update all coverage accumulators.

        Args:
            sample: The sample to keep (typically one returned by :meth:`evaluate`).
        """
        sample.is_accepted = True
        self.accepted_samples.append(sample)

        w, h = self.image_size
        pts = sample.corners.reshape(-1, 2)

        # Per-corner heatmap
        hx = np.clip((pts[:, 0] / w * self.heat_w).astype(int), 0, self.heat_w - 1)
        hy = np.clip((pts[:, 1] / h * self.heat_h).astype(int), 0, self.heat_h - 1)
        np.add.at(self.heatmap, (hy, hx), 1.0)

        # Coarse corner occupancy (position coverage + guidance)
        cols, rows = self.coverage_grid
        cx = np.clip((pts[:, 0] / w * cols).astype(int), 0, cols - 1)
        cy = np.clip((pts[:, 1] / h * rows).astype(int), 0, rows - 1)
        np.add.at(self.cell_counts, (cy, cx), 1)

        # Size / tilt histograms
        s_idx = int(np.clip((sample.size - self.min_size_ratio) /
                            (self.size_cov_max - self.min_size_ratio) * self.n_size_bins,
                            0, self.n_size_bins - 1))
        t_idx = int(np.clip(sample.skew / self.max_skew * self.n_tilt_bins,
                            0, self.n_tilt_bins - 1))
        self.size_bins[s_idx] += 1
        self.tilt_bins[t_idx] += 1

        # Queue the incremental calibration (runs on the background worker)
        self._queue_live_calibration(sample)

        # Kept quiet at INFO (the caller logs a single "✅ Saved <path>"); the
        # per-sample metrics stay available at DEBUG for troubleshooting.
        logging.debug(f"Sample kept: pos=({sample.x:.2f}, {sample.y:.2f}) "
                      f"size={sample.size:.3f} skew={sample.skew:.2f} "
                      f"rot={sample.rotation:.0f}° | total={len(self.accepted_samples)}")

    def _queue_live_calibration(self, sample: CalibrationSample) -> None:
        """Add a committed view's points and wake the background calibration worker.

        The board↔image point match is cheap, so it stays on the caller's thread; the
        expensive ``cv2.calibrateCamera`` is deferred to :meth:`_calibration_worker`.

        Args:
            sample: The freshly committed sample.
        """
        if self.board is None or sample.ids is None:
            return
        try:
            obj, img = self.board.matchImagePoints(sample.corners, sample.ids)
        except Exception:
            return
        if obj is None or img is None or len(obj) < 4:
            return

        with self._cal_lock:
            self.obj_points.append(obj.reshape(-1, 1, 3).astype(np.float32))
            self.img_points.append(img.reshape(-1, 1, 2).astype(np.float32))
        self._cal_dirty.set()  # ask the worker to recompute the live RMS

    def _calibration_worker(self) -> None:
        """Recompute the live reprojection error off the capture thread.

        Waits for new points, snapshots them under the lock, then runs the (GIL-
        releasing) ``cv2.calibrateCamera`` outside the lock. Multiple commits that
        arrive during one calibration coalesce into a single follow-up recompute.
        """
        while not self._cal_stop.is_set():
            if not self._cal_dirty.wait(timeout=0.2):
                continue
            self._cal_dirty.clear()
            with self._cal_lock:
                if len(self.obj_points) < 6:
                    continue
                obj = list(self.obj_points)
                img = list(self.img_points)
            try:
                rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                    obj, img, self.image_size, None, None)
            except cv2.error:
                continue
            self.live_rms = float(rms)
            try:
                err = cv2.projectPoints(obj[-1], rvecs[-1], tvecs[-1], K, dist)[0]
                self.last_view_error = float(cv2.norm(err.reshape(-1, 2),
                                                      img[-1].reshape(-1, 2),
                                                      cv2.NORM_L2) / len(err))
            except cv2.error:
                pass

    def close(self) -> None:
        """Stop the background calibration worker (call when collection ends).

        Waits for any in-flight ``cv2.calibrateCamera`` to finish before returning, so
        the worker is never force-killed mid-native-call at interpreter exit.
        """
        self._cal_stop.set()
        self._cal_dirty.set()  # unblock the worker so it can see the stop flag
        if self._cal_thread is not None:
            self._cal_thread.join(timeout=10.0)
            self._cal_thread = None

    # ──────────────────────────── Coverage ─────────────────────────────

    def coverage(self) -> Dict[str, float]:
        """Compute occupancy-based coverage on each diversity axis.

        Returns:
            Dictionary with position/size/tilt/overall coverage in [0, 1].
        """
        # Position is occupancy (every grid cell touched at least once — already hard
        # to fully satisfy via the frame edges). The size/tilt axes use graduated
        # coverage so a single view no longer saturates them.
        pos = float(np.count_nonzero(self.cell_counts) / self.cell_counts.size)
        size = self._bin_coverage(self.size_bins)
        tilt = self._bin_coverage(self.tilt_bins)
        # Split the size axis: small board = far = ZOOM-OUT (low bins), large board =
        # near = ZOOM-IN (high bins). Tracking them apart tells the operator exactly
        # which one is still missing instead of a vague "vary the distance".
        half = max(1, self.n_size_bins // 2)
        zoom_out = self._bin_coverage(self.size_bins[:half])    # far / small
        zoom_in = self._bin_coverage(self.size_bins[-half:])    # near / large
        overall = float(np.mean([pos, size, tilt]))
        return {"position": pos, "size": size, "tilt": tilt,
                "zoom_in": zoom_in, "zoom_out": zoom_out, "overall": overall}

    def _bin_coverage(self, counts: np.ndarray) -> float:
        """Graduated coverage of a set of bins (full credit only at ``min_per_bin``).

        Unlike plain occupancy (which a single sample saturates), each bin contributes
        ``min(count / min_per_bin, 1)``, so an axis reads as fully covered only once
        every bin has been sampled several times — reflecting real spread.

        Args:
            counts: Per-bin sample counts.

        Returns:
            Mean per-bin coverage in [0, 1].
        """
        if counts.size == 0:
            return 0.0
        return float(np.mean(np.clip(counts / self.min_per_bin, 0.0, 1.0)))

    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information.

        Returns:
            Dictionary with sample counts, coverage and a readiness flag.
        """
        cov = self.coverage()
        n = len(self.accepted_samples)
        is_sufficient = (n >= self.target_samples and cov["position"] >= 0.80 and
                         cov["zoom_in"] >= 0.50 and cov["zoom_out"] >= 0.50 and
                         cov["tilt"] >= 0.75)
        return {
            "accepted_samples": n,
            "target_samples": self.target_samples,
            "progress_ratio": n / self.target_samples if self.target_samples else 0.0,
            "coverage": cov,
            "is_sufficient": is_sufficient,
        }

    def next_action_hint(self) -> Tuple[str, Optional[Tuple[float, float]]]:
        """Suggest the single most useful next move for the operator.

        Returns:
            A tuple ``(instruction, target)`` where ``target`` is a normalized
            (x, y) point to aim the board at, or ``None`` when no target applies.
        """
        cov = self.coverage()

        # Address the single weakest axis first (position carries a concrete target)
        axes = {"position": cov["position"], "zoom_in": cov["zoom_in"],
                "zoom_out": cov["zoom_out"], "tilt": cov["tilt"]}
        weakest = min(axes, key=axes.get)

        if weakest == "position":
            ry, rx = np.unravel_index(np.argmin(self.cell_counts), self.cell_counts.shape)
            cols, rows = self.coverage_grid
            target = ((rx + 0.5) / cols, (ry + 0.5) / rows)
            horiz = ["LEFT", "CENTER", "RIGHT"][min(int(rx / cols * 3), 2)]
            vert = ["TOP", "MIDDLE", "BOTTOM"][min(int(ry / rows * 3), 2)]
            where = "CENTER" if (horiz == "CENTER" and vert == "MIDDLE") else f"{vert}-{horiz}"
            return f"Move board to {where}", target

        if weakest == "zoom_in":
            return "ZOOM IN - move board CLOSER (large/near views)", None
        if weakest == "zoom_out":
            return "ZOOM OUT - move board FARTHER (small/far views)", None
        return "TILT the board in depth (perspective views)", None

    # ─────────────────────────── Visualization ─────────────────────────

    def _heat_field(self, size: Tuple[int, int]) -> np.ndarray:
        """Return a normalized [0, 1] corner-density field at the given (w, h) size.

        Args:
            size: Target field size as (width, height).

        Returns:
            A (height, width) float32 array in [0, 1] (log-scaled and blurred).
        """
        w, h = size
        if self.heatmap.max() <= 0:
            return np.zeros((h, w), dtype=np.float32)

        # Log scaling keeps a few very hot cells from washing out everything else
        heat = np.maximum(cv2.resize(self.heatmap, (w, h), interpolation=cv2.INTER_CUBIC), 0.0)
        heat = np.log1p(heat)
        heat = heat / heat.max()
        return cv2.GaussianBlur(heat, (0, 0), sigmaX=w / 100.0)

    def render_heatmap_overlay(self, frame: np.ndarray, alpha: float = 0.92,
                               gamma: float = 0.45) -> np.ndarray:
        """Blend the corner-coverage heatmap onto a frame.

        Empty regions stay transparent; covered regions become clearly visible. A
        gamma < 1 lifts low/mid densities so even lightly covered areas show up (not
        only the single hottest spot), while truly empty pixels remain see-through.

        Args:
            frame: BGR frame to draw on.
            alpha: Maximum heatmap opacity in [0, 1] (reached at the hottest pixel).
            gamma: Tone curve for opacity/color (< 1 brightens sparse coverage).

        Returns:
            A new frame with the heatmap blended proportionally to coverage.
        """
        h, w = frame.shape[:2]
        field = self._heat_field((w, h))
        if field.max() <= 0:
            return frame.copy()

        vis = np.clip(np.power(field, gamma) * 1.3, 0.0, 1.0)  # lift + boost coverage
        color = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
        a = (vis * alpha)[..., None]  # per-pixel opacity grows with corner density
        return (frame * (1 - a) + color * a).astype(np.uint8)

    def render_frame(self, frame: np.ndarray, sample: Optional[CalibrationSample],
                     show_heatmap: bool = False) -> np.ndarray:
        """Augment the live camera frame (kept at full resolution, view unobstructed).

        Only lightweight, non-blocking overlays go on the frame: a guidance target,
        an optional translucent heatmap, and a thin status border. All textual
        guidance lives in the separate panel (see :meth:`render_panel`).

        Args:
            frame: Current BGR frame (already carrying the detector's corner dots).
            sample: Result of :meth:`evaluate` for this frame (or ``None``).
            show_heatmap: Whether to blend the translucent corner heatmap onto the frame.

        Returns:
            The augmented frame, the same size as the input.
        """
        base = self.render_heatmap_overlay(frame) if show_heatmap else frame.copy()
        s = frame.shape[1] / 1920.0
        progress = self.get_progress_info()

        if sample is not None:
            border_col = (80, 220, 80) if sample.is_accepted else (60, 170, 240)
        else:
            border_col = (60, 60, 230)

        _, target = self.next_action_hint()
        if (not progress["is_sufficient"]) and target is not None:
            self._draw_target(base, target, s)

        # Thin status border (green = good view, orange = skip, red = no board)
        t = max(3, int(8 * s))
        cv2.rectangle(base, (0, 0), (base.shape[1] - 1, base.shape[0] - 1), border_col, t)
        return base

    def render_panel(self, sample: Optional[CalibrationSample],
                     panel_height: int = 900, panel_width: int = 460) -> np.ndarray:
        """Render the stand-alone guidance panel (shown in its own window).

        Args:
            sample: Result of :meth:`evaluate` for this frame (or ``None``).
            panel_height: Panel canvas height in pixels.
            panel_width: Panel canvas width in pixels.

        Returns:
            The panel image (a standalone canvas, not attached to the frame).
        """
        progress = self.get_progress_info()
        return self._build_panel(progress, progress["coverage"],
                                 self.next_action_hint()[0], sample,
                                 panel_height, panel_width)

    def render_report_view(self) -> np.ndarray:
        """Return the full report (heatmap + radar) as a BGR image, cached.

        This is the same figure produced by :meth:`generate_heatmap` (the
        ``final_heatmap.png`` look). It is rebuilt only when a new view is committed,
        so toggling it on during collection is cheap between saves.

        Returns:
            The report as a BGR image ready for ``cv2.imshow``.
        """
        n = len(self.accepted_samples)
        if self._report_cache is None or self._report_cache_n != n:
            rgb = self.generate_heatmap()
            self._report_cache = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            self._report_cache_n = n
        return self._report_cache

    def _draw_target(self, frame: np.ndarray, target: Tuple[float, float], s: float) -> None:
        """Draw a pulsing target marker where the next view is most needed."""
        h, w = frame.shape[:2]
        cx, cy = int(target[0] * w), int(target[1] * h)
        r = int(60 * s)
        cv2.circle(frame, (cx, cy), r, (0, 215, 255), max(2, int(3 * s)), cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), int(r * 0.4), (0, 215, 255), -1, cv2.LINE_AA)
        cv2.drawMarker(frame, (cx, cy), (0, 215, 255), cv2.MARKER_CROSS,
                       int(r * 1.6), max(1, int(2 * s)), cv2.LINE_AA)

    def _build_panel(self, progress: Dict[str, Any], cov: Dict[str, float],
                     instruction: str, sample: Optional[CalibrationSample],
                     panel_h: int, pw: int) -> np.ndarray:
        """Build the stand-alone guidance panel canvas."""
        panel = np.full((panel_h, pw, 3), 35, dtype=np.uint8)
        s = pw / 600.0  # design scale (tuned for a ~600 px wide panel)

        x0 = int(22 * s)
        line = int(40 * s)

        # Title band drawn flush to the top edge, with the heading padded inside it
        # so the glyphs never get clipped against the top of the window.
        band_h = int(line * 1.7)
        cv2.rectangle(panel, (0, 0), (pw, band_h), (52, 52, 52), -1)
        cv2.putText(panel, "CALIBRATION READINESS", (x0, int(band_h * 0.64)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.82 * s, (255, 255, 255),
                    max(1, int(2 * s)), cv2.LINE_AA)

        y = band_h + int(line)

        def text(txt, color=(235, 235, 235), scale=0.7, dy=None, thick=None):
            nonlocal y
            cv2.putText(panel, txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, scale * s,
                        color, thick or max(1, int(2 * s)), cv2.LINE_AA)
            y += dy if dy is not None else line

        # Current-view status (moved off the frame so the camera view stays clear)
        if sample is None:
            text("No board detected", (60, 60, 230), 0.8, int(line * 1.1))
        elif sample.is_accepted:
            text("GOOD - press 's' to keep", (80, 220, 80), 0.75, int(line * 1.1))
        else:
            text(f"SKIP: {sample.reject_reason}", (60, 170, 240), 0.7, int(line * 1.1))

        n, tgt = progress["accepted_samples"], progress["target_samples"]
        text(f"Samples: {n}/{tgt}", (120, 230, 255), 0.8, int(line * 1.1))

        # Real-time quality readout: live reprojection error + current-frame focus
        self._quality_readout(panel, sample, x0, y, pw, s)
        y += int(line * 1.7)

        for name, key in (("Position (where)", "position"),
                          ("Zoom-in (near)", "zoom_in"),
                          ("Zoom-out (far)", "zoom_out"),
                          ("Tilt (angle)", "tilt")):
            self._bar(panel, name, cov[key], x0, y, pw, s)
            y += int(line * 1.4)

        y += int(line * 0.3)
        self._mini_map(panel, x0, y, pw, s)
        cols, rows = self.coverage_grid
        y += int((pw - 2 * x0) / cols * rows) + int(line * 1.4)

        # Next-action call-out
        text("NEXT:", (0, 215, 255), 0.78, int(line * 0.85))
        for chunk in self._wrap(instruction, 26):
            text(chunk, (0, 215, 255), 0.74)

        # Key legend (single-window controls)
        cv2.putText(panel, "s save   h heat   m map   q quit",
                    (x0, panel_h - int(line * 2.0)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, (150, 150, 150), max(1, int(1 * s)), cv2.LINE_AA)

        y = panel_h - int(line * 0.9)
        if progress["is_sufficient"]:
            text("READY TO CALIBRATE", (80, 230, 80), 0.85)
        else:
            text("Keep collecting...", (180, 180, 180), 0.72)

        return panel

    def _quality_readout(self, panel: np.ndarray, sample: Optional[CalibrationSample],
                         x0: int, y: int, pw: int, s: float) -> None:
        """Draw the live calibration-quality readout (reprojection error + focus)."""
        if self.live_rms is None:
            rms_txt, rms_col = "Live RMS: warming up...", (160, 160, 160)
        else:
            rms_col = ((80, 230, 80) if self.live_rms < 0.7 else
                       (60, 200, 230) if self.live_rms < 1.5 else (70, 90, 240))
            rms_txt = f"Live RMS: {self.live_rms:.2f} px"
        cv2.putText(panel, rms_txt, (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7 * s, rms_col, max(1, int(2 * s)), cv2.LINE_AA)

        if sample is not None and sample.sharpness >= 0:
            blurry = self.min_sharpness > 0 and sample.sharpness < self.min_sharpness
            f_col = (60, 170, 240) if blurry else (200, 200, 200)
            thr = f" / min {self.min_sharpness:.0f}" if self.min_sharpness > 0 else ""
            label = f"Focus: {sample.sharpness:.0f}{thr}" + (" BLUR" if blurry else "")
            cv2.putText(panel, label, (x0, y + int(34 * s)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, f_col, max(1, int(2 * s)), cv2.LINE_AA)

    def _bar(self, panel: np.ndarray, name: str, value: float,
             x0: int, y: int, pw: int, s: float) -> None:
        """Draw a labeled progress bar for one coverage axis."""
        cv2.putText(panel, f"{name}: {value:.0%}", (x0, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62 * s, (235, 235, 235), max(1, int(2 * s)), cv2.LINE_AA)
        by = y + int(12 * s)
        bw = pw - 2 * x0
        bh = int(16 * s)
        cv2.rectangle(panel, (x0, by), (x0 + bw, by + bh), (70, 70, 70), -1)
        color = (80, 230, 80) if value >= 0.8 else (60, 200, 230) if value >= 0.5 else (70, 90, 240)
        cv2.rectangle(panel, (x0, by), (x0 + int(bw * min(value, 1.0)), by + bh), color, -1)

    def _mini_map(self, panel: np.ndarray, x0: int, y: int, pw: int, s: float) -> None:
        """Draw a compact occupancy map of the coarse position grid (JET: cold→hot)."""
        cols, rows = self.coverage_grid
        bw = pw - 2 * x0
        cell = bw / cols
        bh = int(cell * rows)
        peak = max(1, int(self.cell_counts.max()))
        jet = cv2.applyColorMap((self.cell_counts / peak * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for r in range(rows):
            for c in range(cols):
                col = (55, 55, 55) if self.cell_counts[r, c] == 0 else tuple(int(v) for v in jet[r, c])
                p1 = (x0 + int(c * cell), y + int(r * cell))
                p2 = (x0 + int((c + 1) * cell), y + int((r + 1) * cell))
                cv2.rectangle(panel, p1, p2, col, -1)
                cv2.rectangle(panel, p1, p2, (35, 35, 35), 1)
        cv2.putText(panel, "corner coverage map (gray = empty)", (x0, y + bh + int(20 * s)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (160, 160, 160), max(1, int(1 * s)), cv2.LINE_AA)

    @staticmethod
    def _wrap(text: str, width: int) -> List[str]:
        """Greedy word-wrap a string to ``width`` characters per line."""
        words, lines, cur = text.split(), [], ""
        for word in words:
            if len(cur) + len(word) + 1 > width:
                lines.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}".strip()
        if cur:
            lines.append(cur)
        return lines

    # ───────────────────────────── Reports ─────────────────────────────

    def generate_heatmap(self, save_path: Optional[str] = None) -> np.ndarray:
        """Render the coverage report (corner heatmap + readiness radar).

        Args:
            save_path: Optional path to save the figure.

        Returns:
            The report image as an RGB numpy array.
        """
        matplotlib.use("Agg", force=False)
        w, h = self.image_size
        cov = self.coverage()

        fig = plt.figure(figsize=(13.5, 6.0), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.0])
        ax_hm = fig.add_subplot(gs[0, 0])
        ax_radar = fig.add_subplot(gs[0, 1], projection="polar")

        # ── Corner-density heatmap ──
        heat = np.log1p(np.maximum(cv2.resize(self.heatmap, (w, h), interpolation=cv2.INTER_CUBIC), 0.0))
        im = ax_hm.imshow(heat, cmap="jet", extent=[0, w, h, 0], aspect="auto")
        fig.colorbar(im, ax=ax_hm, label="corner density (log)", fraction=0.046, pad=0.02)

        # Board centers as crosses (one per kept view), outlined so they read on any hue
        if self.accepted_samples:
            xs = [s.x * w for s in self.accepted_samples]
            ys = [s.y * h for s in self.accepted_samples]
            sc = ax_hm.scatter(xs, ys, s=22, marker="x", c="black", linewidths=1.0)
            sc.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])

        ax_hm.set_xlim(0, w)
        ax_hm.set_ylim(h, 0)
        ax_hm.set_title(f"Corner coverage heatmap — {len(self.accepted_samples)} views", fontsize=13)
        ax_hm.set_xlabel("X (px)")
        ax_hm.set_ylabel("Y (px)")

        # ── Readiness radar ──
        self._draw_radar(ax_radar, cov)

        fig.supxlabel("  •  ".join(self._generate_recommendations()), fontsize=10, color="#334155")
        fig.suptitle("Camera Calibration Data Quality", fontsize=16, fontweight="bold")
        fig.get_layout_engine().set(w_pad=0.02, wspace=0.0)  # pull the two panels together

        if save_path:
            fig.savefig(save_path, dpi=140, bbox_inches="tight", facecolor="white")
            logging.info(f"📊 Coverage report saved to {save_path}")

        fig.canvas.draw()
        report = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return report

    def _draw_radar(self, ax, cov: Dict[str, float]) -> None:
        """Draw the calibration-readiness radar (Position / Distance / Tilt / Count).

        Args:
            ax: A polar matplotlib axes.
            cov: Coverage dictionary from :meth:`coverage`.
        """
        # Fixed star layout (clockwise from top): Quantity, Position, Zoom-out,
        # Zoom-in, Tilt — so the user always reads the same axis in the same spot.
        labels = ["Quantity", "Position", "Zoom-out", "Zoom-in", "Tilt"]
        vals = [min(len(self.accepted_samples) / max(self.target_samples, 1), 1.0),
                cov["position"], cov["zoom_out"], cov["zoom_in"], cov["tilt"]]
        overall = float(np.mean(vals))
        accent = "#16a34a" if overall >= 0.8 else "#d97706" if overall >= 0.5 else "#dc2626"

        ax.set_theta_offset(np.pi / 2)   # first axis at the top
        ax.set_theta_direction(-1)       # remaining axes go clockwise
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        loop = np.concatenate([angles, angles[:1]])
        ring = np.linspace(0, 2 * np.pi, 100)

        ax.plot(ring, [0.8] * 100, color="#94a3b8", linestyle="--", linewidth=1.0)  # target ring
        ax.fill(loop, vals + vals[:1], color=accent, alpha=0.25)
        ax.plot(loop, vals + vals[:1], color=accent, linewidth=2.2)
        ax.scatter(angles, vals, color="#dc2626", s=55, zorder=5,
                   edgecolors="white", linewidths=1.0)  # red dots at the star tips
        for ang, v in zip(angles, vals):
            r_text = v - 0.12 if v > 0.3 else v + 0.12  # keep labels inside the rim
            ax.text(ang, r_text, f"{v:.0%}", ha="center", va="center",
                    fontsize=11, color="#0f172a", fontweight="bold")

        ax.set_xticks(angles)
        ax.set_xticklabels(labels, fontsize=11)
        ax.tick_params(axis="x", pad=14)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color="#64748b")
        ax.set_rlabel_position(22)
        ax.set_title(f"Calibration readiness — {overall:.0%}",
                     fontsize=13, pad=22, color=accent)

    def filter_existing_dataset(self, samples: List[CalibrationSample]) -> List[CalibrationSample]:
        """Select a maximally diverse subset via farthest-point sampling.

        Compared to a greedy threshold pass, farthest-point sampling guarantees the
        kept views are spread across the whole feature space — e.g. 100 near-center
        views collapse to a handful while edge/tilted views are preserved.

        Args:
            samples: Candidate samples (already passing basic quality is not required).

        Returns:
            The kept, diverse subset (also committed into this judge's accumulators).
        """
        self.accepted_samples = []
        self.heatmap[:] = 0
        self.cell_counts[:] = 0
        self.size_bins[:] = 0
        self.tilt_bins[:] = 0
        self.obj_points, self.img_points = [], []
        self.live_rms = self.last_view_error = None

        valid = [s for s in samples if self._passes_basic_quality(s)]
        if not valid:
            logging.warning("⚠️ No samples passed basic quality")
            return []

        feats = np.array([s.feature for s in valid])

        # Seed with the most "central" view, then repeatedly take the farthest one
        order = [int(np.argmin(np.sum(np.abs(feats - feats.mean(axis=0)), axis=1)))]
        dist = np.sum(np.abs(feats - feats[order[0]]), axis=1)
        while len(order) < min(self.target_samples, len(valid)):
            nxt = int(np.argmax(dist))
            if dist[nxt] < self.min_distance_threshold:
                break  # everything left is a near-duplicate of what we already kept
            order.append(nxt)
            dist = np.minimum(dist, np.sum(np.abs(feats - feats[nxt]), axis=1))

        kept = [valid[i] for i in order]
        for s in kept:
            self.commit(s)

        logging.info(f"🧹 Diversity filter: {len(samples)} → {len(kept)} samples")
        return kept

    def _passes_basic_quality(self, sample: CalibrationSample) -> bool:
        """Check size/skew/corner-count limits (no edge rejection — edges are wanted).

        Args:
            sample: Sample to check.

        Returns:
            True if the sample is within the basic quality limits.
        """
        return (len(sample.corners.reshape(-1, 2)) >= self.min_corners and
                self.min_size_ratio <= sample.size <= self.max_size_ratio and
                sample.skew <= self.max_skew)

    def export_summary(self, output_path: str) -> None:
        """Export a JSON summary of coverage and per-sample metrics.

        Args:
            output_path: Path to write the JSON summary.
        """
        cov = self.coverage()
        sizes = [s.size for s in self.accepted_samples]
        skews = [s.skew for s in self.accepted_samples]

        summary = {
            "total_samples": len(self.accepted_samples),
            "target_samples": self.target_samples,
            "coverage": cov,
            "metrics": {
                "size_range": [float(min(sizes)), float(max(sizes))] if sizes else [0.0, 0.0],
                "skew_range": [float(min(skews)), float(max(skews))] if skews else [0.0, 0.0],
                "size_histogram": self.size_bins.tolist(),
                "tilt_histogram": self.tilt_bins.tolist(),
            },
            "recommendations": self._generate_recommendations(),
        }
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=2)
        logging.info(f"📄 Summary exported to {output_path}")

    def _generate_recommendations(self) -> List[str]:
        """Generate human-readable suggestions from current coverage.

        Returns:
            A list of recommendation strings.
        """
        cov = self.coverage()
        recs = []
        if cov["position"] < 0.8:
            recs.append("Cover more of the frame, especially the edges and corners")
        if cov["zoom_in"] < 0.5:
            recs.append("Collect NEAR views (zoom in / move closer)")
        if cov["zoom_out"] < 0.5:
            recs.append("Collect FAR views (zoom out / move farther)")
        if cov["tilt"] < 0.75:
            recs.append("Add more tilted/perspective views")
        if len(self.accepted_samples) < self.target_samples:
            recs.append(f"Collect more samples ({len(self.accepted_samples)}/{self.target_samples})")
        if not recs:
            recs.append("Coverage looks great — ready for calibration!")
        return recs
