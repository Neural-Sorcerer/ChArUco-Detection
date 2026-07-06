"""
Camera Discovery
================

Enumerate the machine's video devices and, where possible, their supported
pixel formats / resolutions / frame rates. Uses ``v4l2-ctl`` on Linux and falls
back to probing OpenCV indices when it is unavailable.
"""
from __future__ import annotations

# === Standard Libraries ===
import re
import shutil
import logging
import subprocess
from dataclasses import dataclass, field

# === Third-Party Libraries ===
import cv2

# === Local ===
from configs.config import Resolution

__all__ = ["CameraDevice", "list_cameras"]

log = logging.getLogger(__name__)

# Resolutions offered when a device exposes no format information.
_FALLBACK_RESOLUTIONS: list[tuple[int, int]] = [
    Resolution.HD, Resolution.SD, Resolution.FHD, Resolution.SS,
]
_FALLBACK_FPS: list[float] = [30.0, 15.0]

# --- v4l2-ctl output parsers ---
_FORMAT_RE = re.compile(r"\[\d+\]:\s*'(\w+)'")          # [0]: 'MJPG' (...)
_SIZE_RE = re.compile(r"Size:\s*Discrete\s*(\d+)x(\d+)")  # Size: Discrete 1280x720
_FPS_RE = re.compile(r"\(([\d.]+)\s*fps\)")              # Interval: ... (30.000 fps)
_VIDEO_NODE_RE = re.compile(r"(/dev/video\d+)")


@dataclass
class CameraDevice:
    """
    A single capture device and the formats it advertises.

    Attributes:
        index: The V4L2 index N of ``/dev/videoN`` (what OpenCV opens)
        path: The device node path, e.g. ``/dev/video0``
        name: Human-readable device name from ``v4l2-ctl --list-devices``
        formats: ``{fourcc: {(w, h): [fps, ...]}}`` capability map
    """

    index: int
    path: str
    name: str
    formats: dict[str, dict[tuple[int, int], list[float]]] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """Return a drop-down-friendly ``name (/dev/videoN)`` label."""
        return f"{self.name} ({self.path})"

    def fourccs(self) -> list[str]:
        """Return the advertised pixel formats (may be empty)."""
        return list(self.formats.keys())

    def resolutions(self, fourcc: str | None = None) -> list[tuple[int, int]]:
        """
        Return supported resolutions, largest first.

        Args:
            fourcc: Restrict to one pixel format, or ``None`` for the union

        Returns:
            Sorted ``(width, height)`` list; falls back to presets if unknown
        """
        sizes: set[tuple[int, int]] = set()
        for fmt, size_map in self.formats.items():
            if fourcc is None or fmt == fourcc:
                sizes.update(size_map.keys())
        if not sizes:
            return list(_FALLBACK_RESOLUTIONS)
        return sorted(sizes, key=lambda wh: wh[0] * wh[1], reverse=True)

    def fps_options(self, resolution: tuple[int, int], fourcc: str | None = None) -> list[float]:
        """
        Return frame rates supported at a resolution, highest first.

        Args:
            resolution: The ``(width, height)`` to look up
            fourcc: Restrict to one pixel format, or ``None`` for the union

        Returns:
            Sorted list of frame rates; falls back to presets if unknown
        """
        rates: set[float] = set()
        for fmt, size_map in self.formats.items():
            if fourcc is None or fmt == fourcc:
                rates.update(size_map.get(resolution, []))
        if not rates:
            return list(_FALLBACK_FPS)
        return sorted(rates, reverse=True)


def list_cameras() -> list[CameraDevice]:
    """
    Discover capture devices, preferring ``v4l2-ctl`` and falling back to OpenCV.

    Returns:
        Capture devices with their capability maps; empty if none are found
    """
    if shutil.which("v4l2-ctl"):
        devices = _list_cameras_v4l2()
        if devices:
            return devices
        log.warning("v4l2-ctl found no usable devices - falling back to OpenCV probe")
    return _probe_opencv_indices()


def _list_cameras_v4l2() -> list[CameraDevice]:
    """Enumerate devices via ``v4l2-ctl``, keeping only capture-capable nodes."""
    node_to_name = _v4l2_node_names()
    devices: list[CameraDevice] = []

    for path, name in node_to_name.items():
        formats = _v4l2_formats(path)
        if not formats:
            # Metadata-only nodes (common on multi-stream cameras) expose no
            # capture formats - skip them so the drop-down stays clean.
            continue
        match = _VIDEO_NODE_RE.search(path)
        index = int(match.group(1).rsplit("video", 1)[1]) if match else -1
        devices.append(CameraDevice(index=index, path=path, name=name, formats=formats))

    devices.sort(key=lambda dev: dev.index)
    log.info("Discovered %d capture device(s) via v4l2-ctl", len(devices))
    return devices


def _v4l2_node_names() -> dict[str, str]:
    """Map each ``/dev/videoN`` node to its parent device name."""
    try:
        output = subprocess.run(
            ["v4l2-ctl", "--list-devices"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError) as error:
        log.warning("v4l2-ctl --list-devices failed: %s", error)
        return {}

    node_to_name: dict[str, str] = {}
    current_name = "Camera"
    for line in output.splitlines():
        if not line.strip():
            continue
        if not line[0].isspace():
            # Header line, e.g. "Integrated_Webcam_HD: ... (usb-...):"
            current_name = line.split("(")[0].strip().rstrip(":").strip()
        elif "/dev/video" in line:
            node_to_name[line.strip()] = current_name
    return node_to_name


def _v4l2_formats(path: str) -> dict[str, dict[tuple[int, int], list[float]]]:
    """Parse ``--list-formats-ext`` for one node into a capability map."""
    try:
        output = subprocess.run(
            ["v4l2-ctl", "-d", path, "--list-formats-ext"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (subprocess.SubprocessError, OSError) as error:
        log.warning("v4l2-ctl --list-formats-ext failed for %s: %s", path, error)
        return {}

    formats: dict[str, dict[tuple[int, int], list[float]]] = {}
    current_fmt: str | None = None
    current_size: tuple[int, int] | None = None

    for line in output.splitlines():
        fmt_match = _FORMAT_RE.search(line)
        if fmt_match:
            current_fmt = fmt_match.group(1)
            formats.setdefault(current_fmt, {})
            current_size = None
            continue

        size_match = _SIZE_RE.search(line)
        if size_match and current_fmt is not None:
            current_size = (int(size_match.group(1)), int(size_match.group(2)))
            formats[current_fmt].setdefault(current_size, [])
            continue

        fps_match = _FPS_RE.search(line)
        if fps_match and current_fmt is not None and current_size is not None:
            formats[current_fmt][current_size].append(float(fps_match.group(1)))

    return formats


def _probe_opencv_indices(max_index: int = 10) -> list[CameraDevice]:
    """Fallback: open indices 0..max_index-1 and keep the ones that respond."""
    devices: list[CameraDevice] = []
    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if cap.isOpened():
            devices.append(
                CameraDevice(index=index, path=f"/dev/video{index}", name=f"Camera {index}")
            )
        cap.release()
    log.info("Discovered %d device(s) via OpenCV probe", len(devices))
    return devices
