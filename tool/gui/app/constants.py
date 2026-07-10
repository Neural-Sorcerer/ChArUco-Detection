"""
UI Choice Constants
===================

Enumerations offered in the GUI drop-downs: resolution presets, pixel formats,
ArUco dictionaries and calibration modes. Resolution presets are reused from the
existing CLI config so the GUI and CLI stay in sync.
"""
from __future__ import annotations

# === Third-Party Libraries ===
import cv2

# === Local ===
from configs.config import Resolution

__all__ = [
    "RESOLUTION_PRESETS",
    "FOURCC_OPTIONS",
    "ARUCO_DICT_NAMES",
    "CALIBRATION_MODES",
    "aruco_dictionary_id",
]

# name -> (width, height); pulled straight from the CLI's Resolution presets.
RESOLUTION_PRESETS: dict[str, tuple[int, int]] = {
    "SS (640x360)": Resolution.SS,
    "SD (640x480)": Resolution.SD,
    "HD (1280x720)": Resolution.HD,
    "FHD (1920x1080)": Resolution.FHD,
    "UHD (3840x2160)": Resolution.UHD,
    "OMS (2592x1800)": Resolution.OMS,
}

# Common V4L2 pixel formats a webcam is likely to expose.
FOURCC_OPTIONS: list[str] = ["MJPG", "YUYV", "H264", "GREY"]

# ArUco dictionaries exposed in the board settings drop-down.
ARUCO_DICT_NAMES: list[str] = [
    "DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
    "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
    "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
    "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
]

# Calibration intent. Only "Intrinsic" and "Data collection only" are wired in
# the MVP; the rest are placeholders the architecture is ready to grow into.
CALIBRATION_MODES: list[str] = [
    "Intrinsic",
    "Multi-camera",
    "Data collection only",
]


def aruco_dictionary_id(name: str) -> int:
    """
    Resolve an ArUco dictionary name to its ``cv2.aruco`` integer id.

    Args:
        name: A dictionary name such as ``"DICT_6X6_1000"``

    Returns:
        The matching ``cv2.aruco.DICT_*`` integer constant

    Raises:
        ValueError: If the name is not a known ``cv2.aruco`` dictionary
    """
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown ArUco dictionary: {name}")
    return getattr(cv2.aruco, name)
