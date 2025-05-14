"""Configuration module for Charuco detection pipeline.

This module contains all configuration parameters for the Charuco detection pipeline,
including board specifications, camera settings, and detector parameters.
"""
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

import cv2
import numpy as np


class Resolution:
    """Standard resolution presets."""
    SS: Tuple[int, int] = (640, 360)    # Small Screen
    SD: Tuple[int, int] = (640, 480)    # Standard Definition
    HD: Tuple[int, int] = (1280, 720)   # High Definition
    FHD: Tuple[int, int] = (1920, 1080) # Full HD
    UHD: Tuple[int, int] = (3840, 2160) # Ultra HD (4K)


@dataclass
class CharucoBoardConfig:
    """Configuration for Charuco board."""
    board_id: int = 0
    x_squares: int = 7
    y_squares: int = 5
    square_length: float = 0.12             # in meters
    marker_length: Optional[float] = None   # in meters, defaults to 75% of square_length
    dictionary_type: int = cv2.aruco.DICT_6X6_1000
    
    def __post_init__(self):
        """Initialize derived parameters after initialization."""
        if self.marker_length is None:
            self.marker_length = self.square_length * 0.75
        
        self.size = (self.x_squares, self.y_squares)
        self.markers_per_board = int(self.x_squares * self.y_squares / 2.0)
        self.start_id = self.board_id * self.markers_per_board
        self.stop_id = (self.board_id + 1) * self.markers_per_board
        self.ids = np.array(range(self.start_id, self.stop_id), dtype=np.int32)
        
        # Create the dictionary
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(self.dictionary_type)
        
        # Create the board
        self.board = cv2.aruco.CharucoBoard(
            size=self.size,
            squareLength=self.square_length,
            markerLength=self.marker_length,
            dictionary=self.aruco_dict,
            ids=self.ids
        )


@dataclass
class DetectorConfig:
    """Configuration for Aruco detector parameters."""
    corner_refinement_method: int = cv2.aruco.CORNER_REFINE_SUBPIX
    corner_refinement_win_size: int = 5
    corner_refinement_max_iterations: int = 30
    corner_refinement_min_accuracy: float = 0.1
    adaptive_thresh_win_size_min: int = 3
    adaptive_thresh_win_size_max: int = 23
    adaptive_thresh_win_size_step: int = 10
    adaptive_thresh_constant: float = 7
    min_marker_perimeter_rate: float = 0.03
    max_marker_perimeter_rate: float = 4.0
    polygonal_approx_accuracy_rate: float = 0.03
    min_corner_distance_rate: float = 0.05
    min_distance_to_border: int = 3
    min_marker_distance_rate: float = 0.05
    use_aruco3_detection: bool = True
    
    def create_detector_params(self) -> cv2.aruco.DetectorParameters:
        """Create and return OpenCV Aruco detector parameters."""
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = self.corner_refinement_method
        params.cornerRefinementWinSize = self.corner_refinement_win_size
        params.cornerRefinementMaxIterations = self.corner_refinement_max_iterations
        params.cornerRefinementMinAccuracy = self.corner_refinement_min_accuracy
        params.adaptiveThreshWinSizeMin = self.adaptive_thresh_win_size_min
        params.adaptiveThreshWinSizeMax = self.adaptive_thresh_win_size_max
        params.adaptiveThreshWinSizeStep = self.adaptive_thresh_win_size_step
        params.adaptiveThreshConstant = self.adaptive_thresh_constant
        params.minMarkerPerimeterRate = self.min_marker_perimeter_rate
        params.maxMarkerPerimeterRate = self.max_marker_perimeter_rate
        params.polygonalApproxAccuracyRate = self.polygonal_approx_accuracy_rate
        params.minCornerDistanceRate = self.min_corner_distance_rate
        params.minDistanceToBorder = self.min_distance_to_border
        params.minMarkerDistanceRate = self.min_marker_distance_rate
        params.useAruco3Detection = self.use_aruco3_detection
        return params


@dataclass
class CharucoDetectorConfig:
    """Configuration for Charuco detector parameters."""
    min_markers: int = 2
    try_refine_markers: bool = True
    
    def create_charuco_params(self) -> cv2.aruco.CharucoParameters:
        """Create and return OpenCV Charuco detector parameters."""
        charuco_params = cv2.aruco.CharucoParameters()
        charuco_params.minMarkers = self.min_markers
        charuco_params.tryRefineMarkers = self.try_refine_markers
        return charuco_params
