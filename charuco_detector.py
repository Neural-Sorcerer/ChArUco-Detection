"""Charuco board detection and visualization module.

This module provides a class for detecting Charuco boards in images and videos,
as well as visualizing the results.
"""
import os
import logging
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, List, Optional, Any, Union

import cv2
import numpy as np

from utils import util
from config import CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig


class CharucoDetector:
    """Class for detecting and visualizing Charuco boards.

    This class encapsulates the functionality for detecting Charuco boards
    in images and videos, as well as visualizing the results.
    """

    def __init__(
        self,
        board_config: CharucoBoardConfig,
        detector_config: DetectorConfig,
        charuco_detector_config: CharucoDetectorConfig
    ):
        """Initialize the CharucoDetector.

        Args:
            board_config: Configuration for the Charuco board
            detector_config: Configuration for the Aruco detector
            charuco_detector_config: Configuration for the Charuco detector
        """
        self.board_config = board_config
        self.detector_config = detector_config
        self.charuco_detector_config = charuco_detector_config

        # Create detector parameters
        self.detector_params = detector_config.create_detector_params()
        self.charuco_params = charuco_detector_config.create_charuco_params()

        # Create the Charuco detector
        self.detector = cv2.aruco.CharucoDetector(
            board=board_config.board,
            detectorParams=self.detector_params,
            charucoParams=self.charuco_params
        )

        # Prepare object points (3D points in real-world space)
        self.objp = np.zeros((board_config.x_squares * board_config.y_squares, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:board_config.x_squares, 0:board_config.y_squares].T.reshape(-1, 2)
        self.objp *= board_config.square_length  # Scale according to real square size
        self.objpoint = self.objp.astype(np.float32).reshape(-1, 1, 3)

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None

    def set_camera_params(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        """Set camera intrinsic parameters.

        Args:
            camera_matrix (K): 3x3 camera intrinsic matrix
            dist_coeffs (D): Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def load_camera_params(self, file_path: str) -> bool:
        """Load camera intrinsic parameters from a file.

        Args:
            file_path: Path to the camera parameter file

        Returns:
            True if parameters were loaded successfully, False otherwise
        """
        try:
            self.camera_matrix, self.dist_coeffs = util.load_camera_params(file_path)
            return True
        except (FileNotFoundError, ET.ParseError, ValueError) as e:
            logging.error(f"Failed to load camera parameters: {str(e)}")
            return False

    def detect_board(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray],
                                                       Optional[np.ndarray],
                                                       Optional[List[np.ndarray]],
                                                       Optional[np.ndarray]]:
        """Detect Charuco board in a frame.

        Args:
            frame: Input frame (BGR or grayscale)

        Returns:
            Tuple containing:
                - charuco_corners: Detected Charuco corners
                - charuco_ids: IDs of detected Charuco corners
                - marker_corners: Detected marker corners
                - marker_ids: IDs of detected markers
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Detect markers and interpolate Charuco corners
        try:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
            return charuco_corners, charuco_ids, marker_corners, marker_ids
        except Exception as e:
            logging.error(f"Error detecting Charuco board: {str(e)}")
            return None, None, None, None

    def draw_detected_markers(self,
                              frame: np.ndarray,
                              marker_corners: List[np.ndarray],
                              marker_ids: np.ndarray,
                              color: Tuple[int, int, int] = (0, 255, 255)) -> None:
        """Draw detected markers on the frame.

        Args:
            frame: Input frame to draw on
            marker_corners: Detected marker corners
            marker_ids: IDs of detected markers
            color: Color to draw markers (BGR)
        """
        if marker_corners and len(marker_corners) > 0:
            cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, borderColor=color)

    def draw_detected_corners(self,
                              frame: np.ndarray,
                              charuco_corners: np.ndarray,
                              charuco_ids: np.ndarray,
                              color: Tuple[int, int, int] = (255, 255, 0)) -> None:
        """Draw detected Charuco corners on the frame.

        Args:
            frame: Input frame to draw on
            charuco_corners: Detected Charuco corners
            charuco_ids: IDs of detected Charuco corners
            color: Color to draw corners (BGR)
        """
        if charuco_corners is not None and len(charuco_corners) > 0:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, cornerColor=color)

    def draw_corner_ids(self,
                        frame: np.ndarray,
                        charuco_corners: np.ndarray,
                        charuco_ids: np.ndarray,
                        font_scale: float = 0.7,
                        color: Tuple[int, int, int] = (255, 255, 0),
                        thickness: int = 2) -> None:
        """Draw corner IDs on the frame.

        Args:
            frame: Input frame to draw on
            charuco_corners: Detected Charuco corners
            charuco_ids: IDs of detected Charuco corners
            font_scale: Font scale for text
            color: Color to draw text (BGR)
            thickness: Thickness of text
        """
        if charuco_corners is not None and charuco_ids is not None:
            for corner, corner_id in zip(charuco_corners, charuco_ids.flatten()):
                pos = (int(corner[0][0]), int(corner[0][1]))
                cv2.putText(frame, str(corner_id), pos, cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, color, thickness, cv2.LINE_AA)

    def estimate_pose(self,
                      charuco_corners: np.ndarray,
                      charuco_ids: np.ndarray
                      ) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate pose of the Charuco board.

        Args:
            charuco_corners: Detected Charuco corners
            charuco_ids: IDs of detected Charuco corners

        Returns:
            Tuple containing:
                - success: True if pose estimation was successful
                - rvec: Rotation vector (None if estimation failed)
                - tvec: Translation vector (None if estimation failed)
        """
        if ((self.camera_matrix is None) or (self.dist_coeffs is None) or
            (charuco_corners is None) or (charuco_ids is None) or len(charuco_corners) < 4):
            return False, None, None

        try:
            ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners, charuco_ids, self.board_config.board,
                self.camera_matrix, self.dist_coeffs, None, None
            )

            if ret:
                rvec = np.array(rvec, dtype=np.float32)
                tvec = np.array(tvec, dtype=np.float32)
                return True, rvec, tvec
            else:
                return False, None, None
        except Exception as e:
            logging.error(f"Error estimating pose: {str(e)}")
            return False, None, None

    def draw_axes(self,
                  frame: np.ndarray,
                  rvec: np.ndarray,
                  tvec: np.ndarray,
                  axis_length: float = 0.1) -> None:
        """Draw 3D axes on the frame.

        Args:
            frame: Input frame to draw on
            rvec: Rotation vector
            tvec: Translation vector
            axis_length: Length of axes to draw
        """
        cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvec, tvec, axis_length)

    def project_points(self,
                       frame: np.ndarray,
                       rvec: np.ndarray,
                       tvec: np.ndarray,
                       color: Tuple[int, int, int] = (0, 0, 255),
                       radius: int = 5) -> None:
        """Project 3D points to image plane and draw them.

        Args:
            frame: Input frame to draw on
            rvec: Rotation vector
            tvec: Translation vector
            color: Color to draw points (BGR)
            radius: Radius of points to draw
        """
        # Project 3D points to image plane
        imgpoints_proj = util.project_points_to_image(
            self.objpoint, rvec, tvec, self.camera_matrix, self.dist_coeffs)

        # Draw projected points
        for imgpoint in imgpoints_proj:
            cv2.circle(frame, (int(imgpoint[0]), int(imgpoint[1])), radius, color, -1)

    def generate_board_image(self,
                             pixels_per_square: int = 100,
                             margin_percent: float = 0.05,
                             border_bits: int = 1) -> np.ndarray:
        """Generate an image of the Charuco board.

        Args:
            pixels_per_square: Number of pixels per square
            margin_percent: Margin around the board as a percentage (0.05 = 5%) of the minimum grid dimension
            border_bits: Width of marker borders

        Returns:
            Image of the Charuco board
        """
        # Calculate grid size
        width_grid = self.board_config.x_squares * pixels_per_square
        height_grid = self.board_config.y_squares * pixels_per_square

        # Calculate margin size in pixels based on percentage
        margin_size = int(margin_percent * min(width_grid, height_grid))

        # Calculate image size
        width = width_grid + (2 * margin_size)
        height = height_grid + (2 * margin_size)

        # Generate board image
        board_img = self.board_config.board.generateImage(
            outSize=(width, height), marginSize=margin_size, borderBits=border_bits)
        return board_img

    def save_board_image(self,
                         output_path: str,
                         pixels_per_square: int = 300,
                         margin_percent: float = 0.05,
                         border_bits: int = 1) -> bool:
        """Generate and save an image of the Charuco board.

        Args:
            output_path: Path to save the board image
            pixels_per_square: Number of pixels per square
            margin_percent: Margin around the board as a percentage (0.05 = 5%) of the minimum grid dimension
            border_bits: Width of marker borders

        Returns:
            True if the image was saved successfully, False otherwise
        """
        try:
            # Generate board image
            board_img = self.generate_board_image(pixels_per_square, margin_percent, border_bits)

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Save image
            cv2.imwrite(output_path, board_img)
            logging.info(f"Saved Charuco board image to {output_path}")
            return True
        except Exception as e:
            logging.error(f"Error saving board image: {str(e)}")
            return False
