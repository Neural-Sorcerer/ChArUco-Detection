"""Charuco board detection and visualization module.

This module provides a class for detecting Charuco boards in images and videos,
as well as visualizing the results.
"""
# === Standard Libraries ===
import os
import logging
from typing import *
import xml.etree.ElementTree as ET

# === Third-Party Libraries ===
import cv2
import numpy as np

# === Local Modules ===
from utils import util
from configs.config import CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig


class CharucoDetector:
    """Class for detecting and visualizing Charuco boards.

    This class encapsulates the functionality for detecting Charuco boards
    in images and videos, as well as visualizing the results.
    """

    def __init__(
        self,
        board_config: CharucoBoardConfig,
        detector_config: DetectorConfig,
        charuco_detector_config: CharucoDetectorConfig,
        fisheye: bool = False
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
        self.obj_names = {"inner-corners": -1, "squares": 0, "all-corners": 1}
        self.objpoints = {"inner-corners": None, "squares": None, "all-corners": None, "temp": None}
        for key, value in self.obj_names.items():
            x_squares = board_config.x_squares + value
            y_squares = board_config.y_squares + value
            
            objp = np.zeros((x_squares * y_squares, 3), np.float64)
            objp[:, :2] = np.mgrid[0:x_squares, 0:y_squares].T.reshape(-1, 2)
            objp *= board_config.square_length  # Scale according to real square size
            self.objpoints[key] = objp.astype(np.float64).reshape(-1, 1, 3)

        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.fisheye = fisheye

    def set_temp_objpoints(self, charuco_ids: np.ndarray):
        """Set temporary object points for current board.

        Args:
            charuco_ids: IDs of detected Charuco corners
        """
        try:
            self.objpoints["temp"] = self.objpoints["inner-corners"][charuco_ids.flatten()]
        
        except Exception as e:
            logging.error(f"❌ Error during setting temporary object points: {str(e)}")
    
    def set_camera_params(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray) -> None:
        """Set camera intrinsic parameters.

        Args:
            camera_matrix (K): 3x3 camera intrinsic matrix
            dist_coeffs (D): Distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    def set_synthetic_camera_params(self, resolution: np.ndarray, fov_deg: float = None) -> None:
        """Set synthetic camera intrinsic parameters.

        Args:
            resolution: Image resolution (width, height)
            fov_deg: Field of view of the camera in degrees (default: None)
        """
        try:
            width = resolution[0]
            height = resolution[1]
            
            # Principal point at the center of the image
            cx = width / 2
            cy = height / 2
            
            # Calculate focal length from FOV
            if fov_deg is not None:
                fov_rad = np.deg2rad(fov_deg)
                fx = fy = width / (2 * np.tan(fov_rad / 2)) # ≈ 1662.0
            else:
                fx = fy = 1000.0    # Arbitrary focal length in pixels (you define it)

            # Set camera matrix and distortion coefficients
            self.camera_matrix = np.array([
                [fx,  0, cx],
                [0,  fy, cy],
                [0,   0,  1]
            ], dtype=np.float64)
            self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            
            logging.info(f"✅ Synthetic camera parameters set.")
        except Exception as e:
            logging.error(f"❌ Error setting synthetic camera parameters: {str(e)}")
    
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
            logging.error(f"❌ Failed to load camera parameters: {str(e)}")
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
            logging.error(f"❌ Error detecting Charuco board: {str(e)}")
            return None, None, None, None

    def draw_detected_markers_cv2(self,
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

    def draw_detected_corners_cv2(self,
                                  frame: np.ndarray,
                                  charuco_corners: np.ndarray,
                                  charuco_ids: np.ndarray,
                                  color: Tuple[int, int, int] = (255, 255, 0)) -> None:
        """Draw detected Charuco inner-corners on the frame.

        Args:
            frame: Input frame to draw on
            charuco_corners: Detected Charuco corners
            charuco_ids: IDs of detected Charuco corners
            color: Color to draw corners (BGR)
        """
        if (charuco_corners is not None) and (len(charuco_corners) > 0):
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, cornerColor=color)

    def draw_detected_corners(self,
                              frame: np.ndarray,
                              charuco_corners: np.ndarray,
                              charuco_ids: np.ndarray,
                              font_scale: float = 0.7,
                              font_color: Tuple[int, int, int] = (255, 255, 0),
                              thickness: int = 2,
                              point_color: Tuple[int, int, int] = (0, 255, 0),
                              point_radius: int = 5) -> None:
        """Draw inner-corner IDs on the frame.

        Args:
            frame: Input frame to draw on
            charuco_corners: Detected Charuco corners
            charuco_ids: IDs of detected Charuco corners
            font_scale: Font scale for text
            color: Color to draw text (BGR)
            thickness: Thickness of text
        """
        if (charuco_corners is not None) and (charuco_ids is not None):
            for corner, corner_id in zip(charuco_corners, charuco_ids.flatten()):
                org = (int(corner[0][0]), int(corner[0][1]))
                cv2.putText(frame,
                            str(corner_id),
                            org,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale,
                            font_color,
                            thickness,
                            cv2.LINE_AA)
                cv2.circle(frame, org, point_radius, point_color, -1)

    def estimate_pose_CharucoBoard(self,
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
        if (
            (self.camera_matrix is None) or
            (self.dist_coeffs is None) or
            (charuco_corners is None) or
            (charuco_ids is None) or
            (len(charuco_corners) < 4)
        ):
            return False, None, None

        try:
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charuco_corners,
                charuco_ids,
                self.board_config.board,
                self.camera_matrix,
                self.dist_coeffs,
                None,
                None
            )

            if success:
                rvec = np.array(rvec, dtype=np.float64)
                tvec = np.array(tvec, dtype=np.float64)
                return True, rvec, tvec
            else:
                return False, None, None
        except Exception as e:
            logging.error(f"❌ Error estimating pose: {str(e)}")
            return False, None, None

    def estimate_pose_solvePnP(self,
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
        if (
            (self.camera_matrix is None) or
            (self.dist_coeffs is None) or
            (charuco_corners is None) or
            (charuco_ids is None) or
            (len(charuco_corners) < 4)
        ):
            return False, None, None

        try:
            success, rvec, tvec = cv2.solvePnP(
                self.objpoints["temp"],
                charuco_corners,
                self.camera_matrix,
                self.dist_coeffs,
                None,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                rvec = np.array(rvec, dtype=np.float64)
                tvec = np.array(tvec, dtype=np.float64)
                return True, rvec, tvec
            else:
                return False, None, None
        except Exception as e:
            logging.error(f"❌ Error estimating pose: {str(e)}")
            return False, None, None
    
    def draw_board_pose_cv2(self,
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
                       objpoints_type: str = "inner-corners",
                       color: Tuple[int, int, int] = (0, 0, 255),
                       radius: int = 5
                       ) -> None:
        """Project 3D points to image plane.

        Args:
            frame: Input frame to draw on
            rvec: Rotation vector
            tvec: Translation vector
            objpoints_type: Type of object points to project
            color: Color to draw points (BGR)
            radius: Radius of points to draw
        """
        if self.fisheye:
            proj_points, _ = cv2.fisheye.projectPoints(
                self.objpoints[objpoints_type],
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs,
            )
        else:
            proj_points, _ = cv2.projectPoints(
                self.objpoints[objpoints_type],
                rvec,
                tvec,
                self.camera_matrix,
                self.dist_coeffs,
            )
        proj_points = proj_points.reshape(-1, 2)
        
        # Draw projected points
        for p in proj_points:
            cv2.circle(frame, (int(p[0]), int(p[1])), radius, color, -1)

    def generate_board_image(self,
                             pixels_per_square: int = 300,
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
            outSize=(width, height),
            marginSize=margin_size,
            borderBits=border_bits)
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
            dir_path = os.path.dirname(output_path)
            if dir_path:  # Only create directory if path contains a directory
                os.makedirs(dir_path, exist_ok=True)
                
            # Save image
            cv2.imwrite(output_path, board_img)
            logging.info(f"✅ Saved Charuco board image to {output_path}")
            return True
        except Exception as e:
            logging.error(f"❌ Error saving board image: {str(e)}")
            return False
