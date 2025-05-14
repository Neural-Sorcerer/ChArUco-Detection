"""Camera calibration module using Charuco boards.

This module provides functionality for calibrating cameras using Charuco boards.
It includes functions for collecting calibration data, performing calibration,
and saving/loading calibration parameters.
"""
import os
import glob
import logging
import xml.etree.ElementTree as ET
from typing import Tuple, List, Dict, Optional, Any, Union

import cv2
import numpy as np

from configs.config import CharucoBoardConfig
from src.charuco_detector import CharucoDetector


class CameraCalibrator:
    """Class for calibrating cameras using Charuco boards."""

    def __init__(self, detector: CharucoDetector):
        """Initialize the CameraCalibrator.

        Args:
            detector: CharucoDetector instance
        """
        self.detector = detector
        self.all_corners = []   # All detected corners
        self.all_ids = []       # All detected IDs
        self.image_size = None  # Image size (width, height)

        # Calibration results
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.std_deviations_intrinsics = None
        self.std_deviations_extrinsics = None
        self.per_view_errors = None
        self.reprojection_error = None

    def add_calibration_image(self, image: np.ndarray) -> bool:
        """Add an image for calibration.

        Args:
            image: Input image

        Returns:
            True if corners were detected, False otherwise
        """
        # Store image size
        if self.image_size is None:
            self.image_size = (image.shape[1], image.shape[0])
        elif self.image_size != (image.shape[1], image.shape[0]):
            logging.warning("Image size does not match previous images. Skipping.")
            return False

        # Detect Charuco board
        charuco_corners, charuco_ids, _, _ = self.detector.detect_board(image)

        # Check if corners were detected
        if charuco_corners is None or len(charuco_corners) < 4:
            logging.warning("Not enough corners detected in image.")
            return False

        # Add corners and IDs to lists
        self.all_corners.append(charuco_corners)
        self.all_ids.append(charuco_ids)

        return True

    def add_calibration_images_from_directory(self, directory: str, pattern: str = "*.png") -> int:
        """Add all images in a directory for calibration.

        Args:
            directory: Directory containing calibration images
            pattern: File pattern to match

        Returns:
            Number of images successfully added
        """
        # Get all image files in directory
        image_files = glob.glob(os.path.join(directory, pattern))

        if not image_files:
            logging.warning(f"No images found in {directory} matching pattern {pattern}")
            return 0

        # Add each image
        count = 0
        for image_file in image_files:
            logging.info(f"Processing {image_file}")
            image = cv2.imread(image_file)

            if image is None:
                logging.warning(f"Could not read image {image_file}")
                continue

            if self.add_calibration_image(image):
                count += 1

        logging.info(f"Added {count} images for calibration")
        return count

    def calibrate(self) -> bool:
        """Perform camera calibration.

        Returns:
            True if calibration was successful, False otherwise
        """
        if not self.all_corners or not self.all_ids:
            logging.error("No calibration data available")
            return False

        if self.image_size is None:
            logging.error("Image size not set")
            return False

        # Prepare object points (3D points in real-world space)
        board = self.detector.board_config.board

        # Perform calibration
        flags = (
            cv2.CALIB_RATIONAL_MODEL +      # Use rational model for distortion
            cv2.CALIB_THIN_PRISM_MODEL +    # Add thin prism distortion
            cv2.CALIB_TILTED_MODEL          # Add tilted sensor model
        )
        criteria = (cv2.TERM_CRITERIA_EPS & cv2.TERM_CRITERIA_COUNT, 10000, 1e-9)

        try:
            (
                self.reprojection_error,
                self.camera_matrix,
                self.dist_coeffs,
                self.rvecs,
                self.tvecs,
                self.std_deviations_intrinsics,
                self.std_deviations_extrinsics,
                self.per_view_errors
            ) = cv2.aruco.calibrateCameraCharucoExtended(
                charucoCorners=self.all_corners,
                charucoIds=self.all_ids,
                board=board,
                imageSize=self.image_size,
                cameraMatrix=None,
                distCoeffs=None,
                flags=flags,
                criteria=criteria
            )

            logging.info(f"Calibration successful. Reprojection error: {self.reprojection_error}")
            return True

        except Exception as e:
            logging.error(f"Calibration failed: {str(e)}")
            return False

    def save_calibration(self, file_path: str) -> bool:
        """Save calibration parameters to a file.

        Args:
            file_path: Path to save calibration parameters

        Returns:
            True if parameters were saved successfully, False otherwise
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            logging.error("No calibration data available")
            return False

        try:
            # Create XML structure
            root = ET.Element("camera_calibration")

            # Add image size
            image_size_elem = ET.SubElement(root, "image_size")
            ET.SubElement(image_size_elem, "width").text = str(self.image_size[0])
            ET.SubElement(image_size_elem, "height").text = str(self.image_size[1])

            # Add camera matrix
            camera_matrix_elem = ET.SubElement(root, "camera_matrix")
            ET.SubElement(camera_matrix_elem, "fx").text = str(self.camera_matrix[0, 0])
            ET.SubElement(camera_matrix_elem, "fy").text = str(self.camera_matrix[1, 1])
            ET.SubElement(camera_matrix_elem, "ppx").text = str(self.camera_matrix[0, 2])
            ET.SubElement(camera_matrix_elem, "ppy").text = str(self.camera_matrix[1, 2])

            # Add distortion coefficients
            dist_coeffs_elem = ET.SubElement(root, "distortion_coefficients")
            for i, coeff in enumerate(self.dist_coeffs.flatten()):
                ET.SubElement(dist_coeffs_elem, f"coeff_{i}").text = str(coeff)

            # Add reprojection error
            ET.SubElement(root, "reprojection_error").text = str(self.reprojection_error)

            # Create directory if it doesn't exist
            dir_path = os.path.dirname(file_path)
            if dir_path:  # Only create directory if path contains a directory
                os.makedirs(dir_path, exist_ok=True)

            # Write to file with proper indentation
            tree = ET.ElementTree(root)

            # Helper function to add indentation
            def indent(elem, level=0):
                i = "\n" + level*"  "
                if len(elem):
                    if not elem.text or not elem.text.strip():
                        elem.text = i + "  "
                    if not elem.tail or not elem.tail.strip():
                        elem.tail = i
                    for elem in elem:
                        indent(elem, level+1)
                    if not elem.tail or not elem.tail.strip():
                        elem.tail = i
                else:
                    if level and (not elem.tail or not elem.tail.strip()):
                        elem.tail = i

            # Apply indentation
            indent(root)

            # Write to file
            tree.write(file_path, encoding="utf-8", xml_declaration=True)

            logging.info(f"Calibration parameters saved to {file_path}")
            return True

        except Exception as e:
            logging.error(f"Error saving calibration parameters: {str(e)}")
            return False

    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get calibration metrics.

        Returns:
            Dictionary containing calibration metrics
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            return {"error": "No calibration data available"}

        metrics = {
            "reprojection_error": self.reprojection_error,
            "image_size": self.image_size,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "num_images": len(self.all_corners)
        }

        return metrics

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """Undistort an image using calibration parameters.

        Args:
            image: Input image

        Returns:
            Undistorted image
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            logging.error("No calibration data available")
            return image

        # Get optimal new camera matrix
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )

        # Undistort
        undistorted = cv2.undistort(
            image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix
        )

        # Crop the image
        x, y, w, h = roi
        if w > 0 and h > 0:
            undistorted = undistorted[y:y+h, x:x+w]

        return undistorted
