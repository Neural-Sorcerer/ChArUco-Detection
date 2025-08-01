"""Camera calibration module using Charuco boards.

This module provides functionality for calibrating cameras using Charuco boards.
It includes functions for collecting calibration data, performing calibration,
and saving/loading calibration parameters.
"""
# === Standard Libraries ===
import os
import glob
import logging
from typing import *
import xml.etree.ElementTree as ET

# === Third-Party Libraries ===
import cv2
import numpy as np

# === Local Modules ===
from src.charuco_detector import CharucoDetector


class CameraCalibrator:
    """Class for calibrating cameras using Charuco boards."""

    def __init__(self, detector: CharucoDetector, fisheye: bool = False):
        """Initialize the CameraCalibrator.

        Args:
            detector: CharucoDetector instance
            fisheye: Whether to use fisheye camera model (default: False for pinhole)
        """
        self.detector = detector
        self.fisheye = fisheye
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
            logging.warning("⚠️ Image size does not match previous images. Skipping.")
            return False

        # Detect Charuco board
        charuco_corners, charuco_ids, _, _ = self.detector.detect_board(image)

        # Check if corners were detected
        if charuco_corners is None or len(charuco_corners) < 4:
            logging.warning("⚠️ Not enough corners detected in image.")
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
        logging.info(f"⭐ ───────────── Adding Images for Calibration ───────────── ⭐")
        
        # Get all image files in directory
        image_files = glob.glob(os.path.join(directory, pattern))

        if not image_files:
            logging.warning(f"⚠️ No images found in {directory} matching pattern {pattern}")
            return 0

        # Add each image
        count = 0
        for image_file in image_files:
            logging.info(f"Processing {image_file}")
            image = cv2.imread(image_file)

            if image is None:
                logging.warning(f"⚠️ Could not read image {image_file}")
                continue

            if self.add_calibration_image(image):
                count += 1

        logging.info(f"✅ Added {count} images for calibration")
        return count

    def calibrate(self) -> bool:
        """Perform camera calibration.

        Returns:
            True if calibration was successful, False otherwise
        """
        if not self.all_corners or not self.all_ids:
            logging.error(f"❌ No calibration data available")
            return False

        if self.image_size is None:
            logging.error(f"❌ Image size not set")
            return False

        # Prepare object points (3D points in real-world space)
        board = self.detector.board_config.board

        # Prepare object points for fisheye calibration
        objpoints = []
        imgpoints = []
        for corners, ids in zip(self.all_corners, self.all_ids):
            if (corners is not None) and (ids is not None) and (len(corners) > 3):
                # Get object points for detected corners
                obj_pts, img_pts = board.matchImagePoints(corners, ids)
                if (obj_pts is not None) and (img_pts is not None):
                    objpoints.append(obj_pts.reshape(-1, 1, 3))
                    imgpoints.append(img_pts.reshape(-1, 1, 2))

        if len(objpoints) < 3:
            logging.error(f"❌ Not enough valid images for calibration")
            return False
        
        try:
            if self.fisheye:
                # Fisheye calibration
                logging.info(f"⭐ ───────────── Performing Fisheye Camera Calibration ───────────── ⭐")
                
                # Fisheye calibration flags
                flags = (
                    cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
                    cv2.fisheye.CALIB_CHECK_COND +
                    cv2.fisheye.CALIB_FIX_SKEW
                )
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
                
                # Initialize camera matrix for fisheye
                K = np.eye(3)
                D = np.zeros((4, 1))
                
                # Perform fisheye calibration
                (
                    self.reprojection_error,
                    self.camera_matrix,
                    self.dist_coeffs,
                    self.rvecs,
                    self.tvecs
                ) = cv2.fisheye.calibrate(
                    objectPoints=objpoints,
                    imagePoints=imgpoints,
                    image_size=self.image_size,
                    K=K,
                    D=D,
                    rvecs=self.rvecs,
                    tvecs=self.tvecs,
                    flags=flags,
                    criteria=criteria
                )

                # Log it
                cond_K = np.linalg.cond(self.camera_matrix)
                logging.info(f"Condition number of camera matrix: {cond_K:.2e}")

                # Check if it's in the range 1e6 to 1e8
                if 1e6 <= cond_K <= 1e8:
                    logging.warning("⚠️ Condition number is high - calibration may be unstable.")
                elif cond_K > 1e8:
                    logging.error(f"❌ Condition number is extremely high - calibration likely invalid.")
                else:
                    logging.info("✅ Condition number is within acceptable range.")
    
                logging.info(f"✅ Fisheye calibration successful.")

            else:
                # Pinhole calibration
                logging.info(f"⭐ ───────────── Performing Pinhole Camera Calibration ───────────── ⭐")

                # Pinhole calibration flags
                flags = 0
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        
                (
                    self.reprojection_error,
                    self.camera_matrix,
                    self.dist_coeffs,
                    self.rvecs,
                    self.tvecs
                ) = cv2.calibrateCamera(
                    objectPoints=objpoints,
                    imagePoints=imgpoints,
                    imageSize=self.image_size,
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=self.dist_coeffs,
                    rvecs=self.rvecs,
                    tvecs=self.tvecs,
                    flags=flags,
                    criteria=criteria
                )

                logging.info(f"✅ Pinhole calibration successful.")
            return True

        except Exception as e:
            logging.error(f"❌ Calibration failed: {str(e)}")
            return False

    def save_calibration_parameters(self, file_path: str) -> bool:
        """Save calibration parameters to a file.

        Args:
            file_path: Path to save calibration parameters

        Returns:
            True if parameters were saved successfully, False otherwise
        """
        if (self.camera_matrix is None) or (self.dist_coeffs is None):
            logging.error(f"❌ No calibration data available")
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
            dist_coeffs_flat = self.dist_coeffs.flatten()
            for i, coeff in enumerate(dist_coeffs_flat):
                ET.SubElement(dist_coeffs_elem, f"coeff_{i}").text = str(coeff)

            # Add camera model type
            ET.SubElement(root, "camera_model").text = "fisheye" if self.fisheye else "pinhole"
            ET.SubElement(root, "num_distortion_coeffs").text = str(len(dist_coeffs_flat))

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

            logging.info(f"✅ Calibration parameters saved to {file_path}")
            return True

        except Exception as e:
            logging.error(f"❌ Error saving calibration parameters: {str(e)}")
            return False

    def load_calibration_parameters(file_path: str
            ) -> Optional[Tuple[np.ndarray, np.ndarray, Tuple[int, int], bool, Optional[float]]]:
        """Load calibration parameters from an XML file.

        Args:
            file_path: Path to the saved XML file

        Returns:
            Tuple of (camera_matrix, dist_coeffs, image_size, fisheye, reprojection_error) or None on failure
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Camera parameter file not found: {file_path}")
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Image size
            width = int(root.find("image_size/width").text)
            height = int(root.find("image_size/height").text)
            image_size = (width, height)

            # Camera intrinsics
            fx = float(root.find("camera_matrix/fx").text)
            fy = float(root.find("camera_matrix/fy").text)
            ppx = float(root.find("camera_matrix/ppx").text)
            ppy = float(root.find("camera_matrix/ppy").text)

            camera_matrix = np.array([
                [fx,  0, ppx],
                [ 0, fy, ppy],
                [ 0,  0,   1]
            ], dtype=np.float64)

            # Distortion coefficients
            num_coeffs = int(root.find("num_distortion_coeffs").text)
            dist_coeffs = np.zeros((num_coeffs, 1), dtype=np.float64)
            for i in range(num_coeffs):
                coeff_elem = root.find(f"distortion_coefficients/coeff_{i}")
                if coeff_elem is not None:
                    dist_coeffs[i] = float(coeff_elem.text)

            # Camera model
            cam_model_text = root.find("camera_model").text
            fisheye = True if cam_model_text.lower() == "fisheye" else False

            # Reprojection error
            reprojection_error_elem = root.find("reprojection_error")
            reprojection_error = float(reprojection_error_elem.text) if reprojection_error_elem is not None else None

            logging.info(f"✅ Calibration parameters loaded from {file_path}")
            return camera_matrix, dist_coeffs, image_size, fisheye, reprojection_error

        except Exception as e:
            logging.error(f"❌ Failed to load calibration from {file_path}: {str(e)}")
            return None
    
    def get_calibration_metrics(self) -> Dict[str, Any]:
        """Get calibration metrics.

        Returns:
            Dictionary containing calibration metrics
        """
        if (self.camera_matrix is None) or (self.dist_coeffs is None):
            return {"error": "No calibration data available"}

        metrics = {
            "reprojection_error": self.reprojection_error,
            "image_size": self.image_size,
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.flatten().tolist(),
            "num_images": len(self.all_corners)
        }
        return metrics
    
    def show_calibration_metrics(self, metrics: Dict[str, Any]) -> bool:
        """Show calibration metrics.

        args:
            metrics: Dictionary containing calibration metrics

        Returns:
            Treue if metrics were shown successfully, False otherwise
        """
        try:
            logging.info(f"⭐ ───────────── Calibration Quality ───────────── ⭐")
            for key, value in metrics.items():
                if isinstance(value, list):
                    result = f"{key}:\n{np.array(value)}\n"
                else:
                    result = f"{key}: {value}"
                if "error" in key:
                    result = f"{result} ⬅️"
                logging.info(result)
            return True
        
        except Exception as e:
            logging.error(f"❌ Error showing calibration metrics: {str(e)}")
            return False

    def undistort_image(self, image: np.ndarray, balance: float = 1.0, simple: bool = False) -> np.ndarray:
        """Undistort an image using calibration parameters.

        Args:
            image: Input image
            balance: Balance value for undistortion (0.0 = crop, 1.0 = stretch)
            simple: Whether to use simple undistortion (no remapping)

        Returns:
            Undistorted image
        """
        if (self.camera_matrix is None) or (self.dist_coeffs is None):
            logging.error(f"❌ No calibration data available")
            return image, None

        R = np.eye(3)
        m1type = cv2.CV_16SC2
        h, w = image.shape[:2]
        img_size = (w, h)

        if self.fisheye:
            # Fisheye undistortion
            # Get optimal new camera matrix for fisheye
            new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                K=self.camera_matrix,
                D=self.dist_coeffs,
                image_size=img_size,
                R=R,
                balance=balance
            )

            # Create undistortion maps
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                K=self.camera_matrix,
                D=self.dist_coeffs,
                R=R,
                P=new_camera_matrix,
                size=img_size,
                m1type=m1type
            )

            # Apply undistortion
            undistorted = cv2.remap(
                src=image,
                map1=map1,
                map2=map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT
            )

        else:
            # Pinhole undistortion
            # Get optimal new camera matrix
            new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                imageSize=img_size,
                alpha=balance,
                newImgSize=img_size
            )

            if simple:
                # Undistort
                undistorted = cv2.undistort(
                    src=image,
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=self.dist_coeffs,
                    dst=None,
                    newCameraMatrix=new_camera_matrix
                )

                # Crop the image
                x, y, w, h = roi
                if w > 0 and h > 0:
                    undistorted = undistorted[y:y+h, x:x+w]
            else:
                
                # Create undistortion maps
                map1, map2 = cv2.initUndistortRectifyMap(
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=self.dist_coeffs,
                    R=R,
                    newCameraMatrix=new_camera_matrix,
                    size=img_size,
                    m1type=m1type
                )
            
                # Apply undistortion
                undistorted = cv2.remap(
                    src=image,
                    map1=map1,
                    map2=map2,
                    interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT
                )
        return undistorted, new_camera_matrix
