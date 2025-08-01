"""Charuco detection pipeline for camera calibration.

This module provides functionality for detecting Charuco boards in images and videos,
visualizing the results, and saving the data for camera calibration.
"""
# === Standard Libraries ===
import os
import logging
import argparse
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Tuple, Dict, List, Optional, Any, Union

# === Third-Party Libraries ===
import cv2
import numpy as np

# === Local Modules ===
from utils import util
from src.charuco_detector import CharucoDetector
from configs.config import Resolution, CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig
from src.calibration import CameraCalibrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s:%(lineno)02d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Removes milliseconds
)
logging = logging.getLogger(__name__)


def load_intrinsic_data_xml(xml_file_path):
    """Load intrinsic parameters from an XML file"""
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    
    # Extract image size
    width = int(root.find('image_size/width').text)
    height = int(root.find('image_size/height').text)
    
    # Extract camera matrix parameters
    fx = float(root.find('camera_matrix/fx').text)
    fy = float(root.find('camera_matrix/fy').text)
    ppx = float(root.find('camera_matrix/ppx').text)
    ppy = float(root.find('camera_matrix/ppy').text)
    
    # Convert to 3x3 camera matrix (row-major order)
    camera_matrix = [fx, 0., ppx, 0., fy, ppy, 0., 0., 1.]
    
    # Extract distortion coefficients
    distortion_coeffs = []
    num_coeffs = int(root.find('num_distortion_coeffs').text)
    for i in range(num_coeffs):
        coeff = float(root.find(f'distortion_coefficients/coeff_{i}').text)
        distortion_coeffs.append(coeff)
    
    # Extract camera model
    camera_model = root.find('camera_model').text
    
    return {
        'image_width': width,
        'image_height': height,
        'camera_matrix': camera_matrix,
        'distortion_coefficients': distortion_coeffs,
        'camera_model': camera_model,
        'num_distortion_coeffs': num_coeffs
    }


def process_frame(detector: CharucoDetector,
                  frame: np.ndarray,
                  draw_charuco_markers_cv2: bool = False,
                  draw_charuco_corners_cv2: bool = False,
                  draw_board_pose_cv2: bool = False,
                  use_estimate_pose_CharucoBoard: bool = False,
                  draw_charuco_corners: bool = False,
                  project_points: bool = False) -> None:
    """Process a single frame to detect and visualize Charuco board.

    Args:
        detector: CharucoDetector instance
        frame: Input frame to process
        draw_marker_corners: Whether to draw marker corners
        draw_charuco_corners: Whether to draw Charuco corners
        show_ids: Whether to show corner IDs
        project_points: Whether to project 3D points to image plane
        evaluate_3d: Whether to evaluate 3D consistency
        pose_matrix: Optional pose matrix for 3D consistency evaluation
    """
    if detector.fisheye:
        calibrator = CameraCalibrator(detector, fisheye=detector.fisheye)
        calibrator.camera_matrix = detector.camera_matrix
        calibrator.dist_coeffs = detector.dist_coeffs
        frame, new_camera_matrix = calibrator.undistort_image(frame, 0.0)
        
        detector.camera_matrix = new_camera_matrix
        detector.dist_coeffs = np.zeros((1, 4))
        detector.fisheye = False
    
    # Detect Charuco board
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detect_board(frame)
    
    # Visualization
    if draw_charuco_markers_cv2:
        detector.draw_detected_markers_cv2(frame, marker_corners, marker_ids)

    if draw_charuco_corners_cv2:
        detector.draw_detected_corners_cv2(frame, charuco_corners, charuco_ids)
    
    # Set temporary object points
    detector.set_temp_objpoints(charuco_ids)
    # =======================================================

    # Estimate pose if camera parameters are available
    if use_estimate_pose_CharucoBoard:
        objpoints_type = "all-corners"
        success, rvec, tvec = detector.estimate_pose_CharucoBoard(charuco_corners, charuco_ids)
    else:
        objpoints_type = "temp"
        success, rvec, tvec = detector.estimate_pose_solvePnP(charuco_corners, charuco_ids)

    if success:
        # Draw axes
        if draw_board_pose_cv2:
            detector.draw_board_pose_cv2(frame, rvec, tvec, axis_length=300)

        # Project 3D points to image plane
        if project_points:
            detector.project_points(frame, rvec, tvec, objpoints_type=objpoints_type)
        
    if draw_charuco_corners:
        detector.draw_detected_corners(frame, charuco_corners, charuco_ids)
    
    return frame


def run_pipeline(args: argparse.Namespace,
                 detectors: CharucoDetector,
                 freeze: int = 0,
                 winname: str = "Charuco Detection") -> None:
    """Run the Charuco detection pipeline on an image or video/camera feed.

    Args:
        args: Command-line arguments
        detector: CharucoDetector instance
        freeze: Frame delay in milliseconds (0 for no delay)
        winname: Window name for display
    """
    paths = {
        "cam_0": "temp_extr/images/Cam_001/000031.png",
    }
    
    for key, value in paths.items():
        frame = cv2.imread(value)
        detector = detectors[key]

        if frame is None:
            logging.error(f"❌ Cannot open image {args.index}")
            return

        # Create a window
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, width=Resolution.HD[0], height=Resolution.HD[1])

        # Process single image
        frame = process_frame(
            detector, frame,
            draw_charuco_markers_cv2=args.draw_charuco_markers_cv2,
            draw_charuco_corners_cv2=args.draw_charuco_corners_cv2,
            draw_board_pose_cv2=args.draw_board_pose_cv2,
            use_estimate_pose_CharucoBoard=args.use_estimate_pose_charuco_board,
            draw_charuco_corners=args.draw_charuco_corners,
            project_points=args.project_points,
        )

        # Show the frame
        cv2.imshow(winname, frame)
        cv2.waitKey(freeze)
        
    cv2.destroyAllWindows()


def main() -> None:
    """Main function to run the Charuco detection pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Charuco board detection for camera calibration")
    parser.add_argument('--draw-charuco-markers-cv2',
                        action="store_true", default=False, help='Draw charuco markers (corner+id)')
    parser.add_argument('--draw-charuco-corners-cv2',
                        action="store_true", default=False, help='Draw charuco inner-corners (corner+id)')
    parser.add_argument('--draw-board-pose-cv2',
                        action="store_true", default=True, help='Draw board pose')
    parser.add_argument('--use-estimate-pose-charuco-board',
                        action="store_true", default=False, help='Use estimatePoseCharucoBoard')
    parser.add_argument('--draw-charuco-corners',
                        action="store_true", default=True, help='Draw charuco inner-corners (id)')
    parser.add_argument('--project-points',
                        action="store_true", default=True, help='Project 3D points to image plane')
    
    # Charuco board arguments
    parser.add_argument('--board-id', type=int, default=0, help='Charuco board ID')
    parser.add_argument('--x-squares', type=int, default=7, help='Number of squares in X direction')
    parser.add_argument('--y-squares', type=int, default=7, help='Number of squares in Y direction')
    parser.add_argument('--square-length', type=float, default=100, help='Square length in meters')
    parser.add_argument('--marker-length', type=float, default=None, help='Marker length in meters (default: 75% of square length)')
    args = parser.parse_args()

    # Create configurations
    board_config = CharucoBoardConfig(
        board_id=args.board_id,
        x_squares=args.x_squares,
        y_squares=args.y_squares,
        square_length=args.square_length,
        marker_length=args.marker_length
    )
    detector_config = DetectorConfig()
    charuco_detector_config = CharucoDetectorConfig()

    detector_0 = CharucoDetector(board_config, detector_config, charuco_detector_config, fisheye=True)

    args.camera_params = "calibration_images/calibration_images_0/calibration.xml"
    params = load_intrinsic_data_xml(args.camera_params)
    K_cam_0 = np.array(params["camera_matrix"]).reshape(3, 3)
    D_cam_0 = np.array(params["distortion_coefficients"]).reshape(1,-1)
    
    detector_0.set_camera_params(K_cam_0, D_cam_0)
    
    detectors = {}
    detectors["cam_0"] = detector_0
    
    # Run the pipeline
    run_pipeline(args, detectors)


if __name__ == '__main__':
    main()
