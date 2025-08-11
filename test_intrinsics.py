"""Charuco detection pipeline for camera calibration.

This module provides functionality for detecting Charuco boards in images and videos,
visualizing the results, and saving the data for camera calibration.
"""
# === Standard Libraries ===
import logging
import argparse
from typing import *

# === Third-Party Libraries ===
import cv2

# === Local Modules ===
from src.calibration import CameraCalibrator
from src.charuco_detector import CharucoDetector
from configs.config import Resolution, CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig


# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s:%(lineno)02d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging = logging.getLogger(__name__)


def run_pipeline(args: argparse.Namespace,
                 detectors: CharucoDetector,
                 freeze: int = 0,
                 winname: str = "Charuco Detection") -> None:
    """Run the Charuco detection pipeline on an image or video/camera feed.

    Args:
        args: Command-line arguments
        detector: CharucoDetector instance
        freeze: Frame delay in milliseconds (0 for no delay)
        resolution: Resolution for video capture
        winname: Window name for display
    """
    # Create a window
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, width=Resolution.HD[0], height=Resolution.HD[1])
    
    paths = {
        "cam_0": "temp/temp_extr/images/Cam_001/000031.png",
    }
    
    for key, value in paths.items():
        frame = cv2.imread(value)
        detector = detectors[key]

        if frame is None:
            logging.error(f"❌ Cannot open image {args.index}")
            return

        # Save original frame
        original = frame.copy()
        
        pt = (2390, 1043)
        # pt = (293, 198)
        original = cv2.circle(original, pt, 10, (0, 0, 255), -1)
        
        calibrator = CameraCalibrator()
        calibrator.camera_matrix = detector.camera_matrix
        calibrator.dist_coeffs = detector.dist_coeffs
        
        # Process single image
        frame = detector.run_charuco_pipeline(frame)
        
        # detector.camera_matrix = new_camera_matrix
        rect_point = calibrator.undistort_point(pt, detector.camera_matrix)
        frame = cv2.circle(frame, rect_point, 10, (0, 255, 0), -1)
        
        frame = cv2.hconcat([original, frame])

        # Show the frame
        cv2.imshow(winname, frame)
        cv2.waitKey(freeze)
    
    cv2.destroyAllWindows()


def main() -> None:
    """Main function to run the Charuco detection pipeline."""
    # Default path for sample image
    intrinsics = "temp/calibration_images/calibration_images_0/calibration.xml"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Charuco board detection for camera calibration")
    parser.add_argument('--index', default=None, type=str, help='Camera index, video file path, or image path')
    parser.add_argument('--camera-params', default=intrinsics, type=str, help='Path to camera calibration file')
    parser.add_argument('--resolution', type=str, default='FHD', choices=['SS', 'SD', 'HD', 'FHD', 'UHD', 'OMS'], help='Camera resolution')
    
    # Visualization arguments
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

    # Output arguments
    parser.add_argument('--output-dir', default="outputs/charuco_detection", type=str, help='Output path')
    parser.add_argument('--save', action="store_true", default=False, help='Save flag')
    parser.add_argument('--save-all', action="store_true", default=False, help='Save all frames flag')
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

    # Create detector
    detector = CharucoDetector(args, board_config, detector_config, charuco_detector_config)

    detectors = {}
    detectors["cam_0"] = detector
    
    # Run the pipeline
    run_pipeline(args, detectors)


if __name__ == '__main__':
    main()
