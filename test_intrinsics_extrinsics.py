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
from src.charuco_detector import CharucoDetector
from configs.config import Resolution, CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
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
        winname: Window name for display
    """
    paths = {
        "cam_4": "temp/temp_extr/images/Cam_005/000031.png",
        "cam_0": "temp/temp_extr/images/Cam_001/000031.png",
        # "cam_1": "temp/temp_extr/images/Cam_002/000006.png",
    }
    
    for key, value in paths.items():
        frame = cv2.imread(value)
        detector = detectors[key]

        if key == "cam_0":
            detector.rvec_obj_in_dst_cam = detectors["cam_4"].rvec_obj_in_dst_cam
            detector.tvec_obj_in_dst_cam = detectors["cam_4"].tvec_obj_in_dst_cam

        if frame is None:
            logging.error(f"❌ Cannot open image {args.index}")
            return

        # Create a window
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, width=Resolution.HD[0], height=Resolution.HD[1])

        # Process single image
        frame = detector.run_charuco_pipeline_extrinsics(frame)

        # Show the frame
        cv2.imshow(winname, frame)
        cv2.waitKey(freeze)
        
    cv2.destroyAllWindows()


def main() -> None:
    """Main function to run the Charuco detection pipeline."""
    # Default path for sample image
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Charuco board detection for camera calibration")
    parser.add_argument('--camera-params', default=None, type=str, help='Path to camera calibration file')
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

    fisheye_intrinsics = "temp/calibration_images/calibration_images_0/calibration.xml"
    pinhole_intrinsics = "temp/calibration_images/calibration_images_8/calibration.xml"

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
    args.camera_params = fisheye_intrinsics
    detector_0 = CharucoDetector(args, board_config, detector_config, charuco_detector_config)
    args.camera_params = pinhole_intrinsics
    detector_1 = CharucoDetector(args, board_config, detector_config, charuco_detector_config)

    detectors = {}
    detectors["cam_0"] = detector_0
    detectors["cam_4"] = detector_1
    
    # Run the pipeline
    run_pipeline(args, detectors)


if __name__ == '__main__':
    main()
