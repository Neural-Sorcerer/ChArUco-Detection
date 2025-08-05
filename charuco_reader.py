"""Charuco detection pipeline for camera calibration.

This module provides functionality for detecting Charuco boards in images and videos,
visualizing the results, and saving the data for camera calibration.
"""
# === Standard Libraries ===
import os
import logging
import argparse
from typing import *

# === Third-Party Libraries ===
import cv2

# === Local Modules ===
from utils import util
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
                 detector: CharucoDetector,
                 freeze: int = 1,
                 resolution: Tuple[int, int] = Resolution.FHD,
                 winname: str = "Charuco Detection") -> None:
    """Run the Charuco detection pipeline on an image or video/camera feed.

    Args:
        args: Command-line arguments
        detector: CharucoDetector instance
        freeze: Frame delay in milliseconds (0 for no delay)
        resolution: Resolution for video capture
        winname: Window name for display
    """
    # Check if input is an image file
    if os.path.isfile(args.index) and args.index.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read image from file
        frame = cv2.imread(args.index)

        if frame is None:
            logging.error(f"❌ Cannot open image {args.index}")
            return

        # Create a window
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, width=Resolution.HD[0], height=Resolution.HD[1])

        # Process single image
        frame = detector.run_charuco_pipeline(frame)

        # Show the frame
        cv2.imshow(winname, frame)
        cv2.waitKey(0)
    else:
        # Process video/camera feed
        if args.index.isdigit():
            # If it's an integer, it's a camera index
            cap = cv2.VideoCapture(int(args.index), cv2.CAP_V4L2)   # Ensure V4L2 backend
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # If it's a string, it's a video file path
            cap = cv2.VideoCapture(args.index)

        if not cap.isOpened():
            logging.error(f"❌ Cannot open camera {args.index}")
            return

        # Create a window
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, width=Resolution.HD[0], height=Resolution.HD[1])

        # For recording
        if args.save_all:
            args.save = True
            freeze = 1000

        if args.save:
            logging.info("⚠️ Press 's' to save an image or 'q' to quit the process!")

        frame_id = 0
        while cap.isOpened():
            success, original = cap.read()

            if not success:
                break

            # Save original frame
            frame = original.copy()

            # Process frame
            frame = detector.run_charuco_pipeline(frame)

            # Show the frame
            cv2.imshow(winname, frame)

            # Get a key
            key = cv2.waitKey(freeze) & 0xFF

            if key == ord('q'):     # Press 'q' to quit
                break
            elif key == ord('f'):   # Toggle freeze mode
                freeze = 0 if freeze else 1
            # Save images
            elif (key == ord('s') and args.save) or args.save_all:
                util.save_frame(original, frame, args.output_dir, frame_id)
                frame_id += 1

        # Destroy all the windows
        cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    """Main function to run the Charuco detection pipeline."""
    # Default path for sample image
    # path = "assets/charuco_boards/charuco_board_7x7.png"
    path = "calibration_images/calibration_images_0/calib_0039.png"
    intrinsics = "calibration_images/calibration_images_0/calibration.xml"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Charuco board detection for camera calibration")
    parser.add_argument('--index', default=path, type=str, help='Camera index, video file path, or image path')
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
    parser.add_argument('--square-length', type=float, default=0.10, help='Square length in meters')
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

    # Run the pipeline
    run_pipeline(args, detector)


if __name__ == '__main__':
    main()
