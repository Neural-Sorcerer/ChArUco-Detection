"""Camera calibration script using Charuco boards.

This script provides a command-line interface for calibrating cameras using Charuco boards.
It can be used to collect calibration data, perform calibration, and save calibration parameters.
"""
# === Standard Libraries ===
import os
import glob
import logging
import argparse
from time import time
from typing import Tuple, Optional

# === Third-Party Libraries ===
import cv2
import numpy as np

# === Local Modules ===
from src.calibration import CameraCalibrator
from src.charuco_detector import CharucoDetector
from configs.config import Resolution, CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s:%(lineno)02d - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'  # Removes milliseconds
)
logging = logging.getLogger(__name__)
np.set_printoptions(suppress=True, precision=14)


def collect_calibration_images(args: argparse.Namespace,
                               detector: CharucoDetector,
                               resolution: Tuple[int, int] = Resolution.FHD) -> None:
    """Collect calibration images from a camera.

    Args:
        args: Command-line arguments
        detector: CharucoDetector instance
        resolution: Resolution for video capture
    """
    logging.info(f"⭐ ───────────── Collecting Calibration Images ───────────── ⭐")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Open camera or video file
    if args.index.isdigit():
        # If it's an integer, it's a camera index
        cap = cv2.VideoCapture(int(args.index), cv2.CAP_V4L2)
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

    # Create window
    winname = "Calibration Image Collection"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, width=Resolution.HD[0], height=Resolution.HD[1])

    logging.info("⚠️ Press 's' to save an image or 'q' to quit the process!")

    frame_id = 0
    save_time = 0
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            break

        # Make a copy for visualization
        display_frame = frame.copy()

        # Detect Charuco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detect_board(display_frame)

        # Visualization
        if marker_corners:
            detector.draw_detected_markers(display_frame, marker_corners, marker_ids)

        if (charuco_corners is not None) and (len(charuco_corners) > 0):
            detector.draw_detected_corners(display_frame, charuco_corners, charuco_ids)

            # Show number of detected corners
            cv2.putText(
                display_frame,
                f"Detected corners: {len(charuco_corners)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )
        else:
            # Show warning
            cv2.putText(
                display_frame,
                "No corners detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

        # Show saved flag
        if (time() - save_time) < 1:
            cv2.putText(
                display_frame,
                f"✅ Image Saved",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

        # Show frame
        cv2.imshow(winname, display_frame)

        # Get key
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Only save if corners are detected
            if (charuco_corners is not None) and (len(charuco_corners) >= 4):
                output_path = os.path.join(args.output_dir, f"calib_{frame_id:04d}.png")
                cv2.imwrite(output_path, frame)
                logging.info(f"Saved {output_path} with {len(charuco_corners)} corners")
                frame_id += 1
                save_time = time()
                
            else:
                logging.warning("⚠️ Not enough corners detected. Image not saved.")

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

    logging.info(f"✅ Collected {frame_id} calibration images")


def calibrate_from_images(args: argparse.Namespace, detector: CharucoDetector) -> Optional[CameraCalibrator]:
    """Calibrate camera from collected images.

    Args:
        args: Command-line arguments
        detector: CharucoDetector instance

    Returns:
        CameraCalibrator instance if calibration was successful, None otherwise
    """
    logging.info(f"⭐ ───────────── Camera Calibration ───────────── ⭐")
    
    # Create calibrator
    calibrator = CameraCalibrator(detector, fisheye=args.fisheye)

    # Add calibration images
    num_images = calibrator.add_calibration_images_from_directory(
        directory=args.input_dir,
        pattern=args.pattern
    )

    if num_images == 0:
        logging.error(f"❌ No calibration images found")
        return None

    # Perform calibration
    if not calibrator.calibrate():
        logging.error("❌ Calibration failed")
        return None

    # Save calibration parameters
    if not calibrator.save_calibration(args.output_file):
        logging.error("❌ Failed to save calibration parameters")

    # Show calibration metrics
    metrics = calibrator.get_calibration_metrics()
    calibrator.show_calibration_metrics(metrics)

    return calibrator


def test_calibration(args: argparse.Namespace, calibrator: CameraCalibrator) -> None:
    """Test calibration by undistorting images.

    Args:
        args: Command-line arguments
        calibrator: CameraCalibrator instance
    """
    # Create output directory
    logging.info(f"⭐ ───────────── Applying Undistortion ───────────── ⭐")
    
    undistort_dir = os.path.join(os.path.dirname(args.output_file), "undistorted")
    os.makedirs(undistort_dir, exist_ok=True)

    # Get all images
    image_files = glob.glob(os.path.join(args.input_dir, args.pattern))

    if not image_files:
        logging.warning(f"⚠️ No images found in {args.input_dir} matching pattern {args.pattern}")
        return

    # Undistort each image
    for i, image_file in enumerate(image_files):
        logging.info(f"Undistorting {image_file}")
        image = cv2.imread(image_file)

        if image is None:
            logging.warning(f"⚠️ Could not read image {image_file}")
            continue

        # Undistort
        undistorted = calibrator.undistort_image(image, args.balance, args.simple)

        # Save undistorted image
        output_path = os.path.join(undistort_dir, f"undistorted_{i:04d}.png")
        cv2.imwrite(output_path, undistorted)

    logging.info(f"✅ Undistorted images saved to {undistort_dir}")


def main() -> None:
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Camera calibration using Charuco boards")

    # Common arguments
    parser.add_argument('--board-id', type=int, default=0, help='Charuco board ID')
    parser.add_argument('--x-squares', type=int, default=7, help='Number of squares in X direction')
    parser.add_argument('--y-squares', type=int, default=7, help='Number of squares in Y direction')
    parser.add_argument('--square-length', type=float, default=0.12, help='Square length in meters')
    parser.add_argument('--marker-length', type=float, default=None, help='Marker length in meters (default: 75% of square length)')

    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Generate board mode
    generate_parser = subparsers.add_parser('generate', help='Generate Charuco board image')
    generate_parser.add_argument('--output-file', type=str, default='charuco_board.png', help='Output file for board image')
    generate_parser.add_argument('--pixels-per-square', type=int, default=300, help='Pixels per square')
    generate_parser.add_argument('--margin-percent', type=float, default=0.05, help='Margin around the board as a percentage (0.05 = 5%) of the minimum grid dimension')

    # Collect mode
    collect_parser = subparsers.add_parser('collect', help='Collect calibration images')
    collect_parser.add_argument('--index', type=str, default="2", help='Camera index or video file path')
    collect_parser.add_argument('--output-dir', type=str, default='calibration_images', help='Output directory for calibration images')
    collect_parser.add_argument('--resolution', type=str, default='FHD', choices=['SS', 'SD', 'HD', 'FHD', 'UHD', 'OMS'], help='Camera resolution')

    # Calibrate mode
    calibrate_parser = subparsers.add_parser('calibrate', help='Calibrate camera from images')
    calibrate_parser.add_argument('--input-dir', type=str, default='calibration_images', help='Input directory with calibration images')
    calibrate_parser.add_argument('--pattern', type=str, default='*.png', help='File pattern for calibration images')
    calibrate_parser.add_argument('--output-file', type=str, default='calibration.xml', help='Output file for calibration parameters')
    calibrate_parser.add_argument('--fisheye', action='store_true', help='Assume fisheye camera model')
    calibrate_parser.add_argument('--undistort', action='store_true', help='Test calibration by undistorting images')
    calibrate_parser.add_argument('--balance', type=float, default=0.0, help='Balance value for undistortion (0.0 = crop, 1.0 = stretch)')
    calibrate_parser.add_argument('--simple', action='store_true', help='Use simple undistortion (no remapping)')

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
    detector = CharucoDetector(board_config, detector_config, charuco_detector_config)

    # Handle different modes
    if args.mode == 'collect':
        # Get resolution
        resolution = getattr(Resolution, args.resolution)
        collect_calibration_images(args, detector, resolution)

    elif args.mode == 'calibrate':
        calibrator = calibrate_from_images(args, detector)

        if calibrator and args.undistort:
            test_calibration(args, calibrator)

    elif args.mode == 'generate':
        detector.save_board_image(args.output_file,
                                  pixels_per_square=args.pixels_per_square,
                                  margin_percent=args.margin_percent)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
""" 
python calibrate_camera.py calibrate \
    --input-dir calibration_images/calibration_images_8 \
    --output-file calibration_images_8/calibration.xml \
    --undistort \
    --balance 0.0
    
python calibrate_camera.py calibrate \
    --input-dir calibration_images/calibration_images_0 \
    --output-file calibration.xml \
    --fisheye \
    --undistort \
    --balance 0.0
"""
