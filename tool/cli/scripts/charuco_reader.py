"""Charuco detection pipeline for camera calibration.

Detects Charuco boards from a single image, a directory of images (recursive
glob), a video file, or a live camera feed, visualizes the results, and
optionally saves frames. The source is chosen from ``--index``.
"""
# === Standard Libraries ===
import os
import glob
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
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging = logging.getLogger(__name__)

# Image extensions recognized when --index is a single file or a directory.
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg')


def _list_images(directory: str) -> List[str]:
    """Return every image under a directory (recursive), sorted by path.

    Args:
        directory: Directory to search.

    Returns:
        Sorted list of image file paths.
    """
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(glob.glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))
    return sorted(paths)


def run_pipeline(args: argparse.Namespace,
                 detector: CharucoDetector,
                 freeze: int = 1,
                 resolution: Tuple[int, int] = Resolution.FHD,
                 winname: str = "Charuco Detection") -> None:
    """Run the Charuco detection pipeline on the source selected by ``args.index``.

    ``args.index`` picks the source:
        - a directory   -> every image inside it (recursive glob)
        - an image file -> that single image
        - a digit       -> camera index
        - anything else -> video file path

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

    # Single image: process, show once, and wait for a key.
    if os.path.isfile(args.index) and args.index.lower().endswith(IMAGE_EXTENSIONS):
        frame = cv2.imread(args.index)
        if frame is None:
            logging.error(f"❌ Cannot open image {args.index}")
            cv2.destroyAllWindows()
            return

        frame = detector.run_charuco_pipeline(frame)
        cv2.imshow(winname, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return

    # For recording (folder / video / camera loops)
    if args.save_all:
        args.save = True
        freeze = 1000
    if args.save:
        logging.info("⚠️ Press 's' to save an image or 'q' to quit the process!")

    # Directory of images: iterate over the globbed frames.
    if os.path.isdir(args.index):
        image_paths = _list_images(args.index)
        if not image_paths:
            logging.error(f"❌ No images found in {args.index}")
            cv2.destroyAllWindows()
            return

        frame_id = 0
        for image_path in image_paths:
            original = cv2.imread(image_path)
            if original is None:
                logging.warning(f"⚠️ Could not read image {image_path}")
                continue

            # Process frame
            frame = detector.run_charuco_pipeline(original.copy())

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

        cv2.destroyAllWindows()
        return

    # Camera index or video file.
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
        cv2.destroyAllWindows()
        return

    frame_id = 0

    # Process video/camera feed
    while cap.isOpened():
        success, original = cap.read()

        if not success:
            break

        # Process frame
        frame = detector.run_charuco_pipeline(original.copy())

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


def main(argv: Optional[List[str]] = None) -> None:
    """Main function to run the Charuco detection pipeline."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Charuco board detection for camera calibration")
    parser.add_argument('--index', default="6", type=str,
                        help='Camera index, video file path, image path, or a directory of images')
    parser.add_argument('--camera-params', default=None, type=str, help='Path to camera calibration file')
    parser.add_argument('--resolution', type=str, default='HD', choices=['SS', 'SD', 'HD', 'FHD', 'UHD', 'OMS'], help='Camera resolution')

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
    parser.add_argument('--y-squares', type=int, default=5, help='Number of squares in Y direction')
    parser.add_argument('--square-length', type=float, default=0.053, help='Square length in meters')
    parser.add_argument('--marker-length', type=float, default=None, help='Marker length in meters (default: 75%% of square length)')

    # Output arguments
    parser.add_argument('--output-dir', default="outputs/charuco_detection", type=str, help='Output path')
    parser.add_argument('--save', action="store_true", default=False, help='Save flag')
    parser.add_argument('--save-all', action="store_true", default=False, help='Save all frames flag')
    args = parser.parse_args(argv)

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
