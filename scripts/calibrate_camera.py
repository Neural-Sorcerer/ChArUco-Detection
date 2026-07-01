"""Camera calibration script using Charuco boards.

This script provides a command-line interface for calibrating cameras using Charuco boards.
It can be used to collect calibration data, perform calibration, and save calibration parameters.
"""
# === Standard Libraries ===
import os
import glob
import logging
import argparse
from typing import *
from time import time

# === Third-Party Libraries ===
import cv2
import numpy as np

# === Local Modules ===
from src.calibration import CameraCalibrator
from utils.data_judgment import DataQualityJudge, CalibrationSample
from src.charuco_detector import CharucoDetector
from configs.config import Resolution, CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logging = logging.getLogger(__name__)
np.set_printoptions(suppress=True, precision=14)

# Every collected and calibrated artifact is rooted here, regardless of the
# --output-dir / --output-file passed on the CLI (see _under_outputs).
OUTPUT_ROOT = "outputs"


def _under_outputs(path: str) -> str:
    """Redirect a relative output path under the top-level ``outputs/`` directory.

    Whatever ``--output-dir`` / ``--output-file`` is given, the result is forced
    to live under ``outputs/`` so every collected and calibrated artifact ends up
    in one place. An absolute path is treated as a manual override and returned
    unchanged.

    Args:
        path: The output directory or file path from the CLI.

    Returns:
        The path rooted under ``outputs/`` (or unchanged if absolute).
    """
    if os.path.isabs(path):
        return path
    normalized = os.path.normpath(path)
    if normalized.split(os.sep)[0] == OUTPUT_ROOT:
        return normalized
    return os.path.join(OUTPUT_ROOT, normalized)


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
            detector.draw_detected_markers_cv2(display_frame, marker_corners, marker_ids)

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
                f"Image Saved",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
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


def _screen_size(default: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    """Best-effort desktop resolution, used to size the preview windows.

    Args:
        default: Fallback size when the screen cannot be queried (e.g. headless).

    Returns:
        The screen (width, height) in pixels.
    """
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return default


def collect_with_quality_assessment(
    args: argparse.Namespace,
    detector: CharucoDetector,
    resolution: Tuple[int, int] = Resolution.FHD
) -> None:
    """Collect calibration images with quality assessment and diversity filtering.

    Args:
        args: Command-line arguments
        detector: CharucoDetector instance
        resolution: Resolution for video capture
    """
    logging.info(f"⭐ ───────────── Collecting Images with Quality Assessment ───────────── ⭐")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Open camera or video file
    if args.index.isdigit():
        cap = cv2.VideoCapture(int(args.index), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(args.index)

    if not cap.isOpened():
        logging.error(f"❌ Cannot open camera {args.index}")
        return

    # Use the resolution the camera ACTUALLY delivers: a webcam silently ignores a
    # request it cannot honor (e.g. 720p hardware asked for UHD). Every metric is
    # normalized by this size, so a mismatch would corrupt all coverage scoring.
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_w > 0 and actual_h > 0 and (actual_w, actual_h) != tuple(resolution):
        logging.warning(f"⚠️ Camera delivered {actual_w}x{actual_h}, not the requested "
                        f"{resolution[0]}x{resolution[1]} — using the actual size")
        resolution = (actual_w, actual_h)

    # Initialize data quality judge (board enables a live reprojection-error readout)
    judge = DataQualityJudge(
        image_size=resolution,
        board=detector.board_config.board,
        min_sharpness=args.min_sharpness,
        target_samples=args.target_samples
    )

    # One mouse-resizable window: the full camera view with the guidance panel
    # concatenated on its right. KEEPRATIO keeps the whole frame undistorted.
    win = "Charuco Calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # Size the window to fill the screen at the combined view's aspect ratio (camera
    # frame + a panel strip scaled to the frame height).
    screen_w, screen_h = _screen_size()
    panel_nat_w, panel_nat_h = 460, 900
    combined_w = resolution[0] + int(panel_nat_w * resolution[1] / panel_nat_h)
    fit = min((screen_w - 40) / combined_w, (screen_h - 80) / resolution[1])
    cv2.resizeWindow(win, max(640, int(combined_w * fit)), max(480, int(resolution[1] * fit)))
    cv2.moveWindow(win, 20, 20)

    logging.info("⚠️ Keys: 's' save | 'h' heatmap tint | 'm' full report | 'q' quit")

    frame_id = 0
    show_heatmap = False
    show_report = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        display_frame = frame.copy()

        # Detect Charuco board
        charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detect_board(display_frame)

        # Draw raw detections (markers + corner dots) on the live frame
        if marker_corners:
            detector.draw_detected_markers_cv2(display_frame, marker_corners, marker_ids)
        if (charuco_corners is not None) and (len(charuco_corners) > 0):
            detector.draw_detected_corners_cv2(display_frame, charuco_corners, charuco_ids)

        # Evaluate the current view (read-only — does not commit the sample)
        sample = None
        if (charuco_corners is not None) and (len(charuco_corners) > 0):
            sample = judge.evaluate(charuco_corners, charuco_ids, timestamp=frame_id, image=frame)

        # Single-window view: the full report ('m') or the live frame with the
        # guidance panel scaled to the frame height and concatenated on its right.
        if show_report:
            view = judge.render_report_view()
        else:
            cam = judge.render_frame(display_frame, sample, show_heatmap=show_heatmap)
            panel_nat = judge.render_panel(sample)
            ph = cam.shape[0]
            pw = max(1, int(panel_nat.shape[1] * ph / panel_nat.shape[0]))
            panel = cv2.resize(panel_nat, (pw, ph), interpolation=cv2.INTER_AREA)
            view = np.hstack([cam, panel])
        cv2.imshow(win, view)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        save_request = (key == ord('s')) or (args.auto_save and sample is not None and sample.is_accepted)

        if key == ord('q'):
            break
        elif key == ord('h'):
            show_heatmap = not show_heatmap
        elif key == ord('m'):
            show_report = not show_report
        elif save_request:
            if sample is not None and sample.is_accepted:
                output_path = os.path.join(args.output_dir, f"calib_{frame_id:04d}.png")
                cv2.imwrite(output_path, frame)
                judge.commit(sample)  # only now is the view recorded into coverage
                logging.info(f"✅ Saved {output_path} — size={sample.size:.3f}, skew={sample.skew:.2f}")
                frame_id += 1
            elif sample is not None:
                logging.warning(f"⚠️ Not saved — {sample.reject_reason}")
            else:
                logging.warning("⚠️ Not saved — no board detected")

    # Cleanup and generate reports
    cap.release()
    cv2.destroyAllWindows()
    judge.close()  # retire the background calibration worker

    heatmap_path = os.path.join(args.output_dir, "final_heatmap.png")
    judge.generate_heatmap(heatmap_path)

    summary_path = os.path.join(args.output_dir, "collection_summary.json")
    judge.export_summary(summary_path)

    progress = judge.get_progress_info()
    logging.info(f"🎯 Collection complete: {progress['accepted_samples']} diverse samples")

    if progress['is_sufficient']:
        logging.info("✅ Sufficient diverse samples for calibration!")
    else:
        logging.warning(f"⚠️ Consider collecting more samples (target: {judge.target_samples})")


def filter_existing_dataset(args: argparse.Namespace, detector: CharucoDetector) -> None:
    """Filter an existing dataset to remove redundant samples.

    Args:
        args: Command-line arguments
        detector: CharucoDetector instance (uses the board from the CLI args)
    """
    logging.info(f"⭐ ───────────── Filtering Existing Dataset ───────────── ⭐")

    from time import time
    import shutil

    # Find all images in input directory
    image_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.png"), recursive=True))
    image_files.extend(glob.glob(os.path.join(args.input_dir, "**", "*.jpg")))

    if not image_files:
        logging.error(f"❌ No images found in {args.input_dir}")
        return

    logging.info(f"Found {len(image_files)} images to filter")

    # Initialize quality judge
    sample_image = cv2.imread(image_files[0])
    image_size = (sample_image.shape[1], sample_image.shape[0])

    judge = DataQualityJudge(
        image_size=image_size,
        target_samples=args.target_samples
    )

    # Process all images and create samples
    samples = []
    for i, image_file in enumerate(image_files):
        logging.info(f"Processing {i+1}/{len(image_files)}: {os.path.basename(image_file)}")

        image = cv2.imread(image_file)
        if image is None:
            continue

        # Detect corners
        charuco_corners, charuco_ids, _, _ = detector.detect_board(image)

        if charuco_corners is not None and len(charuco_corners) >= 4:
            sample = judge.evaluate(charuco_corners, charuco_ids, image_file, time())
            samples.append(sample)

    # Filter samples for diversity
    filtered_samples = judge.filter_existing_dataset(samples)

    # Create output directory and copy filtered images
    os.makedirs(args.output_dir, exist_ok=True)

    for i, sample in enumerate(filtered_samples):
        src_path = sample.image_path
        dst_path = os.path.join(args.output_dir, f"filtered_{i:04d}.png")

        # Copy image
        shutil.copy2(src_path, dst_path)

    # Generate reports
    heatmap_path = os.path.join(args.output_dir, "filtered_heatmap.png")
    judge.generate_heatmap(heatmap_path)

    summary_path = os.path.join(args.output_dir, "filter_summary.json")
    judge.export_summary(summary_path)

    logging.info(f"🎯 Filtering complete: {len(image_files)} -> {len(filtered_samples)} images")
    logging.info(f"📊 Reports saved: {heatmap_path}, {summary_path}")


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
    if not calibrator.save_calibration_parameters(args.output_file):
        logging.error("❌ Failed to save calibration parameters")

    # Save calibration debug data to JSON
    json_output_file = os.path.splitext(args.output_file)[0] + "_corners.json"
    if not calibrator.save_calibration_json(json_output_file):
        logging.error("❌ Failed to save calibration JSON data")

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
    image_files = sorted(glob.glob(os.path.join(args.input_dir, "**", args.pattern), recursive=True))

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

        # Undistort (undistort_image returns the image and the new camera matrix)
        undistorted, _ = calibrator.undistort_image(image, args.balance, args.simple)

        # Save undistorted image
        output_path = os.path.join(undistort_dir, f"undistorted_{i:04d}.png")
        cv2.imwrite(output_path, undistorted)

    logging.info(f"✅ Undistorted images saved to {undistort_dir}")


def main(argv: Optional[List[str]] = None) -> None:
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Camera calibration using Charuco boards")

    # Common arguments
    parser.add_argument('--board-id', type=int, default=0, help='Charuco board ID')
    parser.add_argument('--x-squares', type=int, default=7, help='Number of squares in X direction')
    parser.add_argument('--y-squares', type=int, default=5, help='Number of squares in Y direction')
    parser.add_argument('--square-length', type=float, default=0.10, help='Square length in meters')
    parser.add_argument('--marker-length', type=float, default=None, help='Marker length in meters (default: 75%% of square length)')

    # Subparsers for different modes
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')

    # Generate board mode
    generate_parser = subparsers.add_parser('generate', help='Generate Charuco board image')
    generate_parser.add_argument('--output-file', type=str, default='charuco_board.png', help='Output file for board image')
    generate_parser.add_argument('--pixels-per-square', type=int, default=300, help='Pixels per square')
    generate_parser.add_argument('--margin-percent', type=float, default=0.05, help='Margin around the board as a percentage (0.05 = 5%%) of the minimum grid dimension')

    # Collect mode
    collect_parser = subparsers.add_parser('collect', help='Collect calibration images')
    collect_parser.add_argument('--index', type=str, default="0", help='Camera index or video file path')
    collect_parser.add_argument('--output-dir', type=str, default='calibration_images', help='Output directory for calibration images')
    collect_parser.add_argument('--resolution', type=str, default='HD', choices=['SS', 'SD', 'HD', 'FHD', 'UHD', 'OMS'], help='Camera resolution')
    collect_parser.add_argument('--target-samples', type=int, default=100, help='Target number of diverse samples')
    collect_parser.add_argument('--min-sharpness', type=float, default=400.0, help='Reject blurry/motion-blurred views below this focus score (variance of Laplacian); raise it if you move fast, set 0 to disable')
    collect_parser.add_argument('--use-quality-judge', action='store_true', help='Use data quality assessment during collection')
    collect_parser.add_argument('--auto-save', action='store_true', help='Automatically save good quality samples')

    # Calibrate mode
    calibrate_parser = subparsers.add_parser('calibrate', help='Calibrate camera from images')
    calibrate_parser.add_argument('--input-dir', type=str, default='calibration_images', help='Input directory with calibration images')
    calibrate_parser.add_argument('--pattern', type=str, default='*.png', help='File pattern for calibration images')
    calibrate_parser.add_argument('--output-file', type=str, default='calibration.xml', help='Output file for calibration parameters')
    calibrate_parser.add_argument('--fisheye', action='store_true', help='Assume fisheye camera model')
    calibrate_parser.add_argument('--undistort', action='store_true', help='Test calibration by undistorting images')
    calibrate_parser.add_argument('--balance', type=float, default=0.0, help='Balance value for undistortion (0.0 = crop, 1.0 = stretch)')
    calibrate_parser.add_argument('--simple', action='store_true', help='Use simple undistortion (no remapping)')

    # Filter mode
    filter_parser = subparsers.add_parser('filter', help='Filter existing dataset for diversity')
    filter_parser.add_argument('--input-dir', type=str, default='calibration_images', help='Input directory with images to filter')
    filter_parser.add_argument('--output-dir', type=str, default='filtered_images', help='Output directory for filtered images')
    filter_parser.add_argument('--target-samples', type=int, default=50, help='Target number of diverse samples')

    args = parser.parse_args(argv)

    # Force all collected/calibrated artifacts under outputs/; an absolute path
    # passed manually on the CLI is respected as an override.
    if getattr(args, 'output_dir', None) is not None:
        args.output_dir = _under_outputs(args.output_dir)
    if args.mode == 'calibrate' and getattr(args, 'output_file', None) is not None:
        args.output_file = _under_outputs(args.output_file)

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

    # Handle different modes
    if args.mode == 'collect':
        # Get resolution
        resolution = getattr(Resolution, args.resolution)

        if args.use_quality_judge:
            collect_with_quality_assessment(args, detector, resolution)
        else:
            collect_calibration_images(args, detector, resolution)

    elif args.mode == 'calibrate':
        calibrator = calibrate_from_images(args, detector)

        if calibrator and args.undistort:
            test_calibration(args, calibrator)

    elif args.mode == 'generate':
        detector.save_board_image(args.output_file,
                                  pixels_per_square=args.pixels_per_square,
                                  margin_percent=args.margin_percent)

    elif args.mode == 'filter':
        filter_existing_dataset(args, detector)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
