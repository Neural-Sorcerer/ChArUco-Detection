"""Charuco detection pipeline for camera calibration.

This module provides functionality for detecting Charuco boards in images and videos,
visualizing the results, and saving the data for camera calibration.
"""
import os
import argparse
import logging
from typing import Tuple, Dict, List, Optional, Any, Union

import cv2
import numpy as np

from utils import util
from src.charuco_detector import CharucoDetector
from configs.config import Resolution, CharucoBoardConfig, DetectorConfig, CharucoDetectorConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_frame(detector: CharucoDetector,
                  frame: np.ndarray,
                  draw_marker_corners: bool = True,
                  draw_charuco_corners: bool = True,
                  show_ids: bool = False,
                  project_points: bool = False,
                  evaluate_3d: bool = False,
                  pose_matrix: Optional[np.ndarray] = None) -> None:
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
    # Detect Charuco board
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detect_board(frame)

    # Visualization
    if marker_corners and draw_marker_corners:
        detector.draw_detected_markers(frame, marker_corners, marker_ids)

    # Check if Charuco corners are detected
    if not (charuco_corners is not None and len(charuco_corners) > 0):
        return

    if draw_charuco_corners:
        detector.draw_detected_corners(frame, charuco_corners, charuco_ids)

    if show_ids:
        detector.draw_corner_ids(frame, charuco_corners, charuco_ids)

    # Estimate pose if camera parameters are available
    success, rvec, tvec = detector.estimate_pose(charuco_corners, charuco_ids)

    if not success:
        return

    # Draw axes
    detector.draw_axes(frame, rvec, tvec)

    # Project 3D points to image plane
    if project_points:
        detector.project_points(frame, rvec, tvec)

    if evaluate_3d and pose_matrix is not None:
        # Verify 3D consistency from moving one camera coordinate system to another
        util.verify_consistency_3Dobjpoints(
            detector.objpoint,
            pose_matrix,
            detector.board_config.size,
            detector.board_config.square_length
        )


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
    # Auxiliary camera pose relative to origin camera pose (if needed)
    pose_matrix = np.array([
        [-0.99604347062072573,  -0.078366409130313411,  0.041905972770475794,   -0.29179406479883679    ],
        [-0.020389427994994425, 0.66050204336743168,    0.75054734822893399,    -0.84698754658494668    ],
        [-0.086496681207179654, 0.74672334678076591,    -0.65948659388396635,   1.6650489727770146      ],
        [0.,                    0.,                     0.,                     1.                      ],
    ]) if args.evaluate_3d else None

    # Check if input is an image file
    if os.path.isfile(args.index) and args.index.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read image from file
        frame = cv2.imread(args.index)

        if frame is None:
            logger.error(f"Cannot open image {args.index}")
            return

        # Create a window
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, width=Resolution.HD[0], height=Resolution.HD[1])

        # Process single image
        process_frame(
            detector, frame,
            draw_marker_corners=args.draw_marker_corners,
            draw_charuco_corners=args.draw_charuco_corners,
            show_ids=args.show_ids,
            project_points=args.project_points,
            evaluate_3d=args.evaluate_3d,
            pose_matrix=pose_matrix
        )

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
            logger.error(f"Cannot open camera {args.index}")
            return

        # Create a window
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, width=Resolution.HD[0], height=Resolution.HD[1])

        # For recording
        if args.save_all:
            args.save = True
            freeze = 1000

        if args.save:
            logger.info("Press 's' to save an image or 'q' to quit the process!")

        frame_id = 0
        while cap.isOpened():
            success, original = cap.read()

            if not success:
                break

            # Save original frame
            frame = original.copy()

            # Process frame
            process_frame(
                detector, frame,
                draw_marker_corners=args.draw_marker_corners,
                draw_charuco_corners=args.draw_charuco_corners,
                show_ids=args.show_ids,
                project_points=args.project_points,
                evaluate_3d=args.evaluate_3d,
                pose_matrix=pose_matrix
            )

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
                util.save_frame(original, args.output_dir, frame_id, annotated=frame)
                frame_id += 1

        # Destroy all the windows
        cap.release()
    cv2.destroyAllWindows()


def main() -> None:
    """Main function to run the Charuco detection pipeline."""
    # Default path for sample image
    path = "assets/sample.png"

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Charuco board detection for camera calibration")
    parser.add_argument('--index', default=path, type=str, help='camera index, video file path, or image path')
    parser.add_argument('--output-dir', default="outputs/charuco_detection", type=str, help='output path')
    parser.add_argument('--save', action="store_true", help='save flag')
    parser.add_argument('--save-all', action="store_true", help='save all frames flag')
    parser.add_argument('--camera-params', type=str, default="assets/intrinsics.xml", help='path to camera calibration file')
    parser.add_argument('--draw-marker-corners', action="store_true", default=True, help='draw marker corners')
    parser.add_argument('--draw-charuco-corners', action="store_true", default=True, help='draw charuco corners')
    parser.add_argument('--show-ids', action="store_true", default=False, help='show corner IDs')
    parser.add_argument('--project-points', action="store_true", default=False, help='project 3D points to image plane')
    parser.add_argument('--evaluate-3d', action="store_true", default=False, help='evaluate 3D consistency')
    args = parser.parse_args()

    # Create configurations
    board_config = CharucoBoardConfig()
    detector_config = DetectorConfig()
    charuco_detector_config = CharucoDetectorConfig()

    # Create detector
    detector = CharucoDetector(board_config, detector_config, charuco_detector_config)

    # Load camera parameters if provided
    if args.camera_params and os.path.isfile(args.camera_params):
        detector.load_camera_params(args.camera_params)

    # Run the pipeline
    run_pipeline(args, detector)


if __name__ == '__main__':
    main()
