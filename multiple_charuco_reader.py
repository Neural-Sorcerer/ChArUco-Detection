""" Description: Record data from camera and draw multiple Charuco boards """
import os
import cv2
import argparse 
import numpy as np

from utils.utils import load_camera_params, save_frame


SS = (640, 360)
SD = (640, 480)
HD = (1280, 720)
FHD = (1920, 1080)
UHD = (3840, 2160)

# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
# Define board configurations
NUM_BOARDS = 5
SQUARES_X = 7
SQUARES_Y = 7
SQUARE_LENGTH = 0.053
MARKER_LENGTH = 0.039

# Define the dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_1000)

# Detector parameters - optimized for accuracy and robustness
params = cv2.aruco.DetectorParameters()
params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
params.cornerRefinementWinSize = 5
params.cornerRefinementMaxIterations = 30
params.cornerRefinementMinAccuracy = 0.1
params.adaptiveThreshWinSizeMin = 3
params.adaptiveThreshWinSizeMax = 23
params.adaptiveThreshWinSizeStep = 10
params.adaptiveThreshConstant = 7
params.minMarkerPerimeterRate = 0.03
params.maxMarkerPerimeterRate = 4.0
params.polygonalApproxAccuracyRate = 0.03
params.minCornerDistanceRate = 0.05
params.minDistanceToBorder = 3
params.minMarkerDistanceRate = 0.05
params.useAruco3Detection = True

# Charuco detector parameters
charuco_params = cv2.aruco.CharucoParameters()
charuco_params.minMarkers = 2
charuco_params.tryRefineMarkers = True

# Define board configurations with different ID ranges and colors
board_configs = []
markers_per_board = int(SQUARES_X * SQUARES_Y / 2.0)  # Number of markers per board

for i in range(NUM_BOARDS):
    start_id = i * markers_per_board
    end_id = (i + 1) * markers_per_board
    
    # Create a unique color for each board
    color = [0, 0, 0]
    color[i % 3] = 255
    if i >= 3:
        color[(i+1) % 3] = 255
    
    board_configs.append({
        "name": f"Board {i+1}",
        "id_range": (start_id, end_id),
        "color": tuple(color),
        "board": cv2.aruco.CharucoBoard(
            size=(SQUARES_X, SQUARES_Y),
            squareLength=SQUARE_LENGTH,
            markerLength=MARKER_LENGTH,
            dictionary=aruco_dict
        )
    })


def process_frame(args, frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect all markers first
    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(
        gray, aruco_dict, parameters=params
    )
    
    # Draw all detected markers if requested
    if marker_corners and args.draw_marker_corners:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, borderColor=(0, 255, 255))
    
    # If no markers detected, return early
    if marker_ids is None or len(marker_ids) == 0:
        return
    
    # Process each board separately
    for board_config in board_configs:
        start_id, end_id = board_config["id_range"]
        board_color = board_config["color"]
        board_name = board_config["name"]
        board = board_config["board"]
        
        # Filter markers for this board
        board_marker_indices = [i for i, id in enumerate(marker_ids) if start_id <= id[0] < end_id]
        if not board_marker_indices:
            continue
            
        board_marker_corners = [marker_corners[i] for i in board_marker_indices]
        board_marker_ids = np.array([marker_ids[i] for i in board_marker_indices])
        
        # Interpolate corners for this board
        charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=board_marker_corners, 
            markerIds=board_marker_ids, 
            image=gray,
            board=board
        )
        
        # Skip if no Charuco corners detected
        if charuco_corners is None or len(charuco_corners) == 0:
            continue
            
        # Draw detected Charuco corners
        if args.draw_charuco_corners:
            cv2.aruco.drawDetectedCornersCharuco(
                frame, charuco_corners, charuco_ids, cornerColor=board_color
            )
            
        # Add board name to the image
        if len(charuco_corners) > 0:
            pos = (int(charuco_corners[0][0][0]), int(charuco_corners[0][0][1]) - 20)
            cv2.putText(frame, board_name, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, board_color, 2, cv2.LINE_AA)
        
        # Show IDs if requested
        if args.show_ids:
            for charuco_corner, charuco_id in zip(charuco_corners, charuco_ids.flatten()):
                pos = (int(charuco_corner[0][0]), int(charuco_corner[0][1]))
                cv2.putText(frame, str(charuco_id), pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, board_color, 1, cv2.LINE_AA)
        
        # Estimate pose if camera parameters are provided
        if not (args.camera_params and os.path.isfile(args.camera_params)):
            continue
        
        camera_matrix, dist_coeffs = load_camera_params(args.camera_params)
        if camera_matrix is None or len(charuco_corners) < 4:
            continue

        # Estimate pose
        ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
        )

        # Only draw axes if pose estimation was successful
        if not ret:
            continue
        
        # Ensure rvec and tvec are properly formatted
        rvec = np.array(rvec, dtype=np.float32)
        tvec = np.array(tvec, dtype=np.float32)

        # Draw the axes
        cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
        
        # Optionally display distance information
        if args.show_distance:
            # Calculate distance (in meters)
            distance = np.linalg.norm(tvec)
            distance_text = f"Dist: {distance:.2f}m"
            text_pos = (int(charuco_corners[0][0][0]), int(charuco_corners[0][0][1]) - 40)
            cv2.putText(frame, distance_text, text_pos, 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, board_color, 2, cv2.LINE_AA)


def run_pipeline(args, freeze=1, resolution=FHD, frame_name="Multiple Charuco Detection"):
    # Check if input is an image file
    if os.path.isfile(args.index) and args.index.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read image from file
        frame = cv2.imread(args.index)

        if frame is None:
            print(f"Cannot open image {args.index}")
            return
        
        # Create a window named 'Frame'
        cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(frame_name, width=HD[0], height=HD[1])
    
        # Process single image
        process_frame(args, frame)
        cv2.imshow(frame_name, frame)
        cv2.waitKey(0)
    else:
        # Process video/camera feed
        cap = cv2.VideoCapture(args.index, cv2.CAP_V4L2)  # Ensure V4L2 backend

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
        if not cap.isOpened():
            print(f"Cannot open camera {args.index}")
            return
        
        # Create a window named 'Frame'
        cv2.namedWindow(frame_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(frame_name, width=HD[0], height=HD[1])
        
        # For recording
        if args.save_all:
            args.save = True
            freeze = 1000
            
        if args.save:
            print(f"\nPress 's' to save an image or 'q' to quit the process!\n")
        
        frame_id = 0
        while cap.isOpened():
            success, original = cap.read()
            
            if not success:
                break
        
            # Save original frame
            frame = original.copy()
            
            # Process frame
            process_frame(args, frame)
            
            # Show a frame
            cv2.imshow(frame_name, frame)
        
            # Get a key
            key = cv2.waitKey(freeze) & 0xFF
            
            if key == ord('q'):     # Press 'q' to quit
                break
            elif key == ord('f'):   # Toggle freeze mode
                freeze = 0 if freeze else 1
            # Save images
            elif (key == ord('s') and args.save) or args.save_all:
                save_frame(original, args.output_dir, frame_id, annotated=frame)
                frame_id += 1
            
        # Destroy all the windows
        cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', default=2, type=str, help='camera index, video file path, or image path')
    parser.add_argument('--output-dir', default="outputs/charuco_detection", type=str, help='output path')
    parser.add_argument('--save', action="store_true", help='save flag')
    parser.add_argument('--save-all', action="store_true", help='save all frames flag')
    parser.add_argument('--camera-params', type=str, default="utils/intrinsics.xml", help='path to camera calibration file')
    parser.add_argument('--draw-marker-corners', action="store_true", default=True, help='draw marker corners')
    parser.add_argument('--draw-charuco-corners', action="store_true", default=True, help='draw charuco corners')
    parser.add_argument('--show-ids', action="store_true", help='show corner IDs')
    parser.add_argument('--show-distance', action="store_true", help='show distance to board')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    run_pipeline(args)


if __name__ == "__main__":
    main()
