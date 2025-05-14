""" Description: Record data from camera and draw Charuco board """
import os
import cv2
import argparse 
import numpy as np

from utils import util


SS = (640, 360)
SD = (640, 480)
HD = (1280, 720)
FHD = (1920, 1080)
UHD = (3840, 2160)


# -------------------------------------------
# CONFIGURATION
# -------------------------------------------
boardID = 1
x_squares = 7
y_squares = 5
squareLength = 0.12                 # in meters
markerLength = squareLength * 0.75  # in meters with 75% of the square length

size = (x_squares, y_squares)
markers_per_board = int(x_squares * y_squares / 2.0)
start = boardID * markers_per_board
stop = (boardID + 1) * markers_per_board
ids = np.array(range(start, stop), dtype=np.int32)

# Define the dictionary and board parameters
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
params.useAruco3Detection = True        # Use the faster Aruco3 detection method

# Charuco detector parameters
charuco_params = cv2.aruco.CharucoParameters()
charuco_params.minMarkers = 2           # Minimum markers needed to detect a Charuco corner
charuco_params.tryRefineMarkers = True  # Try to refine marker detection

# Define the Charuco board
board = cv2.aruco.CharucoBoard(
    size=size,
    squareLength=squareLength,
    markerLength=markerLength,
    dictionary=aruco_dict,
    ids=ids
)

# Create CharucoDetector with optimized parameters
charuco_detector = cv2.aruco.CharucoDetector(board,
                                             detectorParams=params,
                                             charucoParams=charuco_params)


# Prepare object points (3D points in real-world space)
objp = np.zeros((x_squares * y_squares, 3), np.float32)
objp[:, :2] = np.mgrid[0:x_squares, 0:y_squares].T.reshape(-1, 2)
objp *= squareLength     # Scale according to real square size
objpoint = objp.astype(np.float32).reshape(-1, 1, 3)

# Auxiliary camera pose relative to origin camera pose
pose_matrix = np.array([
    [-0.99604347062072573,  -0.078366409130313411,  0.041905972770475794,   -0.29179406479883679    ],
    [-0.020389427994994425, 0.66050204336743168,    0.75054734822893399,    -0.84698754658494668    ],
    [-0.086496681207179654, 0.74672334678076591,    -0.65948659388396635,   1.6650489727770146      ],
    [0.,                    0.,                     0.,                     1.                      ],
])

# -------------------------------------------
# -------------------------------------------


def process_frame(args, frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect markers and interpolate Charuco corners in one step
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)

    # Visualization
    if marker_corners and args.draw_marker_corners:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids, borderColor=(0, 255, 255))

    # Check if Charuco corners are detected
    if not (charuco_corners is not None and len(charuco_corners) > 0):
        return

    if args.draw_charuco_corners:
        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, cornerColor=(255, 255, 0))
    
    if args.show_ids:
        for charuco_corner, charuco_id in zip(charuco_corners, charuco_ids.flatten()):
            pos = (int(charuco_corner[0][0]), int(charuco_corner[0][1]))
            cv2.putText(frame, str(charuco_id), pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    
    # Estimate pose if camera parameters are provided
    if not (args.camera_params and os.path.isfile(args.camera_params)):
        return
    
    camera_matrix, dist_coeffs = util.load_camera_params(args.camera_params)
    if (camera_matrix is None) or (charuco_corners is None) or len(charuco_corners) < 4:
        return

    # Estimate pose
    ret, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs, None, None
    )

    # Only draw axes if pose estimation was successful
    if not ret:
        return
    
    # Ensure rvec and tvec are properly formatted
    rvec = np.array(rvec, dtype=np.float32)
    tvec = np.array(tvec, dtype=np.float32)
    
    # Draw the axes
    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
    
    if args.project_points:
        # Project 3D points to image plane
        imgpoints_proj = util.project_points_to_image(objpoint, rvec, tvec, camera_matrix, dist_coeffs)

        # Draw projected points
        for imgpoint in imgpoints_proj:
            cv2.circle(frame, (int(imgpoint[0]), int(imgpoint[1])), 5, (0, 0, 255), -1)

    if args.evaluate_3d:
        # Verify 3D consistency from moving one camera coordinate system to another
        util.verify_consistency_3Dobjpoints(objpoint, pose_matrix, size, squareLength)


def run_pipeline(args, freeze=1, resolution=FHD, winname="Charuco Detection"):
    # Check if input is an image file
    if os.path.isfile(args.index) and args.index.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Read image from file
        frame = cv2.imread(args.index)

        if frame is None:
            print(f"Cannot open image {args.index}")
            return
        
        # Create a window named 'Frame'
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, width=HD[0], height=HD[1])
    
        # Process single image
        process_frame(args, frame)
        
        # Show a frame
        cv2.imshow(winname, frame)
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
        cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winname, width=HD[0], height=HD[1])
        
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


def main():
    path = "assets/sample.png"
    
    parser = argparse.ArgumentParser()
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

    run_pipeline(args)


if __name__ == '__main__':
    main()
