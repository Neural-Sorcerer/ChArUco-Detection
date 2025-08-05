import os
import cv2
import glob
import numpy as np
from pathlib import Path
from typing import Dict, Any


SD = (640, 480)
HD = (1280, 720)
FHD = (1920, 1080)
UHD = (3840, 2160)

SQUARE_SIZE_TV_BIG = 105     # Real-world square size in mm
SQUARE_SIZE_TV_SMALL = 90     # Real-world square size in mm
CHECKERBOARD = (15, 8)
CHECKERBOARD_TV = (17, 10)

""" cm
TV1: widht=7.6 height=8.1    90  Fail
TV2: widht=8.9 height=9.5   105  Fail
TV3: widht=9.0 height=9.0    90  OK
"""

# Prepare object points (3D points in real-world space)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_TV_BIG     # Scale according to real square size

objp_tv_big = np.zeros((CHECKERBOARD_TV[0] * CHECKERBOARD_TV[1], 3), np.float32)
objp_tv_big[:, :2] = np.mgrid[0:CHECKERBOARD_TV[0], 0:CHECKERBOARD_TV[1]].T.reshape(-1, 2)
objp_tv_big *= SQUARE_SIZE_TV_BIG     # Scale according to real square size
objp_tv_big[:, :2] = objp_tv_big[:, :2] - SQUARE_SIZE_TV_BIG * 0     # translate

objp_tv_small = np.zeros((CHECKERBOARD_TV[0] * CHECKERBOARD_TV[1], 3), np.float32)
objp_tv_small[:, :2] = np.mgrid[0:CHECKERBOARD_TV[0], 0:CHECKERBOARD_TV[1]].T.reshape(-1, 2)
objp_tv_small *= SQUARE_SIZE_TV_SMALL     # Scale according to real square size
objp_tv_small[:, :2] = objp_tv_small[:, :2] - SQUARE_SIZE_TV_SMALL * 0     # translate


def load_camera_calibration(yaml_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load camera calibration parameters (intrinsics, distortion, extrinsics) from an OpenCV YAML file.

    Parameters:
    -----------
    yaml_path : Path
        Path to the YAML file containing camera calibration data.

    Returns:
    --------
    cameras : dict
        A dictionary with structure:
        {
            'camera_0': {
                'camera_matrix': np.ndarray (3x3),
                'dist_coeffs': np.ndarray (1xN),
                'pose': np.ndarray (4x4),
                'image_size': (width, height),
                'distortion_type': int,
                'camera_group': int
            },
            ...
        }
    """
    fs = cv2.FileStorage(str(yaml_path), cv2.FILE_STORAGE_READ)
    nb_cameras = int(fs.getNode("nb_camera").real())

    cameras = {}

    for cam_idx in range(nb_cameras):
        cam_name = f"camera_{cam_idx}"
        cam_node = fs.getNode(cam_name)

        camera_matrix = cam_node.getNode("camera_matrix").mat()
        dist_coeffs = cam_node.getNode("distortion_vector").mat()
        pose_matrix = cam_node.getNode("camera_pose_matrix").mat()

        image_width = int(cam_node.getNode("img_width").real())
        image_height = int(cam_node.getNode("img_height").real())
        distortion_type = int(cam_node.getNode("distortion_type").real())
        camera_group = int(cam_node.getNode("camera_group").real())

        cameras[cam_name] = {
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "pose": pose_matrix,
            "image_size": (image_width, image_height),
            "distortion_type": distortion_type,
            "camera_group": camera_group,
        }

    fs.release()
    return cameras


def transform_object_pose_between_cameras(
    rvec_obj_in_src_cam: np.ndarray,
    tvec_obj_in_src_cam: np.ndarray,
    src_cam_extrinsic: np.ndarray,
    dst_cam_extrinsic: np.ndarray,
    verbose: bool = False,
    T_name = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform a 3D object's pose from the source camera's coordinate system 
    to the target (destination) camera's coordinate system.

    Parameters:
    ----------
    rvec_obj_in_src_cam : (3, 1) rotation vector of the object in source camera coordinates
    tvec_obj_in_src_cam : (3, 1) translation vector of the object in source camera coordinates
    src_cam_extrinsic : (4, 4) extrinsic matrix of the source camera in world coordinates
    dst_cam_extrinsic : (4, 4) extrinsic matrix of the target camera in world coordinates
    verbose : bool
        If True, prints intermediate transformation matrices

    Returns:
    -------
    rvec_obj_in_dst_cam : (3, 1) rotation vector of the object in destination camera coordinates
    tvec_obj_in_dst_cam : (3, 1) translation vector of the object in destination camera coordinates
    """
    T_net1_board = {}
    T_net1_board["T_net1_board_tv_1"] = np.array([
        [-0.51220160,   -0.01606844,   -0.85871493, -649.27240765],
        [ 0.41370174,   -0.88080740,   -0.23028072, -130.24335167],
        [-0.75266220,   -0.47320201,    0.45779850, 2868.20136331],
        [ 0.00000000,    0.00000000,    0.00000000,    1.00000000],
    ])
    T_net1_board["T_net1_board_tv_2"] = np.array([
        [-0.98022603,   -0.02042590,   -0.19682410,  932.04949588],
        [ 0.11257119,   -0.87558605,   -0.46976250, -331.13267284],
        [-0.16274110,   -0.48263015,    0.86057159, 3248.46174134],
        [ 0.00000000,    0.00000000,    0.00000000,    1.00000000],
    ])
    T_net1_board["T_net1_board_tv_3"] = np.array([
        [-0.85364990,   -0.03571020,    0.51962161, 2198.38359339],
        [-0.22565539,   -0.87379756,   -0.43076381,  -56.92923050],
        [ 0.46942676,   -0.48497690,    0.73785901, 2654.52493370],
        [ 0.00000000,    0.00000000,    0.00000000,    1.00000000],
    ])
    if T_name is not None:
        T_obj_in_src_cam = T_net1_board[T_name]
    else:
        # Convert object rotation vector to rotation matrix
        R_obj_in_src_cam = cv2.Rodrigues(rvec_obj_in_src_cam)[0]

        # Construct the 4x4 pose matrix (homogeneous) of the object in source camera frame
        T_obj_in_src_cam = np.eye(4)
        T_obj_in_src_cam[:3, :3] = R_obj_in_src_cam
        T_obj_in_src_cam[:3, 3] = tvec_obj_in_src_cam.flatten()

    # Compute the transformation from source camera to destination camera
    T_from_src_to_dst = np.linalg.inv(dst_cam_extrinsic) @ src_cam_extrinsic

    # Apply the transformation to move the object's pose into destination camera coordinates
    T_obj_in_dst_cam = T_from_src_to_dst @ T_obj_in_src_cam

    # Extract rotation and translation from the transformed pose
    R_obj_in_dst_cam = T_obj_in_dst_cam[:3, :3]
    rvec_obj_in_dst_cam = cv2.Rodrigues(R_obj_in_dst_cam)[0]
    tvec_obj_in_dst_cam = T_obj_in_dst_cam[:3, 3].reshape(3, 1)

    # Rotate checkboard into Manoj's frame TODO
    rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = rotate_checkboard_into_manoj(
        rvec_obj_in_dst_cam,
        tvec_obj_in_dst_cam,
        T_name
    )

    if verbose:
        print("T_src_to_dst:\n", T_from_src_to_dst)
        print("T_obj_in_dst_cam:\n", T_obj_in_dst_cam)
        print("rvec_obj_in_dst_cam:\n", rvec_obj_in_dst_cam)
        print("tvec_obj_in_dst_cam:\n", tvec_obj_in_dst_cam)
        
    
    return rvec_obj_in_dst_cam, tvec_obj_in_dst_cam


def rotate_checkboard_into_manoj(rvec_obj_in_dst_cam, tvec_obj_in_dst_cam, T_name):
    if T_name == "T_net1_board_tv_2":
        squre_size = SQUARE_SIZE_TV_BIG
    elif T_name == "T_net1_board_tv_1" or T_name == "T_net1_board_tv_3":
        squre_size = SQUARE_SIZE_TV_SMALL
    else:
        raise ValueError("T_name not recognized")
    
    # Original pose (OpenCV outputs)
    rvec = rvec_obj_in_dst_cam
    tvec = tvec_obj_in_dst_cam

    # Convert to 3x3 rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Compose 4x4 pose matrix
    T_cam_obj = np.eye(4)
    T_cam_obj[:3, :3] = R
    T_cam_obj[:3, 3] = tvec.flatten()

    # Define object transformation: flip + shift origin
    R_flip = np.array([
        [-1, 0, 0],
        [ 0,-1, 0],
        [ 0, 0, 1]
    ])
    
    board_width = squre_size*(CHECKERBOARD_TV[0]-2)
    board_height = squre_size*(CHECKERBOARD_TV[1]-2)

    T_obj_transform = np.eye(4)
    T_obj_transform[:3, :3] = R_flip
    T_obj_transform[:3, 3] = [board_width, board_height, 0]

    # Apply the transformation: new object pose in camera
    T_cam_obj_new = T_cam_obj @ T_obj_transform

    # Extract new rvec and tvec
    R_new = T_cam_obj_new[:3, :3]
    t_new = T_cam_obj_new[:3, 3]

    rvec_obj_in_dst_cam, _ = cv2.Rodrigues(R_new)
    tvec_obj_in_dst_cam = t_new.reshape(3, 1)
    
    return rvec_obj_in_dst_cam, tvec_obj_in_dst_cam

def shift_checkerboard(objp, rows, cols, shift_mm=(0, 0)):
    # Shift first and last columns in X
    for r in range(rows):
        first_idx = r * cols
        last_idx = r * cols + (cols - 1)
        
        # First column → move +X (toward center)
        objp[first_idx, 0] += shift_mm[1]
        
        # Last column → move -X (toward center)
        objp[last_idx, 0] -= shift_mm[1]

    # Shift first and last rows in Y
    for c in range(cols):
        first_row_idx = c
        last_row_idx = (rows - 1) * cols + c
        
        # First row → move +Y (toward center)
        objp[first_row_idx, 1] += shift_mm[0]
        
        # Last row → move -Y (toward center)
        objp[last_row_idx, 1] -= shift_mm[0]
        
    return objp


def pipeline(path, objp_tvs, dst_camera_matrix, dst_dist_coeffs, rvec_obj_in_dst_cam, tvec_obj_in_dst_cam, freeze=0, verbose=False):
    path = str(path)
 
    # Create a window named
    winname = "Chessboard Corners"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, width=HD[0], height=HD[1])
    cv2.moveWindow(winname, x=FHD[0]//2-HD[0]//2, y=FHD[1]//4-HD[1]//4)

    # Load image
    img = cv2.imread(path)

    # From 3D to 2D
    proj_points = cv2.projectPoints(
        objp_tvs,
        rvec_obj_in_dst_cam,
        tvec_obj_in_dst_cam,
        dst_camera_matrix,
        dst_dist_coeffs,
    )[0].reshape(-1, 2)

    # Draw projected points
    for p in proj_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
    
    if verbose:
        cv2.drawFrameAxes(img, dst_camera_matrix, dst_dist_coeffs, rvec_obj_in_dst_cam, tvec_obj_in_dst_cam, 300, thickness=3)
        cv2.imshow(winname, img)
        cv2.waitKey(freeze)
        cv2.destroyAllWindows()


def main():
    folder = "boards_imgs"
    yaml_file = Path(f"{folder}/calibrated_cameras_data.yml")
    image_files = sorted(glob.glob(os.path.join(folder, "*.png")))

    cam_params = load_camera_calibration(yaml_file)
    cam_params["net_1"] = cam_params["camera_6"]
    cam_params["net_2"] = cam_params["camera_7"]
    
    for path in image_files:
        # Pose file
        path = Path(path)
        name = str(path.stem)
        
        pose_path = Path(f"{folder}/{name}.npz")
        if "tv_2" not in str(name):
            continue
        
        # Get camera parameters
        dst_cam_params = cam_params["net_1"] if "net_1" in str(path) else cam_params["net_2"]
        src_cam_params = cam_params["net_1"] if "net_2" in str(path) else cam_params["net_2"]
        
        dst_camera_matrix = dst_cam_params["camera_matrix"]
        dst_dist_coeffs =   dst_cam_params["dist_coeffs"]
        
        # # Solve PnP to get rotation and translation vectors       
        # if "net_1" in name:
        #     pose_path = f"{folder}/net_2_tv_2.npz"
        # else:
        #     pose_path = f"{folder}/net_1_tv_2.npz"
        
        # # Load pose
        # if os.path.exists(pose_path):
        #     board_pose = np.load(pose_path)
        #     board_rvec = board_pose["rvec"]
        #     board_tvec = board_pose["tvec"]
        #     board_pose.close()
        # else:
        #     board_rvec, board_tvec = None, None
        
        # rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = transform_object_pose_between_cameras(
        #     rvec_obj_in_src_cam=board_rvec,
        #     tvec_obj_in_src_cam=board_tvec,
        #     src_cam_extrinsic=src_cam_params["pose"],
        #     dst_cam_extrinsic=dst_cam_params["pose"],
        #     verbose=False,
        # )
        
        # # Calculate pose
        # pipeline(
        #     path,
        #     objp_tv_big,
        #     dst_camera_matrix,
        #     dst_dist_coeffs,
        #     rvec_obj_in_dst_cam,
        #     tvec_obj_in_dst_cam,
        #     freeze=0,
        #     verbose=True
        # )
        
        # if "net_2" in name:
        #     # Solve PnP to get rotation and translation vectors       
        #     pose_path = f"{folder}/net_1_tv_1.npz"
            
        #     # Load pose
        #     if os.path.exists(pose_path):
        #         board_pose = np.load(pose_path)
        #         board_rvec = board_pose["rvec"]
        #         board_tvec = board_pose["tvec"]
        #         board_pose.close()
        #     else:
        #         board_rvec, board_tvec = None, None
            
        #     rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = transform_object_pose_between_cameras(
        #         rvec_obj_in_src_cam=board_rvec,
        #         tvec_obj_in_src_cam=board_tvec,
        #         src_cam_extrinsic=src_cam_params["pose"],
        #         dst_cam_extrinsic=dst_cam_params["pose"],
        #         verbose=False,
        #     )
            
        #     # Calculate pose
        #     pipeline(
        #         path,
        #         objp_tv_small,
        #         dst_camera_matrix,
        #         dst_dist_coeffs,
        #         rvec_obj_in_dst_cam,
        #         tvec_obj_in_dst_cam,
        #         freeze=0,
        #         verbose=True
        #     )
        
        board_rvec, board_tvec = None, None
        if "net_2" in name:
            rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = transform_object_pose_between_cameras(
                rvec_obj_in_src_cam=board_rvec,
                tvec_obj_in_src_cam=board_tvec,
                src_cam_extrinsic=src_cam_params["pose"],
                dst_cam_extrinsic=dst_cam_params["pose"],
                verbose=False,
                T_name="T_net1_board_tv_1"
            )
            
            # TV1: widht=76 height=81    90  Fail
            width_shift = SQUARE_SIZE_TV_SMALL - 76
            height_shift = SQUARE_SIZE_TV_SMALL - 81
            shift_mm = (height_shift, width_shift)
            objp_tv_small_1 = objp_tv_small.copy()
            objp_tv_small_1 = shift_checkerboard(objp_tv_small_1, rows=CHECKERBOARD_TV[1], cols=CHECKERBOARD_TV[0], shift_mm=shift_mm)
            
            # Calculate pose
            pipeline(
                path,
                objp_tv_small_1,
                dst_camera_matrix,
                dst_dist_coeffs,
                rvec_obj_in_dst_cam,
                tvec_obj_in_dst_cam,
                freeze=0,
                verbose=True,
            )
            rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = transform_object_pose_between_cameras(
                rvec_obj_in_src_cam=board_rvec,
                tvec_obj_in_src_cam=board_tvec,
                src_cam_extrinsic=src_cam_params["pose"],
                dst_cam_extrinsic=dst_cam_params["pose"],
                verbose=False,
                T_name="T_net1_board_tv_2"
            )
                        
            # TV2: widht=89 height=95   105  Fail
            width_shift = SQUARE_SIZE_TV_BIG - 89
            height_shift = SQUARE_SIZE_TV_BIG - 95
            shift_mm = (height_shift, width_shift)
            objp_tv_big_2 = objp_tv_big.copy()
            objp_tv_big_2 = shift_checkerboard(objp_tv_big_2, rows=CHECKERBOARD_TV[1], cols=CHECKERBOARD_TV[0], shift_mm=shift_mm)
            
            # Calculate pose
            pipeline(
                path,
                objp_tv_big_2,
                dst_camera_matrix,
                dst_dist_coeffs,
                rvec_obj_in_dst_cam,
                tvec_obj_in_dst_cam,
                freeze=0,
                verbose=True,
            )
            rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = transform_object_pose_between_cameras(
                rvec_obj_in_src_cam=board_rvec,
                tvec_obj_in_src_cam=board_tvec,
                src_cam_extrinsic=src_cam_params["pose"],
                dst_cam_extrinsic=dst_cam_params["pose"],
                verbose=False,
                T_name="T_net1_board_tv_3"
            )
            
            # TV3: widht=90 height=90    90  OK
            objp_tv_small_3 = objp_tv_small.copy()
            # Calculate pose
            pipeline(
                path,
                objp_tv_small_3,
                dst_camera_matrix,
                dst_dist_coeffs,
                rvec_obj_in_dst_cam,
                tvec_obj_in_dst_cam,
                freeze=0,
                verbose=True,
            )

        # if "net_1" in name:
        #     pose_path = f"{folder}/net_2_tv_3.npz"
            
        #     # Load pose
        #     if os.path.exists(pose_path):
        #         board_pose = np.load(pose_path)
        #         board_rvec = board_pose["rvec"]
        #         board_tvec = board_pose["tvec"]
        #         board_pose.close()
        #     else:
        #         board_rvec, board_tvec = None, None
            
        #     rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = transform_object_pose_between_cameras(
        #         rvec_obj_in_src_cam=board_rvec,
        #         tvec_obj_in_src_cam=board_tvec,
        #         src_cam_extrinsic=src_cam_params["pose"],
        #         dst_cam_extrinsic=dst_cam_params["pose"],
        #         verbose=False,
        #     )
            
        #     # Calculate pose
        #     pipeline(
        #         path,
        #         objp_tv_small,
        #         dst_camera_matrix,
        #         dst_dist_coeffs,
        #         rvec_obj_in_dst_cam,
        #         tvec_obj_in_dst_cam,
        #         freeze=0,
        #         verbose=True
        #     )


if __name__ == "__main__":
    main()
