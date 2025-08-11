import cv2
import numpy as np


def load_tv_poses(tv_poses_path, verbose=False):
    with open(tv_poses_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    tv_poses = {}
    matrix_lines = []
    current_name = None

    for line in lines:
        if line.startswith("board_name:"):
            current_name = line.split(':', 1)[1].strip()
        elif (not line.startswith("board_name:")) and (not line.startswith("T_net1_board")) and (len(line.strip()) > 0):
            matrix_lines.append(line)

        # Save the previous matrix if any
        if current_name and len(matrix_lines) == 4:
            matrix = [list(map(float, row.split())) for row in matrix_lines]
            tv_poses[current_name] = np.array(matrix)
            matrix_lines = []
            current_name = None

    if verbose:
        # Optional: print to verify
        for name, mat in tv_poses.items():
            print(f"{name}:\n{mat}\n")
    return tv_poses


def rotate_checkboard_into_manoj(rvec_obj_in_dst_cam, tvec_obj_in_dst_cam, T_name):
    SQUARE_SIZE_TV_BIG = 105        # Real-world square size in mm
    SQUARE_SIZE_TV_SMALL = 90       # Real-world square size in mm
    CHECKERBOARD_TV = (17, 10)

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


def transform_object_pose_between_cameras(
    rvec_obj_in_src_cam: np.ndarray,
    tvec_obj_in_src_cam: np.ndarray,
    src_cam_extrinsic: np.ndarray,
    dst_cam_extrinsic: np.ndarray,
    verbose: bool = False,
    T_name = None,
    use_predefined = False,
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
    if (T_name is not None) and use_predefined:
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
