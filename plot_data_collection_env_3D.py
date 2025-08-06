from typing import List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


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


# MATPLOTLIB


def axis_equal_3d(ax: Axes) -> None:
    extents = np.array([getattr(ax, f"get_{dim}lim")() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, f"set_{dim}lim")(ctr - r, ctr + r)


def prepare_figure(ax: Axes) -> None:
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    axis_equal_3d(ax)
    ax.set_xlabel("x (mm)", fontsize=20)
    ax.set_ylabel("y (mm)", fontsize=20)
    ax.set_zlabel("z (mm)", fontsize=20)
    ax.set_xlim(-1500, 1500)
    ax.set_ylim(-1800, 1500)
    ax.set_zlim(-100, 2500)


def plot_camera(ax: Axes, coords: List[np.ndarray], cam_color: List[float], cam_edge: float) -> None:
    assert len(coords) == 16

    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16 = coords

    ax.plot(
        xs=[x1[0][0], x2[0][0]],
        ys=[x1[1][0], x2[1][0]],
        zs=[x1[2][0], x2[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x1[0][0], x3[0][0]],
        ys=[x1[1][0], x3[1][0]],
        zs=[x1[2][0], x3[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x4[0][0], x3[0][0]],
        ys=[x4[1][0], x3[1][0]],
        zs=[x4[2][0], x4[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x2[0][0], x4[0][0]],
        ys=[x2[1][0], x4[1][0]],
        zs=[x2[2][0], x4[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x5[0][0], x6[0][0]],
        ys=[x5[1][0], x6[1][0]],
        zs=[x5[2][0], x6[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x5[0][0], x7[0][0]],
        ys=[x5[1][0], x7[1][0]],
        zs=[x5[2][0], x7[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x8[0][0], x7[0][0]],
        ys=[x8[1][0], x7[1][0]],
        zs=[x8[2][0], x7[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x6[0][0], x8[0][0]],
        ys=[x6[1][0], x8[1][0]],
        zs=[x6[2][0], x8[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x1[0][0], x5[0][0]],
        ys=[x1[1][0], x5[1][0]],
        zs=[x1[2][0], x5[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x2[0][0], x6[0][0]],
        ys=[x2[1][0], x6[1][0]],
        zs=[x2[2][0], x6[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x3[0][0], x7[0][0]],
        ys=[x3[1][0], x7[1][0]],
        zs=[x3[2][0], x7[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x4[0][0], x8[0][0]],
        ys=[x4[1][0], x8[1][0]],
        zs=[x4[2][0], x8[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x9[0][0], x10[0][0]],
        ys=[x9[1][0], x10[1][0]],
        zs=[x9[2][0], x10[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x9[0][0], x11[0][0]],
        ys=[x9[1][0], x11[1][0]],
        zs=[x9[2][0], x11[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x12[0][0], x11[0][0]],
        ys=[x12[1][0], x11[1][0]],
        zs=[x12[2][0], x11[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x10[0][0], x12[0][0]],
        ys=[x10[1][0], x12[1][0]],
        zs=[x10[2][0], x12[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x5[0][0], x9[0][0]],
        ys=[x5[1][0], x9[1][0]],
        zs=[x5[2][0], x9[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x6[0][0], x10[0][0]],
        ys=[x6[1][0], x10[1][0]],
        zs=[x6[2][0], x10[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x7[0][0], x11[0][0]],
        ys=[x7[1][0], x11[1][0]],
        zs=[x7[2][0], x11[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x8[0][0], x12[0][0]],
        ys=[x8[1][0],  x8[1][0]],
        zs=[x8[2][0], x12[2][0]],
        color=(cam_color[0], cam_color[1], cam_color[2]),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x13[0][0], x14[0][0]],
        ys=[x13[1][0], x14[1][0]],
        zs=[x13[2][0], x14[2][0]],
        color=(1.0, 0.0, 0.0),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x13[0][0], x15[0][0]],
        ys=[x13[1][0], x15[1][0]],
        zs=[x13[2][0], x15[2][0]],
        color=(0.0, 1.0, 0.0),
        linewidth=cam_edge,
    )
    ax.plot(
        xs=[x13[0][0], x16[0][0]],
        ys=[x13[1][0], x16[1][0]],
        zs=[x13[2][0], x16[2][0]],
        color=(0.0, 0.0, 1.0),
        linewidth=cam_edge,
    )


def add_camera_to_subplot(
    ax: Axes,
    cam_rot: np.ndarray,
    cam_trans: np.ndarray,
    cam_size: float,
    cam_edge: float,
    cam_color: Optional[List[float]] = None,
) -> None:
    # CameraSize: half of the camera body length
    # CameraEdge: line width of the camera body
    # CameraColor: 3-vector RGB color of the camera

    if not cam_color:
        cam_color = [0.0, 0.0, 0.0]

    r = cam_size
    r1_5 = r * 1.5 
    r2_0 = r * 2 
    r3_0 = r * 3 

    # Corners of the camera in the camera coordinate system
    x1 = np.asarray([[ r, -r, -r2_0]]).T
    x2 = np.asarray([[ r,  r, -r2_0]]).T
    x3 = np.asarray([[-r, -r, -r2_0]]).T
    x4 = np.asarray([[-r,  r, -r2_0]]).T

    x5 = np.asarray([[ r, -r, r2_0]]).T
    x6 = np.asarray([[ r,  r, r2_0]]).T
    x7 = np.asarray([[-r, -r, r2_0]]).T
    x8 = np.asarray([[-r,  r, r2_0]]).T

    x9 =  np.asarray([[ r1_5, -r1_5, r3_0]]).T
    x10 = np.asarray([[ r1_5,  r1_5, r3_0]]).T
    x11 = np.asarray([[-r1_5, -r1_5, r3_0]]).T
    x12 = np.asarray([[-r1_5,  r1_5, r3_0]]).T

    x13 = np.asarray([[   0,    0,    0]]).T
    x14 = np.asarray([[   0,    0, r2_0]]).T
    x15 = np.asarray([[   0, r2_0,    0]]).T
    x16 = np.asarray([[r2_0,    0,    0]]).T

    # Corners of the camera in the world coordinate system
    x1 = np.matmul(cam_rot, x1) + cam_trans
    x2 = np.matmul(cam_rot, x2) + cam_trans
    x3 = np.matmul(cam_rot, x3) + cam_trans
    x4 = np.matmul(cam_rot, x4) + cam_trans
    x5 = np.matmul(cam_rot, x5) + cam_trans
    x6 = np.matmul(cam_rot, x6) + cam_trans
    x7 = np.matmul(cam_rot, x7) + cam_trans
    x8 = np.matmul(cam_rot, x8) + cam_trans
    x9 = np.matmul(cam_rot, x9) + cam_trans
    x10 = np.matmul(cam_rot, x10) + cam_trans
    x11 = np.matmul(cam_rot, x11) + cam_trans
    x12 = np.matmul(cam_rot, x12) + cam_trans
    x13 = np.matmul(cam_rot, x13) + cam_trans
    x14 = np.matmul(cam_rot, x14) + cam_trans
    x15 = np.matmul(cam_rot, x15) + cam_trans
    x16 = np.matmul(cam_rot, x16) + cam_trans

    coords = [x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16]
    plot_camera(ax, coords, cam_color, cam_edge)


def display_cameras_tvs_in_3D():
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_title("Calibration Result")
    ax.view_init(elev=-70, azim=-70)

    # Parse TV board poses (4x4) relative to camera_6 (net1)
    tv_poses = load_tv_poses("temp/TV_board_poses_optimized.txt", verbose=False)
    fs = cv2.FileStorage("temp/calibrated_cameras_data.yml", cv2.FILE_STORAGE_READ)
    num_cameras = int(fs.getNode("nb_camera").real())
    
    for cam_idx in range(num_cameras):
        cam_name = f"camera_{cam_idx}"
        cam_pose = fs.getNode(cam_name).getNode("camera_pose_matrix").mat()
        cam_trans = np.asarray([cam_pose[0:3, 3]]).T
        cam_rot = cam_pose[0:3, 0:3]
        
        # OMS Fisheye (origin)
        if cam_idx == 0:
            cam_color = [1.0, 0.0, 1.0]
            
        # OMS Fisheye
        if cam_idx in [1, 2, 3]:
            cam_color = [1.0, 0.0, 0.0] # red
        
        # # Anker Pinhole
        elif cam_idx in [4, 5]:
            cam_color = [0.0, 1.0, 0.0] # green
        
        # CCTV Pinhole
        elif cam_idx in [6, 7]:
            cam_color = [0.0, 0.0, 1.0] # blue
        
        # TFT – Helios (Depth)
        elif cam_idx in [8, 9]:
            cam_color = [0.0, 0.0, 0.0] # black
        
        add_camera_to_subplot(ax, cam_rot, cam_trans, cam_size=0.5*100, cam_edge=1.0, cam_color=cam_color)

    # TV poses
    for tv_name, tv_pose in tv_poses.items():
        tv_R = tv_pose[0:3, 0:3]
        tv_tvec = np.asarray([tv_pose[0:3, 3]]).T
        cam_color = [1.0, 1.0, 1.0]     # white
        # cam_color = [0.0, 0.5, 0.5]   # aqua
        
        board_rvec = cv2.Rodrigues(tv_R)[0]
        board_tvec = tv_tvec
        
        rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = transform_object_pose_between_cameras(
            rvec_obj_in_src_cam=board_rvec,
            tvec_obj_in_src_cam=board_tvec,
            src_cam_extrinsic=fs.getNode("camera_6").getNode("camera_pose_matrix").mat(),
            dst_cam_extrinsic=fs.getNode("camera_0").getNode("camera_pose_matrix").mat(),
            verbose=False,
            T_name=f"T_net1_board_{tv_name}"
        )
        
        tv_R = cv2.Rodrigues(rvec_obj_in_dst_cam)[0]
        tv_tvec = tvec_obj_in_dst_cam
            
        add_camera_to_subplot(ax, tv_R, tv_tvec, cam_size=0.5*300, cam_edge=3.0, cam_color=cam_color)
    
    prepare_figure(ax)
    plt.show(block=True)
    plt.close()


if __name__ == "__main__":
    display_cameras_tvs_in_3D()
