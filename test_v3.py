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

# Termination criteria for corner sub-pixel refinement
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
CALIB_CB_FLAG = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE

SQUARE_SIZE_TV_BIG = 105     # Real-world square size in mm
SQUARE_SIZE_TV_SMALL = 90     # Real-world square size in mm
CHECKERBOARD = (15, 8)
CHECKERBOARD_TV = (16, 9)

# Prepare object points (3D points in real-world space)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_TV_BIG     # Scale according to real square size

objp_tv_big = np.zeros((CHECKERBOARD_TV[0] * CHECKERBOARD_TV[1], 3), np.float32)
objp_tv_big[:, :2] = np.mgrid[0:CHECKERBOARD_TV[0], 0:CHECKERBOARD_TV[1]].T.reshape(-1, 2)
objp_tv_big *= SQUARE_SIZE_TV_BIG     # Scale according to real square size
objp_tv_big[:, :2] = objp_tv_big[:, :2] - (SQUARE_SIZE_TV_BIG * 0.5)     # translate

objp_tv_small = np.zeros((CHECKERBOARD_TV[0] * CHECKERBOARD_TV[1], 3), np.float32)
objp_tv_small[:, :2] = np.mgrid[0:CHECKERBOARD_TV[0], 0:CHECKERBOARD_TV[1]].T.reshape(-1, 2)
objp_tv_small *= SQUARE_SIZE_TV_SMALL     # Scale according to real square size
objp_tv_small[:, :2] = objp_tv_small[:, :2] - (SQUARE_SIZE_TV_SMALL * 0.5)     # translate


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
    verbose: bool = False
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

    if verbose:
        print("T_src_to_dst:\n", T_from_src_to_dst)
        print("T_obj_in_dst_cam:\n", T_obj_in_dst_cam)
        print("rvec_obj_in_dst_cam:\n", rvec_obj_in_dst_cam)
        print("tvec_obj_in_dst_cam:\n", tvec_obj_in_dst_cam)
        
    return rvec_obj_in_dst_cam, tvec_obj_in_dst_cam


def pipeline(path, camera_matrix, dist_coeffs, freeze=0, verbose=True, winname="Chessboard Corners"):
    path = str(path)
 
    # Load image
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, CALIB_CB_FLAG)

    if ret:
        # Refine corner locations
        objpoints = objp.astype(np.float32).reshape(-1, 1, 3)
        imgpoints = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), CRITERIA).reshape(-1, 1, 2)[::-1]

    # Solve PnP to get rotation and translation vectors
    rvec, tvec = None, None
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        objpoints,
        imgpoints,
        camera_matrix,
        dist_coeffs,
        rvec=rvec,
        tvec=tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=0.3,
        confidence=0.99,
    )   # Initial fit

    for _ in range(10):
        success, rvec, tvec = cv2.solvePnP(
            objpoints[inliers[:, 0]],
            imgpoints[inliers[:, 0]],
            camera_matrix,
            dist_coeffs,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
    )   # Second fit for higher accuracy

    # Convert rotation vector to rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    if objpoints.shape[0] != len(inliers):
        print(f"❌ Missing inliers: {objpoints.shape[0]-len(inliers)}")
    else:
        print("✅ All inliers")

    # From 3D to 2D
    proj_points = cv2.projectPoints(
        objp_tv_big,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )[0].reshape(-1, 2)

    # Draw projected points
    for p in proj_points:
        cv2.circle(img, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
    
    if verbose:
        # Draw and display corners
        cv2.drawChessboardCorners(img, CHECKERBOARD, imgpoints, ret)
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 300, thickness=3)
        cv2.imshow(winname, img)
        cv2.waitKey(freeze)
        cv2.destroyAllWindows()
    
    return rvec, tvec


def main():
    # Create a window named
    winname = "Chessboard Corners"
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname, width=HD[0], height=HD[1])
    cv2.moveWindow(winname, x=FHD[0]//2-HD[0]//2, y=FHD[1]//4-HD[1]//4)
    
    # Load images
    folder = "temp/boards_imgs"
    image_net_1 = f"{folder}/net_1_tv_2.png"
    image_net_2 = f"{folder}/net_2_tv_2.png"
    
    # Load camera calibration
    yaml_file = f"{folder}/calibrated_cameras_data.yml"

    # Get camera parameters
    cam_params = load_camera_calibration(yaml_file)
    cam_params["net_1"] = cam_params["camera_6"]
    cam_params["net_2"] = cam_params["camera_7"]
    
    K_net_1 = cam_params["net_1"]["camera_matrix"]
    D_net_1 = cam_params["net_1"]["dist_coeffs"]
    E_net_1 = cam_params["net_1"]["pose"]
    
    K_net_2 = cam_params["net_2"]["camera_matrix"]
    D_net_2 = cam_params["net_2"]["dist_coeffs"]
    E_net_2 = cam_params["net_2"]["pose"]
    
    # Board Load poses
    # board_pose_net_1 = np.load(f"{folder}/net_1_tv_2.npz")
    # board_rvec_net_1 = board_pose_net_1["rvec"]
    # board_tvec_net_1 = board_pose_net_1["tvec"]
    # board_pose_net_1.close()

    # board_pose_net_2 = np.load(f"{folder}/net_2_tv_2.npz")
    # board_rvec_net_2 = board_pose_net_2["rvec"]
    # board_tvec_net_2 = board_pose_net_2["tvec"]
    # board_pose_net_2.close()
    
    board_rvec_net_1, board_tvec_net_1 = pipeline(image_net_1, K_net_1, D_net_1, freeze=0, verbose=False)
    board_rvec_net_2, board_tvec_net_2 = pipeline(image_net_2, K_net_2, D_net_2, freeze=0, verbose=False)
    
    rvec_obj_in_dst_cam_net_1, tvec_obj_in_dst_cam_net_1 = transform_object_pose_between_cameras(
        rvec_obj_in_src_cam=board_rvec_net_2,
        tvec_obj_in_src_cam=board_tvec_net_2,
        src_cam_extrinsic=E_net_2,
        dst_cam_extrinsic=E_net_1,
        verbose=False,
    )
    
    rvec_obj_in_dst_cam_net_2, tvec_obj_in_dst_cam_net_2 = transform_object_pose_between_cameras(
        rvec_obj_in_src_cam=board_rvec_net_1,
        tvec_obj_in_src_cam=board_tvec_net_1,
        src_cam_extrinsic=E_net_1,
        dst_cam_extrinsic=E_net_2,
        verbose=False,
    )

    # From 3D to 2D
    proj_points = cv2.projectPoints(
        objp_tv_big,
        rvec_obj_in_dst_cam_net_1,
        tvec_obj_in_dst_cam_net_1,
        K_net_1,
        D_net_1,
    )[0].reshape(-1, 2)

    # Load image
    img_net_1 = cv2.imread(image_net_1)

    # Draw projected points
    for p in proj_points:
        cv2.circle(img_net_1, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
    
    cv2.drawFrameAxes(
        img_net_1,
        K_net_1,
        D_net_1,
        rvec_obj_in_dst_cam_net_1,
        tvec_obj_in_dst_cam_net_1,
        length=300,
        thickness=3
    )
    cv2.imshow(winname, img_net_1)
    cv2.waitKey(0)
    
    # ======================================================================    
    # From 3D to 2D
    proj_points = cv2.projectPoints(
        objp_tv_big,
        rvec_obj_in_dst_cam_net_2,
        tvec_obj_in_dst_cam_net_2,
        K_net_2,
        D_net_2,
    )[0].reshape(-1, 2)

    # Load image
    img_net_2 = cv2.imread(image_net_2)
    
    # Draw projected points
    for p in proj_points:
        cv2.circle(img_net_2, (int(p[0]), int(p[1])), 5, (0, 0, 255), -1)
    
    cv2.drawFrameAxes(
        img_net_2,
        K_net_2,
        D_net_2,
        rvec_obj_in_dst_cam_net_2,
        tvec_obj_in_dst_cam_net_2,
        length=300,
        thickness=3
    )
    cv2.imshow(winname, img_net_2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
