import cv2
import numpy as np

from utils.plot_3d import *
from utils.transform import *


# Load camera and TV poses
tv_poses = load_tv_poses("camera_tv_params/TV_board_poses_optimized.txt", verbose=False)
cam_poses = cv2.FileStorage("camera_tv_params/calibrated_cameras_data_5.yml", cv2.FILE_STORAGE_READ)


def display_cameras_tvs_in_3D():
    # Setup figure
    ax = setup_figure()[1]

    num_cameras = int(cam_poses.getNode("nb_camera").real())
    
    # Display cameras
    for cam_idx in range(num_cameras):
        # Get camera pose
        cam_pose = cam_poses.getNode(f"camera_{cam_idx}").getNode("camera_pose_matrix").mat()
        
        # Get rotation and translation
        cam_R = cam_pose[0:3, 0:3]
        cam_tvec = np.asarray([cam_pose[0:3, 3]]).T
        
        # OMS Fisheye (origin)
        if cam_idx == 0:
            cam_color = (1.0, 0.0, 0.0) # red
            cam_label = f"{cam_idx}: Fisheye OMS (origin)"
            
        # OMS Fisheye
        if cam_idx in [1, 2, 3]:
            cam_color = (0.95, 0.5, 0.07) # orange
            cam_label = f"{cam_idx}: Fisheye OMS"

        # Anker Pinhole
        elif cam_idx in [4, 5]:
            cam_color = (0.3, 0.8, 1.0) # light blue
            cam_label = f"{cam_idx}: Pinhole Anker"
        
        # CCTV Pinhole
        elif cam_idx in [6, 7]:
            cam_color = (0.0, 0.0, 1.0) # blue
            cam_label = f"{cam_idx}: Pinhole CCTV"
        
        # TFT – Helios (Depth)
        elif cam_idx in [8, 9]:
            cam_color = (0.0, 1.0, 0.0) # green
            cam_label = f"{cam_idx}: TFT - Helios (Depth)"
        
        # Add cameras
        add_camera_to_subplot(
            ax,
            R=cam_R,
            tvec=cam_tvec,
            cam_color=cam_color,
            label=cam_label
        )

    # Display TV poses
    for tv_name, tv_pose in tv_poses.items():
        # Get rotation and translation
        tv_R = tv_pose[0:3, 0:3]
        tv_tvec = np.asarray([tv_pose[0:3, 3]]).T
        
        # Transform TV pose to camera 0
        board_rvec = cv2.Rodrigues(tv_R)[0]
        board_tvec = tv_tvec
        cam_R_6 = cam_poses.getNode("camera_6").getNode("camera_pose_matrix").mat()
        cam_R_0 = cam_poses.getNode("camera_2").getNode("camera_pose_matrix").mat()
        
        rvec_obj_in_dst_cam, tvec_obj_in_dst_cam = transform_object_pose_between_cameras(
            rvec_obj_in_src_cam=board_rvec,
            tvec_obj_in_src_cam=board_tvec,
            src_cam_extrinsic=cam_R_6,
            dst_cam_extrinsic=cam_R_0,
            verbose=False,
            T_name=f"T_net1_board_{tv_name}"
        )
        
        # Update TV pose
        tv_R = cv2.Rodrigues(rvec_obj_in_dst_cam)[0]
        tv_tvec = tvec_obj_in_dst_cam
        
        # Only screen with partially displayed fields
        if tv_name == "tv_1":
            tv_label = "TV-1: Left"
            monitor_mm = (1412, 787)
            
        elif tv_name == "tv_2":
            tv_label = "TV-2: Middle"
            monitor_mm = (1648, 919)
            
        elif tv_name == "tv_3":
            tv_label = "TV-3: Right"
            monitor_mm = (1440, 810)
            
        # Add TVs
        add_tv_screen_to_subplot(
            ax,
            R=tv_R,
            tvec=tv_tvec,
            monitor_mm=monitor_mm,
            label=tv_label
        )

    # Show the figure
    show_figure(save=True)


if __name__ == "__main__":
    display_cameras_tvs_in_3D()
