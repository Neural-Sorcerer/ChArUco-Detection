import cv2
import numpy as np

from utils.plot_3d_plotly import PlotlyVisualizer3D
from utils.transform import *


# Load camera and TV poses
tv_poses = load_tv_poses("camera_tv_params/TV_board_poses_optimized_v2.txt", verbose=False)
cam_poses = cv2.FileStorage("camera_tv_params/calibrated_cameras_data_v2.yml", cv2.FILE_STORAGE_READ)


def display_cameras_tvs_in_3D_plotly():
    """
    Display cameras and TV screens in an interactive 3D visualization using Plotly.
    Provides better interactivity and visual quality compared to matplotlib.
    """
    # Create the Plotly visualizer
    viz = PlotlyVisualizer3D(title="Interactive 3D Camera-TV Calibration Visualization")
    
    num_cameras = int(cam_poses.getNode("nb_camera").real())
    
    # Display cameras
    for cam_idx in range(num_cameras):
        # Get camera pose
        cam_pose = cam_poses.getNode(f"camera_{cam_idx}").getNode("camera_pose_matrix").mat()
        
        # Get rotation and translation
        cam_R = cam_pose[0:3, 0:3]
        cam_tvec = np.asarray([cam_pose[0:3, 3]]).T
        
        # Define camera colors and labels
        if cam_idx == 0:
            cam_color = 'red'
            cam_label = f"{cam_idx}: Fisheye OMS (origin)"
            
        elif cam_idx in [1, 2, 3]:
            cam_color = 'orange'
            cam_label = f"{cam_idx}: Fisheye OMS"

        elif cam_idx in [4, 5]:
            cam_color = 'lightblue'
            cam_label = f"{cam_idx}: Pinhole RealSense"
        
        elif cam_idx in [6, 7]:
            cam_color = 'blue'
            cam_label = f"{cam_idx}: Pinhole CCTV"
        
        elif cam_idx == 8:
            cam_color = 'yellow'
            cam_label = f"{cam_idx}: Pinhole Anker"
        
        elif cam_idx == 9:
            cam_color = 'green'
            cam_label = f"{cam_idx}: TFT - Helios (Depth)"
        
        else:
            cam_color = 'gray'
            cam_label = f"{cam_idx}: Unknown Camera"
        
        # Add camera to visualization
        viz.add_camera(
            R=cam_R,
            tvec=cam_tvec,
            cam_color=cam_color,
            label=cam_label,
            cam_size=80
        )

    # Display TV poses
    for tv_name, tv_pose in tv_poses.items():
        # Get rotation and translation
        tv_R = tv_pose[0:3, 0:3]
        tv_tvec = np.asarray([tv_pose[0:3, 3]]).T
        
        # Transform TV pose to camera 0 coordinate system
        board_rvec = cv2.Rodrigues(tv_R)[0]
        board_tvec = tv_tvec
        cam_R_6 = cam_poses.getNode("camera_6").getNode("camera_pose_matrix").mat()
        cam_R_0 = cam_poses.getNode("camera_0").getNode("camera_pose_matrix").mat()
        
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
        
        # Define TV specifications
        if tv_name == "tv_1":
            tv_label = "TV-1: Left"
            monitor_mm = (1412, 787)
            screen_color = 'rgba(255, 200, 200, 0.3)'  # Light red
            
        elif tv_name == "tv_2":
            tv_label = "TV-2: Middle"
            monitor_mm = (1648, 919)
            screen_color = 'rgba(200, 255, 200, 0.3)'  # Light green
            
        elif tv_name == "tv_3":
            tv_label = "TV-3: Right"
            monitor_mm = (1440, 810)
            screen_color = 'rgba(200, 200, 255, 0.3)'  # Light blue
            
        else:
            tv_label = f"TV: {tv_name}"
            monitor_mm = (1500, 800)  # Default size
            screen_color = 'rgba(200, 200, 200, 0.3)'  # Light gray
        
        # Add TV screen to visualization
        viz.add_tv_screen(
            R=tv_R,
            tvec=tv_tvec,
            monitor_mm=monitor_mm,
            label=tv_label,
            screen_color=screen_color
        )

    # Show the interactive visualization
    viz.show(save_html=True, filename="interactive_calibration_result.html")
    
    # Optionally save as static image
    viz.save_image("calibration_result_plotly.png")
    
    print("\n" + "="*60)
    print("🎉 Interactive 3D Visualization Complete!")
    print("="*60)
    print("✅ Interactive HTML saved to: outputs/interactive_calibration_result.html")
    print("✅ Static PNG saved to: outputs/calibration_result_plotly.png")
    print("\n📋 Features of the Plotly visualization:")
    print("   • Interactive rotation, zoom, and pan")
    print("   • Hover information for cameras and screens")
    print("   • Better visual quality with WebGL rendering")
    print("   • Professional appearance with proper lighting")
    print("   • Export capabilities (HTML, PNG, SVG, PDF)")
    print("   • Responsive design that works in web browsers")
    print("\n🔧 Controls:")
    print("   • Left click + drag: Rotate view")
    print("   • Right click + drag: Pan view")
    print("   • Scroll wheel: Zoom in/out")
    print("   • Double click: Reset view")
    print("   • Click legend items: Show/hide elements")
    print("="*60)


def compare_visualizations():
    """
    Run both matplotlib and Plotly visualizations for comparison.
    """
    print("Running matplotlib visualization...")
    from plot_data_collection_env_3D import display_cameras_tvs_in_3D
    display_cameras_tvs_in_3D()
    
    print("\nRunning Plotly visualization...")
    display_cameras_tvs_in_3D_plotly()


if __name__ == "__main__":
    # You can choose which visualization to run:
    
    # Option 1: Run only the new Plotly visualization
    display_cameras_tvs_in_3D_plotly()
    
    # Option 2: Run both for comparison (uncomment the line below)
    compare_visualizations()
