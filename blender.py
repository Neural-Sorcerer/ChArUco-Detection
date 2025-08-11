"""
Blender script to visualize 10 calibrated cameras and TV board poses relative to camera_7.

Requires:
- PyYAML installed in Blender's Python environment.
- Files:
    - calibrated_cameras_data.yml: camera extrinsic matrices fileciteturn1file2L3-L11
    - TV_board_poses_optimized.txt: TV board poses relative to camera_6 (net1) fileciteturn1file0L1-L6
"""
import bpy
import cv2
import mathutils
import numpy as np

# File paths
cam_yaml_path = 'temp/calibrated_cameras_data.yml'
tv_poses_path = 'temp/TV_board_poses_optimized.txt'


def list_to_matrix(data):
    # Utility: convert flat list of 16 values into a 4x4 mathutils.Matrix
    return mathutils.Matrix((
        (data[0][0], data[0][1], data[0][2], data[0][3]),
        (data[1][0], data[1][1], data[1][2], data[1][3]),
        (data[2][0], data[2][1], data[2][2], data[2][3]),
        (data[3][0], data[3][1], data[3][2], data[3][3]),
    ))

def load_camera_calibration(yaml_path):
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


# Load camera extrinsic data
cam_data = load_camera_calibration(cam_yaml_path)

# Create cameras and axes
for cam_name, info in cam_data.items():
    mat = list_to_matrix(info['pose'])
    cam = bpy.data.cameras.new(name=cam_name)
    cam.display_size = 0.5
    cam_obj = bpy.data.objects.new(cam_name, cam)
    cam_obj.matrix_world = mat
    bpy.context.collection.objects.link(cam_obj)
    # Show axes at camera
    bpy.ops.object.empty_add(type='ARROWS')
    empty = bpy.context.active_object
    empty.name = f'{cam_name}_axes'
    empty.matrix_world = mat

# Compute relative boards in camera_7 frame
mat6 = list_to_matrix(cam_data['camera_6']['pose'])
mat7 = list_to_matrix(cam_data['camera_7']['pose'])
inv7 = mat7.inverted()

# Parse TV board poses (4x4) relative to camera_6 (net1)
tv_poses = load_tv_poses(tv_poses_path, verbose=False)

for name, arr in tv_poses.items():
    tv_mat = list_to_matrix(arr)
    world_mat = mat6 @ tv_mat
    rel_mat = inv7 @ world_mat
    bpy.ops.mesh.primitive_plane_add(size=0.5)
    obj = bpy.context.active_object
    obj.name = name
    obj.matrix_world = rel_mat
    mat = bpy.data.materials.new(name=f'{name}_mat')
    mat.diffuse_color = (0.8, 0.2, 0.1, 1)
    obj.data.materials.append(mat)


# Set the active scene camera to camera_7
scene = bpy.context.scene
cam_name = 'camera_7'
if cam_name in bpy.data.objects:
    scene.camera = bpy.data.objects[cam_name]
    # Switch any 3D Viewport to camera view
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == 'VIEW_3D':
                region = next(r for r in area.regions if r.type == 'WINDOW')
                override = {'window': window, 'screen': window.screen, 'area': area, 'region': region, 'scene': scene}
                bpy.ops.view3d.view_camera(override)
                break
        else:
            continue
        break
    
print('Visualization complete: cameras, axes, and TV boards visible. Active camera set to camera_7; press Numpad 0 to view.')

