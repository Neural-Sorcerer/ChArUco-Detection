import os
import cv2
import time
import numpy as np
import xml.etree.ElementTree as ET


def load_camera_params(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Extract intrinsic matrix values
    fx = float(root.findtext(".//fx"))
    fy = float(root.findtext(".//fy"))
    ppx = float(root.findtext(".//ppx"))
    ppy = float(root.findtext(".//ppy"))

    camera_matrix = np.array([
        [fx, 0,  ppx],
        [0,  fy, ppy],
        [0,  0,  1]
    ], dtype=np.float64)

    # Extract distortion coefficients
    distortion_coeffs = [
        float(root.findtext(".//coeff_0")),
        float(root.findtext(".//coeff_1")),
        float(root.findtext(".//coeff_2")),
        float(root.findtext(".//coeff_3")),
        float(root.findtext(".//coeff_4")),
    ]
    dist_coeffs = np.array(distortion_coeffs, dtype=np.float64)

    return camera_matrix, dist_coeffs


def save_frame(original, output_dir, frame_id, annotated=None):
    timestamp = time.time()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{format(frame_id, '06d')}_{timestamp}")
    
    cv2.imwrite(f"{output_path}.png", original, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    if annotated is not None and id(annotated) != id(original):
        cv2.imwrite(f"{output_path}_annotated.png", annotated, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    
    print(f"Saved image-{frame_id}")


def invert_transformation(R, t):
    """
    Invert a rigid transformation (R, t).
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        R_inv: Inverted rotation matrix
        t_inv: Inverted translation vector
    """
    # Transpose of rotation matrix is its inverse
    R_inv = R.T
    
    # Inverted translation is -R^T * t
    t_inv = -np.dot(R_inv, t)
    
    return R_inv, t_inv


def transform_points(points, R, t):
    """
    Transform 3D points using rotation and translation.
    
    Args:
        points: Nx3 array of 3D points
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        transformed_points: Nx3 array of transformed points
    """
    # Ensure t is a column vector
    t = t.reshape(3, 1)
    
    # Transform each point: R * point + t
    transformed_points = np.dot(points, R.T) + t.T
    
    return transformed_points


def project_points_to_image(points_3d, K, D, R, t):
    """
    Project 3D points to image plane.
    
    Args:
        points_3d: Nx3 array of 3D points
        K: 3x3 camera intrinsic matrix
        D: Distortion coefficients
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        
    Returns:
        points_2d: Nx2 array of projected 2D points
    """
    # Use OpenCV's projectPoints function
    points_2d, _ = cv2.projectPoints(
        points_3d.astype(np.float32),
        R.astype(np.float32),
        t.astype(np.float32),
        K.astype(np.float32),
        D.astype(np.float32)
    )
    
    return points_2d.reshape(-1, 2)


def project_points_to_image(objpoint, rvec, tvec, camera_matrix, dist_coeffs):
    """Project 3D points to image plane"""
    imgpoints_proj, _ = cv2.projectPoints(objpoint, rvec, tvec, camera_matrix, dist_coeffs)
    imgpoints_proj = imgpoints_proj.reshape(-1, 2)
    return imgpoints_proj


def verify_consistency_3Dobjpoints(objpoint, pose_matrix, CHECKERBOARD, SQUARE_SIZE):
    # Extract R and t
    R = pose_matrix[:3, :3] # Rotation matrix (3x3)
    t = pose_matrix[:3, 3]  # Translation vector (3,)

    # Invert the transformation to get camera 0 relative to camera 1
    R_inv, t_inv = invert_transformation(R, t)

    # Transform 3D points from camera 0 coordinate system to camera 1 coordinate system
    transform_objpoint = transform_points(objpoint, R_inv, t_inv)

    # Evaluate 3D consistency
    evaluate_checkerboard_3d(transform_objpoint.reshape(-1, 3), CHECKERBOARD, SQUARE_SIZE)


def evaluate_checkerboard_3d(points3d: np.ndarray, pattern_size: tuple[int, int], square_size: float) -> dict:
    """
    Evaluate 3D consistency of a detected checkerboard.

    Args:
      points3d     : (N,3) array of corner coords in camera space.
      pattern_size : (cols, rows) inner‐corner counts (e.g. (7,5)).
      square_size  : physical side length of one checker square.

    Returns:
      dict with keys:
        'planarity': {'mean_dist', 'max_dist'}  – distances (m) to best‐fit plane
        'spacing': {
            'row_mean', 'row_std',  # along each row
            'col_mean', 'col_std'   # along each column
        }
        'rmse_edge': float           – RMSE of |d_ij – square_size|
        'orthogonality': {'mean_deg','std_deg'}  – angle dev from 90°
        'diagonal': {'mean', 'std'}  – diag length dev from √2·square_size
    """
    # reshape into (rows, cols, 3)
    cols, rows = pattern_size
    assert (
        points3d.shape[0] == cols * rows
    ), f"Expected {cols*rows} points, got {points3d.shape[0]}"
    grid = points3d.reshape(rows, cols, 3)

    # 1) Planarity via PCA
    P = points3d - points3d.mean(axis=0)
    _, S, Vt = np.linalg.svd(P, full_matrices=False)
    normal = Vt[-1]                 # plane normal
    dists = np.abs(P.dot(normal))   # signed‐distances
    planarity = {"mean_dist": dists.mean(), "max_dist": dists.max()}

    # 2) Edge‐spacing along rows & cols
    row_ds = []
    col_ds = []
    for i in range(rows):
        for j in range(cols - 1):
            row_ds.append(np.linalg.norm(grid[i, j + 1] - grid[i, j]))
    for j in range(cols):
        for i in range(rows - 1):
            col_ds.append(np.linalg.norm(grid[i + 1, j] - grid[i, j]))
    row_ds = np.array(row_ds)
    col_ds = np.array(col_ds)
    spacing = {
        "row_mean": row_ds.mean(),
        "row_std": row_ds.std(),
        "col_mean": col_ds.mean(),
        "col_std": col_ds.std(),
    }

    # 3) RMSE of edge‐length error
    all_ds = np.concatenate([row_ds, col_ds])
    rmse_edge = np.sqrt(np.mean((all_ds - square_size) ** 2))

    # 4) Orthogonality: for each interior corner compute angle between
    #    its row‐edge and col‐edge vectors
    ang_errors = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            v1 = grid[i, j + 1] - grid[i, j]
            v2 = grid[i + 1, j] - grid[i, j]
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            ang_errors.append(abs(90 - ang))
    ang_errors = np.array(ang_errors)
    orthogonality = {"mean_deg": ang_errors.mean(), "std_deg": ang_errors.std()}

    # 5) Diagonal lengths: should be √2·square_size
    diag_errors = []
    target_diag = square_size * np.sqrt(2)
    for i in range(rows - 1):
        for j in range(cols - 1):
            d1 = np.linalg.norm(grid[i, j] - grid[i + 1, j + 1])
            d2 = np.linalg.norm(grid[i, j + 1] - grid[i + 1, j])
            diag_errors.extend([d1, d2])
    diag_errors = np.array(diag_errors) - target_diag
    diagonal = {"mean": diag_errors.mean(), "std": diag_errors.std()}

    result = {
        "planarity": planarity,
        "spacing": spacing,
        "rmse_edge": rmse_edge,
        "orthogonality": orthogonality,
        "diagonal": diagonal,
    }
    for key, value in result.items():
        print(f"{key}: {value}")

    return result
