"""Utility functions for Charuco detection and camera calibration.

This module provides utility functions for loading camera parameters,
saving frames, transforming 3D points, and evaluating 3D consistency.
"""
# === Standard Libraries ===
import os
import time
import logging
from typing import *
import xml.etree.ElementTree as ET

# === Third-Party Libraries ===
import cv2
import numpy as np


def load_camera_params(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera intrinsic parameters from an XML file.

    Args:
        file_path: Path to the XML file containing camera parameters.

    Returns:
        Tuple containing:
            - camera_matrix: 3x3 camera intrinsic matrix
            - dist_coeffs: Distortion coefficients (k1, k2, p1, p2, k3)

    Raises:
        FileNotFoundError: If the file does not exist
        ET.ParseError: If the XML file is malformed
        ValueError: If required parameters are missing
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Camera parameter file not found: {file_path}")

    try:
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

    except ET.ParseError:
        raise ET.ParseError(f"Error parsing XML file: {file_path}")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Error extracting camera parameters: {str(e)}")


def save_frame(
    original: np.ndarray,
    output_dir: str,
    frame_id: int,
    annotated: Optional[np.ndarray] = None,
    compression: int = 3
) -> str:
    """Save original and optionally annotated frames to disk.

    Args:
        original: Original frame to save
        output_dir: Directory to save frames to
        frame_id: Frame identifier (used in filename)
        annotated: Optional annotated frame to save
        compression: PNG compression level (0-9)

    Returns:
        Base path of the saved files (without extension)

    Raises:
        OSError: If directory creation fails
    """
    timestamp = time.time()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{format(frame_id, '06d')}_{timestamp}")

    cv2.imwrite(f"{output_path}.png", original, [cv2.IMWRITE_PNG_COMPRESSION, compression])

    if annotated is not None and id(annotated) != id(original):
        cv2.imwrite(f"{output_path}_annotated.png", annotated, [cv2.IMWRITE_PNG_COMPRESSION, compression])

    logging.info(f"Saved image-{frame_id} to {output_path}")
    return output_path


def invert_transformation(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a rigid transformation (R, t).

    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector

    Returns:
        Tuple containing:
            - R_inv: Inverted rotation matrix
            - t_inv: Inverted translation vector
    """
    # Transpose of rotation matrix is its inverse
    R_inv = R.T

    # Inverted translation is -R^T * t
    t_inv = -np.dot(R_inv, t)

    return R_inv, t_inv


def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Transform 3D points using rotation and translation.

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


def evaluate_checkerboard_3d(
    points3d: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size: float
) -> Dict[str, Any]:
    """Evaluate 3D consistency of a detected checkerboard.

    Args:
        points3d: (N,3) array of corner coords in camera space
        pattern_size: (cols, rows) inner-corner counts
        square_size: Physical side length of one checker square in meters

    Returns:
        Dictionary with keys:
            'planarity': {'mean_dist', 'max_dist'} - distances (m) to best-fit plane
            'spacing': {
                'row_mean', 'row_std',  # along each row
                'col_mean', 'col_std'   # along each column
            }
            'rmse_edge': float - RMSE of |d_ij - square_size|
            'orthogonality': {'mean_deg','std_deg'} - angle dev from 90°
            'diagonal': {'mean', 'std'} - diag length dev from √2·square_size
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
    dists = np.abs(P.dot(normal))   # signed-distances
    planarity = {"mean_dist": dists.mean(), "max_dist": dists.max()}

    # 2) Edge-spacing along rows & cols
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

    # 3) RMSE of edge-length error
    all_ds = np.concatenate([row_ds, col_ds])
    rmse_edge = np.sqrt(np.mean((all_ds - square_size) ** 2))

    # 4) Orthogonality: for each interior corner compute angle between
    #    its row-edge and col-edge vectors
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
    
    def log_nested_result(result: dict, indent: int = 0):
        tab = "    " * indent
        for key, value in result.items():
            if isinstance(value, dict):
                logging.info(f"{tab}{key}:")
                log_nested_result(value, indent + 1)
            else:
                try:
                    # Convert numpy float to Python float and format in scientific notation
                    if hasattr(value, 'item'):
                        value = value.item()
                    formatted = f"{value:.0E}"
                except Exception:
                    formatted = str(value)
                logging.info(f"{tab}{key}:\t{formatted}")

    # Log the results
    log_nested_result(result)

    return result


def verify_consistency_3Dobjpoints(
    objpoint: np.ndarray,
    pose_matrix: np.ndarray,
    pattern_size: Tuple[int, int],
    square_size: float
) -> Dict[str, Any]:
    """Verify 3D consistency of object points across different camera views.

    Args:
        objpoint: Nx3 array of 3D points in camera 0 coordinate system
        pose_matrix: 4x4 transformation matrix from camera 0 to camera 1
        pattern_size: (cols, rows) inner-corner counts
        square_size: Physical side length of one checker square in meters

    Returns:
        Dictionary with evaluation metrics
    """
    # Extract R and t
    R = pose_matrix[:3, :3]  # Rotation matrix (3x3)
    t = pose_matrix[:3, 3]   # Translation vector (3,)

    # Invert the transformation to get camera 0 relative to camera 1
    R_inv, t_inv = invert_transformation(R, t)

    # Transform 3D points from camera 0 coordinate system to camera 1 coordinate system
    transform_objpoint = transform_points(objpoint.reshape(-1, 3), R_inv, t_inv)

    # Evaluate 3D consistency
    return evaluate_checkerboard_3d(transform_objpoint, pattern_size, square_size)
