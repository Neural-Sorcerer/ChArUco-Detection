

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
