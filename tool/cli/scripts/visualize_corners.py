import os
import cv2
import json
import argparse


def _put_edged_text(image, text, org, font_scale, color, thickness, edge_color=(0, 0, 0)):
    """Draw text with a black outline so ids/labels stay readable over any background."""
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                edge_color, thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                color, thickness, cv2.LINE_AA)


def visualize_corners(json_path, image_dir):
    """
    Visualizes corners and IDs from a JSON file on the corresponding images.

    Args:
        json_path: Path to the JSON file containing corner information.
        image_dir: Directory containing the images.
    """
    if not os.path.exists(json_path):
        print(f"❌ Error: JSON file not found at {json_path}")
        return

    with open(json_path, 'r') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Error decoding JSON: {e}")
            return

    samples = data.get('samples', [])
    if not samples:
        print("⚠️ No samples found in JSON.")
        return

    print(f"Found {len(samples)} samples. Press any key to next image, 'q' to quit.")

    for i, sample in enumerate(samples):
        image_name = sample['image']
        keypoints = sample['keypoints']
        
        # Construct full image path
        image_path = os.path.join(image_dir, image_name)
        
        # Check if image exists
        if not os.path.exists(image_path):
            print(f"⚠️ Image not found: {image_path}")
            continue

        image = cv2.imread(image_path)
        
        if image is None:
            print(f"⚠️ Could not load image: {image_path}")
            continue

        # Draw corners and IDs
        for point in keypoints:
            # Point format is [x, y, id]
            if len(point) >= 3:
                x, y, id_val = point[0], point[1], point[2]
            else:
                x, y = point[0], point[1]
                id_val = "?"

            x, y = int(x), int(y)
            id_str = str(int(id_val)) if isinstance(id_val, (int, float)) else str(id_val)
            
            # Draw circle
            cv2.circle(image, (x, y), 8, (0, 255, 0), -1) # Green filled circle
            cv2.circle(image, (x, y), 8, (0, 0, 0), 2)    # Black outline

            # Draw ID (black-edged so it reads over any background)
            _put_edged_text(image, id_str, (x + 8, y - 8), 1.0, (0, 0, 255), 2)

        # Draw info text
        _put_edged_text(image, f"Image: {image_name} ({i+1}/{len(samples)})", (10, 30),
                        0.7, (255, 255, 255), 2)
        _put_edged_text(image, f"Points: {len(keypoints)}", (10, 60),
                        0.7, (255, 255, 255), 2)

        # Show image
        win_name = "Calibration Corners Visualization"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, image)
        
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break

    cv2.destroyAllWindows()

def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize calibration corners from JSON")
    parser.add_argument("--json", type=str,
                        default="outputs/calibration_images_0/calibration_corners.json",
                        help="Path to calibration_corners.json")
    parser.add_argument("--images", type=str,
                        default="outputs/calibration_images_0",
                        help="Directory containing calibration images")
    args = parser.parse_args(argv)

    visualize_corners(args.json, args.images)


if __name__ == "__main__":
    main()
