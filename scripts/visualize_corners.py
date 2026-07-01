import os
import cv2
import json
import argparse


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
            cv2.circle(image, (x, y), 8, (0, 0, 0), 1)    # Black outline
            
            # Draw ID
            cv2.putText(image, id_str, (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw info text
        cv2.putText(image, f"Image: {image_name} ({i+1}/{len(samples)})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"Points: {len(keypoints)}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

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
                        default="calibration_images/calibration_images_0/calibration_corners.json",
                        help="Path to calibration_corners.json")
    parser.add_argument("--images", type=str,
                        default="calibration_images/calibration_images_0",
                        help="Directory containing calibration images")
    args = parser.parse_args(argv)

    visualize_corners(args.json, args.images)


if __name__ == "__main__":
    main()
