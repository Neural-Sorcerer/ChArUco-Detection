# Charuco Detection and Camera Calibration

A comprehensive toolkit for Charuco board detection, visualization, and camera calibration.

## Features

- **Charuco Board Detection**: Detect Charuco boards in images and videos
- **Visualization**: Visualize detected markers, corners, and 3D axes
- **Camera Calibration**: Calibrate cameras using Charuco boards (pinhole and fisheye models)
- **Board Generation**: Generate Charuco board images for printing with customizable parameters
- **3D Consistency Evaluation**: Evaluate 3D consistency of detected boards
- **Data Quality Assessment**: Intelligent filtering for diverse calibration samples
- **Position Heatmaps**: Visual coverage analysis for calibration data collection
- **Automatic Sample Filtering**: Remove redundant samples based on position, size, and skew
- **Flexible Input Handling**: Support for camera indices, video files, and image files
- **Object-Oriented Design**: Well-structured, maintainable code with proper typing and documentation

## TODO

- [ ] Make a heatmap for board position to see, where the board has already been shown in the camera
- [ ] Filter detected boards by: X, Y, size, skew, repetitive

## Create `charuco` conda env

```bash
conda create --name charuco python=3.11 -y
conda activate charuco
pip install opencv-contrib-python==4.11.0.86
pip install matplotlib PyYAML
```

## Project Structure

- `charuco_reader.py`: Main script for detecting and visualizing Charuco boards
- `calibrate_camera.py`: Script for camera calibration
- `src/charuco_detector.py`: Class for detecting and visualizing Charuco boards
- `src/calibration.py`: Class for camera calibration functionality
- `configs/config.py`: Configuration parameters using dataclasses
- `utils/util.py`: Utility functions for 3D transformations and file operations

## Usage

### Charuco Board Detection

To detect a Charuco board from a camera, video, or image:

```bash
python charuco_reader.py --index [camera_index|video_path|image_path]
```

### Save Frames

```bash
python charuco_reader.py \
    --index=0 \
    --save \
    --output-dir=outputs
```

### Show Charuco Corners

```bash
python charuco_reader.py \
    --index=0 \
    --draw-charuco-corners
```

### Project 3D Points

```bash
python charuco_reader.py \
    --index=0 \
    --camera-params=assets/intrinsics.xml \
    --project-points
```

### Camera Calibration

0. Generate a Charuco board image:

    ```bash
    python calibrate_camera.py generate \
        --output-file=path/to/output.png \
        --pixels-per-square=300 \
        --margin-percent=0.05
    ```

1. Collect calibration images:

    ```bash
    # Basic collection
    python calibrate_camera.py collect \
        --index=0 \
        --output-dir=calibration_images

    # With intelligent quality assessment and diversity filtering
    python calibrate_camera.py collect \
        --index=0 \
        --output-dir=calibration_images \
        --use-quality-judge \
        --target-samples=50 \
        --auto-save
    ```

2. Filter existing dataset for diversity (optional):

    ```bash
    python calibrate_camera.py filter \
        --input-dir=raw_images \
        --output-dir=filtered_images \
        --target-samples=50
    ```

3. Calibrate the camera:

    ```bash
    # Pinhole camera model
    python calibrate_camera.py calibrate \
        --input-dir=calibration_images \
        --output-file=calibration.xml

    # Fisheye camera model
    python calibrate_camera.py calibrate \
        --input-dir=calibration_images \
        --output-file=calibration.xml \
        --fisheye
    ```

4. Test the calibration by undistorting images:

    ```bash
    python calibrate_camera.py calibrate \
        --input-dir=calibration_images \
        --output-file=calibration.xml \
        --fisheye \
        --undistort \
        --balance=1.0
    ```

## Command-Line Arguments

### charuco_reader.py

- `--index`: Camera index, video file path, or image path
- `--output-dir`: Output directory for saved frames
- `--save`: Save frames when 's' key is pressed
- `--save-all`: Save all frames
- `--camera-params`: Path to camera calibration file
- `--draw-marker-corners`: Draw marker corners
- `--draw-charuco-corners`: Draw Charuco corners
- `--show-ids`: Show corner IDs
- `--project-points`: Project 3D points to image plane
- `--evaluate-3d`: Evaluate 3D consistency

### calibrate_camera.py

#### Common arguments

- `--board-id`: Charuco board ID
- `--x-squares`: Number of squares in X direction
- `--y-squares`: Number of squares in Y direction
- `--square-length`: Square length in meters
- `--marker-length`: Marker length in meters (default: 75% of square length)

#### Generate mode

```bash
python calibrate_camera.py generate [options]
```

- `--output-file`: Output file for board image
- `--pixels-per-square`: Pixels per square
- `--margin-percent`: Margin around the board as a percentage (0.05 = 5%) of the minimum grid dimension

#### Collect mode

```bash
python calibrate_camera.py collect [options]
```

- `--index`: Camera index or video file path
- `--output-dir`: Output directory for calibration images
- `--resolution`: Camera resolution (SS, SD, HD, FHD, UHD, OMS)
- `--use-quality-judge`: Enable intelligent quality assessment during collection
- `--target-samples`: Target number of diverse samples (default: 50)
- `--auto-save`: Automatically save good quality samples

#### Calibrate mode

```bash
python calibrate_camera.py calibrate [options]
```

- `--input-dir`: Input directory with calibration images
- `--pattern`: File pattern for calibration images
- `--output-file`: Output file for calibration parameters
- `--fisheye`: Assume fisheye camera model
- `--undistort`: Test calibration by undistorting images
- `--balance`: Balance value for undistortion (0.0 = crop, 1.0 = stretch)
- `--simple`: Use simple undistortion (no remapping)

#### Filter mode

```bash
python calibrate_camera.py filter [options]
```

- `--input-dir`: Input directory with images to filter
- `--output-dir`: Output directory for filtered images
- `--target-samples`: Target number of diverse samples (default: 50)

## Important Note on Board Generation

While this repository provides functionality to generate PNG images of Charuco boards, I recommend to generate vector-based formats (PDF) for printing:

- **For digital use or testing**: The PNG generation in this toolkit is perfectly suitable
- **For printing**: Use vector-based formats (PDF) for high-quality results

For professional printing or high-resolution applications, consider using:
[Calib.io Camera Calibration Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator)
