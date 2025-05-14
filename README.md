# Charuco Detection and Camera Calibration

A comprehensive toolkit for Charuco board detection, visualization, and camera calibration.

## Features

- **Charuco Board Detection**: Detect Charuco boards in images and videos
- **Visualization**: Visualize detected markers, corners, and 3D axes
- **Camera Calibration**: Calibrate cameras using Charuco boards
- **Board Generation**: Generate Charuco board images for printing
- **3D Consistency Evaluation**: Evaluate 3D consistency of detected boards

## Requirements

- Python 3.6+
- OpenCV 4.7.0+ with contrib modules
- NumPy

## Project Structure

- `charuco_reader.py`: Main script for detecting and visualizing Charuco boards
- `charuco_detector.py`: Class for detecting and visualizing Charuco boards
- `calibration.py`: Camera calibration functionality
- `calibrate_camera.py`: Script for camera calibration
- `config.py`: Configuration parameters
- `utils/util.py`: Utility functions

## Usage

### Charuco Board Detection

To detect a Charuco board in an image:

```bash
python charuco_reader.py --index path/to/image.png
```

To detect a Charuco board from a camera:

```bash
python charuco_reader.py --index 0
```

Additional options:

```bash
python charuco_reader.py --help
```

### Generate Charuco Board

To generate a Charuco board image:

```bash
python charuco_reader.py --generate-board --board-output path/to/output.png
```

Or use the dedicated script:

```bash
python calibrate_camera.py generate --output-file path/to/output.png
```

### Camera Calibration

1. Collect calibration images:

```bash
python calibrate_camera.py collect --output-dir calibration_images
```

2. Calibrate the camera:

```bash
python calibrate_camera.py calibrate --input-dir calibration_images --output-file calibration.xml
```

3. Test the calibration:

```bash
python calibrate_camera.py calibrate --input-dir calibration_images --output-file calibration.xml --test
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
- `--generate-board`: Generate and save a Charuco board image
- `--board-output`: Path to save generated board
- `--pixels-per-square`: Pixels per square for generated board

### calibrate_camera.py

#### Collect mode

```bash
python calibrate_camera.py collect [options]
```

- `--camera-index`: Camera index
- `--output-dir`: Output directory for calibration images
- `--resolution`: Camera resolution (SS, SD, HD, FHD, UHD)

#### Calibrate mode

```bash
python calibrate_camera.py calibrate [options]
```

- `--input-dir`: Input directory with calibration images
- `--pattern`: File pattern for calibration images
- `--output-file`: Output file for calibration parameters
- `--test`: Test calibration by undistorting images

#### Generate mode

```bash
python calibrate_camera.py generate [options]
```

- `--output-file`: Output file for board image
- `--pixels-per-square`: Pixels per square
- `--margin`: Margin around the board in pixels

## Examples

### Basic Detection

```bash
python charuco_reader.py --index 0
```

### Save Frames

```bash
python charuco_reader.py --index 0 --save --output-dir outputs
```

### Show Corner IDs

```bash
python charuco_reader.py --index 0 --show-ids
```

### Project 3D Points

```bash
python charuco_reader.py --index 0 --camera-params calibration.xml --project-points
```

### Generate Board

```bash
python calibrate_camera.py generate --output-file charuco_board.png --pixels-per-square 200
```