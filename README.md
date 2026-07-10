# Charuco Studio — Detection & Camera Calibration

A toolkit for **ChArUco** board detection, camera calibration and
calibration-data collection. It ships two front-ends over the same core engine:

- **Charuco Studio** — a PySide6 desktop GUI (`python tool.py`)
- **CLI** — a single command-line entry point (`python run.py <tool> …`)

## Features

- **Desktop GUI (Charuco Studio)**: live camera view, board/geometry settings,
  data-collection controls and a real-time calibration-readiness dashboard.
- **Quality-guided collection**: keeps only sharp, diverse views (variance-of-Laplacian
  blur gate + coverage diversity), with a live coverage heatmap and a
  “ready to calibrate” readiness panel. The GUI reuses the exact same
  `DataQualityJudge` engine the CLI uses.
- **Charuco detection & visualization**: markers, corners, board pose axes and
  projected 3D grid points, on images, folders, videos or a live camera.
- **Camera calibration**: pinhole and fisheye models, with optional undistortion test.
- **Board generation**: printable ChArUco board images with configurable geometry.
- **CLI ⇄ GUI parity**: `python run.py calibrate collect --use-quality-judge`
  mirrors the GUI’s quality-guided collection.

## Install

```bash
conda create --name charuco python=3.11 -y
conda activate charuco
pip install -r requirements.txt
```

> Some CLI visualization tools additionally use `matplotlib`; install it with
> `pip install matplotlib` if you hit an import error.

## Project layout

Everything the tool needs lives under `tool/`; the repo root only holds the two
entry-point scripts and project metadata.

```
.
├── tool.py              # GUI entry point:  python tool.py
├── run.py               # CLI entry point:  python run.py <tool> …
├── requirements.txt
├── README.md
├── outputs/             # generated data (gitignored)
└── tool/                # all code & data, split by tool
    ├── gui/             # GUI tool (Charuco Studio)
    │   ├── app/         # app identity, paths and GUI constants
    │   ├── ui/          # PySide6 panels and the main window
    │   └── core/        # camera worker, frame processor, collectors, config model
    ├── cli/             # simple key-driven OpenCV tool
    │   └── scripts/     # CLI tools dispatched by run.py
    └── engine/          # shared engine used by both tools
        ├── src/         # detection & calibration (CharucoDetector, CameraCalibrator)
        ├── configs/     # engine config (config.py) + GUI defaults (default_config.yaml)
        ├── utils/       # data-quality judge, sample filtering, helpers
        └── assets/      # sample intrinsics + generated board images
```

The two tools are kept separate (`gui/` and `cli/`) and share only the engine in
`tool/engine/`. Each entry point adds its own tool folder plus `tool/engine/` to the
import path, so packages are imported by their short names (`from ui…`, `from core…`,
`from src…`) and the tools run from the repo root.

## GUI usage

```bash
python tool.py
```

Launches Charuco Studio. Pick a camera, set the board geometry, press **Start**,
and (in quality-guided mode) collect diverse sharp frames until the readiness
panel reports the set is ready. Load/save settings via the **File** menu; the
GUI starts from `tool/configs/default_config.yaml`.

## CLI usage

`run.py` is a thin dispatcher; everything after the tool name is forwarded to
that tool. Use `python run.py <tool> -h` for a tool’s own options.

```bash
python run.py detect    --index 0 --resolution HD
python run.py calibrate collect   --index 0 --use-quality-judge --auto-save
python run.py calibrate calibrate --input-dir outputs/… --fisheye --undistort
python run.py visualize --json … --images …
```

### Detection — `python run.py detect …`

```bash
# From a camera, video, or image (index = camera index | video path | image path)
python run.py detect --index 0

# Save frames and project 3D points using known intrinsics
python run.py detect --index 0 --save --output-dir outputs
python run.py detect --index 0 --camera-params tool/engine/assets/intrinsics.xml --project-points
```

- `--index`: camera index, video file path, or image path
- `--output-dir`: output directory for saved frames
- `--save` / `--save-all`: save on `s` keypress / save every frame
- `--camera-params`: path to a camera-intrinsics `.xml`
- `--draw-marker-corners` / `--draw-charuco-corners` / `--show-ids`
- `--project-points`: project the board’s 3D grid back onto the image
- `--evaluate-3d`: evaluate 3D consistency

### Calibration — `python run.py calibrate <mode> …`

Common board arguments: `--board-id`, `--x-squares`, `--y-squares`,
`--square-length` (m), `--marker-length` (m, default 75% of the square).

```bash
# 0) Generate a printable board image
python run.py calibrate generate --output-file board.png --pixels-per-square 300 --margin-percent 0.05

# 1) Collect calibration images (with quality/diversity gating)
python run.py calibrate collect --index 0 --output-dir calibration_images \
    --use-quality-judge --target-samples 50 --auto-save

# 2) (optional) Filter an existing dataset for diversity
python run.py calibrate filter --input-dir raw_images --output-dir filtered_images --target-samples 50

# 3) Calibrate (pinhole or --fisheye)
python run.py calibrate calibrate --input-dir calibration_images --output-file calibration.xml
python run.py calibrate calibrate --input-dir calibration_images --output-file calibration.xml --fisheye

# 4) Test by undistorting (--balance 0.0 crops, 1.0 stretches)
python run.py calibrate calibrate --input-dir calibration_images --output-file calibration.xml \
    --fisheye --undistort --balance 1.0
```

- **generate**: `--output-file`, `--pixels-per-square`, `--margin-percent`
- **collect**: `--index`, `--output-dir`, `--resolution` (SS, SD, HD, FHD, UHD, OMS),
  `--use-quality-judge`, `--target-samples`, `--auto-save`
- **calibrate**: `--input-dir`, `--pattern`, `--output-file`, `--fisheye`,
  `--undistort`, `--balance`, `--simple`
- **filter**: `--input-dir`, `--output-dir`, `--target-samples`

### Visualize — `python run.py visualize …`

```bash
python run.py visualize --json path/to/corners.json --images path/to/images
```

## Note on board generation

This toolkit can generate PNG board images, which are fine for digital use and
testing. For **printing**, prefer a vector format (PDF) for crisp results —
e.g. the [Calib.io Camera Calibration Pattern Generator](https://calib.io/pages/camera-calibration-pattern-generator).
