'''
###########################################################################
Charuco Camera Calibration - single entry point
###########################################################################

Thin dispatcher over the tools in ``scripts/``. Each tool keeps its own CLI;
run.py only selects which one to run and forwards the remaining arguments.

    python run.py detect    --index 0 --resolution HD
    python run.py detect    --index data/ti_camera_input --save
    python run.py calibrate collect   --index 0 --use-quality-judge --auto-save
    python run.py calibrate calibrate --input-dir outputs/... --fisheye --undistort
    python run.py visualize --json ... --images ...

Use ``python run.py <tool> -h`` to see a tool's own options.
'''
import sys
import logging
import argparse
from time import time

from scripts import charuco_reader
from scripts import calibrate_camera
from scripts import visualize_corners


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("run")


class CharucoCalibrationPipeline:
    """Route a tool name to the matching script's ``main(argv)``."""

    def detect(self, argv):
        """Charuco detection on an image, an image folder, a video, or a camera."""
        charuco_reader.main(argv)

    def calibrate(self, argv):
        """Camera calibration: generate / collect / calibrate / filter."""
        calibrate_camera.main(argv)

    def visualize(self, argv):
        """Visualize saved calibration corners from a JSON file."""
        visualize_corners.main(argv)


def main():
    """Select a tool and forward the remaining CLI arguments to it."""
    pipeline = CharucoCalibrationPipeline()
    tools = {
        "detect":    pipeline.detect,
        "calibrate": pipeline.calibrate,
        "visualize": pipeline.visualize,
    }

    parser = argparse.ArgumentParser(
        description="Charuco Camera Calibration - single entry point.",
        epilog="Everything after <tool> is forwarded to that tool; "
               "try `python run.py <tool> -h` for its own options.",
    )
    parser.add_argument("tool", choices=list(tools), help="Which tool to run.")

    # Parse only the tool name; forward the rest verbatim to the chosen tool so
    # its own argparse (including sub-commands) sees exactly what it expects.
    args = parser.parse_args(sys.argv[1:2])
    tools[args.tool](sys.argv[2:])


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()

    elapsed_seconds = int(end_time - start_time)
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60

    logger.info(f"Total execution time: {hours}h {minutes}m")
