"""
Charuco Studio - GUI Entry Point
================================

Launches the PySide6 desktop tool for camera calibration, data collection and
live Charuco visualization. The command-line tools remain available via
``run.py``; this is the graphical front-end over the same core logic.

    python tool.py
"""
from __future__ import annotations

# === Standard Libraries ===
import os
import sys
import logging
from pathlib import Path

# The GUI tool lives in tool/gui and shares the engine in tool/engine; put both on the
# import path so ``from ui`` / ``from core`` and ``from src`` resolve from the repo root.
_TOOL_DIR = Path(__file__).resolve().parent / "tool"
sys.path.insert(0, str(_TOOL_DIR / "engine"))
sys.path.insert(0, str(_TOOL_DIR / "gui"))

# === Third-Party Libraries ===
import PySide6
from PySide6.QtWidgets import QApplication

# === Local ===
from app.config import APP_NAME, ORG_NAME, DEFAULT_THEME
from ui.styles import apply_theme
from ui.main_window import MainWindow


def _use_pyside6_qt_plugins() -> None:
    """
    Force Qt to load PySide6's own platform plugins.

    OpenCV bundles its own Qt build and repoints ``QT_QPA_PLATFORM_PLUGIN_PATH``
    at ``cv2/qt/plugins`` when imported. Loading that plugin into a PySide6
    process mixes two incompatible Qt builds and aborts at startup, so we
    override the path back to the plugins that ship with PySide6.
    """
    plugins_dir = Path(PySide6.__file__).parent / "Qt" / "plugins"
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins_dir)


def main() -> None:
    """Configure logging, build the app, apply the theme and run the loop."""
    _use_pyside6_qt_plugins()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(ORG_NAME)
    apply_theme(app, theme=DEFAULT_THEME)

    window = MainWindow()
    window.show()

    print(f"✅ {APP_NAME} started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
