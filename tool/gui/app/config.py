"""
Application Metadata & Paths
============================

Static app identity and the well-known filesystem locations the GUI relies on.
Kept dependency-free so any layer can import it.
"""
from __future__ import annotations

# === Standard Libraries ===
from pathlib import Path

__all__ = [
    "APP_NAME",
    "ORG_NAME",
    "APP_VERSION",
    "PROJECT_ROOT",
    "CONFIGS_DIR",
    "DEFAULT_CONFIG_PATH",
    "OUTPUTS_DIR",
    "DEFAULT_THEME",
]

APP_NAME = "Charuco Studio"
ORG_NAME = "DeltaX"
APP_VERSION = "0.1.0"

# This module lives at tool/gui/app/config.py. Resolve well-known dirs from the tool/
# root: the shared engine's config sits under tool/engine/, generated data at the repo root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]        # tool/  (app -> gui -> tool)
CONFIGS_DIR = PROJECT_ROOT / "engine" / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default_config.yaml"
OUTPUTS_DIR = PROJECT_ROOT.parent / "outputs"             # generated data stays at the repo root

# qt-material theme applied on startup (see ui/styles.py for the full list).
DEFAULT_THEME = "dark_blue.xml"
