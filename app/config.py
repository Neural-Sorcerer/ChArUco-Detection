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

# Repo root = parent of this app/ package.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIGS_DIR / "default_config.yaml"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# qt-material theme applied on startup (see ui/styles.py for the full list).
DEFAULT_THEME = "dark_blue.xml"
