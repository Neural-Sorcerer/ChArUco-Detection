"""
Theme & Styling
===============

Applies the qt-material dark theme and a few small tweaks so the window reads as
an internal product rather than a default Qt app.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from qt_material import apply_stylesheet
from PySide6.QtWidgets import QApplication

__all__ = ["apply_theme"]

log = logging.getLogger(__name__)

# Extra qss layered on top of qt-material: tighter group boxes with titled borders
# and, crucially, readable button text on hover/checked (the stock qt-material
# hover tints the background but can leave the label low-contrast).
_EXTRA_QSS = """
QGroupBox {
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 6px;
    margin-top: 14px;
    padding: 12px 6px 6px 6px;
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 4px;
}
QLabel#StatusValue {
    font-weight: bold;
}
QPushButton {
    padding: 5px 12px;
    border-radius: 5px;
}
QPushButton:hover {
    color: #ffffff;
    background-color: rgba(68, 138, 255, 0.28);
    border: 1px solid rgba(68, 138, 255, 0.75);
}
QPushButton:pressed {
    color: #ffffff;
    background-color: rgba(68, 138, 255, 0.45);
}
QPushButton:checked {
    color: #ffffff;
    background-color: rgba(68, 138, 255, 0.40);
}
QPushButton:disabled {
    color: rgba(255, 255, 255, 0.35);
}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border: 1px solid rgba(68, 138, 255, 0.75);
}
QMenu::item:selected, QMenuBar::item:selected {
    color: #ffffff;
}
QProgressBar {
    border: none;
    border-radius: 5px;
    background-color: rgba(255, 255, 255, 0.10);
}
QProgressBar::chunk {
    border-radius: 5px;
}
"""


def apply_theme(app: QApplication, theme: str = "dark_blue.xml") -> None:
    """
    Apply the qt-material theme plus local tweaks to the application.

    Args:
        app: The running :class:`QApplication`
        theme: A qt-material theme filename, e.g. ``"dark_teal.xml"``
    """
    apply_stylesheet(app, theme=theme)
    app.setStyleSheet(app.styleSheet() + _EXTRA_QSS)
    log.info("Applied theme '%s'", theme)
