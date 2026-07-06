"""
Shared UI Helpers
=================

Small reusable widget utilities used across the settings panels:

* :func:`block_wheel` stops the mouse wheel from silently changing spin boxes and
  combo boxes (values change only via keyboard or clicks).
* :func:`make_browse_button` builds an icon-only file/folder picker button using
  the current Qt style's standard icons (no image assets needed).
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import QEvent, QObject
from PySide6.QtWidgets import QPushButton, QStyle, QWidget

__all__ = ["block_wheel", "make_browse_button"]

log = logging.getLogger(__name__)


class _WheelGuard(QObject):
    """Event filter that swallows wheel events on the widgets it is installed on."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Consume wheel events so the underlying control keeps its value."""
        if event.type() == QEvent.Type.Wheel:
            return True
        return super().eventFilter(obj, event)


# One shared filter installed on every guarded control; kept module-level so it
# outlives the widgets (an event filter must stay alive to keep working).
_WHEEL_GUARD = _WheelGuard()


def block_wheel(*widgets: QWidget) -> None:
    """
    Stop the mouse wheel from changing the given controls.

    Args:
        *widgets: Spin boxes / combo boxes (or any widget) to guard
    """
    for widget in widgets:
        widget.installEventFilter(_WHEEL_GUARD)


def make_browse_button(parent: QWidget, tooltip: str, folder: bool = True) -> QPushButton:
    """
    Build a compact icon-only browse button using a standard Qt style icon.

    Args:
        parent: Widget whose style provides the standard icon
        tooltip: Hover tooltip describing what the picker opens
        folder: ``True`` for a folder-open icon, ``False`` for a file-open icon

    Returns:
        A ready-to-connect :class:`QPushButton`
    """
    pixmap = (
        QStyle.StandardPixmap.SP_DirOpenIcon if folder
        else QStyle.StandardPixmap.SP_FileDialogDetailedView
    )
    button = QPushButton()
    button.setIcon(parent.style().standardIcon(pixmap))
    button.setToolTip(tooltip)
    button.setFixedWidth(38)
    return button
