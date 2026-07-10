"""
Shared UI Helpers
=================

Small reusable widget utilities used across the settings panels:

* :func:`block_wheel` stops the mouse wheel from silently changing spin boxes and
  combo boxes (values change only via keyboard or clicks).
* :func:`make_browse_button` builds an icon-only file/folder picker button using
  the current Qt style's standard icons (no image assets needed).
* :func:`make_check_row` pairs a checkbox with a *selectable* caption, since a
  native ``QCheckBox`` label cannot be selected or copied.
* :class:`TitledGroupBox` is a group box whose title sits on the top border like
  the native one but is a real, copyable label.
* :class:`TrimmedDoubleSpinBox` shows values without trailing zeros (``0.03``, not
  ``0.0300``) while keeping full editing precision.
"""
from __future__ import annotations

# === Standard Libraries ===
import logging

# === Third-Party Libraries ===
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QMouseEvent, QResizeEvent
from PySide6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QStyle, QWidget,
)

__all__ = ["block_wheel", "make_browse_button", "make_check_row", "TitledGroupBox", "TrimmedDoubleSpinBox"]

log = logging.getLogger(__name__)


class TrimmedDoubleSpinBox(QDoubleSpinBox):
    """A double spin box that displays values without trailing zeros (0.03, not 0.0300)."""

    def textFromValue(self, value: float) -> str:
        """Format ``value`` with up to ``decimals()`` places, dropping trailing zeros."""
        text = f"{value:.{self.decimals()}f}".rstrip("0").rstrip(".")
        return text or "0"


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
    button.setObjectName("BrowseButton")            # neutral edges (no blue ring) via styles.py
    button.setIcon(parent.style().standardIcon(pixmap))
    button.setToolTip(tooltip)
    button.setFixedWidth(38)
    button.setFocusPolicy(Qt.FocusPolicy.NoFocus)   # no blue keyboard-focus rectangle
    return button


class _ClickableLabel(QLabel):
    """
    A selectable caption that toggles its partner checkbox on a plain click.

    The text stays selectable (drag to select, Ctrl+C to copy) - which a bare
    :class:`QCheckBox` label is not - while a click that is not a drag still toggles
    the box, preserving the usual "click the caption" checkbox behaviour.
    """

    def __init__(self, text: str, target: QCheckBox) -> None:
        super().__init__(text)
        self._target = target
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Toggle the partner box on a plain left click (never on a text-selecting drag)."""
        super().mouseReleaseEvent(event)
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._target.isEnabled()
            and not self.selectedText()
        ):
            self._target.toggle()


def make_check_row(text: str, tooltip: str = "") -> tuple[QWidget, QCheckBox]:
    """
    Build a checkbox whose caption is a separate, selectable label.

    Qt does not let a :class:`QCheckBox`'s own text be selected, so the box is created
    without text and paired with a copyable label that still toggles it on click.

    Args:
        text: Caption shown next to the checkbox
        tooltip: Optional hover tooltip applied to both the box and the caption

    Returns:
        ``(row, checkbox)`` - add ``row`` to a layout and wire ``checkbox``
    """
    checkbox = QCheckBox()
    caption = _ClickableLabel(text, checkbox)
    if tooltip:
        checkbox.setToolTip(tooltip)
        caption.setToolTip(tooltip)

    row = QWidget()
    box = QHBoxLayout(row)
    box.setContentsMargins(0, 0, 0, 0)
    box.setSpacing(6)
    box.addWidget(checkbox)
    box.addWidget(caption)
    box.addStretch(1)
    return row, checkbox


class TitledGroupBox(QGroupBox):
    """
    A group box whose title sits on the top border *and* can be selected/copied.

    Qt paints the native ``QGroupBox`` title itself, so its text can never be
    selected. This keeps the classic titled-border look by overlaying a real,
    copyable :class:`QLabel` (styled ``#PanelHeader``) centred on the top border
    line; its background matches the box so it "breaks" the border like the native
    title does.
    """

    _TITLE_LEFT = 8    # inset from the left edge, like the native title
    _TITLE_TOP = 10    # title/border centre line; must match QGroupBox margin-top in styles.py

    def __init__(self, title: str) -> None:
        super().__init__()   # leave the native (non-selectable) title empty
        self._title = QLabel(title, self)
        self._title.setObjectName("PanelHeader")
        self._title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._position_title()

    def _position_title(self) -> None:
        """Centre the title label on the top border line, inset from the left."""
        self._title.adjustSize()
        y = max(0, self._TITLE_TOP - self._title.height() // 2)
        self._title.move(self._TITLE_LEFT, y)
        self._title.raise_()

    def resizeEvent(self, event: QResizeEvent) -> None:  # noqa: N802 (Qt override)
        """Keep the title pinned to the top border when the box is resized."""
        super().resizeEvent(event)
        self._position_title()
