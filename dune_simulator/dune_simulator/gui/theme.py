"""Blender 4.x Dark Studio QSS theme and styling palette for the Desert application."""

BLENDER_DARK_QSS = """
QMainWindow, QDialog, QWidget {
    background-color: #181818;
    color: #e2e2e2;
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
}

QLabel {
    color: #dcdcdc;
    background-color: transparent;
}

QMenuBar {
    background-color: #1f1f1f;
    color: #dcdcdc;
    border-bottom: 1px solid #111111;
    padding: 2px 6px;
    font-weight: 600;
}

QMenuBar::item {
    background: transparent;
    padding: 4px 10px;
    border-radius: 4px;
}

QMenuBar::item:selected {
    background-color: #383838;
    color: #ffffff;
}

QMenu {
    background-color: #242424;
    color: #e0e0e0;
    border: 1px solid #3a3a3a;
    padding: 4px;
}

QMenu::item {
    padding: 5px 24px 5px 14px;
    border-radius: 3px;
}

QMenu::item:selected {
    background-color: #d97724;
    color: #ffffff;
}

QToolBar {
    background-color: #202020;
    border: 1px solid #141414;
    spacing: 5px;
    padding: 4px;
}

QTabWidget::pane {
    border: 1px solid #2c2c2c;
    background-color: #1d1d1d;
    border-radius: 4px;
}

QTabBar::tab {
    background-color: #262626;
    color: #a8a8a8;
    border: 1px solid #191919;
    border-bottom: none;
    padding: 6px 14px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    font-weight: 600;
}

QTabBar::tab:hover {
    background-color: #303030;
    color: #dedede;
}

QTabBar::tab:selected {
    background-color: #333333;
    color: #f5a623;
    border-top: 2px solid #e67e22;
}

QGroupBox {
    background-color: #252525;
    border: 1px solid #333333;
    border-radius: 5px;
    margin-top: 10px;
    padding-top: 12px;
    font-weight: bold;
    color: #eab354;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: #f0a830;
}

QPushButton {
    background-color: #363636;
    color: #e6e6e6;
    border: 1px solid #484848;
    border-radius: 4px;
    padding: 6px 12px;
    font-weight: 600;
}

QPushButton:hover {
    background-color: #444444;
    border-color: #5a5a5a;
}

QPushButton:pressed {
    background-color: #d97724;
    color: #ffffff;
}

QPushButton#PrimaryActionBtn {
    background-color: #d97724;
    color: #ffffff;
    border: 1px solid #f39c12;
    font-weight: bold;
    padding: 7px 14px;
}

QPushButton#PrimaryActionBtn:hover {
    background-color: #e68a35;
}

QPushButton#RunActiveBtn {
    background-color: #27ae60;
    color: #ffffff;
    border: 1px solid #2ecc71;
    font-weight: bold;
    padding: 7px 14px;
}

QPushButton#StopActiveBtn {
    background-color: #c0392b;
    color: #ffffff;
    border: 1px solid #e74c3c;
    font-weight: bold;
    padding: 7px 14px;
}

QComboBox, QDoubleSpinBox, QSpinBox {
    background-color: #1c1c1c;
    color: #ececec;
    border: 1px solid #3c3c3c;
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: #d97724;
}

QComboBox:hover, QDoubleSpinBox:hover, QSpinBox:hover {
    border-color: #606060;
}

QSlider::groove:horizontal {
    border: 1px solid #141414;
    height: 6px;
    background: #1b1b1b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: #d97724;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #f0a830;
    border: 1px solid #181818;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}

QCheckBox {
    color: #dcdcdc;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 3px;
    border: 1px solid #4a4a4a;
    background-color: #1c1c1c;
}

QCheckBox::indicator:checked {
    background-color: #d97724;
    border-color: #f39c12;
}

QScrollArea {
    border: none;
    background-color: transparent;
}

QScrollBar:vertical {
    background: #1c1c1c;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background: #3e3e3e;
    min-height: 20px;
    border-radius: 5px;
}

QScrollBar::handle:vertical:hover {
    background: #585858;
}

QStatusBar {
    background-color: #1f1f1f;
    color: #b8b8b8;
    border-top: 1px solid #111111;
}

QLabel#TelemetryBadge {
    background-color: #252525;
    color: #f5b041;
    border: 1px solid #383838;
    border-radius: 4px;
    padding: 3px 8px;
    font-family: "Consolas", "Courier New", monospace;
    font-weight: bold;
    font-size: 11px;
}
"""
