"""Main Blender-style studio window and launch splash screen for the 'Desert' application."""

import os
import json
import math
from typing import List, Optional
import numpy as np

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QAction, QFont, QPainter, QColor
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QFormLayout,
    QSplitter,
    QTabWidget,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QSlider,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QFrame,
)

import matplotlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LightSource

from ..core.constants import (
    PlanetaryEnvironment,
    EARTH_DESERT,
    MARS_DUNES,
    VENUS_DUNES,
    AQUEOUS_FLUME,
)
from ..core.wind_profile import (
    compute_wind_profile,
    evaluate_log_wind_profile,
    u_star_from_u10,
    compute_impact_ripple_scales,
    synthesize_zoomed_ripple_patch,
    compute_experimental_benchmarks,
)
from ..analysis.dispersion import (
    linear_growth_rate,
    find_neutral_wavenumber,
    find_fastest_growing_wavenumber,
)
from ..analysis.morphometry import compute_kinematic_brink_speed
from ..presets.initial_surfaces import (
    barchan_seed_2d,
    twin_barchans_2d,
    flat_bed_noise_2d,
    ridge_2d as gaussian_ridge_2d,
    backward_step_2d,
    multi_mound_field_2d,
    sahara_dune_sea_2d,
    desert_pyramid_obstacle_2d,
)
from ..simulation.config import SimulationConfig2D, SimulationState2D
from ..simulation.simulator_2d import DuneSimulator2D
from ..visualization.plots_2d import _desert_sand_cmap, plot_barchan_evolution_diagnostics
from .assets import ensure_gui_assets
from .theme import BLENDER_DARK_QSS
from .viewport_3d import DesertViewport3D
from .custom_project_dialog import CustomProjectDialog
from .scientific_help_dialog import EquationsAndLimitationsStrip, ScientificHelpDialog
from .i18n import TRANSLATIONS

NL = chr(10)


class DesertSplashScreen(QDialog):
    """Blender-inspired startup splash dialog showcasing the Desert logo and quick-start presets."""

    def __init__(self, assets: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Desert — Physical Aeolian Morphodynamics Studio")
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        self.setFixedSize(760, 560)
        self.selected_preset: str = "barchan"
        self.open_file_requested: bool = False
        self.custom_project_requested: bool = False

        if os.path.exists(assets.get("icon", "")):
            self.setWindowIcon(QIcon(assets["icon"]))

        self.setStyleSheet(
            "QDialog { background-color: #16161a; border: 2px solid #d97724; border-radius: 8px; }"
            "QPushButton { background-color: #2c2c32; color: #ececec; border: 1px solid #45454d; "
            "border-radius: 5px; padding: 8px 14px; font-weight: 600; font-size: 12px; }"
            "QPushButton:hover { background-color: #d97724; color: #ffffff; border-color: #f5a623; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 12)
        layout.setSpacing(8)

        # Top Logo Banner with painted title overlay
        banner_label = QLabel(self)
        banner_label.setFixedSize(756, 360)
        pix = QPixmap(assets["splash"]).scaled(
            756,
            360,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(pix)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw 'DESERT' emblem text on banner
        font_title = QFont("Segoe UI", 42, QFont.Weight.Bold)
        font_title.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 8.0)
        painter.setFont(font_title)
        painter.setPen(QColor(0, 0, 0, 190))
        painter.drawText(30, 304, "DESERT")
        painter.setPen(QColor(252, 222, 156))
        painter.drawText(27, 301, "DESERT")

        font_sub = QFont("Segoe UI", 11, QFont.Weight.DemiBold)
        painter.setFont(font_sub)
        painter.setPen(QColor(235, 175, 85))
        painter.drawText(
            30, 332, "PHYSICAL AEOLIAN & SUBAQUEOUS DUNE MORPHODYNAMICS STUDIO  |  v1.0"
        )
        painter.end()

        banner_label.setPixmap(pix)
        layout.addWidget(banner_label)

        # Quick-Start Project Templates Row
        sub_lbl = QLabel("   QUICK START — SELECT INITIAL DESERT PROJECT OR OPEN EXISTING:")
        sub_lbl.setStyleSheet("color: #a8acb8; font-weight: bold; font-size: 11px;")
        layout.addWidget(sub_lbl)

        grid = QGridLayout()
        grid.setContentsMargins(16, 2, 16, 4)
        grid.setSpacing(8)

        btn_barchan = QPushButton("Classic 3D Barchan Crescent Dune")
        btn_barchan.setStyleSheet(
            "background-color: #d97724; color: #ffffff; border: 1px solid #f5a623; font-weight: bold;"
        )
        btn_barchan.clicked.connect(lambda: self._pick("barchan"))
        grid.addWidget(btn_barchan, 0, 0)

        btn_sea = QPushButton("Sahara Sinuous Dune Sea (Full Field)")
        btn_sea.setStyleSheet(
            "background-color: #b85d19; color: #ffffff; border: 1px solid #f5a623; font-weight: bold;"
        )
        btn_sea.clicked.connect(lambda: self._pick("dune_sea"))
        grid.addWidget(btn_sea, 0, 1)

        btn_pyr = QPushButton("Desert Pyramid Obstacle && Dunes")
        btn_pyr.clicked.connect(lambda: self._pick("pyramid"))
        grid.addWidget(btn_pyr, 0, 2)

        btn_twin = QPushButton("Twin Interacting Barchans (Collision)")
        btn_twin.clicked.connect(lambda: self._pick("twin_barchan"))
        grid.addWidget(btn_twin, 1, 0)

        btn_mars = QPushButton("Mars Low-Density Mega-Barchan")
        btn_mars.clicked.connect(lambda: self._pick("mars"))
        grid.addWidget(btn_mars, 1, 1)

        btn_open = QPushButton("Open Saved Project (.desert)...")
        btn_open.clicked.connect(self._pick_open)
        grid.addWidget(btn_open, 1, 2)

        btn_custom = QPushButton("New Custom Project (Define Parameters && Sculpt 3D Initial Surface)...")
        btn_custom.setStyleSheet(
            "background-color: #27ae60; color: #ffffff; border: 1px solid #2ecc71; font-weight: bold; padding: 9px;"
        )
        btn_custom.clicked.connect(self._pick_custom)
        grid.addWidget(btn_custom, 2, 0, 1, 3)

        layout.addLayout(grid)

    def _pick(self, preset: str) -> None:
        self.selected_preset = preset
        self.accept()

    def _pick_open(self) -> None:
        self.open_file_requested = True
        self.accept()

    def _pick_custom(self) -> None:
        self.custom_project_requested = True
        self.accept()


class DesertMainWindow(QMainWindow):
    """Blender-inspired interactive 3D studio for physical sand dune and ripple simulation."""

    def __init__(self, assets: Optional[dict] = None):
        super().__init__()
        self.assets = assets or ensure_gui_assets()
        self.lang: str = "en"
        self.setWindowTitle(TRANSLATIONS["en"]["window_title"])
        if os.path.exists(self.assets.get("icon", "")):
            self.setWindowIcon(QIcon(self.assets["icon"]))
        self.resize(1620, 960)
        self.setStyleSheet(BLENDER_DARK_QSS)

        self.simulator: Optional[DuneSimulator2D] = None
        self.state_history: List[SimulationState2D] = []
        self.current_project_path: Optional[str] = None
        self._updating_wind_controls: bool = False
        self._scrubbing_timeline: bool = False

        # Simulation QTimer for dynamic real-time execution
        self.sim_timer = QTimer(self)
        self.sim_timer.setInterval(45)
        self.sim_timer.timeout.connect(self._on_timer_step)

        self._build_menus()
        self._build_main_layout()
        self._update_physical_readouts()
        self.initialize_simulation_from_ui()

    # =========================================================================
    # MENU BAR & TOPBAR (WITH ENGLISH / ARABIC LANGUAGE TOGGLE BUTTON)
    # =========================================================================
    def _build_menus(self) -> None:
        menubar = self.menuBar()

        # Top-Right Language Toggle Button (EN <-> Arabic)
        self.btn_lang_toggle = QPushButton(TRANSLATIONS["en"]["lang_btn"], menubar)
        self.btn_lang_toggle.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_lang_toggle.setStyleSheet(
            "QPushButton { background-color: #d97724; color: #ffffff; border: 1px solid #f5a623; "
            "border-radius: 4px; padding: 3px 12px; font-weight: bold; font-size: 11.5px; margin: 2px 6px; }"
            "QPushButton:hover { background-color: #e68a35; }"
        )
        self.btn_lang_toggle.clicked.connect(self.toggle_language)
        menubar.setCornerWidget(self.btn_lang_toggle, Qt.Corner.TopRightCorner)

        # File Menu
        self.file_menu = menubar.addMenu("&File")
        self.act_custom_new = QAction("New Custom Project (3D Surface Sculptor && Parameters)...", self)
        self.act_custom_new.setShortcut("Ctrl+Shift+N")
        self.act_custom_new.triggered.connect(self.open_custom_project_dialog)
        self.file_menu.addAction(self.act_custom_new)

        self.act_new = QAction("Reset to Active Preset Initial Conditions", self)
        self.act_new.setShortcut("Ctrl+N")
        self.act_new.triggered.connect(self.initialize_simulation_from_ui)
        self.file_menu.addAction(self.act_new)

        self.act_open = QAction("Open Project (.desert)...", self)
        self.act_open.setShortcut("Ctrl+O")
        self.act_open.triggered.connect(self.load_project_dialog)
        self.file_menu.addAction(self.act_open)

        self.act_save = QAction("Save Project (.desert)...", self)
        self.act_save.setShortcut("Ctrl+S")
        self.act_save.triggered.connect(self.save_project_dialog)
        self.file_menu.addAction(self.act_save)

        self.file_menu.addSeparator()

        self.act_exp_png = QAction("Export Publication Diagnostic Figure (.png)...", self)
        self.act_exp_png.triggered.connect(self.export_diagnostic_png_dialog)
        self.file_menu.addAction(self.act_exp_png)

        self.act_exp_obj = QAction("Export 3D Wavefront Mesh (.obj)...", self)
        self.act_exp_obj.triggered.connect(self.export_wavefront_obj_dialog)
        self.file_menu.addAction(self.act_exp_obj)

        self.file_menu.addSeparator()
        self.act_exit = QAction("Exit Desert", self)
        self.act_exit.setShortcut("Ctrl+Q")
        self.act_exit.triggered.connect(self.close)
        self.file_menu.addAction(self.act_exit)

        # View Menu
        self.view_menu = menubar.addMenu("&View")
        self.act_v_lee = QAction("Camera: Leeward Slipface Bay", self)
        self.act_v_lee.triggered.connect(lambda: self.viewport_3d.view_leeward_bay())
        self.view_menu.addAction(self.act_v_lee)

        self.act_v_stoss = QAction("Camera: Windward Stoss Ramp", self)
        self.act_v_stoss.triggered.connect(lambda: self.viewport_3d.view_windward_stoss())
        self.view_menu.addAction(self.act_v_stoss)

        self.act_v_top = QAction("Camera: Overhead Planform", self)
        self.act_v_top.triggered.connect(lambda: self.viewport_3d.view_top_planform())
        self.view_menu.addAction(self.act_v_top)

        self.act_v_rip = QAction("Camera: Close-Up Stoss Ripples", self)
        self.act_v_rip.triggered.connect(lambda: self.viewport_3d.zoom_close_to_stoss_ripples())
        self.view_menu.addAction(self.act_v_rip)

        self.act_v_micro = QAction("Toggle 2m x 2m Micro-Ripple Inspector", self)
        self.act_v_micro.triggered.connect(
            lambda: self.viewport_3d.set_micro_ripple_zoom_mode(
                not self.viewport_3d.micro_ripple_zoom_mode
            )
        )
        self.view_menu.addAction(self.act_v_micro)

        # Presets Menu
        self.preset_menu = menubar.addMenu("&Presets")
        self.preset_actions = []
        for label, key in [
            ("Earth Desert — Classic Barchan Dune", "barchan"),
            ("Earth Desert — Sahara Sinuous Dune Sea (Full Field)", "dune_sea"),
            ("Earth Desert — Rocky Pyramid Obstacle && Shadow Dunes", "pyramid"),
            ("Earth Desert — Twin Colliding Barchans", "twin_barchan"),
            ("Earth Desert — Multi-Mound Barchan Swarm", "swarm"),
            ("Earth Desert — Spontaneous Dune && Ripple Emergence", "instability"),
            ("Earth Desert — Backward Step Shadow Dune", "step"),
            ("Mars — Low-Density Mega-Barchan", "mars"),
            ("Venus — Dense-Atmosphere Micro-Dune", "venus"),
            ("Subaqueous Flume — Water Current Bedforms", "flume"),
        ]:
            act = QAction(label, self)
            act.triggered.connect(lambda checked=False, k=key: self.apply_quick_preset(k))
            self.preset_menu.addAction(act)
            self.preset_actions.append(act)

        # Simulation Menu
        self.sim_menu = menubar.addMenu("&Simulation")
        self.act_sim_run = QAction("Run / Pause Dynamic Simulation (Space)", self)
        self.act_sim_run.setShortcut("Space")
        self.act_sim_run.triggered.connect(self.toggle_run_pause)
        self.sim_menu.addAction(self.act_sim_run)

        self.act_sim_step1 = QAction("Step Forward (+1 dt)", self)
        self.act_sim_step1.triggered.connect(lambda: self.advance_simulation_steps(1))
        self.sim_menu.addAction(self.act_sim_step1)

        self.act_sim_step10 = QAction("Burst Forward (+10 dt)", self)
        self.act_sim_step10.triggered.connect(lambda: self.advance_simulation_steps(10))
        self.sim_menu.addAction(self.act_sim_step10)

        self.act_sim_ff = QAction("Fast-Forward to Mature Barchan (+35 Steps)", self)
        self.act_sim_ff.triggered.connect(lambda: self.advance_simulation_steps(35))
        self.sim_menu.addAction(self.act_sim_ff)

        # Help Menu
        self.help_menu = menubar.addMenu("&Help")
        self.act_sci_ref = QAction("Scientific Reference Manual (Equations, Integration, Citations && Limitations)...", self)
        self.act_sci_ref.setShortcut("F1")
        self.act_sci_ref.triggered.connect(self.show_about_dialog)
        self.help_menu.addAction(self.act_sci_ref)

        self.act_toggle_eq = QAction("Toggle Bottom Equations && Limitations Dock", self)
        self.act_toggle_eq.setCheckable(True)
        self.act_toggle_eq.setChecked(True)
        self.act_toggle_eq.toggled.connect(lambda v: self.eq_strip.setVisible(v))
        self.help_menu.addAction(self.act_toggle_eq)

    # =========================================================================
    # MAIN WORKSPACE LAYOUT (LEFT SHELF | CENTER VIEWPORTS | RIGHT INSPECTOR)
    # =========================================================================
    def _build_main_layout(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_vbox = QVBoxLayout(central)
        root_vbox.setContentsMargins(4, 4, 4, 4)
        root_vbox.setSpacing(4)

        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # --- 1. Left Tool Shelf (Blender T-Panel) ---
        left_shelf = self._build_left_tool_shelf()
        main_splitter.addWidget(left_shelf)

        # --- 2. Center Multi-Viewport Workspace ---
        self.center_tabs = QTabWidget(self)
        self.viewport_3d = DesertViewport3D(self)
        self.viewport_3d.on_surface_sculpted = self._on_main_viewport_sculpted
        self.center_tabs.addTab(self.viewport_3d, "3D Desert Studio Viewport")

        self.panel_2d = self._build_2d_inspector_tab()
        self.center_tabs.addTab(self.panel_2d, "2D Morphodynamics && Ripple Zoom Inspector")

        self.panel_bench = self._build_experimental_validation_tab()
        self.center_tabs.addTab(self.panel_bench, "Experimental Benchmarks && Dispersion Lab")

        self.center_tabs.currentChanged.connect(self._on_center_tab_changed)
        main_splitter.addWidget(self.center_tabs)

        # --- 3. Right Properties Inspector (Blender N-Panel) ---
        right_inspector = self._build_right_inspector()
        main_splitter.addWidget(right_inspector)

        main_splitter.setStretchFactor(0, 0)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setStretchFactor(2, 0)
        main_splitter.setSizes([205, 960, 455])

        root_vbox.addWidget(main_splitter, stretch=1)

        # --- 4. Visible Scientific Equations, Integration Formulas, Citations & Limitations Strip ---
        self.eq_strip = EquationsAndLimitationsStrip(self)
        root_vbox.addWidget(self.eq_strip)

        # --- 5. Bottom Timeline & Telemetry Bar ---
        timeline_bar = self._build_bottom_timeline_bar()
        root_vbox.addWidget(timeline_bar)

    def _build_left_tool_shelf(self) -> QFrame:
        shelf = QFrame(self)
        shelf.setFixedWidth(205)
        shelf.setStyleSheet("QFrame { background-color: #212124; border: 1px solid #2e2e32; border-radius: 4px; }")
        vbox = QVBoxLayout(shelf)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(6)

        self.lbl_shelf_hdr = QLabel("DESERT TOOL SHELF")
        self.lbl_shelf_hdr.setStyleSheet("color: #f0a830; font-weight: bold; font-size: 11px; border: none;")
        vbox.addWidget(self.lbl_shelf_hdr)

        self.btn_new_custom = QPushButton("New Custom Project...")
        self.btn_new_custom.setObjectName("PrimaryActionBtn")
        self.btn_new_custom.setToolTip("Open Custom Project Designer & 3D Initial Surface Sculptor")
        self.btn_new_custom.clicked.connect(self.open_custom_project_dialog)
        vbox.addWidget(self.btn_new_custom)

        self.btn_init_bed = QPushButton("Reset / Rebuild Bed")
        self.btn_init_bed.setToolTip("Rebuild initial sand bed from Right Panel settings")
        self.btn_init_bed.clicked.connect(self.initialize_simulation_from_ui)
        vbox.addWidget(self.btn_init_bed)

        self.btn_run_pause = QPushButton("Run Simulation")
        self.btn_run_pause.setObjectName("RunActiveBtn")
        self.btn_run_pause.clicked.connect(self.toggle_run_pause)
        vbox.addWidget(self.btn_run_pause)

        self.btn_step1 = QPushButton("Step (+1 dt)")
        self.btn_step1.clicked.connect(lambda: self.advance_simulation_steps(1))
        vbox.addWidget(self.btn_step1)

        self.btn_step10 = QPushButton("Burst (+10 dt)")
        self.btn_step10.clicked.connect(lambda: self.advance_simulation_steps(10))
        vbox.addWidget(self.btn_step10)

        self.btn_ff = QPushButton("Form Barchan (+35 dt)")
        self.btn_ff.setToolTip("Advance 35 morphodynamic steps to see full crescent Barchan formation")
        self.btn_ff.clicked.connect(lambda: self.advance_simulation_steps(35))
        vbox.addWidget(self.btn_ff)

        self.lbl_sep1 = QLabel("3D SCULPT & BED TOOLS")
        self.lbl_sep1.setStyleSheet("color: #9ba0ac; font-weight: bold; font-size: 10px; margin-top: 6px; border: none;")
        vbox.addWidget(self.lbl_sep1)

        self.btn_sculpt_3d = QPushButton("3D Brush Sculpt Mode")
        self.btn_sculpt_3d.setToolTip("Paint and shape the 3D sand surface directly with the mouse")
        self.btn_sculpt_3d.clicked.connect(
            lambda: self.viewport_3d.set_sculpt_mode(not self.viewport_3d.sculpt_mode)
        )
        vbox.addWidget(self.btn_sculpt_3d)

        self.btn_add_mound = QPushButton("+ Add Sand Mound")
        self.btn_add_mound.clicked.connect(self.sculpt_add_sand_mound)
        vbox.addWidget(self.btn_add_mound)

        self.btn_smooth = QPushButton("Smooth Sand Surface")
        self.btn_smooth.clicked.connect(self.sculpt_smooth_surface)
        vbox.addWidget(self.btn_smooth)

        self.lbl_sep2 = QLabel("CAMERA & RIPPLE ZOOM")
        self.lbl_sep2.setStyleSheet("color: #9ba0ac; font-weight: bold; font-size: 10px; margin-top: 6px; border: none;")
        vbox.addWidget(self.lbl_sep2)

        self.btn_cam_lee = QPushButton("Cam: Leeward Bay")
        self.btn_cam_lee.clicked.connect(lambda: self.viewport_3d.view_leeward_bay())
        vbox.addWidget(self.btn_cam_lee)

        self.btn_cam_stoss = QPushButton("Cam: Windward Stoss")
        self.btn_cam_stoss.clicked.connect(lambda: self.viewport_3d.view_windward_stoss())
        vbox.addWidget(self.btn_cam_stoss)

        self.btn_cam_zoom_rip = QPushButton("Zoom: Stoss Ripples")
        self.btn_cam_zoom_rip.clicked.connect(lambda: self.viewport_3d.zoom_close_to_stoss_ripples())
        vbox.addWidget(self.btn_cam_zoom_rip)

        self.btn_cam_micro = QPushButton("Zoom: 2m Ripple Patch")
        self.btn_cam_micro.clicked.connect(
            lambda: self.viewport_3d.set_micro_ripple_zoom_mode(
                not self.viewport_3d.micro_ripple_zoom_mode
            )
        )
        vbox.addWidget(self.btn_cam_micro)

        self.lbl_sep3 = QLabel("PROJECT & EXPORT")
        self.lbl_sep3.setStyleSheet("color: #9ba0ac; font-weight: bold; font-size: 10px; margin-top: 6px; border: none;")
        vbox.addWidget(self.lbl_sep3)

        self.btn_save = QPushButton("Save Project (.desert)")
        self.btn_save.clicked.connect(self.save_project_dialog)
        vbox.addWidget(self.btn_save)

        self.btn_load = QPushButton("Open Project (.desert)")
        self.btn_load.clicked.connect(self.load_project_dialog)
        vbox.addWidget(self.btn_load)

        self.btn_exp = QPushButton("Export Figure (.png)")
        self.btn_exp.clicked.connect(self.export_diagnostic_png_dialog)
        vbox.addWidget(self.btn_exp)

        vbox.addStretch(1)
        return shelf

    # =========================================================================
    # CENTER TAB 2: 2D MORPHODYNAMICS & RIPPLE ZOOM INSPECTOR
    # =========================================================================
    def _build_2d_inspector_tab(self) -> QWidget:
        w = QWidget(self)
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(4, 4, 4, 4)

        self.fig_2d = Figure(figsize=(12, 8), dpi=100, facecolor="#1a1a1e")
        self.canvas_2d = FigureCanvas(self.fig_2d)
        vbox.addWidget(self.canvas_2d)
        return w

    # =========================================================================
    # CENTER TAB 3: EXPERIMENTAL VALIDATION & DISPERSION LAB
    # =========================================================================
    def _build_experimental_validation_tab(self) -> QWidget:
        w = QWidget(self)
        vbox = QVBoxLayout(w)
        vbox.setContentsMargins(4, 4, 4, 4)

        self.fig_bench = Figure(figsize=(12, 8), dpi=100, facecolor="#1a1a1e")
        self.canvas_bench = FigureCanvas(self.fig_bench)
        vbox.addWidget(self.canvas_bench)
        return w

    # =========================================================================
    # RIGHT PROPERTIES INSPECTOR (4 TABS: SAND BED | GRAIN | WIND | SHADING)
    # =========================================================================
    def _build_right_inspector(self) -> QWidget:
        panel = QWidget(self)
        panel.setFixedWidth(455)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)

        self.inspector_tabs = QTabWidget(panel)
        self.inspector_tabs.addTab(self._build_sand_bed_tab(), "1. Sand Bed")
        self.inspector_tabs.addTab(self._build_grain_planet_tab(), "2. Grain && Planet")
        self.inspector_tabs.addTab(self._build_wind_profile_tab(), "3. Wind Profile")
        self.inspector_tabs.addTab(self._build_shading_ripples_tab(), "4. Shading && Ripples")

        vbox.addWidget(self.inspector_tabs)
        return panel

    def _build_sand_bed_tab(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self.grp_surf = QGroupBox("Initial Sand Bed State (Topography)")
        form = QFormLayout(self.grp_surf)
        form.setSpacing(7)

        self.combo_bed_preset = QComboBox()
        self.combo_bed_preset.addItems(
            [
                "Isolated Barchan Mound (Parabolic Seed)",
                "Twin Interacting Barchans (Collision)",
                "Multi-Mound Barchan Swarm Field",
                "Flat Sand Sheet + White Noise (Instability)",
                "Gaussian Transverse Ridge",
                "Backward Step Escarpment (Wake Trap)",
                "Sahara Sinuous Dune Sea (Full Field)",
                "Rocky Pyramid Obstacle & Shadow Dunes",
            ]
        )
        self.combo_bed_preset.currentIndexChanged.connect(self._on_bed_preset_combo_changed)
        self.lbl_preset = QLabel("Surface Preset:")
        form.addRow(self.lbl_preset, self.combo_bed_preset)

        self.spin_mound_height = QDoubleSpinBox()
        self.spin_mound_height.setRange(0.2, 20.0)
        self.spin_mound_height.setSingleStep(0.2)
        self.spin_mound_height.setValue(3.2)
        self.spin_mound_height.setSuffix(" m")
        self.lbl_h0 = QLabel("Initial Mound Height H0:")
        form.addRow(self.lbl_h0, self.spin_mound_height)

        self.spin_mound_radius = QDoubleSpinBox()
        self.spin_mound_radius.setRange(4.0, 65.0)
        self.spin_mound_radius.setSingleStep(1.0)
        self.spin_mound_radius.setValue(18.0)
        self.spin_mound_radius.setSuffix(" m")
        self.lbl_r0 = QLabel("Mound Radius / Width R0:")
        form.addRow(self.lbl_r0, self.spin_mound_radius)

        self.spin_base_sand = QDoubleSpinBox()
        self.spin_base_sand.setRange(0.0, 15.0)
        self.spin_base_sand.setSingleStep(0.1)
        self.spin_base_sand.setValue(0.0)
        self.spin_base_sand.setSuffix(" m")
        self.lbl_base_sand = QLabel("Base Sand Layer Thickness:")
        form.addRow(self.lbl_base_sand, self.spin_base_sand)

        self.spin_noise_amp = QDoubleSpinBox()
        self.spin_noise_amp.setRange(0.0, 0.50)
        self.spin_noise_amp.setSingleStep(0.01)
        self.spin_noise_amp.setDecimals(3)
        self.spin_noise_amp.setValue(0.00)
        self.spin_noise_amp.setSuffix(" m")
        self.lbl_noise = QLabel("Surface Roughness Noise:")
        form.addRow(self.lbl_noise, self.spin_noise_amp)

        vbox.addWidget(self.grp_surf)

        self.grp_dom = QGroupBox("Domain & Boundary Conditions")
        form_dom = QFormLayout(self.grp_dom)
        form_dom.setSpacing(7)

        self.spin_lx = QDoubleSpinBox()
        self.spin_lx.setRange(40.0, 1000.0)
        self.spin_lx.setValue(120.0)
        self.spin_lx.setSuffix(" m")
        self.lbl_lx = QLabel("Streamwise Length Lx:")
        form_dom.addRow(self.lbl_lx, self.spin_lx)

        self.spin_ly = QDoubleSpinBox()
        self.spin_ly.setRange(40.0, 1000.0)
        self.spin_ly.setValue(100.0)
        self.spin_ly.setSuffix(" m")
        self.lbl_ly = QLabel("Transverse Width Ly:")
        form_dom.addRow(self.lbl_ly, self.spin_ly)

        self.combo_grid_res = QComboBox()
        self.combo_grid_res.addItems(["96 x 80 (Fast Interactive)", "120 x 100 (Balanced HD)", "160 x 128 (High Precision)"])
        self.combo_grid_res.setCurrentIndex(1)
        self.lbl_grid_res = QLabel("Grid Resolution (Nx x Ny):")
        form_dom.addRow(self.lbl_grid_res, self.combo_grid_res)

        self.chk_bedrock = QCheckBox("Enforce Non-Erodible Bedrock Floor (z = 0)")
        self.chk_bedrock.setChecked(True)
        form_dom.addRow(self.chk_bedrock)

        self.chk_periodic_x = QCheckBox("Periodic Streamwise Boundary (x)")
        self.chk_periodic_x.setChecked(False)
        form_dom.addRow(self.chk_periodic_x)

        self.spin_inflow = QDoubleSpinBox()
        self.spin_inflow.setRange(0.0, 1.0)
        self.spin_inflow.setSingleStep(0.02)
        self.spin_inflow.setValue(0.12)
        self.lbl_inflow = QLabel("Upwind Sand Inflow Ratio:")
        form_dom.addRow(self.lbl_inflow, self.spin_inflow)

        vbox.addWidget(self.grp_dom)

        self.btn_apply_bed = QPushButton("Apply Initial Conditions && Rebuild Desert Bed")
        self.btn_apply_bed.setObjectName("PrimaryActionBtn")
        self.btn_apply_bed.clicked.connect(self.initialize_simulation_from_ui)
        vbox.addWidget(self.btn_apply_bed)

        vbox.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_grain_planet_tab(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self.grp_env = QGroupBox("Planetary Environment & Grain Properties")
        form = QFormLayout(self.grp_env)
        form.setSpacing(7)

        self.combo_planet = QComboBox()
        self.combo_planet.addItems(TRANSLATIONS["en"]["planet_items"])
        self.combo_planet.currentIndexChanged.connect(self._on_planet_preset_changed)
        self.lbl_env_preset = QLabel("Environment Preset:")
        form.addRow(self.lbl_env_preset, self.combo_planet)

        # Grain Diameter d [um] with synced slider + spinbox
        self.spin_grain_d_um = QDoubleSpinBox()
        self.spin_grain_d_um.setRange(40.0, 2500.0)
        self.spin_grain_d_um.setSingleStep(10.0)
        self.spin_grain_d_um.setDecimals(0)
        self.spin_grain_d_um.setValue(200.0)
        self.spin_grain_d_um.setSuffix(" um")
        self.spin_grain_d_um.valueChanged.connect(self._on_grain_or_fluid_changed)
        self.lbl_grain_d = QLabel("Grain Diameter d:")
        form.addRow(self.lbl_grain_d, self.spin_grain_d_um)

        self.slider_grain_d = QSlider(Qt.Orientation.Horizontal)
        self.slider_grain_d.setRange(40, 2500)
        self.slider_grain_d.setValue(200)
        self.slider_grain_d.valueChanged.connect(lambda v: self.spin_grain_d_um.setValue(float(v)))
        self.lbl_grain_slider = QLabel("Grain Size Slider:")
        form.addRow(self.lbl_grain_slider, self.slider_grain_d)

        self.spin_rho_s = QDoubleSpinBox()
        self.spin_rho_s.setRange(500.0, 8000.0)
        self.spin_rho_s.setSingleStep(50.0)
        self.spin_rho_s.setValue(2650.0)
        self.spin_rho_s.setSuffix(" kg/m^3")
        self.spin_rho_s.valueChanged.connect(self._on_grain_or_fluid_changed)
        self.lbl_rho_s = QLabel("Grain Density rho_s:")
        form.addRow(self.lbl_rho_s, self.spin_rho_s)

        self.spin_rho_f = QDoubleSpinBox()
        self.spin_rho_f.setRange(0.005, 2000.0)
        self.spin_rho_f.setDecimals(3)
        self.spin_rho_f.setSingleStep(0.05)
        self.spin_rho_f.setValue(1.225)
        self.spin_rho_f.setSuffix(" kg/m^3")
        self.spin_rho_f.valueChanged.connect(self._on_grain_or_fluid_changed)
        self.lbl_rho_f = QLabel("Fluid Density rho_f:")
        form.addRow(self.lbl_rho_f, self.spin_rho_f)

        self.spin_gravity = QDoubleSpinBox()
        self.spin_gravity.setRange(0.5, 30.0)
        self.spin_gravity.setDecimals(2)
        self.spin_gravity.setValue(9.81)
        self.spin_gravity.setSuffix(" m/s^2")
        self.spin_gravity.valueChanged.connect(self._on_grain_or_fluid_changed)
        self.lbl_gravity = QLabel("Gravity g:")
        form.addRow(self.lbl_gravity, self.spin_gravity)

        vbox.addWidget(self.grp_env)

        # Live Physical Scales Readout Card
        self.grp_scales = QGroupBox("Emergent Physical & Experimental Scales")
        v_scales = QVBoxLayout(self.grp_scales)
        self.lbl_grain_scales = QLabel()
        self.lbl_grain_scales.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 11px; color: #dcdcdc; "
            "background-color: #1b1b1e; padding: 8px; border-radius: 4px; border: 1px solid #34343a;"
        )
        v_scales.addWidget(self.lbl_grain_scales)
        vbox.addWidget(self.grp_scales)

        self.btn_apply_grain = QPushButton("Update Active Simulation Grain && Fluid Parameters")
        self.btn_apply_grain.setObjectName("PrimaryActionBtn")
        self.btn_apply_grain.clicked.connect(self._apply_live_parameters_to_simulator)
        vbox.addWidget(self.btn_apply_grain)

        vbox.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_wind_profile_tab(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self.grp_wind = QGroupBox("Logarithmic Boundary-Layer Wind Profile u(z)")
        form = QFormLayout(self.grp_wind)
        form.setSpacing(7)

        self.spin_u_star = QDoubleSpinBox()
        self.spin_u_star.setRange(0.02, 2.50)
        self.spin_u_star.setDecimals(3)
        self.spin_u_star.setSingleStep(0.02)
        self.spin_u_star.setValue(0.380)
        self.spin_u_star.setSuffix(" m/s")
        self.spin_u_star.valueChanged.connect(self._on_u_star_changed)
        self.lbl_u_star = QLabel("Shear Velocity u*0:")
        form.addRow(self.lbl_u_star, self.spin_u_star)

        self.slider_u_star = QSlider(Qt.Orientation.Horizontal)
        self.slider_u_star.setRange(20, 1500)
        self.slider_u_star.setValue(380)
        self.slider_u_star.valueChanged.connect(lambda v: self.spin_u_star.setValue(v / 1000.0))
        self.lbl_u_slider = QLabel("Shear Velocity Slider:")
        form.addRow(self.lbl_u_slider, self.slider_u_star)

        self.spin_u10 = QDoubleSpinBox()
        self.spin_u10.setRange(0.5, 45.0)
        self.spin_u10.setDecimals(2)
        self.spin_u10.setSingleStep(0.5)
        self.spin_u10.setValue(12.20)
        self.spin_u10.setSuffix(" m/s")
        self.spin_u10.valueChanged.connect(self._on_u10_changed)
        self.lbl_u10 = QLabel("10m Wind Speed U(10m):")
        form.addRow(self.lbl_u10, self.spin_u10)

        self.spin_jh_A = QDoubleSpinBox()
        self.spin_jh_A.setRange(1.0, 6.0)
        self.spin_jh_A.setSingleStep(0.1)
        self.spin_jh_A.setValue(3.2)
        self.spin_jh_A.valueChanged.connect(self._on_grain_or_fluid_changed)
        self.lbl_jh_A = QLabel("Jackson-Hunt In-Phase A:")
        form.addRow(self.lbl_jh_A, self.spin_jh_A)

        self.spin_jh_B = QDoubleSpinBox()
        self.spin_jh_B.setRange(0.4, 4.0)
        self.spin_jh_B.setSingleStep(0.1)
        self.spin_jh_B.setValue(1.6)
        self.spin_jh_B.valueChanged.connect(self._on_grain_or_fluid_changed)
        self.lbl_jh_B = QLabel("Jackson-Hunt Quadrature B:")
        form.addRow(self.lbl_jh_B, self.spin_jh_B)

        self.spin_d_lat = QDoubleSpinBox()
        self.spin_d_lat.setRange(0.05, 1.50)
        self.spin_d_lat.setSingleStep(0.05)
        self.spin_d_lat.setValue(0.35)
        self.spin_d_lat.valueChanged.connect(self._apply_live_parameters_to_simulator)
        self.lbl_d_lat = QLabel("Lateral Splash Diffusion D_lat:")
        form.addRow(self.lbl_d_lat, self.spin_d_lat)

        vbox.addWidget(self.grp_wind)

        # Wind Layer Readout Card
        self.lbl_wind_readout = QLabel()
        self.lbl_wind_readout.setStyleSheet(
            "font-family: 'Consolas', monospace; font-size: 11px; color: #dcdcdc; "
            "background-color: #1b1b1e; padding: 7px; border-radius: 4px; border: 1px solid #34343a;"
        )
        vbox.addWidget(self.lbl_wind_readout)

        # Embedded Mini Logarithmic Wind Profile Canvas
        self.grp_wind_plot = QGroupBox("Live Prandtl-Owen Wind Profile u(z)")
        v_plot = QVBoxLayout(self.grp_wind_plot)
        v_plot.setContentsMargins(4, 8, 4, 4)
        self.fig_wind_mini = Figure(figsize=(3.6, 2.35), dpi=90, facecolor="#1b1b1e")
        self.canvas_wind_mini = FigureCanvas(self.fig_wind_mini)
        self.canvas_wind_mini.setMinimumHeight(195)
        v_plot.addWidget(self.canvas_wind_mini)
        vbox.addWidget(self.grp_wind_plot)

        vbox.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_shading_ripples_tab(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget()
        vbox = QVBoxLayout(content)
        vbox.setContentsMargins(8, 8, 8, 8)
        vbox.setSpacing(8)

        self.grp_sh = QGroupBox("3D Viewport Shader & Sun Illumination")
        form = QFormLayout(self.grp_sh)
        form.setSpacing(7)

        self.combo_shader = QComboBox()
        self.combo_shader.addItems(TRANSLATIONS["en"]["shader_items"])
        self.combo_shader.currentIndexChanged.connect(self._on_shading_changed)
        self.lbl_shader = QLabel("Surface Shader:")
        form.addRow(self.lbl_shader, self.combo_shader)

        self.slider_sun_az = QSlider(Qt.Orientation.Horizontal)
        self.slider_sun_az.setRange(0, 360)
        self.slider_sun_az.setValue(140)
        self.slider_sun_az.valueChanged.connect(self._on_shading_changed)
        self.lbl_sun_az = QLabel("Sun Azimuth [deg]:")
        form.addRow(self.lbl_sun_az, self.slider_sun_az)

        self.slider_sun_el = QSlider(Qt.Orientation.Horizontal)
        self.slider_sun_el.setRange(8, 85)
        self.slider_sun_el.setValue(28)
        self.slider_sun_el.valueChanged.connect(self._on_shading_changed)
        self.lbl_sun_el = QLabel("Sun Elevation [deg]:")
        form.addRow(self.lbl_sun_el, self.slider_sun_el)

        self.slider_vert_exag = QSlider(Qt.Orientation.Horizontal)
        self.slider_vert_exag.setRange(10, 60)
        self.slider_vert_exag.setValue(35)
        self.slider_vert_exag.valueChanged.connect(self._on_shading_changed)
        self.lbl_vert_exag = QLabel("3D Vertical Exaggeration:")
        form.addRow(self.lbl_vert_exag, self.slider_vert_exag)

        vbox.addWidget(self.grp_sh)

        self.grp_rip = QGroupBox("Multi-Scale Desert Ripples & 3D Overlays")
        v_rip = QVBoxLayout(self.grp_rip)
        v_rip.setSpacing(6)

        self.chk_show_ripples = QCheckBox("Synthesize 3D Ballistic Impact Ripples on Dune")
        self.chk_show_ripples.setChecked(True)
        self.chk_show_ripples.toggled.connect(self._on_shading_changed)
        v_rip.addWidget(self.chk_show_ripples)

        form_rip = QFormLayout()
        self.slider_rip_exag = QSlider(Qt.Orientation.Horizontal)
        self.slider_rip_exag.setRange(10, 200)
        self.slider_rip_exag.setValue(55)
        self.slider_rip_exag.valueChanged.connect(self._on_shading_changed)
        self.lbl_rip_contrast = QLabel("Ripple Relief Contrast:")
        form_rip.addRow(self.lbl_rip_contrast, self.slider_rip_exag)
        v_rip.addLayout(form_rip)

        self.chk_diorama = QCheckBox("Render 3D Geological Cutaway Diorama Skirt")
        self.chk_diorama.setChecked(True)
        self.chk_diorama.toggled.connect(self._on_shading_changed)
        v_rip.addWidget(self.chk_diorama)

        self.chk_shadows = QCheckBox("Real-Time Ray-Marched Sun Shadows && HBAO")
        self.chk_shadows.setChecked(True)
        self.chk_shadows.toggled.connect(self._on_shading_changed)
        v_rip.addWidget(self.chk_shadows)

        self.chk_show_wake = QCheckBox("Show 3D Separation Streamline Bubble")
        self.chk_show_wake.setChecked(False)
        self.chk_show_wake.toggled.connect(self._on_shading_changed)
        v_rip.addWidget(self.chk_show_wake)

        self.chk_show_brink = QCheckBox("Highlight 3D Crescent Brink Crestline")
        self.chk_show_brink.setChecked(False)
        self.chk_show_brink.toggled.connect(self._on_shading_changed)
        v_rip.addWidget(self.chk_show_brink)

        self.chk_show_saltation = QCheckBox("Show Active Saltation Grain Cloud")
        self.chk_show_saltation.setChecked(False)
        self.chk_show_saltation.toggled.connect(self._on_shading_changed)
        v_rip.addWidget(self.chk_show_saltation)

        self.chk_show_streamlines = QCheckBox("Show 3D Boundary-Layer Wind Streamlines")
        self.chk_show_streamlines.setChecked(False)
        self.chk_show_streamlines.toggled.connect(self._on_shading_changed)
        v_rip.addWidget(self.chk_show_streamlines)

        self.chk_wireframe = QCheckBox("Overlay 3D Topographic Mesh Wireframe")
        self.chk_wireframe.setChecked(False)
        self.chk_wireframe.toggled.connect(self._on_shading_changed)
        v_rip.addWidget(self.chk_wireframe)

        vbox.addWidget(self.grp_rip)
        vbox.addStretch(1)
        scroll.setWidget(content)
        return scroll

    # =========================================================================
    # BOTTOM TIMELINE & LIVE TELEMETRY DOCK
    # =========================================================================
    def _build_bottom_timeline_bar(self) -> QFrame:
        bar = QFrame(self)
        bar.setFixedHeight(44)
        bar.setStyleSheet("QFrame { background-color: #202024; border: 1px solid #2e2e34; border-radius: 4px; }")
        hbox = QHBoxLayout(bar)
        hbox.setContentsMargins(8, 4, 8, 4)
        hbox.setSpacing(8)

        self.lbl_tl = QLabel("TIMELINE:")
        self.lbl_tl.setStyleSheet("color: #f0a830; font-weight: bold; font-size: 11px; border: none;")
        hbox.addWidget(self.lbl_tl)

        self.slider_timeline = QSlider(Qt.Orientation.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.setValue(0)
        self.slider_timeline.setMinimumWidth(240)
        self.slider_timeline.valueChanged.connect(self._on_timeline_scrubbed)
        hbox.addWidget(self.slider_timeline, stretch=1)

        self.badge_time = QLabel("t = 0.0 d")
        self.badge_time.setObjectName("TelemetryBadge")
        hbox.addWidget(self.badge_time)

        self.badge_height = QLabel("Crest H: 0.00 m")
        self.badge_height.setObjectName("TelemetryBadge")
        hbox.addWidget(self.badge_height)

        self.badge_speed = QLabel("Speed c: 0.0 m/yr")
        self.badge_speed.setObjectName("TelemetryBadge")
        hbox.addWidget(self.badge_speed)

        self.badge_horns = QLabel("Horn Ext: +0.0 m")
        self.badge_horns.setObjectName("TelemetryBadge")
        hbox.addWidget(self.badge_horns)

        self.badge_ripple = QLabel("Ripple: 12.5 cm")
        self.badge_ripple.setObjectName("TelemetryBadge")
        hbox.addWidget(self.badge_ripple)

        self.badge_vol = QLabel("Vol: 0.0 m^3")
        self.badge_vol.setObjectName("TelemetryBadge")
        hbox.addWidget(self.badge_vol)

        return bar

    # =========================================================================
    # ENVIRONMENT, GRAIN & WIND PROFILE SYNCHRONIZATION
    # =========================================================================
    def _build_current_environment(self) -> PlanetaryEnvironment:
        names = ["Earth Desert", "Mars Dunes", "Venus Dunes", "Subaqueous Flume", "Custom Regime"]
        idx = self.combo_planet.currentIndex()
        name = names[idx] if 0 <= idx < len(names) else "Custom Regime"
        return PlanetaryEnvironment(
            name=name,
            gravity=float(self.spin_gravity.value()),
            fluid_density=float(self.spin_rho_f.value()),
            grain_density=float(self.spin_rho_s.value()),
            grain_diameter=float(self.spin_grain_d_um.value()) * 1e-6,
            u_star_0=float(self.spin_u_star.value()),
        )

    def _on_planet_preset_changed(self, idx: int) -> None:
        presets = [EARTH_DESERT, MARS_DUNES, VENUS_DUNES, AQUEOUS_FLUME]
        if 0 <= idx < len(presets):
            p = presets[idx]
            self.spin_grain_d_um.blockSignals(True)
            self.spin_rho_s.blockSignals(True)
            self.spin_rho_f.blockSignals(True)
            self.spin_gravity.blockSignals(True)
            self.spin_u_star.blockSignals(True)

            self.spin_grain_d_um.setValue(p.grain_diameter * 1e6)
            self.slider_grain_d.setValue(int(round(p.grain_diameter * 1e6)))
            self.spin_rho_s.setValue(p.grain_density)
            self.spin_rho_f.setValue(p.fluid_density)
            self.spin_gravity.setValue(p.gravity)
            u_def = 0.38 if idx == 0 else p.u_star_0
            self.spin_u_star.setValue(u_def)
            self.slider_u_star.setValue(int(round(u_def * 1000.0)))

            self.spin_grain_d_um.blockSignals(False)
            self.spin_rho_s.blockSignals(False)
            self.spin_rho_f.blockSignals(False)
            self.spin_gravity.blockSignals(False)
            self.spin_u_star.blockSignals(False)

        self._update_physical_readouts()
        self._apply_live_parameters_to_simulator()

    def _on_grain_or_fluid_changed(self) -> None:
        self.slider_grain_d.blockSignals(True)
        self.slider_grain_d.setValue(int(round(self.spin_grain_d_um.value())))
        self.slider_grain_d.blockSignals(False)
        self._update_physical_readouts()
        self._apply_live_parameters_to_simulator()

    def _on_u_star_changed(self, val: float) -> None:
        if self._updating_wind_controls:
            return
        self._updating_wind_controls = True
        self.slider_u_star.blockSignals(True)
        self.slider_u_star.setValue(int(round(val * 1000.0)))
        self.slider_u_star.blockSignals(False)

        env = self._build_current_environment()
        prof = compute_wind_profile(env, val)
        self.spin_u10.blockSignals(True)
        self.spin_u10.setValue(prof.u_10m)
        self.spin_u10.blockSignals(False)
        self._updating_wind_controls = False

        self._update_physical_readouts()
        self._apply_live_parameters_to_simulator()

    def _on_u10_changed(self, u10_val: float) -> None:
        if self._updating_wind_controls:
            return
        self._updating_wind_controls = True
        env = self._build_current_environment()
        u_star = u_star_from_u10(env, u10_val)
        self.spin_u_star.blockSignals(True)
        self.spin_u_star.setValue(u_star)
        self.spin_u_star.blockSignals(False)
        self.slider_u_star.blockSignals(True)
        self.slider_u_star.setValue(int(round(u_star * 1000.0)))
        self.slider_u_star.blockSignals(False)
        self._updating_wind_controls = False

        self._update_physical_readouts()
        self._apply_live_parameters_to_simulator()

    def _update_physical_readouts(self) -> None:
        env = self._build_current_environment()
        u_star = float(self.spin_u_star.value())
        A = float(self.spin_jh_A.value())
        B = float(self.spin_jh_B.value())

        prof = compute_wind_profile(env, u_star)
        _, lam_min = find_neutral_wavenumber(env, A=A, B=B)
        _, lam_max, sigma_max = find_fastest_growing_wavenumber(env, prof.tau_0, A=A, B=B)
        lam_r, amp_r, c_r = compute_impact_ripple_scales(env, u_star)

        stage = u_star / max(1e-6, env.threshold_friction_velocity)
        if self.lang == "ar":
            self.lbl_grain_scales.setText(
                f"سرعة القص الحرجة u*_th : {env.threshold_friction_velocity:6.3f} م/ث" + NL
                + f"إجهاد القص الحرج tau_th: {env.threshold_shear_stress:6.3f} باسكال" + NL
                + f"طول الكبح الحبيبي L_dr : {env.drag_length:6.3f} م" + NL
                + f"طول التشبع الرسوبي L_sat: {env.saturation_length:6.3f} م" + NL
                + f"الطول الموجي الحرج lambda_m: {lam_min:6.2f} م" + NL
                + f"أسرع طول موجي نمواً lambda*: {lam_max:6.2f} م" + NL
                + f"طول موجة التموجات lambda_r : {lam_r*100.0:6.1f} سم (الارتفاع = {2*amp_r*1000:4.1f} ملم)"
            )
            status_str = "قفز حبيبي نشط (ACTIVE)" if stage > 1.0 else "تحت عتبة الحركة (ساكن)"
            self.lbl_wind_readout.setText(
                f"النظام الحركي : {status_str} (u*/u*th = {stage:.2f})" + NL
                + f"إجهاد القص القاعي tau_0 : {prof.tau_0:.3f} باسكال" + NL
                + f"سرعة الرياح U(2m): {prof.u_2m:.2f} م/ث | U(10m): {prof.u_10m:.2f} م/ث" + NL
                + f"الخشونة الفعالة z0_eff: {prof.z0_eff_m*1000:.4f} ملم (الطبقة الداخلية l = {prof.inner_layer_l_m:.2f} م)"
            )
        else:
            self.lbl_grain_scales.setText(
                f"Threshold u*_th : {env.threshold_friction_velocity:6.3f} m/s" + NL
                + f"Threshold tau_th: {env.threshold_shear_stress:6.3f} Pa" + NL
                + f"Drag Length L_dr: {env.drag_length:6.3f} m" + NL
                + f"Saturation L_sat: {env.saturation_length:6.3f} m" + NL
                + f"Cutoff lambda_m : {lam_min:6.2f} m" + NL
                + f"Fastest lambda* : {lam_max:6.2f} m" + NL
                + f"Ripple lambda_r : {lam_r*100.0:6.1f} cm (H_r = {2*amp_r*1000:4.1f} mm)"
            )
            status_str = "ACTIVE SALTATION" if stage > 1.0 else "BELOW THRESHOLD (QUIESCENT)"
            self.lbl_wind_readout.setText(
                f"Regime : {status_str} (u*/u*th = {stage:.2f})" + NL
                + f"Basal Shear tau_0 : {prof.tau_0:.3f} Pa" + NL
                + f"Wind Speed U(2m)  : {prof.u_2m:.2f} m/s | U(10m): {prof.u_10m:.2f} m/s" + NL
                + f"Roughness z0_eff  : {prof.z0_eff_m*1000:.4f} mm (Inner l = {prof.inner_layer_l_m:.2f} m)"
            )

        # Update embedded mini logarithmic wind profile plot
        self.fig_wind_mini.clear()
        ax = self.fig_wind_mini.add_subplot(111)
        ax.set_facecolor("#151518")
        z_arr, u_arr = evaluate_log_wind_profile(prof, z_min_m=0.005, z_max_m=10.0)
        ax.plot(u_arr, z_arr, color="#00e5ff", lw=2.0, label=r"$u(z) = \frac{u_*}{\kappa}\ln(z/z_0)$")
        ax.axhline(prof.inner_layer_l_m, color="#f5a623", ls="--", lw=1.2, label=f"Inner Layer l = {prof.inner_layer_l_m:.1f} m")
        ax.set_yscale("log")
        ax.set_xlabel("Wind Speed u(z) [m/s]", color="#dcdcdc", fontsize=8.5)
        ax.set_ylabel("Height z [m] (log)", color="#dcdcdc", fontsize=8.5)
        ax.tick_params(colors="#b0b0b0", labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#3a3a42")
        ax.grid(True, linestyle=":", color="#33333a", alpha=0.7)
        ax.legend(loc="lower right", fontsize=7.5, facecolor="#202024", edgecolor="#3a3a42", labelcolor="#e0e0e0")
        self.fig_wind_mini.tight_layout()
        self.canvas_wind_mini.draw_idle()

    def _apply_live_parameters_to_simulator(self) -> None:
        if self.simulator is None:
            return
        env = self._build_current_environment()
        self.simulator.config.env = env
        self.simulator.config.u_star_0 = float(self.spin_u_star.value())
        self.simulator.config.A = float(self.spin_jh_A.value())
        self.simulator.config.B = float(self.spin_jh_B.value())
        self.simulator.config.d_lat = float(self.spin_d_lat.value())
        self.simulator.config.inflow_ratio = float(self.spin_inflow.value())
        self.simulator.config.enforce_bedrock = bool(self.chk_bedrock.isChecked())
        self._refresh_all_views(self.simulator.get_state())

    def _on_shading_changed(self) -> None:
        modes = ["desert", "elevation", "shear", "flux", "wake"]
        idx = self.combo_shader.currentIndex()
        self.viewport_3d.shader_mode = modes[idx] if 0 <= idx < len(modes) else "desert"
        self.viewport_3d.sun_azimuth_deg = float(self.slider_sun_az.value())
        self.viewport_3d.sun_elevation_deg = float(self.slider_sun_el.value())
        self.viewport_3d.vertical_exaggeration = float(self.slider_vert_exag.value()) / 10.0
        self.viewport_3d.show_ripples = bool(self.chk_show_ripples.isChecked())
        self.viewport_3d.ripple_exaggeration = float(self.slider_rip_exag.value()) / 10.0
        self.viewport_3d.show_diorama_skirt = bool(self.chk_diorama.isChecked())
        self.viewport_3d.show_shadows_and_ao = bool(self.chk_shadows.isChecked())
        self.viewport_3d.show_wake_dome = bool(self.chk_show_wake.isChecked())
        self.viewport_3d.show_brink_line = bool(self.chk_show_brink.isChecked())
        self.viewport_3d.show_saltation_cloud = bool(self.chk_show_saltation.isChecked())
        self.viewport_3d.show_wind_streamlines = bool(self.chk_show_streamlines.isChecked())
        self.viewport_3d.show_wireframe = bool(self.chk_wireframe.isChecked())

        if self.simulator is not None:
            self.viewport_3d.update_from_state(
                self.simulator.get_state(),
                self._build_current_environment(),
                float(self.spin_u_star.value()),
                self.simulator.config.bedrock_floor,
            )

    @staticmethod
    def _repopulate_combo(combo: QComboBox, items: list) -> None:
        idx = combo.currentIndex()
        combo.blockSignals(True)
        combo.clear()
        combo.addItems(items)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)

    def toggle_language(self) -> None:
        """Toggles 100% of the application UI between English ('en') and Technical Arabic ('ar')."""
        self.lang = "ar" if self.lang == "en" else "en"
        tr = TRANSLATIONS[self.lang]

        if self.lang == "ar":
            self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
            self.viewport_3d.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        else:
            self.setLayoutDirection(Qt.LayoutDirection.LeftToRight)

        self.setWindowTitle(tr["window_title"])
        self.btn_lang_toggle.setText(tr["lang_btn"])

        # Menus & Sub-actions
        self.file_menu.setTitle(tr["menu_file"])
        self.view_menu.setTitle(tr["menu_view"])
        self.preset_menu.setTitle(tr["menu_presets"])
        self.sim_menu.setTitle(tr["menu_sim"])
        self.help_menu.setTitle(tr["menu_help"])
        self.act_custom_new.setText(tr["act_custom_new"])
        self.act_new.setText(tr["act_reset_bed"])
        self.act_open.setText(tr["act_open"])
        self.act_save.setText(tr["act_save"])
        self.act_exp_png.setText(tr["act_exp_png"])
        self.act_exp_obj.setText(tr["act_exp_obj"])
        self.act_exit.setText(tr["act_exit"])
        self.act_v_lee.setText(tr["act_v_lee"])
        self.act_v_stoss.setText(tr["act_v_stoss"])
        self.act_v_top.setText(tr["act_v_top"])
        self.act_v_rip.setText(tr["act_v_rip"])
        self.act_v_micro.setText(tr["act_v_micro"])
        for act, txt in zip(self.preset_actions, tr["preset_menu_items"]):
            act.setText(txt)
        self.act_sim_run.setText(tr["act_sim_run"])
        self.act_sim_step1.setText(tr["act_sim_step1"])
        self.act_sim_step10.setText(tr["act_sim_step10"])
        self.act_sim_ff.setText(tr["act_sim_ff"])
        self.act_sci_ref.setText(tr["act_sci_ref"])
        self.act_toggle_eq.setText(tr["act_toggle_eq"])

        # Tool Shelf
        self.lbl_shelf_hdr.setText(tr["shelf_header"])
        self.btn_new_custom.setText(tr["btn_new_custom"])
        self.btn_init_bed.setText(tr["btn_reset_bed"])
        self.btn_run_pause.setText(tr["btn_pause"] if self.sim_timer.isActive() else tr["btn_run"])
        self.btn_step1.setText(tr["btn_step1"])
        self.btn_step10.setText(tr["btn_step10"])
        self.btn_ff.setText(tr["btn_form_barchan"])
        self.lbl_sep1.setText(tr["shelf_sculpt"])
        self.btn_sculpt_3d.setText(tr["btn_sculpt_mode"])
        self.btn_add_mound.setText(tr["btn_add_mound"])
        self.btn_smooth.setText(tr["btn_smooth"])
        self.lbl_sep2.setText(tr["shelf_camera"])
        self.btn_cam_lee.setText(tr["btn_cam_lee"])
        self.btn_cam_stoss.setText(tr["btn_cam_stoss"])
        self.btn_cam_zoom_rip.setText(tr["btn_cam_ripples"])
        self.btn_cam_micro.setText(tr["btn_cam_micro"])
        self.lbl_sep3.setText(tr["shelf_project"])
        self.btn_save.setText(tr["btn_save_proj"])
        self.btn_load.setText(tr["btn_open_proj"])
        self.btn_exp.setText(tr["btn_exp_fig"])

        # Center & Right Tabs
        self.center_tabs.setTabText(0, tr["tab_3d"])
        self.center_tabs.setTabText(1, tr["tab_2d"])
        self.center_tabs.setTabText(2, tr["tab_bench"])
        self.inspector_tabs.setTabText(0, tr["insp_tab_bed"])
        self.inspector_tabs.setTabText(1, tr["insp_tab_grain"])
        self.inspector_tabs.setTabText(2, tr["insp_tab_wind"])
        self.inspector_tabs.setTabText(3, tr["insp_tab_shade"])

        # Tab 1: Sand Bed
        self.grp_surf.setTitle(tr["grp_surf"])
        self.lbl_preset.setText(tr["lbl_preset"])
        self._repopulate_combo(self.combo_bed_preset, tr["bed_preset_items"])
        self.lbl_h0.setText(tr["lbl_h0"])
        self.lbl_r0.setText(tr["lbl_r0"])
        self.lbl_base_sand.setText(tr["lbl_base_sand"])
        self.lbl_noise.setText(tr["lbl_noise"])
        self.grp_dom.setTitle(tr["grp_dom"])
        self.lbl_lx.setText(tr["lbl_lx"])
        self.lbl_ly.setText(tr["lbl_ly"])
        self.lbl_grid_res.setText(tr["lbl_grid_res"])
        self._repopulate_combo(self.combo_grid_res, tr["grid_res_items"])
        self.lbl_inflow.setText(tr["lbl_inflow"])
        self.chk_bedrock.setText(tr["chk_bedrock"])
        self.chk_periodic_x.setText(tr["chk_periodic"])
        self.btn_apply_bed.setText(tr["btn_apply_bed"])

        # Tab 2: Grain & Planet
        self.grp_env.setTitle(tr["grp_env"])
        self.lbl_env_preset.setText(tr["lbl_env_preset"])
        self._repopulate_combo(self.combo_planet, tr["planet_items"])
        self.lbl_grain_d.setText(tr["lbl_grain_d"])
        self.lbl_grain_slider.setText(tr["lbl_grain_slider"])
        self.lbl_rho_s.setText(tr["lbl_rho_s"])
        self.lbl_rho_f.setText(tr["lbl_rho_f"])
        self.lbl_gravity.setText(tr["lbl_gravity"])
        self.grp_scales.setTitle(tr["grp_scales"])
        self.btn_apply_grain.setText(tr["btn_apply_grain"])

        # Tab 3: Wind Profile
        self.grp_wind.setTitle(tr["grp_wind"])
        self.lbl_u_star.setText(tr["lbl_u_star"])
        self.lbl_u_slider.setText(tr["lbl_u_slider"])
        self.lbl_u10.setText(tr["lbl_u10"])
        self.lbl_jh_A.setText(tr["lbl_jh_A"])
        self.lbl_jh_B.setText(tr["lbl_jh_B"])
        self.lbl_d_lat.setText(tr["lbl_d_lat"])
        self.grp_wind_plot.setTitle(tr["grp_wind_plot"])

        # Tab 4: Shading & Ripples
        self.grp_sh.setTitle(tr["grp_shade"])
        self.lbl_shader.setText(tr["lbl_shader"])
        self._repopulate_combo(self.combo_shader, tr["shader_items"])
        self.lbl_sun_az.setText(tr["lbl_sun_az"])
        self.lbl_sun_el.setText(tr["lbl_sun_el"])
        self.lbl_vert_exag.setText(tr["lbl_vert_exag"])
        self.grp_rip.setTitle(tr["grp_ripples"])
        self.chk_show_ripples.setText(tr["chk_ripples"])
        self.lbl_rip_contrast.setText(tr["lbl_rip_contrast"])
        self.chk_diorama.setText(tr["chk_diorama"])
        self.chk_shadows.setText(tr["chk_shadows"])
        self.chk_show_wake.setText(tr["chk_wake"])
        self.chk_show_brink.setText(tr["chk_brink"])
        self.chk_show_saltation.setText(tr["chk_saltation"])
        self.chk_show_streamlines.setText(tr["chk_streamlines"])
        self.chk_wireframe.setText(tr["chk_wireframe"])

        # Timeline & Bottom Docks
        self.lbl_tl.setText(tr["tl_label"])
        self.viewport_3d.set_language(self.lang)
        self.eq_strip.set_language(self.lang)
        self._update_physical_readouts()
        if self.simulator is not None:
            self._refresh_all_views(self.simulator.get_state())

    # =========================================================================
    # INITIALIZATION, SIMULATION LOOP & SCULPTING
    # =========================================================================
    def _on_bed_preset_combo_changed(self, idx: int) -> None:
        """Automatically configures default boundary and sand layer settings when Surface Preset changes."""
        if idx in (3, 6):  # Flat Sand Sheet or Sahara Sinuous Dune Sea
            self.spin_base_sand.setValue(2.5)
            self.spin_noise_amp.setValue(0.08 if idx == 3 else 0.0)
            self.chk_periodic_x.setChecked(True)
        elif idx in (5, 7):  # Backward Step or Rocky Pyramid Obstacle
            self.spin_base_sand.setValue(0.5 if idx == 5 else 1.5)
            self.chk_periodic_x.setChecked(idx == 7)
            self.spin_inflow.setValue(0.25)
        else:
            self.spin_base_sand.setValue(0.0)
            self.chk_periodic_x.setChecked(False)
            if self.spin_inflow.value() == 0.0:
                self.spin_inflow.setValue(0.12)
        if self.simulator is not None:
            self.initialize_simulation_from_ui()

    def apply_quick_preset(self, preset_key: str) -> None:
        """Configures UI controls for a named preset and initializes the sand bed."""
        if self.sim_timer.isActive():
            self.toggle_run_pause()

        if preset_key == "barchan":
            self.combo_planet.setCurrentIndex(0)
            self.combo_bed_preset.setCurrentIndex(0)
            self.spin_mound_height.setValue(3.2)
            self.spin_mound_radius.setValue(18.0)
            self.spin_base_sand.setValue(0.0)
            self.spin_noise_amp.setValue(0.0)
            self.chk_bedrock.setChecked(True)
            self.chk_periodic_x.setChecked(False)
            self.spin_inflow.setValue(0.12)
            self.spin_lx.setValue(120.0)
            self.spin_ly.setValue(100.0)
        elif preset_key == "dune_sea":
            self.combo_planet.setCurrentIndex(0)
            self.combo_bed_preset.setCurrentIndex(6)
            self.spin_mound_height.setValue(3.2)
            self.spin_base_sand.setValue(2.6)
            self.chk_bedrock.setChecked(True)
            self.chk_periodic_x.setChecked(True)
            self.spin_lx.setValue(120.0)
            self.spin_ly.setValue(100.0)
        elif preset_key == "pyramid":
            self.combo_planet.setCurrentIndex(0)
            self.combo_bed_preset.setCurrentIndex(7)
            self.spin_mound_height.setValue(5.2)
            self.spin_base_sand.setValue(2.4)
            self.chk_bedrock.setChecked(True)
            self.chk_periodic_x.setChecked(True)
            self.spin_lx.setValue(120.0)
            self.spin_ly.setValue(100.0)
        elif preset_key == "twin_barchan":
            self.combo_planet.setCurrentIndex(0)
            self.combo_bed_preset.setCurrentIndex(1)
            self.spin_mound_height.setValue(3.4)
            self.spin_mound_radius.setValue(17.0)
            self.spin_base_sand.setValue(0.0)
            self.chk_bedrock.setChecked(True)
            self.chk_periodic_x.setChecked(False)
            self.spin_inflow.setValue(0.10)
            self.spin_lx.setValue(140.0)
            self.spin_ly.setValue(110.0)
        elif preset_key == "swarm":
            self.combo_planet.setCurrentIndex(0)
            self.combo_bed_preset.setCurrentIndex(2)
            self.spin_mound_height.setValue(3.0)
            self.spin_mound_radius.setValue(14.0)
            self.spin_base_sand.setValue(0.0)
            self.chk_bedrock.setChecked(True)
            self.chk_periodic_x.setChecked(False)
            self.spin_inflow.setValue(0.12)
            self.spin_lx.setValue(140.0)
            self.spin_ly.setValue(110.0)
        elif preset_key == "instability":
            self.combo_planet.setCurrentIndex(0)
            self.combo_bed_preset.setCurrentIndex(3)
            self.spin_base_sand.setValue(2.5)
            self.spin_noise_amp.setValue(0.08)
            self.chk_bedrock.setChecked(True)
            self.chk_periodic_x.setChecked(True)
            self.spin_lx.setValue(120.0)
            self.spin_ly.setValue(100.0)
        elif preset_key == "step":
            self.combo_planet.setCurrentIndex(0)
            self.combo_bed_preset.setCurrentIndex(5)
            self.spin_mound_height.setValue(2.6)
            self.spin_base_sand.setValue(0.5)
            self.chk_bedrock.setChecked(True)
            self.chk_periodic_x.setChecked(False)
            self.spin_inflow.setValue(0.25)
        elif preset_key == "mars":
            self.combo_planet.setCurrentIndex(1)
            self.combo_bed_preset.setCurrentIndex(0)
            self.spin_mound_height.setValue(8.0)
            self.spin_mound_radius.setValue(60.0)
            self.spin_lx.setValue(450.0)
            self.spin_ly.setValue(360.0)
            self.chk_bedrock.setChecked(True)
            self.chk_periodic_x.setChecked(False)
        elif preset_key == "venus":
            self.combo_planet.setCurrentIndex(2)
            self.combo_bed_preset.setCurrentIndex(0)
            self.spin_mound_height.setValue(0.4)
            self.spin_mound_radius.setValue(4.0)
            self.spin_lx.setValue(40.0)
            self.spin_ly.setValue(40.0)
        elif preset_key == "flume":
            self.combo_planet.setCurrentIndex(3)
            self.combo_bed_preset.setCurrentIndex(0)
            self.spin_mound_height.setValue(0.3)
            self.spin_mound_radius.setValue(4.0)
            self.spin_lx.setValue(40.0)
            self.spin_ly.setValue(40.0)

        self.initialize_simulation_from_ui()

    def initialize_simulation_from_ui(self) -> None:
        """Builds the initial 2D sand bed surface and initializes DuneSimulator2D."""
        if self.sim_timer.isActive():
            self.toggle_run_pause()

        res_options = [(96, 80), (120, 100), (160, 128)]
        r_idx = self.combo_grid_res.currentIndex()
        nx, ny = res_options[r_idx] if 0 <= r_idx < len(res_options) else (120, 100)

        lx = float(self.spin_lx.value())
        ly = float(self.spin_ly.value())
        h0 = float(self.spin_mound_height.value())
        r0 = float(self.spin_mound_radius.value())
        base_h = float(self.spin_base_sand.value())
        noise_a = float(self.spin_noise_amp.value())

        bedrock_floor = 0.0
        preset_idx = self.combo_bed_preset.currentIndex()
        if preset_idx == 0:
            h_init = barchan_seed_2d(
                lx, ly, nx, ny, height=h0, radius=r0, center_x=lx * 0.30, center_y=ly * 0.50, base_height=base_h
            )
            if noise_a > 0.0:
                rng = np.random.default_rng(42)
                mask = h_init > base_h + 0.05
                h_init[mask] = np.maximum(base_h, h_init[mask] + rng.uniform(-noise_a, noise_a, size=np.sum(mask)))
        elif preset_idx == 1:
            h_init = twin_barchans_2d(
                lx, ly, nx, ny, height_main=h0, radius_main=r0, base_height=base_h, noise_amplitude=noise_a
            )
        elif preset_idx == 2:
            h_init = multi_mound_field_2d(lx, ly, nx, ny, height=h0, radius=r0, base_height=base_h)
        elif preset_idx == 3:
            h_init = flat_bed_noise_2d(
                lx, ly, nx, ny, mean_height=max(0.5, base_h if base_h > 0 else h0 * 0.6), noise_amplitude=max(0.03, noise_a)
            )
        elif preset_idx == 4:
            h_init = gaussian_ridge_2d(
                lx, ly, nx, ny, height=h0, width=r0, center_x=lx * 0.32, base_height=base_h
            )
        elif preset_idx == 5:
            h_init = backward_step_2d(lx, ly, nx, ny, step_height=h0, base_height=max(0.2, base_h))
        elif preset_idx == 6:
            h_init = sahara_dune_sea_2d(lx, ly, nx, ny, mean_height=max(1.8, base_h), amplitude=h0 * 0.72)
        else:
            h_init, bedrock_floor = desert_pyramid_obstacle_2d(
                lx, ly, nx, ny, pyramid_height=max(4.5, h0), sand_height=max(2.2, base_h)
            )

        env = self._build_current_environment()
        cfg = SimulationConfig2D(
            nx=nx,
            ny=ny,
            length_x_m=lx,
            length_y_m=ly,
            env=env,
            u_star_0=float(self.spin_u_star.value()),
            A=float(self.spin_jh_A.value()),
            B=float(self.spin_jh_B.value()),
            A_y=1.6,
            periodic_x=bool(self.chk_periodic_x.isChecked()),
            periodic_y=True,
            inflow_ratio=float(self.spin_inflow.value()),
            enforce_bedrock=bool(self.chk_bedrock.isChecked()),
            bedrock_floor=bedrock_floor,
            d_lat=float(self.spin_d_lat.value()),
            cfl=0.45,
            dt_max=14400.0,
        )

        self.simulator = DuneSimulator2D(config=cfg, h_init=h_init)
        st0 = self.simulator.get_state()
        self.state_history = [st0]

        self.slider_timeline.blockSignals(True)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.setValue(0)
        self.slider_timeline.blockSignals(False)

        self._refresh_all_views(st0)
        self.statusBar().showMessage(
            f"Initialized '{self.combo_bed_preset.currentText()}' on {nx}x{ny} grid ({lx:.0f}m x {ly:.0f}m)."
        )

    def toggle_run_pause(self) -> None:
        if self.sim_timer.isActive():
            self.sim_timer.stop()
            self.btn_run_pause.setText("Run Simulation")
            self.btn_run_pause.setObjectName("RunActiveBtn")
            self.btn_run_pause.setStyle(self.btn_run_pause.style())
            self.statusBar().showMessage("Simulation paused.")
        else:
            self.sim_timer.start()
            self.btn_run_pause.setText("Pause Simulation")
            self.btn_run_pause.setObjectName("StopActiveBtn")
            self.btn_run_pause.setStyle(self.btn_run_pause.style())
            self.statusBar().showMessage("Running real-time morphodynamic simulation...")

    def _on_timer_step(self) -> None:
        self.advance_simulation_steps(2)

    def advance_simulation_steps(self, n_steps: int = 1) -> None:
        if self.simulator is None:
            return
        state = None
        nx = self.simulator.config.nx
        for _ in range(max(1, n_steps)):
            state = self.simulator.step()

            # Co-moving frame recentering for isolated dunes on bare bedrock:
            # If the upwind 15% of the domain is bare floor and an isolated dune's leading
            # edge approaches within 6 cells of the downwind outlet, smoothly shift the frame
            # upwind so the dune remains in view and never stops prematurely.
            if not self.simulator.config.periodic_x:
                h_cur = self.simulator.h
                h_floor = float(np.min(h_cur))
                upwind_clear = float(np.max(h_cur[: max(4, nx // 6), :])) < h_floor + 0.05
                downwind_near = float(np.max(h_cur[-6:, :])) > h_floor + 0.15
                if upwind_clear and downwind_near:
                    shift_n = max(4, nx // 8)
                    h_shifted = np.full_like(h_cur, h_floor)
                    h_shifted[:-shift_n, :] = h_cur[shift_n:, :]
                    self.simulator.h = h_shifted
                    self.simulator._update_diagnostics()
                    state = self.simulator.get_state()

        if state is not None:
            self.state_history.append(state)
            if len(self.state_history) > 250:
                self.state_history.pop(0)

            self._scrubbing_timeline = True
            self.slider_timeline.setRange(0, len(self.state_history) - 1)
            self.slider_timeline.setValue(len(self.state_history) - 1)
            self._scrubbing_timeline = False

            self._refresh_all_views(state)

    def _on_timeline_scrubbed(self, idx: int) -> None:
        if self._scrubbing_timeline or not (0 <= idx < len(self.state_history)):
            return
        if self.sim_timer.isActive():
            self.toggle_run_pause()
        st = self.state_history[idx]
        self._refresh_all_views(st)
        self.statusBar().showMessage(
            f"Timeline scrubbed to frame {idx + 1}/{len(self.state_history)} (t = {st.time_days:.1f} days)."
        )

    def sculpt_add_sand_mound(self) -> None:
        """Adds an auxiliary sand mound onto the active desert bed."""
        if self.simulator is None:
            return
        nx, ny = self.simulator.config.nx, self.simulator.config.ny
        lx, ly = self.simulator.config.length_x_m, self.simulator.config.length_y_m
        rng = np.random.default_rng()
        cx = rng.uniform(0.18 * lx, 0.45 * lx)
        cy = rng.uniform(0.30 * ly, 0.70 * ly)
        add_m = barchan_seed_2d(lx, ly, nx, ny, height=2.2, radius=12.0, center_x=cx, center_y=cy)
        self.simulator.h = self.simulator.h + add_m
        st = self.simulator.step()
        self.state_history.append(st)
        self._refresh_all_views(st)
        self.statusBar().showMessage(f"Sculpted new sand mound at (x={cx:.1f} m, y={cy:.1f} m).")

    def sculpt_smooth_surface(self) -> None:
        """Applies a gentle conservative smoothing pass to the sand bed."""
        if self.simulator is None:
            return
        from scipy import ndimage

        self.simulator.h = ndimage.gaussian_filter(self.simulator.h, sigma=1.0)
        st = self.simulator.step()
        self.state_history.append(st)
        self._refresh_all_views(st)
        self.statusBar().showMessage("Applied surface smoothing pass.")

    # =========================================================================
    # VIEWPORT & DASHBOARD RENDERING
    # =========================================================================
    def _on_center_tab_changed(self, idx: int) -> None:
        if self.simulator is not None and len(self.state_history) > 0:
            st = self.state_history[self.slider_timeline.value()]
            self._refresh_all_views(st)

    def _refresh_all_views(self, state: SimulationState2D) -> None:
        env = self._build_current_environment()
        u_star = float(self.spin_u_star.value())

        # Compute telemetry metrics
        ny = state.h.shape[1]
        yc = ny // 2
        h_c = state.h[:, yc]
        ix_c = int(np.argmax(h_c))
        q_brink = float(state.qx[ix_c, yc])
        _, c_m_yr = compute_kinematic_brink_speed(q_brink, max(0.1, float(state.max_height)), env.bulk_density)

        sand = state.h > 0.12
        front_x = float(state.x[np.where(sand.any(axis=1))[0].max()]) if sand.any() else 0.0
        c_sand = np.where(h_c > 0.12)[0]
        toe_c = float(state.x[c_sand[-1]]) if len(c_sand) > 0 else front_x
        horn_ext = max(0.0, front_x - toe_c)
        lam_r, _, _ = compute_impact_ripple_scales(env, u_star)

        if self.lang == "ar":
            self.badge_time.setText(f"الزمن = {state.time_days:5.1f} يوم ({state.time_years:4.2f} سنة)")
            self.badge_height.setText(f"القمة H: {state.max_height:4.2f} م")
            self.badge_speed.setText(f"السرعة c: {c_m_yr:5.1f} م/سنة")
            self.badge_horns.setText(f"امتداد القرنين: +{horn_ext:4.1f} م")
            self.badge_ripple.setText(f"التموجات: {lam_r*100:4.1f} سم")
            self.badge_vol.setText(f"الحجم: {state.total_volume:6.1f} م³")
        else:
            self.badge_time.setText(f"t = {state.time_days:5.1f} d ({state.time_years:4.2f} yr)")
            self.badge_height.setText(f"Crest H: {state.max_height:4.2f} m")
            self.badge_speed.setText(f"Speed c: {c_m_yr:5.1f} m/yr")
            self.badge_horns.setText(f"Horn Ext: +{horn_ext:4.1f} m")
            self.badge_ripple.setText(f"Ripple: {lam_r*100:4.1f} cm")
            self.badge_vol.setText(f"Vol: {state.total_volume:6.1f} m^3")

        active_tab = self.center_tabs.currentIndex()
        if active_tab == 0:
            bf = self.simulator.config.bedrock_floor if self.simulator is not None else None
            self.viewport_3d.update_from_state(state, env, u_star, bf)
        elif active_tab == 1:
            self._render_2d_inspector_tab(state, env, u_star)
        elif active_tab == 2:
            self._render_experimental_validation_tab(state, env, u_star, c_m_yr)

    def _render_2d_inspector_tab(
        self,
        state: SimulationState2D,
        env: PlanetaryEnvironment,
        u_star: float,
    ) -> None:
        self.fig_2d.clear()
        gs = self.fig_2d.add_gridspec(2, 2, hspace=0.32, wspace=0.28)

        x, y, h = state.x, state.y, state.h
        nx, ny = h.shape
        dx = float(x[1] - x[0]) if nx > 1 else 1.0
        dy = float(y[1] - y[0]) if ny > 1 else 1.0
        yc = ny // 2

        # Panel 1: 2D Hillshaded Planform + Brink + Wake + Flux Vectors
        ax1 = self.fig_2d.add_subplot(gs[0, 0])
        ls = LightSource(azdeg=315, altdeg=42)
        h_top = max(float(np.max(h)), 1.0)
        rgb = ls.shade(
            h, cmap=_desert_sand_cmap(), vmin=0.0, vmax=h_top, vert_exag=1.8, blend_mode="overlay", dx=dy, dy=dx
        )
        ax1.imshow(rgb, extent=[y[0], y[-1], x[0], x[-1]], origin="lower", aspect="equal")
        if float(np.max(h)) > 0.15:
            ax1.contour(y, x, h, levels=np.linspace(0.4, h_top * 0.9, 5), colors="white", alpha=0.5, linewidths=0.7)
        if np.any(state.in_wake):
            ax1.contour(y, x, state.in_wake.astype(float), levels=[0.5], colors=["#00e5ff"], linestyles="--", linewidths=1.2)
        ax1.set_title(f"2D Planform Topography & Wake (t = {state.time_days:.1f} d)", color="#f0a830", fontsize=10.5, fontweight="bold")
        ax1.set_xlabel("Transverse y [m]", color="#dcdcdc", fontsize=9)
        ax1.set_ylabel("Downwind x [m]", color="#dcdcdc", fontsize=9)
        ax1.tick_params(colors="#b0b0b0", labelsize=8)

        # Panel 2: Longitudinal Symmetry Slice (h, h_tilde, wake, tau_x)
        ax2 = self.fig_2d.add_subplot(gs[0, 1])
        ax2.set_facecolor("#141418")
        h_c = h[:, yc]
        ht_c = state.h_tilde[:, yc]
        w_c = state.in_wake[:, yc]
        if len(self.state_history) > 1:
            ax2.plot(x, self.state_history[0].h[:, yc], color="#7f8c8d", ls=":", lw=1.4, label="Initial Bed (t=0)")
        ax2.plot(x, h_c, color="#f5a623", lw=2.3, label="Centerline h(x, y_c)")
        ax2.fill_between(x, 0, h_c, color="#d97724", alpha=0.45)
        if np.any(w_c):
            ax2.plot(x, ht_c, color="#00e5ff", ls="--", lw=1.6, label="Separation Envelope")
            ax2.fill_between(x, h_c, ht_c, where=w_c, color="#00e5ff", alpha=0.22, label="Wake (tau=0)")
        ax2.set_title("Longitudinal Symmetry Profile & Separation Bubble", color="#f0a830", fontsize=10.5, fontweight="bold")
        ax2.set_xlabel("Downwind Position x [m]", color="#dcdcdc", fontsize=9)
        ax2.set_ylabel("Elevation h [m]", color="#dcdcdc", fontsize=9)
        ax2.tick_params(colors="#b0b0b0", labelsize=8)
        ax2.grid(True, ls=":", color="#32323a")
        ax2.legend(loc="upper left", fontsize=8, facecolor="#202024", edgecolor="#3a3a42", labelcolor="#e0e0e0")

        # Panel 3: Transverse Cross-Sections (Central Crest vs Lee Horn Bay)
        ax3 = self.fig_2d.add_subplot(gs[1, 0])
        ax3.set_facecolor("#141418")
        ix_crest = int(np.argmax(h_c))
        sand_c = np.where(h_c > 0.12)[0]
        ix_toe = int(sand_c[-1]) if len(sand_c) > 0 else ix_crest + 5
        ix_bay = min(nx - 1, ix_toe + 6)
        ax3.plot(y, h[ix_crest, :], color="#f5a623", lw=2.2, label=f"At Crest (x = {x[ix_crest]:.0f} m)")
        ax3.fill_between(y, 0, h[ix_crest, :], color="#f5a623", alpha=0.35)
        ax3.plot(y, h[ix_bay, :], color="#ff5722", ls="--", lw=2.0, label=f"Lee Horn Bay (x = {x[ix_bay]:.0f} m)")
        ax3.fill_between(y, 0, h[ix_bay, :], color="#ff5722", alpha=0.28)
        ax3.set_title("Transverse Slices: Crest vs Forward Horns", color="#f0a830", fontsize=10.5, fontweight="bold")
        ax3.set_xlabel("Transverse Position y [m]", color="#dcdcdc", fontsize=9)
        ax3.set_ylabel("Elevation h [m]", color="#dcdcdc", fontsize=9)
        ax3.tick_params(colors="#b0b0b0", labelsize=8)
        ax3.grid(True, ls=":", color="#32323a")
        ax3.legend(loc="upper right", fontsize=8, facecolor="#202024", edgecolor="#3a3a42", labelcolor="#e0e0e0")

        # Panel 4: Zoomed 2m x 2m Desert Impact Ripple Surface & Profile
        ax4 = self.fig_2d.add_subplot(gs[1, 1])
        x_p, y_p, z_mm, lam_cm, h_mm = synthesize_zoomed_ripple_patch(
            env=env, u_star_0=u_star, patch_size_m=2.0, n_grid=140, time_s=state.time_s
        )
        ls_r = LightSource(azdeg=315, altdeg=35)
        rgb_r = ls_r.shade(z_mm, cmap=_desert_sand_cmap(), vert_exag=0.8, blend_mode="overlay")
        ax4.imshow(rgb_r, extent=[y_p[0], y_p[-1], x_p[0], x_p[-1]], origin="lower", aspect="equal")
        ax4.set_title(
            f"Zoomed Ripple Patch (2m x 2m): lambda={lam_cm:.1f} cm, H={h_mm:.1f} mm",
            color="#f0a830",
            fontsize=10.0,
            fontweight="bold",
        )
        ax4.set_xlabel("Local y [m]", color="#dcdcdc", fontsize=9)
        ax4.set_ylabel("Local Downwind x [m]", color="#dcdcdc", fontsize=9)
        ax4.tick_params(colors="#b0b0b0", labelsize=8)

        self.fig_2d.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.08, hspace=0.34, wspace=0.26)
        self.canvas_2d.draw_idle()

    def _render_experimental_validation_tab(
        self,
        state: SimulationState2D,
        env: PlanetaryEnvironment,
        u_star: float,
        c_m_yr: float,
    ) -> None:
        self.fig_bench.clear()
        gs = self.fig_bench.add_gridspec(2, 2, hspace=0.34, wspace=0.28)
        bench = compute_experimental_benchmarks(env, u_star)

        # 1. Bagnold (1941) & Iversen-White (1982) Threshold Velocity vs Grain Diameter
        ax1 = self.fig_bench.add_subplot(gs[0, 0])
        ax1.set_facecolor("#141418")
        ax1.plot(bench["d_um"], bench["u_th_curve"], color="#00e5ff", lw=2.2, label="Bagnold Model u*_th(d)")
        ax1.scatter(bench["exp_d_um"], bench["exp_u_th"], color="#ffffff", s=32, zorder=4, label="Wind-Tunnel Data (Bagnold / Iversen)")
        ax1.scatter(
            [env.grain_diameter * 1e6],
            [env.threshold_friction_velocity],
            color="#f5a623",
            marker="*",
            s=160,
            zorder=5,
            label=f"Active Grain (d={env.grain_diameter*1e6:.0f} um)",
        )
        ax1.set_xscale("log")
        ax1.set_title("1. Threshold u*_th vs Grain Size d", color="#f0a830", fontsize=10.5, fontweight="bold")
        ax1.set_xlabel("Grain Diameter d [um] (log)", color="#dcdcdc", fontsize=9)
        ax1.set_ylabel("Threshold u*_th [m/s]", color="#dcdcdc", fontsize=9)
        ax1.tick_params(colors="#b0b0b0", labelsize=8)
        ax1.grid(True, ls=":", color="#32323a")
        ax1.legend(loc="upper left", fontsize=8, facecolor="#202024", edgecolor="#3a3a42", labelcolor="#e0e0e0")

        # 2. Saturated Sand Flux q_sat vs Wind Shear Velocity u_* (Lettau & Lettau 1978)
        ax2 = self.fig_bench.add_subplot(gs[0, 1])
        ax2.set_facecolor("#141418")
        ax2.plot(bench["u_star_arr"], bench["q_sat_curve_g"], color="#ff9800", lw=2.2, label="Lettau-Bagnold Law q_sat(u*)")
        ax2.scatter(bench["exp_u_pts"], bench["exp_q_pts_g"], color="#ffffff", s=32, zorder=4, label="Wind-Tunnel Flux Data")
        tau_curr = env.fluid_density * (u_star ** 2)
        q_curr_g = (
            4.2 * (env.fluid_density / env.gravity) * (u_star ** 3) * max(0.0, 1.0 - env.threshold_shear_stress / max(1e-9, tau_curr)) * 1e3
        )
        ax2.scatter([u_star], [q_curr_g], color="#00e5ff", marker="*", s=160, zorder=5, label=f"Active Wind (u*={u_star:.2f} m/s)")
        ax2.set_title("2. Saturated Flux q_sat vs Shear Velocity u*", color="#f0a830", fontsize=10.5, fontweight="bold")
        ax2.set_xlabel("Wind Friction Velocity u* [m/s]", color="#dcdcdc", fontsize=9)
        ax2.set_ylabel("Saturated Flux q_sat [g/(m s)]", color="#dcdcdc", fontsize=9)
        ax2.tick_params(colors="#b0b0b0", labelsize=8)
        ax2.grid(True, ls=":", color="#32323a")
        ax2.legend(loc="upper left", fontsize=8, facecolor="#202024", edgecolor="#3a3a42", labelcolor="#e0e0e0")

        # 3. Linear Stability Dispersion Relation sigma(lambda) (Claudin & Andreotti 2006)
        ax3 = self.fig_bench.add_subplot(gs[1, 0])
        ax3.set_facecolor("#141418")
        A = float(self.spin_jh_A.value())
        B = float(self.spin_jh_B.value())
        _, lam_min = find_neutral_wavenumber(env, A=A, B=B)
        _, lam_max, _ = find_fastest_growing_wavenumber(env, tau_curr, A=A, B=B)
        lams = np.geomspace(max(0.005, 0.55 * lam_min), 12.0 * lam_max, 250)
        ks = 2.0 * math.pi / lams
        sigmas = linear_growth_rate(ks, env, tau_curr, A=A, B=B)
        ax3.plot(lams, sigmas, color="#2ecc71", lw=2.2, label=r"Linear Growth Rate $\sigma(\lambda)$")
        ax3.axhline(0.0, color="#888888", lw=0.9)
        ax3.axvline(lam_min, color="#e74c3c", ls=":", lw=1.4, label=f"Neutral Cutoff = {lam_min:.2f} m")
        ax3.axvline(lam_max, color="#f5a623", ls="--", lw=1.6, label=f"Fastest Mode = {lam_max:.2f} m")
        ax3.set_xscale("log")
        ax3.set_title("3. Primary Hydrodynamic Instability Dispersion sigma(lambda)", color="#f0a830", fontsize=10.5, fontweight="bold")
        ax3.set_xlabel("Bedform Wavelength lambda [m] (log)", color="#dcdcdc", fontsize=9)
        ax3.set_ylabel("Growth Rate sigma [s^-1]", color="#dcdcdc", fontsize=9)
        ax3.tick_params(colors="#b0b0b0", labelsize=8)
        ax3.grid(True, ls=":", color="#32323a")
        ax3.legend(loc="upper right", fontsize=8, facecolor="#202024", edgecolor="#3a3a42", labelcolor="#e0e0e0")

        # 4. Barchan Migration Speed Scaling c(H) ~ 1/H (Sauermann et al. 2000; Elbelrhiti et al. 2005)
        ax4 = self.fig_bench.add_subplot(gs[1, 1])
        ax4.set_facecolor("#141418")
        ax4.plot(bench["H_arr"], bench["c_yr_curve"], color="#e67e22", lw=2.2, label="Kinematic Scaling c(H) = q_brink / (rho_b H)")
        ax4.scatter(bench["exp_H_pts"], bench["exp_c_pts"], color="#ffffff", s=32, zorder=4, label="Field Barchan Observations")
        if state.max_height > 0.2 and c_m_yr > 0.0:
            ax4.scatter(
                [state.max_height],
                [c_m_yr],
                color="#00e5ff",
                marker="*",
                s=170,
                zorder=5,
                label=f"Simulated Dune (H={state.max_height:.2f}m, c={c_m_yr:.1f}m/yr)",
            )
        ax4.set_title("4. Barchan Migration Velocity c(H) vs Dune Height H", color="#f0a830", fontsize=10.5, fontweight="bold")
        ax4.set_xlabel("Dune Crest Height H [m]", color="#dcdcdc", fontsize=9)
        ax4.set_ylabel("Migration Speed c [m/year]", color="#dcdcdc", fontsize=9)
        ax4.tick_params(colors="#b0b0b0", labelsize=8)
        ax4.grid(True, ls=":", color="#32323a")
        ax4.legend(loc="upper right", fontsize=8, facecolor="#202024", edgecolor="#3a3a42", labelcolor="#e0e0e0")

        self.fig_bench.subplots_adjust(left=0.07, right=0.96, top=0.93, bottom=0.08, hspace=0.34, wspace=0.26)
        self.canvas_bench.draw_idle()

    # =========================================================================
    # PROJECT SAVE / LOAD (.desert) & EXPORTS
    # =========================================================================
    def save_project_to_file(self, filepath: str) -> None:
        if self.simulator is None:
            return
        st = self.simulator.get_state()
        meta = {
            "planet_idx": self.combo_planet.currentIndex(),
            "bed_preset_idx": self.combo_bed_preset.currentIndex(),
            "mound_height": float(self.spin_mound_height.value()),
            "mound_radius": float(self.spin_mound_radius.value()),
            "base_sand": float(self.spin_base_sand.value()),
            "noise_amp": float(self.spin_noise_amp.value()),
            "lx": float(self.spin_lx.value()),
            "ly": float(self.spin_ly.value()),
            "grid_res_idx": self.combo_grid_res.currentIndex(),
            "enforce_bedrock": bool(self.chk_bedrock.isChecked()),
            "periodic_x": bool(self.chk_periodic_x.isChecked()),
            "inflow_ratio": float(self.spin_inflow.value()),
            "grain_d_um": float(self.spin_grain_d_um.value()),
            "rho_s": float(self.spin_rho_s.value()),
            "rho_f": float(self.spin_rho_f.value()),
            "gravity": float(self.spin_gravity.value()),
            "u_star_0": float(self.spin_u_star.value()),
            "A": float(self.spin_jh_A.value()),
            "B": float(self.spin_jh_B.value()),
            "d_lat": float(self.spin_d_lat.value()),
            "time_s": float(st.time_s),
            "step_count": int(st.step_count),
        }
        with open(filepath, "wb") as f:
            np.savez_compressed(f, h=st.h, meta_json=np.array(json.dumps(meta)))
        self.current_project_path = filepath
        self.statusBar().showMessage(f"Saved Desert project to: {filepath}")

    def load_project_from_file(self, filepath: str) -> None:
        if self.sim_timer.isActive():
            self.toggle_run_pause()
        with np.load(filepath, allow_pickle=False) as data:
            h_loaded = np.array(data["h"], dtype=np.float64)
            meta = json.loads(str(data["meta_json"]))

        self.combo_planet.setCurrentIndex(int(meta.get("planet_idx", 0)))
        self.combo_bed_preset.setCurrentIndex(int(meta.get("bed_preset_idx", 0)))
        self.spin_mound_height.setValue(float(meta.get("mound_height", 3.2)))
        self.spin_mound_radius.setValue(float(meta.get("mound_radius", 18.0)))
        self.spin_base_sand.setValue(float(meta.get("base_sand", 0.0)))
        self.spin_noise_amp.setValue(float(meta.get("noise_amp", 0.0)))
        self.spin_lx.setValue(float(meta.get("lx", 120.0)))
        self.spin_ly.setValue(float(meta.get("ly", 100.0)))
        self.combo_grid_res.setCurrentIndex(int(meta.get("grid_res_idx", 1)))
        self.chk_bedrock.setChecked(bool(meta.get("enforce_bedrock", True)))
        self.chk_periodic_x.setChecked(bool(meta.get("periodic_x", False)))
        self.spin_inflow.setValue(float(meta.get("inflow_ratio", 0.12)))
        self.spin_grain_d_um.setValue(float(meta.get("grain_d_um", 200.0)))
        self.spin_rho_s.setValue(float(meta.get("rho_s", 2650.0)))
        self.spin_rho_f.setValue(float(meta.get("rho_f", 1.225)))
        self.spin_gravity.setValue(float(meta.get("gravity", 9.81)))
        self.spin_u_star.setValue(float(meta.get("u_star_0", 0.38)))
        self.spin_jh_A.setValue(float(meta.get("A", 3.2)))
        self.spin_jh_B.setValue(float(meta.get("B", 1.6)))
        self.spin_d_lat.setValue(float(meta.get("d_lat", 0.35)))

        nx, ny = h_loaded.shape
        env = self._build_current_environment()
        cfg = SimulationConfig2D(
            nx=nx,
            ny=ny,
            length_x_m=float(self.spin_lx.value()),
            length_y_m=float(self.spin_ly.value()),
            env=env,
            u_star_0=float(self.spin_u_star.value()),
            A=float(self.spin_jh_A.value()),
            B=float(self.spin_jh_B.value()),
            A_y=1.6,
            periodic_x=bool(self.chk_periodic_x.isChecked()),
            periodic_y=True,
            inflow_ratio=float(self.spin_inflow.value()),
            enforce_bedrock=bool(self.chk_bedrock.isChecked()),
            d_lat=float(self.spin_d_lat.value()),
        )
        self.simulator = DuneSimulator2D(config=cfg, h_init=h_loaded)
        self.simulator.time_s = float(meta.get("time_s", 0.0))
        self.simulator.step_count = int(meta.get("step_count", 0))
        st = self.simulator.get_state()
        self.state_history = [st]
        self._refresh_all_views(st)
        self.current_project_path = filepath
        self.statusBar().showMessage(f"Loaded Desert project from: {filepath}")

    def save_project_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Desert Project", "barchan_project.desert", "Desert Project Files (*.desert)"
        )
        if path:
            if not path.endswith(".desert"):
                path += ".desert"
            self.save_project_to_file(path)

    def load_project_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Desert Project", "", "Desert Project Files (*.desert);;All Files (*)"
        )
        if path and os.path.exists(path):
            self.load_project_from_file(path)

    def export_diagnostic_png_dialog(self) -> None:
        if not self.state_history:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Publication Diagnostic Figure", "desert_barchan_diagnostics.png", "PNG Images (*.png)"
        )
        if path:
            st = self.state_history[-1]
            dx = float(st.x[1] - st.x[0]) if len(st.x) > 1 else 1.0
            dy = float(st.y[1] - st.y[0]) if len(st.y) > 1 else 1.0
            fig = plot_barchan_evolution_diagnostics(self.state_history, dx=dx, dy=dy, save_path=path)
            import matplotlib.pyplot as plt
            plt.close(fig)
            self.statusBar().showMessage(f"Exported publication figure to: {path}")

    def export_wavefront_obj_dialog(self) -> None:
        if self.simulator is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export 3D Wavefront OBJ Mesh", "desert_dune.obj", "Wavefront OBJ (*.obj)"
        )
        if path:
            st = self.simulator.get_state()
            h = st.h
            nx, ny = h.shape
            X, Y = np.meshgrid(st.x, st.y, indexing="ij")
            verts = np.column_stack([X.ravel(), Y.ravel(), h.ravel()])
            from .viewport_3d import _build_grid_faces
            faces = _build_grid_faces(nx, ny) + 1
            with open(path, "w", encoding="utf-8") as f:
                f.write("# Desert Studio 3D Dune Surface Mesh" + NL)
                for v in verts:
                    f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}" + NL)
                for fc in faces:
                    f.write(f"f {fc[0]} {fc[1]} {fc[2]}" + NL)
            self.statusBar().showMessage(f"Exported 3D OBJ mesh to: {path}")

    def _on_main_viewport_sculpted(self, h_updated: np.ndarray) -> None:
        """Callback invoked when the user sculpts the 3D surface directly in the main 3D viewport."""
        if self.simulator is None:
            return
        self.simulator.h = h_updated
        self.simulator._update_diagnostics()
        st = self.simulator.get_state()
        if self.state_history:
            self.state_history[-1] = st
        else:
            self.state_history = [st]
        self._refresh_all_views(st)

    def open_custom_project_dialog(self) -> None:
        """Opens the interactive New Custom Project & 3D Initial Surface Sculptor dialog."""
        if self.sim_timer.isActive():
            self.toggle_run_pause()
        dlg = CustomProjectDialog(self, lang=self.lang)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.custom_h is not None and dlg.custom_config is not None:
            cfg = dlg.custom_config
            # Synchronize main window inspector controls with the user's custom project settings
            self.combo_planet.blockSignals(True)
            self.combo_planet.setCurrentIndex(dlg.planet_idx)
            self.combo_planet.blockSignals(False)

            self.spin_grain_d_um.blockSignals(True)
            self.spin_grain_d_um.setValue(cfg.env.grain_diameter * 1e6)
            self.spin_grain_d_um.blockSignals(False)

            self.spin_rho_s.blockSignals(True)
            self.spin_rho_s.setValue(cfg.env.grain_density)
            self.spin_rho_s.blockSignals(False)

            self.spin_rho_f.blockSignals(True)
            self.spin_rho_f.setValue(cfg.env.fluid_density)
            self.spin_rho_f.blockSignals(False)

            self.spin_gravity.blockSignals(True)
            self.spin_gravity.setValue(cfg.env.gravity)
            self.spin_gravity.blockSignals(False)

            self.spin_u_star.blockSignals(True)
            self.spin_u_star.setValue(cfg.u_star_0)
            self.spin_u_star.blockSignals(False)

            self.spin_lx.setValue(cfg.length_x_m)
            self.spin_ly.setValue(cfg.length_y_m)
            self.combo_grid_res.setCurrentIndex(dlg.grid_res_idx)
            self.chk_bedrock.setChecked(cfg.enforce_bedrock)
            self.spin_inflow.setValue(cfg.inflow_ratio)

            self._update_physical_readouts()
            self.simulator = DuneSimulator2D(config=cfg, h_init=dlg.custom_h)
            st0 = self.simulator.get_state()
            self.state_history = [st0]
            self.slider_timeline.blockSignals(True)
            self.slider_timeline.setRange(0, 0)
            self.slider_timeline.setValue(0)
            self.slider_timeline.blockSignals(False)
            self._refresh_all_views(st0)
            self.statusBar().showMessage(
                f"Loaded custom 3D sculpted project ({cfg.nx}x{cfg.ny}, d={cfg.env.grain_diameter*1e6:.0f} um, u*={cfg.u_star_0:.2f} m/s). Ready to Run!"
            )

    def show_about_dialog(self) -> None:
        dlg = ScientificHelpDialog(self, lang=self.lang)
        dlg.exec()


def launch_desert_app(show_splash: bool = True, auto_close_ms: Optional[int] = None) -> int:
    """Entry point to launch the Desert 3D Studio GUI application."""
    app = QApplication.instance() or QApplication([])
    app.setApplicationName("Desert")
    assets = ensure_gui_assets()
    if os.path.exists(assets.get("icon", "")):
        app.setWindowIcon(QIcon(assets["icon"]))

    selected_preset = "barchan"
    open_requested = False
    custom_requested = False

    if show_splash:
        splash = DesertSplashScreen(assets)
        if auto_close_ms is not None:
            QTimer.singleShot(auto_close_ms, splash.accept)
        splash.exec()
        selected_preset = splash.selected_preset
        open_requested = splash.open_file_requested
        custom_requested = splash.custom_project_requested

    win = DesertMainWindow(assets=assets)
    if selected_preset != "barchan":
        win.apply_quick_preset(selected_preset)
    win.show()

    if open_requested:
        QTimer.singleShot(100, win.load_project_dialog)
    elif custom_requested:
        QTimer.singleShot(100, win.open_custom_project_dialog)

    if auto_close_ms is not None:
        QTimer.singleShot(auto_close_ms, win.close)

    return app.exec()
