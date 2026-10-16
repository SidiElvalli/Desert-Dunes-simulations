"""Interactive 'New Custom Project & 3D Initial Surface Designer' dialog for the Desert Studio.

Supports full bilingual English ('en') and Technical Arabic ('ar') localization.
"""

from typing import Optional
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QSplitter,
    QGroupBox,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QCheckBox,
)

from ..core.constants import (
    PlanetaryEnvironment,
    EARTH_DESERT,
    MARS_DUNES,
    VENUS_DUNES,
    AQUEOUS_FLUME,
)
from ..core.wind_profile import compute_wind_profile, u_star_from_u10
from ..simulation.config import SimulationConfig2D
from ..simulation.simulator_2d import DuneSimulator2D
from .theme import BLENDER_DARK_QSS
from .viewport_3d import DesertViewport3D
from .i18n import TRANSLATIONS


class CustomProjectDialog(QDialog):
    """Blender-style New Custom Project studio with interactive 3D surface sculpting."""

    def __init__(self, parent: Optional[QWidget] = None, lang: str = "en"):
        super().__init__(parent)
        self.lang = lang if lang in TRANSLATIONS else "en"
        tr = TRANSLATIONS[self.lang]
        self.setWindowTitle(tr["cust_title"])
        self.resize(1380, 820)
        self.setStyleSheet(BLENDER_DARK_QSS)
        if self.lang == "ar":
            self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        self.custom_h: Optional[np.ndarray] = None
        self.custom_config: Optional[SimulationConfig2D] = None
        self.planet_idx: int = 0
        self.grid_res_idx: int = 1
        self._updating_wind: bool = False

        self._build_ui()
        self.rebuild_base_surface()

    def _build_ui(self) -> None:
        tr = TRANSLATIONS[self.lang]
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        banner = QLabel(tr["cust_banner"])
        banner.setStyleSheet(
            "background-color: #26211c; color: #f5b041; font-size: 13px; font-weight: bold; "
            "padding: 8px 12px; border: 1px solid #d97724; border-radius: 5px;"
        )
        root.addWidget(banner)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)

        # --- Left Column: Initial Condition Parameters & Surface Formula ---
        left_widget = QWidget(self)
        left_widget.setFixedWidth(440)
        l_vbox = QVBoxLayout(left_widget)
        l_vbox.setContentsMargins(0, 0, 4, 0)
        l_vbox.setSpacing(8)

        # 1. Surface Formula & Parametric Shape Builder
        grp_shape = QGroupBox(tr["cust_grp1"])
        f_shape = QFormLayout(grp_shape)
        f_shape.setSpacing(6)

        self.combo_template = QComboBox()
        self.combo_template.addItems(tr["cust_shape_items"])
        self.combo_template.currentIndexChanged.connect(self._on_template_changed)
        f_shape.addRow(tr["cust_lbl_shape"], self.combo_template)

        self.spin_h0 = QDoubleSpinBox()
        self.spin_h0.setRange(0.0, 20.0)
        self.spin_h0.setValue(3.2)
        self.spin_h0.setSuffix(" m")
        f_shape.addRow(tr["cust_lbl_h0"], self.spin_h0)

        self.spin_rx = QDoubleSpinBox()
        self.spin_rx.setRange(3.0, 80.0)
        self.spin_rx.setValue(18.0)
        self.spin_rx.setSuffix(" m")
        f_shape.addRow(tr["cust_lbl_rx"], self.spin_rx)

        self.spin_ry = QDoubleSpinBox()
        self.spin_ry.setRange(3.0, 80.0)
        self.spin_ry.setValue(18.0)
        self.spin_ry.setSuffix(" m")
        f_shape.addRow(tr["cust_lbl_ry"], self.spin_ry)

        self.spin_base = QDoubleSpinBox()
        self.spin_base.setRange(0.0, 10.0)
        self.spin_base.setSingleStep(0.1)
        self.spin_base.setValue(0.0)
        self.spin_base.setSuffix(" m")
        f_shape.addRow(tr["cust_lbl_base"], self.spin_base)

        self.edit_formula = QLineEdit(
            "base + H0 * np.maximum(0.0, 1.0 - ((x - 0.30*Lx)/Rx)**2 - ((y - 0.50*Ly)/Ry)**2)"
        )
        self.edit_formula.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        f_shape.addRow(tr["cust_lbl_formula"], self.edit_formula)

        btn_eval = QPushButton(tr["cust_btn_eval"])
        btn_eval.clicked.connect(self.rebuild_base_surface)
        f_shape.addRow(btn_eval)

        l_vbox.addWidget(grp_shape)

        # 2. Grain Size & Planetary Environment
        grp_grain = QGroupBox(tr["cust_grp2"])
        f_grain = QFormLayout(grp_grain)
        f_grain.setSpacing(6)

        self.combo_planet = QComboBox()
        self.combo_planet.addItems(tr["planet_items"][:4])
        self.combo_planet.currentIndexChanged.connect(self._on_planet_changed)
        f_grain.addRow(tr["lbl_env_preset"], self.combo_planet)

        self.spin_d_um = QDoubleSpinBox()
        self.spin_d_um.setRange(40.0, 2500.0)
        self.spin_d_um.setDecimals(0)
        self.spin_d_um.setSingleStep(10.0)
        self.spin_d_um.setValue(200.0)
        self.spin_d_um.setSuffix(" um")
        self.spin_d_um.valueChanged.connect(self._refresh_preview_physics)
        f_grain.addRow(tr["lbl_grain_d"], self.spin_d_um)

        self.spin_rho_s = QDoubleSpinBox()
        self.spin_rho_s.setRange(500.0, 8000.0)
        self.spin_rho_s.setValue(2650.0)
        self.spin_rho_s.setSuffix(" kg/m^3")
        f_grain.addRow(tr["lbl_rho_s"], self.spin_rho_s)

        self.spin_rho_f = QDoubleSpinBox()
        self.spin_rho_f.setRange(0.005, 2000.0)
        self.spin_rho_f.setDecimals(3)
        self.spin_rho_f.setValue(1.225)
        self.spin_rho_f.setSuffix(" kg/m^3")
        f_grain.addRow(tr["lbl_rho_f"], self.spin_rho_f)

        self.spin_g = QDoubleSpinBox()
        self.spin_g.setRange(0.5, 30.0)
        self.spin_g.setValue(9.81)
        self.spin_g.setSuffix(" m/s^2")
        f_grain.addRow(tr["lbl_gravity"], self.spin_g)

        l_vbox.addWidget(grp_grain)

        # 3. Wind Profile & Domain Settings
        grp_wind = QGroupBox(tr["cust_grp3"])
        f_wind = QFormLayout(grp_wind)
        f_wind.setSpacing(6)

        self.spin_u_star = QDoubleSpinBox()
        self.spin_u_star.setRange(0.05, 2.50)
        self.spin_u_star.setDecimals(3)
        self.spin_u_star.setSingleStep(0.02)
        self.spin_u_star.setValue(0.380)
        self.spin_u_star.setSuffix(" m/s")
        self.spin_u_star.valueChanged.connect(self._on_u_star_changed)
        f_wind.addRow(tr["lbl_u_star"], self.spin_u_star)

        self.spin_u10 = QDoubleSpinBox()
        self.spin_u10.setRange(0.5, 45.0)
        self.spin_u10.setDecimals(2)
        self.spin_u10.setValue(12.20)
        self.spin_u10.setSuffix(" m/s")
        self.spin_u10.valueChanged.connect(self._on_u10_changed)
        f_wind.addRow(tr["lbl_u10"], self.spin_u10)

        self.spin_lx = QDoubleSpinBox()
        self.spin_lx.setRange(40.0, 1000.0)
        self.spin_lx.setValue(120.0)
        self.spin_lx.setSuffix(" m")
        f_wind.addRow(tr["lbl_lx"], self.spin_lx)

        self.spin_ly = QDoubleSpinBox()
        self.spin_ly.setRange(40.0, 1000.0)
        self.spin_ly.setValue(100.0)
        self.spin_ly.setSuffix(" m")
        f_wind.addRow(tr["lbl_ly"], self.spin_ly)

        self.combo_res = QComboBox()
        self.combo_res.addItems(tr["grid_res_items"])
        self.combo_res.setCurrentIndex(1)
        f_wind.addRow(tr["lbl_grid_res"], self.combo_res)

        self.chk_bedrock = QCheckBox(tr["chk_bedrock"])
        self.chk_bedrock.setChecked(True)
        f_wind.addRow(self.chk_bedrock)

        self.spin_inflow = QDoubleSpinBox()
        self.spin_inflow.setRange(0.0, 1.0)
        self.spin_inflow.setSingleStep(0.02)
        self.spin_inflow.setValue(0.12)
        f_wind.addRow(tr["lbl_inflow"], self.spin_inflow)

        l_vbox.addWidget(grp_wind)
        l_vbox.addStretch(1)
        splitter.addWidget(left_widget)

        # --- Right Column: Interactive 3D Surface Sculpting Viewport ---
        right_widget = QWidget(self)
        r_vbox = QVBoxLayout(right_widget)
        r_vbox.setContentsMargins(4, 0, 0, 0)
        r_vbox.setSpacing(6)

        self.sculpt_viewport = DesertViewport3D(self)
        self.sculpt_viewport.setLayoutDirection(Qt.LayoutDirection.LeftToRight)
        self.sculpt_viewport.set_language(self.lang)
        self.sculpt_viewport.set_sculpt_mode(True)
        self.sculpt_viewport.on_surface_sculpted = self._on_3d_surface_sculpted
        r_vbox.addWidget(self.sculpt_viewport, stretch=1)

        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # --- Bottom Action Buttons ---
        btn_row = QHBoxLayout()
        btn_cancel = QPushButton(tr["cust_btn_cancel"])
        btn_cancel.clicked.connect(self.reject)
        btn_row.addWidget(btn_cancel)

        btn_row.addStretch(1)

        btn_create = QPushButton(tr["cust_btn_create"])
        btn_create.setObjectName("PrimaryActionBtn")
        btn_create.setMinimumWidth(320)
        btn_create.clicked.connect(self._on_accept_project)
        btn_row.addWidget(btn_create)

        root.addLayout(btn_row)

    def _build_env(self) -> PlanetaryEnvironment:
        return PlanetaryEnvironment(
            name=self.combo_planet.currentText(),
            gravity=float(self.spin_g.value()),
            fluid_density=float(self.spin_rho_f.value()),
            grain_density=float(self.spin_rho_s.value()),
            grain_diameter=float(self.spin_d_um.value()) * 1e-6,
            u_star_0=float(self.spin_u_star.value()),
        )

    def _on_planet_changed(self, idx: int) -> None:
        presets = [EARTH_DESERT, MARS_DUNES, VENUS_DUNES, AQUEOUS_FLUME]
        if 0 <= idx < len(presets):
            p = presets[idx]
            self.spin_d_um.setValue(p.grain_diameter * 1e6)
            self.spin_rho_s.setValue(p.grain_density)
            self.spin_rho_f.setValue(p.fluid_density)
            self.spin_g.setValue(p.gravity)
            self.spin_u_star.setValue(0.38 if idx == 0 else p.u_star_0)
        self._refresh_preview_physics()

    def _on_u_star_changed(self, val: float) -> None:
        if self._updating_wind:
            return
        self._updating_wind = True
        prof = compute_wind_profile(self._build_env(), val)
        self.spin_u10.setValue(prof.u_10m)
        self._updating_wind = False
        self._refresh_preview_physics()

    def _on_u10_changed(self, u10: float) -> None:
        if self._updating_wind:
            return
        self._updating_wind = True
        u_star = u_star_from_u10(self._build_env(), u10)
        self.spin_u_star.setValue(u_star)
        self._updating_wind = False
        self._refresh_preview_physics()

    def _on_template_changed(self, idx: int) -> None:
        formulas = [
            "base + H0 * np.maximum(0.0, 1.0 - ((x - 0.30*Lx)/Rx)**2 - ((y - 0.50*Ly)/Ry)**2)",
            "np.full_like(x, base)",
            "base + np.maximum(0.0, H0*(1 - ((x-0.32*Lx)/Rx)**2 - ((y-0.5*Ly)/Ry)**2) - 0.65*H0*np.maximum(0.0, 1 - ((x-0.42*Lx)/(0.7*Rx))**2 - ((y-0.5*Ly)/(0.6*Ry))**2))",
            "base + 0.5*H0 * (1.0 + np.sin(2*np.pi*x/Rx) * np.cos(2*np.pi*y/(2*Ry)))",
            "base + H0 * np.exp(-((x - 0.30*Lx)/Rx)**2 - ((y - 0.50*Ly)/Ry)**2)",
        ]
        if 0 <= idx < len(formulas):
            self.edit_formula.setText(formulas[idx])
        self.rebuild_base_surface()

    def rebuild_base_surface(self) -> None:
        res_options = [(96, 80), (120, 100), (160, 128)]
        r_idx = self.combo_res.currentIndex()
        nx, ny = res_options[r_idx] if 0 <= r_idx < len(res_options) else (120, 100)
        lx = float(self.spin_lx.value())
        ly = float(self.spin_ly.value())
        dx, dy = lx / nx, ly / ny
        x_1d = np.arange(nx) * dx
        y_1d = np.arange(ny) * dy
        X, Y = np.meshgrid(x_1d, y_1d, indexing="ij")

        H0 = float(self.spin_h0.value())
        Rx = max(1.0, float(self.spin_rx.value()))
        Ry = max(1.0, float(self.spin_ry.value()))
        base = float(self.spin_base.value())

        safe_ns = {
            "np": np,
            "sin": np.sin,
            "cos": np.cos,
            "exp": np.exp,
            "sqrt": np.sqrt,
            "maximum": np.maximum,
            "minimum": np.minimum,
            "x": X,
            "y": Y,
            "Lx": lx,
            "Ly": ly,
            "H0": H0,
            "Rx": Rx,
            "Ry": Ry,
            "base": base,
        }
        try:
            h_eval = eval(self.edit_formula.text(), {"__builtins__": {}}, safe_ns)
            h_arr = np.asarray(h_eval, dtype=np.float64)
            if h_arr.shape != (nx, ny):
                h_arr = np.broadcast_to(h_arr, (nx, ny)).copy()
            h_arr = np.maximum(0.0, h_arr)
        except Exception:
            h_arr = base + H0 * np.maximum(0.0, 1.0 - ((X - 0.30 * lx) / Rx) ** 2 - ((Y - 0.50 * ly) / Ry) ** 2)

        self.custom_h = h_arr
        self._refresh_preview_physics()

    def _on_3d_surface_sculpted(self, h_updated: np.ndarray) -> None:
        self.custom_h = h_updated
        self._refresh_preview_physics()

    def _refresh_preview_physics(self) -> None:
        if self.custom_h is None:
            return
        nx, ny = self.custom_h.shape
        env = self._build_env()
        cfg = SimulationConfig2D(
            nx=nx,
            ny=ny,
            length_x_m=float(self.spin_lx.value()),
            length_y_m=float(self.spin_ly.value()),
            env=env,
            u_star_0=float(self.spin_u_star.value()),
            periodic_x=False,
            periodic_y=True,
            inflow_ratio=float(self.spin_inflow.value()),
            enforce_bedrock=bool(self.chk_bedrock.isChecked()),
            cfl=0.45,
            dt_max=14400.0,
        )
        self.custom_config = cfg
        sim_preview = DuneSimulator2D(config=cfg, h_init=self.custom_h)
        st = sim_preview.get_state()
        self.sculpt_viewport.update_from_state(st, env, cfg.u_star_0)

    def _on_accept_project(self) -> None:
        self.planet_idx = self.combo_planet.currentIndex()
        self.grid_res_idx = self.combo_res.currentIndex()
        self._refresh_preview_physics()
        self.accept()
