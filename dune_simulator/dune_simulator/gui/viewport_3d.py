"""Hardware-accelerated OpenGL 3D Desert Diorama Viewport with:
  - Real-time ray-marched directional sun shadows
  - Multi-scale heightfield ambient occlusion (HBAO)
  - Two-tone Sahara golden-stoss / coral-terracotta-slipface slope shading
  - Solid 3D geological cutaway diorama pedestal walls
  - Superimposed ballistic impact ripples & interactive 3D ray-cast surface sculpting
"""

from typing import Optional, Callable
import math
import numpy as np
from scipy import ndimage
from PyQt6.QtCore import Qt, QPointF, QCoreApplication
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
)
import pyqtgraph.opengl as gl
from matplotlib import cm

from ..core.constants import PlanetaryEnvironment
from ..core.wind_profile import (
    synthesize_multiscale_desert_surface,
    synthesize_zoomed_ripple_patch,
    compute_impact_ripple_scales,
)
from ..simulation.config import SimulationState2D
from .i18n import TRANSLATIONS

QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)

NL = chr(10)


def _build_grid_faces(nx: int, ny: int) -> np.ndarray:
    """Builds a vectorized triangle face index array (2*(nx-1)*(ny-1), 3) for a regular (nx, ny) grid."""
    idx = np.arange(nx * ny, dtype=np.int32).reshape(nx, ny)
    v00 = idx[:-1, :-1].ravel()
    v10 = idx[1:, :-1].ravel()
    v01 = idx[:-1, 1:].ravel()
    v11 = idx[1:, 1:].ravel()
    f1 = np.column_stack([v00, v10, v11])
    f2 = np.column_stack([v00, v11, v01])
    return np.vstack([f1, f2])


def _build_diorama_skirt_mesh(
    x_hr: np.ndarray,
    y_hr: np.ndarray,
    z_top: np.ndarray,
    lx: float,
    ly: float,
    z_base: float = -4.2,
) -> tuple:
    """Builds solid 3D geological cutaway side walls (4 vertical skirts) around the terrain perimeter."""
    nx, ny = z_top.shape
    X = x_hr - 0.5 * lx
    Y = y_hr - 0.5 * ly

    verts_list = []
    faces_list = []
    colors_list = []
    v_offset = 0

    # Define the 4 perimeter walls: (top_x, top_y, top_z, shade_factor)
    walls = [
        # Front wall (x = X[-1], y along Y)
        (np.full(ny, X[-1]), Y, z_top[-1, :], 0.72),
        # Back wall (x = X[0], y along Y)
        (np.full(ny, X[0]), Y, z_top[0, :], 0.88),
        # Right wall (x along X, y = Y[-1])
        (X, np.full(nx, Y[-1]), z_top[:, -1], 0.64),
        # Left wall (x along X, y = Y[0])
        (X, np.full(nx, Y[0]), z_top[:, 0], 0.92),
    ]

    for wx, wy, wz, wall_light in walls:
        m = len(wx)
        top_v = np.column_stack([wx, wy, wz])
        bot_v = np.column_stack([wx, wy, np.full(m, z_base)])
        w_verts = np.vstack([top_v, bot_v]).astype(np.float32)

        idx = np.arange(m - 1, dtype=np.int32)
        t0 = v_offset + idx
        t1 = v_offset + idx + 1
        b0 = v_offset + m + idx
        b1 = v_offset + m + idx + 1
        w_faces = np.vstack([np.column_stack([t0, b0, b1]), np.column_stack([t0, b1, t1])])

        # Warm sandstone cutaway coloring: golden sand rim at top, deeper terracotta bedrock at bottom
        top_col = np.tile(
            np.array([0.76 * wall_light, 0.54 * wall_light, 0.34 * wall_light, 1.0], dtype=np.float32),
            (m, 1),
        )
        bot_col = np.tile(
            np.array([0.42 * wall_light, 0.28 * wall_light, 0.19 * wall_light, 1.0], dtype=np.float32),
            (m, 1),
        )
        w_cols = np.vstack([top_col, bot_col])

        verts_list.append(w_verts)
        faces_list.append(w_faces)
        colors_list.append(w_cols)
        v_offset += 2 * m

    return (
        np.vstack(verts_list),
        np.vstack(faces_list),
        np.vstack(colors_list),
    )


class SculptableGLViewWidget(gl.GLViewWidget):
    """Subclass of GLViewWidget supporting 3D ray-plane intersection for direct 3D sand sculpting."""

    def __init__(self, viewport: "DesertViewport3D", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.viewport = viewport
        self.setMouseTracking(True)
        self._is_sculpting_stroke: bool = False

    def _ray_intersect_plane_z0(self, pos: QPointF) -> Optional[tuple]:
        w = max(1, self.width())
        h = max(1, self.height())
        ndc_x = (2.0 * pos.x() / w) - 1.0
        ndc_y = 1.0 - (2.0 * pos.y() / h)

        fov_rad = math.radians(self.opts.get("fov", 60.0))
        tan_half_fov = math.tan(0.5 * fov_rad)
        aspect = w / h

        el = math.radians(self.opts.get("elevation", 30.0))
        az = math.radians(self.opts.get("azimuth", 38.0))
        dist = float(self.opts.get("distance", 102.0))
        center = self.opts.get("center")
        cx = float(center.x()) if center is not None else 0.0
        cy = float(center.y()) if center is not None else 0.0
        cz = float(center.z()) if center is not None else 0.0

        cam_x = cx + dist * math.cos(el) * math.cos(az)
        cam_y = cy + dist * math.cos(el) * math.sin(az)
        cam_z = cz + dist * math.sin(el)

        fwd = np.array([cx - cam_x, cy - cam_y, cz - cam_z], dtype=np.float64)
        fwd_norm = np.linalg.norm(fwd)
        if fwd_norm < 1e-9:
            return None
        fwd /= fwd_norm

        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        right = np.cross(fwd, world_up)
        r_norm = np.linalg.norm(right)
        if r_norm < 1e-6:
            right = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            right /= r_norm
        up = np.cross(right, fwd)

        ray_dir = fwd + (ndc_x * aspect * tan_half_fov) * right + (ndc_y * tan_half_fov) * up
        ray_dir /= np.linalg.norm(ray_dir)

        if abs(ray_dir[2]) < 1e-5:
            return None
        t = -cam_z / ray_dir[2]
        if t <= 0.0:
            return None
        hit_x = cam_x + t * ray_dir[0]
        hit_y = cam_y + t * ray_dir[1]
        return float(hit_x), float(hit_y)

    def mousePressEvent(self, ev: QMouseEvent) -> None:
        if (
            self.viewport.sculpt_mode
            and ev.button() == Qt.MouseButton.LeftButton
            and not (ev.modifiers() & (Qt.KeyboardModifier.AltModifier | Qt.KeyboardModifier.ControlModifier))
        ):
            self._is_sculpting_stroke = True
            hit = self._ray_intersect_plane_z0(ev.position())
            if hit is not None:
                self.viewport.apply_sculpt_brush_at_gl_coords(hit[0], hit[1])
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev: QMouseEvent) -> None:
        if self.viewport.sculpt_mode:
            hit = self._ray_intersect_plane_z0(ev.position())
            if hit is not None:
                self.viewport.update_brush_cursor_gl(hit[0], hit[1])
                if self._is_sculpting_stroke and (ev.buttons() & Qt.MouseButton.LeftButton):
                    self.viewport.apply_sculpt_brush_at_gl_coords(hit[0], hit[1])
                    ev.accept()
                    return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev: QMouseEvent) -> None:
        if self._is_sculpting_stroke and ev.button() == Qt.MouseButton.LeftButton:
            self._is_sculpting_stroke = False
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class DesertViewport3D(QWidget):
    """Blender-inspired 3D OpenGL viewport with SimpleWindErosion-grade diorama shadow/HBAO shading."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.lang: str = "en"
        self.show_ripples: bool = True
        self.show_diorama_skirt: bool = True
        self.show_shadows_and_ao: bool = True
        self.show_wake_dome: bool = False
        self.show_brink_line: bool = False
        self.show_saltation_cloud: bool = False
        self.show_wind_streamlines: bool = False
        self.show_wireframe: bool = False
        self.micro_ripple_zoom_mode: bool = False

        # Interactive 3D Sculpt Mode state
        self.sculpt_mode: bool = False
        self.sculpt_tool: str = "raise"
        self.sculpt_radius_m: float = 12.0
        self.sculpt_strength_m: float = 0.45
        self.on_surface_sculpted: Optional[Callable[[np.ndarray], None]] = None

        self.shader_mode: str = "desert"
        self.sun_azimuth_deg: float = 140.0
        self.sun_elevation_deg: float = 28.0
        self.vertical_exaggeration: float = 3.5
        self.ripple_exaggeration: float = 5.5
        self.upsample_factor: int = 3

        self._cached_faces: Optional[np.ndarray] = None
        self._cached_shape: tuple = (0, 0)
        self._wake_faces: Optional[np.ndarray] = None
        self._wake_shape: tuple = (0, 0)
        self._last_state: Optional[SimulationState2D] = None
        self._last_env: Optional[PlanetaryEnvironment] = None
        self._last_u_star: float = 0.38
        self._last_bedrock_floor: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(101)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # --- Top Viewport Header Bar ---
        header = QFrame(self)
        header.setStyleSheet(
            "QFrame { background-color: #232323; border-bottom: 1px solid #141414; }"
            "QPushButton { padding: 3px 8px; font-size: 11px; background-color: #303030; }"
            "QPushButton:checked { background-color: #d97724; color: #ffffff; border: 1px solid #f5a623; }"
        )
        h_layout = QHBoxLayout(header)
        h_layout.setContentsMargins(8, 3, 8, 3)
        h_layout.setSpacing(5)

        self.mode_badge = QLabel(TRANSLATIONS["en"]["vp_badge_default"])
        self.mode_badge.setStyleSheet("color: #f0a830; font-weight: bold; font-size: 11px;")
        h_layout.addWidget(self.mode_badge)
        h_layout.addStretch(1)

        self.btn_sculpt_mode = QPushButton(TRANSLATIONS["en"]["vp_btn_sculpt"])
        self.btn_sculpt_mode.setCheckable(True)
        self.btn_sculpt_mode.toggled.connect(self.set_sculpt_mode)
        h_layout.addWidget(self.btn_sculpt_mode)

        self.btn_lee = QPushButton(TRANSLATIONS["en"]["vp_btn_lee"])
        self.btn_lee.clicked.connect(self.view_leeward_bay)
        h_layout.addWidget(self.btn_lee)

        self.btn_stoss = QPushButton(TRANSLATIONS["en"]["vp_btn_stoss"])
        self.btn_stoss.clicked.connect(self.view_windward_stoss)
        h_layout.addWidget(self.btn_stoss)

        self.btn_top = QPushButton(TRANSLATIONS["en"]["vp_btn_top"])
        self.btn_top.clicked.connect(self.view_top_planform)
        h_layout.addWidget(self.btn_top)

        self.btn_ripple_zoom = QPushButton(TRANSLATIONS["en"]["vp_btn_micro"])
        self.btn_ripple_zoom.setCheckable(True)
        self.btn_ripple_zoom.toggled.connect(self.set_micro_ripple_zoom_mode)
        h_layout.addWidget(self.btn_ripple_zoom)

        layout.addWidget(header)

        # --- Collapsible 3D Sculpt Toolbar ---
        self.sculpt_bar = QFrame(self)
        self.sculpt_bar.setStyleSheet(
            "QFrame { background-color: #2b241d; border-bottom: 1px solid #d97724; }"
            "QLabel { color: #f5cba7; font-size: 11px; font-weight: 600; }"
        )
        s_layout = QHBoxLayout(self.sculpt_bar)
        s_layout.setContentsMargins(10, 4, 10, 4)
        s_layout.setSpacing(8)

        self.lbl_sculpt_brush = QLabel(TRANSLATIONS["en"]["sculpt_lbl_brush"])
        s_layout.addWidget(self.lbl_sculpt_brush)
        self.combo_brush = QComboBox()
        self.combo_brush.addItems(TRANSLATIONS["en"]["sculpt_brush_items"])
        self.combo_brush.currentIndexChanged.connect(self._on_brush_settings_changed)
        s_layout.addWidget(self.combo_brush)

        self.lbl_sculpt_radius = QLabel(TRANSLATIONS["en"]["sculpt_lbl_radius"])
        s_layout.addWidget(self.lbl_sculpt_radius)
        self.spin_brush_r = QDoubleSpinBox()
        self.spin_brush_r.setRange(2.0, 45.0)
        self.spin_brush_r.setSingleStep(1.0)
        self.spin_brush_r.setValue(12.0)
        self.spin_brush_r.valueChanged.connect(self._on_brush_settings_changed)
        s_layout.addWidget(self.spin_brush_r)

        self.lbl_sculpt_strength = QLabel(TRANSLATIONS["en"]["sculpt_lbl_strength"])
        s_layout.addWidget(self.lbl_sculpt_strength)
        self.spin_brush_s = QDoubleSpinBox()
        self.spin_brush_s.setRange(0.05, 5.0)
        self.spin_brush_s.setSingleStep(0.1)
        self.spin_brush_s.setValue(0.45)
        self.spin_brush_s.valueChanged.connect(self._on_brush_settings_changed)
        s_layout.addWidget(self.spin_brush_s)

        self.btn_clear_bed = QPushButton(TRANSLATIONS["en"]["sculpt_btn_clear"])
        self.btn_clear_bed.clicked.connect(self._clear_sculpted_bed)
        s_layout.addWidget(self.btn_clear_bed)

        s_layout.addStretch(1)
        self.lbl_sculpt_hint = QLabel(TRANSLATIONS["en"]["sculpt_lbl_hint"])
        s_layout.addWidget(self.lbl_sculpt_hint)
        self.sculpt_bar.setVisible(False)
        layout.addWidget(self.sculpt_bar)

        # --- OpenGL 3D View Widget (Atmospheric Slate-Teal Horizon Background like SimpleWindErosion) ---
        self.gl_view = SculptableGLViewWidget(self, self)
        self.gl_view.setBackgroundColor((66, 94, 108, 255))
        layout.addWidget(self.gl_view, stretch=1)

        # Main Terrain Mesh Item
        self.terrain_mesh = gl.GLMeshItem(smooth=True, computeNormals=False, drawEdges=False)
        self.gl_view.addItem(self.terrain_mesh)

        # Solid 3D Geological Cutaway Diorama Skirt Mesh
        self.skirt_mesh = gl.GLMeshItem(smooth=False, computeNormals=False, drawEdges=False)
        self.gl_view.addItem(self.skirt_mesh)

        # Optional Separation Bubble Translucent Streamline Dome
        self.wake_mesh = gl.GLMeshItem(smooth=True, computeNormals=False, drawEdges=False)
        self.wake_mesh.setGLOptions("translucent")
        self.wake_mesh.setVisible(False)
        self.gl_view.addItem(self.wake_mesh)

        # Optional Brink Crestline Curve
        self.brink_line = gl.GLLinePlotItem(color=(1.0, 0.28, 0.15, 0.90), width=2.5, antialias=True)
        self.brink_line.setVisible(False)
        self.gl_view.addItem(self.brink_line)

        # Optional Saltating Grain Cloud
        self.saltation_scatter = gl.GLScatterPlotItem(size=3.0, pxMode=True)
        self.saltation_scatter.setGLOptions("translucent")
        self.saltation_scatter.setVisible(False)
        self.gl_view.addItem(self.saltation_scatter)

        # Optional Wind Streamline Ribbons
        self.streamline_items = []
        for _ in range(7):
            line = gl.GLLinePlotItem(color=(0.35, 0.88, 1.0, 0.50), width=1.5, antialias=True)
            line.setVisible(False)
            self.gl_view.addItem(line)
            self.streamline_items.append(line)

        # 3D Sculpt Brush Ring Cursor
        self.brush_ring = gl.GLLinePlotItem(color=(1.0, 0.85, 0.20, 0.98), width=3.0, antialias=True)
        self.brush_ring.setVisible(False)
        self.gl_view.addItem(self.brush_ring)

        # Bottom-left HUD overlay bar
        self.hud_label = QLabel(self.gl_view)
        self.hud_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.hud_label.setStyleSheet(
            "background-color: rgba(22, 28, 34, 195); color: #f5ece0; "
            "border: 1px solid #5c6e7a; border-radius: 5px; padding: 6px 10px; "
            "font-family: 'Consolas', monospace; font-size: 11px;"
        )
        self.hud_label.move(14, 14)
        self.hud_label.resize(760, 62)

        self.view_leeward_bay()

    def set_language(self, lang: str) -> None:
        self.lang = lang if lang in TRANSLATIONS else "en"
        tr = TRANSLATIONS[self.lang]
        self.btn_sculpt_mode.setText(tr["vp_btn_sculpt"])
        self.btn_lee.setText(tr["vp_btn_lee"])
        self.btn_stoss.setText(tr["vp_btn_stoss"])
        self.btn_top.setText(tr["vp_btn_top"])
        self.btn_ripple_zoom.setText(tr["vp_btn_micro"])

        self.lbl_sculpt_brush.setText(tr["sculpt_lbl_brush"])
        cur_b = self.combo_brush.currentIndex()
        self.combo_brush.blockSignals(True)
        self.combo_brush.clear()
        self.combo_brush.addItems(tr["sculpt_brush_items"])
        self.combo_brush.setCurrentIndex(max(0, cur_b))
        self.combo_brush.blockSignals(False)
        self.lbl_sculpt_radius.setText(tr["sculpt_lbl_radius"])
        self.lbl_sculpt_strength.setText(tr["sculpt_lbl_strength"])
        self.btn_clear_bed.setText(tr["sculpt_btn_clear"])
        self.lbl_sculpt_hint.setText(tr["sculpt_lbl_hint"])

        if self.micro_ripple_zoom_mode:
            self.mode_badge.setText(tr["vp_badge_micro"])
        elif self.sculpt_mode:
            self.mode_badge.setText(tr["vp_badge_sculpt"])
        else:
            self.mode_badge.setText(tr["vp_badge_default"])

        if self._last_state is not None and self._last_env is not None:
            self.update_from_state(
                self._last_state, self._last_env, self._last_u_star, self._last_bedrock_floor
            )

    def set_sculpt_mode(self, enabled: bool) -> None:
        self.sculpt_mode = bool(enabled)
        self.btn_sculpt_mode.blockSignals(True)
        self.btn_sculpt_mode.setChecked(self.sculpt_mode)
        self.btn_sculpt_mode.blockSignals(False)
        self.sculpt_bar.setVisible(self.sculpt_mode)
        self.brush_ring.setVisible(self.sculpt_mode)
        tr = TRANSLATIONS[self.lang]
        self.mode_badge.setText(tr["vp_badge_sculpt"] if self.sculpt_mode else tr["vp_badge_default"])

    def _on_brush_settings_changed(self) -> None:
        tools = ["raise", "lower", "barchan_stamp", "ridge", "smooth", "flatten"]
        idx = self.combo_brush.currentIndex()
        self.sculpt_tool = tools[idx] if 0 <= idx < len(tools) else "raise"
        self.sculpt_radius_m = float(self.spin_brush_r.value())
        self.sculpt_strength_m = float(self.spin_brush_s.value())

    def _clear_sculpted_bed(self) -> None:
        if self._last_state is None or self.on_surface_sculpted is None:
            return
        h_new = np.zeros_like(self._last_state.h)
        self.on_surface_sculpted(h_new)

    def update_brush_cursor_gl(self, gx: float, gy: float) -> None:
        if not self.sculpt_mode or self._last_state is None:
            self.brush_ring.setVisible(False)
            return
        st = self._last_state
        nx, ny = st.h.shape
        dx = float(st.x[1] - st.x[0]) if nx > 1 else 1.0
        dy = float(st.y[1] - st.y[0]) if ny > 1 else 1.0
        lx, ly = nx * dx, ny * dy

        theta = np.linspace(0.0, 2.0 * math.pi, 48)
        r = self.sculpt_radius_m
        rx = gx + r * np.cos(theta)
        ry = gy + r * np.sin(theta)

        ix = np.clip(np.round((rx + 0.5 * lx) / dx).astype(int), 0, nx - 1)
        iy = np.clip(np.round((ry + 0.5 * ly) / dy).astype(int), 0, ny - 1)
        rz = st.h[ix, iy] * self.vertical_exaggeration + 0.30

        pts = np.column_stack([rx, ry, rz]).astype(np.float32)
        self.brush_ring.setData(pos=pts)
        self.brush_ring.setVisible(True)

    def apply_sculpt_brush_at_gl_coords(self, gx: float, gy: float) -> None:
        if self._last_state is None or self.on_surface_sculpted is None:
            return
        st = self._last_state
        h = st.h.copy()
        nx, ny = h.shape
        dx = float(st.x[1] - st.x[0]) if nx > 1 else 1.0
        dy = float(st.y[1] - st.y[0]) if ny > 1 else 1.0
        lx, ly = nx * dx, ny * dy

        cx = gx + 0.5 * lx
        cy = gy + 0.5 * ly
        if not (0.0 <= cx <= lx and 0.0 <= cy <= ly):
            return

        X, Y = np.meshgrid(st.x, st.y, indexing="ij")
        r = max(1.5, self.sculpt_radius_m)
        s = self.sculpt_strength_m
        dist_sq = ((X - cx) / r) ** 2 + ((Y - cy) / r) ** 2

        if self.sculpt_tool == "raise":
            kernel = np.where(dist_sq < 1.0, (1.0 - dist_sq) ** 2, 0.0)
            h = h + s * kernel
        elif self.sculpt_tool == "lower":
            kernel = np.where(dist_sq < 1.0, (1.0 - dist_sq) ** 2, 0.0)
            h = np.maximum(0.0, h - s * kernel)
        elif self.sculpt_tool == "barchan_stamp":
            r_main = ((X - cx) / r) ** 2 + ((Y - cy) / (1.1 * r)) ** 2
            r_lee = ((X - (cx + 0.55 * r)) / (0.75 * r)) ** 2 + ((Y - cy) / (0.65 * r)) ** 2
            mound = np.where(r_main < 1.0, (2.5 * s) * (1.0 - r_main) - (1.4 * s) * np.maximum(0.0, 1.0 - r_lee), 0.0)
            h = np.maximum(h, np.maximum(0.0, mound))
        elif self.sculpt_tool == "ridge":
            rx_sq = ((X - cx) / (0.55 * r)) ** 2
            ry_sq = ((Y - cy) / (1.6 * r)) ** 2
            kernel = np.where(rx_sq + ry_sq < 1.0, np.exp(-2.5 * rx_sq) * (1.0 - ry_sq), 0.0)
            h = h + s * kernel
        elif self.sculpt_tool == "smooth":
            h_smooth = ndimage.gaussian_filter(h, sigma=1.8)
            mask = dist_sq < 1.0
            w = np.where(mask, 1.0 - dist_sq, 0.0)
            h = h * (1.0 - 0.6 * w) + h_smooth * (0.6 * w)
        elif self.sculpt_tool == "flatten":
            target_h = float(s * 4.0)
            w = np.where(dist_sq < 1.0, (1.0 - dist_sq) ** 1.5, 0.0)
            h = h * (1.0 - 0.5 * w) + target_h * (0.5 * w)

        self.on_surface_sculpted(np.maximum(0.0, h))
        self.update_brush_cursor_gl(gx, gy)

    def view_leeward_bay(self) -> None:
        if self.micro_ripple_zoom_mode:
            self.btn_ripple_zoom.setChecked(False)
        self.gl_view.opts["distance"] = 104.0
        self.gl_view.opts["elevation"] = 30.0
        self.gl_view.opts["azimuth"] = 38.0
        self.gl_view.update()

    def view_windward_stoss(self) -> None:
        if self.micro_ripple_zoom_mode:
            self.btn_ripple_zoom.setChecked(False)
        self.gl_view.opts["distance"] = 104.0
        self.gl_view.opts["elevation"] = 30.0
        self.gl_view.opts["azimuth"] = 216.0
        self.gl_view.update()

    def view_top_planform(self) -> None:
        if self.micro_ripple_zoom_mode:
            self.btn_ripple_zoom.setChecked(False)
        self.gl_view.opts["distance"] = 122.0
        self.gl_view.opts["elevation"] = 84.0
        self.gl_view.opts["azimuth"] = 0.0
        self.gl_view.update()

    def zoom_close_to_stoss_ripples(self) -> None:
        if self.micro_ripple_zoom_mode:
            self.btn_ripple_zoom.setChecked(False)
        self.gl_view.opts["distance"] = 36.0
        self.gl_view.opts["elevation"] = 24.0
        self.gl_view.opts["azimuth"] = 205.0
        self.gl_view.update()

    def set_micro_ripple_zoom_mode(self, enabled: bool) -> None:
        self.micro_ripple_zoom_mode = bool(enabled)
        self.btn_ripple_zoom.blockSignals(True)
        self.btn_ripple_zoom.setChecked(self.micro_ripple_zoom_mode)
        self.btn_ripple_zoom.blockSignals(False)

        tr = TRANSLATIONS[self.lang]
        if self.micro_ripple_zoom_mode:
            self.mode_badge.setText(tr["vp_badge_micro"])
            self.gl_view.opts["distance"] = 82.0
            self.gl_view.opts["elevation"] = 30.0
            self.gl_view.opts["azimuth"] = -140.0
        else:
            self.mode_badge.setText(tr["vp_badge_default"])
            self.gl_view.opts["distance"] = 104.0
            self.gl_view.opts["elevation"] = 30.0
            self.gl_view.opts["azimuth"] = 38.0

        if self._last_state is not None and self._last_env is not None:
            self.update_from_state(
                self._last_state, self._last_env, self._last_u_star, self._last_bedrock_floor
            )

    def _compute_ray_marched_shadows_and_ao(
        self,
        z_scaled: np.ndarray,
        dx: float,
        dy: float,
    ) -> tuple:
        """Computes real-time ray-marched heightfield cast shadows and multi-scale HBAO."""
        if not self.show_shadows_and_ao:
            return np.ones_like(z_scaled, dtype=np.float32), np.ones_like(z_scaled, dtype=np.float32)

        az = math.radians(self.sun_azimuth_deg)
        el = math.radians(max(8.0, self.sun_elevation_deg))
        dir_x = math.cos(az)
        dir_y = math.sin(az)
        tan_el = math.tan(el)

        step_dist = 1.35 * min(dx, dy)
        max_horizon_excess = np.zeros_like(z_scaled, dtype=np.float32)

        # Ray-march 14 steps toward the sun to detect cast shadows from dune brinks/pyramids
        for m in (1, 2, 3, 5, 7, 10, 14, 19, 25, 32):
            dist_m = m * step_dist
            sx = int(round((dist_m * dir_x) / dx))
            sy = int(round((dist_m * dir_y) / dy))
            if sx == 0 and sy == 0:
                continue
            z_ray = z_scaled + dist_m * tan_el
            z_occ = np.roll(np.roll(z_scaled, -sx, axis=0), -sy, axis=1)
            # Clear rolled wrap-around edges
            if sx > 0:
                z_occ[-sx:, :] = z_scaled[-sx:, :]
            elif sx < 0:
                z_occ[:-sx, :] = z_scaled[:-sx, :]
            if sy > 0:
                z_occ[:, -sy:] = z_scaled[:, -sy:]
            elif sy < 0:
                z_occ[:, :-sy] = z_scaled[:, :-sy]

            excess = (z_occ - z_ray) / max(0.3, 0.18 * dist_m)
            max_horizon_excess = np.maximum(max_horizon_excess, excess)

        shadow_vis = 1.0 - np.clip(max_horizon_excess, 0.0, 1.0)
        shadow_vis = ndimage.gaussian_filter(shadow_vis, sigma=1.1)

        # Multi-scale Heightfield Ambient Occlusion (HBAO): darkens concave slipface toes & valleys
        blur_fine = ndimage.gaussian_filter(z_scaled, sigma=2.5)
        blur_wide = ndimage.gaussian_filter(z_scaled, sigma=9.0)
        concavity = np.maximum(0.0, (blur_fine - z_scaled) * 0.45 + (blur_wide - z_scaled) * 0.22)
        ao = np.clip(1.0 - 0.38 * concavity, 0.52, 1.0)

        return shadow_vis.astype(np.float32), ao.astype(np.float32)

    def _compute_blinn_phong_colors(
        self,
        h_field: np.ndarray,
        dx: float,
        dy: float,
        base_rgb: np.ndarray,
        vert_exag: float,
    ) -> np.ndarray:
        """Computes SimpleWindErosion-inspired Blinn-Phong + Cast Shadow + HBAO vertex colors."""
        z_scaled = h_field * vert_exag
        gx, gy = np.gradient(z_scaled, dx, dy)
        norm_len = np.sqrt(gx ** 2 + gy ** 2 + 1.0)
        nx_v = -gx / norm_len
        ny_v = -gy / norm_len
        nz_v = 1.0 / norm_len

        az = math.radians(self.sun_azimuth_deg)
        el = math.radians(self.sun_elevation_deg)
        lx = math.cos(el) * math.cos(az)
        ly = math.cos(el) * math.sin(az)
        lz = math.sin(el)

        ndotl = np.clip(nx_v * lx + ny_v * ly + nz_v * lz, 0.05, 1.0)

        shadow_vis, ao = self._compute_ray_marched_shadows_and_ao(z_scaled, dx, dy)

        # Specular quartz glint in sunlit areas
        hx, hy, hz = lx * 0.5, ly * 0.5, lz * 0.5 + 0.6
        h_len = max(1e-6, math.sqrt(hx ** 2 + hy ** 2 + hz ** 2))
        hx, hy, hz = hx / h_len, hy / h_len, hz / h_len
        spec = (np.clip(nx_v * hx + ny_v * hy + nz_v * hz, 0.0, 1.0) ** 48) * 0.06 * shadow_vis

        ambient = 0.52 * ao
        diffuse = 0.72 * ndotl * (0.28 + 0.72 * shadow_vis)
        light_total = 1.12 * (ambient + diffuse)

        # Warm sunlight tint (1.0, 0.99, 0.92) as in SimpleWindErosion model.hpp
        rgb = np.zeros_like(base_rgb)
        rgb[..., 0] = np.clip(base_rgb[..., 0] * light_total * 1.02 + spec, 0.0, 1.0)
        rgb[..., 1] = np.clip(base_rgb[..., 1] * light_total * 0.99 + spec * 0.95, 0.0, 1.0)
        rgb[..., 2] = np.clip(base_rgb[..., 2] * light_total * 0.92 + spec * 0.85, 0.0, 1.0)

        alpha = np.ones((*h_field.shape, 1), dtype=np.float32)
        return np.concatenate([rgb, alpha], axis=-1).astype(np.float32)

    def _compute_base_colormap(
        self,
        h_hr: np.ndarray,
        ripple_hr: np.ndarray,
        state: SimulationState2D,
        up: int,
        dx_hr: float,
        dy_hr: float,
        bedrock_hr: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Computes two-tone slope-dependent Sahara sand & coral-terracotta slipface colors."""
        h_max = max(float(np.max(h_hr)), 0.5)
        h_norm = np.clip(h_hr / h_max, 0.0, 1.0)

        if self.shader_mode == "elevation":
            return cm.turbo(h_norm)[..., :3].astype(np.float32)

        if self.shader_mode == "shear":
            tau_mag = np.hypot(state.tau_x, state.tau_y)
            if up > 1:
                tau_mag = ndimage.zoom(tau_mag, up, order=1)
            t_max = max(float(np.max(tau_mag)), 1e-4)
            return cm.viridis(np.clip(tau_mag / t_max, 0.0, 1.0))[..., :3].astype(np.float32)

        if self.shader_mode == "flux":
            q_mag = np.hypot(state.qx, state.qy)
            if up > 1:
                q_mag = ndimage.zoom(q_mag, up, order=1)
            q_max = max(float(np.max(q_mag)), 1e-6)
            return cm.inferno(np.clip(q_mag / q_max, 0.0, 1.0))[..., :3].astype(np.float32)

        if self.shader_mode == "wake":
            wake = state.in_wake.astype(float)
            if up > 1:
                wake = ndimage.zoom(wake, up, order=1)
            sand_mask = (h_hr > 0.04).astype(float)
            r = np.where(wake > 0.4, 0.15, np.where(sand_mask > 0.5, 0.85, 0.22))
            g = np.where(wake > 0.4, 0.78, np.where(sand_mask > 0.5, 0.64, 0.17))
            b = np.where(wake > 0.4, 0.95, np.where(sand_mask > 0.5, 0.36, 0.12))
            return np.stack([r, g, b], axis=-1).astype(np.float32)

        # Default 'desert': SimpleWindErosion-inspired two-tone slope coloring
        # flatColor = (0.85, 0.71, 0.46) golden-cream stoss sand
        # steepColor = (0.94, 0.50, 0.32) coral-terracotta avalanching slipface & interdune apron
        gx, gy = np.gradient(h_hr, dx_hr, dy_hr)
        slope_mag = np.hypot(gx, gy)
        steep_weight = np.clip((slope_mag - 0.16) / 0.38, 0.0, 1.0) ** 0.85

        # Flat / gentle stoss sand color (brightens slightly toward crest)
        flat_r = 0.80 + 0.10 * h_norm
        flat_g = 0.66 + 0.09 * h_norm
        flat_b = 0.43 + 0.07 * h_norm

        # Steep avalanching slipface / iron-oxide coral-terracotta color
        steep_r = 0.94
        steep_g = 0.49
        steep_b = 0.32

        r_sand = flat_r * (1.0 - steep_weight) + steep_r * steep_weight
        g_sand = flat_g * (1.0 - steep_weight) + steep_g * steep_weight
        b_sand = flat_b * (1.0 - steep_weight) + steep_b * steep_weight

        # Subtle ripple crest highlight
        if self.show_ripples and np.max(np.abs(ripple_hr)) > 1e-6:
            rip_norm = ripple_hr / max(1e-5, float(np.max(np.abs(ripple_hr))))
            r_sand = np.clip(r_sand + 0.038 * rip_norm, 0.0, 1.0)
            g_sand = np.clip(g_sand + 0.030 * rip_norm, 0.0, 1.0)
            b_sand = np.clip(b_sand + 0.018 * rip_norm, 0.0, 1.0)

        # Interdune floor / exposed bedrock or rocky pyramid obstacle coloring
        if bedrock_hr is not None:
            sand_depth = np.maximum(0.0, h_hr - bedrock_hr)
            rock_exposed = (bedrock_hr > 0.05) & (sand_depth < 0.06)
        else:
            sand_depth = h_hr
            rock_exposed = np.zeros_like(h_hr, dtype=bool)

        sand_blend = np.clip((sand_depth - 0.005) / 0.10, 0.0, 1.0)
        # Warm terracotta-ochre desert floor (harmonizes with dune sand instead of black void)
        floor_r, floor_g, floor_b = 0.74, 0.51, 0.33

        r = floor_r * (1.0 - sand_blend) + r_sand * sand_blend
        g = floor_g * (1.0 - sand_blend) + g_sand * sand_blend
        b = floor_b * (1.0 - sand_blend) + b_sand * sand_blend

        # If a rocky pyramid obstacle is exposed, color it with stratified desert sandstone/granite
        if np.any(rock_exposed):
            strata = 0.5 + 0.5 * np.sin(h_hr * 6.0)
            rock_r = 0.52 + 0.08 * strata
            rock_g = 0.43 + 0.06 * strata
            rock_b = 0.35 + 0.05 * strata
            r = np.where(rock_exposed, rock_r, r)
            g = np.where(rock_exposed, rock_g, g)
            b = np.where(rock_exposed, rock_b, b)

        return np.stack([r, g, b], axis=-1).astype(np.float32)

    def update_from_state(
        self,
        state: SimulationState2D,
        env: PlanetaryEnvironment,
        u_star_0: float,
        bedrock_floor: Optional[np.ndarray] = None,
    ) -> None:
        """Updates the 3D OpenGL terrain, cutaway diorama skirt, and HUD from SimulationState2D."""
        self._last_state = state
        self._last_env = env
        self._last_u_star = u_star_0
        self._last_bedrock_floor = bedrock_floor

        if self.micro_ripple_zoom_mode:
            self._render_micro_ripple_patch(state, env, u_star_0)
            return

        h = state.h
        nx, ny = h.shape
        dx = float(state.x[1] - state.x[0]) if nx > 1 else 1.0
        dy = float(state.y[1] - state.y[0]) if ny > 1 else 1.0
        lx = nx * dx
        ly = ny * dy

        up = self.upsample_factor if self.show_ripples else 2
        if self.show_ripples:
            x_hr, y_hr, h_hr, ripple_hr = synthesize_multiscale_desert_surface(
                h=h,
                dx=dx,
                dy=dy,
                env=env,
                u_star_0=u_star_0,
                in_wake=state.in_wake,
                tau_x=state.tau_x,
                tau_y=state.tau_y,
                time_s=state.time_s,
                upsample_factor=up,
                ripple_visual_exaggeration=self.ripple_exaggeration,
            )
        else:
            h_hr = ndimage.zoom(h, up, order=3)
            if np.min(h) >= 0.0:
                h_hr = np.maximum(0.0, h_hr)
            x_hr = np.arange(h_hr.shape[0]) * (dx / up)
            y_hr = np.arange(h_hr.shape[1]) * (dy / up)
            ripple_hr = np.zeros_like(h_hr)

        bedrock_hr = None
        if bedrock_floor is not None and np.ndim(bedrock_floor) == 2:
            bedrock_hr = ndimage.zoom(np.asarray(bedrock_floor, dtype=np.float64), up, order=1)

        nx_hr, ny_hr = h_hr.shape
        dx_hr = dx / up
        dy_hr = dy / up

        if self._cached_shape != (nx_hr, ny_hr) or self._cached_faces is None:
            self._cached_faces = _build_grid_faces(nx_hr, ny_hr)
            self._cached_shape = (nx_hr, ny_hr)

        X_hr, Y_hr = np.meshgrid(x_hr - 0.5 * lx, y_hr - 0.5 * ly, indexing="ij")
        Z_hr = h_hr * self.vertical_exaggeration
        verts = np.column_stack([X_hr.ravel(), Y_hr.ravel(), Z_hr.ravel()]).astype(np.float32)

        base_rgb = self._compute_base_colormap(h_hr, ripple_hr, state, up, dx_hr, dy_hr, bedrock_hr)
        colors = self._compute_blinn_phong_colors(h_hr, dx_hr, dy_hr, base_rgb, self.vertical_exaggeration)

        self.terrain_mesh.setMeshData(
            vertexes=verts,
            faces=self._cached_faces,
            vertexColors=colors.reshape(-1, 4),
            smooth=True,
            computeNormals=False,
        )
        self.terrain_mesh.opts["drawEdges"] = self.show_wireframe
        self.terrain_mesh.opts["edgeColor"] = (0.9, 0.7, 0.3, 0.22)

        # --- Solid 3D Geological Cutaway Diorama Skirt ---
        if self.show_diorama_skirt:
            s_verts, s_faces, s_cols = _build_diorama_skirt_mesh(
                x_hr, y_hr, Z_hr, lx, ly, z_base=-4.2
            )
            self.skirt_mesh.setMeshData(
                vertexes=s_verts,
                faces=s_faces,
                vertexColors=s_cols,
                smooth=False,
                computeNormals=False,
            )
            self.skirt_mesh.setVisible(True)
        else:
            self.skirt_mesh.setVisible(False)

        # --- Optional Diagnostic Overlays ---
        if self.show_wake_dome and np.any(state.in_wake):
            if self._wake_shape != (nx, ny) or self._wake_faces is None:
                self._wake_faces = _build_grid_faces(nx, ny)
                self._wake_shape = (nx, ny)
            X0, Y0 = np.meshgrid(state.x - 0.5 * lx, state.y - 0.5 * ly, indexing="ij")
            Z_wake = np.where(state.in_wake, state.h_tilde * self.vertical_exaggeration + 0.08, h * self.vertical_exaggeration - 0.05)
            w_verts = np.column_stack([X0.ravel(), Y0.ravel(), Z_wake.ravel()]).astype(np.float32)
            w_colors = np.zeros((nx * ny, 4), dtype=np.float32)
            w_mask = state.in_wake.ravel()
            w_colors[w_mask] = [0.95, 0.75, 0.35, 0.18]
            self.wake_mesh.setMeshData(
                vertexes=w_verts,
                faces=self._wake_faces,
                vertexColors=w_colors,
                smooth=True,
                computeNormals=False,
            )
            self.wake_mesh.setVisible(True)
        else:
            self.wake_mesh.setVisible(False)

        if self.show_brink_line and np.any(state.in_wake):
            pts = []
            for j in range(ny):
                w_idx = np.where(state.in_wake[:, j])[0]
                if len(w_idx) > 0 and w_idx[0] > 0:
                    bi = w_idx[0] - 1
                    if h[bi, j] > 0.35:
                        pts.append([state.x[bi] - 0.5 * lx, state.y[j] - 0.5 * ly, h[bi, j] * self.vertical_exaggeration + 0.25])
            if len(pts) > 2:
                self.brink_line.setData(pos=np.array(pts, dtype=np.float32))
                self.brink_line.setVisible(True)
            else:
                self.brink_line.setVisible(False)
        else:
            self.brink_line.setVisible(False)

        if self.show_saltation_cloud:
            q_mag = np.hypot(state.qx, state.qy)
            active_i, active_j = np.where(q_mag > 0.08 * max(float(np.max(q_mag)), 1e-6))
            if len(active_i) > 20:
                n_particles = min(400, len(active_i))
                pick = self._rng.choice(len(active_i), size=n_particles, replace=False)
                pi, pj = active_i[pick], active_j[pick]
                px = state.x[pi] - 0.5 * lx + self._rng.uniform(-0.5 * dx, 0.5 * dx, size=n_particles)
                py = state.y[pj] - 0.5 * ly + self._rng.uniform(-0.5 * dy, 0.5 * dy, size=n_particles)
                pz = (h[pi, pj] + self._rng.exponential(0.15, size=n_particles)) * self.vertical_exaggeration
                pos = np.column_stack([px, py, pz]).astype(np.float32)
                p_col = np.tile(np.array([1.0, 0.86, 0.55, 0.62], dtype=np.float32), (n_particles, 1))
                self.saltation_scatter.setData(pos=pos, color=p_col, size=2.8)
                self.saltation_scatter.setVisible(True)
            else:
                self.saltation_scatter.setVisible(False)
        else:
            self.saltation_scatter.setVisible(False)

        if self.show_wind_streamlines:
            y_seeds = np.linspace(0.15 * ly, 0.85 * ly, len(self.streamline_items))
            for idx_line, y0 in enumerate(y_seeds):
                j_col = int(np.clip(round(y0 / dy), 0, ny - 1))
                sx = state.x - 0.5 * lx
                defl = np.cumsum(state.tau_y[:, j_col]) * dx / max(1e-6, env.fluid_density * (u_star_0 ** 2))
                sy = (y0 - 0.5 * ly) + 1.8 * defl
                sz = (state.h_tilde[:, j_col] + 0.8) * self.vertical_exaggeration
                self.streamline_items[idx_line].setData(pos=np.column_stack([sx, sy, sz]).astype(np.float32))
                self.streamline_items[idx_line].setVisible(True)
        else:
            for line in self.streamline_items:
                line.setVisible(False)

        lam_r, _, _ = compute_impact_ripple_scales(env, u_star_0)
        if self.lang == "ar":
            mode_tag = " [وضع النحت 3D]" if self.sculpt_mode else ""
            self.hud_label.setText(
                f"الزمن: {state.time_days:5.1f} يوم ({state.time_years:4.2f} سنة)  |  "
                f"ارتفاع القمة H: {state.max_height:4.2f} م  |  الحجم: {state.total_volume:7.1f} م³{mode_tag}"
                + NL
                + f"قطر الحبيبات d: {env.grain_diameter*1e6:4.0f} ميكرومتر  |  سرعة القص u*: {u_star_0:4.2f} م/ث  |  "
                f"طول التشبع L_sat: {env.saturation_length:4.2f} م  |  طول موجة التموجات: {lam_r*100:4.1f} سم"
            )
        else:
            mode_tag = " [SCULPT MODE]" if self.sculpt_mode else ""
            self.hud_label.setText(
                f"Time: {state.time_days:6.1f} d ({state.time_years:4.2f} yr)  |  "
                f"Crest H: {state.max_height:4.2f} m  |  Volume: {state.total_volume:7.1f} m^3{mode_tag}"
                + NL
                + f"Grain d: {env.grain_diameter*1e6:4.0f} um  |  u*: {u_star_0:4.2f} m/s  |  "
                f"L_sat: {env.saturation_length:4.2f} m  |  Ripple lambda_r: {lam_r*100:4.1f} cm"
            )

    def _render_micro_ripple_patch(
        self,
        state: SimulationState2D,
        env: PlanetaryEnvironment,
        u_star_0: float,
    ) -> None:
        """Renders a 2m x 2m ultra-high-resolution 3D diorama patch of ballistic impact ripples."""
        x_p, y_p, z_mm, lam_cm, h_mm = synthesize_zoomed_ripple_patch(
            env=env,
            u_star_0=u_star_0,
            patch_size_m=2.0,
            n_grid=180,
            time_s=state.time_s,
        )
        nx_p, ny_p = len(x_p), len(y_p)
        if self._cached_shape != (nx_p, ny_p) or self._cached_faces is None:
            self._cached_faces = _build_grid_faces(nx_p, ny_p)
            self._cached_shape = (nx_p, ny_p)

        scale_xy = 50.0
        X, Y = np.meshgrid((x_p - 1.0) * scale_xy, (y_p - 1.0) * scale_xy, indexing="ij")
        Z = z_mm * 0.45
        verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]).astype(np.float32)

        dx_p = (x_p[1] - x_p[0]) * scale_xy
        gx, gy = np.gradient(Z, dx_p, dx_p)
        slope_mag = np.hypot(gx, gy)
        steep_w = np.clip((slope_mag - 0.12) / 0.30, 0.0, 1.0)

        r = 0.86 * (1.0 - steep_w) + 0.94 * steep_w
        g = 0.72 * (1.0 - steep_w) + 0.50 * steep_w
        b = 0.47 * (1.0 - steep_w) + 0.32 * steep_w
        base_rgb = np.stack([r, g, b], axis=-1).astype(np.float32)
        colors = self._compute_blinn_phong_colors(Z, dx_p, dx_p, base_rgb, 1.0)

        self.terrain_mesh.setMeshData(
            vertexes=verts,
            faces=self._cached_faces,
            vertexColors=colors.reshape(-1, 4),
            smooth=True,
            computeNormals=False,
        )
        if self.show_diorama_skirt:
            s_verts, s_faces, s_cols = _build_diorama_skirt_mesh(
                x_p * scale_xy, y_p * scale_xy, Z, 2.0 * scale_xy, 2.0 * scale_xy, z_base=-3.5
            )
            self.skirt_mesh.setMeshData(
                vertexes=s_verts, faces=s_faces, vertexColors=s_cols, smooth=False, computeNormals=False
            )
            self.skirt_mesh.setVisible(True)
        else:
            self.skirt_mesh.setVisible(False)

        self.wake_mesh.setVisible(False)
        self.brink_line.setVisible(False)
        self.saltation_scatter.setVisible(False)
        for line in self.streamline_items:
            line.setVisible(False)

        if self.lang == "ar":
            self.hud_label.setText(
                "[فاحص التموجات الدقيقة: رقعة 2.0 م × 2.0 م من سطح المنحدر المواجه للرياح]"
                + NL
                + f"طول موجة تموجات التصادم القذفي: {lam_cm:5.1f} سم  |  "
                f"ارتفاع قمة التموج: {h_mm:4.1f} ملم  |  مؤشر التموج RI: 16.0"
            )
        else:
            self.hud_label.setText(
                "[MICRO-RIPPLE ZOOM PATCH: 2.0 m x 2.0 m Stoss Surface]"
                + NL
                + f"Ballistic Impact Ripple Wavelength: {lam_cm:5.1f} cm  |  "
                f"Ripple Crest Height: {h_mm:4.1f} mm  |  Ripple Index RI: 16.0"
            )
