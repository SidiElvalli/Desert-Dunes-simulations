"""Scientific documentation dialog and always-visible bottom Equations, Numerical Integration,
Literature References, and Model Limitations panel for the Desert Studio.
"""

from typing import Optional
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QLabel,
    QPushButton,
    QScrollArea,
    QFrame,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .theme import BLENDER_DARK_QSS
from .i18n import TRANSLATIONS

BS = chr(92)
NL = chr(10)


def _tex(s: str) -> str:
    """Converts '@' prefix to backslash for LaTeX mathtext strings."""
    return s.replace("@", BS)


class EquationsAndLimitationsStrip(QFrame):
    """Always-visible bottom scientific dock displaying rendered Governing Equations,
    Numerical Integration Formulas, Paper References, and Model Limitations.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.lang: str = "en"
        self.setFixedHeight(192)
        self.setStyleSheet(
            "QFrame { background-color: #1b1b1f; border: 1px solid #32323a; border-radius: 5px; }"
        )
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(8)

        # --- Left Half: Rendered Mathematical Governing Equations & Integration Formulas (Matplotlib Canvas) ---
        eq_frame = QFrame(self)
        eq_frame.setStyleSheet("QFrame { background-color: #141418; border: 1px solid #2d2d35; border-radius: 4px; }")
        eq_layout = QVBoxLayout(eq_frame)
        eq_layout.setContentsMargins(6, 4, 6, 2)
        eq_layout.setSpacing(1)

        self.lbl_eq_hdr = QLabel(TRANSLATIONS["en"]["eq_hdr"])
        self.lbl_eq_hdr.setStyleSheet("color: #f0a830; font-weight: bold; font-size: 10.5px; border: none;")
        eq_layout.addWidget(self.lbl_eq_hdr)

        self.fig_eq = Figure(figsize=(7.6, 1.55), dpi=96, facecolor="#141418")
        self.canvas_eq = FigureCanvas(self.fig_eq)
        eq_layout.addWidget(self.canvas_eq, stretch=1)
        self._render_equations_canvas()
        root.addWidget(eq_frame, stretch=13)

        # --- Middle Card: Peer-Reviewed Paper References by Module ---
        ref_frame = QFrame(self)
        ref_frame.setStyleSheet("QFrame { background-color: #16161b; border: 1px solid #2d2d35; border-radius: 4px; }")
        ref_layout = QVBoxLayout(ref_frame)
        ref_layout.setContentsMargins(8, 5, 8, 5)
        ref_layout.setSpacing(2)

        self.lbl_ref_hdr = QLabel(TRANSLATIONS["en"]["ref_hdr"])
        self.lbl_ref_hdr.setStyleSheet("color: #f0a830; font-weight: bold; font-size: 10.5px; border: none;")
        ref_layout.addWidget(self.lbl_ref_hdr)

        self.lbl_refs = QLabel(TRANSLATIONS["en"]["ref_body"])
        self.lbl_refs.setWordWrap(True)
        self.lbl_refs.setStyleSheet("color: #d0d4dc; font-size: 9.8px; border: none;")
        ref_layout.addWidget(self.lbl_refs, stretch=1)
        root.addWidget(ref_frame, stretch=7)

        # --- Right Card: Physical & Numerical Limitations of the Model ---
        lim_frame = QFrame(self)
        lim_frame.setStyleSheet("QFrame { background-color: #191618; border: 1px solid #482e2e; border-radius: 4px; }")
        lim_layout = QVBoxLayout(lim_frame)
        lim_layout.setContentsMargins(8, 5, 8, 5)
        lim_layout.setSpacing(2)

        self.lbl_lim_hdr = QLabel(TRANSLATIONS["en"]["lim_hdr"])
        self.lbl_lim_hdr.setStyleSheet("color: #ff7657; font-weight: bold; font-size: 10.5px; border: none;")
        lim_layout.addWidget(self.lbl_lim_hdr)

        self.lbl_lims = QLabel(TRANSLATIONS["en"]["lim_body"])
        self.lbl_lims.setWordWrap(True)
        self.lbl_lims.setStyleSheet("color: #d8ccd0; font-size: 9.8px; border: none;")
        lim_layout.addWidget(self.lbl_lims, stretch=1)
        root.addWidget(lim_frame, stretch=7)

    def set_language(self, lang: str) -> None:
        self.lang = lang if lang in TRANSLATIONS else "en"
        tr = TRANSLATIONS[self.lang]
        self.lbl_eq_hdr.setText(tr["eq_hdr"])
        self.lbl_ref_hdr.setText(tr["ref_hdr"])
        self.lbl_refs.setText(tr["ref_body"])
        self.lbl_lim_hdr.setText(tr["lim_hdr"])
        self.lbl_lims.setText(tr["lim_body"])

    def _render_equations_canvas(self) -> None:
        self.fig_eq.clear()
        ax = self.fig_eq.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.set_facecolor("#141418")
        ax.axis("off")

        ax.text(
            0.012,
            0.79,
            _tex(
                "1. Wind & Shear: $u(z) = @frac{u_{*0}}{@kappa}@ln@left(@frac{z}{z_{@mathrm{eff}}}@right), @; "
                "@hat{@tau}_x = @left(A@frac{k_x^2}{|k|} + iBk_x@right)@hat{@tilde{h}}, @; "
                "@hat{@tau}_y = A_y@frac{k_x k_y}{|k|}@hat{@tilde{h}}$"
            ),
            color="#e8ecf2",
            fontsize=8.7,
        )
        ax.text(
            0.012,
            0.55,
            _tex(
                "2. Separation Bubble: $s(z) = h_b + (2H_d + h'_d L_b)z^3 - (3H_d + 2h'_d L_b)z^2 + h'_d L_b z + H_d, @; "
                "L_b @approx 6H_d$"
            ),
            color="#e8ecf2",
            fontsize=8.7,
        )
        ax.text(
            0.012,
            0.31,
            _tex(
                "3. Saltation & Lag: $L_{@mathrm{sat}}@partial_x q_x = q_{@mathrm{sat}} - q_x, @; "
                "q_{@mathrm{sat}} = C@frac{@rho_f}{g}u_*^3@left(1 - @frac{@tau_{@mathrm{th,eff}}}{@tau}@right)_+, @; "
                "L_{@mathrm{sat}} = 2.2@frac{@rho_s}{@rho_f}d$"
            ),
            color="#e8ecf2",
            fontsize=8.7,
        )
        ax.text(
            0.012,
            0.07,
            _tex(
                "4. Exner & Avalanche Integration: $q_{x,i} = q_{x,i-1}e^{-@Delta x/L_{@mathrm{sat}}} + q_{@mathrm{sat},i}(1 - e^{-@Delta x/L_{@mathrm{sat}}}), @; "
                "h_i^{n+1} = h_i^n - @frac{@Delta t}{@rho_b}@nabla_{@mathrm{up}}@cdot@mathbf{q} + @Delta h_{@mathrm{aval}}(|@nabla h| > @tan 34^@circ)$"
            ),
            color="#82e0aa",
            fontsize=8.5,
        )
        self.canvas_eq.draw_idle()


class ScientificHelpDialog(QDialog):
    """Comprehensive multi-tab Scientific Documentation, Literature Citations,
    Numerical Integration Schemes, and Model Limitations browser (Bilingual EN / AR).
    """

    def __init__(self, parent: Optional[QWidget] = None, lang: str = "en"):
        super().__init__(parent)
        self.lang = lang if lang in ("en", "ar") else "en"
        if self.lang == "ar":
            self.setWindowTitle(
                "صحراء (Desert) — الدليل العلمي المرجعي: المعادلات الفيزيائية، التكامل العددي، المراجع المحكمة والقيود"
            )
            self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        else:
            self.setWindowTitle(
                "Desert Studio — Scientific Models, Numerical Integration, Citations & Limitations"
            )
        self.resize(1120, 760)
        self.setStyleSheet(BLENDER_DARK_QSS)
        self._build_ui()

    def _build_ui(self) -> None:
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(12, 12, 12, 12)
        vbox.setSpacing(10)

        hdr_txt = (
            "الدليل العلمي المرجعي لاستوديو صحراء — المعادلات الفيزيائية الحاكمة، صيغ التكامل العددي، الأبحاث المحكمة وحدود النموذج"
            if self.lang == "ar"
            else "DESERT SCIENTIFIC REFERENCE MANUAL — Equations, Integration Schemes, Peer-Reviewed Citations & Limitations"
        )
        hdr = QLabel(hdr_txt)
        hdr.setStyleSheet(
            "background-color: #24201b; color: #f5b041; font-size: 13.5px; font-weight: bold; "
            "padding: 10px; border: 1px solid #d97724; border-radius: 5px;"
        )
        vbox.addWidget(hdr)

        tabs = QTabWidget(self)
        if self.lang == "ar":
            tabs.addTab(self._build_equations_tab_ar(), "1. المعادلات الفيزيائية الحاكمة والمراجع")
            tabs.addTab(self._build_integration_tab_ar(), "2. صيغ التكامل العددي والحجوم المنتهية")
            tabs.addTab(self._build_limitations_tab_ar(), "3. الافتراضات الفيزيائية وحدود صلاحية النموذج")
            tabs.addTab(self._build_bibliography_tab(), "4. قائمة الأبحاث والمراجع العلمية المحكمة")
        else:
            tabs.addTab(self._build_equations_tab(), "1. Governing Physical Equations && Citations")
            tabs.addTab(self._build_integration_tab(), "2. Numerical Integration && Discretization Formulas")
            tabs.addTab(self._build_limitations_tab(), "3. Physical && Numerical Limitations of the Model")
            tabs.addTab(self._build_bibliography_tab(), "4. Complete Peer-Reviewed Bibliography")
        vbox.addWidget(tabs, stretch=1)

        btn_close = QPushButton(
            "إغلاق الدليل العلمي المرجعي" if self.lang == "ar" else "Close Scientific Reference Manual"
        )
        btn_close.setObjectName("PrimaryActionBtn")
        btn_close.clicked.connect(self.accept)
        vbox.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)

    def _make_scroll_page(self, blocks: list) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        for title, body, eq_latex in blocks:
            card = QFrame()
            card.setStyleSheet(
                "QFrame { background-color: #222228; border: 1px solid #383842; border-radius: 6px; padding: 6px; }"
            )
            c_vbox = QVBoxLayout(card)
            lbl_t = QLabel(title)
            lbl_t.setStyleSheet("color: #f0a830; font-size: 12.5px; font-weight: bold; border: none;")
            c_vbox.addWidget(lbl_t)

            if eq_latex:
                fig = Figure(figsize=(8.5, 0.72), dpi=100, facecolor="#16161c")
                canvas = FigureCanvas(fig)
                canvas.setFixedHeight(68)
                ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
                ax.set_facecolor("#16161c")
                ax.axis("off")
                ax.text(0.5, 0.5, _tex(eq_latex), color="#ffffff", fontsize=11.5, ha="center", va="center")
                c_vbox.addWidget(canvas)

            lbl_b = QLabel(body)
            lbl_b.setWordWrap(True)
            lbl_b.setStyleSheet("color: #dcdfe6; font-size: 11.5px; line-height: 1.4; border: none;")
            c_vbox.addWidget(lbl_b)
            layout.addWidget(card)

        layout.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _build_equations_tab(self) -> QWidget:
        blocks = [
            (
                "Part 1: Turbulent Logarithmic Wind Profile & Owen Saltation Roughness (core/wind_profile.py)",
                "WHEN USED: Evaluated continuously to couple the 10 m reference wind speed U(10 m) to the basal friction "
                "velocity u_*0 and undisturbed shear stress tau_0 = rho_f u_*0^2." + NL
                + "WHY USED: Above the threshold friction velocity u_*th, saltating grains extract momentum from the near-surface "
                "air layer, creating an elevated apparent aerodynamic roughness z0_salt proportional to (u_*0^2 - u_*th^2)/(2g) "
                "(Bagnold focal point)." + NL
                + "REFERENCES: Prandtl (1925) ZAMM 5, 136-139; Bagnold (1941) The Physics of Blown Sand and Desert Dunes; "
                "Owen (1964) J. Fluid Mech. 20, 225-242.",
                "$u(z) = @frac{u_{*0}}{@kappa}@ln@left(@frac{z}{z_{0,@mathrm{grain}} + z_{0,@mathrm{salt}}}@right), "
                "@qquad z_{0,@mathrm{grain}} = @frac{d}{30}, @qquad z_{0,@mathrm{salt}} = C_{@mathrm{Owen}}@frac{u_{*0}^2 - u_{*@mathrm{th}}^2}{2g}$",
            ),
            (
                "Part 2: Inertial Drag Length, Saturation Length & Threshold Scaling (core/constants.py)",
                "WHEN USED: Computed for every planetary environment (Earth, Mars, Venus, Subaqueous Flume) and grain diameter d." + NL
                + "WHY USED: A grain of density rho_s accelerated by fluid of density rho_f requires an inertial distance "
                "L_drag = (rho_s / rho_f) d to reach the fluid speed. Multi-grain splash cascades set the flux saturation length "
                "L_sat = 2.2 L_drag, which universally determines the minimum unstable dune wavelength lambda_min = 4*pi*L_sat "
                "and fastest-growing wavelength lambda_max ~= 19.8 L_sat across five orders of magnitude." + NL
                + "REFERENCES: Bagnold (1941); Iversen & White (1982) Sedimentology 29, 111-119; Claudin & Andreotti (2006) "
                "Earth Planet. Sci. Lett. 252, 30-44.",
                "$L_{@mathrm{drag}} = @frac{@rho_s}{@rho_f}d, @qquad L_{@mathrm{sat}} = 2.2@frac{@rho_s}{@rho_f}d, @qquad "
                "u_{*@mathrm{th}} = A_B@sqrt{@frac{@rho_s - @rho_f}{@rho_f}gd} @; (A_B = 0.10)$",
            ),
            (
                "Part 3: Flow Separation Streamline Bubble & Aerodynamic Envelope (core/separation.py)",
                "WHEN USED: Step 1 of every time step whenever the downward lee slope exceeds the critical boundary-layer "
                "detachment angle theta_sep = 14 deg (dh/dx < -0.2493)." + NL
                + "WHY USED: Turbulent boundary layers detach at sharp dune brinks due to the strong adverse pressure gradient. "
                "The dividing streamline s(z) is modeled as a cubic Hermite polynomial satisfying C^0 and C^1 continuity at the "
                "brink (s(0)=H_d, s'(0)=h'_d) and smooth tangential reattachment at distance L_b ~= 6 H_d (s(1)=0, s'(1)=0). "
                "Inside the wake bubble (s > h), basal shear stress is set to tau = 0, trapping all incoming saltating sand." + NL
                + "REFERENCES: Kroy, Sauermann & Herrmann (2002) Phys. Rev. E 66, 031302; Duran, Parteli & Herrmann (2010) "
                "Earth Surf. Process. Landforms 35, 1591-1600.",
                "$s(z) = h_b + (2H_d + h'_d L_b)z^3 - (3H_d + 2h'_d L_b)z^2 + h'_d L_b z + H_d, @qquad "
                "L_b = @frac{1.5 H_d}{@bar{s}'}@left(1 + @frac{1}{4}@nu + @frac{1}{8}@nu^2@right)$",
            ),
            (
                "Part 4: 2D Jackson-Hunt Fourier Shear Stress Perturbation Solver (core/wind.py)",
                "WHEN USED: Step 2 of every time step over the smooth aerodynamic envelope h_tilde(x, y)." + NL
                + "WHY USED: Solves the linearized turbulent boundary-layer momentum equations in 2D Fourier space. "
                "The in-phase term A (kx^2 / |k|) represents potential-flow Bernoulli streamline compression over the crest, "
                "while the out-of-phase quadrature term +i B kx shifts the shear stress maximum UPWIND of the topographic crest. "
                "This upwind phase advance is the fundamental hydrodynamic instability mechanism that creates sand dunes." + NL
                + "REFERENCES: Jackson & Hunt (1975) Q. J. R. Meteorol. Soc. 101, 929-955; Hunt, Leibovich & Richards (1988) "
                "QJRMS 114, 1435-1470; Fourriere, Claudin & Andreotti (2010) J. Fluid Mech. 649, 287-328.",
                "$@hat{@tau}_{@mathrm{pert},x}(k_x,k_y) = @left(A@frac{k_x^2}{|k|} + iBk_x@right)@hat{@tilde{h}}(k_x,k_y), @qquad "
                "@hat{@tau}_{@mathrm{pert},y}(k_x,k_y) = A_y@frac{k_x k_y}{|k|}@hat{@tilde{h}}(k_x,k_y)$",
            ),
            (
                "Part 5: Non-Local Saltation Relaxation & Slope-Modulated Threshold (core/transport.py)",
                "WHEN USED: Step 3 of every time step across the entire 2D domain." + NL
                + "WHY USED: Saltation flux q_x(x,y) does not adjust instantaneously to local shear stress tau; it relaxes "
                "spatially over L_sat via L_sat dq_x/dx = q_sat - q_x. Uphill stoss slopes experience an increased threshold "
                "tau_th,eff due to gravity (Bagnold-Howard-Iversen effect), naturally regulating windward stoss slopes to ~10 deg." + NL
                + "REFERENCES: Lettau & Lettau (1978); Iversen & Rasmussen (1994) Sedimentology 41, 721-728; Sauermann, Kroy & "
                "Herrmann (2001) Phys. Rev. E 64, 031305.",
                "$L_{@mathrm{sat}}@frac{@partial q_x}{@partial x} = q_{@mathrm{sat}}(@tau) - q_x, @qquad "
                "q_{@mathrm{sat}}(@tau) = C_{@mathrm{flux}}@frac{@rho_f}{g}u_*^3@left(1 - @frac{@tau_{@mathrm{th,eff}}}{@tau}@right)_+, @qquad "
                "@tau_{@mathrm{th,eff}} = @tau_{@mathrm{th},0}@left(1 + @frac{@partial_x h}{@tan@theta_r}@right)$",
            ),
        ]
        return self._make_scroll_page(blocks)

    def _build_integration_tab(self) -> QWidget:
        blocks = [
            (
                "Integration Formula 1: Unconditionally Positive Exponential Euler Spatial Integrator (core/transport.py)",
                "WHY STANDARD FORWARD EULER FAILS: If L_sat is smaller than or comparable to grid spacing dx (e.g., coarse grids "
                "or subaqueous ripples), explicit Forward Euler q_i = q_{i-1} + (dx/L_sat)(q_sat,i - q_{i-1}) becomes oscillatory "
                "and negative whenever dx > L_sat." + NL
                + "EXACT INTEGRATION FORMULA USED: Treating q_sat as piecewise constant over cell [x_{i-1}, x_i] and integrating "
                "the linear ODE L_sat dq/dx + q = q_sat analytically yields the exact Exponential Euler recurrence below, coupled "
                "with a local finite-volume bedrock entrainment cap so bare bedrock (h = h_floor) can never be eroded:",
                "$@mu = @exp@left(-@frac{@Delta x}{L_{@mathrm{sat}}}@right), @qquad "
                "q_{x,i}^{@mathrm{ideal}} = q_{x,i-1}@,@mu + q_{@mathrm{sat},i}(1 - @mu), @qquad "
                "q_{x,i} = @min@left(q_{x,i}^{@mathrm{ideal}},@; q_{x,i-1} + @frac{@rho_b @Delta x}{@Delta t}(h_i - h_{@mathrm{floor}})_+@right)$",
            ),
            (
                "Integration Formula 2: Staggered Donor-Cell Finite-Volume Transverse Flux (core/transport.py, core/exner.py)",
                "WHY CENTRAL DIFFERENCING FAILS: Evaluating q_y at cell centers and taking central differences (q_{j+1} - q_{j-1})/(2*dy) "
                "creates a wide 5-point stencil (h_{j+2} - 2h_j + h_{j-2})/(4*dy^2) with zero damping at the Nyquist (-1)^j mode "
                "and erodes bare bedrock cells adjacent to the dune." + NL
                + "EXACT INTEGRATION FORMULA USED: Transverse flux q_y is discretized on staggered cell interfaces j + 1/2 using "
                "upwind donor-cell selection for wind deflection and downslope donor-cell selection for splash diffusion:",
                "$q_{y, i, j+1/2} = q_{x,@mathrm{upwind}}@,@frac{@tau_{y,i,j} + @tau_{y,i,j+1}}{2@tau_0} "
                "- D_{@mathrm{lat}}@,q_{x,@mathrm{donor}}@,@frac{h_{i,j+1} - h_{i,j}}{@Delta y}$",
            ),
            (
                "Integration Formula 3: Upwind Finite-Volume Exner Time Step & Adaptive Morphological CFL (core/exner.py)",
                "TIME INTEGRATION FORMULA: Bed elevation h(x, y, t) is advanced in time via conservative upwind finite-volume flux "
                "differences paired with an adaptive morphological Courant-Friedrichs-Lewy (CFL) time step dt that bounds the "
                "maximum elevation change per step to a fraction of the dynamic repose step height dx * tan(32 deg):",
                "$h_{i,j}^{n+1,*} = h_{i,j}^n - @frac{@Delta t}{@rho_{@mathrm{bulk}}}@left(@frac{q_{x,i,j} - q_{x,i-1,j}}{@Delta x} "
                "+ @frac{q_{y,i,j+1/2} - q_{y,i,j-1/2}}{@Delta y}@right), @qquad "
                "@Delta t = @mathrm{clip}@left(@frac{@mathrm{CFL}@,@Delta x@,@tan 32^@circ}{@max |@partial_t h|},@; @Delta t_{@min},@; @Delta t_{@max}@right)$",
            ),
            (
                "Integration Formula 4: Isotropic 2D Euclidean Cellular Avalanche Relaxation (core/avalanche.py)",
                "WHY AXIS-BY-AXIS AVALANCHE FAILS: Checking |dh/dx| > tan(34 deg) and |dh/dy| > tan(34 deg) independently along "
                "grid axes allows 45-deg diagonal slipfaces on Barchan shoulders to stand at tan(32 deg)/cos(45 deg) = tan(41.5 deg), "
                "creating unphysical twin shoulder peaks." + NL
                + "ISOTROPIC INTEGRATION FORMULA USED: At every cell edge, the true 2D Euclidean slope norm |grad h|_e is computed "
                "using centered cross-derivatives, and mass-conserving Jacobi edge fluxes (omega = 0.15 < 1/4) relax over-steepened "
                "slopes isotropically from theta_stat = 34 deg to theta_dyn = 32 deg:",
                "$|@nabla h|_e = @sqrt{s_x^2 + s_y^2}, @qquad "
                "F_{@mathrm{aval},e} = @omega@,@Delta h_e@,@max@left(0,@; 1 - @frac{@tan 32^@circ}{|@nabla h|_e}@right), @qquad "
                "h @leftarrow h + F_{@mathrm{in}} - F_{@mathrm{out}}$",
            ),
        ]
        return self._make_scroll_page(blocks)

    def _build_limitations_tab(self) -> QWidget:
        blocks = [
            (
                "Limitation 1: First-Order Linearized Boundary-Layer Perturbation Theory (Aspect Ratio H/L < 0.25)",
                "The Jackson-Hunt (1975) / Kroy et al. (2002) spectral wind solver derives from a first-order asymptotic expansion "
                "of the turbulent boundary layer in powers of the dune aspect ratio H/L. While highly accurate on windward stoss "
                "slopes (typically 8-12 deg, H/L ~ 0.10-0.15), it cannot directly resolve nonlinear 3D turbulent vortex shedding, "
                "horseshoe vortices around horn tips, or unsteady wake turbulence behind steep cliffs. Those nonlinear separation "
                "effects are approximated via the empirical Kroy cubic Hermite separation bubble closure (theta_sep = 14 deg, L_b ~ 6 H_d).",
                "",
            ),
            (
                "Limitation 2: Quasi-Steady Aerodynamic & Saltation Equilibrium (Time-Scale Separation)",
                "The model assumes strict time-scale separation between fluid/grain transport adaptation (~0.5-5 seconds) and "
                "morphological bed evolution (~hours to years). Consequently, the wind shear field tau(x,y) and saltation flux "
                "q(x,y) are solved as steady diagnostic balances at each morphological time step dt. The model does not resolve "
                "second-by-second turbulent wind gusts, acoustic waves, or suspended dust plumes (grains d < 50 um carried in suspension).",
                "",
            ),
            (
                "Limitation 3: Monodisperse, Dry, Cohesionless Granular Bed",
                "All sand grains in a given run are modeled with a single effective mean diameter d and uniform density rho_s. "
                "Natural desert dunes often contain bimodal or polydisperse grain distributions that produce grain-size segregation "
                "(coarse grains armoring ripple crests, fine grains enriching the slipface). In addition, moisture capillary cohesion, "
                "salt crusting, and vegetation root stabilization (which drive parabolic dune inversion) are not included.",
                "",
            ),
            (
                "Limitation 4: Sub-Grid Scale Separation Between Dunes (lambda ~ 20-200 m) and Impact Ripples (lambda_r ~ 10-15 cm)",
                "Resolving 12 cm ballistic impact ripples directly inside the finite-volume Exner PDE across a 150 m x 120 m Barchan "
                "domain would require dx ~ 1 cm (over 1.8 x 10^8 grid cells per time step). To achieve real-time 60 FPS interactive "
                "performance while preserving physical accuracy, the PDE solves the macroscopic dune topography on dx ~ 1 m and "
                "superimposes the analytical Andreotti-Bagnold ballistic impact ripple field (oriented orthogonal to local tau "
                "and suppressed inside the wake and on 32-deg avalanche slipfaces) in the multi-scale 3D renderer and 2m x 2m zoom inspector.",
                "",
            ),
        ]
        return self._make_scroll_page(blocks)

    def _build_bibliography_tab(self) -> QWidget:
        blocks = [
            (
                "Foundational Aeolian Physics & Boundary-Layer Theory",
                "1. Bagnold, R. A. (1941). The Physics of Blown Sand and Desert Dunes. Methuen, London." + NL
                + "2. Owen, P. R. (1964). Saltation of uniform grains in air. Journal of Fluid Mechanics, 20(2), 225-242." + NL
                + "3. Jackson, P. S., & Hunt, J. C. R. (1975). Turbulent wind flow over a low hill. Quarterly Journal of the Royal Meteorological Society, 101(430), 929-955." + NL
                + "4. Hunt, J. C. R., Leibovich, S., & Richards, K. J. (1988). Turbulent shear flows over low hills. Q. J. R. Meteorol. Soc., 114(484), 1435-1470." + NL
                + "5. Weng, W. S., Hunt, J. C. R., Carruthers, D. J., Warren, A., Wiggs, G. F. S., Livingstone, I., & Castro, I. (1991). Air flow and sand transport over sand-dunes. Acta Mechanica Suppl., 2, 1-22.",
                "",
            ),
            (
                "Continuum Dune Morphodynamics, Saltation Lag & Barchan Modeling",
                "6. Sauermann, G., Kroy, K., & Herrmann, H. J. (2001). Continuum saltation model for sand dunes. Physical Review E, 64(3), 031305." + NL
                + "7. Kroy, K., Sauermann, G., & Herrmann, H. J. (2002). Minimal model for aeolian sand dunes. Physical Review E, 66(3), 031302." + NL
                + "8. Andreotti, B., Claudin, P., & Douady, S. (2002). Selection of dune shapes and velocities — Part 1: Dynamics of sand, wind and barchans. The European Physical Journal B, 28(3), 321-339." + NL
                + "9. Hersen, P., Douady, S., & Andreotti, B. (2002). Relevant length scale of barchan dunes. Physical Review Letters, 89(26), 264301." + NL
                + "10. Schwammle, V., & Herrmann, H. J. (2005). A model for Barchan dunes including lateral shear stress. The European Physical Journal E, 16(1), 57-65." + NL
                + "11. Duran, O., Parteli, E. J. R., & Herrmann, H. J. (2010). A continuous model for sand dunes: Review, new developments and application to barchan dunes and barchan dune fields. Earth Surface Processes and Landforms, 35(13), 1591-1600.",
                "",
            ),
            (
                "Planetary Scaling, Linear Stability, Avalanching & Ballistic Ripples",
                "12. Claudin, P., & Andreotti, B. (2006). A scaling law for aeolian dunes on Mars, Venus, Earth, and for subaqueous ripples. Earth and Planetary Science Letters, 252(1-2), 30-44." + NL
                + "13. Fourriere, A., Claudin, P., & Andreotti, B. (2010). Bedforms in a turbulent stream: formation of ripples by primary linear instability and of dunes by nonlinear pattern coarsening. Journal of Fluid Mechanics, 649, 287-328." + NL
                + "14. Andreotti, B., Claudin, P., & Pouliquen, O. (2006). Aeolian sand ripples: Experimental study of fully developed states. Physical Review Letters, 96(2), 028001." + NL
                + "15. Iversen, J. D., & White, B. R. (1982). Saltation threshold on Earth, Mars and Venus. Sedimentology, 29(1), 111-119." + NL
                + "16. Iversen, J. D., & Rasmussen, K. R. (1994). The effect of surface slope on shear velocity threshold foraving sand. Sedimentology, 41(4), 721-728." + NL
                + "17. Bouchaud, J. P., Cates, M. E., Prakash, J. R., & Edwards, S. F. (1994). A model for the dynamics of sandpile surfaces. Journal de Physique I, 4(10), 1383-1410.",
                "",
            ),
        ]
        return self._make_scroll_page(blocks)

    def _build_equations_tab_ar(self) -> QWidget:
        blocks = [
            (
                "الجزء الأول: الملف اللوغاريتمي للرياح في الطبقة الجدارية المضطربة وخشونة أوين للقفز الحبيبي (core/wind_profile.py)",
                "متى يُستخدم: يُحسب بشكل مستمر لربط سرعة الرياح المرجعية على ارتفاع 10 أمتار U(10m) بسرعة القص الاحتكاكي القاعية u_*0 وإجهاد القص غير المضطرب tau_0 = rho_f u_*0^2." + NL
                + "لماذا يُستخدم: عندما تتجاوز سرعة القص العتبة الحرجة u_*th، تسحب الحبيبات القافزة كمية حركة من طبقة الهواء القريبة من السطح، مما يولّد خشونة أيروديناميكية إضافية z0_salt تتناسب مع (u_*0^2 - u_*th^2)/(2g)." + NL
                + "المراجع المحكمة: Prandtl (1925); Bagnold (1941); Owen (1964) J. Fluid Mech. 20, 225-242.",
                "$u(z) = @frac{u_{*0}}{@kappa}@ln@left(@frac{z}{z_{0,@mathrm{grain}} + z_{0,@mathrm{salt}}}@right), "
                "@qquad z_{0,@mathrm{grain}} = @frac{d}{30}, @qquad z_{0,@mathrm{salt}} = C_{@mathrm{Owen}}@frac{u_{*0}^2 - u_{*@mathrm{th}}^2}{2g}$",
            ),
            (
                "الجزء الثاني: طول الكبح الحبيبي، طول التشبع L_sat وعتبة باغنولد للحركة (core/constants.py)",
                "متى يُستخدم: يُحسب لكل بيئة كوكبية (الأرض، المريخ، الزهرة، أو القناة المائية) ولكل قطر حبيبات d." + NL
                + "لماذا يُستخدم: تحتاج حبيبة كثافتها rho_s في مائع كثافته rho_f إلى مسافة قصور ذاتي L_drag = (rho_s/rho_f)*d لتصل إلى سرعة المائع. ويحدد طول التشبع L_sat = 2.2 L_drag أصغر طول موجي غير مستقر للكثبان lambda_min = 4*pi*L_sat وأسرع طول موجي نمواً lambda_max ~= 19.8 L_sat." + NL
                + "المراجع المحكمة: Bagnold (1941); Iversen & White (1982) Sedimentology 29; Claudin & Andreotti (2006) EPSL 252.",
                "$L_{@mathrm{drag}} = @frac{@rho_s}{@rho_f}d, @qquad L_{@mathrm{sat}} = 2.2@frac{@rho_s}{@rho_f}d, @qquad "
                "u_{*@mathrm{th}} = A_B@sqrt{@frac{@rho_s - @rho_f}{@rho_f}gd} @; (A_B = 0.10)$",
            ),
            (
                "الجزء الثالث: فقاعة انفصال الجريان الهوائي ومنحنى هيرميت التكعيبي (core/separation.py)",
                "متى يُستخدم: الخطوة الأولى في كل خطوة زمنية عندما يتجاوز انحدار الواجهة الخلفية زاوية الانفصال الحرجة theta_sep = 14 درجة (dh/dx < -0.2493)." + NL
                + "لماذا يُستخدم: تنفصل الطبقة الجدارية المضطربة عند حافة الكثيب الحادة (Brink) بسبب تدرج الضغط العكسي. يُنمذج خط الانسياب الفاصل s(z) بكثيرة حدود هيرميت التكعيبية التي تحقق اتصال C^0 و C^1 عند الحافة وتلتقي بالقاعدة عند مسافة إعادة الالتصاق L_b ~= 6 H_d، حيث ينعدم إجهاد القص (tau = 0) داخل منطقة الظل فتترسب جميع الرمال العابرة للحافة." + NL
                + "المراجع المحكمة: Kroy, Sauermann & Herrmann (2002) Phys. Rev. E 66; Duran, Parteli & Herrmann (2010) ESPL 35.",
                "$s(z) = h_b + (2H_d + h'_d L_b)z^3 - (3H_d + 2h'_d L_b)z^2 + h'_d L_b z + H_d, @qquad "
                "L_b = @frac{1.5 H_d}{@bar{s}'}@left(1 + @frac{1}{4}@nu + @frac{1}{8}@nu^2@right)$",
            ),
            (
                "الجزء الرابع: اضطراب إجهاد القص القاعي بطيف فورييه ثنائي الأبعاد (core/wind.py)",
                "متى يُستخدم: الخطوة الثانية في كل خطوة زمنية فوق الغلاف الأيروديناميكي الأملس h_tilde(x, y)." + NL
                + "لماذا يُستخدم: يحل معادلات كمية الحركة الخطية للطبقة الجدارية في فضاء فورييه 2D FFT. يمثل المعامل A تسارع الرياح فوق القمة (تأثير برنولي)، بينما يمثل المعامل التربيعي +i B kx إزاحة أقصى إجهاد قص عكس اتجاه الرياح (قبل القمة)، وهي الآلية الهيدروديناميكية الأساسية المسؤولة عن نمو الكثبان الرملية." + NL
                + "المراجع المحكمة: Jackson & Hunt (1975) QJRMS 101; Fourriere, Claudin & Andreotti (2010) JFM 649.",
                "$@hat{@tau}_{@mathrm{pert},x}(k_x,k_y) = @left(A@frac{k_x^2}{|k|} + iBk_x@right)@hat{@tilde{h}}(k_x,k_y), @qquad "
                "@hat{@tau}_{@mathrm{pert},y}(k_x,k_y) = A_y@frac{k_x k_y}{|k|}@hat{@tilde{h}}(k_x,k_y)$",
            ),
            (
                "الجزء الخامس: استرخاء التدفق بالقفز الحبيبي وتعديل العتبة بانحدار السطح (core/transport.py)",
                "متى يُستخدم: الخطوة الثالثة في كل خطوة زمنية عبر كامل النطاق الحسابي." + NL
                + "لماذا يُستخدم: لا يتكيف تدفق الرمال q_x لحظياً مع إجهاد القص المحلي، بل يسترخي مكانياً على مدى طول التشبع L_sat. كما تزداد عتبة القص الفعالة tau_th,eff على المنحدرات الصاعدة بفعل الجاذبية مما يضبط ميل ظهر الكثيب عند ~10 درجات." + NL
                + "المراجع المحكمة: Lettau & Lettau (1978); Iversen & Rasmussen (1994); Sauermann, Kroy & Herrmann (2001) PRE 64.",
                "$L_{@mathrm{sat}}@frac{@partial q_x}{@partial x} = q_{@mathrm{sat}}(@tau) - q_x, @qquad "
                "q_{@mathrm{sat}}(@tau) = C_{@mathrm{flux}}@frac{@rho_f}{g}u_*^3@left(1 - @frac{@tau_{@mathrm{th,eff}}}{@tau}@right)_+, @qquad "
                "@tau_{@mathrm{th,eff}} = @tau_{@mathrm{th},0}@left(1 + @frac{@partial_x h}{@tan@theta_r}@right)$",
            ),
        ]
        return self._make_scroll_page(blocks)

    def _build_integration_tab_ar(self) -> QWidget:
        blocks = [
            (
                "صيغة التكامل الأولى: مكامل أويلر الأسي المكاني الموجب دوماً (core/transport.py)",
                "لماذا يفشل تكامل أويلر الأمامي التقليدي: عندما يكون طول التشبع L_sat أصغر من خطوة الشبكة dx، فإن صيغة أويلر الصريحة تعطي قيماً سالبة وتذبذبات غير فيزيائية." + NL
                + "الصيغة التكاملية الدقيقة المستخدمة: بتكامل المعادلة التفاضلية الخطية تحليلياً على كل خلية نحصل على صيغة أويلر الأسية المستقرة مطلقاً، مع قيد الحجم المنتهي لصخر الأساس:",
                "$@mu = @exp@left(-@frac{@Delta x}{L_{@mathrm{sat}}}@right), @qquad "
                "q_{x,i}^{@mathrm{ideal}} = q_{x,i-1}@,@mu + q_{@mathrm{sat},i}(1 - @mu), @qquad "
                "q_{x,i} = @min@left(q_{x,i}^{@mathrm{ideal}},@; q_{x,i-1} + @frac{@rho_b @Delta x}{@Delta t}(h_i - h_{@mathrm{floor}})_+@right)$",
            ),
            (
                "صيغة التكامل الثانية: طريقة الخلية المانحة (Donor-Cell Upwind) للتدفق العرضي q_y (core/transport.py, core/exner.py)",
                "الصيغة المستخدمة: يُحسَب التدفق العرضي q_y على أوجه الخلايا المتداخلة j + 1/2 باستخدام انتقاء الخلية المانحة عكس الرياح وفي اتجاه الانحدار، مما يمنع تذبذبات نايكويست ويحافظ على صخر الأساس من الحت الزائف:",
                "$q_{y, i, j+1/2} = q_{x,@mathrm{upwind}}@,@frac{@tau_{y,i,j} + @tau_{y,i,j+1}}{2@tau_0} "
                "- D_{@mathrm{lat}}@,q_{x,@mathrm{donor}}@,@frac{h_{i,j+1} - h_{i,j}}{@Delta y}$",
            ),
            (
                "صيغة التكامل الثالثة: معادلة إكسنر بالحجوم المنتهية والخطوة الزمنية المتكيفة CFL (core/exner.py)",
                "صيغة التكامل الزمني: يتطور ارتفاع السطح الرملي h(x,y,t) زمنياً عبر فروق التدفق المحافظة عكس الرياح مع خطوة زمنية متكيفة تحقق شرط كوران-فريدريكس-ليفي المورفولوجي:",
                "$h_{i,j}^{n+1,*} = h_{i,j}^n - @frac{@Delta t}{@rho_{@mathrm{bulk}}}@left(@frac{q_{x,i,j} - q_{x,i-1,j}}{@Delta x} "
                "+ @frac{q_{y,i,j+1/2} - q_{y,i,j-1/2}}{@Delta y}@right), @qquad "
                "@Delta t = @mathrm{clip}@left(@frac{@mathrm{CFL}@,@Delta x@,@tan 32^@circ}{@max |@partial_t h|},@; @Delta t_{@min},@; @Delta t_{@max}@right)$",
            ),
            (
                "صيغة التكامل الرابعة: استرخاء الانهيار الحبيبي الإقليدي المتساوي المناحي 2D (core/avalanche.py)",
                "الصيغة المستخدمة: يُحسب معيار الانحدار الإقليدي ثنائي الأبعاد |grad h|_e على كل حافة خلية مع المشتقات المتقاطعة، بحيث تنهار الواجهات التي تتجاوز زاوية الاستقرار السكوني (34 درجة) بشكل متماثل في جميع الاتجاهات إلى زاوية الاستقرار الديناميكي (32 درجة):",
                "$|@nabla h|_e = @sqrt{s_x^2 + s_y^2}, @qquad "
                "F_{@mathrm{aval},e} = @omega@,@Delta h_e@,@max@left(0,@; 1 - @frac{@tan 32^@circ}{|@nabla h|_e}@right), @qquad "
                "h @leftarrow h + F_{@mathrm{in}} - F_{@mathrm{out}}$",
            ),
        ]
        return self._make_scroll_page(blocks)

    def _build_limitations_tab_ar(self) -> QWidget:
        blocks = [
            (
                "القيد الأول: نظرية الاضطراب الخطي من الرتبة الأولى للطبقة الجدارية (نسبة الارتفاع إلى الطول H/L < 0.25)",
                "يعتمد حل جاكسون-هانت (1975) الطيفي على مفكوك خطي من الرتبة الأولى لمعادلات الطبقة الجدارية المضطربة. وهو دقيق للغاية على المنحدرات الأمامية للكثبان (8–12 درجة)، لكنه لا يحل دوامات نافييه-ستوكس ثلاثية الأبعاد خلف الجروف الحادة مباشرة، بل يستعيض عنها بنموذج فقاعة الانفصال التكعيبي لـ Kroy et al. (2002).",
                "",
            ),
            (
                "القيد الثاني: فرضية التوازن شبه المستقر للرياح والقفز الحبيبي (فصل المقاييس الزمنية)",
                "يفترض النموذج أن زمن استجابة الرياح وسحابة الحبيبات القافزة (~1-5 ثوانٍ) أصغر بكثير من الزمن المورفولوجي لتطور الكثيب (ساعات إلى سنوات). لذلك يُحل حقل القص والقفز كحالة مستقرة عند كل خطوة مورفولوجية، ولا يشمل النموذج العواصف الغبارية العالقة (حبيبات الطمي والغبار d < 50 ميكرومتر).",
                "",
            ),
            (
                "القيد الثالث: فرضية الحبيبات الجافة المتجانسة غير المتماسكة",
                "تُحاكى الرمال بقطر حبيبي فعال موحد d وكثافة موحدة rho_s دون قوى تماسك شعرية ناتجة عن الرطوبة أو تثبيت جذور النباتات (التي تحول كثبان البرخان إلى كثبان مكافئة Parabolic Dunes) ودون فرز حبيبي متعدد الأقطار.",
                "",
            ),
            (
                "القيد الرابع: التمثيل متعدد المقاييس لتموجات التصادم القذفي (Sub-Grid Ballistic Ripples)",
                "نظراً لأن طول موجة تموجات الرمال (~12 سم) أصغر بمئات المرات من أبعاد حقل الكثبان (~120 م)، فإن حلها مباشرة في شبكة إكسنر يتطلب أكثر من 10^8 خلية. لذا يحل المحرك الكثبان بدقة على الشبكة الماكروسكوبية ويركب تموجات التصادم الفيزيائية تحليلياً وفق قانون أندريوتي وباغنولد (2006) في العرض ثلاثي الأبعاد وفاحص الرقعة المجهرية (2م × 2م).",
                "",
            ),
        ]
        return self._make_scroll_page(blocks)
