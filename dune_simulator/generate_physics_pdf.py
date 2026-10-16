"""Generates a comprehensive 7-page publication-grade PDF guide explaining the physical models,
mathematical formulations, numerical schemes, and when/why each equation is used in dune_simulator.
"""

import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

BS = chr(92)


def _tex(s: str) -> str:
    """Converts '@' LaTeX command prefix to backslash to avoid escape sequence ambiguity."""
    return s.replace("@", BS)


def _add_page_header(fig: plt.Figure, page_num: int, total_pages: int, section_title: str) -> None:
    """Adds a consistent top header bar and page number footer to a PDF page figure."""
    fig.text(
        0.05,
        0.965,
        "GROUNDED AEOLIAN & SUBAQUEOUS DUNE SIMULATOR — PHYSICAL MODELS & EQUATION GUIDE",
        fontsize=8.5,
        fontweight="bold",
        color="#4a5568",
    )
    fig.text(
        0.95,
        0.965,
        f"Page {page_num} of {total_pages}",
        fontsize=8.5,
        fontweight="bold",
        color="#4a5568",
        ha="right",
    )
    fig.text(
        0.05,
        0.935,
        section_title,
        fontsize=14.5,
        fontweight="bold",
        color="#1a202c",
    )
    line = plt.Line2D(
        [0.05, 0.95], [0.922, 0.922], transform=fig.transFigure, color="#b8864b", linewidth=1.8
    )
    fig.add_artist(line)


def _add_text_box(
    fig: plt.Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    body_lines: list,
    bg_color: str = "#f8f9fa",
    border_color: str = "#cbd5e0",
    title_color: str = "#2b6cb0",
    fontsize: float = 9.0,
    line_spacing: float = 0.021,
) -> None:
    """Draws a rounded callout box with a bold header and mathtext-enabled body lines."""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        transform=fig.transFigure,
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.2,
        zorder=1,
    )
    fig.add_artist(patch)

    cur_y = y + h - 0.022
    if title:
        fig.text(
            x + 0.014,
            cur_y,
            title,
            fontsize=fontsize + 0.8,
            fontweight="bold",
            color=title_color,
            zorder=2,
        )
        cur_y -= line_spacing * 1.25

    for line in body_lines:
        if line == "":
            cur_y -= line_spacing * 0.55
            continue
        is_eq = line.startswith("EQ:")
        if is_eq:
            eq_str = _tex(line[3:].strip())
            cur_y -= line_spacing * 0.25
            fig.text(
                x + w * 0.5,
                cur_y,
                eq_str,
                fontsize=fontsize + 0.4,
                color="#1a202c",
                ha="center",
                zorder=2,
            )
            cur_y -= line_spacing * 1.45
        else:
            fig.text(
                x + 0.014,
                cur_y,
                _tex(line),
                fontsize=fontsize,
                color="#2d3748",
                zorder=2,
            )
            cur_y -= line_spacing


def _embed_image(fig: plt.Figure, rect: list, img_path: str, caption: str = "") -> None:
    """Embeds a high-resolution PNG figure into the given figure rectangle [left, bottom, width, height]."""
    if not os.path.exists(img_path):
        return
    ax = fig.add_axes(rect)
    img = plt.imread(img_path)
    ax.imshow(img)
    ax.axis("off")
    if caption:
        fig.text(
            rect[0] + rect[2] * 0.5,
            rect[1] - 0.012,
            caption,
            fontsize=8.2,
            style="italic",
            color="#4a5568",
            ha="center",
        )


def generate_pdf(output_pdf_path: str, img_dir: str) -> None:
    total_pages = 7
    with PdfPages(output_pdf_path) as pdf:
        # =========================================================================
        # PAGE 1: EXECUTIVE OVERVIEW & MORPHODYNAMIC FEEDBACK ARCHITECTURE
        # =========================================================================
        fig = plt.figure(figsize=(8.5, 11.0), dpi=300)
        _add_page_header(
            fig, 1, total_pages, "1. Morphodynamic Feedback Architecture & Master Equation Matrix"
        )

        _add_text_box(
            fig,
            0.05,
            0.72,
            0.90,
            0.185,
            "Physical Foundation: Why Sand Dunes Are Self-Organizing Aerodynamic-Granular Waves",
            [
                "A sand dune is not a static pile of grains; it is a dynamic wave emerging from the closed feedback loop between",
                "a turbulent fluid boundary layer (wind or water) and a mobile granular bed:",
                "  (1) Topography h(x, y, t) perturbs the turbulent boundary layer, accelerating flow over the stoss slope and",
                "      detaching into a recirculating wake eddy behind steep lee slopes (> 14 deg).",
                "  (2) Because of fluid inertia in the inner shear layer, the basal shear stress maximum tau_max is shifted UPWIND",
                "      of the topographic crest (Jackson & Hunt 1975). Conversely, grain inertia causes the actual sand flux q(x)",
                "      to lag DOWNWIND of the shear stress by the saturation length L_sat (Claudin & Andreotti 2006).",
                "  (3) The competition between the aerodynamic upwind phase advance and the inertial downwind saturation lag",
                "      determines whether bed perturbations amplify into dunes (lambda > lambda_min) or decay into a flat bed.",
            ],
            bg_color="#fffaf0",
            border_color="#d69e2e",
            title_color="#975a16",
            fontsize=8.6,
            line_spacing=0.0175,
        )

        _add_text_box(
            fig,
            0.05,
            0.185,
            0.90,
            0.515,
            "Master Guide: Core Equations, When Each Is Triggered & Why It Is Essential",
            [
                "1. Drag & Saturation Scales (constants.py):  L_drag = (rho_s / rho_f) d,   L_sat = 2.2 L_drag",
                "   - WHEN USED: Initialization of any planetary environment and at every spatial flux relaxation step.",
                "   - WHY USED: Sets the distance required for saltating grains to accelerate to fluid speed; controls the minimum",
                "     and fastest-growing dune wavelengths across Earth (0.95 m), Mars (44 m), Venus (1.4 cm), and water (1.2 mm).",
                "",
                "2. Separation Bubble Envelope (separation.py):  h_tilde(x) = max(h(x), s_Hermite(x))  when  dh/dx < -tan(14 deg)",
                "   - WHEN USED: Step 1 of every time step whenever any lee slope exceeds the critical detachment angle (14 deg).",
                "   - WHY USED: Turbulent boundary layers cannot remain attached down a steep slipface. The cubic Hermite streamline",
                "     encloses the zero-shear wake eddy (L_b ~ 6 H_d) and provides a smooth C^1 envelope for the FFT wind solver.",
                "",
                "3. Jackson-Hunt / Kroy FFT Shear Stress (wind.py):  tau_pert_hat(k) = [A (kx^2/|k|) + i B kx] h_tilde_hat(k)",
                "   - WHEN USED: Step 2 of every time step over the aerodynamic envelope h_tilde(x, y); masked to 0 inside wake.",
                "   - WHY USED: Captures nonlocal Bernoulli pressure speed-up over the crest (A |k|) and the inner-layer upwind",
                "     phase shift (+i B kx) that drives sand erosion on the windward stoss and deposition near the crest.",
                "",
                "4. Saturated Flux & Slope Threshold (transport.py):  q_sat = C (rho_f/g) u_*^3 (1 - tau_th,eff / tau)_+",
                "   - WHEN USED: Step 3a of every time step where basal shear tau exceeds slope-adjusted threshold tau_th,eff.",
                "   - WHY USED: Quantifies the maximum steady-state mass transport capacity of saltation, while the Bagnold-Howard",
                "     gravity correction tau_th,eff = tau_th (1 + dh/dx / tan 34 deg) prevents unphysical stoss over-steepening.",
                "",
                "5. Non-Local Exponential Relaxation (transport.py):  L_sat dq_x/dx = q_sat - q_x   &   q_y = q_x(tau_y/tau_0 - D_lat dh/dy)",
                "   - WHEN USED: Step 3b of every time step, sweeping downwind in +x and across transverse faces j + 1/2 in y.",
                "   - WHY USED: Solves the spatial lag of saltation adaptation (essential to damp short-wavelength noise) and models",
                "     lateral wind deflection + splash diffusion in 2D, while capping entrainment on bare non-erodible bedrock.",
                "",
                "6. Exner Conservation & Isotropic Avalanche (exner.py, avalanche.py):  rho_b dh/dt + div(q) = 0,  |grad h| <= tan(32 deg)",
                "   - WHEN USED: Steps 4-5 of every time step (upwind finite-volume step + Jacobi slope relaxation when |grad h| > tan 34 deg).",
                "   - WHY USED: Strictly conserves total sand volume to machine precision (< 1e-13) and relaxes over-steepened lee",
                "     faces isotropically to the dynamic angle of repose (32 deg), forming the razor-sharp Barchan slipface.",
            ],
            bg_color="#f7fafc",
            border_color="#a0aec0",
            title_color="#2c5282",
            fontsize=8.4,
            line_spacing=0.0180,
        )

        _add_text_box(
            fig,
            0.05,
            0.035,
            0.90,
            0.135,
            "Closed-Loop Execution Pipeline in Every Time Step dt (DuneSimulator1D & DuneSimulator2D)",
            [
                "EQ: $h(x,y,t) @;@longrightarrow@; @tilde{h}(x,y),@,@mathrm{wake} @;@longrightarrow@; @tau_x,@tau_y"
                " @;@longrightarrow@; q_x,q_y @;@longrightarrow@; h^*(x,y) @;@longrightarrow@; h(x,y,t+@Delta t)$",
                "Key Numerical Guarantee: Every spatial operator is matched to the physical characteristic direction of information",
                "propagation: pseudo-spectral nonlocal FFT for elliptic pressure-shear perturbation, exact exponential integration",
                "for hyperbolic downwind flux relaxation, donor-cell upwinding for Exner divergence, and flux-conservative",
                "isotropic edge relaxation for granular avalanching.",
            ],
            bg_color="#edf2f7",
            border_color="#718096",
            title_color="#1a202c",
            fontsize=8.4,
            line_spacing=0.0175,
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =========================================================================
        # PAGE 2: PLANETARY SCALINGS & CHARACTERISTIC LENGTHS (constants.py)
        # =========================================================================
        fig = plt.figure(figsize=(8.5, 11.0), dpi=300)
        _add_page_header(
            fig, 2, total_pages, "2. Planetary Boundary Layer Parameters & Universal Length Scales"
        )

        _add_text_box(
            fig,
            0.05,
            0.55,
            0.90,
            0.355,
            "Governing Equations in core/constants.py (Bagnold 1941; Claudin & Andreotti 2006)",
            [
                "Given gravitational acceleration g [m/s^2], fluid density rho_f [kg/m^3], grain density rho_s [kg/m^3],",
                "mean grain diameter d [m], and granular bed packing fraction phi_p = 0.62:",
                "",
                "EQ: $@rho_{@mathrm{bulk}} = @phi_p @, @rho_s, @qquad L_{@mathrm{drag}} = @frac{@rho_s}{@rho_f}@, d,"
                " @qquad L_{@mathrm{sat}} = 2.2@, L_{@mathrm{drag}} = 2.2@, @frac{@rho_s}{@rho_f}@, d$",
                "",
                "EQ: $u_{*{@mathrm{th}}} = A_B @sqrt{@frac{@rho_s - @rho_f}{@rho_f}@, g@, d} @quad (A_B = 0.10),"
                " @qquad @tau_{@mathrm{th}} = @rho_f@, u_{*{@mathrm{th}}}^2, @qquad @tau_0 = @rho_f@, u_{*0}^2$",
                "",
                "EQ: $@lambda_{@min} = 2@pi @frac{A}{B} L_{@mathrm{sat}} = 4@pi L_{@mathrm{sat}} @approx 12.57@, L_{@mathrm{sat}},"
                " @qquad @lambda_{@max} @approx 19.78@, L_{@mathrm{sat}}$",
                "",
                "- WHEN USED: Evaluated when instantiating a PlanetaryEnvironment (EARTH_DESERT, MARS_DUNES, VENUS_DUNES,",
                "  AQUEOUS_FLUME) and referenced at every transport, dispersion, and Exner conservation step.",
                "- WHY USED (Physical Derivation):",
                "  * A stationary grain of mass m ~ rho_s d^3 entrained into a fluid stream of density rho_f experiences an",
                "    aerodynamic drag force F_D ~ rho_f u^2 d^2. By Newton's second law (m du/dt = m u du/dx ~ F_D), the",
                "    distance required to accelerate the grain to the fluid speed is L_drag = (rho_s / rho_f) d — independent of",
                "    wind speed and gravity! Because saltation saturation involves grain-bed splash cascades, wind-tunnel and field",
                "    measurements (Claudin & Andreotti 2006) show L_sat = 2.2 L_drag.",
                "  * Because L_sat scales inversely with fluid density rho_f, thin Martian air (rho_f = 0.015 kg/m^3) produces",
                "    L_sat = 44.0 m and giant 858 m dunes, whereas dense water (rho_f = 1000 kg/m^3) produces L_sat = 1.17 mm",
                "    and 2.3 cm current ripples — explaining five orders of magnitude of bedform sizes with a single law!",
            ],
            bg_color="#f8f9fa",
            border_color="#cbd5e0",
            title_color="#2b6cb0",
            fontsize=8.6,
            line_spacing=0.0175,
        )

        _embed_image(
            fig,
            [0.06, 0.05, 0.88, 0.47],
            os.path.join(img_dir, "05_planetary_comparison.png"),
            "Figure 1 (from 05_planetary_comparison.py): Linear growth rate sigma(lambda) across Earth, Mars, Venus, and Subaqueous regimes.",
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =========================================================================
        # PAGE 3: FLOW SEPARATION BUBBLE & AERODYNAMIC ENVELOPE (separation.py)
        # =========================================================================
        fig = plt.figure(figsize=(8.5, 11.0), dpi=300)
        _add_page_header(
            fig, 3, total_pages, "3. Flow Separation Bubble & Aerodynamic Envelope Model"
        )

        _add_text_box(
            fig,
            0.05,
            0.52,
            0.90,
            0.385,
            "Governing Equations in core/separation.py (Kroy, Sauermann & Herrmann 2002)",
            [
                "Flow detachment occurs at any brink point x_d where the downward lee slope exceeds theta_sep = 14 deg:",
                "EQ: $@left.@frac{@partial h}{@partial x}@right|_{x_d^+} < -s_{@mathrm{crit}} = -@tan(14^@circ) @approx -0.2493,"
                " @qquad H_d = h(x_d) - h_{@mathrm{base}}, @qquad @bar{s}' = 0.25$",
                "",
                "With clamped upwind launch slope h'_d = clip(dh/dx(x_d^-), -s_bar', +s_bar') and dimensionless parameter",
                "nu = h'_d / s_bar', the separation bubble reattachment distance L_b and cubic Hermite streamline s(z) are:",
                "EQ: $L_b = @frac{1.5@, H_d}{@bar{s}'}@left(1 + @frac{1}{4}@nu + @frac{1}{8}@nu^2@right),"
                " @qquad z = @frac{x - x_d}{L_b} @in [0, 1]$",
                "",
                "EQ: $s(z) = h_{@mathrm{base}} + (2 H_d + h'_d L_b) z^3 - (3 H_d + 2 h'_d L_b) z^2 + (h'_d L_b) z + H_d$",
                "EQ: $@tilde{h}(x) = @max@left(h(x),@, s(x)@right), @qquad @mathrm{in@_wake}(x) = @mathbf{1}@left(s(x) > h(x)@right)$",
                "",
                "- WHEN USED: Computed before the wind solver at every time step in 1D and along every streamwise slice in 2D.",
                "- WHY USED (Physical & Mathematical Rationale):",
                "  1. Adverse Pressure Gradient Detachment: When air flows over a sharp brink onto a 32-deg slipface, the rapid",
                "     expansion creates a strong adverse pressure gradient that detaches the boundary layer, forming a",
                "     recirculating Karman-like eddy extending ~6 H_d downwind.",
                "  2. Exact Hermite Boundary Conditions: The cubic polynomial s(z) is the unique lowest-order curve satisfying",
                "     all four physical boundary conditions of a dividing streamline: C^0 & C^1 launch continuity at the brink",
                "     (s(0) = h(x_d), s'(0) = h'_d) and smooth tangential reattachment at the bed (s(1) = h_base, s'(1) = 0).",
                "  3. Outer Flow Regularization: Passing the smooth envelope h_tilde(x) instead of the sharp-kinked bed h(x) to",
                "     the FFT wind solver prevents Gibbs ringing, while setting tau = 0 inside in_wake forces all saltating",
                "     grains crossing the brink to deposit immediately onto the upper slipface.",
            ],
            bg_color="#f8f9fa",
            border_color="#cbd5e0",
            title_color="#2b6cb0",
            fontsize=8.5,
            line_spacing=0.0172,
        )

        _embed_image(
            fig,
            [0.06, 0.04, 0.88, 0.45],
            os.path.join(img_dir, "03_step_topography_response.png"),
            "Figure 2 (from 03_step_topography_response.py): Flow detachment, L_b = 6 H_d wake bubble, and shadow dune trapping behind a step.",
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =========================================================================
        # PAGE 4: TURBULENT WIND SHEAR PERTURBATION & LINEAR INSTABILITY (wind.py, dispersion.py)
        # =========================================================================
        fig = plt.figure(figsize=(8.5, 11.0), dpi=300)
        _add_page_header(
            fig, 4, total_pages, "4. Jackson-Hunt FFT Shear Stress Solver & Primary Instability"
        )

        _add_text_box(
            fig,
            0.05,
            0.52,
            0.90,
            0.385,
            "Governing Equations in core/wind.py & analysis/dispersion.py (Jackson & Hunt 1975; Fourriere et al. 2010)",
            [
                "In Fourier space, the relative basal shear stress perturbation over the aerodynamic envelope h_tilde is:",
                "EQ: $@hat{@tau}_{@mathrm{pert}, x}(k_x, k_y) = @left(A@,@frac{k_x^2}{|k|} + i@,B@,k_x@right)"
                " e^{-@frac{1}{2}(|k| @ell_s)^2}@,@hat{@tilde{h}}(k_x, k_y), @qquad"
                " @hat{@tau}_{@mathrm{pert}, y}(k_x, k_y) = A_y@,@frac{k_x k_y}{|k|}"
                " e^{-@frac{1}{2}(|k| @ell_s)^2}@,@hat{@tilde{h}}(k_x, k_y)$",
                "",
                "In 1D spatial coordinates, this is equivalent to the nonlocal Hilbert transform convolution:",
                "EQ: $@tau_{@mathrm{pert}}(x) = A@,@mathcal{H}@left[@frac{@partial @tilde{h}}{@partial x}@right](x)"
                " + B@,@frac{@partial @tilde{h}}{@partial x}(x), @qquad"
                " @tau(x) = @tau_0@,@max@left(0,@, 1 + @tau_{@mathrm{pert}}(x)@right)@;@; (@tau = 0 @;@mathrm{in@;wake})$",
                "",
                "Linearizing the coupled (tau, q, h) system around a flat bed yields the exact dispersion relation:",
                "EQ: $@sigma(k) = @frac{Q_1}{@rho_{@mathrm{bulk}}}@,@frac{k^2@,(B - A@,|k|@,L_{@mathrm{sat}})}{1 + (k@,L_{@mathrm{sat}})^2},"
                " @qquad c(k) = @frac{Q_1}{@rho_{@mathrm{bulk}}}@,@frac{|k|@,(A + B@,|k|@,L_{@mathrm{sat}})}{1 + (k@,L_{@mathrm{sat}})^2}$",
                "",
                "- WHEN USED: Step 2 of every time step (FFT shear solver) and in spectral stability diagnostics.",
                "- WHY USED (Physical Meaning of Every Term):",
                "  * In-Phase Term A |k| (A = 3.2): Potential outer-flow streamline compression (Bernoulli effect). It is symmetric",
                "    and peaks at the crest. In the dispersion relation sigma(k), it appears as -A |k| L_sat (stabilizing short waves!).",
                "  * Quadrature Term +i B k (B = 1.6): Turbulent shear-stress phase advance in the inner layer (thickness l). It is",
                "    proportional to the upwind slope dh/dx, shifting the peak shear stress UPWIND of the crest by angle arctan(B/A).",
                "    In sigma(k), +B is the SOLE positive destabilizing term that pumps sand from troughs to crests!",
                "  * Transverse Term A_y kx ky / |k| (A_y = 1.6): Lateral pressure gradient deflecting wind around 3D mounds.",
            ],
            bg_color="#f8f9fa",
            border_color="#cbd5e0",
            title_color="#2b6cb0",
            fontsize=8.5,
            line_spacing=0.0172,
        )

        _embed_image(
            fig,
            [0.06, 0.04, 0.88, 0.45],
            os.path.join(img_dir, "01_spontaneous_dune_formation.png"),
            "Figure 3 (from 01_spontaneous_dune_formation.py): Spontaneous emergence of periodic dunes at lambda_max ~= 19.8 L_sat from 1 cm noise.",
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =========================================================================
        # PAGE 5: SALTATION TRANSPORT, SLOPE THRESHOLD & EXPONENTIAL RELAXATION (transport.py)
        # =========================================================================
        fig = plt.figure(figsize=(8.5, 11.0), dpi=300)
        _add_page_header(
            fig, 5, total_pages, "5. Non-Local Saltation Flux, Slope Threshold & Bedrock Limiting"
        )

        _add_text_box(
            fig,
            0.05,
            0.52,
            0.90,
            0.385,
            "Governing Equations in core/transport.py (Sauermann, Kroy & Herrmann 2001; Duran et al. 2010)",
            [
                "1. Slope-Modified Threshold & Saturated Sand Flux Capacity q_sat [kg/(m s)]:",
                "EQ: $@tau_{@mathrm{th,eff}}(x,y) = @tau_{@mathrm{th},0}@,@mathrm{clip}@left(1 + @frac{@partial_x h}{@tan@theta_{@mathrm{rep}}},@; 0.2,@; 2.5@right),"
                " @qquad q_{@mathrm{sat}}(@tau) = C_{@mathrm{flux}}@,@frac{@rho_f}{g}@,u_*^3@left(1 - @frac{@tau_{@mathrm{th,eff}}}{@tau}@right)_+$",
                "",
                "2. Non-Local Spatial Relaxation PDE & Exact Exponential Euler Recurrence (with mu = exp(-dx / L_sat)):",
                "EQ: $L_{@mathrm{sat}}@,@frac{@partial q_x}{@partial x} = q_{@mathrm{sat}}(x) - q_x(x)"
                " @quad@Longrightarrow@quad q_{x, i}^{@mathrm{ideal}} = q_{x, i-1}@,@mu + q_{@mathrm{sat}, i}@,(1 - @mu)$",
                "",
                "3. Non-Erodible Bedrock Limiter & Donor-Cell Transverse Flux across y-interfaces j + 1/2:",
                "EQ: $q_{x, i} = @min@left(q_{x, i}^{@mathrm{ideal}},@; q_{x, i-1} + @frac{@rho_{@mathrm{bulk}}@,@Delta x}{@Delta t}@,@max(0, h_i - h_{@mathrm{floor}})@right),"
                " @qquad q_{y, j+1/2} = q_{x,@mathrm{up}}@,@frac{@tau_y}{@tau_0} - D_{@mathrm{lat}}@,q_{x,@mathrm{donor}}@,@frac{h_{j+1}-h_j}{@Delta y}$",
                "",
                "- WHEN USED: Step 3 of every time step after computing basal shear stress tau.",
                "- WHY USED (Physical & Numerical Necessity):",
                "  1. Why Instantaneous Flux (q = q_sat) Fails Catastrophically: If L_sat -> 0, the growth rate becomes",
                "     sigma(k) ~ B k^2, which grows fastest at k -> infinity (grid-scale 2*dx noise explodes in a few steps!).",
                "     The spatial relaxation ODE L_sat dq/dx = q_sat - q introduces the physical phase delay that makes",
                "     sigma(k) < 0 for all wavelengths shorter than lambda_min = 4*pi*L_sat.",
                "  2. Why Exponential Euler Is Used: When dx >= L_sat (e.g., subaqueous ripples or coarse grids), standard forward",
                "     Euler (1 - dx/L_sat) oscillates and goes negative. Exponential Euler mu = exp(-dx/L_sat) is unconditionally",
                "     positive, monotone, and exact to machine precision (< 1e-12) for piecewise-constant q_sat.",
                "  3. Why Forward Slope dh/dx Is Used in tau_th,eff: Pairing forward difference dh/dx in q_sat with backward",
                "     difference dq/dx in the Exner equation forms a centered diffusive operator (h_{i+1} - 2h_i + h_{i-1})/dx^2",
                "     that damps Nyquist modes while naturally regulating the windward stoss angle to ~10 deg.",
            ],
            bg_color="#f8f9fa",
            border_color="#cbd5e0",
            title_color="#2b6cb0",
            fontsize=8.4,
            line_spacing=0.0170,
        )

        _embed_image(
            fig,
            [0.08, 0.03, 0.84, 0.46],
            os.path.join(img_dir, "02_gaussian_heap_diagnostic.png"),
            "Figure 4 (from 02_gaussian_heap_evolution.py): Synchronized 3-panel profile of h(x), h_tilde(x), tau(x), q_sat(x), and relaxed q(x).",
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =========================================================================
        # PAGE 6: EXNER CONSERVATION, ISOTROPIC AVALANCHE & 1D MIGRATION (exner.py, avalanche.py)
        # =========================================================================
        fig = plt.figure(figsize=(8.5, 11.0), dpi=300)
        _add_page_header(
            fig, 6, total_pages, "6. Exner Mass Conservation, Isotropic Avalanche & 1/H Migration Law"
        )

        _add_text_box(
            fig,
            0.05,
            0.48,
            0.90,
            0.425,
            "Governing Equations in core/exner.py & core/avalanche.py (Exner 1925; Bouchaud et al. 1994)",
            [
                "1. Upwind Finite-Volume Exner Equation & Morphological CFL Time Step:",
                "EQ: $@frac{@partial h}{@partial t} = -@frac{1}{@rho_{@mathrm{bulk}}}@left(@frac{q_{x, i, j} - q_{x, i-1, j}}{@Delta x}"
                " + @frac{q_{y, i, j+1/2} - q_{y, i, j-1/2}}{@Delta y}@right), @qquad"
                " @Delta t = @mathrm{clip}@left(@frac{@mathrm{CFL}@,@Delta x@,@tan 32^@circ}{@max |@partial_t h|},@, @Delta t_{@min},@, @Delta t_{@max}@right)$",
                "",
                "2. Isotropic 2D Cellular Avalanche Relaxation (Triggered where Euclidean slope |grad h| > tan 34 deg):",
                "EQ: $|@nabla h|_e = @sqrt{@left(@frac{@partial h}{@partial x}@right)_e^2 + @left(@frac{@partial h}{@partial y}@right)_e^2},"
                " @qquad F_{@mathrm{aval}, e} = @omega@,@Delta h_e@,@max@left(0,@; 1 - @frac{@tan 32^@circ}{|@nabla h|_e}@right)$",
                "",
                "3. Kinematic Dune Migration Speed & Inverse-Height Scaling Law:",
                "EQ: $c = @frac{q_{@mathrm{brink}}}{@rho_{@mathrm{bulk}}@,H} @propto @frac{1}{H}$",
                "",
                "- WHEN USED: Steps 4 and 5 of every time step.",
                "- WHY USED (Physical & Numerical Rationale):",
                "  1. Upwind Differencing in +x: Because sand transport is a hyperbolic wave propagating downwind (c > 0),",
                "     central differencing of dq_x/dx is unconditionally unstable (dispersive ringing). First-order backward",
                "     (upwind) differencing is monotone, telescoping (sum(div_q) = 0 to < 1e-14), and shock-capturing.",
                "  2. Why Isotropic 2D Euclidean Slope |grad h| Is Critical for Barchan Dunes: A Barchan slipface curves from",
                "     0 deg at the centerline to ~45 deg along the inner horns. If an avalanche solver checks |dh/dx| > tan 34 deg",
                "     and |dh/dy| > tan 34 deg separately along grid axes, diagonal faces at 45 deg can stand at",
                "     tan(32 deg)/cos(45 deg) = tan(41.5 deg) — 30% steeper than the center, creating false twin peaks on the",
                "     shoulders! Using the Euclidean norm |grad h|_e relaxes all slipface orientations isotropically to 32 deg.",
                "  3. Why c ~ 1/H Forms Barchan Horns: Integrating dh/dt = -c dh/dx = -(1/rho_b) dq/dx across the slipface [x_b, x_t]",
                "     where q(x_t) = 0 inside the wake gives c H = q_brink / rho_b. Because the lateral flanks of a 3D mound have",
                "     smaller height H(y) < H_center, they migrate faster (c(y) ~ 1/H(y)), advancing ahead to form the two horns!",
            ],
            bg_color="#f8f9fa",
            border_color="#cbd5e0",
            title_color="#2b6cb0",
            fontsize=8.35,
            line_spacing=0.0168,
        )

        _embed_image(
            fig,
            [0.06, 0.04, 0.88, 0.41],
            os.path.join(img_dir, "02_gaussian_heap_evolution.png"),
            "Figure 5 (from 02_gaussian_heap_evolution.py): Transformation of a symmetric Gaussian heap into a shape-invariant migrating dune.",
        )

        pdf.savefig(fig)
        plt.close(fig)

        # =========================================================================
        # PAGE 7: 2D BARCHAN DUNE MORPHOGENESIS SYNTHESIS
        # =========================================================================
        fig = plt.figure(figsize=(8.5, 11.0), dpi=300)
        _add_page_header(
            fig, 7, total_pages, "7. Synthesis: Emergent 2D Crescentic Barchan Dune Morphodynamics"
        )

        _add_text_box(
            fig,
            0.05,
            0.73,
            0.90,
            0.175,
            "How the Five Physical Equations Jointly Sculpt a Crescentic Barchan Dune from a Circular Mound",
            [
                "Starting from a radially symmetric parabolic sand dome on flat, non-erodible bedrock (t = 0 d):",
                "  - Stage 1 (0 - 15 d, Stoss Flattening & Slipface Birth): Upwind shear shift (+i B kx) erodes the windward",
                "    toe (modulated by gravity threshold tau_th,eff) and deposits near the crest until the lee slope exceeds 14 deg,",
                "    triggering the separation wake (tau = 0) and isotropic 32-deg avalanche slipface.",
                "  - Stage 2 (15 - 35 d, Differential Migration & Horn Emergence): Lateral wind deflection (tau_y) and splash",
                "    diffusion (-D_lat dh/dy) shed sand toward the lower flanks. Because migration speed scales as c(y) ~ 1/H(y),",
                "    the lower flanks outrun the tall central crest, curving the brink into a concave crescent.",
                "  - Stage 3 (35 - 55 d, Equilibrium Barchan Attractor): The dune reaches a steady shape-invariant crescent",
                "    (H = 3.68 m) with a smooth ~10-deg convex stoss ramp, a 32-deg concave slipface, and symmetric forward horns.",
            ],
            bg_color="#fffaf0",
            border_color="#d69e2e",
            title_color="#975a16",
            fontsize=8.4,
            line_spacing=0.0175,
        )

        _embed_image(
            fig,
            [0.04, 0.33, 0.92, 0.385],
            os.path.join(img_dir, "04_barchan_2d_simulation.png"),
            "Figure 6 (from 04_barchan_2d_simulation.py): 4-stage planform evolution, centerline profile, horn cross-sections, and sand flux field.",
        )

        _embed_image(
            fig,
            [0.04, 0.03, 0.92, 0.275],
            os.path.join(img_dir, "04_barchan_3d_surface.png"),
            "Figure 7 (from 04_barchan_2d_simulation.py): Dual-perspective 3D surface of the emergent equilibrium Barchan dune at t = 55 days.",
        )

        pdf.savefig(fig)
        plt.close(fig)

    print(f"Successfully generated 7-page physics guide PDF: {output_pdf_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(base_dir, "examples", "output")
    pdf_out_local = os.path.join(base_dir, "dune_physics_models_guide.pdf")
    pdf_out_root = os.path.abspath(os.path.join(base_dir, "..", "dune_physics_models_guide.pdf"))

    generate_pdf(pdf_out_local, img_dir)
    generate_pdf(pdf_out_root, img_dir)
