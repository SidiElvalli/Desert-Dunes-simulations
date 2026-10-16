"""2D contour, hillshade, diagnostic cross-sections, and 3D surface visualization for topographic dune modeling."""

from typing import Optional, List
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource, LinearSegmentedColormap
import numpy as np
from ..simulation.config import SimulationState2D


def _desert_sand_cmap() -> LinearSegmentedColormap:
    """Returns a custom desert colormap with dark slate bedrock at 0 and warm golden sand above."""
    colors = [
        (0.00, "#141418"),   # Bare non-erodible bedrock floor (dark basalt/slate)
        (0.03, "#3b2d22"),   # Thin sand margin
        (0.20, "#7c5836"),   # Lower stoss / horn apron
        (0.50, "#b8864b"),   # Mid dune body
        (0.80, "#e0af68"),   # Upper stoss / crest shoulder
        (1.00, "#fce2a6"),   # High crest / brink highlight
    ]
    return LinearSegmentedColormap.from_list("desert_barchan", colors, N=256)


def plot_topography_2d(
    h: np.ndarray,
    dx: float,
    dy: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    ax: Optional[plt.Axes] = None,
    show_wind_arrow: bool = True,
    vmax: Optional[float] = None,
    in_wake: Optional[np.ndarray] = None,
    qx: Optional[np.ndarray] = None,
    qy: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Renders a 2D hillshaded topographic surface map showing dune morphology (horns, slipface, brink).

    Parameters:
        h: 2D elevation array of shape (nx, ny).
        dx, dy: Grid spacings [m].
        title: Plot title.
        save_path: Optional file path to save PNG.
        show: Whether to call plt.show().
        ax: Optional existing axes to draw on (used for multi-panel time sequences).
        show_wind_arrow: Draw the wind direction annotation.
        vmax: Optional fixed upper elevation for color/contour scaling across time series.
        in_wake: Optional boolean wake mask (nx, ny) to outline the separation bubble.
        qx, qy: Optional sand flux arrays (nx, ny) to overlay transport vectors.
    """
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8, 6.5))
    else:
        fig = ax.figure

    nx, ny = h.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy
    extent = [y[0], y[-1], x[0], x[-1]]

    h_top = float(vmax) if vmax is not None else max(float(np.max(h)), 1.0)
    cmap = _desert_sand_cmap()

    # Shaded relief with illumination from NW (azimuth 315 deg)
    ls = LightSource(azdeg=315, altdeg=42)
    rgb = ls.shade(
        h,
        cmap=cmap,
        vmin=0.0,
        vmax=h_top,
        vert_exag=1.8,
        blend_mode="overlay",
        dx=dy,
        dy=dx,
    )

    ax.imshow(rgb, extent=extent, origin="lower", aspect="equal")

    # Contour only real sand above bedrock (> 0.15 m)
    if float(np.max(h)) > 0.15 and h_top > 0.15:
        levels = np.linspace(0.4, h_top * 0.92, 6)
        contours = ax.contour(
            y, x, h, levels=levels, colors="white", alpha=0.55, linewidths=0.75
        )
        if contours.allsegs and any(len(s) for s in contours.allsegs):
            ax.clabel(contours, inline=True, fontsize=7.5, fmt="%.1f m")

        # Sand footprint outline on bedrock (h = 0.10 m)
        ax.contour(y, x, h, levels=[0.10], colors=["#d4a359"], alpha=0.8, linewidths=1.0)

    # Highlight separation wake boundary if provided
    if in_wake is not None and np.any(in_wake):
        ax.contour(
            y,
            x,
            in_wake.astype(float),
            levels=[0.5],
            colors=["#00e5ff"],
            linestyles="--",
            linewidths=1.2,
            alpha=0.85,
        )

    # Highlight brink line (upwind edge of separation wake / steep lee face)
    if in_wake is not None and np.any(in_wake):
        brink_x, brink_y = [], []
        for j in range(ny):
            w_idx = np.where(in_wake[:, j])[0]
            if len(w_idx) > 0 and w_idx[0] > 0:
                b_i = w_idx[0] - 1
                if h[b_i, j] > 0.4:
                    brink_x.append(x[b_i])
                    brink_y.append(y[j])
        if len(brink_x) > 2:
            ax.plot(brink_y, brink_x, color="#ff3b30", lw=1.8, label="Brink Line")

    # Optional sand flux quiver vectors
    if qx is not None and qy is not None:
        stride_x = max(1, nx // 16)
        stride_y = max(1, ny // 14)
        Y_g, X_g = np.meshgrid(y[::stride_y], x[::stride_x])
        qx_sub = qx[::stride_x, ::stride_y]
        qy_sub = qy[::stride_x, ::stride_y]
        q_mag = np.hypot(qx_sub, qy_sub)
        mask = q_mag > 0.05 * max(float(np.max(q_mag)), 1e-9)
        if np.any(mask):
            ax.quiver(
                Y_g[mask],
                X_g[mask],
                qy_sub[mask],
                qx_sub[mask],
                color="white",
                alpha=0.55,
                scale=float(np.max(q_mag)) * 14.0,
                width=0.0035,
                headwidth=3.5,
            )

    ax.set_xlabel("Transverse Coordinate $y$ [m]", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Downwind Coordinate $x$ [m]", fontsize=10.5, fontweight="bold")
    ax.set_title(
        title or "2D Dune Topography (Hillshaded)",
        fontsize=12.5 if own_figure else 11,
        fontweight="bold",
        pad=8,
    )

    if show_wind_arrow:
        ax.annotate(
            "Wind (+x)",
            xy=(0.12, 0.92),
            xycoords="axes fraction",
            xytext=(0.12, 0.77),
            ha="center",
            arrowprops=dict(facecolor="#00e5ff", edgecolor="black", width=2.5, headwidth=8),
            fontsize=9.5,
            fontweight="bold",
            color="#00e5ff",
        )

    if own_figure:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

    return fig


def plot_barchan_evolution_diagnostics(
    states: List[SimulationState2D],
    dx: float,
    dy: float,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Renders a comprehensive 2-row publication figure of Barchan dune formation and physics.

    Top Row: 4 time-lapse planform hillshaded snapshots showing evolution from symmetric
             dome to crescentic barchan with brink line, wake bubble, and flux vectors.
    Bottom Row:
        - Left: Centerline longitudinal profile h(x, y_c), separation envelope h_tilde,
                recirculation wake, and basal shear stress tau_x(x, y_c).
        - Center: Transverse cross-sections through the central crest vs the leeward
                  crescent bay (highlighting the two forward-pointing horns).
        - Right: 2D sand flux magnitude |q(x, y)| and transport streamlines around the barchan.
    """
    pick = sorted(set(np.linspace(0, len(states) - 1, 4).round().astype(int)))
    h_ref = max(float(s.h.max()) for s in states)

    fig = plt.figure(figsize=(18.5, 10.8))
    outer_gs = fig.add_gridspec(2, 1, height_ratios=[1.22, 1.0], hspace=0.34)
    gs_top = outer_gs[0].subgridspec(1, 4, wspace=0.24)
    gs_bot = outer_gs[1].subgridspec(1, 3, width_ratios=[1.85, 1.0, 1.0], wspace=0.42)

    # --- Top Row: 4 Time-Lapse Planform Snapshots ---
    stage_labels = [
        "1. Initial Parabolic Seed",
        "2. Lee Steepening & Brink Birth",
        "3. Horn Emergence (c ~ 1/H)",
        "4. Mature Equilibrium Barchan",
    ]
    for k, idx in enumerate(pick):
        st = states[idx]
        ax = fig.add_subplot(gs_top[0, k])
        stage_lbl = stage_labels[k] if k < len(stage_labels) else f"Stage {k+1}"
        plot_topography_2d(
            st.h,
            dx=dx,
            dy=dy,
            title=f"{stage_lbl}\nt = {st.time_days:.0f} d (H = {st.max_height:.2f} m)",
            ax=ax,
            show_wind_arrow=(k == 0),
            vmax=h_ref,
            in_wake=st.in_wake,
            qx=st.qx if k == len(pick) - 1 else None,
            qy=st.qy if k == len(pick) - 1 else None,
        )
        if k > 0:
            ax.set_ylabel("")

    final_st = states[-1]
    init_st = states[0]
    x = final_st.x
    y = final_st.y
    nx, ny = final_st.h.shape
    yc = ny // 2

    # --- Bottom Left: Longitudinal Centerline Morphodynamics ---
    ax_long = fig.add_subplot(gs_bot[0, 0])
    ax_long.plot(
        x, init_st.h[:, yc], color="gray", linestyle=":", lw=1.6, label="Initial Mound (t = 0 d)"
    )
    mid_st = states[pick[1]]
    ax_long.plot(
        x,
        mid_st.h[:, yc],
        color="#c9975b",
        linestyle="-.",
        lw=1.5,
        label=f"Intermediate (t = {mid_st.time_days:.0f} d)",
    )

    h_c = final_st.h[:, yc]
    ht_c = final_st.h_tilde[:, yc]
    wake_c = final_st.in_wake[:, yc]

    ax_long.plot(
        x, h_c, color="#7a3e10", lw=2.5, label=f"Mature Centerline $h(x, y_c)$ (t = {final_st.time_days:.0f} d)"
    )
    ax_long.fill_between(x, 0, h_c, color="#e0af68", alpha=0.65)

    if np.any(wake_c):
        ax_long.plot(x, ht_c, "g--", lw=1.8, label=r"Separation Streamline $\tilde{h}(x, y_c)$")
        ax_long.fill_between(
            x, h_c, ht_c, where=wake_c, color="slategray", alpha=0.35, label=r"Recirculating Wake ($\tau=0$)"
        )

    # Twin axis for streamwise shear stress tau_x
    ax_tau = ax_long.twinx()
    ax_tau.plot(
        x, final_st.tau_x[:, yc], color="tab:blue", lw=1.8, alpha=0.85, label=r"Shear Stress $\tau_x(x, y_c)$ [Pa]"
    )
    ax_tau.set_ylabel(r"Shear Stress $\tau_x$ [Pa]", color="tab:blue", fontsize=9.5, fontweight="bold")
    ax_tau.tick_params(axis="y", labelcolor="tab:blue")
    ax_tau.set_ylim(0.0, max(float(np.max(final_st.tau_x[:, yc])) * 1.35, 0.4))

    ax_long.set_xlabel("Downwind Position $x$ [m]", fontsize=10.5, fontweight="bold")
    ax_long.set_ylabel("Centerline Elevation $h$ [m]", fontsize=10.5, fontweight="bold")
    ax_long.set_ylim(0.0, max(h_ref * 1.28, 4.5))
    ax_long.set_xlim(x[0], x[-1])
    ax_long.set_title(
        "Longitudinal Symmetry Slice ($y = L_y/2$): Stoss Ramp, $32^\\circ$ Slipface & Wake Bubble",
        fontsize=11,
        fontweight="bold",
    )
    ax_long.grid(True, linestyle=":", alpha=0.6)

    lines1, labels1 = ax_long.get_legend_handles_labels()
    lines2, labels2 = ax_tau.get_legend_handles_labels()
    ax_long.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8.5, framealpha=0.92)

    # --- Bottom Center: Transverse Cross-Sections (Crest vs Crescent Bay Horns) ---
    ax_trans = fig.add_subplot(gs_bot[0, 1])
    ix_crest = int(np.argmax(h_c))
    # Find slipface toe at centerline and pick a bay slice 7 m downwind of the toe
    sand_c = np.where(h_c > 0.12)[0]
    ix_toe = int(sand_c[-1]) if len(sand_c) > 0 else ix_crest + 6
    ix_bay = min(nx - 1, ix_toe + 7)

    ax_trans.plot(
        y,
        final_st.h[ix_crest, :],
        color="#8b4513",
        lw=2.3,
        label=f"At Central Brink ($x = {x[ix_crest]:.0f}$ m)",
    )
    ax_trans.fill_between(y, 0, final_st.h[ix_crest, :], color="#e0af68", alpha=0.45)

    ax_trans.plot(
        y,
        final_st.h[ix_bay, :],
        color="#d95f02",
        lw=2.2,
        linestyle="--",
        label=f"Across Lee Horn Bay ($x = {x[ix_bay]:.0f}$ m)",
    )
    ax_trans.fill_between(y, 0, final_st.h[ix_bay, :], color="#d95f02", alpha=0.30)

    ax_trans.set_xlabel("Transverse Coordinate $y$ [m]", fontsize=10.5, fontweight="bold")
    ax_trans.set_ylabel("Elevation $h$ [m]", fontsize=10.5, fontweight="bold")
    ax_trans.set_ylim(0.0, max(h_ref * 1.2, 4.2))
    ax_trans.set_xlim(y[0], y[-1])
    ax_trans.set_title(
        "Transverse Cross-Sections:\nCentral Dome vs Forward Horns",
        fontsize=11,
        fontweight="bold",
    )
    ax_trans.grid(True, linestyle=":", alpha=0.6)
    ax_trans.legend(loc="upper right", fontsize=8.5, framealpha=0.92)

    # --- Bottom Right: 2D Sediment Flux Magnitude & Wake Trapping ---
    ax_flux = fig.add_subplot(gs_bot[0, 2])
    q_mag = np.hypot(final_st.qx, final_st.qy) * 1e3  # g / (m s)
    im = ax_flux.imshow(
        q_mag,
        extent=[y[0], y[-1], x[0], x[-1]],
        origin="lower",
        cmap="inferno",
        aspect="equal",
    )
    if np.any(final_st.in_wake):
        ax_flux.contour(
            y,
            x,
            final_st.in_wake.astype(float),
            levels=[0.5],
            colors=["#00e5ff"],
            linestyles="--",
            linewidths=1.3,
        )
    ax_flux.contour(y, x, final_st.h, levels=[0.15, 1.5, 3.0], colors="white", alpha=0.45, linewidths=0.7)
    cbar = fig.colorbar(im, ax=ax_flux, fraction=0.046, pad=0.04)
    cbar.set_label(r"Sand Flux $|\mathbf{q}|$ [g/(m$\cdot$s)]", fontsize=9)
    ax_flux.set_xlabel("Transverse Coordinate $y$ [m]", fontsize=10.5, fontweight="bold")
    ax_flux.set_ylabel("Downwind Coordinate $x$ [m]", fontsize=10.5, fontweight="bold")
    ax_flux.set_title(
        "Sand Transport Field $|\mathbf{q}(x,y)|$\n& Zero-Flux Lee Wake (Cyan)",
        fontsize=11,
        fontweight="bold",
    )

    fig.suptitle(
        "Physics-Based 2D Barchan Dune Morphodynamics: Crescent Formation, Flow Separation & Horn Elongation",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_surface_3d(
    h: np.ndarray,
    dx: float,
    dy: float,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
) -> plt.Figure:
    """Renders a dual-perspective 3D surface figure highlighting both the concave slipface bay and windward stoss ramp."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    nx, ny = h.shape
    x = np.arange(nx) * dx
    y = np.arange(ny) * dy

    # Crop window around active sand dune (with generous margin) for high-detail 3D rendering
    sand_x, sand_y = np.where(h > 0.02)
    if len(sand_x) > 10:
        i_min = max(0, int(sand_x.min()) - 14)
        i_max = min(nx, int(sand_x.max()) + 16)
        j_min = max(0, int(sand_y.min()) - 14)
        j_max = min(ny, int(sand_y.max()) + 14)
    else:
        i_min, i_max, j_min, j_max = 0, nx, 0, ny

    h_sub = h[i_min:i_max, j_min:j_max]
    x_sub = x[i_min:i_max]
    y_sub = y[j_min:j_max]
    X, Y = np.meshgrid(x_sub, y_sub, indexing="ij")

    cmap = _desert_sand_cmap()
    h_max = max(float(np.max(h_sub)), 1.0)
    ls = LightSource(azdeg=315, altdeg=40)
    rgb = ls.shade(
        h_sub, cmap=cmap, vmin=0.0, vmax=h_max, vert_exag=2.0, blend_mode="soft", dx=dy, dy=dx
    )

    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(1, 2, wspace=0.18)

    lx_sub = max(1.0, float(x_sub[-1] - x_sub[0]))
    ly_sub = max(1.0, float(y_sub[-1] - y_sub[0]))

    # --- Panel 1: Downwind Lee-Bay Oblique Perspective (viewing concave slipface & horns) ---
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax1.plot_surface(
        X,
        Y,
        h_sub,
        facecolors=rgb,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax1.set_xlabel("Downwind $x$ [m]", fontsize=10, fontweight="bold", labelpad=8)
    ax1.set_ylabel("Transverse $y$ [m]", fontsize=10, fontweight="bold", labelpad=8)
    ax1.set_zlabel("Elevation $h$ [m]", fontsize=10, fontweight="bold", labelpad=6)
    ax1.set_zlim(0.0, h_max * 1.22)
    ax1.set_box_aspect((lx_sub, ly_sub, h_max * 6.5))
    ax1.view_init(elev=28, azim=35)
    ax1.set_title(
        "Leeward Perspective:\nConcave $32^\\circ$ Slipface & Forward Horns",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )

    # --- Panel 2: Windward Stoss & Side Oblique Perspective ---
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    surf = ax2.plot_surface(
        X,
        Y,
        h_sub,
        cmap=cmap,
        vmin=0.0,
        vmax=h_max,
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        alpha=0.96,
    )
    ax2.set_xlabel("Downwind $x$ [m]", fontsize=10, fontweight="bold", labelpad=8)
    ax2.set_ylabel("Transverse $y$ [m]", fontsize=10, fontweight="bold", labelpad=8)
    ax2.set_zlabel("Elevation $h$ [m]", fontsize=10, fontweight="bold", labelpad=6)
    ax2.set_zlim(0.0, h_max * 1.22)
    ax2.set_box_aspect((lx_sub, ly_sub, h_max * 6.5))
    ax2.view_init(elev=30, azim=215)
    ax2.set_title(
        "Windward Perspective:\nSmooth Stoss Ramp & Crescent Brink",
        fontsize=11.5,
        fontweight="bold",
        pad=8,
    )

    fig.colorbar(surf, ax=[ax1, ax2], shrink=0.62, aspect=16, pad=0.05, label="Sand Elevation $h$ [m]")
    fig.suptitle(title or "3D Barchan Dune Surface Morphology", fontsize=14, fontweight="bold", y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig

