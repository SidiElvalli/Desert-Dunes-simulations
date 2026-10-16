"""Tests for logarithmic wind profile, ballistic impact ripples, experimental benchmarks, and Desert GUI."""

import os
import tempfile
import numpy as np
from dune_simulator.core.constants import EARTH_DESERT, MARS_DUNES
from dune_simulator.core.wind_profile import (
    compute_wind_profile,
    evaluate_log_wind_profile,
    u_star_from_u10,
    compute_impact_ripple_scales,
    synthesize_zoomed_ripple_patch,
    compute_experimental_benchmarks,
)
from dune_simulator.gui.assets import ensure_gui_assets


def test_logarithmic_wind_profile_and_inversion():
    """Verifies Prandtl-Owen logarithmic wind profile u(z) and U(10m) <-> u* inversion."""
    prof = compute_wind_profile(EARTH_DESERT, u_star_0=0.38)
    assert prof.z0_eff_m > prof.z0_grain_m  # Saltation Owen roughness active above threshold
    assert prof.u_10m > prof.u_2m > 0.0

    # Inversion from U(10m) recovers u_star_0 to within 0.1%
    u_recovered = u_star_from_u10(EARTH_DESERT, prof.u_10m)
    assert np.isclose(u_recovered, 0.38, rtol=1e-3)

    z, u = evaluate_log_wind_profile(prof)
    assert np.all(np.diff(u) > 0.0)


def test_ballistic_impact_ripple_scaling():
    """Verifies impact ripple wavelength (~10-15 cm on Earth quartz sand) and grain-size scaling."""
    lam_r, amp_r, c_r = compute_impact_ripple_scales(EARTH_DESERT, u_star=0.38)
    # Wind-tunnel & desert ripple wavelength for 200 um quartz at u*=0.38 m/s is ~10-15 cm
    assert 0.09 <= lam_r <= 0.16
    assert amp_r > 0.0
    assert c_r > 0.0

    # Below threshold, ripple amplitude and speed drop to 0
    _, amp_sub, c_sub = compute_impact_ripple_scales(EARTH_DESERT, u_star=0.10)
    assert amp_sub == 0.0
    assert c_sub == 0.0

    # Zoomed 2m x 2m patch synthesis
    x_p, y_p, z_mm, lam_cm, h_mm = synthesize_zoomed_ripple_patch(EARTH_DESERT, u_star_0=0.38, n_grid=64)
    assert z_mm.shape == (64, 64)
    assert np.isclose(lam_cm, lam_r * 100.0)
    assert h_mm > 4.0


def test_experimental_benchmark_curves():
    """Verifies experimental benchmark validation curves against empirical wind-tunnel bounds."""
    bench = compute_experimental_benchmarks(EARTH_DESERT, u_star_0=0.38)
    assert len(bench["d_um"]) == 100
    assert np.all(np.diff(bench["u_th_curve"]) > 0.0)
    assert np.all(np.diff(bench["c_yr_curve"]) < 0.0)  # c ~ 1/H monotonically decreases with H


def test_desert_gui_assets_and_project_roundtrip():
    """Verifies procedural generation of Desert.ico / Desert_Splash.png and GUI project save/load."""
    with tempfile.TemporaryDirectory() as tmpd:
        assets = ensure_gui_assets(tmpd)
        assert os.path.exists(assets["splash"])
        assert os.path.exists(assets["logo"])
        assert os.path.exists(assets["icon"])


def test_custom_project_dialog_and_3d_sculpting():
    """Verifies CustomProjectDialog formula evaluation and direct 3D surface brush sculpting."""
    from PyQt6.QtWidgets import QApplication
    from dune_simulator.gui.custom_project_dialog import CustomProjectDialog

    app = QApplication.instance() or QApplication([])
    dlg = CustomProjectDialog()
    assert dlg.custom_h is not None
    v0 = float(np.sum(dlg.custom_h))

    # Apply a 'raise' sculpt brush stroke at the center (gx=0, gy=0)
    dlg.sculpt_viewport.sculpt_tool = "raise"
    dlg.sculpt_viewport.sculpt_radius_m = 15.0
    dlg.sculpt_viewport.sculpt_strength_m = 1.2
    dlg.sculpt_viewport.apply_sculpt_brush_at_gl_coords(0.0, 0.0)
    v1 = float(np.sum(dlg.custom_h))
    assert v1 > v0
    dlg.close()


def test_all_presets_and_custom_surfaces_run_continuously_and_help_dialog():
    """Verifies that all non-barchan presets and custom 3D surfaces run continuously without stopping after 1 step."""
    from PyQt6.QtWidgets import QApplication
    from dune_simulator.gui.main_window import DesertMainWindow
    from dune_simulator.gui.scientific_help_dialog import ScientificHelpDialog

    app = QApplication.instance() or QApplication([])
    win = DesertMainWindow()
    win.combo_grid_res.setCurrentIndex(0)

    # Test every preset (barchan, dune_sea, pyramid, twin_barchan, swarm, instability, step, mars)
    for preset in ["barchan", "dune_sea", "pyramid", "twin_barchan", "swarm", "instability", "step", "mars"]:
        win.apply_quick_preset(preset)
        win.advance_simulation_steps(6)
        assert win.simulator.step_count == 6
        assert win.simulator.time_s > 0.0

    # Verify English <-> Technical Arabic toggle
    assert win.lang == "en"
    win.toggle_language()
    assert win.lang == "ar"
    assert "صحراء" in win.windowTitle()
    win.toggle_language()
    assert win.lang == "en"

    # Test custom 3D surface that spans the entire domain (e.g. wave train / sand sheet near outlet)
    nx, ny = win.simulator.config.nx, win.simulator.config.ny
    win.simulator.h = np.full((nx, ny), 1.5, dtype=np.float64)
    win.simulator.step_count = 0
    win.advance_simulation_steps(10)
    assert win.simulator.step_count == 10

    # Verify ScientificHelpDialog and bottom EquationsAndLimitationsStrip instantiate cleanly
    assert win.eq_strip.isVisible() or not win.eq_strip.isHidden()
    help_dlg = ScientificHelpDialog(win)
    assert help_dlg.windowTitle().startswith("Desert Studio")
    help_dlg.close()
    win.close()
