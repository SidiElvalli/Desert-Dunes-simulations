"""Procedural generation of the 'Desert' application logo, splash banner, and Windows .ico icon."""

import os
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate_desert_logo_image(width: int = 760, height: int = 400) -> Image.Image:
    """Renders a high-resolution artistic Desert logo & splash graphic with a golden crescent
    Barchan dune, wind-sculpted ripples, glowing desert sun, and twilight sky gradient.
    """
    y_norm = np.linspace(0.0, 1.0, height)[:, None]
    x_norm = np.linspace(0.0, 1.0, width)[None, :]

    # 1. Twilight desert sky gradient (deep obsidian-indigo top -> warm amber horizon)
    r = np.clip(18 + 195 * (y_norm ** 1.7), 0, 255)
    g = np.clip(20 + 105 * (y_norm ** 2.0), 0, 255)
    b = np.clip(34 + 55 * (1.0 - (y_norm - 0.45) ** 2), 18, 95)

    # 2. Glowing Desert Sun (centered slightly left of crest)
    sun_x, sun_y = 0.34, 0.36
    aspect = width / height
    dist_sun = np.sqrt(((x_norm - sun_x) * aspect) ** 2 + (y_norm - sun_y) ** 2)
    sun_core = np.exp(-((dist_sun / 0.085) ** 4))
    sun_halo = np.exp(-((dist_sun / 0.28) ** 1.6))

    r = np.clip(r + 240 * sun_core + 115 * sun_halo, 0, 255)
    g = np.clip(g + 205 * sun_core + 70 * sun_halo, 0, 255)
    b = np.clip(b + 135 * sun_core + 20 * sun_halo, 0, 255)

    # 3. Background distant dune silhouette
    bg_crest = 0.62 - 0.10 * np.exp(-(((x_norm - 0.72) / 0.28) ** 2)) - 0.05 * np.sin(5.0 * x_norm)
    bg_mask = y_norm > bg_crest
    r = np.where(bg_mask, 78 + 30 * (y_norm - bg_crest), r)
    g = np.where(bg_mask, 48 + 18 * (y_norm - bg_crest), g)
    b = np.where(bg_mask, 32 + 10 * (y_norm - bg_crest), b)

    # 4. Foreground Sweeping Barchan Crescent Dune with Illuminated Stoss Ripples & Shadowed Slipface
    brink_x = 0.52
    crest_y = 0.46 + 1.15 * ((x_norm - brink_x) ** 2) - 0.04 * np.sin(3.5 * x_norm)
    dune_mask = y_norm > crest_y

    # Divide foreground dune into sunlit windward stoss (left of ridge spine) and shadowed lee slipface (right)
    spine_x = brink_x + 0.24 * (y_norm - 0.46)
    is_stoss = x_norm <= spine_x

    # Physical wind ripples on the sunlit stoss slope
    ripple_phase = 140.0 * (x_norm - 0.35 * y_norm) + 8.0 * np.sin(18.0 * y_norm)
    ripple_shade = 0.88 + 0.12 * (np.sin(ripple_phase) - 0.3 * np.sin(2.0 * ripple_phase))

    stoss_r = (232 - 55 * (y_norm - crest_y)) * ripple_shade
    stoss_g = (174 - 50 * (y_norm - crest_y)) * ripple_shade
    stoss_b = (96 - 35 * (y_norm - crest_y)) * ripple_shade

    # Deep copper-umber lee slipface shadow
    lee_r = 92 - 35 * (x_norm - spine_x)
    lee_g = 56 - 24 * (x_norm - spine_x)
    lee_b = 34 - 14 * (x_norm - spine_x)

    r = np.where(dune_mask, np.where(is_stoss, stoss_r, lee_r), r)
    g = np.where(dune_mask, np.where(is_stoss, stoss_g, lee_g), g)
    b = np.where(dune_mask, np.where(is_stoss, stoss_b, lee_b), b)

    # Highlight the razor-sharp Barchan brink and spine ridge
    brink_glow = np.exp(-(((y_norm - crest_y) / 0.006) ** 2))
    spine_glow = np.where(dune_mask, np.exp(-(((x_norm - spine_x) / 0.005) ** 2)), 0.0)
    glow = np.maximum(brink_glow, 0.75 * spine_glow)
    r = np.clip(r + 80 * glow, 0, 255)
    g = np.clip(g + 65 * glow, 0, 255)
    b = np.clip(b + 35 * glow, 0, 255)

    # Bottom dark vignette bar for crisp title typography
    vignette = np.clip((y_norm - 0.72) / 0.28, 0.0, 1.0) ** 1.4
    r = r * (1.0 - 0.84 * vignette) + 18 * (0.84 * vignette)
    g = g * (1.0 - 0.84 * vignette) + 18 * (0.84 * vignette)
    b = b * (1.0 - 0.84 * vignette) + 22 * (0.84 * vignette)

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")

    # Add wind streamline arcs across the sky
    draw = ImageDraw.Draw(img, "RGBA")
    for k, y_arc in enumerate([0.18, 0.24, 0.30]):
        pts = []
        for px in np.linspace(0.06, 0.94, 80):
            py = y_arc - 0.035 * math.exp(-(((px - 0.52) / 0.22) ** 2)) + 0.008 * math.sin(10 * px + k)
            pts.append((px * width, py * height))
        draw.line(pts, fill=(255, 225, 165, 52 - k * 12), width=2)

    return img


def generate_desert_icon(ico_path: str, png_path: str | None = None) -> str:
    """Generates a square multi-resolution Windows .ico icon and square logo PNG for Desert."""
    size = 256
    base = generate_desert_logo_image(width=size, height=size)

    # Add rounded golden-amber border frame for the desktop icon
    draw = ImageDraw.Draw(base, "RGBA")
    draw.rectangle([2, 2, size - 3, size - 3], outline=(230, 155, 45, 235), width=5)

    os.makedirs(os.path.dirname(os.path.abspath(ico_path)), exist_ok=True)
    base.save(ico_path, format="ICO", sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)])
    if png_path:
        base.save(png_path, format="PNG")
    return ico_path


def ensure_gui_assets(asset_dir: str | None = None) -> dict:
    """Ensures that Desert_Splash.png, Desert_Logo.png, and Desert.ico exist on disk."""
    if asset_dir is None:
        asset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resources")
    os.makedirs(asset_dir, exist_ok=True)

    splash_path = os.path.join(asset_dir, "Desert_Splash.png")
    logo_path = os.path.join(asset_dir, "Desert_Logo.png")
    ico_path = os.path.join(asset_dir, "Desert.ico")

    if not os.path.exists(splash_path):
        splash_img = generate_desert_logo_image(width=760, height=400)
        splash_img.save(splash_path, format="PNG")

    if not os.path.exists(ico_path) or not os.path.exists(logo_path):
        generate_desert_icon(ico_path=ico_path, png_path=logo_path)

    return {
        "splash": splash_path,
        "logo": logo_path,
        "icon": ico_path,
    }
