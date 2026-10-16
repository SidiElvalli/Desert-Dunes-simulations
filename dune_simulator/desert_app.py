"""Main entry point for the standalone 'Desert' 3D morphodynamics studio executable (Desert.exe)."""

import sys
import os
import argparse

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dune_simulator.gui.main_window import launch_desert_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Desert — Physical Aeolian & Subaqueous Dune Studio")
    parser.add_argument("--no-splash", action="store_true", help="Skip the startup splash screen")
    parser.add_argument("--auto-close-ms", type=int, default=None, help="Auto-close after N ms (for automated testing)")
    args = parser.parse_args()

    sys.exit(
        launch_desert_app(
            show_splash=not args.no_splash,
            auto_close_ms=args.auto_close_ms,
        )
    )


if __name__ == "__main__":
    main()
