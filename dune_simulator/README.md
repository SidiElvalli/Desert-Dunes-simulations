# Desert (`dune_simulator`) — Python Package & 3D Morphodynamics Studio

This directory contains the Python source package, test suite, educational examples, and PyInstaller configuration for **Desert** (`dune_simulator`).

## Quick Commands

### 1. Launch the Interactive 3D Studio GUI
```powershell
uv run python desert_app.py
```

### 2. Run the 35 Physical & Numerical Unit Tests
```powershell
uv run pytest tests -v
```

### 3. Rebuild the Standalone `Desert.exe` Executable
```powershell
uv run pyinstaller --noconfirm --clean --windowed --onefile --name Desert --icon="dune_simulator/gui/resources/Desert.ico" --add-data "dune_simulator/gui/resources;dune_simulator/gui/resources" --collect-submodules pyqtgraph.opengl desert_app.py
Copy-Item -Path "dist\Desert.exe" -Destination "..\Desert.exe" -Force
```

### 4. Regenerate the 7-Page Physics Guide PDF
```powershell
uv run python generate_physics_pdf.py
```

See the root [`../README.md`](../README.md) and [`../dune_physics_models_guide.pdf`](../dune_physics_models_guide.pdf) for full documentation of the physical models, equations, and visual tour.
