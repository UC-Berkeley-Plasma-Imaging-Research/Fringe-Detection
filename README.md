# Fringe Detection

This repository contains resources and notebooks for fringe detection in imaging data, likely related to plasma imaging research at UC Berkeley.

## Contents
- `Reference.png`, `ReferenceCropped.png`: Reference images used for calibration or comparison.
- `Shot.png`, `ShotCropped.png`: Sample images for analysis, possibly raw and cropped versions.

## Getting Started
1. **Requirements**: Install Python 3.x and Jupyter Notebook.
2. **Dependencies**: Common packages include `numpy`, `matplotlib`, `opencv-python`, and `scipy`. Install them with:
   ```powershell
   pip install numpy matplotlib opencv-python scipy
   ```
3. **Usage**:
   - Open the notebooks (`.ipynb` files) in Jupyter.
   - Follow the instructions and code cells to process the provided images.
Even Illumination — Standalone App

This repository contains a small cross-platform GUI app that performs shading correction + CLAHE + Sauvola thresholding (useful for fringe extraction).

Files added:
- even_illumination_app.py  : PySimpleGUI-based standalone app (single-file).
- even_illumination_app.py  : tkinter-based standalone app (single-file).
- requirements.txt          : Python dependencies.
- .github/workflows/build.yml: GitHub Actions workflow to build Windows/macOS artifacts on releases.

Run from source (Windows/macOS/Linux):

1. Create a Python 3.9+ virtualenv (recommended):

    python -m venv venv
    # Windows (PowerShell):
    .\\venv\\Scripts\\Activate.ps1
    # macOS / Linux:
    source venv/bin/activate

2. Install dependencies:

    python -m pip install -r requirements.txt

3. Run the app:

    python even_illumination_app.py

Build platform-native executable with PyInstaller

- Windows (build on Windows):
   python -m pip install pyinstaller
   py -3 -m PyInstaller --noconfirm --onefile --windowed even_illumination_app.py
   # The single exe will be in the `dist` directory.

- macOS (build on macOS):
   python3 -m pip install pyinstaller
   python3 -m PyInstaller --noconfirm --onefile --windowed even_illumination_app.py
   # The .app (singlefile binary) will be in `dist`. Note: To produce a macOS app you must build on macOS.

CI: GitHub Actions builds

When you create a GitHub Release, the workflow at `.github/workflows/build.yml` will run and produce platform artifacts for Windows and macOS and attach them to the release as downloadable assets.

Notes and limitations

- Building a macOS .app requires macOS; cross-compilation from Windows is non-trivial. Use GitHub Actions macOS runners for CI builds.
- The GUI uses PySimpleGUI (Tkinter backend by default). This keeps dependencies small and works cross-platform.
- The GUI uses the built-in `tkinter` + `Pillow` for image previews. No paid/private packages are required.
- If you prefer a web-based UI (so users don't have to install anything), I can add a small Flask app and provide instructions for packaging with a lightweight server.

## Project Purpose
This project aims to detect and analyze fringe patterns in images, which can be useful for optical diagnostics, interferometry, or plasma imaging.

## License
Specify your license here (e.g., MIT, GPL, etc.).

## Contact
For questions or collaboration, contact the UC Berkeley Plasma Imaging Research group.
