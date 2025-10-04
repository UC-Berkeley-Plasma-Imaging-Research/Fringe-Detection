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
   # Fringe Detection — Even Illumination App

   This repository provides a small GUI app to perform even-illumination correction and fringe extraction on images. It includes a tkinter-based GUI (`even_illumination_app.py`) and small, focused processing modules under the project root.

   What’s in this repo
   - `even_illumination_app.py` — tkinter GUI that runs the shading/CLAHE/Sauvola pipeline and fringe detection.
   - `fringe_utils.py` — fringe processing helpers (binarize, oriented opening, overlay).
   - `shading_pipeline.py` — illumination/shading helpers (Sauvola, CLAHE pipeline).
   - `ui_helpers.py` — small UI helpers (slider rows, image conversion).
   - `requirements.txt` — runtime dependencies.

   Quick start (Windows/macOS/Linux)
   1. Create and activate a virtual environment:

      python -m venv .venv
      # Windows (PowerShell)
      .\.venv\Scripts\Activate.ps1
      # macOS / Linux
      source .venv/bin/activate

   2. Install dependencies:

      python -m pip install -r requirements.txt

   3. Run the app:

      python even_illumination_app.py

   Notes
   - The app uses OpenCV (`opencv-python`) and scikit-image; installation may build wheels or download prebuilt packages depending on your platform.
   - If you plan to distribute the app to non-developers, consider building an executable with PyInstaller (I can add a build script for that).

   License
   - Add a license file if you want to publish this project (e.g., `LICENSE` with MIT/GPL text).

   Contact
   - For collaboration or questions, add contact info or a project maintainer email.
