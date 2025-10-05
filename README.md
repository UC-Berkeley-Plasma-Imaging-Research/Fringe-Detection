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
      - If you plan to distribute the app to non-developers, consider building an executable with PyInstaller (this repo includes a GitHub Actions workflow that builds a Windows executable when you push a tag starting with `v`). 

   License
   - Add a license file if you want to publish this project (e.g., `LICENSE` with MIT/GPL text).

   Contact
   - For collaboration or questions, add contact info or a project maintainer email.

   ## Downloads

   Pre-built release assets (Windows ZIPs) are produced automatically by the repository's GitHub Actions workflow when a tag starting with `v` is pushed (for example `v1.0.0`). Those assets are attached to the corresponding GitHub Release. To download:

   - Visit the repository on GitHub and click the Releases tab. Download the ZIP for the latest Windows build.

   Building locally

   If you prefer to build locally (Windows), you can produce a single-file executable using PyInstaller:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt pyinstaller
   pyinstaller --noconfirm --onefile --name "Fringe-Detection" even_illumination_app.py
   # The executable will be in the dist\ folder: dist\Fringe-Detection.exe
   ```

   Creating a release

   To create a release that triggers the GitHub Actions build and uploads an asset:

   1. Create and push a tag from your local machine (PowerShell):

   ```powershell
   git tag v1.0.0
   git push origin v1.0.0
   ```

   2. GitHub Actions will run the workflow and attach a ZIP asset to the Release created for that tag.

   Notes and troubleshooting

   - The workflow currently builds a Windows executable using Python 3.11 on `windows-latest` runners. If you need macOS or Linux builds, we can add additional jobs.
   - If PyInstaller misses any dynamic libraries (rare for OpenCV/scikit-image), the local build log will show missing DLLs; bundling fixes can be applied in the workflow or by editing the PyInstaller spec.
