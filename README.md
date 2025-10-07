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

This repository provides a small GUI app to perform even-illumination correction and fringe extraction on images. The code has been reorganized so distribution ZIPs contain a single top-level script plus a package folder containing helpers.

What’s in this repo
- `FringeDetection.py` — the top-level entrypoint to run the GUI app (rename/replace of previous main file).
- `fringe_detection/` — package containing helper modules (`shading_pipeline`, `fringe_utils`, `ui_helpers`).
- `requirements.txt` — runtime dependencies.

Run as Python scripts (no build/release required)

1) Create and activate a virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3) Run the app:

```powershell
python FringeDetection.py
```

   Notes
   - The app uses OpenCV (`opencv-python`), scikit-image, and Pillow; these normally install as binary wheels. On some platforms (or Python versions) pip may attempt to build from source and will require build tools.
   - If you want a single-file EXE later the repo includes PyInstaller support and a GitHub Actions workflow that builds a Windows executable when you push a tag starting with `v`.

   License
   - Add a license file if you want to publish this project (e.g., `LICENSE` with MIT/GPL text).

   Contact
   - For collaboration or questions, add contact info or a project maintainer email.

   ## Downloads

   Pre-built release assets (Windows ZIPs) are produced automatically by the repository's GitHub Actions workflow when a tag starting with `v` is pushed (for example `v1.0.0`). Those assets are attached to the corresponding GitHub Release. To download:

   - Visit the repository on GitHub and click the Releases tab. Download the ZIP for the latest Windows build.

   Building locally

   If you prefer to build locally (Windows), you can produce a single-file executable using PyInstaller — update the command to point at `FringeDetection.py`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --name "Fringe-Detection" FringeDetection.py
# The executable will be in the dist\ folder: dist\Fringe-Detection.exe
```

Run from a downloaded ZIP (Windows)

If you distribute a ZIP with the repo contents, the top-level folder will contain `FringeDetection.py` and the `fringe_detection/` package. A simple set of steps for non-developers:

```powershell
# Extract the ZIP to a folder, open PowerShell in that folder
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python FringeDetection.py
```

Run on macOS

Use the same venv + pip workflow but replace PowerShell commands with shell commands and `python`/`python3` as appropriate:

```bash
cd /path/to/Fringe-Detection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python FringeDetection.py
```

If binary wheel installation fails on macOS (especially Apple Silicon), consider using Miniforge/conda to get prebuilt packages.

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
