Fringe Detection

This repository contains a small GUI app and helper modules for fringe detection and even-illumination correction.

Contents
- FringeDetection.py - top-level script that launches the GUI app.
- fringe_detection/ - helper package with processing and UI utilities.
- requirements.txt - Python dependencies.
- Images/ - example and reference images (optional).

Requirements
- Python 3.8+ (3.11 is recommended)
- A virtual environment is recommended to install dependencies.

Quick start (Windows PowerShell)
1. Create and activate a virtual environment:
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1

2. Install dependencies:
   python -m pip install --upgrade pip
   pip install -r requirements.txt

3. Run the app:
   python FringeDetection.py

Notes
- The app depends on OpenCV (opencv-python), scikit-image and Pillow. On some platforms installing these packages may pull in binary wheels or require build tools.
- This repository is intended to be used as Python source. Build artifacts and packaging scripts were removed to keep the project lightweight.

License
- Add a LICENSE file (e.g., MIT) if you want to publish this project.

Contact
- Add maintainer contact information here if desired.
