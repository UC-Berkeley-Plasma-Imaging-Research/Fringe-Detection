Simple one-file build (PyInstaller)

This file contains the absolute minimal steps to create a single-file executable for each OS.

Windows (run on Windows):

1. Install build deps and PyInstaller (recommended in a venv):

   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt pyinstaller

2. Build one-file exe:

   py -3 -m PyInstaller --noconfirm --onefile --windowed even_illumination_app.py

3. Result: dist\even_illumination_app.exe

macOS (run on macOS):

1. Install build deps and PyInstaller (recommended in a venv):

   python3 -m venv .venv
   source .venv/bin/activate
   python3 -m pip install --upgrade pip
   python3 -m pip install -r requirements.txt pyinstaller

2. Build one-file binary:

   python3 -m PyInstaller --noconfirm --onefile --windowed even_illumination_app.py

3. Result: dist/even_illumination_app

Notes and caveats:
- Build on target OS: you must run the macOS build on macOS (or use macOS CI) and the Windows build on Windows.
- For GUI apps, use --windowed so a console window does not appear.
- To include extra data files (icons, sample images), use --add-data. Remember Windows uses a semicolon separator and macOS/Linux use a colon.
- macOS notarization: distributing macOS apps to broad audiences requires code signing and notarization by Apple (Apple Developer account).
- The produced single-file executables can be large (OpenCV and Python runtime included).

If you want, I can:
- Add a small GitHub Actions job to zip built artifacts and compute SHA256 checksums.
- Add an example --add-data line for any images you want bundled.
