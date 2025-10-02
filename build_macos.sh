#!/usr/bin/env bash
# Simple build script for macOS (run on macOS)
# Usage:
#   chmod +x build_macos.sh
#   ./build_macos.sh

set -euo pipefail

# 1) Create and activate a venv (optional but recommended)
python3 -m venv .venv
# activate: source .venv/bin/activate
. .venv/bin/activate

# 2) Upgrade pip and install build deps
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt pyinstaller

# 3) Build a single-file macOS app (one-file binary)
# If you need to include extra data, add --add-data "path:dest" (macOS uses : separator)
python3 -m PyInstaller --noconfirm --onefile --windowed even_illumination_app.py

echo "Build finished. Check the dist/ directory for the built binary."