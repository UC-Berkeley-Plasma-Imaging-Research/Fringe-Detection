# Simple build script for Windows (PowerShell)
# Usage (PowerShell):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\build_windows.ps1

# 1) Create and activate a venv (optional but recommended)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Upgrade pip and install build deps
python -m pip install --upgrade pip
python -m pip install -r requirements.txt pyinstaller

# 3) Build a single-file Windows exe (GUI, no console)
# Use the currently-active Python (the venv's python) to run PyInstaller so the
# module installed into the venv is used (calling `py -3` can invoke a system
# Python instead).
$distExe = Join-Path -Path (Get-Location) -ChildPath 'dist\even_illumination_app.exe'

# If a previous build artifact exists, try to remove it first. If removal fails,
# most likely the executable is still running or is locked by antivirus/explorer.
if (Test-Path $distExe) {
	Write-Host ('Found existing build artifact at {0} - attempting to remove it...' -f $distExe)
	try {
		Remove-Item -Path $distExe -Force -ErrorAction Stop
		Write-Host 'Removed previous artifact. Continuing with build...' -ForegroundColor Green
	}
	catch {
	Write-Host 'Could not remove existing artifact. It may be running or locked by another process.' -ForegroundColor Yellow
	Write-Host 'Please ensure the program is not running, close any Explorer preview, disable antivirus locking the file, or reboot, then re-run this script.' -ForegroundColor Yellow
	Write-Host ('Path: {0}' -f $distExe) -ForegroundColor Yellow
		exit 1
	}
}

# Run PyInstaller via the active venv python
python -m PyInstaller --noconfirm --onefile --windowed even_illumination_app.py

Write-Host 'Build finished. Check the dist\ directory for even_illumination_app.exe' -ForegroundColor Green
