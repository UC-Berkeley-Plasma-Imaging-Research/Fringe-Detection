# SpellBook — Fringe Detection & Overlay App

SpellBook is a Tkinter-based desktop tool for:
1. Preprocessing (even illumination / contrast enhancement)
2. Automatic fringe (line) detection & skeletonization
3. Manual mask touch‑up (add/remove/link endpoints)
4. Overlay visualization for quality review

---
## 1. Repository Layout

| Path | Purpose |
|------|---------|
| `SpellBook.py` | Entry point launching the GUI (Overlay, Detection, Editor tabs). |
| `fringe_detection/` | Processing helpers: shading pipeline, binarization & oriented opening utilities. |
| `tabs/overlay_tab.py` | Overlay/registration & cropping UI. |
| `tabs/fringe_editor.py` | Interactive binary mask editor (paint + link endpoints). |
| `mixins/viewport_rendering.py` | Shared viewport zoom/pan helpers (legacy mixin). |
| `requirements.txt` | Python dependencies. |

---
## 2. Quick Start (Windows PowerShell)

```powershell
python -m venv .venv
./.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python SpellBook.py
```

macOS / Linux (bash):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python SpellBook.py
```

If wheel installation fails (e.g. on Apple Silicon), use a Conda/Miniforge environment.

---
## 3. Tutorial Walkthrough

### Step 0: Prepare Your Images
Place raw input images in a convenient folder. Supported formats: PNG / TIFF / JPG. High bit‑depth TIFFs are internally normalized to 8‑bit for display.

### Step 1: Launch & Load (Detection Tab)
1. Click `Browse & Load` to pick an image.
2. Adjust preprocessing sliders:
   - Blur σ: mild Gaussian smoothing for noise reduction.
   - CLAHE clip & tile: local contrast expansion; high clip increases contrast, tile defines grid size.
3. Original overlay (optional): blend some of the original intensity back into processed views for context.

The two center viewers show:
| Viewer | Content |
|--------|---------|
| Illumination | Enhanced grayscale (after shading + contrast). |
| Fringe Overlay | Detected fringe skeleton overlayed (green/colored) on processed background. |

Zoom: mouse wheel (each viewer independent). Pan: right‑click drag. Status bar shows current zoom.

### Step 2: Tune Fringe Detection
Right panel sliders:
| Slider | Effect |
|--------|--------|
| Kernel length | Line structuring element length for oriented opening (line extraction). |
| Kernel thickness | Line thickness assumption. |
| Angle ± / Angle step | Angular sweep around horizontal for oriented opening (coverage vs. speed). |
| Dilate px | Thickens detected ridges before skeletonization. |
| Min area | Removes small specks before skeletonizing. |
| Background fade | Dims background under overlay for visibility. |

After adjustments the overlay viewer updates automatically (debounced ~180 ms).

### Step 3: Save Automatic Result
Click `Save Fringes as Binary` to export the current binary mask (0 = fringe, 255 = background).

### Step 4: Fine Editing (Editor Tab)
Switch to `Editor`:
1. `Open Binary` – load a saved mask OR create one from an image (thresholded automatically).
2. (Optional) `Open Background` – load a grayscale backdrop for contextual editing.
3. Mode radio buttons: `Add Black` paints fringe (sets pixels to 0); `Remove Black` erases fringe (sets to 255).
4. Mouse:
   - Left drag: paint / erase.
   - Ctrl + wheel: change brush radius.
   - Wheel: zoom.
   - Right drag: pan.
5. `Link endpoints` + Angle / Link tol (px): connect nearby skeleton endpoints within tolerance & angular constraint.
6. `½ Angle, 2× Tol` quickly broadens search tolerance while tightening angle.
7. `Color comps` optionally pseudo‑colors connected components.
8. `Undo` reverts last stroke (stack depth 20).

No mask is auto‑loaded into the Editor; you decide when to load or import one.

### Step 5: Iterate & Export
Refine, then `Save As…` in the Editor for a cleaned fringe mask. Use saved masks for downstream analysis or comparison.

---
## 4. Design Notes
| Aspect | Choice |
|--------|--------|
| Independent zoom | Each viewer maintains its own zoom state for local inspection. |
| Skeletonization | Uses `skimage.morphology.skeletonize` on prefiltered binary. |
| Oriented opening | Sweeps discrete angles (± range, given step) with line structuring elements for ridge isolation. |
| Endpoint linking | Brute-force within radius + angle gate + component separation. |
| Performance | Debounced slider changes; uses integer structuring elements. |

---
## 5. Building a Standalone Executable (Windows)
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller
pyinstaller --noconfirm --onefile --name "SpellBook" SpellBook.py
```
Binary appears in `dist/SpellBook.exe`.

GitHub Actions can be configured to run this on tag push (e.g. `v1.2.0`).

---
## 6. Troubleshooting
| Issue | Fix |
|-------|-----|
| Wheel zoom not working | Ensure window focus; on Linux use `<Button-4>/<Button-5>` events. |
| Missing DLL (OpenCV) | Reinstall with `pip install --force-reinstall opencv-python`. |
| Slow large images | Reduce Blur σ & CLAHE tile; disable Color comps during editing. |
| No fringes detected | Increase kernel length, lower Min area, adjust Angle ±. |

---
## 7. Contributing
1. Fork & branch: `git checkout -b feature/xyz`.
2. Keep UI changes minimal per commit.
3. Run lint/tests (add if missing) before PR.

---
## 8. License & Contact
Add `LICENSE` (MIT recommended) and maintainer contact/email here.

---
## 9. At-a-Glance Commands
```powershell
# Setup
python -m venv .venv; ./.venv/Scripts/Activate.ps1
pip install -r requirements.txt

# Run
python SpellBook.py

# Build executable
pyinstaller --onefile --name SpellBook SpellBook.py
```

---
## 10. FAQ
**Q: Why two viewers?** Independent inspection of raw enhancement vs. overlay.
**Q: Why isn’t the Editor auto-filled?** Manual control prevents accidental edits; load explicitly.
**Q: Units of link tolerance?** Pixels in original image coordinates.

---
Happy detecting and editing! Feel free to open issues for feature requests.
