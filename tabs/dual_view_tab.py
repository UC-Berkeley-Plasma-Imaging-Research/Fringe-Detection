import os
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

from fringe_detection import to_photoimage_from_bgr_with_scale


class DualViewTabFrame(ttk.Frame):
    """Tab that loads a single image and shows it twice (stacked),
    with independent zoom and pan per viewer. Wheel zoom anchors at the cursor.
    """

    def __init__(self, master, status_callback=None):
        super().__init__(master)
        self._status = status_callback or (lambda txt: None)

        # Data/state
        self._img_bgr = None

        # Per-viewer state
        self._zoom_top = 1.0
        self._zoom_bottom = 1.0
        self._photo_top = None
        self._photo_bottom = None
        self._img_id_top = None
        self._img_id_bottom = None

        self._build_ui()
        self._bind_events()

    # ---------------------------- UI ----------------------------
    def _build_ui(self):
        # Left controls
        ctrl = ttk.Frame(self)
        ctrl.pack(side='left', fill='y', padx=8, pady=8)
        ctrl.config(width=220)
        ctrl.pack_propagate(False)

        title = ttk.Label(ctrl, text='Dual View', font=('Segoe UI', 10, 'bold'))
        title.pack(anchor='w', pady=(0, 6))

        ttk.Button(ctrl, text='Load Image', command=self._load_image).pack(anchor='w')
        ttk.Label(ctrl, text='Tip: Wheel to zoom, Right-drag to pan').pack(anchor='w', pady=(8,0))

        # Middle: two stacked viewers
        mid = ttk.Frame(self)
        mid.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        mid.pack_propagate(False)

        self.top_canvas = self._make_viewer(mid, 'Top Image')
        self.bottom_canvas = self._make_viewer(mid, 'Bottom Image')

    def _make_viewer(self, parent, title):
        lf = ttk.LabelFrame(parent, text=title)
        lf.pack(fill='both', expand=True, pady=(0,6))
        lf.pack_propagate(False)
        outer = ttk.Frame(lf)
        outer.pack(fill='both', expand=True)
        canvas = tk.Canvas(outer, bg='black', highlightthickness=0)
        hbar = ttk.Scrollbar(outer, orient='horizontal', command=canvas.xview)
        vbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='ew')
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        return canvas

    def _bind_events(self):
        # Top viewer
        self.top_canvas.bind('<MouseWheel>', lambda e: self._on_wheel(self.top_canvas, 'top', e))
        self.top_canvas.bind('<Button-3>', lambda e: self.top_canvas.scan_mark(e.x, e.y))
        self.top_canvas.bind('<B3-Motion>', lambda e: self.top_canvas.scan_dragto(e.x, e.y, gain=1))
        # Linux wheel
        self.top_canvas.bind('<Button-4>', lambda e: self._on_wheel(self.top_canvas, 'top', e))
        self.top_canvas.bind('<Button-5>', lambda e: self._on_wheel(self.top_canvas, 'top', e))

        # Bottom viewer
        self.bottom_canvas.bind('<MouseWheel>', lambda e: self._on_wheel(self.bottom_canvas, 'bottom', e))
        self.bottom_canvas.bind('<Button-3>', lambda e: self.bottom_canvas.scan_mark(e.x, e.y))
        self.bottom_canvas.bind('<B3-Motion>', lambda e: self.bottom_canvas.scan_dragto(e.x, e.y, gain=1))
        # Linux wheel
        self.bottom_canvas.bind('<Button-4>', lambda e: self._on_wheel(self.bottom_canvas, 'bottom', e))
        self.bottom_canvas.bind('<Button-5>', lambda e: self._on_wheel(self.bottom_canvas, 'bottom', e))

    # ------------------------- Image IO -------------------------
    def _load_image(self):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        raw_dir = os.path.join(base_dir, 'RawImages')
        initial = raw_dir if os.path.isdir(raw_dir) else base_dir
        p = filedialog.askopenfilename(parent=self, title='Select image', initialdir=initial,
                                       filetypes=[('Images', ('*.png','*.jpg','*.jpeg','*.tif','*.tiff'))])
        if not p:
            return
        img0 = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img0 is None:
            messagebox.showerror('Load error', f'Could not read image: {os.path.basename(p)}')
            return
        img_u8 = self._normalize_image_dtype(img0)
        self._img_bgr = self._ensure_bgr(img_u8)
        self._zoom_top = 1.0
        self._zoom_bottom = 1.0
        self._render_both()
        self._status(f'Loaded: {os.path.basename(p)}')

    def _normalize_image_dtype(self, img):
        if img is None:
            return None
        if img.dtype == np.uint8:
            return img
        # Normalize any other dtype to 0..255 uint8 for display
        try:
            info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
        except Exception:
            info = None
        if info is not None:
            rng = float(info.max - info.min) or 1.0
            out = ((img.astype(np.float32) - info.min) * (255.0 / rng)).clip(0, 255).astype(np.uint8)
            return out
        else:
            # float or unknown
            min_v = float(np.nanmin(img)) if np.size(img) else 0.0
            max_v = float(np.nanmax(img)) if np.size(img) else 1.0
            rng = (max_v - min_v) or 1.0
            out = (((img.astype(np.float32) - min_v) * (255.0 / rng))).clip(0, 255).astype(np.uint8)
            return out

    def _ensure_bgr(self, img):
        if img is None:
            return None
        try:
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
        except Exception:
            return img

    # -------------------------- Render --------------------------
    def _render_both(self):
        self._render_view(self.top_canvas, 'top')
        self._render_view(self.bottom_canvas, 'bottom')

    def _render_view(self, canvas, which):
        if self._img_bgr is None:
            canvas.delete('all')
            return
        scale = self._zoom_top if which == 'top' else self._zoom_bottom
        photo = to_photoimage_from_bgr_with_scale(self._img_bgr, scale=max(0.1, float(scale)))
        if which == 'top':
            self._photo_top = photo
            if self._img_id_top is None:
                self._img_id_top = canvas.create_image(0, 0, anchor='nw', image=photo)
            else:
                canvas.itemconfig(self._img_id_top, image=photo)
        else:
            self._photo_bottom = photo
            if self._img_id_bottom is None:
                self._img_id_bottom = canvas.create_image(0, 0, anchor='nw', image=photo)
            else:
                canvas.itemconfig(self._img_id_bottom, image=photo)
        canvas.config(scrollregion=(0, 0, photo.width(), photo.height()))

    # --------------------------- Zoom ---------------------------
    def _on_wheel(self, canvas, which, event):
        if self._img_bgr is None:
            return
        # Determine wheel delta across platforms
        delta = int(getattr(event, 'delta', 0))
        if delta == 0:
            # Linux X11 fallback
            num = getattr(event, 'num', None)
            if num == 4:
                delta = 120
            elif num == 5:
                delta = -120
        if delta == 0:
            return

        old_zoom = self._zoom_top if which == 'top' else self._zoom_bottom
        factor = 1.1 if delta > 0 else 0.9
        new_zoom = max(0.1, min(16.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        # Anchor point (cursor) in content coordinates before change
        try:
            mx, my = int(event.x), int(event.y)
        except Exception:
            mx = my = 0
        bbox = canvas.bbox('all')
        if not bbox:
            return
        content_w_before = bbox[2] - bbox[0]
        content_h_before = bbox[3] - bbox[1]
        vp_w = max(1, canvas.winfo_width())
        vp_h = max(1, canvas.winfo_height())
        left_frac, top_frac = canvas.xview()[0], canvas.yview()[0]
        left_px_before = left_frac * content_w_before
        top_px_before = top_frac * content_h_before
        anchor_abs_x = left_px_before + mx
        anchor_abs_y = top_px_before + my

        # Apply zoom and redraw
        if which == 'top':
            self._zoom_top = new_zoom
        else:
            self._zoom_bottom = new_zoom
        self._render_view(canvas, which)

        # Restore view so cursor stays on the same content point
        bbox_after = canvas.bbox('all')
        if not bbox_after:
            return
        content_w_after = bbox_after[2] - bbox_after[0]
        content_h_after = bbox_after[3] - bbox_after[1]
        rx = (content_w_after / content_w_before) if content_w_before > 0 else 1.0
        ry = (content_h_after / content_h_before) if content_h_before > 0 else 1.0
        new_left_px = (rx * anchor_abs_x) - mx
        new_top_px = (ry * anchor_abs_y) - my
        max_left_px = max(0, content_w_after - vp_w)
        max_top_px = max(0, content_h_after - vp_h)
        new_left_px = max(0, min(max_left_px, new_left_px))
        new_top_px = max(0, min(max_top_px, new_top_px))
        try:
            canvas.xview_moveto(new_left_px / max(1, content_w_after))
            canvas.yview_moveto(new_top_px / max(1, content_h_after))
        except Exception:
            pass
