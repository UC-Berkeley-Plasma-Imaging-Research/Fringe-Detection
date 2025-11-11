"""
Dual-view tab: load an image once, display two sections.
- Top section (controlled by LEFT panel): convert to binary via threshold (manual or Otsu).
- Bottom section (controlled by RIGHT panel): overlay the TOP binary on the original with adjustable opacity.

Each viewer supports mouse wheel zoom and right-button pan.
"""
from __future__ import annotations

import os
import math
from typing import Optional

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
import numpy as np

from fringe_detection import read_gray, to_photoimage_from_bgr_with_scale


class DualViewTabFrame(ttk.Frame):
    def __init__(self, master, status_callback=None):
        super().__init__(master)
        self._status = status_callback or (lambda txt: None)

        # Source and products
        self._src_gray: Optional[np.ndarray] = None
        self._bin_mask: Optional[np.ndarray] = None  # 0/255

        # Viewer state (independent zoom per canvas)
        self._zoom_top_orig = 1.0
        self._zoom_top_bin = 1.0
        self._zoom_bottom = 1.0
        self._img_id_top_orig = None
        self._img_id_top_bin = None
        self._img_id_bottom = None
        self._photo_top_orig = None
        self._photo_top_bin = None
        self._photo_bottom = None

        self._build_ui()

    def _build_ui(self):
        # Left controls for Binary Conversion (top section)
        left = ttk.Frame(self)
        left.pack(side='left', fill='y', padx=8, pady=8)
        left.config(width=240)
        left.pack_propagate(False)

        titleL = ttk.Frame(left)
        titleL.pack(anchor='w', fill='x')
        ttk.Label(titleL, text='Binary Conversion', font=('Segoe UI', 10, 'bold')).pack(side='left')
        self._make_help_icon(titleL, (
            'Top section is controlled here.\n'
            '- Browse & Load to choose an image\n'
            '- Choose Auto (Otsu) or manual threshold\n'
            '- Invert flips which pixels are black (0) vs white (255)\n'
            '\nMouse: wheel to zoom, right-drag to pan in viewers.'
        ))

        row = ttk.Frame(left)
        row.pack(anchor='w', pady=4)
        ttk.Button(row, text='Browse & Load', command=self._on_browse).pack(side='left')

        ttk.Separator(left, orient='horizontal').pack(fill='x', pady=6)

        self.auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text='Auto (Otsu)', variable=self.auto_var, command=self._on_params_changed).pack(anchor='w')
        thr_row = ttk.Frame(left)
        thr_row.pack(fill='x', pady=4)
        ttk.Label(thr_row, text='Threshold').pack(side='left')
        self.thr_val_lbl = ttk.Label(thr_row, width=4, anchor='e')
        self.thr_val_lbl.pack(side='right')
        def on_thr(val=None):
            self.thr_val_lbl.config(text=f"{int(self.thr_var.get()):3d}")
            if not self.auto_var.get():
                self._on_params_changed()
        self.thr_var = tk.IntVar(value=128)
        thr_scale = ttk.Scale(thr_row, from_=0, to=255, orient='horizontal', command=lambda *_: on_thr())
        thr_scale.pack(side='left', fill='x', expand=True, padx=8)
        thr_scale.configure(variable=self.thr_var)
        on_thr()

        self.invert_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(left, text='Invert', variable=self.invert_var, command=self._on_params_changed).pack(anchor='w', pady=(2,0))

        # Center area with two sections: top has two viewers (original and binary), bottom has one overlay viewer
        center = ttk.Frame(self)
        center.pack(side='left', fill='both', expand=True, padx=8, pady=8)
        center.pack_propagate(False)

        top_group = ttk.LabelFrame(center, text='Top: Original vs Binary (left UI)')
        top_group.pack(fill='both', expand=True)
        top_group.pack_propagate(False)
        top_row = ttk.Frame(top_group)
        top_row.pack(fill='both', expand=True)
        # Original viewer
        self.canvas_top_orig = self._make_viewer(top_row)
        # Binary viewer
        self.canvas_top_bin = self._make_viewer(top_row)

        bottom_group = ttk.LabelFrame(center, text='Bottom: Binary overlay opacity (right UI)')
        bottom_group.pack(fill='both', expand=True, pady=(6,0))
        bottom_group.pack_propagate(False)
        self.canvas_bottom = self._make_viewer(bottom_group)

        # Right controls for Opacity (bottom section)
        right = ttk.Frame(self)
        right.pack(side='right', fill='y', padx=8, pady=8)
        right.config(width=220)
        right.pack_propagate(False)

        titleR = ttk.Frame(right)
        titleR.pack(anchor='w', fill='x')
        ttk.Label(titleR, text='Overlay Opacity', font=('Segoe UI', 10, 'bold')).pack(side='left')
        self._make_help_icon(titleR, (
            'Bottom section blends the top binary mask over the original image.\n'
            'Use the slider to adjust overlay opacity.'
        ), side='left')

        self.opacity_var = tk.DoubleVar(value=0.5)
        self._make_slider_row(right, 'Opacity', self.opacity_var, 0.0, 1.0, fmt='{:.2f}', command=lambda *_: self._render_bottom())

        self.status = ttk.Label(right, text='Ready', wraplength=180)
        self.status.pack(pady=6)

    def _make_slider_row(self, parent, label_text, var, frm, to, fmt='{:.2f}', command=None):
        row = ttk.Frame(parent)
        row.pack(fill='x', pady=4)
        ttk.Label(row, text=label_text).pack(side='left')
        val_lbl = ttk.Label(row, width=6, anchor='e')
        val_lbl.pack(side='right')
        def on_slide(_=None):
            try:
                val_lbl.config(text=fmt.format(float(var.get())))
            except Exception:
                pass
            if command:
                command()
        scale = ttk.Scale(row, from_=frm, to=to, orient='horizontal', command=on_slide)
        scale.pack(side='left', fill='x', expand=True, padx=8)
        try:
            scale.configure(variable=var)
        except Exception:
            pass
        on_slide()
        return scale

    def _make_help_icon(self, parent, tooltip_text, side='right'):
        try:
            bg = self.cget('background')
        except Exception:
            bg = '#f0f0f0'
        c = tk.Canvas(parent, width=18, height=18, highlightthickness=0, bg=bg)
        c.create_oval(2, 2, 16, 16, outline='#666', width=1)
        c.create_text(9, 9, text='?', font=('Segoe UI', 9))
        c.pack(side='left', padx=(6,0))
        self._attach_tooltip(c, tooltip_text, side=side)
        return c

    def _attach_tooltip(self, widget, text, side='right'):
        tip = {'win': None}
        def show_tip(_e=None):
            if tip['win'] is not None:
                return
            win = tk.Toplevel(widget); tip['win'] = win
            try: win.wm_overrideredirect(True)
            except Exception: pass
            frame = ttk.Frame(win, borderwidth=1, relief='solid'); frame.pack()
            ttk.Label(frame, text=text, justify='left', padding=6).pack()
            try:
                win.update_idletasks()
                wrx = widget.winfo_rootx(); wry = widget.winfo_rooty()
                ww = widget.winfo_width(); wh = widget.winfo_height()
                win_w = win.winfo_width() or win.winfo_reqwidth()
                if str(side).lower() == 'left':
                    x = int(wrx - 8 - win_w)
                else:
                    x = int(wrx + ww + 8)
                y = int(wry + (wh * 0.5))
                if x < 0: x = 0
                if y < 0: y = 0
                win.wm_geometry(f"+{x}+{y}")
            except Exception:
                pass
        def hide_tip(_e=None):
            w = tip.get('win')
            if w is not None:
                try: w.destroy()
                except Exception: pass
                tip['win'] = None
        try:
            widget.bind('<Enter>', show_tip)
            widget.bind('<Leave>', hide_tip)
        except Exception:
            pass

    def _make_viewer(self, parent):
        outer = ttk.Frame(parent)
        outer.pack(side='left', fill='both', expand=True)
        outer.pack_propagate(False)
        canvas = tk.Canvas(outer, bg='black', highlightthickness=0)
        hbar = ttk.Scrollbar(outer, orient='horizontal', command=canvas.xview)
        vbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
        canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        canvas.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='ew')
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)
        # Bind interactions
        canvas.bind('<MouseWheel>', lambda e, c=canvas: self._on_wheel(e, c))
        canvas.bind('<Button-3>', lambda e, c=canvas: c.scan_mark(e.x, e.y))
        canvas.bind('<B3-Motion>', lambda e, c=canvas: c.scan_dragto(e.x, e.y, gain=1))
        return canvas

    def _on_browse(self):
        # Default open in RawImages folder if present
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        raw_dir = os.path.join(base_dir, 'RawImages')
        initial = raw_dir if os.path.isdir(raw_dir) else base_dir
        p = filedialog.askopenfilename(parent=self, title='Select image', initialdir=initial,
                                       filetypes=[('Images', ('*.png','*.jpg','*.jpeg','*.tif','*.tiff'))])
        if not p:
            return
        try:
            g = read_gray(p)
        except Exception:
            g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if g is None:
            messagebox.showerror('Load error', 'Failed to read image')
            return
        if g.dtype != np.uint8:
            g = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        self._src_gray = g
        self._status(f'Loaded {os.path.basename(p)} — {g.shape[1]}×{g.shape[0]}')
        self._on_params_changed()

    def _on_params_changed(self):
        if self._src_gray is None:
            return
        self._compute_binary()
        self._render_top()
        self._render_bottom()

    def _compute_binary(self):
        g = self._src_gray
        inv = bool(self.invert_var.get())
        if bool(self.auto_var.get()):
            # Otsu
            _, thr = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            t = int(max(0, min(255, self.thr_var.get())))
            _, thr = cv2.threshold(g, t, 255, cv2.THRESH_BINARY)
        if inv:
            thr = cv2.bitwise_not(thr)
        self._bin_mask = thr

    def _render_top(self):
        if self._src_gray is None:
            return
        # Original
        bgr = cv2.cvtColor(self._src_gray, cv2.COLOR_GRAY2BGR)
        self._photo_top_orig = to_photoimage_from_bgr_with_scale(bgr, scale=self._zoom_top_orig)
        if self._img_id_top_orig is None:
            self._img_id_top_orig = self.canvas_top_orig.create_image(0,0, anchor='nw', image=self._photo_top_orig)
        else:
            self.canvas_top_orig.itemconfig(self._img_id_top_orig, image=self._photo_top_orig)
        self.canvas_top_orig.config(scrollregion=(0,0,self._photo_top_orig.width(), self._photo_top_orig.height()))

        # Binary (ensure 3-channel for display)
        if self._bin_mask is None:
            bin_bgr = bgr
        else:
            bin_bgr = cv2.cvtColor(self._bin_mask, cv2.COLOR_GRAY2BGR)
        self._photo_top_bin = to_photoimage_from_bgr_with_scale(bin_bgr, scale=self._zoom_top_bin)
        if self._img_id_top_bin is None:
            self._img_id_top_bin = self.canvas_top_bin.create_image(0,0, anchor='nw', image=self._photo_top_bin)
        else:
            self.canvas_top_bin.itemconfig(self._img_id_top_bin, image=self._photo_top_bin)
        self.canvas_top_bin.config(scrollregion=(0,0,self._photo_top_bin.width(), self._photo_top_bin.height()))

    def _render_bottom(self):
        if self._src_gray is None:
            return
        alpha = float(self.opacity_var.get())
        alpha = max(0.0, min(1.0, alpha))
        base = cv2.cvtColor(self._src_gray, cv2.COLOR_GRAY2BGR)
        if self._bin_mask is None:
            out = base
        else:
            # Colorize mask pixels and alpha-blend only where mask is 0 (black) or 255 depending on convention.
            # We'll highlight black (0) pixels by default.
            mask = (self._bin_mask == 0)
            if mask.any():
                color = np.array([0, 255, 0], dtype=np.uint8)  # green overlay
                out = base.copy()
                roi = out[mask]
                blended = cv2.addWeighted(roi, 1.0 - alpha, np.full_like(roi, color), alpha, 0)
                out[mask] = blended
            else:
                out = base
        self._photo_bottom = to_photoimage_from_bgr_with_scale(out, scale=self._zoom_bottom)
        if self._img_id_bottom is None:
            self._img_id_bottom = self.canvas_bottom.create_image(0,0, anchor='nw', image=self._photo_bottom)
        else:
            self.canvas_bottom.itemconfig(self._img_id_bottom, image=self._photo_bottom)
        self.canvas_bottom.config(scrollregion=(0,0,self._photo_bottom.width(), self._photo_bottom.height()))

    def _on_wheel(self, event, canvas):
        # Map canvas to zoom var
        delta = int(getattr(event, 'delta', 0))
        if delta == 0:
            return
        factor = 1.1 if delta > 0 else 0.9
        if canvas is self.canvas_top_orig:
            self._zoom_top_orig = max(0.1, min(10.0, self._zoom_top_orig * factor))
            self._render_top()
        elif canvas is self.canvas_top_bin:
            self._zoom_top_bin = max(0.1, min(10.0, self._zoom_top_bin * factor))
            self._render_top()
        else:
            self._zoom_bottom = max(0.1, min(10.0, self._zoom_bottom * factor))
            self._render_bottom()
