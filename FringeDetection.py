"""
Entry point renamed to FringeDetection.py. Imports helpers from the `fringe_detection` package.
"""
import os
import threading
import traceback
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
# PIL is used inside ui_helpers; not imported directly here
from skimage.morphology import skeletonize, remove_small_objects

# package imports (moved helpers into fringe_detection/)
from fringe_detection import pipeline_shading_sauvola, read_gray
from fringe_detection import binarize, oriented_opening, overlay_mask_on_gray
from fringe_detection import make_slider_row, to_photoimage_from_bgr_with_scale
from crop_overlay import CropMixin
from mixins.viewport_rendering import ViewportRenderingMixin


# passthrough helper removed


class EvenApp(ViewportRenderingMixin, CropMixin, tk.Tk):
    def _make_slider_row(self, parent, label_text, var, frm, to, resolution=None, is_int=False, fmt=None, command=None):
        if command is None:
            command = self.on_param_change
        return make_slider_row(parent, label_text, var, frm, to, resolution=resolution, is_int=is_int, fmt=fmt, command=command)

    def __init__(self):
        super().__init__()
        self.title('Even Illumination - Fringe Extraction')
        self.geometry('1100x700')
        try:
            # Bring window to front briefly so it isn't hidden behind others
            self.after(100, self.lift)
            self.after(120, lambda: self.attributes('-topmost', True))
            self.after(700, lambda: self.attributes('-topmost', False))
        except Exception:
            pass

        # state
        self.src_img = None
        # illumination outputs (keep only what's used)
        self.enh_img = None
        self.lock = threading.Lock()
        self._after_id = None
        
        # Root layout: Notebook with two tabs: Detection and Editor
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        # Detection tab (existing UI)
        detect_tab = ttk.Frame(self.notebook)
        self.notebook.add(detect_tab, text='Detection')

        # controls frame inside detection tab
        ctrl = ttk.Frame(detect_tab)
        ctrl.pack(side='left', fill='y', padx=8, pady=8)
        ctrl.config(width=260)
        ctrl.pack_propagate(False)

        ttk.Button(ctrl, text='Browse & Load', command=lambda: self.load_image_dialog()).pack(anchor='w', pady=4)
        ttk.Button(ctrl, text='Save Fringes as Binary', command=self.save_result).pack(anchor='w', pady=4)

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        # sliders
        self.sigma_var = tk.DoubleVar(value=5.0)
        self.clip_var = tk.DoubleVar(value=50.0)
        self.tile_var = tk.IntVar(value=8)
        self.win_var = tk.IntVar(value=31)
        self.k_var = tk.DoubleVar(value=0.20)
        self.post_var = tk.IntVar(value=1)

        # keep metadata for sliders so we can edit ranges at runtime
        self._slider_meta = {}

        # Illumination sliders with value readouts
        s1 = self._make_slider_row(ctrl, 'Blur σ', self.sigma_var, 0.1, 10.0, is_int=False, fmt="{:.1f}")
        self._slider_meta['Blur σ'] = {'scale': s1, 'var': self.sigma_var, 'is_int': False, 'frm': 0.1, 'to': 10.0}
        s2 = self._make_slider_row(ctrl, 'CLAHE clip', self.clip_var, 10.0, 100.0, is_int=False, fmt="{:.2f}")
        self._slider_meta['CLAHE clip'] = {'scale': s2, 'var': self.clip_var, 'is_int': False, 'frm': 10.0, 'to': 100.0}
        s3 = self._make_slider_row(ctrl, 'CLAHE tile', self.tile_var, 2, 128, is_int=True)
        self._slider_meta['CLAHE tile'] = {'scale': s3, 'var': self.tile_var, 'is_int': True, 'frm': 2, 'to': 128}

        # button to edit slider ranges
        ttk.Button(ctrl, text='Edit slider ranges', command=self.open_slider_ranges).pack(anchor='w', pady=4)

        self.status = ttk.Label(ctrl, text='Ready', wraplength=220)
        self.status.pack(pady=6)

        self.orig_alpha = tk.DoubleVar(value=0.0)
        s_orig = self._make_slider_row(ctrl, 'Original overlay', self.orig_alpha, 0.0, 1.0, is_int=False, fmt="{:.2f}")
        self._slider_meta['Original overlay'] = {'scale': s_orig, 'var': self.orig_alpha, 'is_int': False, 'frm': 0.0, 'to': 1.0}

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        img_frame = ttk.Frame(detect_tab)
        img_frame.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        fringe_ctrl = ttk.Frame(detect_tab)
        fringe_ctrl.pack(side='right', fill='y', padx=8, pady=8)
        fringe_ctrl.config(width=260)
        fringe_ctrl.pack_propagate(False)

        img_frame.pack_propagate(False)
        self.viewport = tk.Canvas(img_frame, bg='black', highlightthickness=0)
        self.viewport.pack(side='left', fill='both', expand=True)

        self.inner_frame = ttk.Frame(self.viewport)
        self._inner_window = self.viewport.create_window((0, 0), window=self.inner_frame, anchor='nw')
        # update scrollregion when inner_frame changes
        self.inner_frame.bind('<Configure>', lambda e: self.viewport.configure(scrollregion=self.viewport.bbox('all')))

        # stacked canvases inside inner_frame
        self.illum_canvas = tk.Canvas(self.inner_frame, bg='black', highlightthickness=0)
        self.illum_canvas.pack(fill='x')
        self._illum_img_id = None

        self.fringe_canvas = tk.Canvas(self.inner_frame, bg='black', highlightthickness=0)
        self.fringe_canvas.pack(fill='x')
        self._fringe_img_id = None

        # keep reference to PhotoImage to avoid GC
        self._photo_illum = None
        self._photo_fringe = None
        # store last display images (BGR numpy arrays) so we can rescale on window resize
        self._last_illum_bgr = None
        self._last_overlay_bgr = None
        self._resize_after_id = None
        self._zoom_level = 1.0  # track current zoom level

        # mouse wheel scrolling for viewport (Windows wheel: <MouseWheel>)
        def _bind_mousewheel(widget):
            try:
                widget.bind('<Enter>', lambda e: widget.focus_set())
                widget.bind('<MouseWheel>', self._on_mousewheel)
                # also allow shift+wheel for horizontal if needed
                widget.bind('<Shift-MouseWheel>', self._on_mousewheel)
            except Exception:
                pass

        _bind_mousewheel(self.viewport)
        _bind_mousewheel(self.inner_frame)
        _bind_mousewheel(self.illum_canvas)
        _bind_mousewheel(self.fringe_canvas)

        # Click-and-drag panning state
        self._is_dragging = False
        self._drag_start_root = (0, 0)
        self._drag_scroll_start_px = (0.0, 0.0)

        def _bind_panning(widget):
            try:
                # Use right mouse button for panning
                widget.bind('<Button-3>', self._on_pan_start)
                widget.bind('<B3-Motion>', self._on_pan_move)
                widget.bind('<ButtonRelease-3>', self._on_pan_end)
            except Exception:
                pass

        _bind_panning(self.viewport)
        _bind_panning(self.inner_frame)
        _bind_panning(self.illum_canvas)
        _bind_panning(self.fringe_canvas)

        # When the viewport changes size, rescale displayed images to fit width
        self.viewport.bind('<Configure>', self._on_viewport_configure)

        # --- Crop feature state & UI ---
        self._crop_mode = False
        self._crop_rect = None  # (x, y, w, h) in image coordinates
        self._crop_drag = None  # None | 'move' | 'tl' | 'tr' | 'bl' | 'br'
        self._crop_last = None  # last mouse pos in image coords (float)
        self._crop_items = []   # canvas item ids for overlay
        self._crop_handle_px = 12  # size in canvas pixels (not scaled by zoom)

        ttk.Separator(fringe_ctrl, orient='horizontal').pack(fill='x', pady=(6, 6))
        ttk.Label(fringe_ctrl, text='Crop').pack(anchor='w')
        btns = ttk.Frame(fringe_ctrl)
        btns.pack(anchor='w', pady=(2, 4))
        ttk.Button(btns, text='Start crop', command=self._start_crop_mode).pack(side='left', padx=(0, 6))
        ttk.Button(btns, text='Apply', command=self._apply_crop).pack(side='left', padx=(0, 6))
        ttk.Button(btns, text='Cancel', command=self._cancel_crop_mode).pack(side='left')

        # X/Y/W/H controls
        crop_form = ttk.Frame(fringe_ctrl)
        crop_form.pack(fill='x', pady=(2, 6))
        self._crop_x = tk.IntVar(value=0)
        self._crop_y = tk.IntVar(value=0)
        self._crop_w = tk.IntVar(value=0)
        self._crop_h = tk.IntVar(value=0)
        def row(lbl, var):
            fr = ttk.Frame(crop_form)
            fr.pack(fill='x', pady=1)
            ttk.Label(fr, text=lbl, width=6).pack(side='left')
            try:
                sp = tk.Spinbox(fr, from_=0, to=99999, width=8, textvariable=var)
            except Exception:
                sp = ttk.Entry(fr, width=10, textvariable=var)
            sp.pack(side='left')
            return sp
        self._crop_x_sp = row('X', self._crop_x)
        self._crop_y_sp = row('Y', self._crop_y)
        self._crop_w_sp = row('W', self._crop_w)
        self._crop_h_sp = row('H', self._crop_h)
        ttk.Button(crop_form, text='Update rect from fields', command=self._update_rect_from_fields).pack(anchor='w', pady=(3, 0))
        ttk.Button(crop_form, text='Save crop to TXT', command=self._save_crop_dims).pack(anchor='w', pady=(4, 0))

        # --- Editor tab ---
        editor_tab = ttk.Frame(self.notebook)
        self.notebook.add(editor_tab, text='Editor')

        # Lazy import to avoid circulars
        try:
            from fringe_editor import FringeEditorFrame
        except Exception:
            FringeEditorFrame = None

        self._editor_frame = None
        if FringeEditorFrame is not None:
            def on_apply(mask, _bg):
                # Accept edited mask (0 black, 255 white) and refresh overlay view
                try:
                    self._binary_mask = mask.copy()
                except Exception:
                    self._binary_mask = mask
                # If we have an enhanced image, rebuild the overlay to preview
                if self.enh_img is not None:
                    try:
                        traced = (255 - self._binary_mask) // 255  # 1 on lines
                        overlay = overlay_mask_on_gray(self.enh_img, traced.astype(np.uint8), line_alpha=1.0,
                                                       bg_fade=float(self.k_bgfade.get()) if hasattr(self, 'k_bgfade') else 0.4,
                                                       bg_to='white')
                        self._update_illum_and_fringe(cv2.cvtColor(self.enh_img, cv2.COLOR_GRAY2BGR), overlay)
                        self.set_status('Applied edited mask from Editor')
                    except Exception:
                        pass

            def on_close():
                # optional future: prompt to apply/discard
                pass

            self._editor_frame = FringeEditorFrame(editor_tab, on_apply=on_apply, on_close=on_close)
            self._editor_frame.pack(fill='both', expand=True)

        # start
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        # --- Fringe detection controls (defaults tuned from notebook) ---
        ttk.Separator(fringe_ctrl, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(fringe_ctrl, text='Fringe detection').pack(anchor='w')
    
        self.k_len = tk.IntVar(value=41)
        s_klen = self._make_slider_row(fringe_ctrl, 'Kernel length', self.k_len, 5, 151, is_int=True)
        self._slider_meta['Kernel length'] = {'scale': s_klen, 'var': self.k_len, 'is_int': True, 'frm': 5, 'to': 151}
        self.k_thk = tk.IntVar(value=1)
        s_kthk = self._make_slider_row(fringe_ctrl, 'Kernel thickness', self.k_thk, 1, 11, is_int=True)
        self._slider_meta['Kernel thickness'] = {'scale': s_kthk, 'var': self.k_thk, 'is_int': True, 'frm': 1, 'to': 11}
        self.k_ang = tk.DoubleVar(value=8.0)
        s_kang = self._make_slider_row(fringe_ctrl, 'Angle ±', self.k_ang, 0.0, 20.0, is_int=False, fmt="{:.1f}")
        self._slider_meta['Angle ±'] = {'scale': s_kang, 'var': self.k_ang, 'is_int': False, 'frm': 0.0, 'to': 20.0}
        self.k_step = tk.DoubleVar(value=2.0)
        s_kstep = self._make_slider_row(fringe_ctrl, 'Angle step', self.k_step, 0.5, 5.0, is_int=False, fmt="{:.2f}")
        self._slider_meta['Angle step'] = {'scale': s_kstep, 'var': self.k_step, 'is_int': False, 'frm': 0.5, 'to': 5.0}
        self.k_dilate = tk.IntVar(value=1)
        s_kdil = self._make_slider_row(fringe_ctrl, 'Dilate px', self.k_dilate, 0, 6, is_int=True)
        self._slider_meta['Dilate px'] = {'scale': s_kdil, 'var': self.k_dilate, 'is_int': True, 'frm': 0, 'to': 6}
        self.k_area = tk.IntVar(value=50)
        s_karea = self._make_slider_row(fringe_ctrl, 'Min area', self.k_area, 0, 500, is_int=True)
        self._slider_meta['Min area'] = {'scale': s_karea, 'var': self.k_area, 'is_int': True, 'frm': 0, 'to': 500}
    
        self.k_bgfade = tk.DoubleVar(value=0.4)
        s_kbg = self._make_slider_row(fringe_ctrl, 'Background fade', self.k_bgfade, 0.0, 1.0, is_int=False, fmt="{:.2f}")
        self._slider_meta['Background fade'] = {'scale': s_kbg, 'var': self.k_bgfade, 'is_int': False, 'frm': 0.0, 'to': 1.0}

    def set_status(self, txt):
        self.status.config(text=txt)


    def load_image_dialog(self, entry_widget=None):
        # Ensure the app is raised so the file dialog gets focus on macOS
        try:
            self.lift()
        except Exception:
            pass
        try:
            p = filedialog.askopenfilename(parent=self, title='Select image', initialdir=os.path.expanduser('~'),
                                           filetypes=[('Images', ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff'))])
        except Exception as e:
            raise
        if not p:
            return
        if entry_widget is not None:
            try:
                entry_widget.delete(0, 'end')
                entry_widget.insert(0, p)
            except Exception:
                pass
        try:
            self.src_img = read_gray(p)
            self.set_status(f'Loaded: {os.path.basename(p)}')
            self.start_render_now()
        except Exception as e:
            messagebox.showerror('Load error', str(e))

    def open_slider_ranges(self):
        """Open a dialog to edit min/max ranges for each registered slider."""
        dlg = tk.Toplevel(self)
        dlg.title('Edit slider ranges')
        dlg.transient(self)
        dlg.grab_set()

        rows = []
        for i, (name, meta) in enumerate(self._slider_meta.items()):
            ttk.Label(dlg, text=name).grid(row=i, column=0, sticky='w', padx=6, pady=4)
            frm_val = tk.StringVar(value=str(meta.get('frm', '')))
            to_val = tk.StringVar(value=str(meta.get('to', '')))
            e1 = ttk.Entry(dlg, textvariable=frm_val, width=10)
            e1.grid(row=i, column=1, padx=6, pady=4)
            e2 = ttk.Entry(dlg, textvariable=to_val, width=10)
            e2.grid(row=i, column=2, padx=6, pady=4)
            rows.append((name, meta, frm_val, to_val))

        def on_apply():
            # Validate and apply ranges
            for name, meta, frm_var, to_var in rows:
                try:
                    if meta['is_int']:
                        new_frm = int(float(frm_var.get()))
                        new_to = int(float(to_var.get()))
                    else:
                        new_frm = float(frm_var.get())
                        new_to = float(to_var.get())
                except Exception:
                    messagebox.showerror('Invalid value', f'Invalid numeric value for {name}')
                    return
                if new_to <= new_frm:
                    messagebox.showerror('Invalid range', f'Max must be > Min for {name}')
                    return
                # update metadata
                meta['frm'] = new_frm
                meta['to'] = new_to
                # update corresponding scale if we have one
                scale = meta.get('scale')
                if scale is not None:
                    try:
                        scale.config(from_=new_frm, to=new_to)
                        # clamp variable to new range
                        var = meta.get('var')
                        if var is not None:
                            v = var.get()
                            if v < new_frm:
                                var.set(new_frm)
                            elif v > new_to:
                                var.set(new_to)
                    except Exception:
                        pass
            dlg.destroy()

        def on_cancel():
            dlg.destroy()

        btn_fr = ttk.Frame(dlg)
        btn_fr.grid(row=len(rows), column=0, columnspan=3, pady=8)
        ttk.Button(btn_fr, text='Apply', command=on_apply).pack(side='left', padx=8)
        ttk.Button(btn_fr, text='Cancel', command=on_cancel).pack(side='left', padx=8)

    def save_result(self):
        # Save the black-and-white binary mask instead of the red overlay
        if hasattr(self, '_binary_mask') and self._binary_mask is not None:
            to_save = self._binary_mask
        else:
            messagebox.showinfo('No result', 'No result to save')
            return
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not p:
            return
        try:
            cv2.imwrite(p, to_save)
            self.set_status(f'Saved: {os.path.basename(p)}')
        except Exception as e:
            messagebox.showerror('Save error', str(e))

    def on_param_change(self, _=None):
        if self._after_id:
            self.after_cancel(self._after_id)
        self._after_id = self.after(180, self.start_render_now)

    def start_render_now(self):
        if self.src_img is None:
            self.set_status('No image loaded')
            return
        params = (float(self.sigma_var.get()), float(self.clip_var.get()), int(self.tile_var.get()), int(self.win_var.get()), float(self.k_var.get()), int(self.post_var.get()))
        t = threading.Thread(target=self._render_worker, args=(params,), daemon=True)
        t.start()

    def _render_worker(self, params):
        if not self.lock.acquire(blocking=False):
            return
        try:
            flat, enh, binary = pipeline_shading_sauvola(self.src_img, sigma=params[0], clip=params[1], tile=params[2], win=params[3], k=params[4], post_open=params[5])
            self.enh_img = enh

            # Fringe binarization is fixed to Otsu (no threshold slider)
            method = 'Otsu'
            bw = binarize(enh, method=method, blur=0)
            bw01 = (bw > 0).astype(np.uint8)
            opened = oriented_opening(bw01, length=int(self.k_len.get()) if hasattr(self, 'k_len') else 41,
                                      thickness=int(self.k_thk.get()) if hasattr(self, 'k_thk') else 1,
                                      max_angle=float(self.k_ang.get()) if hasattr(self, 'k_ang') else 8.0,
                                      step=float(self.k_step.get()) if hasattr(self, 'k_step') else 2.0)
            if hasattr(self, 'k_dilate') and int(self.k_dilate.get()) > 0:
                K = cv2.getStructuringElement(cv2.MORPH_RECT, (int(self.k_dilate.get()), int(self.k_dilate.get())))
                opened = cv2.dilate(opened, K, 1)
            if hasattr(self, 'k_area') and int(self.k_area.get()) > 0:
                opened_bool = opened.astype(bool)
                opened_bool = remove_small_objects(opened_bool, min_size=int(self.k_area.get()))
                opened = opened_bool.astype(np.uint8)
            # Always skeletonize
            traced = skeletonize(opened.astype(bool)).astype(np.uint8)

            # Store the binary mask (black fringes on white background) for saving
            self._binary_mask = 255 - (traced * 255).astype(np.uint8)

            overlay = overlay_mask_on_gray(enh, traced, line_alpha=1.0,
                                           bg_fade=float(self.k_bgfade.get()) if hasattr(self, 'k_bgfade') else 0.4,
                                           bg_to='white')

            # no need to cache overlay separately

            illum_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR) if enh.ndim == 2 else enh

            self.after(0, lambda: self._update_illum_and_fringe(illum_bgr, overlay))
            # Push latest data into editor if it's open and empty
            try:
                if self._editor_frame is not None and self._editor_frame.get_mask() is None:
                    # Provide current binary result to editor tab for manual tweaks
                    self._editor_frame.set_data(self._binary_mask)
            except Exception:
                pass
        except Exception:
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror('Render error', 'An error occurred during rendering'))
        finally:
            try:
                self.lock.release()
            except Exception:
                pass

    def _update_illum_and_fringe(self, illum_bgr, overlay_bgr):
        try:
            # Capture scroll state before update (both fractions and absolute px)
            try:
                pre_x = self.viewport.xview()
                pre_y = self.viewport.yview()
                pre_bbox = self.viewport.bbox('all')
                pre_w = float(pre_bbox[2] - pre_bbox[0]) if pre_bbox else 1.0
                pre_h = float(pre_bbox[3] - pre_bbox[1]) if pre_bbox else 1.0
                pre_x_px = float(pre_x[0]) * pre_w if pre_w > 0 else 0.0
                pre_y_px = float(pre_y[0]) * pre_h if pre_h > 0 else 0.0
            except Exception:
                pre_x_px = 0.0
                pre_y_px = 0.0
            try:
                self._last_illum_bgr = illum_bgr.copy() if illum_bgr is not None else None
            except Exception:
                self._last_illum_bgr = illum_bgr
            try:
                self._last_overlay_bgr = overlay_bgr.copy() if overlay_bgr is not None else None
            except Exception:
                self._last_overlay_bgr = overlay_bgr
            try:
                cur_first, cur_last = self.viewport.yview()
            except Exception:
                cur_first, cur_last = (0.0, 1.0)
            orig_alpha = float(self.orig_alpha.get()) if hasattr(self, 'orig_alpha') else 0.0
            if orig_alpha > 0.0 and self.src_img is not None:
                src_bgr = cv2.cvtColor(self.src_img, cv2.COLOR_GRAY2BGR) if self.src_img.ndim == 2 else self.src_img
                if src_bgr.shape[:2] != illum_bgr.shape[:2]:
                    src_resized = cv2.resize(src_bgr, (illum_bgr.shape[1], illum_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    src_resized = src_bgr
                illum_disp = cv2.addWeighted(src_resized, orig_alpha, illum_bgr, 1.0 - orig_alpha, 0)
                if overlay_bgr.shape[:2] != src_resized.shape[:2]:
                    src2 = cv2.resize(src_bgr, (overlay_bgr.shape[1], overlay_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    src2 = src_resized
                fringe_disp = cv2.addWeighted(src2, orig_alpha, overlay_bgr, 1.0 - orig_alpha, 0)
            else:
                illum_disp = illum_bgr
                fringe_disp = overlay_bgr

            try:
                vp_w = max(1, self.viewport.winfo_width())
            except Exception:
                vp_w = 800
            orig_iw = illum_disp.shape[1]
            # Apply zoom level to the scale calculation
            base_scale_illum = min(1.0, float(vp_w) / float(orig_iw)) if orig_iw > 0 else 1.0
            scale_illum = base_scale_illum * self._zoom_level
            photo_illum = to_photoimage_from_bgr_with_scale(illum_disp, scale=scale_illum)
            self._photo_illum = photo_illum
            iw, ih = photo_illum.width(), photo_illum.height()
            # For correct zoom/pan, set canvas width to content width (no centering here)
            self.illum_canvas.config(width=iw, height=ih)
            self._illum_size = (iw, ih)
            x = 0
            y = 0
            if self._illum_img_id is None:
                self._illum_img_id = self.illum_canvas.create_image(x, y, anchor='nw', image=photo_illum)
            else:
                self.illum_canvas.itemconfig(self._illum_img_id, image=photo_illum)
                self.illum_canvas.coords(self._illum_img_id, x, y)

            orig_fw = fringe_disp.shape[1]
            # Apply zoom level to the scale calculation
            base_scale_fringe = min(1.0, float(vp_w) / float(orig_fw)) if orig_fw > 0 else 1.0
            scale_fringe = base_scale_fringe * self._zoom_level
            photo_fringe = to_photoimage_from_bgr_with_scale(fringe_disp, scale=scale_fringe)
            self._photo_fringe = photo_fringe
            fw, fh = photo_fringe.width(), photo_fringe.height()
            self.fringe_canvas.config(width=fw, height=fh)
            self._fringe_size = (fw, fh)
            x2 = 0
            y2 = 0
            if self._fringe_img_id is None:
                self._fringe_img_id = self.fringe_canvas.create_image(x2, y2, anchor='nw', image=photo_fringe)
            else:
                self.fringe_canvas.itemconfig(self._fringe_img_id, image=photo_fringe)
                self.fringe_canvas.coords(self._fringe_img_id, x2, y2)
            # Update inner_frame width to the max content width so viewport can scroll horizontally
            try:
                content_w = max(iw, fw)
                self.inner_frame.config(width=content_w)
            except Exception:
                pass

            # no-op: overlay caching removed

            self.set_status('Rendered')
        except Exception:
            traceback.print_exc()
        finally:
            try:
                self.viewport.update_idletasks()
                bbox = self.viewport.bbox('all')
                if bbox is None:
                    return
                inner_w = bbox[2] - bbox[0]
                inner_h = bbox[3] - bbox[1]
                vp_w = self.viewport.winfo_width()
                vp_h = self.viewport.winfo_height()
                self.viewport.configure(scrollregion=(0, 0, inner_w, inner_h))
                # Attempt to restore previous visual position using absolute px, with retries if needed
                try:
                    self._restore_scroll_after_update(pre_x_px, pre_y_px)
                except Exception:
                    pass
                # Redraw crop overlay if active (after images updated)
                try:
                    self._redraw_crop_overlay()
                except Exception:
                    pass
            except Exception:
                pass

    # removed unused _on_canvas_resize handler

    

    # Viewport & rendering methods moved to ViewportRenderingMixin

    def on_close(self):
        self.destroy()

    def _img_size(self):
        try:
            # Prefer illumination size (top image), fallback to overlay size
            if self._last_illum_bgr is not None:
                h, w = self._last_illum_bgr.shape[:2]
                return (w, h)
            if self._last_overlay_bgr is not None:
                h, w = self._last_overlay_bgr.shape[:2]
                return (w, h)
        except Exception:
            pass
        return (0, 0)

    # crop methods are provided by CropMixin

def main():
    app = EvenApp()
    app.mainloop()


if __name__ == '__main__':
    main()
