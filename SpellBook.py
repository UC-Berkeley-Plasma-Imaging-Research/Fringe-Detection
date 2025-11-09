import os
import threading
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects
# package imports (moved helpers into fringe_detection/)
from fringe_detection import pipeline_shading_sauvola, read_gray
from fringe_detection import binarize, oriented_opening, overlay_mask_on_gray
from fringe_detection import make_slider_row, to_photoimage_from_bgr_with_scale
from tabs.overlay_tab import OverlayTabFrame
from mixins.viewport_rendering import ViewportRenderingMixin


class EvenApp(ViewportRenderingMixin, tk.Tk):
    def _make_slider_row(self, parent, label_text, var, frm, to, resolution=None, is_int=False, fmt=None, command=None):
        if command is None:
            command = self.on_param_change
        return make_slider_row(parent, label_text, var, frm, to, resolution=resolution, is_int=is_int, fmt=fmt, command=command)

    def __init__(self):
        super().__init__()
        self.title('SpellBook — Fringe Detection')
        self.geometry('1100x700')
        try:
            self.after(100, self.lift)
            self.after(120, lambda: self.attributes('-topmost', True))
            self.after(700, lambda: self.attributes('-topmost', False))
        except Exception:
            pass

        # state
        self.src_img = None
        self.enh_img = None
        self.lock = threading.Lock()
        self._after_id = None

        # Notebook tabs: Overlay, Detection, Editor
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill='both', expand=True)

        try:
            self._overlay_frame = OverlayTabFrame(self.notebook, status_callback=self.set_status)
            self.notebook.add(self._overlay_frame, text='Overlay')
        except Exception:
            self._overlay_frame = None

        detect_tab = ttk.Frame(self.notebook)
        self.notebook.add(detect_tab, text='Detection')

        ctrl = ttk.Frame(detect_tab)
        ctrl.pack(side='left', fill='y', padx=8, pady=8)
        ctrl.config(width=260)
        ctrl.pack_propagate(False)

        # Left panel title
        ttk.Label(ctrl, text='Binary Masking', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 6))

        # Row with Browse on the left and Save on the right
        btn_row = ttk.Frame(ctrl)
        btn_row.pack(anchor='w', pady=4)
        ttk.Button(btn_row, text='Browse & Load', command=lambda: self.load_image_dialog()).pack(side='left')
        ttk.Button(btn_row, text='Save Fringes as Binary', command=self.save_result).pack(side='left', padx=(6, 0))

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        self.sigma_var = tk.DoubleVar(value=5.0)
        self.clip_var = tk.DoubleVar(value=50.0)
        self.tile_var = tk.IntVar(value=8)
        self.win_var = tk.IntVar(value=31)
        self.k_var = tk.DoubleVar(value=0.20)
        self.post_var = tk.IntVar(value=1)
        self._slider_meta = {}

        s1 = self._make_slider_row(ctrl, 'Blur σ', self.sigma_var, 0.1, 10.0, is_int=False, fmt="{:.1f}")
        self._slider_meta['Blur σ'] = {'scale': s1, 'var': self.sigma_var, 'is_int': False, 'frm': 0.1, 'to': 10.0}
        s2 = self._make_slider_row(ctrl, 'CLAHE clip', self.clip_var, 10.0, 100.0, is_int=False, fmt="{:.2f}")
        self._slider_meta['CLAHE clip'] = {'scale': s2, 'var': self.clip_var, 'is_int': False, 'frm': 10.0, 'to': 100.0}
        s3 = self._make_slider_row(ctrl, 'CLAHE tile', self.tile_var, 2, 128, is_int=True)
        self._slider_meta['CLAHE tile'] = {'scale': s3, 'var': self.tile_var, 'is_int': True, 'frm': 2, 'to': 128}

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

        # Right panel title
        ttk.Label(fringe_ctrl, text='Fringe Detection', font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=(0, 6))

        img_frame.pack_propagate(False)
        self.viewport = tk.Canvas(img_frame, bg='black', highlightthickness=0)
        self.viewport.pack(side='left', fill='both', expand=True)

        self.inner_frame = ttk.Frame(self.viewport)
        self._inner_window = self.viewport.create_window((0, 0), window=self.inner_frame, anchor='nw')
        self.inner_frame.bind('<Configure>', lambda e: self.viewport.configure(scrollregion=self.viewport.bbox('all')))

        self.illum_canvas = tk.Canvas(self.inner_frame, bg='black', highlightthickness=0)
        self.illum_canvas.pack(fill='x')
        self._illum_img_id = None

        self.fringe_canvas = tk.Canvas(self.inner_frame, bg='black', highlightthickness=0)
        self.fringe_canvas.pack(fill='x')
        self._fringe_img_id = None

        self._photo_illum = None
        self._photo_fringe = None
        self._last_illum_bgr = None
        self._last_overlay_bgr = None
        self._resize_after_id = None
        self._zoom_level = 1.0

        # Removed zoom debug log usage; keeping attribute for compatibility.
        self._zoom_debug_log = None

        self._pan_active = False
        self._pan_start_root = (0, 0)
        self._pan_start_scroll = (0.0, 0.0)
        self._pan_content_size = (1, 1)

        def bind_all(targets):
            for w in targets:
                try:
                    w.bind('<MouseWheel>', self._on_det_wheel, add='+')
                except Exception:
                    pass
                try:
                    w.bind('<Button-3>', self._on_det_pan_start, add='+')
                    w.bind('<B3-Motion>', self._on_det_pan_move, add='+')
                    w.bind('<ButtonRelease-3>', self._on_det_pan_end, add='+')
                except Exception:
                    pass
        bind_all((self.viewport, self.inner_frame, self.illum_canvas, self.fringe_canvas))

        self._is_dragging = False
        self.viewport.bind('<Configure>', self._on_viewport_configure)

        editor_tab = ttk.Frame(self.notebook)
        self.notebook.add(editor_tab, text='Editor')

        try:
            from tabs.fringe_editor import FringeEditorFrame
        except Exception:
            FringeEditorFrame = None

        self._editor_frame = None
        if FringeEditorFrame is not None:
            def on_apply(mask, _bg):
                try:
                    self._binary_mask = mask.copy()
                except Exception:
                    self._binary_mask = mask
                if self.enh_img is not None:
                    try:
                        traced = (255 - self._binary_mask) // 255
                        overlay = overlay_mask_on_gray(self.enh_img, traced.astype(np.uint8), line_alpha=1.0,
                                                       bg_fade=float(self.k_bgfade.get()) if hasattr(self, 'k_bgfade') else 0.4,
                                                       bg_to='white')
                        self._update_illum_and_fringe(cv2.cvtColor(self.enh_img, cv2.COLOR_GRAY2BGR), overlay)
                        self.set_status('Applied edited mask from Editor')
                    except Exception:
                        pass

            def on_close():
                pass

            self._editor_frame = FringeEditorFrame(editor_tab, on_apply=on_apply, on_close=on_close)
            self._editor_frame.pack(fill='both', expand=True)

    # (Title already added at top of right panel)

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
        try:
            zoom_txt = f" (zoom {self._zoom_level:.2f}x)" if hasattr(self, '_zoom_level') else ''
        except Exception:
            zoom_txt = ''
        self.status.config(text=txt + zoom_txt)
        # Simplified status setter: removed file logging of zoom/pan state.

    def load_image_dialog(self, entry_widget=None):
        try:
            self.lift()
        except Exception:
            pass
        try:
            # Default to EditedImages folder for browsing
            base_dir = os.path.abspath(os.path.dirname(__file__))
            edited_dir = os.path.join(base_dir, 'EditedImages')
            initial_dir = edited_dir if os.path.isdir(edited_dir) else base_dir
            p = filedialog.askopenfilename(parent=self, title='Select image', initialdir=initial_dir,
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
                meta['frm'] = new_frm
                meta['to'] = new_to
                scale = meta.get('scale')
                if scale is not None:
                    try:
                        scale.config(from_=new_frm, to=new_to)
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
        if hasattr(self, '_binary_mask') and self._binary_mask is not None:
            to_save = self._binary_mask
        else:
            messagebox.showinfo('No result', 'No result to save')
            return
        # Default save location to EditedImages
        base_dir = os.path.abspath(os.path.dirname(__file__))
        edited_dir = os.path.join(base_dir, 'EditedImages')
        try:
            os.makedirs(edited_dir, exist_ok=True)
        except Exception:
            pass
        p = filedialog.asksaveasfilename(initialdir=edited_dir, defaultextension='.png', filetypes=[('PNG', '*.png')])
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
            traced = skeletonize(opened.astype(bool)).astype(np.uint8)
            self._binary_mask = 255 - (traced * 255).astype(np.uint8)
            overlay = overlay_mask_on_gray(enh, traced, line_alpha=1.0,
                                           bg_fade=float(self.k_bgfade.get()) if hasattr(self, 'k_bgfade') else 0.4,
                                           bg_to='white')
            illum_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR) if enh.ndim == 2 else enh
            self.after(0, lambda: self._update_illum_and_fringe(illum_bgr, overlay))
            # If Editor tab is present and has no mask yet, seed it with the current result.
            try:
                if self._editor_frame is not None and self._editor_frame.get_mask() is None:
                    self._editor_frame.set_data(self._binary_mask)
            except Exception:
                pass
        except Exception:
            self.after(0, lambda: messagebox.showerror('Render error', 'An error occurred during rendering'))
        finally:
            try:
                self.lock.release()
            except Exception:
                pass

    def _update_illum_and_fringe(self, illum_bgr, overlay_bgr):
        try:
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
            base_scale_illum = (float(vp_w) / float(orig_iw)) if orig_iw > 0 else 1.0
            scale_illum = base_scale_illum * self._zoom_level
            photo_illum = to_photoimage_from_bgr_with_scale(illum_disp, scale=scale_illum)
            self._photo_illum = photo_illum
            iw, ih = photo_illum.width(), photo_illum.height()
            self.illum_canvas.config(width=iw, height=ih)
            self._illum_size = (iw, ih)
            x = 0; y = 0
            if self._illum_img_id is None:
                self._illum_img_id = self.illum_canvas.create_image(x, y, anchor='nw', image=photo_illum)
            else:
                self.illum_canvas.itemconfig(self._illum_img_id, image=photo_illum)
                self.illum_canvas.coords(self._illum_img_id, x, y)

            orig_fw = fringe_disp.shape[1]
            base_scale_fringe = (float(vp_w) / float(orig_fw)) if orig_fw > 0 else 1.0
            scale_fringe = base_scale_fringe * self._zoom_level
            photo_fringe = to_photoimage_from_bgr_with_scale(fringe_disp, scale=scale_fringe)
            self._photo_fringe = photo_fringe
            fw, fh = photo_fringe.width(), photo_fringe.height()
            self.fringe_canvas.config(width=fw, height=fh)
            self._fringe_size = (fw, fh)
            x2 = 0; y2 = 0
            if self._fringe_img_id is None:
                self._fringe_img_id = self.fringe_canvas.create_image(x2, y2, anchor='nw', image=photo_fringe)
            else:
                self.fringe_canvas.itemconfig(self._fringe_img_id, image=photo_fringe)
                self.fringe_canvas.coords(self._fringe_img_id, x2, y2)
            try:
                content_w = max(iw, fw)
                self.inner_frame.config(width=content_w)
            except Exception:
                pass

            self.set_status('Rendered')
        except Exception:
            pass  # Error while rendering display; suppressed debug traceback.
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
                try:
                    self._restore_scroll_after_update(pre_x_px, pre_y_px)
                except Exception:
                    pass
            except Exception:
                pass

    def _log_zoom_pan(self, msg):
        # Removed debug logging; function retained as a no-op to avoid dangling calls.
        pass

    def _on_det_wheel(self, event):
        if self._last_illum_bgr is None and self._last_overlay_bgr is None:
            return
        delta = int(getattr(event, 'delta', 0))
        if delta == 0:
            return
        factor = 1.1 if delta > 0 else 0.9
        old_zoom = self._zoom_level
        new_zoom = max(0.1, min(10.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return
        try:
            mx_root, my_root = int(event.x_root), int(event.y_root)
            vp_rx, vp_ry = self.viewport.winfo_rootx(), self.viewport.winfo_rooty()
            mx = mx_root - vp_rx
            my = my_root - vp_ry
        except Exception:
            mx, my = getattr(event, 'x', 0), getattr(event, 'y', 0)
        vp_w = max(1, self.viewport.winfo_width())
        vp_h = max(1, self.viewport.winfo_height())
        bbox_before = self.viewport.bbox('all')
        if not bbox_before:
            return
        content_w_before = bbox_before[2] - bbox_before[0]
        content_h_before = bbox_before[3] - bbox_before[1]
        left_frac_before = self.viewport.xview()[0]
        top_frac_before = self.viewport.yview()[0]
        left_px_before = left_frac_before * content_w_before
        top_px_before = top_frac_before * content_h_before
        anchor_abs_x = left_px_before + mx
        anchor_abs_y = top_px_before + my

        self._zoom_level = new_zoom
        try:
            if self._last_illum_bgr is not None and self._last_overlay_bgr is not None:
                self._update_illum_and_fringe(self._last_illum_bgr, self._last_overlay_bgr)
        except Exception:
            pass
        bbox_after = self.viewport.bbox('all')
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
            self.viewport.xview_moveto(new_left_px / max(1, content_w_after))
            self.viewport.yview_moveto(new_top_px / max(1, content_h_after))
        except Exception:
            pass
        # Debug logging removed.

    def _on_det_pan_start(self, event):
        bbox = self.viewport.bbox('all')
        if not bbox:
            return
        self._pan_active = True
        self._pan_start_root = (int(event.x_root), int(event.y_root))
        self._pan_start_scroll = (self.viewport.xview()[0], self.viewport.yview()[0])
        self._pan_content_size = (bbox[2] - bbox[0], bbox[3] - bbox[1])
        try:
            self.config(cursor='fleur')
        except Exception:
            pass
        # Debug logging removed.

    def _on_det_pan_move(self, event):
        if not self._pan_active:
            return
        (sx, sy) = self._pan_start_root
        dx = int(event.x_root) - sx
        dy = int(event.y_root) - sy
        content_w, content_h = self._pan_content_size
        vp_w = max(1, self.viewport.winfo_width())
        vp_h = max(1, self.viewport.winfo_height())
        start_x_frac, start_y_frac = self._pan_start_scroll
        start_x_px = start_x_frac * content_w
        start_y_px = start_y_frac * content_h
        new_left_px = start_x_px - dx
        new_top_px = start_y_px - dy
        max_left_px = max(0, content_w - vp_w)
        max_top_px = max(0, content_h - vp_h)
        new_left_px = max(0, min(max_left_px, new_left_px))
        new_top_px = max(0, min(max_top_px, new_top_px))
        try:
            self.viewport.xview_moveto(new_left_px / max(1, content_w))
            self.viewport.yview_moveto(new_top_px / max(1, content_h))
        except Exception:
            pass
        # Debug logging removed.

    def _on_det_pan_end(self, event):
        if not self._pan_active:
            return
        self._pan_active = False
        try:
            self.config(cursor='')
        except Exception:
            pass
        # Debug logging removed.

    def on_close(self):
        self.destroy()

    def _img_size(self):
        try:
            if self._last_illum_bgr is not None:
                h, w = self._last_illum_bgr.shape[:2]
                return (w, h)
            if self._last_overlay_bgr is not None:
                h, w = self._last_overlay_bgr.shape[:2]
                return (w, h)
        except Exception:
            pass
        return (0, 0)


def main():
    app = EvenApp()
    app.mainloop()


if __name__ == '__main__':
    main()
