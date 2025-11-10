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
        ctrl.config(width=200)  # 50% smaller than previous 260
        ctrl.pack_propagate(False)

        # Left panel title + help icon
        left_title = ttk.Frame(ctrl)
        left_title.pack(anchor='w', fill='x')
        ttk.Label(left_title, text='Binary Masking', font=('Segoe UI', 10, 'bold')).pack(side='left')
        def make_help_icon(parent, tooltip_text, side='right'):
            # ttk frames/labels use style; fallback to system bg via winfo_rgb hack instead of cget('background')
            try:
                bg = self.cget('background')
            except Exception:
                bg = '#f0f0f0'
            c = tk.Canvas(parent, width=18, height=18, highlightthickness=0, bg=bg)
            c.create_oval(2,2,16,16, outline='#666', width=1)
            c.create_text(9,9, text='?', font=('Segoe UI', 9))
            c.pack(side='left', padx=(6,0))
            self._attach_tooltip(c, tooltip_text, side=side)
            return c
        make_help_icon(left_title, (
            'Detection Tab Purpose:\n'
            'This tab allows you to detect fringes in an image using various preprocessing and detection parameters.\n'
            '\n'
            'Controls:\n'
            '- Right-click drag to move image\n'
            '- Mouse wheel to zoom\n'
            '\n'
            'Features:\n'
            '- Load an image to detect fringes\n'
            '- Adjustable preprocessing sliders (blur, CLAHE)\n'
            '- Adjustable Overlay Opacity\n'
        ))

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

        self.status = ttk.Label(ctrl, text='Ready', wraplength=110)
        self.status.pack(pady=6)

        self.orig_alpha = tk.DoubleVar(value=0.0)
        s_orig = self._make_slider_row(ctrl, 'Original overlay', self.orig_alpha, 0.0, 1.0, is_int=False, fmt="{:.2f}")
        self._slider_meta['Original overlay'] = {'scale': s_orig, 'var': self.orig_alpha, 'is_int': False, 'frm': 0.0, 'to': 1.0}

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        # Middle display area with two independent viewers
        img_frame = ttk.Frame(detect_tab)
        img_frame.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        # Right control panel
        fringe_ctrl = ttk.Frame(detect_tab)
        fringe_ctrl.pack(side='right', fill='y', padx=8, pady=8)
        fringe_ctrl.config(width=195)  # 25% smaller than previous 260
        fringe_ctrl.pack_propagate(False)

        # Right panel title + help icon
        right_title = ttk.Frame(fringe_ctrl)
        right_title.pack(anchor='w', fill='x')
        ttk.Label(right_title, text='Fringe Detection', font=('Segoe UI', 10, 'bold')).pack(side='left')
        make_help_icon(right_title, (
            'Fringe Detection Sliders:\n'
            '- Kernel length/thickness: sets oriented opening size.\n'
            '- Angle ± and step: max angle and step at which fringes are drawn.\n'
            '- Dilate px & Min area: post-filter specks before skeletonize.\n'
            '- Background fade: sets overlay dimming for visibility.'
        ), side='left')

        # Two viewers: one for illumination, one for overlay — independent zoom/pan
        img_frame.pack_propagate(False)
        viewers_container = ttk.Frame(img_frame)
        viewers_container.pack(fill='both', expand=True)

        def make_viewer(parent, title):
            lf = ttk.LabelFrame(parent, text=title)
            lf.pack(fill='both', expand=True, pady=(0,6) if title!='Fringe Overlay' else (0,0))
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

        self.illum_canvas = make_viewer(viewers_container, 'Illumination')
        self.fringe_canvas = make_viewer(viewers_container, 'Fringe Overlay')

        # Image / viewer state
        self._illum_img_id = None
        self._fringe_img_id = None
        self._photo_illum = None
        self._photo_fringe = None
        self._last_illum_bgr = None
        self._last_overlay_bgr = None
        self._illum_zoom = 1.0
        self._fringe_zoom = 1.0
        self._zoom_level = 1.0  # for status (use overlay zoom)

        # Bind zoom & pan per viewer
        def bind_viewer(canvas, which):
            canvas.bind('<MouseWheel>', lambda e, w=which: self._on_viewer_wheel(e, w))
            canvas.bind('<Button-3>', lambda e, c=canvas: c.scan_mark(e.x, e.y))
            canvas.bind('<B3-Motion>', lambda e, c=canvas: c.scan_dragto(e.x, e.y, gain=1))
        bind_viewer(self.illum_canvas, 'illum')
        bind_viewer(self.fringe_canvas, 'fringe')

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

    def _attach_tooltip(self, widget, text, side='right'):
        # Minimal tooltip on hover using a small Toplevel
        tip = {'win': None}

        def show_tip(_e=None):
            if tip['win'] is not None:
                return
            win = tk.Toplevel(widget)
            tip['win'] = win
            try:
                win.wm_overrideredirect(True)
            except Exception:
                pass
            frame = ttk.Frame(win, borderwidth=1, relief='solid')
            frame.pack()
            lbl = ttk.Label(frame, text=text, justify='left', padding=6)
            lbl.pack()
            # Position after measuring to support left-of-icon placement
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
                try:
                    w.destroy()
                except Exception:
                    pass
                tip['win'] = None

        try:
            widget.bind('<Enter>', show_tip)
            widget.bind('<Leave>', hide_tip)
        except Exception:
            pass

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
        except Exception:
            self.after(0, lambda: messagebox.showerror('Render error', 'An error occurred during rendering'))
        finally:
            try:
                self.lock.release()
            except Exception:
                pass

    def _update_illum_and_fringe(self, illum_bgr, overlay_bgr):
        try:
            # Store originals
            self._last_illum_bgr = None if illum_bgr is None else illum_bgr.copy()
            self._last_overlay_bgr = None if overlay_bgr is None else overlay_bgr.copy()
            orig_alpha = float(self.orig_alpha.get()) if hasattr(self, 'orig_alpha') else 0.0
            if orig_alpha > 0.0 and self.src_img is not None and illum_bgr is not None:
                src_bgr = cv2.cvtColor(self.src_img, cv2.COLOR_GRAY2BGR) if self.src_img.ndim == 2 else self.src_img
                if src_bgr.shape[:2] != illum_bgr.shape[:2]:
                    src_resized = cv2.resize(src_bgr, (illum_bgr.shape[1], illum_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    src_resized = src_bgr
                illum_disp = cv2.addWeighted(src_resized, orig_alpha, illum_bgr, 1.0 - orig_alpha, 0)
                if overlay_bgr is not None:
                    if overlay_bgr.shape[:2] != src_resized.shape[:2]:
                        src2 = cv2.resize(src_bgr, (overlay_bgr.shape[1], overlay_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                    else:
                        src2 = src_resized
                    fringe_disp = cv2.addWeighted(src2, orig_alpha, overlay_bgr, 1.0 - orig_alpha, 0)
                else:
                    fringe_disp = overlay_bgr
            else:
                illum_disp = illum_bgr
                fringe_disp = overlay_bgr

            # Render illumination viewer
            if illum_disp is not None:
                photo_illum = to_photoimage_from_bgr_with_scale(illum_disp, scale=self._illum_zoom)
                self._photo_illum = photo_illum
                if self._illum_img_id is None:
                    self._illum_img_id = self.illum_canvas.create_image(0,0, anchor='nw', image=photo_illum)
                else:
                    self.illum_canvas.itemconfig(self._illum_img_id, image=photo_illum)
                self.illum_canvas.config(scrollregion=(0,0,photo_illum.width(), photo_illum.height()))
            # Render fringe viewer
            if fringe_disp is not None:
                photo_fringe = to_photoimage_from_bgr_with_scale(fringe_disp, scale=self._fringe_zoom)
                self._photo_fringe = photo_fringe
                if self._fringe_img_id is None:
                    self._fringe_img_id = self.fringe_canvas.create_image(0,0, anchor='nw', image=photo_fringe)
                else:
                    self.fringe_canvas.itemconfig(self._fringe_img_id, image=photo_fringe)
                self.fringe_canvas.config(scrollregion=(0,0,photo_fringe.width(), photo_fringe.height()))
            self._zoom_level = self._fringe_zoom
            self.set_status('Rendered')
        except Exception:
            pass

    # Per-viewer wheel handler
    def _on_viewer_wheel(self, event, which):
        delta = int(getattr(event, 'delta', 0))
        if delta == 0:
            return
        factor = 1.1 if delta > 0 else 0.9
        if which == 'illum':
            old = self._illum_zoom
            self._illum_zoom = max(0.1, min(10.0, old * factor))
        else:
            old = self._fringe_zoom
            self._fringe_zoom = max(0.1, min(10.0, old * factor))
        self._zoom_level = self._fringe_zoom
        if self._last_illum_bgr is not None or self._last_overlay_bgr is not None:
            self._update_illum_and_fringe(self._last_illum_bgr, self._last_overlay_bgr)

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
