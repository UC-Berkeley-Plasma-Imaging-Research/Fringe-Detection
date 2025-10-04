"""
Standalone Even Illumination GUI using tkinter
Uses tkinter + Pillow for GUI, OpenCV + scikit-image for processing.

Run from source:
  python -m pip install -r requirements.txt
  python even_illumination_app.py

Build with PyInstaller (Windows):
  python -m pip install pyinstaller
  python -m PyInstaller --noconfirm --onefile --windowed even_illumination_app.py

"""

import os
import threading
import traceback
from tkinter import filedialog, messagebox
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize, remove_small_objects

# import processing helpers extracted into small modules
from shading_pipeline import pipeline_shading_sauvola, read_gray
from fringe_utils import binarize, line_kernel, oriented_opening, overlay_mask_on_gray
from ui_helpers import make_slider_row, to_photoimage_from_bgr_with_scale


def to_photoimage_from_bgr(bgr):
    # Convert BGR (OpenCV) to PIL PhotoImage for tkinter
    # kept for backward compatibility: default scale = 1.0
    return to_photoimage_from_bgr_with_scale(bgr, scale=1.0)


def to_photoimage_from_bgr_with_scale(bgr, scale=1.0):
    # bgr: numpy array (H,W) gray or (H,W,3) BGR
    if bgr is None:
        return ImageTk.PhotoImage(Image.new('RGB', (1, 1)))
    if bgr.ndim == 2:
        img = Image.fromarray(bgr)
    else:
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if scale is None or float(scale) == 1.0:
        return ImageTk.PhotoImage(img)
    # resize image by scale (keep integer sizes)
    w, h = img.size
    new_w = max(1, int(round(w * float(scale))))
    new_h = max(1, int(round(h * float(scale))))
    img2 = img.resize((new_w, new_h), resample=Image.BILINEAR)
    return ImageTk.PhotoImage(img2)



class EvenApp(tk.Tk):
    def _make_slider_row(self, parent, label_text, var, frm, to, resolution=None, is_int=False, fmt=None, command=None):
        # Thin wrapper to maintain compatibility with code that calls
        # self._make_slider_row while delegating the implementation to ui_helpers.
        if command is None:
            command = self.on_param_change
        return make_slider_row(parent, label_text, var, frm, to, resolution=resolution, is_int=is_int, fmt=fmt, command=command)

    def __init__(self):
        super().__init__()
        self.title('Even Illumination - Fringe Extraction')
        self.geometry('1100x700')

        # state
        self.src_img = None
        # illumination outputs
        self.flat_img = None
        self.enh_img = None
        self.last_result = None
        self.lock = threading.Lock()
        self._after_id = None

        # controls frame
        ctrl = ttk.Frame(self)
        ctrl.pack(side='left', fill='y', padx=8, pady=8)
        # set a target control column width and prevent propagation so width is respected
        ctrl.config(width=260)
        ctrl.pack_propagate(False)

        ttk.Label(ctrl, text='Open image:').pack(anchor='w')
        path_entry = ttk.Entry(ctrl, width=30)
        path_entry.pack(anchor='w')
        ttk.Button(ctrl, text='Browse & Load', command=lambda: self.load_image_dialog(path_entry)).pack(anchor='w', pady=4)
        ttk.Button(ctrl, text='Save result', command=self.save_result).pack(anchor='w', pady=4)

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        # sliders
        self.sigma_var = tk.DoubleVar(value=10.0)
        self.clip_var = tk.DoubleVar(value=2.5)
        self.tile_var = tk.IntVar(value=8)
        self.win_var = tk.IntVar(value=31)
        self.k_var = tk.DoubleVar(value=0.20)
        self.post_var = tk.IntVar(value=1)

        # Illumination sliders with value readouts
        s1 = self._make_slider_row(ctrl, 'Blur σ', self.sigma_var, 1, 50, is_int=False, fmt="{:.1f}")
        s2 = self._make_slider_row(ctrl, 'CLAHE clip', self.clip_var, 1.0, 16.0, is_int=False, fmt="{:.2f}")
        s3 = self._make_slider_row(ctrl, 'CLAHE tile', self.tile_var, 2, 128, is_int=True)
        s4 = self._make_slider_row(ctrl, 'Sauvola win', self.win_var, 3, 402, is_int=True)
        s5 = self._make_slider_row(ctrl, 'Sauvola k', self.k_var, 0.0, 2.0, is_int=False, fmt="{:.3f}")
        s6 = self._make_slider_row(ctrl, 'Post open', self.post_var, 0, 40, is_int=True)

        # real-time rendering — button removed
        self.status = ttk.Label(ctrl, text='Ready', wraplength=220)
        self.status.pack(pady=6)

        # UI scale removed: images are auto-sized to fit between panels

        # overlay original image over results (0.0 = off, 1.0 = full original)
        self.orig_alpha = tk.DoubleVar(value=0.0)
        s_orig = self._make_slider_row(ctrl, 'Original overlay', self.orig_alpha, 0.0, 1.0, is_int=False, fmt="{:.2f}")

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        # image column (center)
        img_frame = ttk.Frame(self)
        img_frame.pack(side='left', fill='both', expand=True, padx=8, pady=8)

        # right-side fringe control panel (move fringe controls here)
        fringe_ctrl = ttk.Frame(self)
        fringe_ctrl.pack(side='right', fill='y', padx=8, pady=8)
        # match width to left control panel
        fringe_ctrl.config(width=260)
        fringe_ctrl.pack_propagate(False)

        # Use a viewport Canvas to contain two stacked canvases (illumination above, fringe below).
        # This lets the top canvas expand (pushing the bottom down) while the viewport provides clipping.
        img_frame.pack_propagate(False)
        # viewport canvas (clips horizontally and vertically)
        self.viewport = tk.Canvas(img_frame, bg='black', highlightthickness=0)
        self.viewport.pack(side='left', fill='both', expand=True)

        # inner frame inside viewport holds the two canvases stacked vertically
        self.inner_frame = ttk.Frame(self.viewport)
        self._inner_window = self.viewport.create_window((0, 0), window=self.inner_frame, anchor='nw')
        # update scrollregion when inner_frame changes
        self.inner_frame.bind('<Configure>', lambda e: self.viewport.configure(scrollregion=self.viewport.bbox('all')))

        # vertical image scroll slider (appears on right of img area)
        self.img_scroll_var = tk.DoubleVar(value=0.0)
        self.img_scroll = ttk.Scale(img_frame, orient='vertical', from_=0.0, to=0.0, variable=self.img_scroll_var, command=self._on_img_scroll)
        self.img_scroll.pack(side='right', fill='y')

        # stacked canvases inside inner_frame
        self.illum_canvas = tk.Canvas(self.inner_frame, bg='black', highlightthickness=0)
        self.illum_canvas.pack(fill='x')
        self._illum_img_id = None
        self._illum_size = (0, 0)

        self.fringe_canvas = tk.Canvas(self.inner_frame, bg='black', highlightthickness=0)
        self.fringe_canvas.pack(fill='x')
        self._fringe_img_id = None
        self._fringe_size = (0, 0)

    # keep reference to PhotoImage to avoid GC
        self._photo_illum = None
        self._photo_fringe = None
        # store last display images (BGR numpy arrays) so we can rescale on window resize
        self._last_illum_bgr = None
        self._last_overlay_bgr = None
        self._resize_after_id = None

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

        # When the viewport changes size, rescale displayed images to fit width
        self.viewport.bind('<Configure>', self._on_viewport_configure)

        # start
        self.protocol('WM_DELETE_WINDOW', self.on_close)

        # --- Fringe detection controls (defaults tuned from notebook) ---
        ttk.Separator(fringe_ctrl, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(fringe_ctrl, text='Fringe detection').pack(anchor='w')
        self.bin_method = tk.StringVar(value='Otsu')
        ttk.Combobox(fringe_ctrl, textvariable=self.bin_method, values=['Otsu', 'Adaptive', 'Fixed'], width=10).pack(anchor='w')
        self.bin_thresh = tk.IntVar(value=128)
        # sliders with value readouts
        self._make_slider_row(fringe_ctrl, 'Binary thresh', self.bin_thresh, 0, 255, is_int=True)
        self.k_len = tk.IntVar(value=41)
        self._make_slider_row(fringe_ctrl, 'Kernel length', self.k_len, 5, 151, is_int=True)
        self.k_thk = tk.IntVar(value=1)
        self._make_slider_row(fringe_ctrl, 'Kernel thickness', self.k_thk, 1, 11, is_int=True)
        self.k_ang = tk.DoubleVar(value=8.0)
        self._make_slider_row(fringe_ctrl, 'Angle ±', self.k_ang, 0.0, 20.0, is_int=False, fmt="{:.1f}")
        self.k_step = tk.DoubleVar(value=2.0)
        self._make_slider_row(fringe_ctrl, 'Angle step', self.k_step, 0.5, 5.0, is_int=False, fmt="{:.2f}")
        self.k_dilate = tk.IntVar(value=1)
        self._make_slider_row(fringe_ctrl, 'Dilate px', self.k_dilate, 0, 6, is_int=True)
        self.k_area = tk.IntVar(value=50)
        self._make_slider_row(fringe_ctrl, 'Min area', self.k_area, 0, 500, is_int=True)
        self.k_skel = tk.IntVar(value=1)
        ttk.Checkbutton(fringe_ctrl, text='Skeletonize', variable=self.k_skel, command=self.on_param_change).pack(anchor='w')
        self.k_alpha = tk.DoubleVar(value=0.9)
        self._make_slider_row(fringe_ctrl, 'Overlay alpha', self.k_alpha, 0.1, 1.0, is_int=False, fmt="{:.2f}")
        self.k_bgfade = tk.DoubleVar(value=0.4)
        self._make_slider_row(fringe_ctrl, 'Background fade', self.k_bgfade, 0.0, 1.0, is_int=False, fmt="{:.2f}")
    # background target selection removed (always use 'white')

    def set_status(self, txt):
        self.status.config(text=txt)

    def load_image_dialog(self, entry_widget):
        p = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.tif;*.tiff')])
        if not p:
            return
        entry_widget.delete(0, 'end')
        entry_widget.insert(0, p)
        try:
            self.src_img = read_gray(p)
            self.set_status(f'Loaded: {os.path.basename(p)}')
            self.start_render_now()
        except Exception as e:
            messagebox.showerror('Load error', str(e))

    def load_ref_dialog(self, entry_widget):
        pass

    def apply_reference(self):
        pass

    def save_result(self):
        # prefer saving fringe overlay if available, else last_result
        to_save = None
        if hasattr(self, '_latest_overlay') and self._latest_overlay is not None:
            to_save = cv2.cvtColor(self._latest_overlay, cv2.COLOR_RGB2BGR)
        elif self.last_result is not None:
            to_save = self.last_result
        if to_save is None:
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
        # debounce using after
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
            # compute illumination-corrected images (flat, enhanced, binary)
            flat, enh, binary = pipeline_shading_sauvola(self.src_img, sigma=params[0], clip=params[1], tile=params[2], win=params[3], k=params[4], post_open=params[5])
            # store illumination outputs
            self.flat_img = flat
            self.enh_img = enh

            # prepare binary for display (from illumination pipeline)
            b = binary.astype(np.uint8)
            out = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            self.last_result = out

            # run fringe detection on enhanced image (use current fringe params)
            method = self.bin_method.get() if hasattr(self, 'bin_method') else 'Otsu'
            bw = binarize(enh, method=method, thresh=int(self.bin_thresh.get()) if hasattr(self, 'bin_thresh') else 128, blur=0)
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
            traced = skeletonize(opened.astype(bool)).astype(np.uint8) if (hasattr(self, 'k_skel') and int(self.k_skel.get())) else opened

            overlay = overlay_mask_on_gray(enh, traced, line_alpha=float(self.k_alpha.get()) if hasattr(self, 'k_alpha') else 0.9,
                                           bg_fade=float(self.k_bgfade.get()) if hasattr(self, 'k_bgfade') else 0.4,
                                           bg_to='white')

            # store latest overlay (RGB)
            self._latest_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

            # prepare illumination (enh) as BGR for display
            illum_bgr = cv2.cvtColor(enh, cv2.COLOR_GRAY2BGR) if enh.ndim == 2 else enh

            # schedule UI update on main thread to update illumination + overlay
            self.after(0, lambda: self._update_illum_and_fringe(illum_bgr, overlay))
        except Exception:
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror('Render error', 'An error occurred during rendering'))
        finally:
            try:
                self.lock.release()
            except Exception:
                pass

    def _update_illum_and_fringe(self, illum_bgr, overlay_bgr):
        # Update two stacked canvases: top (illumination), bottom (fringe overlay)
        try:
            # remember the raw display images so we can rescale on viewport resize
            try:
                # ensure copies so further changes don't mutate stored versions
                self._last_illum_bgr = illum_bgr.copy() if illum_bgr is not None else None
            except Exception:
                self._last_illum_bgr = illum_bgr
            try:
                self._last_overlay_bgr = overlay_bgr.copy() if overlay_bgr is not None else None
            except Exception:
                self._last_overlay_bgr = overlay_bgr
            # remember current vertical view so we can restore it after redraw
            try:
                cur_first, cur_last = self.viewport.yview()
            except Exception:
                cur_first, cur_last = (0.0, 1.0)
            # optionally blend original image over outputs
            orig_alpha = float(self.orig_alpha.get()) if hasattr(self, 'orig_alpha') else 0.0
            if orig_alpha > 0.0 and self.src_img is not None:
                # src_img is grayscale; convert to BGR
                src_bgr = cv2.cvtColor(self.src_img, cv2.COLOR_GRAY2BGR) if self.src_img.ndim == 2 else self.src_img
                # resize source to match illum/overlay if needed (they should be same size)
                if src_bgr.shape[:2] != illum_bgr.shape[:2]:
                    src_resized = cv2.resize(src_bgr, (illum_bgr.shape[1], illum_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    src_resized = src_bgr
                # blend for illumination display
                illum_disp = cv2.addWeighted(src_resized, orig_alpha, illum_bgr, 1.0 - orig_alpha, 0)
                # blend for fringe overlay display
                if overlay_bgr.shape[:2] != src_resized.shape[:2]:
                    src2 = cv2.resize(src_bgr, (overlay_bgr.shape[1], overlay_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                else:
                    src2 = src_resized
                fringe_disp = cv2.addWeighted(src2, orig_alpha, overlay_bgr, 1.0 - orig_alpha, 0)
            else:
                illum_disp = illum_bgr
                fringe_disp = overlay_bgr

            # Illumination: scale to fit viewport width (so images fit between left/right panels)
            try:
                vp_w = max(1, self.viewport.winfo_width())
            except Exception:
                vp_w = 800
            orig_iw = illum_disp.shape[1]
            scale_illum = min(1.0, float(vp_w) / float(orig_iw)) if orig_iw > 0 else 1.0
            photo_illum = to_photoimage_from_bgr_with_scale(illum_disp, scale=scale_illum)
            self._photo_illum = photo_illum
            iw, ih = photo_illum.width(), photo_illum.height()
            # set canvas size to match displayed image so the stacked layout is preserved
            self.illum_canvas.config(width=vp_w, height=ih)
            self._illum_size = (vp_w, ih)
            x = max(0, (vp_w - iw) // 2)
            y = 0
            if self._illum_img_id is None:
                self._illum_img_id = self.illum_canvas.create_image(x, y, anchor='nw', image=photo_illum)
            else:
                self.illum_canvas.itemconfig(self._illum_img_id, image=photo_illum)
                self.illum_canvas.coords(self._illum_img_id, x, y)

            # Fringe overlay
            # Fringe: scale to same viewport width
            orig_fw = fringe_disp.shape[1]
            scale_fringe = min(1.0, float(vp_w) / float(orig_fw)) if orig_fw > 0 else 1.0
            photo_fringe = to_photoimage_from_bgr_with_scale(fringe_disp, scale=scale_fringe)
            self._photo_fringe = photo_fringe
            fw, fh = photo_fringe.width(), photo_fringe.height()
            # set canvas size to match displayed image so it sits below the top image
            self.fringe_canvas.config(width=vp_w, height=fh)
            self._fringe_size = (vp_w, fh)
            x2 = max(0, (vp_w - fw) // 2)
            y2 = 0
            if self._fringe_img_id is None:
                self._fringe_img_id = self.fringe_canvas.create_image(x2, y2, anchor='nw', image=photo_fringe)
            else:
                self.fringe_canvas.itemconfig(self._fringe_img_id, image=photo_fringe)
                self.fringe_canvas.coords(self._fringe_img_id, x2, y2)
            # ensure inner_frame width matches vp_w so canvases align
            try:
                self.inner_frame.config(width=vp_w)
            except Exception:
                pass

            # update latest overlay (store as RGB for saving)
            try:
                # fringe_disp may be BGR; convert to RGB
                self._latest_overlay = cv2.cvtColor(fringe_disp, cv2.COLOR_BGR2RGB)
            except Exception:
                self._latest_overlay = fringe_disp

            self.set_status('Rendered')
        except Exception:
            traceback.print_exc()
        finally:
            # After updating images, update viewport scrollregion and slider range
            try:
                self.viewport.update_idletasks()
                bbox = self.viewport.bbox('all')
                if bbox is None:
                    return
                inner_w = bbox[2] - bbox[0]
                inner_h = bbox[3] - bbox[1]
                vp_w = self.viewport.winfo_width()
                vp_h = self.viewport.winfo_height()
                # update scrollregion explicitly
                self.viewport.configure(scrollregion=(0, 0, inner_w, inner_h))
                max_scroll = max(0, inner_h - vp_h)
                # set the slider range to pixel scrollable amount (so value maps to pixels)
                try:
                    self.img_scroll.config(to=float(max_scroll))
                except Exception:
                    pass
                # restore previous yview (clamped) so sliders don't reset view to top
                try:
                    # compute new first based on previous fraction
                    first, last = self.viewport.yview()
                    # prefer to restore cur_first but clamp to available range
                    new_first = max(0.0, min(cur_first, first))
                    self.viewport.yview_moveto(new_first)
                    to_val = float(self.img_scroll['to']) if 'to' in self.img_scroll.keys() else 0.0
                    if to_val > 0:
                        self.img_scroll_var.set(new_first * to_val)
                except Exception:
                    pass
            except Exception:
                pass

    def _on_canvas_resize(self, which, width, height):
        # store canvas size and reposition image if present
        try:
            if which == 'illum':
                self._illum_size = (width, height)
                if getattr(self, '_photo_illum', None) is not None and getattr(self, '_illum_img_id', None) is not None:
                    w, h = self._photo_illum.width(), self._photo_illum.height()
                    x = max(0, (width - w) // 2)
                    y = max(0, (height - h) // 2)
                    self.illum_canvas.coords(self._illum_img_id, x, y)
            elif which == 'fringe':
                self._fringe_size = (width, height)
                if getattr(self, '_photo_fringe', None) is not None and getattr(self, '_fringe_img_id', None) is not None:
                    w, h = self._photo_fringe.width(), self._photo_fringe.height()
                    x = max(0, (width - w) // 2)
                    y = max(0, (height - h) // 2)
                    self.fringe_canvas.coords(self._fringe_img_id, x, y)
        except Exception:
            pass

    def _on_img_scroll(self, v):
        # v is a string from ttk.Scale; map 0..1 to viewport yview
        try:
            fv = float(v)
            # compute the fractional shift. ttk.Scale returns value in the scale's from_/to_ bounds
            # we kept from_=0.0; to_ will be set to max_scroll later. Normalize using current to_
            to_val = float(self.img_scroll['to']) if 'to' in self.img_scroll.keys() else 0.0
            if to_val <= 0:
                return
            frac = fv / to_val
            frac = max(0.0, min(1.0, frac))
            self.viewport.yview_moveto(frac)
        except Exception:
            pass

    def _on_mousewheel(self, event):
        try:
            # Windows: event.delta is multiple of 120
            delta = int(event.delta / 120) if hasattr(event, 'delta') else 0
            # scroll by 3 lines per wheel notch
            amount = -delta * 3
            self.viewport.yview_scroll(amount, 'units')
            # sync slider
            first, last = self.viewport.yview()
            to_val = float(self.img_scroll['to']) if 'to' in self.img_scroll.keys() else 0.0
            if to_val > 0:
                self.img_scroll_var.set(first * to_val)
        except Exception:
            pass

    def _on_viewport_configure(self, event=None):
        # Debounced handler: when viewport size changes, rescale displayed images
        try:
            if self._resize_after_id:
                try:
                    self.after_cancel(self._resize_after_id)
                except Exception:
                    pass
            self._resize_after_id = self.after(120, self._rescale_display_images)
        except Exception:
            pass

    def _rescale_display_images(self):
        # Rescale stored BGR images to viewport width and update canvases
        try:
            self._resize_after_id = None
            if self._last_illum_bgr is None and self._last_overlay_bgr is None:
                return
            try:
                vp_w = max(1, self.viewport.winfo_width())
            except Exception:
                vp_w = 800

            # Rescale and replace illumination display
            if self._last_illum_bgr is not None:
                illum_bgr = self._last_illum_bgr
                scale_illum = min(1.0, float(vp_w) / float(illum_bgr.shape[1])) if illum_bgr.shape[1] > 0 else 1.0
                photo_illum = to_photoimage_from_bgr_with_scale(illum_bgr, scale=scale_illum)
                self._photo_illum = photo_illum
                iw, ih = photo_illum.width(), photo_illum.height()
                self.illum_canvas.config(width=vp_w, height=ih)
                x = max(0, (vp_w - iw) // 2)
                if self._illum_img_id is None:
                    self._illum_img_id = self.illum_canvas.create_image(x, 0, anchor='nw', image=photo_illum)
                else:
                    self.illum_canvas.itemconfig(self._illum_img_id, image=photo_illum)
                    self.illum_canvas.coords(self._illum_img_id, x, 0)

            # Rescale and replace fringe display
            if self._last_overlay_bgr is not None:
                fringe_bgr = self._last_overlay_bgr
                scale_fringe = min(1.0, float(vp_w) / float(fringe_bgr.shape[1])) if fringe_bgr.shape[1] > 0 else 1.0
                photo_fringe = to_photoimage_from_bgr_with_scale(fringe_bgr, scale=scale_fringe)
                self._photo_fringe = photo_fringe
                fw, fh = photo_fringe.width(), photo_fringe.height()
                self.fringe_canvas.config(width=vp_w, height=fh)
                x2 = max(0, (vp_w - fw) // 2)
                if self._fringe_img_id is None:
                    self._fringe_img_id = self.fringe_canvas.create_image(x2, 0, anchor='nw', image=photo_fringe)
                else:
                    self.fringe_canvas.itemconfig(self._fringe_img_id, image=photo_fringe)
                    self.fringe_canvas.coords(self._fringe_img_id, x2, 0)

            try:
                self.inner_frame.config(width=vp_w)
            except Exception:
                pass
        except Exception:
            pass

    # UI scale removed: images auto-fit the available width

    def on_close(self):
        self.destroy()


def main():
    app = EvenApp()
    app.mainloop()


if __name__ == '__main__':
    main()
