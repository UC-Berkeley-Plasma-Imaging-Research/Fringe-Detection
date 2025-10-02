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


# --- Image processing pipeline (same logic as notebook) ---


def pipeline_shading_sauvola(img_gray, sigma=35.0, clip=2.5, tile=8, win=31, k=0.20, post_open=1):
    """Return (flat, enh, binary) where binary is uint8 {0,255}."""
    bg = cv2.GaussianBlur(img_gray, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    flat = cv2.divide(img_gray, bg, scale=255)
    tile = max(2, int(tile))
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
    enh = clahe.apply(flat)
    win = int(win) if int(win) % 2 == 1 else int(win) + 1
    thv = threshold_sauvola(enh, window_size=win, k=float(k))
    binary = (enh > thv).astype(np.uint8) * 255
    if post_open > 0:
        ksz = max(1, int(post_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((ksz, ksz), np.uint8))
    return flat, enh, binary


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def to_photoimage_from_bgr(bgr):
    # Convert BGR (OpenCV) to PIL PhotoImage for tkinter
    if bgr.ndim == 2:
        img = Image.fromarray(bgr)
    else:
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return ImageTk.PhotoImage(img)


class EvenApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Even Illumination - Fringe Extraction')
        self.geometry('1100x700')

        # state
        self.src_img = None
        self.ref_img = None
        self.last_result = None
        self.lock = threading.Lock()
        self._after_id = None

        # controls frame
        ctrl = ttk.Frame(self)
        ctrl.pack(side='left', fill='y', padx=8, pady=8)

        ttk.Label(ctrl, text='Open image:').pack(anchor='w')
        path_entry = ttk.Entry(ctrl, width=30)
        path_entry.pack(anchor='w')
        ttk.Button(ctrl, text='Browse & Load', command=lambda: self.load_image_dialog(path_entry)).pack(anchor='w', pady=4)
        ttk.Button(ctrl, text='Save result', command=self.save_result).pack(anchor='w', pady=4)

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)

        # sliders
        self.sigma_var = tk.DoubleVar(value=50.0)
        self.clip_var = tk.DoubleVar(value=2.5)
        self.tile_var = tk.IntVar(value=8)
        self.win_var = tk.IntVar(value=31)
        self.k_var = tk.DoubleVar(value=0.20)
        self.post_var = tk.IntVar(value=1)

        ttk.Label(ctrl, text='Blur σ').pack(anchor='w')
        s1 = ttk.Scale(ctrl, from_=0, to=200, variable=self.sigma_var, command=self.on_param_change)
        s1.pack(fill='x')
        ttk.Label(ctrl, text='CLAHE clip').pack(anchor='w')
        s2 = ttk.Scale(ctrl, from_=1.0, to=8.0, variable=self.clip_var, command=self.on_param_change)
        s2.pack(fill='x')
        ttk.Label(ctrl, text='CLAHE tile').pack(anchor='w')
        s3 = ttk.Scale(ctrl, from_=2, to=64, variable=self.tile_var, command=self.on_param_change)
        s3.pack(fill='x')
        ttk.Label(ctrl, text='Sauvola win').pack(anchor='w')
        s4 = ttk.Scale(ctrl, from_=3, to=201, variable=self.win_var, command=self.on_param_change)
        s4.pack(fill='x')
        ttk.Label(ctrl, text='Sauvola k').pack(anchor='w')
        s5 = ttk.Scale(ctrl, from_=0.0, to=1.0, variable=self.k_var, command=self.on_param_change)
        s5.pack(fill='x')
        ttk.Label(ctrl, text='Post open').pack(anchor='w')
        s6 = ttk.Scale(ctrl, from_=0, to=20, variable=self.post_var, command=self.on_param_change)
        s6.pack(fill='x')

        ttk.Button(ctrl, text='Render now', command=self.start_render_now).pack(pady=8)
        self.status = ttk.Label(ctrl, text='Ready', wraplength=220)
        self.status.pack(pady=6)

        ttk.Separator(ctrl, orient='horizontal').pack(fill='x', pady=6)
        ttk.Label(ctrl, text='Reference image').pack(anchor='w')
        ref_entry = ttk.Entry(ctrl, width=30)
        ref_entry.pack(anchor='w')
        ttk.Button(ctrl, text='Browse Reference', command=lambda: self.load_ref_dialog(ref_entry)).pack(anchor='w', pady=4)
        ttk.Button(ctrl, text='Apply reference as input', command=self.apply_reference).pack(anchor='w', pady=4)

        # image column
        img_frame = ttk.Frame(self)
        img_frame.pack(side='right', fill='both', expand=True, padx=8, pady=8)

        self.ref_label = ttk.Label(img_frame)
        self.ref_label.pack()
        ttk.Label(img_frame, text='Reference').pack()
        ttk.Separator(img_frame, orient='horizontal').pack(fill='x', pady=6)

        self.img_label = ttk.Label(img_frame)
        self.img_label.pack(fill='both', expand=True)

        # keep reference to PhotoImage to avoid GC
        self._photo_ref = None
        self._photo_main = None

        # start
        self.protocol('WM_DELETE_WINDOW', self.on_close)

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
        p = filedialog.askopenfilename(filetypes=[('Images', '*.png;*.jpg;*.jpeg;*.tif;*.tiff')])
        if not p:
            return
        entry_widget.delete(0, 'end')
        entry_widget.insert(0, p)
        try:
            self.ref_img = read_gray(p)
            # preview
            ref_bgr = cv2.cvtColor(self.ref_img, cv2.COLOR_GRAY2BGR) if self.ref_img.ndim == 2 else self.ref_img
            self._photo_ref = to_photoimage_from_bgr(ref_bgr)
            self.ref_label.config(image=self._photo_ref)
            self.set_status(f'Reference loaded: {os.path.basename(p)}')
        except Exception as e:
            messagebox.showerror('Reference load error', str(e))

    def apply_reference(self):
        if self.ref_img is None:
            self.set_status('No reference loaded')
            return
        self.src_img = self.ref_img.copy()
        self.set_status('Reference applied as input')
        self.start_render_now()

    def save_result(self):
        if self.last_result is None:
            messagebox.showinfo('No result', 'No result to save')
            return
        p = filedialog.asksaveasfilename(defaultextension='.png', filetypes=[('PNG', '*.png')])
        if not p:
            return
        try:
            cv2.imwrite(p, self.last_result)
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
            _, _, binary = pipeline_shading_sauvola(self.src_img, sigma=params[0], clip=params[1], tile=params[2], win=params[3], k=params[4], post_open=params[5])
            b = binary.astype(np.uint8)
            out = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
            self.last_result = out
            # schedule UI update on main thread
            self.after(0, lambda: self._update_image(out))
        except Exception:
            traceback.print_exc()
            self.after(0, lambda: messagebox.showerror('Render error', 'An error occurred during rendering'))
        finally:
            try:
                self.lock.release()
            except Exception:
                pass

    def _update_image(self, out_bgr):
        self._photo_main = to_photoimage_from_bgr(out_bgr)
        self.img_label.config(image=self._photo_main)
        self.set_status('Rendered')

    def on_close(self):
        self.destroy()


def main():
    app = EvenApp()
    app.mainloop()


if __name__ == '__main__':
    main()
