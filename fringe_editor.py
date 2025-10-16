"""
Simple standalone Fringe Mask Editor

Features:
- Load a binary image (0 = black, 255 = white)
- Paint to add black pixels or remove black pixels (paint white)
- Adjustable brush size (Ctrl + Mouse Wheel shortcut)
- Undo per stroke
- Scroll to zoom at cursor position
- Persistent brush-size circle overlay around the cursor
- Optional background underlay with brightness control (white in mask is transparent)
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps


class FringeEditorFrame(tk.Frame):
    """
    Embeddable fringe editor as a tkinter Frame.
    - on_apply(mask, background) callback will be invoked when user clicks 'Apply to App'
    - on_close() callback will be invoked when user clicks 'Close' (when embedded)
    """
    def __init__(self, master=None, on_apply=None, on_close=None):
        super().__init__(master)
        self._on_apply = on_apply
        self._on_close = on_close
        self._own_root = None  # set when launched standalone via main()

        # Data
        self.mask = None  # uint8 (0 or 255), shape (H, W)
        self._bg = None   # optional grayscale background, same shape as mask
        self._undo_stack = []  # list of mask snapshots
        self._photo = None

        # View/zoom state
        self._base_scale = 1.0  # fit-to-window scale
        self._zoom = 1.0        # user zoom multiplier
        self._scale = 1.0       # effective scale = base_scale * zoom
        self._offset = None     # (x0, y0) of image top-left on canvas

        # Toolbar
        toolbar = ttk.Frame(self)
        toolbar.pack(side="top", fill="x")

        ttk.Button(toolbar, text="Open Binary", command=self.open_binary).pack(side="left", padx=4, pady=4)
        ttk.Button(toolbar, text="Open Background", command=self.open_background).pack(side="left", padx=4, pady=4)
        ttk.Button(toolbar, text="Save As…", command=self.save_binary).pack(side="left", padx=4, pady=4)
        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)

        self.mode_var = tk.StringVar(value="add")
        ttk.Radiobutton(toolbar, text="Add Black", value="add", variable=self.mode_var).pack(side="left", padx=4)
        ttk.Radiobutton(toolbar, text="Remove Black", value="erase", variable=self.mode_var).pack(side="left", padx=4)

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        # Brush size is controlled via Ctrl+MouseWheel only; keep an internal variable
        # (no visible slider per user request)
        self.brush_var = tk.DoubleVar(value=10.0)

        # Background brightness control
        ttk.Label(toolbar, text="BG bright").pack(side="left", padx=(8, 2))
        self.bg_brightness = tk.DoubleVar(value=1.0)
        self.bg_scale = ttk.Scale(
            toolbar,
            from_=0.0,
            to=1.0,
            orient="horizontal",
            variable=self.bg_brightness,
            command=self._on_bg_brightness_changed,
        )
        self.bg_scale.pack(side="left", padx=4)

        # Blue overlay toggle for better visibility of black regions
        self.red_overlay_var = tk.BooleanVar(value=False)
        self.red_overlay_btn = ttk.Checkbutton(
            toolbar,
            text="Blue overlay",
            variable=self.red_overlay_var,
            command=self._on_red_overlay_changed,
        )
        self.red_overlay_btn.pack(side="left", padx=(8, 4))

        ttk.Separator(toolbar, orient="vertical").pack(side="left", fill="y", padx=6)
        ttk.Button(toolbar, text="Connect gaps", command=self._connect_gaps).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Undo", command=self.undo).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Apply to App", command=self._handle_apply).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Close", command=self._handle_close).pack(side="left", padx=4)

        # Status bar
        self.status = ttk.Label(self, text="Open a binary image (0/255) to edit…", anchor="w")
        self.status.pack(side="bottom", fill="x")

        # Canvas
        self.canvas = tk.Canvas(self, bg="gray20", highlightthickness=0)
        self.canvas.pack(side="top", fill="both", expand=True)

        # Bindings
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Button-1>", self._on_paint_start)
        self.canvas.bind("<B1-Motion>", self._on_paint_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_paint_end)
        self.canvas.bind("<Motion>", self._on_mouse_move)
        self.canvas.bind("<Leave>", self._on_mouse_leave)
        self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
        self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mouse_wheel)
        # Right-click drag to pan
        self.canvas.bind("<Button-3>", self._on_pan_start)
        self.canvas.bind("<B3-Motion>", self._on_pan_move)
        self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)

        # Bind to toplevel for shortcuts
        self._bind_to_toplevel("<Control-z>", lambda e: self.undo())
        self._bind_to_toplevel("<Key-a>", lambda e: self._set_mode("add"))
        self._bind_to_toplevel("<Key-e>", lambda e: self._set_mode("erase"))

        # Stroke and cursor overlay state
        self._painting = False
        self._cursor_circle_id = None
        self._last_mouse_pos = None
        # Pan state
        self._panning = False
        self._pan_start = (0, 0)
        self._offset_start = (0, 0)

    # -------------- Public API --------------
    def set_data(self, mask: np.ndarray, background: np.ndarray | None = None):
        """Set editor data from external app. Arrays should be uint8.
        mask: 0 = black line, 255 = white background
        background: grayscale image matching mask shape, optional
        """
        if mask is None:
            return
        try:
            m = mask.copy()
        except Exception:
            m = mask
        if m.dtype != np.uint8:
            m = m.astype(np.uint8)
        self.mask = m
        if background is not None:
            try:
                b = background.copy()
            except Exception:
                b = background
            if b.dtype != np.uint8:
                b = b.astype(np.uint8)
            if b.shape != self.mask.shape:
                try:
                    b = cv2.resize(b, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                except Exception:
                    b = None
            self._bg = b
        self._undo_stack.clear()
        self._zoom = 1.0
        self._base_scale = 1.0
        self._offset = None
        self._refresh_display(force_recompute_base=True)

    def get_mask(self) -> np.ndarray | None:
        return None if self.mask is None else self.mask.copy()

    # -------------- Internal helpers --------------
    def _bind_to_toplevel(self, sequence, func):
        try:
            tl = self.winfo_toplevel()
            tl.bind(sequence, func)
        except Exception:
            try:
                self.bind(sequence, func)
            except Exception:
                pass

    def _handle_apply(self):
        if callable(self._on_apply) and self.mask is not None:
            try:
                self._on_apply(self.mask.copy(), None if self._bg is None else self._bg.copy())
                self.set_status("Applied to app")
            except Exception:
                messagebox.showerror("Apply error", "Failed to apply edited mask to the app")

    def _handle_close(self):
        if callable(self._on_close):
            try:
                self._on_close()
                return
            except Exception:
                pass
        # Standalone: close the root window if we own it
        try:
            if self._own_root is not None:
                self._own_root.destroy()
            else:
                # Fallback: destroy just this frame
                self.destroy()
        except Exception:
            pass

    # -------------- UI helpers --------------
    def _set_mode(self, mode):
        if mode in ("add", "erase"):
            self.mode_var.set(mode)

    def set_status(self, text):
        try:
            self.status.config(text=text)
        except Exception:
            pass

    # -------------- File actions --------------
    def open_binary(self):
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title="Open binary image",
            filetypes=[("Images", ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"))],
            initialdir=os.path.expanduser("~"),
        )
        if not path:
            return
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Open error", "Failed to read the image")
            return
        # Normalize to strict binary 0/255
        mask = np.where(img <= 127, 0, 255).astype(np.uint8)
        self.mask = mask
        self._undo_stack.clear()
        # If a background is loaded and size mismatched, resize bg to match mask
        if self._bg is not None:
            try:
                if self._bg.shape != self.mask.shape:
                    self._bg = cv2.resize(self._bg, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            except Exception:
                self._bg = None
        self.set_status(f"Loaded: {os.path.basename(path)} — {mask.shape[1]}×{mask.shape[0]}")
        # Reset view
        self._zoom = 1.0
        self._base_scale = 1.0
        self._offset = None
        self._refresh_display(force_recompute_base=True)

    def _on_red_overlay_changed(self):
        self._refresh_display(False)

    def open_background(self):
        path = filedialog.askopenfilename(
            parent=self.winfo_toplevel(),
            title="Open background (grayscale)",
            filetypes=[("Images", ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"))],
            initialdir=os.path.expanduser("~"),
        )
        if not path:
            return
        bg = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if bg is None:
            messagebox.showerror("Open error", "Failed to read the background image")
            return
        # If no mask loaded yet, start a new white mask matching background size
        if self.mask is None:
            self.mask = np.full(bg.shape, 255, dtype=np.uint8)
            self._undo_stack.clear()
            self._zoom = 1.0
            self._base_scale = 1.0
            self._offset = None
        # Resize background to match mask if needed
        if bg.shape != self.mask.shape:
            try:
                bg = cv2.resize(bg, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            except Exception:
                messagebox.showerror("Background error", "Failed to resize background to match mask size")
                return
        self._bg = bg.astype(np.uint8)
        self.set_status(f"Background loaded: {os.path.basename(path)} — brightness {self.bg_brightness.get():.2f}")
        self._refresh_display(force_recompute_base=True)

    def _on_bg_brightness_changed(self, value=None):
        # Re-render preview when background brightness changes
        self._refresh_display(False)

    def save_binary(self):
        if self.mask is None:
            messagebox.showinfo("Nothing to save", "Load an image first")
            return
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Save edited mask",
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("TIFF", "*.tif"), ("JPEG", "*.jpg")],
        )
        if not path:
            return
        try:
            ok = cv2.imwrite(path, self.mask)
            if not ok:
                raise RuntimeError("cv2.imwrite returned False")
            self.set_status(f"Saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Save error", str(e))

    def undo(self):
        if not self._undo_stack:
            return
        try:
            self.mask = self._undo_stack.pop()
            self._refresh_display()
            self.set_status("Undid last stroke")
        except Exception:
            pass

    def _connect_gaps(self):
        """Bridge small gaps in the binary mask using morphological closing.
        Operates on the in-editor mask (0=black fringe, 255=white background).
        """
        if self.mask is None:
            return
        try:
            # Save for undo
            self._undo_stack.append(self.mask.copy())
            if len(self._undo_stack) > 20:
                self._undo_stack = self._undo_stack[-20:]

            # Convert to 0/1 with 1 indicating a fringe pixel
            bw = (self.mask == 0).astype(np.uint8)

            # Close tiny gaps (diagonal and 1px breaks)
            k_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_small, iterations=1)

            # Bridge short horizontal gaps (fringes are often horizontal)
            k_h = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_h, iterations=1)

            # Bridge short diagonal gaps using custom 5x5 diagonal kernels
            k_d1 = np.eye(5, dtype=np.uint8)
            k_d2 = np.fliplr(k_d1)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_d1, iterations=1)
            bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k_d2, iterations=1)

            # Convert back to 0/255 mask: 1 (fringe) -> 0 black, 0 -> 255 white
            self.mask = np.where(bw > 0, 0, 255).astype(np.uint8)
            self._refresh_display()
            self.set_status("Connected gaps (3x3 + horiz 7 + diagonals)")
        except Exception:
            messagebox.showerror("Connect gaps", "Failed to connect gaps on the current mask")

    # -------------- Rendering & transforms --------------
    def _on_canvas_configure(self, event=None):
        # Recompute base scale on resize only when zoom is 1.0 (fit mode)
        self._refresh_display(force_recompute_base=(self._zoom == 1.0))

    def _refresh_display(self, force_recompute_base=False):
        if self.mask is None:
            self.canvas.delete("all")
            self._cursor_circle_id = None
            return
        h, w = self.mask.shape[:2]
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        if force_recompute_base or self._base_scale <= 0:
            self._base_scale = max(1e-6, min(cw / w, ch / h))
        self._scale = float(self._base_scale * self._zoom)
        disp_w = max(1, int(round(w * self._scale)))
        disp_h = max(1, int(round(h * self._scale)))

        # Preserve thin lines when zoomed out by thickening them before resize (preview only)
        preview_mask = self.mask
        try:
            if self._scale < 1.0:
                import math
                k = int(min(12, max(1, math.ceil(1.0 / max(1e-6, self._scale)))))
                if k > 1:
                    kernel = np.ones((k, k), np.uint8)
                    inv = 255 - preview_mask
                    inv = cv2.dilate(inv, kernel, iterations=1)
                    preview_mask = 255 - inv
        except Exception:
            preview_mask = self.mask

        # Determine visible region on canvas and corresponding image crop
        if self._offset is None:
            x0 = (cw - disp_w) // 2
            y0 = (ch - disp_h) // 2
            self._offset = (x0, y0)
        x0, y0 = self._offset
        vx0 = max(0, x0)
        vy0 = max(0, y0)
        vx1 = min(cw, x0 + disp_w)
        vy1 = min(ch, y0 + disp_h)

        self.canvas.delete("all")
        self._cursor_circle_id = None

        if vx1 <= vx0 or vy1 <= vy0:
            self._img_topleft = (x0, y0)
            return

        # Source crop in image coordinates
        scale = self._scale if self._scale > 0 else 1.0
        import math
        ix0 = int(max(0, math.floor((vx0 - x0) / scale)))
        iy0 = int(max(0, math.floor((vy0 - y0) / scale)))
        ix1 = int(min(w, math.ceil((vx1 - x0) / scale)))
        iy1 = int(min(h, math.ceil((vy1 - y0) / scale)))

        if ix1 <= ix0 or iy1 <= iy0:
            self._img_topleft = (x0, y0)
            return

        # Destination size on canvas
        dst_w = int(vx1 - vx0)
        dst_h = int(vy1 - vy0)

        # Build preview for the cropped region
        region_mask = preview_mask[iy0:iy1, ix0:ix1]
        if self._bg is not None:
            try:
                bg_region = self._bg[iy0:iy1, ix0:ix1]
                bg_img = Image.fromarray(bg_region, mode="L").resize((dst_w, dst_h), Image.BILINEAR)
            except Exception:
                bg_img = None
            mask_img = Image.fromarray(region_mask, mode="L").resize((dst_w, dst_h), Image.NEAREST)
            if bg_img is not None:
                b = float(self.bg_brightness.get()) if self.bg_brightness is not None else 1.0
                b = max(0.0, min(1.0, b))
                white_L = Image.new("L", (dst_w, dst_h), 255)
                blended = Image.blend(white_L, bg_img, b)
                if bool(self.red_overlay_var.get()):
                    result = blended.convert("RGB")
                    blue_img = Image.new("RGB", (dst_w, dst_h), (0, 200, 255))
                    mask_inv = ImageOps.invert(mask_img)
                    result.paste(blue_img, (0, 0), mask_inv)
                    pil_img = result
                else:
                    result = blended.copy()
                    black_L = Image.new("L", (dst_w, dst_h), 0)
                    mask_inv = ImageOps.invert(mask_img)
                    result.paste(black_L, (0, 0), mask_inv)
                    pil_img = result
            else:
                pil_img = mask_img
        else:
            mask_img = Image.fromarray(region_mask, mode="L").resize((dst_w, dst_h), Image.NEAREST)
            if bool(self.red_overlay_var.get()):
                base = Image.new("RGB", (dst_w, dst_h), (255, 255, 255))
                blue_img = Image.new("RGB", (dst_w, dst_h), (0, 200, 255))
                mask_inv = ImageOps.invert(mask_img)
                base.paste(blue_img, (0, 0), mask_inv)
                pil_img = base
            else:
                pil_img = mask_img

        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(vx0, vy0, anchor="nw", image=self._photo)
        self._img_topleft = (x0, y0)

        # Redraw cursor circle if we know the last mouse position
        if self._last_mouse_pos is not None:
            try:
                self._update_cursor_circle(self._last_mouse_pos[0], self._last_mouse_pos[1])
            except Exception:
                pass

    def _canvas_to_image_coords(self, x_canvas, y_canvas):
        if self.mask is None:
            return None
        x0, y0 = getattr(self, "_img_topleft", (0, 0))
        scale = self._scale if self._scale > 0 else 1.0
        xi = int((x_canvas - x0) / scale)
        yi = int((y_canvas - y0) / scale)
        h, w = self.mask.shape[:2]
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            return None
        return xi, yi

    # -------------- Painting --------------
    def _apply_brush(self, xi, yi):
        if self.mask is None:
            return
        # Use a float brush radius (no integer rounding) by generating a distance mask for the affected region
        r = float(max(0.1, self.brush_var.get()))
        color = 0 if self.mode_var.get() == "add" else 255
        h, w = self.mask.shape[:2]
        # integer bounds for the affected box
        y_min = int(max(0, np.floor(yi - r)))
        y_max = int(min(h - 1, np.ceil(yi + r)))
        x_min = int(max(0, np.floor(xi - r)))
        x_max = int(min(w - 1, np.ceil(xi + r)))
        yy, xx = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
        # compute boolean mask using float radius
        circle = (xx - xi) ** 2 + (yy - yi) ** 2 <= (r * r)
        sub = self.mask[y_min:y_max + 1, x_min:x_max + 1]
        sub[circle] = color

    def _on_paint_start(self, event):
        if self.mask is None:
            return
        try:
            self._undo_stack.append(self.mask.copy())
            if len(self._undo_stack) > 20:
                self._undo_stack = self._undo_stack[-20:]
        except Exception:
            pass

        pt = self._canvas_to_image_coords(event.x, event.y)
        if pt is not None:
            xi, yi = pt
            self._apply_brush(xi, yi)
            self._refresh_display()
            self._painting = True
        self._last_mouse_pos = (event.x, event.y)
        self._update_cursor_circle(event.x, event.y)

    def _on_paint_move(self, event):
        if self.mask is None or not self._painting:
            return
        pt = self._canvas_to_image_coords(event.x, event.y)
        if pt is not None:
            xi, yi = pt
            self._apply_brush(xi, yi)
            self._refresh_display()
        self._last_mouse_pos = (event.x, event.y)
        self._update_cursor_circle(event.x, event.y)

    def _on_paint_end(self, event):
        self._painting = False

    # -------------- Right-click panning --------------
    def _on_pan_start(self, event):
        self._panning = True
        self._pan_start = (event.x, event.y)
        self._offset_start = self._offset if self._offset is not None else getattr(self, "_img_topleft", (0, 0))
        try:
            self.config(cursor='fleur')
        except Exception:
            pass

    def _on_pan_move(self, event):
        if not self._panning:
            return
        sx, sy = self._pan_start
        dx = event.x - sx
        dy = event.y - sy
        ox0, oy0 = self._offset_start
        self._offset = (ox0 + dx, oy0 + dy)
        # Redraw at new offset
        self._refresh_display(False)
        # Keep cursor circle at current mouse
        self._last_mouse_pos = (event.x, event.y)
        self._update_cursor_circle(event.x, event.y)

    def _on_pan_end(self, event):
        self._panning = False
        try:
            self.config(cursor='')
        except Exception:
            pass

    # -------------- Cursor overlay --------------
    def _on_mouse_move(self, event):
        self._last_mouse_pos = (event.x, event.y)
        self._update_cursor_circle(event.x, event.y)

    def _on_mouse_leave(self, event):
        if self._cursor_circle_id is not None:
            try:
                self.canvas.delete(self._cursor_circle_id)
            except Exception:
                pass
            self._cursor_circle_id = None
        self._last_mouse_pos = None

    def _update_cursor_circle(self, x, y):
        if self.mask is None:
            if self._cursor_circle_id is not None:
                try:
                    self.canvas.delete(self._cursor_circle_id)
                except Exception:
                    pass
                self._cursor_circle_id = None
            return
        scale = self._scale if self._scale > 0 else 1.0
        # Use float brush size for the on-canvas cursor circle so it matches brush_var precisely
        r_img = float(max(0.1, self.brush_var.get()))
        r_canvas = max(1.0, (r_img * scale))
        x0 = x - r_canvas
        y0 = y - r_canvas
        x1 = x + r_canvas
        y1 = y + r_canvas
        color = "#00ff88" if self.mode_var.get() == "add" else "#ff5555"
        if self._cursor_circle_id is None:
            try:
                # canvas supports float coordinates
                self._cursor_circle_id = self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=1)
            except Exception:
                self._cursor_circle_id = None
        else:
            try:
                self.canvas.coords(self._cursor_circle_id, x0, y0, x1, y1)
                self.canvas.itemconfig(self._cursor_circle_id, outline=color)
            except Exception:
                try:
                    self._cursor_circle_id = self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=1)
                except Exception:
                    self._cursor_circle_id = None

    # -------------- Zoom and brush size via wheel --------------
    def _on_mouse_wheel(self, event):
        # Ignore here if Control is pressed (handled by _on_ctrl_mouse_wheel)
        try:
            if getattr(event, "state", 0) & 0x4:
                return
        except Exception:
            pass
        if self.mask is None:
            return
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        zoom_factor = 1.1 if delta > 0 else 0.9
        old_zoom = self._zoom
        new_zoom = max(0.1, min(16.0, old_zoom * zoom_factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        x_c = event.x
        y_c = event.y
        s_before = max(1e-6, self._base_scale * old_zoom)
        s_after = max(1e-6, self._base_scale * new_zoom)
        ox, oy = self._offset if self._offset is not None else getattr(self, "_img_topleft", (0, 0))
        x_img = (x_c - ox) / s_before
        y_img = (y_c - oy) / s_before
        self._zoom = new_zoom
        new_ox = int(round(x_c - x_img * s_after))
        new_oy = int(round(y_c - y_img * s_after))
        self._offset = (new_ox, new_oy)
        self._refresh_display(False)
        self._last_mouse_pos = (x_c, y_c)
        self._update_cursor_circle(x_c, y_c)

    def _on_ctrl_mouse_wheel(self, event):
        delta = int(getattr(event, "delta", 0))
        step = 0.5 if delta > 0 else -0.5
        new_size = float(self.brush_var.get()) + step
        new_size = max(0.1, min(200.0, new_size))
        self.brush_var.set(new_size)
        self._last_mouse_pos = (event.x, event.y)
        self._update_cursor_circle(event.x, event.y)


def main():
    root = tk.Tk()
    root.title("Fringe Mask Editor (MVP)")
    root.geometry("1000x700")
    editor = FringeEditorFrame(root)
    editor._own_root = root
    editor.pack(fill="both", expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
