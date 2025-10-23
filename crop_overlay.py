import os
import tkinter as tk
from tkinter import filedialog


class CropMixin:
    # ---- scale helpers for illumination (top image) ----
    def _illum_scale(self):
        try:
            if getattr(self, '_photo_illum', None) is not None and getattr(self, '_last_illum_bgr', None) is not None:
                w_img = int(self._last_illum_bgr.shape[1])
                w_disp = int(self._photo_illum.width())
                if w_img > 0:
                    return float(w_disp) / float(w_img)
        except Exception:
            pass
        return 1.0

    # ---- crop lifecycle ----
    def _start_crop_mode(self):
        if getattr(self, 'src_img', None) is None:
            self.set_status('Load an image first')
            return
        w, h = self._img_size()
        if w <= 0 or h <= 0:
            self.set_status('Nothing to crop')
            return
        # default rect to centered half
        x = max(0, w // 4)
        y = max(0, h // 4)
        rw = max(1, w // 2)
        rh = max(1, h // 2)
        self._crop_rect = [x, y, rw, rh]
        self._crop_mode = True
        self._sync_fields_from_rect()
        # bind events on illum (top) canvas
        try:
            self.illum_canvas.bind('<Button-1>', self._on_crop_press)
            self.illum_canvas.bind('<B1-Motion>', self._on_crop_drag)
            self.illum_canvas.bind('<ButtonRelease-1>', self._on_crop_release)
        except Exception:
            pass
        self._redraw_crop_overlay()
        self.set_status('Crop mode: drag inside to move, drag corners to resize; edit X/Y/W/H to set exact')

    def _cancel_crop_mode(self):
        self._crop_mode = False
        self._crop_rect = None
        self._crop_drag = None
        self._crop_last = None
        # unbind
        try:
            self.illum_canvas.unbind('<Button-1>')
            self.illum_canvas.unbind('<B1-Motion>')
            self.illum_canvas.unbind('<ButtonRelease-1>')
        except Exception:
            pass
        self._clear_crop_overlay()
        self.set_status('Crop canceled')

    def _apply_crop(self):
        if not self._crop_mode or self._crop_rect is None or getattr(self, 'src_img', None) is None:
            self.set_status('No crop to apply')
            return
        x, y, w, h = self._crop_rect
        iw, ih = self._img_size()
        # clamp
        x = max(0, min(iw - 1, int(x)))
        y = max(0, min(ih - 1, int(y)))
        w = max(1, min(iw - x, int(w)))
        h = max(1, min(ih - y, int(h)))
        try:
            cropped = self.src_img[y:y+h, x:x+w]
            if cropped is None or cropped.size == 0:
                self.set_status('Invalid crop area')
                return
            self.src_img = cropped.copy()
        except Exception:
            self.set_status('Crop failed')
            return
        # reset view state
        self._cancel_crop_mode()
        self._zoom_level = 1.0
        # recompute pipeline
        self.set_status(f'Cropped to ({x},{y}) {w}×{h}')
        self.start_render_now()

    # ---- UI field helpers ----
    def _update_rect_from_fields(self):
        if not self._crop_mode:
            return
        try:
            x = int(self._crop_x.get())
            y = int(self._crop_y.get())
            w = int(self._crop_w.get())
            h = int(self._crop_h.get())
        except Exception:
            return
        iw, ih = self._img_size()
        x = max(0, min(iw - 1, x))
        y = max(0, min(ih - 1, y))
        w = max(1, min(iw - x, w))
        h = max(1, min(ih - y, h))
        self._crop_rect = [x, y, w, h]
        self._redraw_crop_overlay()

    def _sync_fields_from_rect(self):
        if self._crop_rect is None:
            return
        x, y, w, h = self._crop_rect
        try:
            self._crop_x.set(int(x))
            self._crop_y.set(int(y))
            self._crop_w.set(int(w))
            self._crop_h.set(int(h))
        except Exception:
            pass

    def _save_crop_dims(self):
        if self._crop_rect is None:
            self.set_status('No crop to save')
            return
        try:
            path = filedialog.asksaveasfilename(
                title='Save crop dimensions',
                defaultextension='.txt',
                filetypes=[('Text files', '*.txt'), ('All files', '*.*')],
                initialfile='crop.txt')
            if not path:
                return
            x, y, w, h = self._crop_rect
            content = f"x={int(x)}\ny={int(y)}\nw={int(w)}\nh={int(h)}\n"
            with open(path, 'w', encoding='utf-8') as f:
                f.write(content)
            try:
                self.set_status(f'Saved crop to {os.path.basename(path)}')
            except Exception:
                pass
        except Exception:
            self.set_status('Failed to save crop')

    # ---- drawing ----
    def _clear_crop_overlay(self):
        if not getattr(self, '_crop_items', None):
            return
        try:
            for item in self._crop_items:
                try:
                    self.illum_canvas.delete(item)
                except Exception:
                    pass
        finally:
            self._crop_items = []

    def _redraw_crop_overlay(self):
        self._clear_crop_overlay()
        if not self._crop_mode or self._crop_rect is None:
            return
        scale = self._illum_scale()
        x, y, w, h = self._crop_rect
        # Canvas coordinates
        x0 = int(round(x * scale))
        y0 = int(round(y * scale))
        x1 = int(round((x + w) * scale))
        y1 = int(round((y + h) * scale))
        # rectangle
        try:
            rect_id = self.illum_canvas.create_rectangle(x0, y0, x1, y1, outline='#ffd400', width=2, dash=(4, 3))
            self._crop_items.append(rect_id)
        except Exception:
            return
        # shaded regions outside crop to emphasize selection
        try:
            # Canvas should be sized to the displayed image; use canvas size
            fw = int(self.illum_canvas.winfo_width())
            fh = int(self.illum_canvas.winfo_height())
            shade_opts = dict(fill='#000000', stipple='gray50', outline='')
            for (ax0, ay0, ax1, ay1) in [
                (0, 0, fw, max(0, y0)),            # top
                (0, y0, max(0, x0), y1),           # left
                (x1, y0, fw, y1),                   # right
                (0, min(fh, y1), fw, fh),          # bottom
            ]:
                if ax1 > ax0 and ay1 > ay0:
                    sid = self.illum_canvas.create_rectangle(ax0, ay0, ax1, ay1, **shade_opts)
                    self._crop_items.append(sid)
        except Exception:
            pass
        # handles
        r = int(self._crop_handle_px)
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        for cx, cy in corners:
            try:
                hid = self.illum_canvas.create_rectangle(cx - r, cy - r, cx + r, cy + r, fill='#ffd400', outline='')
                self._crop_items.append(hid)
            except Exception:
                pass
        # label text
        try:
            txt = f'({int(x)},{int(y)})  {int(w)}×{int(h)}'
            tid = self.illum_canvas.create_text(x0 + 6, y0 - 10, anchor='w', text=txt, fill='#ffd400')
            self._crop_items.append(tid)
        except Exception:
            pass
        # ensure overlay is above image
        try:
            for it in self._crop_items:
                self.illum_canvas.tag_raise(it)
        except Exception:
            pass

    # ---- interactions ----
    def _on_crop_press(self, event):
        if not self._crop_mode or self._crop_rect is None:
            return 'break'
        scale = self._illum_scale()
        x, y, w, h = self._crop_rect
        x0 = x * scale
        y0 = y * scale
        x1 = (x + w) * scale
        y1 = (y + h) * scale
        ex, ey = float(event.x), float(event.y)
        # hit test corners
        r = float(self._crop_handle_px) * 1.8
        def near(ax, ay):
            return (abs(ex - ax) <= r) and (abs(ey - ay) <= r)
        if near(x0, y0):
            self._crop_drag = 'tl'
        elif near(x1, y0):
            self._crop_drag = 'tr'
        elif near(x0, y1):
            self._crop_drag = 'bl'
        elif near(x1, y1):
            self._crop_drag = 'br'
        elif (ex >= x0 and ex <= x1 and ey >= y0 and ey <= y1):
            self._crop_drag = 'move'
        else:
            self._crop_drag = None
        self._crop_last = (ex / scale, ey / scale)
        return 'break'

    def _on_crop_drag(self, event):
        if not self._crop_mode or self._crop_rect is None or self._crop_drag is None:
            return 'break'
        scale = self._illum_scale()
        xi, yi = float(event.x) / scale, float(event.y) / scale
        last = self._crop_last
        if last is None:
            self._crop_last = (xi, yi)
            return 'break'
        dx = xi - last[0]
        dy = yi - last[1]
        x, y, w, h = self._crop_rect
        iw, ih = self._img_size()
        # update based on drag type
        if self._crop_drag == 'move':
            x += dx
            y += dy
        elif self._crop_drag == 'tl':
            x += dx
            y += dy
            w -= dx
            h -= dy
        elif self._crop_drag == 'tr':
            y += dy
            w += dx
            h -= dy
        elif self._crop_drag == 'bl':
            x += dx
            w -= dx
            h += dy
        elif self._crop_drag == 'br':
            w += dx
            h += dy
        # normalize to ensure w,h>=1 and clamp to image bounds
        x = max(0.0, x)
        y = max(0.0, y)
        w = max(1.0, w)
        h = max(1.0, h)
        if x + w > iw:
            if self._crop_drag in ('move',):
                x = iw - w
            else:
                w = iw - x
        if y + h > ih:
            if self._crop_drag in ('move',):
                y = ih - h
            else:
                h = ih - y
        self._crop_rect = [x, y, w, h]
        self._crop_last = (xi, yi)
        self._sync_fields_from_rect()
        self._redraw_crop_overlay()
        return 'break'

    def _on_crop_release(self, event):
        self._crop_drag = None
        self._crop_last = None
        return 'break'
