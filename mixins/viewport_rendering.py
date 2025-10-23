import cv2
from fringe_detection import to_photoimage_from_bgr_with_scale


class ViewportRenderingMixin:
    def _on_mousewheel(self, event):
        try:
            # Zoom in/out at mouse position (like Google Maps)
            delta = int(event.delta / 120) if hasattr(event, 'delta') else 0
            zoom_factor = 1.1 if delta > 0 else 0.9
            old_zoom = self._zoom_level
            new_zoom = max(0.1, min(10.0, old_zoom * zoom_factor))
            if new_zoom != old_zoom:
                try:
                    mouse_x = int(event.x_root) - int(self.viewport.winfo_rootx())
                    mouse_y = int(event.y_root) - int(self.viewport.winfo_rooty())
                except Exception:
                    mouse_x = self.viewport.winfo_width() // 2
                    mouse_y = self.viewport.winfo_height() // 2
                left_px_before = float(self.viewport.canvasx(0))
                top_px_before = float(self.viewport.canvasy(0))
                canvas_x_before = left_px_before + float(mouse_x)
                canvas_y_before = top_px_before + float(mouse_y)
                zoom_ratio = new_zoom / old_zoom
                self._zoom_level = new_zoom
                self._rescale_display_images()
                self.viewport.update_idletasks()
                canvas_x_after = canvas_x_before * zoom_ratio
                canvas_y_after = canvas_y_before * zoom_ratio
                bbox_after = self.viewport.bbox('all')
                if bbox_after:
                    bbox_width = float(bbox_after[2] - bbox_after[0])
                    bbox_height = float(bbox_after[3] - bbox_after[1])
                    vp_w = max(1.0, float(self.viewport.winfo_width()))
                    vp_h = max(1.0, float(self.viewport.winfo_height()))
                    new_left_px = canvas_x_after - float(mouse_x)
                    new_top_px = canvas_y_after - float(mouse_y)
                    max_left = max(0.0, bbox_width - vp_w)
                    max_top = max(0.0, bbox_height - vp_h)
                    new_left_px = min(max(new_left_px, 0.0), max_left)
                    new_top_px = min(max(new_top_px, 0.0), max_top)
                    new_x_fraction = new_left_px / bbox_width if bbox_width > 0 else 0.0
                    new_y_fraction = new_top_px / bbox_height if bbox_height > 0 else 0.0
                    self.viewport.xview_moveto(max(0.0, min(1.0, new_x_fraction)))
                    self.viewport.yview_moveto(max(0.0, min(1.0, new_y_fraction)))
        except Exception:
            pass

    def _on_viewport_configure(self, event=None):
        try:
            if self._resize_after_id:
                try:
                    self.after_cancel(self._resize_after_id)
                except Exception:
                    pass
            self._resize_after_id = self.after(120, self._rescale_display_images)
        except Exception:
            pass

    def _get_scroll_offsets_px(self):
        try:
            bbox = self.viewport.bbox('all')
            if not bbox:
                return 0.0, 0.0, 1.0, 1.0
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            x_frac0 = self.viewport.xview()[0]
            y_frac0 = self.viewport.yview()[0]
            return x_frac0 * width, y_frac0 * height, width, height
        except Exception:
            return 0.0, 0.0, 1.0, 1.0

    def _on_pan_start(self, event):
        try:
            self._is_dragging = True
            self._drag_start_root = (event.x_root, event.y_root)
            sx, sy, _, _ = self._get_scroll_offsets_px()
            self._drag_scroll_start_px = (sx, sy)
            try:
                self.configure(cursor='fleur')
            except Exception:
                pass
            return 'break'
        except Exception:
            return None

    def _on_pan_move(self, event):
        if not self._is_dragging:
            return None
        try:
            dx = event.x_root - self._drag_start_root[0]
            dy = event.y_root - self._drag_start_root[1]
            start_x_px, start_y_px = self._drag_scroll_start_px
            new_x_px = start_x_px - dx
            new_y_px = start_y_px - dy
            _, _, width, height = self._get_scroll_offsets_px()
            new_x_frac = 0.0 if width <= 0 else max(0.0, min(1.0, new_x_px / width))
            new_y_frac = 0.0 if height <= 0 else max(0.0, min(1.0, new_y_px / height))
            self.viewport.xview_moveto(new_x_frac)
            self.viewport.yview_moveto(new_y_frac)
            return 'break'
        except Exception:
            return None

    def _on_pan_end(self, event):
        try:
            self._is_dragging = False
            try:
                self.configure(cursor='')
            except Exception:
                pass
            return 'break'
        except Exception:
            return None

    def _restore_scroll_after_update(self, old_x_px: float, old_y_px: float, attempt: int = 0):
        try:
            bbox = self.viewport.bbox('all')
            if not bbox:
                return
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            if (height <= 1.0 or width <= 1.0) and attempt < 3:
                self.after(16, lambda: self._restore_scroll_after_update(old_x_px, old_y_px, attempt + 1))
                return
            new_x_frac = 0.0 if width <= 0 else old_x_px / width
            new_y_frac = 0.0 if height <= 0 else old_y_px / height
            new_x_frac = max(0.0, min(1.0, new_x_frac))
            new_y_frac = max(0.0, min(1.0, new_y_frac))
            self.viewport.xview_moveto(new_x_frac)
            self.viewport.yview_moveto(new_y_frac)
        except Exception:
            pass

    def _rescale_display_images(self):
        try:
            self._resize_after_id = None
            if self._last_illum_bgr is None and self._last_overlay_bgr is None:
                return
            try:
                vp_w = max(1, self.viewport.winfo_width())
            except Exception:
                vp_w = 800

            if self._last_illum_bgr is not None:
                illum_bgr = self._last_illum_bgr
                try:
                    orig_alpha = float(self.orig_alpha.get()) if hasattr(self, 'orig_alpha') else 0.0
                except Exception:
                    orig_alpha = 0.0
                if orig_alpha > 0.0 and self.src_img is not None:
                    try:
                        import numpy as np  # local import safeguard
                        src_bgr = cv2.cvtColor(self.src_img, cv2.COLOR_GRAY2BGR) if self.src_img.ndim == 2 else self.src_img
                        if src_bgr.shape[:2] != illum_bgr.shape[:2]:
                            src_resized = cv2.resize(src_bgr, (illum_bgr.shape[1], illum_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                        else:
                            src_resized = src_bgr
                        illum_bgr = cv2.addWeighted(src_resized, orig_alpha, illum_bgr, 1.0 - orig_alpha, 0)
                    except Exception:
                        pass
                base_scale = min(1.0, float(vp_w) / float(illum_bgr.shape[1])) if illum_bgr.shape[1] > 0 else 1.0
                scale_illum = base_scale * self._zoom_level
                photo_illum = to_photoimage_from_bgr_with_scale(illum_bgr, scale=scale_illum)
                self._photo_illum = photo_illum
                iw, ih = photo_illum.width(), photo_illum.height()
                self.illum_canvas.config(width=iw, height=ih)
                x = 0
                if self._illum_img_id is None:
                    self._illum_img_id = self.illum_canvas.create_image(x, 0, anchor='nw', image=photo_illum)
                else:
                    self.illum_canvas.itemconfig(self._illum_img_id, image=photo_illum)
                    self.illum_canvas.coords(self._illum_img_id, x, 0)

            if self._last_overlay_bgr is not None:
                fringe_bgr = self._last_overlay_bgr
                try:
                    orig_alpha = float(self.orig_alpha.get()) if hasattr(self, 'orig_alpha') else 0.0
                except Exception:
                    orig_alpha = 0.0
                if orig_alpha > 0.0 and self.src_img is not None:
                    try:
                        src_bgr = cv2.cvtColor(self.src_img, cv2.COLOR_GRAY2BGR) if self.src_img.ndim == 2 else self.src_img
                        if src_bgr.shape[:2] != fringe_bgr.shape[:2]:
                            src_resized = cv2.resize(src_bgr, (fringe_bgr.shape[1], fringe_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
                        else:
                            src_resized = src_bgr
                        fringe_bgr = cv2.addWeighted(src_resized, orig_alpha, fringe_bgr, 1.0 - orig_alpha, 0)
                    except Exception:
                        pass
                base_scale_fringe = min(1.0, float(vp_w) / float(fringe_bgr.shape[1])) if fringe_bgr.shape[1] > 0 else 1.0
                scale_fringe = base_scale_fringe * self._zoom_level
                photo_fringe = to_photoimage_from_bgr_with_scale(fringe_bgr, scale=scale_fringe)
                self._photo_fringe = photo_fringe
                fw, fh = photo_fringe.width(), photo_fringe.height()
                self.fringe_canvas.config(width=fw, height=fh)
                x2 = 0
                if self._fringe_img_id is None:
                    self._fringe_img_id = self.fringe_canvas.create_image(x2, 0, anchor='nw', image=photo_fringe)
                else:
                    self.fringe_canvas.itemconfig(self._fringe_img_id, image=photo_fringe)
                    self.fringe_canvas.coords(self._fringe_img_id, x2, 0)

            try:
                content_w = 0
                if getattr(self, '_photo_illum', None):
                    content_w = max(content_w, self._photo_illum.width())
                if getattr(self, '_photo_fringe', None):
                    content_w = max(content_w, self._photo_fringe.width())
                if content_w <= 0:
                    content_w = vp_w
                self.inner_frame.config(width=content_w)
            except Exception:
                pass
            try:
                self._redraw_crop_overlay()
            except Exception:
                pass
        except Exception:
            pass

    def _fringe_scale(self):
        try:
            if self._photo_fringe is not None and self._last_overlay_bgr is not None:
                w_img = int(self._last_overlay_bgr.shape[1])
                w_disp = int(self._photo_fringe.width())
                if w_img > 0:
                    return float(w_disp) / float(w_img)
        except Exception:
            pass
        return 1.0
