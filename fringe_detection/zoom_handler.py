"""Reusable zoom & pan handler for a Tkinter canvas.

Provides right-click-and-drag panning and mouse-wheel zooming centered on
the pointer position. The handler is intentionally lightweight and
generic: it requires callbacks to get/set a numeric zoom level and to
trigger whatever redraw/rescale operation the host widget uses.

Usage example (from an object that has `viewport` canvas, `_zoom_level`,
and `_rescale_display_images` method):

    handler = ZoomPanHandler(
        widget=self.viewport,
        get_zoom=lambda: self._zoom_level,
        set_zoom=lambda z: setattr(self, '_zoom_level', z),
        rescale_callback=self._rescale_display_images,
    )

The handler attaches event bindings on construction and can be detached
with `handler.detach()` if needed.
"""
from typing import Callable, Optional


class ZoomPanHandler:
    def __init__(
        self,
        widget,
        get_zoom: Callable[[], float],
        set_zoom: Callable[[float], None],
        rescale_callback: Callable[[], None],
        min_zoom: float = 0.1,
        max_zoom: float = 10.0,
        zoom_step: float = 1.1,
    ):
        """Create and attach a zoom/pan handler to a Tk widget (usually Canvas).

        widget: a Tkinter widget (Canvas) that supports canvasx/canvasy,
                bbox('all'), xview_moveto, yview_moveto and update_idletasks.
        get_zoom/set_zoom: callbacks to read and write the numeric zoom level.
        rescale_callback: called after zoom level changes to redraw/rescale items.
        """
        self.widget = widget
        self.get_zoom = get_zoom
        self.set_zoom = set_zoom
        self.rescale_callback = rescale_callback
        self.min_zoom = min_zoom
        self.max_zoom = max_zoom
        self.zoom_step = zoom_step

        # drag state
        self._is_dragging = False
        self._drag_start_root: Optional[tuple] = None
        self._drag_scroll_start_px: Optional[tuple] = None

        # bind events (Windows MouseWheel event)
        try:
            self.widget.bind('<MouseWheel>', self._on_mousewheel)
            # Right button pan
            self.widget.bind('<Button-3>', self._on_pan_start)
            self.widget.bind('<B3-Motion>', self._on_pan_move)
            self.widget.bind('<ButtonRelease-3>', self._on_pan_end)
        except Exception:
            # Best-effort attach; failures will be silent so callers can handle
            pass

    def detach(self):
        """Remove event bindings created by this handler."""
        try:
            self.widget.unbind('<MouseWheel>')
            self.widget.unbind('<Button-3>')
            self.widget.unbind('<B3-Motion>')
            self.widget.unbind('<ButtonRelease-3>')
        except Exception:
            pass

    def _on_mousewheel(self, event):
        """Zoom in/out centered at mouse pointer position.

        Uses event.delta (Windows) sign to determine direction.
        """
        try:
            # Normalize delta -- on Windows event.delta is multiple of 120
            delta = int(event.delta / 120) if hasattr(event, 'delta') else 0
            if delta == 0:
                return
            old_zoom = float(self.get_zoom())
            zoom_factor = self.zoom_step if delta > 0 else (1.0 / self.zoom_step)
            new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * zoom_factor))
            if new_zoom == old_zoom:
                return

            # Mouse position relative to widget
            try:
                mouse_x = int(event.x)
                mouse_y = int(event.y)
            except Exception:
                # Fallback to center
                mouse_x = self.widget.winfo_width() // 2
                mouse_y = self.widget.winfo_height() // 2

            left_px_before = float(self.widget.canvasx(0))
            top_px_before = float(self.widget.canvasy(0))
            canvas_x_before = left_px_before + float(mouse_x)
            canvas_y_before = top_px_before + float(mouse_y)

            zoom_ratio = float(new_zoom) / float(old_zoom) if old_zoom != 0 else 1.0

            # update zoom and redraw
            self.set_zoom(new_zoom)
            try:
                self.rescale_callback()
            except Exception:
                pass
            try:
                self.widget.update_idletasks()
            except Exception:
                pass

            # Compute where the canvas point moved to after rescale and move view to keep pointer
            canvas_x_after = canvas_x_before * zoom_ratio
            canvas_y_after = canvas_y_before * zoom_ratio

            bbox_after = self.widget.bbox('all')
            if bbox_after:
                bbox_width = float(bbox_after[2] - bbox_after[0])
                bbox_height = float(bbox_after[3] - bbox_after[1])
                vp_w = max(1.0, float(self.widget.winfo_width()))
                vp_h = max(1.0, float(self.widget.winfo_height()))
                new_left_px = canvas_x_after - float(mouse_x)
                new_top_px = canvas_y_after - float(mouse_y)
                max_left = max(0.0, bbox_width - vp_w)
                max_top = max(0.0, bbox_height - vp_h)
                new_left_px = min(max(new_left_px, 0.0), max_left)
                new_top_px = min(max(new_top_px, 0.0), max_top)
                new_x_fraction = new_left_px / bbox_width if bbox_width > 0 else 0.0
                new_y_fraction = new_top_px / bbox_height if bbox_height > 0 else 0.0
                try:
                    self.widget.xview_moveto(max(0.0, min(1.0, new_x_fraction)))
                    self.widget.yview_moveto(max(0.0, min(1.0, new_y_fraction)))
                except Exception:
                    pass
        except Exception:
            # swallow exceptions to avoid breaking UI
            pass

    def _get_scroll_offsets_px(self):
        try:
            bbox = self.widget.bbox('all')
            if not bbox:
                return 0.0, 0.0, 1.0, 1.0
            width = float(bbox[2] - bbox[0])
            height = float(bbox[3] - bbox[1])
            x_frac0 = self.widget.xview()[0]
            y_frac0 = self.widget.yview()[0]
            return x_frac0 * width, y_frac0 * height, width, height
        except Exception:
            return 0.0, 0.0, 1.0, 1.0

    def _on_pan_start(self, event):
        try:
            self._is_dragging = True
            # record root coords so movement works even if cursor leaves widget
            self._drag_start_root = (event.x_root, event.y_root)
            sx, sy, _, _ = self._get_scroll_offsets_px()
            self._drag_scroll_start_px = (sx, sy)
            try:
                # set a panning cursor if widget supports configure
                self.widget.configure(cursor='fleur')
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
            try:
                self.widget.xview_moveto(new_x_frac)
                self.widget.yview_moveto(new_y_frac)
            except Exception:
                pass
            return 'break'
        except Exception:
            return None

    def _on_pan_end(self, event):
        try:
            self._is_dragging = False
            try:
                self.widget.configure(cursor='')
            except Exception:
                pass
            return 'break'
        except Exception:
            return None


__all__ = ['ZoomPanHandler']
