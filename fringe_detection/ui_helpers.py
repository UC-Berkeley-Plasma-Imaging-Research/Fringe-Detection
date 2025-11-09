import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


def to_photoimage_from_bgr_with_scale(bgr, scale=1.0, interpolation=Image.BILINEAR):
    """Convert a BGR/gray numpy array to a Tk PhotoImage (optional scaling).
    
    Args:
        bgr: BGR or grayscale numpy array
        scale: Scale factor (default=1.0)
        interpolation: PIL interpolation mode (default=Image.BILINEAR)
    """
    if bgr is None:
        return ImageTk.PhotoImage(Image.new('RGB', (1, 1)))

    # Convert only if needed
    if hasattr(to_photoimage_from_bgr_with_scale, '_last_bgr') and \
       hasattr(to_photoimage_from_bgr_with_scale, '_last_img') and \
       bgr is to_photoimage_from_bgr_with_scale._last_bgr:
        img = to_photoimage_from_bgr_with_scale._last_img
    else:
        if bgr.ndim == 2:
            img = Image.fromarray(bgr)
        else:
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        to_photoimage_from_bgr_with_scale._last_bgr = bgr
        to_photoimage_from_bgr_with_scale._last_img = img

    # Scale only if needed
    if scale is None or float(scale) == 1.0:
        return ImageTk.PhotoImage(img)

    w, h = img.size
    new_w = max(1, int(w * float(scale)))
    new_h = max(1, int(h * float(scale)))
    
    # Check if we have a cached version of this size
    cache_key = (id(img), new_w, new_h, interpolation)
    if hasattr(to_photoimage_from_bgr_with_scale, '_size_cache'):
        cached = to_photoimage_from_bgr_with_scale._size_cache.get(cache_key)
        if cached is not None:
            return cached
    else:
        to_photoimage_from_bgr_with_scale._size_cache = {}

    # Create new scaled image
    img2 = img.resize((new_w, new_h), resample=interpolation)
    photo = ImageTk.PhotoImage(img2)
    
    # Cache result
    to_photoimage_from_bgr_with_scale._size_cache[cache_key] = photo
    
    # Limit cache size
    if len(to_photoimage_from_bgr_with_scale._size_cache) > 10:
        # Remove oldest entries
        for k in list(to_photoimage_from_bgr_with_scale._size_cache.keys())[:-10]:
            del to_photoimage_from_bgr_with_scale._size_cache[k]
            
    return photo


def make_slider_row(parent, label_text, var, frm, to, resolution=None, is_int=False, fmt=None, command=None):
    """Create a labeled slider with a live value label; returns the Scale widget."""
    if command is None:
        # no-op default
        def command(_=None):
            return
    # label above
    ttk.Label(parent, text=label_text).pack(anchor='w')
    row = ttk.Frame(parent)
    row.pack(fill='x')
    scale = ttk.Scale(row, from_=frm, to=to, variable=var, command=command)
    scale.pack(side='left', fill='x', expand=True)
    # value label
    val_var = tk.StringVar()
    if fmt is None:
        fmt = "{}"

    def _update_val(*a):
        try:
            v = var.get()
            if is_int:
                val_var.set(f"{int(round(v))}")
            else:
                if isinstance(v, float) and abs(v - round(v)) < 1e-6:
                    val_var.set(f"{int(round(v))}")
                else:
                    try:
                        val_var.set(fmt.format(v))
                    except Exception:
                        val_var.set(str(v))
        except Exception:
            val_var.set('')

    _update_val()
    try:
        var.trace_add('write', lambda *a: _update_val())
    except Exception:
        var.trace('w', lambda *a: _update_val())

    lbl = ttk.Label(row, textvariable=val_var, width=6, anchor='e')
    lbl.pack(side='left', padx=(6, 0))
    return scale
