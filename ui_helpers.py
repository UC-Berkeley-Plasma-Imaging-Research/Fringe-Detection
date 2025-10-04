import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2


def to_photoimage_from_bgr_with_scale(bgr, scale=1.0):
    """Convert an OpenCV BGR/gray numpy array to a Tk PhotoImage, optionally scaled.
    Returns an ImageTk.PhotoImage."""
    if bgr is None:
        return ImageTk.PhotoImage(Image.new('RGB', (1, 1)))
    if bgr.ndim == 2:
        img = Image.fromarray(bgr)
    else:
        # convert BGR -> RGB for PIL
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if scale is None or float(scale) == 1.0:
        return ImageTk.PhotoImage(img)
    w, h = img.size
    new_w = max(1, int(round(w * float(scale))))
    new_h = max(1, int(round(h * float(scale))))
    img2 = img.resize((new_w, new_h), resample=Image.BILINEAR)
    return ImageTk.PhotoImage(img2)


def make_slider_row(parent, label_text, var, frm, to, resolution=None, is_int=False, fmt=None, command=None):
    """Create a labeled slider row in parent with a live value label on the right.
    Returns the scale widget. The "command" callable will be set as the scale callback.
    This mirrors the behaviour previously implemented as an instance method.
    """
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

    # initial
    _update_val()
    # trace changes
    try:
        var.trace_add('write', lambda *a: _update_val())
    except Exception:
        var.trace('w', lambda *a: _update_val())

    lbl = ttk.Label(row, textvariable=val_var, width=6, anchor='e')
    lbl.pack(side='left', padx=(6, 0))
    return scale
