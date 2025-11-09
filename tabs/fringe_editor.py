"""
Standalone Fringe Mask Editor (moved to tabs/).
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk, ImageOps
from skimage.morphology import skeletonize


class FringeEditorFrame(tk.Frame):
	"""Embeddable fringe editor as a tkinter Frame (copied from project root)."""
	def __init__(self, master=None, on_apply=None, on_close=None):
		super().__init__(master)
		self._on_apply = on_apply
		self._on_close = on_close
		self._own_root = None

		# Data
		self.mask = None
		self._bg = None
		self._undo_stack = []
		self._photo = None

		# View/zoom state
		self._base_scale = 1.0
		self._zoom = 1.0
		self._scale = 1.0
		self._offset = None
		self._labels_cache = None
		self._labels_dirty = True

		# Toolbar
		self.toolbar = ttk.Frame(self)
		self.toolbar.pack(side="top", fill="x")
		self._toolbar_items = []

		def add(item):
			self._toolbar_items.append(item)
			return item

		add(ttk.Button(self.toolbar, text="Open Binary", command=self.open_binary))
		add(ttk.Button(self.toolbar, text="Open Background", command=self.open_background))
		add(ttk.Button(self.toolbar, text="Save As…", command=self.save_binary))
		add(ttk.Separator(self.toolbar, orient="vertical"))

		self.mode_var = tk.StringVar(value="add")
		add(ttk.Radiobutton(self.toolbar, text="Add Black", value="add", variable=self.mode_var,
							 command=lambda: self._set_mode("add")))
		add(ttk.Radiobutton(self.toolbar, text="Remove Black", value="erase", variable=self.mode_var,
							 command=lambda: self._set_mode("erase")))
		add(ttk.Separator(self.toolbar, orient="vertical"))

		self.brush_var = tk.DoubleVar(value=10.0)

		add(ttk.Label(self.toolbar, text="Background brightness"))
		self.bg_brightness = tk.DoubleVar(value=5.0)
		self.bg_scale = ttk.Scale(self.toolbar, from_=1.0, to=10.0, orient="horizontal",
								  variable=self.bg_brightness, command=self._on_bg_brightness_changed)
		add(self.bg_scale)
		add(ttk.Separator(self.toolbar, orient="vertical"))

		add(ttk.Label(self.toolbar, text="Angle°"))
		self.angle_deg_var = tk.IntVar(value=40)
		try:
			ang_spin = tk.Spinbox(self.toolbar, from_=0, to=45, width=4, textvariable=self.angle_deg_var)
		except Exception:
			ang_spin = tk.Entry(self.toolbar, width=4, textvariable=self.angle_deg_var)
		add(ang_spin)

		add(ttk.Label(self.toolbar, text="Link tol"))
		self.link_tol_var = tk.IntVar(value=10)
		try:
			link_spin = tk.Spinbox(self.toolbar, from_=1, to=300, width=4, textvariable=self.link_tol_var)
		except Exception:
			link_spin = tk.Entry(self.toolbar, width=4, textvariable=self.link_tol_var)
		add(link_spin)

		add(ttk.Button(self.toolbar, text="½ Angle, 2× Tol", command=self._halve_angle_double_tol))
		add(ttk.Separator(self.toolbar, orient="vertical"))

		self.show_components_var = tk.BooleanVar(value=False)
		add(ttk.Checkbutton(self.toolbar, text="Color comps", variable=self.show_components_var,
							 command=lambda: self._refresh_display(False)))

		add(ttk.Button(self.toolbar, text="Link endpoints", command=self._link_endpoints))
		add(ttk.Button(self.toolbar, text="Undo", command=self.undo))

		def _layout_toolbar(event=None):
			if not self._toolbar_items:
				return
			try:
				self.update_idletasks()
				avail = max(1, self.toolbar.winfo_width())
			except Exception:
				avail = 800
			for w in self._toolbar_items:
				try:
					w.grid_forget()
				except Exception:
					pass
			row = 0; col = 0; cur_w = 0; pad_x = 6
			for w in self._toolbar_items:
				try:
					req = max(1, w.winfo_reqwidth())
				except Exception:
					req = 60
				sticky = 'w'
				if isinstance(w, ttk.Separator):
					sticky = 'ns'; req = max(6, min(req, 6))
				if col > 0 and (cur_w + req + pad_x) > avail:
					row += 1; col = 0; cur_w = 0
				w.grid(row=row, column=col, padx=3, pady=2, sticky=sticky)
				cur_w += req + pad_x; col += 1
		self.toolbar.bind('<Configure>', _layout_toolbar)
		_layout_toolbar()

		self.status = ttk.Label(self, text="Open a binary image (0/255) to edit…", anchor="w")
		self.status.pack(side="bottom", fill="x")

		self.canvas = tk.Canvas(self, bg="gray20", highlightthickness=0)
		self.canvas.pack(side="top", fill="both", expand=True)

		self.canvas.bind("<Configure>", self._on_canvas_configure)
		self.canvas.bind("<Button-1>", self._on_paint_start)
		self.canvas.bind("<B1-Motion>", self._on_paint_move)
		self.canvas.bind("<ButtonRelease-1>", self._on_paint_end)
		self.canvas.bind("<Motion>", self._on_mouse_move)
		self.canvas.bind("<Leave>", self._on_mouse_leave)
		self.canvas.bind("<Control-MouseWheel>", self._on_ctrl_mouse_wheel)
		self.canvas.bind("<MouseWheel>", self._on_mouse_wheel)
		self.canvas.bind("<Button-3>", self._on_pan_start)
		self.canvas.bind("<B3-Motion>", self._on_pan_move)
		self.canvas.bind("<ButtonRelease-3>", self._on_pan_end)

		self._bind_to_toplevel("<Control-z>", lambda e: self.undo())
		self._bind_to_toplevel("<Key-a>", lambda e: self._set_mode("add"))
		self._bind_to_toplevel("<Key-e>", lambda e: self._set_mode("erase"))

		self._painting = False
		self._cursor_circle_id = None
		self._last_mouse_pos = None
		self._panning = False
		self._pan_start = (0, 0)
		self._offset_start = (0, 0)

		self._notebook_ref = None
		try:
			tl = self.winfo_toplevel()
			for child in tl.winfo_children():
				try:
					if isinstance(child, ttk.Notebook):
						self._notebook_ref = child
						break
				except Exception:
					pass
		except Exception:
			self._notebook_ref = None

	def set_data(self, mask: np.ndarray, background: np.ndarray | None = None):
		if mask is None:
			return
		try: m = mask.copy()
		except Exception: m = mask
		if m.dtype != np.uint8: m = m.astype(np.uint8)
		self.mask = m
		if background is not None:
			try: b = background.copy()
			except Exception: b = background
			if b.dtype != np.uint8: b = b.astype(np.uint8)
			if b.shape != self.mask.shape:
				try: b = cv2.resize(b, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
				except Exception: b = None
			self._bg = b
		self._undo_stack.clear(); self._zoom = 1.0; self._base_scale = 1.0; self._offset = None
		self._labels_dirty = True
		self._refresh_display(force_recompute_base=True)

	def get_mask(self) -> np.ndarray | None:
		return None if self.mask is None else self.mask.copy()

	def _bind_to_toplevel(self, sequence, func):
		try: self.winfo_toplevel().bind(sequence, func)
		except Exception:
			try: self.bind(sequence, func)
			except Exception: pass

	def _set_mode(self, mode):
		if mode in ("add","erase"):
			self.mode_var.set(mode)
			if mode == "add":
				try: self.brush_var.set(0.5)
				except Exception: pass

	def set_status(self, text):
		try: self.status.config(text=text)
		except Exception: pass

	def open_binary(self):
		# Default to EditedImages folder within the repo
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		edited_dir = os.path.join(base_dir, 'EditedImages')
		initial_dir = edited_dir if os.path.isdir(edited_dir) else base_dir
		path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title="Open binary image",
										  filetypes=[("Images", ("*.png","*.jpg","*.jpeg","*.tif","*.tiff"))], initialdir=initial_dir)
		if not path: return
		img0 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
		if img0 is None: messagebox.showerror("Open error","Failed to read the image"); return
		if img0.ndim == 3: img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
		img = cv2.normalize(img0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if img0.dtype != np.uint8 else img0
		mask = np.where(img <= 127, 0, 255).astype(np.uint8)
		self.mask = mask; self._undo_stack.clear()
		if self._bg is not None:
			try:
				if self._bg.shape != self.mask.shape:
					self._bg = cv2.resize(self._bg, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
			except Exception: self._bg = None
		self.set_status(f"Loaded: {os.path.basename(path)} — {mask.shape[1]}×{mask.shape[0]}")
		self._zoom=1.0; self._base_scale=1.0; self._offset=None; self._labels_dirty=True
		self._refresh_display(force_recompute_base=True)

	def open_background(self):
		# Default to EditedImages folder within the repo
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		edited_dir = os.path.join(base_dir, 'EditedImages')
		initial_dir = edited_dir if os.path.isdir(edited_dir) else base_dir
		path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title="Open background (grayscale)",
										  filetypes=[("Images", ("*.png","*.jpg","*.jpeg","*.tif","*.tiff"))], initialdir=initial_dir)
		if not path: return
		bg0 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
		if bg0 is None: messagebox.showerror("Open error","Failed to read the background image"); return
		if bg0.ndim == 3: bg0 = cv2.cvtColor(bg0, cv2.COLOR_BGR2GRAY)
		bg = cv2.normalize(bg0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if bg0.dtype != np.uint8 else bg0
		if self.mask is None:
			self.mask = np.full(bg.shape, 255, dtype=np.uint8)
			self._undo_stack.clear(); self._zoom=1.0; self._base_scale=1.0; self._offset=None
		if bg.shape != self.mask.shape:
			try: bg = cv2.resize(bg, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
			except Exception: messagebox.showerror("Background error","Failed to resize background"); return
		self._bg = bg.astype(np.uint8)
		self.set_status(f"Background loaded: {os.path.basename(path)} — brightness {self.bg_brightness.get():.2f}")
		self._refresh_display(force_recompute_base=True)

	def _on_bg_brightness_changed(self,_=None): self._refresh_display(False)

	def save_binary(self):
		if self.mask is None: messagebox.showinfo("Nothing to save","Load an image first"); return
		# Default save location to EditedImages and ensure it exists
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		edited_dir = os.path.join(base_dir, 'EditedImages')
		try:
			os.makedirs(edited_dir, exist_ok=True)
		except Exception:
			pass
		path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title="Save edited mask", initialdir=edited_dir, defaultextension=".png",
											filetypes=[("PNG","*.png"),("TIFF","*.tif"),("JPEG","*.jpg")])
		if not path: return
		try:
			ok = cv2.imwrite(path, self.mask)
			if not ok: raise RuntimeError("cv2.imwrite returned False")
			self.set_status(f"Saved: {os.path.basename(path)}")
		except Exception as e:
			messagebox.showerror("Save error", str(e))

	def undo(self):
		if not self._undo_stack: return
		try: self.mask = self._undo_stack.pop(); self._labels_dirty = True; self._refresh_display(); self.set_status("Undid last stroke")
		except Exception: pass

	def _halve_angle_double_tol(self):
		try: ang=int(self.angle_deg_var.get()) if hasattr(self,'angle_deg_var') else 0
		except Exception: ang=0
		ang=max(0, ang//2)
		try: tol=int(self.link_tol_var.get()) if hasattr(self,'link_tol_var') else 1
		except Exception: tol=1
		tol=max(1, min(300, tol*2))
		try: self.angle_deg_var.set(ang); self.link_tol_var.set(tol)
		except Exception: pass
		self.set_status(f"Angle set to {ang}°, Link tol set to {tol}px")

	def _link_endpoints(self):
		if self.mask is None: return
		try:
			R = int(self.link_tol_var.get()) if hasattr(self,'link_tol_var') else 12; R=max(1,min(300,R))
			try: theta_deg = int(self.angle_deg_var.get()) if hasattr(self,'angle_deg_var') else 0
			except Exception: theta_deg = 0
			import math
			tan_theta = math.tan(math.radians(max(0,min(89,theta_deg)))) if theta_deg > 0 else None
			self._undo_stack.append(self.mask.copy())
			if len(self._undo_stack) > 20: self._undo_stack = self._undo_stack[-20:]
			bw = (self.mask == 0).astype(np.uint8)
			if bw.max() == 0: return
			skel = skeletonize(bw > 0).astype(np.uint8)
			ones3 = np.ones((3,3), np.uint8)
			nbr = cv2.filter2D(skel, -1, ones3, borderType=cv2.BORDER_CONSTANT)
			endpoints = (skel == 1) & (nbr == 2)
			ys, xs = np.where(endpoints); n = len(xs)
			if n < 2: self.set_status("No endpoints to link"); return
			try: _, labels = cv2.connectedComponents(skel, connectivity=8)
			except Exception: labels = None
			used=set(); links=0; h,w=bw.shape
			for i in range(n):
				if i in used: continue
				x0,y0=int(xs[i]),int(ys[i])
				x_min=max(0,x0-R); x_max=min(w-1,x0+R); y_min=max(0,y0-R); y_max=min(h-1,y0+R)
				best_j=None; best_d2=None
				for j in range(n):
					if j==i or j in used: continue
					x1,y1=int(xs[j]),int(ys[j])
					if x1<x_min or x1>x_max or y1<y_min or y1>y_max: continue
					dx=x1-x0; dy=y1-y0; d2=dx*dx+dy*dy
					if d2 > R*R: continue
					if tan_theta is not None:
						if dx == 0: continue
						if abs(dy) > tan_theta * abs(dx): continue
					if labels is not None:
						lid=int(labels[y0,x0]); rid=int(labels[y1,x1])
						if lid>0 and rid>0 and lid==rid: continue
					if best_d2 is None or d2 < best_d2:
						best_d2=d2; best_j=j
				if best_j is not None:
					x1,y1=int(xs[best_j]),int(ys[best_j])
					try: cv2.line(bw,(x0,y0),(x1,y1),1,1); used.add(i); used.add(best_j); links+=1
					except Exception: pass
			bw = skeletonize(bw > 0).astype(np.uint8)
			self.mask = np.where(bw>0, 0, 255).astype(np.uint8)
			self._labels_dirty=True; self._refresh_display(); self.set_status(f"Linked {links} endpoint pairs (tol {R}px, angle ±{theta_deg}°)")
		except Exception:
			messagebox.showerror("Link endpoints","Failed to link endpoints on the current mask")

	def _on_canvas_configure(self, event=None):
		self._refresh_display(force_recompute_base=(self._zoom == 1.0))

	def _refresh_display(self, force_recompute_base=False):
		if self.mask is None:
			self.canvas.delete("all"); self._cursor_circle_id=None; return
		h, w = self.mask.shape[:2]; cw=max(1,self.canvas.winfo_width()); ch=max(1,self.canvas.winfo_height())
		if force_recompute_base or self._base_scale <= 0: self._base_scale = max(1e-6, min(cw/w, ch/h))
		self._scale = float(self._base_scale * self._zoom)
		disp_w = max(1, int(round(w * self._scale))); disp_h = max(1, int(round(h * self._scale)))
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
		if self._offset is None:
			x0 = (cw - disp_w) // 2; y0 = (ch - disp_h) // 2; self._offset = (x0, y0)
		x0, y0 = self._offset
		vx0 = max(0, x0); vy0 = max(0, y0); vx1 = min(cw, x0 + disp_w); vy1 = min(ch, y0 + disp_h)
		self.canvas.delete("all"); self._cursor_circle_id=None
		if vx1 <= vx0 or vy1 <= vy0:
			self._img_topleft = (x0, y0); return
		scale = self._scale if self._scale > 0 else 1.0
		import math
		ix0 = int(max(0, math.floor((vx0 - x0) / scale)))
		iy0 = int(max(0, math.floor((vy0 - y0) / scale)))
		ix1 = int(min(w, math.ceil((vx1 - x0) / scale)))
		iy1 = int(min(h, math.ceil((vy1 - y0) / scale)))
		if ix1 <= ix0 or iy1 <= iy0:
			self._img_topleft = (x0, y0); return
		dst_w = int(vx1 - vx0); dst_h = int(vy1 - vy0)
		region_mask = preview_mask[iy0:iy1, ix0:ix1]
		colorize = bool(getattr(self, 'show_components_var', tk.BooleanVar(value=False)).get())
		comp_rgb = None
		if colorize and self.mask is not None:
			try:
				if self._labels_cache is None or self._labels_dirty:
					bw_full = (self.mask == 0).astype(np.uint8)
					num, labels_full = cv2.connectedComponents(bw_full, connectivity=8)
					idx = np.arange(num, dtype=np.uint32)
					x = (idx * np.uint32(2654435761)) & np.uint32(0xFFFFFFFF)
					r = (x & np.uint32(0xFF)).astype(np.uint8)
					g = ((x >> np.uint32(8)) & np.uint32(0xFF)).astype(np.uint8)
					b = ((x >> np.uint32(16)) & np.uint32(0xFF)).astype(np.uint8)
					colors = np.stack([r, g, b], axis=1)
					colors = np.maximum(colors, 40)
					colors[0] = (255, 255, 255)
					self._labels_cache = (labels_full, colors)
					self._labels_dirty = False
				labels_full, colors = self._labels_cache
				labels_crop = labels_full[iy0:iy1, ix0:ix1]
				comp_rgb = colors[labels_crop]
			except Exception:
				comp_rgb = None
		if comp_rgb is not None:
			try: pil_img = Image.fromarray(comp_rgb, mode="RGB").resize((dst_w, dst_h), Image.NEAREST)
			except Exception: pil_img = Image.fromarray(region_mask, mode="L").resize((dst_w, dst_h), Image.NEAREST)
		elif self._bg is not None:
			try: bg_region = self._bg[iy0:iy1, ix0:ix1]; bg_img = Image.fromarray(bg_region, mode="L").resize((dst_w, dst_h), Image.BILINEAR)
			except Exception: bg_img = None
			mask_img = Image.fromarray(region_mask, mode="L").resize((dst_w, dst_h), Image.NEAREST)
			if bg_img is not None:
				try:
					factor = float(self.bg_brightness.get()) if self.bg_brightness is not None else 1.0
				except Exception: factor = 1.0
				factor = max(1.0, min(10.0, factor))
				try:
					arr = np.asarray(bg_img, dtype=np.float32)
					arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
					result = Image.fromarray(arr, mode="L")
				except Exception:
					result = bg_img.copy()
				black_L = Image.new("L", (dst_w, dst_h), 0)
				mask_inv = ImageOps.invert(mask_img)
				result.paste(black_L, (0, 0), mask_inv)
				pil_img = result
			else:
				pil_img = mask_img
		else:
			mask_img = Image.fromarray(region_mask, mode="L").resize((dst_w, dst_h), Image.NEAREST)
			pil_img = mask_img
		self._photo = ImageTk.PhotoImage(pil_img)
		self.canvas.create_image(vx0, vy0, anchor="nw", image=self._photo)
		self._img_topleft = (x0, y0)
		if self._last_mouse_pos is not None:
			try: self._update_cursor_circle(self._last_mouse_pos[0], self._last_mouse_pos[1])
			except Exception: pass

	def _canvas_to_image_coords(self, x_canvas, y_canvas):
		if self.mask is None: return None
		x0, y0 = getattr(self, "_img_topleft", (0, 0))
		scale = self._scale if self._scale > 0 else 1.0
		xi = int((x_canvas - x0) / scale); yi = int((y_canvas - y0) / scale)
		h, w = self.mask.shape[:2]
		if xi < 0 or yi < 0 or xi >= w or yi >= h: return None
		return xi, yi

	def _apply_brush(self, xi, yi):
		if self.mask is None: return
		r = float(max(0.1, self.brush_var.get()))
		color = 0 if self.mode_var.get() == "add" else 255
		h, w = self.mask.shape[:2]
		y_min = int(max(0, np.floor(yi - r))); y_max = int(min(h - 1, np.ceil(yi + r)))
		x_min = int(max(0, np.floor(xi - r))); x_max = int(min(w - 1, np.ceil(xi + r)))
		yy, xx = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
		circle = (xx - xi) ** 2 + (yy - yi) ** 2 <= (r * r)
		sub = self.mask[y_min:y_max + 1, x_min:x_max + 1]
		sub[circle] = color
		self._labels_dirty = True

	def _apply_brush_line(self, x0, y0, x1, y1):
		if self.mask is None: return
		r = float(max(0.1, self.brush_var.get()))
		dx = float(x1 - x0); dy = float(y1 - y0)
		dist = float(np.hypot(dx, dy))
		step_len = max(1.0, r * 0.5)
		steps = int(max(1, np.ceil(dist / step_len)))
		if steps <= 1:
			self._apply_brush(x1, y1); return
		for i in range(1, steps + 1):
			t = i / float(steps)
			xi = x0 + dx * t; yi = y0 + dy * t
			self._apply_brush(xi, yi)

	def _on_paint_start(self, event):
		if self.mask is None: return
		try:
			self._undo_stack.append(self.mask.copy())
			if len(self._undo_stack) > 20: self._undo_stack = self._undo_stack[-20:]
		except Exception: pass
		pt = self._canvas_to_image_coords_f(event.x, event.y)
		if pt is not None:
			xi, yi = pt
			try: self._last_paint_xy = (xi, yi)
			except Exception: self._last_paint_xy = None
			self._apply_brush(xi, yi); self._refresh_display(); self._painting = True
		self._last_mouse_pos = (event.x, event.y); self._update_cursor_circle(event.x, event.y)

	def _on_paint_move(self, event):
		if self.mask is None or not self._painting: return
		pt = self._canvas_to_image_coords_f(event.x, event.y)
		if pt is not None:
			xi, yi = pt; lp = getattr(self, "_last_paint_xy", None)
			if lp is None: self._apply_brush(xi, yi)
			else: self._apply_brush_line(lp[0], lp[1], xi, yi)
			self._last_paint_xy = (xi, yi); self._refresh_display()
		self._last_mouse_pos = (event.x, event.y); self._update_cursor_circle(event.x, event.y)

	def _on_paint_end(self, event):
		self._painting = False; self._last_paint_xy = None

	def _canvas_to_image_coords_f(self, x_canvas, y_canvas):
		if self.mask is None: return None
		x0, y0 = getattr(self, "_img_topleft", (0, 0))
		scale = self._scale if self._scale > 0 else 1.0
		xi = (x_canvas - x0) / scale; yi = (y_canvas - y0) / scale
		h, w = self.mask.shape[:2]
		if xi < 0 or yi < 0 or xi >= w or yi >= h: return None
		return xi, yi

	def _global_event_to_image_coords(self, event):
		if self.mask is None: return None
		try:
			x_c = int(event.x_root) - int(self.canvas.winfo_rootx())
			y_c = int(event.y_root) - int(self.canvas.winfo_rooty())
		except Exception:
			return None
		x0, y0 = getattr(self, "_img_topleft", (0, 0))
		scale = self._scale if self._scale > 0 else 1.0
		xi = (x_c - x0) / scale; yi = (y_c - y0) / scale
		return xi, yi

	def _on_global_paint_start(self, event):
		if self.mask is None: return
		self._painting = True; self._on_global_paint_move(event)

	def _on_global_paint_move(self, event):
		if self.mask is None or not self._painting: return
		pt = self._global_event_to_image_coords(event)
		if pt is None: return
		xi_f, yi_f = pt
		try: r = float(max(0.1, self.brush_var.get()))
		except Exception: r = 1.0
		h, w = self.mask.shape[:2]
		if (xi_f + r) < 0 or (yi_f + r) < 0 or (xi_f - r) > (w - 1) or (yi_f - r) > (h - 1): return
		xi = float(xi_f); yi = float(yi_f)
		y_min = int(max(0, np.floor(yi - r))); y_max = int(min(h - 1, np.ceil(yi + r)))
		x_min = int(max(0, np.floor(xi - r))); x_max = int(min(w - 1, np.ceil(xi + r)))
		if y_max >= y_min and x_max >= x_min:
			yy, xx = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
			circle = (xx - xi) ** 2 + (yy - yi) ** 2 <= (r * r)
			color = 0 if self.mode_var.get() == "add" else 255
			sub = self.mask[y_min:y_max + 1, x_min:x_max + 1]
			sub[circle] = color
			self._labels_dirty = True; self._refresh_display()

	def _on_global_paint_end(self, event):
		self._painting = False

	def _on_pan_start(self, event):
		self._panning = True
		self._pan_start = (event.x, event.y)
		self._offset_start = self._offset if self._offset is not None else getattr(self, "_img_topleft", (0, 0))
		try: self.config(cursor='fleur')
		except Exception: pass

	def _on_pan_move(self, event):
		if not self._panning: return
		sx, sy = self._pan_start; dx = event.x - sx; dy = event.y - sy
		ox0, oy0 = self._offset_start; self._offset = (ox0 + dx, oy0 + dy)
		self._refresh_display(False)
		self._last_mouse_pos = (event.x, event.y); self._update_cursor_circle(event.x, event.y)

	def _on_pan_end(self, event):
		self._panning = False
		try: self.config(cursor='')
		except Exception: pass

	def _on_mouse_move(self, event):
		self._last_mouse_pos = (event.x, event.y)
		self._update_cursor_circle(event.x, event.y)

	def _on_mouse_leave(self, event):
		if self._cursor_circle_id is not None:
			try: self.canvas.delete(self._cursor_circle_id)
			except Exception: pass
			self._cursor_circle_id = None
		self._last_mouse_pos = None

	def _update_cursor_circle(self, x, y):
		if self.mask is None:
			if self._cursor_circle_id is not None:
				try: self.canvas.delete(self._cursor_circle_id)
				except Exception: pass
				self._cursor_circle_id = None
			return
		scale = self._scale if self._scale > 0 else 1.0
		r_img = float(max(0.1, self.brush_var.get()))
		r_canvas = max(1.0, (r_img * scale))
		x0 = x - r_canvas; y0 = y - r_canvas; x1 = x + r_canvas; y1 = y + r_canvas
		color = "#00ff88" if self.mode_var.get() == "add" else "#ff5555"
		if self._cursor_circle_id is None:
			try: self._cursor_circle_id = self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=1)
			except Exception: self._cursor_circle_id = None
		else:
			try:
				self.canvas.coords(self._cursor_circle_id, x0, y0, x1, y1)
				self.canvas.itemconfig(self._cursor_circle_id, outline=color)
			except Exception:
				try: self._cursor_circle_id = self.canvas.create_oval(x0, y0, x1, y1, outline=color, width=1)
				except Exception: self._cursor_circle_id = None

	def _on_mouse_wheel(self, event):
		try:
			if getattr(event, "state", 0) & 0x4: return
		except Exception: pass
		if self.mask is None: return
		delta = int(getattr(event, "delta", 0));
		if delta == 0: return
		zoom_factor = 1.1 if delta > 0 else 0.9
		old_zoom = self._zoom; new_zoom = max(0.1, min(16.0, old_zoom * zoom_factor))
		if abs(new_zoom - old_zoom) < 1e-6: return
		x_c = event.x; y_c = event.y
		s_before = max(1e-6, self._base_scale * old_zoom); s_after = max(1e-6, self._base_scale * new_zoom)
		ox, oy = self._offset if self._offset is not None else getattr(self, "_img_topleft", (0, 0))
		x_img = (x_c - ox) / s_before; y_img = (y_c - oy) / s_before
		self._zoom = new_zoom; new_ox = int(round(x_c - x_img * s_after)); new_oy = int(round(y_c - y_img * s_after))
		self._offset = (new_ox, new_oy); self._refresh_display(False)
		self._last_mouse_pos = (x_c, y_c); self._update_cursor_circle(x_c, y_c)

	def _on_ctrl_mouse_wheel(self, event):
		delta = int(getattr(event, "delta", 0)); step = 1.0 if delta > 0 else -1.0
		new_size = float(self.brush_var.get()) + step; new_size = max(0.1, min(200.0, new_size))
		self.brush_var.set(new_size); self._last_mouse_pos = (event.x, event.y); self._update_cursor_circle(event.x, event.y)

