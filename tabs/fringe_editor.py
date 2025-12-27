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

		# Data
		self.mask = None  # Binary mask (black/white)
		self.gray_mask = None  # Gray overlay layer (separate)
		self._bg = None
		self._undo_stack = []
		self._gray_undo_stack = []
		self._photo = None

		# Overlay state
		self.overlay_mask = None
		self.overlay_pos = (0, 0)  # Top-left in image coordinates
		self.overlay_active = False
		self._moving_overlay = False

		# Magic2 state
		self.magic2_active = False
		self.magic2_labels = set()

		# View/zoom state
		self._base_scale = 1.0
		self._zoom = 1.0
		self._scale = 1.0
		self._offset = None
		self._labels_cache = None
		self._labels_dirty = True

		# Brush settings per mode
		self._last_mode = "add"
		self._brush_sizes = {
			"add": 1.0,
			"erase": 1.0,
			"add_gray": 10.0,
			"erase_gray": 10.0
		}
		self._brush_offsets = {
			"add": (-0.4, -0.8),
			"erase": (-0.5, -0.6),
			"add_gray": (0.0, 0.0),
			"erase_gray": (0.0, 0.0)
		}
		# Visual cursor offsets (red circle) - separate from actual brush application
		self._cursor_offsets = {
			"add": (0, 0),
			"erase": (0, 0),
			"add_gray": (0.0, 0.0),
			"erase_gray": (0.0, 0.0)
		}

		# Toolbar with title + help icon (header) and a separate body for controls
		self.toolbar = ttk.Frame(self)
		self.toolbar.pack(side="top", fill="x")
		header = ttk.Frame(self.toolbar)
		header.pack(side='top', anchor='w', fill='x', pady=(0,4))
		ttk.Label(header, text='Editor', font=('Segoe UI', 10, 'bold')).pack(side='left')
		# Canvas-based help icon: thin circle, non-bold '?'
		try:
			bg = self.cget('background')
		except Exception:
			bg = '#f0f0f0'
		help_icon = tk.Canvas(header, width=18, height=18, highlightthickness=0, bg=bg)
		help_icon.create_oval(2,2,16,16, outline='#666', width=1)
		help_icon.create_text(9,9, text='?', font=('Segoe UI', 9))
		help_icon.pack(side='left', padx=(6,0))
		self._attach_tooltip(help_icon, (
            'Editor Tab Purpose:\n'
            'This tab allows you to add, remove, connect, or edit binary fringes with\n'
            'manual precision. It also allows a gray mask to be added and edited\n'
            '\n'
            'Controls:\n'
            '- Right-click drag to move image\n'
            '- Mouse wheel to zoom\n'
			'- Ctrl + Mouse wheel to adjust brush size\n'
            '- Left-click drag to Add/Remove any black/gray pixels\n'
            '\n'
            'Features:\n'
            '- Load/Save a binary/background image to edit fringes\n'
			'- Load/Save a grayscale image for masking\n'
			'- Brush modes: Add Black, Remove Black, Add Gray Mask, Remove Gray Mask\n'
			'- Background brightness: Adjusts background images brightness\n'
			'- Fringe Opacity: Adjusts fringe overlay opacity\n'
			'- Link endpoints: Connects nearby fringe endpoints based on angle and distance\n'
            '- Angle: Adjusts angle tolerance of Link endpoint Feature\n'
			'- Link tolerance: Adjusts distance tolerance of Link endpoint Feature\n'
			'ie. Endpoints will only links if another endpoint is within the\n'
			'specified distance and angle constraints.\n'
			'- Magic2 Tester: Draws an invisable vertical line in the center of the canvas\n' \
			'Any fringes that touch the line will be highlighted, used to split fringes.\n'
			'- Color comps: Toggle coloring of connected components for easier visualization\n'
			'- Undo: Undo last strokes\n'
			))
		
		# Body frame to hold all interactive toolbar widgets (isolated from header layout)
		self.toolbar_body = ttk.Frame(self.toolbar)
		self.toolbar_body.pack(side='top', fill='x')

		self._toolbar_items = []
		def add(item): self._toolbar_items.append(item); return item
		add(ttk.Button(self.toolbar_body, text="Open Binary", command=self.open_binary))
		add(ttk.Button(self.toolbar_body, text="Open Background", command=self.open_background))
		add(ttk.Button(self.toolbar_body, text="Save As…", command=self.save_binary))
		add(ttk.Separator(self.toolbar_body, orient="vertical"))
		add(ttk.Button(self.toolbar_body, text="Open Binary Overlay", command=self.open_overlay_binary))
		self.btn_merge_overlay = ttk.Button(self.toolbar_body, text="Merge Overlay", command=self.merge_overlay, state="disabled")
		add(self.btn_merge_overlay)
		add(ttk.Separator(self.toolbar_body, orient="vertical"))
		add(ttk.Button(self.toolbar_body, text="Open Gray Mask", command=self.open_gray_mask))
		add(ttk.Button(self.toolbar_body, text="Save Gray Mask", command=self.save_gray_mask))
		add(ttk.Separator(self.toolbar_body, orient="vertical"))
		self.mode_var = tk.StringVar(value="add")
		add(ttk.Radiobutton(self.toolbar_body, text="Add Black", value="add", variable=self.mode_var, command=lambda: self._set_mode("add")))
		add(ttk.Radiobutton(self.toolbar_body, text="Remove Black", value="erase", variable=self.mode_var, command=lambda: self._set_mode("erase")))
		add(ttk.Separator(self.toolbar_body, orient="vertical"))
		add(ttk.Radiobutton(self.toolbar_body, text="Add Mask (gray)", value="add_gray", variable=self.mode_var, command=lambda: self._set_mode("add_gray")))
		add(ttk.Radiobutton(self.toolbar_body, text="Remove Mask", value="erase_gray", variable=self.mode_var, command=lambda: self._set_mode("erase_gray")))
		add(ttk.Separator(self.toolbar_body, orient="vertical"))
		self.brush_var = tk.DoubleVar(value=1.0)
		add(ttk.Label(self.toolbar_body, text="Background brightness"))
		self.bg_brightness = tk.DoubleVar(value=5.0)
		self.bg_scale = ttk.Scale(self.toolbar_body, from_=1.0, to=75.0, orient="horizontal", variable=self.bg_brightness, command=self._on_bg_brightness_changed)
		add(self.bg_scale)
		add(ttk.Label(self.toolbar_body, text="Fringe Opacity"))
		self.fringe_opacity = tk.DoubleVar(value=1.0)
		self.fringe_opacity_scale = ttk.Scale(self.toolbar_body, from_=0.0, to=1.0, orient="horizontal", variable=self.fringe_opacity, command=self._on_fringe_opacity_changed)
		add(self.fringe_opacity_scale)
		add(ttk.Separator(self.toolbar_body, orient="vertical"))
		# Place 'Link endpoints' before the angle controls and in the same section
		add(ttk.Button(self.toolbar_body, text="Link endpoints", command=self._link_endpoints))
		add(ttk.Label(self.toolbar_body, text="Angle°"))
		self.angle_deg_var = tk.IntVar(value=40)
		try:
			ang_spin = tk.Spinbox(self.toolbar_body, from_=0, to=45, width=4, textvariable=self.angle_deg_var)
		except Exception:
			ang_spin = tk.Entry(self.toolbar_body, width=4, textvariable=self.angle_deg_var)
		add(ang_spin)
		add(ttk.Label(self.toolbar_body, text="Link tol (px)"))
		self.link_tol_var = tk.IntVar(value=10)
		try:
			link_spin = tk.Spinbox(self.toolbar_body, from_=1, to=300, width=4, textvariable=self.link_tol_var)
		except Exception:
			link_spin = tk.Entry(self.toolbar_body, width=4, textvariable=self.link_tol_var)
		add(link_spin)
		add(ttk.Button(self.toolbar_body, text="½ Angle, 2× Tol", command=self._halve_angle_double_tol))
		add(ttk.Separator(self.toolbar_body, orient="vertical"))
		add(ttk.Button(self.toolbar_body, text="Magic2 Tester", command=self.run_magic2_tester))
		add(ttk.Separator(self.toolbar_body, orient="vertical"))
		self.show_components_var = tk.BooleanVar(value=False)
		add(ttk.Checkbutton(self.toolbar_body, text="Color comps", variable=self.show_components_var, command=lambda: self._refresh_display(False)))
		add(ttk.Button(self.toolbar_body, text="Undo", command=self.undo))
		def _layout_toolbar(event=None):
			if not self._toolbar_items: return
			avail = self.toolbar_body.winfo_width()
			if event: avail = event.width
			if avail < 20: avail = 800
			
			self.toolbar_body.pack_propagate(False)
			self.toolbar_body.grid_propagate(False)
			
			for w in self._toolbar_items:
				w.place_forget(); w.grid_forget(); w.pack_forget()
			
			pad = 2
			
			# 1. Chunk items into groups (split after Separator)
			groups = []
			current_group = []
			for w in self._toolbar_items:
				try: req_w = max(1, w.winfo_reqwidth())
				except Exception: req_w = 60
				try: req_h = max(1, w.winfo_reqheight())
				except Exception: req_h = 24
				
				if isinstance(w, ttk.Separator): req_w = 6
				
				current_group.append((w, req_w, req_h))
				if isinstance(w, ttk.Separator):
					groups.append(current_group)
					current_group = []
			if current_group:
				groups.append(current_group)

			# 2. Layout groups
			x = 0; y = 0; row_h = 0
			row_items = []
			
			def layout_row(items, cy, rh):
				cx = 0
				for it, rw, rh_req in items:
					if isinstance(it, ttk.Separator):
						it.place(x=cx, y=cy, width=rw, height=rh)
					else:
						y_off = (rh - rh_req) // 2
						it.place(x=cx, y=cy + y_off, width=rw, height=rh_req)
					cx += rw + pad

			for group in groups:
				# Calculate total width of this group
				group_w = sum(item[1] + pad for item in group)
				
				# If adding this group exceeds width, wrap to next line
				# (unless it's the first group on the line, then we must place it)
				if x > 0 and (x + group_w) > avail:
					layout_row(row_items, y, row_h)
					y += row_h + pad
					x = 0; row_h = 0; row_items = []
				
				# Add group to current row
				for item in group:
					row_items.append(item)
					row_h = max(row_h, item[2])
				x += group_w
			
			if row_items:
				layout_row(row_items, y, row_h)
				y += row_h + pad
			
			self.toolbar_body.configure(height=max(1, y))
		self.toolbar_body.bind('<Configure>', _layout_toolbar)
		_layout_toolbar()

		self.status = ttk.Label(self, text="Open a binary image (0/255) to edit…", anchor="w")
		self.status.pack(side="bottom", fill="x")

		self.canvas = tk.Canvas(self, bg="gray20", highlightthickness=0)
		self.canvas.pack(side="top", fill="both", expand=True)

		# Bind canvas interactions: paint (LMB), pan (RMB), zoom (wheel), brush resize (Ctrl+wheel)
		try:
			self.canvas.bind('<Configure>', self._on_canvas_configure)
			# Painting with left mouse
			self.canvas.bind('<Button-1>', self._on_paint_start)
			self.canvas.bind('<B1-Motion>', self._on_paint_move)
			self.canvas.bind('<ButtonRelease-1>', self._on_paint_end)
			# Panning with right mouse
			self.canvas.bind('<Button-3>', self._on_pan_start)
			self.canvas.bind('<B3-Motion>', self._on_pan_move)
			self.canvas.bind('<ButtonRelease-3>', self._on_pan_end)
			# Cursor preview
			self.canvas.bind('<Motion>', self._on_mouse_move)
			self.canvas.bind('<Leave>', self._on_mouse_leave)
			# Zoom and brush size
			self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
			self.canvas.bind('<Control-MouseWheel>', self._on_ctrl_mouse_wheel)
			# Linux/X11 wheel events
			self.canvas.bind('<Button-4>', self._on_mouse_wheel)
			self.canvas.bind('<Button-5>', self._on_mouse_wheel)
			self.canvas.bind('<Control-Button-4>', self._on_ctrl_mouse_wheel)
			self.canvas.bind('<Control-Button-5>', self._on_ctrl_mouse_wheel)
		except Exception:
			pass

	def _attach_tooltip(self, widget, text):
		"""Attach a hover tooltip to a widget (no side effects on canvas)."""
		_tip = {'win': None}
		def show_tip(_e=None):
			if _tip['win'] is not None: return
			try:
				x = widget.winfo_rootx() + widget.winfo_width() + 8
				y = widget.winfo_rooty() + int(widget.winfo_height()*0.5)
			except Exception:
				x = y = 0
			win = tk.Toplevel(widget); _tip['win'] = win
			try: win.wm_overrideredirect(True)
			except Exception: pass
			try: win.wm_geometry(f"+{x}+{y}")
			except Exception: pass
			frame = ttk.Frame(win, borderwidth=1, relief='solid'); frame.pack()
			lbl = ttk.Label(frame, text=text, justify='left', padding=6); lbl.pack()
		def hide_tip(_e=None):
			w = _tip.get('win')
			if w is not None:
				try: w.destroy()
				except Exception: pass
				_tip['win'] = None
		try:
			widget.bind('<Enter>', show_tip); widget.bind('<Leave>', hide_tip)
		except Exception:
			pass

		self._bind_to_toplevel("<Control-z>", lambda e: self.undo())
		self._bind_to_toplevel("<Key-a>", lambda e: self._set_mode("add"))
		self._bind_to_toplevel("<Key-e>", lambda e: self._set_mode("erase"))
		self._bind_to_toplevel("<Left>", lambda e: self._nudge_overlay(-1, 0))
		self._bind_to_toplevel("<Right>", lambda e: self._nudge_overlay(1, 0))
		self._bind_to_toplevel("<Up>", lambda e: self._nudge_overlay(0, -1))
		self._bind_to_toplevel("<Down>", lambda e: self._nudge_overlay(0, 1))

		self._painting = False
		self._cursor_circle_id = None
		self._last_mouse_pos = None
		self._panning = False
		self._pan_start = (0, 0)
		self._offset_start = (0, 0)

		# _notebook_ref discovery removed (unused)

	def set_data(self, mask: np.ndarray, background: np.ndarray | None = None):
		if mask is None:
			return
		try: m = mask.copy()
		except Exception: m = mask
		if m.dtype != np.uint8: m = m.astype(np.uint8)
		# Separate binary mask from gray layer
		self.mask = np.where(m < 128, 0, 255).astype(np.uint8)  # Binary only
		self.gray_mask = np.where((m > 0) & (m < 255), 255, 0).astype(np.uint8)  # Gray layer
		if background is not None:
			try: b = background.copy()
			except Exception: b = background
			if b.dtype != np.uint8: b = b.astype(np.uint8)
			if b.shape != self.mask.shape:
				try: b = cv2.resize(b, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
				except Exception: b = None
			self._bg = b
		self._undo_stack.clear(); self._gray_undo_stack.clear()
		self._zoom = 1.0; self._base_scale = 1.0; self._offset = None
		self._labels_dirty = True
		self._refresh_display(force_recompute_base=True)

	def get_mask(self) -> np.ndarray | None:
		"""Return merged mask with gray overlay for export."""
		if self.mask is None:
			return None
		# Merge binary mask with gray layer
		result = self.mask.copy()
		if self.gray_mask is not None:
			# Where gray_mask has content (255), set result to gray (128)
			result = np.where(self.gray_mask > 0, 128, result)
		return result

	def _bind_to_toplevel(self, sequence, func):
		try: self.winfo_toplevel().bind(sequence, func)
		except Exception:
			try: self.bind(sequence, func)
			except Exception: pass

	def _set_mode(self, mode):
		if mode in ("add", "erase", "add_gray", "erase_gray"):
			# Save current size to old mode
			try:
				old_size = float(self.brush_var.get())
				self._brush_sizes[self._last_mode] = old_size
			except Exception: pass

			self.mode_var.set(mode)
			# Determine new brush size for this mode.
			# Special rule: when switching to 'erase', make its brush size 1px larger
			# than the 'add' brush size so remove appears slightly bigger.
			new_size = self._brush_sizes.get(mode, None)
			try:
				if mode == 'erase':
					# base add size (fallback to current or 1.0)
					add_size = float(self._brush_sizes.get('add', self.brush_var.get() if self.brush_var is not None else 1.0))
					# make erase 1 pixel larger than add unless an explicit erase size was stored
					if new_size is None:
						new_size = max(0.1, add_size + 1.0)
					# remember it
					self._brush_sizes['erase'] = new_size
				elif mode == 'add':
					# Always reset add brush to 0.5 pixels when selected
					new_size = 0.5
					self._brush_sizes['add'] = new_size
				else:
					if new_size is None:
						new_size = 10.0
			except Exception:
				new_size = (new_size if new_size is not None else 10.0)
			self._last_mode = mode
			try: self.brush_var.set(float(new_size))
			except Exception: pass

			if mode == "add_gray":
				self.set_status("Gray mask mode: Paint to add gray pixels")
			elif mode == "erase_gray":
				self.set_status("Erase gray mode: Paint to remove gray pixels")

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
		# Separate binary mask from gray layer
		# Values 0-127: binary black, 128-254: gray mask, 255: white
		self.mask = np.where(img < 128, 0, 255).astype(np.uint8)  # Binary only
		self.gray_mask = np.where((img > 0) & (img < 255), 255, 0).astype(np.uint8)  # Gray layer
		self._undo_stack.clear(); self._gray_undo_stack.clear()
		
		# Count pixels in each layer for status
		gray_pixels = np.sum(self.gray_mask > 0)
		black_pixels = np.sum(self.mask == 0)
		
		if self._bg is not None:
			try:
				if self._bg.shape != self.mask.shape:
					self._bg = cv2.resize(self._bg, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
			except Exception: self._bg = None
		
		# Informative status message
		status_msg = f"Loaded: {os.path.basename(path)} — {self.mask.shape[1]}×{self.mask.shape[0]}"
		if gray_pixels > 0:
			status_msg += f" (with {gray_pixels} gray mask pixels)"
		self.set_status(status_msg)
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
			self.gray_mask = np.zeros(bg.shape, dtype=np.uint8)
			self._undo_stack.clear(); self._gray_undo_stack.clear()
			self._zoom=1.0; self._base_scale=1.0; self._offset=None
		if bg.shape != self.mask.shape:
			try: bg = cv2.resize(bg, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_LINEAR)
			except Exception: messagebox.showerror("Background error","Failed to resize background"); return
		self._bg = bg.astype(np.uint8)
		self.set_status(f"Background loaded: {os.path.basename(path)} — brightness {self.bg_brightness.get():.2f}")
		self._refresh_display(force_recompute_base=True)

	def open_gray_mask(self):
		# Default to EditedImages folder within the repo
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		edited_dir = os.path.join(base_dir, 'EditedImages')
		initial_dir = edited_dir if os.path.isdir(edited_dir) else base_dir
		path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title="Open gray mask",
										  filetypes=[("Images", ("*.png","*.jpg","*.jpeg","*.tif","*.tiff"))], initialdir=initial_dir)
		if not path: return
		img0 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
		if img0 is None: messagebox.showerror("Open error","Failed to read the image"); return
		if img0.ndim == 3: img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
		img = cv2.normalize(img0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if img0.dtype != np.uint8 else img0
		
		# If no binary mask exists, create one matching this gray mask
		if self.mask is None:
			self.mask = np.full(img.shape, 255, dtype=np.uint8)
			self._undo_stack.clear(); self._gray_undo_stack.clear()
			self._zoom=1.0; self._base_scale=1.0; self._offset=None

		# Resize if needed
		if img.shape != self.mask.shape:
			try: img = cv2.resize(img, (self.mask.shape[1], self.mask.shape[0]), interpolation=cv2.INTER_NEAREST)
			except Exception: messagebox.showerror("Open error","Failed to resize gray mask"); return
			
		self.gray_mask = np.where(img > 127, 255, 0).astype(np.uint8)
		self._gray_undo_stack.clear()
		
		self.set_status(f"Loaded gray mask: {os.path.basename(path)}")
		self._refresh_display(force_recompute_base=(self._zoom==1.0))

	def open_overlay_binary(self):
		if self.mask is None:
			messagebox.showinfo("Overlay error", "Please load a base binary image first.")
			return
		
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		edited_dir = os.path.join(base_dir, 'EditedImages')
		initial_dir = edited_dir if os.path.isdir(edited_dir) else base_dir
		path = filedialog.askopenfilename(parent=self.winfo_toplevel(), title="Open overlay binary",
										  filetypes=[("Images", ("*.png","*.jpg","*.jpeg","*.tif","*.tiff"))], initialdir=initial_dir)
		if not path: return
		
		img0 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
		if img0 is None: messagebox.showerror("Open error","Failed to read the image"); return
		if img0.ndim == 3: img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
		img = cv2.normalize(img0, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) if img0.dtype != np.uint8 else img0
		
		# Create binary mask from loaded image (0=black, 255=white)
		# Assuming input is standard image where dark is ink.
		# If it's already a mask (0/255), we keep it.
		# Let's assume standard thresholding: <128 is black (ink), >128 is white (bg)
		# We want to overlay the BLACK parts (fringes).
		# So we'll store the overlay where 0 is ink, 255 is bg.
		self.overlay_mask = np.where(img < 128, 0, 255).astype(np.uint8)
		
		# Center the overlay initially
		h, w = self.mask.shape[:2]
		oh, ow = self.overlay_mask.shape[:2]
		self.overlay_pos = ((w - ow) // 2, (h - oh) // 2)
		
		self.overlay_active = True
		self.btn_merge_overlay.config(state="normal")
		self.set_status(f"Overlay loaded. Drag to position, then click 'Merge Overlay'.")
		self._refresh_display()

	def merge_overlay(self):
		if not self.overlay_active or self.overlay_mask is None or self.mask is None:
			return
		
		self._undo_stack.append(self.mask.copy())
		if len(self._undo_stack) > 20: self._undo_stack = self._undo_stack[-20:]
		
		ox, oy = self.overlay_pos
		oh, ow = self.overlay_mask.shape[:2]
		h, w = self.mask.shape[:2]
		
		# Calculate intersection
		x0 = max(0, ox); y0 = max(0, oy)
		x1 = min(w, ox + ow); y1 = min(h, oy + oh)
		
		if x1 > x0 and y1 > y0:
			# Source coordinates in overlay
			sx0 = x0 - ox; sy0 = y0 - oy
			sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
			
			# Get the overlay region
			ov_region = self.overlay_mask[sy0:sy1, sx0:sx1]
			
			# Merge logic:
			# We want to add the black pixels (0) from overlay to the mask.
			# Mask: 0 is black, 255 is white.
			# Result should be 0 if EITHER mask is 0 OR overlay is 0.
			# This is equivalent to logical AND if 0=False, but here 0 is ink.
			# So it's min(mask, overlay) or bitwise AND.
			
			current_region = self.mask[y0:y1, x0:x1]
			merged_region = cv2.bitwise_and(current_region, ov_region)
			self.mask[y0:y1, x0:x1] = merged_region
			
		self.overlay_active = False
		self.overlay_mask = None
		self.btn_merge_overlay.config(state="disabled")
		self.set_status("Overlay merged.")
		self._refresh_display()

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
			# Merge binary mask with gray layer for export
			merged = self.mask.copy()
			if self.gray_mask is not None:
				merged = np.where(self.gray_mask > 0, 128, merged)
			ok = cv2.imwrite(path, merged)
			if not ok: raise RuntimeError("cv2.imwrite returned False")
			self.set_status(f"Saved: {os.path.basename(path)}")
		except Exception as e:
			messagebox.showerror("Save error", str(e))

	def save_gray_mask(self):
		if self.gray_mask is None: messagebox.showinfo("Nothing to save","No gray mask present"); return
		# Default save location to EditedImages and ensure it exists
		base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
		edited_dir = os.path.join(base_dir, 'EditedImages')
		try:
			os.makedirs(edited_dir, exist_ok=True)
		except Exception:
			pass
		path = filedialog.asksaveasfilename(parent=self.winfo_toplevel(), title="Save gray mask", initialdir=edited_dir, defaultextension=".png",
									filetypes=[("PNG","*.png"),("TIFF","*.tif"),("JPEG","*.jpg")])
		if not path: return
		try:
			ok = cv2.imwrite(path, self.gray_mask)
			if not ok: raise RuntimeError("cv2.imwrite returned False")
			self.set_status(f"Saved gray mask: {os.path.basename(path)}")
		except Exception as e:
			messagebox.showerror("Save error", str(e))

	def undo(self):
		if not self._undo_stack and not self._gray_undo_stack: return
		try:
			if self._undo_stack:
				self.mask = self._undo_stack.pop()
			if self._gray_undo_stack:
				self.gray_mask = self._gray_undo_stack.pop()
			self._labels_dirty = True
			self._refresh_display()
			self.set_status("Undid last stroke")
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
			if self.gray_mask is not None:
				self._gray_undo_stack.append(self.gray_mask.copy())
				if len(self._gray_undo_stack) > 20: self._gray_undo_stack = self._gray_undo_stack[-20:]
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

	def run_magic2_tester(self):
		if self.mask is None: return
		
		# Toggle off if already active
		if self.magic2_active:
			self.magic2_active = False
			self.magic2_labels = set()
			self._refresh_display()
			self.set_status("Magic2 Tester deactivated")
			return

		try:
			# Ensure labels are computed
			if self._labels_cache is None or self._labels_dirty:
				bw_full = (self.mask == 0).astype(np.uint8)
				num, labels_full = cv2.connectedComponents(bw_full, connectivity=8)
				# Generate colors (same logic as in _refresh_display)
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
			
			labels_full, _ = self._labels_cache
			
			h, w = self.mask.shape[:2]
			mid_x = int(w // 2)
			
			# Find labels along the vertical line at mid_x
			col_labels = labels_full[:, mid_x]
			touched = set(np.unique(col_labels)) - {0}
			
			self.magic2_active = True
			self.magic2_labels = touched
			self._refresh_display()
			self.set_status(f"Magic2: Highlighted {len(touched)} fringes touching the center line")
			
		except Exception as e:
			print(f"Magic2 error: {e}")
			self.set_status("Magic2 Tester failed")

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
		
		# Get gray layer region for compositing
		region_gray = None
		if self.gray_mask is not None:
			try:
				region_gray = self.gray_mask[iy0:iy1, ix0:ix1]
			except Exception:
				region_gray = None
		
		colorize = bool(getattr(self, 'show_components_var', tk.BooleanVar(value=False)).get())
		magic_active = getattr(self, 'magic2_active', False)
		comp_rgb = None
		if (colorize or magic_active) and self.mask is not None:
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
					
					if magic_active:
						h, w = self.mask.shape[:2]
						mid_x = int(w // 2)
						col_labels = labels_full[:, mid_x]
						self.magic2_labels = set(np.unique(col_labels)) - {0}

				labels_full, colors = self._labels_cache
				labels_crop = labels_full[iy0:iy1, ix0:ix1]
				
				if magic_active:
					if colorize:
						comp_rgb = colors[labels_crop].copy()
					else:
						# White bg, Black fringes
						h_c, w_c = labels_crop.shape
						comp_rgb = np.full((h_c, w_c, 3), 255, dtype=np.uint8)
						comp_rgb[labels_crop > 0] = [0, 0, 0]
					
					# Apply highlights
					if self.magic2_labels:
						mask_hl = np.isin(labels_crop, list(self.magic2_labels))
						comp_rgb[mask_hl] = [255, 0, 255]
				elif colorize:
					comp_rgb = colors[labels_crop]
			except Exception:
				comp_rgb = None
		
		# Prepare background image if available
		bg_pil = None
		if self._bg is not None:
			try:
				bg_region = self._bg[iy0:iy1, ix0:ix1]
				bg_img = Image.fromarray(bg_region, mode="L").resize((dst_w, dst_h), Image.BILINEAR)
				factor = float(self.bg_brightness.get()) if self.bg_brightness is not None else 1.0
				factor = max(1.0, min(100.0, factor))
				arr = np.asarray(bg_img, dtype=np.float32)
				arr = np.clip(arr * factor, 0, 255).astype(np.uint8)
				bg_pil = Image.fromarray(arr, mode="L").convert("RGB")
			except Exception:
				bg_pil = None

		if comp_rgb is not None:
			try: 
				comp_pil = Image.fromarray(comp_rgb, mode="RGB").resize((dst_w, dst_h), Image.NEAREST)
				if bg_pil is not None:
					# Create mask from components (where not white background)
					# Assuming background in comp_rgb is white (255,255,255)
					comp_arr = np.array(comp_pil)
					# Create mask: True where pixel is NOT white
					is_not_white = np.any(comp_arr != [255, 255, 255], axis=2)
					
					# Start with background
					pil_img = bg_pil.copy()
					pil_arr = np.array(pil_img)
					
					# Overlay components
					pil_arr[is_not_white] = comp_arr[is_not_white]
					pil_img = Image.fromarray(pil_arr)
				else:
					pil_img = comp_pil
			except Exception: 
				pil_img = Image.fromarray(region_mask, mode="L").resize((dst_w, dst_h), Image.NEAREST)
		elif bg_pil is not None:
			# Standard display with background
			mask_img = Image.fromarray(region_mask, mode="L").resize((dst_w, dst_h), Image.NEAREST)
			
			# Convert mask to RGBA for compositing or just use L
			# Here we stick to the logic: Background + Black Fringes
			
			result = bg_pil.convert("L") # Convert back to L for existing logic or keep RGB?
			# Existing logic used L, let's stick to L for consistency unless we want color
			# But wait, if we want to support color overlays later, RGB is better.
			# Let's keep it simple and match previous behavior for non-color-comp mode
			
			black_L = Image.new("L", (dst_w, dst_h), 0)
			mask_inv = ImageOps.invert(mask_img)
			
			# Apply opacity
			opacity = float(self.fringe_opacity.get()) if hasattr(self, 'fringe_opacity') else 1.0
			if opacity < 1.0:
				arr = np.asarray(mask_inv, dtype=np.float32)
				arr = arr * opacity
				mask_inv = Image.fromarray(arr.astype(np.uint8), mode="L")

			result.paste(black_L, (0, 0), mask_inv)
			pil_img = result
		else:
			mask_img = Image.fromarray(region_mask, mode="L").resize((dst_w, dst_h), Image.NEAREST)
			
			# Apply opacity
			opacity = float(self.fringe_opacity.get()) if hasattr(self, 'fringe_opacity') else 1.0
			if opacity < 1.0:
				mask_inv = ImageOps.invert(mask_img)
				arr = np.asarray(mask_inv, dtype=np.float32)
				arr = arr * opacity
				mask_inv = Image.fromarray(arr.astype(np.uint8), mode="L")
				mask_img = ImageOps.invert(mask_inv)
			
			pil_img = mask_img
		
		# Composite gray layer on top if present
		if region_gray is not None:
			try:
				# Resize gray region to match display size
				gray_resized = Image.fromarray(region_gray, mode="L").resize((dst_w, dst_h), Image.NEAREST)
				gray_arr = np.array(gray_resized)
				
				# Convert pil_img to RGB if it's grayscale so we can show gray as actual gray
				if pil_img.mode == "L":
					pil_img = pil_img.convert("RGB")
				
				img_arr = np.array(pil_img)
				
				# Where gray_mask is present (255), overlay gray color (128, 128, 128)
				gray_overlay = gray_arr > 0
				if len(img_arr.shape) == 3:  # RGB
					img_arr[gray_overlay] = [128, 128, 128]
				else:  # Grayscale (shouldn't happen but handle it)
					img_arr[gray_overlay] = 128
				
				pil_img = Image.fromarray(img_arr)
			except Exception:
				pass  # If compositing fails, just show the base image
		
		# Draw Overlay if active
		if self.overlay_active and self.overlay_mask is not None:
			try:
				# Overlay position in image coords
				ox, oy = self.overlay_pos
				oh, ow = self.overlay_mask.shape[:2]
				
				# Visible region in image coords: ix0, iy0, ix1, iy1
				# Intersection of overlay and visible region
				inter_x0 = max(ix0, ox)
				inter_y0 = max(iy0, oy)
				inter_x1 = min(ix1, ox + ow)
				inter_y1 = min(iy1, oy + oh)
				
				if inter_x1 > inter_x0 and inter_y1 > inter_y0:
					# Slice from overlay
					sx0 = inter_x0 - ox
					sy0 = inter_y0 - oy
					sx1 = sx0 + (inter_x1 - inter_x0)
					sy1 = sy0 + (inter_y1 - inter_y0)
					
					ov_sub = self.overlay_mask[sy0:sy1, sx0:sx1]
					
					# Scale to display size
					# Display region corresponds to [inter_x0:inter_x1, inter_y0:inter_y1]
					# Its position on canvas relative to vx0, vy0 is:
					# dx = (inter_x0 - ix0) * scale
					# dy = (inter_y0 - iy0) * scale
					# w = (inter_x1 - inter_x0) * scale
					# h = (inter_y1 - inter_y0) * scale
					
					scale = self._scale if self._scale > 0 else 1.0
					
					# Calculate destination rect on the pil_img (which is size dst_w x dst_h)
					# pil_img top-left corresponds to ix0, iy0
					
					dest_x = int((inter_x0 - ix0) * scale)
					dest_y = int((inter_y0 - iy0) * scale)
					dest_w = int((inter_x1 - inter_x0) * scale)
					dest_h = int((inter_y1 - inter_y0) * scale)
					
					if dest_w > 0 and dest_h > 0:
						# Resize overlay slice
						ov_img = Image.fromarray(ov_sub, mode="L").resize((dest_w, dest_h), Image.NEAREST)
						
						# Create a colored version for visibility (e.g. Cyan for black pixels)
						# Overlay mask: 0 is ink, 255 is bg.
						# We want to show ink as Cyan, transparent elsewhere.
						
						ov_arr = np.array(ov_img)
						# Create RGBA
						rgba = np.zeros((dest_h, dest_w, 4), dtype=np.uint8)
						# Ink pixels (low value) -> Cyan (0, 255, 255) with alpha
						ink_mask = ov_arr < 128
						rgba[ink_mask] = [0, 255, 255, 180] # Cyan, semi-transparent
						
						ov_rgba = Image.fromarray(rgba, mode="RGBA")
						
						if pil_img.mode != "RGBA":
							pil_img = pil_img.convert("RGBA")
						
						pil_img.paste(ov_rgba, (dest_x, dest_y), ov_rgba)
			except Exception as e:
				print(f"Overlay render error: {e}")
				pass

		self._photo = ImageTk.PhotoImage(pil_img)
		self.canvas.create_image(vx0, vy0, anchor="nw", image=self._photo)
		self._img_topleft = (x0, y0)
		if self._last_mouse_pos is not None:
			try: self._update_cursor_circle(self._last_mouse_pos[0], self._last_mouse_pos[1])
			except Exception: pass

	# int-rounded canvas->image helper removed (unused)

	def _apply_brush(self, xi, yi):
		if self.mask is None: return
		if self.gray_mask is None:
			self.gray_mask = np.zeros(self.mask.shape, dtype=np.uint8)
		
		r = float(max(0.1, self.brush_var.get()))
		mode = self.mode_var.get()

		# Apply offset for current mode
		off_x, off_y = self._brush_offsets.get(mode, (0.0, 0.0))
		xi += off_x
		yi += off_y

		h, w = self.mask.shape[:2]
		y_min = int(max(0, np.floor(yi - r))); y_max = int(min(h - 1, np.ceil(yi + r)))
		x_min = int(max(0, np.floor(xi - r))); x_max = int(min(w - 1, np.ceil(xi + r)))
		
		if x_min > x_max or y_min > y_max:
			return

		yy, xx = np.ogrid[y_min:y_max + 1, x_min:x_max + 1]
		circle = (xx - xi) ** 2 + (yy - yi) ** 2 <= (r * r)
		
		if mode == "add":
			# Paint on binary mask
			sub = self.mask[y_min:y_max + 1, x_min:x_max + 1]
			sub[circle] = 0
		elif mode == "erase":
			# Erase from binary mask
			sub = self.mask[y_min:y_max + 1, x_min:x_max + 1]
			sub[circle] = 255
		elif mode == "add_gray":
			# Paint on gray layer (255 = gray present)
			sub_gray = self.gray_mask[y_min:y_max + 1, x_min:x_max + 1]
			sub_gray[circle] = 255
		elif mode == "erase_gray":
			# Erase from gray layer only
			sub_gray = self.gray_mask[y_min:y_max + 1, x_min:x_max + 1]
			sub_gray[circle] = 0
		
		self._labels_dirty = True

	def _apply_brush_line(self, x0, y0, x1, y1):
		if self.mask is None: return
		r = float(max(0.1, self.brush_var.get()))
		dx = float(x1 - x0); dy = float(y1 - y0)
		dist = float(np.hypot(dx, dy))
		# Use very small step for ultra-smooth strokes without gaps
		step_len = max(0.3, r * 0.15)
		steps = int(max(1, np.ceil(dist / step_len)))
		if steps <= 1:
			self._apply_brush(x1, y1); return
		# Paint all intermediate points
		for i in range(steps + 1):
			t = i / float(steps)
			xi = x0 + dx * t; yi = y0 + dy * t
			self._apply_brush(xi, yi)

	def _on_paint_start(self, event):
		if self.mask is None: return
		try:
			self._undo_stack.append(self.mask.copy())
			if len(self._undo_stack) > 20: self._undo_stack = self._undo_stack[-20:]
			if self.gray_mask is not None:
				self._gray_undo_stack.append(self.gray_mask.copy())
				if len(self._gray_undo_stack) > 20: self._gray_undo_stack = self._gray_undo_stack[-20:]
		except Exception: pass
		pt = self._canvas_to_image_coords_f(event.x, event.y)
		if pt is not None:
			xi, yi = pt
			try: self._last_paint_xy = (xi, yi)
			except Exception: self._last_paint_xy = None
			self._apply_brush(xi, yi)
			self._painting = True
			self._paint_update_counter = 0
			self._refresh_display()
		self._last_mouse_pos = (event.x, event.y); self._update_cursor_circle(event.x, event.y)

	def _on_paint_move(self, event):
		if self.mask is None or not self._painting: return
		pt = self._canvas_to_image_coords_f(event.x, event.y)
		if pt is not None:
			xi, yi = pt; lp = getattr(self, "_last_paint_xy", None)
			if lp is None: self._apply_brush(xi, yi)
			else: self._apply_brush_line(lp[0], lp[1], xi, yi)
			self._last_paint_xy = (xi, yi)
			# Update display every 3 paint events to reduce lag
			self._paint_update_counter = getattr(self, '_paint_update_counter', 0) + 1
			if self._paint_update_counter >= 3:
				self._refresh_display()
				self._paint_update_counter = 0
		self._last_mouse_pos = (event.x, event.y); self._update_cursor_circle(event.x, event.y)

	def _on_paint_end(self, event):
		self._painting = False
		self._last_paint_xy = None
		# Force final refresh to show complete stroke
		if hasattr(self, '_paint_update_counter'):
			self._refresh_display()
			self._paint_update_counter = 0

	def _canvas_to_image_coords_f(self, x_canvas, y_canvas):
		if self.mask is None: return None
		x0, y0 = getattr(self, "_img_topleft", (0, 0))
		scale = self._scale if self._scale > 0 else 1.0
		xi = (x_canvas - x0) / scale; yi = (y_canvas - y0) / scale
		# Removed bounds check to allow drawing from outside
		return xi, yi

	# Global paint handlers removed (unused)

	def _nudge_overlay(self, dx, dy):
		if not self.overlay_active or self.overlay_mask is None: return
		ox, oy = self.overlay_pos
		self.overlay_pos = (ox + dx, oy + dy)
		self._refresh_display(False)

	def _on_pan_start(self, event):
		# Check for overlay drag first
		# Require Control key (state & 4) to move overlay
		is_ctrl = (event.state & 0x4) != 0
		if is_ctrl and self.overlay_active and self.overlay_mask is not None:
			ox, oy = self.overlay_pos
			oh, ow = self.overlay_mask.shape[:2]
			scale = self._scale if self._scale > 0 else 1.0
			# self._offset is the canvas coordinate of the image top-left (0,0)
			img_x0, img_y0 = self._offset if self._offset is not None else (0, 0)
			
			# Overlay rect on canvas
			cv_ox = img_x0 + ox * scale
			cv_oy = img_y0 + oy * scale
			cv_w = ow * scale
			cv_h = oh * scale
			
			if cv_ox <= event.x <= cv_ox + cv_w and cv_oy <= event.y <= cv_oy + cv_h:
				self._moving_overlay = True
				self._pan_start = (event.x, event.y)
				self._overlay_start_pos = (ox, oy)
				return

		self._panning = True
		self._pan_start = (event.x, event.y)
		self._offset_start = self._offset if self._offset is not None else getattr(self, "_img_topleft", (0, 0))
		try: self.config(cursor='fleur')
		except Exception: pass

	def _on_pan_move(self, event):
		if self._moving_overlay:
			sx, sy = self._pan_start
			dx = event.x - sx
			dy = event.y - sy
			
			scale = self._scale if self._scale > 0 else 1.0
			
			# Convert delta to image coords
			idx = dx / scale
			idy = dy / scale
			
			ox_start, oy_start = self._overlay_start_pos
			self.overlay_pos = (int(ox_start + idx), int(oy_start + idy))
			self._refresh_display(False)
			return

		if not self._panning: return
		sx, sy = self._pan_start; dx = event.x - sx; dy = event.y - sy
		ox0, oy0 = self._offset_start; self._offset = (ox0 + dx, oy0 + dy)
		self._refresh_display(False)
		self._last_mouse_pos = (event.x, event.y); self._update_cursor_circle(event.x, event.y)

	def _on_pan_end(self, event):
		if self._moving_overlay:
			self._moving_overlay = False
			return

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
		
		mode = self.mode_var.get()
		scale = self._scale if self._scale > 0 else 1.0
		
		# Apply offset to visual cursor
		off_x, off_y = self._cursor_offsets.get(mode, (0.0, 0.0))
		# Offset is in image pixels, convert to canvas pixels
		# Canvas x = x0 + xi * scale
		# We have canvas x, y. We want to shift them by offset * scale
		x += off_x * scale
		y += off_y * scale

		r_img = float(max(0.1, self.brush_var.get()))
		# For erase mode, make the visual cursor 1 pixel smaller than the actual brush
		if mode == 'erase':
			r_img = max(0.1, r_img - 0.3)

		r_canvas = max(1.0, (r_img * scale))
		x0 = x - r_canvas; y0 = y - r_canvas; x1 = x + r_canvas; y1 = y + r_canvas
		
		if mode == "add":
			color = "#00ff88"
		elif mode == "add_gray":
			color = "#888888"
		elif mode == "erase_gray":
			color = "#ffaa00"
		else:  # erase (remove black)
			color = "#ff5555"
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
		# Support Windows (delta) and X11 (<Button-4>/<Button-5>) events
		delta = int(getattr(event, "delta", 0))
		if delta == 0:
			# Fallback for Linux where MouseWheel isn't delivered; use event.num
			num = getattr(event, 'num', None)
			if num == 4: delta = 120
			elif num == 5: delta = -120
		if delta == 0: return
		zoom_factor = 1.1 if delta > 0 else 0.9
		old_zoom = self._zoom; new_zoom = max(0.1, min(64.0, old_zoom * zoom_factor))
		if abs(new_zoom - old_zoom) < 1e-6: return
		x_c = event.x; y_c = event.y
		s_before = max(1e-6, self._base_scale * old_zoom); s_after = max(1e-6, self._base_scale * new_zoom)
		ox, oy = self._offset if self._offset is not None else getattr(self, "_img_topleft", (0, 0))
		x_img = (x_c - ox) / s_before; y_img = (y_c - oy) / s_before
		self._zoom = new_zoom; new_ox = int(round(x_c - x_img * s_after)); new_oy = int(round(y_c - y_img * s_after))
		self._offset = (new_ox, new_oy); self._refresh_display(False)
		self._last_mouse_pos = (x_c, y_c); self._update_cursor_circle(x_c, y_c)

	def _on_ctrl_mouse_wheel(self, event):
		# Brush size adjust with Ctrl+Wheel (Windows) or Control-Button-4/5 (X11)
		delta = int(getattr(event, "delta", 0))
		if delta == 0:
			num = getattr(event, 'num', None)
			if num == 4: delta = 120
			elif num == 5: delta = -120
		factor = 1.1 if delta > 0 else 0.9
		new_size = float(self.brush_var.get()) * factor; new_size = max(0.1, min(1000.0, new_size))
		self.brush_var.set(new_size); self._last_mouse_pos = (event.x, event.y); self._update_cursor_circle(event.x, event.y)

	def _on_bg_brightness_changed(self, _=None):
		self._refresh_display(False)

	def _on_fringe_opacity_changed(self, _=None):
		self._refresh_display(False)

