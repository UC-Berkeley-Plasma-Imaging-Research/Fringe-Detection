import cv2
import numpy as np
from skimage.morphology import skeletonize, remove_small_objects


def binarize(gray, method="Otsu", thresh=128, invert=False, blur=0):
    g = gray
    if blur > 0:
        k = int(2 * round(blur / 2) + 1)
        g = cv2.GaussianBlur(g, (k, k), 0)
    if method == "Otsu":
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == "Adaptive":
        bksz = int(thresh) | 1
        bksz = max(3, min(151, bksz))
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, bksz, 2)
    else:
        _, bw = cv2.threshold(g, int(thresh), 255, cv2.THRESH_BINARY)
    if invert:
        bw = 255 - bw
    return bw


def line_kernel(length, thickness=1, angle_deg=0):
    L = max(3, int(length)); T = max(1, int(thickness))
    size = int(np.ceil(np.sqrt(2) * L)) + 4
    canv = np.zeros((size, size), np.uint8); c = size // 2
    cv2.line(canv, (c - L // 2, c), (c + L // 2, c), 255, T)
    M = cv2.getRotationMatrix2D((c, c), angle_deg, 1.0)
    rot = cv2.warpAffine(canv, M, (size, size), flags=cv2.INTER_NEAREST, borderValue=0)
    rot = (rot > 0).astype(np.uint8)
    return rot


def oriented_opening(bw01, length, thickness, max_angle=8.0, step=2.0):
    angles = np.arange(-float(max_angle), float(max_angle) + 1e-6, float(step))
    out = np.zeros_like(bw01, np.uint8)
    for a in angles:
        k = line_kernel(length, thickness, a)
        er = cv2.erode(bw01, k, iterations=1)
        op = cv2.dilate(er, k, iterations=1)
        out = np.maximum(out, op)
    return out


def overlay_mask_on_gray(gray, mask01, line_alpha=0.85, bg_fade=0.4, bg_to='white', line_color=(255, 0, 0)):
    # gray: uint8, mask01: 0/1 uint8
    base = gray.astype(np.float32)
    target = 255.0 if bg_to == 'white' else 0.0
    base = (1.0 - bg_fade) * base + bg_fade * target
    base = np.clip(base, 0, 255).astype(np.uint8)
    base_bgr = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    color = np.zeros_like(base_bgr)
    # line_color provided as RGB; convert to BGR for OpenCV
    color[..., 0] = line_color[2]
    color[..., 1] = line_color[1]
    color[..., 2] = line_color[0]
    m = mask01.astype(bool)
    out = base_bgr.copy()
    out[m] = (line_alpha * color[m] + (1.0 - line_alpha) * base_bgr[m]).astype(np.uint8)
    return out
