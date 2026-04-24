"""Pytest configuration: path setup and cv2 stub for environments with broken cv2."""
import sys
import os
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ---------------------------------------------------------------------------
# Install a numpy-based cv2 stub when the real cv2 is broken or absent.
# This lets test_image_source.py run without OpenCV being importable.
# ---------------------------------------------------------------------------
def _try_import_cv2():
    try:
        import cv2 as _real
        _real.imread   # probe a symbol to surface DLL errors early
        return True
    except Exception:
        return False


if not _try_import_cv2():
    _cv2 = types.ModuleType("cv2")

    # Constants used by image_source.py
    _cv2.IMREAD_COLOR = 1
    _cv2.INTER_AREA = 3
    _cv2.COLOR_BGR2RGB = 4
    _cv2.COLOR_BGR2GRAY = 6
    _cv2.CV_64F = 6

    def _resize(img, dsize, interpolation=None):
        """Nearest-neighbour resize matching OpenCV's (width, height) dsize order."""
        w, h = dsize
        if img.ndim == 3:
            src_h, src_w = img.shape[:2]
            ys = (np.arange(h) * src_h / h).astype(int).clip(0, src_h - 1)
            xs = (np.arange(w) * src_w / w).astype(int).clip(0, src_w - 1)
            return img[np.ix_(ys, xs)].copy()
        else:
            src_h, src_w = img.shape
            ys = (np.arange(h) * src_h / h).astype(int).clip(0, src_h - 1)
            xs = (np.arange(w) * src_w / w).astype(int).clip(0, src_w - 1)
            return img[np.ix_(ys, xs)].copy()

    def _cvtColor(img, code):
        if code == 4:  # BGR → RGB
            return img[:, :, ::-1].copy()
        if code == 6:  # BGR → GRAY  (0.299R + 0.587G + 0.114B, OpenCV formula)
            b = img[:, :, 0].astype(np.float32)
            g = img[:, :, 1].astype(np.float32)
            r = img[:, :, 2].astype(np.float32)
            return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)
        return img.copy()

    def _Canny(img, t1, t2):
        return np.zeros_like(img, dtype=np.uint8)

    def _Sobel(img, ddepth, dx, dy, ksize=3):
        return np.zeros(img.shape, dtype=np.float64)

    # Default imread returns None; tests override with patch('cv2.imread')
    _cv2.imread = lambda path, flags=None: None
    _cv2.resize = _resize
    _cv2.cvtColor = _cvtColor
    _cv2.Canny = _Canny
    _cv2.Sobel = _Sobel

    sys.modules["cv2"] = _cv2
