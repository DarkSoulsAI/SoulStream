import os
import glob
import cv2
import numpy as np

PROCESS_W = 240
PREVIEW_W, PREVIEW_H = 160, 120


class ImageSource:
    def __init__(self, image_dir, screen_w=1280, screen_h=720):
        self._screen_w = screen_w
        self._screen_h = screen_h
        self._image_dir = image_dir

        # Discover all images
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
        self._paths = []
        for ext in exts:
            self._paths.extend(glob.glob(os.path.join(image_dir, ext)))
        self._paths.sort()
        self._index = 0

        # Prefer darksouls1.jpg as starting image if present
        for i, p in enumerate(self._paths):
            if os.path.basename(p).lower() == "darksouls1.jpg":
                self._index = i
                break

        # Grid dimensions (set by _load)
        self.grid_w = 0
        self.grid_h = 0

        # Processed data
        self._spawn_probs = None
        self._color_map = None
        self._preview = np.zeros((PREVIEW_H, PREVIEW_W, 3), dtype=np.uint8)

        # Dummy data for ModeController compat
        self._brightness = None
        self._weight_map = None

        if self._paths:
            self._load(self._paths[self._index])

    @property
    def image_name(self):
        if not self._paths:
            return "(no images)"
        return os.path.basename(self._paths[self._index])

    @property
    def image_count(self):
        return len(self._paths)

    def next_image(self):
        if not self._paths:
            return
        self._index = (self._index + 1) % len(self._paths)
        self._load(self._paths[self._index])

    def prev_image(self):
        if not self._paths:
            return
        self._index = (self._index - 1) % len(self._paths)
        self._load(self._paths[self._index])

    def _load(self, path):
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            return

        # Resize to fit screen while maintaining aspect ratio
        h, w = bgr.shape[:2]
        scale = min(self._screen_w / w, self._screen_h / h)
        fit_w = int(w * scale)
        fit_h = int(h * scale)
        bgr_fit = cv2.resize(bgr, (fit_w, fit_h), interpolation=cv2.INTER_AREA)

        # Compute processing dimensions (PROCESS_W wide, proportional height)
        proc_h = int(PROCESS_W * fit_h / fit_w)
        bgr_proc = cv2.resize(bgr_fit, (PROCESS_W, proc_h), interpolation=cv2.INTER_AREA)

        self.grid_w = PROCESS_W
        self.grid_h = proc_h
        self._fit_w = fit_w
        self._fit_h = fit_h

        # Convert to RGB and grayscale
        rgb_proc = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr_proc, cv2.COLOR_BGR2GRAY)

        # --- Processing pipeline ---

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

        # Sobel gradient magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_max = sobel_mag.max()
        if sobel_max > 0:
            sobel_mag /= sobel_max
        sobel = sobel_mag.astype(np.float32)

        # Brightness from grayscale, floor at 0.03
        brightness = np.maximum(gray.astype(np.float32) / 255.0, 0.03)

        # Combined spawn weight
        weight_map = edges * 0.5 + sobel * 0.3 + brightness * 0.2

        # Pre-flatten and normalize to probability distribution
        flat = weight_map.ravel()
        total = flat.sum()
        if total < 1e-6:
            flat = np.ones_like(flat)
            total = flat.sum()
        self._spawn_probs = flat / total

        # Store RGB color map (float32)
        self._color_map = rgb_proc.astype(np.float32) / 255.0

        # Store for ModeController compat
        self._brightness = brightness
        self._weight_map = weight_map

        # Preview image (160x120)
        self._preview = cv2.resize(rgb_proc, (PREVIEW_W, PREVIEW_H),
                                   interpolation=cv2.INTER_AREA)

    def get_spawn_indices(self, n):
        if self._spawn_probs is None:
            return np.zeros(n, dtype=np.int32), np.zeros(n, dtype=np.int32)
        indices = np.random.choice(len(self._spawn_probs), size=n, replace=True,
                                   p=self._spawn_probs)
        gy, gx = np.unravel_index(indices, (self.grid_h, self.grid_w))
        return gy.astype(np.int32), gx.astype(np.int32)

    def sample_colors(self, gy, gx):
        if self._color_map is None:
            n = len(gy)
            return np.zeros(n, np.float32), np.zeros(n, np.float32), np.zeros(n, np.float32)
        gy_c = np.clip(gy, 0, self.grid_h - 1)
        gx_c = np.clip(gx, 0, self.grid_w - 1)
        r = self._color_map[gy_c, gx_c, 0]
        g = self._color_map[gy_c, gx_c, 1]
        b = self._color_map[gy_c, gx_c, 2]
        return r.astype(np.float32), g.astype(np.float32), b.astype(np.float32)

    def grid_to_ndc(self, gy, gx):
        # Map grid coords to NDC, centered on screen
        # Image is centered within 1280x720, so we need to account for letterboxing
        # Grid cell size in screen pixels
        cell_w = self._fit_w / self.grid_w
        cell_h = self._fit_h / self.grid_h

        # Screen pixel position (centered)
        offset_x = (self._screen_w - self._fit_w) / 2.0
        offset_y = (self._screen_h - self._fit_h) / 2.0

        px = offset_x + gx.astype(np.float32) * cell_w + cell_w * 0.5
        py = offset_y + gy.astype(np.float32) * cell_h + cell_h * 0.5

        # Sub-cell jitter
        n = len(gx)
        px += np.random.uniform(-cell_w * 0.5, cell_w * 0.5, n).astype(np.float32)
        py += np.random.uniform(-cell_h * 0.5, cell_h * 0.5, n).astype(np.float32)

        # Convert to NDC (-1..1)
        nx = (px / self._screen_w) * 2.0 - 1.0
        ny = 1.0 - (py / self._screen_h) * 2.0  # flip Y (screen top = NDC +1)

        return nx.astype(np.float32), ny.astype(np.float32)

    def get_preview(self):
        return self._preview.copy()

    def get_data(self):
        if self._brightness is None:
            z = np.zeros((1, 1), dtype=np.float32)
            return z, z, 0.0
        return self._brightness.copy(), self._weight_map.copy(), 0.0

    def stop(self):
        pass
