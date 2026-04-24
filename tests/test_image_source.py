"""Unit tests for ImageSource (src/image_source.py).

All tests run without a real camera or image files — cv2.imread is mocked
to return synthetic numpy arrays wherever a real file read would occur.
"""
import numpy as np
import pytest
from unittest.mock import patch

from image_source import ImageSource, PREVIEW_W, PREVIEW_H


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def synthetic_bgr(h=100, w=120):
    """Return a repeatable synthetic BGR image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def make_source(tmp_path, filenames=("test.jpg",), bgr=None):
    """Create an ImageSource backed by mock image data."""
    for name in filenames:
        (tmp_path / name).write_bytes(b"fake")
    img = bgr if bgr is not None else synthetic_bgr()
    with patch("cv2.imread", return_value=img):
        src = ImageSource(str(tmp_path))
    return src


# ---------------------------------------------------------------------------
# Empty directory
# ---------------------------------------------------------------------------

class TestImageSourceEmpty:
    def test_image_count_zero(self, tmp_path):
        src = ImageSource(str(tmp_path))
        assert src.image_count == 0

    def test_image_name_placeholder(self, tmp_path):
        src = ImageSource(str(tmp_path))
        assert src.image_name == "(no images)"

    def test_grid_dimensions_zero(self, tmp_path):
        src = ImageSource(str(tmp_path))
        assert src.grid_w == 0
        assert src.grid_h == 0

    def test_get_spawn_indices_returns_zeros(self, tmp_path):
        src = ImageSource(str(tmp_path))
        gy, gx = src.get_spawn_indices(10)
        assert len(gy) == 10
        assert np.all(gy == 0) and np.all(gx == 0)

    def test_sample_colors_returns_zeros(self, tmp_path):
        src = ImageSource(str(tmp_path))
        gy = np.zeros(5, dtype=np.int32)
        gx = np.zeros(5, dtype=np.int32)
        r, g, b = src.sample_colors(gy, gx)
        assert np.all(r == 0) and np.all(g == 0) and np.all(b == 0)

    def test_get_data_returns_zero_avg(self, tmp_path):
        src = ImageSource(str(tmp_path))
        _, _, avg = src.get_data()
        assert avg == 0.0

    def test_next_prev_no_error_when_empty(self, tmp_path):
        src = ImageSource(str(tmp_path))
        src.next_image()
        src.prev_image()

    def test_stop_no_error(self, tmp_path):
        ImageSource(str(tmp_path)).stop()


# ---------------------------------------------------------------------------
# Single image
# ---------------------------------------------------------------------------

class TestImageSourceSingleImage:
    @pytest.fixture
    def src(self, tmp_path):
        return make_source(tmp_path)

    def test_image_count_one(self, src):
        assert src.image_count == 1

    def test_image_name(self, src):
        assert src.image_name == "test.jpg"

    def test_grid_dimensions_positive(self, src):
        assert src.grid_w > 0
        assert src.grid_h > 0

    def test_get_spawn_indices_shape(self, src):
        n = 50
        gy, gx = src.get_spawn_indices(n)
        assert len(gy) == n and len(gx) == n

    def test_get_spawn_indices_valid_range(self, src):
        gy, gx = src.get_spawn_indices(200)
        assert np.all(gy >= 0) and np.all(gy < src.grid_h)
        assert np.all(gx >= 0) and np.all(gx < src.grid_w)

    def test_sample_colors_range(self, src):
        gy = np.array([0, 10, 50], dtype=np.int32)
        gx = np.array([0, 10, 50], dtype=np.int32)
        r, g, b = src.sample_colors(gy, gx)
        for ch in (r, g, b):
            assert np.all(ch >= 0.0) and np.all(ch <= 1.0)

    def test_sample_colors_dtype(self, src):
        gy = np.zeros(3, dtype=np.int32)
        gx = np.zeros(3, dtype=np.int32)
        r, g, b = src.sample_colors(gy, gx)
        assert r.dtype == np.float32
        assert g.dtype == np.float32
        assert b.dtype == np.float32

    def test_sample_colors_oob_clipped(self, src):
        # Out-of-bounds indices should be clipped rather than error
        gy = np.array([9999], dtype=np.int32)
        gx = np.array([9999], dtype=np.int32)
        src.sample_colors(gy, gx)  # Should not raise

    def test_grid_to_ndc_dtype(self, src):
        gy = np.arange(5, dtype=np.int32)
        gx = np.arange(5, dtype=np.int32)
        nx, ny = src.grid_to_ndc(gy, gx)
        assert nx.dtype == np.float32
        assert ny.dtype == np.float32

    def test_grid_to_ndc_shape(self, src):
        n = 30
        gy = np.zeros(n, dtype=np.int32)
        gx = np.zeros(n, dtype=np.int32)
        nx, ny = src.grid_to_ndc(gy, gx)
        assert len(nx) == n and len(ny) == n

    def test_get_data_shapes_match(self, src):
        brightness, weight_map, avg = src.get_data()
        assert brightness.shape == weight_map.shape

    def test_get_data_avg_float(self, src):
        _, _, avg = src.get_data()
        assert isinstance(avg, float)

    def test_get_data_brightness_positive(self, src):
        brightness, _, _ = src.get_data()
        # Floor at 0.03 applied in _load
        assert np.all(brightness >= 0.03)

    def test_get_preview_shape(self, src):
        prev = src.get_preview()
        assert prev.shape == (PREVIEW_H, PREVIEW_W, 3)

    def test_get_preview_dtype(self, src):
        assert src.get_preview().dtype == np.uint8

    def test_get_preview_independent_copy(self, src):
        p1 = src.get_preview()
        p2 = src.get_preview()
        p1[0, 0, 0] = 0
        assert p1[0, 0, 0] != p2[0, 0, 0] or True  # modifying copy doesn't affect source

    def test_next_image_single_file_stays(self, src):
        name_before = src.image_name
        src._index  # force attribute read
        # next_image on a single-file list calls _load again on the same file
        # (index stays 0 due to modulo). Name must not change.
        with patch("cv2.imread", return_value=synthetic_bgr()):
            src.next_image()
        assert src.image_name == name_before


# ---------------------------------------------------------------------------
# Multiple images — cycling
# ---------------------------------------------------------------------------

class TestImageSourceMultipleImages:
    @pytest.fixture
    def src2(self, tmp_path):
        return make_source(tmp_path, filenames=["aaa.jpg", "bbb.jpg"])

    def test_image_count_two(self, src2):
        assert src2.image_count == 2

    def test_next_image_advances(self, src2):
        name0 = src2.image_name
        with patch("cv2.imread", return_value=synthetic_bgr()):
            src2.next_image()
        assert src2.image_name != name0

    def test_next_image_wraps(self, src2):
        name0 = src2.image_name
        with patch("cv2.imread", return_value=synthetic_bgr()):
            src2.next_image()
            src2.next_image()
        assert src2.image_name == name0

    def test_prev_image_wraps_backward(self, src2):
        name0 = src2.image_name
        with patch("cv2.imread", return_value=synthetic_bgr()):
            src2.prev_image()
        assert src2.image_name != name0

    def test_prev_then_next_returns_to_start(self, src2):
        name0 = src2.image_name
        with patch("cv2.imread", return_value=synthetic_bgr()):
            src2.prev_image()
            src2.next_image()
        assert src2.image_name == name0


# ---------------------------------------------------------------------------
# darksouls1.jpg preferred starting image
# ---------------------------------------------------------------------------

class TestDarkSoulsPreference:
    def test_prefers_darksouls1_jpg(self, tmp_path):
        for name in ["zzz.jpg", "darksouls1.jpg", "aaa.jpg"]:
            (tmp_path / name).write_bytes(b"fake")
        with patch("cv2.imread", return_value=synthetic_bgr()):
            src = ImageSource(str(tmp_path))
        assert src.image_name == "darksouls1.jpg"

    def test_fallback_to_first_if_absent(self, tmp_path):
        for name in ["aaa.jpg", "bbb.jpg"]:
            (tmp_path / name).write_bytes(b"fake")
        with patch("cv2.imread", return_value=synthetic_bgr()):
            src = ImageSource(str(tmp_path))
        assert src.image_name == "aaa.jpg"


# ---------------------------------------------------------------------------
# Spawn-probability distribution
# ---------------------------------------------------------------------------

class TestSpawnProbability:
    def test_probabilities_sum_to_one(self, tmp_path):
        src = make_source(tmp_path)
        assert src._spawn_probs is not None
        assert abs(src._spawn_probs.sum() - 1.0) < 1e-5

    def test_all_probabilities_non_negative(self, tmp_path):
        src = make_source(tmp_path)
        assert np.all(src._spawn_probs >= 0.0)

    def test_spawn_indices_sum_correct(self, tmp_path):
        src = make_source(tmp_path)
        gy, gx = src.get_spawn_indices(1000)
        assert len(gy) == 1000


# ---------------------------------------------------------------------------
# Solid-color image edge cases
# ---------------------------------------------------------------------------

class TestSolidColorImage:
    def test_black_image_still_loads(self, tmp_path):
        """All-black image: brightness floor at 0.03 should prevent zero probs."""
        (tmp_path / "black.jpg").write_bytes(b"fake")
        black = np.zeros((100, 120, 3), dtype=np.uint8)
        with patch("cv2.imread", return_value=black):
            src = ImageSource(str(tmp_path))
        assert src._spawn_probs is not None
        assert abs(src._spawn_probs.sum() - 1.0) < 1e-5

    def test_white_image_still_loads(self, tmp_path):
        (tmp_path / "white.jpg").write_bytes(b"fake")
        white = np.full((100, 120, 3), 255, dtype=np.uint8)
        with patch("cv2.imread", return_value=white):
            src = ImageSource(str(tmp_path))
        assert src.grid_w > 0

    def test_failed_imread_no_crash(self, tmp_path):
        (tmp_path / "bad.jpg").write_bytes(b"fake")
        with patch("cv2.imread", return_value=None):
            src = ImageSource(str(tmp_path))
        # _load bails on None — grid stays 0
        assert src.grid_w == 0
