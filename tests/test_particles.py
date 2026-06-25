"""Unit tests for ParticleSystem (src/particles.py).

All tests run without GPU, camera, or MediaPipe — pure numpy logic.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from particles import MAX_PARTICLES, SPAWN_PER_FRAME, ParticleSystem


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_mock_image_source(grid_h=50, grid_w=60):
    src = MagicMock()
    src.grid_h = grid_h
    src.grid_w = grid_w

    def get_spawn_indices(n):
        return (
            np.random.randint(0, grid_h, n).astype(np.int32),
            np.random.randint(0, grid_w, n).astype(np.int32),
        )

    def grid_to_ndc(gy, gx):
        n = len(gy)
        return (
            np.random.uniform(-1, 1, n).astype(np.float32),
            np.random.uniform(-1, 1, n).astype(np.float32),
        )

    def sample_colors(gy, gx):
        n = len(gy)
        return (
            np.random.uniform(0, 1, n).astype(np.float32),
            np.random.uniform(0, 1, n).astype(np.float32),
            np.random.uniform(0, 1, n).astype(np.float32),
        )

    src.get_spawn_indices.side_effect = get_spawn_indices
    src.grid_to_ndc.side_effect = grid_to_ndc
    src.sample_colors.side_effect = sample_colors
    return src


def _ps_with_particles(n=100):
    ps = ParticleSystem()
    ps.count = n
    ps.pos_x[:n] = np.random.uniform(-0.5, 0.5, n).astype(np.float32)
    ps.pos_y[:n] = np.random.uniform(-0.5, 0.5, n).astype(np.float32)
    ps.prev_x[:n] = ps.pos_x[:n].copy()
    ps.prev_y[:n] = ps.pos_y[:n].copy()
    ps.vel_x[:n] = np.random.uniform(-0.1, 0.1, n).astype(np.float32)
    ps.vel_y[:n] = np.random.uniform(0.01, 0.1, n).astype(np.float32)
    ps.life[:n] = np.random.uniform(1.0, 3.0, n).astype(np.float32)
    ps.max_life[:n] = ps.life[:n].copy()
    ps.color_r[:n] = np.random.uniform(0, 1, n).astype(np.float32)
    ps.color_g[:n] = np.random.uniform(0, 1, n).astype(np.float32)
    ps.color_b[:n] = np.random.uniform(0, 1, n).astype(np.float32)
    ps._phase[:n] = np.random.uniform(0, np.pi * 2, n).astype(np.float32)
    return ps


def _fake_camera_module(w=80, h=60):
    cam = types.ModuleType("camera")
    cam.CAPTURE_W = w
    cam.CAPTURE_H = h
    return cam


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestParticleSystemInit:
    def test_count_zero(self):
        assert ParticleSystem().count == 0

    def test_array_lengths(self):
        ps = ParticleSystem()
        for arr in (ps.pos_x, ps.pos_y, ps.vel_x, ps.vel_y,
                    ps.life, ps.max_life, ps.color_r, ps.color_g, ps.color_b,
                    ps._phase):
            assert len(arr) == MAX_PARTICLES

    def test_arrays_float32(self):
        ps = ParticleSystem()
        for arr in (ps.pos_x, ps.pos_y, ps.vel_x, ps.vel_y,
                    ps.life, ps.max_life, ps.color_r, ps.color_g, ps.color_b):
            assert arr.dtype == np.float32

    def test_time_zero(self):
        assert ParticleSystem()._time == 0.0


# ---------------------------------------------------------------------------
# spawn (image-source mode)
# ---------------------------------------------------------------------------

class TestSpawn:
    def test_humanity_increases_count(self):
        ps = ParticleSystem()
        ps.spawn(make_mock_image_source(), is_ember=False)
        assert 0 < ps.count <= SPAWN_PER_FRAME

    def test_ember_increases_count(self):
        ps = ParticleSystem()
        ps.spawn(make_mock_image_source(), is_ember=True)
        assert ps.count > 0

    def test_humanity_colors_in_range(self):
        ps = ParticleSystem()
        ps.spawn(make_mock_image_source(), is_ember=False)
        n = ps.count
        for ch in (ps.color_r, ps.color_g, ps.color_b):
            assert np.all(ch[:n] >= 0.0) and np.all(ch[:n] <= 1.0)

    def test_ember_colors_in_range(self):
        ps = ParticleSystem()
        ps.spawn(make_mock_image_source(), is_ember=True)
        n = ps.count
        for ch in (ps.color_r, ps.color_g, ps.color_b):
            assert np.all(ch[:n] >= 0.0) and np.all(ch[:n] <= 1.0)

    def test_life_positive_after_spawn(self):
        ps = ParticleSystem()
        ps.spawn(make_mock_image_source(), is_ember=False)
        assert np.all(ps.life[:ps.count] > 0.0)

    def test_max_life_equals_initial_life(self):
        ps = ParticleSystem()
        ps.spawn(make_mock_image_source(), is_ember=False)
        n = ps.count
        assert np.allclose(ps.life[:n], ps.max_life[:n])

    def test_respects_max_particles(self):
        ps = ParticleSystem()
        ps.count = MAX_PARTICLES
        ps.spawn(make_mock_image_source(), is_ember=False)
        assert ps.count == MAX_PARTICLES

    def test_multiple_spawns_accumulate(self):
        ps = ParticleSystem()
        src = make_mock_image_source()
        ps.spawn(src, is_ember=False)
        c1 = ps.count
        ps.spawn(src, is_ember=False)
        assert ps.count >= c1

    def test_ember_velocity_upward(self):
        ps = ParticleSystem()
        ps.spawn(make_mock_image_source(), is_ember=True)
        assert np.all(ps.vel_y[:ps.count] > 0.0)

    def test_humanity_velocity_gentle(self):
        ps = ParticleSystem()
        ps.spawn(make_mock_image_source(), is_ember=False)
        n = ps.count
        assert np.all(np.abs(ps.vel_x[:n]) < 0.02)


# ---------------------------------------------------------------------------
# spawn_palm_sparks
# ---------------------------------------------------------------------------

class TestSpawnPalmSparks:
    def test_spawns_15_particles(self):
        ps = ParticleSystem()
        ps.spawn_palm_sparks(0.0, 0.0)
        assert ps.count == 15

    def test_custom_count(self):
        ps = ParticleSystem()
        ps.spawn_palm_sparks(0.0, 0.0, count=32)
        assert ps.count == 32

    def test_positions_near_palm(self):
        ps = ParticleSystem()
        ps.spawn_palm_sparks(0.5, 0.3)
        n = ps.count
        assert np.all(np.abs(ps.pos_x[:n] - 0.5) <= 0.04)
        assert np.all(np.abs(ps.pos_y[:n] - 0.3) <= 0.04)

    def test_custom_spread(self):
        ps = ParticleSystem()
        ps.spawn_palm_sparks(0.5, 0.3, spread=0.01)
        n = ps.count
        assert np.all(np.abs(ps.pos_x[:n] - 0.5) <= 0.011)
        assert np.all(np.abs(ps.pos_y[:n] - 0.3) <= 0.011)

    def test_velocity_upward(self):
        ps = ParticleSystem()
        ps.spawn_palm_sparks(0.0, 0.0)
        assert np.all(ps.vel_y[:ps.count] > 0)

    def test_red_channel_full(self):
        ps = ParticleSystem()
        ps.spawn_palm_sparks(0.0, 0.0)
        assert np.all(ps.color_r[:ps.count] == 1.0)

    def test_respects_max_particles(self):
        ps = ParticleSystem()
        ps.count = MAX_PARTICLES
        ps.spawn_palm_sparks(0.0, 0.0)
        assert ps.count == MAX_PARTICLES

    def test_life_positive(self):
        ps = ParticleSystem()
        ps.spawn_palm_sparks(0.0, 0.0)
        assert np.all(ps.life[:ps.count] > 0)


# ---------------------------------------------------------------------------
# spawn_you_died
# ---------------------------------------------------------------------------

class TestSpawnYouDied:
    def test_spawns_300_by_default(self):
        ps = ParticleSystem()
        ps.spawn_you_died()
        assert ps.count == 300

    def test_respects_max_particles(self):
        ps = ParticleSystem()
        ps.count = MAX_PARTICLES - 100
        ps.spawn_you_died(n=300)
        assert ps.count <= MAX_PARTICLES

    def test_colors_are_red(self):
        ps = ParticleSystem()
        ps.spawn_you_died()
        n = ps.count
        assert np.all(ps.color_r[:n] >= 0.7)
        assert np.all(ps.color_g[:n] <= 0.08)
        assert np.all(ps.color_b[:n] <= 0.04)

    def test_velocity_downward(self):
        ps = ParticleSystem()
        ps.spawn_you_died()
        assert np.all(ps.vel_y[:ps.count] < 0)

    def test_custom_n(self):
        ps = ParticleSystem()
        ps.spawn_you_died(n=50)
        assert ps.count == 50


# ---------------------------------------------------------------------------
# spawn_praise_sun
# ---------------------------------------------------------------------------

class TestSpawnPraiseSun:
    def test_spawns_from_both_hands(self):
        ps = ParticleSystem()
        ps.spawn_praise_sun((-0.5, 0.5), (0.5, 0.5))
        assert ps.count > 0

    def test_radial_velocities_nonzero(self):
        ps = ParticleSystem()
        ps.spawn_praise_sun((0.0, 0.0), (0.0, 0.0))
        n = ps.count
        speed = np.sqrt(ps.vel_x[:n] ** 2 + ps.vel_y[:n] ** 2)
        assert np.all(speed > 0)

    def test_red_channel_full(self):
        ps = ParticleSystem()
        ps.spawn_praise_sun((0.0, 0.0), (0.0, 0.0))
        assert np.all(ps.color_r[:ps.count] == 1.0)

    def test_life_positive(self):
        ps = ParticleSystem()
        ps.spawn_praise_sun((0.0, 0.0), (0.0, 0.0))
        assert np.all(ps.life[:ps.count] > 0)


# ---------------------------------------------------------------------------
# spawn_camera
# ---------------------------------------------------------------------------

class TestSpawnCamera:
    def _call(self, ps, brightness, motion, is_ember=False, w=80, h=60):
        with patch.dict("sys.modules", {"camera": _fake_camera_module(w, h)}):
            ps.spawn_camera(brightness, motion, is_ember=is_ember)

    def test_increases_count(self):
        ps = ParticleSystem()
        h, w = 60, 80
        brightness = np.random.uniform(0.1, 1, (h, w)).astype(np.float32)
        motion = np.random.uniform(0.1, 1, (h, w)).astype(np.float32)
        self._call(ps, brightness, motion)
        assert ps.count > 0

    def test_zero_weights_no_spawn(self):
        ps = ParticleSystem()
        h, w = 60, 80
        brightness = np.zeros((h, w), dtype=np.float32)
        motion = np.zeros((h, w), dtype=np.float32)
        self._call(ps, brightness, motion)
        assert ps.count == 0

    def test_ember_mode_upward_velocity(self):
        ps = ParticleSystem()
        h, w = 60, 80
        brightness = np.ones((h, w), dtype=np.float32)
        motion = np.ones((h, w), dtype=np.float32)
        self._call(ps, brightness, motion, is_ember=True)
        assert np.all(ps.vel_y[:ps.count] > 0)

    def test_respects_max_particles(self):
        ps = ParticleSystem()
        ps.count = MAX_PARTICLES
        h, w = 60, 80
        brightness = np.ones((h, w), dtype=np.float32)
        motion = np.ones((h, w), dtype=np.float32)
        self._call(ps, brightness, motion)
        assert ps.count == MAX_PARTICLES


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------

class TestUpdate:
    def test_positions_change(self):
        ps = _ps_with_particles()
        x_before = ps.pos_x[:ps.count].copy()
        ps.update(0.016)
        assert not np.allclose(ps.pos_x[:ps.count], x_before)

    def test_life_decreases(self):
        ps = _ps_with_particles()
        life_before = ps.life[:ps.count].copy()
        n_before = ps.count
        ps.update(0.016)
        assert np.all(ps.life[:ps.count] < life_before[:ps.count])

    def test_dead_particles_removed(self):
        ps = _ps_with_particles(20)
        ps.life[:5] = -1.0
        ps.update(0.001)
        assert ps.count <= 15

    def test_time_advances(self):
        ps = _ps_with_particles()
        ps.update(0.016)
        assert ps._time == pytest.approx(0.016)

    def test_empty_system_no_error(self):
        ParticleSystem().update(0.016)

    def test_prev_position_saved(self):
        ps = _ps_with_particles()
        pos_before = ps.pos_x[:ps.count].copy()
        ps.update(0.016)
        assert np.allclose(ps.prev_x[:ps.count], pos_before[:ps.count])

    def test_flow_field_shifts_velocity(self):
        ps = _ps_with_particles(50)
        flow = np.zeros((2, 10, 10), dtype=np.float32)
        flow[0, :, :] = 1.0  # rightward
        vel_x_before = ps.vel_x[:ps.count].copy()
        ps.update(0.1, flow_field=flow)
        assert np.mean(ps.vel_x[:ps.count]) > np.mean(vel_x_before)

    def test_all_dead_clears_count(self):
        ps = _ps_with_particles(10)
        ps.life[:10] = -1.0
        ps.update(0.001)
        assert ps.count == 0


# ---------------------------------------------------------------------------
# kindle_nearby
# ---------------------------------------------------------------------------

class TestKindleNearby:
    def test_warms_nearby_colors(self):
        ps = ParticleSystem()
        n = 10
        ps.count = n
        ps.pos_x[:n] = 0.0
        ps.pos_y[:n] = 0.0
        ps.color_r[:n] = 0.0
        ps.color_g[:n] = 0.0
        ps.color_b[:n] = 0.0
        ps.vel_x[:n] = 0.0
        ps.vel_y[:n] = 0.0
        ps.kindle_nearby(0.0, 0.0, radius=0.3)
        assert np.all(ps.color_r[:n] > 0.0)

    def test_empty_system_no_error(self):
        ParticleSystem().kindle_nearby(0.0, 0.0)

    def test_far_particles_unchanged(self):
        ps = ParticleSystem()
        n = 5
        ps.count = n
        ps.pos_x[:n] = 5.0
        ps.pos_y[:n] = 5.0
        ps.color_r[:n] = 0.2
        ps.color_g[:n] = 0.2
        ps.color_b[:n] = 0.2
        ps.vel_x[:n] = 0.0
        ps.vel_y[:n] = 0.0
        ps.kindle_nearby(0.0, 0.0, radius=0.3)
        assert np.allclose(ps.color_r[:n], 0.2)
        assert np.allclose(ps.color_g[:n], 0.2)

    def test_pull_velocity_toward_palm(self):
        ps = ParticleSystem()
        n = 1
        ps.count = n
        ps.pos_x[:1] = 0.1  # just inside radius
        ps.pos_y[:1] = 0.0
        ps.vel_x[:1] = 0.0
        ps.vel_y[:1] = 0.0
        ps.kindle_nearby(0.0, 0.0, radius=0.3)
        # Particle is to the right of palm, so vel_x should be pulled left (negative)
        assert ps.vel_x[0] < 0.0


# ---------------------------------------------------------------------------
# pack_gpu / pack_gpu_trails
# ---------------------------------------------------------------------------

class TestPackGpu:
    def test_empty_returns_empty(self):
        assert len(ParticleSystem().pack_gpu()) == 0

    def test_shape_7_floats_per_particle(self):
        ps = _ps_with_particles(50)
        assert len(ps.pack_gpu()) == 50 * 7

    def test_alpha_in_01(self):
        ps = _ps_with_particles(50)
        alpha = ps.pack_gpu()[5::7]
        assert np.all(alpha >= 0.0) and np.all(alpha <= 1.0)

    def test_size_positive(self):
        ps = _ps_with_particles(50)
        size = ps.pack_gpu()[6::7]
        assert np.all(size > 0)

    def test_positions_match(self):
        ps = _ps_with_particles(10)
        buf = ps.pack_gpu()
        assert np.allclose(buf[0::7], ps.pos_x[:10])
        assert np.allclose(buf[1::7], ps.pos_y[:10])

    def test_trails_empty_returns_empty(self):
        assert len(ParticleSystem().pack_gpu_trails()) == 0

    def test_trails_shape_14_floats_per_particle(self):
        ps = _ps_with_particles(30)
        ps.prev_x[:30] = ps.pos_x[:30] - 0.01
        ps.prev_y[:30] = ps.pos_y[:30] - 0.01
        assert len(ps.pack_gpu_trails()) == 30 * 14

    def test_trails_tail_alpha_zero(self):
        ps = _ps_with_particles(20)
        ps.prev_x[:20] = ps.pos_x[:20]
        ps.prev_y[:20] = ps.pos_y[:20]
        buf = ps.pack_gpu_trails()
        tail_alpha = buf[5::14]
        assert np.all(tail_alpha == 0.0)

    def test_trails_head_positions_match(self):
        ps = _ps_with_particles(10)
        ps.prev_x[:10] = 0.0
        ps.prev_y[:10] = 0.0
        buf = ps.pack_gpu_trails()
        assert np.allclose(buf[7::14], ps.pos_x[:10])
        assert np.allclose(buf[8::14], ps.pos_y[:10])
