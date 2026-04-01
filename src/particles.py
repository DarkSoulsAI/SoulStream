import numpy as np

MAX_PARTICLES = 25000
SPAWN_PER_FRAME = 150


class ParticleSystem:
    def __init__(self):
        self.count = 0
        self.pos_x = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.pos_y = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.prev_x = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.prev_y = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.vel_x = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.vel_y = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.life = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.max_life = np.ones(MAX_PARTICLES, dtype=np.float32)
        self.color_r = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.color_g = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self.color_b = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self._phase = np.zeros(MAX_PARTICLES, dtype=np.float32)
        self._time = 0.0

    def spawn(self, image_source, is_ember):
        slots = MAX_PARTICLES - self.count
        if slots <= 0:
            return

        n = min(SPAWN_PER_FRAME, slots)

        # Sample grid positions from image source
        gy, gx = image_source.get_spawn_indices(n)

        # Convert to NDC with sub-cell jitter
        nx, ny = image_source.grid_to_ndc(gy, gx)

        # Sample base colors from the image
        cr, cg, cb = image_source.sample_colors(gy, gx)

        s = self.count
        e = s + n

        self.pos_x[s:e] = nx
        self.pos_y[s:e] = ny
        self.prev_x[s:e] = nx
        self.prev_y[s:e] = ny

        if is_ember:
            # Ember mode: warm-shift colors
            self.vel_x[s:e] = np.random.uniform(-0.06, 0.06, n).astype(np.float32)
            self.vel_y[s:e] = np.random.uniform(0.02, 0.12, n).astype(np.float32)

            # Warm-shift: boost R, reduce B
            cr_mod = np.minimum(cr * 1.3 + 0.1, 1.0)
            cg_mod = cg * 0.8
            cb_mod = cb * 0.3

            # 8% white-gold spark chance
            spark = np.random.uniform(0.0, 1.0, n) < 0.08
            self.color_r[s:e] = np.where(spark, 1.0, cr_mod)
            self.color_g[s:e] = np.where(spark, 0.9, cg_mod)
            self.color_b[s:e] = np.where(spark, 0.6, cb_mod)

            life_vals = np.random.uniform(1.5, 3.0, n).astype(np.float32)
        else:
            # Humanity mode: desaturate colors
            self.vel_x[s:e] = np.random.uniform(-0.01, 0.01, n).astype(np.float32)
            self.vel_y[s:e] = np.random.uniform(0.005, 0.025, n).astype(np.float32)

            # Desaturate 50-80% toward luminance
            lum = 0.299 * cr + 0.587 * cg + 0.114 * cb
            desat = np.random.uniform(0.5, 0.8, n).astype(np.float32)
            cr_mod = cr * (1.0 - desat) + lum * desat
            cg_mod = cg * (1.0 - desat) + lum * desat
            cb_mod = cb * (1.0 - desat) + lum * desat

            # Boost dark pixels
            cr_mod = np.maximum(cr_mod, 0.15)
            cg_mod = np.maximum(cg_mod, 0.15)
            cb_mod = np.maximum(cb_mod, 0.15)

            # 3% magenta accent, 3% indigo accent
            roll = np.random.uniform(0.0, 1.0, n)
            magenta = roll < 0.03
            indigo = (roll >= 0.03) & (roll < 0.06)

            self.color_r[s:e] = np.where(magenta, 1.0, np.where(indigo, 0.29, cr_mod))
            self.color_g[s:e] = np.where(magenta, 0.0, np.where(indigo, 0.0, cg_mod))
            self.color_b[s:e] = np.where(magenta, 1.0, np.where(indigo, 0.51, cb_mod))

            life_vals = np.random.uniform(2.5, 4.5, n).astype(np.float32)

        self.life[s:e] = life_vals
        self.max_life[s:e] = life_vals
        self._phase[s:e] = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)

        self.count = e

    def spawn_camera(self, brightness, motion, is_ember):
        """Camera-based spawn for webcam mode."""
        from camera import CAPTURE_W, CAPTURE_H

        slots = MAX_PARTICLES - self.count
        if slots <= 0:
            return

        n = min(SPAWN_PER_FRAME, slots)

        weights = brightness * 0.6 + motion * 0.4
        flat = weights.ravel()
        total = flat.sum()
        if total < 1e-6:
            return

        probs = flat / total
        indices = np.random.choice(len(flat), size=n, replace=True, p=probs)

        gy, gx = np.unravel_index(indices, (CAPTURE_H, CAPTURE_W))

        # Map grid coords to NDC (-1..1), mirror x so webcam feels natural
        nx = 1.0 - (gx.astype(np.float32) / CAPTURE_W) * 2.0
        ny = 1.0 - (gy.astype(np.float32) / CAPTURE_H) * 2.0

        nx += np.random.uniform(-1.0 / CAPTURE_W, 1.0 / CAPTURE_W, n).astype(np.float32)
        ny += np.random.uniform(-1.0 / CAPTURE_H, 1.0 / CAPTURE_H, n).astype(np.float32)

        s = self.count
        e = s + n

        self.pos_x[s:e] = nx
        self.pos_y[s:e] = ny
        self.prev_x[s:e] = nx
        self.prev_y[s:e] = ny

        if is_ember:
            self.vel_x[s:e] = np.random.uniform(-0.15, 0.15, n).astype(np.float32)
            self.vel_y[s:e] = np.random.uniform(0.05, 0.35, n).astype(np.float32)
            t = np.random.uniform(0.0, 1.0, n).astype(np.float32)
            spark = np.random.uniform(0.0, 1.0, n) < 0.1
            self.color_r[s:e] = np.where(spark, 1.0, 1.0)
            self.color_g[s:e] = np.where(spark, 1.0, 0.27 + t * 0.57)
            self.color_b[s:e] = np.where(spark, 1.0, 0.0)
        else:
            self.vel_x[s:e] = np.random.uniform(-0.03, 0.03, n).astype(np.float32)
            self.vel_y[s:e] = np.random.uniform(0.01, 0.08, n).astype(np.float32)
            roll = np.random.uniform(0.0, 1.0, n)
            magenta = roll < 0.05
            indigo = (roll >= 0.05) & (roll < 0.10)
            gray_val = np.random.uniform(0.15, 0.4, n).astype(np.float32)
            self.color_r[s:e] = np.where(magenta, 1.0, np.where(indigo, 0.29, gray_val))
            self.color_g[s:e] = np.where(magenta, 0.0, np.where(indigo, 0.0, gray_val))
            self.color_b[s:e] = np.where(magenta, 1.0, np.where(indigo, 0.51, gray_val))

        life_vals = np.random.uniform(1.0, 3.0, n).astype(np.float32)
        self.life[s:e] = life_vals
        self.max_life[s:e] = life_vals
        self._phase[s:e] = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)

        self.count = e

    def spawn_palm_sparks(self, palm_ndc_x, palm_ndc_y):
        slots = MAX_PARTICLES - self.count
        if slots <= 0:
            return

        n = min(15, slots)

        s = self.count
        e = s + n

        px = palm_ndc_x + np.random.uniform(-0.03, 0.03, n).astype(np.float32)
        py = palm_ndc_y + np.random.uniform(-0.03, 0.03, n).astype(np.float32)
        self.pos_x[s:e] = px
        self.pos_y[s:e] = py
        self.prev_x[s:e] = px
        self.prev_y[s:e] = py

        self.vel_x[s:e] = np.random.uniform(-0.10, 0.10, n).astype(np.float32)
        self.vel_y[s:e] = np.random.uniform(0.15, 0.50, n).astype(np.float32)

        # Colors: orange #FF8C00 to gold #FFD700, 15% white-hot sparks
        t = np.random.uniform(0.0, 1.0, n).astype(np.float32)
        spark = np.random.uniform(0.0, 1.0, n) < 0.15
        self.color_r[s:e] = np.where(spark, 1.0, 1.0)
        self.color_g[s:e] = np.where(spark, 1.0, 0.55 + t * 0.29)
        self.color_b[s:e] = np.where(spark, 0.9, 0.0)

        life_vals = np.random.uniform(0.4, 1.2, n).astype(np.float32)
        self.life[s:e] = life_vals
        self.max_life[s:e] = life_vals
        self._phase[s:e] = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)

        self.count = e

    def spawn_praise_sun(self, left_ndc, right_ndc, n_per_hand=60):
        """Radial golden sun-ray burst from both raised wrists."""
        for cx, cy in (left_ndc, right_ndc):
            slots = MAX_PARTICLES - self.count
            if slots <= 0:
                return
            n = min(n_per_hand, slots)
            s, e = self.count, self.count + n

            angles = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
            speeds = np.random.uniform(0.25, 0.9, n).astype(np.float32)

            px = cx + np.random.uniform(-0.02, 0.02, n).astype(np.float32)
            py = cy + np.random.uniform(-0.02, 0.02, n).astype(np.float32)
            self.pos_x[s:e] = px
            self.pos_y[s:e] = py
            self.prev_x[s:e] = px
            self.prev_y[s:e] = py
            self.vel_x[s:e] = np.cos(angles) * speeds
            self.vel_y[s:e] = np.sin(angles) * speeds

            # Gold (#FFD700) → white-hot sparks (20%)
            t = np.random.uniform(0, 1, n).astype(np.float32)
            white = np.random.uniform(0, 1, n) < 0.20
            self.color_r[s:e] = 1.0
            self.color_g[s:e] = np.where(white, 1.0, 0.65 + t * 0.30).astype(np.float32)
            self.color_b[s:e] = np.where(white, 0.9, 0.0).astype(np.float32)

            life_vals = np.random.uniform(0.6, 1.8, n).astype(np.float32)
            self.life[s:e] = life_vals
            self.max_life[s:e] = life_vals
            self._phase[s:e] = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
            self.count = e

    def spawn_you_died(self, n=300):
        """Blood-red particles cascading down — YOU DIED effect."""
        slots = MAX_PARTICLES - self.count
        if slots <= 0:
            return
        n = min(n, slots)
        s, e = self.count, self.count + n

        px = np.random.uniform(-1.0, 1.0, n).astype(np.float32)
        py = np.random.uniform(0.1, 1.0, n).astype(np.float32)
        self.pos_x[s:e] = px
        self.pos_y[s:e] = py
        self.prev_x[s:e] = px
        self.prev_y[s:e] = py

        self.vel_x[s:e] = np.random.uniform(-0.04, 0.04, n).astype(np.float32)
        self.vel_y[s:e] = np.random.uniform(-0.6, -0.1, n).astype(np.float32)

        self.color_r[s:e] = np.random.uniform(0.7, 1.0, n).astype(np.float32)
        self.color_g[s:e] = np.random.uniform(0.0, 0.08, n).astype(np.float32)
        self.color_b[s:e] = np.random.uniform(0.0, 0.04, n).astype(np.float32)

        life_vals = np.random.uniform(1.2, 3.5, n).astype(np.float32)
        self.life[s:e] = life_vals
        self.max_life[s:e] = life_vals
        self._phase[s:e] = np.random.uniform(0, 2 * np.pi, n).astype(np.float32)
        self.count = e

    def kindle_nearby(self, palm_x, palm_y, radius=0.3):
        """Warm particles near the palm toward ember colors + gentle pull."""
        if self.count == 0:
            return

        n = self.count
        dx = self.pos_x[:n] - palm_x
        dy = self.pos_y[:n] - palm_y
        dist = np.sqrt(dx * dx + dy * dy)

        mask = dist < radius
        if not np.any(mask):
            return

        t = np.zeros(n, dtype=np.float32)
        t[mask] = 1.0 - dist[mask] / radius

        t_smooth = t * t * (3.0 - 2.0 * t)

        blend = t_smooth * 0.6
        self.color_r[:n] = self.color_r[:n] * (1.0 - blend) + 1.0 * blend
        self.color_g[:n] = self.color_g[:n] * (1.0 - blend) + 0.65 * blend
        self.color_b[:n] = self.color_b[:n] * (1.0 - blend) + 0.1 * blend

        pull_strength = t_smooth * 0.15
        self.vel_x[:n] -= dx * pull_strength
        self.vel_y[:n] -= dy * pull_strength

    def update(self, dt, is_ember=False, flow_field=None):
        if self.count == 0:
            return

        self._time += dt
        n = self.count

        # Save previous positions for trail rendering
        self.prev_x[:n] = self.pos_x[:n]
        self.prev_y[:n] = self.pos_y[:n]

        # Apply flow field nudge if provided
        if flow_field is not None:
            fh, fw = flow_field.shape[1], flow_field.shape[2]
            FLOW_STRENGTH = 0.8
            fx = ((self.pos_x[:n] + 1.0) / 2.0 * (fw - 1)).astype(np.int32).clip(0, fw - 1)
            fy = ((self.pos_y[:n] + 1.0) / 2.0 * (fh - 1)).astype(np.int32).clip(0, fh - 1)
            self.vel_x[:n] += flow_field[0, fy, fx] * dt * FLOW_STRENGTH
            self.vel_y[:n] += flow_field[1, fy, fx] * dt * FLOW_STRENGTH

        # Mode-dependent wobble amplitude
        wobble_amp = 0.025 if is_ember else 0.012
        wobble = np.sin(self._time * 2.0 + self._phase[:n]) * wobble_amp
        self.pos_x[:n] += (self.vel_x[:n] + wobble) * dt
        self.pos_y[:n] += self.vel_y[:n] * dt

        self.life[:n] -= dt

        # Compact: swap-and-pop dead particles
        alive = self.life[:n] > 0.0
        dead_count = n - int(alive.sum())
        if dead_count > 0:
            alive_idx = np.where(alive)[0]
            new_count = len(alive_idx)

            for arr in (self.pos_x, self.pos_y, self.prev_x, self.prev_y,
                        self.vel_x, self.vel_y,
                        self.life, self.max_life, self.color_r, self.color_g,
                        self.color_b, self._phase):
                arr[:new_count] = arr[alive_idx]

            self.count = new_count

    def pack_gpu(self):
        n = self.count
        if n == 0:
            return np.empty(0, dtype=np.float32)

        ratio = self.life[:n] / self.max_life[:n]

        # Brief fade-in: peak at 85% remaining life, then fade out
        alpha = np.where(ratio > 0.85, (1.0 - ratio) / 0.15, ratio / 0.85)
        alpha = np.clip(alpha, 0.0, 1.0)

        size = 1.5 + ratio * 4.0

        buf = np.empty(n * 7, dtype=np.float32)
        buf[0::7] = self.pos_x[:n]
        buf[1::7] = self.pos_y[:n]
        buf[2::7] = self.color_r[:n]
        buf[3::7] = self.color_g[:n]
        buf[4::7] = self.color_b[:n]
        buf[5::7] = alpha
        buf[6::7] = size

        return buf

    def pack_gpu_trails(self):
        """Pack trail data: 2 vertices per particle (tail → head) for GL_LINES."""
        n = self.count
        if n == 0:
            return np.empty(0, dtype=np.float32)

        ratio = self.life[:n] / self.max_life[:n]
        alpha = np.where(ratio > 0.85, (1.0 - ratio) / 0.15, ratio / 0.85)
        alpha = np.clip(alpha, 0.0, 1.0)
        size = 1.5 + ratio * 4.0

        # 2 verts per particle, 7 floats each = 14 floats per particle
        buf = np.empty(n * 14, dtype=np.float32)

        # Tail (previous position, alpha=0)
        buf[0::14] = self.prev_x[:n]
        buf[1::14] = self.prev_y[:n]
        buf[2::14] = self.color_r[:n]
        buf[3::14] = self.color_g[:n]
        buf[4::14] = self.color_b[:n]
        buf[5::14] = 0.0
        buf[6::14] = size

        # Head (current position, full alpha)
        buf[7::14] = self.pos_x[:n]
        buf[8::14] = self.pos_y[:n]
        buf[9::14] = self.color_r[:n]
        buf[10::14] = self.color_g[:n]
        buf[11::14] = self.color_b[:n]
        buf[12::14] = alpha
        buf[13::14] = size

        return buf
