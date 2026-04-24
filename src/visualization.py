"""
Visualization system for SoulStream.

Five rendering modes, each encapsulated in its own class:
  0  POINTS   — circular glowing point cloud (original behaviour)
  1  TRAILS   — particles leave fading line-segment tails
  2  SKELETON — MediaPipe hand/body skeleton rendered in Dark Souls gold glow
  3  FLOW     — curl-noise velocity field bends particle paths
  4  SHAPES   — small rotating triangle sigils derived from particle positions
"""

import os
import math
import numpy as np
import moderngl
from enum import IntEnum

_SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
SHADER_DIR = os.path.join(_ROOT_DIR, "shaders")

VIZ_NAMES = ["Points", "Trails", "Skeleton", "Flow", "Shapes"]


class VisualizationMode(IntEnum):
    POINTS   = 0
    TRAILS   = 1
    SKELETON = 2
    FLOW     = 3
    SHAPES   = 4


# MediaPipe hand connections (21 landmarks)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]

# Compact body skeleton connections (33 landmarks)
_POSE_CONNECTIONS = [
    (11, 12),            # shoulders
    (11, 13), (13, 15),  # left arm
    (12, 14), (14, 16),  # right arm
    (11, 23), (12, 24),  # torso sides
    (23, 24),            # hips
    (23, 25), (25, 27),  # left leg
    (24, 26), (26, 28),  # right leg
]

_POSE_JOINTS = {11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28}


# ═══════════════════════════════════════════════════════════════
#  PointsRenderer  — wraps the existing GL point cloud
# ═══════════════════════════════════════════════════════════════

class PointsRenderer:
    """Renders particles as circular glowing points (original behaviour)."""

    def setup(self, ctx: moderngl.Context):
        self._ctx = ctx
        with open(os.path.join(SHADER_DIR, "particle.vert")) as f:
            vert_src = f.read()
        with open(os.path.join(SHADER_DIR, "particle.frag")) as f:
            frag_src = f.read()
        self._prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # Size for trails (2 verts per particle) so both modes share one VBO
        from particles import MAX_PARTICLES
        self._vbo = ctx.buffer(reserve=MAX_PARTICLES * 2 * 7 * 4)
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 3f 1f 1f", "in_pos", "in_color", "in_alpha", "in_size")],
        )

    def render(self, particles, is_ember: bool = False):
        n = particles.count
        if n == 0:
            return
        self._prog['u_ember'] = 1 if is_ember else 0
        data = particles.pack_gpu()
        self._vbo.orphan()
        self._vbo.write(data.tobytes())
        self._vao.render(moderngl.POINTS, vertices=n)

    def cleanup(self):
        pass


# ═══════════════════════════════════════════════════════════════
#  TrailsRenderer  — GL_LINES from prev → cur position
# ═══════════════════════════════════════════════════════════════

class TrailsRenderer:
    """Each particle emits a fading tail from its previous to current position."""

    def setup(self, ctx: moderngl.Context):
        self._ctx = ctx
        with open(os.path.join(SHADER_DIR, "particle.vert")) as f:
            vert_src = f.read()
        with open(os.path.join(SHADER_DIR, "particle.frag")) as f:
            frag_src = f.read()
        self._prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        from particles import MAX_PARTICLES
        self._vbo = ctx.buffer(reserve=MAX_PARTICLES * 2 * 7 * 4)
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 3f 1f 1f", "in_pos", "in_color", "in_alpha", "in_size")],
        )

    def render(self, particles, is_ember: bool = False):
        n = particles.count
        if n == 0:
            return
        self._prog['u_ember'] = 1 if is_ember else 0
        data = particles.pack_gpu_trails()
        self._vbo.orphan()
        self._vbo.write(data.tobytes())
        self._vao.render(moderngl.LINES, vertices=n * 2)

    def cleanup(self):
        pass


# ═══════════════════════════════════════════════════════════════
#  SkeletonRenderer  — MediaPipe landmarks as Dark Souls glow
# ═══════════════════════════════════════════════════════════════

class SkeletonRenderer:
    """
    Renders the detected hand / body skeleton as a styled glowing overlay on
    top of the particle field.  Uses Dark Souls gold (#C8A84E) for bones and
    bright gold (#FFD700) for joints.
    """

    # Bone color: Dark Souls gold
    _BONE_R, _BONE_G, _BONE_B = 0.784, 0.659, 0.306   # #C8A84E
    # Joint color: bright gold
    _JOINT_R, _JOINT_G, _JOINT_B = 1.0, 0.843, 0.0    # #FFD700
    # Palm / wrist highlight: ember orange
    _PALM_R, _PALM_G, _PALM_B = 1.0, 0.549, 0.0       # #FF8C00

    def setup(self, ctx: moderngl.Context):
        self._ctx = ctx

        skel_vert_path = os.path.join(SHADER_DIR, "skeleton.vert")
        skel_frag_path = os.path.join(SHADER_DIR, "skeleton.frag")
        with open(skel_vert_path) as f:
            vert_src = f.read()
        with open(skel_frag_path) as f:
            frag_src = f.read()

        self._prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # 64 hand bones*2 + 21 joints + 13 pose bones*2 + 12 pose joints ≈ 200 verts max
        self._vbo = ctx.buffer(reserve=512 * 6 * 4)
        self._line_vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 4f", "in_pos", "in_color")],
        )

    def render(self, hand_data, pose_data, time_val: float):
        buf = []

        # ── Hand skeleton ──
        if hand_data.detected and hand_data.landmarks:
            lm = hand_data.landmarks
            alpha = 0.9 + 0.1 * math.sin(time_val * 4.0)  # subtle pulse

            for a, b in _HAND_CONNECTIONS:
                ax, ay = lm[a]
                bx, by = lm[b]
                buf.extend([ax, ay, self._BONE_R, self._BONE_G, self._BONE_B, alpha,
                             bx, by, self._BONE_R, self._BONE_G, self._BONE_B, alpha])

            line_count = len(_HAND_CONNECTIONS) * 2

            # Joints
            for i, (x, y) in enumerate(lm):
                if i == 0 or i == 9:   # wrist + middle MCP → palm highlight
                    r, g, b = self._PALM_R, self._PALM_G, self._PALM_B
                else:
                    r, g, b = self._JOINT_R, self._JOINT_G, self._JOINT_B
                buf.extend([x, y, r, g, b, alpha])

            joint_count = len(lm)

            if buf:
                data = np.array(buf, dtype="f4")
                self._vbo.orphan()
                self._vbo.write(data.tobytes())
                self._line_vao.render(moderngl.LINES, vertices=line_count)
                self._line_vao.render(moderngl.POINTS, vertices=joint_count, first=line_count)
            buf = []

        # ── Body skeleton ──
        if pose_data.detected and pose_data.landmarks:
            lm = pose_data.landmarks
            alpha = 0.85 + 0.1 * math.sin(time_val * 3.0 + 1.0)

            for a, b in _POSE_CONNECTIONS:
                if a < len(lm) and b < len(lm):
                    ax, ay = lm[a]
                    bx, by = lm[b]
                    buf.extend([ax, ay, self._BONE_R, self._BONE_G, self._BONE_B, alpha,
                                 bx, by, self._BONE_R, self._BONE_G, self._BONE_B, alpha])

            pose_line_count = len([c for c in _POSE_CONNECTIONS
                                   if c[0] < len(lm) and c[1] < len(lm)]) * 2

            for idx in _POSE_JOINTS:
                if idx < len(lm):
                    x, y = lm[idx]
                    buf.extend([x, y, self._JOINT_R, self._JOINT_G, self._JOINT_B, alpha])

            pose_joint_count = sum(1 for idx in _POSE_JOINTS if idx < len(lm))

            if buf:
                data = np.array(buf, dtype="f4")
                self._vbo.orphan()
                self._vbo.write(data.tobytes())
                self._line_vao.render(moderngl.LINES, vertices=pose_line_count)
                self._line_vao.render(moderngl.POINTS, vertices=pose_joint_count,
                                      first=pose_line_count)

    def cleanup(self):
        pass


# ═══════════════════════════════════════════════════════════════
#  FlowRenderer  — curl-noise velocity field (CPU, vectorised)
# ═══════════════════════════════════════════════════════════════

class FlowRenderer:
    """
    Computes a 2D curl-noise flow field each frame and exposes it for
    particles.update().  No additional GPU draw call — particles still
    render via PointsRenderer or TrailsRenderer.
    """

    FLOW_W = 64
    FLOW_H = 36

    def __init__(self):
        self._field = np.zeros((2, self.FLOW_H, self.FLOW_W), dtype=np.float32)
        self._time = 0.0
        h = 0.05  # finite-difference step for gradient
        self._h = h

        # Pre-build meshgrid once
        xs = np.linspace(0.0, 4.0, self.FLOW_W, dtype=np.float32)
        ys = np.linspace(0.0, 4.0, self.FLOW_H, dtype=np.float32)
        self._X, self._Y = np.meshgrid(xs, ys)

    def _potential(self, X, Y, t):
        return (np.sin(X * 1.3 + t) * np.cos(Y * 0.9 + t * 0.7)
                + np.sin(X * 2.1 + t * 1.4) * 0.5
                + np.cos(Y * 1.7 + X * 0.8 + t * 0.5) * 0.5)

    def update_field(self, dt) -> np.ndarray:
        """Update and return the (2, H, W) flow field."""
        self._time += dt * 0.3
        t = self._time
        h = self._h
        X, Y = self._X, self._Y

        phi    = self._potential(X,     Y,     t)
        phi_dy = self._potential(X,     Y + h, t)
        phi_dx = self._potential(X + h, Y,     t)

        self._field[0] = (phi_dy - phi) / h   # curl x component
        self._field[1] = -(phi_dx - phi) / h  # curl y component
        return self._field

    def get_field(self) -> np.ndarray:
        return self._field

    def setup(self, ctx: moderngl.Context):
        pass  # no GPU resources needed

    def cleanup(self):
        pass


# ═══════════════════════════════════════════════════════════════
#  ShapesRenderer  — rotating triangle sigils
# ═══════════════════════════════════════════════════════════════

class ShapesRenderer:
    """
    Derives up to MAX_SHAPES small rotating equilateral triangles from live
    particle positions and renders them as a separate GL_TRIANGLES pass on top
    of the particle field.
    """

    MAX_SHAPES = 400

    def __init__(self):
        self._time = 0.0
        self._shape_buf = None

    def setup(self, ctx: moderngl.Context):
        self._ctx = ctx

        shapes_vert_path = os.path.join(SHADER_DIR, "shapes.vert")
        shapes_frag_path = os.path.join(SHADER_DIR, "shapes.frag")
        with open(shapes_vert_path) as f:
            vert_src = f.read()
        with open(shapes_frag_path) as f:
            frag_src = f.read()

        self._prog = ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        # 3 verts per triangle × 6 floats (x,y,r,g,b,a) × MAX_SHAPES
        self._vbo = ctx.buffer(reserve=self.MAX_SHAPES * 3 * 6 * 4)
        self._vao = ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 4f", "in_pos", "in_color")],
        )

    def update(self, particles, is_ember, dt):
        """Build triangle geometry from a subset of live particles."""
        self._time += dt
        n = particles.count
        if n == 0:
            self._shape_buf = None
            return

        # Sample up to MAX_SHAPES particle indices
        k = min(self.MAX_SHAPES, n)
        step = max(1, n // k)
        indices = np.arange(0, n, step, dtype=np.int32)[:k]

        cx = particles.pos_x[indices]
        cy = particles.pos_y[indices]
        cr = particles.color_r[indices]
        cg = particles.color_g[indices]
        cb = particles.color_b[indices]
        ratio = particles.life[indices] / particles.max_life[indices]

        alpha = np.where(ratio > 0.85, (1.0 - ratio) / 0.15, ratio / 0.85)
        alpha = np.clip(alpha * 0.6, 0.0, 0.6).astype(np.float32)  # semi-transparent

        size = (0.015 + ratio * 0.025).astype(np.float32)
        # Slowly rotate based on time + particle phase
        angle = self._time * 0.8 + indices.astype(np.float32) * 0.17

        # Build 3 verts per triangle (equilateral, rotated)
        a0 = angle
        a1 = angle + 2.094395   # +120 deg
        a2 = angle + 4.18879    # +240 deg

        v0x = cx + np.cos(a0) * size
        v0y = cy + np.sin(a0) * size
        v1x = cx + np.cos(a1) * size
        v1y = cy + np.sin(a1) * size
        v2x = cx + np.cos(a2) * size
        v2y = cy + np.sin(a2) * size

        m = len(indices)
        buf = np.empty(m * 3 * 6, dtype=np.float32)
        for vi, (vx, vy) in enumerate(((v0x, v0y), (v1x, v1y), (v2x, v2y))):
            base = vi * 6
            buf[base::18] = vx
            buf[base+1::18] = vy
            buf[base+2::18] = cr
            buf[base+3::18] = cg
            buf[base+4::18] = cb
            buf[base+5::18] = alpha

        self._shape_buf = buf
        self._shape_count = m * 3

    def render(self):
        if self._shape_buf is None or self._shape_count == 0:
            return
        self._vbo.orphan()
        self._vbo.write(self._shape_buf.tobytes())
        self._vao.render(moderngl.TRIANGLES, vertices=self._shape_count)

    def cleanup(self):
        pass
