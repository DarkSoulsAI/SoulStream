import os
import time
import random
import numpy as np
import pyglet
from pyglet.window import key
import moderngl

from image_source import ImageSource
from particles import ParticleSystem, MAX_PARTICLES

WIDTH, HEIGHT = 1280, 720
SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")

# --- Mode Controller ---

MODE_AUTO = 0
MODE_FORCE_HUMANITY = 1
MODE_FORCE_EMBER = 2
MODE_NAMES = ["Auto", "Humanity (forced)", "Ember (forced)"]

# Camera mode thresholds (unchanged)
EMBER_ENTER = 0.06
EMBER_EXIT = 0.03
EMBER_COOLDOWN = 2.0

# Image mode time-based cycle
IMAGE_HUMANITY_DURATION = 12.0
IMAGE_EMBER_DURATION = 8.0


class ModeController:
    def __init__(self):
        self.mode = MODE_AUTO
        self.is_ember = False
        self._ember_since = 0.0
        self._last_high = 0.0
        self._cycle_start = time.monotonic()

    def cycle(self):
        self.mode = (self.mode + 1) % 3
        self._cycle_start = time.monotonic()

    def update_camera(self, avg_motion, now):
        """Motion-based hysteresis for camera mode."""
        if self.mode == MODE_FORCE_HUMANITY:
            self.is_ember = False
            return
        if self.mode == MODE_FORCE_EMBER:
            self.is_ember = True
            return

        if avg_motion > EMBER_ENTER:
            self._last_high = now
            if not self.is_ember:
                self.is_ember = True
                self._ember_since = now

        if self.is_ember and avg_motion < EMBER_EXIT:
            if now - self._last_high > EMBER_COOLDOWN:
                self.is_ember = False

    def update_image(self, now):
        """Time-based auto-cycle for image mode."""
        if self.mode == MODE_FORCE_HUMANITY:
            self.is_ember = False
            return
        if self.mode == MODE_FORCE_EMBER:
            self.is_ember = True
            return

        # Auto cycle: 12s Humanity -> 8s Ember -> repeat
        cycle_len = IMAGE_HUMANITY_DURATION + IMAGE_EMBER_DURATION
        elapsed = (now - self._cycle_start) % cycle_len
        self.is_ember = elapsed >= IMAGE_HUMANITY_DURATION


# --- Debug Overlay ---

class DebugOverlay:
    def __init__(self, ctx):
        self.enabled = False
        self._ctx = ctx
        self._tex = ctx.texture((160, 120), 3)
        self._tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        vert = """
        #version 330 core
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_uv = in_uv;
        }
        """
        frag = """
        #version 330 core
        uniform sampler2D tex;
        in vec2 v_uv;
        out vec4 frag_color;
        void main() {
            frag_color = vec4(texture(tex, v_uv).rgb, 0.8);
        }
        """
        self._prog = ctx.program(vertex_shader=vert, fragment_shader=frag)

        # Bottom-left quad: NDC coords for 160x120 preview
        qw = 160 / WIDTH * 2.0
        qh = 120 / HEIGHT * 2.0
        x0, y0 = -1.0, -1.0
        x1, y1 = x0 + qw, y0 + qh

        verts = np.array([
            x0, y0, 0, 0,
            x1, y0, 1, 0,
            x1, y1, 1, 1,
            x0, y0, 0, 0,
            x1, y1, 1, 1,
            x0, y1, 0, 1,
        ], dtype="f4")

        vbo = ctx.buffer(verts)
        self._vao = ctx.vertex_array(self._prog, [(vbo, "2f 2f", "in_pos", "in_uv")])

    def draw(self, preview_rgb):
        if not self.enabled:
            return
        self._tex.write(preview_rgb.tobytes())
        self._tex.use(0)
        self._vao.render(moderngl.TRIANGLES)


# --- Soul Overlay (Dark Souls GUI) ---

_SOUL_QUOTES = [
    "Don't you dare go hollow.",
    "Praise the Sun!",
    "If only I could be so grossly incandescent...",
    "Fear not the dark, my friend, and let the feast begin.",
    "Ashen one, hearest thou my voice still?",
    "In the Age of Ancients, the world was unformed...",
    "Every soul has its dark.",
    "Bearer of the curse, seek misery.",
    "I am a warrior of the sun!",
    "Fire for Ariandel... Fire for Ariandel...",
    "Rise, if you would. For that is our curse.",
    "Perhaps you've seen it, maybe in a dream.",
    "The fire fades, and the lords go without thrones.",
    "Touch the darkness within me.",
    "Hand it over. That thing, your dark soul.",
]

_HELP_TEXT = (
    "\u2500\u2500 Soul Stream \u2500\u2500\n"
    "\n"
    "Souls rise from darkness.\n"
    "Each particle is born from edge\n"
    "detection on Dark Souls artwork \u2014\n"
    "Canny edges, Sobel gradients,\n"
    "and brightness maps guide where\n"
    "25,000 souls materialize.\n"
    "\n"
    "They drift upward like freed\n"
    "spirits, carrying colors sampled\n"
    "from the original image.\n"
    "\n"
    "\u2500\u2500 Modes \u2500\u2500\n"
    "Humanity: desaturated, slow drift\n"
    "Ember: warm gold, faster rise\n"
    "\n"
    "\u2500\u2500 Controls \u2500\u2500\n"
    "SPACE   Cycle modes\n"
    "\u2190 \u2192     Change image\n"
    "C       Toggle webcam\n"
    "D       Debug overlay\n"
    "H       This help\n"
    "ESC     Quit"
)

# Banner timing (seconds)
_BANNER_FADE_IN = 0.5
_BANNER_HOLD = 2.0
_BANNER_FADE_OUT = 1.0
_BANNER_TOTAL = _BANNER_FADE_IN + _BANNER_HOLD + _BANNER_FADE_OUT

# Quote timing (seconds)
_QUOTE_DISPLAY = 8.0
_QUOTE_FADE_OUT = 2.0
_QUOTE_FADE_IN = 2.0
_QUOTE_CYCLE = _QUOTE_DISPLAY + _QUOTE_FADE_OUT + _QUOTE_FADE_IN


class SoulOverlay:
    def __init__(self):
        # --- Banner ---
        self._banner_label = pyglet.text.Label(
            "", font_name="Georgia", font_size=48,
            x=WIDTH // 2, y=HEIGHT // 2,
            anchor_x="center", anchor_y="center",
            color=(255, 255, 255, 0),
        )
        self._banner_timer = 0.0
        self._banner_active = False
        self._banner_color = (255, 255, 255)

        # --- Quotes ---
        self._quote_label = pyglet.text.Label(
            "", font_name="Georgia", font_size=18, italic=True,
            x=WIDTH // 2, y=40,
            anchor_x="center", anchor_y="center",
            color=(160, 150, 130, 0),
        )
        self._quotes = list(_SOUL_QUOTES)
        random.shuffle(self._quotes)
        self._quote_idx = 0
        self._quote_timer = 0.0
        self._quote_label.text = self._quotes[0]

        # --- Help Panel ---
        self._help_visible = False
        self._help_labels = []
        lines = _HELP_TEXT.split("\n")
        for i, line in enumerate(lines):
            lbl = pyglet.text.Label(
                line, font_name="Consolas", font_size=13,
                x=WIDTH - 20, y=HEIGHT - 30 - i * 20,
                anchor_x="right", anchor_y="center",
                color=(160, 160, 160, 180),
            )
            self._help_labels.append(lbl)

        # --- State tracking ---
        self._prev_ember = False
        self._prev_image_name = None

    def trigger_banner(self, text, color):
        """Start a banner fade-in -> hold -> fade-out animation."""
        self._banner_label.text = text
        self._banner_color = color
        self._banner_timer = 0.0
        self._banner_active = True

    def toggle_help(self):
        self._help_visible = not self._help_visible

    def update(self, dt, is_ember, image_name=None):
        # Detect mode transitions
        if is_ember != self._prev_ember:
            if is_ember:
                self.trigger_banner("HEIR OF FIRE RESTORED", (255, 200, 80))
            else:
                self.trigger_banner("HUMANITY RESTORED", (200, 210, 220))
        self._prev_ember = is_ember

        # Detect image change
        if image_name is not None and self._prev_image_name is not None:
            if image_name != self._prev_image_name:
                self.trigger_banner("BONFIRE LIT", (255, 160, 40))
        self._prev_image_name = image_name

        # Advance banner timer
        if self._banner_active:
            self._banner_timer += dt
            if self._banner_timer >= _BANNER_TOTAL:
                self._banner_active = False

        # Advance quote timer
        self._quote_timer += dt
        if self._quote_timer >= _QUOTE_CYCLE:
            self._quote_timer = 0.0
            self._quote_idx = (self._quote_idx + 1) % len(self._quotes)
            self._quote_label.text = self._quotes[self._quote_idx]

    def draw(self):
        # --- Banner ---
        if self._banner_active:
            t = self._banner_timer
            if t < _BANNER_FADE_IN:
                alpha = t / _BANNER_FADE_IN
            elif t < _BANNER_FADE_IN + _BANNER_HOLD:
                alpha = 1.0
            else:
                alpha = 1.0 - (t - _BANNER_FADE_IN - _BANNER_HOLD) / _BANNER_FADE_OUT
            alpha = max(0.0, min(1.0, alpha))
            r, g, b = self._banner_color
            self._banner_label.color = (r, g, b, int(alpha * 255))
            self._banner_label.draw()

        # --- Quote ---
        t = self._quote_timer
        if t < _QUOTE_DISPLAY:
            # Fully visible (after initial fade-in completes from prev cycle)
            alpha = 1.0
        elif t < _QUOTE_DISPLAY + _QUOTE_FADE_OUT:
            # Fading out
            alpha = 1.0 - (t - _QUOTE_DISPLAY) / _QUOTE_FADE_OUT
        else:
            # Fading in (new quote already set at cycle boundary)
            alpha = (t - _QUOTE_DISPLAY - _QUOTE_FADE_OUT) / _QUOTE_FADE_IN
        alpha = max(0.0, min(1.0, alpha))
        self._quote_label.color = (160, 150, 130, int(alpha * 200))
        self._quote_label.draw()

        # --- Help Panel ---
        if self._help_visible:
            for lbl in self._help_labels:
                lbl.draw()


# --- Main Application ---

class SoulStreamApp(pyglet.window.Window):
    def __init__(self):
        super().__init__(WIDTH, HEIGHT, caption="Soul Stream", resizable=False,
                         config=pyglet.gl.Config(
                             major_version=3, minor_version=3,
                             double_buffer=True,
                         ))

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

        # Image source as primary
        self.image_source = ImageSource(IMAGE_DIR, WIDTH, HEIGHT)
        self.use_camera = False
        self.camera = None  # Lazy-initialized on C key

        self.particles = ParticleSystem()
        self.mode_ctrl = ModeController()
        self.debug = DebugOverlay(self.ctx)
        self.overlay = SoulOverlay()

        # Load particle shaders
        with open(os.path.join(SHADER_DIR, "particle.vert")) as f:
            vert_src = f.read()
        with open(os.path.join(SHADER_DIR, "particle.frag")) as f:
            frag_src = f.read()
        self._prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # GPU buffer — pre-allocate for max particles (7 floats each)
        self._vbo = self.ctx.buffer(reserve=MAX_PARTICLES * 7 * 4)
        self._vao = self.ctx.vertex_array(
            self._prog,
            [(self._vbo, "2f 3f 1f 1f", "in_pos", "in_color", "in_alpha", "in_size")],
        )

        self._mode_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=12,
            x=10, y=HEIGHT - 20, color=(180, 180, 180, 200),
        )
        self._particle_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=12,
            x=10, y=HEIGHT - 40, color=(180, 180, 180, 200),
        )
        self._source_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=12,
            x=10, y=HEIGHT - 60, color=(180, 180, 180, 200),
        )

    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            if self.camera:
                self.camera.stop()
            self.close()
        elif symbol == key.D:
            self.debug.enabled = not self.debug.enabled
        elif symbol == key.SPACE:
            self.mode_ctrl.cycle()
        elif symbol == key.C:
            # Toggle camera mode (lazy-init webcam)
            if self.use_camera:
                self.use_camera = False
            else:
                if self.camera is None:
                    from camera import Camera
                    self.camera = Camera()
                self.use_camera = True
        elif symbol == key.H:
            self.overlay.toggle_help()
        elif symbol == key.LEFT:
            if not self.use_camera:
                self.image_source.prev_image()
        elif symbol == key.RIGHT:
            if not self.use_camera:
                self.image_source.next_image()

    def on_draw(self):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        now = time.monotonic()
        dt = 1.0 / 60.0

        if self.use_camera and self.camera:
            # Camera path: motion-based mode switching
            brightness, motion, avg_motion = self.camera.get_data()
            self.mode_ctrl.update_camera(avg_motion, now)
            self.particles.spawn_camera(brightness, motion, self.mode_ctrl.is_ember)
        else:
            # Image path: time-based mode cycling
            self.mode_ctrl.update_image(now)
            self.particles.spawn(self.image_source, self.mode_ctrl.is_ember)

        self.particles.update(dt, self.mode_ctrl.is_ember)

        # Pack and upload to GPU
        gpu_data = self.particles.pack_gpu()
        n = self.particles.count
        if n > 0:
            data_bytes = gpu_data.tobytes()
            self._vbo.orphan()
            self._vbo.write(data_bytes)
            self._vao.render(moderngl.POINTS, vertices=n)

        # Debug overlay + HUD
        if self.debug.enabled:
            if self.use_camera and self.camera:
                preview = self.camera.get_preview()
            else:
                preview = self.image_source.get_preview()
            self.debug.draw(preview)

            mode_name = MODE_NAMES[self.mode_ctrl.mode]
            state = "EMBER" if self.mode_ctrl.is_ember else "Humanity"
            source = "Camera" if self.use_camera else f"Image: {self.image_source.image_name}"
            self._mode_label.text = f"Mode: {mode_name} | State: {state}"
            self._mode_label.draw()
            self._particle_label.text = f"Particles: {self.particles.count}"
            self._particle_label.draw()
            self._source_label.text = f"Source: {source} [{self.image_source.image_count} images]"
            self._source_label.draw()

        # Soul overlay (banners, quotes, help) — always last
        image_name = None if self.use_camera else self.image_source.image_name
        self.overlay.update(dt, self.mode_ctrl.is_ember, image_name)
        self.overlay.draw()

    def on_close(self):
        if self.camera:
            self.camera.stop()
        super().on_close()


def main():
    app = SoulStreamApp()
    pyglet.clock.schedule_interval(lambda dt: None, 1 / 60)
    pyglet.app.run()


if __name__ == "__main__":
    main()
