import os
import math
import time
import random
from datetime import datetime, timezone
import numpy as np
import pyglet
from pyglet.window import key
import moderngl

from image_source import ImageSource
from particles import ParticleSystem, MAX_PARTICLES
from gui import GameMenu

WIDTH, HEIGHT = 1280, 720
SHADER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shaders")
IMAGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "image")
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio")
RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# App states
STATE_LOADING = 0
STATE_INTRO = 1
STATE_RUNNING = 2
INTRO_DURATION = 5.0

# Audio file assignments
AUDIO_OPENING = "dark-souls-the-ancient-dragon-choir.mp3"   # Looping ambience
AUDIO_EMBER_IGNITE = "dark-souls-kill.mp3"                  # Humanity -> Ember
AUDIO_HUMANITY_RESTORED = "dark-souls-im-sorry.mp3"         # Ember -> Humanity
AUDIO_BONFIRE_LIT = "darksoul_bonfire_jump.mp3"             # Image change
AUDIO_CAMERA_ON = "hello-darksoul3.mp3"                     # Camera toggle on
AUDIO_HELP = "firekeeper.mp3"                               # Help panel toggle
AUDIO_QUIT = "thank-you-dark-souls.mp3"                     # ESC quit
AUDIO_MODE_CYCLE = "darksouls-pain.mp3"                     # SPACE mode cycle
AUDIO_BOSS_OUT = "bossout.mp3"                              # Palm ember burst
AUDIO_START = "i-offer-you-an-accord.mp3"                   # Loading -> intro transition

OPENING_VOLUME = 0.25
SFX_VOLUME = 0.60

# Floating intro key definitions
_INTRO_KEYS = [
    ("SPACE",  "Cycle Modes",   (255, 140, 50)),    # orange
    ("\u2190 \u2192",   "Change Image",  (80, 220, 255)),    # cyan
    ("C",      "Toggle Camera", (80, 255, 120)),    # green
    ("H",      "Help",          (255, 100, 220)),   # magenta
    ("TAB",    "Menu",          (200, 168, 78)),     # gold
    ("ESC",    "Quit",          (255, 80, 80)),      # red
]

# --- Sound Manager ---

class SoundManager:
    """Handles background ambience and all interaction sound effects."""

    def __init__(self):
        self._ambience_player = None
        self._ambience_source = None
        self._sounds = {}
        self._prev_ember = False

        # Pre-load ambience source but don't play yet
        self._ambience_source = self._load_source(AUDIO_OPENING)

        # Load all one-shot sounds
        for name in (AUDIO_EMBER_IGNITE, AUDIO_HUMANITY_RESTORED, AUDIO_BONFIRE_LIT,
                     AUDIO_CAMERA_ON, AUDIO_HELP, AUDIO_QUIT, AUDIO_MODE_CYCLE,
                     AUDIO_BOSS_OUT, AUDIO_START):
            src = self._load_source(name)
            if src is not None:
                self._sounds[name] = src

    def start_ambience(self):
        """Start looping ambience playback (called on intro state entry)."""
        if self._ambience_player is not None:
            return  # already playing
        if self._ambience_source is None:
            return
        try:
            player = pyglet.media.Player()
            player.queue(self._ambience_source)
            player.loop = True
            player.volume = OPENING_VOLUME
            player.play()
            self._ambience_player = player
            print(f"[SoundManager] Ambience started")
        except Exception as e:
            print(f"[SoundManager] Could not start ambience: {e}")

    def _load_source(self, filename):
        try:
            path = os.path.join(AUDIO_DIR, filename)
            source = pyglet.media.load(path, streaming=False)
            dur = source.duration
            if dur is None or dur < 0.05 or dur > 600.0:
                print(f"[SoundManager] '{filename}' unusual duration ({dur}s), skipping.")
                return None
            print(f"[SoundManager] Loaded: {filename} ({dur:.1f}s)")
            return source
        except Exception as e:
            print(f"[SoundManager] Could not load '{filename}': {e}")
            return None

    def play(self, filename, volume=None):
        """Play a one-shot sound effect."""
        src = self._sounds.get(filename)
        if src is None:
            return
        try:
            player = src.play()
            player.volume = volume if volume is not None else SFX_VOLUME
        except Exception as e:
            print(f"[SoundManager] Error playing '{filename}': {e}")

    def play_quit(self):
        """Play quit sound and return its duration for delayed close."""
        src = self._sounds.get(AUDIO_QUIT)
        if src is None:
            return 0.0
        try:
            player = src.play()
            player.volume = SFX_VOLUME
            return src.duration or 0.0
        except Exception:
            return 0.0

    def update(self, is_ember):
        """Call every frame. Plays transition sounds on state change."""
        if is_ember and not self._prev_ember:
            self.play(AUDIO_EMBER_IGNITE)
        elif not is_ember and self._prev_ember:
            self.play(AUDIO_HUMANITY_RESTORED)
        self._prev_ember = is_ember

    def cleanup(self):
        """Stop all audio playback."""
        try:
            if self._ambience_player:
                self._ambience_player.pause()
                self._ambience_player = None
        except Exception:
            pass


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

    def update_camera(self, avg_motion, now, hand_is_open_palm=False):
        """Motion-based hysteresis for camera mode, supplemented by open palm."""
        if self.mode == MODE_FORCE_HUMANITY:
            self.is_ember = False
            return
        if self.mode == MODE_FORCE_EMBER:
            self.is_ember = True
            return

        # Open palm supplements motion: either trigger -> ember active
        if avg_motion > EMBER_ENTER or hand_is_open_palm:
            self._last_high = now
            if not self.is_ember:
                self.is_ember = True
                self._ember_since = now

        if self.is_ember and avg_motion < EMBER_EXIT and not hand_is_open_palm:
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

# MediaPipe hand skeleton connections (21 landmarks, 21 bones)
_HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (5, 6), (6, 7), (7, 8),
    # Middle
    (9, 10), (10, 11), (11, 12),
    # Ring
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (17, 18), (18, 19), (19, 20),
    # Palm
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]

_FINGERTIPS = {4, 8, 12, 16, 20}
_FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]


class DebugOverlay:
    def __init__(self, ctx, win_w=WIDTH, win_h=HEIGHT):
        self.enabled = False
        self._ctx = ctx
        self._win_w = win_w
        self._win_h = win_h
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

        self._preview_vbo = ctx.buffer(reserve=6 * 4 * 4)
        self._vao = ctx.vertex_array(self._prog, [(self._preview_vbo, "2f 2f", "in_pos", "in_uv")])
        self._rebuild_preview_quad(win_w, win_h)

    def _rebuild_preview_quad(self, win_w, win_h):
        """Rebuild the preview quad VBO for current window size."""
        qw = 160 / win_w * 2.0
        qh = 120 / win_h * 2.0
        x0, y0 = -1.0, -1.0
        x1, y1 = x0 + qw, y0 + qh
        verts = np.array([
            x0, y0, 0, 1,
            x1, y0, 1, 1,
            x1, y1, 1, 0,
            x0, y0, 0, 1,
            x1, y1, 1, 0,
            x0, y1, 0, 0,
        ], dtype="f4")
        self._preview_vbo.orphan()
        self._preview_vbo.write(verts.tobytes())

    def resize(self, win_w, win_h):
        """Update positions for new window dimensions."""
        self._win_w = win_w
        self._win_h = win_h
        self._rebuild_preview_quad(win_w, win_h)
        # Reset lazy-init flag so hand panel labels get recreated at new positions
        if hasattr(self, "_hand_panel_labels"):
            del self._hand_panel_labels

        # Hand skeleton line/point shader
        hand_vert = """
        #version 330 core
        in vec2 in_pos;
        in vec3 in_color;
        out vec3 v_color;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            gl_PointSize = 6.0;
            v_color = in_color;
        }
        """
        hand_frag = """
        #version 330 core
        in vec3 v_color;
        out vec4 frag_color;
        void main() {
            frag_color = vec4(v_color, 0.85);
        }
        """
        self._hand_prog = self._ctx.program(vertex_shader=hand_vert, fragment_shader=hand_frag)
        # 21 connections * 2 verts = 42 line verts + 21 joint points = 63 max
        self._hand_vbo = self._ctx.buffer(reserve=63 * 5 * 4)
        self._hand_line_vao = self._ctx.vertex_array(
            self._hand_prog,
            [(self._hand_vbo, "2f 3f", "in_pos", "in_color")],
        )

    def draw(self, preview_rgb):
        if not self.enabled:
            return
        self._tex.write(preview_rgb.tobytes())
        self._tex.use(0)
        self._vao.render(moderngl.TRIANGLES)

    def draw_hand(self, hand_data):
        if not self.enabled or not hand_data.detected or hand_data.landmarks is None:
            return

        lm = hand_data.landmarks
        is_open = hand_data.is_open_palm
        finger_states = hand_data.finger_states or {}

        # Skeleton color: green when open palm, cyan when closed
        if is_open:
            line_color = (0.2, 1.0, 0.2)
            joint_color = (1.0, 1.0, 0.0)
        else:
            line_color = (0.0, 0.8, 0.8)
            joint_color = (0.0, 0.9, 1.0)

        # Fingertip index -> finger name for per-finger coloring
        tip_to_name = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

        # Build line vertices: 2 verts per connection
        buf = []
        for a, b in _HAND_CONNECTIONS:
            ax, ay = lm[a]
            bx, by = lm[b]
            buf.extend([ax, ay, *line_color, bx, by, *line_color])

        line_count = len(_HAND_CONNECTIONS) * 2

        # Build joint point vertices: 1 vert per landmark
        # Fingertips: green if extended, red if not
        for i, (x, y) in enumerate(lm):
            if i in _FINGERTIPS:
                fname = tip_to_name[i]
                if finger_states.get(fname, False):
                    c = (0.2, 1.0, 0.2)  # green = extended
                else:
                    c = (1.0, 0.1, 0.1)  # red = closed
            else:
                c = joint_color
            buf.extend([x, y, *c])

        joint_count = len(lm)

        data = np.array(buf, dtype="f4")
        self._hand_vbo.orphan()
        self._hand_vbo.write(data.tobytes())

        # Draw lines first, then points on top
        self._hand_line_vao.render(moderngl.LINES, vertices=line_count)
        self._hand_line_vao.render(moderngl.POINTS, vertices=joint_count, first=line_count)

    def draw_hand_panel(self, hand_data, ema_confidence):
        """Draw hand tracking status panel (pyglet labels) in bottom-right."""
        if not self.enabled:
            return

        if not hasattr(self, "_hand_panel_labels"):
            # Lazy-init labels for the hand panel
            rx = self._win_w - 20
            self._hand_status_label = pyglet.text.Label(
                "", font_name="Consolas", font_size=14,
                x=rx, y=160,
                anchor_x="right", anchor_y="center",
            )
            self._hand_ema_label = pyglet.text.Label(
                "", font_name="Consolas", font_size=10,
                x=rx, y=140,
                anchor_x="right", anchor_y="center",
                color=(180, 180, 180, 200),
            )
            self._hand_finger_labels = []
            for i in range(5):
                lbl = pyglet.text.Label(
                    "", font_name="Consolas", font_size=11,
                    x=rx, y=118 - i * 18,
                    anchor_x="right", anchor_y="center",
                )
                self._hand_finger_labels.append(lbl)
            self._hand_ndc_label = pyglet.text.Label(
                "", font_name="Consolas", font_size=10,
                x=rx, y=20,
                anchor_x="right", anchor_y="center",
                color=(180, 180, 180, 200),
            )
            self._hand_panel_labels = True

        if not hand_data.detected:
            self._hand_status_label.text = "Hand: not detected"
            self._hand_status_label.color = (180, 80, 80, 220)
            self._hand_status_label.draw()
            self._hand_ema_label.text = f"EMA: {ema_confidence:.3f}"
            self._hand_ema_label.draw()
            return

        finger_states = hand_data.finger_states or {}
        is_open = hand_data.is_open_palm

        # Status
        if is_open:
            self._hand_status_label.text = "OPEN PALM"
            self._hand_status_label.color = (80, 255, 80, 255)
        else:
            self._hand_status_label.text = "CLOSED"
            self._hand_status_label.color = (80, 200, 255, 220)
        self._hand_status_label.draw()

        # EMA bar as text
        bar_len = 20
        filled = int(min(ema_confidence, 1.0) * bar_len)
        bar = "|" + "#" * filled + "-" * (bar_len - filled) + "|"
        thresh_pos = int(0.5 * bar_len) + 1  # +1 for leading |
        self._hand_ema_label.text = f"EMA: {ema_confidence:.3f} {bar}"
        self._hand_ema_label.draw()

        # Per-finger status
        for i, fname in enumerate(_FINGER_NAMES):
            extended = finger_states.get(fname, False)
            marker = "[X]" if extended else "[ ]"
            self._hand_finger_labels[i].text = f"{marker} {fname}"
            if extended:
                self._hand_finger_labels[i].color = (80, 255, 80, 220)
            else:
                self._hand_finger_labels[i].color = (255, 80, 80, 220)
            self._hand_finger_labels[i].draw()

        # NDC
        self._hand_ndc_label.text = (
            f"Palm NDC: ({hand_data.palm_ndc_x:.2f}, {hand_data.palm_ndc_y:.2f})"
        )
        self._hand_ndc_label.draw()


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
    "PALM    Open palm = kindle ember\n"
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

    def resize(self, win_w, win_h):
        """Update label positions for new window dimensions."""
        self._banner_label.x = win_w // 2
        self._banner_label.y = win_h // 2
        self._quote_label.x = win_w // 2
        for i, lbl in enumerate(self._help_labels):
            lbl.x = win_w - 20
            lbl.y = win_h - 30 - i * 20

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
        super().__init__(WIDTH, HEIGHT, caption="Soul Stream", resizable=True,
                         config=pyglet.gl.Config(
                             major_version=3, minor_version=3,
                             double_buffer=True,
                         ))
        self._is_fullscreen = False

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)

        # --- App state machine ---
        self._state = STATE_LOADING
        self._float_keys_timer = 0.0
        self._float_keys_active = False

        # Image source as primary
        self.image_source = ImageSource(IMAGE_DIR, WIDTH, HEIGHT)
        self.use_camera = False
        self.camera = None  # Lazy-initialized on C key

        self.particles = ParticleSystem()
        self.mode_ctrl = ModeController()
        self.debug = DebugOverlay(self.ctx)
        self.overlay = SoulOverlay()
        self.sound = SoundManager()
        self._prev_palm_open = False

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
        self._hand_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=12,
            x=10, y=HEIGHT - 80, color=(180, 180, 180, 200),
        )

        # Wall clock (bottom-right)
        self._clock_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=12,
            x=WIDTH - 10, y=10,
            anchor_x="right", anchor_y="bottom",
            color=(160, 150, 130, 180),
        )
        self._tz_name = datetime.now().astimezone().strftime("%Z")

        # GUI overlay menu
        self.menu = GameMenu(WIDTH, HEIGHT, callbacks={
            "toggle_camera": self._gui_toggle_camera,
            "prev_image":    self._gui_prev_image,
            "next_image":    self._gui_next_image,
            "set_mode_auto":     lambda: self._gui_set_mode(0),
            "set_mode_humanity": lambda: self._gui_set_mode(1),
            "set_mode_ember":    lambda: self._gui_set_mode(2),
            "set_volume":    self._gui_set_volume,
            "toggle_debug":  self._gui_toggle_debug,
            "toggle_help":   self._gui_toggle_help,
            "quit":          self._gui_quit,
        })

        # --- Loading screen assets ---
        self._loading_bg_sprite = None
        try:
            bg_path = os.path.join(IMAGE_DIR, "darksouls1.jpg")
            bg_img = pyglet.image.load(bg_path)
            self._loading_bg_sprite = pyglet.sprite.Sprite(bg_img)
            # Scale to fill window
            sx = WIDTH / bg_img.width
            sy = HEIGHT / bg_img.height
            scale = max(sx, sy)
            self._loading_bg_sprite.scale = scale
            self._loading_bg_sprite.x = (WIDTH - bg_img.width * scale) / 2
            self._loading_bg_sprite.y = (HEIGHT - bg_img.height * scale) / 2
        except Exception as e:
            print(f"[Loading] Could not load background: {e}")

        # Dark overlay rectangle (reused every frame)
        self._loading_overlay = pyglet.shapes.Rectangle(0, 0, WIDTH, HEIGHT, color=(0, 0, 0))
        self._loading_overlay.opacity = 160

        self._loading_title = pyglet.text.Label(
            "SoulStream", font_name="Georgia", font_size=64,
            x=WIDTH // 2, y=HEIGHT // 2 + 60,
            anchor_x="center", anchor_y="center",
            color=(200, 168, 78, 255),
        )
        self._loading_subtitle = pyglet.text.Label(
            "by \u6eaf\u6d41\u5149", font_name="Georgia", font_size=22,
            x=WIDTH // 2, y=HEIGHT // 2 - 10,
            anchor_x="center", anchor_y="center",
            color=(230, 220, 200, 220),
        )
        self._loading_version = pyglet.text.Label(
            "v1.0", font_name="Consolas", font_size=12,
            x=WIDTH // 2, y=40,
            anchor_x="center", anchor_y="center",
            color=(140, 130, 120, 160),
        )
        self._loading_start = pyglet.text.Label(
            "PRESS ENTER", font_name="Georgia", font_size=20,
            x=WIDTH // 2, y=HEIGHT // 2 - 80,
            anchor_x="center", anchor_y="center",
            color=(200, 168, 78, 255),
        )
        self._loading_time = 0.0

        # --- Intro floating key labels ---
        self._intro_labels = []
        n_keys = len(_INTRO_KEYS)
        for i, (key_name, desc, color) in enumerate(_INTRO_KEYS):
            # Scatter across screen in a loose 3x2 grid
            col = i % 3
            row = i // 3
            lx = int(WIDTH * (0.2 + col * 0.3))
            ly = int(HEIGHT * (0.6 - row * 0.25))
            lbl = pyglet.text.Label(
                f"  [{key_name}]  {desc}  ",
                font_name="Consolas", font_size=16,
                x=lx, y=ly,
                anchor_x="center", anchor_y="center",
                color=(*color, 0),
            )
            self._intro_labels.append({
                "label": lbl,
                "base_x": lx,
                "base_y": ly,
                "color": color,
                "phase": i * 1.1,  # different sinusoidal phase per key
            })

    # ── GUI menu callbacks ──────────────────────────────────

    def _gui_toggle_camera(self):
        if self.use_camera:
            self.use_camera = False
        else:
            if self.camera is None:
                from camera import Camera
                self.camera = Camera()
            self.use_camera = True
            self.sound.play(AUDIO_CAMERA_ON)

    def _gui_prev_image(self):
        if not self.use_camera:
            self.image_source.prev_image()
            self.sound.play(AUDIO_BONFIRE_LIT)

    def _gui_next_image(self):
        if not self.use_camera:
            self.image_source.next_image()
            self.sound.play(AUDIO_BONFIRE_LIT)

    def _gui_set_mode(self, mode):
        self.mode_ctrl.mode = mode
        self.mode_ctrl._cycle_start = time.monotonic()
        self.sound.play(AUDIO_MODE_CYCLE)

    def _gui_set_volume(self, value):
        if self.sound._ambience_player:
            self.sound._ambience_player.volume = value

    def _gui_toggle_debug(self):
        self.debug.enabled = not self.debug.enabled

    def _gui_toggle_help(self):
        self._show_float_keys()
        self.sound.play(AUDIO_HELP, volume=0.40)

    def _save_screenshot(self):
        """Save current frame to result/ with auto-generated name."""
        os.makedirs(RESULT_DIR, exist_ok=True)
        mode = MODE_NAMES[self.mode_ctrl.mode].split()[0].lower()  # auto/humanity/ember
        state = "ember" if self.mode_ctrl.is_ember else "humanity"
        source = "cam" if self.use_camera else self.image_source.image_name.rsplit(".", 1)[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stamp}_{source}_{mode}_{state}.png"
        path = os.path.join(RESULT_DIR, filename)
        pyglet.image.get_buffer_manager().get_color_buffer().save(path)
        print(f"[Screenshot] Saved: {path}")
        self.overlay.trigger_banner("SCREENSHOT SAVED", (180, 200, 220))

    def _gui_quit(self):
        dur = self.sound.play_quit()
        self.sound.cleanup()
        pyglet.clock.schedule_once(lambda dt: self._do_close(), min(dur, 2.0))

    def _transition_to_intro(self):
        """Transition from loading screen to intro state."""
        self._state = STATE_INTRO
        self._show_float_keys()
        self.sound.play(AUDIO_START)
        self.sound.start_ambience()
        self.sound.play(AUDIO_HELP, volume=0.30)

    def on_key_press(self, symbol, modifiers):
        # Loading screen: only ENTER proceeds
        if self._state == STATE_LOADING:
            if symbol == key.RETURN:
                self._transition_to_intro()
            return

        # Intro state: ignore most keys, allow ESC
        if self._state == STATE_INTRO:
            if symbol == key.ESCAPE:
                self._state = STATE_RUNNING
            return

        if symbol == key.TAB:
            self.menu.sync_state(
                use_camera=self.use_camera,
                mode=self.mode_ctrl.mode,
                debug=self.debug.enabled,
                help_visible=self.overlay._help_visible,
                volume=(self.sound._ambience_player.volume
                        if self.sound._ambience_player else 0.25),
            )
            self.menu.toggle()
            return
        if symbol == key.ESCAPE:
            if self.menu.visible:
                self.menu.toggle()
                return
            dur = self.sound.play_quit()
            self.sound.cleanup()
            # Delay close slightly so quit sound can be heard
            pyglet.clock.schedule_once(lambda dt: self._do_close(), min(dur, 2.0))
        elif symbol == key.D:
            self.debug.enabled = not self.debug.enabled
        elif symbol == key.SPACE:
            self.mode_ctrl.cycle()
            self.sound.play(AUDIO_MODE_CYCLE)
        elif symbol == key.C:
            # Toggle camera mode (lazy-init webcam)
            if self.use_camera:
                self.use_camera = False
            else:
                if self.camera is None:
                    from camera import Camera
                    self.camera = Camera()
                self.use_camera = True
                self.sound.play(AUDIO_CAMERA_ON)
        elif symbol == key.F11:
            self._toggle_fullscreen()
        elif symbol == key.S:
            self._save_screenshot()
        elif symbol == key.H:
            self._show_float_keys()
            self.sound.play(AUDIO_HELP, volume=0.40)
        elif symbol == key.LEFT:
            if not self.use_camera:
                self.image_source.prev_image()
                self.sound.play(AUDIO_BONFIRE_LIT)
        elif symbol == key.RIGHT:
            if not self.use_camera:
                self.image_source.next_image()
                self.sound.play(AUDIO_BONFIRE_LIT)

    def _do_close(self):
        if self.camera:
            self.camera.stop()
        self.close()

    def on_mouse_motion(self, x, y, dx, dy):
        self.menu.on_mouse_motion(x, y)

    def on_mouse_press(self, x, y, button, modifiers):
        if self._state == STATE_LOADING:
            # Click anywhere to start
            self._transition_to_intro()
            return
        self.menu.on_mouse_press(x, y, button)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.menu.on_mouse_drag(x, y)

    def on_mouse_release(self, x, y, button, modifiers):
        self.menu.on_mouse_release(x, y)

    def on_resize(self, width, height):
        super().on_resize(width, height)
        self.ctx.viewport = (0, 0, width, height)

        # Update overlay / debug positions for new dimensions
        self.overlay.resize(width, height)
        self.debug.resize(width, height)

        # Reposition HUD labels
        self._mode_label.y = height - 20
        self._particle_label.y = height - 40
        self._source_label.y = height - 60
        self._hand_label.y = height - 80

        # Reposition clock
        self._clock_label.x = width - 10

        # Reposition loading screen
        if self._loading_bg_sprite:
            img = self._loading_bg_sprite.image
            sx = width / img.width
            sy = height / img.height
            scale = max(sx, sy)
            self._loading_bg_sprite.scale = scale
            self._loading_bg_sprite.x = (width - img.width * scale) / 2
            self._loading_bg_sprite.y = (height - img.height * scale) / 2
        self._loading_overlay.width = width
        self._loading_overlay.height = height
        self._loading_title.x = width // 2
        self._loading_title.y = height // 2 + 60
        self._loading_subtitle.x = width // 2
        self._loading_subtitle.y = height // 2 - 10
        self._loading_version.x = width // 2
        self._loading_start.x = width // 2
        self._loading_start.y = height // 2 - 80

        # Reposition intro floating key labels
        for i, entry in enumerate(self._intro_labels):
            col = i % 3
            row = i // 3
            lx = int(width * (0.2 + col * 0.3))
            ly = int(height * (0.6 - row * 0.25))
            entry["base_x"] = lx
            entry["base_y"] = ly
            entry["label"].x = lx
            entry["label"].y = ly

        # Recreate the GUI menu at the new dimensions
        self.menu = GameMenu(width, height, callbacks={
            "toggle_camera": self._gui_toggle_camera,
            "prev_image":    self._gui_prev_image,
            "next_image":    self._gui_next_image,
            "set_mode_auto":     lambda: self._gui_set_mode(0),
            "set_mode_humanity": lambda: self._gui_set_mode(1),
            "set_mode_ember":    lambda: self._gui_set_mode(2),
            "set_volume":    self._gui_set_volume,
            "toggle_debug":  self._gui_toggle_debug,
            "toggle_help":   self._gui_toggle_help,
            "quit":          self._gui_quit,
        })

    def _toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen
        self.set_fullscreen(self._is_fullscreen)

    def _draw_loading(self, dt):
        """Draw the loading/title screen."""
        self._loading_time += dt

        # Draw background image scaled to fill window
        if self._loading_bg_sprite:
            self._loading_bg_sprite.draw()

        # Dark overlay for readability
        self._loading_overlay.draw()

        # Title, subtitle, version
        self._loading_title.draw()
        self._loading_subtitle.draw()
        self._loading_version.draw()

        # Pulsing "PRESS ENTER" text
        pulse = int((math.sin(self._loading_time * 2.5) * 0.5 + 0.5) * 255)
        self._loading_start.color = (200, 168, 78, pulse)
        self._loading_start.draw()

    def _show_float_keys(self):
        """Start (or restart) the floating key help animation."""
        self._float_keys_timer = 0.0
        self._float_keys_active = True

    def _draw_float_keys(self, dt):
        """Draw floating key labels with fade and drift. Returns True while active."""
        if not self._float_keys_active:
            return False

        self._float_keys_timer += dt
        t = self._float_keys_timer

        # Fade in 0-1s, full 1-4s, fade out 4-5s
        if t < 1.0:
            alpha_factor = t
        elif t < 4.0:
            alpha_factor = 1.0
        elif t < INTRO_DURATION:
            alpha_factor = max(0.0, 1.0 - (t - 4.0))
        else:
            self._float_keys_active = False
            return False

        for entry in self._intro_labels:
            lbl = entry["label"]
            r, g, b = entry["color"]
            y_offset = math.sin(t * 1.5 + entry["phase"]) * 12.0
            lbl.x = entry["base_x"]
            lbl.y = int(entry["base_y"] + y_offset)
            lbl.color = (r, g, b, int(alpha_factor * 230))
            lbl.draw()
        return True

    def _draw_intro(self, dt):
        """Draw particle system + floating help keys during intro."""
        # Run the normal particle simulation
        now = time.monotonic()
        self.mode_ctrl.update_image(now)
        self.particles.spawn(self.image_source, self.mode_ctrl.is_ember)
        self.particles.update(dt, self.mode_ctrl.is_ember)
        self.sound.update(self.mode_ctrl.is_ember)

        gpu_data = self.particles.pack_gpu()
        n = self.particles.count
        if n > 0:
            data_bytes = gpu_data.tobytes()
            self._vbo.orphan()
            self._vbo.write(data_bytes)
            self._vao.render(moderngl.POINTS, vertices=n)

        # Draw floating keys; transition when done
        if not self._draw_float_keys(dt):
            self._state = STATE_RUNNING

    def on_draw(self):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        dt = 1.0 / 60.0

        if self._state == STATE_LOADING:
            self._draw_loading(dt)
            return

        if self._state == STATE_INTRO:
            self._draw_intro(dt)
            return

        # --- STATE_RUNNING (original logic) ---
        now = time.monotonic()

        if self.use_camera and self.camera:
            # Camera path: motion-based mode switching + hand gestures
            brightness, motion, avg_motion = self.camera.get_data()
            hand_data = self.camera.get_hand_data()
            self.mode_ctrl.update_camera(avg_motion, now, hand_data.is_open_palm)
            self.particles.spawn_camera(brightness, motion, self.mode_ctrl.is_ember)
            if hand_data.detected and hand_data.is_open_palm:
                self.particles.recolor_fire_gradient(hand_data.palm_ndc_x, hand_data.palm_ndc_y)
                self.particles.spawn_palm_sparks(hand_data.palm_ndc_x, hand_data.palm_ndc_y)
                # Play palm burst sound on open palm transition
                if not self._prev_palm_open:
                    self.sound.play(AUDIO_BOSS_OUT, volume=0.35)
                self._prev_palm_open = True
            else:
                self._prev_palm_open = False
        else:
            # Image path: time-based mode cycling
            self.mode_ctrl.update_image(now)
            self.particles.spawn(self.image_source, self.mode_ctrl.is_ember)

        self.particles.update(dt, self.mode_ctrl.is_ember)
        self.sound.update(self.mode_ctrl.is_ember)

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
                hand_data_dbg = self.camera.get_hand_data()
            else:
                preview = self.image_source.get_preview()
                hand_data_dbg = None
            self.debug.draw(preview)
            if hand_data_dbg is not None:
                self.debug.draw_hand(hand_data_dbg)

            mode_name = MODE_NAMES[self.mode_ctrl.mode]
            state = "EMBER" if self.mode_ctrl.is_ember else "Humanity"
            source = "Camera" if self.use_camera else f"Image: {self.image_source.image_name}"
            self._mode_label.text = f"Mode: {mode_name} | State: {state}"
            self._mode_label.draw()
            self._particle_label.text = f"Particles: {self.particles.count}"
            self._particle_label.draw()
            self._source_label.text = f"Source: {source} [{self.image_source.image_count} images]"
            self._source_label.draw()

            # Hand tracking debug panel (bottom-right: skeleton + finger status)
            if self.use_camera and self.camera:
                hand_data_panel = self.camera.get_hand_data()
                ema = self.camera.get_hand_ema()
                self.debug.draw_hand_panel(hand_data_panel, ema)

                # Top-left summary line
                if hand_data_panel.detected:
                    palm_state = "OPEN PALM" if hand_data_panel.is_open_palm else "CLOSED"
                    self._hand_label.text = (
                        f"Hand: {palm_state} | Palm NDC: "
                        f"({hand_data_panel.palm_ndc_x:.2f}, {hand_data_panel.palm_ndc_y:.2f})"
                    )
                else:
                    self._hand_label.text = "Hand: not detected"
                self._hand_label.draw()

        # Soul overlay (banners, quotes)
        image_name = None if self.use_camera else self.image_source.image_name
        self.overlay.update(dt, self.mode_ctrl.is_ember, image_name)
        self.overlay.draw()

        # Floating key help (triggered by H key)
        self._draw_float_keys(dt)

        # Wall clock (bottom-right)
        now_str = datetime.now().strftime("%H:%M:%S")
        self._clock_label.text = f"{now_str}  {self._tz_name}"
        self._clock_label.draw()

        # GUI menu overlay — always last (on top of everything)
        self.menu.draw()

    def on_close(self):
        self.sound.cleanup()
        super().on_close()


def main():
    app = SoulStreamApp()
    pyglet.clock.schedule_interval(lambda dt: None, 1 / 60)
    pyglet.app.run()


if __name__ == "__main__":
    main()
