"""
Overlay and controller classes extracted from main.py.

Contains: SoundManager, ModeController, DebugOverlay, SoulOverlay,
          GestureCornerOverlay, YouDiedOverlay
"""

import os
import time
import random
import numpy as np
import pyglet
import moderngl

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SRC_DIR)
AUDIO_DIR = os.path.join(_ROOT_DIR, "audio")
IMAGE_DIR  = os.path.join(_ROOT_DIR, "image")

WIDTH, HEIGHT = 1280, 720

# Audio file assignments
AUDIO_OPENING           = "dark-souls-the-ancient-dragon-choir.mp3"
AUDIO_EMBER_IGNITE      = "dark-souls-kill.mp3"
AUDIO_HUMANITY_RESTORED = "dark-souls-im-sorry.mp3"
AUDIO_BONFIRE_LIT       = "darksoul_bonfire_jump.mp3"
AUDIO_CAMERA_ON         = "hello-darksoul3.mp3"
AUDIO_HELP              = "firekeeper.mp3"
AUDIO_QUIT              = "thank-you-dark-souls.mp3"
AUDIO_MODE_CYCLE        = "darksouls-pain.mp3"
AUDIO_BOSS_OUT          = "bossout.mp3"
AUDIO_START             = "i-offer-you-an-accord.mp3"
AUDIO_PRAISE_SUN        = "bossout.mp3"
AUDIO_YOU_DIED          = "you-died-dark-souls.mp3"

OPENING_VOLUME = 0.25
SFX_VOLUME     = 0.60

# Mode constants
MODE_AUTO           = 0
MODE_FORCE_HUMANITY = 1
MODE_FORCE_EMBER    = 2
MODE_NAMES          = ["Auto", "Humanity (forced)", "Ember (forced)"]

# Camera mode thresholds
EMBER_ENTER   = 0.06
EMBER_EXIT    = 0.03
EMBER_COOLDOWN = 2.0

# Image mode time-based cycle
IMAGE_HUMANITY_DURATION = 12.0
IMAGE_EMBER_DURATION    = 8.0

# MediaPipe hand skeleton connections (21 landmarks, 21 bones)
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]

_FINGERTIPS    = {4, 8, 12, 16, 20}
_FINGER_NAMES  = ["thumb", "index", "middle", "ring", "pinky"]

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
    "H       Hand tracker toggle\n"
    "P       Pose detect toggle\n"
    "V / 1-5 Visualization mode\n"
    "PALM    Open palm = kindle ember\n"
    "D       Debug overlay\n"
    "F1      This help\n"
    "ESC     Quit"
)

# Banner timing (seconds)
_BANNER_FADE_IN  = 0.5
_BANNER_HOLD     = 2.0
_BANNER_FADE_OUT = 1.0
_BANNER_TOTAL    = _BANNER_FADE_IN + _BANNER_HOLD + _BANNER_FADE_OUT

# Quote timing (seconds)
_QUOTE_DISPLAY  = 8.0
_QUOTE_FADE_OUT = 2.0
_QUOTE_FADE_IN  = 2.0
_QUOTE_CYCLE    = _QUOTE_DISPLAY + _QUOTE_FADE_OUT + _QUOTE_FADE_IN

# Pose info: name → (image_file, display_label, banner_text, audio_file)
_POSE_INFO = {
    'praise_sun': ("praise_the_sun.png", "Praise the Sun",  "PRAISE THE SUN!",     AUDIO_PRAISE_SUN),
    'point_down': ("_point_down.png",    "Point Down",       "YOU DIED",            AUDIO_YOU_DIED),
    'point_up':   ("_point_up.png",      "Point Up",         "SEEK THE DARK SOUL",  "firekeeper.mp3"),
    'bow':        ("_bow.png",           "Bow",              "I HAVE FAILED...",    "dark-souls-im-sorry.mp3"),
    'wave':       ("_wave.png",          "Wave",             "FAREWELL, UNKINDLED", "hello-darksoul3.mp3"),
    'welcome':    ("welcome.png",        "Welcome",          "WELCOME, UNDEAD",     "i-offer-you-an-accord.mp3"),
    'applause':   ("applause.png",       "Applause",         "MAGNIFICENT!",        "darksoul_bonfire_jump.mp3"),
}

_GESTURE_DISPLAY_H = 150
_GESTURE_PADDING   = 20
_GESTURE_TEXT_GAP  = 8
_GESTURE_FADE_IN   = 0.4
_GESTURE_HOLD      = 3.2
_GESTURE_FADE_OUT  = 0.9
_GESTURE_TOTAL     = _GESTURE_FADE_IN + _GESTURE_HOLD + _GESTURE_FADE_OUT

_YOU_DIED_DURATION = 4.0
_YOU_DIED_FADE_IN  = 0.8
_YOU_DIED_HOLD     = 2.4
_YOU_DIED_FADE_OUT = 0.8


# ── Sound Manager ──────────────────────────────────────────────────────────────

class SoundManager:
    """Handles background ambience and all interaction sound effects."""

    def __init__(self):
        self._ambience_player = None
        self._ambience_source = None
        self._sounds = {}
        self._prev_ember = False

        self._ambience_source = self._load_source(AUDIO_OPENING)

        for name in (AUDIO_EMBER_IGNITE, AUDIO_HUMANITY_RESTORED, AUDIO_BONFIRE_LIT,
                     AUDIO_CAMERA_ON, AUDIO_HELP, AUDIO_QUIT, AUDIO_MODE_CYCLE,
                     AUDIO_BOSS_OUT, AUDIO_START, AUDIO_YOU_DIED):
            src = self._load_source(name)
            if src is not None:
                self._sounds[name] = src

    def start_ambience(self):
        if self._ambience_player is not None:
            return
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
        src = self._sounds.get(filename)
        if src is None:
            return
        try:
            player = src.play()
            player.volume = volume if volume is not None else SFX_VOLUME
        except Exception as e:
            print(f"[SoundManager] Error playing '{filename}': {e}")

    def play_quit(self):
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
        if is_ember and not self._prev_ember:
            self.play(AUDIO_EMBER_IGNITE)
        elif not is_ember and self._prev_ember:
            self.play(AUDIO_HUMANITY_RESTORED)
        self._prev_ember = is_ember

    def cleanup(self):
        try:
            if self._ambience_player:
                self._ambience_player.pause()
                self._ambience_player = None
        except Exception:
            pass


# ── Mode Controller ────────────────────────────────────────────────────────────

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
        if self.mode == MODE_FORCE_HUMANITY:
            self.is_ember = False
            return
        if self.mode == MODE_FORCE_EMBER:
            self.is_ember = True
            return

        if avg_motion > EMBER_ENTER or hand_is_open_palm:
            self._last_high = now
            if not self.is_ember:
                self.is_ember = True
                self._ember_since = now

        if self.is_ember and avg_motion < EMBER_EXIT and not hand_is_open_palm:
            if now - self._last_high > EMBER_COOLDOWN:
                self.is_ember = False

    def update_image(self, now):
        if self.mode == MODE_FORCE_HUMANITY:
            self.is_ember = False
            return
        if self.mode == MODE_FORCE_EMBER:
            self.is_ember = True
            return

        cycle_len = IMAGE_HUMANITY_DURATION + IMAGE_EMBER_DURATION
        elapsed = (now - self._cycle_start) % cycle_len
        self.is_ember = elapsed >= IMAGE_HUMANITY_DURATION


# ── Debug Overlay ──────────────────────────────────────────────────────────────

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
        self._hand_vbo = self._ctx.buffer(reserve=63 * 5 * 4)
        self._hand_line_vao = self._ctx.vertex_array(
            self._hand_prog,
            [(self._hand_vbo, "2f 3f", "in_pos", "in_color")],
        )

    def _rebuild_preview_quad(self, win_w, win_h):
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
        self._win_w = win_w
        self._win_h = win_h
        self._rebuild_preview_quad(win_w, win_h)
        if hasattr(self, "_hand_panel_labels"):
            del self._hand_panel_labels

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

        if is_open:
            line_color = (0.2, 1.0, 0.2)
            joint_color = (1.0, 1.0, 0.0)
        else:
            line_color = (0.0, 0.8, 0.8)
            joint_color = (0.0, 0.9, 1.0)

        tip_to_name = {4: "thumb", 8: "index", 12: "middle", 16: "ring", 20: "pinky"}

        buf = []
        for a, b in _HAND_CONNECTIONS:
            ax, ay = lm[a]
            bx, by = lm[b]
            buf.extend([ax, ay, *line_color, bx, by, *line_color])

        line_count = len(_HAND_CONNECTIONS) * 2

        for i, (x, y) in enumerate(lm):
            if i in _FINGERTIPS:
                fname = tip_to_name[i]
                if finger_states.get(fname, False):
                    c = (0.2, 1.0, 0.2)
                else:
                    c = (1.0, 0.1, 0.1)
            else:
                c = joint_color
            buf.extend([x, y, *c])

        joint_count = len(lm)

        data = np.array(buf, dtype="f4")
        self._hand_vbo.orphan()
        self._hand_vbo.write(data.tobytes())

        self._hand_line_vao.render(moderngl.LINES, vertices=line_count)
        self._hand_line_vao.render(moderngl.POINTS, vertices=joint_count, first=line_count)

    def draw_hand_panel(self, hand_data, ema_confidence):
        """Draw hand tracking status panel (pyglet labels) in bottom-right."""
        if not self.enabled:
            return

        if not hasattr(self, "_hand_panel_labels"):
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

        if is_open:
            self._hand_status_label.text = "OPEN PALM"
            self._hand_status_label.color = (80, 255, 80, 255)
        else:
            self._hand_status_label.text = "CLOSED"
            self._hand_status_label.color = (80, 200, 255, 220)
        self._hand_status_label.draw()

        bar_len = 20
        filled = int(min(ema_confidence, 1.0) * bar_len)
        bar = "|" + "#" * filled + "-" * (bar_len - filled) + "|"
        self._hand_ema_label.text = f"EMA: {ema_confidence:.3f} {bar}"
        self._hand_ema_label.draw()

        for i, fname in enumerate(_FINGER_NAMES):
            extended = finger_states.get(fname, False)
            marker = "[X]" if extended else "[ ]"
            self._hand_finger_labels[i].text = f"{marker} {fname}"
            if extended:
                self._hand_finger_labels[i].color = (80, 255, 80, 220)
            else:
                self._hand_finger_labels[i].color = (255, 80, 80, 220)
            self._hand_finger_labels[i].draw()

        self._hand_ndc_label.text = (
            f"Palm NDC: ({hand_data.palm_ndc_x:.2f}, {hand_data.palm_ndc_y:.2f})"
        )
        self._hand_ndc_label.draw()


# ── Soul Overlay ───────────────────────────────────────────────────────────────

class SoulOverlay:
    def __init__(self):
        self._banner_label = pyglet.text.Label(
            "", font_name="Georgia", font_size=48,
            x=WIDTH // 2, y=HEIGHT // 2,
            anchor_x="center", anchor_y="center",
            color=(255, 255, 255, 0),
        )
        self._banner_timer = 0.0
        self._banner_active = False
        self._banner_color = (255, 255, 255)

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

        self._prev_ember = False
        self._prev_image_name = None

    def resize(self, win_w, win_h):
        self._banner_label.x = win_w // 2
        self._banner_label.y = win_h // 2
        self._quote_label.x = win_w // 2
        for i, lbl in enumerate(self._help_labels):
            lbl.x = win_w - 20
            lbl.y = win_h - 30 - i * 20

    def trigger_banner(self, text, color):
        self._banner_label.text = text
        self._banner_color = color
        self._banner_timer = 0.0
        self._banner_active = True

    def toggle_help(self):
        self._help_visible = not self._help_visible

    def update(self, dt, is_ember, image_name=None):
        if is_ember != self._prev_ember:
            if is_ember:
                self.trigger_banner("HEIR OF FIRE RESTORED", (255, 200, 80))
            else:
                self.trigger_banner("HUMANITY RESTORED", (200, 210, 220))
        self._prev_ember = is_ember

        if image_name is not None and self._prev_image_name is not None:
            if image_name != self._prev_image_name:
                self.trigger_banner("BONFIRE LIT", (255, 160, 40))
        self._prev_image_name = image_name

        if self._banner_active:
            self._banner_timer += dt
            if self._banner_timer >= _BANNER_TOTAL:
                self._banner_active = False

        self._quote_timer += dt
        if self._quote_timer >= _QUOTE_CYCLE:
            self._quote_timer = 0.0
            self._quote_idx = (self._quote_idx + 1) % len(self._quotes)
            self._quote_label.text = self._quotes[self._quote_idx]

    def draw(self):
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

        t = self._quote_timer
        if t < _QUOTE_DISPLAY:
            alpha = 1.0
        elif t < _QUOTE_DISPLAY + _QUOTE_FADE_OUT:
            alpha = 1.0 - (t - _QUOTE_DISPLAY) / _QUOTE_FADE_OUT
        else:
            alpha = (t - _QUOTE_DISPLAY - _QUOTE_FADE_OUT) / _QUOTE_FADE_IN
        alpha = max(0.0, min(1.0, alpha))
        self._quote_label.color = (160, 150, 130, int(alpha * 200))
        self._quote_label.draw()

        if self._help_visible:
            for lbl in self._help_labels:
                lbl.draw()


# ── Gesture Corner Overlay ─────────────────────────────────────────────────────

class GestureCornerOverlay:
    """Top-right corner gesture image + pose name label."""

    GESTURE_DIR = os.path.join(IMAGE_DIR, "gesture")

    def __init__(self, win_w, win_h):
        self._sprites = {}
        self._positions = {}
        self._current = None
        self._timer = 0.0
        self._active = False
        self._win_w = win_w
        self._win_h = win_h

        self._label = pyglet.text.Label(
            "", font_name="Georgia", font_size=15,
            x=0, y=0,
            anchor_x="center", anchor_y="top",
            color=(200, 168, 78, 0),
        )

        self._load_sprites()
        self._calc_positions(win_w, win_h)

    def _load_sprites(self):
        for pose, (filename, *_) in _POSE_INFO.items():
            path = os.path.join(self.GESTURE_DIR, filename)
            try:
                img = pyglet.image.load(path)
                sprite = pyglet.sprite.Sprite(img)
                scale = _GESTURE_DISPLAY_H / img.height
                sprite.scale = scale
                self._sprites[pose] = sprite
            except Exception as e:
                print(f"[GestureOverlay] Could not load {filename}: {e}")

    def _calc_positions(self, win_w, win_h):
        for pose, sprite in self._sprites.items():
            sw = sprite.image.width * sprite.scale
            sh = sprite.image.height * sprite.scale
            sx = win_w - _GESTURE_PADDING - sw
            sy = win_h - _GESTURE_PADDING - sh
            label_x = sx + sw / 2
            label_y = sy - _GESTURE_TEXT_GAP
            self._positions[pose] = (int(sx), int(sy), label_x, label_y)

    def show(self, pose_name):
        if pose_name not in self._sprites:
            return
        self._current = pose_name
        self._timer = 0.0
        self._active = True
        sx, sy, lx, ly = self._positions[pose_name]
        self._sprites[pose_name].x = sx
        self._sprites[pose_name].y = sy
        self._label.x = lx
        self._label.y = ly
        self._label.text = _POSE_INFO[pose_name][1]

    def update(self, dt):
        if self._active:
            self._timer += dt
            if self._timer >= _GESTURE_TOTAL:
                self._active = False

    def resize(self, win_w, win_h):
        self._win_w = win_w
        self._win_h = win_h
        self._calc_positions(win_w, win_h)
        if self._active and self._current and self._current in self._positions:
            sx, sy, lx, ly = self._positions[self._current]
            self._sprites[self._current].x = sx
            self._sprites[self._current].y = sy
            self._label.x = lx
            self._label.y = ly

    def draw(self):
        if not self._active or self._current not in self._sprites:
            return
        t = self._timer
        if t < _GESTURE_FADE_IN:
            alpha = t / _GESTURE_FADE_IN
        elif t < _GESTURE_FADE_IN + _GESTURE_HOLD:
            alpha = 1.0
        elif t < _GESTURE_TOTAL:
            alpha = 1.0 - (t - _GESTURE_FADE_IN - _GESTURE_HOLD) / _GESTURE_FADE_OUT
        else:
            return
        alpha = max(0.0, min(1.0, alpha))
        a_int = int(alpha * 230)
        self._sprites[self._current].opacity = a_int
        self._sprites[self._current].draw()
        self._label.color = (200, 168, 78, int(alpha * 220))
        self._label.draw()


# ── You Died Overlay ───────────────────────────────────────────────────────────

class YouDiedOverlay:
    """Full-screen blood-red "YOU DIED" overlay, Dark Souls style."""

    def __init__(self, win_w, win_h):
        self._bg = pyglet.shapes.Rectangle(0, 0, win_w, win_h, color=(60, 0, 0))
        self._bg.opacity = 0
        self._label = pyglet.text.Label(
            "YOU DIED", font_name="Georgia", font_size=80,
            x=win_w // 2, y=win_h // 2,
            anchor_x="center", anchor_y="center",
            color=(200, 20, 20, 0),
        )
        self._timer = 0.0
        self._active = False

    def trigger(self):
        self._timer = 0.0
        self._active = True

    def update(self, dt):
        if self._active:
            self._timer += dt
            if self._timer >= _YOU_DIED_DURATION:
                self._active = False

    def resize(self, win_w, win_h):
        self._bg.width = win_w
        self._bg.height = win_h
        self._label.x = win_w // 2
        self._label.y = win_h // 2

    def draw(self):
        if not self._active:
            return
        t = self._timer
        if t < _YOU_DIED_FADE_IN:
            alpha = t / _YOU_DIED_FADE_IN
        elif t < _YOU_DIED_FADE_IN + _YOU_DIED_HOLD:
            alpha = 1.0
        elif t < _YOU_DIED_DURATION:
            alpha = 1.0 - (t - _YOU_DIED_FADE_IN - _YOU_DIED_HOLD) / _YOU_DIED_FADE_OUT
        else:
            return
        alpha = max(0.0, min(1.0, alpha))
        self._bg.opacity = int(alpha * 140)
        self._bg.draw()
        self._label.color = (200, 20, 20, int(alpha * 255))
        self._label.draw()
