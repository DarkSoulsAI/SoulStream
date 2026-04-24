import os
import math
import time
from datetime import datetime
import numpy as np
import pyglet
from pyglet.window import key
import moderngl

from image_source import ImageSource
import particles as _particles_mod
from particles import ParticleSystem, MAX_PARTICLES, SPAWN_PER_FRAME
from gui import GameMenu
from overlays import (
    SoundManager, ModeController, DebugOverlay, SoulOverlay,
    GestureCornerOverlay, YouDiedOverlay,
    MODE_NAMES, _POSE_INFO,
    AUDIO_BONFIRE_LIT, AUDIO_CAMERA_ON, AUDIO_HELP, AUDIO_MODE_CYCLE,
    AUDIO_BOSS_OUT, AUDIO_START,
)
from visualization import (
    VisualizationMode, VIZ_NAMES,
    PointsRenderer, TrailsRenderer, SkeletonRenderer,
    FlowRenderer, ShapesRenderer,
)

_SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR  = os.path.dirname(_SRC_DIR)
SHADER_DIR = os.path.join(_ROOT_DIR, "shaders")
IMAGE_DIR  = os.path.join(_ROOT_DIR, "image")
AUDIO_DIR  = os.path.join(_ROOT_DIR, "audio")
RESULT_DIR = os.path.join(_ROOT_DIR, "result")
TIMELAPSE_DIR = os.path.join(RESULT_DIR, "timelapse")

WIDTH, HEIGHT = 1280, 720

# App states
STATE_LOADING = 0
STATE_INTRO   = 1
STATE_RUNNING = 2
INTRO_DURATION = 5.0

# Floating intro key definitions  (H → hand tracker, F1 → help)
_INTRO_KEYS = [
    ("SPACE",  "Cycle Modes",   (255, 140, 50)),
    ("\u2190 \u2192",   "Change Image",  (80, 220, 255)),
    ("C",      "Toggle Camera", (80, 255, 120)),
    ("F1",     "Help",          (255, 100, 220)),
    ("TAB",    "Menu",          (200, 168, 78)),
    ("ESC",    "Quit",          (255, 80, 80)),
]


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
        self.camera = None  # lazy-initialized on C key
        self._source_transition = 0.0

        self.particles = ParticleSystem()
        self.mode_ctrl = ModeController()
        self.debug = DebugOverlay(self.ctx)
        self.overlay = SoulOverlay()
        self.sound = SoundManager()
        self._prev_palm_open = False

        # Timelapse auto-screenshot
        self._timelapse_enabled = False
        self._timelapse_timer = 0.0
        self._timelapse_index = 0

        # Pose detection state
        self._gesture_overlay = GestureCornerOverlay(WIDTH, HEIGHT)
        self._you_died_overlay = YouDiedOverlay(WIDTH, HEIGHT)
        self._prev_pose = None
        self._pose_cooldown = 0.0

        # Camera feature toggles (independent of use_camera)
        self._hand_enabled = True
        self._pose_enabled = True

        # Visualization mode
        self._viz_mode = VisualizationMode.POINTS
        self._renderers = {
            VisualizationMode.POINTS:   PointsRenderer(),
            VisualizationMode.TRAILS:   TrailsRenderer(),
            VisualizationMode.SKELETON: SkeletonRenderer(),
            VisualizationMode.FLOW:     FlowRenderer(),
            VisualizationMode.SHAPES:   ShapesRenderer(),
        }
        for r in self._renderers.values():
            r.setup(self.ctx)

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

        self._clock_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=12,
            x=WIDTH - 10, y=10,
            anchor_x="right", anchor_y="bottom",
            color=(160, 150, 130, 180),
        )
        self._tz_name = datetime.now().astimezone().strftime("%Z")

        self.menu = GameMenu(WIDTH, HEIGHT, callbacks=self._build_callbacks())

        # --- Loading screen assets ---
        self._loading_bg_sprite = None
        try:
            bg_path = os.path.join(IMAGE_DIR, "darksouls1.jpg")
            bg_img = pyglet.image.load(bg_path)
            self._loading_bg_sprite = pyglet.sprite.Sprite(bg_img)
            sx = WIDTH / bg_img.width
            sy = HEIGHT / bg_img.height
            scale = max(sx, sy)
            self._loading_bg_sprite.scale = scale
            self._loading_bg_sprite.x = (WIDTH - bg_img.width * scale) / 2
            self._loading_bg_sprite.y = (HEIGHT - bg_img.height * scale) / 2
        except Exception as e:
            print(f"[Loading] Could not load background: {e}")

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
            "v2.0", font_name="Consolas", font_size=12,
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
        for i, (key_name, desc, color) in enumerate(_INTRO_KEYS):
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
                "phase": i * 1.1,
            })

    # ── callbacks dict ──────────────────────────────────────

    def _build_callbacks(self):
        return {
            "toggle_camera":     self._gui_toggle_camera,
            "prev_image":        self._gui_prev_image,
            "next_image":        self._gui_next_image,
            "set_mode_auto":     lambda: self._gui_set_mode(0),
            "set_mode_humanity": lambda: self._gui_set_mode(1),
            "set_mode_ember":    lambda: self._gui_set_mode(2),
            "toggle_hand":       self._gui_toggle_hand,
            "toggle_pose":       self._gui_toggle_pose,
            "set_viz":           self._gui_set_viz,
            "set_volume":        self._gui_set_volume,
            "toggle_debug":      self._gui_toggle_debug,
            "toggle_help":       self._gui_toggle_help,
            "quit":              self._gui_quit,
        }

    # ── GUI menu callbacks ──────────────────────────────────

    def _gui_toggle_camera(self):
        if self.use_camera:
            self.use_camera = False
            self._source_transition = 1.5
        else:
            if self.camera is None:
                from camera import Camera
                self.camera = Camera(
                    enable_hand=self._hand_enabled,
                    enable_pose=self._pose_enabled,
                )
                self.overlay.trigger_banner("KINDLING CAMERA...", (255, 160, 40))
            self.use_camera = True
            self._source_transition = 1.5
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

    def _gui_toggle_hand(self):
        self._hand_enabled = not self._hand_enabled
        if self.camera:
            self.camera.hand_enabled = self._hand_enabled
        label = "HAND TRACKER ON" if self._hand_enabled else "HAND TRACKER OFF"
        self.overlay.trigger_banner(label, (80, 220, 255))

    def _gui_toggle_pose(self):
        self._pose_enabled = not self._pose_enabled
        if self.camera:
            self.camera.pose_enabled = self._pose_enabled
        label = "POSE DETECT ON" if self._pose_enabled else "POSE DETECT OFF"
        self.overlay.trigger_banner(label, (80, 220, 255))

    def _gui_set_viz(self, mode_int: int):
        self._viz_mode = VisualizationMode(mode_int)
        self.overlay.trigger_banner(f"VIZ: {VIZ_NAMES[mode_int]}", (200, 168, 78))
        # Sync viz radio buttons when called from menu
        for btn in self.menu._viz_buttons:
            btn.active = (btn.group_value == mode_int)

    def _save_screenshot(self):
        os.makedirs(RESULT_DIR, exist_ok=True)
        mode = MODE_NAMES[self.mode_ctrl.mode].split()[0].lower()
        state = "ember" if self.mode_ctrl.is_ember else "humanity"
        source = "cam" if self.use_camera else self.image_source.image_name.rsplit(".", 1)[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{stamp}_{source}_{mode}_{state}.png"
        path = os.path.join(RESULT_DIR, filename)
        pyglet.image.get_buffer_manager().get_color_buffer().save(path)
        print(f"[Screenshot] Saved: {path}")
        self.overlay.trigger_banner("SCREENSHOT SAVED", (180, 200, 220))

    def _toggle_timelapse(self):
        self._timelapse_enabled = not self._timelapse_enabled
        if self._timelapse_enabled:
            self._timelapse_index = 0
            self._timelapse_timer = 0.0
            os.makedirs(TIMELAPSE_DIR, exist_ok=True)
            self.overlay.trigger_banner("TIMELAPSE ON", (180, 220, 255))
        else:
            self.overlay.trigger_banner("TIMELAPSE OFF", (180, 180, 180))

    def _timelapse_save(self):
        path = os.path.join(TIMELAPSE_DIR, f"{self._timelapse_index}.png")
        pyglet.image.get_buffer_manager().get_color_buffer().save(path)
        self._timelapse_index += 1

    def _gui_quit(self):
        dur = self.sound.play_quit()
        self.sound.cleanup()
        pyglet.clock.schedule_once(lambda dt: self._do_close(), min(dur, 2.0))

    def _trigger_pose_effect(self, pose_name, pose_data):
        if pose_name not in _POSE_INFO:
            return
        _, _label, banner_text, audio_file = _POSE_INFO[pose_name]

        self._gesture_overlay.show(pose_name)

        if pose_name == 'point_down':
            self.overlay.trigger_banner(banner_text, (220, 30, 30))
        elif pose_name == 'praise_sun':
            self.overlay.trigger_banner(banner_text, (255, 220, 50))
        elif pose_name == 'bow':
            self.overlay.trigger_banner(banner_text, (180, 180, 200))
        else:
            self.overlay.trigger_banner(banner_text, (200, 168, 78))

        self.sound.play(audio_file, volume=0.55)

        if pose_name == 'praise_sun':
            self.particles.spawn_praise_sun(pose_data.left_wrist_ndc, pose_data.right_wrist_ndc)
        elif pose_name == 'point_down':
            self._you_died_overlay.trigger()
            self.particles.spawn_you_died()

    def _transition_to_intro(self):
        self._state = STATE_INTRO
        self._show_float_keys()
        self.sound.play(AUDIO_START)
        self.sound.start_ambience()
        self.sound.play(AUDIO_HELP, volume=0.30)

    # ── Input handlers ──────────────────────────────────────

    def on_key_press(self, symbol, modifiers):
        if self._state == STATE_LOADING:
            if symbol == key.RETURN:
                self._transition_to_intro()
            return

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
                hand_enabled=self._hand_enabled,
                pose_enabled=self._pose_enabled,
                viz_mode=int(self._viz_mode),
            )
            self.menu.toggle()
            return
        if symbol == key.ESCAPE:
            if self.menu.visible:
                self.menu.toggle()
                return
            dur = self.sound.play_quit()
            self.sound.cleanup()
            pyglet.clock.schedule_once(lambda dt: self._do_close(), min(dur, 2.0))
        elif symbol == key.D:
            self.debug.enabled = not self.debug.enabled
        elif symbol == key.SPACE:
            self.mode_ctrl.cycle()
            self.sound.play(AUDIO_MODE_CYCLE)
        elif symbol == key.C:
            if self.use_camera:
                self.use_camera = False
                self._source_transition = 1.5
            else:
                if self.camera is None:
                    from camera import Camera
                    self.camera = Camera(
                        enable_hand=self._hand_enabled,
                        enable_pose=self._pose_enabled,
                    )
                    self.overlay.trigger_banner("KINDLING CAMERA...", (255, 160, 40))
                self.use_camera = True
                self._source_transition = 1.5
                self.sound.play(AUDIO_CAMERA_ON)
        elif symbol == key.H:
            self._gui_toggle_hand()
        elif symbol == key.P:
            self._gui_toggle_pose()
        elif symbol == key.V:
            self._gui_set_viz((int(self._viz_mode) + 1) % 5)
        elif symbol == key._1:
            self._gui_set_viz(0)
        elif symbol == key._2:
            self._gui_set_viz(1)
        elif symbol == key._3:
            self._gui_set_viz(2)
        elif symbol == key._4:
            self._gui_set_viz(3)
        elif symbol == key._5:
            self._gui_set_viz(4)
        elif symbol == key.F1:
            self._show_float_keys()
            self.sound.play(AUDIO_HELP, volume=0.40)
        elif symbol == key.F11:
            self._toggle_fullscreen()
        elif symbol == key.S:
            self._save_screenshot()
        elif symbol == key.T:
            self._toggle_timelapse()
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

        self.overlay.resize(width, height)
        self.debug.resize(width, height)

        self._mode_label.y = height - 20
        self._particle_label.y = height - 40
        self._source_label.y = height - 60
        self._hand_label.y = height - 80

        self._clock_label.x = width - 10

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

        for i, entry in enumerate(self._intro_labels):
            col = i % 3
            row = i // 3
            lx = int(width * (0.2 + col * 0.3))
            ly = int(height * (0.6 - row * 0.25))
            entry["base_x"] = lx
            entry["base_y"] = ly
            entry["label"].x = lx
            entry["label"].y = ly

        self._you_died_overlay.resize(width, height)
        self._gesture_overlay.resize(width, height)

        self.menu = GameMenu(width, height, callbacks=self._build_callbacks())
        self.menu.sync_state(
            use_camera=self.use_camera,
            mode=self.mode_ctrl.mode,
            debug=self.debug.enabled,
            help_visible=self.overlay._help_visible,
            volume=(self.sound._ambience_player.volume
                    if self.sound._ambience_player else 0.25),
            hand_enabled=self._hand_enabled,
            pose_enabled=self._pose_enabled,
            viz_mode=int(self._viz_mode),
        )

    def _toggle_fullscreen(self):
        self._is_fullscreen = not self._is_fullscreen
        self.set_fullscreen(self._is_fullscreen)

    # ── Draw helpers ────────────────────────────────────────

    def _draw_loading(self, dt):
        self._loading_time += dt
        if self._loading_bg_sprite:
            self._loading_bg_sprite.draw()
        self._loading_overlay.draw()
        self._loading_title.draw()
        self._loading_subtitle.draw()
        self._loading_version.draw()
        pulse = int((math.sin(self._loading_time * 2.5) * 0.5 + 0.5) * 255)
        self._loading_start.color = (200, 168, 78, pulse)
        self._loading_start.draw()

    def _show_float_keys(self):
        self._float_keys_timer = 0.0
        self._float_keys_active = True

    def _draw_float_keys(self, dt):
        if not self._float_keys_active:
            return False

        self._float_keys_timer += dt
        t = self._float_keys_timer

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
        now = time.monotonic()
        self.mode_ctrl.update_image(now)
        self.particles.spawn(self.image_source, self.mode_ctrl.is_ember)
        self.particles.update(dt, self.mode_ctrl.is_ember)
        self.sound.update(self.mode_ctrl.is_ember)

        renderer = self._renderers[VisualizationMode.POINTS]
        renderer.render(self.particles)

        if not self._draw_float_keys(dt):
            self._state = STATE_RUNNING

    def _render_particles(self, dt, hand_data=None, pose_data=None):
        """Dispatch rendering for the current visualization mode."""
        from hand_tracker import HandData
        from pose_tracker import PoseData

        if hand_data is None:
            hand_data = HandData()
        if pose_data is None:
            pose_data = PoseData()

        mode = self._viz_mode

        is_ember = self.mode_ctrl.is_ember

        if mode == VisualizationMode.POINTS:
            self._renderers[VisualizationMode.POINTS].render(self.particles, is_ember)

        elif mode == VisualizationMode.TRAILS:
            self._renderers[VisualizationMode.TRAILS].render(self.particles, is_ember)

        elif mode == VisualizationMode.SKELETON:
            # Particles underneath, skeleton on top
            self._renderers[VisualizationMode.POINTS].render(self.particles, is_ember)
            self._renderers[VisualizationMode.SKELETON].render(
                hand_data, pose_data, time.monotonic()
            )

        elif mode == VisualizationMode.FLOW:
            self._renderers[VisualizationMode.POINTS].render(self.particles, is_ember)

        elif mode == VisualizationMode.SHAPES:
            self._renderers[VisualizationMode.POINTS].render(self.particles, is_ember)
            self._renderers[VisualizationMode.SHAPES].update(
                self.particles, is_ember, dt
            )
            self._renderers[VisualizationMode.SHAPES].render()

    # ── Main draw loop ──────────────────────────────────────

    def on_draw(self):
        self.ctx.clear(0.0, 0.0, 0.0, 1.0)

        dt = 1.0 / 60.0

        if self._state == STATE_LOADING:
            self._draw_loading(dt)
            return

        if self._state == STATE_INTRO:
            self._draw_intro(dt)
            return

        # --- STATE_RUNNING ---
        now = time.monotonic()

        # Source transition: ramp spawn rate from 50 → 150 over 1.5s
        if self._source_transition > 0.0:
            self._source_transition = max(0.0, self._source_transition - dt)
            t = 1.0 - self._source_transition / 1.5
            _particles_mod.SPAWN_PER_FRAME = int(50 + 100 * t)
        else:
            _particles_mod.SPAWN_PER_FRAME = SPAWN_PER_FRAME

        hand_data = None
        pose_data = None

        if self.use_camera and self.camera:
            if not self.camera.ready:
                self.mode_ctrl.update_image(now)
                self.particles.spawn(self.image_source, self.mode_ctrl.is_ember)
            else:
                brightness, motion, avg_motion = self.camera.get_data()

                hand_data = self.camera.get_hand_data() if self._hand_enabled else None
                open_palm = hand_data.is_open_palm if (hand_data and hand_data.detected) else False

                self.mode_ctrl.update_camera(avg_motion, now, open_palm)
                self.particles.spawn_camera(brightness, motion, self.mode_ctrl.is_ember)

                if hand_data and hand_data.detected and hand_data.is_open_palm:
                    self.particles.kindle_nearby(hand_data.palm_ndc_x, hand_data.palm_ndc_y)
                    self.particles.spawn_palm_sparks(hand_data.palm_ndc_x, hand_data.palm_ndc_y)
                    if not self._prev_palm_open:
                        self.sound.play(AUDIO_BOSS_OUT, volume=0.35)
                    self._prev_palm_open = True
                else:
                    self._prev_palm_open = False

                if self._pose_enabled:
                    pose_data = self.camera.get_pose_data()
                    self._pose_cooldown = max(0.0, self._pose_cooldown - dt)
                    pose_name = pose_data.pose if pose_data.detected else None
                    if pose_name and pose_name != self._prev_pose and self._pose_cooldown <= 0:
                        self._trigger_pose_effect(pose_name, pose_data)
                        self._pose_cooldown = 5.0
                    self._prev_pose = pose_name
                else:
                    pose_data = None
        else:
            self.mode_ctrl.update_image(now)
            self.particles.spawn(self.image_source, self.mode_ctrl.is_ember)

        # Determine flow field for this frame
        flow = None
        if self._viz_mode == VisualizationMode.FLOW:
            flow = self._renderers[VisualizationMode.FLOW].update_field(dt)

        self.particles.update(dt, self.mode_ctrl.is_ember, flow_field=flow)
        self.sound.update(self.mode_ctrl.is_ember)
        self._you_died_overlay.update(dt)
        self._gesture_overlay.update(dt)

        # Render particles / visualization
        self._render_particles(dt, hand_data, pose_data)

        # YOU DIED overlay
        self._you_died_overlay.draw()

        # Debug overlay + HUD
        if self.debug.enabled:
            if self.use_camera and self.camera:
                preview = self.camera.get_preview()
                hand_data_dbg = self.camera.get_hand_data() if self._hand_enabled else None
            else:
                preview = self.image_source.get_preview()
                hand_data_dbg = None
            self.debug.draw(preview)
            if hand_data_dbg is not None:
                self.debug.draw_hand(hand_data_dbg)

            mode_name = MODE_NAMES[self.mode_ctrl.mode]
            state = "EMBER" if self.mode_ctrl.is_ember else "Humanity"
            source = "Camera" if self.use_camera else f"Image: {self.image_source.image_name}"
            viz_name = VIZ_NAMES[int(self._viz_mode)]
            self._mode_label.text = f"Mode: {mode_name} | State: {state} | Viz: {viz_name}"
            self._mode_label.draw()
            self._particle_label.text = f"Particles: {self.particles.count}"
            self._particle_label.draw()
            self._source_label.text = f"Source: {source} [{self.image_source.image_count} images]"
            self._source_label.draw()

            if self.use_camera and self.camera and self._hand_enabled:
                hand_data_panel = self.camera.get_hand_data()
                ema = self.camera.get_hand_ema()
                self.debug.draw_hand_panel(hand_data_panel, ema)
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

        # Gesture corner image
        self._gesture_overlay.draw()

        # Floating key help
        self._draw_float_keys(dt)

        # Wall clock
        now_str = datetime.now().strftime("%H:%M:%S")
        self._clock_label.text = f"{now_str}  {self._tz_name}"
        self._clock_label.draw()

        # GUI menu — always last
        self.menu.draw()

        # Timelapse: save after all drawing is done
        if self._timelapse_enabled:
            self._timelapse_timer += dt
            if self._timelapse_timer >= 1.0:
                self._timelapse_timer -= 1.0
                self._timelapse_save()

    def on_close(self):
        self.sound.cleanup()
        super().on_close()
