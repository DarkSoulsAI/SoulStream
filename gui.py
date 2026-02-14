"""
Dark Souls themed in-app GUI overlay.

Pure pyglet shapes + labels — no external GUI library.
Toggle with TAB. Provides clickable buttons, a volume slider,
and tooltip descriptions for every control.
"""

import pyglet

# ── Theme colors ──────────────────────────────────────────────
COL_PANEL_BG    = (20, 15, 10, 200)
COL_BORDER      = (200, 168, 78, 255)      # gold #C8A84E
COL_BORDER_HOT  = (255, 215, 0, 255)       # bright gold #FFD700
COL_TEXT         = (232, 224, 208, 255)     # warm white #E8E0D0
COL_ACTIVE       = (255, 140, 0, 255)      # ember orange #FF8C00
COL_BTN_BG       = (35, 28, 20, 220)
COL_BTN_BG_HOT   = (50, 40, 30, 230)
COL_BTN_BG_PRESS = (70, 55, 35, 240)
COL_TOOLTIP_BG   = (30, 25, 18, 220)


# ── helpers ───────────────────────────────────────────────────

def _rgba(color):
    """Ensure an RGBA tuple (int 0-255)."""
    if len(color) == 3:
        return (*color, 255)
    return color


def _hit(x, y, bx, by, bw, bh):
    """Point-in-rect test."""
    return bx <= x <= bx + bw and by <= y <= by + bh


# ═════════════════════════════════════════════════════════════
#  GuiButton
# ═════════════════════════════════════════════════════════════

class GuiButton:
    """Clickable rectangle with hover/press states and tooltip."""

    def __init__(self, x, y, w, h, text, tooltip="", callback=None,
                 toggle=False, group_value=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.text = text
        self.tooltip = tooltip
        self.callback = callback
        self.hovered = False
        self.pressed = False

        # Toggle / radio support
        self.toggle = toggle            # if True, stays highlighted when active
        self.active = False             # current toggle state
        self.group_value = group_value  # used for radio groups

        # Shapes (batch-drawn)
        self._bg = pyglet.shapes.Rectangle(x, y, w, h,
                                           color=COL_BTN_BG[:3])
        self._bg.opacity = COL_BTN_BG[3]
        self._border = pyglet.shapes.BorderedRectangle(
            x, y, w, h, border=2,
            color=COL_BTN_BG[:3],
            border_color=COL_BORDER[:3],
        )
        self._border.opacity = COL_BTN_BG[3]

        self._label = pyglet.text.Label(
            text, font_name="Consolas", font_size=12,
            x=x + w // 2, y=y + h // 2,
            anchor_x="center", anchor_y="center",
            color=COL_TEXT,
        )

    def hit_test(self, mx, my):
        return _hit(mx, my, self.x, self.y, self.w, self.h)

    def on_hover(self, inside):
        self.hovered = inside

    def on_press(self):
        self.pressed = True
        if self.callback:
            self.callback()

    def on_release(self):
        self.pressed = False

    def draw(self):
        # Decide colors
        if self.pressed:
            bg = COL_BTN_BG_PRESS
        elif self.hovered:
            bg = COL_BTN_BG_HOT
        else:
            bg = COL_BTN_BG

        border_col = COL_BORDER_HOT[:3] if self.hovered else COL_BORDER[:3]

        if self.active or (self.toggle and self.active):
            border_col = COL_ACTIVE[:3]
            self._label.color = COL_ACTIVE
        else:
            self._label.color = COL_TEXT

        self._border.color = bg[:3]
        self._border.opacity = bg[3]
        self._border.border_color = border_col
        self._border.draw()
        self._label.draw()


# ═════════════════════════════════════════════════════════════
#  GuiSlider
# ═════════════════════════════════════════════════════════════

class GuiSlider:
    """Horizontal drag slider (0.0 – 1.0) with label."""

    KNOB_W = 14
    TRACK_H = 6

    def __init__(self, x, y, w, label_text, value=0.5,
                 tooltip="", on_change=None):
        self.x, self.y, self.w = x, y, w
        self.h = 28                        # total hit-area height
        self.value = value
        self.tooltip = tooltip
        self.on_change = on_change
        self.dragging = False
        self.hovered = False

        self._track = pyglet.shapes.Rectangle(
            x, y + self.h // 2 - self.TRACK_H // 2,
            w, self.TRACK_H,
            color=COL_BORDER[:3],
        )
        self._track.opacity = 120

        self._knob = pyglet.shapes.Rectangle(
            0, y + self.h // 2 - 10,
            self.KNOB_W, 20,
            color=COL_BORDER[:3],
        )
        self._knob.opacity = 255

        self._title = pyglet.text.Label(
            label_text, font_name="Consolas", font_size=11,
            x=x, y=y + self.h + 4,
            anchor_x="left", anchor_y="bottom",
            color=COL_TEXT,
        )
        self._val_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=11,
            x=x + w, y=y + self.h + 4,
            anchor_x="right", anchor_y="bottom",
            color=COL_TEXT,
        )

    def _knob_x(self):
        return self.x + int(self.value * (self.w - self.KNOB_W))

    def hit_test(self, mx, my):
        return _hit(mx, my, self.x, self.y, self.w, self.h)

    def on_hover(self, inside):
        self.hovered = inside

    def begin_drag(self, mx):
        self.dragging = True
        self._move(mx)

    def drag(self, mx):
        if self.dragging:
            self._move(mx)

    def end_drag(self):
        self.dragging = False

    def _move(self, mx):
        rel = (mx - self.x) / max(1, self.w)
        self.value = max(0.0, min(1.0, rel))
        if self.on_change:
            self.on_change(self.value)

    def draw(self):
        self._track.draw()
        kx = self._knob_x()
        self._knob.x = kx
        if self.hovered or self.dragging:
            self._knob.color = COL_BORDER_HOT[:3]
        else:
            self._knob.color = COL_BORDER[:3]
        self._knob.draw()
        self._title.draw()
        self._val_label.text = f"{int(self.value * 100)}%"
        self._val_label.draw()


# ═════════════════════════════════════════════════════════════
#  GuiPanel
# ═════════════════════════════════════════════════════════════

class GuiPanel:
    """Semi-transparent dark background container with a title."""

    def __init__(self, x, y, w, h, title=""):
        self.x, self.y, self.w, self.h = x, y, w, h
        self._bg = pyglet.shapes.Rectangle(x, y, w, h,
                                           color=COL_PANEL_BG[:3])
        self._bg.opacity = COL_PANEL_BG[3]
        self._border = pyglet.shapes.BorderedRectangle(
            x, y, w, h, border=1,
            color=COL_PANEL_BG[:3],
            border_color=COL_BORDER[:3],
        )
        self._border.opacity = COL_PANEL_BG[3]

        self._title = None
        if title:
            self._title = pyglet.text.Label(
                title, font_name="Georgia", font_size=12,
                x=x + 10, y=y + h - 8,
                anchor_x="left", anchor_y="top",
                color=COL_BORDER,
            )

    def draw(self):
        self._border.draw()
        if self._title:
            self._title.draw()


# ═════════════════════════════════════════════════════════════
#  GameMenu  —  main orchestrator
# ═════════════════════════════════════════════════════════════

class GameMenu:
    """
    Full-screen overlay menu toggled with TAB.

    Sections
    --------
    Source  : [Camera ON/OFF]  [< Prev]  [Next >]
    Mode    : [Auto]  [Humanity]  [Ember]          (radio)
    Audio   : Volume slider
    Tools   : [Debug]  [Help]
    Quit    : [Quit]
    """

    def __init__(self, width, height, callbacks):
        """
        Parameters
        ----------
        width, height : window dimensions
        callbacks : dict of str -> callable
            Required keys:
                toggle_camera, prev_image, next_image,
                set_mode_auto, set_mode_humanity, set_mode_ember,
                set_volume, toggle_debug, toggle_help, quit
        """
        self.width = width
        self.height = height
        self.visible = False
        self._callbacks = callbacks

        # Tooltip state
        self._tooltip_text = ""

        # Dim overlay behind menu
        self._dim = pyglet.shapes.Rectangle(0, 0, width, height,
                                            color=(0, 0, 0))
        self._dim.opacity = 140

        # ── Layout ────────────────────────────────────────────
        panel_w = 360
        px = (width - panel_w) // 2      # panel left x

        # Build from top to bottom — store y cursor
        top = height // 2 + 190
        row_h = 36
        gap = 14
        panel_pad = 32

        # Title
        self._title_label = pyglet.text.Label(
            "~ SOUL STREAM ~", font_name="Georgia", font_size=22,
            x=width // 2, y=top + 10,
            anchor_x="center", anchor_y="center",
            color=COL_BORDER,
        )

        y = top - 30  # start below title

        # ── Source panel ──────────────────────────────────────
        src_h = row_h + panel_pad
        self._source_panel = GuiPanel(px, y - src_h, panel_w, src_h, "Source")
        btn_y = y - src_h + 10
        bw = (panel_w - 40) // 3
        self._btn_camera = GuiButton(
            px + 10, btn_y, bw, row_h, "Camera",
            tooltip="Toggle webcam input (key: C)",
            callback=callbacks["toggle_camera"],
            toggle=True,
        )
        self._btn_prev = GuiButton(
            px + 15 + bw, btn_y, bw, row_h, "< Prev",
            tooltip="Previous image (key: Left Arrow)",
            callback=callbacks["prev_image"],
        )
        self._btn_next = GuiButton(
            px + 20 + bw * 2, btn_y, bw, row_h, "Next >",
            tooltip="Next image (key: Right Arrow)",
            callback=callbacks["next_image"],
        )
        y -= src_h + gap

        # ── Mode panel ────────────────────────────────────────
        mode_h = row_h + panel_pad
        self._mode_panel = GuiPanel(px, y - mode_h, panel_w, mode_h, "Mode")
        btn_y = y - mode_h + 10
        bw3 = (panel_w - 40) // 3
        self._btn_auto = GuiButton(
            px + 10, btn_y, bw3, row_h, "Auto",
            tooltip="Automatic mode cycling (key: Space)",
            callback=callbacks["set_mode_auto"],
            toggle=True, group_value=0,
        )
        self._btn_humanity = GuiButton(
            px + 15 + bw3, btn_y, bw3, row_h, "Humanity",
            tooltip="Force Humanity state — desaturated, slow drift",
            callback=callbacks["set_mode_humanity"],
            toggle=True, group_value=1,
        )
        self._btn_ember = GuiButton(
            px + 20 + bw3 * 2, btn_y, bw3, row_h, "Ember",
            tooltip="Force Ember state — warm gold, fast rise",
            callback=callbacks["set_mode_ember"],
            toggle=True, group_value=2,
        )
        self._mode_buttons = [self._btn_auto, self._btn_humanity, self._btn_ember]
        self._btn_auto.active = True  # default
        y -= mode_h + gap

        # ── Audio panel ───────────────────────────────────────
        audio_h = 60 + panel_pad
        self._audio_panel = GuiPanel(px, y - audio_h, panel_w, audio_h, "Audio")
        self._slider_vol = GuiSlider(
            px + 10, y - audio_h + 10, panel_w - 20,
            "Volume", value=0.25,
            tooltip="Adjust ambient background volume",
            on_change=callbacks["set_volume"],
        )
        y -= audio_h + gap

        # ── Tools panel ───────────────────────────────────────
        tools_h = row_h + panel_pad
        self._tools_panel = GuiPanel(px, y - tools_h, panel_w, tools_h, "Tools")
        btn_y = y - tools_h + 10
        bw2 = (panel_w - 30) // 2
        self._btn_debug = GuiButton(
            px + 10, btn_y, bw2, row_h, "Debug",
            tooltip="Toggle debug overlay with camera preview & stats (key: D)",
            callback=callbacks["toggle_debug"],
            toggle=True,
        )
        self._btn_help = GuiButton(
            px + 20 + bw2, btn_y, bw2, row_h, "Help",
            tooltip="Show help panel with controls & lore (key: H)",
            callback=callbacks["toggle_help"],
            toggle=True,
        )
        y -= tools_h + gap

        # ── Quit button ──────────────────────────────────────
        quit_h = row_h + 8
        self._btn_quit = GuiButton(
            px + panel_w // 4, y - quit_h, panel_w // 2, quit_h, "Quit",
            tooltip="Thank you, Dark Souls. (key: Esc)",
            callback=callbacks["quit"],
        )
        y -= quit_h + gap

        # ── Tooltip area ─────────────────────────────────────
        self._tooltip_bg = pyglet.shapes.Rectangle(
            px, y - 30, panel_w, 28,
            color=COL_TOOLTIP_BG[:3],
        )
        self._tooltip_bg.opacity = COL_TOOLTIP_BG[3]
        self._tooltip_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=11, italic=True,
            x=width // 2, y=y - 16,
            anchor_x="center", anchor_y="center",
            color=(180, 170, 150, 200),
        )

        # Collect all interactive elements for iteration
        self._buttons = [
            self._btn_camera, self._btn_prev, self._btn_next,
            self._btn_auto, self._btn_humanity, self._btn_ember,
            self._btn_debug, self._btn_help,
            self._btn_quit,
        ]
        self._panels = [
            self._source_panel, self._mode_panel,
            self._audio_panel, self._tools_panel,
        ]

    # ── public sync helpers ───────────────────────────────────

    def sync_state(self, *, use_camera=False, mode=0,
                   debug=False, help_visible=False, volume=0.25):
        """Sync button active states with app state."""
        self._btn_camera.active = use_camera
        for btn in self._mode_buttons:
            btn.active = (btn.group_value == mode)
        self._btn_debug.active = debug
        self._btn_help.active = help_visible
        self._slider_vol.value = volume

    # ── event handlers ────────────────────────────────────────

    def toggle(self):
        self.visible = not self.visible

    def on_mouse_motion(self, mx, my):
        if not self.visible:
            return
        self._tooltip_text = ""
        for btn in self._buttons:
            inside = btn.hit_test(mx, my)
            btn.on_hover(inside)
            if inside and btn.tooltip:
                self._tooltip_text = btn.tooltip
        self._slider_vol.on_hover(self._slider_vol.hit_test(mx, my))
        if self._slider_vol.hovered and self._slider_vol.tooltip:
            self._tooltip_text = self._slider_vol.tooltip

    def on_mouse_press(self, mx, my, button):
        """Returns True if the click was consumed by the menu."""
        if not self.visible:
            return False
        # Slider
        if self._slider_vol.hit_test(mx, my):
            self._slider_vol.begin_drag(mx)
            return True
        # Buttons
        for btn in self._buttons:
            if btn.hit_test(mx, my):
                btn.on_press()
                # Radio group logic for mode buttons
                if btn in self._mode_buttons:
                    for b in self._mode_buttons:
                        b.active = (b is btn)
                elif btn.toggle:
                    btn.active = not btn.active
                return True
        return True  # consume all clicks when menu is open

    def on_mouse_drag(self, mx, my):
        if not self.visible:
            return
        self._slider_vol.drag(mx)

    def on_mouse_release(self, mx, my):
        if not self.visible:
            return
        self._slider_vol.end_drag()
        for btn in self._buttons:
            btn.on_release()

    # ── drawing ───────────────────────────────────────────────

    def draw(self):
        if not self.visible:
            return

        # Dim background
        self._dim.draw()

        # Title
        self._title_label.draw()

        # Panels
        for panel in self._panels:
            panel.draw()

        # Buttons
        for btn in self._buttons:
            btn.draw()

        # Slider
        self._slider_vol.draw()

        # Tooltip
        if self._tooltip_text:
            self._tooltip_bg.draw()
            self._tooltip_label.text = self._tooltip_text
            self._tooltip_label.draw()
