from dataclasses import dataclass

import pyglet


@dataclass
class BonfirePalmState:
    detected: bool = False
    is_open: bool = False
    palm_ndc_x: float = 0.0
    palm_ndc_y: float = 0.0
    charge: float = 0.0
    intensity: float = 0.0
    kindle_radius: float = 0.0
    sun_sync: float = 0.0
    phase: str = "Camera Off"
    just_ignited: bool = False


class BonfirePalmController:
    """Product-level interaction state for the open-palm bonfire gesture."""

    MIN_RADIUS = 0.18
    MAX_RADIUS = 0.42

    def __init__(self):
        self.enabled = True
        self.state = BonfirePalmState()
        self._was_open = False

    def reset(self):
        self.state = BonfirePalmState()
        self._was_open = False

    def update(self, hand_data, dt, particles=None) -> BonfirePalmState:
        state = self.state
        state.just_ignited = False

        if not self.enabled:
            state.detected = False
            state.is_open = False
            state.phase = "Disabled"
            state.charge = max(0.0, state.charge - dt * 1.4)
            self._refresh_derived_state(state)
            self._was_open = False
            return state

        detected = bool(hand_data and getattr(hand_data, "detected", False))
        is_open = bool(detected and getattr(hand_data, "is_open_palm", False))

        state.detected = detected
        state.is_open = is_open
        if detected:
            state.palm_ndc_x = float(getattr(hand_data, "palm_ndc_x", 0.0))
            state.palm_ndc_y = float(getattr(hand_data, "palm_ndc_y", 0.0))

        if is_open:
            state.charge = min(1.0, state.charge + dt * 1.35)
            state.phase = "Kindling" if state.charge < 0.78 else "Sun Warrior Map"
        elif detected:
            state.charge = max(0.0, state.charge - dt * 0.75)
            state.phase = "Open Palm Needed"
        else:
            state.charge = max(0.0, state.charge - dt * 1.0)
            state.phase = "Seeking Hand"

        state.just_ignited = is_open and not self._was_open
        self._refresh_derived_state(state)

        if particles is not None and is_open:
            spark_count = int(10 + 34 * state.intensity)
            spark_spread = 0.018 + 0.026 * state.intensity
            particles.kindle_nearby(
                state.palm_ndc_x,
                state.palm_ndc_y,
                radius=state.kindle_radius,
            )
            particles.spawn_palm_sparks(
                state.palm_ndc_x,
                state.palm_ndc_y,
                count=spark_count,
                spread=spark_spread,
                intensity=0.65 + state.intensity * 0.9,
            )

        self._was_open = is_open
        return state

    def _refresh_derived_state(self, state):
        state.intensity = state.charge if state.is_open else state.charge * 0.35
        state.kindle_radius = (
            self.MIN_RADIUS + (self.MAX_RADIUS - self.MIN_RADIUS) * state.charge
        )
        state.sun_sync = max(0.0, min(1.0, (state.charge - 0.35) / 0.65))


class BonfirePalmOverlay:
    """Persistent product HUD for open-palm kindling."""

    PANEL_W = 300
    PANEL_H = 178
    PAD = 18

    def __init__(self, win_w, win_h):
        self._win_w = win_w
        self._win_h = win_h

        self._panel = pyglet.shapes.BorderedRectangle(
            0, 0, self.PANEL_W, self.PANEL_H, border=1,
            color=(12, 10, 8), border_color=(200, 168, 78),
        )
        self._panel.opacity = 185

        self._title = pyglet.text.Label(
            "Bonfire Palm", font_name="Georgia", font_size=16,
            x=0, y=0, anchor_x="left", anchor_y="top",
            color=(255, 215, 120, 240),
        )
        self._status = pyglet.text.Label(
            "", font_name="Consolas", font_size=12,
            x=0, y=0, anchor_x="left", anchor_y="top",
            color=(220, 210, 190, 220),
        )
        self._charge_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=11,
            x=0, y=0, anchor_x="left", anchor_y="top",
            color=(180, 170, 150, 210),
        )
        self._phase_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=11,
            x=0, y=0, anchor_x="left", anchor_y="top",
            color=(200, 168, 78, 220),
        )
        self._coords_label = pyglet.text.Label(
            "", font_name="Consolas", font_size=10,
            x=0, y=0, anchor_x="left", anchor_y="bottom",
            color=(150, 170, 175, 190),
        )

        self._bar_bg = pyglet.shapes.Rectangle(0, 0, 1, 8, color=(55, 44, 30))
        self._bar_bg.opacity = 190
        self._bar_fill = pyglet.shapes.Rectangle(0, 0, 1, 8, color=(255, 140, 0))
        self._bar_fill.opacity = 230

        self._step_labels = [
            pyglet.text.Label("", font_name="Consolas", font_size=10,
                              x=0, y=0, anchor_x="center", anchor_y="center"),
            pyglet.text.Label("", font_name="Consolas", font_size=10,
                              x=0, y=0, anchor_x="center", anchor_y="center"),
            pyglet.text.Label("", font_name="Consolas", font_size=10,
                              x=0, y=0, anchor_x="center", anchor_y="center"),
        ]
        self._step_dots = [
            pyglet.shapes.Circle(0, 0, 5, color=(80, 70, 55)),
            pyglet.shapes.Circle(0, 0, 5, color=(80, 70, 55)),
            pyglet.shapes.Circle(0, 0, 5, color=(80, 70, 55)),
        ]
        self._step_lines = [
            pyglet.shapes.Line(0, 0, 1, 1, thickness=1, color=(120, 90, 40)),
            pyglet.shapes.Line(0, 0, 1, 1, thickness=1, color=(120, 90, 40)),
        ]

        self._reticle_outer = pyglet.shapes.Arc(
            0, 0, 30, segments=64, thickness=2, color=(255, 180, 70)
        )
        self._reticle_inner = pyglet.shapes.Arc(
            0, 0, 10, segments=32, thickness=2, color=(80, 220, 255)
        )
        self._reticle_core = pyglet.shapes.Circle(0, 0, 4, color=(255, 215, 0))
        self._reticle_h = pyglet.shapes.Line(0, 0, 1, 1, thickness=1, color=(255, 180, 70))
        self._reticle_v = pyglet.shapes.Line(0, 0, 1, 1, thickness=1, color=(255, 180, 70))

        self.resize(win_w, win_h)

    def resize(self, win_w, win_h):
        self._win_w = win_w
        self._win_h = win_h

        x = win_w - self.PANEL_W - self.PAD
        y = win_h - self.PANEL_H - self.PAD
        self._panel.x = x
        self._panel.y = y

        self._title.x = x + 14
        self._title.y = y + self.PANEL_H - 14
        self._status.x = x + 14
        self._status.y = y + self.PANEL_H - 44
        self._charge_label.x = x + 14
        self._charge_label.y = y + self.PANEL_H - 70
        self._phase_label.x = x + 14
        self._phase_label.y = y + 58
        self._coords_label.x = x + 14
        self._coords_label.y = y + 12

        bar_x = x + 14
        bar_y = y + self.PANEL_H - 96
        self._bar_bg.x = bar_x
        self._bar_bg.y = bar_y
        self._bar_bg.width = self.PANEL_W - 28
        self._bar_fill.x = bar_x
        self._bar_fill.y = bar_y

        step_y = y + 34
        step_xs = [x + 58, x + self.PANEL_W // 2, x + self.PANEL_W - 58]
        for dot, label, sx in zip(self._step_dots, self._step_labels, step_xs):
            dot.x = sx
            dot.y = step_y + 16
            label.x = sx
            label.y = step_y

        for i, line in enumerate(self._step_lines):
            line.x = step_xs[i] + 8
            line.y = step_y + 16
            line.x2 = step_xs[i + 1] - 8
            line.y2 = step_y + 16

    def draw(self, state, visible=True):
        if not visible:
            return

        self._draw_reticle(state)
        self._draw_panel(state)

    def _draw_reticle(self, state):
        if not state.detected:
            return

        sx = int((state.palm_ndc_x + 1.0) * 0.5 * self._win_w)
        sy = int((state.palm_ndc_y + 1.0) * 0.5 * self._win_h)
        radius = 24 + int(30 * state.charge)

        color = (255, 180, 70) if state.is_open else (80, 220, 255)
        alpha = int(90 + 150 * max(state.intensity, 0.25))

        for shape in (self._reticle_outer, self._reticle_inner,
                      self._reticle_core, self._reticle_h, self._reticle_v):
            shape.opacity = alpha
            shape.color = color

        self._reticle_outer.x = sx
        self._reticle_outer.y = sy
        self._reticle_outer.radius = radius
        self._reticle_inner.x = sx
        self._reticle_inner.y = sy
        self._reticle_inner.radius = max(8, radius * 0.35)
        self._reticle_core.x = sx
        self._reticle_core.y = sy
        self._reticle_h.x = sx - radius - 8
        self._reticle_h.y = sy
        self._reticle_h.x2 = sx + radius + 8
        self._reticle_h.y2 = sy
        self._reticle_v.x = sx
        self._reticle_v.y = sy - radius - 8
        self._reticle_v.x2 = sx
        self._reticle_v.y2 = sy + radius + 8

        self._reticle_h.draw()
        self._reticle_v.draw()
        self._reticle_outer.draw()
        self._reticle_inner.draw()
        self._reticle_core.draw()

    def _draw_panel(self, state):
        self._panel.draw()
        self._title.draw()

        status = "OPEN PALM" if state.is_open else ("HAND FOUND" if state.detected else "SEEKING HAND")
        self._status.text = f"Status: {status}"
        self._status.color = (255, 190, 80, 235) if state.is_open else (160, 210, 220, 210)
        self._status.draw()

        self._charge_label.text = f"Kindle Radius: {state.kindle_radius:.2f}   Charge: {int(state.charge * 100):02d}%"
        self._charge_label.draw()

        self._bar_fill.width = max(1, int((self.PANEL_W - 28) * state.charge))
        self._bar_fill.color = (255, 140 + int(70 * state.sun_sync), 0)
        self._bar_bg.draw()
        self._bar_fill.draw()

        self._phase_label.text = f"Palm -> Sun: {state.phase}"
        self._phase_label.draw()
        self._coords_label.text = f"Palm NDC ({state.palm_ndc_x:+.2f}, {state.palm_ndc_y:+.2f})"
        self._coords_label.draw()

        steps = ("Open Palm", "Kindle", "Sun Map")
        active = (state.detected, state.charge > 0.18, state.sun_sync > 0.55)
        for label, text, is_active in zip(self._step_labels, steps, active):
            label.text = text
            label.color = (255, 215, 120, 230) if is_active else (130, 120, 100, 180)
            label.draw()

        for dot, is_active in zip(self._step_dots, active):
            dot.color = (255, 180, 70) if is_active else (80, 70, 55)
            dot.opacity = 235 if is_active else 160
            dot.draw()

        for line in self._step_lines:
            line.opacity = 170
            line.draw()
