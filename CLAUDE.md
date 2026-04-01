# SoulStream - Project Notes

## Pyglet Pitfalls

- `pyglet.text.Label()` does NOT accept `bold=True`. To use bold, set the `font_name` to a bold variant (e.g., `"Consolas Bold"`) or use `pyglet.text.HTMLLabel`.
- Do NOT create `pyglet.sprite.Sprite` or `pyglet.shapes.Rectangle` objects inside `on_draw()` or per-frame methods. Creating and discarding them every frame causes `__del__` errors (`DocumentLabel.__del__` / `'Label' object has no attribute '_boxes'`). Always create them once in `__init__` and reuse.

## Project Structure

All Python source lives under `src/`. `main.py` is a thin launcher that adds `src/` to `sys.path` and calls `SoulStreamApp().run()`.

```
main.py                  ← launcher only
test_pose.py             ← standalone webcam pose debug tool
test_hand_tracker.py     ← standalone webcam hand debug tool
src/
  app.py                 ← SoulStreamApp (window, state machine, on_draw)
  overlays.py            ← SoundManager, ModeController, DebugOverlay,
                            SoulOverlay, GestureCornerOverlay, YouDiedOverlay
  visualization.py       ← VisualizationMode enum + 5 renderer classes
  gui.py                 ← GameMenu (Source, Camera Features, Mode,
                            Visualization, Audio, Tools, Quit panels)
  particles.py           ← ParticleSystem (prev positions, flow_field support)
  camera.py              ← Camera (enable_hand / enable_pose feature flags)
  hand_tracker.py        ← HandTracker, HandData
  pose_tracker.py        ← PoseTracker, PoseData
  image_source.py        ← ImageSource
shaders/
  particle.vert / .frag  ← point cloud (original)
  skeleton.vert / .frag  ← SkeletonRenderer
  shapes.vert / .frag    ← ShapesRenderer
```

## Camera Feature Flags

`Camera` accepts `enable_hand` and `enable_pose` boolean kwargs. Both default to `True`. Each tracker is only instantiated and run when its flag is enabled, saving CPU. Runtime toggling via property setters (`camera.hand_enabled`, `camera.pose_enabled`) with lazy init.

## Visualization Modes

Five modes in `VisualizationMode` (IntEnum 0–4): `POINTS`, `TRAILS`, `SKELETON`, `FLOW`, `SHAPES`. Active mode is stored in `SoulStreamApp._viz_mode`. Each mode has a renderer in `src/visualization.py` with a `setup(ctx)` / `render(...)` interface. `FlowRenderer` owns the curl-noise field; it does NOT have its own draw call — it modifies particle velocities via `particles.update(flow_field=...)`.

## Key Bindings (v2)

`H` = hand tracker toggle (was "help" in v1). `F1` = floating help. `P` = pose detector toggle. `V` = cycle viz mode. `1–5` = jump to viz mode.
