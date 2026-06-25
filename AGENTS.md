# SoulStream - Project Notes

## Pyglet Pitfalls

- `pyglet.text.Label()` does NOT accept `bold=True`. To use bold, set `font_name` to a bold variant such as `"Consolas Bold"`, or use `pyglet.text.HTMLLabel`.
- Do NOT create `pyglet.sprite.Sprite`, `pyglet.shapes.Rectangle`, `pyglet.shapes.Circle`, `pyglet.shapes.Line`, `pyglet.text.Label`, or other pyglet draw objects inside `on_draw()` or per-frame methods. Create them once in `__init__` and only update their properties per frame.
- Per-frame code should update state, write buffers, and draw existing objects. Object allocation in draw paths has caused `DocumentLabel.__del__` / `'Label' object has no attribute '_boxes'` shutdown errors.

## Project Structure

All Python source lives under `src/`. `main.py` is a thin launcher that adds `src/` to `sys.path` and calls `SoulStreamApp().run()`.

```text
main.py                  -> launcher only
test_pose.py             -> standalone webcam pose debug tool
test_hand_tracker.py     -> standalone webcam hand debug tool
src/
  app.py                 -> SoulStreamApp, window, state machine,
                            high-level runtime pipeline
  bonfire_palm.py        -> BonfirePalmState, BonfirePalmController,
                            BonfirePalmOverlay product interaction
  overlays.py            -> SoundManager, ModeController, DebugOverlay,
                            SoulOverlay, GestureCornerOverlay, YouDiedOverlay
  visualization.py       -> VisualizationMode enum + 5 renderer classes
  gui.py                 -> GameMenu panels and controls
  particles.py           -> ParticleSystem, prev positions, flow_field,
                            palm sparks and pose burst effects
  camera.py              -> Camera, enable_hand / enable_pose feature flags
  hand_tracker.py        -> HandTracker, HandData
  pose_tracker.py        -> PoseTracker, PoseData
  image_source.py        -> ImageSource
tools/
  simulate_bonfire_palm.py -> pure-Python Bonfire Palm state simulation
shaders/
  particle.vert / .frag  -> point cloud
  skeleton.vert / .frag  -> SkeletonRenderer
  shapes.vert / .frag    -> ShapesRenderer
```

## Runtime Architecture

- Keep `SoulStreamApp.on_draw()` as a readable pipeline. It should call helper stages rather than contain raw source, hand, pose, particle, overlay, and menu logic inline.
- Source/input stages belong in helpers such as `_update_source_transition()`, `_update_running_sources()`, `_spawn_from_image_source()`, `_spawn_from_camera_source()`, and `_update_pose_interaction()`.
- Product interactions should get their own module when they have state plus UI. Bonfire Palm lives in `src/bonfire_palm.py`; do not move its controller or HUD back into `app.py` or `overlays.py`.
- `overlays.py` should remain for shared overlays and managers. Avoid adding product-specific state machines there.
- `particles.py` owns particle data and effects only. It should not know about UI panels, menu state, or product phases.

## Bonfire Palm

Bonfire Palm is product interaction #2 from the concept work:

- Human input: webcam hand tracking, open palm.
- Mapping: palm reticle -> kindle radius -> Palm -> Sun / Sun Warrior Map.
- State: `BonfirePalmController` updates `BonfirePalmState`.
- UI: `BonfirePalmOverlay` draws the reticle, charge, kindle radius, and three-step mapping HUD.
- Particle effects: controller calls `particles.kindle_nearby(...)` and `particles.spawn_palm_sparks(...)` when the palm is open.

When changing Bonfire Palm, run the simulation and check the output:

```powershell
python tools\simulate_bonfire_palm.py
```

Expected behavior:

- `none` and `closed` inputs do not spawn sparks.
- The first `open` frame has `ignited=True`; later open frames do not.
- During sustained open palm, charge rises toward `1.00`, radius rises toward `0.42`, and phase becomes `Sun Warrior Map`.
- After closed/no hand input, charge and sun sync decay smoothly.

## Simulation and Verification

For each staged behavior change:

1. Run the relevant simulation before/after the change.
2. Inspect the printed values for plausible state transitions, not just pass/fail.
3. Run targeted tests.
4. Run full tests before final handoff.

Useful commands:

```powershell
python tools\simulate_bonfire_palm.py
pytest -q tests\test_interactions.py tests\test_particles.py
pytest -q
python -c "from pathlib import Path; files=['src/app.py','src/gui.py','src/overlays.py','src/particles.py','src/bonfire_palm.py','tools/simulate_bonfire_palm.py']; [compile(Path(f).read_text(encoding='utf-8'), f, 'exec') for f in files]; print('syntax ok')"
```

On Windows, direct `py_compile` can occasionally fail if a `.pyc` file in `__pycache__` is locked. Use the in-memory `compile(...)` syntax check above when that happens.

## MediaPipe Model Files

`hand_landmarker.task` and `pose_landmarker_lite.task` are binary model files that must be downloaded separately. They are not in git. Place them in the project root.

```powershell
curl -L -o hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
curl -L -o pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task
```

## Camera Feature Flags

`Camera` accepts `enable_hand` and `enable_pose` boolean kwargs. Both default to `True`. Each tracker is only instantiated and run when its flag is enabled, saving CPU. Runtime toggling is handled by property setters: `camera.hand_enabled` and `camera.pose_enabled`.

## Visualization Modes

Five modes in `VisualizationMode` (`IntEnum` values `0` through `4`):

- `POINTS`
- `TRAILS`
- `SKELETON`
- `FLOW`
- `SHAPES`

Active mode is stored in `SoulStreamApp._viz_mode`. Each mode has a renderer in `src/visualization.py` with a `setup(ctx)` / `render(...)` interface. `FlowRenderer` owns the curl-noise field; it does not have its own draw call. It modifies particle velocities via `particles.update(flow_field=...)`.

## Key Bindings

- `H` = hand tracker toggle
- `P` = pose detector toggle
- `B` = Bonfire Palm toggle
- `F1` = floating help
- `V` = cycle visualization mode
- `1` through `5` = jump to visualization mode
- `C` = toggle camera
- `TAB` = menu
- `SPACE` = cycle mode
