# SoulStream - Project Notes

## Pyglet Pitfalls

- `pyglet.text.Label()` does NOT accept `bold=True`. To use bold, set the `font_name` to a bold variant (e.g., `"Consolas Bold"`) or use `pyglet.text.HTMLLabel`.
- Do NOT create `pyglet.sprite.Sprite` or `pyglet.shapes.Rectangle` objects inside `on_draw()` or per-frame methods. Creating and discarding them every frame causes `__del__` errors (`DocumentLabel.__del__` / `'Label' object has no attribute '_boxes'`). Always create them once in `__init__` and reuse.
