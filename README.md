# webcam-01 — Webcam Overlay Effect

**Version 1.1.0**

> **Doc staleness note (2026-08-09):** this README's Description/Hotkeys
> sections below describe an older `WebcamOverlay` playlist-effect
> architecture and reference `unicornviz/camera_overlay.py`, neither of
> which exist in the current code. The actual current implementation is
> `WebcamSystem` in `webcam_overlay.py` — the always-on system-level PiP
> overlay described in that file's own module docstring. Pre-existing
> staleness, not introduced by this change; flagging rather than doing an
> unscoped rewrite. The two sections below ("Auto-cycle timing" and
> "Selfie-segmentation background dim") describe the current code
> accurately.


A drop-in effect for [unicorn-viz](https://github.com/iDoMeteor/unicorn-viz).

## Description

A standalone playlist effect that places the live camera feed over an animated, audio-reactive background. Selected from the playlist like any other effect.

> **Note:** This is separate from the *system-level camera overlay* (`unicornviz/camera_overlay.py`) which renders a PiP over every effect regardless of which is active. This drop-in is a full-screen effect with its own animated background.

## Visual Features

- Animated neon-grid background driven by bass and treble
- Camera feed rendered as a PiP or fullscreen overlay
- Cyan border that pulses with bass
- Keypad-controlled PiP position (9 presets + fullscreen + hide)

## Audio Reactivity

| Signal  | Effect |
|---------|--------|
| `bass`  | Background wave intensity, border glow |
| `treble`| Background colour temperature |

## Requirements

```
opencv-python-headless >= 4.9
```

Already listed in the main `requirements.txt`.

## Installer

This drop-in ships a bundle installer script at `install.sh`.

```bash
bash install.sh --root ~/.local/share/unicorn-viz
```

The installer copies the bundle into `~/.local/share/unicorn-viz/drop-ins/webcam-01`
by default. Use `--uninstall` to remove that bundle again.

## Config

```toml
[effects.WebcamOverlay]
camera_device   = 0              # /dev/video index (Linux) or DirectShow (Windows)
camera_width    = 1280
camera_height   = 720
camera_fps      = 30
pip_position    = "bottom_right" # top_left | top_center | top_right |
                                 # left | center | right |
                                 # bottom_left | bottom_center | bottom_right |
                                 # fullscreen
pip_scale       = 0.33           # fraction of viewport width (0.12–0.80)
border_thickness = 8.0            # border/backplate thickness in pixels
border_color     = [0.04, 0.72, 1.0, 0.28]  # RGBA values 0.0–1.0
enabled         = true           # false disables camera worker
```

## Hotkeys

While this effect is active, the standard keypad layout controls apply:

| Key     | Action |
|---------|--------|
| `KP 1–9`| PiP position |
| `KP 0`  | Fullscreen camera |
| `KP .`  | Hide camera |

## Auto-cycle timing (current, accurate)

`Ctrl+KP Enter` toggles auto-cycle. It alternates the plain camera feed
(`raw`) with the next treatment, rather than round-robining flatly through
every treatment including raw:

```text
normal -> effect1 -> normal -> effect2 -> normal -> effect3 -> ...
```

An effect airs for half of `[webcam] cycle_interval`; the normal interlude
between effects airs for the full (un-halved) `cycle_interval`, so effects
read as brief highlights rather than eating equal screen time to plain
video. Manual treatment switching (`Ctrl+KP /`, `Ctrl+KP *`) is unaffected
— it still cycles flatly through every treatment, raw included.

## Selfie-segmentation background dim (current, accurate)

Optional, per-camera, off by default. Runs MediaPipe's selfie-segmentation
model on the capture thread (never the render thread — inference costs
~8-15ms, too much of the 16.67ms frame budget to spend there) and dims
every background pixel to 25% of its original brightness, leaving the
detected person untouched. Applied *before* every treatment shader, so
raw/neon/glitch/vintage/thermal all receive an already-dimmed background
for free.

Requires `mediapipe` (see `requirements.txt` — needs `--no-deps` to avoid
a conflict with this project's pinned `opencv-python-headless`) and the
bundled model at `assets/models/selfie_segmenter.tflite`. Absent either,
the toggle silently does nothing rather than erroring.

Toggle per-camera from the Config Editor's **Visuals** tab — one 0/1
slider per detected camera (the generic Config Editor has no native
checkbox widget, so booleans ride a min=0/max=1 slider by the same
convention used elsewhere in the app). Not configurable from
`config.toml` directly, since it's inherently per-camera rather than a
single scalar.

## Planning

- docs/planning.md

## Tags

`webcam` · `overlay` · `camera` · `drop-in`

## Changelog

- **1.1.0** — Two additions:
  1. Auto-cycle (`Ctrl+KP Enter`) now alternates normal (raw) and the next
     treatment instead of round-robining flatly through every treatment
     including raw — effects air for half of `cycle_interval`, raw airs
     for the full `cycle_interval` in between. Fixed a same-effect-twice
     bug caught by the new regression tests: the first implementation
     derived "next effect" from `_treatment_index`, which reads `'raw'`
     throughout the whole normal phase and so could never tell which
     effect played last; it now tracks its own sequence index.
  2. Optional per-camera selfie-segmentation background dim (MediaPipe),
     applied before every existing treatment shader. Off by default;
     toggle per-camera from the Config Editor's Visuals tab. Runs on the
     capture thread, never the render thread. Requires `mediapipe`
     installed with `--no-deps` (see `requirements.txt`) to avoid a
     conflict with this project's pinned `opencv-python-headless` — both
     packages own the same `cv2` import path.
- **1.0.0-rc.4** — **Stops re-uploading a frame the camera never changed.**
  The render path copied the frame, ran `tobytes()` and pushed it to the GPU
  on *every rendered frame* — three full passes over 1280x720x3 at up to
  60 fps to show a picture the 30 fps camera had not touched, roughly
  490 MB/s of pointless memory traffic. Alone it fit the budget; alongside a
  4K mirror composite it did not, which is exactly the "webcam + mirror
  together" fault the owner isolated (measured: `subsys_upd` was the single
  largest frame stage at 21 ms median, more than `draw`). The capture worker
  now stamps a sequence number and the whole path — copy, flip,
  brightness/contrast float32 conversion, upload — short-circuits when
  nothing new has arrived. 3 regression tests.
- **1.0.0-rc.3** — **Fixes a total-failure regression introduced in rc.2.** The privacy change routed `__init__` through the side-effecting `set_layout()`, which now calls `_ensure_worker()` — reaching `_worker`/`_available_cameras` before either existed. The `AttributeError` escaped the constructor, core's guarded loader swallowed it, and the entire webcam drop-in was silently disabled for the session. `__init__` now resolves the configured position through a pure `_resolve_layout_token()` with no side effects; `start()` remains the only thing that opens a camera. Two new tests actually **construct** the object across every position value — every previous test built it with `__new__` and hand-set attributes, which is precisely why a construction-order bug could ship unnoticed.
- **1.0.0-rc.2** — privacy: the camera never opens (or probes) while the overlay is hidden — with the shipped `pip_position = "hidden"` default the webcam LED previously lit at first boot capturing 30 fps nobody rendered. Capture now starts lazily on first show, a deliberate hide releases the device, and the transient camera-switch shim still keeps it alive. 5 regression tests.
- **1.0.0-rc.1** — initial versioned release.
