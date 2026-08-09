# webcam-01 — Webcam Overlay

**Version 1.1.1**

A drop-in for [unicorn-viz](https://github.com/iDoMeteor/unicorn-viz).

## Description

An always-on, system-level picture-in-picture camera overlay, implemented
by `WebcamSystem` in `webcam_overlay.py`. It renders above whatever main
effect is currently active and below the HUD — it is **not** a playlist
effect and does not appear in the effect playlist. It starts automatically
whenever `[webcam] enabled = true` (the default) and is controlled at any
time via the keypad, regardless of which effect is showing.

The camera feed can be shown as a corner/edge/center picture-in-picture
(nine grid positions), fullscreen, or hidden entirely — hiding fully
releases the camera device (LED off) rather than just not drawing it, and
the camera never opens at all if the configured startup position is
`hidden`.

## Visual treatments

Five shaders the live feed can be rendered through, cycled with
`Ctrl+KP /` and `Ctrl+KP *` (see `_TREATMENT_FRAGS` / `_TREATMENT_NAMES`
in `webcam_overlay.py`):

| Treatment | What it does |
|---|---|
| `raw`     | Clean feed, no processing |
| `neon`    | Bass-reactive exposure punch + scanline modulation, slight blue push |
| `glitch`  | Bass/treble-driven chromatic aberration and beat-triggered scanline scramble |
| `vintage` | Sepia tone, vignette, animated film grain, bass-reactive red push |
| `thermal` | Luminance mapped through a false-colour heat palette, bass-boosted |

The default treatment is `neon` (or `[webcam] treatment`, if set).

### Auto-cycle

`Ctrl+KP Enter` toggles auto-cycle, which alternates the plain feed
(`raw`) with the next treatment rather than round-robining flatly through
every treatment including raw:

```text
normal -> effect1 -> normal -> effect2 -> normal -> effect3 -> ...
```

An effect airs for half of `[webcam] cycle_interval`; the normal
interlude between effects airs for the full (un-halved) `cycle_interval`,
so effects read as brief highlights rather than eating equal screen time
to plain video. Manual treatment switching (`Ctrl+KP /`, `Ctrl+KP *`) is
unaffected — it still cycles flatly through every treatment, raw
included.

### Selfie-segmentation background dim

Optional, per-camera, off by default. Runs MediaPipe's selfie-segmentation
model on the capture thread (never the render thread — inference costs
~8–15 ms, too much of the 16.67 ms frame budget to spend there) and dims
every background pixel to 25% of its original brightness, leaving the
detected person untouched. Applied *before* every treatment shader, so
raw/neon/glitch/vintage/thermal all receive an already-dimmed background
for free.

Requires `mediapipe` (see `requirements.txt` — needs `--no-deps` to avoid
a conflict with this project's pinned `opencv-python-headless`) and the
bundled model at `assets/models/selfie_segmenter.tflite` (Apache 2.0 —
see `assets/models/NOTICE.md` for provenance/attribution). Absent either,
the toggle silently does nothing rather than erroring.

Toggle per-camera from the Config Editor's **Visuals** tab — one 0/1
slider per detected camera (the generic Config Editor has no native
checkbox widget, so booleans ride a min=0/max=1 slider by the same
convention used elsewhere in the app). Not configurable from
`config.toml` directly, since it's inherently per-camera rather than a
single scalar.

## Audio reactivity

Only `raw` is silent. Every other treatment reads `iBass`; `glitch` is
the only one that also reads `iTreble`.

| Signal | Effect |
|---|---|
| `bass` (`iBass`) | `neon`: exposure punch + faint blue push. `glitch`: chromatic-aberration spread + probability/magnitude of the beat-triggered scanline scramble. `vintage`: red-channel push. `thermal`: luminance boost before the heat-palette lookup. Also drives the PiP border's glow/thickness pulse in `render()`, independent of treatment. |
| `treble` (`iTreble`) | `glitch` only: adds to the chromatic-aberration spread. |

## Requirements

```
opencv-python-headless >= 4.9
```

Already listed in the main `requirements.txt`.

Optional (selfie-segmentation only, not auto-installed — see
`requirements.txt` in this directory for the install command):

```
mediapipe==1.0.0
```

## Installer

This drop-in ships a bundle installer script at `install.sh`.

```bash
bash install.sh --root ~/.local/share/unicorn-viz
```

The installer copies the bundle into `~/.local/share/unicorn-viz/drop-ins/webcam-01`
by default. Use `--uninstall` to remove that bundle again.

## Config

`[webcam]` in `config.toml` (see the section's own comments there for the
live, authoritative defaults — this is a summary, not a second source of
truth):

```toml
[webcam]
device            = 0              # /dev/video index (Linux) or DirectShow index (Windows)
width             = 1280
height            = 720
fps               = 30
pip_position      = "bottom_right" # top_left | top_center | top_right |
                                    # left | center | right |
                                    # bottom_left | bottom_center | bottom_right |
                                    # fullscreen | hidden
pip_scale         = 0.33           # fraction of viewport width (0.12-0.80)
border_thickness  = 8.0            # border/backplate thickness in pixels
border_color      = [0.04, 0.72, 1.0, 0.28]  # RGBA, 0.0-1.0
treatment         = "neon"         # raw | neon | glitch | vintage | thermal
cycle_interval    = 30             # seconds; see Auto-cycle above
brightness        = 1.0            # software multiplier, 0.2-2.0
contrast          = 1.0            # software multiplier, 0.2-2.0
flip_horizontal   = true           # mirror by default
flip_vertical     = false
switch_hide_duration_s = 1.2       # PiP hidden briefly during a device switch
enabled           = true           # false disables the whole overlay + camera worker
# selfie_seg is per-camera, not a single scalar here -- see Config Editor
# Visuals tab, above.
```

`pip_position = "hidden"` is the shipped default — the camera device is
never opened (or even probed) until the overlay is shown for the first
time; see the Description above.

## Hotkeys

Keypad, PageUp/PageDown, and `Ctrl+Alt+K` controls are active at all
times, independent of which effect is on screen. The full, current list
lives in code as the project's single source of truth for keybindings
(see CLAUDE.md's Drop-In policy) rather than being duplicated here where
it can drift stale again: `HELP_ENTRIES` and the module docstring at the
top of `webcam_overlay.py`, both merged automatically into the in-app `H`
help overlay. `Alt+[` / `Alt+]` also resize the PiP as an alias for
`Ctrl+KP -` / `Ctrl+KP +`.

## Planning

- docs/planning.md

## Tags

`webcam` · `overlay` · `camera` · `drop-in`

## Changelog

- **1.1.1** — Fixes the Config Editor's Visuals tab showing zero
  selfie-seg rows to toggle. `config_editor_settings()` only iterated
  `_available_cameras`, which stays empty until the camera has been
  shown at least once — `start()` deliberately skips `rediscover_cameras()`
  while `pip_position = "hidden"` (the shipped default), for camera
  privacy. Now unions in the configured/active device id and any
  persisted per-camera settings, without probing hardware. 2 regression
  tests.
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
