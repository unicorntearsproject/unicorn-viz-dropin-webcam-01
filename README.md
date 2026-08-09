# webcam-01 — Webcam Overlay

**Version 1.4.0**

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

### Selfie-segmentation background cutout

Optional, per-camera, off by default. Four modes:

| Mode | Effect |
|---|---|
| `off` | No segmentation runs at all |
| `dim` | Background dims to 25% of its original brightness |
| `blur` | Background gets a strong Gaussian blur instead of dimming — foreground stays sharp |
| `transparent` | Background cuts to a true per-pixel alpha hole — whatever is already rendered behind the PiP (the show, not a solid backing colour) shows through, and the PiP border/backplate is skipped entirely so the cutout floats free rather than sitting in a coloured frame |

Runs MediaPipe's selfie-segmentation model on the capture thread (never
the render thread — inference costs ~8–15 ms, too much of the 16.67 ms
frame budget to spend there). Applied *before* every treatment shader, so
raw/neon/glitch/vintage/thermal all receive an already-processed
background for free, alpha included — each shader samples `cam_tex.a`
and carries it straight through to `fragColor.a` (for `glitch`, from the
un-shifted centre uv, so the cutout edge doesn't wobble with the
chromatic-aberration offset).

Every non-`off` mode mattes the raw segmentation confidence through the
same grow-then-feather technique you'd use selecting a background in an
image editor — grow the selection a couple px, feather, then cut:
binarize, dilate the kept region (past stray fringe pixels like hair),
then Gaussian-blur the edge so it reads as a smooth ramp instead of a
sticker cutout (`_feather_mask()`). Before that, the raw per-frame
confidence mask is blended toward the *previous* frame's raw mask
(`selfie_seg_temporal`) — MediaPipe has no notion of "last frame", so
movement makes the raw mask noisy at the boundary frame to frame, which
reads as a shaky/flickery edge. Blending smooths the boundary's actual
position, not just the already-feathered alpha values.

`transparent` needs real GPU alpha blending to look right, not just an
RGBA texture: the border/backplate clear (`_clear_border_ring()`) only
paints the four ring strips around the PiP in every other mode, never
the inner camera rect, so cutout pixels blend against whatever was
already drawn there that frame instead of a flat backing colour — and in
`transparent` specifically, the border is skipped altogether.

Requires `mediapipe` (see `requirements.txt` — needs `--no-deps` to avoid
a conflict with this project's pinned `opencv-python-headless`) and the
bundled model at `assets/models/selfie_segmenter.tflite` (Apache 2.0 —
see `assets/models/NOTICE.md` for provenance/attribution). Absent either,
the selector silently does nothing rather than erroring.

Select per-camera from the Config Editor's **Visuals** tab — one 0-3
stepped slider per detected camera (the generic Config Editor has no
native dropdown widget, so the mode index rides a slider by the same
convention used elsewhere in the app). The mode itself is per-camera; the
matte tuning below it is global (shapes every camera's cutout the same
way, rather than needing to be re-tuned per camera):

| Slider | Meaning | Default | Range |
|---|---|---|---|
| `selfie_seg_grow_px` | How far the kept (foreground) region grows past its original edge, in px | 3 | 0–20 |
| `selfie_seg_feather_px` | Gaussian-blur radius applied to the grown edge, in px | 4 | 0–30 |
| `selfie_seg_blur_px` | Gaussian-blur radius applied to the whole background in `blur` mode, in px | 35 | 0–100 |
| `selfie_seg_temporal` | How much of the previous frame's raw mask carries forward — higher is smoother but laggier | 0.5 | 0–0.95 |

None of this is configurable from `config.toml` directly — mode is
inherently per-camera, and the tuning sliders are meant for live
adjustment while watching the feed.

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

- **1.4.0** — Temporal smoothing for the selfie-seg matte: the raw
  per-frame confidence mask now blends toward the previous frame's raw
  mask (`selfie_seg_temporal`, default 0.5, new global live slider)
  before grow+feather runs, so the cutout boundary doesn't visibly
  flicker/shake during movement (MediaPipe scores every frame from
  scratch with no memory of the last one). `_CameraWorker` carries
  `_prev_raw_mask` across frames; a resolution change (fresh worker
  instance, or a mismatched cached shape) just starts smoothing over.
  6 new regression tests (prev-mask bookkeeping, shape-mismatch safety,
  a direct instant-vs-smoothed comparison), 52 total in this drop-in.
- **1.3.0** — Selfie-seg gains a `blur` mode (background gets a strong
  Gaussian blur instead of dimming; default radius 35px — "DEF huge",
  per the owner) between `dim` and `transparent`. The `transparent`
  mode no longer draws the PiP border/backplate at all — a cutout
  floating over the show read wrong boxed in a coloured frame. The
  grow/feather/blur radii (previously fixed module constants) are now
  three global live sliders on the Config Editor's Visuals tab
  (`selfie_seg_grow_px`, `selfie_seg_feather_px`, `selfie_seg_blur_px`),
  pushed straight through to the running capture worker. Per-camera mode
  selector is now a 0–3 stepped slider (was 0–2). 8 new regression
  tests (blur math, tuning setters/push-to-worker, render()'s
  border-skip branch), 47 total in this drop-in.
- **1.2.0** — Selfie-seg gains two more modes: `off` / `dim` (existing) /
  `transparent` (new) — a true per-pixel alpha cutout that reveals
  whatever is already rendered behind the PiP, not a solid backing
  colour. Both modes now matte the raw segmentation confidence through a
  grow (3px) + feather (4px) pass (`_feather_mask()`) instead of using
  it raw, so the cutout edge reads as a smooth ramp past stray fringe
  pixels rather than a sticker cutout — the same "select background,
  grow a couple px, feather, cut" technique from an image editor.
  Required: all 5 treatment shaders now sample and propagate `cam_tex.a`
  instead of hardcoding opaque output (`glitch` reads alpha from the
  un-shifted centre uv so the cutout doesn't wobble with the chromatic
  aberration offset); `_get_camera_texture()` tracks channel count and
  only brightness/contrast-adjusts the RGB channels, never alpha; the
  PiP border clear (`_clear_border_ring()`) now paints only the four
  ring strips around the camera rect instead of the whole backing
  rectangle, so transparent-mode pixels blend against the actual frame
  behind the PiP rather than the border colour. The per-camera Config
  Editor selector is now a 0/1/2 stepped slider (was 0/1). 6 new
  regression tests (grow/feather math + both modes' pixel output), 40
  total in this drop-in.
- **1.1.2** — selfie-seg Config Editor rows declare `step: 1.0` so one
  editor notch flips the 0/1 toggle (with the core default step of
  (max-min)/40, the >= 0.5 quantizing setter floored every nudge back
  to 0 and the toggle was unreachable by click or scroll).
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
