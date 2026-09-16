# webcam-01 — Webcam Overlay

**Version 1.6.0**

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

Seven shaders the live feed can be rendered through, cycled with
`Ctrl+KP /` and `Ctrl+KP *` (see `_TREATMENT_FRAGS` / `_TREATMENT_NAMES`
in `webcam_overlay.py`):

| Treatment | What it does |
|---|---|
| `raw`       | Clean feed, no processing |
| `neon`      | Bass-reactive exposure punch + scanline modulation, slight blue push |
| `glitch`    | Bass/treble-driven chromatic aberration and beat-triggered scanline scramble |
| `vintage`   | Sepia tone, vignette, animated film grain, bass-reactive red push |
| `thermal`   | Luminance mapped through a false-colour heat palette, bass-boosted |
| `posterize` | Cel-shaded colour quantization (bass briefly coarsens the levels) + a Sobel-derived neon edge outline (hue-cycling, treble-reactive) |
| `halftone`  | Retro CMYK-print dot-matrix look — bass shrinks the dot pitch and widens each dot's radius on a hit, treble spins the grid angle, hue drifts slowly |

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

### PiP border

`[webcam] border_style` picks the look of the ring drawn around the PiP
(`_clear_border_ring()` for `classic`, `_draw_border_futuristic()` +
`_BORDER_FRAG` for `futuristic`):

| Style | Look |
|---|---|
| `off` | No border/backplate drawn at all |
| `classic` | The original flat-colour ring (`border_color`, `border_thickness`) |
| `futuristic` (default) | Glowing neon/HUD frame: crisp corner-bracket accents, a treble-reactive scanline sweep, and organic bass/beat-reactive "drip" tendrils that locally thicken the ring — heavier toward the bottom (gravity) and on a bass hit (sag), the same wave-widens-with-bass / extra-sag-on-beat idea as banner-01's `_draw_drippy_text`, adapted from per-character text motion to a border ring's local thickness |

`futuristic` is configurable, not one hardcoded look — `border_intensity`
(0–2) dials the overall glow/drip strength, and it keeps varying on its
own regardless of config: the drip motion is always alive (never fully
static, like the base wobble in `_draw_drippy_text`), and the neon hue
slowly drifts over time at `border_hue_speed` full turns/second, starting
from `border_color`'s own hue (`_hue_from_color()`) so the configured
tint still sets the starting point. A bass-transient ("beat-ish") pulse
is approximated by thresholding `iBass` — this drop-in's `render()` only
receives bass/treble, not a true beat trigger, the same idiom the
`glitch` treatment already uses for its scan-scramble.

Interior pixels of the ring shader are always fully transparent; the
opaque camera quad drawn immediately afterward redraws its own exact
rect on top regardless, so drips only ever grow the ring *outward* (into
a reserved padding margin) rather than bulging into the frame, where
they'd just be invisible.

None of `border_style` / `border_intensity` / `border_hue_speed` are
per-camera — set them in `config.toml`, or live from the Config Editor's
Visuals tab (`border_style` rides a 0–2 stepped slider by the same
"no native dropdown" convention as the selfie-seg mode selector below).

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
all seven treatments receive an already-processed background for free,
alpha included — each shader samples `cam_tex.a`
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
RGBA texture: whichever border style is active only ever paints the ring
around the PiP (never the inner camera rect) in every other selfie-seg
mode, so cutout pixels blend against whatever was already drawn there
that frame instead of a flat backing colour or a border shader's glow —
and in `transparent` specifically, no border is drawn at all, regardless
of `border_style`.

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

Only `raw` is silent. Every other treatment reads `iBass`; `glitch`,
`posterize` and `halftone` also read `iTreble`. The `futuristic` border
style reads both, independent of whichever treatment is showing.

| Signal | Effect |
|---|---|
| `bass` (`iBass`) | `neon`: exposure punch + faint blue push. `glitch`: chromatic-aberration spread + probability/magnitude of the beat-triggered scanline scramble. `vintage`: red-channel push. `thermal`: luminance boost before the heat-palette lookup. `posterize`: briefly coarsens the colour quantization on a hit. `halftone`: shrinks the dot pitch and widens each dot's radius on a hit. Also drives the PiP border's ring-thickness pulse in every style, and the `futuristic` style's drip-tendril length/sag (a bass-transient pulse approximates a "beat" since `render()` has no `iBeat`). |
| `treble` (`iTreble`) | `glitch`: adds to the chromatic-aberration spread. `posterize`: speeds up and brightens the edge outline's hue cycle. `halftone`: spins the dot-grid angle. `futuristic` border: brightens/densifies the scanline sweep. |

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
pip_position      = "hidden"       # top_left | top_center | top_right |
                                    # left | center | right |
                                    # bottom_left | bottom_center | bottom_right |
                                    # fullscreen | hidden
pip_scale         = 0.33           # fraction of viewport width (0.12-0.80)
border_thickness  = 8.0            # base ring thickness in pixels (all styles)
border_color      = [0.04, 0.72, 1.0, 0.28]  # RGBA, 0.0-1.0; also seeds the
                                    # futuristic style's starting neon hue
border_style      = "futuristic"   # off | classic | futuristic -- see PiP border above
border_intensity  = 1.0            # futuristic style: overall glow/drip strength, 0.0-2.0
border_hue_speed  = 0.05           # futuristic style: neon hue drift, full turns/second, 0.0-1.0
treatment         = "neon"         # raw | neon | glitch | vintage | thermal |
                                    # posterize | halftone
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

- **1.6.0** — Config editor 2.0 rows: segmentation mode per camera and the matte radii move to the Performance tab as labeled choice/slider rows; new capture rate and capture size rows (next launch) and a treatment-cycle row (live) join them. Border rows keep Visuals with proper labels.
- **1.5.4** — Moved the Control Room panel to the shared `overlays` page
  (was `main`), where it belongs alongside banner/chat/lyrics per
  `docs/planning/control-room-drop-in-integration-plan-2026-09-05.md`
  section 3.7 — and where it fixes a real regression: once several
  drop-ins were registering panels on `main`, the band there was only
  ever sized for one row (see control-room-01's own changelog), so the
  webcam panel silently stopped rendering at all. Also polishes the
  panel now that it has real room: brightness/contrast are `PanelMeter`
  bars (normalized against the actual `[0.2, 2.0]` clamp range) instead
  of a plain text row, and a new `Devices` row reports detected/enabled
  camera counts. 4 new tests.
- **1.5.3** — the *code* default for `pip_position` is now `"hidden"`, matching
  what the README and the privacy comments have called the shipped default
  since rc.2. It was still `"bottom_right"`, so a config without a `[webcam]`
  section (the bare-bones `config.dist.toml` the installers seed) probed and
  opened every camera at boot — 90 s of black screen on the first Windows
  beta run (three cameras through Media Foundation).
- **1.5.2** — Control Room panel is unregistered in `destroy()` so a mid-session teardown never leaves a dead panel (error box) behind — the `operator_panels.py` lifecycle rule from `docs/planning/control-room-drop-in-integration-plan-2026-09-05.md`.
- **1.5.1** — **Control Room panel, P2 of
  `docs/planning/control-room-panel-registry-plan-2026-09-04.md`.**
  Registers a declarative `Webcam` panel via
  `vj_api.register_operator_panel()` (guarded, degrades to no panel on
  an older core), replacing Control Room's hardcoded panel, which always
  drew full controls even with webcam-01 not loaded. The panel shows
  active camera + brightness/contrast (or a switching countdown), and
  PREV/NEXT/REDISC/B±/C±/FLIP H/FLIP V buttons wired to the existing
  camera and image-control methods, unchanged. Per-device select/enable
  chips (capped at 2 devices in the old panel) are not carried forward —
  the declarative button row is single-row-only, and PREV/NEXT/REDISC
  already cover cycling through and refreshing every detected camera.
  11 new tests.
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
- **1.5.0** — Two additions:
  1. Two new treatments: `posterize` (cel-shaded colour quantization +
     a Sobel-derived, hue-cycling neon edge outline) and `halftone`
     (retro CMYK-print dot matrix, bass/treble-reactive pitch and grid
     angle). `_TREATMENT_NAMES` grows from five entries to seven; both
     read `iBass`/`iTreble`/`iTime` like the existing shaders.
  2. The PiP border is redesigned. A new `border_style` config key picks
     `off` / `classic` (the original flat ring, unchanged) / `futuristic`
     (new default): a glowing neon/HUD frame with corner-bracket accents,
     a treble-reactive scanline sweep, and organic bass/beat-reactive
     "drip" tendrils that locally thicken the ring — heavier at the
     bottom (gravity) and on a bass hit (sag), adapting banner-01's
     `_draw_drippy_text` wave-widens-with-bass/extra-sag-on-beat idea
     from per-character text motion to a border ring's local thickness.
     Rendered via a new `_BORDER_FRAG` shader + `_border_prog`/
     `_border_vao`/`_border_vbo` GL resources (replacing the flat
     `_clear_border_ring()` scissor-clear for this style only —
     `classic` still uses it unchanged). Configurable via
     `border_intensity` (0-2, overall glow/drip strength) and
     `border_hue_speed` (0-1, neon hue drift in turns/second, seeded
     from `border_color`'s own hue via the new `_hue_from_color()`); it
     also keeps varying on its own regardless of config — the drip
     motion is always alive and the hue keeps drifting — so a long
     session doesn't read static. All three are live sliders on the
     Config Editor's Visuals tab, same "no native dropdown" stepped-
     slider convention as the selfie-seg mode selector. `_quad_verts()`
     factors the shared pixel-to-clip-space quad math out of
     `_write_cam_quad()` so the new `_write_border_quad()` (camera rect
     grown by a padding margin that reserves room for drips) can reuse
     it exactly. No border of any style is drawn in `transparent`
     selfie-seg mode, matching the 1.3.0 behaviour. 24 new regression
     tests (style dispatch, hue extraction, uniform/geometry math,
     Config Editor round-trip, construction), 76 total in this drop-in.
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
