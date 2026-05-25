# webcam-01 — Webcam Overlay Effect

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

## Tags

`webcam` · `overlay` · `camera` · `drop-in`
