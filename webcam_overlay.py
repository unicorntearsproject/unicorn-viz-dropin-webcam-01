"""webcam-01 — WebcamSystem subsystem drop-in.

Always-on camera overlay rendered above every main effect, below the HUD.
Provides seven visual treatments that cycle with KP-/ and KP-*.
Does not appear in the main effect playlist.

Keypad controls (active at all times regardless of which effect is running):
    Ctrl+KP 0     Fullscreen camera
    Ctrl+KP .     Hide camera
    Ctrl+KP 1..9  PiP position (bottom-left .. top-right grid)
    Ctrl+KP -     Shrink PiP
    Ctrl+KP +     Grow PiP
    Ctrl+KP /     Previous camera treatment
    Ctrl+KP *     Next camera treatment
    Ctrl+KP Enter Toggle auto-cycle camera treatments
    PageUp        Previous camera device
    PageDown      Next camera device
    Ctrl+Alt+K    Webcam editor modal (core-dispatched)

Camera treatments:
    raw       Clean camera feed, minimal processing
    neon      Scanline punch + bass-reactive exposure
    glitch    Chromatic aberration + scan scramble on beats
    vintage   Sepia tone, film grain, vignette
    thermal   False-colour luminance heat map
    posterize Cel-shaded colour quantization + neon Sobel-edge outline
    halftone  CMYK-print dot matrix, bass-reactive pitch, spinning grid

PiP border (see `_border_style` / `border_style` in config.toml): 'off'
draws nothing, 'classic' is the original flat ring, and the default
'futuristic' is a glowing neon/HUD frame with corner-bracket accents, a
treble-reactive scanline sweep, and organic bass/beat-reactive "drip"
tendrils (see `_BORDER_FRAG`) -- configurable via `border_style` /
`border_intensity` / `border_hue_speed` and always animating on its own
(drip motion + a slow hue drift) so a long session doesn't look static.

Auto-cycle (Ctrl+KP Enter) alternates normal (raw) and the next treatment
rather than round-robining flatly through every treatment including raw:
normal -> effect -> normal -> effect -> ... An effect airs for half of
`cycle_interval`; the normal interlude between effects airs for the full
(un-halved) `cycle_interval`. Manual treatment switching (Ctrl+KP /, *) is
unaffected -- it still cycles flatly through every treatment.

Selfie-segmentation background cutout (optional, per-camera, off by
default): applied before every treatment above, so all seven treatments
get an already-processed background for free. Four modes -- off / dim /
blur / transparent -- selected per-camera from the
Config Editor's Visuals tab (one 0/1/2/3 slider per detected camera), not
from config.toml. Needs `mediapipe` (see requirements.txt) -- absent, the
selector just does nothing. `transparent` outputs a genuine per-pixel
alpha cutout, not a solid backing colour -- pixels the segmenter drops
reveal whatever was already drawn behind the PiP that frame (needs the
render's blend state, enabled globally at context init), and the PiP
border/backplate is skipped entirely in this mode so the cutout floats
free rather than sitting in a coloured frame.

All non-off modes matte the raw segmentation confidence through the same
grow-then-feather technique as selecting a background in an image editor:
binarize, dilate the kept region by `grow_px` to push the edge past stray
fringe pixels, then Gaussian-blur by `feather_px` so the now-hard edge
reads as a smooth ramp instead of a sticker cutout. `blur` additionally
Gaussian-blurs the whole background by `blur_px` instead of dimming it.
Before any of that, the raw per-frame confidence mask is blended toward
the *previous* frame's raw mask by `temporal` (the model has no notion of
"last frame", so movement makes the boundary noisy frame to frame without
this) -- smooths the boundary position itself, not just the final alpha.
All four values are global (not per-camera) live sliders on the Config
Editor's Visuals tab, defaulting to `_SELFIE_SEG_GROW_PX_DEFAULT` /
`_SELFIE_SEG_FEATHER_PX_DEFAULT` / `_SELFIE_SEG_BLUR_PX_DEFAULT` /
`_SELFIE_SEG_TEMPORAL_DEFAULT`.
"""
from __future__ import annotations

__version__ = '1.5.0'

import colorsys
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import moderngl
import numpy as np

try:
    import cv2  # type: ignore
    _CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2 = False

# Optional: selfie-segmentation background dim. Guarded exactly like cv2
# above -- absent mediapipe just disables the per-camera toggle rather than
# breaking the drop-in. See requirements.txt for the install command (it
# needs --no-deps to avoid an opencv-contrib-python/opencv-python-headless
# conflict).
try:
    import mediapipe as _mp  # type: ignore
    from mediapipe.tasks.python import BaseOptions as _MpBaseOptions  # type: ignore
    from mediapipe.tasks.python import vision as _mp_vision  # type: ignore
    _MEDIAPIPE = True
except Exception:  # pragma: no cover
    _mp = None  # type: ignore
    _MpBaseOptions = None  # type: ignore
    _mp_vision = None  # type: ignore
    _MEDIAPIPE = False

_SELFIE_SEG_MODEL_PATH = Path(__file__).resolve().parent / 'assets' / 'models' / 'selfie_segmenter.tflite'
# Background pixels are scaled to this fraction of their original
# brightness in 'dim' mode; foreground (person) pixels are left untouched.
# Applied before every existing treatment shader runs, so
# raw/neon/glitch/vintage/thermal all receive an already-processed
# background for free.
_SELFIE_SEG_BG_DIM = 0.25
# Grow-then-feather matte parameters (image-editor "select background,
# grow selection a couple px, feather, cut" technique) and the 'blur'
# mode's Gaussian radius. These are defaults only -- all four are
# tunable live from the Config Editor's Visuals tab (global, not
# per-camera; see WebcamSystem._selfie_seg_grow_px and friends).
_SELFIE_SEG_GROW_PX_DEFAULT = 3
_SELFIE_SEG_FEATHER_PX_DEFAULT = 4
_SELFIE_SEG_BLUR_PX_DEFAULT = 35  # "DEF huge" -- a strong blur out of the box
# Temporal smoothing: how much of the *previous* frame's raw confidence
# mask carries into this frame's blend, in [0, 1) -- 0 disables it
# entirely (each frame stands alone), higher values trade responsiveness
# for stability. Movement makes the raw per-frame mask noisy at the
# boundary (the model has no notion of the previous frame), which reads
# as a shaky/flickery edge; blending toward the prior mask before the
# grow+feather pass smooths the *boundary position* itself, not just the
# already-feathered alpha.
_SELFIE_SEG_TEMPORAL_DEFAULT = 0.5
_SELFIE_SEG_MODES: tuple[str, ...] = ('off', 'dim', 'blur', 'transparent')

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

_CAM_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_TREATMENT_FRAGS: dict[str, str] = {

    'raw': """
#version 330
uniform sampler2D cam_tex;
in  vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 cam = texture(cam_tex, v_uv);
    fragColor = vec4(cam.rgb, cam.a);
}
""",

    'neon': """
#version 330
uniform sampler2D cam_tex;
uniform float iBass;
uniform float iTime;
in  vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 cam = texture(cam_tex, v_uv);
    vec3 col = cam.rgb;
    float punch = 1.0 + iBass * 0.30;
    col *= punch;
    col *= 0.96 + 0.04 * sin((v_uv.y + iTime * 0.2) * 700.0);
    col.b += iBass * 0.06;
    fragColor = vec4(clamp(col, 0.0, 1.0), cam.a);
}
""",

    'glitch': """
#version 330
uniform sampler2D cam_tex;
uniform float iBass;
uniform float iTreble;
uniform float iTime;
in  vec2 v_uv;
out vec4 fragColor;
void main() {
    vec2 uv = v_uv;
    // Alpha (the segmentation matte) always comes from the un-shifted
    // centre uv -- the cutout edge must not wobble with the chromatic
    // aberration / scan-glitch colour offsets below.
    float a = texture(cam_tex, uv).a;
    float ca = iBass * 0.018 + iTreble * 0.004;
    vec3 col;
    col.r = texture(cam_tex, uv + vec2( ca, 0.0)).r;
    col.g = texture(cam_tex, uv              ).g;
    col.b = texture(cam_tex, uv - vec2( ca, 0.0)).b;
    float scan = step(0.72, iBass) * step(0.0, sin(uv.y * 180.0 + iTime * 28.0));
    float shift = scan * iBass * 0.045;
    col = mix(col, texture(cam_tex, vec2(uv.x + shift, uv.y)).rgb, scan);
    fragColor = vec4(clamp(col, 0.0, 1.0), a);
}
""",

    'vintage': """
#version 330
uniform sampler2D cam_tex;
uniform float iBass;
uniform float iTime;
in  vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 cam = texture(cam_tex, v_uv);
    vec3 col = cam.rgb;
    vec3 sepia;
    sepia.r = dot(col, vec3(0.393, 0.769, 0.189));
    sepia.g = dot(col, vec3(0.349, 0.686, 0.168));
    sepia.b = dot(col, vec3(0.272, 0.534, 0.131));
    col = mix(col, sepia, 0.68);
    vec2 uv = v_uv - 0.5;
    float vig = 1.0 - dot(uv, uv) * 2.4;
    col *= clamp(vig, 0.0, 1.0);
    float grain = fract(sin(dot(v_uv + iTime * 0.01, vec2(12.9898, 78.233))) * 43758.5453);
    col += (grain - 0.5) * 0.045;
    col.r += iBass * 0.05;
    fragColor = vec4(clamp(col, 0.0, 1.0), cam.a);
}
""",

    'thermal': """
#version 330
uniform sampler2D cam_tex;
uniform float iBass;
in  vec2 v_uv;
out vec4 fragColor;
vec3 thermalPalette(float t) {
    t = clamp(t, 0.0, 1.0);
    vec3 col = vec3(0.0);
    col += smoothstep(0.0,  0.25, t) * vec3(0.0,  0.0,  1.0);
    col += smoothstep(0.25, 0.5,  t) * vec3(0.0,  1.0,  0.0);
    col += smoothstep(0.5,  0.75, t) * vec3(1.0,  1.0, -1.0);
    col += smoothstep(0.75, 1.0,  t) * vec3(0.0, -1.0,  0.0);
    return clamp(col, 0.0, 1.0);
}
void main() {
    vec4 cam = texture(cam_tex, v_uv);
    float luma = dot(cam.rgb, vec3(0.299, 0.587, 0.114));
    float boosted = clamp(luma + iBass * 0.18, 0.0, 1.0);
    fragColor = vec4(thermalPalette(boosted), cam.a);
}
""",

    # Cel-shaded colour quantization with a Sobel-derived neon edge
    # outline. Bass briefly coarsens the quantization (fewer levels,
    # bigger colour jumps) on a hit; treble speeds up and brightens the
    # outline's hue-cycling glow.
    'posterize': """
#version 330
uniform sampler2D cam_tex;
uniform float iBass;
uniform float iTreble;
uniform float iTime;
in  vec2 v_uv;
out vec4 fragColor;
vec3 hsvToRgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    vec2 texel = 1.0 / vec2(textureSize(cam_tex, 0));
    vec4 cam = texture(cam_tex, v_uv);

    float levels = mix(10.0, 4.0, clamp(iBass * 1.4, 0.0, 1.0));
    vec3 quant = floor(cam.rgb * levels + 0.5) / levels;

    float tl = dot(texture(cam_tex, v_uv + texel * vec2(-1.0, -1.0)).rgb, vec3(0.299, 0.587, 0.114));
    float t  = dot(texture(cam_tex, v_uv + texel * vec2( 0.0, -1.0)).rgb, vec3(0.299, 0.587, 0.114));
    float tr = dot(texture(cam_tex, v_uv + texel * vec2( 1.0, -1.0)).rgb, vec3(0.299, 0.587, 0.114));
    float l  = dot(texture(cam_tex, v_uv + texel * vec2(-1.0,  0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float r  = dot(texture(cam_tex, v_uv + texel * vec2( 1.0,  0.0)).rgb, vec3(0.299, 0.587, 0.114));
    float bl = dot(texture(cam_tex, v_uv + texel * vec2(-1.0,  1.0)).rgb, vec3(0.299, 0.587, 0.114));
    float bt = dot(texture(cam_tex, v_uv + texel * vec2( 0.0,  1.0)).rgb, vec3(0.299, 0.587, 0.114));
    float br = dot(texture(cam_tex, v_uv + texel * vec2( 1.0,  1.0)).rgb, vec3(0.299, 0.587, 0.114));
    float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
    float gy = -tl - 2.0 * t - tr + bl + 2.0 * bt + br;
    float edge = clamp(sqrt(gx * gx + gy * gy), 0.0, 1.0);

    float hue = fract(iTime * (0.05 + iTreble * 0.25));
    vec3 edgeCol = hsvToRgb(vec3(hue, 0.85, 1.0));
    vec3 col = mix(quant, edgeCol, edge * (0.65 + iTreble * 0.35));
    fragColor = vec4(clamp(col, 0.0, 1.0), cam.a);
}
""",

    # Retro CMYK-print dot-matrix halftone. Luminance is resampled per
    # grid cell and drawn as a variable-radius dot; bass shrinks the dot
    # pitch (denser dots) and widens each dot's radius on a hit, treble
    # spins the grid angle, and a slow hue drift keeps a long session
    # from reading static.
    'halftone': """
#version 330
uniform sampler2D cam_tex;
uniform float iBass;
uniform float iTreble;
uniform float iTime;
in  vec2 v_uv;
out vec4 fragColor;
vec3 hsvToRgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}
void main() {
    vec2 res = vec2(textureSize(cam_tex, 0));
    float cellPx = max(3.0, 9.0 - iBass * 3.0);
    float angle = 0.35 + iTreble * 0.6 + sin(iTime * 0.07) * 0.15;
    float c = cos(angle);
    float s = sin(angle);
    mat2 rot = mat2(c, -s, s, c);

    vec2 uvPx = v_uv * res;
    vec2 ruv = rot * uvPx;
    vec2 cellCenterR = (floor(ruv / cellPx) + 0.5) * cellPx;
    vec2 cellCenterPx = transpose(rot) * cellCenterR;
    vec2 cellCenterUv = clamp(cellCenterPx / res, 0.0, 1.0);

    vec4 cell = texture(cam_tex, cellCenterUv);
    float luma = dot(cell.rgb, vec3(0.299, 0.587, 0.114));
    float radius = (1.0 - luma) * cellPx * 0.5 * (0.7 + iBass * 0.5);
    float d = length(ruv - cellCenterR);
    float dotMask = 1.0 - smoothstep(radius - 1.0, radius + 1.0, d);

    float hue = fract(iTime * 0.03);
    vec3 dotCol = hsvToRgb(vec3(hue, 0.55, 1.0));
    vec3 col = mix(vec3(0.03, 0.03, 0.06), dotCol, dotMask);
    fragColor = vec4(clamp(col, 0.0, 1.0), cell.a);
}
""",
}

_TREATMENT_NAMES: list[str] = ['raw', 'neon', 'glitch', 'vintage', 'thermal', 'posterize', 'halftone']

# PiP border look: 'off' draws nothing, 'classic' is the original flat
# scissor-clear ring (_clear_border_ring), 'futuristic' is the glowing
# neon/HUD/drippy shader border (_draw_border_futuristic / _BORDER_FRAG).
# See WebcamSystem._border_style and the `border_style` / `border_intensity`
# / `border_hue_speed` config.toml keys.
_BORDER_STYLES: tuple[str, ...] = ('off', 'classic', 'futuristic')

# Fragment shader for the 'futuristic' border style: a glowing neon/HUD
# frame around the PiP with crisp corner-bracket accents, a treble-reactive
# scanline sweep, and organic bass/beat-reactive "drip" tendrils that
# locally thicken the ring (heavier toward the bottom of the frame --
# gravity -- and on a bass hit -- "sag"). The drip concept mirrors
# banner-01's `_draw_drippy_text` (wave amplitude widens with bass, extra
# sag on a beat hit) adapted from per-character text motion to a border
# ring's local thickness. Interior pixels always stay fully transparent --
# the opaque cam quad drawn right after this always redraws its own exact
# rect on top, so any inward bulge here would be invisible anyway; drips
# only ever grow the ring *outward* into the padding reserved for them
# (see the `headroom` margin in _draw_border_futuristic).
#
# Expects: iRectPx (inner camera rect size, px), iPadPx (total quad
# padding around that rect, px -- how far glow/drips can reach, always
# >= 1), iBorderPx (base ring thickness within that padding, px, already
# bass-pulsed on the CPU side), iIntensity (0-2 overall strength dial),
# iHueBase/iHueSpeed (colour drift -- see WebcamSystem._border_hue_base),
# iTime/iBass/iTreble. Produces the border's RGBA colour; v_uv spans the
# OUTER padded rect, (0,0) at its top-left, (1,1) at its bottom-right
# (image-space, matching the camera texture's own row order).
_BORDER_FRAG = """
#version 330
uniform vec2  iRectPx;
uniform float iPadPx;
uniform float iBorderPx;
uniform float iIntensity;
uniform float iHueBase;
uniform float iHueSpeed;
uniform float iTime;
uniform float iBass;
uniform float iTreble;
in  vec2 v_uv;
out vec4 fragColor;

vec3 hsvToRgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

float hash11(float n) { return fract(sin(n) * 43758.5453123); }

// Signed "outside a rect" distance: <=0 inside [innerMin, innerMax],
// grows beyond it. Not a full rounded-box SDF -- this border is a plain
// right-angled HUD frame, no rounding.
float sdBoxOutside(vec2 p, vec2 innerMin, vec2 innerMax) {
    vec2 o = max(innerMin - p, p - innerMax);
    return max(o.x, o.y);
}

void main() {
    vec2 outerPx  = iRectPx + vec2(iPadPx * 2.0);
    vec2 fragPx   = v_uv * outerPx;
    vec2 innerMin = vec2(iPadPx);
    vec2 innerMax = iRectPx + vec2(iPadPx);
    float dist = sdBoxOutside(fragPx, innerMin, innerMax);

    // Bass-transient ("beat-ish") pulse -- this drop-in's render() only
    // receives bass/treble (no iBeat), so a hit is approximated by
    // thresholding iBass, the same idiom the 'glitch' treatment uses.
    float pulse = smoothstep(0.55, 0.95, iBass);

    // Organic per-column drip wobble, heavier toward the bottom (gravity)
    // and on a bass hit (sag) -- widens the ring locally, outward only.
    float cell = floor((fragPx.x + fragPx.y * 0.35) / 14.0);
    float jitter = hash11(cell);
    float phase = iTime * (0.5 + jitter * 0.9) + jitter * 6.2831;
    float drip = pow(0.5 + 0.5 * sin(phase), 3.0);
    float gravity = mix(0.5, 1.4, smoothstep(0.0, outerPx.y, fragPx.y));
    float dripPx = (6.0 + iBass * 22.0 + pulse * 16.0) * iIntensity * drip * gravity;
    float localBorder = max(0.5, min(iBorderPx + dripPx, iPadPx * 0.95));

    float band = smoothstep(0.0, 2.0, dist)
               * (1.0 - smoothstep(localBorder * 0.5, localBorder, max(dist, 0.0)));
    float halo = (1.0 - smoothstep(localBorder, iPadPx, max(dist, 0.0))) * 0.35;
    float glow = clamp(band + halo, 0.0, 1.0);

    // HUD corner brackets -- crisp accents at the camera rect's four
    // corners, deliberately NOT drippy (contrast against the melt).
    float dc = min(
        min(length(fragPx - innerMin), length(fragPx - vec2(innerMax.x, innerMin.y))),
        min(length(fragPx - vec2(innerMin.x, innerMax.y)), length(fragPx - innerMax))
    );
    float bracket = (1.0 - smoothstep(0.0, iBorderPx * 2.2, dc)) * step(dist, iBorderPx * 1.6);

    // Scanline sweep; treble adds sparkle density/brightness.
    float scan = pow(max(0.0, 0.5 + 0.5 * sin(fragPx.y * 0.9 - iTime * 40.0)), 8.0) * (0.4 + iTreble * 1.2);

    float hue = fract(iHueBase + iTime * iHueSpeed + fragPx.x * 0.0006 - fragPx.y * 0.0004);
    vec3 neon = hsvToRgb(vec3(hue, 0.85, 1.0));

    vec3 col = neon * (0.55 + scan * 0.6) + vec3(1.0) * bracket * 0.9;
    float alpha = clamp((glow + bracket * 0.6) * iIntensity * (1.0 + pulse * 0.4), 0.0, 1.0);
    fragColor = vec4(clamp(col, 0.0, 1.0), alpha);
}
"""


def _feather_mask(mask: np.ndarray, grow_px: int, feather_px: int) -> np.ndarray:
    """Grow the foreground selection, then feather the edge.

    Same technique as an image editor's "select the background, grow the
    selection a couple px, feather, then cut": dilating the binarized
    foreground pushes the matte edge past stray fringe pixels (hair,
    motion blur), then a Gaussian blur turns the now-hard edge back into a
    smooth alpha ramp so the cutout doesn't read as a sticker. Requires
    cv2 -- only ever called from the capture thread, which already
    requires _CV2 to run at all (see _CameraWorker.run()).

    ``mask`` is a single-channel float32 array in [0, 1] (foreground
    confidence); returns a same-shaped float32 array in [0, 1].
    """
    binary = (mask >= 0.5).astype(np.uint8) * 255
    if grow_px > 0:
        k = 2 * grow_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        binary = cv2.dilate(binary, kernel)
    alpha = binary.astype(np.float32) / 255.0
    if feather_px > 0:
        k = 2 * feather_px + 1  # cv2.GaussianBlur requires an odd kernel
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
    return np.clip(alpha, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------

class _CameraWorker(threading.Thread):
    """Background camera capture thread."""

    _MAX_OPEN_FAILURES = 5  # give up after this many consecutive not-found attempts

    def __init__(
        self, device: int, width: int, height: int, fps: int,
        selfie_seg_mode: str = 'off',
        selfie_seg_grow_px: int = _SELFIE_SEG_GROW_PX_DEFAULT,
        selfie_seg_feather_px: int = _SELFIE_SEG_FEATHER_PX_DEFAULT,
        selfie_seg_blur_px: int = _SELFIE_SEG_BLUR_PX_DEFAULT,
        selfie_seg_temporal: float = _SELFIE_SEG_TEMPORAL_DEFAULT,
    ) -> None:
        super().__init__(daemon=True, name='webcam-capture')
        self._device = int(device)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._lock = threading.Lock()
        self._device_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._switch_event = threading.Event()
        self._requested_device = int(device)
        self._latest: np.ndarray | None = None
        # Selfie segmentation runs here (dedicated capture thread), never on
        # the render thread -- inference costs ~8ms, which is a third of
        # the whole 16.67ms render budget on its own. The segmenter model
        # is loaded lazily, only if this ever leaves 'off', so cameras that
        # never use it never pay MediaPipe's ~250KB model load / init cost.
        self._selfie_seg_lock = threading.Lock()
        self._selfie_seg_mode = selfie_seg_mode if selfie_seg_mode in _SELFIE_SEG_MODES else 'off'
        self._selfie_seg_grow_px = max(0, int(selfie_seg_grow_px))
        self._selfie_seg_feather_px = max(0, int(selfie_seg_feather_px))
        self._selfie_seg_blur_px = max(0, int(selfie_seg_blur_px))
        self._selfie_seg_temporal = max(0.0, min(0.95, float(selfie_seg_temporal)))
        # Previous frame's raw (pre-feather) confidence mask, blended into
        # this frame's before grow+feather runs -- smooths the boundary
        # position itself across frames, not just the alpha values.
        # Capture-thread-only state; reset happens for free on device
        # switch / restart since a new _CameraWorker is a fresh instance.
        self._prev_raw_mask: np.ndarray | None = None
        self._segmenter: Any = None
        self._segmenter_failed = False
        # Bumped on every captured frame. Lets the render thread ask "is
        # there anything new?" without paying for a copy of a frame it has
        # already uploaded -- the camera runs at ~30 fps while the display
        # attempts 60, so most render frames have nothing new to show.
        self._seq: int = 0
        self._active = False

    def _try_open(self, dev: int):
        if sys.platform != 'win32':
            log.debug('WebcamSystem: trying /dev/video%d (V4L2)', dev)
            cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(dev)
        if not cap.isOpened():
            cap.release()
            return None
        if sys.platform != 'win32':
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        return cap

    def run(self) -> None:
        if not _CV2:
            log.error('WebcamSystem: opencv-python-headless not installed')
            return

        period = 1.0 / max(1, self._fps)

        initial_open = True  # True until first successful cap.isOpened()
        open_failures = 0
        while not self._stop_event.is_set():
            with self._device_lock:
                self._device = int(self._requested_device)
            cap = None
            for offset in range(3):
                cap = self._try_open(self._device + offset)
                if cap is not None:
                    if offset:
                        log.info('WebcamSystem: opened /dev/video%d (fallback)', self._device + offset)
                    break

            if cap is None:
                if initial_open:
                    log.warning(
                        'WebcamSystem: no camera at device %d..%d — giving up (reconnect and restart)',
                        self._device, self._device + 2,
                    )
                    return
                open_failures += 1
                if open_failures == 1:
                    log.warning(
                        'WebcamSystem: camera lost at device %d, retrying',
                        self._device,
                    )
                if open_failures >= self._MAX_OPEN_FAILURES:
                    log.warning(
                        'WebcamSystem: camera not available after %d reconnect attempts — giving up',
                        open_failures,
                    )
                    return
                self._stop_event.wait(3.0)
                continue

            initial_open = False
            open_failures = 0  # reset on successful open

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(self._width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
            cap.set(cv2.CAP_PROP_FPS,          float(self._fps))
            self._active = True
            log.info('WebcamSystem: camera %d opened at %dx%d',
                     self._device,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            fail_count = 0
            switched_device = False
            while not self._stop_event.is_set():
                if self._switch_event.is_set():
                    self._switch_event.clear()
                    switched_device = True
                    break
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)
                    if self._selfie_seg_mode != 'off':
                        frame = self._apply_selfie_seg(frame)
                    with self._lock:
                        self._latest = frame
                        self._seq += 1
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count >= 20:
                        log.warning('WebcamSystem: camera stopped responding, reconnecting')
                        break
                self._stop_event.wait(period * 0.45)

            cap.release()
            self._active = False
            if not self._stop_event.is_set() and not switched_device:
                self._stop_event.wait(2.0)

    def stop(self) -> None:
        self._stop_event.set()
        self._switch_event.set()

    def request_device(self, device: int) -> None:
        with self._device_lock:
            self._requested_device = int(device)
        log.debug('WebcamSystem: switch requested -> camera %d', int(device))
        self._switch_event.set()

    def set_selfie_seg_mode(self, mode: str) -> None:
        """Set live, from the render thread. Checked once per captured frame."""
        with self._selfie_seg_lock:
            self._selfie_seg_mode = mode if mode in _SELFIE_SEG_MODES else 'off'

    def set_selfie_seg_tuning(self, grow_px: int, feather_px: int, blur_px: int, temporal: float) -> None:
        """Set the matte grow/feather/blur radii and temporal smoothing live, from the render thread."""
        with self._selfie_seg_lock:
            self._selfie_seg_grow_px = max(0, int(grow_px))
            self._selfie_seg_feather_px = max(0, int(feather_px))
            self._selfie_seg_blur_px = max(0, int(blur_px))
            self._selfie_seg_temporal = max(0.0, min(0.95, float(temporal)))

    def _get_segmenter(self):
        """Lazily create the MediaPipe segmenter. Returns None if unavailable.

        Called only from this capture thread. A load/init failure is
        remembered so we don't retry (and re-log) on every single frame.
        """
        if self._segmenter is not None or self._segmenter_failed:
            return self._segmenter
        if not _MEDIAPIPE:
            log.warning('WebcamSystem: selfie_seg requested but mediapipe is not installed — '
                        'see drop-ins/webcam-01/requirements.txt')
            self._segmenter_failed = True
            return None
        if not _SELFIE_SEG_MODEL_PATH.is_file():
            log.warning('WebcamSystem: selfie_seg model missing at %s', _SELFIE_SEG_MODEL_PATH)
            self._segmenter_failed = True
            return None
        try:
            options = _mp_vision.ImageSegmenterOptions(
                base_options=_MpBaseOptions(model_asset_path=str(_SELFIE_SEG_MODEL_PATH)),
                output_category_mask=False,
                output_confidence_masks=True,
            )
            self._segmenter = _mp_vision.ImageSegmenter.create_from_options(options)
        except Exception as exc:
            log.warning('WebcamSystem: failed to load selfie_seg model: %s', exc)
            self._segmenter_failed = True
            return None
        return self._segmenter

    def _apply_selfie_seg(self, frame: np.ndarray) -> np.ndarray:
        """Cut MediaPipe's idea of the background: dim it, blur it, or cut it to alpha.

        Returns the frame unchanged on any failure -- a bad frame from a
        broken segmenter is a much worse experience than a missed effect.
        """
        segmenter = self._get_segmenter()
        if segmenter is None:
            return frame
        try:
            mp_image = _mp.Image(image_format=_mp.ImageFormat.SRGB, data=frame)
            result = segmenter.segment(mp_image)
            raw_mask = result.confidence_masks[0].numpy_view()  # (H, W) or (H, W, 1) float32, 0..1
            h, w = frame.shape[0], frame.shape[1]
            raw_mask = np.clip(raw_mask, 0.0, 1.0).reshape(h, w)
            # Temporal smoothing: blend toward the previous frame's raw
            # mask before grow+feather runs, so the boundary itself moves
            # smoothly across frames instead of re-deriving from scratch
            # every frame (the model has no notion of "last frame").
            t = self._selfie_seg_temporal
            if t > 0.0 and self._prev_raw_mask is not None and self._prev_raw_mask.shape == raw_mask.shape:
                raw_mask = t * self._prev_raw_mask + (1.0 - t) * raw_mask
            self._prev_raw_mask = raw_mask
            alpha = _feather_mask(raw_mask, self._selfie_seg_grow_px, self._selfie_seg_feather_px)
            if self._selfie_seg_mode == 'transparent':
                return np.dstack((frame, (alpha * 255.0).astype(np.uint8)))
            alpha3 = alpha.reshape(h, w, 1)
            if self._selfie_seg_mode == 'blur':
                blur_px = self._selfie_seg_blur_px
                if blur_px <= 0:
                    return frame
                k = 2 * blur_px + 1
                blurred = cv2.GaussianBlur(frame, (k, k), 0)
                out = frame.astype(np.float32) * alpha3 + blurred.astype(np.float32) * (1.0 - alpha3)
                return np.clip(out, 0.0, 255.0).astype(np.uint8)
            scale = alpha3 + (1.0 - alpha3) * _SELFIE_SEG_BG_DIM
            return np.clip(frame.astype(np.float32) * scale, 0.0, 255.0).astype(np.uint8)
        except Exception as exc:
            log.debug('WebcamSystem: selfie_seg inference failed on a frame: %s', exc)
            return frame

    @property
    def sequence(self) -> int:
        """Monotonic count of captured frames; compare to skip stale work."""
        with self._lock:
            return self._seq

    def latest(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def latest_if_newer(self, seen_seq: int) -> tuple[np.ndarray | None, int]:
        """Return ``(frame, seq)`` only when ``seq`` has moved past *seen_seq*.

        Returns ``(None, seen_seq)`` when nothing new has arrived, so the
        caller can skip the copy, the ``tobytes()`` and the GPU upload
        entirely. One lock acquisition either way.
        """
        with self._lock:
            if self._latest is None or self._seq == seen_seq:
                return None, self._seq
            return self._latest.copy(), self._seq

    @property
    def active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# WebcamSystem
# ---------------------------------------------------------------------------

class WebcamSystem:
    """Always-on camera overlay.  Not a playlist effect."""

    HELP_ENTRIES: list[tuple[str, str, str]] = [
        ('Camera Overlay', 'Ctrl+KP 1-9', 'PiP position'),
        ('Camera Overlay', 'Ctrl+KP 0 / .', 'PiP fullscreen / hide'),
        ('Camera Overlay', 'Ctrl+KP / *', 'Treatment prev / next'),
        ('Camera Overlay', 'Ctrl+KP - +', 'PiP size'),
        ('Camera Overlay', 'Ctrl+KP Enter', 'Treatment auto-cycle'),
        ('Camera Overlay', 'PageUp / PageDown', 'Camera device prev / next'),
        ('Camera Overlay', 'Ctrl+Alt+K', 'Webcam editor modal'),
    ]

    _LAYOUTS: dict[str, tuple[float, float]] = {
        'top_left':      (0.04, 0.04),
        'top_center':    (0.50, 0.04),
        'top_right':     (0.96, 0.04),
        'left':          (0.04, 0.50),
        'center':        (0.50, 0.50),
        'right':         (0.96, 0.50),
        'bottom_left':   (0.04, 0.96),
        'bottom_center': (0.50, 0.96),
        'bottom_right':  (0.96, 0.96),
    }

    _KEYPAD_TO_LAYOUT: dict[str, str] = {
        '7': 'top_left',    '8': 'top_center',    '9': 'top_right',
        '4': 'left',        '5': 'center',         '6': 'right',
        '1': 'bottom_left', '2': 'bottom_center',  '3': 'bottom_right',
        '0': 'fullscreen',  '.': 'hide',
    }

    def __init__(self, ctx: moderngl.Context, width: int, height: int, cfg: dict) -> None:
        self._ctx = ctx
        self._width = width
        self._height = height

        self.enabled: bool = str(cfg.get('enabled', 'true')).lower() != 'false'

        self._device     = int(cfg.get('device', 0))
        self._cam_width  = int(cfg.get('width',  1280))
        self._cam_height = int(cfg.get('height', 720))
        self._cam_fps    = int(cfg.get('fps',    30))
        self._pip_scale  = float(cfg.get('pip_scale', 0.33))
        self._border_thickness = max(0.0, float(cfg.get('border_thickness', 8.0)))
        self._border_color = self._parse_border_color(
            cfg.get('border_color', (0.04, 0.72, 1.0, 0.28))
        )
        # Border look: 'off' | 'classic' (original flat ring) | 'futuristic'
        # (glowing neon/HUD/drippy shader border, see _BORDER_FRAG) --
        # default is the new futuristic look. `border_intensity` scales the
        # whole effect's glow/drip strength; `border_hue_speed` is how many
        # full colour turns per second the neon hue drifts through, seeded
        # from `border_color`'s own hue so the look still starts from the
        # configured tint even though it's no longer static.
        self._border_style = self._parse_border_style(cfg.get('border_style', 'futuristic'))
        self._border_intensity = max(0.0, min(2.0, float(cfg.get('border_intensity', 1.0))))
        self._border_hue_speed = max(0.0, min(1.0, float(cfg.get('border_hue_speed', 0.05))))
        self._border_hue_base = self._hue_from_color(self._border_color)
        self._layout     = 'bottom_right'
        self._hidden     = False
        self._switch_hide_active = False
        self._switch_hide_deadline = 0.0
        self._switch_hide_duration_s = max(0.0, float(cfg.get('switch_hide_duration_s', 1.2)))
        # Resolve the configured position WITHOUT touching the capture
        # worker: __init__ has not created it yet, and start() is the
        # lifecycle method that opens the camera. Calling the full
        # set_layout() here reached _ensure_worker() before _worker and
        # _available_cameras existed, raising AttributeError out of the
        # constructor -- which the core's guarded loader swallowed, silently
        # disabling the whole drop-in (regression in 1.0.0-rc.2).
        self._layout, self._hidden = self._resolve_layout_token(
            str(cfg.get('pip_position', 'bottom_right')).lower(), self._layout)

        default_treatment = str(cfg.get('treatment', 'neon')).lower()
        if default_treatment not in _TREATMENT_NAMES:
            default_treatment = 'neon'
        self._treatment_index = _TREATMENT_NAMES.index(default_treatment)

        self._auto_cycle = False
        self._auto_cycle_interval = float(cfg.get('cycle_interval', 30.0)) or 30.0
        self._auto_cycle_timer = 0.0
        # Auto-cycle alternates raw (normal) <-> the next real effect treatment,
        # rather than round-robining flatly through every treatment including
        # raw: normal -> effect1 -> normal -> effect2 -> ... An effect airs for
        # half of cycle_interval; the normal interlude between effects airs for
        # twice that (i.e. the un-halved cycle_interval), so effects read as
        # brief highlights rather than eating equal screen time to plain video.
        self._auto_cycle_effect_duration = self._auto_cycle_interval / 2.0
        self._auto_cycle_normal_duration = self._auto_cycle_interval
        self._auto_cycle_showing_effect = False
        # Tracked separately from _treatment_index: that field is 'raw'
        # throughout the whole normal phase, so it can't tell us which
        # effect played last once we're back to raw between effects.
        self._auto_cycle_effect_seq_index = -1

        self._available_cameras: list[int] = []
        self._detected_cameras: list[int] = []
        self._camera_list_index: int = 0
        self._disabled_cameras: set[int] = set()
        self._active_device: int | None = None
        self._camera_settings: dict[int, dict[str, float | bool]] = {}
        self._brightness = max(0.2, min(2.0, float(cfg.get('brightness', 1.0))))
        self._contrast = max(0.2, min(2.0, float(cfg.get('contrast', 1.0))))
        self._flip_horizontal = str(cfg.get('flip_horizontal', 'true')).lower() != 'false'
        self._flip_vertical = str(cfg.get('flip_vertical', 'false')).lower() == 'true'
        # Selfie-segmentation background cutout: 'off' | 'dim' | 'blur' |
        # 'transparent', per-camera, off by default. Applied before every
        # treatment shader (raw/neon/glitch/vintage/thermal) -- see
        # _CameraWorker._apply_selfie_seg. Mode is configured per-camera
        # from the Config Editor (Visuals tab), not here.
        # _parse_selfie_seg_mode also accepts the legacy true/false spelling
        # for anyone with an old `selfie_seg = false` line in config.toml.
        self._selfie_seg = self._parse_selfie_seg_mode(cfg.get('selfie_seg', 'off'))
        # Matte tuning: global (not per-camera), live-adjustable from the
        # Config Editor. Optional config.toml overrides for the initial
        # values only -- see _SELFIE_SEG_*_DEFAULT for what these mean.
        self._selfie_seg_grow_px = max(0, int(cfg.get('selfie_seg_grow_px', _SELFIE_SEG_GROW_PX_DEFAULT)))
        self._selfie_seg_feather_px = max(0, int(cfg.get('selfie_seg_feather_px', _SELFIE_SEG_FEATHER_PX_DEFAULT)))
        self._selfie_seg_blur_px = max(0, int(cfg.get('selfie_seg_blur_px', _SELFIE_SEG_BLUR_PX_DEFAULT)))
        self._selfie_seg_temporal = max(0.0, min(
            0.95, float(cfg.get('selfie_seg_temporal', _SELFIE_SEG_TEMPORAL_DEFAULT))))
        self._default_brightness = float(self._brightness)
        self._default_contrast = float(self._contrast)
        self._default_flip_horizontal = bool(self._flip_horizontal)
        self._default_flip_vertical = bool(self._flip_vertical)
        self._default_selfie_seg = str(self._selfie_seg)
        self._remember_current_camera_settings(int(self._device))

        self._programs: dict[str, moderngl.Program] = {}
        for name, frag in _TREATMENT_FRAGS.items():
            self._programs[name] = ctx.program(
                vertex_shader=_CAM_VERT,
                fragment_shader=frag,
            )

        self._cam_vbo = ctx.buffer(reserve=6 * 4 * 4)
        self._cam_vao_map: dict[str, moderngl.VertexArray] = {}
        for name, prog in self._programs.items():
            self._cam_vao_map[name] = ctx.vertex_array(
                prog, [(self._cam_vbo, '2f 2f', 'in_pos', 'in_uv')]
            )

        self._border_prog = ctx.program(vertex_shader=_CAM_VERT, fragment_shader=_BORDER_FRAG)
        self._border_vbo = ctx.buffer(reserve=6 * 4 * 4)
        self._border_vao = ctx.vertex_array(
            self._border_prog, [(self._border_vbo, '2f 2f', 'in_pos', 'in_uv')]
        )

        fb = np.zeros((256, 256, 3), dtype=np.uint8)
        yy, xx = np.indices((256, 256))
        fb[..., 0] = ((xx * 2 + yy) % 255).astype(np.uint8)
        fb[..., 1] = ((yy * 2) % 255).astype(np.uint8)
        fb[..., 2] = 100
        self._fallback_tex = ctx.texture((256, 256), 3, fb.tobytes())
        self._fallback_tex.filter = moderngl.LINEAR, moderngl.LINEAR

        self._cam_tex: moderngl.Texture | None = None
        self._cam_tex_size = (0, 0)
        self._cam_tex_channels = 3
        # Capture sequence already uploaded; -1 forces the first upload.
        self._cam_seq: int = -1
        self._worker: _CameraWorker | None = None
        self._time = 0.0
        self._vj_api: Any = None

    def set_vj_api(self, api: Any) -> None:
        """Inject the VJApi instance after construction."""
        self._vj_api = api
        try:
            api.register_midi_actions('Webcam', [
                ('webcam_toggle',       'Toggle webcam overlay'),
                ('webcam_next_camera',  'Next camera device'),
                ('webcam_prev_camera',  'Previous camera device'),
                ('webcam_auto_cycle',   'Toggle webcam auto-cycle'),
                ('webcam_next_effect',  'Next webcam effect'),
                ('webcam_prev_effect',  'Prev webcam effect'),
                ('webcam_flip_h',       'Flip webcam horizontal'),
                ('webcam_flip_v',       'Flip webcam vertical'),
            ])
            api.register_midi_action_handler('webcam_toggle',      self._midi_toggle)
            api.register_midi_action_handler('webcam_next_camera', self._midi_next_camera)
            api.register_midi_action_handler('webcam_prev_camera', self._midi_prev_camera)
            api.register_midi_action_handler('webcam_auto_cycle',  self.toggle_auto_cycle)
            api.register_midi_action_handler('webcam_next_effect', self._midi_next_effect)
            api.register_midi_action_handler('webcam_prev_effect', self._midi_prev_effect)
            api.register_midi_action_handler('webcam_flip_h',      self.toggle_flip_horizontal)
            api.register_midi_action_handler('webcam_flip_v',      self.toggle_flip_vertical)
        except Exception as exc:
            import logging  # noqa: PLC0415
            logging.getLogger(__name__).debug('webcam-01: MIDI registration failed: %s', exc)

    def _midi_toggle(self) -> None:
        """MIDI handler: toggle webcam overlay on/off."""
        self.enabled = not self.enabled

    def _midi_next_camera(self) -> None:
        """MIDI handler: advance to next camera device."""
        if self._vj_api is not None:
            self._vj_api.goto_next_camera()
        else:
            self.next_camera()

    def _midi_prev_camera(self) -> None:
        """MIDI handler: retreat to previous camera device."""
        if self._vj_api is not None:
            self._vj_api.goto_prev_camera()
        else:
            self.prev_camera()

    def _midi_next_effect(self) -> None:
        """MIDI handler: advance to next webcam visual effect."""
        if self._vj_api is not None:
            self._vj_api.goto_next_webcam_effect()

    def _midi_prev_effect(self) -> None:
        """MIDI handler: retreat to previous webcam visual effect."""
        if self._vj_api is not None:
            self._vj_api.goto_prev_webcam_effect()

    def handle_key(self, sym: int, mod: int) -> 'str | None | bool':
        """Handle webcam numpad layout + Alt+[/] PiP sizing keys."""
        import sdl2  # noqa: PLC0415

        # Alt+[ / Alt+] — PiP size down/up.
        if mod & sdl2.KMOD_ALT:
            if sym == sdl2.SDLK_LEFTBRACKET:
                val: float = (self._vj_api.scale_pip(-0.05) if self._vj_api is not None
                              else self.scale_pip(-0.05))
                return f'Camera PiP: {val:.0%}' if val > 0 else 'Camera PiP not available'
            if sym == sdl2.SDLK_RIGHTBRACKET:
                val = (self._vj_api.scale_pip(0.05) if self._vj_api is not None
                       else self.scale_pip(0.05))
                return f'Camera PiP: {val:.0%}' if val > 0 else 'Camera PiP not available'

        # PgUp / PgDown — switch camera device.
        if sym == sdl2.SDLK_PAGEUP:
            label = self._vj_api.goto_prev_camera() if self._vj_api else self.prev_camera()
            return f'Camera device: {label}' if label else 'Camera: only one device'
        if sym == sdl2.SDLK_PAGEDOWN:
            label = self._vj_api.goto_next_camera() if self._vj_api else self.next_camera()
            return f'Camera device: {label}' if label else 'Camera: only one device'

        # Ctrl+Numpad — layout positions, treatment cycling, PiP, auto-cycle.
        if not (mod & sdl2.KMOD_CTRL):
            return False

        layout_map = {
            sdl2.SDLK_KP_1: '1', sdl2.SDLK_KP_2: '2', sdl2.SDLK_KP_3: '3',
            sdl2.SDLK_KP_4: '4', sdl2.SDLK_KP_5: '5', sdl2.SDLK_KP_6: '6',
            sdl2.SDLK_KP_7: '7', sdl2.SDLK_KP_8: '8', sdl2.SDLK_KP_9: '9',
        }
        labels = {
            '1': 'bottom-left', '2': 'bottom-center', '3': 'bottom-right',
            '4': 'left', '5': 'center', '6': 'right',
            '7': 'top-left', '8': 'top-center', '9': 'top-right',
        }
        if sym in layout_map:
            token = layout_map[sym]
            if self._vj_api is not None:
                if self._vj_api.set_camera_layout(token):
                    return f'Camera: {labels[token]}'
            elif self.set_layout(token):
                return f'Camera: {labels[token]}'
            return 'Camera: not available'
        if sym == sdl2.SDLK_KP_0:
            if self._vj_api is not None:
                success = self._vj_api.set_camera_layout('0')
            else:
                success = bool(self.set_layout('0'))
            return 'Camera: fullscreen' if success else 'Camera: not available'
        if sym == sdl2.SDLK_KP_PERIOD:
            if self._vj_api is not None:
                success = self._vj_api.set_camera_layout('.')
            else:
                self.set_layout('hide')
                success = True
            return 'Camera: hidden' if success else 'Camera: not available'
        if sym == sdl2.SDLK_KP_DIVIDE:
            name = self._vj_api.goto_prev_webcam_effect() if self._vj_api else self.prev_treatment()
            return f'Camera treatment: {name}' if name else 'Camera: not available'
        if sym == sdl2.SDLK_KP_MULTIPLY:
            name = self._vj_api.goto_next_webcam_effect() if self._vj_api else self.next_treatment()
            return f'Camera treatment: {name}' if name else 'Camera: not available'
        if sym == sdl2.SDLK_KP_MINUS:
            pip_val = self._vj_api.scale_pip(-0.05) if self._vj_api else self.scale_pip(-0.05)
            return f'Camera PiP: {pip_val:.0%}' if pip_val > 0 else 'Camera: not available'
        if sym == sdl2.SDLK_KP_PLUS:
            pip_val = self._vj_api.scale_pip(0.05) if self._vj_api else self.scale_pip(0.05)
            return f'Camera PiP: {pip_val:.0%}' if pip_val > 0 else 'Camera: not available'
        if sym == sdl2.SDLK_KP_ENTER:
            active = self._vj_api.toggle_webcam_auto_cycle() if self._vj_api else self.toggle_auto_cycle()
            return f'Camera treatment auto-cycle: {"ON" if active else "OFF"}'

        return False

    def start(self) -> None:
        if not self.enabled:
            return
        if not _CV2:
            log.error('WebcamSystem: opencv-python-headless not installed')
            return
        # Privacy: never open (or even probe) the camera while the overlay is
        # hidden — the shipped default is pip_position = "hidden", which used
        # to light the webcam LED at first boot and capture at 30 fps into a
        # frame nobody rendered. The worker starts lazily on first show.
        if self._hidden:
            log.info('WebcamSystem: started hidden — camera stays off until shown')
            return
        self.rediscover_cameras()
        self._restart_worker()
        log.info('WebcamSystem: started (treatment=%s, layout=%s, cameras=%s)',
                 self.current_treatment, self._layout, self._available_cameras)

    def resize(self, w: int, h: int) -> None:
        self._width = w
        self._height = h

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _resolve_layout_token(self, token: str,
                              current: str) -> tuple[str, bool]:
        """Map a layout token to ``(layout, hidden)``. Pure -- no side effects.

        Split out so ``__init__`` can resolve the configured position
        without starting capture; ``set_layout()`` adds the worker
        lifecycle on top.
        """
        token = token.lower().strip()
        if token == 'hidden':
            token = 'hide'
        token = self._KEYPAD_TO_LAYOUT.get(token, token)
        if token == 'hide':
            return current, True
        if token == 'fullscreen' or token in self._LAYOUTS:
            return token, False
        return current, False

    def set_layout(self, token: str) -> str:
        layout, hidden = self._resolve_layout_token(token, self._layout)
        self._layout = layout
        self._hidden = hidden
        if hidden:
            # Privacy: a deliberate hide releases the camera entirely (LED
            # off), unlike the transient camera-switch shim which keeps the
            # worker alive across a device change.
            self._stop_worker()
            return 'hide'
        self._ensure_worker()
        return self._layout

    def _invalidate_camera_frame(self) -> None:
        """Force the next render to re-upload (device switch / restart)."""
        self._cam_seq = -1

    def _stop_worker(self) -> None:
        """Stop and release the capture worker (camera LED goes off)."""
        if self._worker is not None:
            try:
                self._worker.stop()
                self._worker.join(timeout=2.0)
            except Exception:
                pass
            self._worker = None

    def _ensure_worker(self) -> None:
        """Lazily start capture on first show (no camera while hidden)."""
        if self._hidden or not self.enabled or not _CV2:
            return
        if not self._available_cameras:
            self.rediscover_cameras()
        if self._worker is None or not self._worker.is_alive():
            self._restart_worker()

    def _active_or_default_device(self) -> int:
        if self._active_device is not None:
            return int(self._active_device)
        return int(self._device)

    @staticmethod
    def _parse_selfie_seg_mode(value: Any) -> str:
        """Accept a mode name, or the legacy true/false spelling (pre-modes,
        'true' only ever meant the one mode that existed then: dim)."""
        text = str(value).strip().lower()
        if text in _SELFIE_SEG_MODES:
            return text
        if text == 'true':
            return 'dim'
        return 'off'

    def _remember_current_camera_settings(self, camera_id: int | None = None) -> None:
        target_id = self._active_or_default_device() if camera_id is None else int(camera_id)
        self._camera_settings[target_id] = {
            'brightness': float(self._brightness),
            'contrast': float(self._contrast),
            'flip_horizontal': bool(self._flip_horizontal),
            'flip_vertical': bool(self._flip_vertical),
            'selfie_seg': str(self._selfie_seg),
        }

    def _apply_camera_settings(self, camera_id: int) -> None:
        settings = self._camera_settings.get(int(camera_id))
        if not isinstance(settings, dict):
            self._brightness = float(self._default_brightness)
            self._contrast = float(self._default_contrast)
            self._flip_horizontal = bool(self._default_flip_horizontal)
            self._flip_vertical = bool(self._default_flip_vertical)
            self._selfie_seg = str(self._default_selfie_seg)
            return
        self._brightness = max(0.2, min(2.0, float(settings.get('brightness', self._brightness))))
        self._contrast = max(0.2, min(2.0, float(settings.get('contrast', self._contrast))))
        self._flip_horizontal = bool(settings.get('flip_horizontal', self._flip_horizontal))
        self._flip_vertical = bool(settings.get('flip_vertical', self._flip_vertical))
        self._selfie_seg = self._parse_selfie_seg_mode(settings.get('selfie_seg', self._selfie_seg))

    def scale_pip(self, delta: float) -> float:
        self._pip_scale = max(0.12, min(0.80, self._pip_scale + delta))
        return self._pip_scale

    def _parse_border_color(self, value) -> tuple[float, float, float, float]:
        if isinstance(value, (tuple, list)):
            parts = list(value)
        elif isinstance(value, str):
            text = value.strip()
            if text.startswith('#') and len(text) in (7, 9):
                r = int(text[1:3], 16) / 255.0
                g = int(text[3:5], 16) / 255.0
                b = int(text[5:7], 16) / 255.0
                a = int(text[7:9], 16) / 255.0 if len(text) == 9 else 1.0
                return (r, g, b, a)
            parts = [p.strip() for p in text.split(',') if p.strip()]
        else:
            parts = []

        floats = [float(v) for v in parts[:4]]
        if len(floats) == 3:
            floats.append(1.0)
        while len(floats) < 4:
            floats.append(0.0 if len(floats) < 3 else 1.0)
        return tuple(max(0.0, min(1.0, v)) for v in floats[:4])

    @staticmethod
    def _parse_border_style(value: Any) -> str:
        """Accept a border style name; anything unrecognized falls back to
        'futuristic' (the redesigned default look), not 'off'/'classic' --
        an unrecognized config value should not silently disable the
        border, it should fall back to the intended new default."""
        text = str(value).strip().lower()
        return text if text in _BORDER_STYLES else 'futuristic'

    @staticmethod
    def _hue_from_color(color: tuple[float, float, float, float]) -> float:
        """Extract the hue (0-1) from an RGBA border colour -- the
        futuristic border's starting hue before `border_hue_speed` drifts
        it over time."""
        hue, _saturation, _value = colorsys.rgb_to_hsv(color[0], color[1], color[2])
        return float(hue)

    def next_treatment(self) -> str:
        self._treatment_index = (self._treatment_index + 1) % len(_TREATMENT_NAMES)
        return _TREATMENT_NAMES[self._treatment_index]

    def prev_treatment(self) -> str:
        self._treatment_index = (self._treatment_index - 1) % len(_TREATMENT_NAMES)
        return _TREATMENT_NAMES[self._treatment_index]

    def _next_cycle_effect_treatment(self) -> str:
        """Advance to the next non-raw treatment, for auto-cycle's effect phase.

        Manual next/prev_treatment() still round-robin flatly through every
        treatment including raw; only auto-cycle skips raw here since raw is
        auto-cycle's own separate "normal" phase, not one of the effects it
        rotates through. Uses its own sequence index rather than deriving
        "current" from _treatment_index, which reads 'raw' throughout the
        whole normal phase and so can't say which effect played last.
        """
        effects = [t for t in _TREATMENT_NAMES if t != 'raw']
        self._auto_cycle_effect_seq_index = (self._auto_cycle_effect_seq_index + 1) % len(effects)
        next_name = effects[self._auto_cycle_effect_seq_index]
        self._treatment_index = _TREATMENT_NAMES.index(next_name)
        return next_name

    def toggle_auto_cycle(self) -> bool:
        self._auto_cycle = not self._auto_cycle
        self._auto_cycle_timer = 0.0
        self._auto_cycle_showing_effect = False
        if self._auto_cycle:
            self._treatment_index = _TREATMENT_NAMES.index('raw')
        return self._auto_cycle

    def _advance_auto_cycle(self, dt: float) -> None:
        """Tick the normal/effect alternation; no-op unless auto-cycle is on.

        Pattern: normal -> effect -> normal -> effect -> ... An effect airs
        for _auto_cycle_effect_duration; the normal interlude between
        effects airs for _auto_cycle_normal_duration (2x the effect
        duration). Split out from render() so it's unit-testable without a
        camera/GL context.
        """
        if not self._auto_cycle:
            return
        self._auto_cycle_timer += dt
        duration = (
            self._auto_cycle_effect_duration if self._auto_cycle_showing_effect
            else self._auto_cycle_normal_duration
        )
        if self._auto_cycle_timer < duration:
            return
        self._auto_cycle_timer = 0.0
        if self._auto_cycle_showing_effect:
            self._treatment_index = _TREATMENT_NAMES.index('raw')
            self._auto_cycle_showing_effect = False
        else:
            self._next_cycle_effect_treatment()
            self._auto_cycle_showing_effect = True

    @staticmethod
    def _enumerate_cameras() -> list[int]:
        """Probe available video capture devices and return their indices."""
        if not _CV2:
            return []
        found: list[int] = []
        if sys.platform != 'win32':
            import glob as _glob
            for p in sorted(_glob.glob('/dev/video*')):
                try:
                    idx = int(Path(p).name.replace('video', ''))
                except ValueError:
                    continue
                cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
                if cap.isOpened():
                    found.append(idx)
                cap.release()
        else:
            for i in range(8):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    found.append(i)
                cap.release()
        return found

    def _apply_camera_filter(self, detected: list[int]) -> list[int]:
        return [idx for idx in detected if idx not in self._disabled_cameras]

    def _rebuild_camera_list(self, detected: list[int]) -> list[int]:
        self._detected_cameras = list(detected)
        current_device = self._active_device
        self._available_cameras = self._apply_camera_filter(detected)
        if not self._available_cameras:
            self._camera_list_index = 0
            return self._available_cameras

        preferred = current_device if current_device is not None else self._device
        if preferred in self._available_cameras:
            self._camera_list_index = self._available_cameras.index(preferred)
        else:
            self._camera_list_index = 0
        return self._available_cameras

    def list_cameras(self) -> list[dict[str, Any]]:
        """Return detected camera list with enable/disable and selection state."""
        detected = self._detected_cameras
        selected = self._active_device
        if selected is None and self._available_cameras:
            selected = self._available_cameras[self._camera_list_index]
        return [
            {
                'id': int(idx),
                'label': f'/dev/video{idx}' if sys.platform != 'win32' else str(idx),
                'enabled': idx not in self._disabled_cameras,
                'selected': idx == selected,
            }
            for idx in detected
        ]

    def rediscover_cameras(self) -> list[int]:
        """Re-probe camera devices and refresh enabled/selected state."""
        detected = self._enumerate_cameras()
        return self._rebuild_camera_list(detected)

    def set_camera_enabled(self, camera_id: int, enabled: bool) -> bool:
        """Enable or disable a detected camera index; returns success."""
        camera_id = int(camera_id)
        detected = self._detected_cameras
        if camera_id not in detected:
            return False
        if enabled:
            self._disabled_cameras.discard(camera_id)
        else:
            self._disabled_cameras.add(camera_id)
        self._rebuild_camera_list(detected)
        if self._worker is not None:
            if not self._available_cameras:
                self._worker.stop()
                self._worker.join(timeout=2.0)
                self._worker = None
                self._active_device = None
            else:
                self._restart_worker()
        return True

    def set_active_camera(self, camera_id: int) -> str | None:
        """Switch directly to a specific enabled camera id and return its label."""
        camera_id = int(camera_id)
        self._rebuild_camera_list(self._detected_cameras)
        if camera_id not in self._available_cameras:
            return None
        self._prepare_camera_switch_hide()
        self._camera_list_index = self._available_cameras.index(camera_id)
        self._restart_worker()
        return f'/dev/video{camera_id}' if sys.platform != 'win32' else str(camera_id)

    def set_brightness(self, value: float) -> float:
        """Set software brightness multiplier and return clamped value."""
        self._brightness = max(0.2, min(2.0, float(value)))
        self._remember_current_camera_settings()
        return self._brightness

    def set_contrast(self, value: float) -> float:
        """Set software contrast multiplier and return clamped value."""
        self._contrast = max(0.2, min(2.0, float(value)))
        self._remember_current_camera_settings()
        return self._contrast

    def adjust_brightness(self, delta: float) -> float:
        """Adjust brightness by delta and return updated value."""
        return self.set_brightness(self._brightness + float(delta))

    def adjust_contrast(self, delta: float) -> float:
        """Adjust contrast by delta and return updated value."""
        return self.set_contrast(self._contrast + float(delta))

    def _prepare_camera_switch_hide(self) -> None:
        """Hide PiP during camera switch lag, then restore automatically."""
        if self._hidden:
            return
        self._hidden = True
        self._switch_hide_active = True
        self._switch_hide_deadline = time.monotonic() + self._switch_hide_duration_s

    def set_flip_horizontal(self, enabled: bool) -> bool:
        """Enable or disable horizontal mirror flip on webcam frames."""
        self._flip_horizontal = bool(enabled)
        self._remember_current_camera_settings()
        return self._flip_horizontal

    def set_flip_vertical(self, enabled: bool) -> bool:
        """Enable or disable vertical flip on webcam frames."""
        self._flip_vertical = bool(enabled)
        self._remember_current_camera_settings()
        return self._flip_vertical

    def toggle_flip_horizontal(self) -> bool:
        """Toggle horizontal mirror flip on webcam frames."""
        self._flip_horizontal = not self._flip_horizontal
        self._remember_current_camera_settings()
        return self._flip_horizontal

    def toggle_flip_vertical(self) -> bool:
        """Toggle vertical flip on webcam frames."""
        self._flip_vertical = not self._flip_vertical
        self._remember_current_camera_settings()
        return self._flip_vertical

    def set_camera_selfie_seg(self, device: int, mode: str) -> None:
        """Set the selfie-seg background mode for one camera (any camera, not just active).

        ``mode`` is one of _SELFIE_SEG_MODES ('off' | 'dim' | 'blur' |
        'transparent'); an unrecognized value falls back to 'off'. Called
        from the Config Editor's per-camera row. Updates the persisted
        per-camera setting; if `device` is the currently active camera,
        also flips the live instance mode and pushes it to the running
        capture worker so the change is visible immediately.
        """
        device = int(device)
        mode = mode if mode in _SELFIE_SEG_MODES else 'off'
        settings = self._camera_settings.setdefault(device, {})
        settings['selfie_seg'] = mode
        if device == self._active_or_default_device():
            self._selfie_seg = mode
            if self._worker is not None:
                self._worker.set_selfie_seg_mode(self._selfie_seg)

    def _push_selfie_seg_tuning(self) -> None:
        """Push the current grow/feather/blur/temporal values to the live capture worker."""
        if self._worker is not None:
            self._worker.set_selfie_seg_tuning(
                self._selfie_seg_grow_px, self._selfie_seg_feather_px,
                self._selfie_seg_blur_px, self._selfie_seg_temporal)

    # -- Configuration-editor convention -------------------------------------
    # One 0/1/2/3 selector per detected camera: selfie-segmentation
    # background mode (off/dim/blur/transparent). The generic Config Editor
    # is slider-only (no native checkbox or dropdown widget), so the mode
    # index rides a stepped slider by the same convention used elsewhere in
    # the app (e.g. App._set_audio_latency_index). Four more rows tune the
    # grow/feather/blur/temporal-smoothing values -- global (not
    # per-camera), since they shape how the matte looks and behaves rather
    # than which camera wants segmentation.
    CONFIG_EDITOR_CATEGORY = 'Visuals'
    _SELFIE_SEG_SETTING_PREFIX = 'cam'
    _SELFIE_SEG_SETTING_SUFFIX = '_selfie_seg'
    _SELFIE_SEG_GROW_SETTING = 'selfie_seg_grow_px'
    _SELFIE_SEG_FEATHER_SETTING = 'selfie_seg_feather_px'
    _SELFIE_SEG_BLUR_SETTING = 'selfie_seg_blur_px'
    _SELFIE_SEG_TEMPORAL_SETTING = 'selfie_seg_temporal'
    # Border look: same "no native dropdown, ride a stepped slider"
    # convention as the selfie-seg mode selector above. Style is a 0-2
    # stepped slider (off/classic/futuristic); intensity and hue-cycle
    # speed are continuous global sliders.
    _BORDER_STYLE_SETTING = 'border_style'
    _BORDER_INTENSITY_SETTING = 'border_intensity'
    _BORDER_HUE_SPEED_SETTING = 'border_hue_speed'

    def config_editor_settings(self) -> list[dict[str, float | str]]:
        # Union _available_cameras with anything already known in memory
        # (persisted per-camera settings, the configured default device) --
        # NOT a fresh probe. The shipped default is pip_position = "hidden",
        # under which start() deliberately never calls rediscover_cameras()
        # (camera privacy: no probing while hidden), so _available_cameras
        # alone stays empty and this tab would otherwise show zero rows
        # until the operator manually shows the camera once.
        devices = set(self._available_cameras) | set(self._camera_settings.keys())
        devices.add(self._active_or_default_device())
        settings: list[dict[str, float | str]] = []
        for dev in sorted(devices):
            mode = self._camera_settings.get(dev, {}).get('selfie_seg', 'off')
            mode = mode if mode in _SELFIE_SEG_MODES else 'off'
            settings.append({
                'name': f'{self._SELFIE_SEG_SETTING_PREFIX}{dev}{self._SELFIE_SEG_SETTING_SUFFIX}',
                'value': float(_SELFIE_SEG_MODES.index(mode)),
                'min': 0.0,
                'max': float(len(_SELFIE_SEG_MODES) - 1),
                # 0=off 1=dim 2=blur 3=transparent: one editor notch must
                # land on an exact integer, or set_config_setting()'s
                # round() would read a value it never actually stepped
                # through.
                'step': 1.0,
            })
        settings.append({
            'name': self._SELFIE_SEG_GROW_SETTING, 'value': float(self._selfie_seg_grow_px),
            'min': 0.0, 'max': 20.0, 'step': 1.0,
        })
        settings.append({
            'name': self._SELFIE_SEG_FEATHER_SETTING, 'value': float(self._selfie_seg_feather_px),
            'min': 0.0, 'max': 30.0, 'step': 1.0,
        })
        settings.append({
            'name': self._SELFIE_SEG_BLUR_SETTING, 'value': float(self._selfie_seg_blur_px),
            'min': 0.0, 'max': 100.0, 'step': 1.0,
        })
        settings.append({
            # Continuous 0-0.95 -- how much of the previous frame's raw
            # mask carries forward; higher = smoother but laggier.
            'name': self._SELFIE_SEG_TEMPORAL_SETTING, 'value': float(self._selfie_seg_temporal),
            'min': 0.0, 'max': 0.95,
        })
        settings.append({
            'name': self._BORDER_STYLE_SETTING,
            'value': float(_BORDER_STYLES.index(self._border_style)),
            'min': 0.0, 'max': float(len(_BORDER_STYLES) - 1), 'step': 1.0,
        })
        settings.append({
            'name': self._BORDER_INTENSITY_SETTING, 'value': float(self._border_intensity),
            'min': 0.0, 'max': 2.0,
        })
        settings.append({
            'name': self._BORDER_HUE_SPEED_SETTING, 'value': float(self._border_hue_speed),
            'min': 0.0, 'max': 1.0,
        })
        return settings

    def set_config_setting(self, name: str, value: float) -> None:
        if name == self._SELFIE_SEG_GROW_SETTING:
            self._selfie_seg_grow_px = max(0, int(round(value)))
            self._push_selfie_seg_tuning()
            return
        if name == self._SELFIE_SEG_FEATHER_SETTING:
            self._selfie_seg_feather_px = max(0, int(round(value)))
            self._push_selfie_seg_tuning()
            return
        if name == self._SELFIE_SEG_BLUR_SETTING:
            self._selfie_seg_blur_px = max(0, int(round(value)))
            self._push_selfie_seg_tuning()
            return
        if name == self._SELFIE_SEG_TEMPORAL_SETTING:
            self._selfie_seg_temporal = max(0.0, min(0.95, float(value)))
            self._push_selfie_seg_tuning()
            return
        if name == self._BORDER_STYLE_SETTING:
            index = max(0, min(len(_BORDER_STYLES) - 1, int(round(value))))
            self._border_style = _BORDER_STYLES[index]
            return
        if name == self._BORDER_INTENSITY_SETTING:
            self._border_intensity = max(0.0, min(2.0, float(value)))
            return
        if name == self._BORDER_HUE_SPEED_SETTING:
            self._border_hue_speed = max(0.0, min(1.0, float(value)))
            return
        prefix, suffix = self._SELFIE_SEG_SETTING_PREFIX, self._SELFIE_SEG_SETTING_SUFFIX
        if not (name.startswith(prefix) and name.endswith(suffix)):
            return
        try:
            device = int(name[len(prefix):-len(suffix)])
        except ValueError:
            return
        index = max(0, min(len(_SELFIE_SEG_MODES) - 1, int(round(value))))
        self.set_camera_selfie_seg(device, _SELFIE_SEG_MODES[index])

    def apply_persistence_state(self, payload: dict[str, Any]) -> None:
        """Apply persisted webcam state before startup."""
        if not isinstance(payload, dict):
            return
        selected = payload.get('selected_camera')
        if isinstance(selected, int):
            self._device = int(selected)

        disabled = payload.get('disabled_cameras', [])
        if isinstance(disabled, list):
            self._disabled_cameras = {int(item) for item in disabled if isinstance(item, int)}

        per_camera = payload.get('per_camera', {})
        camera_settings: dict[int, dict[str, float | bool]] = {}
        if isinstance(per_camera, dict):
            for key, value in per_camera.items():
                if not isinstance(value, dict):
                    continue
                try:
                    cam_id = int(key)
                except Exception:
                    continue
                camera_settings[cam_id] = {
                    'brightness': max(0.2, min(2.0, float(value.get('brightness', 1.0)))),
                    'contrast': max(0.2, min(2.0, float(value.get('contrast', 1.0)))),
                    'flip_horizontal': bool(value.get('flip_horizontal', True)),
                    'flip_vertical': bool(value.get('flip_vertical', False)),
                }
        self._camera_settings = camera_settings

        self._layout = str(payload.get('layout', self._layout))
        if self._layout == 'hidden':
            self._layout = 'hide'
        if self._layout in self._KEYPAD_TO_LAYOUT:
            self._layout = self._KEYPAD_TO_LAYOUT[self._layout]
        if self._layout == 'hide':
            self._hidden = True
            self._layout = 'bottom_right'
        elif self._layout != 'fullscreen' and self._layout not in self._LAYOUTS:
            self._layout = 'bottom_right'

        self._pip_scale = max(0.12, min(0.80, float(payload.get('pip_scale', self._pip_scale))))
        treatment = str(payload.get('treatment', self.current_treatment)).lower()
        if treatment in _TREATMENT_NAMES:
            self._treatment_index = _TREATMENT_NAMES.index(treatment)
        self._auto_cycle = bool(payload.get('auto_cycle', self._auto_cycle))

        self._apply_camera_settings(int(self._device))
        self._remember_current_camera_settings(int(self._device))

    def get_persistence_state(self) -> dict[str, Any]:
        """Return webcam settings payload for shared runtime persistence."""
        self._remember_current_camera_settings()
        selected_device = self._active_or_default_device()
        return {
            'selected_camera': int(selected_device),
            'disabled_cameras': sorted(int(idx) for idx in self._disabled_cameras),
            'layout': 'hide' if self._hidden else str(self._layout),
            'pip_scale': float(self._pip_scale),
            'treatment': str(self.current_treatment),
            'auto_cycle': bool(self._auto_cycle),
            'per_camera': {
                str(cam_id): {
                    'brightness': float(values.get('brightness', 1.0)),
                    'contrast': float(values.get('contrast', 1.0)),
                    'flip_horizontal': bool(values.get('flip_horizontal', True)),
                    'flip_vertical': bool(values.get('flip_vertical', False)),
                }
                for cam_id, values in sorted(self._camera_settings.items(), key=lambda item: item[0])
            },
        }

    def webcam_flip_state(self) -> dict[str, bool]:
        """Return current webcam flip state for editor and control-room UI."""
        return {
            'horizontal': self._flip_horizontal,
            'vertical': self._flip_vertical,
        }

    def webcam_image_state(self) -> dict[str, float | bool]:
        """Return webcam image-control state for operator UIs."""
        switching = bool(self._switch_hide_active and time.monotonic() < self._switch_hide_deadline)
        remaining_s = max(0.0, self._switch_hide_deadline - time.monotonic()) if switching else 0.0
        return {
            'brightness': float(self._brightness),
            'contrast': float(self._contrast),
            'flip_horizontal': bool(self._flip_horizontal),
            'flip_vertical': bool(self._flip_vertical),
            'switching': switching,
            'switch_hide_remaining_s': float(remaining_s),
        }

    def _restart_worker(self) -> None:
        """Start capture worker if needed, or request in-thread device switch."""
        # A new/!switched worker restarts its own sequence, so drop the
        # cached one or the first frames of the new camera would be skipped
        # as "already uploaded".
        self._invalidate_camera_frame()
        if not self._available_cameras:
            if self._worker is not None:
                self._worker.stop()
                self._worker.join(timeout=0.25)
                self._worker = None
            self._active_device = None
            return
        device = (
            self._available_cameras[self._camera_list_index]
            if self._available_cameras
            else self._device
        )
        self._active_device = int(device)
        self._apply_camera_settings(self._active_device)
        log.debug(
            'WebcamSystem: apply camera selection idx=%d device=%d available=%s',
            self._camera_list_index,
            self._active_device,
            self._available_cameras,
        )
        if self._worker is None or not self._worker.is_alive():
            self._worker = _CameraWorker(
                device, self._cam_width, self._cam_height, self._cam_fps,
                selfie_seg_mode=self._selfie_seg,
                selfie_seg_grow_px=self._selfie_seg_grow_px,
                selfie_seg_feather_px=self._selfie_seg_feather_px,
                selfie_seg_blur_px=self._selfie_seg_blur_px,
                selfie_seg_temporal=self._selfie_seg_temporal,
            )
            self._worker.start()
            return
        self._worker.request_device(device)
        self._worker.set_selfie_seg_mode(self._selfie_seg)
        self._worker.set_selfie_seg_tuning(
            self._selfie_seg_grow_px, self._selfie_seg_feather_px,
            self._selfie_seg_blur_px, self._selfie_seg_temporal)

    def next_camera(self) -> str | None:
        """Switch to the next detected camera device. Returns a label or None if only one."""
        if len(self._available_cameras) <= 1:
            return None
        self._prepare_camera_switch_hide()
        self._camera_list_index = (self._camera_list_index + 1) % len(self._available_cameras)
        log.debug('WebcamSystem: next camera index -> %d', self._camera_list_index)
        self._restart_worker()
        dev = self._available_cameras[self._camera_list_index]
        return f'/dev/video{dev}' if sys.platform != 'win32' else str(dev)

    def prev_camera(self) -> str | None:
        """Switch to the previous detected camera device. Returns a label or None if only one."""
        if len(self._available_cameras) <= 1:
            return None
        self._prepare_camera_switch_hide()
        self._camera_list_index = (self._camera_list_index - 1) % len(self._available_cameras)
        log.debug('WebcamSystem: prev camera index -> %d', self._camera_list_index)
        self._restart_worker()
        dev = self._available_cameras[self._camera_list_index]
        return f'/dev/video{dev}' if sys.platform != 'win32' else str(dev)

    @property
    def current_treatment(self) -> str:
        return _TREATMENT_NAMES[self._treatment_index]

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _clear_border_ring(self, x: float, y: float, w: float, h: float, b: float) -> None:
        """Clear only the border margin around the PiP, never the inner rect.

        The cam quad always fully overwrites the inner rect for the opaque
        treatments (raw/dim), so clearing the whole outer rect first was
        harmless there. But in 'transparent' selfie-seg mode, background
        pixels carry alpha < 1 and blend against whatever is already in the
        framebuffer -- if that were the border colour (from a full-rect
        clear), a cutout would reveal a flat backplate instead of the show
        already rendered behind the PiP. Clearing only the four ring strips
        leaves the inner rect's existing framebuffer content alone for the
        cam quad to blend against.
        """
        sx = max(0, int(x - b))
        sy = max(0, int(self._height - (y + h + b)))
        sw = int(w + 2 * b)
        sh = int(h + 2 * b)
        bpx = max(0, int(round(b)))
        if bpx <= 0 or bpx * 2 >= min(sw, sh):
            self._ctx.scissor = (sx, sy, sw, sh)
            self._ctx.clear(*self._border_color)
            self._ctx.scissor = None
            return
        inner_sy = max(0, int(self._height - (y + h)))
        inner_sh = int(h)
        for rx, ry, rw, rh in (
            (sx, sy + sh - bpx, sw, bpx),               # top
            (sx, sy, sw, bpx),                          # bottom
            (sx, inner_sy, bpx, inner_sh),               # left
            (sx + sw - bpx, inner_sy, bpx, inner_sh),    # right
        ):
            if rw <= 0 or rh <= 0:
                continue
            self._ctx.scissor = (rx, ry, rw, rh)
            self._ctx.clear(*self._border_color)
        self._ctx.scissor = None

    def _draw_border_futuristic(self, x: float, y: float, w: float, h: float,
                                 bass: float, treble: float) -> None:
        """Render the glowing neon/HUD/drippy border via `_BORDER_FRAG`.

        Ring thickness pulses with bass exactly like the classic border
        did; a `headroom` margin is added on top so the shader's organic
        drip tendrils (see `_BORDER_FRAG`) have room to bulge outward
        without clipping at the quad's own edge. Drawn before the cam
        quad, with fully transparent interior pixels, so the opaque cam
        quad drawn right after this shows through unobstructed.
        """
        b = self._border_thickness + bass * (self._border_thickness * 0.75)
        headroom = 26.0 * self._border_intensity
        pad = max(1.0, b + headroom)
        self._write_border_quad(x, y, w, h, pad)
        prog = self._border_prog
        prog['iRectPx'].value = (w, h)
        prog['iPadPx'].value = pad
        prog['iBorderPx'].value = b
        prog['iIntensity'].value = self._border_intensity
        prog['iHueBase'].value = self._border_hue_base
        prog['iHueSpeed'].value = self._border_hue_speed
        prog['iTime'].value = self._time
        prog['iBass'].value = bass
        prog['iTreble'].value = treble
        self._border_vao.render(moderngl.TRIANGLES)

    def render(self, dt: float, bass: float, treble: float) -> None:
        if self._switch_hide_active and time.monotonic() >= self._switch_hide_deadline:
            self._switch_hide_active = False
            self._hidden = False
        if not self.enabled or self._hidden:
            return

        self._time += dt
        self._advance_auto_cycle(dt)

        tex = self._get_camera_texture()
        x, y, w, h = self._rect_from_layout()

        # No backplate in 'transparent' mode -- a cutout floating over the
        # show reads wrong with a coloured frame boxing it in.
        if self._selfie_seg != 'transparent':
            if self._border_style == 'classic':
                b = self._border_thickness + bass * (self._border_thickness * 0.75)
                self._clear_border_ring(x, y, w, h, b)
            elif self._border_style == 'futuristic':
                self._draw_border_futuristic(x, y, w, h, bass, treble)
            # 'off' draws nothing.

        self._write_cam_quad(x, y, w, h)
        tex.use(location=0)
        name = self.current_treatment
        prog = self._programs[name]
        vao  = self._cam_vao_map[name]

        def _u(k: str, v) -> None:
            try:
                prog[k].value = v
            except KeyError:
                pass

        _u('cam_tex', 0)
        _u('iBass',   bass)
        _u('iTreble', treble)
        _u('iTime',   self._time)
        vao.render(moderngl.TRIANGLES)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_camera_texture(self) -> moderngl.Texture:
        """Return the camera texture, re-uploading only on a NEW frame.

        This used to copy the frame, convert it with ``tobytes()`` and push
        it to the GPU on *every rendered frame* — three full passes over
        1280x720x3 at up to 60 fps to display a picture the 30 fps camera
        had not changed, roughly 490 MB/s of pointless memory traffic. It
        fitted the budget alone, but not alongside a 4K mirror composite,
        which is exactly the "webcam + mirror together" fault the owner
        isolated (2026-08-08 capture: subsys_upd median 21 ms, the largest
        single stage). Now the whole path short-circuits when the capture
        sequence has not advanced, and the flip/brightness/contrast work —
        which allocated a float32 copy of the entire frame — rides along
        only on frames that are actually new.
        """
        if self._worker is None:
            return self._fallback_tex
        frame, seq = self._worker.latest_if_newer(self._cam_seq)
        if frame is None:
            # Nothing new: reuse the texture already on the GPU.
            return self._cam_tex if self._cam_tex is not None else self._fallback_tex
        self._cam_seq = seq
        if self._flip_horizontal and self._flip_vertical:
            frame = np.ascontiguousarray(frame[::-1, ::-1, :])
        elif self._flip_horizontal:
            frame = np.ascontiguousarray(frame[:, ::-1, :])
        elif self._flip_vertical:
            frame = np.ascontiguousarray(frame[::-1, :, :])
        if self._brightness != 1.0 or self._contrast != 1.0:
            # Brightness/contrast are colour adjustments; a transparent-mode
            # frame's 4th channel is a segmentation matte, not colour, and
            # must ride through unscaled.
            rgb = frame[..., :3].astype(np.float32) / 255.0
            rgb = (rgb - 0.5) * self._contrast + 0.5
            rgb *= self._brightness
            rgb = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            frame = np.dstack((rgb, frame[..., 3])) if frame.shape[2] == 4 else rgb
        h, w, channels = frame.shape
        if (self._cam_tex is None or self._cam_tex_size != (w, h)
                or self._cam_tex_channels != channels):
            if self._cam_tex is not None:
                self._cam_tex.release()
            self._cam_tex = self._ctx.texture((w, h), channels, frame.tobytes())
            self._cam_tex.filter = moderngl.LINEAR, moderngl.LINEAR
            self._cam_tex_size = (w, h)
            self._cam_tex_channels = channels
        else:
            self._cam_tex.write(frame.tobytes())
        return self._cam_tex

    def _rect_from_layout(self) -> tuple[float, float, float, float]:
        if self._layout == 'fullscreen':
            return 0.0, 0.0, float(self._width), float(self._height)

        scale = max(0.12, min(0.80, self._pip_scale))
        w = self._width * scale
        h = w * (self._cam_height / max(1, self._cam_width))
        h = min(h, self._height * 0.70)

        anchor = self._LAYOUTS.get(self._layout, self._LAYOUTS['bottom_right'])
        ax, ay = anchor

        x = 24.0 if ax <= 0.05 else (self._width - w - 24.0 if ax >= 0.95 else (self._width - w) * 0.5)
        y = 24.0 if ay <= 0.05 else (self._height - h - 24.0 if ay >= 0.95 else (self._height - h) * 0.5)

        return x, y, w, h

    def _quad_verts(self, x: float, y: float, w: float, h: float) -> np.ndarray:
        """Build a two-triangle fullscreen-quad vertex buffer for the rect
        ``(x, y, w, h)`` in viewport pixel space (origin top-left), with
        ``in_uv`` in [0,1]^2 -- uv (0,0) at the rect's top-left, matching
        the camera texture's own row order. Shared by `_write_cam_quad`
        and `_write_border_quad` so the two quads (camera image, border
        ring) always use identical pixel-to-clip-space math.
        """
        def px(v: float) -> float:
            return (v / self._width) * 2.0 - 1.0
        def py(v: float) -> float:
            return 1.0 - (v / self._height) * 2.0

        x0, x1 = px(x), px(x + w)
        y0, y1 = py(y), py(y + h)

        return np.array([
            x0, y0, 0.0, 0.0,
            x1, y0, 1.0, 0.0,
            x0, y1, 0.0, 1.0,
            x1, y0, 1.0, 0.0,
            x1, y1, 1.0, 1.0,
            x0, y1, 0.0, 1.0,
        ], dtype=np.float32)

    def _write_cam_quad(self, x: float, y: float, w: float, h: float) -> None:
        self._cam_vbo.write(self._quad_verts(x, y, w, h))

    def _write_border_quad(self, x: float, y: float, w: float, h: float, pad: float) -> None:
        """Write the border ring's quad: the camera rect grown by ``pad``
        px on every side, so the border shader has room to render its
        glow/halo/drip tendrils around the (untouched) inner camera rect.
        """
        self._border_vbo.write(self._quad_verts(x - pad, y - pad, w + 2.0 * pad, h + 2.0 * pad))

    def destroy(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker.join(timeout=2.0)
            self._worker = None
        for vao in self._cam_vao_map.values():
            vao.release()
        self._cam_vbo.release()
        for prog in self._programs.values():
            prog.release()
        self._border_vao.release()
        self._border_vbo.release()
        self._border_prog.release()
        self._fallback_tex.release()
        if self._cam_tex is not None:
            self._cam_tex.release()
