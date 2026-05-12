"""webcam-01 — WebcamSystem subsystem drop-in.

Always-on camera overlay rendered above every main effect, below the HUD.
Provides five visual treatments that cycle with KP-/ and KP-*.
Does not appear in the main effect playlist.

Keypad controls (active at all times regardless of which effect is running):
    KP 0          Fullscreen camera
    KP .          Hide camera
    KP 1..9       PiP position (bottom-left .. top-right grid)
    KP -          Shrink PiP
    KP +          Grow PiP
    KP /          Previous camera treatment
    KP *          Next camera treatment
    KP Enter      Toggle auto-cycle camera treatments

Camera treatments:
    raw       Clean camera feed, minimal processing
    neon      Scanline punch + bass-reactive exposure
    glitch    Chromatic aberration + scan scramble on beats
    vintage   Sepia tone, film grain, vignette
    thermal   False-colour luminance heat map
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path

import moderngl
import numpy as np

try:
    import cv2  # type: ignore
    _CV2 = True
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
    _CV2 = False

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
    fragColor = vec4(texture(cam_tex, v_uv).rgb, 1.0);
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
    vec3 col = texture(cam_tex, v_uv).rgb;
    float punch = 1.0 + iBass * 0.30;
    col *= punch;
    col *= 0.96 + 0.04 * sin((v_uv.y + iTime * 0.2) * 700.0);
    col.b += iBass * 0.06;
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
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
    float ca = iBass * 0.018 + iTreble * 0.004;
    vec3 col;
    col.r = texture(cam_tex, uv + vec2( ca, 0.0)).r;
    col.g = texture(cam_tex, uv              ).g;
    col.b = texture(cam_tex, uv - vec2( ca, 0.0)).b;
    float scan = step(0.72, iBass) * step(0.0, sin(uv.y * 180.0 + iTime * 28.0));
    float shift = scan * iBass * 0.045;
    col = mix(col, texture(cam_tex, vec2(uv.x + shift, uv.y)).rgb, scan);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
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
    vec3 col = texture(cam_tex, v_uv).rgb;
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
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
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
    vec3 col = texture(cam_tex, v_uv).rgb;
    float luma = dot(col, vec3(0.299, 0.587, 0.114));
    float boosted = clamp(luma + iBass * 0.18, 0.0, 1.0);
    fragColor = vec4(thermalPalette(boosted), 1.0);
}
""",
}

_TREATMENT_NAMES: list[str] = ['raw', 'neon', 'glitch', 'vintage', 'thermal']


# ---------------------------------------------------------------------------
# Camera capture thread
# ---------------------------------------------------------------------------

class _CameraWorker(threading.Thread):
    """Background camera capture thread."""

    def __init__(self, device: int, width: int, height: int, fps: int) -> None:
        super().__init__(daemon=True, name='webcam-capture')
        self._device = int(device)
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest: np.ndarray | None = None
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

        while not self._stop.is_set():
            cap = None
            for offset in range(3):
                cap = self._try_open(self._device + offset)
                if cap is not None:
                    if offset:
                        log.info('WebcamSystem: opened /dev/video%d (fallback)', self._device + offset)
                    break

            if cap is None:
                log.warning('WebcamSystem: no camera found at devices %d..%d, retrying in 3 s',
                            self._device, self._device + 2)
                self._stop.wait(3.0)
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  float(self._width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
            cap.set(cv2.CAP_PROP_FPS,          float(self._fps))
            self._active = True
            log.info('WebcamSystem: camera %d opened at %dx%d',
                     self._device,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

            fail_count = 0
            while not self._stop.is_set():
                ok, frame = cap.read()
                if ok and frame is not None:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.flip(frame, 1)
                    with self._lock:
                        self._latest = frame
                    fail_count = 0
                else:
                    fail_count += 1
                    if fail_count >= 20:
                        log.warning('WebcamSystem: camera stopped responding, reconnecting')
                        break
                self._stop.wait(period * 0.45)

            cap.release()
            self._active = False
            if not self._stop.is_set():
                self._stop.wait(2.0)

    def stop(self) -> None:
        self._stop.set()

    def latest(self) -> np.ndarray | None:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    @property
    def active(self) -> bool:
        return self._active


# ---------------------------------------------------------------------------
# WebcamSystem
# ---------------------------------------------------------------------------

class WebcamSystem:
    """Always-on camera overlay.  Not a playlist effect."""

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
        self._layout     = 'bottom_right'
        self._hidden     = False
        # Route through set_layout so 'hidden' correctly sets self._hidden.
        self.set_layout(str(cfg.get('pip_position', 'bottom_right')).lower())

        default_treatment = str(cfg.get('treatment', 'neon')).lower()
        if default_treatment not in _TREATMENT_NAMES:
            default_treatment = 'neon'
        self._treatment_index = _TREATMENT_NAMES.index(default_treatment)

        self._auto_cycle = False
        self._auto_cycle_interval = float(cfg.get('cycle_interval', 30.0)) or 30.0
        self._auto_cycle_timer = 0.0

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

        fb = np.zeros((256, 256, 3), dtype=np.uint8)
        yy, xx = np.indices((256, 256))
        fb[..., 0] = ((xx * 2 + yy) % 255).astype(np.uint8)
        fb[..., 1] = ((yy * 2) % 255).astype(np.uint8)
        fb[..., 2] = 100
        self._fallback_tex = ctx.texture((256, 256), 3, fb.tobytes())
        self._fallback_tex.filter = moderngl.LINEAR, moderngl.LINEAR

        self._cam_tex: moderngl.Texture | None = None
        self._cam_tex_size = (0, 0)
        self._worker: _CameraWorker | None = None
        self._time = 0.0

    def start(self) -> None:
        if not self.enabled:
            return
        if not _CV2:
            log.error('WebcamSystem: opencv-python-headless not installed')
            return
        self._worker = _CameraWorker(
            self._device, self._cam_width, self._cam_height, self._cam_fps
        )
        self._worker.start()
        log.info('WebcamSystem: started (treatment=%s, layout=%s)',
                 self.current_treatment, self._layout)

    def resize(self, w: int, h: int) -> None:
        self._width = w
        self._height = h

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def set_layout(self, token: str) -> str:
        token = token.lower().strip()
        if token == 'hidden':
            token = 'hide'
        if token in self._KEYPAD_TO_LAYOUT:
            token = self._KEYPAD_TO_LAYOUT[token]
        if token == 'hide':
            self._hidden = True
            return 'hide'
        self._hidden = False
        if token == 'fullscreen' or token in self._LAYOUTS:
            self._layout = token
        return self._layout

    def scale_pip(self, delta: float) -> float:
        self._pip_scale = max(0.12, min(0.80, self._pip_scale + delta))
        return self._pip_scale

    def next_treatment(self) -> str:
        self._treatment_index = (self._treatment_index + 1) % len(_TREATMENT_NAMES)
        return _TREATMENT_NAMES[self._treatment_index]

    def prev_treatment(self) -> str:
        self._treatment_index = (self._treatment_index - 1) % len(_TREATMENT_NAMES)
        return _TREATMENT_NAMES[self._treatment_index]

    def toggle_auto_cycle(self) -> bool:
        self._auto_cycle = not self._auto_cycle
        self._auto_cycle_timer = 0.0
        return self._auto_cycle

    @property
    def current_treatment(self) -> str:
        return _TREATMENT_NAMES[self._treatment_index]

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, dt: float, bass: float, treble: float) -> None:
        if not self.enabled or self._hidden:
            return

        self._time += dt

        if self._auto_cycle:
            self._auto_cycle_timer += dt
            if self._auto_cycle_timer >= self._auto_cycle_interval:
                self._auto_cycle_timer = 0.0
                self.next_treatment()

        tex = self._get_camera_texture()
        x, y, w, h = self._rect_from_layout()

        b = 6.0 + bass * 12.0
        sx = max(0, int(x - b))
        sy = max(0, int(self._height - (y + h + b)))
        sw = int(w + 2 * b)
        sh = int(h + 2 * b)
        self._ctx.scissor = (sx, sy, sw, sh)
        self._ctx.clear(0.04, 0.72, 1.0, 0.28)
        self._ctx.scissor = None

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
        if self._worker is None:
            return self._fallback_tex
        frame = self._worker.latest()
        if frame is None:
            return self._fallback_tex
        h, w = frame.shape[:2]
        if self._cam_tex is None or self._cam_tex_size != (w, h):
            if self._cam_tex is not None:
                self._cam_tex.release()
            self._cam_tex = self._ctx.texture((w, h), 3, frame.tobytes())
            self._cam_tex.filter = moderngl.LINEAR, moderngl.LINEAR
            self._cam_tex_size = (w, h)
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

    def _write_cam_quad(self, x: float, y: float, w: float, h: float) -> None:
        def px(v: float) -> float:
            return (v / self._width) * 2.0 - 1.0
        def py(v: float) -> float:
            return 1.0 - (v / self._height) * 2.0

        x0, x1 = px(x), px(x + w)
        y0, y1 = py(y), py(y + h)

        verts = np.array([
            x0, y0, 0.0, 0.0,
            x1, y0, 1.0, 0.0,
            x0, y1, 0.0, 1.0,
            x1, y0, 1.0, 0.0,
            x1, y1, 1.0, 1.0,
            x0, y1, 0.0, 1.0,
        ], dtype=np.float32)
        self._cam_vbo.write(verts)

    def destroy(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._worker = None
        for vao in self._cam_vao_map.values():
            vao.release()
        self._cam_vbo.release()
        for prog in self._programs.values():
            prog.release()
        self._fallback_tex.release()
        if self._cam_tex is not None:
            self._cam_tex.release()
