"""Webcam Overlay drop-in effect.

Live camera feed rendered as fullscreen or PiP with keypad-position presets.
Supports optional custom asset directories for future overlays via
``webcam_asset_dirs`` config key (comma-separated paths).

Keypad layout presets expected by hotkeys:
- KP0: fullscreen
- KP.: hide
- KP1..KP9: bottom-left..top-right grid positions
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import moderngl
import numpy as np

from unicornviz.effects.base import BaseEffect, AudioData

try:
    import cv2  # type: ignore
    _CV2 = True
except Exception:  # pragma: no cover
    cv2 = None
    _CV2 = False

log = logging.getLogger(__name__)

_BG_VERT = """
#version 330
in vec2 in_vert;
out vec2 v_uv;
void main() {
    v_uv = in_vert * 0.5 + 0.5;
    gl_Position = vec4(in_vert, 0.0, 1.0);
}
"""

_BG_FRAG = """
#version 330
uniform float iTime;
uniform float iBass;
uniform float iMid;
uniform float iTreble;
in vec2 v_uv;
out vec4 fragColor;

vec3 palette(float t) {
    vec3 a = vec3(0.5);
    vec3 b = vec3(0.5);
    vec3 c = vec3(1.0);
    vec3 d = vec3(0.02, 0.15, 0.35);
    return a + b * cos(6.28318 * (c * t + d));
}

void main() {
    vec2 uv = v_uv;
    float t = iTime;
    float f = 0.0;
    f += sin(uv.x * 16.0 + t * 1.2) * 0.25;
    f += sin(uv.y * 13.0 - t * 1.1) * 0.22;
    f += sin((uv.x + uv.y) * 8.0 + t * 0.7) * 0.18;

    float beat = iBass * 0.35 + iTreble * 0.15;
    vec3 col = palette(f + t * 0.05 + beat);
    col = mix(col, vec3(0.04, 0.05, 0.09), 0.35);

    float grid = smoothstep(0.02, 0.0, abs(fract(uv.x * 12.0) - 0.5))
               + smoothstep(0.02, 0.0, abs(fract(uv.y * 7.0) - 0.5));
    col += vec3(0.08, 0.16, 0.25) * grid * 0.25;

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""

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

_CAM_FRAG = """
#version 330
uniform sampler2D cam_tex;
uniform float iBass;
uniform float iTime;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec2 uv = v_uv;
    vec3 col = texture(cam_tex, uv).rgb;

    // Gentle punch + scanline vibe
    float punch = 1.0 + iBass * 0.25;
    col *= punch;
    col *= 0.96 + 0.04 * sin((uv.y + iTime * 0.2) * 700.0);

    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


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

    def run(self) -> None:
        if not _CV2:
            return

        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            log.warning('WebcamOverlay: could not open camera device %d', self._device)
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))
        cap.set(cv2.CAP_PROP_FPS, float(self._fps))
        self._active = True

        period = 1.0 / max(1, self._fps)
        while not self._stop.is_set():
            ok, frame = cap.read()
            if ok and frame is not None:
                # Convert BGR->RGB and mirror for selfie feel.
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.flip(frame, 1)
                with self._lock:
                    self._latest = frame
            time.sleep(period * 0.45)

        cap.release()
        self._active = False

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


class WebcamOverlay(BaseEffect):
    NAME = 'Webcam Overlay'
    AUTHOR = 'unicorn-viz drop-in'
    TAGS = ['webcam', 'overlay', 'camera', 'drop-in']

    _LAYOUTS: dict[str, tuple[float, float]] = {
        'top_left': (0.04, 0.04),
        'top_center': (0.5, 0.04),
        'top_right': (0.96, 0.04),
        'left': (0.04, 0.5),
        'center': (0.5, 0.5),
        'right': (0.96, 0.5),
        'bottom_left': (0.04, 0.96),
        'bottom_center': (0.5, 0.96),
        'bottom_right': (0.96, 0.96),
    }

    _KEYPAD_TO_LAYOUT = {
        '7': 'top_left',
        '8': 'top_center',
        '9': 'top_right',
        '4': 'left',
        '5': 'center',
        '6': 'right',
        '1': 'bottom_left',
        '2': 'bottom_center',
        '3': 'bottom_right',
        '0': 'fullscreen',
        '.': 'hide',
    }

    def _init(self) -> None:
        self.parameters = {
            'speed': 1.0,
            'pip_scale': float(self.config.get('pip_scale', 0.33)),
            'border': 1.0,
        }

        self._bg_prog = self._make_program(_BG_VERT, _BG_FRAG)
        self._bg_vao, self._bg_vbo = self._fullscreen_quad(self._bg_prog)

        self._cam_prog = self._make_program(_CAM_VERT, _CAM_FRAG)
        self._cam_vbo = self.ctx.buffer(reserve=6 * 4 * 4)
        self._cam_vao = self.ctx.vertex_array(
            self._cam_prog,
            [(self._cam_vbo, '2f 2f', 'in_pos', 'in_uv')],
        )

        self._bass = 0.0
        self._mid = 0.0
        self._treble = 0.0

        # Camera configuration from effect config section.
        self._cam_device = int(self.config.get('camera_device', 0))
        self._cam_width = int(self.config.get('camera_width', 1280))
        self._cam_height = int(self.config.get('camera_height', 720))
        self._cam_fps = int(self.config.get('camera_fps', 30))

        self._layout = str(self.config.get('pip_position', 'bottom_right')).lower()
        self._hidden = False

        self._worker = _CameraWorker(self._cam_device, self._cam_width, self._cam_height, self._cam_fps)
        self._worker.start()

        # Fallback texture shown when camera is unavailable.
        fallback = np.zeros((256, 256, 3), dtype=np.uint8)
        yy, xx = np.indices((256, 256))
        fallback[..., 0] = ((xx * 2 + yy) % 255).astype(np.uint8)
        fallback[..., 1] = ((yy * 2) % 255).astype(np.uint8)
        fallback[..., 2] = 120
        self._fallback_tex = self.ctx.texture((256, 256), 3, fallback.tobytes())
        self._fallback_tex.filter = moderngl.LINEAR, moderngl.LINEAR

        self._cam_tex: moderngl.Texture | None = None
        self._cam_tex_size = (0, 0)

        self._asset_dirs = self._resolve_asset_dirs()
        log.info('WebcamOverlay assets: %s', ', '.join(str(p) for p in self._asset_dirs))

    def _resolve_asset_dirs(self) -> list[Path]:
        default_dir = Path(__file__).resolve().parent / 'assets' / 'webcam'
        extra = str(self.config.get('webcam_asset_dirs', '')).strip()
        dirs = [default_dir]
        if extra:
            for raw in extra.split(','):
                p = Path(raw.strip()).expanduser()
                if p.exists():
                    dirs.append(p)
        seen = set()
        out: list[Path] = []
        for d in dirs:
            key = str(d.resolve())
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    def set_camera_layout(self, layout: str) -> str:
        """Set camera layout preset. Returns resolved layout name."""
        layout = layout.lower().strip()
        if layout in self._KEYPAD_TO_LAYOUT:
            layout = self._KEYPAD_TO_LAYOUT[layout]

        if layout == 'hide':
            self._hidden = True
            return 'hide'

        self._hidden = False
        if layout == 'fullscreen':
            self._layout = 'fullscreen'
            return self._layout

        if layout in self._LAYOUTS:
            self._layout = layout
            return self._layout

        return self._layout

    def _rect_from_layout(self) -> tuple[float, float, float, float]:
        if self._layout == 'fullscreen':
            return 0.0, 0.0, float(self.width), float(self.height)

        scale = max(0.12, min(0.8, float(self.parameters['pip_scale'])))
        w = self.width * scale
        h = w * (self._cam_height / max(1, self._cam_width))
        h = min(h, self.height * 0.7)

        anchor = self._LAYOUTS.get(self._layout, self._LAYOUTS['bottom_right'])
        ax, ay = anchor

        if ax <= 0.05:
            x = 24.0
        elif ax >= 0.95:
            x = self.width - w - 24.0
        else:
            x = (self.width - w) * 0.5

        if ay <= 0.05:
            y = 24.0
        elif ay >= 0.95:
            y = self.height - h - 24.0
        else:
            y = (self.height - h) * 0.5

        return x, y, w, h

    def _update_camera_texture(self) -> moderngl.Texture:
        frame = self._worker.latest()
        if frame is None:
            return self._fallback_tex

        h, w = frame.shape[:2]
        if self._cam_tex is None or self._cam_tex_size != (w, h):
            if self._cam_tex is not None:
                self._cam_tex.release()
            self._cam_tex = self.ctx.texture((w, h), 3, frame.tobytes())
            self._cam_tex.filter = moderngl.LINEAR, moderngl.LINEAR
            self._cam_tex_size = (w, h)
        else:
            self._cam_tex.write(frame.tobytes())

        return self._cam_tex

    def _write_cam_quad(self, x: float, y: float, w: float, h: float) -> None:
        def px(v: float) -> float:
            return (v / self.width) * 2.0 - 1.0

        def py(v: float) -> float:
            return 1.0 - (v / self.height) * 2.0

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

    def update(self, dt: float, audio: AudioData) -> None:
        super().update(dt, audio)
        self._bass = audio.bass
        self._mid = audio.mid
        self._treble = audio.treble

    def render(self) -> None:
        def set_bg(name: str, value) -> None:
            try:
                self._bg_prog[name].value = value
            except KeyError:
                pass

        def set_cam(name: str, value) -> None:
            try:
                self._cam_prog[name].value = value
            except KeyError:
                pass

        # Animated background always present.
        set_bg('iTime', self.time)
        set_bg('iBass', self._bass)
        set_bg('iMid', self._mid)
        set_bg('iTreble', self._treble)
        self._bg_vao.render(moderngl.TRIANGLE_STRIP)

        if self._hidden:
            return

        tex = self._update_camera_texture()
        x, y, w, h = self._rect_from_layout()

        # Border/backplate
        b = 8.0 + self._bass * 10.0 * float(self.parameters['border'])
        self.ctx.enable(moderngl.BLEND)

        # Draw backplate using scissor + clear trick
        self.ctx.scissor = (max(0, int(x - b)), max(0, int(self.height - (y + h + b))),
                            int(w + 2 * b), int(h + 2 * b))
        self.ctx.clear(0.05, 0.75, 1.0, 0.35)
        self.ctx.scissor = None

        # Camera textured quad
        self._write_cam_quad(x, y, w, h)
        tex.use(location=0)
        set_cam('cam_tex', 0)
        set_cam('iBass', self._bass)
        set_cam('iTime', self.time)
        self._cam_vao.render(moderngl.TRIANGLES)

    def destroy(self) -> None:
        self._worker.stop()
        self._bg_vao.release()
        self._bg_vbo.release()
        self._bg_prog.release()
        self._cam_vao.release()
        self._cam_vbo.release()
        self._cam_prog.release()
        self._fallback_tex.release()
        if self._cam_tex is not None:
            self._cam_tex.release()
