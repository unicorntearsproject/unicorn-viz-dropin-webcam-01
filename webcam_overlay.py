"""webcam-01 — WebcamSystem subsystem drop-in.

Always-on camera overlay rendered above every main effect, below the HUD.
Provides five visual treatments that cycle with KP-/ and KP-*.
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
"""
from __future__ import annotations

__version__ = '1.0.0-rc.4'

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

    _MAX_OPEN_FAILURES = 5  # give up after this many consecutive not-found attempts

    def __init__(self, device: int, width: int, height: int, fps: int) -> None:
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
        self._default_brightness = float(self._brightness)
        self._default_contrast = float(self._contrast)
        self._default_flip_horizontal = bool(self._flip_horizontal)
        self._default_flip_vertical = bool(self._flip_vertical)
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

        fb = np.zeros((256, 256, 3), dtype=np.uint8)
        yy, xx = np.indices((256, 256))
        fb[..., 0] = ((xx * 2 + yy) % 255).astype(np.uint8)
        fb[..., 1] = ((yy * 2) % 255).astype(np.uint8)
        fb[..., 2] = 100
        self._fallback_tex = ctx.texture((256, 256), 3, fb.tobytes())
        self._fallback_tex.filter = moderngl.LINEAR, moderngl.LINEAR

        self._cam_tex: moderngl.Texture | None = None
        self._cam_tex_size = (0, 0)
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

    def _remember_current_camera_settings(self, camera_id: int | None = None) -> None:
        target_id = self._active_or_default_device() if camera_id is None else int(camera_id)
        self._camera_settings[target_id] = {
            'brightness': float(self._brightness),
            'contrast': float(self._contrast),
            'flip_horizontal': bool(self._flip_horizontal),
            'flip_vertical': bool(self._flip_vertical),
        }

    def _apply_camera_settings(self, camera_id: int) -> None:
        settings = self._camera_settings.get(int(camera_id))
        if not isinstance(settings, dict):
            self._brightness = float(self._default_brightness)
            self._contrast = float(self._default_contrast)
            self._flip_horizontal = bool(self._default_flip_horizontal)
            self._flip_vertical = bool(self._default_flip_vertical)
            return
        self._brightness = max(0.2, min(2.0, float(settings.get('brightness', self._brightness))))
        self._contrast = max(0.2, min(2.0, float(settings.get('contrast', self._contrast))))
        self._flip_horizontal = bool(settings.get('flip_horizontal', self._flip_horizontal))
        self._flip_vertical = bool(settings.get('flip_vertical', self._flip_vertical))

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
            self._worker = _CameraWorker(device, self._cam_width, self._cam_height, self._cam_fps)
            self._worker.start()
            return
        self._worker.request_device(device)

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

    def render(self, dt: float, bass: float, treble: float) -> None:
        if self._switch_hide_active and time.monotonic() >= self._switch_hide_deadline:
            self._switch_hide_active = False
            self._hidden = False
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

        b = self._border_thickness + bass * (self._border_thickness * 0.75)
        sx = max(0, int(x - b))
        sy = max(0, int(self._height - (y + h + b)))
        sw = int(w + 2 * b)
        sh = int(h + 2 * b)
        self._ctx.scissor = (sx, sy, sw, sh)
        self._ctx.clear(*self._border_color)
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
            adjusted = frame.astype(np.float32) / 255.0
            adjusted = (adjusted - 0.5) * self._contrast + 0.5
            adjusted *= self._brightness
            frame = np.clip(adjusted * 255.0, 0.0, 255.0).astype(np.uint8)
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
            self._worker.join(timeout=2.0)
            self._worker = None
        for vao in self._cam_vao_map.values():
            vao.release()
        self._cam_vbo.release()
        for prog in self._programs.values():
            prog.release()
        self._fallback_tex.release()
        if self._cam_tex is not None:
            self._cam_tex.release()
