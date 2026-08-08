"""Regression tests: the camera must not run while the overlay is hidden.

The shipped default (`pip_position = "hidden"` with enabled unset → true)
used to light the webcam LED at first boot and capture at 30 fps into a
frame nobody rendered — a privacy problem for a public release. Contract:
start() while hidden opens nothing; a deliberate hide releases the worker;
showing starts capture lazily; the transient camera-switch shim keeps the
worker alive.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webcam_overlay as wo  # noqa: E402


class _Worker:
    def __init__(self) -> None:
        self.stopped = False

    def is_alive(self) -> bool:
        return not self.stopped

    def stop(self) -> None:
        self.stopped = True

    def join(self, timeout=None) -> None:
        pass


def _system(hidden: bool) -> 'wo.WebcamSystem':
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s.enabled = True
    s._hidden = hidden
    s._worker = None
    s._available_cameras = [0]
    s._layout = 'bottom_right'
    s._switch_hide_active = False
    s._switch_hide_deadline = 0.0
    s._switch_hide_duration_s = 1.2
    s._treatment_index = 0
    calls = {'restart': 0, 'discover': 0}
    s._test_calls = calls  # type: ignore[attr-defined]
    s._restart_worker = lambda: calls.__setitem__('restart', calls['restart'] + 1)  # type: ignore[method-assign]
    s.rediscover_cameras = lambda: calls.__setitem__('discover', calls['discover'] + 1)  # type: ignore[method-assign]
    return s


def test_start_while_hidden_never_touches_the_camera(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_CV2', True)
    s = _system(hidden=True)
    s.start()
    assert s._test_calls == {'restart': 0, 'discover': 0}


def test_start_while_visible_opens_camera(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_CV2', True)
    s = _system(hidden=False)
    s.start()
    assert s._test_calls['restart'] == 1


def test_hide_layout_releases_worker(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_CV2', True)
    s = _system(hidden=False)
    worker = _Worker()
    s._worker = worker
    assert s.set_layout('hide') == 'hide'
    assert worker.stopped is True
    assert s._worker is None


def test_show_after_hide_starts_capture_lazily(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_CV2', True)
    s = _system(hidden=True)
    s.set_layout('bottom_right')
    assert s._hidden is False
    assert s._test_calls['restart'] == 1


def test_camera_switch_shim_keeps_worker_alive(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_CV2', True)
    s = _system(hidden=False)
    worker = _Worker()
    s._worker = worker
    s._prepare_camera_switch_hide()
    assert s._hidden is True          # visually hidden during the switch…
    assert worker.stopped is False    # …but capture continues (transient)


# ---------------------------------------------------------------- construction

class _FakeTexture:
    def __init__(self, *_a, **_kw) -> None:
        self.filter = None

    def write(self, *_a, **_kw) -> None: pass
    def use(self, *_a, **_kw) -> None: pass
    def release(self) -> None: pass


class _FakeCtx:
    """Just enough moderngl surface for WebcamSystem.__init__."""

    def texture(self, *_a, **_kw): return _FakeTexture()
    def program(self, **_kw): return {}
    def buffer(self, *_a, **_kw): return object()
    def vertex_array(self, *_a, **_kw): return object()
    def simple_vertex_array(self, *_a, **_kw): return object()


def test_constructor_completes_for_every_configured_position(monkeypatch) -> None:
    """Regression: 1.0.0-rc.2 called the side-effecting set_layout() from
    __init__, which reached _ensure_worker() before _worker and
    _available_cameras existed. The AttributeError escaped the constructor
    and core's guarded loader swallowed it, silently disabling the whole
    drop-in for the session.

    Every earlier test in this file builds the object with __new__ and hand-
    sets attributes, so none of them could ever have caught a construction
    -order bug. This one constructs it for real.
    """
    monkeypatch.setattr(wo, '_CV2', False)   # never touch a real camera
    for position in ('hidden', 'hide', 'bottom_right', 'fullscreen', '7', 'nonsense'):
        system = wo.WebcamSystem(_FakeCtx(), 1920, 1080, {'pip_position': position})
        assert hasattr(system, '_worker')
        assert hasattr(system, '_available_cameras')
        # 'hidden'/'hide' must land hidden; everything else visible.
        assert system._hidden is (position in ('hidden', 'hide'))


def test_constructor_does_not_start_capture(monkeypatch) -> None:
    """__init__ records the layout; start() is what opens the camera."""
    monkeypatch.setattr(wo, '_CV2', False)
    started: list[int] = []
    monkeypatch.setattr(
        wo.WebcamSystem, '_restart_worker',
        lambda self: started.append(1))
    wo.WebcamSystem(_FakeCtx(), 1920, 1080, {'pip_position': 'bottom_right'})
    assert started == []


# ------------------------------------------------- redundant-upload fix (rc.4)

class _SeqWorker:
    """Capture worker stand-in with the real sequence semantics."""

    def __init__(self) -> None:
        import numpy as _np
        self._frame = _np.zeros((4, 4, 3), dtype=_np.uint8)
        self._seq = 0

    def bump(self) -> None:
        self._seq += 1

    @property
    def sequence(self) -> int:
        return self._seq

    def latest_if_newer(self, seen):
        if self._seq == seen:
            return None, self._seq
        return self._frame.copy(), self._seq

    def is_alive(self) -> bool:
        return True


def _system_with_worker():
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._worker = _SeqWorker()
    s._cam_seq = -1
    s._cam_tex = None
    s._cam_tex_size = (0, 0)
    s._fallback_tex = object()
    s._flip_horizontal = s._flip_vertical = False
    s._brightness = s._contrast = 1.0
    return s


def test_unchanged_frame_skips_the_upload_entirely() -> None:
    """The 2026-08-08 fault: the camera runs ~30fps while the display
    attempts 60, so most rendered frames re-uploaded a byte-identical
    frame — ~490 MB/s of memory traffic that only fell over once the 4K
    mirror composite was competing for the same bus."""
    uploaded: list[int] = []

    class _Tex:
        size = (4, 4)
        def write(self, _data): uploaded.append(1)
        def release(self): pass

    s = _system_with_worker()
    s._cam_tex = _Tex()
    s._cam_tex_size = (4, 4)

    s._worker.bump()                       # one new camera frame
    wo.WebcamSystem._get_camera_texture(s)
    assert uploaded == [1]

    for _ in range(10):                    # ten renders, no new camera frame
        wo.WebcamSystem._get_camera_texture(s)
    assert uploaded == [1], 'must not re-upload an unchanged frame'

    s._worker.bump()                       # next camera frame arrives
    wo.WebcamSystem._get_camera_texture(s)
    assert uploaded == [1, 1]


def test_stale_frame_returns_the_existing_texture_not_the_fallback() -> None:
    """Skipping the upload must keep showing the camera, not flash the
    placeholder."""
    class _Tex:
        size = (4, 4)
        def write(self, _data): pass
        def release(self): pass

    s = _system_with_worker()
    tex = _Tex()
    s._cam_tex = tex
    s._cam_tex_size = (4, 4)
    s._cam_seq = 0                          # already current
    assert wo.WebcamSystem._get_camera_texture(s) is tex


def test_worker_restart_forces_a_fresh_upload() -> None:
    """A new worker restarts its own sequence, so the cached one must be
    dropped or the new camera's first frames look 'already uploaded'."""
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._cam_seq = 47
    s._available_cameras = []
    s._worker = None
    s._active_device = None
    wo.WebcamSystem._restart_worker(s)
    assert s._cam_seq == -1
