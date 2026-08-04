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
