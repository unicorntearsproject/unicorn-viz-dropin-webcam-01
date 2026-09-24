"""Per-camera settings: each camera keeps its own brightness/contrast/flips/
segmentation, within a session and across restarts (2026-09-23).

Found live with three USB cameras (C615 /dev/video0, C922 /dev/video2,
generic /dev/video4, each with a metadata node at N+1):
- REDISC while a camera was live dropped it from the list (a V4L2 camera
  we stream from refuses the probe's second open), desyncing NEXT/PREV
  from the camera on air for the rest of the session.
- the capture worker fell through to N+1/N+2 when N would not open, i.e.
  aired the *next physical camera* under camera N's settings.
- a camera with a partial settings entry inherited the previous camera's
  live values for the missing keys.
- per-camera state was keyed by /dev/videoN (renumbers with USB ports),
  never saved segmentation mode, only saved on clean shutdown when changed
  from the Control Room, and saved a camera-switch blink as "hidden".

Hermetic: the real constructor over a fake GL context, with the capture
worker, device probe and udev identity scan all replaced -- no camera is
ever opened.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webcam_overlay as wo  # noqa: E402


class _FakeTexture:
    filter = None

    def write(self, *_a, **_kw) -> None: pass
    def use(self, *_a, **_kw) -> None: pass
    def release(self) -> None: pass


class _FakeCtx:
    def texture(self, *_a, **_kw): return _FakeTexture()
    def program(self, **_kw): return {}
    def buffer(self, *_a, **_kw): return object()
    def vertex_array(self, *_a, **_kw): return object()


class _FakeWorker:
    def __init__(self, device, *_a, **_kw) -> None:
        self.device = device
        self.alive = True

    def start(self) -> None: pass
    def is_alive(self) -> bool: return self.alive
    def stop(self) -> None: self.alive = False
    def join(self, timeout=None) -> None: pass
    def request_device(self, device) -> None: self.device = device
    def set_selfie_seg_mode(self, _mode) -> None: pass
    def set_selfie_seg_tuning(self, *_a) -> None: pass


class _FakeVJApi:
    def __init__(self) -> None:
        self.saved: dict | None = None
        self.saves = 0

    def set_runtime_state(self, key, value) -> bool:
        assert key == 'webcam'
        self.saved = value
        self.saves += 1
        return True


class _Rig:
    """Mutable stand-ins for what's plugged in right now."""

    def __init__(self) -> None:
        self.detected = [0, 2, 4]
        self.ids = {0: 'id:C615', 2: 'id:C922', 4: 'id:Generic'}


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> _Rig:
    r = _Rig()
    monkeypatch.setattr(wo, '_CV2', True)
    monkeypatch.setattr(wo, '_CameraWorker', _FakeWorker)
    monkeypatch.setattr(wo, '_scan_camera_identities', lambda: dict(r.ids))
    monkeypatch.setattr(wo.WebcamSystem, '_enumerate_cameras', staticmethod(lambda: list(r.detected)))
    return r


def _boot(payload: dict | None = None, show: bool = True) -> 'wo.WebcamSystem':
    s = wo.WebcamSystem(_FakeCtx(), 1920, 1080, {'pip_position': 'hidden'})
    s.apply_persistence_state(payload or {})
    s._vj_api = _FakeVJApi()
    s.start()
    if show:
        s.set_layout('3')
    return s


def _look(s: 'wo.WebcamSystem') -> tuple:
    return (s._active_device, round(s._brightness, 2), s._flip_horizontal, s._flip_vertical)


def _act(s: 'wo.WebcamSystem', *actions: str) -> None:
    for action in actions:
        s._operator_panel_action(action, None)


# ------------------------------------------------------------- within a session

def test_each_camera_keeps_its_own_settings_across_switches(rig) -> None:
    s = _boot()
    _act(s, 'brightness_up', 'flip_v')                 # camera 0
    _act(s, 'next_camera', 'flip_h')                   # camera 2
    _act(s, 'next_camera')                             # camera 4, untouched
    assert _look(s) == (4, 1.0, True, False)           # configured defaults
    _act(s, 'prev_camera')
    assert _look(s) == (2, 1.0, False, False)
    _act(s, 'prev_camera')
    assert _look(s) == (0, 1.05, True, True)


def test_missing_keys_fall_back_to_defaults_not_the_previous_camera(rig) -> None:
    s = _boot()
    _act(s, 'brightness_up', 'brightness_up', 'flip_v')          # camera 0 set up
    s.set_camera_selfie_seg(2, 'dim')                            # camera 2: seg entry only
    _act(s, 'next_camera')
    assert _look(s) == (2, 1.0, True, False)
    assert s._selfie_seg == 'dim'


def test_rediscover_while_live_keeps_the_camera_on_air(rig) -> None:
    s = _boot()
    rig.detected = [2, 4]                  # camera 0 refuses the probe: we hold it
    _act(s, 'rediscover')
    assert s._available_cameras == [0, 2, 4]
    assert s._available_cameras[s._camera_list_index] == 0
    _act(s, 'next_camera')
    assert s._active_device == 2


def test_rediscover_without_a_live_worker_adds_nothing(rig) -> None:
    s = _boot(show=False)
    s._active_device = 0
    rig.detected = [2, 4]
    s.rediscover_cameras()
    assert s._available_cameras == [2, 4]


def test_worker_opens_only_the_requested_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wo, '_CV2', True)
    tried: list[int] = []

    def _try_open(self, dev):
        tried.append(dev)
        return None

    monkeypatch.setattr(wo._CameraWorker, '_try_open', _try_open)
    wo._CameraWorker(2, 640, 360, 30).run()    # initial open fails -> gives up
    assert tried == [2]                         # never 3 (metadata) or 4 (next camera)


# ------------------------------------------------------------------ saving now

def test_panel_actions_save_immediately(rig) -> None:
    s = _boot()
    _act(s, 'flip_v')
    assert s._vj_api.saves == 1
    assert s._vj_api.saved['per_camera_by_id']['id:C615']['flip_vertical'] is True


def test_unknown_panel_action_does_not_save(rig) -> None:
    s = _boot()
    assert s._operator_panel_action('nonsense', None) is None
    assert s._vj_api.saves == 0


def test_segmentation_mode_saves_and_round_trips(rig) -> None:
    s = _boot()
    s.set_camera_selfie_seg(2, 'blur')
    saved = s._vj_api.saved
    assert saved['per_camera_by_id']['id:C922']['selfie_seg'] == 'blur'
    s2 = _boot(saved)
    _act(s2, 'next_camera')
    assert s2._active_device == 2 and s2._selfie_seg == 'blur'


def test_camera_switch_blink_is_not_saved_as_hidden(rig) -> None:
    s = _boot()
    _act(s, 'next_camera')
    assert s._hidden is True                    # the transient switch hide
    assert s._vj_api.saved['layout'] == 'bottom_right'


def test_deliberate_hide_is_saved(rig) -> None:
    s = _boot()
    s.set_layout('hide')
    assert s.get_persistence_state()['layout'] == 'hide'


# ---------------------------------------------------------------- across restarts

def test_restart_with_renumbered_usb_restores_each_camera(rig) -> None:
    s = _boot()
    _act(s, 'brightness_up', 'flip_v')          # C615 at 0
    _act(s, 'next_camera', 'flip_h')            # C922 at 2
    saved = s._vj_api.saved
    rig.ids = {0: 'id:C922', 2: 'id:Generic', 4: 'id:C615'}   # different ports
    s2 = _boot(saved)
    looks = {}
    for _ in range(3):
        looks[rig.ids[s2._active_device]] = _look(s2)[1:]
        _act(s2, 'next_camera')
    assert looks == {
        'id:C922': (1.0, False, False),
        'id:Generic': (1.0, True, False),
        'id:C615': (1.05, True, True),
    }


def test_selected_camera_is_restored_by_identity(rig) -> None:
    s = _boot()
    _act(s, 'next_camera')                      # C922 selected
    saved = s._vj_api.saved
    rig.ids = {0: 'id:C922', 2: 'id:C615', 4: 'id:Generic'}
    s2 = _boot(saved)
    assert s2._active_device == 0


def test_disabled_camera_is_restored_by_identity(rig) -> None:
    s = _boot()
    s.set_camera_enabled(4, False)              # Generic off
    saved = s.get_persistence_state()
    rig.ids = {0: 'id:Generic', 2: 'id:C615', 4: 'id:C922'}
    s2 = _boot(saved)
    assert s2._disabled_cameras == {0}


def test_legacy_index_payload_still_loads(rig) -> None:
    legacy = {'selected_camera': 2, 'layout': 'hide', 'per_camera': {
        '2': {'brightness': 1.3, 'contrast': 1.0, 'flip_horizontal': False, 'flip_vertical': True},
    }}
    s = _boot(legacy)
    assert _look(s) == (2, 1.3, False, True)


def test_identity_payload_ignores_stale_legacy_index_fields(rig) -> None:
    payload = {
        'selected_camera': 4, 'selected_camera_id': 'id:C615',
        'per_camera': {'0': {'brightness': 1.9}},              # stale index data
        'per_camera_by_id': {'id:C615': {'brightness': 1.2}},
    }
    rig.ids = {0: 'id:C922', 2: 'id:C615', 4: 'id:Generic'}
    s = _boot(payload)
    assert _look(s)[:2] == (2, 1.2)
    _act(s, 'prev_camera')
    assert _look(s)[:2] == (0, 1.0)             # C922 never saw the 1.9


def test_unplugged_camera_keeps_its_saved_settings(rig) -> None:
    payload = {'per_camera_by_id': {'id:Webcam-at-home': {'brightness': 1.7}}}
    s = _boot(payload)
    _act(s, 'flip_v')
    assert s._vj_api.saved['per_camera_by_id']['id:Webcam-at-home']['brightness'] == 1.7


def test_replug_mid_session_moves_settings_with_the_camera(rig) -> None:
    s = _boot()
    _act(s, 'next_camera', 'next_camera', 'brightness_up', 'brightness_up')   # Generic at 4
    _act(s, 'prev_camera', 'prev_camera')                                     # back to C615
    rig.ids = {0: 'id:C615', 2: 'id:C922', 6: 'id:Generic'}                   # replugged
    rig.detected = [0, 2, 6]
    _act(s, 'rediscover')
    assert s._camera_settings[6]['brightness'] == pytest.approx(1.1)
    assert 4 not in s._camera_settings


# --------------------------------------------------------------- identity scan

def test_scan_camera_identities_prefers_by_id_and_skips_metadata_nodes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dev = tmp_path / 'dev'
    by_id, by_path = tmp_path / 'by-id', tmp_path / 'by-path'
    for d in (dev, by_id, by_path):
        d.mkdir()
    for n in range(6):
        (dev / f'video{n}').touch()
    (by_id / 'usb-C922_B2D9-video-index0').symlink_to(dev / 'video2')
    (by_id / 'usb-C922_B2D9-video-index1').symlink_to(dev / 'video3')
    (by_path / 'pci-usb-0:1.2-video-index0').symlink_to(dev / 'video2')
    (by_path / 'pci-usb-0:1.3-video-index0').symlink_to(dev / 'video4')  # no serial
    monkeypatch.setattr(wo, '_V4L_ID_DIRS', ((by_id, 'id:'), (by_path, 'path:')))
    assert wo._scan_camera_identities() == {
        2: 'id:usb-C922_B2D9',
        4: 'path:pci-usb-0:1.3',
    }


def test_scan_camera_identities_without_v4l_is_empty(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wo, '_V4L_ID_DIRS', ((tmp_path / 'missing', 'id:'),))
    assert wo._scan_camera_identities() == {}
