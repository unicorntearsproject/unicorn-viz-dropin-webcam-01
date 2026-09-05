"""webcam-01's Control Room operator panel (P2 of
docs/planning/control-room-panel-registry-plan-2026-09-04.md).

Replaces the old hardcoded CR "WEBCAM" panel, which always drew full
controls against vj_api.list_webcam_cameras()/webcam_image_state() even
with no webcam-01 loaded and nothing behind them.
_operator_panel_content()/_operator_panel_action() read/act through the
same list_cameras()/webcam_image_state()/next_camera()/etc. every other
integration already uses. Same object.__new__() + manual-attrs pattern
as test_auto_cycle.py -- no capture worker, no threading, no hardware.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webcam_overlay as wo  # noqa: E402

from unicornviz.operator_panels import OperatorPanel, PanelContent  # noqa: E402


def _system(**overrides) -> 'wo.WebcamSystem':
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s.enabled = True
    s._detected_cameras = [0]
    s._active_device = 0
    s._available_cameras = [0]
    s._camera_list_index = 0
    s._disabled_cameras = set()
    s._brightness = 1.0
    s._contrast = 1.0
    s._flip_horizontal = False
    s._flip_vertical = False
    s._switch_hide_active = False
    s._switch_hide_deadline = 0.0
    for k, v in overrides.items():
        setattr(s, k, v)
    return s


class _FakeVJApi:
    def __init__(self) -> None:
        self.panels: dict[str, OperatorPanel] = {}

    def register_midi_actions(self, section, actions) -> None:
        pass

    def register_midi_action_handler(self, action, fn) -> None:
        pass

    def register_operator_panel(self, panel: OperatorPanel) -> None:
        self.panels[panel.name] = panel


def test_registration_call_registers_an_operator_panel() -> None:
    s = _system()
    api = _FakeVJApi()
    s.set_vj_api(api)
    panel = api.panels['webcam']
    assert panel.title == 'Webcam' and panel.page == 'main' and panel.size == 'full'
    assert panel.content == s._operator_panel_content
    assert panel.on_action == s._operator_panel_action
    assert panel.tooltips['prev_camera'] == 'Switch to the previous detected camera'


def test_content_reflects_active_camera_and_brightness_contrast() -> None:
    s = _system(_brightness=1.25, _contrast=0.75)
    content = s._operator_panel_content()
    assert isinstance(content, PanelContent)
    assert content.status == 'ON'
    by_label = {row.label: row.value for row in content.rows}
    expected_label = '/dev/video0' if sys.platform != 'win32' else '0'
    assert by_label['Active'] == expected_label
    assert by_label['B / C'] == '1.25 / 0.75'
    assert 'Switching' not in by_label


def test_content_shows_switching_instead_of_brightness_contrast() -> None:
    s = _system(_switch_hide_active=True, _switch_hide_deadline=time.monotonic() + 0.4)
    content = s._operator_panel_content()
    by_label = {row.label: row.value for row in content.rows}
    assert 'Switching' in by_label
    assert 'B / C' not in by_label
    switching_row = next(r for r in content.rows if r.label == 'Switching')
    assert switching_row.emphasis == 'warn'


def test_content_active_dash_with_no_cameras() -> None:
    s = _system(_detected_cameras=[], _active_device=None, _available_cameras=[])
    content = s._operator_panel_content()
    by_label = {row.label: row.value for row in content.rows}
    assert by_label['Active'] == '-'


def test_content_flip_buttons_reflect_state() -> None:
    s = _system(_flip_horizontal=True, _flip_vertical=False)
    content = s._operator_panel_content()
    flip_h = next(b for b in content.buttons if b.action == 'flip_h')
    flip_v = next(b for b in content.buttons if b.action == 'flip_v')
    assert flip_h.active is True and flip_v.active is False
    assert flip_h.accent == 'accent_2'


def test_content_status_off_when_disabled() -> None:
    s = _system(enabled=False)
    content = s._operator_panel_content()
    assert content.status == 'OFF'


# --------------------------------------------------------------------------- #
# _operator_panel_action -- routes to the real camera/image methods
# --------------------------------------------------------------------------- #

def test_action_prev_next_camera_report_the_new_label() -> None:
    s = _system()
    s.next_camera = lambda: '/dev/video1'
    assert s._operator_panel_action('next_camera', None) == 'Camera: /dev/video1'
    s.prev_camera = lambda: None
    assert s._operator_panel_action('prev_camera', None) == 'Camera: only one device'


def test_action_rediscover_reports_count() -> None:
    s = _system()
    s.rediscover_cameras = lambda: [0, 1, 2]
    assert s._operator_panel_action('rediscover', None) == 'Webcam rediscover: 3 found'


def test_action_brightness_contrast_report_new_value() -> None:
    s = _system()
    s.adjust_brightness = lambda delta: 1.30
    assert s._operator_panel_action('brightness_up', None) == 'Webcam brightness 1.30'
    s.adjust_contrast = lambda delta: 0.70
    assert s._operator_panel_action('contrast_down', None) == 'Webcam contrast 0.70'


def test_action_flip_reports_new_state() -> None:
    s = _system()
    s.toggle_flip_horizontal = lambda: True
    assert s._operator_panel_action('flip_h', None) == 'Webcam flip H: ON'
    s.toggle_flip_vertical = lambda: False
    assert s._operator_panel_action('flip_v', None) == 'Webcam flip V: OFF'


def test_action_unknown_returns_none() -> None:
    s = _system()
    assert s._operator_panel_action('bogus', None) is None
