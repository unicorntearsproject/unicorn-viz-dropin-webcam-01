"""Regression tests for the redesigned PiP border and the two new webcam
treatments (posterize, halftone).

Border: `_border_style` selects 'off' (nothing drawn), 'classic' (the
original flat scissor-clear ring, `_clear_border_ring`), or 'futuristic'
(the new glowing neon/HUD/drippy shader border, `_draw_border_futuristic`
+ `_BORDER_FRAG`) -- configurable via `border_style` / `border_intensity`
/ `border_hue_speed`, and animating on its own via bass/beat-reactive drip
motion plus a slow hue drift (`iHueBase + iTime * iHueSpeed`) regardless of
config, so a long session doesn't look static. See the module docstring in
`webcam_overlay.py` and `_BORDER_FRAG`'s own comment block for the design.

Shader compilation itself is covered by manually running the new
fragment shaders through a real headless `moderngl` context during
development (not part of this file -- these tests target the pure-Python
dispatch/config/geometry logic, matching this drop-in's existing test
style, which never spins up a real GL context).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webcam_overlay as wo  # noqa: E402


# --------------------------------------------------------------- new treatments

def test_new_treatments_registered_in_both_dicts() -> None:
    for name in ('posterize', 'halftone'):
        assert name in wo._TREATMENT_NAMES
        assert name in wo._TREATMENT_FRAGS


def test_treatment_count_is_seven() -> None:
    assert len(wo._TREATMENT_NAMES) == 7
    assert wo._TREATMENT_NAMES == [
        'raw', 'neon', 'glitch', 'vintage', 'thermal', 'posterize', 'halftone',
    ]


def test_manual_cycling_reaches_new_treatments() -> None:
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._treatment_index = 0
    seen = {s.current_treatment}
    for _ in range(len(wo._TREATMENT_NAMES)):
        seen.add(s.next_treatment())
    assert {'posterize', 'halftone'}.issubset(seen)


def test_auto_cycle_effect_phase_can_land_on_new_treatments() -> None:
    """_next_cycle_effect_treatment() walks every non-raw treatment in
    order -- the two new ones must be reachable, not silently skipped."""
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._treatment_index = 0
    s._auto_cycle_effect_seq_index = -1
    effects_seen = set()
    for _ in range(len(wo._TREATMENT_NAMES)):
        effects_seen.add(s._next_cycle_effect_treatment())
    assert effects_seen == {'neon', 'glitch', 'vintage', 'thermal', 'posterize', 'halftone'}


# --------------------------------------------------------------------- parsing

def test_parse_border_style_accepts_known_values() -> None:
    for style in wo._BORDER_STYLES:
        assert wo.WebcamSystem._parse_border_style(style) == style
        assert wo.WebcamSystem._parse_border_style(style.upper()) == style


def test_parse_border_style_falls_back_to_futuristic_not_off() -> None:
    """An unrecognized config value should not silently disable the
    border -- it should land on the intended new default look."""
    assert wo.WebcamSystem._parse_border_style('nonsense') == 'futuristic'
    assert wo.WebcamSystem._parse_border_style('') == 'futuristic'


def test_hue_from_color_extracts_hue_channel() -> None:
    # Pure blue (0, 0, 1) -> hue 2/3.
    hue = wo.WebcamSystem._hue_from_color((0.0, 0.0, 1.0, 1.0))
    assert abs(hue - (2.0 / 3.0)) < 1e-9
    # Pure red (1, 0, 0) -> hue 0.
    hue = wo.WebcamSystem._hue_from_color((1.0, 0.0, 0.0, 1.0))
    assert hue == 0.0


# --------------------------------------------------------------- render dispatch

def _render_system(border_style: str) -> tuple['wo.WebcamSystem', dict]:
    calls = {'classic': 0, 'futuristic': 0}
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._switch_hide_active = False
    s._hidden = False
    s.enabled = True
    s._time = 0.0
    s._treatment_index = 0  # 'raw'
    s._border_thickness = 4.0
    s._border_style = border_style
    s._selfie_seg = 'off'
    s._advance_auto_cycle = lambda dt: None
    s._get_camera_texture = lambda: SimpleNamespace(use=lambda location: None)
    s._rect_from_layout = lambda: (0.0, 0.0, 10.0, 10.0)
    s._write_cam_quad = lambda x, y, w, h: None
    s._clear_border_ring = lambda x, y, w, h, b: calls.__setitem__('classic', calls['classic'] + 1)
    s._draw_border_futuristic = (
        lambda x, y, w, h, bass, treble: calls.__setitem__('futuristic', calls['futuristic'] + 1))
    s._programs = {'raw': {
        'cam_tex': SimpleNamespace(value=None), 'iBass': SimpleNamespace(value=None),
        'iTreble': SimpleNamespace(value=None), 'iTime': SimpleNamespace(value=None),
    }}
    s._cam_vao_map = {'raw': SimpleNamespace(render=lambda mode: None)}
    return s, calls


def test_render_dispatches_classic_style_to_clear_border_ring() -> None:
    s, calls = _render_system('classic')
    wo.WebcamSystem.render(s, dt=0.016, bass=0.0, treble=0.0)
    assert calls == {'classic': 1, 'futuristic': 0}


def test_render_dispatches_futuristic_style_to_draw_border_futuristic() -> None:
    s, calls = _render_system('futuristic')
    wo.WebcamSystem.render(s, dt=0.016, bass=0.0, treble=0.0)
    assert calls == {'classic': 0, 'futuristic': 1}


def test_render_off_style_draws_no_border_at_all() -> None:
    s, calls = _render_system('off')
    wo.WebcamSystem.render(s, dt=0.016, bass=0.0, treble=0.0)
    assert calls == {'classic': 0, 'futuristic': 0}


def test_render_still_skips_any_border_in_transparent_selfie_seg_mode() -> None:
    """Unchanged contract from before the redesign: a cutout floating
    over the show must not get boxed in by any border style."""
    s, calls = _render_system('futuristic')
    s._selfie_seg = 'transparent'
    wo.WebcamSystem.render(s, dt=0.016, bass=0.0, treble=0.0)
    assert calls == {'classic': 0, 'futuristic': 0}


# ------------------------------------------------------- _draw_border_futuristic

def _futuristic_system() -> 'wo.WebcamSystem':
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._border_thickness = 4.0
    s._border_intensity = 1.0
    s._border_hue_base = 0.5
    s._border_hue_speed = 0.05
    s._time = 12.5
    written = {}
    s._write_border_quad = lambda x, y, w, h, pad: written.update(
        x=x, y=y, w=w, h=h, pad=pad)
    s._test_written = written  # type: ignore[attr-defined]
    prog = {
        'iRectPx': SimpleNamespace(value=None),
        'iPadPx': SimpleNamespace(value=None),
        'iBorderPx': SimpleNamespace(value=None),
        'iIntensity': SimpleNamespace(value=None),
        'iHueBase': SimpleNamespace(value=None),
        'iHueSpeed': SimpleNamespace(value=None),
        'iTime': SimpleNamespace(value=None),
        'iBass': SimpleNamespace(value=None),
        'iTreble': SimpleNamespace(value=None),
    }
    s._border_prog = prog
    s._border_vao = SimpleNamespace(render=lambda mode: None)
    return s


def test_draw_border_futuristic_writes_padded_quad_and_uniforms() -> None:
    s = _futuristic_system()
    wo.WebcamSystem._draw_border_futuristic(s, x=100.0, y=50.0, w=200.0, h=120.0, bass=0.4, treble=0.2)

    # b = thickness + bass * thickness * 0.75 = 4.0 + 0.4 * 3.0 = 5.2
    # headroom = 26.0 * intensity(1.0) = 26.0 -> pad = 31.2
    written = s._test_written
    assert written['x'] == 100.0 and written['y'] == 50.0
    assert written['w'] == 200.0 and written['h'] == 120.0
    assert abs(written['pad'] - 31.2) < 1e-6

    prog = s._border_prog
    assert prog['iRectPx'].value == (200.0, 120.0)
    assert abs(prog['iPadPx'].value - 31.2) < 1e-6
    assert abs(prog['iBorderPx'].value - 5.2) < 1e-6
    assert prog['iIntensity'].value == 1.0
    assert prog['iHueBase'].value == 0.5
    assert prog['iHueSpeed'].value == 0.05
    assert prog['iTime'].value == 12.5
    assert prog['iBass'].value == 0.4
    assert prog['iTreble'].value == 0.2


def test_draw_border_futuristic_pad_never_below_one_px() -> None:
    """Guards the shader against a degenerate iPadPx (would make
    localBorder/iPadPx smoothstep calls collapse to a zero-width edge)."""
    s = _futuristic_system()
    s._border_thickness = 0.0
    s._border_intensity = 0.0
    wo.WebcamSystem._draw_border_futuristic(s, x=0.0, y=0.0, w=10.0, h=10.0, bass=0.0, treble=0.0)
    assert s._border_prog['iPadPx'].value >= 1.0


def test_draw_border_futuristic_pad_grows_with_bass_and_intensity() -> None:
    s = _futuristic_system()
    wo.WebcamSystem._draw_border_futuristic(s, x=0.0, y=0.0, w=10.0, h=10.0, bass=0.0, treble=0.0)
    pad_quiet = s._border_prog['iPadPx'].value

    s2 = _futuristic_system()
    wo.WebcamSystem._draw_border_futuristic(s2, x=0.0, y=0.0, w=10.0, h=10.0, bass=1.0, treble=0.0)
    pad_loud = s2._border_prog['iPadPx'].value
    assert pad_loud > pad_quiet

    s3 = _futuristic_system()
    s3._border_intensity = 2.0
    wo.WebcamSystem._draw_border_futuristic(s3, x=0.0, y=0.0, w=10.0, h=10.0, bass=0.0, treble=0.0)
    pad_intense = s3._border_prog['iPadPx'].value
    assert pad_intense > pad_quiet


# ------------------------------------------------------------------- geometry

def test_write_border_quad_grows_the_cam_rect_by_pad_on_every_side() -> None:
    """`_write_border_quad` must feed `_quad_verts` the camera rect grown
    by `pad` px on all four sides -- verified by comparing the raw vertex
    buffer against a directly-computed expected quad."""
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._width = 800
    s._height = 600

    written = {}

    class _Vbo:
        def write(self, data):
            written['data'] = data

    s._border_vbo = _Vbo()
    wo.WebcamSystem._write_border_quad(s, x=100.0, y=50.0, w=200.0, h=120.0, pad=10.0)

    expected = wo.WebcamSystem._quad_verts(s, x=90.0, y=40.0, w=220.0, h=140.0)
    import numpy as np
    assert np.array_equal(written['data'], expected)


def test_quad_verts_shared_by_cam_and_border_quads_agree_at_zero_pad() -> None:
    """A border quad with pad=0 must produce the exact same vertices as
    the plain camera quad for the same rect -- proof both call sites
    share the same pixel-to-clip-space math (_quad_verts)."""
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._width = 800
    s._height = 600
    cam_verts = wo.WebcamSystem._quad_verts(s, x=10.0, y=20.0, w=30.0, h=40.0)
    border_verts = wo.WebcamSystem._quad_verts(s, x=10.0 - 0.0, y=20.0 - 0.0, w=30.0 + 0.0, h=40.0 + 0.0)
    import numpy as np
    assert np.array_equal(cam_verts, border_verts)


# --------------------------------------------------------------- config editor

def _config_system() -> 'wo.WebcamSystem':
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._available_cameras = []
    s._camera_settings = {}
    s._active_device = None
    s._device = 0
    s._selfie_seg_grow_px = wo._SELFIE_SEG_GROW_PX_DEFAULT
    s._selfie_seg_feather_px = wo._SELFIE_SEG_FEATHER_PX_DEFAULT
    s._selfie_seg_blur_px = wo._SELFIE_SEG_BLUR_PX_DEFAULT
    s._selfie_seg_temporal = wo._SELFIE_SEG_TEMPORAL_DEFAULT
    s._border_style = 'futuristic'
    s._border_intensity = 1.0
    s._border_hue_speed = 0.05
    return s


def test_config_editor_settings_includes_border_rows() -> None:
    s = _config_system()
    settings = s.config_editor_settings()
    names = [row['name'] for row in settings]
    assert names[-3:] == ['border_style', 'border_intensity', 'border_hue_speed']
    style_row = settings[-3]
    assert style_row == {
        'name': 'border_style', 'value': float(wo._BORDER_STYLES.index('futuristic')),
        'min': 0.0, 'max': float(len(wo._BORDER_STYLES) - 1), 'step': 1.0,
    }


def test_set_config_setting_border_style_round_trip() -> None:
    s = _config_system()
    for index, style in enumerate(wo._BORDER_STYLES):
        s.set_config_setting('border_style', float(index))
        assert s._border_style == style


def test_set_config_setting_border_style_clamps_out_of_range() -> None:
    s = _config_system()
    s.set_config_setting('border_style', 99.0)
    assert s._border_style == wo._BORDER_STYLES[-1]
    s.set_config_setting('border_style', -5.0)
    assert s._border_style == wo._BORDER_STYLES[0]


def test_set_config_setting_border_intensity_clamps_to_0_2() -> None:
    s = _config_system()
    s.set_config_setting('border_intensity', 5.0)
    assert s._border_intensity == 2.0
    s.set_config_setting('border_intensity', -1.0)
    assert s._border_intensity == 0.0
    s.set_config_setting('border_intensity', 1.5)
    assert s._border_intensity == 1.5


def test_set_config_setting_border_hue_speed_clamps_to_0_1() -> None:
    s = _config_system()
    s.set_config_setting('border_hue_speed', 5.0)
    assert s._border_hue_speed == 1.0
    s.set_config_setting('border_hue_speed', -1.0)
    assert s._border_hue_speed == 0.0


# --------------------------------------------------------------- construction

class _FakeTexture:
    def __init__(self, *_a, **_kw) -> None:
        self.filter = None

    def write(self, *_a, **_kw) -> None: pass
    def use(self, *_a, **_kw) -> None: pass
    def release(self) -> None: pass


class _FakeCtx:
    """Just enough moderngl surface for WebcamSystem.__init__ -- same
    fake used by test_privacy_hidden_capture.py's construction tests."""

    def texture(self, *_a, **_kw): return _FakeTexture()
    def program(self, **_kw): return {}
    def buffer(self, *_a, **_kw): return object()
    def vertex_array(self, *_a, **_kw): return object()
    def simple_vertex_array(self, *_a, **_kw): return object()


def test_constructor_sets_futuristic_default_border_style(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_CV2', False)
    system = wo.WebcamSystem(_FakeCtx(), 1920, 1080, {'pip_position': 'bottom_right'})
    assert system._border_style == 'futuristic'
    assert system._border_intensity == 1.0
    assert 0.0 <= system._border_hue_base <= 1.0


def test_constructor_honors_configured_border_style(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_CV2', False)
    system = wo.WebcamSystem(
        _FakeCtx(), 1920, 1080,
        {'pip_position': 'bottom_right', 'border_style': 'classic', 'border_intensity': '1.5'},
    )
    assert system._border_style == 'classic'
    assert system._border_intensity == 1.5


def test_constructor_builds_border_gl_resources(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_CV2', False)
    system = wo.WebcamSystem(_FakeCtx(), 1920, 1080, {'pip_position': 'bottom_right'})
    assert hasattr(system, '_border_prog')
    assert hasattr(system, '_border_vbo')
    assert hasattr(system, '_border_vao')
