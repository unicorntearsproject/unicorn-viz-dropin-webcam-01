"""Regression tests for selfie-segmentation background cutout.

Covers: per-camera mode persistence (off/dim/blur/transparent), Config
Editor integration (the generic editor is slider-only, so the 4-way mode
selector rides a stepped 0-3 slider, and the grow/feather/blur radii are
separate global stepped sliders, all by the same convention used
elsewhere in the app), the grow-then-feather matte helper, and the
dim/blur/transparent output math (mocking MediaPipe's segmenter -- no
model file or camera needed).
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webcam_overlay as wo  # noqa: E402


# --------------------------------------------------------------- WebcamSystem

def _system() -> 'wo.WebcamSystem':
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._border_thickness = 8.0
    s._available_cameras = [0, 2]
    s._camera_settings = {}
    s._active_device = 0
    s._device = 0
    s._brightness = 1.0
    s._contrast = 1.0
    s._flip_horizontal = False
    s._flip_vertical = False
    s._default_brightness = 1.0
    s._default_contrast = 1.0
    s._default_flip_horizontal = False
    s._default_flip_vertical = False
    s._selfie_seg = 'off'
    s._default_selfie_seg = 'off'
    s._selfie_seg_grow_px = wo._SELFIE_SEG_GROW_PX_DEFAULT
    s._selfie_seg_feather_px = wo._SELFIE_SEG_FEATHER_PX_DEFAULT
    s._selfie_seg_blur_px = wo._SELFIE_SEG_BLUR_PX_DEFAULT
    s._selfie_seg_temporal = wo._SELFIE_SEG_TEMPORAL_DEFAULT
    s._border_style = 'futuristic'
    s._border_intensity = 1.0
    s._border_hue_speed = 0.05
    s._cam_fps = 30
    s._cam_width = 1280
    s._cam_height = 720
    s._auto_cycle_interval = 30.0
    s._worker = None
    return s


_TUNING_ROWS = [
    {'name': 'selfie_seg_grow_px', 'value': float(wo._SELFIE_SEG_GROW_PX_DEFAULT),
     'min': 0.0, 'max': 20.0, 'step': 1.0},
    {'name': 'selfie_seg_feather_px', 'value': float(wo._SELFIE_SEG_FEATHER_PX_DEFAULT),
     'min': 0.0, 'max': 30.0, 'step': 1.0},
    {'name': 'selfie_seg_blur_px', 'value': float(wo._SELFIE_SEG_BLUR_PX_DEFAULT),
     'min': 0.0, 'max': 100.0, 'step': 1.0},
    {'name': 'selfie_seg_temporal', 'value': float(wo._SELFIE_SEG_TEMPORAL_DEFAULT),
     'min': 0.0, 'max': 0.95},
    {'name': 'capture_fps', 'value': 2},
    {'name': 'capture_size', 'value': 1},
    {'name': 'cycle_interval', 'value': 30.0, 'min': 5.0, 'max': 120.0},
    {'name': 'border_style', 'value': float(wo._BORDER_STYLES.index('futuristic')),
     'min': 0.0, 'max': float(len(wo._BORDER_STYLES) - 1), 'step': 1.0},
    {'name': 'border_intensity', 'value': 1.0, 'min': 0.0, 'max': 2.0},
    {'name': 'border_hue_speed', 'value': 0.05, 'min': 0.0, 'max': 1.0},
    {'name': 'border_thickness', 'value': 8.0, 'min': 0.0, 'max': 32.0, 'step': 1.0},
]


_CORE_KEYS = ('name', 'value', 'min', 'max', 'step')


def _core(rows):
    """The 1.x row shape (the 2.0 presentation keys are covered separately)."""
    return [{k: r[k] for k in _CORE_KEYS if k in r} for r in rows]


def test_selfie_seg_defaults_off() -> None:
    s = _system()
    assert _core(s.config_editor_settings()) == [
        {'name': 'cam0_selfie_seg', 'value': 0.0, 'min': 0.0, 'max': 3.0, 'step': 1.0},
        {'name': 'cam2_selfie_seg', 'value': 0.0, 'min': 0.0, 'max': 3.0, 'step': 1.0},
        *_TUNING_ROWS,
    ]


def test_config_editor_settings_shows_default_device_when_hidden_at_startup() -> None:
    """pip_position = 'hidden' (the shipped default) means start() never
    calls rediscover_cameras() (camera privacy: no probing while hidden),
    so _available_cameras stays empty until the operator manually shows
    the camera once. The Visuals tab must still offer a selector for the
    configured device, or there's no way to pre-set the mode before first
    show."""
    s = _system()
    s._available_cameras = []
    s._active_device = None
    s._device = 0
    assert _core(s.config_editor_settings()) == [
        {'name': 'cam0_selfie_seg', 'value': 0.0, 'min': 0.0, 'max': 3.0, 'step': 1.0},
        *_TUNING_ROWS,
    ]


def test_config_editor_settings_includes_persisted_cameras_not_currently_detected() -> None:
    s = _system()
    s._available_cameras = []
    s._active_device = None
    s._device = 0
    s._camera_settings = {2: {'selfie_seg': 'transparent'}}
    result = _core(s.config_editor_settings())
    assert result == [
        {'name': 'cam0_selfie_seg', 'value': 0.0, 'min': 0.0, 'max': 3.0, 'step': 1.0},
        {'name': 'cam2_selfie_seg', 'value': 3.0, 'min': 0.0, 'max': 3.0, 'step': 1.0},
        *_TUNING_ROWS,
    ]


def test_set_camera_selfie_seg_updates_persisted_setting() -> None:
    s = _system()
    s.set_camera_selfie_seg(2, 'dim')
    assert s._camera_settings[2]['selfie_seg'] == 'dim'
    # Camera 2 isn't active -- the live mode must not change.
    assert s._selfie_seg == 'off'


def test_set_camera_selfie_seg_on_active_camera_flips_live_mode() -> None:
    s = _system()
    s.set_camera_selfie_seg(0, 'transparent')  # camera 0 is active (see _system())
    assert s._selfie_seg == 'transparent'


def test_set_camera_selfie_seg_rejects_unknown_mode() -> None:
    s = _system()
    s.set_camera_selfie_seg(0, 'bogus')
    assert s._selfie_seg == 'off'
    assert s._camera_settings[0]['selfie_seg'] == 'off'


def test_config_setting_round_trip() -> None:
    s = _system()
    s.set_config_setting('cam2_selfie_seg', 1.0)
    assert s._camera_settings[2]['selfie_seg'] == 'dim'
    s.set_config_setting('cam2_selfie_seg', 2.0)
    assert s._camera_settings[2]['selfie_seg'] == 'blur'
    s.set_config_setting('cam2_selfie_seg', 3.0)
    assert s._camera_settings[2]['selfie_seg'] == 'transparent'
    s.set_config_setting('cam2_selfie_seg', 0.0)
    assert s._camera_settings[2]['selfie_seg'] == 'off'


def test_config_setting_clamps_out_of_range_values() -> None:
    s = _system()
    s.set_config_setting('cam2_selfie_seg', 99.0)
    assert s._camera_settings[2]['selfie_seg'] == 'transparent'  # clamps to the top mode
    s.set_config_setting('cam2_selfie_seg', -5.0)
    assert s._camera_settings[2]['selfie_seg'] == 'off'  # clamps to the bottom mode


def test_config_setting_ignores_unrelated_names() -> None:
    s = _system()
    s.set_config_setting('intensity', 1.0)
    s.set_config_setting('camX_selfie_seg', 1.0)  # not an int -- must not raise
    assert s._camera_settings == {}


def test_config_setting_tunes_grow_feather_blur_temporal_globally() -> None:
    s = _system()
    s.set_config_setting('selfie_seg_grow_px', 7.0)
    s.set_config_setting('selfie_seg_feather_px', 12.0)
    s.set_config_setting('selfie_seg_blur_px', 60.0)
    s.set_config_setting('selfie_seg_temporal', 0.8)
    assert s._selfie_seg_grow_px == 7
    assert s._selfie_seg_feather_px == 12
    assert s._selfie_seg_blur_px == 60
    assert s._selfie_seg_temporal == 0.8
    # No worker attached -- must not raise (no live push target).
    assert s._worker is None


def test_config_setting_clamps_temporal_to_valid_range() -> None:
    s = _system()
    s.set_config_setting('selfie_seg_temporal', 5.0)
    assert s._selfie_seg_temporal == 0.95  # never lets the mask freeze entirely
    s.set_config_setting('selfie_seg_temporal', -1.0)
    assert s._selfie_seg_temporal == 0.0


def test_config_setting_pushes_tuning_to_live_worker() -> None:
    pushed = []

    class _FakeWorker:
        def set_selfie_seg_tuning(self, grow_px, feather_px, blur_px, temporal):
            pushed.append((grow_px, feather_px, blur_px, temporal))

    s = _system()
    s._worker = _FakeWorker()
    s.set_config_setting('selfie_seg_grow_px', 9.0)
    assert pushed == [(9, wo._SELFIE_SEG_FEATHER_PX_DEFAULT, wo._SELFIE_SEG_BLUR_PX_DEFAULT,
                        wo._SELFIE_SEG_TEMPORAL_DEFAULT)]


def test_remember_and_apply_camera_settings_round_trip_selfie_seg() -> None:
    s = _system()
    s._brightness = 1.0
    s._contrast = 1.0
    s._flip_horizontal = True
    s._flip_vertical = False
    s._selfie_seg = 'transparent'
    s._remember_current_camera_settings(0)
    assert s._camera_settings[0]['selfie_seg'] == 'transparent'

    s._selfie_seg = 'off'  # simulate switching away and back
    s._apply_camera_settings(0)
    assert s._selfie_seg == 'transparent'


def test_apply_camera_settings_falls_back_to_default_when_unknown() -> None:
    s = _system()
    s._default_selfie_seg = 'off'
    s._selfie_seg = 'dim'
    s._apply_camera_settings(99)  # never remembered
    assert s._selfie_seg == 'off'


# ---------------------------------------------------------------------- render

def test_render_skips_border_clear_in_transparent_mode() -> None:
    """A coloured backplate boxing in a cutout that's supposed to float
    free over the show reads wrong -- transparent mode must not call
    _clear_border_ring() at all. Every other mode still gets a border."""
    calls: list[bool] = []
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._switch_hide_active = False
    s._hidden = False
    s.enabled = True
    s._time = 0.0
    s._treatment_index = 0  # 'raw' -- current_treatment is a real property
    s._border_thickness = 4.0
    s._border_style = 'classic'  # exercise the classic _clear_border_ring path
    s._advance_auto_cycle = lambda dt: None
    s._get_camera_texture = lambda: SimpleNamespace(use=lambda location: None)
    s._rect_from_layout = lambda: (0.0, 0.0, 10.0, 10.0)
    s._write_cam_quad = lambda x, y, w, h: None
    s._clear_border_ring = lambda x, y, w, h, b: calls.append(True)
    s._programs = {'raw': {
        'cam_tex': SimpleNamespace(value=None), 'iBass': SimpleNamespace(value=None),
        'iTreble': SimpleNamespace(value=None), 'iTime': SimpleNamespace(value=None),
    }}
    s._cam_vao_map = {'raw': SimpleNamespace(render=lambda mode: None)}

    s._selfie_seg = 'transparent'
    wo.WebcamSystem.render(s, dt=0.016, bass=0.0, treble=0.0)
    assert calls == []

    s._selfie_seg = 'dim'
    wo.WebcamSystem.render(s, dt=0.016, bass=0.0, treble=0.0)
    assert calls == [True]


# ---------------------------------------------------------------- _feather_mask

def test_feather_mask_grows_foreground_past_original_edge() -> None:
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[:, :20] = 1.0  # left half is foreground
    alpha = wo._feather_mask(mask, grow_px=3, feather_px=4)
    assert alpha.shape == mask.shape
    # Deep foreground / deep background, far from any edge, stay near-pure.
    assert alpha[20, 5] > 0.95
    assert alpha[20, 35] < 0.05
    # A couple of px past the original edge, into what was background,
    # must now read as at least partially foreground -- proof the
    # selection grew outward rather than just being blurred in place.
    assert alpha[20, 21] > 0.5


def test_feather_mask_produces_a_smooth_ramp_not_a_hard_edge() -> None:
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[:, :20] = 1.0
    alpha = wo._feather_mask(mask, grow_px=3, feather_px=4)
    row = alpha[20, :]
    # A hard step would only ever take values near 0.0 or 1.0; feathering
    # must produce genuine intermediate values along the transition.
    intermediate = [v for v in row if 0.05 < v < 0.95]
    assert len(intermediate) >= 3


def test_feather_mask_zero_radii_is_a_no_op_binarize() -> None:
    mask = np.array([[0.2, 0.6], [0.9, 0.4]], dtype=np.float32)
    alpha = wo._feather_mask(mask, grow_px=0, feather_px=0)
    assert np.array_equal(alpha, np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32))


# ---------------------------------------------------------------- _CameraWorker

def _worker() -> 'wo._CameraWorker':
    import threading
    w = wo._CameraWorker.__new__(wo._CameraWorker)
    w._selfie_seg_lock = threading.Lock()
    w._selfie_seg_mode = 'dim'
    w._selfie_seg_grow_px = wo._SELFIE_SEG_GROW_PX_DEFAULT
    w._selfie_seg_feather_px = wo._SELFIE_SEG_FEATHER_PX_DEFAULT
    w._selfie_seg_blur_px = wo._SELFIE_SEG_BLUR_PX_DEFAULT
    w._selfie_seg_temporal = 0.0  # deterministic single-frame tests by default
    w._prev_raw_mask = None
    w._segmenter = None
    w._segmenter_failed = False
    return w


def test_set_selfie_seg_mode_is_thread_safe() -> None:
    w = _worker()
    w.set_selfie_seg_mode('transparent')
    assert w._selfie_seg_mode == 'transparent'
    w.set_selfie_seg_mode('off')
    assert w._selfie_seg_mode == 'off'


def test_set_selfie_seg_mode_rejects_unknown_value() -> None:
    w = _worker()
    w.set_selfie_seg_mode('bogus')
    assert w._selfie_seg_mode == 'off'


def test_set_selfie_seg_tuning_updates_radii_and_temporal() -> None:
    w = _worker()
    w.set_selfie_seg_tuning(1, 2, 3, 0.7)
    assert (w._selfie_seg_grow_px, w._selfie_seg_feather_px,
            w._selfie_seg_blur_px, w._selfie_seg_temporal) == (1, 2, 3, 0.7)


def test_set_selfie_seg_tuning_clamps_temporal() -> None:
    w = _worker()
    w.set_selfie_seg_tuning(0, 0, 0, 5.0)
    assert w._selfie_seg_temporal == 0.95


def test_get_segmenter_returns_none_and_remembers_when_mediapipe_absent(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_MEDIAPIPE', False)
    w = _worker()
    assert w._get_segmenter() is None
    assert w._segmenter_failed is True
    # Second call must not re-attempt (no mediapipe import touched again).
    assert w._get_segmenter() is None


def test_get_segmenter_returns_none_when_model_file_missing(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_MEDIAPIPE', True)
    monkeypatch.setattr(wo, '_SELFIE_SEG_MODEL_PATH', Path('/nonexistent/selfie_segmenter.tflite'))
    w = _worker()
    assert w._get_segmenter() is None
    assert w._segmenter_failed is True


def test_apply_selfie_seg_returns_frame_unchanged_when_segmenter_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_MEDIAPIPE', False)
    w = _worker()
    frame = np.full((4, 4, 3), 200, dtype=np.uint8)
    out = w._apply_selfie_seg(frame)
    assert np.array_equal(out, frame)


def _fake_segmenter_for(mask_2d: np.ndarray):
    class _FakeMask:
        def numpy_view(self):
            return mask_2d

    class _FakeResult:
        confidence_masks = [_FakeMask()]

    class _FakeSegmenter:
        def segment(self, mp_image):  # noqa: ARG002
            return _FakeResult()

    return _FakeSegmenter()


def _patch_mp(monkeypatch) -> None:
    monkeypatch.setattr(wo, '_MEDIAPIPE', True)
    monkeypatch.setattr(
        wo, '_mp',
        SimpleNamespace(
            Image=lambda image_format, data: data,  # noqa: ARG005
            ImageFormat=SimpleNamespace(SRGB=1),
        ),
    )


def test_apply_selfie_seg_dim_mode_dims_background_leaves_foreground(monkeypatch) -> None:
    """Foreground (mask=1, deep inside the region) must survive
    essentially untouched; background (mask=0, deep inside the region)
    must scale down toward _SELFIE_SEG_BG_DIM of its original value. The
    mask is deliberately larger than the grow+feather kernels so the
    plateaus away from the edge are unaffected by the transition."""
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[:, :20] = 1.0  # left half foreground, right half background

    _patch_mp(monkeypatch)
    w = _worker()
    w._selfie_seg_mode = 'dim'
    w._segmenter = _fake_segmenter_for(mask)  # skip _get_segmenter()'s lazy-load path
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    out = w._apply_selfie_seg(frame)

    assert out.shape == frame.shape  # dim mode stays 3-channel
    # Deep foreground column: untouched.
    assert np.allclose(out[20, 5, :], 200, atol=1)
    # Deep background column: dimmed to _SELFIE_SEG_BG_DIM of original.
    expected_bg = int(round(200 * wo._SELFIE_SEG_BG_DIM))
    assert np.allclose(out[20, 35, :], expected_bg, atol=2)


def test_apply_selfie_seg_updates_prev_raw_mask_for_next_frame(monkeypatch) -> None:
    mask = np.full((10, 10), 0.8, dtype=np.float32)
    _patch_mp(monkeypatch)
    w = _worker()
    w._selfie_seg_mode = 'dim'
    w._segmenter = _fake_segmenter_for(mask)
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    assert w._prev_raw_mask is None
    w._apply_selfie_seg(frame)
    assert w._prev_raw_mask is not None
    assert np.allclose(w._prev_raw_mask, 0.8)


def test_apply_selfie_seg_ignores_stale_prev_mask_of_different_shape(monkeypatch) -> None:
    """A resolution change (e.g. a device switch mid-session) must not try
    to blend masks of two different shapes."""
    _patch_mp(monkeypatch)
    w = _worker()
    w._selfie_seg_mode = 'dim'
    w._selfie_seg_temporal = 0.9
    w._prev_raw_mask = np.zeros((5, 5), dtype=np.float32)  # stale, wrong shape
    mask = np.ones((10, 10), dtype=np.float32)
    w._segmenter = _fake_segmenter_for(mask)
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    out = w._apply_selfie_seg(frame)
    # Must use this frame's mask directly (all foreground) rather than
    # crashing or blending against a mismatched shape.
    assert np.allclose(out, 100, atol=1)


def test_apply_selfie_seg_temporal_smoothing_retains_previous_frame_influence(monkeypatch) -> None:
    """With temporal smoothing on, a frame where the person briefly drops
    out of the mask must not immediately snap the matte to fully
    background -- the previous frame's mask should still carry weight.
    Without smoothing, the same sequence snaps instantly."""
    mask_frame1 = np.zeros((40, 40), dtype=np.float32)
    mask_frame1[:, :20] = 1.0  # left half foreground
    mask_frame2 = np.zeros((40, 40), dtype=np.float32)  # nothing detected this frame

    _patch_mp(monkeypatch)
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    probe = (20, 5)  # deep foreground in frame 1

    w_instant = _worker()
    w_instant._selfie_seg_mode = 'dim'
    w_instant._selfie_seg_temporal = 0.0
    w_instant._segmenter = _fake_segmenter_for(mask_frame1)
    w_instant._apply_selfie_seg(frame)
    w_instant._segmenter = _fake_segmenter_for(mask_frame2)
    out_instant = w_instant._apply_selfie_seg(frame)

    w_smoothed = _worker()
    w_smoothed._selfie_seg_mode = 'dim'
    w_smoothed._selfie_seg_temporal = 0.8
    w_smoothed._segmenter = _fake_segmenter_for(mask_frame1)
    w_smoothed._apply_selfie_seg(frame)
    w_smoothed._segmenter = _fake_segmenter_for(mask_frame2)
    out_smoothed = w_smoothed._apply_selfie_seg(frame)

    assert out_instant[probe][0] < 100        # instant: dimmed toward background
    assert out_smoothed[probe][0] > out_instant[probe][0]  # smoothed: still carries frame 1


def test_apply_selfie_seg_blur_mode_blurs_background_leaves_foreground(monkeypatch) -> None:
    """Blur mode must leave the deep foreground essentially untouched and
    replace the deep background with a blurred version -- i.e. a flat
    background must stay flat (nothing to blur into), so what actually
    distinguishes blur from dim here is that a *patterned* background
    gets smoothed out."""
    mask = np.zeros((60, 60), dtype=np.float32)
    mask[:, :30] = 1.0  # left half foreground, right half background

    _patch_mp(monkeypatch)
    w = _worker()
    w._selfie_seg_mode = 'blur'
    w._selfie_seg_blur_px = 10
    w._segmenter = _fake_segmenter_for(mask)
    frame = np.full((60, 60, 3), 128, dtype=np.uint8)
    # Sharp checkerboard noise on the background half only.
    frame[::2, 30:, :] = 250
    frame[1::2, 30:, :] = 10
    out = w._apply_selfie_seg(frame)

    assert out.shape == frame.shape  # blur mode stays 3-channel
    # Deep foreground: untouched.
    assert np.allclose(out[30, 5, :], 128, atol=1)
    # Deep background: the checkerboard's per-row 250/10 swing must have
    # been smoothed toward the middle -- no single pixel this far from
    # the edge should still read near either extreme.
    bg_column = out[:, 55, 0].astype(np.int16)
    assert bg_column.max() < 200
    assert bg_column.min() > 60


def test_apply_selfie_seg_blur_mode_with_zero_radius_is_a_no_op(monkeypatch) -> None:
    mask = np.ones((10, 10), dtype=np.float32)
    _patch_mp(monkeypatch)
    w = _worker()
    w._selfie_seg_mode = 'blur'
    w._selfie_seg_blur_px = 0
    w._segmenter = _fake_segmenter_for(mask)
    frame = np.full((10, 10, 3), 100, dtype=np.uint8)
    out = w._apply_selfie_seg(frame)
    assert np.array_equal(out, frame)


def test_apply_selfie_seg_transparent_mode_cuts_alpha(monkeypatch) -> None:
    """Transparent mode must return an RGBA frame: colour untouched
    everywhere, alpha near-opaque deep in the foreground and near-zero
    deep in the background."""
    mask = np.zeros((40, 40), dtype=np.float32)
    mask[:, :20] = 1.0

    _patch_mp(monkeypatch)
    w = _worker()
    w._selfie_seg_mode = 'transparent'
    w._segmenter = _fake_segmenter_for(mask)
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    out = w._apply_selfie_seg(frame)

    assert out.shape == (40, 40, 4)
    # Colour channels are never touched by transparent mode -- only alpha.
    assert np.array_equal(out[..., :3], frame)
    assert out[20, 5, 3] > 240   # deep foreground: near-opaque
    assert out[20, 35, 3] < 15   # deep background: near-transparent


def test_apply_selfie_seg_survives_a_bad_frame() -> None:
    """A segmenter that raises must not propagate -- return the original frame."""
    class _BrokenSegmenter:
        def segment(self, mp_image):  # noqa: ARG002
            raise RuntimeError('boom')

    w = _worker()
    w._segmenter = _BrokenSegmenter()
    frame = np.full((2, 2, 3), 123, dtype=np.uint8)
    out = w._apply_selfie_seg(frame)
    assert np.array_equal(out, frame)


def test_rows_carry_2_0_presentation_and_capture_rows_are_restart_rows() -> None:
    s = _system()
    rows = {r['name']: r for r in s.config_editor_settings()}
    cam = rows['cam0_selfie_seg']
    assert (cam['tab'], cam['kind'], cam['choices'], cam['label']) == (
        'Performance', 'choice', ('OFF', 'DIM', 'BLUR', 'TRANSPARENT'), 'Camera 0 segmentation')
    assert rows['selfie_seg_blur_px']['display'].endswith(' px')
    assert rows['capture_fps']['restart'] == 'webcam' and rows['capture_size']['restart'] == 'webcam'
    assert rows['capture_size']['choices'] == ('640x360', '1280x720', '1920x1080')
    assert 'tab' not in rows['border_style'] and rows['border_style']['kind'] == 'choice'
    assert s.set_config_setting('capture_fps', 3.0) == {'fps': 60}
    assert s._cam_fps == 60
    assert s.set_config_setting('capture_size', 0.0) == {'width': 640, 'height': 360}
    assert s.set_config_setting('cycle_interval', 40.0) is None
    assert (s._auto_cycle_interval, s._auto_cycle_effect_duration,
            s._auto_cycle_normal_duration) == (40.0, 20.0, 40.0)
    assert s.set_config_setting('selfie_seg_grow_px', 3.0) is None
