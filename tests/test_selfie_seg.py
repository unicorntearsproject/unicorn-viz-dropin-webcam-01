"""Regression tests for selfie-segmentation background dim.

Covers: per-camera setting persistence, Config Editor integration (the
generic editor is slider-only, so the on/off toggle rides a 0/1 slider by
the same convention used elsewhere in the app), and the actual dim math
(mocking MediaPipe's segmenter -- no model file or camera needed).
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
    s._selfie_seg = False
    s._default_selfie_seg = False
    s._worker = None
    return s


def test_selfie_seg_defaults_off() -> None:
    s = _system()
    assert s.config_editor_settings() == [
        {'name': 'cam0_selfie_seg', 'value': 0.0, 'min': 0.0, 'max': 1.0},
        {'name': 'cam2_selfie_seg', 'value': 0.0, 'min': 0.0, 'max': 1.0},
    ]


def test_set_camera_selfie_seg_updates_persisted_setting() -> None:
    s = _system()
    s.set_camera_selfie_seg(2, True)
    assert s._camera_settings[2]['selfie_seg'] is True
    # Camera 2 isn't active -- the live flag must not change.
    assert s._selfie_seg is False


def test_set_camera_selfie_seg_on_active_camera_flips_live_flag() -> None:
    s = _system()
    s.set_camera_selfie_seg(0, True)  # camera 0 is active (see _system())
    assert s._selfie_seg is True


def test_config_setting_round_trip() -> None:
    s = _system()
    s.set_config_setting('cam2_selfie_seg', 1.0)
    assert s._camera_settings[2]['selfie_seg'] is True
    s.set_config_setting('cam2_selfie_seg', 0.0)
    assert s._camera_settings[2]['selfie_seg'] is False


def test_config_setting_ignores_unrelated_names() -> None:
    s = _system()
    s.set_config_setting('intensity', 1.0)
    s.set_config_setting('camX_selfie_seg', 1.0)  # not an int -- must not raise
    assert s._camera_settings == {}


def test_remember_and_apply_camera_settings_round_trip_selfie_seg() -> None:
    s = _system()
    s._brightness = 1.0
    s._contrast = 1.0
    s._flip_horizontal = True
    s._flip_vertical = False
    s._selfie_seg = True
    s._remember_current_camera_settings(0)
    assert s._camera_settings[0]['selfie_seg'] is True

    s._selfie_seg = False  # simulate switching away and back
    s._apply_camera_settings(0)
    assert s._selfie_seg is True


def test_apply_camera_settings_falls_back_to_default_when_unknown() -> None:
    s = _system()
    s._default_selfie_seg = False
    s._selfie_seg = True
    s._apply_camera_settings(99)  # never remembered
    assert s._selfie_seg is False


# ---------------------------------------------------------------- _CameraWorker

def _worker() -> 'wo._CameraWorker':
    import threading
    w = wo._CameraWorker.__new__(wo._CameraWorker)
    w._selfie_seg_lock = threading.Lock()
    w._selfie_seg_enabled = False
    w._segmenter = None
    w._segmenter_failed = False
    return w


def test_set_selfie_seg_is_thread_safe_toggle() -> None:
    w = _worker()
    w.set_selfie_seg(True)
    assert w._selfie_seg_enabled is True
    w.set_selfie_seg(False)
    assert w._selfie_seg_enabled is False


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


def test_apply_selfie_seg_dims_background_leaves_foreground(monkeypatch) -> None:
    """Foreground (mask=1) pixels must survive untouched; background
    (mask=0) pixels must scale to _SELFIE_SEG_BG_DIM of their original value."""
    class _FakeMask:
        def numpy_view(self):
            # Left column = foreground (1.0), right column = background (0.0).
            return np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

    class _FakeResult:
        confidence_masks = [_FakeMask()]

    class _FakeSegmenter:
        def segment(self, mp_image):  # noqa: ARG002
            return _FakeResult()

    monkeypatch.setattr(wo, '_MEDIAPIPE', True)
    monkeypatch.setattr(
        wo, '_mp',
        SimpleNamespace(
            Image=lambda image_format, data: data,  # noqa: ARG005
            ImageFormat=SimpleNamespace(SRGB=1),
        ),
    )

    w = _worker()
    w._segmenter = _FakeSegmenter()  # skip _get_segmenter()'s lazy-load path
    frame = np.full((2, 2, 3), 200, dtype=np.uint8)
    out = w._apply_selfie_seg(frame)

    # Foreground column (index 0) untouched.
    assert np.array_equal(out[:, 0, :], frame[:, 0, :])
    # Background column (index 1) dimmed to _SELFIE_SEG_BG_DIM of original.
    expected_bg = int(round(200 * wo._SELFIE_SEG_BG_DIM))
    assert np.allclose(out[:, 1, :], expected_bg, atol=1)


def test_apply_selfie_seg_survives_a_bad_frame(monkeypatch) -> None:
    """A segmenter that raises must not propagate -- return the original frame."""
    class _BrokenSegmenter:
        def segment(self, mp_image):  # noqa: ARG002
            raise RuntimeError('boom')

    monkeypatch.setattr(wo, '_MEDIAPIPE', True)
    monkeypatch.setattr(
        wo, '_mp',
        SimpleNamespace(
            Image=lambda image_format, data: data,  # noqa: ARG005
            ImageFormat=SimpleNamespace(SRGB=1),
        ),
    )
    w = _worker()
    w._segmenter = _BrokenSegmenter()
    frame = np.full((2, 2, 3), 123, dtype=np.uint8)
    out = w._apply_selfie_seg(frame)
    assert np.array_equal(out, frame)
