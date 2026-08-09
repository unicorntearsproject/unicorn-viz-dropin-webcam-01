"""Regression tests for webcam treatment auto-cycle.

Pattern: normal -> effect -> normal -> effect -> ... An effect airs for
half of cycle_interval; the normal (raw) interlude between effects airs
for 2x that (the un-halved cycle_interval). Only auto-cycle behaves this
way -- manual next/prev_treatment() still round-robins flatly through
every treatment including raw.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import webcam_overlay as wo  # noqa: E402


def _system(cycle_interval: float = 30.0) -> 'wo.WebcamSystem':
    s = wo.WebcamSystem.__new__(wo.WebcamSystem)
    s._treatment_index = wo._TREATMENT_NAMES.index('raw')
    s._auto_cycle = False
    s._auto_cycle_interval = cycle_interval
    s._auto_cycle_timer = 0.0
    s._auto_cycle_effect_duration = cycle_interval / 2.0
    s._auto_cycle_normal_duration = cycle_interval
    s._auto_cycle_showing_effect = False
    s._auto_cycle_effect_seq_index = -1
    return s


def test_durations_derived_from_cycle_interval() -> None:
    s = _system(cycle_interval=30.0)
    assert s._auto_cycle_effect_duration == 15.0
    assert s._auto_cycle_normal_duration == 30.0


def test_starts_on_raw_and_stays_until_normal_duration_elapses() -> None:
    s = _system(cycle_interval=30.0)
    s._auto_cycle = True
    s._advance_auto_cycle(29.9)
    assert s.current_treatment == 'raw'
    assert s._auto_cycle_showing_effect is False


def test_normal_phase_advances_to_first_effect() -> None:
    s = _system(cycle_interval=30.0)
    s._auto_cycle = True
    s._advance_auto_cycle(30.0)
    assert s.current_treatment != 'raw'
    assert s._auto_cycle_showing_effect is True


def test_effect_phase_is_half_duration_then_returns_to_raw() -> None:
    s = _system(cycle_interval=30.0)
    s._auto_cycle = True
    s._advance_auto_cycle(30.0)          # normal -> effect1
    first_effect = s.current_treatment
    assert first_effect != 'raw'
    s._advance_auto_cycle(14.9)          # short of the 15s effect duration
    assert s.current_treatment == first_effect
    s._advance_auto_cycle(0.2)           # crosses 15s -> back to raw
    assert s.current_treatment == 'raw'
    assert s._auto_cycle_showing_effect is False


def test_full_sequence_alternates_normal_and_effect() -> None:
    s = _system(cycle_interval=30.0)
    s._auto_cycle = True
    seen: list[str] = [s.current_treatment]  # starts on raw
    for _ in range(8):
        # Step by exactly whichever duration is currently active so each
        # call crosses precisely one phase boundary.
        step = s._auto_cycle_effect_duration if s._auto_cycle_showing_effect else s._auto_cycle_normal_duration
        s._advance_auto_cycle(step)
        seen.append(s.current_treatment)
    # raw, effect, raw, effect, raw, effect, raw, effect, raw
    is_raw = [name == 'raw' for name in seen]
    assert is_raw == [True, False, True, False, True, False, True, False, True]


def test_cycle_effects_never_repeats_immediately_and_skips_raw() -> None:
    s = _system(cycle_interval=2.0)  # effect=1s, normal=2s -- fast cycle for the test
    s._auto_cycle = True
    effects_seen: list[str] = []
    for _ in range(20):
        s._advance_auto_cycle(1.0)
        if s._auto_cycle_showing_effect:
            effects_seen.append(s.current_treatment)
    assert effects_seen  # cycled through at least one effect
    assert all(name != 'raw' for name in effects_seen)
    # Consecutive effects (separated by a raw interlude) must differ --
    # walking the effect list in order, never re-showing the same one twice
    # running (only relevant once len(effects) > 1, true for this drop-in).
    for a, b in zip(effects_seen, effects_seen[1:]):
        assert a != b


def test_disabled_auto_cycle_never_advances() -> None:
    s = _system(cycle_interval=1.0)
    s._auto_cycle = False
    s._advance_auto_cycle(100.0)
    assert s.current_treatment == 'raw'
    assert s._auto_cycle_timer == 0.0


def test_toggle_auto_cycle_resets_to_normal_phase() -> None:
    s = _system(cycle_interval=2.0)
    s._auto_cycle = True
    s._advance_auto_cycle(2.0)  # normal duration is 2.0 here -> normal -> effect
    assert s._auto_cycle_showing_effect is True
    s.toggle_auto_cycle()  # off
    assert s._auto_cycle is False
    s.toggle_auto_cycle()  # back on
    assert s._auto_cycle is True
    assert s._auto_cycle_showing_effect is False
    assert s.current_treatment == 'raw'
    assert s._auto_cycle_timer == 0.0


def test_manual_next_treatment_still_includes_raw() -> None:
    """Manual cycling (Ctrl+KP*) is unaffected -- flat round robin, raw included."""
    s = _system()
    names = [s.current_treatment]
    for _ in range(len(wo._TREATMENT_NAMES)):
        names.append(s.next_treatment())
    assert 'raw' in names
    assert names[0] == names[-1]  # full loop back to start
