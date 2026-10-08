"""Tests for :mod:`app.pipeline.prosody`.

Signals are synthesised, so nothing is downloaded and the expected answer is known
exactly: a sine at a known frequency has a known pitch, and noise has none.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline import prosody
from app.pipeline.prosody import (
    F0_MAX_HZ,
    F0_MIN_HZ,
    InvalidAudioError,
    compare_profiles,
    estimate_f0,
    profile_of_file,
)

RATE = 24_000


def _tone(hz: float, seconds: float = 2.0, rate: int = RATE, amplitude: float = 0.4) -> np.ndarray:
    times = np.arange(int(seconds * rate), dtype=np.float64) / rate
    return (amplitude * np.sin(2.0 * np.pi * hz * times)).astype(np.float32)


def _sweep(low: float, high: float, seconds: float = 2.0, rate: int = RATE) -> np.ndarray:
    """A tone whose pitch moves, so it has a wide range by construction."""

    times = np.arange(int(seconds * rate), dtype=np.float64) / rate
    centre = (low + high) / 2.0
    depth = (high - low) / 2.0
    frequency = centre + depth * np.sin(2.0 * np.pi * 0.5 * times)
    return (0.4 * np.sin(2.0 * np.pi * np.cumsum(frequency) / rate)).astype(np.float32)


# ---------------------------------------------------------------------------
# pitch estimation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("hz", [65.0, 70.0, 80.0, 90.0, 100.0, 120.0, 150.0, 200.0, 250.0, 300.0, 350.0, 380.0])
def test_a_pure_tone_is_measured_at_its_own_frequency(hz: float) -> None:
    """Across the whole search range, including both ends.

    The low end is what a lag-dependent bias would break, and the high end is what
    an octave error would break, so both are checked explicitly.
    """

    profile = estimate_f0(_tone(hz), RATE)

    assert profile.median_hz is not None
    assert profile.median_hz == pytest.approx(hz, rel=0.06)


def test_a_frequency_above_the_range_is_reported_as_a_sub_multiple() -> None:
    """Documented limitation, asserted so it cannot change unnoticed.

    A pitch higher than :data:`F0_MAX_HZ` cannot be found, because its period is
    shorter than the shortest lag searched. What is found instead is a sub-multiple
    of it that falls inside the range. Nothing in film dialogue is above the
    ceiling, but the behaviour should be known rather than discovered.
    """

    assert F0_MIN_HZ < 65.0
    assert F0_MAX_HZ >= 380.0

    profile = estimate_f0(_tone(1000.0), RATE)

    assert profile.median_hz is not None
    assert F0_MIN_HZ <= profile.median_hz <= F0_MAX_HZ
    multiple = round(1000.0 / profile.median_hz)
    assert multiple >= 2
    assert profile.median_hz == pytest.approx(1000.0 / multiple, rel=0.02)


def test_noise_has_no_pitch() -> None:
    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(RATE * 2)).astype(np.float32)

    profile = estimate_f0(noise, RATE)

    assert profile.median_hz is None
    assert profile.is_voiced is False
    assert profile.voiced_ratio == pytest.approx(0.0, abs=0.05)


def test_silence_has_no_pitch_and_does_not_raise() -> None:
    profile = estimate_f0(np.zeros(RATE * 2, dtype=np.float32), RATE)

    assert profile.median_hz is None
    assert profile.speech_frames == 0
    assert profile.is_voiced is False


def test_audio_shorter_than_one_frame_is_not_pitched() -> None:
    profile = estimate_f0(_tone(120.0, seconds=0.005), RATE)

    assert profile.frames == 0
    assert profile.median_hz is None


def test_a_quieter_tone_measures_the_same_pitch() -> None:
    """Normalisation is by the frame's own energy, so level must not matter."""

    loud = estimate_f0(_tone(150.0, amplitude=0.8), RATE)
    quiet = estimate_f0(_tone(150.0, amplitude=0.02), RATE)

    assert loud.median_hz == pytest.approx(150.0, rel=0.06)
    assert quiet.median_hz == pytest.approx(150.0, rel=0.06)


def test_a_sustained_tone_is_still_measured() -> None:
    """A held note has no dynamic range, so a floor-relative test alone would miss it.

    That is what a long shout or a held vowel looks like, and reporting it as
    unpitched would make the preservation measurement blind to exactly the
    deliveries most likely to be flattened.
    """

    profile = estimate_f0(_tone(140.0), RATE)

    assert profile.is_voiced is True
    assert profile.voiced_ratio > 0.9


def test_a_moving_pitch_has_a_wide_range() -> None:
    moving = estimate_f0(_sweep(90.0, 180.0), RATE)
    flat = estimate_f0(_tone(135.0), RATE)

    assert moving.range_semitones is not None and flat.range_semitones is not None
    assert moving.range_semitones > flat.range_semitones * 2
    assert moving.range_semitones == pytest.approx(12.0, abs=6.0)


def test_other_sample_rates_are_resampled() -> None:
    """A 48 kHz take and its 24 kHz copy must measure the same pitch."""

    at_48 = estimate_f0(_tone(200.0, rate=48_000), 48_000)
    at_24 = estimate_f0(_tone(200.0), RATE)

    assert at_48.median_hz == pytest.approx(at_24.median_hz, rel=0.03)


def test_stereo_is_mixed_to_mono() -> None:
    mono = _tone(180.0)
    stereo = np.stack([mono, mono], axis=1)

    profile = estimate_f0(stereo, RATE)

    assert profile.median_hz == pytest.approx(180.0, rel=0.06)


def test_a_profile_is_json_safe() -> None:
    import json

    profile = estimate_f0(_tone(120.0), RATE)

    assert json.loads(json.dumps(profile.as_dict())) == profile.as_dict()


def test_an_impossible_sample_rate_is_rejected() -> None:
    with pytest.raises(InvalidAudioError, match="sample rate"):
        estimate_f0(_tone(120.0), 0)


def test_a_three_dimensional_signal_is_rejected() -> None:
    with pytest.raises(InvalidAudioError, match="1-D or 2-D"):
        estimate_f0(np.zeros((2, 2, 2), dtype=np.float32), RATE)


def test_measuring_a_missing_file_is_reported(tmp_path) -> None:
    with pytest.raises(InvalidAudioError, match="no audio to measure"):
        profile_of_file(tmp_path / "missing.wav")


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------


def test_an_identical_take_preserves_the_performance() -> None:
    profile = estimate_f0(_sweep(90.0, 180.0), RATE)
    comparison = compare_profiles(profile, profile)

    assert comparison.usable is True
    assert comparison.preserved is True
    assert comparison.median_shift_semitones == pytest.approx(0.0, abs=1e-9)
    assert comparison.range_ratio == pytest.approx(1.0)


def test_a_flattened_take_is_reported_as_not_preserved() -> None:
    """Losing the performance is what the measurement exists to catch."""

    performance = estimate_f0(_sweep(90.0, 180.0), RATE)
    flattened = estimate_f0(_tone(135.0), RATE)

    comparison = compare_profiles(performance, flattened)

    assert comparison.usable is True
    assert comparison.preserved is False
    assert comparison.range_ratio is not None and comparison.range_ratio < 0.5


def test_an_octave_error_is_reported_as_a_pitch_shift() -> None:
    comparison = compare_profiles(
        estimate_f0(_tone(120.0), RATE), estimate_f0(_tone(240.0), RATE)
    )

    assert comparison.median_shift_semitones == pytest.approx(12.0, abs=0.5)


def test_an_unpitched_take_is_not_scored() -> None:
    """A whisper or a silence cannot be compared, and saying so is the honest answer."""

    rng = np.random.default_rng(1)
    noise = (0.3 * rng.standard_normal(RATE * 2)).astype(np.float32)

    comparison = compare_profiles(estimate_f0(_sweep(90.0, 180.0), RATE), estimate_f0(noise, RATE))

    assert comparison.usable is False
    assert comparison.preserved is False
    assert comparison.median_shift_semitones is None
    assert comparison.range_ratio is None


def test_a_comparison_is_json_safe() -> None:
    import json

    comparison = compare_profiles(
        estimate_f0(_tone(120.0), RATE), estimate_f0(_tone(120.0), RATE)
    )

    assert json.loads(json.dumps(comparison.as_dict())) == comparison.as_dict()
