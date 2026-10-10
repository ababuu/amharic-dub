"""The original dialogue must not be left in the bed that gets re-mixed.

The music and effects stems are summed back in untouched, so the only way the source
language can reach a finished dub is the separator leaving it there. On the GPU run
under investigation this was checked line by line against the real stems: 0 of 42
dialogue windows carried the original speech above -56 dBFS. The measurement is now
part of every run, because the alternative is hearing about it from a viewer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipeline import separation
from app.pipeline.separation import StemPaths, measure_bed_bleed, measure_dub_bleed

RATE = 8_000


def _write(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples.astype(np.float32), RATE, format="WAV", subtype="PCM_16")
    return path


def _noise(seconds: float, seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(0.0, 0.2, int(round(seconds * RATE)))


def _stems(root: Path, *, speech: np.ndarray, music: np.ndarray, effects: np.ndarray) -> StemPaths:
    return StemPaths(
        speech=_write(root / "speech.wav", speech),
        music=_write(root / "music.wav", music),
        effects=_write(root / "effects.wav", effects),
    )


def test_a_clean_separation_reports_no_leak(tmp_path: Path) -> None:
    speech = _noise(3.0, 1)
    stems = _stems(
        tmp_path,
        speech=speech,
        music=np.zeros_like(speech),
        effects=np.zeros_like(speech),
    )

    report = measure_bed_bleed(stems, [(0.5, 1.5), (1.8, 2.6)])

    assert report.clean
    assert report.windows == 2
    assert report.leaked == 0
    assert "no original dialogue" in report.summary()


def test_dialogue_left_in_the_bed_is_reported(tmp_path: Path) -> None:
    """The separator failing is exactly what this exists to catch."""

    speech = _noise(3.0, 2)
    stems = _stems(
        tmp_path,
        speech=speech,
        # The "music" stem is a copy of the dialogue: separation did nothing.
        music=speech * 0.9,
        effects=np.zeros_like(speech),
    )

    report = measure_bed_bleed(stems, [(0.5, 1.5)])

    assert not report.clean
    assert report.leaked == 1
    assert "WARNING" in report.summary()
    assert report.worst_relative_db > -12.0
    assert report.worst_correlation > 0.5


def test_loud_music_alone_is_not_a_leak(tmp_path: Path) -> None:
    """Level is not enough: a bed has to *carry the performance* to be a leak."""

    speech = _noise(3.0, 3)
    unrelated = _noise(3.0, 99)
    stems = _stems(
        tmp_path,
        speech=speech,
        music=unrelated,
        effects=np.zeros_like(speech),
    )

    report = measure_bed_bleed(stems, [(0.5, 1.5)])

    assert report.clean
    assert report.worst_relative_db > -15.0  # loud, but not the dialogue
    assert abs(report.worst_correlation) < 0.5


def test_quiet_bleed_below_the_threshold_is_clean(tmp_path: Path) -> None:
    """A little separation residue is normal and must not fail a run."""

    speech = _noise(3.0, 4)
    stems = _stems(
        tmp_path,
        speech=speech,
        music=speech * 0.01,  # -40 dB
        effects=np.zeros_like(speech),
    )

    report = measure_bed_bleed(stems, [(0.5, 1.5)])

    assert report.clean
    assert report.worst_relative_db < -35.0


def test_the_threshold_is_configurable(tmp_path: Path) -> None:
    """A run that demands a cleaner bed can ask for one."""

    speech = _noise(3.0, 5)
    stems = _stems(
        tmp_path,
        speech=speech,
        music=speech * 0.05,  # about -26 dB
        effects=np.zeros_like(speech),
    )

    assert measure_bed_bleed(stems, [(0.5, 1.5)]).clean
    assert not measure_bed_bleed(stems, [(0.5, 1.5)], threshold_db=-28.0).clean


def test_a_window_shorter_than_the_floor_is_not_counted(tmp_path: Path) -> None:
    speech = _noise(3.0, 6)
    stems = _stems(
        tmp_path,
        speech=speech,
        music=speech,
        effects=np.zeros_like(speech),
    )

    report = measure_bed_bleed(stems, [(1.0, 1.001)])

    assert report.windows == 0
    assert report.clean
    assert "not measured" in report.summary()


def test_a_missing_stem_raises_rather_than_reporting_clean(tmp_path: Path) -> None:
    """Reporting "no leak" because a file was missing would be a false pass."""

    speech = _write(tmp_path / "speech.wav", _noise(1.0, 7))
    stems = StemPaths(
        speech=speech,
        music=tmp_path / "absent_music.wav",
        effects=tmp_path / "absent_effects.wav",
    )

    with pytest.raises(separation.MissingInputError):
        measure_bed_bleed(stems, [(0.0, 0.5)])


def test_a_window_far_into_a_long_file_is_read_correctly(tmp_path: Path) -> None:
    """Windows are read out of the file one at a time, not by loading whole stems.

    A pair of two-hour 48 kHz stereo stems is about 8 GB of float32. Nothing here may
    load that.
    """

    speech = _noise(600.0, 11)  # ten minutes
    stems = _stems(
        tmp_path,
        speech=speech,
        # Leak only in the last second: a seek that ignored the offset would miss it.
        music=np.concatenate([np.zeros_like(speech[:-RATE]), speech[-RATE:]]),
        effects=np.zeros_like(speech),
    )

    assert measure_bed_bleed(stems, [(1.0, 2.0)]).clean

    late = measure_bed_bleed(stems, [(599.0, 600.0)])
    assert not late.clean
    assert late.leaked == 1
    assert late.worst_seconds == pytest.approx(599.0)


def test_the_report_is_json_safe(tmp_path: Path) -> None:
    import json

    speech = _noise(2.0, 8)
    stems = _stems(
        tmp_path, speech=speech, music=speech, effects=np.zeros_like(speech)
    )

    payload = measure_bed_bleed(stems, [(0.2, 1.0)]).as_dict()

    assert json.loads(json.dumps(payload)) == payload


# ---------------------------------------------------------------------------
# The finished dub, which the bed check cannot see
# ---------------------------------------------------------------------------


def test_the_bed_check_cannot_see_the_dubbed_dialogue(
    tmp_path: Path,
) -> None:
    """Why a second measurement exists.

    ``measure_bed_bleed`` inspects the music and effects stems. The dubbed dialogue is
    not part of them, so a dub that carried the original would be reported clean. This
    is that scenario, measured both ways.
    """

    original = _noise(3.0, 20)
    source = _write(tmp_path / "original.wav", original)
    # A dialogue track that is simply the original: the worst possible leak.
    dialogue = _write(tmp_path / "dialogue.wav", original)
    unrelated = _write(tmp_path / "unrelated.wav", _noise(3.0, 21))

    windows = [(0.5, 2.5)]

    dub = measure_dub_bleed(source, dialogue, windows)
    assert not dub.clean
    assert dub.worst_correlation > 0.9
    assert "dubbed dialogue" in dub.summary()

    # The bed is a different file here, and it is silent: nothing to find.
    stems = StemPaths(
        speech=unrelated,
        music=_write(tmp_path / "music.wav", _noise(3.0, 22)),
        effects=_write(tmp_path / "effects.wav", _noise(3.0, 23)),
    )
    assert measure_bed_bleed(stems, windows).clean


def test_a_dub_in_another_language_is_not_a_leak(tmp_path: Path) -> None:
    """A dub says the same thing in a different language, so it must not correlate."""

    original = _noise(3.0, 24)
    source = _write(tmp_path / "original.wav", original)
    # Speech-like material of the same level but different content - what a dub is.
    dialogue = _write(tmp_path / "dialogue.wav", _noise(3.0, 25) * 0.9)

    report = measure_dub_bleed(source, dialogue, [(0.5, 2.5)])

    assert report.clean


def test_a_leak_at_a_shifted_position_is_still_found(tmp_path: Path) -> None:
    """A placement or resampling fault moves the original; a lag search catches it."""

    original = _noise(4.0, 26)
    source = _write(tmp_path / "original.wav", original)
    # The original, delayed by 0.2s inside the window.
    shifted = np.concatenate([np.zeros(int(0.2 * RATE), dtype=np.float32), original])
    dialogue = _write(tmp_path / "dialogue.wav", shifted[: original.size])

    report = measure_dub_bleed(source, dialogue, [(1.0, 3.0)])

    assert not report.clean
    assert report.worst_correlation > 0.5


def test_quiet_leakage_is_not_reported(tmp_path: Path) -> None:
    """A trace amount is not audible, and crying wolf would make the check useless."""

    original = _noise(3.0, 27)
    source = _write(tmp_path / "original.wav", original)
    dialogue = _write(tmp_path / "dialogue.wav", original * 0.01)  # -40 dB

    assert measure_dub_bleed(source, dialogue, [(0.5, 2.5)]).clean


def test_a_dub_at_a_different_rate_is_refused(tmp_path: Path) -> None:
    """Comparing two time bases would silently measure nothing."""

    source = _write(tmp_path / "original.wav", _noise(2.0, 28))
    other = tmp_path / "dialogue.wav"
    other.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(other), _noise(2.0, 29), RATE * 2, format="WAV", subtype="PCM_16")

    with pytest.raises(separation.UnsupportedSampleRateError):
        measure_dub_bleed(source, other, [(0.0, 1.0)])


def test_a_missing_dialogue_track_raises(tmp_path: Path) -> None:
    source = _write(tmp_path / "original.wav", _noise(2.0, 30))

    with pytest.raises(separation.MissingInputError):
        measure_dub_bleed(source, tmp_path / "absent.wav", [(0.0, 1.0)])


def test_the_dub_report_is_json_safe(tmp_path: Path) -> None:
    import json

    original = _noise(2.0, 31)
    source = _write(tmp_path / "original.wav", original)
    dialogue = _write(tmp_path / "dialogue.wav", original)

    first = measure_dub_bleed(source, dialogue, [(0.2, 1.0)]).as_dict()
    second = measure_dub_bleed(source, dialogue, [(0.2, 1.0)]).as_dict()

    assert first == second
    assert json.loads(json.dumps(first)) == first
