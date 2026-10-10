"""A voice reference must be one voice, and its transcript must describe it.

Both rules come from a real cloning failure. The reference that produced it was the
10.0s window 20.0-30.0s of the dialogue stem, chosen from a stride grid. Two things
were wrong with it, and neither was visible in the measurements the selection used:

* the diarization reported simultaneous speech at 26.27-26.85s and 27.82-28.57s, both
  inside the window, so the recording held two people - yet the profile stated
  "0.0s overlapped by other speakers", because the per-line overlap check compares
  *exclusive* turns and those can never overlap;
* the window began 2.6s before the first line it recorded and ended 1.9s after the
  last, so about 4.5s of the prompt audio had no transcript at all.

OmniVoice, like the other cloning engines, is handed the reference transcript and the
line to speak as one text stream, so a transcript that covers only part of its own
audio is an invitation to speak the difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline.diarization import CrosstalkRegion, SpeakerSegment
from app.pipeline.transcription import TranscriptSegment
from app.pipeline.voice_profiles import build_voice_profiles

STEM_RATE = 16_000

#: Syllabic rhythm: a burst followed by a gap. The module's VAD correctly treats a
#: steady tone as non-speech, so these tests must hand it speech-like audio.
SPEECH_SYLLABLE_HZ = 4.0
SPEECH_DUTY = 0.6
SPEECH_FLOOR_RATIO = 0.02


def _render(level: float, seconds: float) -> np.ndarray:
    """Return ``seconds`` of speech-like audio at ``level``."""

    total = int(round(seconds * STEM_RATE))
    time = np.arange(total) / STEM_RATE
    voice = np.sin(2.0 * np.pi * 220.0 * time)
    phase = (time * SPEECH_SYLLABLE_HZ) % 1.0
    burst = np.where(phase < SPEECH_DUTY, np.sin(np.pi * phase / SPEECH_DUTY) ** 2, 0.0)
    return (level * (burst * voice + SPEECH_FLOOR_RATIO * voice)).astype(np.float32)


def _write_stem(path: Path, regions: list[tuple[float, float, float]]) -> Path:
    """Write a mono stem; ``regions`` are ``(start, end, level)`` triangles."""

    total = max(end for _, end, _ in regions)
    audio = np.zeros(int(round(total * STEM_RATE)), dtype=np.float32)
    for start, end, level in regions:
        first, last = int(round(start * STEM_RATE)), int(round(end * STEM_RATE))
        audio[first:last] = _render(level, (last - first) / STEM_RATE)
    sf.write(str(path), audio, STEM_RATE, subtype="PCM_16")
    return path


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root,
        "output_dir": root,
        "model_cache_dir": root,
        "voice_profile_dir": root / "voices",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_a_window_with_simultaneous_speech_is_avoided(tmp_path: Path) -> None:
    """A reference recording two voices teaches the model both of them."""

    stem = _write_stem(
        tmp_path / "speech.wav", [(10.0, 20.0, 0.1), (30.0, 40.0, 0.1)]
    )
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]
    crosstalk = [CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 10.4, 14.4)]

    profile = build_voice_profiles(
        segments,
        stem,
        crosstalk=crosstalk,
        settings=_settings(tmp_path),
    )["SPEAKER_00"]

    assert profile.reference_start == 30.0
    assert "0.0s overlapped by other speakers" in profile.selection_reason
    assert "WARNING" not in profile.selection_reason


def test_the_other_character_is_never_in_the_reference(tmp_path: Path) -> None:
    """Even a much better-sounding window is refused if it holds two voices.

    This is the case the old check could not see: the turns are exclusive, so the
    overlap term read zero for a window full of crosstalk, and the loud window won.
    """

    stem = _write_stem(
        tmp_path / "speech.wav", [(10.0, 20.0, 0.9), (30.0, 40.0, 0.02)]
    )
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]
    # The loud window is the one with both speakers in it.
    crosstalk = [CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 12.0, 18.0)]

    profile = build_voice_profiles(
        segments, stem, crosstalk=crosstalk, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_start == 30.0


def test_a_film_with_no_clean_window_still_gets_a_voice(tmp_path: Path) -> None:
    """Loud crosstalk throughout must not end a run: it is reported instead."""

    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, 0.1)])
    crosstalk = [CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 10.0, 20.0)]

    profile = build_voice_profiles(
        [SpeakerSegment("SPEAKER_00", 10.0, 20.0)],
        stem,
        crosstalk=crosstalk,
        settings=_settings(tmp_path),
    )["SPEAKER_00"]

    assert profile.reference_start == 10.0
    assert "WARNING" in profile.selection_reason
    assert "simultaneous speech" in profile.selection_reason


def test_crosstalk_between_other_speakers_still_counts(tmp_path: Path) -> None:
    """A window is only clean if *nobody* else is talking in it.

    Two other characters overlapping inside this speaker's turn means the turn
    boundary is wrong, and the audio there is not a safe identity reference.
    """

    stem = _write_stem(
        tmp_path / "speech.wav", [(10.0, 20.0, 0.9), (30.0, 40.0, 0.1)]
    )
    segments = [
        SpeakerSegment("SPEAKER_00", 10.0, 20.0),
        SpeakerSegment("SPEAKER_00", 30.0, 40.0),
    ]
    crosstalk = [CrosstalkRegion(("SPEAKER_01", "SPEAKER_02"), 11.0, 19.0)]

    profile = build_voice_profiles(
        segments, stem, crosstalk=crosstalk, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_start == 30.0


def test_the_transcript_sent_matches_the_audio_sent(tmp_path: Path) -> None:
    """The invariant, over a speaker whose lines are ragged.

    Whatever window is chosen, ``reference_text`` is either absent or the exact words
    of that window - never a partial description of it.
    """

    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 60.0, 0.1)])
    segments = [SpeakerSegment("SPEAKER_00", 10.0, 60.0)]
    transcript = [
        TranscriptSegment("SPEAKER_00", 10.3, 12.9, "first thing said"),
        TranscriptSegment("SPEAKER_00", 13.4, 15.1, "second thing said"),
        TranscriptSegment("SPEAKER_00", 20.0, 21.0, "a lone interjection"),
        TranscriptSegment("SPEAKER_00", 40.0, 48.0, "a much longer stretch of speech"),
    ]

    profile = build_voice_profiles(
        segments, stem, transcript=transcript, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_text is not None
    starts = {line.start for line in transcript}
    ends = {line.end for line in transcript}
    assert profile.reference_start in starts
    assert profile.reference_end in ends
    words = profile.reference_text.split()
    covered = " ".join(
        line.text
        for line in sorted(transcript, key=lambda item: item.start)
        if line.start >= profile.reference_start - 1e-3
        and line.end <= profile.reference_end + 1e-3
    ).split()
    assert words == covered


def test_a_reference_is_never_cut_on_an_arbitrary_grid_when_lines_exist(
    tmp_path: Path,
) -> None:
    """The stride grid is for speakers with no transcript, not for everyone."""

    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 40.0, 0.1)])
    segments = [SpeakerSegment("SPEAKER_00", 10.0, 40.0)]
    # Lines that happen not to start on the 2.0s stride the old code used.
    transcript = [
        TranscriptSegment("SPEAKER_00", 11.37, 15.11, "alpha beta gamma"),
        TranscriptSegment("SPEAKER_00", 15.42, 19.03, "delta epsilon zeta"),
    ]

    profile = build_voice_profiles(
        segments, stem, transcript=transcript, settings=_settings(tmp_path)
    )["SPEAKER_00"]

    assert profile.reference_start == pytest.approx(11.37)
    assert profile.reference_end == pytest.approx(19.03)
    assert profile.reference_text == "alpha beta gamma delta epsilon zeta"


def test_crosstalk_does_not_change_the_result_when_none_is_reported(
    tmp_path: Path,
) -> None:
    """Passing no crosstalk is the old behaviour, and must stay available."""

    stem = _write_stem(tmp_path / "speech.wav", [(10.0, 20.0, 0.1)])
    segments = [SpeakerSegment("SPEAKER_00", 10.0, 20.0)]

    without = build_voice_profiles(segments, stem, settings=_settings(tmp_path))
    with_empty = build_voice_profiles(
        segments, stem, crosstalk=[], settings=_settings(tmp_path)
    )

    assert without["SPEAKER_00"].reference_start == 10.0
    assert (
        with_empty["SPEAKER_00"].reference_start
        == without["SPEAKER_00"].reference_start
    )
