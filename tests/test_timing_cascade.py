"""A line that runs long must move the next one, never speak over it.

The GPU run of the test film produced 29 overlaps, including two passages where the
same character talked over themselves for 2.72s and 2.26s. Every one of them was a
line whose Amharic needed more time than the film had left it, placed on top of the
next line's original start. Two voices at once is the one outcome a dub cannot have,
so the overrun is paid for in position instead of in intelligibility.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.config import DEFAULT_TIMING_MAX_OVERLAP_SECONDS
from app.pipeline import timing
from app.pipeline.timing import AlignedClip, align_dialogue
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import PerformanceControls, TtsClip

CLIP_RATE = 24_000


def _write(path: Path, seconds: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(int(round(seconds * CLIP_RATE)), dtype=np.float64) / CLIP_RATE
    samples = (0.5 * np.sin(2.0 * np.pi * 220.0 * times)).astype(np.float32)
    sf.write(str(path), samples, CLIP_RATE, format="WAV", subtype="PCM_16")
    return path


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "cache",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _clip(
    root: Path,
    *,
    index: int,
    start: float,
    end: float,
    speech: float,
    lead: float = 0.0,
    trail: float = 0.0,
) -> TtsClip:
    dialogue = AdaptedDialogue(
        speaker_id="SPEAKER_00",
        start=start,
        end=end,
        source_text="line",
        amharic="መስመር",
        emotion="neutral",
        intensity=0.5,
        delivery="flat",
        pause_before=lead,
        pause_after=trail,
    )
    path = _write(root / f"clip{index}.wav", speech)
    return TtsClip(
        index=index,
        dialogue=dialogue,
        performance=PerformanceControls.from_dialogue(dialogue, seed=index + 1),
        audio_path=path,
        take_path=path,
        performance_reference_path=None,
        voice_reference_path=None,
        sample_rate=CLIP_RATE,
        speech_duration=speech,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
        performance_engine="omnivoice",
        style_engine="none",
    )


def test_an_over_long_line_moves_the_next_one_instead_of_overlapping(
    tmp_path: Path,
) -> None:
    """Two back-to-back lines, the first needing far more time than it has."""

    clips = [
        # 0.4s window, but the Amharic needs 1.6s of speech.
        _clip(tmp_path, index=0, start=5.0, end=5.4, speech=1.6),
        # Starts 0.6s after the first one did - well before it has finished.
        _clip(tmp_path, index=1, start=5.6, end=6.4, speech=0.6),
    ]

    aligned = align_dialogue(clips, output_dir=tmp_path / "w", settings=_settings(tmp_path))

    first, second = aligned
    assert first.drift == pytest.approx(0.0)
    assert second.drift > 0.0
    # What is left of the first line when the second begins is at most the hand-over
    # the configuration allows, never the seconds of doubled speech that were audible.
    assert first.end - second.start <= DEFAULT_TIMING_MAX_OVERLAP_SECONDS + 1e-6


def test_no_two_lines_ever_run_at_once_beyond_the_hand_over(tmp_path: Path) -> None:
    clips = [
        _clip(tmp_path, index=index, start=5.0 + index, end=5.6 + index, speech=1.4)
        for index in range(5)
    ]

    aligned = align_dialogue(clips, output_dir=tmp_path / "w", settings=_settings(tmp_path))

    assert [line.index for line in aligned] == [0, 1, 2, 3, 4]
    for earlier, later in zip(aligned, aligned[1:]):
        assert earlier.end - later.start <= DEFAULT_TIMING_MAX_OVERLAP_SECONDS + 1e-6


def test_the_hand_over_is_configurable_and_can_be_turned_off(tmp_path: Path) -> None:
    """A run that wants strict separation gets it."""

    clips = [
        _clip(tmp_path, index=0, start=5.0, end=5.4, speech=1.6),
        _clip(tmp_path, index=1, start=5.6, end=6.4, speech=0.6),
    ]

    aligned = align_dialogue(
        clips,
        output_dir=tmp_path / "w",
        settings=_settings(tmp_path, timing_max_overlap_seconds=0.0),
    )

    for earlier, later in zip(aligned, aligned[1:]):
        assert earlier.end <= later.start + 1e-6


def test_a_line_returns_to_its_original_place_when_the_silence_allows_it(
    tmp_path: Path,
) -> None:
    """Drift must not accumulate across a gap: the picture stays authoritative."""

    clips = [
        _clip(tmp_path, index=0, start=5.0, end=5.2, speech=1.2),
        # A long silence follows, so line 1 has no reason to move at all.
        _clip(tmp_path, index=1, start=20.0, end=20.8, speech=0.5),
    ]

    aligned = align_dialogue(clips, output_dir=tmp_path / "w", settings=_settings(tmp_path))

    assert aligned[0].drift == pytest.approx(0.0)
    assert aligned[1].drift == pytest.approx(0.0)
    assert aligned[1].start == pytest.approx(20.0)


def test_genuine_simultaneous_dialogue_is_preserved(tmp_path: Path) -> None:
    """When the source overlapped two performances, the overlap *is* the scene."""

    clips = [
        _clip(tmp_path, index=0, start=5.0, end=6.0, speech=1.0),
        # Line 1 began 0.6s into line 0 in the source: two people talking at once.
        _clip(tmp_path, index=1, start=5.6, end=6.6, speech=1.0),
    ]

    aligned = align_dialogue(clips, output_dir=tmp_path / "w", settings=_settings(tmp_path))

    first, second = aligned
    assert second.drift == pytest.approx(0.0)
    assert second.start == pytest.approx(5.6)
    assert second.start < first.end  # the overlap the source actually had


def test_a_moved_line_says_so_on_the_clip(tmp_path: Path) -> None:
    clips = [
        _clip(tmp_path, index=0, start=5.0, end=5.4, speech=1.6),
        _clip(tmp_path, index=1, start=5.6, end=6.4, speech=0.6),
    ]

    aligned = align_dialogue(clips, output_dir=tmp_path / "w", settings=_settings(tmp_path))

    assert any("starts later" in note for note in aligned[1].notes)
    assert aligned[1].to_dict()["drift"] == pytest.approx(aligned[1].drift)
    assert aligned[0].to_dict()["drift"] == 0.0


def test_the_room_is_never_smaller_than_the_lines_own_window(tmp_path: Path) -> None:
    """A window is where the actor spoke; nothing may squash a line below it.

    The room is measured to the next line's start, and that came back *smaller* than
    the line's own window on the test film (line 8: a 4.56s window with 4.22s of room).
    Compressing a line below its own window to make room for a neighbour that has not
    even been placed yet is how a dub ends up sounding rushed.
    """

    clips = [
        _clip(tmp_path, index=0, start=5.0, end=6.0, speech=1.0),
        _clip(tmp_path, index=1, start=6.05, end=6.5, speech=0.4),
    ]

    aligned = align_dialogue(clips, output_dir=tmp_path / "w", settings=_settings(tmp_path))

    assert aligned[0].available >= aligned[0].original_window


def test_drift_is_reported_as_a_number_not_left_to_be_inferred(tmp_path: Path) -> None:
    clips = [
        _clip(tmp_path, index=0, start=5.0, end=5.4, speech=1.6),
        _clip(tmp_path, index=1, start=5.6, end=6.4, speech=0.6),
    ]

    aligned = align_dialogue(clips, output_dir=tmp_path / "w", settings=_settings(tmp_path))

    for line in aligned:
        assert isinstance(line.drift, float)
        assert line.drift >= 0.0


def test_a_negative_drift_is_rejected() -> None:
    with pytest.raises(timing.InvalidClipError):
        AlignedClip(
            index=0,
            clip=object(),  # type: ignore[arg-type]
            audio_path="a.wav",
            sample_rate=CLIP_RATE,
            start=0.0,
            tempo=1.0,
            required_tempo=1.0,
            speech_duration=1.0,
            original_window=1.0,
            rendered_pause_before=0.0,
            rendered_pause_after=0.0,
            drift=-0.5,
        )
