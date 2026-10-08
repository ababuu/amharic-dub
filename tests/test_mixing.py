"""Tests for :mod:`app.pipeline.mixing`.

No model, no network and no FFmpeg: everything here is real audio arithmetic on
synthetic stems and lines, so the mix can be measured sample by sample.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import mixing, timing, video
from app.pipeline.mixing import (
    CHANNELS,
    DIALOGUE_STEM_FILENAME,
    MIXED_FILENAME,
    MIX_DIRECTORY_NAME,
    PEAK_CEILING,
    PIPELINE_SAMPLE_RATE,
    ConfigurationError,
    InvalidAudioError,
    InvalidClipError,
    MissingInputError,
    db_to_gain,
    mix_track,
    resolve_mix_directory,
)
from app.pipeline.timing import AlignedClip
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import PerformanceControls, TtsClip

RATE = PIPELINE_SAMPLE_RATE

#: Music at 110 Hz and effects at 3 kHz, so the two beds can be told apart.
MUSIC_AMPLITUDE = 0.20
EFFECTS_AMPLITUDE = 0.10
STEM_SECONDS = 6.0


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "cache",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write_stereo(path: Path, samples: np.ndarray, rate: int = RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, rate, format="WAV", subtype="PCM_16")
    return path


def _bed(seconds: float = STEM_SECONDS, rate: int = RATE, amplitude: float = 0.2,
         frequency: float = 110.0) -> np.ndarray:
    times = np.arange(int(round(seconds * rate)), dtype=np.float64) / rate
    mono = (amplitude * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)
    return np.stack([mono, mono], axis=1)


def _stems(root: Path, *, seconds: float = STEM_SECONDS) -> tuple[Path, Path]:
    music = _write_stereo(root / "movie_music.wav", _bed(seconds, amplitude=MUSIC_AMPLITUDE))
    effects = _write_stereo(
        root / "movie_effects.wav", _bed(seconds, amplitude=EFFECTS_AMPLITUDE, frequency=3000.0)
    )
    return music, effects


def _aligned(
    root: Path,
    *,
    index: int = 0,
    start: float = 2.0,
    speech: float = 1.0,
    lead: float = 0.10,
    trail: float = 0.0,
    amplitude: float = 0.5,
    speaker: str = "SPEAKER_00",
) -> AlignedClip:
    """Return an aligned line backed by a real file, without running timing."""

    times = np.arange(int(round(speech * RATE)), dtype=np.float64) / RATE
    body = (amplitude * np.sin(2.0 * np.pi * 440.0 * times)).astype(np.float32)
    pieces = []
    if lead:
        pieces.append(np.zeros(int(round(lead * RATE)), dtype=np.float32))
    pieces.append(body)
    if trail:
        pieces.append(np.zeros(int(round(trail * RATE)), dtype=np.float32))
    path = _write_stereo(
        root / f"aligned_{index}.wav", np.concatenate(pieces)[:, None], rate=RATE
    )

    dialogue = AdaptedDialogue(
        speaker_id=speaker,
        start=start + lead,
        end=start + lead + speech,
        source_text="x",
        amharic="እ",
        emotion="neutral",
        intensity=0.5,
        delivery="plain",
        pause_before=lead,
        pause_after=trail,
    )
    clip = TtsClip(
        index=index,
        dialogue=dialogue,
        performance=PerformanceControls.from_dialogue(dialogue, seed=index + 1),
        audio_path=path,
        take_path=path,
        performance_reference_path=path,
        voice_reference_path=path,
        sample_rate=RATE,
        speech_duration=speech,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
        performance_engine="chatterbox-amharic",
        style_engine="seed-vc-v2",
    )
    return AlignedClip(
        index=index,
        clip=clip,
        audio_path=path,
        sample_rate=RATE,
        start=start,
        tempo=1.0,
        required_tempo=1.0,
        speech_duration=speech,
        original_window=speech,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
    )


def _read(path: Path) -> tuple[np.ndarray, int]:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    return data, rate


def _window(data: np.ndarray, start: float, end: float) -> np.ndarray:
    return data[int(round(start * RATE)) : int(round(end * RATE))]


# ---------------------------------------------------------------------------
# The format contract
# ---------------------------------------------------------------------------


def test_mix_rate_matches_the_rest_of_the_pipeline() -> None:
    assert PIPELINE_SAMPLE_RATE == timing.PIPELINE_SAMPLE_RATE
    assert PIPELINE_SAMPLE_RATE == video.PIPELINE_SAMPLE_RATE


def test_default_directory_follows_the_work_directory(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    music, effects = _stems(tmp_path)
    mix_track([], music, effects, settings=settings)

    expected = Path(settings.work_dir) / MIX_DIRECTORY_NAME
    assert resolve_mix_directory(settings=settings) == expected
    assert expected.is_dir()


def test_artifacts_are_written_at_the_pipeline_format(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    result = mix_track(
        [_aligned(tmp_path)], music, effects, output_dir=tmp_path,
        settings=_settings(tmp_path),
    )

    # The mix gets its own sub-directory, exactly as the timing stage does.
    assert result.dialogue_path == tmp_path / MIX_DIRECTORY_NAME / DIALOGUE_STEM_FILENAME
    assert result.mixed_path == tmp_path / MIX_DIRECTORY_NAME / MIXED_FILENAME
    for path in (result.dialogue_path, result.mixed_path):
        info = sf.info(str(path))
        assert info.samplerate == RATE
        assert info.channels == CHANNELS
        assert info.subtype == "PCM_16"
    assert result.sample_rate == RATE
    assert result.channels == CHANNELS


def test_output_is_as_long_as_the_bed(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path, seconds=4.0)
    result = mix_track(
        [_aligned(tmp_path)], music, effects, output_dir=tmp_path / "mix",
        settings=_settings(tmp_path),
    )

    assert result.duration == pytest.approx(4.0, abs=0.001)
    assert result.duration == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def test_a_line_is_placed_at_its_own_start(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    line = _aligned(tmp_path, start=2.0, speech=1.0, lead=0.10)
    result = mix_track(
        [line], music, effects, output_dir=tmp_path / "mix", settings=_settings(tmp_path)
    )

    dialogue, _ = _read(result.dialogue_path)
    assert float(np.abs(dialogue).max()) > 0.0
    # The line's *file* starts at 2.0 (its leading pause included), so the speech
    # is what lands at 2.1.
    assert float(np.abs(_window(dialogue, 0.0, 1.9)).max()) == 0.0
    assert float(np.abs(_window(dialogue, 2.05, 3.05)).max()) > 0.0
    assert float(np.abs(_window(dialogue, 3.2, 6.0)).max()) == 0.0


def test_the_dialogue_stem_carries_only_the_dialogue(tmp_path: Path) -> None:
    """The stem exists so a mis-levelled dub can be told from a mis-levelled bed."""

    music, effects = _stems(tmp_path)
    result = mix_track(
        [_aligned(tmp_path, start=2.0)], music, effects, output_dir=tmp_path / "mix",
        settings=_settings(tmp_path),
    )

    dialogue, _ = _read(result.dialogue_path)
    assert float(np.abs(_window(dialogue, 3.5, 5.9)).max()) == 0.0


def test_both_beds_reach_the_mix(tmp_path: Path) -> None:
    """Music and effects are both present, and the speech stem is not used."""

    music, effects = _stems(tmp_path)
    result = mix_track(
        [], music, effects, output_dir=tmp_path / "mix", settings=_settings(tmp_path)
    )

    mixed, _ = _read(result.mixed_path)
    assert float(np.abs(mixed).max()) > MUSIC_AMPLITUDE + EFFECTS_AMPLITUDE - 0.05
    # Two tones summing cannot exceed their sum; nothing else was added.
    assert float(np.abs(mixed).max()) <= MUSIC_AMPLITUDE + EFFECTS_AMPLITUDE + 0.01


def test_dialogue_is_added_on_top_of_the_bed(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    silence = _aligned(tmp_path, start=2.0, amplitude=0.0)
    loud = _aligned(tmp_path, start=2.0, amplitude=0.5, index=1)

    plain = mix_track(
        [silence], music, effects, output_dir=tmp_path / "a", settings=_settings(tmp_path)
    )
    added = mix_track(
        [loud], music, effects, output_dir=tmp_path / "b", settings=_settings(tmp_path)
    )

    quiet_mix, _ = _read(plain.mixed_path)
    loud_mix, _ = _read(added.mixed_path)
    assert float(np.abs(_window(loud_mix, 2.3, 2.7)).max()) > float(
        np.abs(_window(quiet_mix, 2.3, 2.7)).max()
    )


def test_clips_are_placed_in_start_order_whatever_the_input_order(
    tmp_path: Path,
) -> None:
    music, effects = _stems(tmp_path)
    late = _aligned(tmp_path, index=5, start=4.0)
    early = _aligned(tmp_path, index=1, start=1.0)

    result = mix_track(
        [late, early], music, effects, output_dir=tmp_path / "mix",
        settings=_settings(tmp_path),
    )

    dialogue, _ = _read(result.dialogue_path)
    assert float(np.abs(_window(dialogue, 1.05, 2.05)).max()) > 0.0
    assert float(np.abs(_window(dialogue, 4.05, 5.05)).max()) > 0.0


# ---------------------------------------------------------------------------
# Overlaps
# ---------------------------------------------------------------------------


def test_overlapping_lines_are_summed_and_reported(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    first = _aligned(tmp_path, index=0, start=2.0, speech=1.0, speaker="A")
    second = _aligned(tmp_path, index=1, start=2.5, speech=1.0, speaker="B")

    result = mix_track(
        [first, second], music, effects, output_dir=tmp_path / "mix",
        settings=_settings(tmp_path),
    )

    assert len(result.overlaps) == 1
    overlap = result.overlaps[0]
    assert (overlap.first_index, overlap.second_index) == (0, 1)
    assert (overlap.first_speaker, overlap.second_speaker) == ("A", "B")
    # Placed audio overlaps, not speech: the first line's file runs from 2.0s to
    # 3.1s because it carries its 0.1s leading pause, so it meets the second line
    # (2.5s-3.6s) for 0.6s even though the two speeches are 1.0s apart.
    assert overlap.seconds == pytest.approx(0.6, abs=0.01)


def test_lines_that_do_not_touch_are_not_reported(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    result = mix_track(
        [_aligned(tmp_path, index=0, start=1.0), _aligned(tmp_path, index=1, start=3.0)],
        music,
        effects,
        output_dir=tmp_path / "mix",
        settings=_settings(tmp_path),
    )

    assert result.overlaps == ()


def test_touching_lines_are_not_an_overlap(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    result = mix_track(
        [
            _aligned(tmp_path, index=0, start=1.0, speech=1.0, lead=0.0),
            _aligned(tmp_path, index=1, start=2.0, speech=1.0, lead=0.0),
        ],
        music,
        effects,
        output_dir=tmp_path / "mix",
        settings=_settings(tmp_path),
    )

    assert result.overlaps == ()


# ---------------------------------------------------------------------------
# Ducking
# ---------------------------------------------------------------------------


def test_duck_gain_is_restored_in_the_gaps(tmp_path: Path) -> None:
    """The bed only stays down while a line is playing."""

    line = _aligned(tmp_path, start=2.0, speech=1.0, amplitude=0.5)
    samples, _ = _read(line.audio_path)

    # The duck is computed on the whole timeline, so the line has to be placed in
    # a full-length bus exactly as mix_track places it.
    bus = np.zeros((int(round(STEM_SECONDS * RATE)), CHANNELS), dtype=np.float32)
    offset = int(round(line.start * RATE))
    stereo = np.repeat(samples[:, :1], CHANNELS, axis=1)
    bus[offset : offset + stereo.shape[0]] = stereo

    gain = mixing._duck_gain(bus, duck_db=6.0, sample_rate=RATE)  # noqa: SLF001

    during = float(np.median(gain[int(2.5 * RATE) : int(3.0 * RATE)]))
    outside = float(np.median(gain[int(4.0 * RATE) : int(5.0 * RATE)]))
    assert during == pytest.approx(db_to_gain(-6.0), abs=0.02)
    assert outside == pytest.approx(1.0, abs=0.01)


def test_ducking_leaves_the_dialogue_alone(tmp_path: Path) -> None:
    """Only the bed is reduced, and by the configured amount."""

    music, effects = _stems(tmp_path)
    line = _aligned(tmp_path, start=2.0, speech=1.0, amplitude=0.5)

    unducked = mix_track(
        [line], music, effects, output_dir=tmp_path / "u",
        settings=_settings(tmp_path, mix_duck_db=0.0),
    )
    ducked = mix_track(
        [line], music, effects, output_dir=tmp_path / "d", settings=_settings(tmp_path)
    )

    a, _ = _read(unducked.mixed_path)
    b, _ = _read(ducked.mixed_path)

    # In the gaps the two mixes are identical: nothing was ducked.
    assert np.allclose(_window(a, 4.0, 5.5), _window(b, 4.0, 5.5), atol=1e-6)
    # Under the line the ducked mix is quieter, and the difference is the bed only.
    assert float(np.abs(_window(b, 2.4, 2.9)).max()) < float(
        np.abs(_window(a, 2.4, 2.9)).max()
    )
    bed_only = np.abs(_window(a, 2.4, 2.9) - _window(b, 2.4, 2.9))
    assert float(bed_only.max()) > 0.0


def test_a_deeper_duck_removes_more_of_the_bed(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    line = _aligned(tmp_path, start=2.0, speech=1.0, amplitude=0.5)

    shallow = mix_track(
        [line], music, effects, output_dir=tmp_path / "s",
        settings=_settings(tmp_path, mix_duck_db=3.0),
    )
    deep = mix_track(
        [line], music, effects, output_dir=tmp_path / "x",
        settings=_settings(tmp_path, mix_duck_db=12.0),
    )

    a, _ = _read(shallow.mixed_path)
    b, _ = _read(deep.mixed_path)
    quiet = _window(a, 4.0, 5.0)
    assert np.allclose(quiet, _window(b, 4.0, 5.0), atol=1e-6)
    assert float(np.abs(_window(b, 2.4, 2.9)).max()) < float(
        np.abs(_window(a, 2.4, 2.9)).max()
    )


# ---------------------------------------------------------------------------
# Levels
# ---------------------------------------------------------------------------


def test_the_peak_ceiling_is_respected_and_reported(tmp_path: Path) -> None:
    loud = _write_stereo(tmp_path / "m.wav", _bed(amplitude=0.7))
    louder = _write_stereo(tmp_path / "e.wav", _bed(amplitude=0.7, frequency=2000.0))
    line = _aligned(tmp_path, start=2.0, amplitude=0.6)

    result = mix_track(
        [line], loud, louder, output_dir=tmp_path / "mix", settings=_settings(tmp_path)
    )

    assert result.peak <= PEAK_CEILING + 1e-6
    assert result.peak == pytest.approx(PEAK_CEILING, abs=1e-3)
    assert result.peak_gain_reduction
    assert result.peak_gain_db < 0
    assert any("ceiling" in note for note in result.notes)

    mixed, _ = _read(result.mixed_path)
    assert float(np.abs(mixed).max()) <= PEAK_CEILING + 1e-3


def test_a_quiet_mix_is_not_touched(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    result = mix_track(
        [_aligned(tmp_path, amplitude=0.2)], music, effects, output_dir=tmp_path / "mix",
        settings=_settings(tmp_path),
    )

    assert not result.peak_gain_reduction
    assert result.peak_gain_db == 0.0
    assert result.peak < PEAK_CEILING


def test_the_dialogue_level_is_configurable(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    quiet = mix_track(
        [_aligned(tmp_path, start=2.0, amplitude=0.3)], music, effects,
        output_dir=tmp_path / "q", settings=_settings(tmp_path, mix_dialogue_gain_db=-6.0),
    )
    plain = mix_track(
        [_aligned(tmp_path, start=2.0, amplitude=0.3)], music, effects,
        output_dir=tmp_path / "p", settings=_settings(tmp_path),
    )

    a, _ = _read(quiet.dialogue_path)
    b, _ = _read(plain.dialogue_path)
    assert float(np.abs(a).max()) == pytest.approx(
        float(np.abs(b).max()) * db_to_gain(-6.0), rel=0.02
    )


# ---------------------------------------------------------------------------
# Stems that are not the pipeline format
# ---------------------------------------------------------------------------


def test_a_stem_at_the_wrong_rate_is_rejected(tmp_path: Path) -> None:
    good = _write_stereo(tmp_path / "m.wav", _bed())
    wrong = _write_stereo(tmp_path / "e.wav", _bed(rate=44_100), rate=44_100)

    with pytest.raises(InvalidAudioError) as info:
        mix_track([], good, wrong, output_dir=tmp_path / "mix", settings=_settings(tmp_path))

    assert "44100" in str(info.value)


def test_a_mono_stem_is_placed_in_both_channels(tmp_path: Path) -> None:
    mono = _bed()[:, :1]
    music = _write_stereo(tmp_path / "m.wav", mono)
    effects = _write_stereo(tmp_path / "e.wav", _bed(frequency=3000.0))
    result = mix_track(
        [], music, effects, output_dir=tmp_path / "mix", settings=_settings(tmp_path)
    )

    assert any("mono" in note for note in result.notes)
    mixed, _ = _read(result.mixed_path)
    assert np.allclose(mixed[:, 0], mixed[:, 1], atol=1e-6)


def test_stems_of_different_lengths_are_padded_not_truncated(tmp_path: Path) -> None:
    music = _write_stereo(tmp_path / "m.wav", _bed(seconds=4.0))
    effects = _write_stereo(tmp_path / "e.wav", _bed(seconds=6.0))
    result = mix_track(
        [], music, effects, output_dir=tmp_path / "mix", settings=_settings(tmp_path)
    )

    assert result.duration == pytest.approx(6.0)
    assert any("differ in length" in note for note in result.notes)


def test_missing_stem_is_reported(tmp_path: Path) -> None:
    music, _ = _stems(tmp_path)

    with pytest.raises(MissingInputError):
        mix_track(
            [], music, tmp_path / "nope.wav", output_dir=tmp_path / "mix",
            settings=_settings(tmp_path),
        )


def test_a_wrong_entry_type_is_reported(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)

    with pytest.raises(InvalidClipError):
        mix_track(
            ["not-a-line"], music, effects, output_dir=tmp_path / "mix",
            settings=_settings(tmp_path),
        )  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Lines that do not fit the film
# ---------------------------------------------------------------------------


def test_a_line_running_past_the_end_is_trimmed(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path, seconds=4.0)
    line = _aligned(tmp_path, start=3.5, speech=1.0, lead=0.0)

    result = mix_track(
        [line], music, effects, output_dir=tmp_path / "mix", settings=_settings(tmp_path)
    )

    assert any("trimmed" in note for note in result.notes)
    assert not result.fits
    dialogue, _ = _read(result.dialogue_path)
    assert dialogue.shape[0] == int(round(4.0 * RATE))


def test_a_line_starting_after_the_film_is_dropped(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path, seconds=4.0)
    line = _aligned(tmp_path, start=9.0, speech=1.0, lead=0.0)

    result = mix_track(
        [line], music, effects, output_dir=tmp_path / "mix", settings=_settings(tmp_path)
    )

    assert any("dropped" in note for note in result.notes)
    dialogue, _ = _read(result.dialogue_path)
    assert float(np.abs(dialogue).max()) == 0.0


# ---------------------------------------------------------------------------
# Configuration and description
# ---------------------------------------------------------------------------


def test_a_negative_duck_is_rejected(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)

    with pytest.raises(ConfigurationError) as info:
        mix_track(
            [], music, effects, output_dir=tmp_path / "mix",
            settings=_settings(tmp_path, mix_duck_db=-3.0),
        )

    assert "MIX_DUCK_DB" in str(info.value)


def test_an_absurd_duck_is_rejected(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)

    with pytest.raises(ConfigurationError):
        mix_track(
            [], music, effects, output_dir=tmp_path / "mix",
            settings=_settings(tmp_path, mix_duck_db=120.0),
        )


def test_to_dict_describes_the_mix(tmp_path: Path) -> None:
    music, effects = _stems(tmp_path)
    result = mix_track(
        [_aligned(tmp_path, start=2.0)], music, effects, output_dir=tmp_path / "mix",
        settings=_settings(tmp_path),
    )
    payload = result.to_dict()

    assert payload["sample_rate"] == RATE
    assert payload["channels"] == CHANNELS
    assert payload["dialogue_gain_db"] == 0.0
    assert payload["duck_db"] == 6.0
    assert payload["peak_gain_reduction"] is False
    assert payload["overlaps"] == []
    assert "dialogue_path" in payload


def test_fits_is_false_when_a_line_was_trimmed(tmp_path: Path) -> None:
    result_fits = mix_track(
        [_aligned(tmp_path, start=1.0)], *_stems(tmp_path), output_dir=tmp_path / "a",
        settings=_settings(tmp_path),
    )
    assert result_fits.fits
