"""Tests for native-speaker voice samples.

The reason this mechanism exists is a specific audible defect: a reference cut from the
film is the original actor speaking English, and the cloning engine conditions on its
reference, so the Amharic arrives in that performance's phonetics. These tests cover the
parts that decide whether a native recording actually reaches the engine - the
assignment, the trimming, the format and the transcript - rather than the audio itself,
which only a listen can judge.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import voice_profiles as vp
from app.pipeline.voice_profiles import (
    MissingVoiceSampleError,
    SAMPLE_REFERENCE_FILENAME,
    VoiceProfile,
    apply_voice_samples,
    assign_voice_samples,
    available_voice_samples,
)

RATE = 44_100


def _settings(root: Path, **overrides) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "models",
        "voice_sample_dir": root / "samples",
        "voice_profile_dir": root / "voices",
        "voice_reference_min_duration": 3.0,
        "voice_reference_target_duration": 10.0,
        "voice_reference_max_duration": 15.0,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write_sample(
    directory: Path,
    name: str,
    *,
    seconds: float = 20.0,
    lead_silence: float = 0.0,
    amplitude: float = 0.4,
) -> Path:
    """Write a tone-ish recording, optionally with silence in front of the speech."""

    directory.mkdir(parents=True, exist_ok=True)
    total = int(round(seconds * RATE))
    rng = np.random.default_rng(0)
    samples = (rng.standard_normal(total) * amplitude).astype(np.float32)
    if lead_silence:
        head = int(round(lead_silence * RATE))
        samples[:head] = 0.0
    path = directory / name
    sf.write(str(path), samples, RATE, subtype="PCM_16")
    return path


def _profile(speaker: str, root: Path, seconds: float = 8.0) -> VoiceProfile:
    return VoiceProfile(
        speaker_id=speaker,
        reference_audio=root / f"{speaker}.wav",
        reference_start=0.0,
        reference_end=seconds,
    )


# ---------------------------------------------------------------------------
# Finding and assigning the recordings
# ---------------------------------------------------------------------------


def test_samples_are_found_in_a_stable_order(tmp_path: Path) -> None:
    """A character whose voice changed between runs would be invisible in the logs."""

    _write_sample(tmp_path / "samples", "b-voice.wav")
    _write_sample(tmp_path / "samples", "a-voice.wav")

    found = available_voice_samples(tmp_path / "samples")

    assert [path.name for path in found] == ["a-voice.wav", "b-voice.wav"]


def test_a_missing_directory_means_the_mechanism_is_off(tmp_path: Path) -> None:
    """The default path exists in a fresh checkout; an empty one is not an error."""

    assert available_voice_samples(tmp_path / "nope") == []


def test_an_unreadable_extension_is_not_offered(tmp_path: Path) -> None:
    """A stray README in the samples directory must not be treated as a voice."""

    _write_sample(tmp_path / "samples", "real.wav")
    (tmp_path / "samples" / "notes.txt").write_text("hi", encoding="utf-8")

    assert [path.name for path in available_voice_samples(tmp_path / "samples")] == [
        "real.wav"
    ]


def test_two_samples_and_two_speakers_need_no_configuration(tmp_path: Path) -> None:
    samples = [
        _write_sample(tmp_path / "samples", "a.wav"),
        _write_sample(tmp_path / "samples", "b.wav"),
    ]

    assignment = assign_voice_samples(["SPEAKER_00", "SPEAKER_01"], samples)

    assert assignment == {"SPEAKER_00": samples[0], "SPEAKER_01": samples[1]}


def test_the_mapping_decides_which_voice_a_speaker_gets(tmp_path: Path) -> None:
    """The numbering rarely lines up with the cast, so the choice has to be explicit."""

    male = _write_sample(tmp_path / "samples", "male.wav")
    female = _write_sample(tmp_path / "samples", "female.wav")

    assignment = assign_voice_samples(
        ["SPEAKER_00", "SPEAKER_01"],
        [female, male],
        mapping="SPEAKER_00=male.wav,SPEAKER_01=female.wav",
    )

    assert assignment == {"SPEAKER_00": male, "SPEAKER_01": female}


def test_a_mapped_speaker_does_not_free_its_default_file(tmp_path: Path) -> None:
    """Otherwise a third speaker would be given a voice the first one is already using."""

    first = _write_sample(tmp_path / "samples", "a.wav")
    second = _write_sample(tmp_path / "samples", "b.wav")
    third = _write_sample(tmp_path / "samples", "c.wav")

    assignment = assign_voice_samples(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"],
        [first, second, third],
        mapping="SPEAKER_01=c.wav",
    )

    assert assignment["SPEAKER_01"] == third
    assert {assignment["SPEAKER_00"], assignment["SPEAKER_02"]} == {first, second}


def test_more_speakers_than_samples_reuses_them(tmp_path: Path) -> None:
    """Two recordings and a crowd is a real case; crashing on it would be useless."""

    samples = [
        _write_sample(tmp_path / "samples", "a.wav"),
        _write_sample(tmp_path / "samples", "b.wav"),
    ]

    assignment = assign_voice_samples(
        ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02"], samples
    )

    assert assignment["SPEAKER_02"] == samples[0]


def test_a_mapping_that_names_a_missing_file_is_reported(tmp_path: Path) -> None:
    samples = [_write_sample(tmp_path / "samples", "a.wav")]

    with pytest.raises(MissingVoiceSampleError, match="no such file"):
        assign_voice_samples(
            ["SPEAKER_00"], samples, mapping="SPEAKER_00=nope.wav"
        )


def test_a_malformed_mapping_entry_is_reported(tmp_path: Path) -> None:
    samples = [_write_sample(tmp_path / "samples", "a.wav")]

    with pytest.raises(vp.ConfigurationError, match="SPEAKER_00=file"):
        assign_voice_samples(["SPEAKER_00"], samples, mapping="SPEAKER_00")


# ---------------------------------------------------------------------------
# Turning a recording into a reference the engine can use
# ---------------------------------------------------------------------------


def test_a_long_recording_is_trimmed_to_a_reference(tmp_path: Path) -> None:
    """Your own recordings are 30-odd seconds; a cloning reference is about ten."""

    _write_sample(tmp_path / "samples", "long.wav", seconds=33.9)
    settings = _settings(tmp_path)

    replaced = apply_voice_samples(
        {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)},
        settings=settings,
    )

    profile = replaced["SPEAKER_00"]
    assert profile.reference_end == pytest.approx(10.0, abs=0.05)
    assert profile.reference_audio.exists()


def test_the_trimmed_reference_is_mono_24k_pcm(tmp_path: Path) -> None:
    """The engine and the rest of the pipeline expect one reference format."""

    _write_sample(tmp_path / "samples", "long.wav", seconds=25.0)

    apply_voice_samples(
        {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)},
        settings=_settings(tmp_path),
    )

    written = (
        tmp_path / "voices" / "SPEAKER_00" / SAMPLE_REFERENCE_FILENAME
    )
    info = sf.info(str(written))
    assert info.samplerate == vp.REFERENCE_SAMPLE_RATE
    assert info.channels == vp.REFERENCE_CHANNELS


def test_leading_silence_is_not_spent_on_the_reference(tmp_path: Path) -> None:
    """A breath at the top of a recording would otherwise cost a fifth of the voice."""

    _write_sample(tmp_path / "samples", "late.wav", seconds=20.0, lead_silence=3.0)
    settings = _settings(tmp_path)

    apply_voice_samples(
        {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)}, settings=settings
    )

    written = tmp_path / "voices" / "SPEAKER_00" / SAMPLE_REFERENCE_FILENAME
    samples, _ = sf.read(str(written), dtype="float32", always_2d=True)
    # The window starts on sound, so it is not the silent stretch of the original.
    assert float(np.max(np.abs(samples))) > 0.1


def test_a_recording_that_is_too_short_is_reported(tmp_path: Path) -> None:
    """Better to name the recording than to hand the engine a reference it cannot use."""

    _write_sample(tmp_path / "samples", "tiny.wav", seconds=1.5)

    with pytest.raises(MissingVoiceSampleError, match="at least"):
        apply_voice_samples(
            {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)},
            settings=_settings(tmp_path),
        )


def test_the_transcript_is_taken_from_the_trimmed_audio(tmp_path: Path) -> None:
    """The transcript has to describe the audio that was kept, not the original file."""

    _write_sample(tmp_path / "samples", "long.wav", seconds=33.9)
    seen: list[Path] = []

    def transcribe(path: Path) -> str:
        seen.append(path)
        return "ሰላም ልጆቼ"

    replaced = apply_voice_samples(
        {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)},
        settings=_settings(tmp_path),
        transcribe=transcribe,
    )

    profile = replaced["SPEAKER_00"]
    assert profile.reference_text == "ሰላም ልጆቼ"
    # Transcribed the file the engine will read, not the operator's original.
    assert seen == [profile.reference_audio]
    assert profile.reference_audio.name == SAMPLE_REFERENCE_FILENAME
    assert "native-speaker sample long.wav" in profile.selection_reason


def test_without_a_transcriber_the_reference_carries_no_text(tmp_path: Path) -> None:
    """No text is honest: the library then transcribes it rather than us guessing."""

    _write_sample(tmp_path / "samples", "long.wav", seconds=20.0)

    replaced = apply_voice_samples(
        {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)},
        settings=_settings(tmp_path),
    )

    assert replaced["SPEAKER_00"].reference_text is None
    assert "no transcript" in replaced["SPEAKER_00"].selection_reason


def test_an_empty_transcript_is_not_stored_as_an_empty_string(tmp_path: Path) -> None:
    """"no text" must stay distinguishable from "" for the engine's own ASR path."""

    _write_sample(tmp_path / "samples", "long.wav", seconds=20.0)

    replaced = apply_voice_samples(
        {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)},
        settings=_settings(tmp_path),
        transcribe=lambda _: "   ",
    )

    assert replaced["SPEAKER_00"].reference_text is None


# ---------------------------------------------------------------------------
# Whole-run behaviour
# ---------------------------------------------------------------------------


def test_every_speaker_is_changed_or_none_is(tmp_path: Path) -> None:
    """One character left on the film's audio would be a dub with two of everything."""

    _write_sample(tmp_path / "samples", "a.wav")
    _write_sample(tmp_path / "samples", "b.wav")
    profiles = {
        "SPEAKER_00": _profile("SPEAKER_00", tmp_path),
        "SPEAKER_01": _profile("SPEAKER_01", tmp_path),
    }

    replaced = apply_voice_samples(profiles, settings=_settings(tmp_path))

    assert set(replaced) == set(profiles)
    for speaker, profile in replaced.items():
        assert profile.reference_audio != profiles[speaker].reference_audio
        assert profile.reference_audio.name == SAMPLE_REFERENCE_FILENAME


def test_with_no_samples_the_profiles_are_untouched(tmp_path: Path) -> None:
    """The mechanism is additive: a checkout without samples behaves as before."""

    profiles = {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)}

    assert apply_voice_samples(profiles, settings=_settings(tmp_path)) == profiles


def test_the_films_own_reference_is_not_overwritten(tmp_path: Path) -> None:
    """Keeping the two apart is what lets the mechanism be switched back."""

    _write_sample(tmp_path / "samples", "a.wav")
    film_reference = tmp_path / "voices" / "SPEAKER_00" / "reference.wav"
    film_reference.parent.mkdir(parents=True)
    film_reference.write_bytes(b"pretend reference")

    apply_voice_samples(
        {"SPEAKER_00": _profile("SPEAKER_00", tmp_path)},
        settings=_settings(tmp_path),
    )

    assert film_reference.read_bytes() == b"pretend reference"
