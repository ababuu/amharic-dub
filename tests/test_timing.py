"""Tests for :mod:`app.pipeline.timing`.

No model and no network: the FFmpeg subprocess seam is replaced by an in-process
fake that produces real audio at the pipeline rate, so every check runs against
actual files. One end-to-end test uses the real FFmpeg, and is skipped when it is
not installed.
"""

from __future__ import annotations

import shutil
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import separation, timing, video
from app.pipeline.timing import (
    ALIGNED_DIRECTORY_NAME,
    EXACT_FIT_TOLERANCE_SECONDS,
    PIPELINE_SAMPLE_RATE,
    TIMING_DIRECTORY_NAME,
    AlignedClip,
    ConfigurationError,
    InvalidAudioError,
    InvalidClipError,
    MissingFfmpegError,
    MissingInputError,
    StretchError,
    align_clip,
    align_dialogue,
    resolve_timing_directory,
)
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import PerformanceControls, TtsClip

#: What the TTS stage actually produces: Seed-VC writes its own converter rate.
CLIP_RATE = 22_050

LEAD = 0.10
SPEECH = 1.00
TRAIL = 0.20


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "cache",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, rate, format="WAV", subtype="PCM_16")
    return path


def _tone(seconds: float, rate: int, frequency: float = 220.0) -> np.ndarray:
    times = np.arange(int(round(seconds * rate)), dtype=np.float64) / rate
    return (0.5 * np.sin(2.0 * np.pi * frequency * times)).astype(np.float32)


def _clip_file(root: Path, *, lead: float = LEAD, speech: float = SPEECH,
               trail: float = TRAIL, name: str = "SPEAKER_00_abc") -> Path:
    """Write a clip shaped like the TTS stage's own: pauses around the speech."""

    pieces = []
    if lead:
        pieces.append(np.zeros(int(round(lead * CLIP_RATE)), dtype=np.float32))
    pieces.append(_tone(speech, CLIP_RATE))
    if trail:
        pieces.append(np.zeros(int(round(trail * CLIP_RATE)), dtype=np.float32))
    return _write_wav(root / f"{name}.wav", np.concatenate(pieces), CLIP_RATE)


def _clip(
    root: Path,
    *,
    index: int = 0,
    start: float = 5.0,
    end: float = 5.8,
    audio: Path | None = None,
    speech: float = SPEECH,
    lead: float = LEAD,
    trail: float = TRAIL,
) -> TtsClip:
    """Return a TtsClip whose original window is ``start``-``end``."""

    dialogue = AdaptedDialogue(
        speaker_id="SPEAKER_00",
        start=start,
        end=end,
        source_text="I love you.",
        amharic="እወድሃለሁ።",
        emotion="romantic",
        intensity=0.7,
        delivery="soft",
        pause_before=lead,
        pause_after=trail,
    )
    path = audio if audio is not None else _clip_file(root, lead=lead, speech=speech, trail=trail)
    return TtsClip(
        index=index,
        dialogue=dialogue,
        performance=PerformanceControls.from_dialogue(dialogue, seed=index + 1),
        audio_path=path,
        take_path=path,
        performance_reference_path=path,
        voice_reference_path=path,
        sample_rate=CLIP_RATE,
        speech_duration=speech,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
        performance_engine="chatterbox-amharic",
        style_engine="seed-vc-v2",
    )


class FakeFfmpeg:
    """Stand-in for FFmpeg that stretches and resamples for real.

    It reads the ``atempo`` factor and the requested output rate out of the
    arguments it is handed, so a wrong argument shows up as a wrong duration
    rather than being papered over.
    """

    def __init__(self, *, write: bool = True) -> None:
        self.write = write
        self.arguments: list[list[str]] = []

    @property
    def argv(self) -> list[str]:
        assert self.arguments, "ffmpeg was never called"
        return self.arguments[-1]

    @property
    def filters(self) -> list[str]:
        """Every ``-af`` value this fake was called with."""

        values = []
        for arguments in self.arguments:
            if "-af" in arguments:
                values.append(arguments[arguments.index("-af") + 1])
        return values

    @staticmethod
    def _after(arguments: list[str], flag: str) -> str | None:
        if flag not in arguments:
            return None
        index = arguments.index(flag)
        return arguments[index + 1] if index + 1 < len(arguments) else None

    def __call__(self, arguments: list[str]) -> None:
        self.arguments.append(list(arguments))
        if not self.write:
            return

        source = Path(self._after(arguments, "-i") or "")
        samples, source_rate = sf.read(str(source), dtype="float32", always_2d=True)
        mono = samples.mean(axis=1, dtype=np.float32)

        factor = 1.0
        filter_value = self._after(arguments, "-af")
        if filter_value:
            assert filter_value.startswith("atempo="), filter_value
            factor = float(filter_value.split("=", 1)[1])

        out_rate = int(self._after(arguments, "-ar") or source_rate)
        target_seconds = (mono.shape[0] / source_rate) / factor
        frames = max(1, int(round(target_seconds * out_rate)))

        # Linear interpolation: enough to give the right length and rate.
        positions = np.linspace(0.0, max(mono.shape[0] - 1, 1), frames)
        resampled = np.interp(positions, np.arange(mono.shape[0]), mono).astype(np.float32)

        destination = Path(arguments[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(destination), resampled, out_rate, format="WAV", subtype="PCM_16")


@pytest.fixture
def ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeFfmpeg:
    fake = FakeFfmpeg()
    monkeypatch.setattr(timing, "_run_ffmpeg", fake)
    return fake


# ---------------------------------------------------------------------------
# The format contract
# ---------------------------------------------------------------------------


def test_alignment_rate_matches_the_rest_of_the_pipeline() -> None:
    assert PIPELINE_SAMPLE_RATE == separation.REQUIRED_SAMPLE_RATE
    assert PIPELINE_SAMPLE_RATE == video.PIPELINE_SAMPLE_RATE


def test_default_directory_follows_the_work_directory(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    settings = _settings(tmp_path)
    align_dialogue([_clip(tmp_path)], settings=settings)

    expected = Path(settings.work_dir) / TIMING_DIRECTORY_NAME / ALIGNED_DIRECTORY_NAME
    assert resolve_timing_directory(settings=settings) == expected
    assert expected.is_dir()


def test_explicit_directory_wins(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    align_dialogue([_clip(tmp_path)], output_dir=tmp_path / "run", settings=_settings(tmp_path))

    assert resolve_timing_directory(tmp_path / "run") == (
        tmp_path / "run" / TIMING_DIRECTORY_NAME / ALIGNED_DIRECTORY_NAME
    )


def test_empty_input_never_touches_ffmpeg(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    assert align_dialogue([], settings=_settings(tmp_path)) == []
    assert ffmpeg.arguments == []


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------


def test_line_is_stretched_to_its_window(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    """A 1.0 s line in a 0.8 s window is sped up by 1.25x."""

    aligned = align_dialogue([_clip(tmp_path, start=5.0, end=5.8)], settings=_settings(tmp_path))[0]

    assert ffmpeg.filters == ["atempo=1.250000"]
    assert aligned.required_tempo == pytest.approx(1.25)
    assert aligned.tempo == pytest.approx(1.25)
    assert aligned.speech_duration == pytest.approx(0.8, abs=0.01)
    assert aligned.fits
    assert aligned.original_window == pytest.approx(0.8)


def test_slow_down_is_also_pitch_preserving(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    aligned = align_dialogue([_clip(tmp_path, start=2.0, end=3.25)], settings=_settings(tmp_path))[0]

    assert ffmpeg.filters == ["atempo=0.800000"]
    assert aligned.tempo < 1.0
    assert aligned.speech_duration == pytest.approx(1.25, abs=0.01)
    assert aligned.fits


def test_a_line_that_already_fits_is_left_alone(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    aligned = align_dialogue([_clip(tmp_path, start=1.0, end=2.0)], settings=_settings(tmp_path))[0]

    assert aligned.tempo == 1.0
    assert ffmpeg.filters == []
    assert aligned.fits
    assert any("tolerance" in note for note in aligned.notes)


def test_a_tiny_difference_is_not_worth_a_filter(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    """A deviation inside the tolerance stays unstretched, so nothing is mangled."""

    window = SPEECH - EXACT_FIT_TOLERANCE_SECONDS / 2
    aligned = align_dialogue(
        [_clip(tmp_path, start=4.0, end=4.0 + window)], settings=_settings(tmp_path)
    )[0]

    assert aligned.tempo == 1.0
    assert ffmpeg.filters == []
    assert aligned.fits


def test_a_line_that_cannot_fit_is_reported_not_mangled(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    """A 2.5x speed-up is outside the band, so the line is clamped and reported."""

    aligned = align_dialogue([_clip(tmp_path, start=3.0, end=3.4)], settings=_settings(tmp_path))[0]

    assert aligned.tempo == pytest.approx(1.25)
    assert aligned.required_tempo == pytest.approx(2.5)
    assert not aligned.fits
    assert aligned.residual > 0
    assert any("outside" in note for note in aligned.notes)
    assert any("long" in note for note in aligned.notes)


def test_a_clamped_line_that_still_fits_says_nothing(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    """Clamping to the very edge of the band is not a defect worth reporting."""

    aligned = align_dialogue([_clip(tmp_path, start=5.0, end=5.8)], settings=_settings(tmp_path))[0]

    assert aligned.tempo == pytest.approx(aligned.required_tempo)
    assert aligned.fits
    assert not any("outside" in note for note in aligned.notes)


def test_the_tempo_band_is_configurable(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    """A line the default band cannot fit is fitted exactly by a wider one."""

    # A 1.0 s line in a two-thirds-of-a-second window needs 1.5x.
    clip = _clip(tmp_path, start=3.0, end=3.0 + SPEECH / 1.5)

    default = align_dialogue([clip], output_dir=tmp_path / "d", settings=_settings(tmp_path))[0]
    wider = align_dialogue(
        [clip],
        output_dir=tmp_path / "w",
        settings=_settings(tmp_path, timing_min_tempo=0.5, timing_max_tempo=2.0),
    )[0]

    assert default.tempo == pytest.approx(1.25)
    assert not default.fits
    assert wider.tempo == pytest.approx(1.5)
    assert wider.fits


# ---------------------------------------------------------------------------
# Pauses and placement
# ---------------------------------------------------------------------------


def test_the_line_keeps_its_original_start(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    """The leading pause moves the file, not the line: speech still starts on time."""

    aligned = align_dialogue([_clip(tmp_path, start=5.0, end=5.8)], settings=_settings(tmp_path))[0]

    assert aligned.start == pytest.approx(5.0 - LEAD)
    assert aligned.start + aligned.rendered_pause_before == pytest.approx(5.0)


def test_rendered_pauses_are_preserved(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    aligned = align_dialogue([_clip(tmp_path, start=5.0, end=5.8)], settings=_settings(tmp_path))[0]

    assert aligned.rendered_pause_before == pytest.approx(LEAD)
    assert aligned.rendered_pause_after == pytest.approx(TRAIL)
    assert aligned.duration == pytest.approx(LEAD + 0.8 + TRAIL, abs=0.01)


def test_a_line_at_the_very_start_has_its_leading_pause_trimmed(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    """The pause gives way, so the speech keeps its start instead of running late."""

    aligned = align_dialogue([_clip(tmp_path, start=0.05, end=1.05)], settings=_settings(tmp_path))[0]

    assert aligned.start == 0.0
    assert aligned.rendered_pause_before == pytest.approx(0.05)
    assert aligned.start + aligned.rendered_pause_before == pytest.approx(0.05)
    assert any("trimmed" in note for note in aligned.notes)


def test_the_written_file_has_the_duration_the_metadata_claims(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    aligned = align_dialogue([_clip(tmp_path, start=5.0, end=5.8)], settings=_settings(tmp_path))[0]

    info = sf.info(str(aligned.audio_path))
    assert info.samplerate == PIPELINE_SAMPLE_RATE
    assert info.channels == 1
    assert info.frames / info.samplerate == pytest.approx(aligned.duration, abs=0.01)


def test_only_the_speech_is_stretched(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    """The pauses keep their length; the speech absorbs the whole tempo change."""

    aligned = align_dialogue([_clip(tmp_path, start=5.0, end=5.8)], settings=_settings(tmp_path))[0]
    samples, rate = sf.read(str(aligned.audio_path), dtype="float32")

    lead_frames = int(round(LEAD * rate))
    assert float(np.abs(samples[:lead_frames]).max()) == 0.0
    assert float(np.abs(samples[lead_frames : lead_frames + 100]).max()) > 0.0


def test_aligning_one_clip_reports_the_same_result(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    clip = _clip(tmp_path, start=5.0, end=5.8)
    single = align_clip(clip, tmp_path / "one.wav", settings=_settings(tmp_path))
    batch = align_dialogue([clip], output_dir=tmp_path / "batch", settings=_settings(tmp_path))[0]

    assert single.tempo == batch.tempo
    assert single.start == batch.start
    assert single.duration == pytest.approx(batch.duration, abs=0.001)


def test_clips_keep_their_input_order(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    clips = [
        _clip(tmp_path, index=0, start=9.0, end=9.8, audio=_clip_file(tmp_path, name="a")),
        _clip(tmp_path, index=1, start=1.0, end=1.8, audio=_clip_file(tmp_path, name="b")),
    ]
    aligned = align_dialogue(clips, settings=_settings(tmp_path))

    assert [line.index for line in aligned] == [0, 1]
    assert aligned[0].start > aligned[1].start


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_missing_clip_audio_is_reported(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    with pytest.raises(MissingInputError):
        align_dialogue(
            [_clip(tmp_path, audio=tmp_path / "nope.wav")], settings=_settings(tmp_path)
        )


def test_a_wrong_entry_type_is_reported(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    with pytest.raises(InvalidClipError):
        align_dialogue(["not-a-clip"], settings=_settings(tmp_path))  # type: ignore[list-item]


def test_missing_ffmpeg_is_reported_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(timing, "shutil", types.SimpleNamespace(which=lambda name: None))

    with pytest.raises(MissingFfmpegError) as info:
        align_dialogue([_clip(tmp_path)], settings=_settings(tmp_path))

    assert "ffmpeg" in str(info.value)


def test_ffmpeg_failure_is_wrapped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        timing, "shutil", types.SimpleNamespace(which=lambda name: "/usr/bin/ffmpeg")
    )
    monkeypatch.setattr(
        timing,
        "subprocess",
        types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(
                returncode=1, stderr="Conversion failed!\n"
            )
        ),
    )

    with pytest.raises(StretchError) as info:
        align_dialogue([_clip(tmp_path)], settings=_settings(tmp_path))

    assert "Conversion failed" in str(info.value)


def test_a_wrong_output_rate_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class WrongRate(FakeFfmpeg):
        def __call__(self, arguments: list[str]) -> None:
            super().__call__(arguments)
            destination = Path(arguments[-1])
            samples, _ = sf.read(str(destination), dtype="float32")
            sf.write(str(destination), samples, 44_100, format="WAV", subtype="PCM_16")

    monkeypatch.setattr(timing, "_run_ffmpeg", WrongRate())

    with pytest.raises(InvalidAudioError):
        align_dialogue([_clip(tmp_path)], settings=_settings(tmp_path))


def test_silent_ffmpeg_output_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(timing, "_run_ffmpeg", FakeFfmpeg(write=False))

    with pytest.raises(MissingInputError):
        align_dialogue([_clip(tmp_path)], settings=_settings(tmp_path))


def test_tempo_bounds_must_not_be_inverted(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    settings = _settings(tmp_path, timing_min_tempo=1.5, timing_max_tempo=1.1)

    with pytest.raises(ConfigurationError) as info:
        align_dialogue([_clip(tmp_path)], settings=settings)

    assert "TIMING_MIN_TEMPO" in str(info.value)


def test_tempo_bounds_outside_atempo_are_rejected(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    settings = _settings(tmp_path, timing_min_tempo=0.25, timing_max_tempo=1.0)

    with pytest.raises(ConfigurationError) as info:
        align_dialogue([_clip(tmp_path)], settings=settings)

    assert "atempo" in str(info.value)


# ---------------------------------------------------------------------------
# The description of an aligned line
# ---------------------------------------------------------------------------


def test_to_dict_describes_the_line(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    aligned = align_dialogue([_clip(tmp_path, start=5.0, end=5.8)], settings=_settings(tmp_path))[0]
    payload = aligned.to_dict()

    assert payload["index"] == 0
    assert payload["speaker_id"] == "SPEAKER_00"
    assert payload["original_start"] == 5.0
    assert payload["original_window"] == pytest.approx(0.8)
    assert payload["fits"] is True
    assert payload["tempo"] == pytest.approx(1.25)
    assert payload["notes"] == list(aligned.notes)


def test_aligned_clip_rejects_a_bad_window(tmp_path: Path) -> None:
    clip = _clip(tmp_path)

    with pytest.raises(InvalidClipError):
        AlignedClip(
            index=0,
            clip=clip,
            audio_path=tmp_path / "x.wav",
            sample_rate=PIPELINE_SAMPLE_RATE,
            start=0.0,
            tempo=1.0,
            required_tempo=1.0,
            speech_duration=1.0,
            original_window=0.0,
            rendered_pause_before=0.0,
            rendered_pause_after=0.0,
        )


def test_aligned_clip_rejects_a_non_clip(tmp_path: Path) -> None:
    with pytest.raises(InvalidClipError):
        AlignedClip(
            index=0,
            clip="not a clip",  # type: ignore[arg-type]
            audio_path=tmp_path / "x.wav",
            sample_rate=PIPELINE_SAMPLE_RATE,
            start=0.0,
            tempo=1.0,
            required_tempo=1.0,
            speech_duration=1.0,
            original_window=1.0,
            rendered_pause_before=0.0,
            rendered_pause_after=0.0,
        )


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_real_ffmpeg_stretches_a_line_to_its_window(tmp_path: Path) -> None:
    """End-to-end: a 1.0 s line really does come back as a 0.8 s line.

    This covers the actual ``atempo`` behaviour - that it preserves length as
    expected and that the resampling to the pipeline rate happens in the same
    pass - rather than only the argument list.
    """

    clip = _clip(tmp_path, start=5.0, end=5.8)
    aligned = align_clip(clip, tmp_path / "aligned.wav", settings=_settings(tmp_path))

    assert aligned.tempo == pytest.approx(1.25)
    assert aligned.speech_duration == pytest.approx(0.8, abs=0.02)
    assert aligned.start == pytest.approx(4.9)
    assert aligned.fits

    info = sf.info(str(aligned.audio_path))
    assert info.samplerate == PIPELINE_SAMPLE_RATE
    assert info.frames / info.samplerate == pytest.approx(LEAD + 0.8 + TRAIL, abs=0.03)

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The room a line has before the next one starts
# ---------------------------------------------------------------------------


def _pair(
    root: Path,
    *,
    first_speech: float = SPEECH,
    first_window: float = 0.8,
    gap: float = 0.4,
) -> list[TtsClip]:
    """Two clips, with real audio files, spaced ``gap`` apart on the timeline."""

    first = _clip(
        root,
        index=0,
        start=5.0,
        end=5.0 + first_window,
        audio=_clip_file(root, speech=first_speech, name="first"),
        speech=first_speech,
    )
    second = _clip(
        root,
        index=1,
        start=5.0 + first_window + gap,
        end=5.0 + first_window + gap + 0.8,
        audio=_clip_file(root, speech=SPEECH, name="second"),
    )
    return [first, second]


def test_a_line_may_use_the_silence_after_it(tmp_path: Path) -> None:
    """Amharic needs longer than the English it replaces, and a gap is spare room.

    Fitting strictly to the original window is what forces a longer language to be
    squashed; the silence before the next line is already empty, and using it is what
    lets the line keep its natural pace.
    """

    # 1.0s of speech in a 0.8s window, with 2.0s of silence before the next line.
    clips = _pair(tmp_path, first_speech=1.0, first_window=0.8, gap=2.0)

    aligned = align_dialogue(
        clips, output_dir=tmp_path / "w", settings=_settings(tmp_path)
    )

    first = aligned[0]
    assert first.available is not None and first.available > 2.0
    assert first.tempo == pytest.approx(1.0, abs=0.01)
    assert first.trimmed == 0.0


def test_a_line_still_too_long_is_cut_only_when_that_is_asked_for(
    tmp_path: Path,
) -> None:
    """Cutting damages the performance, so it is opt-in and off by default.

    Two voices briefly at once is the lesser evil: a listener notices a word clipped off
    the end immediately, and on a real run where the text was too long this fired on 31 of
    38 lines, which sounds broken rather than fast.
    """

    # 4.0s of speech in a 0.8s window with the next line only 0.4s away: even the tempo
    # limit cannot fit it.
    clips = _pair(tmp_path, first_speech=4.0, first_window=0.8, gap=0.4)

    default = align_dialogue(
        clips, output_dir=tmp_path / "d", settings=_settings(tmp_path)
    )
    assert default[0].trimmed == 0.0
    assert default[0].overrun > 0
    assert any("left intact" in note for note in default[0].notes)

    cut = align_dialogue(
        clips,
        output_dir=tmp_path / "w",
        settings=_settings(tmp_path, timing_trim_to_fit=True),
    )
    assert cut[0].trimmed > 0
    assert any("not spoken over" in note for note in cut[0].notes)
    # The guarantee, when it is asked for.
    assert cut[0].end <= cut[1].start + 1e-6


def test_lines_never_overlap_when_trimming_is_asked_for(tmp_path: Path) -> None:
    """The property the deadline gives: no two dubbed lines sound at once."""

    clips = [
        _clip(
            tmp_path,
            index=i,
            start=5.0 + i * 1.1,
            end=5.8 + i * 1.1,
            audio=_clip_file(tmp_path, speech=1.4, name=f"c{i}"),
            speech=1.4,
        )
        for i in range(6)
    ]

    aligned = align_dialogue(
        clips,
        output_dir=tmp_path / "w",
        settings=_settings(tmp_path, timing_trim_to_fit=True),
    )

    for earlier, later in zip(aligned, aligned[1:]):
        assert earlier.end <= later.start + 1e-6


def test_nothing_is_cut_when_there_is_no_next_line(tmp_path: Path) -> None:
    """A lone line harms nobody by running long, so cutting it would be gratuitous."""

    clip = _clip(
        tmp_path,
        start=5.0,
        end=5.8,
        audio=_clip_file(tmp_path, speech=4.0, name="lone"),
        speech=4.0,
    )

    aligned = align_clip(clip, tmp_path / "a.wav", settings=_settings(tmp_path))

    assert aligned.trimmed == 0.0
    assert aligned.available == pytest.approx(0.8)


