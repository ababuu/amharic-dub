"""Tests for :mod:`app.pipeline.video`.

No model and no network: the FFmpeg subprocess seam is replaced by an in-process
fake that writes a real WAV, so every check runs against actual audio files. One
end-to-end test uses the real FFmpeg, and is skipped when it is not installed.
"""

from __future__ import annotations

import shutil
import subprocess
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import separation, video
from app.pipeline.video import (
    DEFAULT_DUB_SUFFIX,
    DEFAULT_TRACK_SUFFIX,
    MUX_AUDIO_LANGUAGE,
    PIPELINE_CHANNELS,
    PIPELINE_CODEC,
    PIPELINE_SAMPLE_RATE,
    ExtractionError,
    InvalidAudioError,
    MissingFfmpegError,
    MissingInputError,
    extract_audio,
)

SOURCE_RATE = 44_100


def _settings(root: Path, **overrides: object) -> Settings:
    """Return settings pointing every directory at ``root``."""

    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "cache",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write_video_stand_in(root: Path, name: str = "movie.mp4") -> Path:
    """Write a file the extractor will accept as a source video.

    The extractor never parses the container itself - FFmpeg does - so a file
    that merely exists is enough for everything except the real-FFmpeg test.
    """

    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not really an mp4")
    return path


class FakeFfmpeg:
    """Stand-in for the FFmpeg subprocess that writes a real WAV.

    The rate, channel count and duration are read back out of the arguments the
    extractor passed, so the fake honours the request the way FFmpeg would and a
    wrong request shows up as a wrong result.
    """

    def __init__(
        self,
        *,
        rate: int | None = None,
        channels: int | None = None,
        seconds: float = 1.0,
        write: bool = True,
    ) -> None:
        self.rate = rate
        self.channels = channels
        self.seconds = seconds
        self.write = write
        self.arguments: list[list[str]] = []

    @property
    def argv(self) -> list[str]:
        """The argv of the most recent call."""

        assert self.arguments, "ffmpeg was never called"
        return self.arguments[-1]

    @staticmethod
    def _value_after(arguments: list[str], flag: str) -> str | None:
        if flag not in arguments:
            return None
        index = arguments.index(flag)
        return arguments[index + 1] if index + 1 < len(arguments) else None

    def __call__(self, arguments: list[str]) -> None:
        self.arguments.append(list(arguments))
        if not self.write:
            return

        rate = self.rate or int(self._value_after(arguments, "-ar") or PIPELINE_SAMPLE_RATE)
        channels = self.channels or int(
            self._value_after(arguments, "-ac") or PIPELINE_CHANNELS
        )
        destination = Path(arguments[-1])
        destination.parent.mkdir(parents=True, exist_ok=True)
        frames = int(self.seconds * rate)
        samples = np.full((frames, channels), 0.25, dtype=np.float32)
        sf.write(str(destination), samples, rate, format="WAV", subtype="PCM_16")


@pytest.fixture
def ffmpeg(monkeypatch: pytest.MonkeyPatch) -> FakeFfmpeg:
    """Replace the FFmpeg subprocess seam with an in-process fake."""

    fake = FakeFfmpeg()
    monkeypatch.setattr(video, "_run_ffmpeg", fake)
    return fake


# ---------------------------------------------------------------------------
# The format contract
# ---------------------------------------------------------------------------


def test_extraction_rate_matches_what_separation_requires() -> None:
    """The extractor and the separator must agree on the sample rate."""

    assert PIPELINE_SAMPLE_RATE == separation.REQUIRED_SAMPLE_RATE


def test_default_destination_follows_the_work_directory(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    source = _write_video_stand_in(tmp_path)
    settings = _settings(tmp_path)

    result = extract_audio(source, settings=settings)

    assert result == Path(settings.work_dir) / f"movie{DEFAULT_TRACK_SUFFIX}"
    assert result.is_file()


def test_explicit_destination_is_created(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    source = _write_video_stand_in(tmp_path)
    destination = tmp_path / "deep" / "nested" / "track.wav"

    result = extract_audio(source, destination, settings=_settings(tmp_path))

    assert result == destination
    assert destination.is_file()


def test_requested_format_is_exactly_the_pipeline_format(
    tmp_path: Path, ffmpeg: FakeFfmpeg
) -> None:
    source = _write_video_stand_in(tmp_path)

    extract_audio(source, tmp_path / "track.wav", settings=_settings(tmp_path))

    argv = ffmpeg.argv
    assert argv[0] == "-hide_banner"
    assert argv[argv.index("-i") + 1] == str(source)
    assert "-vn" in argv
    assert argv[argv.index("-ac") + 1] == str(PIPELINE_CHANNELS)
    assert argv[argv.index("-ar") + 1] == str(PIPELINE_SAMPLE_RATE)
    assert argv[argv.index("-c:a") + 1] == PIPELINE_CODEC
    assert argv[-1] == str(tmp_path / "track.wav")


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_missing_source_is_reported(tmp_path: Path, ffmpeg: FakeFfmpeg) -> None:
    with pytest.raises(MissingInputError):
        extract_audio(tmp_path / "nope.mp4", settings=_settings(tmp_path))
    assert ffmpeg.arguments == []


def test_missing_ffmpeg_is_reported_clearly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video_stand_in(tmp_path)
    monkeypatch.setattr(
        video, "shutil", types.SimpleNamespace(which=lambda name: None)
    )

    with pytest.raises(MissingFfmpegError) as info:
        extract_audio(source, tmp_path / "track.wav", settings=_settings(tmp_path))

    assert "ffmpeg" in str(info.value)


def test_ffmpeg_failure_is_wrapped_with_its_last_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video_stand_in(tmp_path)
    monkeypatch.setattr(
        video, "shutil", types.SimpleNamespace(which=lambda name: "/usr/bin/ffmpeg")
    )
    monkeypatch.setattr(
        video,
        "subprocess",
        types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(
                returncode=1, stderr="warn\nInvalid data found\n"
            )
        ),
    )

    with pytest.raises(ExtractionError) as info:
        extract_audio(source, tmp_path / "track.wav", settings=_settings(tmp_path))

    assert "Invalid data found" in str(info.value)


def test_ffmpeg_writing_nothing_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video_stand_in(tmp_path)
    monkeypatch.setattr(video, "_run_ffmpeg", FakeFfmpeg(write=False))

    with pytest.raises(ExtractionError):
        extract_audio(source, tmp_path / "track.wav", settings=_settings(tmp_path))


def test_empty_output_file_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video_stand_in(tmp_path)
    destination = tmp_path / "track.wav"

    def _touch_empty(arguments: list[str]) -> None:
        Path(arguments[-1]).write_bytes(b"")

    monkeypatch.setattr(video, "_run_ffmpeg", _touch_empty)

    with pytest.raises(ExtractionError):
        extract_audio(source, destination, settings=_settings(tmp_path))


def test_wrong_sample_rate_output_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video_stand_in(tmp_path)
    monkeypatch.setattr(video, "_run_ffmpeg", FakeFfmpeg(rate=SOURCE_RATE))

    with pytest.raises(InvalidAudioError) as info:
        extract_audio(source, tmp_path / "track.wav", settings=_settings(tmp_path))

    assert str(SOURCE_RATE) in str(info.value)


def test_mono_output_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video_stand_in(tmp_path)
    monkeypatch.setattr(video, "_run_ffmpeg", FakeFfmpeg(channels=1))

    with pytest.raises(InvalidAudioError):
        extract_audio(source, tmp_path / "track.wav", settings=_settings(tmp_path))


def test_audio_without_samples_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video_stand_in(tmp_path)
    monkeypatch.setattr(video, "_run_ffmpeg", FakeFfmpeg(seconds=0.0))

    with pytest.raises(InvalidAudioError):
        extract_audio(source, tmp_path / "track.wav", settings=_settings(tmp_path))


def test_unreadable_output_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video_stand_in(tmp_path)

    def _write_garbage(arguments: list[str]) -> None:
        Path(arguments[-1]).write_bytes(b"RIFF not really a wav")

    monkeypatch.setattr(video, "_run_ffmpeg", _write_garbage)

    with pytest.raises(InvalidAudioError) as info:
        extract_audio(source, tmp_path / "track.wav", settings=_settings(tmp_path))

    assert "could not read" in str(info.value)


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg is not installed")
def test_real_ffmpeg_resamples_a_44k_mono_source(tmp_path: Path) -> None:
    """End-to-end extraction of the format the rest of the pipeline consumes.

    The source is deliberately 44.1 kHz mono, the shape that produced the earlier
    decode errors, so this covers the resampling and up-mixing FFmpeg has to do
    rather than only the argument list.
    """

    source = tmp_path / "source.wav"
    frames = int(0.5 * SOURCE_RATE)
    sf.write(
        str(source),
        np.full(frames, 0.3, dtype=np.float32),
        SOURCE_RATE,
        format="WAV",
        subtype="PCM_16",
    )
    destination = tmp_path / "track.wav"

    result = extract_audio(source, destination, settings=_settings(tmp_path))

    info = sf.info(str(result))
    assert result == destination
    assert info.samplerate == PIPELINE_SAMPLE_RATE
    assert info.channels == PIPELINE_CHANNELS
    assert info.frames > 0


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------


def _payload(
    *,
    video_codecs: tuple[str, ...] = ("h264",),
    audio: tuple[tuple[str, int, int], ...] = (("aac", 48_000, 2),),
    duration: float = 1.0,
    format_name: str = "mov,mp4,m4a,3gp,3g2,mj2",
) -> dict[str, object]:
    """Build an ``ffprobe -of json`` payload."""

    streams: list[dict[str, object]] = []
    for codec in video_codecs:
        streams.append(
            {
                "index": len(streams),
                "codec_type": "video",
                "codec_name": codec,
                "duration": str(duration),
            }
        )
    for codec, rate, channels in audio:
        streams.append(
            {
                "index": len(streams),
                "codec_type": "audio",
                "codec_name": codec,
                "sample_rate": str(rate),
                "channels": channels,
                "duration": str(duration),
            }
        )
    return {
        "streams": streams,
        "format": {"format_name": format_name, "duration": str(duration)},
    }


class FakeProbe:
    """Stand-in for the ffprobe seam, keyed by file name."""

    def __init__(self, payloads: dict[str, dict[str, object]]) -> None:
        self.payloads = payloads
        self.arguments: list[list[str]] = []

    def __call__(self, arguments: list[str]) -> dict[str, object]:
        self.arguments.append(list(arguments))
        name = Path(arguments[-1]).name
        if name not in self.payloads:
            raise AssertionError(f"no probe payload prepared for {name}")
        return self.payloads[name]


@pytest.fixture
def probe(monkeypatch: pytest.MonkeyPatch) -> FakeProbe:
    """Replace the ffprobe seam with prepared payloads."""

    fake = FakeProbe(
        {
            "source.mp4": _payload(video_codecs=("h264",), audio=(("aac", 44_100, 2),)),
            "dub.mp4": _payload(video_codecs=("h264",), audio=(("aac", 48_000, 2),)),
            "dub.mkv": _payload(video_codecs=("h264",), audio=(("aac", 48_000, 2),)),
            "mix.wav": _payload(video_codecs=(), duration=1.0),
        }
    )
    monkeypatch.setattr(video, "_run_ffprobe", fake)
    return fake


def _write_mix(
    root: Path,
    name: str = "mix.wav",
    *,
    seconds: float = 1.0,
    rate: int = PIPELINE_SAMPLE_RATE,
    channels: int = PIPELINE_CHANNELS,
) -> Path:
    """Write a stand-in final mix."""

    frames = int(round(seconds * rate))
    samples = np.full((frames, channels), 0.25, dtype=np.float32)
    path = root / name
    sf.write(str(path), samples, rate, format="WAV", subtype="PCM_16")
    return path


def _write_video(root: Path, name: str = "source.mp4") -> Path:
    path = root / name
    path.write_bytes(b"stand-in video")
    return path


def test_probe_reads_streams_and_duration(tmp_path: Path, probe: FakeProbe) -> None:
    source = _write_video(tmp_path)

    info = video.probe(source)

    assert info.path == source
    assert info.duration == pytest.approx(1.0)
    assert [stream.codec_name for stream in info.video_streams] == ["h264"]
    assert info.audio_streams[0].sample_rate == 44_100
    assert info.audio_streams[0].channels == 2


def test_probe_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(MissingInputError):
        video.probe(tmp_path / "nope.mp4")


def test_probe_reports_a_failed_ffprobe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _write_video(tmp_path)
    monkeypatch.setattr(
        video, "shutil", types.SimpleNamespace(which=lambda name: "/usr/bin/ffprobe")
    )
    monkeypatch.setattr(
        video,
        "subprocess",
        types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(
                returncode=1, stdout="", stderr="No such file or directory\n"
            )
        ),
    )

    with pytest.raises(video.ProbeError) as info:
        video.probe(source)

    assert "No such file or directory" in str(info.value)


def test_probe_reports_missing_ffprobe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    monkeypatch.setattr(video, "shutil", types.SimpleNamespace(which=lambda name: None))

    with pytest.raises(video.MissingFfprobeError):
        video.probe(source)


def test_probe_rejects_unparseable_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    monkeypatch.setattr(
        video, "shutil", types.SimpleNamespace(which=lambda name: "/usr/bin/ffprobe")
    )
    monkeypatch.setattr(
        video,
        "subprocess",
        types.SimpleNamespace(
            run=lambda *a, **k: types.SimpleNamespace(
                returncode=0, stdout="not json at all", stderr=""
            )
        ),
    )

    with pytest.raises(video.ProbeError):
        video.probe(source)


# ---------------------------------------------------------------------------
# Muxing the dub back in
# ---------------------------------------------------------------------------


def test_mux_copies_the_video_and_replaces_the_audio(
    tmp_path: Path, ffmpeg: FakeFfmpeg, probe: FakeProbe
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)

    video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    argv = ffmpeg.argv
    assert argv[argv.index("-i") + 1] == str(source)
    assert argv.index("-i", argv.index("-i") + 1) >= 0
    assert argv[argv.index("-map") + 1] == "0:v"
    assert argv[argv.index("-map", argv.index("-map") + 1) + 1] == "1:a:0"
    assert argv[argv.index("-c:v") + 1] == "copy"
    assert argv[argv.index("-c:a") + 1] == "aac"
    assert argv[argv.index("-b:a") + 1] == "192k"
    assert argv[argv.index("-ar") + 1] == str(PIPELINE_SAMPLE_RATE)
    assert argv[argv.index("-ac") + 1] == str(PIPELINE_CHANNELS)
    assert f"language={MUX_AUDIO_LANGUAGE}" in argv
    assert argv[-1] == str(tmp_path / "dub.mp4")


def test_mux_never_maps_the_original_audio(
    tmp_path: Path, ffmpeg: FakeFfmpeg, probe: FakeProbe
) -> None:
    """Carrying the source audio over would leave the English dialogue audible."""

    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)

    video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    maps = [
        ffmpeg.argv[index + 1]
        for index, value in enumerate(ffmpeg.argv)
        if value == "-map"
    ]
    assert maps == ["0:v", "1:a:0"]
    assert not any(entry.startswith("0:a") for entry in maps)


def test_mux_asks_for_faststart_only_where_it_is_valid(
    tmp_path: Path, ffmpeg: FakeFfmpeg, probe: FakeProbe
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)

    video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))
    assert "+faststart" in ffmpeg.argv

    ffmpeg.arguments.clear()
    video.mux_dub(source, mix, tmp_path / "dub.mkv", settings=_settings(tmp_path))
    assert "+faststart" not in ffmpeg.argv


def test_mux_defaults_to_the_output_directory(
    tmp_path: Path, ffmpeg: FakeFfmpeg, probe: FakeProbe
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    settings = _settings(tmp_path)
    probe.payloads["source_amharic.mp4"] = _payload()

    result = video.mux_dub(source, mix, settings=settings)

    assert result.path == Path(settings.output_dir) / f"source{DEFAULT_DUB_SUFFIX}"
    assert result.path.is_file()


def test_mux_reports_what_it_verified(
    tmp_path: Path, ffmpeg: FakeFfmpeg, probe: FakeProbe
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)

    result = video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    assert result.video_codec == "h264"
    assert result.audio_codec == "aac"
    assert result.audio_sample_rate == PIPELINE_SAMPLE_RATE
    assert result.audio_channels == 2
    assert result.video_streams == 1
    assert result.audio_streams == 1
    assert any("replaced" in note for note in result.notes)
    assert result.to_dict()["path"].endswith("dub.mp4")


def test_mux_notes_a_length_mismatch(
    tmp_path: Path, ffmpeg: FakeFfmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    monkeypatch.setattr(
        video,
        "_run_ffprobe",
        FakeProbe(
            {
                "source.mp4": _payload(duration=1.0),
                "dub.mp4": _payload(video_codecs=("h264",), duration=3.0),
            }
        ),
    )

    result = video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    assert any("drift apart" in note for note in result.notes)


def test_mux_notes_a_changed_video_codec(
    tmp_path: Path, ffmpeg: FakeFfmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    monkeypatch.setattr(
        video,
        "_run_ffprobe",
        FakeProbe(
            {
                "source.mp4": _payload(video_codecs=("h264",)),
                "dub.mp4": _payload(video_codecs=("hevc",)),
            }
        ),
    )

    result = video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    assert any("codec changed" in note for note in result.notes)


# ---------------------------------------------------------------------------
# Mux failures
# ---------------------------------------------------------------------------


def test_mux_reports_a_source_without_video(
    tmp_path: Path, ffmpeg: FakeFfmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    monkeypatch.setattr(
        video, "_run_ffprobe", FakeProbe({"source.mp4": _payload(video_codecs=())})
    )

    with pytest.raises(InvalidAudioError) as info:
        video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    assert "no video stream" in str(info.value)


def test_mux_rejects_a_mix_that_is_not_the_pipeline_rate(
    tmp_path: Path, ffmpeg: FakeFfmpeg, probe: FakeProbe
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path, rate=44_100)

    with pytest.raises(InvalidAudioError) as info:
        video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    assert "44100" in str(info.value)


def test_mux_rejects_a_mono_mix(
    tmp_path: Path, ffmpeg: FakeFfmpeg, probe: FakeProbe
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path, channels=1)

    with pytest.raises(InvalidAudioError):
        video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))


def test_mux_reports_missing_inputs(
    tmp_path: Path, ffmpeg: FakeFfmpeg, probe: FakeProbe
) -> None:
    with pytest.raises(MissingInputError):
        video.mux_dub(
            tmp_path / "nope.mp4", _write_mix(tmp_path), tmp_path / "dub.mp4",
            settings=_settings(tmp_path),
        )
    with pytest.raises(MissingInputError):
        video.mux_dub(
            _write_video(tmp_path), tmp_path / "nope.wav", tmp_path / "dub.mp4",
            settings=_settings(tmp_path),
        )


def test_mux_rejects_a_result_with_two_audio_streams(
    tmp_path: Path, ffmpeg: FakeFfmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    monkeypatch.setattr(
        video,
        "_run_ffprobe",
        FakeProbe(
            {
                "source.mp4": _payload(),
                "dub.mp4": _payload(audio=(("aac", 48_000, 2), ("aac", 48_000, 2))),
            }
        ),
    )

    with pytest.raises(video.MuxError) as info:
        video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    assert "audio streams" in str(info.value)


def test_mux_rejects_a_result_with_the_wrong_audio_codec(
    tmp_path: Path, ffmpeg: FakeFfmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    monkeypatch.setattr(
        video,
        "_run_ffprobe",
        FakeProbe(
            {
                "source.mp4": _payload(),
                "dub.mp4": _payload(audio=(("pcm_s16le", 48_000, 2),)),
            }
        ),
    )

    with pytest.raises(video.MuxError) as info:
        video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    assert "aac" in str(info.value)


def test_mux_rejects_a_result_at_the_wrong_rate(
    tmp_path: Path, ffmpeg: FakeFfmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    monkeypatch.setattr(
        video,
        "_run_ffprobe",
        FakeProbe(
            {
                "source.mp4": _payload(),
                "dub.mp4": _payload(audio=(("aac", 44_100, 2),)),
            }
        ),
    )

    with pytest.raises(video.MuxError):
        video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))


def test_mux_rejects_a_result_without_video(
    tmp_path: Path, ffmpeg: FakeFfmpeg, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    monkeypatch.setattr(
        video,
        "_run_ffprobe",
        FakeProbe(
            {"source.mp4": _payload(), "dub.mp4": _payload(video_codecs=())}
        ),
    )

    with pytest.raises(video.MuxError):
        video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))


def test_mux_reports_ffmpeg_writing_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, probe: FakeProbe
) -> None:
    source = _write_video(tmp_path)
    mix = _write_mix(tmp_path)
    monkeypatch.setattr(video, "_run_ffmpeg", FakeFfmpeg(write=False))

    with pytest.raises(video.MuxError):
        video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


def _make_test_video(destination: Path) -> bool:
    """Encode a tiny real MP4, returning ``False`` if this build cannot."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        return False
    process = subprocess.run(  # noqa: S603 - argv list, no shell involved
        [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x48:rate=10:duration=1",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1",
            "-c:v",
            "mpeg4",
            "-q:v",
            "5",
            "-c:a",
            "aac",
            "-shortest",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return process.returncode == 0 and destination.is_file()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="FFmpeg is not installed",
)
def test_real_ffmpeg_replaces_the_audio_track_of_a_real_video(tmp_path: Path) -> None:
    """End-to-end: the picture survives and the audio is really the new track.

    This is the check that would have caught the earlier mux failure, where the
    generated audio's sample rate did not match the video's and the result would
    not decode.
    """

    source = tmp_path / "source.mp4"
    if not _make_test_video(source):
        pytest.skip("this FFmpeg build cannot encode the test video")

    before = video.probe(source)
    assert before.audio_streams, "the test video should have an audio track"
    assert before.audio_streams[0].sample_rate != PIPELINE_SAMPLE_RATE

    mix = _write_mix(tmp_path, seconds=1.0)
    result = video.mux_dub(source, mix, tmp_path / "dub.mp4", settings=_settings(tmp_path))

    assert result.video_codec == before.video_streams[0].codec_name
    assert result.audio_codec == "aac"
    assert result.audio_sample_rate == PIPELINE_SAMPLE_RATE
    assert result.audio_channels == PIPELINE_CHANNELS
    assert result.audio_streams == 1
    assert result.video_streams == len(before.video_streams)
    assert result.path.stat().st_size > 0

    after = video.probe(result.path)
    assert len(after.audio_streams) == 1
    assert after.audio_streams[0].sample_rate == PIPELINE_SAMPLE_RATE
    assert after.video_streams[0].codec_name == before.video_streams[0].codec_name
