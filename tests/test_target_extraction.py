"""Tests for :mod:`app.pipeline.target_extraction`.

No model is imported or executed here. The separators are replaced either by a
:class:`FakeSeparator` that returns harmonic tones, or - for the backend API
tests - by a fake pipeline/model whose output has the exact shape the real
package produces. The suite therefore needs no torch, no checkpoint, and no
network.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.pipeline import target_extraction
from app.pipeline.diarization import CrosstalkRegion
from app.pipeline.target_extraction import (
    ExtractedTrack,
    ExtractionError,
    ExtractionRequest,
    MAX_REGISTER_RATIO,
    PyannoteSeparator,
    Refusal,
    SAMPLE_RATE,
    SpeechBrainSeparator,
    UnavailableError,
    associate,
    build_separator,
    extract_tracks,
)


def _tone(seconds: float, hz: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Return a voiced harmonic signal at ``hz`` - something the tracker can measure."""

    time = np.arange(int(round(seconds * rate))) / rate
    signal = np.sin(2 * np.pi * hz * time)
    signal += 0.5 * np.sin(2 * np.pi * 2 * hz * time)
    return (0.3 * signal).astype(np.float32)


def _region(start: float = 10.0, end: float = 10.6) -> CrosstalkRegion:
    return CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), start, end)


def _settings(
    tmp_path: Path,
    *,
    enabled: bool = True,
    backend: str = "pyannote",
    model: str = "pyannote/speech-separation-ami-1.0",
    token: str | None = "hf_test",
    context: float = 1.0,
) -> Settings:
    return Settings(
        input_dir=tmp_path / "input",
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "output",
        model_cache_dir=tmp_path / "models",
        huggingface_token=token,
        crosstalk_separation=enabled,
        crosstalk_separation_backend=backend,
        crosstalk_separation_model=model,
        crosstalk_separation_context_seconds=context,
    )


class FakeSeparator:
    """Return one harmonic tone per configured frequency, over the model window."""

    backend = "fake"
    model = "fake/separator"
    sample_rate = SAMPLE_RATE

    def __init__(self, frequencies: list[float | None], *, error: Exception | None = None):
        self.frequencies = frequencies
        self.error = error
        self.calls: list[tuple[int, int]] = []

    def separate(self, samples: np.ndarray, num_speakers: int) -> list[np.ndarray]:
        self.calls.append((len(samples), num_speakers))
        if self.error is not None:
            raise self.error
        sources: list[np.ndarray] = []
        for frequency in self.frequencies:
            if frequency is None:
                sources.append(np.zeros(len(samples), dtype=np.float32))
            else:
                sources.append(_tone(len(samples) / SAMPLE_RATE, frequency))
        return sources


# ---------------------------------------------------------------------------
# Availability: every way the stage can be unable to run must be a reason
# ---------------------------------------------------------------------------


def test_the_stage_is_off_by_default_and_says_so(tmp_path: Path) -> None:
    separator, reason = build_separator(_settings(tmp_path, enabled=False))

    assert separator is None
    assert reason is not None and "disabled" in reason


def test_an_unknown_backend_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    separator, reason = build_separator(_settings(tmp_path, backend="magic"))

    assert separator is None
    assert "unknown" in (reason or "")


def test_a_missing_token_is_a_reason_not_an_import_error(tmp_path: Path) -> None:
    """Without a token the gated checkpoint cannot be fetched, so say that first."""

    separator, reason = build_separator(_settings(tmp_path, token=None))

    assert separator is None
    assert "HUGGINGFACE_TOKEN" in (reason or "")


def test_a_checkpoint_that_cannot_be_fetched_degrades_gracefully(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A 403 from the gated model is expected; it must not end the run."""

    class FakePipeline:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return None

    torch_module = types.ModuleType("torch")
    torch_module.device = lambda value: value  # type: ignore[attr-defined]
    audio_module = types.ModuleType("pyannote.audio")
    audio_module.Pipeline = FakePipeline  # type: ignore[attr-defined]
    pyannote_module = types.ModuleType("pyannote")
    pyannote_module.audio = audio_module  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "pyannote", pyannote_module)
    monkeypatch.setitem(sys.modules, "pyannote.audio", audio_module)

    separator, reason = build_separator(_settings(tmp_path))

    assert separator is None
    assert "could not be downloaded" in (reason or "")


def test_the_backend_default_model_wins_when_the_other_default_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``--backend speechbrain`` must not be handed pyannote's checkpoint."""

    seen: dict[str, str] = {}

    def load(settings, model):
        seen["model"] = model
        return FakeSeparator([120.0])

    monkeypatch.setattr(target_extraction, "_load_speechbrain", load)

    separator, reason = build_separator(
        _settings(tmp_path, backend="speechbrain", model="pyannote/speech-separation-ami-1.0")
    )

    assert reason is None and separator is not None
    assert seen["model"] == "speechbrain/sepformer-whamr16k"


def test_an_explicit_custom_model_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, str] = {}

    def load(settings, model):
        seen["model"] = model
        return FakeSeparator([120.0])

    monkeypatch.setattr(target_extraction, "_load_speechbrain", load)

    build_separator(
        _settings(tmp_path, backend="speechbrain", model="someone/custom-sepformer")
    )

    assert seen["model"] == "someone/custom-sepformer"


def test_a_separator_that_fails_to_load_is_a_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def explode(settings, model):
        raise UnavailableError("device cuda is not available")

    monkeypatch.setattr(target_extraction, "_load_pyannote", explode)

    separator, reason = build_separator(_settings(tmp_path))

    assert separator is None
    assert reason == "device cuda is not available"


# ---------------------------------------------------------------------------
# Association: a blind source becomes a character only when the register says so
# ---------------------------------------------------------------------------


def test_each_voice_is_credited_by_its_register() -> None:
    region = _region()
    sources = [_tone(region.duration, 120.0), _tone(region.duration, 210.0)]

    tracks, refusals = associate(
        sources,
        region,
        ("SPEAKER_00", "SPEAKER_01"),
        {"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
        mixture_rms=0.3,
    )

    assert refusals == []
    by_speaker = {track.speaker_id: track for track in tracks}
    assert set(by_speaker) == {"SPEAKER_00", "SPEAKER_01"}
    assert by_speaker["SPEAKER_01"].measured_hz == pytest.approx(120.0, abs=3.0)
    assert by_speaker["SPEAKER_00"].measured_hz == pytest.approx(210.0, abs=3.0)
    assert by_speaker["SPEAKER_00"].start == region.start
    assert by_speaker["SPEAKER_00"].end == region.end
    assert by_speaker["SPEAKER_00"].duration == pytest.approx(region.duration)


def test_a_source_between_two_registers_is_refused() -> None:
    """Refusing is the only safe answer when the split is not decisive."""

    region = _region()
    sources = [_tone(region.duration, 155.0), _tone(region.duration, 210.0)]

    tracks, refusals = associate(
        sources,
        region,
        ("SPEAKER_00", "SPEAKER_01"),
        {"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
        mixture_rms=0.3,
    )

    assert [track.speaker_id for track in tracks] == ["SPEAKER_00"]
    refused = {refusal.speaker_id: refusal for refusal in refusals}
    assert set(refused) == {"SPEAKER_01"}
    assert "decisively" in refused["SPEAKER_01"].reason
    assert refused["SPEAKER_01"].measured_hz == pytest.approx(155.0, abs=3.0)


def test_a_voice_outside_the_characters_band_is_refused() -> None:
    region = _region()
    sources = [_tone(region.duration, 320.0)]

    tracks, refusals = associate(
        sources,
        region,
        ("SPEAKER_00",),
        {"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
        mixture_rms=0.3,
    )

    assert tracks == []
    assert refusals[0].speaker_id == "SPEAKER_00"
    assert f"{MAX_REGISTER_RATIO:.2f}x" in refusals[0].reason


def test_without_a_register_no_source_can_be_attributed() -> None:
    region = _region()
    sources = [_tone(region.duration, 120.0)]

    tracks, refusals = associate(
        sources,
        region,
        ("SPEAKER_01",),
        None,
        mixture_rms=0.3,
    )

    assert tracks == []
    assert "no register" in refusals[0].reason


def test_a_silent_source_is_not_a_voice() -> None:
    region = _region()

    tracks, refusals = associate(
        [np.zeros(int(region.duration * SAMPLE_RATE), dtype=np.float32)],
        region,
        ("SPEAKER_01",),
        {"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
        mixture_rms=0.3,
    )

    assert tracks == []
    assert "no measurable voice" in refusals[0].reason


# ---------------------------------------------------------------------------
# extract_tracks: window, crop and per-region failure isolation
# ---------------------------------------------------------------------------


def test_the_model_sees_context_and_the_track_is_cropped_back() -> None:
    audio = np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32)
    region = CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 1.0, 1.6)
    separator = FakeSeparator([120.0, 210.0])

    report = extract_tracks(
        audio,
        [ExtractionRequest(region, ("SPEAKER_00", "SPEAKER_01"))],
        separator=separator,
        registers={"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
        context_seconds=0.5,
    )

    # 0.5s before + region + 0.5s after = 1.6s of model input.
    assert separator.calls == [(int(1.6 * SAMPLE_RATE), 2)]
    assert report.available
    assert report.separated_regions == 1
    assert report.tracks == 2
    for track in report.regions[0].tracks:
        assert track.duration == pytest.approx(region.duration)
        assert len(track.samples) == int(round(region.duration * SAMPLE_RATE))


def test_one_failing_region_does_not_stop_the_next() -> None:
    audio = np.zeros(int(3.0 * SAMPLE_RATE), dtype=np.float32)
    regions = [
        CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 0.2, 0.8),
        CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 1.5, 2.1),
    ]

    class FlakySeparator(FakeSeparator):
        def separate(self, samples, num_speakers):  # noqa: ANN001, ANN201
            self.calls.append((len(samples), num_speakers))
            if len(self.calls) == 1:
                raise ExtractionError("model produced no sources")
            return [np.zeros(len(samples), dtype=np.float32)]

    separator = FlakySeparator([])
    report = extract_tracks(
        audio,
        [ExtractionRequest(region, ("SPEAKER_01",)) for region in regions],
        separator=separator,
        registers={"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
        context_seconds=0.0,
    )

    assert len(report.regions) == 2
    assert report.regions[0].tracks == ()
    assert "no sources" in report.regions[0].refusals[0].reason
    assert report.regions[1].tracks == ()  # the zero source is not a voice
    assert report.separated_regions == 0
    assert "0/2" in report.summary()
    assert len(separator.calls) == 2


def test_a_region_shorter_than_a_track_is_not_run_through_the_model() -> None:
    audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
    region = CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 0.0, 0.05)
    separator = FakeSeparator([120.0, 210.0])

    report = extract_tracks(
        audio,
        [ExtractionRequest(region, ("SPEAKER_01",))],
        separator=separator,
        registers={"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
    )

    assert separator.calls == []
    assert "too short" in report.regions[0].refusals[0].reason


def test_a_separator_at_the_wrong_rate_is_refused() -> None:
    class WrongRateSeparator(FakeSeparator):
        sample_rate = 8_000

    audio = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    report = extract_tracks(
        audio,
        [ExtractionRequest(_region(0.1, 0.7), ("SPEAKER_01",))],
        separator=WrongRateSeparator([120.0]),
        registers={"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
    )

    assert report.available is False
    assert "8000" in (report.unavailable_reason or "")


def test_the_report_is_json_safe_and_carries_no_samples() -> None:
    audio = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    region = _region(0.1, 0.7)
    report = extract_tracks(
        audio,
        [ExtractionRequest(region, ("SPEAKER_00", "SPEAKER_01"))],
        separator=FakeSeparator([120.0, 210.0]),
        registers={"SPEAKER_00": 200.0, "SPEAKER_01": 120.3},
        context_seconds=0.1,
    )

    payload = json.loads(json.dumps(report.as_dict()))

    assert payload["available"] is True
    assert payload["tracks"] == 2
    assert "samples" not in json.dumps(payload)
    assert set(payload["detail"][0]["tracks"][0]) == {
        "speaker_id",
        "start",
        "end",
        "measured_hz",
    }


# ---------------------------------------------------------------------------
# Backend API contracts (checked against the packages' real output shapes)
# ---------------------------------------------------------------------------


class _FakeTensor:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def unsqueeze(self, axis: int) -> "_FakeTensor":
        return _FakeTensor(np.expand_dims(self.array, axis))


class _FakeOutput:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def detach(self) -> "_FakeOutput":
        return self

    def cpu(self) -> "_FakeOutput":
        return self

    def numpy(self) -> np.ndarray:
        return self.array


def test_pyannote_sources_are_read_from_the_samples_by_source_matrix() -> None:
    region = _region()
    n = int(round(region.duration * SAMPLE_RATE))
    matrix = np.stack(
        [_tone(region.duration, 120.0), _tone(region.duration, 210.0)], axis=1
    )

    class FakeSources:
        data = matrix

    class FakePipeline:
        def __call__(self, file, **kwargs):
            assert file["sample_rate"] == SAMPLE_RATE
            assert file["waveform"].array.shape == (1, n)
            assert kwargs == {"num_speakers": 2}
            return object(), FakeSources()

    torch_module = types.SimpleNamespace(
        from_numpy=lambda array: _FakeTensor(array),
    )

    separator = PyannoteSeparator(FakePipeline(), torch_module, "fake/model")
    sources = separator.separate(np.zeros(n, dtype=np.float32), 2)

    assert len(sources) == 2
    first, second = (target_extraction._measure(source) for source in sources)
    assert first[0] == pytest.approx(120.0, abs=3.0)
    assert second[0] == pytest.approx(210.0, abs=3.0)


def test_speechbrain_sources_are_read_from_the_time_by_source_matrix() -> None:
    """SepFormer returns ``(batch, time, sources)``; reading axis 1 would return time."""

    n = int(round(0.6 * SAMPLE_RATE))
    matrix = np.stack([_tone(0.6, 120.0), _tone(0.6, 210.0)], axis=1)[None, ...]

    class FakeModel:
        def separate_batch(self, batch):  # noqa: ANN001, ANN201
            return _FakeOutput(matrix)

    torch_module = types.SimpleNamespace(
        from_numpy=lambda array: _FakeTensor(array),
    )

    separator = SpeechBrainSeparator(
        FakeModel(), torch_module, "fake/sepformer", SAMPLE_RATE
    )
    sources = separator.separate(np.zeros(n, dtype=np.float32), 2)

    assert len(sources) == 2
    assert all(len(source) == n for source in sources)
    medians = [target_extraction._measure(source)[0] for source in sources]
    assert medians[0] == pytest.approx(120.0, abs=3.0)
    assert medians[1] == pytest.approx(210.0, abs=3.0)


def test_extracted_track_slice_uses_absolute_times() -> None:
    track = ExtractedTrack(
        speaker_id="SPEAKER_00",
        start=10.0,
        end=10.5,
        samples=_tone(0.5, 200.0),
        measured_hz=200.0,
    )

    middle = track.slice(10.1, 10.3)

    # Floor/ceil at the edges may add a sample on either side; the span is right.
    assert abs(len(middle) - int(round(0.2 * SAMPLE_RATE))) <= 2
    assert float(np.max(np.abs(middle))) > 0.0
    assert len(track.slice(11.0, 12.0)) == 0


def test_refusal_describes_itself_in_the_region_report() -> None:
    region = _region()
    item = target_extraction.RegionExtraction(
        region=region,
        refusals=(Refusal("SPEAKER_01", "no measurable voice"),),
    )

    assert "SPEAKER_01 refused (no measurable voice)" in item.describe()
