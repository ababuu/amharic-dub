"""Tests for choosing a separator, and for the check that its stems kept the audio.

Separation is the most expensive stage - about 1.3x real time on a small workstation
GPU - so two things need proving without a GPU: that the lighter backend is wired up
correctly, and that a split which silently loses most of the signal is reported
instead of reaching the mixer as a bed with the music missing.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import separation
from app.pipeline.separation import (
    InvalidAudioError,
    REQUIRED_SAMPLE_RATE,
    SeparationInferenceError,
    separate_stems,
)


class FakeCuda:
    def __init__(self, available: bool = True, count: int = 1) -> None:
        self.available = available
        self.count = count
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return self.count

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    """Just enough torch for the demucs path: tensors are numpy arrays."""

    float32 = np.float32

    def __init__(self, cuda: FakeCuda | None = None) -> None:
        self.cuda = cuda or FakeCuda()

    @staticmethod
    def as_tensor(value: Any, dtype: Any = None) -> np.ndarray:
        return np.asarray(value, dtype=dtype or np.float32)


class FakeModel:
    """A demucs model, as ``demucs.pretrained.get_model`` returns one."""

    sources = ["drums", "bass", "other", "vocals"]
    samplerate = 44_100
    audio_channels = 2
    instances: list["FakeModel"] = []
    #: Each stem's amplitude as a fraction of the (normalised) mixture's own scale,
    #: which is how a real model's output behaves.
    levels = {"drums": 0.4, "bass": 0.5, "other": 0.6, "vocals": 0.7}

    def __init__(self, name: str) -> None:
        self.name = name
        self.device: str | None = None
        self.eval_calls = 0
        FakeModel.instances.append(self)

    def to(self, device: str) -> "FakeModel":
        self.device = device
        return self

    def eval(self) -> "FakeModel":
        self.eval_calls += 1
        return self


def _fake_apply_model(
    model: FakeModel, mix: Any, **_kwargs: Any
) -> np.ndarray:
    """Return ``(batch, stems, channels, samples)`` like ``demucs.apply.apply_model``.

    The returned levels follow the input's scale, because a real model's output is
    in the same units as the mixture it was given - and ``_run_demucs`` divides the
    mixture by its own standard deviation and multiplies it back afterwards.
    """

    array = np.asarray(mix, dtype=np.float32)
    channels, samples = array.shape[1], array.shape[2]
    scale = float(array.std()) or 1.0
    stems = np.stack(
        [
            np.full((channels, samples), model.levels[name] * scale, dtype=np.float32)
            for name in model.sources
        ]
    )
    return stems[None]


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "models",
        "device": "cuda",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


@pytest.fixture
def demucs(monkeypatch: pytest.MonkeyPatch) -> FakeTorch:
    """Install fake ``demucs.pretrained``, ``demucs.apply`` and ``torchaudio`` modules."""

    FakeModel.instances = []
    torch = FakeTorch()
    monkeypatch.setattr(separation, "_require_cuda", lambda: torch)

    pretrained = types.ModuleType("demucs.pretrained")
    pretrained.get_model = lambda name="htdemucs": FakeModel(name)  # type: ignore[attr-defined]
    apply = types.ModuleType("demucs.apply")
    apply.apply_model = _fake_apply_model  # type: ignore[attr-defined]
    package = types.ModuleType("demucs")
    package.pretrained = pretrained  # type: ignore[attr-defined]
    package.apply = apply  # type: ignore[attr-defined]
    functional = types.ModuleType("torchaudio.functional")
    # The fake source is 48 kHz, so no resample happens: identity keeps the levels
    # comparable with the assertions.
    functional.resample = lambda track, _old, _new: track  # type: ignore[attr-defined]
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.functional = functional  # type: ignore[attr-defined]
    for name, module in (
        ("demucs", package),
        ("demucs.pretrained", pretrained),
        ("demucs.apply", apply),
        ("torchaudio", torchaudio),
        ("torchaudio.functional", functional),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return torch


def _source(root: Path, seconds: float = 2.0) -> Path:
    """Write a source with real energy, so the stem probe has something to compare."""

    path = root / "clip.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.arange(int(seconds * REQUIRED_SAMPLE_RATE)) / REQUIRED_SAMPLE_RATE
    tone = 0.2 * np.sin(2.0 * np.pi * 220.0 * times)
    sf.write(
        str(path),
        np.stack([tone, tone], axis=1).astype(np.float32),
        REQUIRED_SAMPLE_RATE,
        subtype="FLOAT",
    )
    return path


def test_the_demucs_backend_writes_both_stems_at_the_pipeline_rate(
    demucs: FakeTorch, tmp_path: Path
) -> None:
    source = _source(tmp_path)

    paths = separate_stems(
        source, tmp_path / "out", settings=_settings(tmp_path, separation_backend="demucs")
    )

    (model,) = FakeModel.instances
    assert model.name == "htdemucs"
    assert model.device == "cuda:0"
    assert model.eval_calls == 1
    assert sf.info(str(paths.speech)).samplerate == REQUIRED_SAMPLE_RATE
    assert sf.info(str(paths.music)).samplerate == REQUIRED_SAMPLE_RATE

    # demucs works on a normalised mixture and is scaled back afterwards, so each
    # stem's level is a fraction of the source's own standard deviation.
    audio, _rate = sf.read(str(source), dtype="float32", always_2d=True)
    deviation = float(audio.mean(axis=1).std())
    speech, _ = sf.read(str(paths.speech), dtype="float32")
    assert np.allclose(speech, FakeModel.levels["vocals"] * deviation, atol=1e-3)
    # The bed is everything that is not the voice, and exactly that: dropping a stem
    # would leave the music thinner than the film's own.
    bed, _ = sf.read(str(paths.music), dtype="float32")
    expected = sum(FakeModel.levels[name] for name in ("drums", "bass", "other"))
    assert np.allclose(bed, expected * deviation, atol=1e-3)


def test_an_unknown_backend_is_named_rather_than_silently_replaced(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)

    with pytest.raises(SeparationInferenceError, match="unknown SEPARATION_BACKEND"):
        separate_stems(
            source,
            tmp_path / "out",
            settings=_settings(tmp_path, separation_backend="spleeter"),
        )


def test_demucs_refuses_a_cpu_device_instead_of_running_for_days(
    demucs: FakeTorch, tmp_path: Path
) -> None:
    """Separation on the CPU takes days, so DEVICE=cpu must be refused, not obeyed."""

    source = _source(tmp_path)

    with pytest.raises(SeparationInferenceError, match="does not select CUDA"):
        separate_stems(
            source,
            tmp_path / "out",
            settings=_settings(tmp_path, separation_backend="demucs", device="cpu"),
        )


def test_a_split_that_loses_the_audio_is_reported(tmp_path: Path) -> None:
    """A model that returns near-silence must fail loudly, not become a silent bed."""

    source = _source(tmp_path)
    quiet = tmp_path / "quiet.wav"
    loud = tmp_path / "loud.wav"
    frames = int(2.0 * REQUIRED_SAMPLE_RATE)
    sf.write(
        str(quiet),
        np.full((frames, 2), 0.001, dtype=np.float32),
        REQUIRED_SAMPLE_RATE,
        subtype="FLOAT",
    )
    sf.write(
        str(loud),
        np.full((frames, 2), 0.001, dtype=np.float32),
        REQUIRED_SAMPLE_RATE,
        subtype="FLOAT",
    )

    with pytest.raises(InvalidAudioError, match="lost most of the signal"):
        separation._probe_stem_energy(source, quiet, loud)


def test_a_healthy_split_passes_the_energy_check(tmp_path: Path) -> None:
    source = _source(tmp_path)
    speech = tmp_path / "speech.wav"
    music = tmp_path / "music.wav"
    frames = int(2.0 * REQUIRED_SAMPLE_RATE)
    # Split the 0.2 amplitude tone into two halves that add back up.
    sf.write(
        str(speech),
        np.full((frames, 2), 0.1, dtype=np.float32),
        REQUIRED_SAMPLE_RATE,
        subtype="FLOAT",
    )
    sf.write(
        str(music),
        np.full((frames, 2), 0.1, dtype=np.float32),
        REQUIRED_SAMPLE_RATE,
        subtype="FLOAT",
    )

    separation._probe_stem_energy(source, speech, music)


def test_nan_in_a_stem_is_reported(tmp_path: Path) -> None:
    """Half precision that overflows returns NaN, which must never reach the mixer."""

    source = _source(tmp_path)
    speech = tmp_path / "speech.wav"
    music = tmp_path / "music.wav"
    frames = int(2.0 * REQUIRED_SAMPLE_RATE)
    sf.write(
        str(speech),
        np.full((frames, 2), np.nan, dtype=np.float32),
        REQUIRED_SAMPLE_RATE,
        subtype="FLOAT",
    )
    sf.write(
        str(music),
        np.zeros((frames, 2), dtype=np.float32),
        REQUIRED_SAMPLE_RATE,
        subtype="FLOAT",
    )

    with pytest.raises(InvalidAudioError, match="NaN"):
        separation._probe_stem_energy(source, speech, music)


def test_options_the_installed_library_does_not_declare_are_dropped() -> None:
    """audio-separator gains and loses constructor options between releases."""

    def old_factory(model_file_dir: str, output_dir: str) -> None:
        pass

    accepted = separation._accepted_kwargs(
        old_factory,
        {"model_file_dir": "a", "output_dir": "b", "use_autocast": True},
    )

    assert accepted == {"model_file_dir": "a", "output_dir": "b"}


def test_options_are_passed_through_to_a_permissive_constructor() -> None:
    def permissive(**kwargs: Any) -> None:
        pass

    accepted = separation._accepted_kwargs(permissive, {"use_autocast": True})

    assert accepted == {"use_autocast": True}
