"""Tests for the audio-separator CUDA adapter without model downloads."""

from __future__ import annotations

import gc
import sys
import types
import weakref
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.pipeline import separation
from app.pipeline.separation import (
    MissingStemError,
    REQUIRED_SAMPLE_RATE,
    SeparationInferenceError,
    StemPaths,
    UnsupportedSampleRateError,
    separate_stems,
)


class FakeCuda:
    def __init__(self, available: bool = True) -> None:
        self.available = available
        self.empty_cache_calls = 0

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


class FakeSeparator:
    instances: list["FakeSeparator"] = []
    device = "cuda:0"
    output_stems = ("Vocals", "Instrumental")

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.torch_device = self.device
        self.loaded_model: str | None = None
        self.separate_call: tuple[str, dict[str, str]] | None = None
        FakeSeparator.instances.append(self)

    def load_model(self, *, model_filename: str) -> None:
        self.loaded_model = model_filename

    def separate(
        self, audio_path: str, *, custom_output_names: dict[str, str]
    ) -> list[str]:
        self.separate_call = audio_path, custom_output_names
        output_paths: list[str] = []
        for label in self.output_stems:
            stem = custom_output_names[label]
            path = Path(self.kwargs["output_dir"]) / f"{stem}.wav"
            sf.write(
                path,
                np.ones((REQUIRED_SAMPLE_RATE, 2), dtype=np.float32) * 0.1,
                REQUIRED_SAMPLE_RATE,
            )
            output_paths.append(str(path))
        return output_paths


@pytest.fixture
def fake_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeCuda, Path]:
    FakeSeparator.instances = []
    FakeSeparator.device = "cuda:0"
    FakeSeparator.output_stems = ("Vocals", "Instrumental")
    cuda = FakeCuda()
    monkeypatch.setattr(separation, "_require_cuda", lambda: FakeTorch(cuda))
    package = types.ModuleType("audio_separator")
    package.__path__ = []  # type: ignore[attr-defined]
    module = types.ModuleType("audio_separator.separator")
    module.Separator = FakeSeparator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "audio_separator", package)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", module)
    return cuda, tmp_path


def _settings(root: Path) -> Settings:
    return Settings(
        input_dir=root,
        work_dir=root / "work",
        output_dir=root / "out",
        model_cache_dir=root / "models",
        device="cpu",
    )


def _write_audio(path: Path, sample_rate: int = REQUIRED_SAMPLE_RATE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        path,
        np.zeros((sample_rate, 2), dtype=np.float32),
        sample_rate,
        subtype="FLOAT",
    )
    return path


def test_two_stems_map_to_existing_contract_and_effects_are_silent(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    cuda, root = fake_separator
    source = _write_audio(root / "clip.wav")

    paths = separate_stems(
        source, root / "out", settings=_settings(root)
    )

    assert isinstance(paths, StemPaths)
    assert paths.as_dict() == {
        "speech": paths.speech,
        "music": paths.music,
        "effects": paths.effects,
    }
    assert paths.speech.name == "clip_speech.wav"
    assert paths.music.name == "clip_music.wav"
    assert paths.effects.name == "clip_effects.wav"
    assert sf.info(paths.speech).samplerate == REQUIRED_SAMPLE_RATE
    assert sf.info(paths.music).samplerate == REQUIRED_SAMPLE_RATE
    effects, _ = sf.read(paths.effects, dtype="float32")
    assert np.count_nonzero(effects) == 0

    (separator,) = FakeSeparator.instances
    assert separator.loaded_model == separation.MODEL_FILENAME
    assert separator.kwargs["chunk_duration"] is None
    assert separator.kwargs["output_format"] == "WAV"
    assert separator.separate_call is not None
    assert separator.separate_call[0] == str(source)
    assert separator.separate_call[1] == {
        "Vocals": "clip_speech",
        "Instrumental": "clip_music",
    }
    assert cuda.empty_cache_calls == 1
    separator_ref = weakref.ref(separator)
    del separator
    FakeSeparator.instances.clear()
    gc.collect()
    assert separator_ref() is None


def test_cuda_is_mandatory_and_device_fallback_is_rejected(
    fake_separator: tuple[FakeCuda, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = fake_separator
    FakeSeparator.device = "cpu"

    with pytest.raises(SeparationInferenceError, match="instead of CUDA"):
        separate_stems(
            _write_audio(root / "clip.wav"),
            root / "out",
            settings=_settings(root),
        )


def test_missing_output_stem_is_an_explicit_error(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    _, root = fake_separator
    FakeSeparator.output_stems = ("Vocals",)

    with pytest.raises(MissingStemError, match="music/instrumental"):
        separate_stems(
            _write_audio(root / "clip.wav"),
            root / "out",
            settings=_settings(root),
        )


def test_missing_cuda_fails_before_loading_separator(
    fake_separator: tuple[FakeCuda, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    cuda, root = fake_separator
    cuda.available = False
    monkeypatch.setattr(
        separation,
        "_require_cuda",
        lambda: (_ for _ in ()).throw(
            SeparationInferenceError("torch.cuda.is_available() is false")
        ),
    )

    with pytest.raises(SeparationInferenceError, match="is_available"):
        separate_stems(
            _write_audio(root / "clip.wav"),
            root / "out",
            settings=_settings(root),
        )
    assert FakeSeparator.instances == []


def test_non_48khz_input_is_rejected_without_resampling(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    _, root = fake_separator

    with pytest.raises(UnsupportedSampleRateError, match="48000 Hz"):
        separate_stems(
            _write_audio(root / "clip.wav", sample_rate=44_100),
            root / "out",
            settings=_settings(root),
        )

    assert FakeSeparator.instances == []


def test_progress_reports_load_and_inference_steps(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    _, root = fake_separator
    updates: list[str] = []

    separate_stems(
        _write_audio(root / "clip.wav"),
        root / "out",
        settings=_settings(root),
        on_progress=updates.append,
    )

    assert any("audio ready" in update for update in updates)
    assert any("loading BS-Roformer" in update for update in updates)
    assert any("loading model" in update for update in updates)
    assert any("inference started" in update for update in updates)
    assert any("separation completed" in update for update in updates)
