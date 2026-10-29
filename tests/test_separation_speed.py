"""Tests for the audio-separator CUDA adapter without model downloads."""

from __future__ import annotations

import builtins
import dataclasses
import gc
import json
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
    InvalidAudioError,
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
        self.selected_device: Any | None = None

    def is_available(self) -> bool:
        return self.available

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def set_device(self, device: Any) -> None:
        self.selected_device = device

    def device_count(self) -> int:
        return 1

    def get_device_name(self, device: Any) -> str:
        return "Fake A40"


class FakeDevice:
    def __init__(self, name: str) -> None:
        self.type = name.split(":")[0]
        self.index = int(name.split(":")[1]) if ":" in name else None

    def __str__(self) -> str:
        return f"{self.type}:{self.index}" if self.index is not None else self.type


class FakeParameter:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device


class FakeModel:
    def __init__(self, device: FakeDevice) -> None:
        self.device = device

    def parameters(self):
        yield FakeParameter(self.device)


class FakeModelInstance:
    device = "cuda:0"

    def __init__(self, device: Any) -> None:
        self.torch_device = device
        self.model_run = FakeModel(device)


class FakeTorch:
    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda
        self.version = types.SimpleNamespace(cuda="12.8")

    @staticmethod
    def device(name: str) -> FakeDevice:
        return FakeDevice(name)


class FakeSeparator:
    instances: list["FakeSeparator"] = []
    model_device = "cuda:0"
    output_stems = ("Vocals", "Instrumental")

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.torch_device = FakeDevice("cpu")
        self.model_instance: FakeModelInstance | None = None
        self.loaded_model: str | None = None
        self.separate_call: tuple[str, dict[str, str]] | None = None
        FakeSeparator.instances.append(self)

    def load_model(self, *, model_filename: str) -> None:
        self.loaded_model = model_filename
        self.model_instance = FakeModelInstance(FakeDevice(self.model_device))

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
            # audio-separator returns its bare filename although it writes into output_dir.
            output_paths.append(path.name)
        return output_paths


@pytest.fixture
def fake_separator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[FakeCuda, Path]:
    FakeSeparator.instances = []
    FakeSeparator.model_device = "cuda:0"
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
    assert str(cuda.selected_device) == "cuda:0"
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


def test_stems_already_beside_the_source_are_reused_instead_of_separated_again(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    """Separation costs over an hour for a feature film, so a rerun must not redo it."""

    _, root = fake_separator
    source = _write_audio(root / "clip.wav")
    out = root / "out"
    first = separate_stems(source, out, settings=_settings(root))
    FakeSeparator.instances.clear()

    second = separate_stems(source, out, settings=_settings(root))

    assert second.speech == first.speech
    assert second.music == first.music
    # No model was loaded and no inference ran.
    assert FakeSeparator.instances == []


def test_stems_from_a_different_backend_are_not_reused(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    """Switching SEPARATION_BACKEND must take effect, not silently reuse old stems."""

    _, root = fake_separator
    source = _write_audio(root / "clip.wav")
    out = root / "out"
    separate_stems(source, out, settings=_settings(root))

    record = json.loads((out / "clip_stems.json").read_text(encoding="utf-8"))
    info = sf.info(str(source))
    demucs = separation._reusable_stems(
        source,
        out,
        info,
        settings=_settings(root, separation_backend="demucs"),
        on_progress=None,
    )

    assert record["backend"] == "roformer"
    assert demucs is None


def test_stems_without_a_record_are_reused_by_the_default_backend(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    """Stems made before the record existed are RoFormer's, so they are still reusable."""

    _, root = fake_separator
    source = _write_audio(root / "clip.wav")
    out = root / "out"
    first = separate_stems(source, out, settings=_settings(root))
    (out / "clip_stems.json").unlink()

    second = separation._reusable_stems(
        source,
        out,
        sf.info(str(source)),
        settings=_settings(root, separation_backend="roformer"),
        on_progress=None,
    )

    assert second is not None
    assert second.speech == first.speech


def test_a_stem_of_the_wrong_length_is_separated_again(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    """A stem whose length does not match the audio belongs to a different cut."""

    _, root = fake_separator
    source = _write_audio(root / "clip.wav")
    out = root / "out"
    separate_stems(source, out, settings=_settings(root))
    FakeSeparator.instances.clear()

    # Same name, half the audio: the stored stems no longer describe it.
    sf.write(
        source,
        np.zeros((REQUIRED_SAMPLE_RATE // 2, 2), dtype=np.float32),
        REQUIRED_SAMPLE_RATE,
        subtype="FLOAT",
    )
    separate_stems(source, out, settings=_settings(root))

    assert FakeSeparator.instances != []


def test_cuda_is_mandatory_and_device_fallback_is_rejected(
    fake_separator: tuple[FakeCuda, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = fake_separator
    FakeSeparator.model_device = "cpu"

    with pytest.raises(SeparationInferenceError, match="parameters are on cpu, not CUDA"):
        separate_stems(
            _write_audio(root / "clip.wav"),
            root / "out",
            settings=_settings(root),
        )


def test_cpu_device_setting_is_rejected(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    _, root = fake_separator
    settings = dataclasses.replace(_settings(root), device="cpu")

    with pytest.raises(SeparationInferenceError, match="requires CUDA"):
        separate_stems(
            _write_audio(root / "clip.wav"),
            root / "out",
            settings=settings,
        )


def test_unresolvable_returned_paths_report_where_it_looked(
    fake_separator: tuple[FakeCuda, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The wrapper writes into output_dir but returns bare filenames."""

    _, root = fake_separator

    def separate_without_writing(
        self: FakeSeparator, audio_path: str, *, custom_output_names: dict[str, str]
    ) -> list[str]:
        return [f"{name}.wav" for name in custom_output_names.values()]

    monkeypatch.setattr(FakeSeparator, "separate", separate_without_writing)

    with pytest.raises(InvalidAudioError, match="returned or expected paths"):
        separate_stems(
            _write_audio(root / "clip.wav"),
            root / "out",
            settings=_settings(root),
        )


def test_a_gpu_index_the_pod_does_not_have_is_rejected(
    fake_separator: tuple[FakeCuda, Path],
) -> None:
    _, root = fake_separator
    settings = dataclasses.replace(_settings(root), device="cuda:3")

    with pytest.raises(SeparationInferenceError, match="does not exist"):
        separate_stems(
            _write_audio(root / "clip.wav"),
            root / "out",
            settings=settings,
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


def test_nested_import_error_is_reported_instead_of_claiming_package_absent(
    fake_separator: tuple[FakeCuda, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, root = fake_separator
    original_import = builtins.__import__

    def import_without_audioread(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "audio_separator.separator":
            raise ModuleNotFoundError("No module named 'audioread'", name="audioread")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_audioread)

    with pytest.raises(
        SeparationInferenceError,
        match=r"audio-separator or one of its dependencies.*audioread",
    ):
        separate_stems(
            _write_audio(root / "clip.wav"),
            root / "out",
            settings=_settings(root),
        )


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
