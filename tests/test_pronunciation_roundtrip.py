"""The pronunciation round trip, and the factory that makes it runnable.

`qc.measure_pronunciation` is deliberately model-free: it takes a callable and reports a
character error rate against the text that was synthesized. That is the *only* automated
check that can see content the speech model invented, because a stray word, a fragment of
an English cloning prompt, or a mispronunciation is not the film's original audio - so
comparing the output with the original, which is what the bleed measurements do, can
never find any of it.

The factory that supplies the callable is what these tests cover, plus the wiring that
makes a run report a pronunciation figure instead of "not measured".
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest

from app.config import Settings, get_settings
from app.pipeline import transcription


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root,
        "output_dir": root,
        "model_cache_dir": root,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


class FakeWhisperModel:
    """A stand-in that records how it was called and returns scripted text."""

    instances: list["FakeWhisperModel"] = []
    fail: Exception | None = None
    text: str = "ሰላም"

    def __init__(self, model: str, **kwargs: object) -> None:
        self.model = model
        self.kwargs = kwargs
        self.calls: list[tuple[str, dict[str, object]]] = []
        FakeWhisperModel.instances.append(self)

    def transcribe(self, path: str, **kwargs: object):
        self.calls.append((path, kwargs))
        if FakeWhisperModel.fail is not None:
            raise FakeWhisperModel.fail
        return [types.SimpleNamespace(text=FakeWhisperModel.text)], object()


@pytest.fixture
def fake_whisper(monkeypatch: pytest.MonkeyPatch):
    FakeWhisperModel.instances = []
    FakeWhisperModel.fail = None
    FakeWhisperModel.text = "ሰላም"
    monkeypatch.setattr(transcription, "WhisperModel", FakeWhisperModel)
    return FakeWhisperModel


def test_the_transcriber_asks_for_amharic(tmp_path: Path, fake_whisper) -> None:
    """Detection on one short line is unreliable, and the answer is known."""

    transcribe = transcription.build_amharic_transcriber(_settings(tmp_path))

    heard = transcribe(tmp_path / "clip.wav")

    assert heard == "ሰላም"
    (call,) = fake_whisper.instances[0].calls
    assert call[1]["language"] == "am"
    # Word timestamps and a beam search are for building a transcript; this only has to
    # read one line back, and both cost time on every clip of a film.
    assert call[1]["beam_size"] == 1
    assert call[1]["vad_filter"] is False


def test_the_model_is_loaded_once(tmp_path: Path, fake_whisper) -> None:
    """A film is thousands of clips; loading the model per clip would be absurd."""

    transcribe = transcription.build_amharic_transcriber(_settings(tmp_path))

    for index in range(4):
        transcribe(tmp_path / f"clip{index}.wav")

    assert len(fake_whisper.instances) == 1
    assert len(fake_whisper.instances[0].calls) == 4


def test_the_configured_model_is_the_one_loaded(tmp_path: Path, fake_whisper) -> None:
    transcribe = transcription.build_amharic_transcriber(
        _settings(tmp_path, transcription_model="large-v3")
    )

    transcribe(tmp_path / "clip.wav")

    assert fake_whisper.instances[0].model == "large-v3"


def test_a_line_that_cannot_be_read_back_is_not_fatal(
    tmp_path: Path, fake_whisper
) -> None:
    """One unreadable clip must not lose a finished dub, or abort the measurement."""

    fake_whisper.fail = RuntimeError("decode failed")
    transcribe = transcription.build_amharic_transcriber(_settings(tmp_path))

    assert transcribe(tmp_path / "clip.wav") == ""


def test_the_factory_is_exported(tmp_path: Path) -> None:
    assert "build_amharic_transcriber" in transcription.__all__


def test_pronunciation_is_off_unless_asked_for(monkeypatch: pytest.MonkeyPatch) -> None:
    """It loads a second model and reads every clip, so it is opt-in."""

    monkeypatch.delenv("QC_PRONUNCIATION", raising=False)

    assert Settings.from_env().qc_pronunciation is False


def test_the_pronunciation_setting_is_read_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QC_PRONUNCIATION", "true")

    settings = Settings.from_env()

    assert settings.qc_pronunciation is True
    assert settings.as_dict()["qc_pronunciation"] is True
