"""Tests for the two things that made separation the slowest stage in a real run.

A measured run spent **354 s of its 590 s** in separation on a 135 s clip. Two causes,
both in this project's own code rather than in Bandit:

* the package's chunk handler was left at its default of one 8-second chunk per forward
  pass, so a 48 GB card was nearly idle for the whole stage;
* the handler builds a tensor proportional to the *entire* input and holds it, which a
  feature-length film cannot fit.

Nothing here loads a model or separates real audio: the session is a fake, and the
budget is patched down to a few seconds so a split happens at a test-sized input.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from app.config import Settings
from app.pipeline import separation
from app.pipeline.separation import (
    BANDIT_STITCH_SECONDS,
    MAX_CHUNK_TENSOR_BYTES,
    MissingStemError,
    REQUIRED_SAMPLE_RATE,
    SeparationInferenceError,
    StemPaths,
    separate_stems,
)


class FakeHandler:
    """Stand-in for BandIt's chunk handler, recording the batch size it was given."""

    def __init__(self) -> None:
        self.inference_batch_size = 1
        self.chunk_size_samples = int(separation.BANDIT_CHUNK_SECONDS * REQUIRED_SAMPLE_RATE)
        self.hop_size_samples = int(
            separation.BANDIT_UPSTREAM_HOP_SECONDS * REQUIRED_SAMPLE_RATE
        )
        self.overlap_samples = self.chunk_size_samples - self.hop_size_samples
        self.scaler = self.chunk_size_samples / (2 * self.hop_size_samples)
        self.front_pad_samples = 2 * self.overlap_samples
        self.scaled_window = 1.0 / self.scaler


class FakeRuntime:
    def __init__(self) -> None:
        self.handler = FakeHandler()


class FakeSession:
    """A session whose "model" scales the input per stem, so stitching is checkable.

    Being linear and memoryless, it must come back out of the segment cross-fade
    unchanged: anything the blend loses or doubles shows up as a difference from the
    input rather than being hidden by a model's nonlinearity.
    """

    instances: list["FakeSession"] = []
    scale = {"speech": 1.0, "music": 0.5, "effects": 0.25}

    def __init__(self, model=None, *, device=None, weights_dir=None) -> None:
        self.model = model
        self.device = device
        self.weights_dir = weights_dir
        self._runtime = FakeRuntime()
        self.loaded = False
        self.closed = False
        self.calls: list[tuple[int, int]] = []
        FakeSession.instances.append(self)

    def load(self):
        self.loaded = True
        return self

    def infer(self, audio, *, sample_rate):
        self.calls.append(audio.shape)
        return {name: audio * factor for name, factor in self.scale.items()}

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _isolate(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeSession.instances = []
    monkeypatch.setattr(separation, "BanditSession", FakeSession)


def _settings(root: Path, **overrides) -> Settings:
    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "models",
        "device": "cpu",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write_video_audio(path: Path, seconds: float, channels: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(round(seconds * REQUIRED_SAMPLE_RATE))
    rng = np.random.default_rng(0)
    data = rng.standard_normal((frames, channels)).astype(np.float32) * 0.1
    import soundfile as sf

    sf.write(str(path), data, REQUIRED_SAMPLE_RATE, subtype="FLOAT")
    return path


# ---------------------------------------------------------------------------
# Batching: the fix for the 354-second stage
# ---------------------------------------------------------------------------


def test_several_chunks_are_pushed_per_forward_pass(tmp_path: Path) -> None:
    """The handler's default of one chunk per forward is what left the GPU idle."""

    source = _write_video_audio(tmp_path / "clip.wav", 1.0)

    separate_stems(
        source, tmp_path / "out", settings=_settings(tmp_path, separation_inference_batch_size=8)
    )

    (session,) = FakeSession.instances
    assert session._runtime.handler.inference_batch_size == 8
    assert session._runtime.handler.hop_size_samples == int(
        separation.BANDIT_HOP_SECONDS * REQUIRED_SAMPLE_RATE
    )
    assert session._runtime.handler.overlap_samples == int(
        (separation.BANDIT_CHUNK_SECONDS - separation.BANDIT_HOP_SECONDS)
        * REQUIRED_SAMPLE_RATE
    )


def test_default_hop_reduces_chunk_count_and_rescales_the_overlap_window() -> None:
    handler = FakeHandler()
    session = type("Session", (), {"_runtime": FakeRuntime()})()
    session._runtime.handler = handler
    original_window = handler.scaled_window

    actual_hop = separation._configure_chunking(
        session, 8, separation.BANDIT_HOP_SECONDS
    )

    assert actual_hop == separation.BANDIT_HOP_SECONDS
    assert handler.inference_batch_size == 8
    assert handler.scaled_window == pytest.approx(original_window * 6.0)
    assert handler.scaler == pytest.approx(2.0 / 3.0)
    assert handler.front_pad_samples == 2 * handler.overlap_samples


def test_separation_reports_io_loading_and_inference_progress(tmp_path: Path) -> None:
    source = _write_video_audio(tmp_path / "clip.wav", 1.0)
    updates: list[str] = []

    separate_stems(
        source,
        tmp_path / "out",
        settings=_settings(tmp_path),
        on_progress=updates.append,
    )

    assert any("separation audio read in" in update for update in updates)
    assert "  separation model loading started" in updates
    assert any("separation model loading completed in" in update for update in updates)
    assert any("inference segment 1/1 started" in update for update in updates)
    assert any("inference segment 1/1 completed in" in update for update in updates)
    assert any("separation inference completed in" in update for update in updates)
    assert any("separation stem writes completed in" in update for update in updates)


def test_a_batch_size_below_one_is_clamped(tmp_path: Path) -> None:
    """A zero or negative batch would make the handler loop forever."""

    source = _write_video_audio(tmp_path / "clip.wav", 1.0)

    separate_stems(
        source, tmp_path / "out", settings=_settings(tmp_path, separation_inference_batch_size=0)
    )

    (session,) = FakeSession.instances
    assert session._runtime.handler.inference_batch_size == 1


def test_default_hop_uses_one_sixth_as_many_chunks_as_upstream() -> None:
    upstream = separation._max_segment_seconds(
        2, REQUIRED_SAMPLE_RATE, separation.BANDIT_UPSTREAM_HOP_SECONDS
    )
    configured = separation._max_segment_seconds(
        2, REQUIRED_SAMPLE_RATE, separation.BANDIT_HOP_SECONDS
    )

    assert separation.BANDIT_HOP_SECONDS == 6.0
    assert configured == pytest.approx(upstream * 6.0)


def test_invalid_hop_is_rejected_before_reading_audio(tmp_path: Path) -> None:
    with pytest.raises(SeparationInferenceError, match="SEPARATION_HOP_SECONDS"):
        separate_stems(
            tmp_path / "missing.wav",
            tmp_path / "out",
            settings=_settings(tmp_path, separation_hop_seconds=9.0),
        )


def test_a_session_without_that_handler_still_runs(tmp_path: Path) -> None:
    """Reaching into the package must never be able to fail a run."""

    class Bare:
        loaded = False

        def load(self):
            self.loaded = True
            return self

        def infer(self, audio, *, sample_rate):
            return {name: audio for name in separation.STEM_NAMES}

        def close(self) -> None:
            pass

    class BareSession(Bare):
        def __init__(self, *_, **__) -> None:
            super().__init__()

    separation.BanditSession = BareSession  # type: ignore[assignment]
    source = _write_video_audio(tmp_path / "clip.wav", 1.0)

    paths = separate_stems(source, tmp_path / "out", settings=_settings(tmp_path))

    assert isinstance(paths, StemPaths)


# ---------------------------------------------------------------------------
# The chunk-tensor budget
# ---------------------------------------------------------------------------


def test_the_budget_shrinks_as_channels_grow() -> None:
    """The tensor the handler holds is proportional to channels * chunks * samples."""

    mono = separation._max_segment_seconds(
        1, REQUIRED_SAMPLE_RATE, separation.BANDIT_HOP_SECONDS
    )
    stereo = separation._max_segment_seconds(
        2, REQUIRED_SAMPLE_RATE, separation.BANDIT_HOP_SECONDS
    )

    assert stereo == pytest.approx(mono / 2)


def test_the_budget_matches_the_tensor_it_bounds() -> None:
    """The number is only meaningful if it really keeps that tensor inside the ceiling."""

    channels = 2
    seconds = separation._max_segment_seconds(
        channels, REQUIRED_SAMPLE_RATE, separation.BANDIT_HOP_SECONDS
    )

    chunks = seconds / separation.BANDIT_HOP_SECONDS
    tensor_bytes = (
        channels * chunks * separation.BANDIT_CHUNK_SECONDS * REQUIRED_SAMPLE_RATE * 4
    )

    assert tensor_bytes == pytest.approx(MAX_CHUNK_TENSOR_BYTES, rel=0.01)


def test_a_clip_that_already_fits_is_still_one_call(tmp_path: Path) -> None:
    """The fix must not change what a short input does, or the audio of a clip would change."""

    source = _write_video_audio(tmp_path / "clip.wav", 2.0)
    settings = _settings(tmp_path)

    separate_stems(source, tmp_path / "out", settings=settings)

    (session,) = FakeSession.instances
    assert len(session.calls) == 1


# ---------------------------------------------------------------------------
# Segmenting a long film, and stitching it back together
# ---------------------------------------------------------------------------


def test_a_long_input_is_split_into_several_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A feature-length film cannot be handed to the handler in one piece."""

    monkeypatch.setattr(separation, "_max_segment_seconds", lambda *_: 2.0)
    source = _write_video_audio(tmp_path / "clip.wav", 3.0)

    separate_stems(source, tmp_path / "out", settings=_settings(tmp_path))

    (session,) = FakeSession.instances
    assert len(session.calls) > 1
    # Every call but the last covers one full segment.
    assert session.calls[0][1] == int(2.0 * REQUIRED_SAMPLE_RATE)


def test_the_segments_overlap_by_more_than_the_handler_pads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each call's own edges are only correct because of that internal padding."""

    monkeypatch.setattr(separation, "_max_segment_seconds", lambda *_: 2.0)
    source = _write_video_audio(tmp_path / "clip.wav", 5.0)

    separate_stems(source, tmp_path / "out", settings=_settings(tmp_path))

    (session,) = FakeSession.instances
    lengths = [length for _, length in session.calls]
    segment = int(2.0 * REQUIRED_SAMPLE_RATE)
    # The overlap the blend uses has to cover the padding the handler adds internally,
    # or a join would fall inside the region whose output is *not* yet correct.
    front_pad_seconds = 2 * (separation.BANDIT_CHUNK_SECONDS - separation.BANDIT_HOP_SECONDS)
    assert BANDIT_STITCH_SECONDS > front_pad_seconds
    assert lengths[:-1] == [segment] * (len(lengths) - 1)


def test_the_cross_fade_reconstructs_the_signal_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A blend that dipped or doubled at a join would be audible as a click or a hole."""

    monkeypatch.setattr(separation, "_max_segment_seconds", lambda *_: 2.0)
    source = _write_video_audio(tmp_path / "clip.wav", 5.0)
    original, _ = separation.read_audio(source)

    paths = separate_stems(source, tmp_path / "out", settings=_settings(tmp_path))

    speech, rate = separation.read_audio(paths.speech)
    assert rate == REQUIRED_SAMPLE_RATE
    # The fake model is a pure gain, so the stitched speech stem must equal the input.
    # PCM-16 output quantises, hence the tolerance.
    assert speech.shape == original.shape
    assert np.max(np.abs(speech - original)) < 1e-3
    # And no sample may have been lost to a gap in coverage.
    assert np.all(np.isfinite(speech))


def test_each_segment_is_scaled_into_its_own_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stitching is per stem, so a mix-up would cross music into the speech stem."""

    monkeypatch.setattr(separation, "_max_segment_seconds", lambda *_: 2.0)
    source = _write_video_audio(tmp_path / "clip.wav", 4.0)
    original, _ = separation.read_audio(source)

    paths = separate_stems(source, tmp_path / "out", settings=_settings(tmp_path))

    speech, _ = separation.read_audio(paths.speech)
    music, _ = separation.read_audio(paths.music)
    assert np.max(np.abs(music - original * 0.5)) < 1e-3
    assert np.max(np.abs(speech - original)) < 1e-3


# ---------------------------------------------------------------------------
# Failure reporting is unchanged
# ---------------------------------------------------------------------------


def test_a_result_that_is_not_a_mapping_is_still_reported(tmp_path: Path) -> None:
    """The type check moved into the collector and must still name the problem."""

    class OddSession(FakeSession):
        def infer(self, audio, *, sample_rate):
            return [audio, audio, audio]

    separation.BanditSession = OddSession  # type: ignore[assignment]
    source = _write_video_audio(tmp_path / "clip.wav", 1.0)

    with pytest.raises(MissingStemError, match="expected a mapping"):
        separate_stems(source, tmp_path / "out", settings=_settings(tmp_path))


def test_a_missing_stem_is_reported(tmp_path: Path) -> None:
    class PartialSession(FakeSession):
        def infer(self, audio, *, sample_rate):
            return {"speech": audio, "music": audio}

    separation.BanditSession = PartialSession  # type: ignore[assignment]
    source = _write_video_audio(tmp_path / "clip.wav", 1.0)

    with pytest.raises(MissingStemError, match="effects"):
        separate_stems(source, tmp_path / "out", settings=_settings(tmp_path))


# ---------------------------------------------------------------------------
# Aiming the batch at a 48 GB card without risking the run
# ---------------------------------------------------------------------------


class MemoryLimitedSession(FakeSession):
    """A session that runs out of memory above a batch size, as a smaller card would."""

    limit = 4
    attempts: list[int] = []

    def infer(self, audio, *, sample_rate):
        batch = self._runtime.handler.inference_batch_size
        MemoryLimitedSession.attempts.append(batch)
        if batch > self.limit:
            raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        return super().infer(audio, sample_rate=sample_rate)


def test_an_out_of_memory_batch_is_halved_instead_of_ending_the_run(
    tmp_path: Path,
) -> None:
    """The default is sized for a 48 GB card; a smaller one must cost time, not the run."""

    MemoryLimitedSession.attempts = []
    separation.BanditSession = MemoryLimitedSession  # type: ignore[assignment]
    source = _write_video_audio(tmp_path / "clip.wav", 1.0)
    updates: list[str] = []

    with pytest.warns(RuntimeWarning, match="ran out of memory"):
        paths = separate_stems(
            source,
            tmp_path / "out",
            settings=_settings(tmp_path, separation_inference_batch_size=32),
            on_progress=updates.append,
        )

    assert MemoryLimitedSession.attempts == [32, 16, 8, 4]
    assert any("out of memory at batch size 32" in update for update in updates)
    assert any("retrying at 16" in update for update in updates)
    assert isinstance(paths, StemPaths)
    # And the audio is still produced at the size that worked.
    assert paths.speech.is_file()


def test_the_retry_warns_once_with_the_size_it_settled_on(tmp_path: Path) -> None:
    """One actionable line, not one per halving: the recovery worked and that is the news."""

    MemoryLimitedSession.attempts = []
    separation.BanditSession = MemoryLimitedSession  # type: ignore[assignment]
    source = _write_video_audio(tmp_path / "clip.wav", 1.0)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        separate_stems(
            source,
            tmp_path / "out",
            settings=_settings(tmp_path, separation_inference_batch_size=32),
        )

    assert len(caught) == 1
    assert "settled on 4" in str(caught[0].message)
    assert "SEPARATION_INFERENCE_BATCH_SIZE=4" in str(caught[0].message)


def test_an_out_of_memory_error_at_one_chunk_still_fails(tmp_path: Path) -> None:
    """One chunk per forward is where the library started, so there is nowhere left to go."""

    class AlwaysOutOfMemory(FakeSession):
        attempts = 0

        def infer(self, audio, *, sample_rate):
            AlwaysOutOfMemory.attempts += 1
            raise RuntimeError("CUDA out of memory")

    separation.BanditSession = AlwaysOutOfMemory  # type: ignore[assignment]
    source = _write_video_audio(tmp_path / "clip.wav", 1.0)

    # It exhausts the sizes it can try and gives up, rather than looping.
    with pytest.raises(separation.SeparationInferenceError, match="out of memory"):
        separate_stems(
            source,
            tmp_path / "out",
            settings=_settings(tmp_path, separation_inference_batch_size=32),
        )

    # 32, 16, 8, 4, 2, 1 - then the error is reported.
    assert AlwaysOutOfMemory.attempts == 6


def test_a_failure_that_is_not_memory_is_not_retried(tmp_path: Path) -> None:
    """Retrying a device-side assert or a shape error would just fail six times slowly."""

    class Broken(FakeSession):
        attempts = 0

        def infer(self, audio, *, sample_rate):
            Broken.attempts += 1
            raise RuntimeError("CUDA error: device-side assert triggered")

    separation.BanditSession = Broken  # type: ignore[assignment]
    source = _write_video_audio(tmp_path / "clip.wav", 1.0)

    with pytest.raises(separation.SeparationInferenceError, match="device-side assert"):
        separate_stems(source, tmp_path / "out", settings=_settings(tmp_path))

    assert Broken.attempts == 1


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "torch.cuda.OutOfMemoryError: CUDA out of memory",
        "Out of memory: an allocation failed",
    ],
)
def test_the_memory_error_is_recognised_by_its_message(message: str) -> None:
    """Recognising it by text keeps torch out of this module's imports."""

    assert separation._is_out_of_memory(RuntimeError(message))


def test_an_unrelated_error_is_not_mistaken_for_memory() -> None:
    assert not separation._is_out_of_memory(RuntimeError("invalid audio shape"))
    assert not separation._is_out_of_memory(ValueError("sample rate must be 48000"))
