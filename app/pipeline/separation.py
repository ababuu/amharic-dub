"""GPU-accelerated two-stem separation using audio-separator.

The separator writes isolated vocals and the combined instrumental bed. The latter
feeds the existing music input; a silent effects stem preserves the downstream
three-path contract without mixing the instrumental twice.
"""

from __future__ import annotations

import gc
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
REQUIRED_SAMPLE_RATE = 48_000
STEM_NAMES: tuple[str, ...] = ("speech", "music", "effects")


class SeparationError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingInputError(SeparationError):
    """The requested input audio file does not exist."""


class UnsupportedSampleRateError(SeparationError):
    """The input is not the 48 kHz audio required by this pipeline."""


class InvalidAudioError(SeparationError):
    """The input, or a produced stem, could not be used as audio."""


class SeparationInferenceError(SeparationError):
    """The separator failed to load or run on the requested CUDA device."""


class MissingStemError(SeparationError):
    """The separator did not produce one of the expected stems."""


@dataclass(frozen=True)
class StemPaths:
    """Filesystem paths of the three stems produced by separate_stems."""

    speech: Path
    music: Path
    effects: Path

    def as_dict(self) -> dict[str, Path]:
        """Return the stems as a mapping keyed by stem name."""

        return {"speech": self.speech, "music": self.music, "effects": self.effects}


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file into a float32 (channels, samples) array."""

    audio_path = Path(path)
    if not audio_path.is_file():
        raise MissingInputError(f"input audio file not found: {audio_path}")

    try:
        data, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read audio file {audio_path}: {exc}") from exc

    audio = np.ascontiguousarray(np.asarray(data, dtype=np.float32).T)
    if audio.ndim != 2 or audio.shape[0] == 0 or audio.shape[1] == 0:
        raise InvalidAudioError(f"audio file {audio_path} contains no usable samples")
    return audio, int(sample_rate)


def _validate_sample_rate(sample_rate: int) -> None:
    if sample_rate != REQUIRED_SAMPLE_RATE:
        raise UnsupportedSampleRateError(
            f"separation requires {REQUIRED_SAMPLE_RATE} Hz audio, got {sample_rate} Hz; "
            "resample the input before calling separate_stems()"
        )


def _require_cuda() -> Any:
    """Import the runtime lazily and fail before model loading when CUDA is absent."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - required on a deployed worker
        raise SeparationInferenceError(
            "PyTorch is required for CUDA audio separation"
        ) from exc

    if not torch.cuda.is_available():
        raise SeparationInferenceError(
            "audio-separator is configured for CUDA, but torch.cuda.is_available() is false"
        )
    return torch


def _configure_cuda_device(torch: Any, separator: Any, requested_device: str) -> Any:
    """Force audio-separator and its BS-Roformer model onto the configured CUDA GPU."""

    if requested_device != "cuda" and not requested_device.startswith("cuda:"):
        raise SeparationInferenceError(
            f"separation requires CUDA, but DEVICE is {requested_device!r}; "
            "set DEVICE=cuda or DEVICE=cuda:N"
        )

    device = torch.device("cuda:0" if requested_device == "cuda" else requested_device)
    if device.index is not None and device.index >= torch.cuda.device_count():
        raise SeparationInferenceError(
            f"configured CUDA device {device} does not exist "
            f"({torch.cuda.device_count()} CUDA device(s) visible)"
        )

    torch.cuda.set_device(device)
    # audio-separator detects CUDA automatically, but explicitly overwrite its
    # selected device before load_model passes it to the architecture implementation.
    separator.torch_device = device
    separator.torch_device_cpu = torch.device("cpu")
    return device


def _verify_model_cuda(separator: Any) -> str:
    """Verify the loaded BS-Roformer model—not only the wrapper—is set to CUDA."""

    model_instance = getattr(separator, "model_instance", None)
    if model_instance is None:
        raise SeparationInferenceError(
            "audio-separator reported success but loaded no model instance"
        )

    # ``model_run`` is the architecture's own PyTorch module, so its parameter
    # device is the authoritative answer to "will inference use the GPU?".
    parameter_device = None
    model = getattr(model_instance, "model_run", None)
    if model is not None:
        try:
            parameter_device = next(model.parameters()).device
        except (AttributeError, StopIteration):
            parameter_device = None

    declared_device = getattr(model_instance, "torch_device", None)

    if parameter_device is not None and getattr(parameter_device, "type", None) != "cuda":
        raise SeparationInferenceError(
            f"loaded BS-Roformer parameters are on {parameter_device}, not CUDA"
        )
    if declared_device is not None and getattr(declared_device, "type", None) != "cuda":
        raise SeparationInferenceError(
            f"loaded audio-separator model selected {declared_device}, not CUDA"
        )
    if parameter_device is None and declared_device is None:
        raise SeparationInferenceError(
            "could not confirm the loaded BS-Roformer model's CUDA placement"
        )
    if parameter_device is None:
        logger.warning(
            "Verified CUDA from audio-separator's declared device (%s); this "
            "version does not expose the model parameters for a stricter check",
            declared_device,
        )
    return str(parameter_device or declared_device)


def _resolve_separator_output(
    item: str | Path,
    destination: Path,
    expected_name: str,
) -> Path:
    """Resolve a returned file path, accounting for relative paths from the wrapper."""

    returned = Path(item).expanduser()
    candidates = (
        [returned] if returned.is_absolute() else [returned, destination / returned]
    )
    candidates.append(destination / expected_name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise InvalidAudioError(
        f"audio-separator returned {str(item)!r}, but no output file exists at "
        f"the returned or expected paths: {searched}"
    )


def _write_silence(path: Path, info: Any) -> None:
    """Create the effects stem without allocating a full-duration audio array."""

    try:
        with sf.SoundFile(
            str(path),
            mode="w",
            samplerate=info.samplerate,
            channels=info.channels,
            format="WAV",
            subtype="PCM_16",
        ) as output:
            block = np.zeros(
                (min(info.samplerate * 10, info.frames), info.channels),
                dtype=np.float32,
            )
            frames_written = 0
            while frames_written < info.frames:
                frames = min(len(block), info.frames - frames_written)
                output.write(block[:frames])
                frames_written += frames
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not write silent effects stem {path}: {exc}") from exc


def _run_audio_separator(
    source: Path,
    destination: Path,
    settings: Settings,
    *,
    on_progress: Callable[[str], None] | None,
) -> tuple[Path, Path]:
    """Run one native-chunked BS-Roformer inference and return vocals/background."""

    torch = _require_cuda()
    try:
        from audio_separator.separator import Separator
    except ImportError as exc:  # pragma: no cover - dependency install failure
        raise SeparationInferenceError(
            "could not import audio-separator or one of its dependencies "
            f"({exc}); install or update the project requirements"
        ) from exc

    cache_dir = Path(settings.model_cache_dir) / "audio-separator"
    cache_dir.mkdir(parents=True, exist_ok=True)
    separator: Any | None = None
    try:
        if on_progress is not None:
            on_progress("  separation CUDA available; loading BS-Roformer model")
        separator = Separator(
            model_file_dir=str(cache_dir),
            output_dir=str(destination),
            output_format="WAV",
            sample_rate=REQUIRED_SAMPLE_RATE,
            use_soundfile=True,
            # Keep file-level chunking disabled: the BS-Roformer MDXC implementation
            # already uses overlapping model segments, while the wrapper's optional
            # file chunker rejoins chunks with simple concatenation.
            chunk_duration=None,
        )
        device = _configure_cuda_device(torch, separator, settings.device)
        device_name = torch.cuda.get_device_name(device)
        logger.info(
            "Separation runtime: torch CUDA device %s (%s), CUDA runtime %s",
            device,
            device_name,
            torch.version.cuda,
        )

        if on_progress is not None:
            on_progress(
                f"  separation loading model {MODEL_FILENAME} on "
                f"{device} ({device_name})"
            )
        separator.load_model(model_filename=MODEL_FILENAME)
        loaded_device = _verify_model_cuda(separator)
        if on_progress is not None:
            on_progress(
                f"  separation inference started on {loaded_device} "
                "(BS-Roformer model-managed overlap-add)"
            )

        generated = separator.separate(
            str(source),
            custom_output_names={
                "Vocals": f"{source.stem}_speech",
                "Instrumental": f"{source.stem}_music",
            },
        )
        if not isinstance(generated, Sequence) or isinstance(generated, (str, bytes)):
            raise MissingStemError(
                "audio-separator did not return the expected sequence of output paths"
            )

        speech: Path | None = None
        music: Path | None = None
        for item in generated:
            returned = Path(item)
            label = returned.stem.casefold()
            if "speech" in label or "vocal" in label:
                output = _resolve_separator_output(
                    item, destination, f"{source.stem}_speech.wav"
                )
                speech = output
            elif "music" in label or "instrumental" in label:
                output = _resolve_separator_output(
                    item, destination, f"{source.stem}_music.wav"
                )
                music = output

        if speech is None or music is None:
            returned_paths = ", ".join(str(path) for path in generated) or "<none>"
            missing = [
                name
                for name, output in (
                    ("speech/vocals", speech),
                    ("music/instrumental", music),
                )
                if output is None
            ]
            raise MissingStemError(
                "audio-separator output is missing expected stem(s): "
                + ", ".join(missing)
                + f"; returned paths: {returned_paths}"
            )
        for name, output in (("speech/vocals", speech), ("music/instrumental", music)):
            if output.stat().st_size == 0:
                raise InvalidAudioError(
                    f"audio-separator produced a zero-byte {name} file: {output}"
                )
            try:
                output_info = sf.info(str(output))
            except (OSError, RuntimeError) as exc:
                raise InvalidAudioError(
                    f"could not read audio-separator {name} output {output}: {exc}"
                ) from exc
            if output_info.frames <= 0 or output_info.channels <= 0:
                raise InvalidAudioError(
                    f"audio-separator produced an empty {name} output"
                )
            if output_info.samplerate != REQUIRED_SAMPLE_RATE:
                raise InvalidAudioError(
                    f"audio-separator {name} output is {output_info.samplerate} Hz, "
                    f"expected {REQUIRED_SAMPLE_RATE} Hz"
                )
        return speech, music
    except SeparationError:
        raise
    except Exception as exc:
        raise SeparationInferenceError(f"audio-separator inference failed: {exc}") from exc
    finally:
        separator = None
        gc.collect()
        try:
            torch.cuda.empty_cache()
        except RuntimeError as exc:  # cleanup must not conceal the inference error
            logger.warning("Could not empty the CUDA cache after separation: %s", exc)


def _reusable_stems(
    source: Path,
    destination: Path,
    info: Any,
    *,
    on_progress: Callable[[str], None] | None,
) -> StemPaths | None:
    """Return the stems already sitting beside ``source``, or ``None`` to separate.

    Separation is deterministic and costs about 1.3x real time, so re-running it for
    the same audio is pure waste - and a run that dies in a later stage would pay it
    again. Stems are reused only when both exist, were written after the source, and
    describe the same audio; anything else means separating for real.
    """

    speech = destination / f"{source.stem}_speech.wav"
    music = destination / f"{source.stem}_music.wav"
    effects = destination / f"{source.stem}_effects.wav"
    try:
        source_mtime = source.stat().st_mtime
    except OSError:
        return None
    for path in (speech, music):
        try:
            stat = path.stat()
            if stat.st_size == 0 or stat.st_mtime < source_mtime:
                return None
            stem_info = sf.info(str(path))
        except (OSError, RuntimeError):
            return None
        if stem_info.samplerate != info.samplerate:
            return None
        # One second of slack: a writer may pad or trim the tail, but a stem of a
        # different length is a different film.
        if abs(stem_info.frames - info.frames) > info.samplerate:
            return None

    if not effects.is_file() or effects.stat().st_size == 0:
        _write_silence(effects, info)
    if on_progress is not None:
        on_progress(
            f"  separation reused the stems already in {destination} "
            "(delete them to separate again)"
        )
    return StemPaths(speech=speech, music=music, effects=effects)


def separate_stems(
    input_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    settings: Settings | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> StemPaths:
    """Separate 48 kHz audio into vocals and the combined instrumental bed.

    CUDA is required. The model library chunks long inputs internally; this adapter
    avoids loading a full waveform or intermediate stem arrays. The instrumental bed
    is returned as music and effects is silent for compatibility with the mixer.
    """

    resolved = settings if settings is not None else get_settings()
    source = Path(input_path)
    if not source.is_file():
        raise MissingInputError(f"input audio file not found: {source}")

    try:
        info = sf.info(str(source))
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read audio metadata for {source}: {exc}") from exc
    if info.frames <= 0 or info.channels <= 0:
        raise InvalidAudioError(f"audio file {source} contains no usable samples")
    _validate_sample_rate(info.samplerate)

    destination = Path(output_dir) if output_dir is not None else Path(resolved.work_dir)
    destination.mkdir(parents=True, exist_ok=True)
    if on_progress is not None:
        on_progress(
            f"  separation audio ready ({info.frames / info.samplerate:.1f}s, "
            f"{info.channels} channel(s))"
        )

    reused = _reusable_stems(source, destination, info, on_progress=on_progress)
    if reused is not None:
        return reused

    started = time.perf_counter()
    speech, music = _run_audio_separator(
        source, destination, resolved, on_progress=on_progress
    )
    effects = destination / f"{source.stem}_effects.wav"
    _write_silence(effects, info)
    if on_progress is not None:
        on_progress(f"  separation completed in {time.perf_counter() - started:.1f}s")

    return StemPaths(speech=speech, music=music, effects=effects)


#: How far below the original performance the bed has to sit for the English to be
#: inaudible under the dub. Speech stays intelligible about 15 dB under competing
#: sound, so a bed carrying the original that close to the mix is a leak, not a
#: level choice.
AUDIBLE_BLEED_DB = -15.0


@dataclass(frozen=True, slots=True)
class BleedReport:
    """How much of the film's **original dialogue** survived into the bed.

    The music and effects stems are re-mixed untouched, so anything the separator
    failed to take out of them is played under the Amharic. That is the one way the
    source language can still be heard in a finished dub, and it is worth measuring
    rather than assuming: a separator that works on ten minutes of dialogue can still
    misclassify a whispered line at minute 95 of a feature.

    ``leaked`` counts the windows where the bed is both *level with* and *correlated
    with* the isolated speech - two independent conditions, because loud music that
    happens to sit under a line is not a leak.
    """

    windows: int
    leaked: int
    worst_seconds: float
    worst_index: int | None
    worst_relative_db: float
    worst_correlation: float
    #: What was measured, for the summary line: the bed, or the dubbed dialogue.
    subject: str = "the bed"
    #: What it was measured against, for the summary line.
    baseline: str = "the speech"

    @property
    def clean(self) -> bool:
        """``True`` when no window carries the original dialogue audibly."""

        return self.leaked == 0

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the measurement."""

        return {
            "subject": self.subject,
            "baseline": self.baseline,
            "windows": self.windows,
            "leaked": self.leaked,
            "worst_seconds": round(self.worst_seconds, 3),
            "worst_index": self.worst_index,
            "worst_relative_db": round(self.worst_relative_db, 2),
            "worst_correlation": round(self.worst_correlation, 4),
        }

    def summary(self) -> str:
        """Return the one-line summary a run prints."""

        if not self.windows:
            return f"original-dialogue bleed in {self.subject} not measured"
        if self.clean:
            return (
                f"no original dialogue in {self.subject} over {self.windows} line "
                f"window(s); worst {self.worst_relative_db:.1f} dB below "
                f"{self.baseline}"
            )
        return (
            f"WARNING: {self.subject} still carries the original dialogue in "
            f"{self.leaked} of {self.windows} window(s); worst "
            f"{self.worst_relative_db:.1f} dB below {self.baseline} at "
            f"{self.worst_seconds:.2f}s"
        )


def _open_stem(path: Path, *, what: str) -> sf.SoundFile:
    """Open a stem for windowed reading, or raise the module's own error."""

    try:
        return sf.SoundFile(str(path))
    except (OSError, RuntimeError) as exc:  # soundfile errors subclass RuntimeError
        if not Path(path).is_file():
            raise MissingInputError(f"{what} not found: {path}") from exc
        raise InvalidAudioError(f"could not open {what} {path}: {exc}") from exc


def measure_bed_bleed(
    stems: StemPaths,
    windows: Iterable[tuple[float, float]],
    *,
    threshold_db: float = AUDIBLE_BLEED_DB,
) -> BleedReport:
    """Measure how much original dialogue is left in the music and effects stems.

    Parameters
    ----------
    stems:
        The three stems one run produced.
    windows:
        ``(start, end)`` of every moment the film has dialogue. The speech stem is
        what the separator thinks the dialogue is; these windows say when it happened.
    threshold_db:
        How close to the isolated speech the bed may come before the bleed is called
        audible.

    Returns
    -------
    BleedReport

    Notes
    -----
    Each window is read straight out of the files, so memory stays flat: three
    two-hour 48 kHz stems are about 8 GB of float32 in total, and an A40 doing the
    rest of the pipeline does not have that to spare. Nothing is ever loaded whole.

    Both stems are summed to one channel: the question is whether the *performance*
    is present, and a mono sum answers it without a channel-wise alignment
    assumption.
    """

    windows = list(windows)
    speech_file = _open_stem(Path(stems.speech), what="speech stem")
    with speech_file, _open_stem(Path(stems.music), what="music stem") as music_file, (
        _open_stem(Path(stems.effects), what="effects stem")
    ) as effects_file:
        rate = int(speech_file.samplerate)
        for other, name in ((music_file, "music"), (effects_file, "effects")):
            if int(other.samplerate) != rate:
                raise UnsupportedSampleRateError(
                    f"the {name} stem is {other.samplerate} Hz but the speech stem is "
                    f"{rate} Hz; the stems of one run must share a rate"
                )

        frames = min(len(speech_file), len(music_file), len(effects_file))
        floor = max(1, int(round(0.02 * rate)))

        leaked = 0
        counted = 0
        worst_index: int | None = None
        worst_relative = -math.inf
        worst_correlation = 0.0
        worst_seconds = 0.0

        for index, (start, end) in enumerate(windows):
            first = max(0, int(round(start * rate)))
            last = min(frames, int(round(end * rate)))
            if last - first < floor:
                continue
            counted += 1

            try:
                reference = _read_window(speech_file, first, last - first)
                candidate = _read_window(music_file, first, last - first) + _read_window(
                    effects_file, first, last - first
                )
            except (OSError, RuntimeError) as exc:
                raise InvalidAudioError(
                    f"could not read the stems at {start:.3f}s-{end:.3f}s: {exc}"
                ) from exc

            reference_rms = float(np.sqrt(np.mean(reference**2)))
            bed_rms = float(np.sqrt(np.mean(candidate**2)))
            if reference_rms <= 0.0 or bed_rms <= 0.0:
                continue
            relative_db = 20.0 * math.log10(bed_rms / reference_rms)

            if reference.std() > 0.0 and candidate.std() > 0.0:
                correlation = float(np.corrcoef(candidate, reference)[0, 1])
            else:
                correlation = 0.0

            if relative_db >= threshold_db and correlation >= 0.5:
                leaked += 1
            if relative_db > worst_relative:
                worst_relative = relative_db
                worst_correlation = correlation
                worst_seconds = start
                worst_index = index

    if not math.isfinite(worst_relative):
        worst_relative = -math.inf

    return BleedReport(
        windows=counted,
        leaked=leaked,
        worst_seconds=worst_seconds,
        worst_index=worst_index,
        worst_relative_db=worst_relative,
        worst_correlation=worst_correlation,
        subject="the bed",
    )


#: How close a window of the dubbed dialogue may come to the film's original audio
#: before the original is judged audible in the dub. A dub is *not* expected to
#: resemble the original: it says the same thing in another language, in a cloned
#: voice, so anything above this is the original surviving rather than a coincidence.
DUB_BLEED_DB = -18.0

#: How far either side of a line's own position to look for the original, in seconds.
#: A mixing or placement fault puts the original *at* its own timestamp; a resampling
#: or offset fault can shift it, and this catches both without a full search.
DUB_BLEED_LAG_SECONDS = 0.25

#: How finely that search is stepped, in seconds. A verbatim copy correlates strongly
#: across a window of lags many milliseconds wide, so stepping at sample resolution
#: tests the same thing thousands of times over - and at 48 kHz the full resolution
#: search is 24,000 offsets per line, which costs minutes for no extra detection.
DUB_BLEED_LAG_STEP_SECONDS = 0.005

#: Waveform correlation above which two windows are judged to be the same audio.
DUB_BLEED_CORRELATION = 0.5


def _open_audio(path: Path, *, what: str) -> sf.SoundFile:
    """Open an audio file for windowed reading, or raise the module's own error."""

    try:
        return sf.SoundFile(str(path))
    except (OSError, RuntimeError) as exc:  # soundfile errors subclass RuntimeError
        if not Path(path).is_file():
            raise MissingInputError(f"{what} not found: {path}") from exc
        raise InvalidAudioError(f"could not open {what} {path}: {exc}") from exc


def measure_dub_bleed(
    original: str | Path,
    dialogue: str | Path,
    windows: Iterable[tuple[float, float]],
    *,
    threshold_db: float = DUB_BLEED_DB,
    correlation_threshold: float = DUB_BLEED_CORRELATION,
) -> BleedReport:
    """Measure how much of the **original audio** survives in the dubbed dialogue.

    :func:`measure_bed_bleed` answers "did separation leave the original in the music
    and effects". It cannot answer "is the original audible in the finished dub",
    because the dubbed dialogue track is not part of what it inspects - and the
    finished dub is what a listener hears. This measures the other half.

    ``dialogue`` is the mix's own dialogue stem, before the bed is added, so anything
    the original shares with it came from the original rather than from the music.
    The two are compared both in level and in waveform, over a small lag search,
    because a copy is only a copy if it is *the same audio*: an Amharic line spoken by
    a cloned voice of the same character resembles the original not at all.

    Parameters
    ----------
    original:
        The film's own soundtrack - the extracted track, not a stem.
    dialogue:
        The dubbed dialogue stem written by :mod:`app.pipeline.mixing`.
    windows:
        ``(start, end)`` of every line the dub contains - the moments a line should be
        the *only* thing in the dialogue stem.
    threshold_db, correlation_threshold:
        How close, and how alike, the two have to be before the original is judged
        audible. Both are required, so loud unrelated material is not a leak.

    Returns
    -------
    BleedReport
        With ``subject`` set to ``"the dubbed dialogue"``.
    """

    windows = list(windows)
    counted = 0
    leaked = 0
    leak: tuple[float, float, float, int] | None = None
    loudest: tuple[float, float, float, int] | None = None

    with _open_audio(Path(original), what="the original soundtrack") as source, (
        _open_audio(Path(dialogue), what="the dubbed dialogue")
    ) as dub:
        rate = int(source.samplerate)
        if int(dub.samplerate) != rate:
            raise UnsupportedSampleRateError(
                f"the dubbed dialogue is {dub.samplerate} Hz but the original is "
                f"{rate} Hz; the two cannot be compared"
            )
        frames = min(len(source), len(dub))
        floor = max(1, int(round(0.02 * rate)))
        # The lag search is in samples of *this* file, so a rate other than the
        # pipeline's would otherwise search a different distance than it says.
        span = int(round(DUB_BLEED_LAG_SECONDS * rate))
        step = max(1, int(round(DUB_BLEED_LAG_STEP_SECONDS * rate)))
        offsets = list(range(-span, span + 1, step))

        for index, (start, end) in enumerate(windows):
            first = max(0, int(round(start * rate)))
            last = min(frames, int(round(end * rate)))
            if last - first < floor:
                continue
            counted += 1

            try:
                produced = _read_window(dub, first, last - first)
                # Read the original once, padded by the lag search, rather than
                # re-seeking for every offset: a window is compared against a whole
                # neighbourhood of the original, and a seek per offset is thousands of
                # reads for one line.
                padded_first = max(0, first - span)
                padded_last = min(frames, last + span)
                neighbourhood = _read_window(
                    source, padded_first, padded_last - padded_first
                )
            except (OSError, RuntimeError) as exc:
                raise InvalidAudioError(
                    f"could not read the audio at {start:.3f}s-{end:.3f}s: {exc}"
                ) from exc

            if produced.size == 0 or produced.std() <= 0.0:
                continue

            best_correlation = 0.0
            best_ratio = 0.0
            produced_rms = float(np.sqrt(np.mean(produced**2)))
            for offset in offsets:
                begin = (first + offset) - padded_first
                if begin < 0 or begin + produced.size > neighbourhood.size:
                    continue
                candidate = neighbourhood[begin : begin + produced.size]
                if candidate.std() <= 0.0:
                    continue
                correlation = float(np.corrcoef(candidate, produced)[0, 1])
                if abs(correlation) > abs(best_correlation):
                    best_correlation = correlation
                    candidate_rms = float(np.sqrt(np.mean(candidate**2)))
                    best_ratio = (
                        20.0 * math.log10(produced_rms / candidate_rms)
                        if candidate_rms > 0.0
                        else -math.inf
                    )

            flagged = (
                best_ratio >= threshold_db
                and abs(best_correlation) >= correlation_threshold
            )
            if flagged:
                leaked += 1
                if leak is None or best_ratio > leak[0]:
                    leak = (best_ratio, best_correlation, start, index)
            if loudest is None or best_ratio > loudest[0]:
                loudest = (best_ratio, best_correlation, start, index)

    # Report the worst *leak*, not the loudest window: a window can be loud and
    # unrelated, and naming that one would send a reader to inspect audio that is
    # fine. When nothing was flagged, the loudest window is the most useful bound.
    chosen = leak if leak is not None else loudest
    worst_relative = chosen[0] if chosen is not None else -math.inf
    worst_correlation = chosen[1] if chosen is not None else 0.0
    worst_seconds = chosen[2] if chosen is not None else 0.0
    worst_index = chosen[3] if chosen is not None else None

    return BleedReport(
        windows=counted,
        leaked=leaked,
        worst_seconds=worst_seconds,
        worst_index=worst_index,
        worst_relative_db=worst_relative,
        worst_correlation=worst_correlation,
        subject="the dubbed dialogue",
        baseline="the original audio",
    )


def _read_window(handle: sf.SoundFile, first: int, frames: int) -> np.ndarray:
    """Read ``frames`` samples starting at ``first`` and return them as one channel.

    ``frames`` is always within every stem - the caller clamps to the shortest - so a
    short read is not a case that arises.
    """

    handle.seek(first)
    block = handle.read(frames, dtype="float32", always_2d=True)
    return np.asarray(block, dtype=np.float32).mean(axis=1)


__all__ = [
    "AUDIBLE_BLEED_DB",
    "DUB_BLEED_CORRELATION",
    "DUB_BLEED_DB",
    "DUB_BLEED_LAG_SECONDS",
    "DUB_BLEED_LAG_STEP_SECONDS",
    "MODEL_NAME",
    "REQUIRED_SAMPLE_RATE",
    "STEM_NAMES",
    "WEIGHTS_DIR_ENV_VAR",
    "BleedReport",
    "SeparationError",
    "MissingInputError",
    "UnsupportedSampleRateError",
    "InvalidAudioError",
    "SeparationInferenceError",
    "MissingStemError",
    "StemPaths",
    "measure_bed_bleed",
    "measure_dub_bleed",
    "read_audio",
    "resolve_weights_dir",
    "separate_stems",
]
