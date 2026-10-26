"""Target-speaker extraction for simultaneous speech (crosstalk).

Why this module exists
----------------------
The exclusive diarization this pipeline uses assigns every instant of speech to
exactly one speaker. Inside a crosstalk region that is a lie the mixer tells: two
characters are talking at once, and whichever cluster won the instant has their
words transcribed while the other character's words are simply absent. Re-decoding
the *mixture* does not fix it - a decoder returns the dominant voice, which is the
speaker who was kept. That is why the transcription stage's register gate refuses
those recoveries: the audio it is offered belongs to the wrong character.

The fix is to give each character their own waveform inside the region. That is
what this module does: for a crosstalk region, a separation model is run on the
region (with a little context), one source waveform is associated with each
character, and the transcription stage decodes *that* track. The register gate
stays the final arbiter on anything credited to a character, so a bad separation
can still only cause a refusal - never a wrong-voice line.

Backends
--------
Two model families are supported, both behind the same tiny interface:

``pyannote`` (the default)
    ``pyannote.audio`` 4.0.7 ships a :class:`SpeechSeparation` pipeline for the
    gated ``pyannote/speech-separation-ami-1.0`` checkpoint (the PixIT model).
    Verified against the 4.0.7 wheel: ``Pipeline.from_pretrained(model, token=...,
    cache_dir=...)`` builds it, ``pipeline.to(torch.device(...))`` moves it, and
    ``pipeline({"waveform": (1, samples) tensor, "sample_rate": 16000},
    num_speakers=N)`` returns ``(diarization, SlidingWindowFeature)`` whose
    ``.data`` is ``(samples, sources)``. The checkpoint is gated: the token's
    account must have accepted its conditions, and ``from_pretrained`` returns
    ``None`` when it cannot be fetched.

``speechbrain``
    SpeechBrain's SepFormer (``speechbrain/sepformer-whamr16k`` by default) is a
    blind two-source separator. ``SepformerSeparation.from_hparams(...)`` loads it
    and ``separate_batch((1, samples) tensor)`` returns ``(1, samples, sources)``
    (time before sources - the opposite axis order from pyannote's
    ``(samples, sources)``). It is not gated, but needs the extra ``speechbrain``
    package.

Both are blind: they return sources, not identities. The association with a
character is made here from the register the pipeline already measured for that
character (median F0), and it is deliberately strict: a source is only credited
when it measures within :data:`MAX_REGISTER_RATIO` of the character's register
*and* clearly nearer that register than any other speaker's in the region. An
unclear split is reported as a refusal, because the alternative is a wrong voice.

Deliberate non-goals
--------------------
* **No model downloads at import time.** Heavy imports happen inside
  :func:`build_separator`, and the model is loaded lazily by the transcription
  stage, only when a crosstalk region actually dropped a speaker.
* **No hiding of failures.** A missing package, token, checkpoint or a failed
  inference is returned as an explicit reason and the pipeline falls back to the
  pre-existing refusal path. The run continues.
* **No numpy arrays in the public JSON view.** Tracks carry samples for the
  decoder; :meth:`ExtractionReport.as_dict` reports only their measurements.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from math import gcd
from typing import Any, Protocol

import numpy as np

from app.config import (
    DEFAULT_CROSSTALK_SEPARATION_MODEL,
    SPEECHBRAIN_SEPARATION_MODEL,
    Settings,
    default_crosstalk_separation_model,
)
from app.pipeline.diarization import CrosstalkRegion
from app.pipeline.prosody import ProsodyError, estimate_f0

logger = logging.getLogger(__name__)

#: Rate every extracted track is delivered at. It is the transcription stage's
#: rate, so a track can be decoded without another conversion.
SAMPLE_RATE = 16_000

#: Gated pyannote PixIT pipeline. Its model repository is
#: ``pyannote/separation-ami-1.0``; the pipeline config downloads both.
DEFAULT_PYANNOTE_MODEL = DEFAULT_CROSSTALK_SEPARATION_MODEL

#: SpeechBrain SepFormer trained on noisy/reverberated two-speaker mixtures,
#: which is the closest match to film dialogue of the published checkpoints.
DEFAULT_SPEECHBRAIN_MODEL = SPEECHBRAIN_SEPARATION_MODEL

#: Audio kept on each side of the region when the model is run. A separator needs
#: context to decide which voice is which; 0.14-0.75 s of overlap is too little on
#: its own. The tracks are cropped back to the region afterwards.
DEFAULT_CONTEXT_SECONDS = 1.0

#: A separated source shorter than this is a fragment, not a track.
MIN_TRACK_SECONDS = 0.10

#: Same floor the register and recovery stages use: below it autocorrelation
#: reports the room rather than a voice, so the source cannot be attributed.
MIN_VOICED_FRAMES = 8

#: A source is only credited to a character when its measured F0 is within this
#: ratio of that character's register. 1.35 is the outlier band the register stage
#: already uses for one voice's expressive range.
MAX_REGISTER_RATIO = 1.35

#: The target's register has to explain the source this much better than any
#: other speaker's register. 1.15 is about two semitones of log-distance; closer
#: than that and the source is a mixture or the two voices are too alike to tell.
ASSOCIATION_MARGIN = 1.15

#: A source with less energy than this relative to the region's mixture is not a
#: voice at all - it is the residue a separator leaves when it found nothing.
MIN_SOURCE_RMS_RATIO = 0.02


class TargetExtractionError(RuntimeError):
    """Base class for every error raised by this module."""


class UnavailableError(TargetExtractionError):
    """The stage is disabled, or its model/token/package cannot be used."""


class ExtractionError(TargetExtractionError):
    """The separator failed on one region."""


class Separator(Protocol):
    """A blind source separator invoked on one crosstalk window.

    ``sample_rate`` is the rate of the *returned* waveforms and must be
    :data:`SAMPLE_RATE`; the backend converts from whatever rate its checkpoint
    expects internally, so the rest of the stage has one rate to reason about.
    """

    model: str
    backend: str
    sample_rate: int

    def separate(self, samples: np.ndarray, num_speakers: int) -> list[np.ndarray]:
        """Return one waveform per detected source, all ``len(samples)`` long."""


@dataclass(frozen=True, slots=True)
class ExtractedTrack:
    """One character's waveform for a crosstalk region, on the movie timeline.

    ``samples`` is mono float32 at :attr:`sample_rate` and spans ``start``..``end``
    (the crosstalk region, not the padded model window). ``measured_hz`` is the
    pitch this track measured at association time; it is evidence, not a guess,
    and is reported so a run can be audited.
    """

    speaker_id: str
    start: float
    end: float
    samples: np.ndarray = field(compare=False, repr=False)
    sample_rate: int = SAMPLE_RATE
    measured_hz: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise ExtractionError("a separated track needs a non-empty speaker id")
        if self.end <= self.start:
            raise ExtractionError("a separated track must end after it starts")
        if self.sample_rate <= 0:
            raise ExtractionError("a separated track needs a positive sample rate")

    @property
    def duration(self) -> float:
        """Length of the track in seconds."""

        return self.end - self.start

    def slice(self, start: float, end: float) -> np.ndarray:
        """Return the track's samples between two absolute times.

        The returned array is empty when the requested span falls outside the
        track, which the caller treats as "nothing to decode".
        """

        # Floor/ceil, like ``transcription._region_samples``: a boundary must never
        # cut a sample away from the edge of the window being decoded.
        first = int(math.floor((start - self.start) * self.sample_rate))
        last = int(math.ceil((end - self.start) * self.sample_rate))
        first = min(max(first, 0), len(self.samples))
        last = min(max(last, first), len(self.samples))
        return np.ascontiguousarray(self.samples[first:last], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class Refusal:
    """Why one character could not be given a separated track."""

    speaker_id: str
    reason: str
    measured_hz: float | None = None


@dataclass(frozen=True, slots=True)
class RegionExtraction:
    """What target extraction produced for one crosstalk region."""

    region: CrosstalkRegion
    tracks: tuple[ExtractedTrack, ...] = ()
    refusals: tuple[Refusal, ...] = ()

    @property
    def separated(self) -> bool:
        """``True`` when at least one character got their own waveform."""

        return bool(self.tracks)

    def track_for(self, speaker_id: str) -> ExtractedTrack | None:
        """Return the track separated for ``speaker_id``, if there is one."""

        for track in self.tracks:
            if track.speaker_id == speaker_id:
                return track
        return None

    def describe(self) -> str:
        """Return a short human-readable account of this region."""

        span = f"{self.region.start:.2f}-{self.region.end:.2f}s"
        parts: list[str] = []
        for track in self.tracks:
            measured = (
                "unmeasured"
                if track.measured_hz is None
                else f"{track.measured_hz:.1f} Hz"
            )
            parts.append(f"{track.speaker_id} extracted at {measured}")
        for refusal in self.refusals:
            parts.append(f"{refusal.speaker_id} refused ({refusal.reason})")
        return f"{span}: " + "; ".join(parts) if parts else f"{span}: no track"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view (no samples)."""

        return {
            "start": self.region.start,
            "end": self.region.end,
            "speakers": list(self.region.speaker_ids),
            "tracks": [
                {
                    "speaker_id": track.speaker_id,
                    "start": round(track.start, 4),
                    "end": round(track.end, 4),
                    "measured_hz": (
                        None if track.measured_hz is None else round(track.measured_hz, 2)
                    ),
                }
                for track in self.tracks
            ],
            "refusals": [
                {
                    "speaker_id": refusal.speaker_id,
                    "reason": refusal.reason,
                    "measured_hz": (
                        None
                        if refusal.measured_hz is None
                        else round(refusal.measured_hz, 2)
                    ),
                }
                for refusal in self.refusals
            ],
        }


@dataclass(frozen=True, slots=True)
class ExtractionReport:
    """The outcome of target extraction for every requested region.

    ``available`` is ``False`` when the stage could not run at all (disabled,
    missing token/package/checkpoint). In that case no region is listed and the
    transcription stage keeps its previous, refusing behaviour.
    """

    backend: str
    model: str
    available: bool = False
    unavailable_reason: str | None = None
    regions: tuple[RegionExtraction, ...] = ()

    @property
    def separated_regions(self) -> int:
        """Number of regions that produced at least one per-speaker track."""

        return sum(1 for region in self.regions if region.separated)

    @property
    def tracks(self) -> int:
        """Number of per-speaker waveforms produced."""

        return sum(len(region.tracks) for region in self.regions)

    @property
    def refused_tracks(self) -> int:
        """Number of characters inside a requested region that got no track."""

        return sum(len(region.refusals) for region in self.regions)

    def track_for(
        self, region: CrosstalkRegion, speaker_id: str
    ) -> ExtractedTrack | None:
        """Return the track for ``speaker_id`` in ``region``, if any."""

        for item in self.regions:
            if item.region == region:
                return item.track_for(speaker_id)
        return None

    def summary(self) -> str:
        """Return the one-line account a run prints."""

        if not self.available:
            return self.unavailable_reason or "unavailable"
        if not self.regions:
            return "no crosstalk region needed separation"
        text = (
            f"{self.separated_regions}/{len(self.regions)} crosstalk region(s) "
            f"separated into {self.tracks} per-speaker track(s)"
        )
        if self.refused_tracks:
            text += f", {self.refused_tracks} track(s) refused"
        return text

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view for the run manifest."""

        return {
            "backend": self.backend,
            "model": self.model,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "regions_requested": len(self.regions),
            "regions_separated": self.separated_regions,
            "tracks": self.tracks,
            "refused_tracks": self.refused_tracks,
            "detail": [region.as_dict() for region in self.regions],
        }


@dataclass(frozen=True, slots=True)
class ExtractionRequest:
    """A crosstalk region plus the characters that need a track inside it."""

    region: CrosstalkRegion
    speakers: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.speakers:
            raise ExtractionError("an extraction request needs at least one speaker")


def _register_of(registers: Mapping[str, float] | None, speaker_id: str) -> float | None:
    """Return a speaker's median pitch in Hz, when it is known."""

    if not registers:
        return None
    value = registers.get(speaker_id)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        median = float(value)
        if math.isfinite(median) and median > 0:
            return median
    return None


def _measure(samples: np.ndarray) -> tuple[float | None, int]:
    """Return the source's median F0 and its voiced frame count."""

    try:
        profile = estimate_f0(samples, SAMPLE_RATE)
    except ProsodyError:
        return None, 0
    return (
        None if profile.median_hz is None else float(profile.median_hz),
        profile.voiced_frames,
    )


def _rms(samples: np.ndarray) -> float:
    """Return the root-mean-square level of a waveform."""

    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def _distance(measured: float, register: float) -> float:
    """Return the log-ratio distance between a measured F0 and a register."""

    return abs(math.log2(measured / register))


def associate(
    sources: Sequence[np.ndarray],
    region: CrosstalkRegion,
    requested: Sequence[str],
    registers: Mapping[str, float] | None,
    *,
    mixture_rms: float,
) -> tuple[list[ExtractedTrack], list[Refusal]]:
    """Credit separated sources to characters, refusing anything ambiguous.

    A blind separator says "these are two voices", not "this one is Skyler". The
    association therefore uses the registers the pipeline measured before
    transcription existed: a source is credited only when its own median F0 sits
    inside the character's expressive band *and* is decisively nearer that band
    than any other region speaker's. Everything else is refused with the measured
    value, because a refusal can be audited while a wrong voice cannot be unheard.
    """

    # ``(source index, samples, median F0, voiced frames, RMS)`` per source.
    measured: list[tuple[int, np.ndarray, float | None, int, float]] = []
    for index, source in enumerate(sources):
        if source.size == 0:
            continue
        median, voiced = _measure(source)
        measured.append((index, source, median, voiced, _rms(source)))

    tracks: list[ExtractedTrack] = []
    refusals: list[Refusal] = []
    taken: dict[int, str] = {}
    for speaker_id in requested:
        wanted = _register_of(registers, speaker_id)
        if wanted is None:
            refusals.append(
                Refusal(
                    speaker_id,
                    "no register was measured for this character, so the separated "
                    "voices cannot be told apart",
                )
            )
            continue

        candidates = [
            item
            for item in measured
            if item[2] is not None
            and item[3] >= MIN_VOICED_FRAMES
            and item[4] >= mixture_rms * MIN_SOURCE_RMS_RATIO
        ]
        if not candidates:
            refusals.append(
                Refusal(
                    speaker_id,
                    "separation produced no measurable voice for this character",
                    None,
                )
            )
            continue

        index, source, median, _voiced, _level = min(
            candidates, key=lambda item: _distance(float(item[2]), wanted)
        )
        assert median is not None
        distance = _distance(median, wanted)
        if distance > math.log2(MAX_REGISTER_RATIO):
            refusals.append(
                Refusal(
                    speaker_id,
                    f"nearest separated voice measured {median:.1f} Hz, outside "
                    f"{MAX_REGISTER_RATIO:.2f}x of this character's {wanted:.1f} Hz",
                    median,
                )
            )
            continue

        ambiguous_with: str | None = None
        for other_id in region.speaker_ids:
            if other_id == speaker_id:
                continue
            other = _register_of(registers, other_id)
            if other is None or other == wanted:
                continue
            if distance * ASSOCIATION_MARGIN >= _distance(median, other):
                ambiguous_with = other_id
                break
        if ambiguous_with is not None:
            refusals.append(
                Refusal(
                    speaker_id,
                    f"separated voice measured {median:.1f} Hz, which is not "
                    f"decisively nearer {wanted:.1f} Hz than {ambiguous_with}'s "
                    "register",
                    median,
                )
            )
            continue

        if index in taken:
            # Two characters cannot be the same waveform. Withdraw the earlier
            # credit as well: whichever of the two it is, it is not both.
            first = taken[index]
            refusals = [item for item in refusals if item.speaker_id != first]
            refusals.append(
                Refusal(
                    first,
                    f"the same separated source was the best match for {speaker_id}",
                    median,
                )
            )
            refusals.append(
                Refusal(
                    speaker_id,
                    f"the same separated source was the best match for {first}",
                    median,
                )
            )
            tracks = [track for track in tracks if track.speaker_id != first]
            continue

        taken[index] = speaker_id
        tracks.append(
            ExtractedTrack(
                speaker_id=speaker_id,
                start=region.start,
                end=region.end,
                samples=np.ascontiguousarray(source, dtype=np.float32),
                sample_rate=SAMPLE_RATE,
                measured_hz=median,
            )
        )

    return tracks, refusals


def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    """Resample mono audio with the project's usual polyphase filter."""

    if source_rate == target_rate:
        return np.ascontiguousarray(samples, dtype=np.float32)
    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    converted = resample_poly(
        samples, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)
    return np.ascontiguousarray(converted)


class PyannoteSeparator:
    """``pyannote.audio`` speech-separation pipeline (the PixIT model)."""

    backend = "pyannote"
    sample_rate = SAMPLE_RATE

    def __init__(self, pipeline: Any, torch_module: Any, model: str) -> None:
        self._pipeline = pipeline
        self._torch = torch_module
        self.model = model

    def separate(self, samples: np.ndarray, num_speakers: int) -> list[np.ndarray]:
        """Run the pipeline on one in-memory window and return its sources."""

        waveform = self._torch.from_numpy(
            np.ascontiguousarray(samples, dtype=np.float32)
        ).unsqueeze(0)
        try:
            output = self._pipeline(
                {"waveform": waveform, "sample_rate": self.sample_rate},
                num_speakers=int(num_speakers),
            )
        except Exception as exc:  # model/runtime failure: the caller falls back
            raise ExtractionError(
                f"pyannote speech separation failed: {type(exc).__name__}: {exc}"
            ) from exc

        if not isinstance(output, tuple) or len(output) < 2:
            return []
        sources = output[1]
        data = getattr(sources, "data", None)
        if data is None:
            return []
        array = np.asarray(data, dtype=np.float32)
        if array.ndim == 1:
            array = array[:, None]

        tracks: list[np.ndarray] = []
        for column in range(array.shape[1]):
            source = array[:, column]
            if source.size >= samples.size:
                source = source[: samples.size]
            else:
                source = np.pad(source, (0, samples.size - source.size))
            tracks.append(np.ascontiguousarray(source, dtype=np.float32))
        return tracks


class SpeechBrainSeparator:
    """SpeechBrain SepFormer, a blind two-source separator."""

    backend = "speechbrain"
    sample_rate = SAMPLE_RATE

    def __init__(
        self, model: Any, torch_module: Any, model_name: str, model_rate: int
    ) -> None:
        self._model = model
        self._torch = torch_module
        self.model = model_name
        #: Rate the checkpoint expects. Sources are converted back to
        #: :data:`SAMPLE_RATE` so every caller sees one rate.
        self.input_rate = model_rate

    def separate(self, samples: np.ndarray, num_speakers: int) -> list[np.ndarray]:
        """Run SepFormer on one window and return up to ``num_speakers`` sources."""

        prepared = _resample(samples, self.sample_rate, self.input_rate)
        batch = self._torch.from_numpy(prepared).unsqueeze(0)
        try:
            estimated = self._model.separate_batch(batch)
        except Exception as exc:
            raise ExtractionError(
                f"speechbrain separation failed: {type(exc).__name__}: {exc}"
            ) from exc

        # SepFormer returns ``(batch, time, sources)``, not ``(batch, sources, time)``.
        array = estimated.detach().cpu().numpy()
        if array.ndim != 3:
            return []
        tracks: list[np.ndarray] = []
        for index in range(min(array.shape[2], max(1, num_speakers))):
            source = _resample(array[0, :, index], self.input_rate, self.sample_rate)
            if source.size >= samples.size:
                source = source[: samples.size]
            else:
                source = np.pad(source, (0, samples.size - source.size))
            tracks.append(np.ascontiguousarray(source, dtype=np.float32))
        return tracks


def _load_pyannote(settings: Settings, model: str) -> Separator:
    if not settings.huggingface_token:
        raise UnavailableError(
            "no HUGGINGFACE_TOKEN is configured, and the separation checkpoint is "
            "gated on Hugging Face"
        )

    try:
        import torch
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise UnavailableError(
            "pyannote.audio/torch are not installed, so crosstalk separation "
            "cannot run"
        ) from exc

    try:
        pipeline = Pipeline.from_pretrained(
            model,
            token=settings.huggingface_token,
            cache_dir=str(settings.model_cache_dir),
        )
    except Exception as exc:
        raise UnavailableError(
            f"could not load separation pipeline {model}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if pipeline is None:
        raise UnavailableError(
            f"separation pipeline {model} could not be downloaded; accept its "
            "conditions on Hugging Face for the configured token"
        )

    try:
        pipeline.to(torch.device(settings.device))
    except Exception as exc:
        raise UnavailableError(
            f"could not move separation pipeline {model} to {settings.device}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not getattr(pipeline, "instantiated", False):
        try:
            pipeline.instantiate(
                {"separation": {"leakage_removal": True, "asr_collar": 0.0}}
            )
        except Exception as exc:
            raise UnavailableError(
                f"separation pipeline {model} could not be instantiated: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    return PyannoteSeparator(pipeline, torch, model)


def _load_speechbrain(settings: Settings, model: str) -> Separator:
    try:
        import torch
        from speechbrain.inference.separation import SepformerSeparation
        from speechbrain.utils.fetching import LocalStrategy
    except ImportError as exc:
        raise UnavailableError(
            "speechbrain is not installed, so the speechbrain separation backend "
            "cannot run"
        ) from exc

    # One directory per checkpoint: speechbrain writes ``hyperparams.yaml`` and the
    # weights by fixed names, so two models must not share a save directory.
    cache = settings.model_cache_dir / "speechbrain" / model.replace("/", "--")
    try:
        loaded = SepformerSeparation.from_hparams(
            source=model,
            savedir=str(cache),
            run_opts={"device": settings.device},
            # Copy instead of symlinking: speechbrain's default requires a
            # privilege Windows does not grant a normal user, and the A40 pod has
            # no reason to keep a second copy in its Hub cache either.
            local_strategy=LocalStrategy.COPY_SKIP_CACHE,
        )
    except Exception as exc:
        raise UnavailableError(
            f"could not load speechbrain separation model {model}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    # ``sample_rate`` lives in the checkpoint's hyperparameters, not on the class.
    rate = getattr(getattr(loaded, "hparams", None), "sample_rate", 8_000)
    try:
        rate = int(rate)
    except (TypeError, ValueError):
        rate = 8_000
    return SpeechBrainSeparator(loaded, torch, model, rate)


def resolve_backend_model(settings: Settings) -> tuple[str, str]:
    """Return the configured ``(backend, checkpoint)`` for this run.

    A settings object built by hand (or a backend overridden on the command line)
    can still carry the other backend's default model. A known default is treated
    as "not chosen" so the selected backend gets the checkpoint it understands;
    an explicit custom id is always kept.
    """

    backend = (settings.crosstalk_separation_backend or "pyannote").lower()
    configured = settings.crosstalk_separation_model.strip()
    backend_default = default_crosstalk_separation_model(backend)
    known_defaults = {DEFAULT_PYANNOTE_MODEL, DEFAULT_SPEECHBRAIN_MODEL}
    model = (
        backend_default
        if not configured
        or (configured in known_defaults and configured != backend_default)
        else configured
    )
    return backend, model


def build_separator(settings: Settings) -> tuple[Separator | None, str | None]:
    """Return a ready separator, or ``(None, reason)`` when the stage cannot run.

    This function never raises: a missing package, token, gated checkpoint or an
    unavailable device is a *reason*, and the caller keeps today's behaviour.
    """

    if not settings.crosstalk_separation:
        return None, (
            "disabled (set CROSSTALK_SEPARATION=1 to give each character in a "
            "crosstalk their own waveform)"
        )
    backend, model = resolve_backend_model(settings)
    if backend not in ("pyannote", "speechbrain"):
        return None, f"unknown CROSSTALK_SEPARATION_BACKEND {backend!r}"

    try:
        if backend == "pyannote":
            return _load_pyannote(settings, model), None
        return _load_speechbrain(settings, model), None
    except UnavailableError as exc:
        logger.info("crosstalk separation unavailable: %s", exc)
        return None, str(exc)
    except Exception as exc:  # defensive: a model load must never end a run
        logger.warning(
            "crosstalk separation could not be started: %s: %s",
            type(exc).__name__,
            exc,
        )
        return None, f"{backend} separator could not be started: {type(exc).__name__}: {exc}"


def _crop(
    source: np.ndarray,
    *,
    window_start: float,
    region: CrosstalkRegion,
) -> np.ndarray:
    """Crop a source that covers the padded window back to the region."""

    first = int(math.floor((region.start - window_start) * SAMPLE_RATE))
    last = int(math.ceil((region.end - window_start) * SAMPLE_RATE))
    first = min(max(first, 0), len(source))
    last = min(max(last, first), len(source))
    return np.ascontiguousarray(source[first:last], dtype=np.float32)


def extract_tracks(
    audio: np.ndarray,
    requests: Sequence[ExtractionRequest],
    *,
    separator: Separator,
    registers: Mapping[str, float] | None,
    context_seconds: float = DEFAULT_CONTEXT_SECONDS,
) -> ExtractionReport:
    """Separate every requested crosstalk region into per-speaker tracks.

    The model sees the region plus ``context_seconds`` on each side, because a
    separator needs more than a fraction of a second to learn two voices; the
    returned tracks are cropped back to the region so their timestamps are the
    film's. A failure on one region is recorded and does not stop the others.
    """

    backend = getattr(separator, "backend", "unknown")
    model = getattr(separator, "model", "unknown")
    rate = int(getattr(separator, "sample_rate", SAMPLE_RATE))
    if rate != SAMPLE_RATE:
        return ExtractionReport(
            backend=backend,
            model=model,
            available=False,
            unavailable_reason=(
                f"separator {backend} returns {rate} Hz audio, but this stage "
                f"requires {SAMPLE_RATE} Hz"
            ),
        )
    report = ExtractionReport(backend=backend, model=model, available=True)
    if not requests:
        return report

    total_seconds = len(audio) / SAMPLE_RATE
    context = max(0.0, float(context_seconds))
    regions: list[RegionExtraction] = []
    for request in requests:
        region = request.region
        if region.duration < MIN_TRACK_SECONDS:
            reasons = tuple(
                Refusal(speaker_id, "the region is too short to hold a separated track")
                for speaker_id in request.speakers
            )
            regions.append(RegionExtraction(region=region, refusals=reasons))
            continue

        window_start = max(0.0, region.start - context)
        window_end = min(total_seconds, region.end + context)
        first = int(round(window_start * SAMPLE_RATE))
        last = int(round(window_end * SAMPLE_RATE))
        window = audio[first:last]
        if window.size == 0:
            reasons = tuple(
                Refusal(speaker_id, "the region lies outside the decoded audio")
                for speaker_id in request.speakers
            )
            regions.append(RegionExtraction(region=region, refusals=reasons))
            continue

        try:
            sources = separator.separate(window, len(region.speaker_ids))
        except ExtractionError as exc:
            reasons = tuple(
                Refusal(speaker_id, str(exc)) for speaker_id in request.speakers
            )
            regions.append(RegionExtraction(region=region, refusals=reasons))
            logger.warning("crosstalk separation failed for %s: %s", region, exc)
            continue
        except Exception as exc:  # defensive: never fail the run
            reasons = tuple(
                Refusal(
                    speaker_id,
                    f"separation failed: {type(exc).__name__}: {exc}",
                )
                for speaker_id in request.speakers
            )
            regions.append(RegionExtraction(region=region, refusals=reasons))
            logger.warning(
                "crosstalk separation failed for %s: %s: %s",
                region,
                type(exc).__name__,
                exc,
            )
            continue

        cropped = [
            _crop(source, window_start=window_start, region=region)
            for source in sources
        ]
        # Compare each source with the mixture *inside the region*, not with the
        # padded window: a quiet overlap in a loud neighbourhood is still a region
        # whose tracks should be judged against its own level.
        mixture_rms = _rms(_crop(window, window_start=window_start, region=region))
        tracks, refusals = associate(
            cropped,
            region,
            request.speakers,
            registers,
            mixture_rms=mixture_rms,
        )
        regions.append(
            RegionExtraction(region=region, tracks=tuple(tracks), refusals=tuple(refusals))
        )

    return ExtractionReport(
        backend=report.backend,
        model=report.model,
        available=True,
        regions=tuple(regions),
    )


__all__ = [
    "ASSOCIATION_MARGIN",
    "DEFAULT_CONTEXT_SECONDS",
    "DEFAULT_PYANNOTE_MODEL",
    "DEFAULT_SPEECHBRAIN_MODEL",
    "ExtractedTrack",
    "ExtractionError",
    "ExtractionReport",
    "ExtractionRequest",
    "MAX_REGISTER_RATIO",
    "MIN_TRACK_SECONDS",
    "MIN_VOICED_FRAMES",
    "PyannoteSeparator",
    "Refusal",
    "RegionExtraction",
    "SAMPLE_RATE",
    "Separator",
    "SpeechBrainSeparator",
    "TargetExtractionError",
    "UnavailableError",
    "associate",
    "build_separator",
    "extract_tracks",
    "resolve_backend_model",
]
