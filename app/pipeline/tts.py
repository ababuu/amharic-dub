"""Amharic speech synthesis in each character's own voice.

This module is the pipeline's TTS stage. It turns the adapted dialogue of
:mod:`app.pipeline.translation` into audio spoken in each character's cloned
voice::

    [AdaptedDialogue] + {speaker: VoiceProfile}
        -> synthesize_dialogue() -> [TtsClip]

Engines
-------
Two engines are supported, and which one is in use decides whether the
voice-profile stage has anything to do:

* :class:`OmniVoiceEngine` (the default) - ``african-low-resource/omnivoice-amharic``
  clones the character's voice from their reference recording and speaks the
  Amharic as it, honouring a per-line pacing request. It reads a per-speaker
  reference, so :func:`engine_needs_profiles` is true for it.
* :class:`MmsAmharicEngine` - MMS-TTS Amharic speaks one voice for the whole
  film. It has no reference to follow and no identity to convert to, so a run
  using it builds no voice profiles at all.

Voice identity
--------------
``VoiceProfile.reference_audio`` is used for one thing only: as the voice the
cloning engine speaks *as*, which is what makes a character sound like themselves
across a whole film. A
:class:`~app.pipeline.voice_profiles.VoiceProfile` is never modified here - this
module only reads it. Because the engine takes reference audio directly, no
clone-prompt serialization happens at this stage, so
``VoiceProfile.clone_prompt_path`` stays untouched.

Performance metadata
--------------------
:class:`~app.pipeline.translation.AdaptedDialogue` carries emotion, intensity,
delivery and the pauses around the line. None of it is discarded:

* ``emotion``, ``intensity`` and ``delivery`` are collected into
  :class:`PerformanceControls` and recorded on every clip, so what the dialogue
  model asked for travels with the audio.
* ``pause_before`` and ``pause_after`` are rendered as silence around the
  converted line by :func:`synthesize_dialogue`, capped at
  ``TTS_MAX_PAUSE_SECONDS``. Both the declared and the rendered value are kept in
  the clip, so nothing the dialogue model asked for is lost.

What this stage deliberately does not do
----------------------------------------
* **No duration fitting.** The clip is returned as generated, together with the
  metadata :mod:`app.pipeline.timing` needs (``speech_duration``, the rendered
  pauses, the original window). Nothing here time-stretches to the original line,
  and no target duration is forced.
* **No mixing and no muxing.** The music and effects stems are not touched.
* **No resampling.** The returned clip keeps the rate the engine produced;
  bringing the whole dub to one rate is the mixing stage's job.
* **No model downloads at import time** (and none in the test suite). Both
  engines are lazy: nothing is imported, downloaded or loaded until a clip is
  actually synthesized, and a loaded model is cached and reused for every
  following line instead of being loaded once per line.
* **No external process.** Rendering pauses and clips uses ``soundfile``, so this
  stage shells out to nothing.

A line that cannot be dubbed at all - a window too short to hold a word, or text
with nothing to pronounce - is skipped before the engine is called and reported,
so a fragment cannot end a run over a film. An engine that genuinely *fails* on a
line still stops the run with that line named, unless
``TTS_CONTINUE_ON_FAILURE=true`` asks for a reported hole instead.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import (
    DEFAULT_MMS_SAMPLE_RATE,
    DEFAULT_MMS_SEED,
    DEFAULT_MMS_SPEAKING_RATE,
    DEFAULT_TTS_MIN_LINE_SECONDS,
    DEFAULT_TTS_MODEL,
    Settings,
    get_settings,
)
from app.pipeline.amharic_text import count_syllables, has_pronounceable_text
from app.pipeline.dialogue_context import (
    PacingPlan,
    plan_pacing,
)
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.voice_profiles import (
    VoiceProfile,
    portable_path,
    speaker_directory_name,
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

#: Stage directory inside ``WORK_DIR``; every path this module writes lives here.
TTS_DIRECTORY_NAME = "tts"

#: Sub-directory holding the takes an engine produced, before pauses are rendered.
TAKE_DIRECTORY_NAME = "takes"

#: Sub-directory holding the clips handed to :mod:`app.pipeline.timing`.
CLIP_DIRECTORY_NAME = "clips"

#: Bump when a change here would make an existing take or clip wrong; it is part
#: of every generated file name, so old artifacts can never be mistaken for new
#: ones.
#:
#: 3: OmniVoice prompts keep the reference audio untouched when its transcript is
#: known, so takes made while silence removal could leave the reference's last word
#: at the start of every line must not be reused.
SYNTHESIS_RECIPE_VERSION = 3

#: Shortest original window a line is *reported* for, in seconds. It is not a reason to
#: leave a line unsaid: a brief English line is normal, and the Amharic that replaces it
#: needs more time, which :mod:`app.pipeline.timing` finds by moving the lines after it.
#: Lines below this are voiced and counted in :attr:`SynthesisResult.notes`.
MINIMUM_REPORTABLE_LINE_SECONDS = DEFAULT_TTS_MIN_LINE_SECONDS

#: Shortest Amharic line worth synthesizing, in syllables. One Fidel character is one
#: syllable, so this counts the script's own unit.
#:
#: The check that enforces this is :func:`app.pipeline.amharic_text.has_pronounceable_text`,
#: not a bare syllable count, and deliberately so: Roman-script text counts as
#: pronounceable too - an English-derived word is a word - so a borrowed word is never
#: dropped for being written in Roman letters while it waits to be rewritten in Fidel.
#: Counting syllables here would reject exactly those lines.
MINIMUM_SPEAKABLE_SYLLABLES = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TtsError(RuntimeError):
    """Base class for every error raised by this module."""


class ConfigurationError(TtsError, ValueError):
    """Settings or engine options contradict each other or are out of range."""


class InvalidDialogueError(TtsError, ValueError):
    """The supplied dialogue is not an iterable of AdaptedDialogue lines."""


class InvalidVoiceProfileError(TtsError, ValueError):
    """The supplied voice profiles are not a speaker-keyed VoiceProfile mapping."""


class MissingVoiceProfileError(TtsError):
    """A speaker who speaks in the dialogue has no voice profile."""


class InvalidInputError(TtsError, ValueError):
    """A path, a timestamp or a duration is not usable."""


class MissingInputError(TtsError):
    """The dialogue stem, or a profile's reference audio, does not exist."""


class InvalidAudioError(TtsError):
    """An input could not be read as audio at all."""


class InvalidEngineError(TtsError, ValueError):
    """An injected engine is not an adapter of the expected kind."""


class EngineLoadError(TtsError):
    """An engine could not be loaded on the configured device."""


class SynthesisError(TtsError):
    """An engine failed to synthesize a line."""


class MissingOutputError(TtsError):
    """An engine reported success without writing any audio."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _finite_seconds(name: str, value: object) -> float:
    """Return ``value`` as a finite number of seconds."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidInputError(
            f"{name} must be a number of seconds, got {type(value).__name__}"
        )

    seconds = float(value)
    if not math.isfinite(seconds):
        raise InvalidInputError(f"{name} must be finite, got {value!r}")
    return seconds


def _clip_path(name: str, value: object) -> Path:
    """Return ``value`` as a :class:`~pathlib.Path`, or raise."""

    if isinstance(value, Path):
        return value
    if isinstance(value, str) and value.strip():
        return Path(value)
    raise InvalidInputError(f"{name} must be a path or a non-empty path string")


def _positive(name: str, value: object) -> float:
    """Return ``value`` as a finite, strictly positive float."""

    number = _finite_seconds(name, value)
    if number <= 0:
        raise ConfigurationError(f"{name} must be positive, got {number}")
    return number


def _rate(name: str, value: object) -> float:
    """Return ``value`` as a finite float in ``[0.0, 1.0]``."""

    number = _finite_seconds(name, value)
    if not 0.0 <= number <= 1.0:
        raise ConfigurationError(f"{name} must be between 0.0 and 1.0, got {number}")
    return number


def _usable_audio(path: Path) -> bool:
    """Return whether ``path`` is an existing, non-empty audio artifact."""

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:  # pragma: no cover - a failing stat is simply not usable
        return False


def _validate_audio_file(value: str | Path, *, label: str) -> Path:
    """Return ``value`` as an existing, readable audio file."""

    path = Path(value)
    if not path.exists():
        raise MissingInputError(f"{label} not found: {path}")
    if not path.is_file():
        raise InvalidInputError(f"{label} is not a regular file: {path}")
    return path


def _read_audio_info(path: Path, *, label: str) -> tuple[int, int]:
    """Return ``(frames, sample_rate)`` of ``path``, or raise clearly."""

    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as exc:  # soundfile errors subclass RuntimeError
        raise InvalidAudioError(f"could not read {label} {path}: {exc}") from exc

    if info.frames < 1 or info.samplerate < 1:
        raise InvalidAudioError(f"{label} {path} contains no usable samples")
    return int(info.frames), int(info.samplerate)


def _read_mono(path: Path, *, label: str) -> tuple[np.ndarray, int]:
    """Read ``path`` as a mono ``float32`` array plus its sample rate."""

    try:
        data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read {label} {path}: {exc}") from exc

    if data.size == 0:
        raise InvalidAudioError(f"{label} {path} contains no usable samples")

    # The pipeline is mono: a multi-channel engine output is folded down rather
    # than rejected, because the identity, not the channel count, is what a take
    # is judged on.
    samples = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


#: Peak a written clip is held below. Anything above this would be truncated by the
#: PCM-16 conversion, which is irreversible distortion - a harsh, buzzy timbre that
#: no later stage can undo. Attenuating instead costs a little level and preserves
#: the waveform, which is the better trade.
CLIP_CEILING = 0.999


def _peak_limit(samples: np.ndarray) -> tuple[np.ndarray, float]:
    """Return ``samples`` below the ceiling, and the gain applied in dB.

    A signal that is already inside the ceiling is returned untouched with ``0.0``.
    One that would clip is scaled *as a whole*, so the waveform keeps its shape:
    clipping the peaks would change the timbre, which is exactly the defect this
    exists to prevent.
    """

    data = np.asarray(samples, dtype=np.float32)
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak <= CLIP_CEILING or peak <= 0.0:
        return data, 0.0

    gain = CLIP_CEILING / peak
    return (data * gain).astype(np.float32), 20.0 * math.log10(gain)


def _write_mono(path: Path, samples: np.ndarray, sample_rate: int, *, label: str) -> Path:
    """Write a mono PCM-16 WAV, peak-limited rather than clipped, and return ``path``."""

    limited, _ = _peak_limit(samples)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(path),
            limited,
            int(sample_rate),
            format="WAV",
            subtype="PCM_16",
        )
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not write {label} {path}: {exc}") from exc
    return path


def _as_samples(waveform: Any) -> np.ndarray:
    """Return a generated waveform as a mono ``float32`` array.

    A diffusion engine hands back a ``torch.Tensor`` and some models a NumPy array;
    both are accepted without importing either runtime here.
    """

    data = waveform
    for attribute in ("detach", "cpu", "numpy"):
        method = getattr(data, attribute, None)
        if callable(method):
            data = method()

    array = np.asarray(data, dtype=np.float32)
    if array.ndim == 1:
        return np.ascontiguousarray(array, dtype=np.float32)
    if array.ndim == 2:
        if array.shape[0] == 1:
            return np.ascontiguousarray(array[0], dtype=np.float32)
        return np.ascontiguousarray(array.mean(axis=0), dtype=np.float32)
    raise InvalidAudioError(
        f"an engine returned a {array.ndim}-dimensional waveform; mono or "
        "'(channels, samples)' audio was expected"
    )


# ---------------------------------------------------------------------------
# Performance metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PerformanceControls:
    """One line's performance direction, carried over from the dialogue stage.

    ``emotion``, ``intensity`` and ``delivery`` are the dialogue model's own
    words, kept verbatim rather than reinterpreted: they are what the adaptation
    asked for, and they travel into the manifest beside the audio produced from
    them. ``pause_before`` and ``pause_after`` are validated here and rendered as
    silence by :func:`_render_clip`.
    """

    emotion: str
    intensity: float
    delivery: str
    pause_before: float
    pause_after: float
    seed: int
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name, text in (("emotion", self.emotion), ("delivery", self.delivery)):
            if not isinstance(text, str) or not text.strip():
                raise InvalidDialogueError(f"{name} must be a non-empty string")

        intensity = _finite_seconds("intensity", self.intensity)
        if not 0.0 <= intensity <= 1.0:
            raise InvalidDialogueError(
                f"intensity must be between 0.0 and 1.0, got {intensity}"
            )
        object.__setattr__(self, "intensity", intensity)

        for name, value in (
            ("pause_before", self.pause_before),
            ("pause_after", self.pause_after),
        ):
            seconds = _finite_seconds(name, value)
            if seconds < 0:
                raise InvalidDialogueError(f"{name} must be >= 0 seconds, got {seconds}")
            object.__setattr__(self, name, seconds)

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise InvalidDialogueError(
                f"seed must be a non-negative integer, got {self.seed!r}"
            )

    @classmethod
    def from_dialogue(cls, dialogue: AdaptedDialogue, *, seed: int) -> "PerformanceControls":
        """Carry ``dialogue``'s direction into the clip's record.

        The direction is copied, not interpreted: both supported engines take their
        delivery from the voice reference and the run's pacing plan, so there is no
        per-line knob here to map onto. Keeping the dialogue model's own fields
        intact is what lets a manifest state what was asked for, whichever engine
        spoke it.
        """

        if not isinstance(dialogue, AdaptedDialogue):
            raise InvalidDialogueError(
                f"expected an AdaptedDialogue, got {type(dialogue).__name__}"
            )

        return cls(
            emotion=dialogue.emotion,
            intensity=float(dialogue.intensity),
            delivery=dialogue.delivery,
            pause_before=float(dialogue.pause_before),
            pause_after=float(dialogue.pause_after),
            seed=seed,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the performance."""

        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "delivery": self.delivery,
            "pause_before": self.pause_before,
            "pause_after": self.pause_after,
            "seed": self.seed,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TtsClip:
    """One synthesized Amharic line, ready for :mod:`app.pipeline.timing`.

    The clip is the character's voice with the line's pauses already rendered into
    it, so it is a self-contained performance. ``duration`` is what timing has to
    place; ``speech_duration`` is the generated speech alone, and the rendered
    pauses are reported separately, so a fitter can trim the silence or account
    for it without guessing. No target duration is applied here.
    """

    index: int
    dialogue: AdaptedDialogue
    performance: PerformanceControls
    audio_path: Path
    #: The raw take the engine wrote, before the line's pauses were rendered into
    #: ``audio_path``; the character's reference recording, when the engine spoke as
    #: one; and the performance prompt, which no supported engine uses any more so it
    #: is ``None``. Kept in the record because it is part of the artifact schema the
    #: later stages and the manifest read.
    take_path: Path
    performance_reference_path: Path | None
    voice_reference_path: Path | None
    sample_rate: int
    speech_duration: float
    rendered_pause_before: float
    rendered_pause_after: float
    performance_engine: str
    style_engine: str

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise InvalidInputError(
                f"index must be a non-negative integer, got {self.index!r}"
            )
        if not isinstance(self.dialogue, AdaptedDialogue):
            raise InvalidDialogueError(
                f"dialogue must be an AdaptedDialogue, got {type(self.dialogue).__name__}"
            )
        if not isinstance(self.performance, PerformanceControls):
            raise InvalidInputError(
                "performance must be PerformanceControls, got "
                f"{type(self.performance).__name__}"
            )

        for name, value in (
            ("audio_path", self.audio_path),
            ("take_path", self.take_path),
        ):
            object.__setattr__(self, name, _clip_path(name, value))

        # Optional: a single-voice engine has neither a performance prompt nor an
        # identity reference, so these are absent rather than fabricated.
        for name, value in (
            ("performance_reference_path", self.performance_reference_path),
            ("voice_reference_path", self.voice_reference_path),
        ):
            if value is not None:
                object.__setattr__(self, name, _clip_path(name, value))

        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise InvalidInputError(
                f"sample_rate must be an integer, got {self.sample_rate!r}"
            )
        if self.sample_rate < 1:
            raise InvalidInputError(f"sample_rate must be positive, got {self.sample_rate}")

        speech = _finite_seconds("speech_duration", self.speech_duration)
        if speech <= 0:
            raise InvalidAudioError(f"speech_duration must be positive, got {speech}")
        object.__setattr__(self, "speech_duration", speech)

        for name, value in (
            ("rendered_pause_before", self.rendered_pause_before),
            ("rendered_pause_after", self.rendered_pause_after),
        ):
            seconds = _finite_seconds(name, value)
            if seconds < 0:
                raise InvalidInputError(f"{name} must be >= 0 seconds, got {seconds}")
            object.__setattr__(self, name, seconds)

        for name in ("performance_engine", "style_engine"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise InvalidInputError(f"{name} must be a non-empty string")

    @property
    def speaker_id(self) -> str:
        """The diarization speaker id this clip belongs to."""

        return self.dialogue.speaker_id

    @property
    def start(self) -> float:
        """Start of the original line in the source timeline, in seconds."""

        return self.dialogue.start

    @property
    def end(self) -> float:
        """End of the original line in the source timeline, in seconds."""

        return self.dialogue.end

    @property
    def amharic(self) -> str:
        """The Amharic text that was spoken."""

        return self.dialogue.amharic

    @property
    def original_duration(self) -> float:
        """Length of the original line the dub has to fit into, in seconds."""

        return self.dialogue.duration

    @property
    def duration(self) -> float:
        """Length of the delivered audio: generated speech plus rendered pauses."""

        return (
            self.speech_duration + self.rendered_pause_before + self.rendered_pause_after
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the clip for a manifest or a report."""

        entry: dict[str, object] = {
            "index": self.index,
            "speaker_id": self.speaker_id,
            "start": self.start,
            "end": self.end,
            "original_duration": self.original_duration,
            "amharic": self.amharic,
            "source_text": self.dialogue.source_text,
            "performance": self.performance.to_dict(),
            "audio_path": portable_path(self.audio_path),
            "take_path": portable_path(self.take_path),
            "performance_reference_path": (
                None
                if self.performance_reference_path is None
                else portable_path(self.performance_reference_path)
            ),
            "voice_reference_path": (
                None
                if self.voice_reference_path is None
                else portable_path(self.voice_reference_path)
            ),
            "sample_rate": self.sample_rate,
            "speech_duration": self.speech_duration,
            "duration": self.duration,
            "rendered_pause_before": self.rendered_pause_before,
            "rendered_pause_after": self.rendered_pause_after,
            "performance_engine": self.performance_engine,
            "style_engine": self.style_engine,
        }
        if self.dialogue.beats:
            # The sentence was spoken as one utterance, but the performance it was
            # written from changed inside it. Recorded so a later engine or a human
            # director can reproduce that arc instead of inferring it.
            entry["beats"] = [beat.as_dict() for beat in self.dialogue.beats]
        return entry


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------


class TextToSpeechEngine(ABC):
    """An engine that speaks plain text in a single voice.

    This is the simpler contract: no performance reference to follow and no identity
    to convert to. It exists because a single-speaker model such as MMS-TTS has
    neither - it speaks Amharic in one voice, always - and pretending otherwise by
    feeding it a prompt it ignores would be dishonest plumbing.

    Implementations write a mono WAV to ``destination`` and return that path.
    """

    #: Short, stable name recorded in the clip metadata.
    name: str

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """``True`` once the underlying model has actually been loaded."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate of the audio this engine writes."""

    @abstractmethod
    def synthesize(self, *, text: str, destination: Path) -> Path:
        """Speak ``text`` into ``destination``."""


#: The Amharic TTS engine name recorded in clip metadata.
MMS_ENGINE_NAME = "mms-tts-amharic"


def _require_vits() -> tuple[Any, Any, Any]:
    """Return ``(torch, AutoTokenizer, VitsModel)``, or explain what is missing."""

    try:
        import torch
        from transformers import AutoTokenizer, VitsModel
    except ImportError as exc:  # pragma: no cover - only without the runtime
        raise EngineLoadError(
            "MMS-TTS needs torch and transformers; install the runtime dependencies "
            f"before running speech synthesis ({type(exc).__name__}: {exc})"
        ) from exc

    return torch, AutoTokenizer, VitsModel


def _accepts_speaking_rate(network: Any) -> bool:
    """Say whether this build's forward pass takes a ``speaking_rate`` argument.

    Which is not the same as the config having the attribute. ``VitsModel`` copies
    ``config.speaking_rate`` into ``self.speaking_rate`` **in its constructor**, so
    assigning to the config afterwards changes nothing at all - the duration
    predictor reads the attribute captured at construction time. The rate is honoured
    only as an argument to the forward call (``length_scale = 1.0 / speaking_rate``),
    which is what this checks for, by signature rather than by version number so that
    a build that gains or loses the argument is handled either way.

    Only an explicit parameter counts: a ``**kwargs`` catch-all would swallow the
    argument and leave the run silently speaking at the wrong rate, which is the
    failure this exists to prevent.
    """

    target = getattr(network, "forward", None)
    if target is None:
        return False
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):  # a signature this build cannot report
        return False
    return "speaking_rate" in parameters


def _require_uroman() -> Any:
    """Return the ``uroman`` romanizer class, or explain what is missing.

    MMS-TTS Amharic takes its text in the Latin alphabet, so Fidel script has to be
    romanised first. ``uroman`` is Meta's own choice for that step, which is why it
    is used here rather than a hand-rolled transliteration: a wrong romanisation is a
    mispronunciation, and there is no way to test a hand-rolled one against the
    model's own expectations.
    """

    try:
        import uroman as uroman_module
    except ImportError as exc:  # pragma: no cover - only without the runtime
        raise EngineLoadError(
            "MMS-TTS needs the 'uroman' package to romanise Fidel script before "
            f"synthesis; install it from requirements.txt ({type(exc).__name__}: {exc})"
        ) from exc

    return uroman_module.Uroman


class MmsAmharicEngine(TextToSpeechEngine):
    """Adapter around **MMS-TTS Amharic** (``facebook/mms-tts-amh``).

    A VITS model trained on Amharic alone. Two properties shape this adapter, and
    both are properties of the model rather than choices made here:

    * **One voice for the whole film.** VITS for Amharic was trained on a single
      speaker, so there is no identity to clone and no per-character voice. Every
      line comes out in the same voice, and the identity/conversion stages are
      bypassed entirely rather than fed a reference they would ignore.
    * **Latin input.** The checkpoint expects romanised text, so Fidel script is
      converted with ``uroman`` before synthesis. It is handled here so the rest of
      the pipeline can keep working in Fidel.

    The duration predictor samples its rhythm, so generation is seeded: without a
    fixed seed the same line would change length between runs, and neither a
    reproducible dub nor a meaningful baseline would be possible.
    """

    name = MMS_ENGINE_NAME

    def __init__(
        self,
        *,
        model: str = DEFAULT_TTS_MODEL,
        device: str = "cuda",
        cache_dir: Path | None = None,
        seed: int = DEFAULT_MMS_SEED,
        speaking_rate: float = DEFAULT_MMS_SPEAKING_RATE,
        romanizer: Any | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("the TTS model must be a non-empty repository id")
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError("the TTS device must be a non-empty string")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ConfigurationError(f"MMS_SEED must be a non-negative integer, got {seed!r}")
        if not math.isfinite(speaking_rate) or speaking_rate <= 0:
            raise ConfigurationError(
                f"MMS_SPEAKING_RATE must be a positive number, got {speaking_rate!r}"
            )

        self._model = model.strip()
        self._device = device.strip()
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._seed = seed
        self._speaking_rate = float(speaking_rate)
        self._torch: Any | None = None
        self._tokenizer: Any | None = None
        self._network: Any | None = None
        self._uroman: Any | None = romanizer
        self._rate: int | None = None
        self._rate_supported: bool = False

    @property
    def model(self) -> str:
        """The checkpoint this engine was constructed for."""

        return self._model

    @property
    def device(self) -> str:
        """The device this engine runs on."""

        return self._device

    @property
    def is_loaded(self) -> bool:
        """``True`` once the weights are in memory."""

        return self._network is not None

    @property
    def sample_rate(self) -> int:
        """The rate the checkpoint produces, known before loading if configured."""

        return self._rate if self._rate is not None else DEFAULT_MMS_SAMPLE_RATE

    def _load(self) -> tuple[Any, Any, Any]:
        """Load the tokenizer and model once, and return them with torch."""

        if (
            self._tokenizer is not None
            and self._network is not None
            and self._torch is not None
        ):
            return self._torch, self._tokenizer, self._network

        torch, auto_tokenizer, auto_model = _require_vits()
        self._torch = torch

        try:
            self._tokenizer = auto_tokenizer.from_pretrained(
                self._model, cache_dir=self._cache_dir
            )
        except Exception as exc:
            raise EngineLoadError(
                f"could not load the MMS tokenizer for {self._model!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            self._network = auto_model.from_pretrained(
                self._model, cache_dir=self._cache_dir
            )
        except Exception as exc:
            raise EngineLoadError(
                f"could not load the MMS model {self._model!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            self._network.to(torch.device(self._device))
        except Exception as exc:
            raise EngineLoadError(
                f"could not move {self._model!r} to device {self._device!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        self._network.eval()

        # Asking the model for a duration is what lets a line fit its window without
        # being time-stretched afterwards. Doing so is a property of the installed
        # build, not something to assume: an argument the forward pass does not take
        # would be ignored, and the run would speak at the wrong rate with nothing to
        # show for it. A build that cannot honour a *requested* rate therefore fails
        # here; the default rate of 1.0 requests nothing, so it is never a reason to
        # stop.
        self._rate_supported = _accepts_speaking_rate(self._network)
        if not self._rate_supported and self._speaking_rate != DEFAULT_MMS_SPEAKING_RATE:
            raise EngineLoadError(
                f"MMS_SPEAKING_RATE is {self._speaking_rate} but the installed "
                "transformers build cannot control the duration predictor: "
                f"{type(self._network).__name__}.forward takes no 'speaking_rate' "
                "argument, so the request cannot be honoured. Upgrade transformers, "
                "or set MMS_SPEAKING_RATE=1.0 to leave the model's own rhythm alone."
            )

        self._rate = int(
            getattr(self._network.config, "sampling_rate", DEFAULT_MMS_SAMPLE_RATE)
        )
        return self._torch, self._tokenizer, self._network

    def romanize(self, text: str) -> str:
        """Return ``text`` in the Latin alphabet, as this checkpoint expects it.

        Latin text already in the line - a borrowed word, a name - is left as it is:
        romanising it again could only corrupt it, and it is already the script the
        model wants.
        """

        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("there is nothing to synthesize")

        if self._uroman is None:
            uroman_class = _require_uroman()
            try:
                self._uroman = uroman_class()
            except Exception as exc:
                raise EngineLoadError(
                    f"could not initialise uroman: {type(exc).__name__}: {exc}"
                ) from exc

        try:
            return self._uroman.romanize_string(text)
        except Exception as exc:
            raise SynthesisError(
                f"uroman could not romanise Amharic text: {type(exc).__name__}: {exc}"
            ) from exc

    def synthesize(self, *, text: str, destination: Path) -> Path:
        """Speak ``text`` into a mono PCM WAV at ``destination``."""

        target = Path(destination)
        romanized = self.romanize(text)
        torch, tokenizer, network = self._load()

        try:
            inputs = tokenizer(romanized, return_tensors="pt")
        except Exception as exc:
            raise SynthesisError(
                f"the MMS tokenizer rejected a line: {type(exc).__name__}: {exc}"
            ) from exc

        device = next(network.parameters()).device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        try:
            # Seeded, because the duration predictor samples: an unseeded run would
            # give the same line a different length every time.
            torch.manual_seed(self._seed)
            with torch.no_grad():
                if self._rate_supported:
                    waveform = network(
                        **inputs, speaking_rate=self._speaking_rate
                    ).waveform
                else:
                    waveform = network(**inputs).waveform
        except Exception as exc:
            raise SynthesisError(
                f"MMS-TTS failed to synthesize {text!r}: {type(exc).__name__}: {exc}"
            ) from exc

        try:
            samples = waveform.squeeze().detach().to("cpu").numpy().astype(np.float32)
        except Exception as exc:
            raise SynthesisError(
                f"MMS-TTS returned an unusable waveform: {type(exc).__name__}: {exc}"
            ) from exc

        if samples.size == 0:
            raise SynthesisError(f"MMS-TTS returned an empty waveform for {text!r}")

        return _write_mono(
            target, samples, self.sample_rate, label="MMS-TTS take"
        )


# ---------------------------------------------------------------------------
# OmniVoice
# ---------------------------------------------------------------------------

#: The OmniVoice engine name recorded in clip metadata.
OMNIVOICE_ENGINE_NAME = "omnivoice"

#: The Amharic-tuned checkpoint's documented API takes the language name.
OMNIVOICE_AMHARIC = "Amharic"

#: The rate OmniVoice generates at.
OMNIVOICE_SAMPLE_RATE = 24_000

#: Characters that already end a reference transcript as a complete sentence.
_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？", "።", "፧")


def _terminate_reference_text(text: str) -> str:
    """Return ``text`` with terminal punctuation, as OmniVoice's preprocessing would.

    The punctuation is what marks where the reference stops and the line to speak
    begins, so it has to survive when that preprocessing is turned off.
    """

    text = text.strip()
    if text and not text.endswith(_SENTENCE_END):
        return text + "."
    return text


def _require_omnivoice() -> tuple[Any, Any, Any]:
    """Return ``(torch, OmniVoice, OmniVoiceGenerationConfig)``, or say what is missing."""

    try:
        import torch
        from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    except ImportError as exc:  # pragma: no cover - only without the runtime
        raise EngineLoadError(
            "the OmniVoice engine needs the 'omnivoice' package; install the runtime "
            f"dependencies before running speech synthesis ({type(exc).__name__}: {exc})"
        ) from exc

    return torch, OmniVoice, OmniVoiceGenerationConfig


class VoiceCloningEngine(ABC):
    """An engine that speaks text in a voice taken from a reference recording.

    This is the richer of the two contracts, and it exists because a single-voice
    engine cannot express what a zero-shot cloning model does:

    * :class:`TextToSpeechEngine` has one voice for the whole film.
    * Here the reference *is* the identity. Passing the same reference for every line
      of a speaker is what makes a character sound like one person across a film.

    ``reference_text`` is the transcript of the reference when it is known. It is
    optional, but supplying it is what keeps a run from loading an ASR model just to
    transcribe a clip the project already has a transcript for.
    """

    #: Short, stable name recorded in the clip metadata.
    name: str

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """``True`` once the underlying model has actually been loaded."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Sample rate of the audio this engine writes."""

    @property
    def supports_speaking_rate(self) -> bool:
        """``True`` when :meth:`synthesize` can honour a requested rate.

        A rate asked of the model *before* synthesis is the difference between a line
        that fits its window and a line that has to be time-stretched into it, so it is
        reported rather than assumed by the caller.
        """

        return False

    @abstractmethod
    def synthesize(
        self,
        *,
        text: str,
        voice_reference: Path,
        destination: Path,
        speaking_rate: float | None = None,
        reference_text: str | None = None,
    ) -> Path:
        """Speak ``text`` as the voice heard in ``voice_reference``."""


class OmniVoiceEngine(VoiceCloningEngine):
    """Amharic speech in a cloned voice, through k2-fsa's OmniVoice.

    OmniVoice is a zero-shot text-to-speech model spanning hundreds of languages -
    Amharic among them - which clones a voice from a short reference recording. It is
    the engine the project reaches for when each character must keep one voice for a
    whole film, because it takes that voice from the character's own audio rather than
    converting a take afterwards.

    Two of its properties are used deliberately here:

    * **A requested rate.** ``speed`` is passed to the model, so a line can be asked
      for at the length its window allows instead of being stretched into it. Whether
      the request can be honoured is a property of the installed build, so it is
      probed rather than assumed - see :meth:`supports_speaking_rate`.
    * **A deterministic mode.** The model samples by default; both temperatures are
      pinned to zero so the same line, reference and rate give the same audio twice.
      Without that, a re-run could not be compared with the run it is meant to improve.

    This project's configured Amharic-tuned checkpoint documents voice cloning, not
    voice design. Per-character actor references are therefore used to keep voices
    distinct and preserve their vocal presentation.
    """

    name = OMNIVOICE_ENGINE_NAME

    def __init__(
        self,
        *,
        model: str,
        device: str = "cuda",
        steps: int = 32,
        guidance_scale: float = 2.0,
        torch_dtype: str = "float16",
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("the OmniVoice model must be a non-empty id")
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError("the TTS device must be a non-empty string")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ConfigurationError(
                f"OMNIVOICE_STEPS must be a positive integer, got {steps!r}"
            )
        if not math.isfinite(guidance_scale) or guidance_scale <= 0:
            raise ConfigurationError(
                f"OMNIVOICE_GUIDANCE_SCALE must be positive, got {guidance_scale!r}"
            )

        self._model = model.strip()
        self._device = device.strip()
        self._steps = steps
        self._guidance_scale = float(guidance_scale)
        self._torch_dtype = torch_dtype
        self._torch: Any | None = None
        self._network: Any | None = None
        self._config_class: Any | None = None
        #: Cloning prompts, one per (reference, transcript). Encoding a reference is
        #: the per-speaker cost, so a film pays it once per character, not per line.
        self._prompts: dict[tuple[str, str], Any] = {}

    @property
    def model(self) -> str:
        """The checkpoint this engine was constructed for."""

        return self._model

    @property
    def device(self) -> str:
        """The device this engine runs on."""

        return self._device

    @property
    def is_loaded(self) -> bool:
        """``True`` once the weights are in memory."""

        return self._network is not None

    @property
    def sample_rate(self) -> int:
        """OmniVoice generates at 24 kHz."""

        return OMNIVOICE_SAMPLE_RATE

    @property
    def supports_speaking_rate(self) -> bool:
        """``True``: OmniVoice accepts a per-item ``speed`` on its generate call."""

        return True

    def _load(self) -> tuple[Any, Any, Any]:
        """Load the model once and return it with torch and the config class."""

        if self._network is not None and self._config_class is not None:
            return self._torch, self._network, self._config_class

        torch, model_class, config_class = _require_omnivoice()
        self._torch = torch
        self._config_class = config_class

        dtype = getattr(torch, self._torch_dtype, None)
        if dtype is None:
            raise ConfigurationError(
                f"OMNIVOICE_TORCH_DTYPE={self._torch_dtype!r} is not a torch dtype"
            )

        try:
            self._network = model_class.from_pretrained(
                self._model, device_map=self._device, dtype=dtype
            )
        except Exception as exc:
            raise EngineLoadError(
                f"could not load the OmniVoice model {self._model!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return self._torch, self._network, self._config_class

    def _prompt_for(self, reference: Path, reference_text: str | None) -> Any:
        """Return the cached cloning prompt for one reference recording.

        The prompt is what carries a character's identity, so it is created once per
        speaker and reused for every line. ``reference_text`` is passed through when it
        describes the reference exactly: without it the model transcribes the reference
        with an ASR model of its own, which is both slower and one more download.

        OmniVoice preprocessing is disabled when the transcript is known. Its
        mid-silence removal shortens the prompt audio while the transcript still
        describes the original length, so the model treats the reference's trailing
        words as unspoken and repeats them at the start of every generated line (a
        reference ending "in this very room." was heard as "room" before each line).
        Without a transcript, preprocessing runs first and the model transcribes the
        processed audio, so the two stay aligned.
        """

        key = (str(reference), (reference_text or "").strip())
        prompt = self._prompts.get(key)
        if prompt is not None:
            return prompt

        _torch, network, _config = self._load()
        text = _terminate_reference_text(key[1]) or None
        try:
            prompt = network.create_voice_clone_prompt(
                ref_audio=str(reference),
                ref_text=text,
                preprocess_prompt=text is None,
            )
        except Exception as exc:
            raise SynthesisError(
                f"OmniVoice could not build a voice from {reference}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._prompts[key] = prompt
        return prompt

    def synthesize(
        self,
        *,
        text: str,
        voice_reference: Path,
        destination: Path,
        speaking_rate: float | None = None,
        reference_text: str | None = None,
    ) -> Path:
        """Speak ``text`` as the voice in ``voice_reference``, at ``speaking_rate``."""

        if not isinstance(text, str) or not text.strip():
            raise InvalidInputError("there is nothing to synthesize")

        target = Path(destination)
        reference = _validate_audio_file(
            Path(voice_reference), label="the voice reference"
        )

        rate: float | None = None
        if speaking_rate is not None:
            candidate = float(speaking_rate)
            if not math.isfinite(candidate) or candidate <= 0:
                raise ConfigurationError(
                    f"a speaking rate must be a positive number, got {speaking_rate!r}"
                )
            rate = candidate

        _torch, network, config_class = self._load()
        prompt = self._prompt_for(reference, reference_text)

        # Keep token selection greedy, but leave position selection at OmniVoice's
        # recommended default. Greedy position selection can produce unnatural,
        # repeated openings in cloned speech.
        config = config_class(
            num_step=self._steps,
            guidance_scale=self._guidance_scale,
            position_temperature=5.0,
            class_temperature=0.0,
        )
        # ``instruct`` is OmniVoice's *voice-design* control: it accepts only a closed
        # vocabulary of accent/age/style words and rejects free-form instructions.
        # Cloning takes the voice from ``voice_clone_prompt``, so it stays unset.
        request: dict[str, Any] = {
            "text": text,
            "language": OMNIVOICE_AMHARIC,
            "voice_clone_prompt": prompt,
            "generation_config": config,
        }
        if rate is not None:
            request["speed"] = rate

        try:
            produced = network.generate(**request)
        except Exception as exc:
            raise SynthesisError(
                f"OmniVoice failed to synthesize {text!r}: {type(exc).__name__}: {exc}"
            ) from exc

        samples = _first_waveform(produced, engine="OmniVoice", text=text)
        return _write_mono(target, samples, self.sample_rate, label="OmniVoice take")


def _first_waveform(produced: Any, *, engine: str, text: str) -> np.ndarray:
    """Return the first waveform of a batch result as a mono ``float32`` array.

    OmniVoice returns a list of arrays, one per input text. One line is requested at a
    time here, so anything other than a non-empty result is a bug worth naming rather
    than a shape to guess at.
    """

    if isinstance(produced, (list, tuple)):
        if not produced:
            raise SynthesisError(f"{engine} returned no audio for {text!r}")
        produced = produced[0]

    samples = _as_samples(produced).squeeze()
    if samples.size == 0:
        raise SynthesisError(f"{engine} returned an empty waveform for {text!r}")
    return samples


# ---------------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------------

#: Single-voice engines, keyed by the settings that shape one.
_MMS_ENGINES: dict[tuple[str, str, int, float], MmsAmharicEngine] = {}
#: Voice-cloning engines, keyed by the settings that shape one.
_OMNIVOICE_ENGINES: dict[tuple[str, str, int, float, str], OmniVoiceEngine] = {}


def load_mms_engine(*, settings: Settings | None = None) -> MmsAmharicEngine:
    """Return the process-wide MMS-TTS Amharic engine for these settings."""

    resolved = settings if settings is not None else get_settings()
    key = (
        resolved.device,
        resolved.tts_model,
        resolved.mms_seed,
        resolved.mms_speaking_rate,
    )
    engine = _MMS_ENGINES.get(key)
    if engine is None:
        engine = MmsAmharicEngine(
            model=resolved.tts_model,
            device=resolved.device,
            cache_dir=Path(resolved.model_cache_dir),
            seed=resolved.mms_seed,
            speaking_rate=resolved.mms_speaking_rate,
        )
        _MMS_ENGINES[key] = engine
    return engine


def load_omnivoice_engine(*, settings: Settings | None = None) -> OmniVoiceEngine:
    """Return the process-wide OmniVoice engine for these settings.

    The model's own downloads follow ``HF_HOME`` rather than ``MODEL_CACHE_DIR``: the
    OmniVoice loader resolves a repository id through ``snapshot_download`` and takes
    no cache directory of its own, so there is nothing here to point at the project's
    cache. A Pod that sets ``HF_HOME`` on the volume - as the runbook requires - keeps
    these weights there with everything else.
    """

    resolved = settings if settings is not None else get_settings()
    key = (
        resolved.device,
        resolved.omnivoice_model,
        resolved.omnivoice_steps,
        resolved.omnivoice_guidance_scale,
        resolved.omnivoice_torch_dtype,
    )
    engine = _OMNIVOICE_ENGINES.get(key)
    if engine is None:
        engine = OmniVoiceEngine(
            model=resolved.omnivoice_model,
            device=resolved.device,
            steps=resolved.omnivoice_steps,
            guidance_scale=resolved.omnivoice_guidance_scale,
            torch_dtype=resolved.omnivoice_torch_dtype,
        )
        _OMNIVOICE_ENGINES[key] = engine
    return engine


def load_tts_engine(*, settings: Settings | None = None) -> TextToSpeechEngine:
    """Return the single-voice engine named by ``TTS_ENGINE``.

    Only the engines that speak without a prompt or a conversion live here. The
    prompt-and-convert pair is resolved separately by
    :func:`synthesize_dialogue_detailed`, because it needs a voice profile per
    speaker that a single-voice engine has no use for.
    """

    resolved = settings if settings is not None else get_settings()
    if resolved.tts_engine == "mms":
        return load_mms_engine(settings=resolved)
    raise ConfigurationError(
        f"TTS_ENGINE={resolved.tts_engine!r} is not a single-voice engine; "
        "supported values here are 'mms'"
    )


#: Engines that speak in a voice taken from a per-speaker reference recording.
VOICE_CLONING_ENGINES: tuple[str, ...] = ("omnivoice",)

#: Engines that need a per-speaker voice reference: the cloning engines, which speak
#: *as* that voice. ``mms`` needs none - it has one voice for the whole film - so a run
#: using it skips the voice-profile stage instead of building references nothing would
#: read.
PROFILE_ENGINES: tuple[str, ...] = VOICE_CLONING_ENGINES

#: Every engine ``TTS_ENGINE`` may name.
SUPPORTED_ENGINES: tuple[str, ...] = ("mms",) + VOICE_CLONING_ENGINES

#: Loaders for the voice-cloning engines, by the ``TTS_ENGINE`` value selecting them.
VOICE_CLONING_LOADERS: dict[str, Callable[..., VoiceCloningEngine]] = {
    "omnivoice": load_omnivoice_engine,
}


def engine_needs_profiles(engine: str) -> bool:
    """``True`` when a run with this engine reads a per-speaker voice reference.

    The orchestrator asks this before deciding whether to build voice profiles: doing
    that work for an engine that cannot read them would spend minutes and a model
    download on references nothing uses.
    """

    return engine in PROFILE_ENGINES


def artifact_model_name(engine: str, *, settings: Settings | None = None) -> str:
    """Return the checkpoint id that shapes this engine's takes.

    Per-line artifacts are content-addressed from their text *and* this string, so
    switching engine cannot silently reuse another engine's audio for the same line.
    Naming the engine as well as the checkpoint is what distinguishes two engines that
    happen to share a repository id.
    """

    resolved = settings if settings is not None else get_settings()
    if engine == "omnivoice":
        return f"{engine}:{resolved.omnivoice_model}"
    return f"{engine}:{resolved.tts_model}"


def reset_engine_cache() -> None:
    """Drop every cached engine and loader, releasing the loaded models.

    An engine is useless once its model is gone, so the cache is cleared as a
    whole. The next call to a ``load_*_engine`` function starts from a clean slate.
    """

    _MMS_ENGINES.clear()
    _OMNIVOICE_ENGINES.clear()


# ---------------------------------------------------------------------------
# Stage layout
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TtsDirectories:
    """The artifact directories of one TTS run."""

    clips: Path
    takes: Path

    @classmethod
    def under(cls, base: Path) -> "_TtsDirectories":
        return cls(
            clips=base / CLIP_DIRECTORY_NAME,
            takes=base / TAKE_DIRECTORY_NAME,
        )

    def create(self) -> None:
        """Create every directory, creating parents as needed."""

        for directory in (self.clips, self.takes):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise InvalidInputError(f"could not create {directory}: {exc}") from exc


def resolve_tts_directory(
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Return the directory holding this stage's artifacts."""

    if output_dir is not None:
        return Path(output_dir)
    resolved = settings if settings is not None else get_settings()
    return Path(resolved.work_dir) / TTS_DIRECTORY_NAME


def _line_digest(
    dialogue: AdaptedDialogue, *, model: str, voice_identity: str | None = None
) -> str:
    """Return a stable digest of everything that shapes one line's audio."""

    payload = {
        "recipe": SYNTHESIS_RECIPE_VERSION,
        "model": model,
        "speaker_id": dialogue.speaker_id,
        "start": dialogue.start,
        "end": dialogue.end,
        "source_text": dialogue.source_text,
        "amharic": dialogue.amharic,
        "emotion": dialogue.emotion,
        "intensity": dialogue.intensity,
        "delivery": dialogue.delivery,
        "pause_before": dialogue.pause_before,
        "pause_after": dialogue.pause_after,
    }
    if voice_identity is not None:
        payload["voice_identity"] = voice_identity
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _line_artifact_name(
    dialogue: AdaptedDialogue, *, model: str, voice_identity: str | None = None
) -> str:
    """Return the deterministic, order-independent stem shared by a line's files."""

    digest = _line_digest(
        dialogue, model=model, voice_identity=voice_identity
    )[:16]
    return f"{speaker_directory_name(dialogue.speaker_id)}_{digest}"


def _voice_profile_identity(profile: VoiceProfile) -> str:
    """Hash the reference audio and transcript that shape cloned speech."""

    reference = profile.resolve_reference_audio()
    digest = hashlib.sha256()
    try:
        with reference.open("rb") as audio:
            for chunk in iter(lambda: audio.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise InvalidInputError(
            f"could not read voice reference for speaker {profile.speaker_id!r} "
            f"at {reference}: {exc}"
        ) from exc
    digest.update((profile.reference_text or "").encode("utf-8"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SkippedLine:
    """A dialogue line that was deliberately not synthesized, and why.

    Reported rather than silently dropped: a line missing from the dub is something
    a listener will notice, so the run has to be able to say which ones and why.
    """

    index: int
    speaker_id: str
    start: float
    end: float
    amharic: str
    reason: str

    @property
    def duration(self) -> float:
        """Length of the original window in seconds."""

        return self.end - self.start

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the skipped line."""

        return {
            "index": self.index,
            "speaker_id": self.speaker_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "amharic": self.amharic,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class SynthesisResult:
    """Every clip the stage produced, plus every line it could not.

    A film is thousands of lines and a paid run is hours long, so one line the engine
    cannot voice must not end the run: the failures are collected here and reported,
    and the stages after this one work with the clips that exist.
    """

    clips: tuple[TtsClip, ...]
    skipped: tuple[SkippedLine, ...] = ()
    #: The film-wide delivery plan these clips were spoken to. Recorded so a run can say
    #: what rate it asked for rather than leaving it to be inferred from the audio.
    pacing: PacingPlan | None = None
    #: Anything the stage wants the run to know without treating it as a failure: lines
    #: voiced despite a window shorter than the engine minimum, for instance.
    notes: tuple[str, ...] = ()

    @property
    def attempted(self) -> int:
        """How many lines were considered."""

        return len(self.clips) + len(self.skipped)

    @property
    def failed(self) -> tuple[SkippedLine, ...]:
        """Lines skipped because an engine failed, rather than as unusable input."""

        return tuple(
            line for line in self.skipped if line.reason.startswith(FAILURE_REASON_PREFIX)
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the result."""

        return {
            "clips": len(self.clips),
            "skipped": len(self.skipped),
            "attempted": self.attempted,
            "failed": len(self.failed),
            "skipped_lines": [line.as_dict() for line in self.skipped],
            "notes": list(self.notes),
        }


#: Prefix of the reason recorded for a line an engine failed on, so a report can tell
#: a broken line apart from an unusable one.
FAILURE_REASON_PREFIX = "synthesis failed"
#: Prefix of the reason recorded for a line that cannot be dubbed at all.
UNSPEAKABLE_REASON_PREFIX = "cannot be dubbed"


def _skip_reason(line: AdaptedDialogue) -> str | None:
    """Return why ``line`` cannot be dubbed at all, or ``None`` when it can.

    The only input that rules a line out is having nothing to pronounce. A short
    original window does *not* rule a line out, and used to: four real lines of the
    test film - "Walt.", "Thank you.", "We're done." and "No." - were dropped from the
    dub because the English actor said them in a tenth of a second, leaving a hole in
    the dialogue where the audience expects an answer.

    A brief English line is normal rather than a defect. Amharic needs more time than
    English to say the same thing, so the honest response is to say the line and let
    :mod:`app.pipeline.timing` move the lines that follow, not to leave the line out.
    """

    if not has_pronounceable_text(line.amharic):
        return (
            f"{UNSPEAKABLE_REASON_PREFIX}: the Amharic line has nothing to "
            "pronounce"
        )
    return None


def _validate_dialogue(dialogue: Iterable[AdaptedDialogue]) -> list[AdaptedDialogue]:
    """Return the dialogue as a chronological list of valid lines."""

    try:
        lines = list(dialogue)
    except TypeError as exc:
        raise InvalidDialogueError(
            f"dialogue must be an iterable of AdaptedDialogue lines: {exc}"
        ) from exc

    for index, line in enumerate(lines):
        if not isinstance(line, AdaptedDialogue):
            raise InvalidDialogueError(
                f"dialogue[{index}] is {type(line).__name__}, expected an AdaptedDialogue"
            )

    starts = [line.start for line in lines]
    if any(later < earlier for earlier, later in zip(starts, starts[1:])):
        raise InvalidDialogueError(
            "dialogue lines must be in chronological order; got starts "
            + ", ".join(f"{value:g}" for value in starts)
        )
    return lines


def _validate_profiles(
    voice_profiles: Mapping[str, VoiceProfile],
    *,
    speakers: Iterable[str],
) -> dict[str, VoiceProfile]:
    """Return the profiles as a validated ``{speaker_id: VoiceProfile}`` mapping."""

    if not isinstance(voice_profiles, Mapping):
        raise InvalidVoiceProfileError(
            f"voice_profiles must be a mapping, got {type(voice_profiles).__name__}"
        )

    profiles: dict[str, VoiceProfile] = {}
    for key, profile in voice_profiles.items():
        if not isinstance(profile, VoiceProfile):
            raise InvalidVoiceProfileError(
                f"voice profile {key!r} is {type(profile).__name__}, expected a VoiceProfile"
            )
        if key != profile.speaker_id:
            raise InvalidVoiceProfileError(
                f"voice profile {profile.speaker_id!r} is stored under the key {key!r}"
            )
        profiles[profile.speaker_id] = profile

    for speaker_id in dict.fromkeys(speakers):
        if speaker_id not in profiles:
            raise MissingVoiceProfileError(
                f"speaker {speaker_id!r} has no voice profile; build one with "
                "app.pipeline.voice_profiles.build_voice_profiles first"
            )
    return profiles


def _require_output(path: Path, *, what: str, engine: str) -> Path:
    """Return ``path`` if the engine actually wrote audio there."""

    if not _usable_audio(path):
        raise MissingOutputError(
            f"{engine} reported success but wrote no {what} to {path}"
        )
    return path


def _render_clip(
    source: Path,
    destination: Path,
    *,
    pause_before: float,
    pause_after: float,
    max_pause: float,
) -> tuple[float, float, float]:
    """Write the delivered clip: the take with its pauses around it.

    Returns ``(speech_duration, rendered_pause_before, rendered_pause_after)``.
    The pauses are capped at ``max_pause``; the dialogue's own estimate is kept
    untouched in the clip metadata, so a cap never hides what was asked for.
    """

    samples, sample_rate = _read_mono(source, label="the take")
    speech_duration = samples.shape[0] / sample_rate

    lead = min(max(pause_before, 0.0), max_pause)
    trail = min(max(pause_after, 0.0), max_pause)

    if lead or trail:
        pieces = [samples]
        if lead:
            pieces.insert(0, np.zeros(int(round(lead * sample_rate)), dtype=np.float32))
        if trail:
            pieces.append(np.zeros(int(round(trail * sample_rate)), dtype=np.float32))
        samples = np.concatenate(pieces)

    _write_mono(destination, samples, sample_rate, label="dubbed clip")
    return speech_duration, lead, trail


def _synthesize_line_single_voice(
    index: int,
    dialogue: AdaptedDialogue,
    *,
    name: str,
    directories: _TtsDirectories,
    engine: TextToSpeechEngine,
    settings: Settings,
) -> TtsClip:
    """Run a single-voice engine for one line and describe the result.

    No performance prompt is cut from the stem and no identity conversion happens,
    because this kind of engine accepts neither: it speaks Amharic in one voice. The
    pauses are still rendered, and the clip is still the same
    :class:`TtsClip` the rest of the pipeline consumes, so timing and mixing do not
    need to know which engine produced it.
    """

    take = directories.takes / f"{name}.wav"
    if not _usable_audio(take):
        engine.synthesize(text=dialogue.amharic, destination=take)
        _require_output(take, what="take", engine=engine.name)

    clip_path = directories.clips / f"{name}.wav"
    speech_duration, lead, trail = _render_clip(
        take,
        clip_path,
        pause_before=dialogue.pause_before,
        pause_after=dialogue.pause_after,
        max_pause=settings.tts_max_pause_seconds,
    )

    _, clip_rate = _read_audio_info(clip_path, label="the dubbed clip")
    return TtsClip(
        index=index,
        dialogue=dialogue,
        performance=PerformanceControls.from_dialogue(
            dialogue, seed=int(name[:8], 16) if _hex_prefix(name) else 0
        ),
        audio_path=clip_path,
        take_path=take,
        performance_reference_path=None,
        voice_reference_path=None,
        sample_rate=clip_rate,
        speech_duration=speech_duration,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
        performance_engine=engine.name,
        style_engine="none",
    )


def requested_speaking_rate(
    dialogue: AdaptedDialogue,
    *,
    plan: PacingPlan,
) -> float | None:
    """Return the rate to ask an engine for, or ``None`` to leave the model alone.

    Asking a model for a duration is materially better than stretching its output
    afterwards: the rate changes how the line is *spoken*, while a time-stretch changes
    the audio that was already spoken.

    The rate comes from the film's :class:`~app.pipeline.dialogue_context.PacingPlan`
    rather than from this line alone - see that class for why two levels are used instead
    of one. The estimate behind it is deliberately coarse, since it only has to move the
    line near its window before :mod:`app.pipeline.timing` fits the residual.
    """

    return plan.rate_for(
        seconds=dialogue.end - dialogue.start,
        syllables=count_syllables(dialogue.amharic),
    )


def pacing_plan_for(
    lines: Iterable[AdaptedDialogue], *, settings: Settings
) -> PacingPlan:
    """Measure a whole film's dialogue and return the delivery plan for it.

    Called once per run, before any line is synthesized, so every line is spoken to the
    same film-wide policy instead of each one negotiating its own.
    """

    return plan_pacing(
        (
            (max(line.end - line.start, 0.0), count_syllables(line.amharic))
            for line in lines
        ),
        minimum=settings.timing_min_tempo,
        maximum=settings.timing_max_tempo,
    )


def _synthesize_line_cloned(
    index: int,
    dialogue: AdaptedDialogue,
    *,
    name: str,
    directories: _TtsDirectories,
    engine: VoiceCloningEngine,
    profile: VoiceProfile,
    settings: Settings,
    pacing: PacingPlan,
) -> TtsClip:
    """Speak one line as its character, cloning the voice from their reference.

    The character's reference recording *is* the voice: the same reference for every
    line of a speaker is what keeps a character recognisable across a film, which is
    the property the performance-prompt-and-convert path approximates from the other
    direction.

    ``VoiceProfile.reference_text`` is passed along when it is known, so the engine does
    not have to transcribe a clip the project already has a transcript for - which is
    both a saved model download and a better prompt.
    """

    take = directories.takes / f"{name}.wav"
    if not _usable_audio(take):
        request: dict[str, Any] = {
            "text": dialogue.amharic,
            "voice_reference": profile.resolve_reference_audio(),
            "destination": take,
        }
        reference_text = (profile.reference_text or "").strip()
        if reference_text:
            request["reference_text"] = reference_text

        # Ask for a length only when the engine can honour one: an engine that cannot
        # would either ignore the request or fail on it, and the timing stage still
        # fits whatever comes back. See :data:`DEFAULT_TTS_REQUEST_RATE`.
        if settings.tts_request_rate and engine.supports_speaking_rate:
            rate = requested_speaking_rate(dialogue, plan=pacing)
            if rate is not None:
                request["speaking_rate"] = rate

        engine.synthesize(**request)
        _require_output(take, what="take", engine=engine.name)

    clip_path = directories.clips / f"{name}.wav"
    speech_duration, lead, trail = _render_clip(
        take,
        clip_path,
        pause_before=dialogue.pause_before,
        pause_after=dialogue.pause_after,
        max_pause=settings.tts_max_pause_seconds,
    )

    _, clip_rate = _read_audio_info(clip_path, label="the dubbed clip")
    return TtsClip(
        index=index,
        dialogue=dialogue,
        performance=PerformanceControls.from_dialogue(
            dialogue, seed=int(name[:8], 16) if _hex_prefix(name) else 0
        ),
        audio_path=clip_path,
        take_path=take,
        performance_reference_path=None,
        voice_reference_path=profile.resolve_reference_audio(),
        sample_rate=clip_rate,
        speech_duration=speech_duration,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
        performance_engine=engine.name,
        style_engine="none",
    )


def _hex_prefix(name: str) -> bool:
    """``True`` when ``name`` starts with at least eight hex characters."""

    head = name[:8]
    return len(head) == 8 and all(c in "0123456789abcdefABCDEF" for c in head)


def synthesize_dialogue_detailed(
    dialogue: Iterable[AdaptedDialogue],
    speech_stem: str | Path | None = None,
    voice_profiles: Mapping[str, VoiceProfile] | None = None,
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    tts_engine: TextToSpeechEngine | VoiceCloningEngine | None = None,
) -> SynthesisResult:
    """Synthesize every dubbable line, and report the ones that could not be.

    Parameters
    ----------
    dialogue:
        Lines from :func:`app.pipeline.translation.adapt_dialogue`, in
        chronological order. Their performance metadata is carried into the clip,
        never dropped.
    speech_stem:
        Accepted and ignored. It used to be the clean ``speech`` stem the
        performance prompt was cut from, and no supported engine reads a prompt any
        more; the parameter stays so an existing caller does not have to be edited
        to keep working. Voice references come from ``voice_profiles``.
    voice_profiles:
        Profiles keyed by speaker id, as returned by
        :func:`app.pipeline.voice_profiles.build_voice_profiles` and persisted by
        :func:`~app.pipeline.voice_profiles.save_voice_profiles`. Only
        ``reference_audio`` and ``reference_text`` are read, and only by a cloning
        engine; a single-voice engine needs no profiles at all.
    output_dir:
        Directory for this stage's artifacts. Defaults to ``WORK_DIR/tts``;
        ``takes`` and ``clips`` sub-directories are created below it.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
        ``DEVICE`` reaches the engine from here.
    tts_engine:
        An engine adapter to use instead of the configured one, for a test or a
        controlled comparison. What it *is* decides how it is driven: a
        :class:`VoiceCloningEngine` is passed a speaker's reference, a
        :class:`TextToSpeechEngine` is not.

    Returns
    -------
    list[TtsClip]
        One clip per input line, in the same order. Audio for each line is written
        under the stage directory, at a path derived from the line's content, so
        re-running the stage reuses the take instead of redoing it. Durations are
        reported, not fitted: fitting each line into its original window is
        :mod:`app.pipeline.timing`'s job.

    Raises
    ------
    InvalidDialogueError, InvalidVoiceProfileError, InvalidInputError
        The supplied lines, profiles or paths are not usable.
    MissingVoiceProfileError
        A speaker in the dialogue has no profile.
    MissingInputError, InvalidAudioError
        A profile's reference audio is missing or unreadable.
    ConfigurationError
        ``TTS_ENGINE`` names no supported engine.
    InvalidEngineError
        An injected engine is not the expected kind of adapter.
    EngineLoadError, SynthesisError, MissingOutputError
        An engine could not be loaded, or failed in a way that is not specific to
        one line. A failure on a *single* line is reported in the result's
        ``skipped`` rather than raised, so one unvoiceable line cannot end a run.
    """

    lines = _validate_dialogue(dialogue)
    if not lines:
        return SynthesisResult(clips=())

    resolved = settings if settings is not None else get_settings()

    # Two ways to speak a line, and which one is in use decides whether the
    # voice-profile stage has anything to do:
    #   omnivoice   clone the character's own voice and speak as it
    #   mms         one voice for the whole film
    # The first reads a per-speaker reference; the second has none to read.
    cloning_engine: VoiceCloningEngine | None = None
    single_voice = False
    if tts_engine is not None:
        # An engine handed in directly is classified by what it *is* rather than by
        # the configured name: a caller that passes one is stating which contract it
        # meets, and the two contracts need different stages.
        if isinstance(tts_engine, VoiceCloningEngine):
            cloning_engine = tts_engine
        elif isinstance(tts_engine, TextToSpeechEngine):
            single_voice = True
        else:
            raise InvalidEngineError(
                "tts_engine must implement VoiceCloningEngine or TextToSpeechEngine, "
                f"got {type(tts_engine).__name__}"
            )
    elif resolved.tts_engine in VOICE_CLONING_LOADERS:
        cloning_engine = VOICE_CLONING_LOADERS[resolved.tts_engine](settings=resolved)
    elif resolved.tts_engine == "mms":
        single_voice = True
    else:
        raise ConfigurationError(
            f"TTS_ENGINE must be one of {', '.join(SUPPORTED_ENGINES)}, got "
            f"{resolved.tts_engine!r}"
        )

    if single_voice:
        # A single-voice engine has no use for voice profiles, and requiring them
        # would mean building identities the engine cannot honour.
        profiles: Mapping[str, VoiceProfile] = {}
        voice_identities: Mapping[str, str] = {}
    else:
        profiles = _validate_profiles(
            voice_profiles, speakers=[line.speaker_id for line in lines]
        )

        # Validate the reference audio of every speaker that is about to be used
        # before any model runs: a missing reference should fail the run the same way
        # whether or not earlier lines could already be synthesized.
        identities: dict[str, str] = {}
        for speaker_id in dict.fromkeys(line.speaker_id for line in lines):
            profile = profiles[speaker_id]
            reference = profile.resolve_reference_audio()
            _validate_audio_file(
                reference, label=f"the voice reference of speaker {speaker_id!r}"
            )
            identities[speaker_id] = _voice_profile_identity(profile)
        voice_identities = identities

    _positive("TTS_MAX_PAUSE_SECONDS", resolved.tts_max_pause_seconds)
    min_line_seconds = _positive(
        "TTS_MIN_LINE_SECONDS", resolved.tts_min_line_seconds
    )
    continue_on_failure = bool(resolved.tts_continue_on_failure)

    if cloning_engine is not None:
        engine = None
    elif tts_engine is not None:
        engine = tts_engine
    elif single_voice:
        engine = load_tts_engine(settings=resolved)
    else:
        engine = None
    if engine is not None and not isinstance(engine, TextToSpeechEngine):
        raise InvalidEngineError(
            f"tts_engine must be a TextToSpeechEngine adapter, got "
            f"{type(engine).__name__}"
        )

    if cloning_engine is not None:
        artifact_model = f"{cloning_engine.name}:{cloning_engine.model}"
    else:
        artifact_model = f"{engine.name}:{resolved.tts_model}"

    # Measured over the whole film, once, before any line is spoken: the film-wide rate
    # is what absorbs the systematic difference between Amharic and the English it
    # replaces, so every line is delivered to one policy rather than negotiating its own.
    pacing = pacing_plan_for(lines, settings=resolved)

    directories = _TtsDirectories.under(
        resolve_tts_directory(output_dir=output_dir, settings=resolved)
    )
    directories.create()

    clips: list[TtsClip] = []
    skipped: list[SkippedLine] = []
    short_windows: list[int] = []
    for index, line in enumerate(lines):
        reason = _skip_reason(line)
        if reason is not None:
            skipped.append(
                SkippedLine(
                    index=index,
                    speaker_id=line.speaker_id,
                    start=line.start,
                    end=line.end,
                    amharic=line.amharic,
                    reason=reason,
                )
            )
            continue

        if line.duration < min_line_seconds:
            short_windows.append(index)

        try:
            if engine is not None:
                clips.append(
                    _synthesize_line_single_voice(
                        index,
                        line,
                        name=_line_artifact_name(line, model=artifact_model),
                        directories=directories,
                        engine=engine,
                        settings=resolved,
                    )
                )
                continue

            clips.append(
                _synthesize_line_cloned(
                    index,
                    line,
                    name=_line_artifact_name(
                        line,
                        model=artifact_model,
                        voice_identity=voice_identities[line.speaker_id],
                    ),
                    directories=directories,
                    engine=cloning_engine,
                    profile=profiles[line.speaker_id],
                    settings=resolved,
                    pacing=pacing,
                )
            )
        except TtsError as exc:
            if not continue_on_failure:
                # The line is identified so the failure can be found and the input
                # fixed. A film's run is long, so a caller that would rather ship a
                # dub with a reported hole can set TTS_CONTINUE_ON_FAILURE.
                raise type(exc)(
                    f"line {index} of speaker {line.speaker_id!r} "
                    f"({line.start:.3f}s-{line.end:.3f}s): {exc}"
                ) from exc
            skipped.append(
                SkippedLine(
                    index=index,
                    speaker_id=line.speaker_id,
                    start=line.start,
                    end=line.end,
                    amharic=line.amharic,
                    reason=f"{FAILURE_REASON_PREFIX}: {type(exc).__name__}: {exc}",
                )
            )

    notes: list[str] = []
    if short_windows:
        shown = ", ".join(str(index) for index in short_windows[:10])
        if len(short_windows) > 10:
            shown += f", and {len(short_windows) - 10} more"
        notes.append(
            f"{len(short_windows)} line(s) have an original window shorter than the "
            f"{min_line_seconds:g}s engine minimum (line(s) {shown}); they were "
            "synthesized and the timing stage moves the lines after them, because "
            "dropping them would leave real dialogue unsaid"
        )

    return SynthesisResult(
        clips=tuple(clips), skipped=tuple(skipped), pacing=pacing, notes=tuple(notes)
    )


def synthesize_dialogue(
    dialogue: Iterable[AdaptedDialogue],
    speech_stem: str | Path | None = None,
    voice_profiles: Mapping[str, VoiceProfile] | None = None,
    *,
    settings: Settings | None = None,
    output_dir: str | Path | None = None,
) -> list[TtsClip]:
    """Synthesize every dubbable line and return the clips.

    The clips alone, in the order the lines were given. Lines the stage could not
    voice - text with nothing to pronounce, or a line an engine failed on - are absent
    rather than raising; call :func:`synthesize_dialogue_detailed` when the reason for
    each one is wanted, which is what the orchestrator does so the manifest can record
    them. A short original window is not a reason to leave a line out, so those lines
    are voiced.

    ``speech_stem`` is accepted and ignored - see
    :func:`synthesize_dialogue_detailed`.
    """

    return list(
        synthesize_dialogue_detailed(
            dialogue,
            speech_stem,
            voice_profiles,
            settings=settings,
            output_dir=output_dir,
        ).clips
    )


__all__ = [
    "CLIP_CEILING",
    "MINIMUM_REPORTABLE_LINE_SECONDS",
    "MINIMUM_SPEAKABLE_SYLLABLES",
    "PerformanceControls",
    "SYNTHESIS_RECIPE_VERSION",
    "TTS_DIRECTORY_NAME",
    "ConfigurationError",
    "EngineLoadError",
    "InvalidAudioError",
    "InvalidDialogueError",
    "InvalidEngineError",
    "InvalidInputError",
    "InvalidVoiceProfileError",
    "MissingInputError",
    "MissingOutputError",
    "MissingVoiceProfileError",
    "MmsAmharicEngine",
    "OmniVoiceEngine",
    "VoiceCloningEngine",
    "SkippedLine",
    "SynthesisError",
    "SynthesisResult",
    "TtsClip",
    "TtsError",
    "TextToSpeechEngine",
    "engine_needs_profiles",
    "artifact_model_name",
    "load_mms_engine",
    "load_omnivoice_engine",
    "load_tts_engine",
    "SUPPORTED_ENGINES",
    "PROFILE_ENGINES",
    "pacing_plan_for",
    "requested_speaking_rate",
    "reset_engine_cache",
    "resolve_tts_directory",
    "synthesize_dialogue",
    "synthesize_dialogue_detailed",
]
