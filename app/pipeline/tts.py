"""Amharic speech synthesis with character voice adaptation.

This module is the pipeline's TTS stage. It turns the adapted dialogue of
:mod:`app.pipeline.translation` into audio that is spoken in each character's
cloned voice, using the clean dialogue stem of
:mod:`app.pipeline.separation` as the performance reference::

    [AdaptedDialogue] + speech.wav + {speaker: VoiceProfile}
        -> synthesize_dialogue() -> [TtsClip]

Architecture (locked)
---------------------
Two engines run in sequence for every line, each behind its own adapter:

1. :class:`ChatterboxAmharicEngine` - ``gabar-tech/chatterbox-amharic`` speaks
   the Amharic text. It is a LoRA adapter plus a Fidel tokenizer on top of
   Chatterbox Multilingual v3, loaded through the loader that ships with the
   adapter repository (``amharic_tts.py`` -> ``load_amharic_tts``), so training
   and inference see the same text front-end.
2. :class:`SeedVcV2Engine` - Seed-VC V2 converts that take into the character's
   identity. It runs in its **timbre-only** mode (``convert_style=False``), which
   replaces the voice without touching the delivery. The style-converting mode
   must not be used here: Seed-VC V2's style branch conditions its autoregressive
   stage on the *reference's* acoustic tokens and content indices, so it speaks the
   source content in the reference's accent and emotion. The character reference is
   the original actor's English audio, so switching that mode on would re-impose an
   English accent on the Amharic and overwrite the very performance the Chatterbox
   prompt carried over.

Why each reference is what it is
--------------------------------
Chatterbox conditions *both* timbre and prosody on its prompt audio, so the
prompt decides how the line is performed. The performance reference is therefore
the original actor's own audio for that line, cut from the clean BandIt
``speech`` stem (see :func:`extract_performance_reference`): the original
delivery, emotion and pauses are carried into the Amharic take instead of being
invented by the model. Music and effects are absent from that stem, so the prompt
contains speech only.

``VoiceProfile.reference_audio`` is used for one thing only: as the Seed-VC
target identity reference, which is what makes a character sound like themselves
across a whole film. It is never handed to Chatterbox, and a
:class:`~app.pipeline.voice_profiles.VoiceProfile` is never modified here - this
module only reads it. Because the Amharic adapter takes reference audio directly,
no clone-prompt serialization happens at this stage, so
``VoiceProfile.clone_prompt_path`` stays untouched.

Performance metadata
--------------------
:class:`~app.pipeline.translation.AdaptedDialogue` carries emotion, intensity,
delivery and the pauses around the line. None of it is discarded:

* ``emotion``, ``delivery`` and ``intensity`` are mapped - conservatively and
  within narrow bounds - onto the controls Chatterbox actually exposes
  (``exaggeration``, ``cfg_weight``, ``temperature``), and every decision is
  recorded in :class:`PerformanceControls` so the mapping is auditable. An
  emotion or delivery direction the tables do not know changes nothing and says
  so, rather than being invented.
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
* **No resampling.** The performance reference keeps the stem's own sample rate
  (Chatterbox's loader resamples the prompt to 16 kHz itself) and the returned
  clip keeps the rate Seed-VC produced; bringing the whole dub to one rate is the
  mixing stage's job.
* **No model downloads at import time** (and none in the test suite). Both
  engines are lazy: nothing is imported, downloaded or loaded until a clip is
  actually synthesized, and a loaded model is cached and reused for every
  following line instead of being loaded once per line. The Amharic adapter,
  its pinned Chatterbox base and the loader file are snapshotted into
  ``MODEL_CACHE_DIR``; Seed-VC's own checkpoints, vocoder and feature extractors
  are fetched by its code, which takes no cache argument, so they follow
  ``HF_HOME`` - see ``.env.example``.
* **No Docker, no ffmpeg.** Cutting the performance reference uses ``soundfile``
  seeking, so this stage adds no external process.

A line that cannot be dubbed at all - a window too short to hold a word, or text
with nothing to pronounce - is skipped before either engine is called and reported,
so a fragment cannot end a run over a film. An engine that genuinely *fails* on a
line still stops the run with that line named, unless
``TTS_CONTINUE_ON_FAILURE=true`` asks for a reported hole instead.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import inspect
import json
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import (
    DEFAULT_CHATTERBOX_MODEL,
    DEFAULT_MMS_SAMPLE_RATE,
    DEFAULT_MMS_SEED,
    DEFAULT_MMS_SPEAKING_RATE,
    DEFAULT_SEED_VC_CONVERT_STYLE,
    DEFAULT_SEED_VC_DIFFUSION_STEPS,
    DEFAULT_TTS_ENGINE,
    DEFAULT_TTS_MIN_LINE_SECONDS,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION,
    DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION,
    Settings,
    get_settings,
)
from app.pipeline.amharic_text import count_syllables, has_pronounceable_text
from app.pipeline.dialogue_context import (
    DEFAULT_SYLLABLES_PER_SECOND,
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

#: Sub-directory holding the original-performance prompts cut from the stem.
PERFORMANCE_DIRECTORY_NAME = "performance"

#: Sub-directory holding the raw Chatterbox takes, before identity conversion.
CHATTERBOX_DIRECTORY_NAME = "chatterbox"

#: Sub-directory holding the Seed-VC output, still without pauses.
CONVERTED_DIRECTORY_NAME = "converted"

#: Sub-directory holding the clips handed to :mod:`app.pipeline.timing`.
#: Sub-directory holding the takes a single-voice engine produced.
TAKE_DIRECTORY_NAME = "takes"

#: Sub-directory holding the aligned, pause-rendered clips every engine ends with.
CLIP_DIRECTORY_NAME = "clips"

#: Bump when a change here would make an existing take or clip wrong; it is part
#: of every generated file name, so old artifacts can never be mistaken for new
#: ones.
SYNTHESIS_RECIPE_VERSION = 1


# ---------------------------------------------------------------------------
# Chatterbox Amharic
# ---------------------------------------------------------------------------

#: Loader module that ships inside the adapter repository.
AMHARIC_LOADER_FILENAME = "amharic_tts.py"

#: Name the loader module is registered under.
AMHARIC_LOADER_MODULE = "amharic_tts"

#: Chatterbox's S3Gen output rate. Known up front so a clip's rate never has to
#: be guessed before the model is loaded.
CHATTERBOX_SAMPLE_RATE = 24_000

#: The adapter's own defaults, which is where every mapping starts.
BASE_EXAGGERATION = 0.5
BASE_CFG_WEIGHT = 0.5
BASE_TEMPERATURE = 0.6

#: Hard bounds for the mapped controls. The adapter was validated at its
#: defaults, so a performance direction may only nudge them, never move them out
#: of a band around those defaults.
EXAGGERATION_BOUNDS = (0.30, 0.70)
CFG_WEIGHT_BOUNDS = (0.35, 0.65)
TEMPERATURE_BOUNDS = (0.40, 0.80)

#: How far ``intensity`` alone may move a control, between intensity 0.0 and 1.0.
INTENSITY_EXAGGERATION_SWING = 0.10
INTENSITY_TEMPERATURE_SWING = 0.10


# ---------------------------------------------------------------------------
# Seed-VC V2
# ---------------------------------------------------------------------------

#: Name reported by the identity-conversion engine.
SEED_VC_ENGINE_NAME = "seed-vc-v2"

#: Configuration that describes Seed-VC V2, relative to its checkout.
SEED_VC_CONFIG_PARTS = ("configs", "v2", "vc_wrapper.yaml")

#: Shortest original window that can be dubbed, in seconds. Below this the original
#: is a fragment rather than a spoken line - there is no room for a word in the time
#: it occupied - and the engine can fail outright on it: Chatterbox does, with an
#: empty mel spectrogram that trips a convolution inside its vocoder.
MINIMUM_SPEAKABLE_LINE_SECONDS = DEFAULT_TTS_MIN_LINE_SECONDS

#: Shortest Amharic line worth synthesizing, in syllables. One Fidel character is one
#: syllable, so this counts the script's own unit. Latin text counts as pronounceable
#: too - an English-derived word is a word - so a borrowed word is never dropped for
#: being written in Roman script; see :func:`app.pipeline.amharic_text.has_latin`.
MINIMUM_SPEAKABLE_SYLLABLES = 1

#: Syllables per second an Amharic performer delivers, used to estimate how long a line
#: would naturally take before a rate is requested of the engine. Measured at ~4.6
#: syllables/second on real synthesized output; the value here is a little slower on
#: purpose, because asking for a line that comes out too fast is worse than one that
#: comes out slightly short and is then fitted. The film-wide part of the pacing policy
#: lives in :mod:`app.pipeline.dialogue_context`, which owns how long a line should be.
RATE_ESTIMATE_SYLLABLES_PER_SECOND = DEFAULT_SYLLABLES_PER_SECOND

#: Seed-VC V2 runs in timbre-only mode by default: it replaces the character's
#: voice and leaves the take's delivery alone. The flag means "also convert the
#: *reference's* accent and style", which is the opposite of what this stage needs,
#: so it is exposed as :data:`SEED_VC_CONVERT_STYLE` - overridable through settings
#: for a single controlled comparison - rather than hard-coded.
SEED_VC_CONVERT_STYLE = DEFAULT_SEED_VC_CONVERT_STYLE

#: The V2 wrapper caches the autoregressive model for one sequence at a time,
#: which is how the repository's own inference script sets it up.
SEED_VC_AR_CACHE_MAX_BATCH_SIZE = 1
SEED_VC_AR_CACHE_MAX_SEQ_LEN = 4096


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
    """Chatterbox or Seed-VC could not be loaded on the configured device."""


class SynthesisError(TtsError):
    """Chatterbox failed to synthesize a line."""


class ConversionError(TtsError):
    """Seed-VC failed to convert a take into the character's voice."""


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

    Chatterbox hands back a ``[1, N]`` ``torch.Tensor`` and Seed-VC a NumPy
    array; both are accepted without importing either runtime here.
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
# Performance mapping
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Cue:
    """A bounded nudge applied to the Chatterbox controls when a cue matches."""

    terms: tuple[str, ...]
    exaggeration: float = 0.0
    cfg_weight: float = 0.0
    temperature: float = 0.0
    note: str = ""


#: Emotion directions, matched by substring against ``AdaptedDialogue.emotion``.
#: The translation stage is free to return any wording - its list of emotions is
#: explicitly a set of examples - so the table is matched in order, the first hit
#: wins, and an unknown emotion changes nothing.
_EMOTION_CUES: tuple[_Cue, ...] = (
    _Cue(
        ("angry", "anger", "furious", "rage", "irate", "outrag", "hostile", "aggress"),
        exaggeration=0.12,
        temperature=0.05,
    ),
    _Cue(
        ("excited", "excitement", "happy", "joy", "deligh", "cheer", "elat", "triumph"),
        exaggeration=0.08,
        temperature=0.05,
    ),
    _Cue(
        ("afraid", "fear", "terrif", "panic", "anxious", "nervous", "worried", "tense"),
        exaggeration=0.05,
        temperature=0.05,
    ),
    _Cue(("surpris", "startled", "shock", "astonish"), exaggeration=0.06),
    _Cue(
        ("sarcast", "playful", "amused", "teasing", "mischiev", "smug"),
        exaggeration=0.04,
    ),
    _Cue(
        ("sad", "sorrow", "grief", "mourn", "weep", "cry", "disappoint", "hurt", "heartbrok"),
        exaggeration=-0.08,
        temperature=-0.05,
    ),
    _Cue(
        ("embarrass", "ashamed", "vulnerab", "guilt", "timid", "sheepish"),
        exaggeration=-0.06,
    ),
    _Cue(("romantic", "tender", "loving", "warm", "affection", "soothing"), exaggeration=-0.04),
    _Cue(
        ("neutral", "calm", "composed", "flat", "serene", "matter-of-fact", "resigned"),
        exaggeration=-0.04,
    ),
)

#: Delivery directions, matched the same way against
#: ``AdaptedDialogue.delivery``. ``cfg_weight`` is the control that decides how
#: strongly the prompt's own delivery is followed, which is exactly what a
#: delivery direction talks about. Singing comes first because it is the one
#: direction the available controls cannot express; it matches before any
#: descriptive word in the same sentence.
_DELIVERY_CUES: tuple[_Cue, ...] = (
    _Cue(
        ("sing", "singing", "song", "chant", "melod"),
        note=(
            "singing needs pitch conditioning, which Seed-VC V2 runs without in "
            "this pipeline, so this direction changed no control"
        ),
    ),
    _Cue(
        ("whisper", "murmur", "hushed", "under his breath", "under her breath"),
        exaggeration=-0.08,
        cfg_weight=0.05,
    ),
    _Cue(
        ("shout", "yell", "scream", "bellow", "holler", "at the top"),
        exaggeration=0.08,
        cfg_weight=0.05,
    ),
    _Cue(
        ("hesitant", "nervous", "trembl", "stammer", "falter", "unsure", "uncertain"),
        exaggeration=-0.04,
        temperature=0.05,
    ),
    _Cue(("deadpan", "monotone", "robotic", "mechanical"), exaggeration=-0.08, cfg_weight=-0.05),
    _Cue(("warm", "gentle", "tender", "soothing", "soft"), exaggeration=-0.04),
    _Cue(("sarcast", "dry", "wry"), exaggeration=0.04, cfg_weight=0.05),
)

def _match_cue(text: str, cues: tuple[_Cue, ...]) -> tuple[_Cue | None, str | None]:
    """Return the first cue whose terminology appears in ``text`` at a word start.

    A cue term only matches where a word begins, so "singing" is singing while
    the "sing" inside "musing to himself" is not. Cues are tried in order and the
    first hit wins; nothing is guessed when no term matches.
    """

    lowered = text.casefold()
    for cue in cues:
        for term in cue.terms:
            start = 0
            while True:
                index = lowered.find(term, start)
                if index < 0:
                    break
                if index == 0 or not lowered[index - 1].isalnum():
                    return cue, term
                start = index + 1
    return None, None


def _clamp(
    value: float,
    bounds: tuple[float, float],
    *,
    name: str,
    notes: list[str],
) -> float:
    """Clamp ``value`` into ``bounds``, recording when it had to be clamped."""

    low, high = bounds
    if value < low:
        notes.append(f"{name} {value:.3f} clamped up to {low:.3f}")
        return low
    if value > high:
        notes.append(f"{name} {value:.3f} clamped down to {high:.3f}")
        return high
    return value


@dataclass(frozen=True, slots=True)
class PerformanceControls:
    """One line's performance direction, as Chatterbox can accept it.

    The original fields are carried through untouched (``emotion``,
    ``intensity``, ``delivery``, ``pause_before``, ``pause_after``) next to the
    bounded controls they were mapped onto, the matched cue terms and the notes
    that explain every adjustment. The mapping is a pure function of the line, so
    the same line always produces the same performance.
    """

    emotion: str
    intensity: float
    delivery: str
    pause_before: float
    pause_after: float
    exaggeration: float
    cfg_weight: float
    temperature: float
    seed: int
    matched_emotion_cue: str | None = None
    matched_delivery_cue: str | None = None
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

        for name, value, bounds in (
            ("exaggeration", self.exaggeration, EXAGGERATION_BOUNDS),
            ("cfg_weight", self.cfg_weight, CFG_WEIGHT_BOUNDS),
            ("temperature", self.temperature, TEMPERATURE_BOUNDS),
        ):
            control = _finite_seconds(name, value)
            if not bounds[0] <= control <= bounds[1]:
                raise InvalidDialogueError(
                    f"{name} must be between {bounds[0]} and {bounds[1]}, got {control}"
                )
            object.__setattr__(self, name, control)

        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise InvalidDialogueError(
                f"seed must be a non-negative integer, got {self.seed!r}"
            )

    @classmethod
    def from_dialogue(cls, dialogue: AdaptedDialogue, *, seed: int) -> "PerformanceControls":
        """Map ``dialogue``'s direction onto the Chatterbox controls.

        Every adjustment starts from the adapter's own defaults and stays inside
        :data:`EXAGGERATION_BOUNDS`, :data:`CFG_WEIGHT_BOUNDS` and
        :data:`TEMPERATURE_BOUNDS`:

        * ``intensity`` is the only numeric field, so it is the one that moves a
          control predictably - a louder line is a little more exaggerated and a
          little less deterministic;
        * ``emotion`` and ``delivery`` only ever add a small bias on top;
        * a direction that matches no cue changes nothing, and says so.
        """

        if not isinstance(dialogue, AdaptedDialogue):
            raise InvalidDialogueError(
                f"expected an AdaptedDialogue, got {type(dialogue).__name__}"
            )

        notes: list[str] = []

        emotion_cue, emotion_term = _match_cue(dialogue.emotion, _EMOTION_CUES)
        if emotion_cue is None:
            notes.append(
                f"emotion {dialogue.emotion!r} matches no cue; exaggeration follows "
                "intensity only"
            )
        else:
            notes.append(f"emotion {dialogue.emotion!r} matched cue {emotion_term!r}")

        delivery_cue, delivery_term = _match_cue(dialogue.delivery, _DELIVERY_CUES)
        if delivery_cue is None:
            notes.append(
                f"delivery {dialogue.delivery!r} matches no cue; it changed no control"
            )
        else:
            notes.append(f"delivery {dialogue.delivery!r} matched cue {delivery_term!r}")

        swing = 2.0 * float(dialogue.intensity) - 1.0
        exaggeration_delta = INTENSITY_EXAGGERATION_SWING * swing
        temperature_delta = INTENSITY_TEMPERATURE_SWING * swing
        notes.append(
            f"intensity {float(dialogue.intensity):.3f} moves exaggeration by "
            f"{exaggeration_delta:+.3f} and temperature by {temperature_delta:+.3f}"
        )

        exaggeration = BASE_EXAGGERATION + exaggeration_delta
        cfg_weight = BASE_CFG_WEIGHT
        temperature = BASE_TEMPERATURE + temperature_delta
        for cue in (emotion_cue, delivery_cue):
            if cue is None:
                continue
            if cue.note:
                notes.append(cue.note)
            exaggeration += cue.exaggeration
            cfg_weight += cue.cfg_weight
            temperature += cue.temperature

        return cls(
            emotion=dialogue.emotion,
            intensity=float(dialogue.intensity),
            delivery=dialogue.delivery,
            pause_before=float(dialogue.pause_before),
            pause_after=float(dialogue.pause_after),
            exaggeration=_clamp(
                exaggeration, EXAGGERATION_BOUNDS, name="exaggeration", notes=notes
            ),
            cfg_weight=_clamp(cfg_weight, CFG_WEIGHT_BOUNDS, name="cfg_weight", notes=notes),
            temperature=_clamp(
                temperature, TEMPERATURE_BOUNDS, name="temperature", notes=notes
            ),
            seed=seed,
            matched_emotion_cue=emotion_term,
            matched_delivery_cue=delivery_term,
            notes=tuple(notes),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the mapped performance."""

        return {
            "emotion": self.emotion,
            "intensity": self.intensity,
            "delivery": self.delivery,
            "pause_before": self.pause_before,
            "pause_after": self.pause_after,
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "temperature": self.temperature,
            "seed": self.seed,
            "matched_emotion_cue": self.matched_emotion_cue,
            "matched_delivery_cue": self.matched_delivery_cue,
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
    #: The character's *performance* prompt, cut from the clean speech stem, and the
    #: identity reference the take was converted towards. Both are ``None`` for a
    #: single-voice engine: it follows no prompt and converts no identity, so there is
    #: no reference to point at, and pretending otherwise would be worse than saying so.
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

        return {
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


# ---------------------------------------------------------------------------
# Engine adapters
# ---------------------------------------------------------------------------


class ChatterboxPerformanceEngine(ABC):
    """An engine that speaks Amharic text with a performance reference.

    Implementations are adapters: they own one loaded model, they load it lazily,
    and they never decide *what* to say. ``synthesize`` must write a mono WAV to
    ``destination`` and return that path.
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
    def synthesize(
        self,
        *,
        text: str,
        performance_reference: Path,
        controls: PerformanceControls,
        destination: Path,
    ) -> Path:
        """Speak ``text`` using ``performance_reference`` as the prompt."""


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


class VoiceConversionEngine(ABC):
    """An engine that re-voices a take as a character.

    ``identity_reference`` is the character's own reference audio, and the only
    thing it may be used for is target identity: the content, accent and emotion
    of the delivery come from ``source_audio``.
    """

    #: Short, stable name recorded in the clip metadata.
    name: str

    @property
    @abstractmethod
    def is_loaded(self) -> bool:
        """``True`` once the underlying model has actually been loaded."""

    @abstractmethod
    def convert(
        self,
        *,
        source_audio: Path,
        identity_reference: Path,
        destination: Path,
    ) -> Path:
        """Convert ``source_audio`` into ``identity_reference``'s voice."""


def _seed_torch(seed: int) -> None:
    """Seed the global RNG so a line is reproducible across runs."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - no runtime installed
        raise EngineLoadError(
            "PyTorch is not installed; install the runtime dependencies before "
            "synthesizing speech"
        ) from exc
    torch.manual_seed(seed)


_LOADER_MODULES: dict[str, Any] = {}


def _download_amharic_loader(model: str, *, cache_dir: Path | None) -> Path:
    """Fetch ``amharic_tts.py`` from the adapter repository on the Hub.

    ``huggingface_hub`` is imported here rather than at module scope so that
    importing this module - and the test suite - never touches the runtime stack.
    """

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - no runtime installed
        raise EngineLoadError(
            "huggingface_hub is not installed; install the runtime dependencies "
            f"before loading the Amharic TTS adapter {model!r}"
        ) from exc

    try:
        return Path(
            hf_hub_download(
                repo_id=model,
                filename=AMHARIC_LOADER_FILENAME,
                cache_dir=None if cache_dir is None else str(cache_dir),
            )
        )
    except Exception as exc:
        raise EngineLoadError(
            f"could not download {AMHARIC_LOADER_FILENAME} from {model!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def _load_amharic_loader(model: str, *, cache_dir: Path | None) -> Any:
    """Return the Amharic loader module that ships with ``model``.

    The adapter repository publishes ``amharic_tts.py`` next to its weights and
    expects callers to load it from there, so the file is fetched from the Hub and
    executed as a module. The module is cached: it is fetched and executed once per
    process, not once per line.
    """

    cached = _LOADER_MODULES.get(model)
    if cached is not None:
        return cached

    loader_path = _download_amharic_loader(model, cache_dir=cache_dir)

    try:
        spec = importlib.util.spec_from_file_location(AMHARIC_LOADER_MODULE, loader_path)
        if spec is None or spec.loader is None:
            raise EngineLoadError(f"{loader_path} is not an importable Python module")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except EngineLoadError:
        raise
    except Exception as exc:
        raise EngineLoadError(
            f"could not load the Amharic TTS adapter from {loader_path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if not hasattr(module, "load_amharic_tts"):
        raise EngineLoadError(
            f"{loader_path} does not expose load_amharic_tts(); {model!r} is not a "
            "Chatterbox Amharic adapter repository"
        )

    _LOADER_MODULES[model] = module
    return module


def _cached_snapshot(
    repo_id: str,
    *,
    cache_dir: Path,
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Snapshot ``repo_id`` into ``cache_dir`` and return the local directory."""

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - no runtime installed
        raise EngineLoadError(
            "huggingface_hub is not installed; install the runtime dependencies "
            f"before loading {repo_id!r}"
        ) from exc

    try:
        return Path(
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                allow_patterns=allow_patterns,
                cache_dir=str(cache_dir),
            )
        )
    except Exception as exc:
        raise EngineLoadError(
            f"could not download {repo_id!r} into {cache_dir}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


class ChatterboxAmharicEngine(ChatterboxPerformanceEngine):
    """Adapter around Chatterbox Multilingual v3 plus the Amharic adapter.

    The model is loaded through the adapter repository's own loader, lazily on the
    first :meth:`synthesize` call and exactly once per engine instance. The
    adapter and its pinned base checkpoint are snapshotted into the configured
    model cache first, so the weights land where the rest of the project keeps
    them rather than in Hugging Face's default cache.
    """

    name = "chatterbox-amharic"

    def __init__(
        self,
        *,
        model: str = DEFAULT_CHATTERBOX_MODEL,
        device: str = "cuda",
        cache_dir: Path | None = None,
        adapter_dir: Path | None = None,
        base_dir: Path | None = None,
        merge_adapter: bool = True,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ConfigurationError("the TTS model must be a non-empty repository id")
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError("the TTS device must be a non-empty string")

        self._model = model.strip()
        self._device = device.strip()
        self._cache_dir = None if cache_dir is None else Path(cache_dir)
        self._adapter_dir = None if adapter_dir is None else Path(adapter_dir)
        self._base_dir = None if base_dir is None else Path(base_dir)
        self._merge_adapter = bool(merge_adapter)
        self._tts: Any | None = None

    @property
    def model(self) -> str:
        """The adapter repository this engine loads from."""

        return self._model

    @property
    def device(self) -> str:
        """The device the project settings asked for."""

        return self._device

    @property
    def is_loaded(self) -> bool:
        """``True`` once the adapter has been loaded."""

        return self._tts is not None

    @property
    def sample_rate(self) -> int:
        """Chatterbox's S3Gen output rate."""

        return CHATTERBOX_SAMPLE_RATE

    def _load(self) -> Any:
        """Load the adapter once and return it."""

        if self._tts is not None:
            return self._tts

        loader = _load_amharic_loader(self._model, cache_dir=self._cache_dir)
        adapter_dir, base_dir = self._local_checkout(loader)
        try:
            tts = loader.load_amharic_tts(
                device=self._device,
                adapter_dir=adapter_dir,
                base_dir=base_dir,
                merge_adapter=self._merge_adapter,
            )
        except Exception as exc:
            raise EngineLoadError(
                f"could not load the Amharic TTS adapter {self._model!r} on device "
                f"{self._device!r}: {type(exc).__name__}: {exc}"
            ) from exc

        if tts is None or not hasattr(tts, "generate"):
            raise EngineLoadError(
                f"the loader of {self._model!r} returned no usable text-to-speech model"
            )

        self._tts = tts
        return tts

    def _local_checkout(self, loader: Any) -> tuple[Path | None, Path | None]:
        """Return the adapter and base directories to load from.

        The adapter's loader downloads both itself when it is given no
        directories, but those downloads then land in Hugging Face's *default*
        cache rather than in ``MODEL_CACHE_DIR``, which is the persistent volume
        on the worker. Snapshotting them here - with the adapter's own pinned
        revision and file list, read from the loader module so there is no second
        copy of that pin to drift - keeps every Chatterbox weight where the rest
        of the project keeps its weights. Directories passed in explicitly always
        win, and a loader that does not publish its base constants keeps its own
        download behaviour.
        """

        adapter_dir = self._adapter_dir
        base_dir = self._base_dir
        if self._cache_dir is None:
            return adapter_dir, base_dir

        if adapter_dir is None:
            adapter_dir = _cached_snapshot(self._model, cache_dir=self._cache_dir)

        repo = getattr(loader, "BASE_REPO", None)
        revision = getattr(loader, "BASE_REVISION", None)
        files = getattr(loader, "BASE_FILES", None)
        if base_dir is None and repo and revision and files:
            base_dir = _cached_snapshot(
                repo,
                revision=revision,
                allow_patterns=list(files),
                cache_dir=self._cache_dir,
            )
        return adapter_dir, base_dir

    def synthesize(
        self,
        *,
        text: str,
        performance_reference: Path,
        controls: PerformanceControls,
        destination: Path,
    ) -> Path:
        """Speak ``text``, performed like ``performance_reference``.

        The adapter normalizes the Amharic text and splits it into sentences
        itself - that front-end is the one the adapter was trained with, so it is
        never bypassed here. No language tag is passed: the adapter was trained
        without a language token.
        """

        if not isinstance(text, str) or not text.strip():
            raise InvalidDialogueError("the text to synthesize must be non-empty")
        if not isinstance(controls, PerformanceControls):
            raise InvalidInputError(
                f"controls must be PerformanceControls, got {type(controls).__name__}"
            )
        reference = _validate_audio_file(
            performance_reference, label="performance reference audio"
        )
        target = Path(destination)

        tts = self._load()
        _seed_torch(controls.seed)
        try:
            waveform = tts.generate(
                text,
                audio_prompt_path=str(reference),
                temperature=controls.temperature,
                cfg_weight=controls.cfg_weight,
                exaggeration=controls.exaggeration,
            )
        except Exception as exc:
            raise SynthesisError(
                f"Chatterbox failed to synthesize {text[:40]!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        try:
            samples = _as_samples(waveform)
        except InvalidAudioError as exc:
            raise SynthesisError(f"Chatterbox returned unusable audio: {exc}") from exc
        if samples.size == 0:
            raise SynthesisError("Chatterbox returned an empty waveform")

        return _write_mono(target, samples, self.sample_rate, label="Chatterbox take")


#: Attributes that ``hf_hub_download`` carried before huggingface_hub 1.0, with the
#: value used to satisfy Seed-VC's vocoder. ``proxies`` was removed outright in
#: that release and ``resume_download`` after a deprecation, so a checkout written
#: against the 0.x line fails on the machines this project targets.
SEED_VC_LEGACY_DOWNLOAD_ARGS: dict[str, Any] = {
    "proxies": None,
    "resume_download": False,
}

#: Marks a compatibility wrapper so it is only ever installed once.
SEED_VC_SHIM_MARKER = "_amharic_dub_hub_compatibility_shim"

#: The module in the Seed-VC checkout whose vocoder needs adapting.
SEED_VC_VOCODER_MODULE = "modules.bigvgan.bigvgan"


def _import_seed_vc_vocoder(repo_path: Path) -> Any:
    """Import the checkout's ``BigVGAN`` module, or explain why it could not be."""

    try:
        return importlib.import_module(SEED_VC_VOCODER_MODULE)
    except Exception as exc:
        raise EngineLoadError(
            f"could not import {SEED_VC_VOCODER_MODULE} from the Seed-VC checkout at "
            f"{repo_path}: {type(exc).__name__}: {exc}"
        ) from exc


def _make_vocoder_args_optional(module: Any) -> None:
    """Stop Seed-VC's vocoder requiring two arguments hub no longer passes.

    ``BigVGAN._from_pretrained`` declares ``proxies`` and ``resume_download`` as
    *required* keyword-only parameters, and huggingface_hub dropped both in 1.0 -
    its ``from_pretrained`` stopped supplying them. Seed-VC's own requirements ask
    only for ``huggingface-hub>=0.28.1``, so it was written against the 0.x line.

    Supplying them through the config does **not** work: hub's argument validator
    pops both names out of the keyword arguments before the call happens (see
    ``utils/_validators.py``, ``proxies = new_kwargs.pop("proxies", None)``, which
    exists to retire them silently). Passing them is therefore invisible, and they
    have to be defaulted *inside* the method, where nothing can remove them. Both
    only ever selected defaults hub now applies itself, so nothing observable
    changes.
    """

    vocoder = getattr(module, "BigVGAN", None)
    if vocoder is None:
        raise EngineLoadError(
            f"{SEED_VC_VOCODER_MODULE} does not define BigVGAN; the Seed-VC checkout "
            "is not the one this project expects"
        )

    descriptor = vocoder.__dict__.get("_from_pretrained")
    if descriptor is None:
        # Inherited from the mixin, which already defaults both arguments.
        return

    function = getattr(descriptor, "__func__", descriptor)
    if getattr(function, SEED_VC_SHIM_MARKER, False):
        return

    @functools.wraps(function)
    def with_defaults(cls: Any, *args: Any, **kwargs: Any) -> Any:
        for name, value in SEED_VC_LEGACY_DOWNLOAD_ARGS.items():
            kwargs.setdefault(name, value)
        return function(cls, *args, **kwargs)

    setattr(with_defaults, SEED_VC_SHIM_MARKER, True)
    setattr(vocoder, "_from_pretrained", classmethod(with_defaults))


def _tolerate_removed_download_kwargs(module: Any) -> None:
    """Let the vocoder's own downloads drop the arguments hub removed.

    ``_from_pretrained`` passes the same two names on to ``hf_hub_download``, which
    dropped them in the same release - so downloading would fail even once the
    method's signature is satisfied. The patch is confined to the vocoder module
    rather than applied to ``huggingface_hub`` itself, so nothing else in the
    process sees it.

    It is the only place in the checkout that needs adapting: every other hub call
    there - ``hf_utils.py``, ``modules/v2/vc_wrapper.py``,
    ``modules/astral_quantization/default_model.py`` - passes only arguments that
    still exist.
    """

    original = getattr(module, "hf_hub_download", None)
    if original is None or getattr(original, SEED_VC_SHIM_MARKER, False):
        return

    @functools.wraps(original)
    def without_removed_kwargs(*args: Any, **kwargs: Any) -> Any:
        for name in SEED_VC_LEGACY_DOWNLOAD_ARGS:
            kwargs.pop(name, None)
        return original(*args, **kwargs)

    setattr(without_removed_kwargs, SEED_VC_SHIM_MARKER, True)
    module.hf_hub_download = without_removed_kwargs


def _adapt_seed_vc_vocoder(repo_path: Path) -> None:
    """Make the checkout's vocoder work with this project's ``huggingface_hub``."""

    module = _import_seed_vc_vocoder(repo_path)
    _make_vocoder_args_optional(module)
    _tolerate_removed_download_kwargs(module)


def _seed_vc_runtime(repo_path: Path, config_path: Path) -> Any:
    """Instantiate Seed-VC V2's ``VCWrapper`` from its checkout.

    This mirrors the repository's own ``inference_v2.py``: the wrapper is built
    from ``configs/v2/vc_wrapper.yaml``, and its checkpoints are downloaded by the
    wrapper itself when no explicit paths are given. Everything heavy - Hydra,
    OmegaConf, PyTorch - is imported here and nowhere else, so importing this
    module never touches the Seed-VC runtime.

    Two adjustments make the checkout work with this project's ``huggingface_hub``,
    which is newer than the one Seed-VC was written against; both are explained at
    their definitions below.
    """

    import yaml  # noqa: F401 - documented part of the Seed-VC environment
    from hydra.utils import instantiate
    from omegaconf import DictConfig

    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    _adapt_seed_vc_vocoder(repo_path)

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return instantiate(DictConfig(payload))


def _seed_vc_dtype(device: str) -> Any:
    """Return the Seed-VC dtype for ``device``: fp16 on GPU, fp32 elsewhere."""

    import torch

    return torch.float16 if device.split(":")[0] == "cuda" else torch.float32


def _seed_vc_device(device: str) -> Any:
    """Return ``device`` as a :class:`torch.device`.

    Seed-VC V2's conversion does ``torch.autocast(device_type=device.type, ...)``,
    so it needs a real ``torch.device`` and not the settings string. Imported
    here rather than at module scope, so importing this module stays cheap.
    """

    import torch

    return torch.device(device)


class SeedVcV2Engine(VoiceConversionEngine):
    """Adapter around Seed-VC V2, run in timbre-only mode by default.

    Seed-VC is not a package, so this engine runs the ``VCWrapper`` that the
    repository's ``inference_v2.py`` builds from a checkout. The wrapper is created
    lazily and cached on the instance; ``load_checkpoints`` downloads the V2
    checkpoints the first time it runs.

    Two details of that API are load-bearing here: the device must be a real
    :class:`torch.device` (the converter reads ``device.type``), and with
    ``stream_output=True`` the generator yields ``(mp3_bytes, full_audio)`` pairs
    where only the final chunk carries the completed ``(sample_rate, samples)``.

    ``convert_style`` is off unless explicitly asked for: on, Seed-VC V2's
    autoregressive stage is primed with the *reference's* acoustic tokens and
    teacher-forced against the *reference's* content indices, so the output
    inherits the reference's accent and emotion rather than the take's. Because the
    identity reference is the original actor's audio in the source language, that
    mode would both anglicise the Amharic and discard the performance this stage
    exists to preserve.
    """

    name = SEED_VC_ENGINE_NAME

    def __init__(
        self,
        *,
        repo_path: str | Path,
        device: str = "cuda",
        diffusion_steps: int = DEFAULT_SEED_VC_DIFFUSION_STEPS,
        length_adjust: float = 1.0,
        intelligibility_cfg_rate: float = 0.7,
        similarity_cfg_rate: float = 0.7,
        top_p: float = 0.9,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        convert_style: bool = DEFAULT_SEED_VC_CONVERT_STYLE,
    ) -> None:
        if not isinstance(device, str) or not device.strip():
            raise ConfigurationError("the Seed-VC device must be a non-empty string")
        if isinstance(diffusion_steps, bool) or not isinstance(diffusion_steps, int):
            raise ConfigurationError(
                f"SEED_VC_DIFFUSION_STEPS must be an integer, got {diffusion_steps!r}"
            )
        if diffusion_steps < 1:
            raise ConfigurationError(
                f"SEED_VC_DIFFUSION_STEPS must be at least 1, got {diffusion_steps}"
            )
        if not isinstance(convert_style, bool):
            raise ConfigurationError(
                f"SEED_VC_CONVERT_STYLE must be a boolean, got {convert_style!r}"
            )

        self._repo_path = _clip_path("repo_path", repo_path)
        self._device = device.strip()
        self._diffusion_steps = diffusion_steps
        self._length_adjust = _positive("length_adjust", length_adjust)
        self._convert_style = convert_style
        self._intelligibility_cfg_rate = _rate(
            "intelligibility_cfg_rate", intelligibility_cfg_rate
        )
        self._similarity_cfg_rate = _rate("similarity_cfg_rate", similarity_cfg_rate)
        self._top_p = _rate("top_p", top_p)
        self._temperature = _positive("temperature", temperature)
        self._repetition_penalty = _positive("repetition_penalty", repetition_penalty)
        self._wrapper: Any | None = None
        #: Resolved on first load, because building a ``torch.device`` needs torch.
        self._torch_device: Any | None = None

    @property
    def repo_path(self) -> Path:
        """The Seed-VC checkout this engine runs from."""

        return self._repo_path

    @property
    def is_loaded(self) -> bool:
        """``True`` once the Seed-VC wrapper has been built."""

        return self._wrapper is not None

    @property
    def convert_style(self) -> bool:
        """Whether Seed-VC also converts the reference's accent and style.

        ``False`` by default, which is the mode that leaves the take's delivery
        intact; see the class docstring for why that is the correct default here.
        """

        return self._convert_style

    def _load(self) -> Any:
        """Build the Seed-VC V2 wrapper once and return it."""

        if self._wrapper is not None:
            return self._wrapper

        repo = self._repo_path
        if not repo.is_dir():
            raise EngineLoadError(
                f"the Seed-VC checkout was not found at {repo}; clone "
                "https://github.com/Plachtaa/seed-vc there or point SEED_VC_REPO_PATH "
                "at an existing checkout"
            )

        config = repo.joinpath(*SEED_VC_CONFIG_PARTS)
        if not config.is_file():
            raise EngineLoadError(
                f"the Seed-VC checkout at {repo} has no {Path(*SEED_VC_CONFIG_PARTS)}"
            )

        try:
            # Seed-VC V2 dereferences ``device.type``, so it is handed a real
            # torch.device; the settings string is only the configuration form.
            device = _seed_vc_device(self._device)
            wrapper = _seed_vc_runtime(repo, config)
            wrapper.load_checkpoints(ar_checkpoint_path=None, cfm_checkpoint_path=None)
            wrapper.to(device)
            wrapper.eval()
            wrapper.setup_ar_caches(
                max_batch_size=SEED_VC_AR_CACHE_MAX_BATCH_SIZE,
                max_seq_len=SEED_VC_AR_CACHE_MAX_SEQ_LEN,
                dtype=_seed_vc_dtype(self._device),
                device=device,
            )
        except Exception as exc:
            raise EngineLoadError(
                f"could not load Seed-VC V2 from {repo} on device {self._device!r}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        self._torch_device = device
        self._wrapper = wrapper
        return wrapper

    def convert(
        self,
        *,
        source_audio: Path,
        identity_reference: Path,
        destination: Path,
    ) -> Path:
        """Convert ``source_audio`` into the voice of ``identity_reference``."""

        source = _validate_audio_file(source_audio, label="Seed-VC source audio")
        reference = _validate_audio_file(
            identity_reference, label="Seed-VC identity reference"
        )
        target = Path(destination)

        wrapper = self._load()
        try:
            generator = wrapper.convert_voice_with_streaming(
                source_audio_path=str(source),
                target_audio_path=str(reference),
                diffusion_steps=self._diffusion_steps,
                length_adjust=self._length_adjust,
                # Seed-VC's own spelling of the argument, kept as-is on purpose.
                intelligebility_cfg_rate=self._intelligibility_cfg_rate,
                similarity_cfg_rate=self._similarity_cfg_rate,
                top_p=self._top_p,
                temperature=self._temperature,
                repetition_penalty=self._repetition_penalty,
                convert_style=self._convert_style,
                anonymization_only=False,
                device=self._torch_device,
                dtype=_seed_vc_dtype(self._device),
                stream_output=True,
            )
            # ``stream_output=True`` makes this a generator of
            # ``(mp3_bytes, full_audio)`` pairs, where ``full_audio`` is None on
            # every chunk but the last and ``(sample_rate, samples)`` there - the
            # same shape the repository's own ``inference_v2.py`` consumes.
            collected: Any = None
            for chunk in generator:
                try:
                    _, full_audio = chunk
                except (TypeError, ValueError) as exc:
                    raise ConversionError(
                        "Seed-VC V2 yielded an unexpected result of type "
                        f"{type(chunk).__name__}"
                    ) from exc
                if full_audio is not None:
                    collected = full_audio
        except ConversionError:
            raise
        except Exception as exc:
            raise ConversionError(
                "Seed-VC V2 failed to convert a take into the character's voice: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        if collected is None:
            raise ConversionError("Seed-VC V2 produced no audio")
        try:
            sample_rate, waveform = collected
        except (TypeError, ValueError) as exc:
            raise ConversionError(
                "Seed-VC V2 returned an unexpected result of type "
                f"{type(collected).__name__}"
            ) from exc

        try:
            samples = _as_samples(waveform)
        except InvalidAudioError as exc:
            raise ConversionError(f"Seed-VC V2 returned unusable audio: {exc}") from exc
        if samples.size == 0:
            raise ConversionError("Seed-VC V2 returned an empty waveform")

        return _write_mono(target, samples, int(sample_rate), label="converted take")


# ---------------------------------------------------------------------------
# Seed-VC provenance
# ---------------------------------------------------------------------------

#: Directory holding the checkout's git metadata.
SEED_VC_GIT_DIRECTORY = ".git"

#: A commit id is 40 hex characters; abbreviated ones are accepted too.
_GIT_SHA_LENGTHS = range(7, 41)


def _looks_like_commit(value: str) -> bool:
    """``True`` when ``value`` is shaped like a git object id."""

    return len(value) in _GIT_SHA_LENGTHS and all(
        character in "0123456789abcdefABCDEF" for character in value
    )


def _read_git_file(path: Path) -> str | None:
    """Return ``path``'s stripped contents, or ``None`` when it cannot be read."""

    try:
        return path.read_text(encoding="utf-8", errors="replace").strip() or None
    except OSError:
        return None


def _resolve_git_directory(repo_path: Path) -> Path | None:
    """Return the checkout's real git directory, following a ``.git`` file.

    A normal clone has a ``.git`` directory; a worktree or a submodule has a
    ``.git`` file containing ``gitdir: <path>``, which is followed when it
    resolves to an existing directory.
    """

    candidate = repo_path / SEED_VC_GIT_DIRECTORY
    if candidate.is_dir():
        return candidate
    if not candidate.is_file():
        return None

    pointer = _read_git_file(candidate)
    if pointer is None or not pointer.lower().startswith("gitdir:"):
        return None

    resolved = Path(pointer.split(":", 1)[1].strip())
    if not resolved.is_absolute():
        resolved = (candidate.parent / resolved).resolve()
    return resolved if resolved.is_dir() else None


def _read_packed_ref(git_dir: Path, ref: str) -> str | None:
    """Return ``ref`` from ``packed-refs``, or ``None`` when it is not packed."""

    packed = _read_git_file(git_dir / "packed-refs")
    if packed is None:
        return None

    for line in packed.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("^"):
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            return parts[0].strip() or None
    return None


def seed_vc_revision(repo_path: str | Path) -> str | None:
    """Return the commit the Seed-VC checkout is sitting on, or ``None``.

    Seed-VC is not a package: it is a ``git clone`` that the pipeline runs code
    from, and upstream has been read-only since April 2025. Recording the exact
    commit a run used is therefore the only way to say what produced a dub, and
    the only way to notice that a re-clone silently changed the engine. Nothing
    here raises: an unreadable checkout is reported as unknown revision, not turned
    into a failed run.
    """

    git_dir = _resolve_git_directory(Path(repo_path))
    if git_dir is None:
        return None

    head = _read_git_file(git_dir / "HEAD")
    if head is None:
        return None

    if not head.lower().startswith("ref:"):
        return head if _looks_like_commit(head) else None

    ref = head.split(":", 1)[1].strip()
    revision = _read_git_file(git_dir / ref) or _read_packed_ref(git_dir, ref)
    if revision is None or not _looks_like_commit(revision):
        return None
    return revision


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
      converted with ``uroman`` before synthesis. That is the reverse of what the
      Chatterbox path needs, and it is handled here so the rest of the pipeline can
      keep working in Fidel.

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

#: OmniVoice's own code for Amharic. Its language map is keyed by name
#: (``"amharic": "am"``), so the ISO 639-3 code ``amh`` is *not* what it accepts.
OMNIVOICE_AMHARIC = "am"

#: The rate OmniVoice generates at.
OMNIVOICE_SAMPLE_RATE = 24_000


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

    This is a third contract, and it exists because the other two cannot express what
    a zero-shot cloning model does:

    * :class:`ChatterboxPerformanceEngine` follows a *performance* prompt and leaves
      identity to a conversion stage, so the take is not the character's voice.
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

    The voice design mode (``instruct=``) is *not* used: the model was trained for it
    on English and Chinese only, so asking it for "whispering" in Amharic would be a
    request it cannot reliably honour. Emotion is left to the pacing and mixing stages
    rather than faked here.
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
        is known: without it the model transcribes the reference with an ASR model of
        its own, which is both slower and one more download.
        """

        key = (str(reference), (reference_text or "").strip())
        prompt = self._prompts.get(key)
        if prompt is not None:
            return prompt

        _torch, network, _config = self._load()
        text = key[1] or None
        try:
            prompt = network.create_voice_clone_prompt(
                ref_audio=str(reference), ref_text=text
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

        # Both temperatures at zero is what makes the result reproducible: `generate`
        # samples by default, and an unseeded run could not be compared with the run it
        # is meant to improve.
        config = config_class(
            num_step=self._steps,
            guidance_scale=self._guidance_scale,
            position_temperature=0.0,
            class_temperature=0.0,
        )
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

_CHATTERBOX_ENGINES: dict[tuple[str, str], ChatterboxAmharicEngine] = {}
#: Keyed by device, checkout and diffusion steps *and* the style flag, because two
#: settings that differ only in that flag need two different engines.
_SEED_VC_ENGINES: dict[tuple[str, str, int, bool], SeedVcV2Engine] = {}
#: Single-voice engines, keyed by the settings that shape one.
_MMS_ENGINES: dict[tuple[str, str, int, float], MmsAmharicEngine] = {}
#: Voice-cloning engines, keyed by the settings that shape one.
_OMNIVOICE_ENGINES: dict[tuple[str, str, int, float, str], OmniVoiceEngine] = {}


def load_chatterbox_engine(*, settings: Settings | None = None) -> ChatterboxAmharicEngine:
    """Return the process-wide Chatterbox Amharic engine for these settings.

    Building an engine is cheap - the model itself is only loaded on the first
    :meth:`ChatterboxAmharicEngine.synthesize` call - but the engine is cached
    anyway, so a stage run and every line in it share one instance and one model.
    """

    resolved = settings if settings is not None else get_settings()
    key = (resolved.device, resolved.chatterbox_model)
    engine = _CHATTERBOX_ENGINES.get(key)
    if engine is None:
        engine = ChatterboxAmharicEngine(
            model=resolved.chatterbox_model,
            device=resolved.device,
            cache_dir=Path(resolved.model_cache_dir),
        )
        _CHATTERBOX_ENGINES[key] = engine
    return engine


def load_seed_vc_engine(*, settings: Settings | None = None) -> SeedVcV2Engine:
    """Return the process-wide Seed-VC V2 engine for these settings."""

    resolved = settings if settings is not None else get_settings()
    key = (
        resolved.device,
        str(resolved.seed_vc_repo_path),
        resolved.seed_vc_diffusion_steps,
        resolved.seed_vc_convert_style,
    )
    engine = _SEED_VC_ENGINES.get(key)
    if engine is None:
        engine = SeedVcV2Engine(
            repo_path=resolved.seed_vc_repo_path,
            device=resolved.device,
            diffusion_steps=resolved.seed_vc_diffusion_steps,
            convert_style=resolved.seed_vc_convert_style,
        )
        _SEED_VC_ENGINES[key] = engine
    return engine


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

#: Engines that need a per-speaker voice reference at all: the cloning engines, which
#: speak *as* that voice, and ``chatterbox``, which converts a take into it. ``mms``
#: needs none - it has one voice for the whole film - so a run using it skips the
#: voice-profile stage instead of building references nothing would read.
PROFILE_ENGINES: tuple[str, ...] = ("chatterbox",) + VOICE_CLONING_ENGINES

#: Every engine ``TTS_ENGINE`` may name.
SUPPORTED_ENGINES: tuple[str, ...] = ("chatterbox", "mms") + VOICE_CLONING_ENGINES

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
    if engine == "chatterbox":
        return f"{engine}:{resolved.chatterbox_model}"
    if engine == "omnivoice":
        return f"{engine}:{resolved.omnivoice_model}"
    return f"{engine}:{resolved.tts_model}"


def reset_engine_cache() -> None:
    """Drop every cached engine and loader, releasing the loaded models.

    An engine is useless once its model is gone, so the cache is cleared as a
    whole. The next call to a ``load_*_engine`` function starts from a clean slate.
    """

    _CHATTERBOX_ENGINES.clear()
    _SEED_VC_ENGINES.clear()
    _MMS_ENGINES.clear()
    _OMNIVOICE_ENGINES.clear()
    _LOADER_MODULES.clear()


# ---------------------------------------------------------------------------
# The original performance reference
# ---------------------------------------------------------------------------


def extract_performance_reference(
    speech_stem: str | Path,
    *,
    start: float,
    end: float,
    destination: str | Path,
    minimum_duration: float = DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION,
    maximum_duration: float = DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION,
) -> Path:
    """Cut the original performance of one line out of the clean speech stem.

    The window is centred on the line: a line shorter than ``minimum_duration`` is
    padded with its own surrounding dialogue, a longer one is trimmed, and the
    window is always shifted to stay inside the stem. Feeding only the line itself
    would leave a very short utterance with too little prompt audio, and feeding
    the whole stem would lose which of the film's performances this line is.

    The result is written mono, at the stem's own sample rate. Resampling is
    deliberately left to Chatterbox's loader, which reads the prompt at 16 kHz
    itself, so this module makes no audio-quality decision of its own.
    """

    stem = _validate_audio_file(speech_stem, label="the BandIt speech stem")
    target = Path(destination)

    low = _positive("minimum_duration", minimum_duration)
    high = _positive("maximum_duration", maximum_duration)
    if high < low:
        raise ConfigurationError(
            f"TTS_PERFORMANCE_REFERENCE_MAX_DURATION ({high}) must not be smaller than "
            f"TTS_PERFORMANCE_REFERENCE_MIN_DURATION ({low})"
        )

    line_start = _finite_seconds("start", start)
    line_end = _finite_seconds("end", end)
    if line_start < 0:
        raise InvalidInputError(f"start must be >= 0 seconds, got {line_start}")
    if line_end <= line_start:
        raise InvalidInputError(
            f"end must be greater than start ({line_start} seconds), got {line_end}"
        )

    frames, sample_rate = _read_audio_info(stem, label="the BandIt speech stem")
    total = frames / sample_rate
    if line_start >= total:
        raise InvalidInputError(
            f"the dialogue line starts at {line_start:.3f}s but the speech stem is only "
            f"{total:.3f}s long"
        )

    begin, finish = _performance_window(
        line_start,
        min(line_end, total),
        total=total,
        minimum=low,
        maximum=high,
    )
    first = int(round(begin * sample_rate))
    last = min(max(first + 1, int(round(finish * sample_rate))), frames)

    try:
        with sf.SoundFile(str(stem)) as handle:
            handle.seek(first)
            data = handle.read(last - first, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise InvalidAudioError(f"could not read {stem} at {begin:.3f}s: {exc}") from exc

    if data.size == 0:
        raise InvalidAudioError(
            f"the speech stem {stem} has no samples between {begin:.3f}s and {finish:.3f}s"
        )

    samples = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    return _write_mono(
        target,
        np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate,
        label="performance reference",
    )


def _performance_window(
    start: float,
    end: float,
    *,
    total: float,
    minimum: float,
    maximum: float,
) -> tuple[float, float]:
    """Return the prompt window centred on ``[start, end]``, inside ``[0, total]``."""

    length = min(max(end - start, minimum), maximum)
    length = min(length, total)

    centre = (start + end) / 2.0
    begin = centre - length / 2.0
    begin = max(0.0, min(begin, total - length))
    return begin, begin + length


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TtsDirectories:
    """The artifact directories of one TTS run."""

    performance: Path
    chatterbox: Path
    converted: Path
    clips: Path
    takes: Path

    @classmethod
    def under(cls, base: Path) -> "_TtsDirectories":
        return cls(
            performance=base / PERFORMANCE_DIRECTORY_NAME,
            chatterbox=base / CHATTERBOX_DIRECTORY_NAME,
            converted=base / CONVERTED_DIRECTORY_NAME,
            clips=base / CLIP_DIRECTORY_NAME,
            takes=base / TAKE_DIRECTORY_NAME,
        )

    def create(self) -> None:
        """Create every directory, creating parents as needed."""

        for directory in (
            self.performance,
            self.chatterbox,
            self.converted,
            self.clips,
            self.takes,
        ):
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


def _line_digest(dialogue: AdaptedDialogue, *, model: str) -> str:
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
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _line_artifact_name(dialogue: AdaptedDialogue, *, model: str) -> str:
    """Return the deterministic, order-independent stem shared by a line's files."""

    digest = _line_digest(dialogue, model=model)[:16]
    return f"{speaker_directory_name(dialogue.speaker_id)}_{digest}"


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
        }


#: Prefix of the reason recorded for a line an engine failed on, so a report can tell
#: a broken line apart from an unusable one.
FAILURE_REASON_PREFIX = "synthesis failed"
#: Prefix of the reason recorded for a line that cannot be dubbed at all.
UNSPEAKABLE_REASON_PREFIX = "cannot be dubbed"


def _skip_reason(line: AdaptedDialogue, *, min_line_seconds: float) -> str | None:
    """Return why ``line`` cannot be dubbed, or ``None`` when it can.

    Both checks are about the *input*, not about an engine: a window with no room for
    a word, or text with nothing to pronounce. Skipping these is not tolerating a
    failure, it is refusing to ask for something that cannot work - and it is what
    keeps a broken fragment from ending a three-hour run.
    """

    if line.duration < min_line_seconds:
        return (
            f"{UNSPEAKABLE_REASON_PREFIX}: the original window is "
            f"{line.duration:.3f}s, shorter than the {min_line_seconds:g}s a line "
            "needs to hold a word"
        )
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


def _resolve_engines(
    *,
    settings: Settings,
    performance_engine: ChatterboxPerformanceEngine | None,
    style_engine: VoiceConversionEngine | None,
) -> tuple[ChatterboxPerformanceEngine, VoiceConversionEngine]:
    """Return the two engines to use, resolving them once for the whole run."""

    if performance_engine is None:
        performance: ChatterboxPerformanceEngine = load_chatterbox_engine(settings=settings)
    elif isinstance(performance_engine, ChatterboxPerformanceEngine):
        performance = performance_engine
    else:
        raise InvalidEngineError(
            "performance_engine must be a ChatterboxPerformanceEngine adapter, got "
            f"{type(performance_engine).__name__}"
        )

    if style_engine is None:
        style: VoiceConversionEngine = load_seed_vc_engine(settings=settings)
    elif isinstance(style_engine, VoiceConversionEngine):
        style = style_engine
    else:
        raise InvalidEngineError(
            f"style_engine must be a VoiceConversionEngine adapter, got "
            f"{type(style_engine).__name__}"
        )

    return performance, style


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
    """Write the delivered clip: the converted line with its pauses around it.

    Returns ``(speech_duration, rendered_pause_before, rendered_pause_after)``.
    The pauses are capped at ``max_pause``; the dialogue's own estimate is kept
    untouched in the clip metadata, so a cap never hides what was asked for.
    """

    samples, sample_rate = _read_mono(source, label="the converted take")
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


def _synthesize_line(
    index: int,
    dialogue: AdaptedDialogue,
    *,
    name: str,
    digest: str,
    stem: Path,
    profile: VoiceProfile,
    directories: _TtsDirectories,
    performance_engine: ChatterboxPerformanceEngine,
    style_engine: VoiceConversionEngine,
    settings: Settings,
) -> TtsClip:
    """Run both engines for one line and describe the result."""

    controls = PerformanceControls.from_dialogue(dialogue, seed=int(digest[:8], 16))

    performance_reference = directories.performance / f"{name}.wav"
    if not _usable_audio(performance_reference):
        extract_performance_reference(
            stem,
            start=dialogue.start,
            end=dialogue.end,
            destination=performance_reference,
            minimum_duration=settings.tts_performance_reference_min_duration,
            maximum_duration=settings.tts_performance_reference_max_duration,
        )

    # Chatterbox first: it performs the Amharic text. The prompt is the original
    # performance from the clean stem, never the character's identity reference.
    take = directories.chatterbox / f"{name}.wav"
    if not _usable_audio(take):
        performance_engine.synthesize(
            text=dialogue.amharic,
            performance_reference=performance_reference,
            controls=controls,
            destination=take,
        )
        _require_output(take, what="take", engine=performance_engine.name)

    # Then Seed-VC: same performance, the character's own voice. The profile's
    # reference audio is used here and only here.
    identity_reference = profile.resolve_reference_audio()
    converted = directories.converted / f"{name}.wav"
    if not _usable_audio(converted):
        style_engine.convert(
            source_audio=take,
            identity_reference=identity_reference,
            destination=converted,
        )
        _require_output(converted, what="converted take", engine=style_engine.name)

    clip_path = directories.clips / f"{name}.wav"
    speech_duration, lead, trail = _render_clip(
        converted,
        clip_path,
        pause_before=controls.pause_before,
        pause_after=controls.pause_after,
        max_pause=settings.tts_max_pause_seconds,
    )

    _, clip_rate = _read_audio_info(clip_path, label="the dubbed clip")
    return TtsClip(
        index=index,
        dialogue=dialogue,
        performance=controls,
        audio_path=clip_path,
        take_path=take,
        performance_reference_path=performance_reference,
        voice_reference_path=identity_reference,
        sample_rate=clip_rate,
        speech_duration=speech_duration,
        rendered_pause_before=lead,
        rendered_pause_after=trail,
        performance_engine=performance_engine.name,
        style_engine=style_engine.name,
    )


def synthesize_dialogue_detailed(
    dialogue: Iterable[AdaptedDialogue],
    speech_stem: str | Path,
    voice_profiles: Mapping[str, VoiceProfile] | None = None,
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    performance_engine: ChatterboxPerformanceEngine | None = None,
    style_engine: VoiceConversionEngine | None = None,
    tts_engine: TextToSpeechEngine | None = None,
) -> SynthesisResult:
    """Synthesize every dubbable line, and report the ones that could not be.

    Parameters
    ----------
    dialogue:
        Lines from :func:`app.pipeline.translation.adapt_dialogue`, in
        chronological order. Their performance metadata is mapped onto the
        engines, never dropped.
    speech_stem:
        The clean ``speech`` stem written by
        :func:`app.pipeline.separation.separate_stems` - the path held in
        ``StemPaths.speech``. Each line's original performance is cut from it and
        used as the Chatterbox prompt.
    voice_profiles:
        Profiles keyed by speaker id, as returned by
        :func:`app.pipeline.voice_profiles.build_voice_profiles` and persisted by
        :func:`~app.pipeline.voice_profiles.save_voice_profiles`. Only
        ``reference_audio`` is read, and only as the Seed-VC identity reference.
    output_dir:
        Directory for this stage's artifacts. Defaults to ``WORK_DIR/tts``;
        ``performance``, ``chatterbox``, ``converted`` and ``clips``
        sub-directories are created below it.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
        ``DEVICE`` reaches both engines from here.
    performance_engine, style_engine:
        Engine adapters to use. When omitted, the cached Chatterbox Amharic and
        Seed-VC V2 engines are resolved once for the whole run, which is also how a
        loaded model is never loaded per line.

    Returns
    -------
    list[TtsClip]
        One clip per input line, in the same order. Audio for each line is written
        under the stage directory, at a path derived from the line's content, so
        re-running the stage reuses the take and the conversion instead of redoing
        them. Durations are reported, not fitted: fitting each line into its
        original window is :mod:`app.pipeline.timing`'s job.

    Raises
    ------
    InvalidDialogueError, InvalidVoiceProfileError, InvalidInputError
        The supplied lines, profiles or paths are not usable.
    MissingVoiceProfileError
        A speaker in the dialogue has no profile.
    MissingInputError, InvalidAudioError
        The speech stem, or a profile's reference audio, is missing or unreadable.
    ConfigurationError
        The configured reference durations or engine options contradict each other.
    InvalidEngineError
        An injected engine is not the expected kind of adapter.
    EngineLoadError, SynthesisError, ConversionError, MissingOutputError
        An engine could not be loaded, or failed in a way that is not specific to
        one line. A failure on a *single* line is reported in the result's
        ``skipped`` rather than raised, so one unvoiceable line cannot end a run.
    """

    lines = _validate_dialogue(dialogue)
    if not lines:
        return SynthesisResult(clips=())

    resolved = settings if settings is not None else get_settings()

    # Three ways to speak a line, and which one is in use decides whether the
    # voice-profile stage has anything to do:
    #   chatterbox  prompt a take with the performance, then convert its identity
    #   omnivoice   clone the character's own voice and speak as it
    #   mms         one voice for the whole film
    # The first two both read a per-speaker reference; the third has none to read.
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
    elif resolved.tts_engine != "chatterbox":
        raise ConfigurationError(
            f"TTS_ENGINE must be one of {', '.join(SUPPORTED_ENGINES)}, got "
            f"{resolved.tts_engine!r}"
        )

    if single_voice:
        # A single-voice engine has no use for voice profiles, and requiring them
        # would mean building identities the engine cannot honour.
        profiles: Mapping[str, VoiceProfile] = {}
        speech = _validate_audio_file(speech_stem, label="the speech stem")
        _read_audio_info(speech, label="the speech stem")
        performance = None
        style = None
    else:
        profiles = _validate_profiles(
            voice_profiles, speakers=[line.speaker_id for line in lines]
        )
        stem = _validate_audio_file(speech_stem, label="the BandIt speech stem")
        _read_audio_info(stem, label="the BandIt speech stem")

        # Validate the reference audio of every speaker that is about to be used
        # before any model runs: a missing reference should fail the run the same way
        # whether or not earlier lines could already be synthesized.
        for speaker_id in dict.fromkeys(line.speaker_id for line in lines):
            reference = profiles[speaker_id].resolve_reference_audio()
            _validate_audio_file(
                reference, label=f"the voice reference of speaker {speaker_id!r}"
            )

        if cloning_engine is not None:
            # A cloning engine takes the character's reference and speaks as it, so
            # there is no performance prompt to cut and no conversion to run. The
            # prompt bounds below describe the Chatterbox prompt, so they are not
            # validated here - a run that cannot use them should not fail on them.
            performance = None
            style = None
        else:
            minimum = _positive(
                "TTS_PERFORMANCE_REFERENCE_MIN_DURATION",
                resolved.tts_performance_reference_min_duration,
            )
            maximum = _positive(
                "TTS_PERFORMANCE_REFERENCE_MAX_DURATION",
                resolved.tts_performance_reference_max_duration,
            )
            if maximum < minimum:
                raise ConfigurationError(
                    f"TTS_PERFORMANCE_REFERENCE_MAX_DURATION ({maximum}) must not be "
                    f"smaller than TTS_PERFORMANCE_REFERENCE_MIN_DURATION ({minimum})"
                )
            performance, style = _resolve_engines(
                settings=resolved,
                performance_engine=performance_engine,
                style_engine=style_engine,
            )

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
    elif engine is not None:
        artifact_model = f"{engine.name}:{resolved.tts_model}"
    else:
        artifact_model = artifact_model_name("chatterbox", settings=resolved)

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
    for index, line in enumerate(lines):
        reason = _skip_reason(line, min_line_seconds=min_line_seconds)
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

        digest = _line_digest(line, model=artifact_model)
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

            if cloning_engine is not None:
                clips.append(
                    _synthesize_line_cloned(
                        index,
                        line,
                        name=_line_artifact_name(line, model=artifact_model),
                        directories=directories,
                        engine=cloning_engine,
                        profile=profiles[line.speaker_id],
                        settings=resolved,
                        pacing=pacing,
                    )
                )
                continue

            clips.append(
                _synthesize_line(
                    index,
                    line,
                    name=_line_artifact_name(line, model=artifact_model),
                    digest=digest,
                    stem=stem,
                    profile=profiles[line.speaker_id],
                    directories=directories,
                    performance_engine=performance,
                    style_engine=style,
                    settings=resolved,
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

    return SynthesisResult(clips=tuple(clips), skipped=tuple(skipped), pacing=pacing)


def synthesize_dialogue(
    dialogue: Iterable[AdaptedDialogue],
    speech_stem: str | Path,
    voice_profiles: Mapping[str, VoiceProfile],
    *,
    settings: Settings | None = None,
    output_dir: str | Path | None = None,
    performance_engine: ChatterboxPerformanceEngine | None = None,
    style_engine: VoiceConversionEngine | None = None,
) -> list[TtsClip]:
    """Synthesize every dubbable line and return the clips.

    The clips alone, in the order the lines were given. Lines the stage could not
    voice - a window too short to hold a word, text with nothing to pronounce, or a
    line an engine failed on - are absent rather than raising; call
    :func:`synthesize_dialogue_detailed` when the reason for each one is wanted,
    which is what the orchestrator does so the manifest can record them.
    """

    return list(
        synthesize_dialogue_detailed(
            dialogue,
            speech_stem,
            voice_profiles,
            settings=settings,
            output_dir=output_dir,
            performance_engine=performance_engine,
            style_engine=style_engine,
        ).clips
    )


__all__ = [
    "AMHARIC_LOADER_FILENAME",
    "BASE_CFG_WEIGHT",
    "BASE_EXAGGERATION",
    "BASE_TEMPERATURE",
    "CFG_WEIGHT_BOUNDS",
    "CHATTERBOX_SAMPLE_RATE",
    "CLIP_CEILING",
    "EXAGGERATION_BOUNDS",
    "MINIMUM_SPEAKABLE_LINE_SECONDS",
    "MINIMUM_SPEAKABLE_SYLLABLES",
    "PerformanceControls",
    "SEED_VC_CONVERT_STYLE",
    "SEED_VC_ENGINE_NAME",
    "SYNTHESIS_RECIPE_VERSION",
    "TEMPERATURE_BOUNDS",
    "TTS_DIRECTORY_NAME",
    "ChatterboxAmharicEngine",
    "ChatterboxPerformanceEngine",
    "ConfigurationError",
    "ConversionError",
    "EngineLoadError",
    "InvalidAudioError",
    "InvalidDialogueError",
    "InvalidEngineError",
    "InvalidInputError",
    "InvalidVoiceProfileError",
    "MissingInputError",
    "MissingOutputError",
    "MissingVoiceProfileError",
    "SeedVcV2Engine",
    "MmsAmharicEngine",
    "OmniVoiceEngine",
    "VoiceCloningEngine",
    "SkippedLine",
    "SynthesisError",
    "SynthesisResult",
    "TtsClip",
    "TtsError",
    "VoiceConversionEngine",
    "TextToSpeechEngine",
    "engine_needs_profiles",
    "artifact_model_name",
    "extract_performance_reference",
    "load_chatterbox_engine",
    "load_mms_engine",
    "load_omnivoice_engine",
    "load_seed_vc_engine",
    "load_tts_engine",
    "SUPPORTED_ENGINES",
    "PROFILE_ENGINES",
    "pacing_plan_for",
    "requested_speaking_rate",
    "reset_engine_cache",
    "seed_vc_revision",
    "resolve_tts_directory",
    "synthesize_dialogue",
    "synthesize_dialogue_detailed",
]
