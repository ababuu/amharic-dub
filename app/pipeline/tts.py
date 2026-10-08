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
import json
import math
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from app.config import (
    DEFAULT_SEED_VC_CONVERT_STYLE,
    DEFAULT_SEED_VC_DIFFUSION_STEPS,
    DEFAULT_TTS_MIN_LINE_SECONDS,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION,
    DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION,
    Settings,
    get_settings,
)
from app.pipeline.amharic_text import count_syllables
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
#: syllable, so this counts the script's own unit: text with nothing to pronounce
#: produces no speech, and asking the engine for it is what fails.
MINIMUM_SPEAKABLE_SYLLABLES = 1

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


def _write_mono(path: Path, samples: np.ndarray, sample_rate: int, *, label: str) -> Path:
    """Write a mono PCM-16 WAV and return ``path``."""

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(path),
            np.clip(samples, -1.0, 1.0).astype(np.float32),
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
    take_path: Path
    performance_reference_path: Path
    voice_reference_path: Path
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
            ("performance_reference_path", self.performance_reference_path),
            ("voice_reference_path", self.voice_reference_path),
        ):
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
            "performance_reference_path": portable_path(self.performance_reference_path),
            "voice_reference_path": portable_path(self.voice_reference_path),
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
        model: str = DEFAULT_TTS_MODEL,
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


# ---------------------------------------------------------------------------
# Engine cache
# ---------------------------------------------------------------------------

_CHATTERBOX_ENGINES: dict[tuple[str, str], ChatterboxAmharicEngine] = {}
#: Keyed by device, checkout and diffusion steps *and* the style flag, because two
#: settings that differ only in that flag need two different engines.
_SEED_VC_ENGINES: dict[tuple[str, str, int, bool], SeedVcV2Engine] = {}


def load_chatterbox_engine(*, settings: Settings | None = None) -> ChatterboxAmharicEngine:
    """Return the process-wide Chatterbox Amharic engine for these settings.

    Building an engine is cheap - the model itself is only loaded on the first
    :meth:`ChatterboxAmharicEngine.synthesize` call - but the engine is cached
    anyway, so a stage run and every line in it share one instance and one model.
    """

    resolved = settings if settings is not None else get_settings()
    key = (resolved.device, resolved.tts_model)
    engine = _CHATTERBOX_ENGINES.get(key)
    if engine is None:
        engine = ChatterboxAmharicEngine(
            model=resolved.tts_model,
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


def reset_engine_cache() -> None:
    """Drop every cached engine and loader, releasing the loaded models.

    An engine is useless once its model is gone, so the cache is cleared as a
    whole. The next call to :func:`load_chatterbox_engine` or
    :func:`load_seed_vc_engine` starts from a clean slate.
    """

    _CHATTERBOX_ENGINES.clear()
    _SEED_VC_ENGINES.clear()
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
    """The four artifact directories of one TTS run."""

    performance: Path
    chatterbox: Path
    converted: Path
    clips: Path

    @classmethod
    def under(cls, base: Path) -> "_TtsDirectories":
        return cls(
            performance=base / PERFORMANCE_DIRECTORY_NAME,
            chatterbox=base / CHATTERBOX_DIRECTORY_NAME,
            converted=base / CONVERTED_DIRECTORY_NAME,
            clips=base / CLIP_DIRECTORY_NAME,
        )

    def create(self) -> None:
        """Create every directory, creating parents as needed."""

        for directory in (self.performance, self.chatterbox, self.converted, self.clips):
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
    if count_syllables(line.amharic) < MINIMUM_SPEAKABLE_SYLLABLES:
        return (
            f"{UNSPEAKABLE_REASON_PREFIX}: the Amharic line has no syllables to "
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
    voice_profiles: Mapping[str, VoiceProfile],
    *,
    output_dir: str | Path | None = None,
    settings: Settings | None = None,
    performance_engine: ChatterboxPerformanceEngine | None = None,
    style_engine: VoiceConversionEngine | None = None,
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
        _validate_audio_file(reference, label=f"the voice reference of speaker {speaker_id!r}")

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
            f"TTS_PERFORMANCE_REFERENCE_MAX_DURATION ({maximum}) must not be smaller "
            f"than TTS_PERFORMANCE_REFERENCE_MIN_DURATION ({minimum})"
        )
    _positive("TTS_MAX_PAUSE_SECONDS", resolved.tts_max_pause_seconds)
    min_line_seconds = _positive(
        "TTS_MIN_LINE_SECONDS", resolved.tts_min_line_seconds
    )
    continue_on_failure = bool(resolved.tts_continue_on_failure)

    performance, style = _resolve_engines(
        settings=resolved,
        performance_engine=performance_engine,
        style_engine=style_engine,
    )

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

        digest = _line_digest(line, model=resolved.tts_model)
        try:
            clips.append(
                _synthesize_line(
                    index,
                    line,
                    name=_line_artifact_name(line, model=resolved.tts_model),
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

    return SynthesisResult(clips=tuple(clips), skipped=tuple(skipped))


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
    "SkippedLine",
    "SynthesisError",
    "SynthesisResult",
    "TtsClip",
    "TtsError",
    "VoiceConversionEngine",
    "extract_performance_reference",
    "load_chatterbox_engine",
    "load_seed_vc_engine",
    "reset_engine_cache",
    "seed_vc_revision",
    "resolve_tts_directory",
    "synthesize_dialogue",
    "synthesize_dialogue_detailed",
]
