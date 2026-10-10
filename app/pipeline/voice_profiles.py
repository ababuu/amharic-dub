"""Per-speaker voice identities: reference selection and clone-prompt cache.

This module sits between dialogue adaptation and TTS, and answers exactly one
question per speaker::

    "What should this character sound like?"

It picks the cleanest continuous stretch of that speaker's own dialogue out of
the separated dialogue stem, cuts it into a canonical voice-cloning reference,
and records where an optionally cached voice-clone prompt for that reference
lives::

    [SpeakerSegment] + speech.wav  ->  build_voice_profiles()
        -> {"SPEAKER_00": VoiceProfile(...), ...}

Voice identity vs. performance
------------------------------
The pipeline keeps the two concepts apart on purpose:

* :class:`~app.pipeline.translation.AdaptedDialogue` answers *how* a particular
  line is performed - emotion, intensity, delivery, pauses.
* :class:`VoiceProfile` answers *what the character's voice is* - nothing else.

One character has to be able to sound calm, furious or frightened while staying
recognizably the same character, so emotion is deliberately **not** part of a
voice identity and never becomes a permanent field of a profile.

Reference selection
-------------------
The reference is not "the first 10 seconds this speaker says". For every speaker
the component builds candidate windows out of their own turns, measures them,
and keeps the best one:

* **Speaker confidence.** Candidate windows only ever come from that speaker's
  own diarized turns, so a reference is never taken from another speaker.
* **Continuous speech.** Turns by the same speaker separated by no more than
  :data:`CONTIGUOUS_GAP_SECONDS` form one run. Anything else (a different
  speaker interjecting) breaks the run, so unrelated speech is never stitched
  together just to reach the target length.
* **Speech presence.** A frame counts as active only when it rises
  :data:`ACTIVITY_MARGIN_DB` above the window's *own* noise floor, estimated from
  its quietest frames. Steady material - music, tone, hum, bleed - stays close to
  its own floor whatever level it sits at and yields almost no active frames,
  while speech, which is bursty, yields many. A window with virtually no active
  frames is rejected outright, so a loud music-only window no longer outranks a
  quieter window that actually contains dialogue.
* **Duration.** The score rewards windows close to
  ``VOICE_REFERENCE_TARGET_DURATION`` and rejects anything shorter than
  ``VOICE_REFERENCE_MIN_DURATION``. A long run is scanned with a sliding window
  rather than used whole, because the best 10 seconds inside 40 seconds of
  shouting is often not at the top.
* **Dynamics and loudness.** Speech has a wide dynamic range between its loud and
  quiet frames; a window that is flat, mostly silence, or heavily clipped scores
  lower.
* **Overlap.** Windows overlapping another speaker's turns score lower, and a
  window that contains **simultaneous speech** is rejected outright. Diarization
  reports those regions separately, and a reference cut from one is a recording of
  two people: cloning from it teaches the model both voices, which is heard as a
  fragment of the wrong character at the start of a line.
* **The text has to match the audio.** A reference is only usable when the words
  that will be sent with it describe exactly the audio that will be sent. The
  reference is therefore cut on the speaker's own line boundaries rather than on a
  fixed grid: a window that begins or ends mid-line contains speech no transcript
  accounts for, and a model given an audio prompt longer than its own transcript
  can speak the difference.
* **Phonetic variety.** When a transcript covers the window it contributes a
  modest orthographic proxy for how much of the speaker's symbol and word
  inventory the window exercises. Without a transcript the term is dropped and
  the other weights are renormalised - nothing is guessed.
* A clean 5 second reference beats a contaminated 10 second one. Ties are broken
  deterministically by the earliest window, so the same input always selects the
  same reference.

Speaker ids and the filesystem
------------------------------
``speaker_id`` is preserved exactly as diarization reported it - in the profile
and in the JSON - because it is the application's identity for a character. Only
its use as a *directory name* is constrained, by :func:`speaker_directory_name`,
so an unusual or hostile label cannot place a reference outside
``VOICE_PROFILE_DIR``.

What cannot be judged without a model
-------------------------------------
* Music or effects that stay **audible above** the dialogue still pass the
  activity test: an energy VAD cannot separate two simultaneous sounds, and dense
  music that modulates like speech will be accepted.
* Whether a window is laughter, shouting, whispering or singing rather than
  normal speech cannot be told from levels or from a transcript.
* The variety signal is **orthographic, not phonetic** - it counts distinct
  symbols and words rather than phonemes. That is a decent proxy for Amharic's
  syllabic script and a weaker one for a language written with a small alphabet.
* Dropping the variety term when a window has no transcript coverage gives such a
  window a small advantage over a covered one with repetitive text. The term is
  deliberately modest (0.10) so it cannot outweigh a real quality difference.

All of these need validation with real cloned-voice weights on RunPod.

Contract with the ``ffmpeg`` CLI
--------------------------------
* Extraction is a single subprocess call, ``ffmpeg -ss <start> -t <duration>
  -i <stem> ...``, which is why the component needs FFmpeg on ``PATH``. The stem
  is analysed by seeking, never loaded whole, because a dialogue stem for a
  feature film is larger than memory.
* The reference is cut to mono 24 kHz PCM WAV (``-ac 1 -ar 24000 -c:a
  pcm_s16le``) and only ever attenuated (``-af volume=<negative>dB``) towards
  :data:`REFERENCE_PEAK_DBFS`, never amplified, so no processing alters the
  speaker's identity.

Deliberate non-goals
--------------------
* **No TTS import.** Generating speech belongs to :mod:`app.pipeline.tts`; this
  module only reserves the prompt path and calls the injected encoder. Nothing in
  the pipeline bundles an encoder either - the TTS stage clones from reference
  audio directly - so a prompt is written only when the caller asks for one.
* **No second voice-cloning system**, no speaker embeddings of its own, and no
  character names - speaker ids stay exactly as diarization produced them.
* **No denoising, EQ, reverb removal or pitch shifting.** Identity preservation
  matters more than a superficially cleaner clip.
* **No model downloads at import time** (and none in the test suite).
* **No audio arrays in a profile.** A profile is paths plus light metadata, so it
  stays JSON-serializable.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import numpy as np
import soundfile as sf

from app.config import PROJECT_ROOT, Settings, get_settings
from app.pipeline.diarization import CrosstalkRegion, SpeakerSegment
from app.pipeline.transcription import TranscriptSegment

#: A voice-clone encoder: ``(reference_audio, destination) -> prompt path``.
#: It must write the serialized prompt to ``destination``. No encoder is bundled:
#: :mod:`app.pipeline.tts` clones from reference audio directly, so this hook is
#: only used when a caller injects one, which keeps profile selection testable
#: without any cloning model.
ClonePromptEncoder = Callable[[Path, Path], Path]

#: Canonical format of a stored voice reference, as the cloning models expect it.
REFERENCE_SAMPLE_RATE = 24_000
REFERENCE_CHANNELS = 1
REFERENCE_FILENAME = "reference.wav"
CLONE_PROMPT_FILENAME = "voice_clone.pt"
PROFILE_FILENAME = "profile.json"

#: Voice cloning stays usable anywhere in this absolute range, whatever the
#: configurable window around it says. It is the hard bound a
#: :class:`VoiceProfile` can never violate, so a hand-written profile file
#: cannot describe a reference no cloning model could use.
MIN_REFERENCE_DURATION = 1.0
MAX_REFERENCE_DURATION = 30.0

#: Two turns by the same speaker closer together than this are one continuous
#: stretch of speech. A different speaker always breaks the run, so this only
#: ever joins a speaker with themself.
CONTIGUOUS_GAP_SECONDS = 0.2

#: Hop between the sliding windows cut out of a stretch longer than the target.
WINDOW_STRIDE_SECONDS = 2.0

#: Level analysis resolution: short enough to see the gaps between words, long
#: enough to ignore single-sample noise.
ANALYSIS_FRAME_SECONDS = 0.02

#: Speech-presence (VAD) parameters. A frame counts as *active* when it rises
#: :data:`ACTIVITY_MARGIN_DB` above the noise floor of the window it belongs to,
#: which is estimated from that window's quietest frames. Tracking the local
#: floor is what separates this from a fixed level threshold: steady material
#: stays close to its own floor however loud it is, while speech is bursty.
ACTIVITY_MARGIN_DB = 10.0
NOISE_FLOOR_PERCENTILE = 15.0
PEAK_PERCENTILE = 95.0

#: Below this frame level there is only noise, whatever the local floor says.
#: Without it, a window of digital silence would look uniformly "active".
SILENCE_FLOOR_DBFS = -70.0

#: The share of a window that must be active for the window to be usable at all.
MIN_SPEECH_RATIO = 0.05

#: Share of a window that naturally carries speech. Dialogue is bursty - roughly
#: this much of a window sits above the noise floor, the rest is the gaps between
#: syllables, words and lines - so coverage saturates here.
SPEECH_COVERAGE_TARGET = 0.6

#: Dynamic range (95th - 15th percentile frame level) of speech-like audio, in dB.
MIN_DYNAMIC_RANGE_DB = 12.0
GOOD_DYNAMIC_RANGE_DB = 25.0

#: Loudness ramp: at or below the quiet end there is nothing worth cloning, at
#: the loud end the window is as good as this measure can tell.
QUIET_RMS_DBFS = -45.0
GOOD_RMS_DBFS = -30.0

#: Sample magnitude treated as clipped, and the clipped fraction at which the
#: clipping term of the score reaches zero.
CLIPPING_THRESHOLD = 0.999
CLIPPING_TOLERANCE = 0.01

#: Peak the extracted reference is attenuated to. Never boosted.
REFERENCE_PEAK_DBFS = -1.0

#: Orthographic targets of the phonetic-variety proxy: distinct symbols and
#: distinct words a reference should exercise, and the number of words below
#: which a window simply has not said enough to characterise a speaker.
VARIETY_LETTER_TARGET = 15.0
VARIETY_WORD_TARGET = 10.0
VARIETY_MIN_WORDS = 4.0

#: Shares of the variety proxy (symbol coverage, word coverage).
VARIETY_LETTER_PART = 0.6
VARIETY_WORD_PART = 0.4

#: Weights of the reference score. The six audio terms sum to 0.90 and the
#: transcript-driven variety term adds 0.10 for a maximum of 1.0. When no
#: transcript covers a window the variety term is dropped and the remaining
#: weights are renormalised, so the score always stays within 0..1.
SCORE_DURATION_WEIGHT = 0.30
SCORE_SPEECH_WEIGHT = 0.20
SCORE_DYNAMICS_WEIGHT = 0.10
SCORE_LOUDNESS_WEIGHT = 0.15
SCORE_OVERLAP_WEIGHT = 0.10
SCORE_CLIPPING_WEIGHT = 0.05
SCORE_VARIETY_WEIGHT = 0.10

#: Timestamps closer than this are treated as equal when deciding whether a
#: transcript line lies inside the selected window.
TIMESTAMP_TOLERANCE = 1e-3

#: Tolerance on the duration of the extracted reference, to absorb the
#: sample-rounding of the ``-ss``/``-t`` seek pair.
DURATION_TOLERANCE = 0.05


class VoiceProfileError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidInputError(VoiceProfileError, ValueError):
    """The supplied input is unusable (bad transcript, missing directory)."""


class InvalidSegmentError(VoiceProfileError, ValueError):
    """A diarization segment supplied to this module is not a SpeakerSegment."""


class ConfigurationError(VoiceProfileError, ValueError):
    """A voice-profile setting is unusable."""


class InvalidProfileError(VoiceProfileError, ValueError):
    """A voice profile has an empty field or an out-of-range value."""


class MissingInputError(VoiceProfileError):
    """The source audio file does not exist."""


class NoUsableReferenceError(VoiceProfileError):
    """No usable reference was found for one of the speakers."""


class AudioAnalysisError(VoiceProfileError):
    """The dialogue stem could not be opened or measured."""


class ReferenceExtractionError(VoiceProfileError):
    """FFmpeg could not cut the reference out of the dialogue stem."""


class ReferencePreprocessingError(ReferenceExtractionError):
    """The extracted reference is not in the required mono 24 kHz PCM format."""


class ClonePromptError(VoiceProfileError):
    """The voice-clone prompt could not be created or cached."""


class ProfilePersistenceError(VoiceProfileError):
    """A voice profile could not be written to or read from disk."""


def _finite_number(name: str, value: object, error: type[Exception]) -> float:
    """Return ``value`` as a finite ``float``, or fail with ``error``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{name} must be a number, got {type(value).__name__}")

    number = float(value)
    if not math.isfinite(number):
        raise error(f"{name} must be finite, got {value!r}")
    return number


def _profile_path(name: str, value: object) -> Path:
    """Return ``value`` as a usable file :class:`~pathlib.Path`.

    Existence is deliberately not required: a profile may be loaded on a machine
    that has not extracted its references yet, and the build path always writes
    the reference before constructing the profile.
    """

    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value.strip())
    else:
        raise InvalidProfileError(f"{name} must be a path or a non-empty path string")

    if not path.name:
        raise InvalidProfileError(f"{name} must not be an empty path")
    if path.is_dir():
        raise InvalidProfileError(f"{name} must be a file path, not a directory: {path}")
    return path


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Return ``value`` limited to the ``low``..``high`` range."""

    return max(low, min(high, value))


def _dbfs(amplitude: Any) -> Any:
    """Return ``amplitude`` (linear, 0..1) in dBFS, flooring silence."""

    return 20.0 * np.log10(np.maximum(np.asarray(amplitude, dtype=np.float64), 1e-10))


#: Anything that is not a plain, portable filename character.
_UNSAFE_DIRECTORY_CHARACTERS = re.compile(r"[^A-Za-z0-9._-]+")

#: Names Windows refuses to use for a file or directory, whatever the extension.
_RESERVED_DIRECTORY_NAMES = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
)


def speaker_directory_name(speaker_id: str) -> str:
    """Return the directory name used to store one speaker's profile.

    Speaker ids are *not* rewritten: diarization owns the identity of a character
    and renaming it would break the link between a profile and the dialogue it
    belongs to. They are also used as a single filesystem component though, and a
    label must never be able to place a file outside ``VOICE_PROFILE_DIR``. This
    therefore reduces the id to one safe component:

    * anything outside ``[A-Za-z0-9._-]`` (separators, ``:``, spaces, control
      characters, ...) becomes ``_``;
    * leading and trailing dots are dropped, so ``.`` and ``..`` cannot survive
      as a directory name;
    * a name Windows reserves is never used as-is.

    Whenever the name actually had to change, a short digest of the original id
    is appended. That keeps the mapping injective: two different speaker ids can
    never end up sharing one profile directory. A plain ``SPEAKER_00`` is
    therefore stored under ``SPEAKER_00``, exactly as before.
    """

    if not isinstance(speaker_id, str) or not speaker_id.strip():
        raise InvalidInputError("speaker_id must be a non-empty string")

    name = _UNSAFE_DIRECTORY_CHARACTERS.sub("_", speaker_id.strip())
    name = name.strip(".").strip()
    if not name:
        name = "speaker"

    if name != speaker_id or name.upper() in _RESERVED_DIRECTORY_NAMES:
        digest = hashlib.sha1(speaker_id.encode("utf-8")).hexdigest()[:8]
        name = f"{name}-{digest}"
    return name


def _speaker_directory(directory: Path, speaker_id: str) -> Path:
    """Return the profile directory of ``speaker_id``, inside ``directory``."""

    name = speaker_directory_name(speaker_id)
    # Defence in depth: the sanitised name can never contain a separator, so this
    # fails loudly if a future change ever lets one through.
    if name in {".", ".."} or len(PurePath(name).parts) != 1:
        raise InvalidInputError(
            f"unsafe speaker directory name for {speaker_id!r}: {name!r}"
        )
    return Path(directory) / name


@dataclass(frozen=True, slots=True)
class VoiceProfile:
    """The reusable voice identity of one diarized speaker.

    A profile never contains audio: only the path of the extracted reference, the
    window inside the dialogue stem it came from, the transcript of that window
    when one is known, and the optionally cached voice-clone prompt. It also
    never contains performance metadata - emotion, intensity and delivery belong
    to a later stage and change per line, while this identity must stay stable.
    """

    speaker_id: str
    reference_audio: Path
    reference_start: float
    reference_end: float
    reference_text: str | None = None
    clone_prompt_path: Path | None = None
    quality_score: float = 0.0
    selection_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise InvalidProfileError("speaker_id must be a non-empty string")

        object.__setattr__(
            self, "reference_audio", _profile_path("reference_audio", self.reference_audio)
        )

        start = _finite_number("reference_start", self.reference_start, InvalidProfileError)
        end = _finite_number("reference_end", self.reference_end, InvalidProfileError)
        if start < 0:
            raise InvalidProfileError(f"reference_start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidProfileError(
                f"reference_end must be greater than reference_start ({start} seconds), got {end}"
            )

        duration = end - start
        if not MIN_REFERENCE_DURATION <= duration <= MAX_REFERENCE_DURATION:
            raise InvalidProfileError(
                f"the reference for {self.speaker_id!r} is {duration:.3f}s long; voice cloning "
                f"needs between {MIN_REFERENCE_DURATION:g} and {MAX_REFERENCE_DURATION:g} seconds"
            )

        object.__setattr__(self, "reference_start", start)
        object.__setattr__(self, "reference_end", end)

        text = self.reference_text
        if text is not None:
            if not isinstance(text, str):
                raise InvalidProfileError("reference_text must be a string or None")
            # An empty or whitespace-only transcript is "no text", never ""
            text = text.strip() or None
        object.__setattr__(self, "reference_text", text)

        prompt = self.clone_prompt_path
        if prompt is not None:
            prompt = _profile_path("clone_prompt_path", prompt)
        object.__setattr__(self, "clone_prompt_path", prompt)

        score = _finite_number("quality_score", self.quality_score, InvalidProfileError)
        if not 0.0 <= score <= 1.0:
            raise InvalidProfileError(f"quality_score must be between 0 and 1, got {score}")
        object.__setattr__(self, "quality_score", score)

        if not isinstance(self.selection_reason, str):
            raise InvalidProfileError("selection_reason must be a string")
        object.__setattr__(self, "selection_reason", self.selection_reason.strip())

    @property
    def reference_duration(self) -> float:
        """Length of the reference window in seconds.

        Derived from the timestamps rather than stored, so a profile can never
        disagree with itself about how long its reference is.
        """

        return self.reference_end - self.reference_start

    def resolve_reference_audio(self) -> Path:
        """Return the reference audio path, absolute if it was stored relative."""

        return resolve_project_path(self.reference_audio)

    def resolve_clone_prompt(self) -> Path | None:
        """Return the cached clone-prompt path, absolute if stored relative."""

        if self.clone_prompt_path is None:
            return None
        return resolve_project_path(self.clone_prompt_path)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the profile, with portable paths."""

        return {
            "speaker_id": self.speaker_id,
            "reference_audio": portable_path(self.reference_audio),
            "reference_start": self.reference_start,
            "reference_end": self.reference_end,
            "reference_text": self.reference_text,
            "clone_prompt_path": (
                None
                if self.clone_prompt_path is None
                else portable_path(self.clone_prompt_path)
            ),
            "quality_score": self.quality_score,
            "selection_reason": self.selection_reason,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "VoiceProfile":
        """Rebuild a profile from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise InvalidProfileError("a voice profile must be a JSON object")

        required = ("speaker_id", "reference_audio", "reference_start", "reference_end")
        missing = [name for name in required if name not in payload]
        if missing:
            raise InvalidProfileError(
                "voice profile is missing " + ", ".join(sorted(missing))
            )

        text = payload.get("reference_text")
        prompt = payload.get("clone_prompt_path")
        return cls(
            speaker_id=payload["speaker_id"],  # type: ignore[arg-type]
            reference_audio=resolve_project_path(
                payload["reference_audio"]  # type: ignore[arg-type]
            ),
            reference_start=payload["reference_start"],  # type: ignore[arg-type]
            reference_end=payload["reference_end"],  # type: ignore[arg-type]
            reference_text=None if text is None else text,  # type: ignore[arg-type]
            clone_prompt_path=(
                None
                if prompt is None
                else resolve_project_path(prompt)  # type: ignore[arg-type]
            ),
            quality_score=payload.get("quality_score", 0.0),  # type: ignore[arg-type]
            selection_reason=payload.get("selection_reason", ""),  # type: ignore[arg-type]
        )


def portable_path(path: str | Path) -> str:
    """Return ``path`` in a form that survives a move to another machine.

    Files inside the project are stored relative to the project root and always
    with forward slashes, so a profile written on Windows stays usable on Linux.
    Anything outside the project keeps its absolute path.
    """

    candidate = Path(path)
    absolute = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    try:
        return absolute.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return absolute.as_posix()


def resolve_project_path(path: str | Path) -> Path:
    """Return ``path`` absolute, resolving project-relative paths."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def resolve_voice_profile_dir(
    *,
    directory: str | Path | None = None,
    settings: Settings | None = None,
) -> Path:
    """Return the directory holding the per-speaker voice profiles."""

    if directory is not None:
        return Path(directory)
    resolved = settings if settings is not None else get_settings()
    return Path(resolved.voice_profile_dir)


def save_voice_profiles(
    profiles: Mapping[str, VoiceProfile],
    *,
    directory: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Path]:
    """Write one ``profile.json`` per speaker and return the written paths.

    The profiles are keyed by speaker id, exactly as
    :func:`build_voice_profiles` returns them.
    """

    base = resolve_voice_profile_dir(directory=directory, settings=settings)
    written: dict[str, Path] = {}

    for key, profile in profiles.items():
        if key != profile.speaker_id:
            raise InvalidProfileError(
                f"voice profile {profile.speaker_id!r} is stored under the key {key!r}"
            )

        target = base / speaker_directory_name(profile.speaker_id) / PROFILE_FILENAME
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(profile.to_dict(), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            raise ProfilePersistenceError(
                f"could not write the voice profile for speaker {profile.speaker_id!r}: {exc}"
            ) from exc
        written[profile.speaker_id] = target

    return written


def load_voice_profiles(
    *,
    directory: str | Path | None = None,
    settings: Settings | None = None,
) -> dict[str, VoiceProfile]:
    """Read every ``profile.json`` under the voice-profile directory."""

    base = resolve_voice_profile_dir(directory=directory, settings=settings)
    if not base.is_dir():
        raise InvalidInputError(f"voice profile directory not found: {base}")

    profiles: dict[str, VoiceProfile] = {}
    for child in sorted(base.iterdir()):
        manifest = child / PROFILE_FILENAME
        if not child.is_dir() or not manifest.is_file():
            continue

        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfilePersistenceError(f"could not read {manifest}: {exc}") from exc

        profile = VoiceProfile.from_dict(payload)
        profiles[profile.speaker_id] = profile

    return profiles


@dataclass(frozen=True, slots=True)
class _WindowStats:
    """Level measurements of one candidate window."""

    speech_ratio: float
    dynamic_range_db: float
    rms_dbfs: float
    peak_dbfs: float
    clipped_fraction: float


@dataclass(frozen=True, slots=True)
class _ReferenceCandidate:
    """One window of a single speaker's own speech, with its measurements."""

    speaker_id: str
    start: float
    end: float
    stats: _WindowStats
    overlap_seconds: float
    text: str | None = None
    variety: float | None = None
    #: ``True`` when the window shares time with simultaneous speech, so the
    #: recording contains more than one voice.
    contaminated: bool = False

    @property
    def duration(self) -> float:
        """Length of the window in seconds."""

        return self.end - self.start

    def is_usable(self, min_duration: float) -> bool:
        """Return whether the window is long enough and carries actual speech."""

        return self.duration >= min_duration and self.stats.speech_ratio >= MIN_SPEECH_RATIO

    def _terms(self, target_duration: float) -> list[tuple[float, float]]:
        """Return the ``(weight, score)`` pairs that apply to this window."""

        terms = [
            (SCORE_DURATION_WEIGHT, min(self.duration / target_duration, 1.0)),
            (
                SCORE_SPEECH_WEIGHT,
                _clamp(self.stats.speech_ratio / SPEECH_COVERAGE_TARGET),
            ),
            (
                SCORE_DYNAMICS_WEIGHT,
                _clamp(
                    (self.stats.dynamic_range_db - MIN_DYNAMIC_RANGE_DB)
                    / (GOOD_DYNAMIC_RANGE_DB - MIN_DYNAMIC_RANGE_DB)
                ),
            ),
            (
                SCORE_LOUDNESS_WEIGHT,
                _clamp(
                    (self.stats.rms_dbfs - QUIET_RMS_DBFS) / (GOOD_RMS_DBFS - QUIET_RMS_DBFS)
                ),
            ),
            (SCORE_OVERLAP_WEIGHT, 1.0 - min(self.overlap_seconds / self.duration, 1.0)),
            (
                SCORE_CLIPPING_WEIGHT,
                _clamp(1.0 - self.stats.clipped_fraction / CLIPPING_TOLERANCE),
            ),
        ]

        # Variety is only added when a transcript actually covers this window: it
        # is never guessed, and its weight is then spread over the terms that do
        # apply.
        if self.variety is not None:
            terms.append((SCORE_VARIETY_WEIGHT, _clamp(self.variety)))
        return terms

    def quality_score(self, target_duration: float) -> float:
        """Return the 0..1 score used to rank this window against its siblings."""

        terms = self._terms(target_duration)
        total_weight = sum(weight for weight, _ in terms)
        return sum(weight * score for weight, score in terms) / total_weight

    def selection_reason(self) -> str:
        """Explain, in one line, why this window was preferred."""

        reason = (
            f"{self.duration:.1f}s of continuous speech, "
            f"{self.stats.speech_ratio:.0%} active speech, "
            f"{self.stats.dynamic_range_db:.0f} dB dynamic range, "
            f"{self.overlap_seconds:.1f}s overlapped by other speakers, "
            f"{self.stats.rms_dbfs:.1f} dBFS, "
            f"{self.stats.clipped_fraction:.1%} clipped"
        )
        if self.variety is not None:
            reason += f", phonetic variety {self.variety:.2f}"
        if self.text is None:
            reason += ", no transcript describes this window (the engine will transcribe it)"
        if self.contaminated:
            reason += (
                ", WARNING: no clean window was available, so this reference contains "
                "simultaneous speech and the cloned voice may carry the other speaker"
            )
        return reason


class _StemReader:
    """Sequential access to a dialogue stem, without loading it whole.

    A stem for a feature film does not fit in memory comfortably, so candidates
    are measured by seeking to each window instead of reading the file once.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: Any | None = None
        self._sample_rate = 0

    def __enter__(self) -> "_StemReader":
        try:
            self._handle = sf.SoundFile(str(self._path))
        except (OSError, RuntimeError) as exc:
            raise AudioAnalysisError(
                f"could not open the dialogue stem {self._path}: {exc}"
            ) from exc

        self._sample_rate = int(self._handle.samplerate)
        if self._sample_rate <= 0:
            raise AudioAnalysisError(
                f"the dialogue stem {self._path} reports an unusable sample rate"
            )
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def sample_rate(self) -> int:
        """Sample rate of the stem in Hz."""

        return self._sample_rate

    def read(self, start: float, end: float) -> np.ndarray:
        """Return the mono samples between ``start`` and ``end`` seconds."""

        if self._handle is None:  # pragma: no cover - guarded by the context manager
            raise AudioAnalysisError("the dialogue stem is not open")

        first = max(0, int(round(start * self._sample_rate)))
        last = max(first, int(round(end * self._sample_rate)))

        try:
            self._handle.seek(first)
            frames = self._handle.read(last - first, dtype="float32", always_2d=True)
        except (OSError, RuntimeError) as exc:
            raise AudioAnalysisError(
                f"could not read {self._path} between {start:.3f}s and {end:.3f}s: {exc}"
            ) from exc

        # Analysis is mono; the reference itself is downmixed by FFmpeg.
        return np.asarray(frames, dtype=np.float64).mean(axis=1)


def _validate_segments(segments: Iterable[SpeakerSegment]) -> list[SpeakerSegment]:
    """Return ``segments`` as a list, rejecting anything that is not a segment."""

    if isinstance(segments, (str, bytes)) or not isinstance(segments, Iterable):
        raise InvalidSegmentError("speaker_segments must be an iterable of SpeakerSegment")

    items = list(segments)
    for item in items:
        if not isinstance(item, SpeakerSegment):
            raise InvalidSegmentError(
                "speaker_segments must contain SpeakerSegment objects, got "
                f"{type(item).__name__}"
            )
    return items


def _validate_transcript(
    transcript: Iterable[TranscriptSegment] | None,
) -> list[TranscriptSegment]:
    """Return ``transcript`` as a chronological list of transcript lines."""

    if transcript is None:
        return []

    if isinstance(transcript, (str, bytes)) or not isinstance(transcript, Iterable):
        raise InvalidInputError("transcript must be an iterable of TranscriptSegment or None")

    items = list(transcript)
    for item in items:
        if not isinstance(item, TranscriptSegment):
            raise InvalidInputError(
                "transcript must contain TranscriptSegment objects, got "
                f"{type(item).__name__}"
            )
    return sorted(items, key=lambda item: (item.start, item.end))


def _validate_audio_path(path: str | Path) -> Path:
    """Return ``path`` as a :class:`~pathlib.Path` pointing at an existing file."""

    source = Path(path)
    if not source.exists():
        raise MissingInputError(f"source audio file not found: {source}")
    if not source.is_file():
        raise InvalidInputError(f"source audio path is not a file: {source}")
    return source


def _resolve_duration_bounds(settings: Settings) -> tuple[float, float, float]:
    """Return the validated ``(minimum, target, maximum)`` reference durations."""

    minimum = _finite_number(
        "voice_reference_min_duration", settings.voice_reference_min_duration, ConfigurationError
    )
    target = _finite_number(
        "voice_reference_target_duration",
        settings.voice_reference_target_duration,
        ConfigurationError,
    )
    maximum = _finite_number(
        "voice_reference_max_duration", settings.voice_reference_max_duration, ConfigurationError
    )

    if not MIN_REFERENCE_DURATION <= minimum:
        raise ConfigurationError(
            "VOICE_REFERENCE_MIN_DURATION must be at least "
            f"{MIN_REFERENCE_DURATION:g} seconds, got {minimum}"
        )
    if target < minimum:
        raise ConfigurationError(
            f"VOICE_REFERENCE_TARGET_DURATION ({target}) must not be shorter than "
            f"VOICE_REFERENCE_MIN_DURATION ({minimum})"
        )
    if maximum < target:
        raise ConfigurationError(
            f"VOICE_REFERENCE_MAX_DURATION ({maximum}) must not be shorter than "
            f"VOICE_REFERENCE_TARGET_DURATION ({target})"
        )
    if maximum > MAX_REFERENCE_DURATION:
        raise ConfigurationError(
            "VOICE_REFERENCE_MAX_DURATION must be at most "
            f"{MAX_REFERENCE_DURATION:g} seconds, got {maximum}"
        )
    return minimum, target, maximum


def _group_by_speaker(segments: list[SpeakerSegment]) -> dict[str, list[SpeakerSegment]]:
    """Group ``segments`` by speaker id, each group sorted chronologically."""

    grouped: dict[str, list[SpeakerSegment]] = {}
    for segment in sorted(segments, key=lambda item: (item.start, item.end)):
        grouped.setdefault(segment.speaker_id, []).append(segment)
    return grouped


def _continuous_runs(segments: list[SpeakerSegment]) -> list[tuple[float, float]]:
    """Merge one speaker's turns that are not separated by a real pause."""

    runs: list[tuple[float, float]] = []
    start: float | None = None
    end = 0.0

    for segment in segments:
        if start is None:
            start, end = segment.start, segment.end
        elif segment.start - end <= CONTIGUOUS_GAP_SECONDS:
            end = max(end, segment.end)
        else:
            runs.append((start, end))
            start, end = segment.start, segment.end

    if start is not None:
        runs.append((start, end))
    return runs


def _windows(
    run: tuple[float, float],
    *,
    target: float,
    maximum: float,
) -> list[tuple[float, float]]:
    """Return the candidate windows to measure inside one continuous run.

    Only used when no transcript is available to align to - see
    :func:`_aligned_windows` for the case that matters. A window cut on a fixed
    grid cannot be described exactly by the lines inside it, and the TTS stage
    requires the transcript it sends to describe the audio it sends.
    """

    run_start, run_end = run
    duration = run_end - run_start

    # A run that already fits is used as it is: windowing a short clean run would
    # only risk cutting the very speech we want to keep.
    if duration <= maximum:
        return [(run_start, run_end)]

    length = min(target, duration)
    windows: list[tuple[float, float]] = []
    offset = run_start
    while offset + length <= run_end:
        windows.append((offset, offset + length))
        offset += WINDOW_STRIDE_SECONDS

    # A long monologue often ends on its best material, so always offer the tail.
    tail = (run_end - length, run_end)
    if not windows or abs(windows[-1][0] - tail[0]) > TIMESTAMP_TOLERANCE:
        windows.append(tail)

    return windows


def _aligned_windows(
    lines: list[TranscriptSegment],
    *,
    minimum: float,
    target: float,
    maximum: float,
) -> list[tuple[float, float]]:
    """Return candidate windows that begin and end on the speaker's own lines.

    A voice-cloning reference travels with its transcript, and the model is asked to
    speak the two together. That only works when they agree. Cutting the window on a
    fixed grid breaks that agreement at both edges: on the run that prompted this,
    the 10.0s window 20.0-30.0s contained 2.6s of the speaker's previous line and
    1.9s of their next one, neither of which the recorded text mentioned - so the
    prompt audio carried about 4.5s of speech with no transcript. A window taken from
    line boundaries cannot have that problem.

    ``lines`` must already belong to one speaker and be chronological. Windows are
    grown from each line in turn, so every candidate is a whole number of complete
    lines, and gaps between lines are included because that silence is part of the
    recording either way.
    """

    windows: list[tuple[float, float]] = []
    for first in range(len(lines)):
        start = lines[first].start
        for last in range(first, len(lines)):
            end = lines[last].end
            duration = end - start
            if duration > maximum:
                break
            if duration >= minimum:
                windows.append((start, end))
            if duration >= target:
                # Past the target the score only falls, so there is nothing to gain
                # by growing this window further; the next line starts a fresh one.
                break
    return windows


def _contaminated(start: float, end: float, regions: Iterable[tuple[float, float]]) -> bool:
    """Return whether ``start``..``end`` shares time with any of ``regions``."""

    for region_start, region_end in regions:
        if min(end, region_end) - max(start, region_start) > TIMESTAMP_TOLERANCE:
            return True
    return False


def _overlap_seconds(start: float, end: float, others: list[SpeakerSegment]) -> float:
    """Return how much of the window is shared with other speakers."""

    total = 0.0
    for other in others:
        shared = min(end, other.end) - max(start, other.start)
        if shared > 0.0:
            total += shared
    return total


def _frame_levels(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return the level in dBFS of every analysis frame of ``samples``."""

    frame_samples = max(1, int(round(ANALYSIS_FRAME_SECONDS * sample_rate)))
    frames = samples.size // frame_samples

    if not frames:
        return _dbfs(np.array([float(np.sqrt(np.mean(np.square(samples))))]))

    blocks = samples[: frames * frame_samples].reshape(frames, frame_samples)
    return _dbfs(np.sqrt(np.mean(np.square(blocks), axis=1)))


def _measure_window(samples: np.ndarray, sample_rate: int) -> _WindowStats:
    """Measure one window: activity, dynamics, loudness and clipping.

    The activity signal is a small energy VAD rather than a fixed level
    threshold. The noise floor is estimated from the quietest frames of the
    window *itself*, and a frame counts as active only when it rises
    :data:`ACTIVITY_MARGIN_DB` above that floor. Steady material - a music bed, a
    tone, hum or bleed - sits close to its own floor at any level, so it produces
    almost no active frames, while speech is bursty and produces many. That is
    what lets a window containing real dialogue outrank a louder window that
    contains none.
    """

    levels = _frame_levels(samples, sample_rate)
    noise_floor = float(np.percentile(levels, NOISE_FLOOR_PERCENTILE))
    peak = float(np.percentile(levels, PEAK_PERCENTILE))
    threshold = max(noise_floor + ACTIVITY_MARGIN_DB, SILENCE_FLOOR_DBFS)
    speech_ratio = float(np.mean(levels >= threshold))

    rms = float(np.sqrt(np.mean(np.square(samples))))
    loudest = float(np.max(np.abs(samples)))
    clipped_fraction = float(np.mean(np.abs(samples) >= CLIPPING_THRESHOLD))

    return _WindowStats(
        speech_ratio=speech_ratio,
        dynamic_range_db=max(0.0, peak - noise_floor),
        rms_dbfs=float(_dbfs(rms)),
        peak_dbfs=float(_dbfs(loudest)),
        clipped_fraction=clipped_fraction,
    )


def _candidates_for(
    speaker_id: str,
    own: list[SpeakerSegment],
    others: list[SpeakerSegment],
    *,
    reader: _StemReader,
    minimum: float,
    target: float,
    maximum: float,
    transcript: list[TranscriptSegment],
    contaminated: Iterable[tuple[float, float]] = (),
) -> list[_ReferenceCandidate]:
    """Build and measure every candidate window for one speaker.

    Two families of window are offered: ones cut on the speaker's own line
    boundaries, which a transcript can describe exactly, and the older sliding
    windows for a speaker whose lines are too short to reach the minimum on their
    own. Every candidate records whether it contains simultaneous speech; the choice
    between a clean and a contaminated window is made by the caller, so a film with
    no clean stretch still gets a voice instead of failing.
    """

    contaminated = list(contaminated)
    own_lines = sorted(
        (line for line in transcript if line.speaker_id == speaker_id),
        key=lambda line: (line.start, line.end),
    )

    spans: list[tuple[float, float]] = []
    if own_lines:
        spans.extend(
            _aligned_windows(own_lines, minimum=minimum, target=target, maximum=maximum)
        )
        # A run-based window is still worth offering for a speaker whose lines are
        # too short to reach the minimum on their own.
        for run in _continuous_runs(own):
            spans.extend(_windows(run, target=target, maximum=maximum))
    else:
        for run in _continuous_runs(own):
            spans.extend(_windows(run, target=target, maximum=maximum))

    candidates: list[_ReferenceCandidate] = []
    seen: set[tuple[float, float]] = set()
    for start, end in spans:
        key = (round(start, 4), round(end, 4))
        if key in seen:
            continue
        seen.add(key)
        samples = reader.read(start, end)
        if samples.size == 0:
            continue
        text = _window_text(speaker_id, start, end, transcript)
        covered = _covered_text(speaker_id, start, end, transcript)
        candidates.append(
            _ReferenceCandidate(
                speaker_id=speaker_id,
                start=start,
                end=end,
                stats=_measure_window(samples, reader.sample_rate),
                overlap_seconds=_overlap_seconds(start, end, others),
                text=text,
                variety=_phonetic_variety(covered or text),
                contaminated=_contaminated(start, end, contaminated),
            )
        )
    return candidates


def _no_usable_reference(
    speaker_id: str,
    candidates: list[_ReferenceCandidate],
    *,
    minimum: float,
) -> NoUsableReferenceError:
    """Explain why a speaker has no usable reference."""

    if not candidates:
        return NoUsableReferenceError(
            f"no usable voice reference for speaker {speaker_id!r}: the speaker has no "
            "diarized speech inside the dialogue stem"
        )

    longest = max(candidate.duration for candidate in candidates)
    active = max(candidate.stats.speech_ratio for candidate in candidates)

    reasons = [f"the longest continuous stretch of speech is {longest:.2f}s"]
    if longest < minimum:
        reasons.append(
            f"at least {minimum:.2f}s is required (VOICE_REFERENCE_MIN_DURATION)"
        )
    if active < MIN_SPEECH_RATIO:
        reasons.append(
            "no speech was detected in any candidate "
            "(the audio is steady, silent, or carries no speech louder than its own noise floor)"
        )

    return NoUsableReferenceError(
        f"no usable voice reference for speaker {speaker_id!r}: " + ", and ".join(reasons)
    )


#: A run of letters, i.e. a word, in any script. Digits and underscores are not
#: letters, so timestamps and markup never count as vocabulary.
_WORD_PATTERN = re.compile(r"[^\W\d_]+", re.UNICODE)


def _covered_text(
    speaker_id: str,
    start: float,
    end: float,
    transcript: list[TranscriptSegment],
) -> str | None:
    """Return the lines lying wholly inside ``start``..``end``, or ``None``.

    Used only for *scoring* a window: which lines a window happens to contain is a
    fair hint about how much of a speaker's range it exercises, even when the window
    cannot be described to the model exactly. What is *sent* is decided by
    :func:`_window_text`, which is stricter.
    """

    lines = [
        line.text.strip()
        for line in transcript
        if line.speaker_id == speaker_id
        and line.start >= start - TIMESTAMP_TOLERANCE
        and line.end <= end + TIMESTAMP_TOLERANCE
    ]
    joined = " ".join(line for line in lines if line)
    return joined or None


def _window_text(
    speaker_id: str,
    start: float,
    end: float,
    transcript: list[TranscriptSegment],
) -> str | None:
    """Return the transcript describing ``start``..``end`` exactly, or ``None``.

    ``None`` means "no transcript describes this window", and it is the safe answer:
    the TTS stage sends a reference transcript only when it has one, and OmniVoice
    transcribes the reference itself when it does not. What must never happen is
    sending a transcript that describes *part* of the audio, because the model is
    asked to speak the reference and the line together and can make up the
    difference.

    So a window is described only when the lines it covers begin exactly where it
    begins and end exactly where it ends. A window cut on a fixed grid does not, and
    gets ``None``.
    """

    lines = [
        line
        for line in transcript
        if line.speaker_id == speaker_id
        and line.end > start + TIMESTAMP_TOLERANCE
        and line.start < end - TIMESTAMP_TOLERANCE
    ]
    lines.sort(key=lambda line: (line.start, line.end))
    if not lines:
        return None

    if abs(lines[0].start - start) > TIMESTAMP_TOLERANCE:
        return None
    if abs(lines[-1].end - end) > TIMESTAMP_TOLERANCE:
        return None

    joined = " ".join(line.text.strip() for line in lines if line.text.strip())
    return joined or None


def _phonetic_variety(text: str | None) -> float | None:
    """Return a 0..1 orthographic proxy for phonetic variety, or ``None``.

    This is deliberately *not* phoneme recognition: nothing already available to
    the project can do that, and a fabricated phonetic model would be worse than
    no signal at all. It measures how much of the speaker's symbol and word
    inventory a window actually exercises, which is enough to reject a reference
    built from one repeated word or a single interjection.

    ``None`` means "no transcript covers this window" - the caller then drops the
    term from the score instead of guessing a value.
    """

    if text is None:
        return None

    words = _WORD_PATTERN.findall(text.lower())
    symbols = "".join(words)
    if not words or not symbols:
        return None

    symbol_score = _clamp(len(set(symbols)) / VARIETY_LETTER_TARGET)
    word_score = _clamp(len(set(words)) / VARIETY_WORD_TARGET)
    length_score = _clamp(len(words) / VARIETY_MIN_WORDS)

    return length_score * (
        VARIETY_LETTER_PART * symbol_score + VARIETY_WORD_PART * word_score
    )


def _run_ffmpeg(arguments: list[str]) -> None:
    """Run ``ffmpeg`` with ``arguments`` and fail clearly when it does not work."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ReferenceExtractionError(
            "ffmpeg was not found on PATH; extracting a voice reference needs FFmpeg "
            "(see the requirements in README.md)"
        )

    process = subprocess.run(  # noqa: S603 - argv list, no shell involved
        [executable, *arguments],
        capture_output=True,
        text=True,
        errors="replace",
        check=False,
    )
    if process.returncode != 0:
        detail = (process.stderr or "").strip().splitlines()
        raise ReferenceExtractionError(
            "ffmpeg failed to extract the voice reference "
            f"(exit code {process.returncode}): {detail[-1] if detail else 'no output'}"
        )


def _verify_reference(path: Path, *, expected_duration: float, speaker_id: str) -> None:
    """Check that the extracted reference is the canonical mono 24 kHz WAV."""

    try:
        info = sf.info(str(path))
    except (OSError, RuntimeError) as exc:
        raise ReferencePreprocessingError(
            f"the extracted voice reference for speaker {speaker_id!r} is unreadable: {exc}"
        ) from exc

    if info.samplerate != REFERENCE_SAMPLE_RATE or info.channels != REFERENCE_CHANNELS:
        raise ReferencePreprocessingError(
            f"the voice reference for speaker {speaker_id!r} is {info.samplerate} Hz with "
            f"{info.channels} channel(s); {REFERENCE_SAMPLE_RATE} Hz mono is required"
        )
    if info.frames <= 0:
        raise ReferencePreprocessingError(
            f"the extracted voice reference for speaker {speaker_id!r} is empty"
        )
    if info.duration + DURATION_TOLERANCE < expected_duration:
        raise ReferencePreprocessingError(
            f"only {info.duration:.3f}s of audio were extracted for the "
            f"{expected_duration:.3f}s reference requested for speaker {speaker_id!r}"
        )


def _extract_reference(
    source: Path,
    candidate: _ReferenceCandidate,
    *,
    destination: Path,
) -> None:
    """Cut the selected window out of the dialogue stem as a mono 24 kHz WAV."""

    # Only attenuate: a reference that is already quiet must not be amplified,
    # because that would raise its noise floor and change how it clones.
    gain_db = min(0.0, REFERENCE_PEAK_DBFS - candidate.stats.peak_dbfs)

    arguments = [
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{candidate.start:.6f}",
        "-t",
        f"{candidate.duration:.6f}",
        "-i",
        str(source),
        "-vn",
        "-map_metadata",
        "-1",
        "-ac",
        str(REFERENCE_CHANNELS),
        "-ar",
        str(REFERENCE_SAMPLE_RATE),
    ]
    if gain_db < -0.01:
        arguments += ["-af", f"volume={gain_db:.2f}dB"]
    arguments += ["-c:a", "pcm_s16le", "-f", "wav", str(destination)]

    _run_ffmpeg(arguments)
    _verify_reference(
        destination, expected_duration=candidate.duration, speaker_id=candidate.speaker_id
    )


def _is_usable_prompt(path: Path) -> bool:
    """Return whether a cached clone prompt exists and is not empty."""

    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:  # pragma: no cover - a stat that fails is simply not usable
        return False


def _clone_prompt(
    destination: Path,
    reference_audio: Path,
    *,
    speaker_id: str,
    encoder: ClonePromptEncoder | None,
) -> Path | None:
    """Return the cached clone-prompt path, encoding it only when needed."""

    # Encoding a prompt is the expensive step, so an existing one is always
    # reused and never re-encoded.
    if _is_usable_prompt(destination):
        return destination
    if encoder is None:
        return None

    try:
        encoder(reference_audio, destination)
    except Exception as exc:
        raise ClonePromptError(
            f"could not create the voice-clone prompt for speaker {speaker_id!r}: {exc}"
        ) from exc

    if not _is_usable_prompt(destination):
        raise ClonePromptError(
            f"the voice-clone encoder did not write a prompt for speaker {speaker_id!r} "
            f"to {destination}"
        )
    return destination


def _build_profile(
    candidate: _ReferenceCandidate,
    *,
    source: Path,
    directory: Path,
    target: float,
    clone_encoder: ClonePromptEncoder | None,
) -> VoiceProfile:
    """Extract the reference, cache its clone prompt, and describe the profile."""

    speaker_id = candidate.speaker_id
    speaker_dir = _speaker_directory(directory, speaker_id)
    try:
        speaker_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ReferenceExtractionError(
            f"could not create the voice-profile directory for speaker {speaker_id!r}: {exc}"
        ) from exc

    reference = speaker_dir / REFERENCE_FILENAME
    _extract_reference(source, candidate, destination=reference)

    return VoiceProfile(
        speaker_id=speaker_id,
        reference_audio=reference,
        reference_start=candidate.start,
        reference_end=candidate.end,
        reference_text=candidate.text,
        clone_prompt_path=_clone_prompt(
            speaker_dir / CLONE_PROMPT_FILENAME,
            reference,
            speaker_id=speaker_id,
            encoder=clone_encoder,
        ),
        quality_score=candidate.quality_score(target),
        selection_reason=candidate.selection_reason(),
    )


def build_voice_profiles(
    speaker_segments: Iterable[SpeakerSegment],
    audio_path: str | Path,
    *,
    transcript: Iterable[TranscriptSegment] | None = None,
    crosstalk: Iterable[CrosstalkRegion] | None = None,
    settings: Settings | None = None,
    clone_encoder: ClonePromptEncoder | None = None,
) -> dict[str, VoiceProfile]:
    """Build one voice identity per diarized speaker.

    Parameters
    ----------
    speaker_segments:
        Diarized turns from :mod:`app.pipeline.diarization`. Speaker ids are
        authoritative and are copied through unchanged.
    audio_path:
        The **dialogue stem** written by
        :func:`app.pipeline.separation.separate_stems`, not the original movie
        mix, so the reference is free of music and effects.
    transcript:
        Optional lines from :mod:`app.pipeline.transcription`. They do two jobs:
        they let a window be cut on line boundaries, so the transcript sent with the
        reference describes exactly the audio sent with it, and they feed the
        phonetic-variety term of the reference score. A window no line describes
        exactly is scored without that term and is sent without a transcript.
    crosstalk:
        Optional simultaneous-speech regions from
        :func:`app.pipeline.diarization.diarize_detailed`. A window sharing time
        with one of them contains more than one voice and is avoided. It is only
        used when the film offers nothing clean, and the profile records the
        compromise.
    settings:
        Project settings override; defaults to
        :func:`app.config.get_settings`. The profile directory, the reference
        duration window and the model cache all come from here.
    clone_encoder:
        Optional callable that writes a serialized voice-clone prompt for a
        reference. It is the only model-specific hook, so a cloning encoder can
        be injected without this module depending on one. When it is omitted,
        ``clone_prompt_path`` stays ``None``.

    Returns
    -------
    dict[str, VoiceProfile]
        Profiles keyed by diarization speaker id, in the order the speakers first
        speak. An empty input returns an empty mapping without touching the
        audio. Speaker ids are kept verbatim; only the directory each profile is
        stored in is sanitised, see :func:`speaker_directory_name`.

    Raises
    ------
    InvalidSegmentError, InvalidInputError
        The segments or the transcript are not the expected pipeline types.
    MissingInputError
        The dialogue stem does not exist.
    ConfigurationError
        The configured reference durations contradict each other.
    AudioAnalysisError
        The dialogue stem could not be measured.
    NoUsableReferenceError
        A speaker has no clean, long enough continuous speech. A reference from
        another speaker is never substituted.
    ReferenceExtractionError, ReferencePreprocessingError
        FFmpeg failed, or produced audio in the wrong format.
    ClonePromptError
        The injected encoder failed.
    """

    segments = _validate_segments(speaker_segments)
    lines = _validate_transcript(transcript)
    if not segments:
        return {}

    resolved = settings if settings is not None else get_settings()
    source = _validate_audio_path(audio_path)
    minimum, target, maximum = _resolve_duration_bounds(resolved)
    directory = Path(resolved.voice_profile_dir)

    # Simultaneous speech is where a reference goes wrong: a window cut across it is a
    # recording of two people, and cloning from it teaches the model both voices. The
    # diarization already reports these regions, but the per-line overlap check cannot
    # see them, because the turns it compares are exclusive by construction - so a
    # window full of crosstalk still scores "0.0s overlapped by other speakers".
    regions = [
        (region.start, region.end)
        for region in (crosstalk if crosstalk is not None else ())
        if isinstance(region, CrosstalkRegion)
    ]

    profiles: dict[str, VoiceProfile] = {}
    with _StemReader(source) as reader:
        for speaker_id, own in _group_by_speaker(segments).items():
            others = [segment for segment in segments if segment.speaker_id != speaker_id]
            candidates = _candidates_for(
                speaker_id,
                own,
                others,
                reader=reader,
                minimum=minimum,
                target=target,
                maximum=maximum,
                transcript=lines,
                contaminated=regions,
            )
            usable = [item for item in candidates if item.is_usable(minimum)]
            if not usable:
                raise _no_usable_reference(speaker_id, candidates, minimum=minimum)

            # Reference quality is a matter of what the window *is*, not of a weighted
            # score that can trade one defect for another. Four kinds of window exist,
            # in descending order of trust:
            #
            #   1. clean, and described exactly by a transcript - what we want;
            #   2. clean, but undescribed - safe, because the engine transcribes the
            #      audio itself rather than being handed a transcript that only covers
            #      part of it;
            #   3. containing simultaneous speech, described - safe to send, but the
            #      recording holds two voices;
            #   4. containing simultaneous speech, undescribed.
            #
            # The best window of the highest non-empty tier wins. A film that offers
            # nothing clean still gets a voice, and the profile says which compromise
            # was made.
            def tier(item: "_ReferenceCandidate") -> int:
                described = item.text is not None
                if not item.contaminated:
                    return 0 if described else 1
                return 2 if described else 3

            chosen_tier = min(tier(item) for item in usable)
            pool = [item for item in usable if tier(item) == chosen_tier]

            # Earliest window wins an exact tie, so the same input always selects
            # the same reference.
            best = max(
                pool,
                key=lambda item: (item.quality_score(target), -item.start, -item.end),
            )
            profiles[speaker_id] = _build_profile(
                best,
                source=source,
                directory=directory,
                target=target,
                clone_encoder=clone_encoder,
            )

    return profiles


__all__ = [
    "AudioAnalysisError",
    "CLONE_PROMPT_FILENAME",
    "ClonePromptEncoder",
    "ClonePromptError",
    "ConfigurationError",
    "InvalidInputError",
    "InvalidProfileError",
    "InvalidSegmentError",
    "MissingInputError",
    "NoUsableReferenceError",
    "PROFILE_FILENAME",
    "ProfilePersistenceError",
    "REFERENCE_FILENAME",
    "REFERENCE_SAMPLE_RATE",
    "ReferenceExtractionError",
    "ReferencePreprocessingError",
    "SpeakerSegment",
    "TranscriptSegment",
    "VoiceProfile",
    "VoiceProfileError",
    "build_voice_profiles",
    "load_voice_profiles",
    "portable_path",
    "resolve_project_path",
    "resolve_voice_profile_dir",
    "save_voice_profiles",
    "speaker_directory_name",
]
