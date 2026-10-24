"""A curated bank of native-Amharic voice references, and casting them to characters.

Why this exists
---------------
A reference cut from the film is the original actor speaking **English**, and a
cloning engine conditions on its reference: OmniVoice's own documentation states
that cross-lingual cloning carries the reference language's accent into the
generated speech. Voice design cannot fix that either - its accent vocabulary is
English accents and Chinese dialects only - so the only lever is a reference in
the target language.

This module supplies that reference. It reads a manifest of human-recorded
Amharic clips with exact transcripts, and swaps each character's film reference
for one of them::

    {"SPEAKER_00": VoiceProfile(...film reference...)}
        -> apply_voice_bank()
        -> {"SPEAKER_00": VoiceProfile(...bank reference...)}

What it costs, and why that is a decision rather than a bug
-----------------------------------------------------------
A bank voice is **not the actor**. Swapping the reference keeps the accent and
gives up the identity, so this is a casting mode, not a repair: the dub becomes
"the cast speaks native Amharic" instead of "the actors speak Amharic". It is
opt-in per run - no configured bank means no change - and every profile it
produces records ``reference_source="voice_bank"`` and the voice it was cast
with, so a manifest can never be mistaken for the film path.

Everything downstream is untouched. The bank clip becomes the profile's
reference audio and transcript, exactly as a film window would, so scoring,
persistence, the TTS engine and its prompt cache all keep their contracts.

Why the clips are trustworthy as references
-------------------------------------------
* **The transcript matches the audio.** The corpus pairs each recording with the
  sentence that was read, which is the hardest thing to obtain from a film and
  the thing OmniVoice requires: ``ref_text`` describes ``ref_audio`` exactly.
* **Silence is trimmed, speech is not.** Only the lead-in before the first word
  and the tail after the last are cut, so the transcript still accounts for every
  sample that is sent. Cutting into speech is how a prompt that ends mid-word
  makes the model finish that word before it speaks the line.
* **The clip is canonical.** Mono 24 kHz PCM, attenuation only towards
  :data:`~app.pipeline.voice_profiles.REFERENCE_PEAK_DBFS`, never amplified, so no
  processing alters the voice.

Casting is deterministic
------------------------
The same bank and the same characters always produce the same pairing: voices are
taken in manifest order, ties are broken by voice id, and a speaker whose
estimated vocal presentation matches a voice's recorded gender is served before
one that does not. Pitch is used to pick the closest voice *within* a gender, so
a low-pitched character does not get the highest voice in the bank.

Deliberate non-goals
--------------------
* **No network at run time.** :func:`build_voice_bank` talks to Hugging Face; the
  pipeline only ever reads the manifest and the clips. Nothing here imports a
  model, and importing this module downloads nothing.
* **No cloning of a real, identified person for its own sake.** The bank is a
  set of consented, published research recordings used as pronunciation anchors;
  it is not a persona library, and a voice is never claimed to be any character.
* **No pitch shifting, denoising or enhancement.** A clip either passes the
  filters as recorded or is not used.
* **No re-use of one voice across characters while an unused one remains.** The
  cast is meant to be distinguishable; exhaustion is reported rather than hidden.
"""

from __future__ import annotations

import io
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import soundfile as sf

from app.config import Settings, get_settings
from app.pipeline.amharic_text import count_syllables, has_latin
from app.pipeline.voice_profiles import (
    MAX_REFERENCE_DURATION,
    MIN_REFERENCE_DURATION,
    REFERENCE_PEAK_DBFS,
    REFERENCE_SAMPLE_RATE,
    VoiceProfile,
    estimate_vocal_presentation,
    speaker_directory_name,
)

#: Manifest describing a built bank. Its presence is what switches casting on.
BANK_MANIFEST_FILENAME = "manifest.json"

#: Attribution written beside a built bank. The corpus is CC BY 4.0, so the
#: credit has to travel with the clips rather than live only in a chat message.
BANK_ATTRIBUTION_FILENAME = "ATTRIBUTION.md"

#: Sub-directory of a bank holding the canonical clips.
BANK_CLIP_DIRNAME = "clips"

#: Name of the reference copied into a speaker's profile directory. Deliberately
#: not ``reference.wav``: that name belongs to the window cut from the film, and
#: keeping them apart means switching casting off does not overwrite anything.
BANK_REFERENCE_FILENAME = "bank_reference.wav"

#: Bumped when the manifest layout changes in a way a reader has to know about.
BANK_MANIFEST_VERSION = 1

#: Dataset the builder defaults to: human-recorded Amharic, Common Voice lineage,
#: with self-reported gender, age band, region, loudness and real speech bounds.
DEFAULT_BANK_DATASET = "snapwre/amharic-speech"

#: Split read by the builder. The corpus splits are speaker-disjoint, and the
#: training split offers the widest choice of voices.
DEFAULT_BANK_SPLIT = "train"

#: Rows fetched per request. The datasets-server caps this at 100.
PAGE_SIZE = 100

#: Pause between page requests, in seconds. The public endpoint is rate limited,
#: and a bank takes well under a minute even with a pause.
PAGE_DELAY_SECONDS = 0.2

#: Statuses worth retrying: the dataset server sheds load and rate limits, and a
#: full scan makes hundreds of requests, so one 502 must not discard the scan.
_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: Silence kept after the last word when the recording has it, mirroring the
#: film path's rule that a reference ends where the speaker stopped.
REFERENCE_TAIL_SILENCE_SECONDS = 0.30

#: Silence kept before the first word, so a reference does not begin on a click.
REFERENCE_LEAD_SILENCE_SECONDS = 0.05

#: Vocal-presentation values a profile can carry, and the bank gender each implies.
_PRESENTATION_TO_GENDER = {"masculine": "male", "feminine": "female"}

#: The presentation a bank gender is reported as on a profile.
_GENDER_TO_PRESENTATION = {"male": "masculine", "female": "feminine", "unknown": "unknown"}

_BANK_GENDERS = ("male", "female", "unknown")

_PRESENTATIONS = ("unknown", "uncertain", "masculine", "feminine")


class VoiceBankError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidBankError(VoiceBankError, ValueError):
    """A manifest, a voice, or a request does not describe a usable bank."""


class MissingBankError(VoiceBankError):
    """A bank was required but its manifest or a clip is absent."""


class CastingError(VoiceBankError):
    """Characters could not be cast from the bank as configured."""


class BankBuildError(VoiceBankError):
    """The builder could not read the corpus or write the bank."""


def _finite(name: str, value: object, error: type[Exception]) -> float:
    """Return ``value`` as a finite ``float`` or fail with ``error``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{name} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise error(f"{name} must be finite, got {value!r}")
    return number


def _non_empty(name: str, value: object, error: type[Exception]) -> str:
    """Return ``value`` stripped as a non-empty string or fail with ``error``."""

    if not isinstance(value, str) or not value.strip():
        raise error(f"{name} must be a non-empty string")
    return value.strip()


def _normalise_text(text: str) -> str:
    """Return ``text`` with corpus whitespace flattened to single spaces.

    Common Voice prompts carry non-breaking spaces and double spaces; a reference
    transcript that differs from the recording only in whitespace is still a
    mismatch worth removing.
    """

    flattened = text.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"\s+", " ", flattened).strip()


@dataclass(frozen=True, slots=True)
class BankVoice:
    """One usable native-speaker reference, as recorded and then made canonical."""

    voice_id: str
    audio: Path
    text: str
    duration: float
    #: Gallery gender of the recording as the corpus reports it: ``male``,
    #: ``female`` or ``unknown``. Used to cast, never to alter audio.
    gender: str
    presentation: str
    pitch_hz: float | None = None
    age_band: str = ""
    region: str = ""
    source_speaker: str = ""
    source_clip_id: str = ""
    source_row: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "voice_id", _non_empty("voice_id", self.voice_id, InvalidBankError)
        )
        object.__setattr__(
            self, "text", _non_empty("text", self.text, InvalidBankError)
        )

        if isinstance(self.audio, Path):
            audio = self.audio
        elif isinstance(self.audio, str) and self.audio.strip():
            audio = Path(self.audio.strip())
        else:
            raise InvalidBankError("audio must be a path or a non-empty path string")
        object.__setattr__(self, "audio", audio)

        duration = _finite("duration", self.duration, InvalidBankError)
        if not MIN_REFERENCE_DURATION <= duration <= MAX_REFERENCE_DURATION:
            raise InvalidBankError(
                f"bank voice {self.voice_id!r} is {duration:.3f}s long; voice cloning "
                f"accepts between {MIN_REFERENCE_DURATION:g} and "
                f"{MAX_REFERENCE_DURATION:g} seconds"
            )
        object.__setattr__(self, "duration", duration)

        if self.gender not in _BANK_GENDERS:
            raise InvalidBankError(
                f"bank voice {self.voice_id!r} has gender {self.gender!r}; "
                f"expected one of {', '.join(_BANK_GENDERS)}"
            )
        if self.presentation not in _PRESENTATIONS:
            raise InvalidBankError(
                f"bank voice {self.voice_id!r} has presentation {self.presentation!r}; "
                f"expected one of {', '.join(_PRESENTATIONS)}"
            )

        if self.pitch_hz is not None:
            pitch = _finite("pitch_hz", self.pitch_hz, InvalidBankError)
            if pitch <= 0:
                raise InvalidBankError(f"pitch_hz must be positive, got {pitch}")
            object.__setattr__(self, "pitch_hz", pitch)

    def describe(self) -> str:
        """Return a short human-readable description of the voice."""

        details = [self.gender, f"{self.duration:.1f}s"]
        if self.region:
            details.append(self.region)
        return ", ".join(details)

    def to_dict(self, *, directory: Path | None = None) -> dict[str, object]:
        """Return a JSON-safe view, storing ``audio`` relative to ``directory``."""

        audio = self.audio
        if directory is not None:
            try:
                audio = Path(audio).relative_to(directory)
            except ValueError:
                pass

        return {
            "voice_id": self.voice_id,
            "audio": Path(audio).as_posix(),
            "text": self.text,
            "duration": self.duration,
            "gender": self.gender,
            "presentation": self.presentation,
            "pitch_hz": self.pitch_hz,
            "age_band": self.age_band,
            "region": self.region,
            "source_speaker": self.source_speaker,
            "source_clip_id": self.source_clip_id,
            "source_row": self.source_row,
        }

    @classmethod
    def from_dict(cls, payload: object, *, directory: Path) -> "BankVoice":
        """Rebuild a voice from :meth:`to_dict` output."""

        if not isinstance(payload, Mapping):
            raise InvalidBankError("a bank voice must be a JSON object")

        required = ("voice_id", "audio", "text", "duration")
        missing = [name for name in required if name not in payload]
        if missing:
            raise InvalidBankError("bank voice is missing " + ", ".join(sorted(missing)))

        audio = Path(str(payload["audio"]))
        if not audio.is_absolute():
            audio = directory / audio

        return cls(
            voice_id=payload["voice_id"],  # type: ignore[arg-type]
            audio=audio,
            text=payload["text"],  # type: ignore[arg-type]
            duration=payload["duration"],  # type: ignore[arg-type]
            gender=str(payload.get("gender", "unknown")),
            presentation=str(payload.get("presentation", "unknown")),
            pitch_hz=payload.get("pitch_hz"),  # type: ignore[arg-type]
            age_band=str(payload.get("age_band", "")),
            region=str(payload.get("region", "")),
            source_speaker=str(payload.get("source_speaker", "")),
            source_clip_id=str(payload.get("source_clip_id", "")),
            source_row=payload.get("source_row"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class VoiceBank:
    """A built bank: the voices plus where they came from."""

    directory: Path
    voices: tuple[BankVoice, ...]
    dataset: str = DEFAULT_BANK_DATASET
    split: str = DEFAULT_BANK_SPLIT
    revision: str = ""
    license: str = ""

    def __post_init__(self) -> None:
        voices = tuple(self.voices)
        if not voices:
            raise InvalidBankError("a voice bank needs at least one voice")

        ids = [voice.voice_id for voice in voices]
        duplicates = sorted({voice_id for voice_id in ids if ids.count(voice_id) > 1})
        if duplicates:
            raise InvalidBankError(
                "duplicate voice ids in the bank: " + ", ".join(duplicates)
            )
        object.__setattr__(self, "voices", voices)

    def __len__(self) -> int:
        return len(self.voices)

    def __iter__(self) -> Iterator[BankVoice]:
        return iter(self.voices)

    def voice(self, voice_id: str) -> BankVoice:
        """Return the voice with ``voice_id``."""

        for voice in self.voices:
            if voice.voice_id == voice_id:
                return voice
        raise InvalidBankError(
            f"unknown bank voice {voice_id!r}; available: "
            + ", ".join(voice.voice_id for voice in self.voices)
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe view of the whole bank."""

        return {
            "version": BANK_MANIFEST_VERSION,
            "dataset": self.dataset,
            "split": self.split,
            "revision": self.revision,
            "license": self.license,
            "voices": [voice.to_dict(directory=self.directory) for voice in self.voices],
        }


@dataclass(frozen=True, slots=True)
class VoiceRequest:
    """What a character needs from the bank."""

    speaker_id: str
    presentation: str = "unknown"
    pitch_hz: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "speaker_id",
            _non_empty("speaker_id", self.speaker_id, CastingError),
        )
        if self.presentation not in _PRESENTATIONS:
            raise CastingError(
                f"presentation for {self.speaker_id!r} is {self.presentation!r}; "
                f"expected one of {', '.join(_PRESENTATIONS)}"
            )
        if self.pitch_hz is not None:
            pitch = _finite("pitch_hz", self.pitch_hz, CastingError)
            if pitch <= 0:
                raise CastingError(f"pitch_hz must be positive, got {pitch}")
            object.__setattr__(self, "pitch_hz", pitch)

    @property
    def gender(self) -> str:
        """Return the bank gender this character asks for, or ``unknown``."""

        return _PRESENTATION_TO_GENDER.get(self.presentation, "unknown")


def load_voice_bank(
    *, directory: str | Path | None = None, settings: Settings | None = None
) -> VoiceBank | None:
    """Return the bank in ``directory``, or ``None`` when there is no manifest.

    A missing bank is not an error: casting is opt-in, and an installation that
    never built one runs exactly as before. A bank that *is* present but broken is
    an error, because silently falling back to the film reference would produce a
    different dub from the one the operator asked for.
    """

    if directory is None:
        resolved = settings if settings is not None else get_settings()
        directory = resolved.voice_bank_dir

    base = Path(directory)
    manifest = base / BANK_MANIFEST_FILENAME
    if not manifest.is_file():
        return None

    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InvalidBankError(f"could not read the voice bank manifest {manifest}: {exc}") from exc

    if not isinstance(payload, Mapping):
        raise InvalidBankError(f"the voice bank manifest {manifest} is not a JSON object")

    version = payload.get("version", 0)
    if version != BANK_MANIFEST_VERSION:
        raise InvalidBankError(
            f"voice bank manifest {manifest} is version {version!r}; this build reads "
            f"version {BANK_MANIFEST_VERSION}. Rebuild the bank."
        )

    entries = payload.get("voices")
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        raise InvalidBankError(f"the voice bank manifest {manifest} has no voice list")
    if not entries:
        raise InvalidBankError(f"the voice bank manifest {manifest} lists no voices")

    voices = tuple(BankVoice.from_dict(entry, directory=base) for entry in entries)

    missing = [voice.voice_id for voice in voices if not voice.audio.is_file()]
    if missing:
        raise MissingBankError(
            "the voice bank manifest names clips that are not on disk: "
            + ", ".join(missing)
            + f" (looked under {base / BANK_CLIP_DIRNAME})"
        )

    return VoiceBank(
        directory=base,
        voices=voices,
        dataset=str(payload.get("dataset", DEFAULT_BANK_DATASET)),
        split=str(payload.get("split", DEFAULT_BANK_SPLIT)),
        revision=str(payload.get("revision", "")),
        license=str(payload.get("license", "")),
    )


def _parse_mapping(mapping: str) -> dict[str, str]:
    """Return ``{speaker: voice_id}`` from ``SPEAKER_00=voice-id,...``."""

    chosen: dict[str, str] = {}
    for entry in (mapping or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        speaker, separator, voice_id = entry.partition("=")
        speaker, voice_id = speaker.strip(), voice_id.strip()
        if not separator or not speaker or not voice_id:
            raise CastingError(
                f"VOICE_BANK_MAP entries look like SPEAKER_00=voice-id, got {entry!r}"
            )
        chosen[speaker] = voice_id
    return chosen


def _pitch_distance(voice: BankVoice, request: VoiceRequest) -> float:
    """Return how far a voice's measured pitch is from a character's, in Hz.

    A voice or character without a measurement is neither favoured nor punished;
    ``inf`` keeps it behind any voice that does have one, and the caller's tie
    break on voice id keeps the result deterministic.
    """

    if voice.pitch_hz is None or request.pitch_hz is None:
        return math.inf
    return abs(voice.pitch_hz - request.pitch_hz)


def assign_bank_voices(
    requests: Iterable[VoiceRequest],
    bank: VoiceBank,
    *,
    mapping: str = "",
) -> dict[str, BankVoice]:
    """Return the bank voice each character should be cast with.

    ``mapping`` is ``SPEAKER_00=voice-id,SPEAKER_01=other-id`` and wins where it
    names a speaker, so an operator can put a specific voice on a specific
    character. Everyone else is served in order from what is left:

    1. an unused voice whose recorded gender matches the character's estimated
       presentation, closest pitch first;
    2. failing that, an unused voice of any gender, chosen to keep the cast's
       gender mix balanced when the character has no presentation estimate;
    3. failing that - more characters than voices - a voice already in use. The
       reuse is recorded in the profile's ``selection_reason``; a bank that is too
       small for the cast is reported, never hidden.

    Raises
    ------
    CastingError
        The mapping is malformed, names a speaker that was not asked for, or
        names a voice the bank does not have.
    InvalidBankError
        The same voice is mapped explicitly to more than one speaker.
    """

    ordered = [request for request in dict.fromkeys(requests)]
    if not ordered:
        return {}

    explicit = _parse_mapping(mapping)
    known = {request.speaker_id for request in ordered}
    strangers = sorted(set(explicit) - known)
    if strangers:
        raise CastingError(
            "VOICE_BANK_MAP names speaker(s) that are not in this film: "
            + ", ".join(strangers)
        )

    chosen: dict[str, BankVoice] = {}
    for speaker_id, voice_id in explicit.items():
        chosen[speaker_id] = bank.voice(voice_id)

    duplicated = sorted(
        {
            voice_id
            for voice_id in explicit.values()
            if list(explicit.values()).count(voice_id) > 1
        }
    )
    if duplicated:
        raise InvalidBankError(
            "VOICE_BANK_MAP gives the same voice to more than one character: "
            + ", ".join(duplicated)
        )

    used = {voice.voice_id for voice in chosen.values()}

    for request in ordered:
        if request.speaker_id in chosen:
            continue

        best = _choose_voice(bank, request, used)
        chosen[request.speaker_id] = best
        used.add(best.voice_id)

    return {request.speaker_id: chosen[request.speaker_id] for request in ordered}


def _choose_voice(bank: VoiceBank, request: VoiceRequest, used: set[str]) -> BankVoice:
    """Return the best unused voice for one character, or a reuse when none is left."""

    unused = [voice for voice in bank if voice.voice_id not in used]

    if request.gender == "unknown":
        # Nothing to match on, so the point of the decision is a balanced cast. Ranking
        # by voice id alone would give every unknown character the alphabetically first
        # voices, and this bank's ids put all the female ones first - a film whose
        # diarization could not estimate presentations would be cast all-female. Ranking
        # by how often each gender has already been used keeps it mixed, and stays
        # deterministic because the tie-break is the voice id.
        counts = Counter(voice.gender for voice in bank if voice.voice_id in used)
        pool = unused or list(bank)
        return min(
            pool, key=lambda voice: (counts.get(voice.gender, 0), voice.voice_id)
        )

    matching = [voice for voice in unused if voice.gender == request.gender]
    if matching:
        # Closest pitch within the requested gender; ties break on voice id, so the
        # pairing never depends on manifest order.
        return min(matching, key=lambda voice: (_pitch_distance(voice, request), voice.voice_id))

    # No unused voice of this gender left: widen to the rest of the bank before
    # reusing a voice, and only reuse when the bank is smaller than the cast.
    pool = unused or [voice for voice in bank if voice.gender == request.gender] or list(bank)
    return min(pool, key=lambda voice: (_pitch_distance(voice, request), voice.voice_id))


def _read_reference(path: Path) -> tuple[np.ndarray, int]:
    """Return a recording as mono ``float32`` samples and its rate."""

    try:
        data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise MissingBankError(f"could not read the bank clip {path}: {exc}") from exc

    if data.size == 0:
        raise MissingBankError(f"the bank clip {path} contains no audio")
    return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32)), int(rate)


def _resample(samples: np.ndarray, rate: int, target_rate: int) -> np.ndarray:
    """Return ``samples`` resampled to ``target_rate``."""

    if rate == target_rate:
        return samples

    from scipy.signal import resample_poly

    divisor = math.gcd(int(rate), int(target_rate))
    resampled = resample_poly(samples, target_rate // divisor, rate // divisor)
    return np.ascontiguousarray(resampled.astype(np.float32))


def _canonical_reference(samples: np.ndarray, rate: int) -> np.ndarray:
    """Return ``samples`` as mono 24 kHz audio with the peak attenuated if needed."""

    mono = samples if samples.ndim == 1 else samples.mean(axis=1, dtype=np.float32)
    resampled = _resample(np.ascontiguousarray(mono, dtype=np.float32), rate, REFERENCE_SAMPLE_RATE)

    peak = float(np.max(np.abs(resampled))) if resampled.size else 0.0
    limit = float(10.0 ** (REFERENCE_PEAK_DBFS / 20.0))
    if peak > limit > 0.0:
        resampled = (resampled * (limit / peak)).astype(np.float32)
    return np.ascontiguousarray(resampled, dtype=np.float32)


def apply_voice_bank(
    profiles: Mapping[str, VoiceProfile],
    *,
    settings: Settings | None = None,
    bank: VoiceBank | None = None,
) -> dict[str, VoiceProfile]:
    """Cast each character from the bank instead of the film's own audio.

    Every character is replaced, or none is: a run that quietly kept one speaker
    on the actor's reference while the rest were cast would be a dub with two
    accents in it.

    The new profile keeps the speaker id and the film window's ``quality_score``
    - that score describes the window the profile was built from and is not a
    claim about the bank clip - and records the casting in ``selection_reason``,
    ``reference_source`` and ``reference_voice_id``. ``clone_prompt_path`` is
    cleared, because a prompt cached for the film reference does not describe the
    bank clip.

    Notes
    -----
    The clip is copied into the speaker's profile directory as
    :data:`BANK_REFERENCE_FILENAME`, so a profile stays self-contained and
    rebuilding the bank cannot silently change a dub that has already been run.

    Returns
    -------
    dict[str, VoiceProfile]
        One profile per input speaker, in the input order. When no bank is
        configured the input profiles are returned unchanged.
    """

    resolved = settings if settings is not None else get_settings()
    library = bank if bank is not None else load_voice_bank(settings=resolved)
    if library is None or not profiles:
        return dict(profiles)

    requests = [
        VoiceRequest(
            speaker_id=speaker_id,
            presentation=profile.estimated_vocal_presentation,
            pitch_hz=profile.estimated_pitch_hz,
        )
        for speaker_id, profile in profiles.items()
    ]
    casting = assign_bank_voices(
        requests, library, mapping=resolved.voice_bank_voice_map
    )

    base = Path(resolved.voice_profile_dir)
    reused = {
        voice.voice_id
        for voice in casting.values()
        if sum(1 for other in casting.values() if other.voice_id == voice.voice_id) > 1
    }

    replaced: dict[str, VoiceProfile] = {}
    for speaker_id, profile in profiles.items():
        voice = casting[speaker_id]
        directory = base / speaker_directory_name(speaker_id)
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VoiceBankError(
                f"could not create the voice-profile directory for speaker "
                f"{speaker_id!r}: {exc}"
            ) from exc

        destination = directory / BANK_REFERENCE_FILENAME
        duration = _write_reference(voice.audio, destination)

        reason = f"voice bank {voice.voice_id} ({voice.describe()}), humans recorded"
        if voice.voice_id in reused:
            reason += "; reused because the bank is smaller than the cast"

        replaced[speaker_id] = VoiceProfile(
            speaker_id=speaker_id,
            reference_audio=destination,
            reference_start=0.0,
            reference_end=duration,
            reference_text=voice.text,
            clone_prompt_path=None,
            quality_score=profile.quality_score,
            selection_reason=reason,
            estimated_vocal_presentation=voice.presentation,
            estimated_pitch_hz=voice.pitch_hz,
            reference_source="voice_bank",
            reference_voice_id=voice.voice_id,
        )

    return replaced


def _write_reference(source: Path, destination: Path) -> float:
    """Write ``source`` to ``destination`` canonically and return its duration."""

    samples, rate = _read_reference(source)
    canonical = _canonical_reference(samples, rate)
    if canonical.size == 0:
        raise MissingBankError(f"the bank clip {source} is empty after conversion")

    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(destination), canonical, REFERENCE_SAMPLE_RATE, subtype="PCM_16"
        )
    except (OSError, RuntimeError) as exc:
        raise VoiceBankError(
            f"could not write the bank reference {destination}: {exc}"
        ) from exc

    try:
        info = sf.info(str(destination))
    except (OSError, RuntimeError) as exc:
        raise VoiceBankError(
            f"could not read back the bank reference {destination}: {exc}"
        ) from exc

    return float(info.frames) / float(info.samplerate)


@dataclass(frozen=True, slots=True)
class BankFilters:
    """What a corpus row has to look like to become a reference.

    The bounds are deliberately narrower than OmniVoice's own 3-10 second
    guidance. A reference near five seconds is long enough to carry the voice and
    short enough that the engine has little extra material to echo, which is the
    same trade the film path makes with its target duration.
    """

    min_speech_seconds: float = 4.0
    target_speech_seconds: float = 5.0
    max_speech_seconds: float = 9.0
    min_clip_seconds: float = 4.0
    max_clip_seconds: float = 10.0
    lead_silence_seconds: float = REFERENCE_LEAD_SILENCE_SECONDS
    tail_silence_seconds: float = REFERENCE_TAIL_SILENCE_SECONDS
    min_up_votes: int = 2
    max_down_votes: int = 0
    min_lufs: float = -34.0
    max_lufs: float = -14.0
    min_text_characters: int = 20
    max_text_characters: int = 160
    min_syllables: int = 8
    min_syllables_per_second: float = 3.0
    max_syllables_per_second: float = 9.0
    require_gender: bool = True

    def __post_init__(self) -> None:
        if self.min_speech_seconds >= self.max_speech_seconds:
            raise InvalidBankError(
                "min_speech_seconds must be below max_speech_seconds"
            )
        if self.min_clip_seconds < MIN_REFERENCE_DURATION:
            raise InvalidBankError(
                f"min_clip_seconds must be at least {MIN_REFERENCE_DURATION:g}"
            )
        if self.max_clip_seconds > MAX_REFERENCE_DURATION:
            raise InvalidBankError(
                f"max_clip_seconds must be at most {MAX_REFERENCE_DURATION:g}"
            )
        if self.min_lufs >= self.max_lufs:
            raise InvalidBankError("min_lufs must be below max_lufs")

    def window(self, row: Mapping[str, Any]) -> tuple[float, float] | None:
        """Return the ``(start, end)`` seconds to keep from ``row``'s recording.

        Only silence is trimmed, so the row's transcript still describes every
        sample that is kept.
        """

        start, speech, end, duration = _speech_bounds(row)
        if start is None or speech is None or end is None or duration is None:
            return None

        keep_from = max(0.0, start - self.lead_silence_seconds)
        keep_to = min(duration, end + self.tail_silence_seconds)
        return (keep_from, keep_to)

    def rejection(self, row: Mapping[str, Any]) -> str | None:
        """Return why ``row`` cannot be a reference, or ``None`` when it can."""

        if not isinstance(row, Mapping):
            return "not a row"

        speaker = _row_text(row, "speaker_id")
        if not speaker:
            return "no speaker id"

        text = _row_text(row, "sentence", "text")
        if not text:
            return "no transcript"
        text = _normalise_text(text)

        if self.require_gender and row.get("gender") not in {"male", "female"}:
            return "no usable gender"

        up_votes = _row_number(row, "up_votes")
        if up_votes is not None and up_votes < self.min_up_votes:
            return "too few validation votes"
        down_votes = _row_number(row, "down_votes")
        if down_votes is not None and down_votes > self.max_down_votes:
            return "reported as a bad recording"

        lufs = _row_number(row, "lufs")
        if lufs is not None and not self.min_lufs <= lufs <= self.max_lufs:
            return "loudness outside the usable band"

        start, speech, end, duration = _speech_bounds(row)
        if start is None or speech is None or end is None or duration is None:
            return "no speech bounds"
        if start < 0 or speech <= 0 or duration <= 0:
            return "inconsistent speech bounds"
        if start + speech > duration + 0.25:
            return "speech runs past the recording"
        if not self.min_speech_seconds <= speech <= self.max_speech_seconds:
            return "speech length outside the cloning window"

        window = self.window(row)
        if window is None:
            return "no keep window"
        kept = window[1] - window[0]
        if not self.min_clip_seconds <= kept <= self.max_clip_seconds:
            return "kept clip outside the cloning window"

        if has_latin(text):
            return "transcript contains Latin script"
        if any(character.isdigit() for character in text):
            return "transcript contains numerals"
        if not self.min_text_characters <= len(text) <= self.max_text_characters:
            return "transcript length outside the usable band"

        syllables = count_syllables(text)
        if syllables < self.min_syllables:
            return "too few syllables to identify a voice"
        rate = syllables / speech
        if not self.min_syllables_per_second <= rate <= self.max_syllables_per_second:
            return "transcript does not fit the audio"

        return None


def _row_text(row: Mapping[str, Any], *names: str) -> str:
    """Return the first non-empty string among ``names`` in ``row``."""

    for name in names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _row_number(row: Mapping[str, Any], name: str) -> float | None:
    """Return ``row[name]`` as a finite float, or ``None``."""

    value = row.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _speech_bounds(
    row: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    """Return ``(speech_start, speech, speech_end, duration)`` from ``row``."""

    start = _row_number(row, "speech_start_s")
    speech = _row_number(row, "speech_s")
    end = _row_number(row, "speech_end_s")
    duration = _row_number(row, "duration_s")

    if speech is None and start is not None and end is not None:
        speech = end - start
    if end is None and start is not None and speech is not None:
        end = start + speech
    return start, speech, end, duration


class RowSource(Protocol):
    """Where a builder gets corpus rows from.

    Implemented by :class:`HuggingFaceRowSource` and injected in tests, so the
    builder can be exercised without a network.
    """

    dataset: str
    split: str
    revision: str
    license: str

    def count(self) -> int:
        """Return how many rows the split has."""

    def rows(self, offset: int, length: int) -> Sequence[Mapping[str, Any]]:
        """Return up to ``length`` rows starting at ``offset``."""

    def download(self, url: str) -> bytes:
        """Return the bytes of a clip the corpus pointed at."""


class HuggingFaceRowSource:
    """Read rows and clip audio from Hugging Face's dataset server.

    Rows come from ``/rows``, which returns metadata *and* a signed URL per clip.
    That keeps a bank build to a few hundred small requests and a handful of
    downloaded clips, instead of the multi-gigabyte parquet download the same
    corpus would cost, and it needs no client library of its own.
    """

    BASE_URL = "https://datasets-server.huggingface.co"

    def __init__(
        self,
        dataset: str = DEFAULT_BANK_DATASET,
        *,
        split: str = DEFAULT_BANK_SPLIT,
        token: str | None = None,
        timeout: float = 120.0,
        page_delay: float = PAGE_DELAY_SECONDS,
        attempts: int = 6,
        backoff: float = 1.5,
    ) -> None:
        self.dataset = dataset
        self.split = split
        self.revision = ""
        self.license = ""
        self._token = token
        self._timeout = timeout
        self._page_delay = page_delay
        self._attempts = max(1, int(attempts))
        self._backoff = max(0.0, float(backoff))
        self._describe()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def _get(self, url: str) -> bytes:
        """Return ``url``'s bytes, retrying the failures that are worth retrying.

        A full scan is dozens of requests to a shared public endpoint, so a single 502
        or 429 must not throw away the pages already scanned. Only server-side and
        rate-limit statuses are retried; a 401 or 404 is a real answer and is raised
        immediately.

        A rate limit is not a server fault, so it is waited out on its own terms: the
        server's ``Retry-After`` is honoured when it sends one, and otherwise the delay
        grows geometrically (1.5s, 3s, 6s, ...). Retrying a 429 after a fixed 1.5s just
        spends the whole retry budget inside the window that is limiting us.
        """

        last_error = ""
        for attempt in range(1, self._attempts + 1):
            request = urllib.request.Request(url, headers=self._headers())
            try:
                with urllib.request.urlopen(request, timeout=self._timeout) as response:
                    return response.read()
            except urllib.error.HTTPError as exc:
                if exc.code not in _RETRYABLE_STATUS:
                    hint = (
                        ". If the corpus is gated, set HUGGINGFACE_TOKEN"
                        if exc.code in {401, 403}
                        else ""
                    )
                    raise BankBuildError(
                        f"Hugging Face returned HTTP {exc.code} for {url}{hint}"
                    ) from exc
                last_error = f"HTTP {exc.code}"
                if attempt < self._attempts:
                    time.sleep(self._retry_delay(exc, attempt))
                continue
            except urllib.error.URLError as exc:
                last_error = str(exc)

            if attempt < self._attempts:
                time.sleep(self._backoff * (2 ** (attempt - 1)))

        raise BankBuildError(
            f"could not read {url} after {self._attempts} attempts: {last_error}"
        )

    def _retry_delay(self, error: urllib.error.HTTPError, attempt: int) -> float:
        """Return how long to wait before retrying ``error``."""

        header = error.headers.get("Retry-After") if error.headers else None
        if header:
            try:
                return max(0.0, float(header))
            except ValueError:
                pass
        return self._backoff * (2 ** (attempt - 1))

    def _describe(self) -> None:
        """Record the dataset revision and license, when the API offers them."""

        url = f"https://huggingface.co/api/datasets/{self.dataset}"
        try:
            payload = json.loads(self._get(url))
        except (BankBuildError, json.JSONDecodeError):
            return
        if not isinstance(payload, Mapping):
            return
        self.revision = str(payload.get("sha", ""))
        tags = payload.get("tags")
        if isinstance(tags, Sequence):
            for tag in tags:
                if isinstance(tag, str) and tag.startswith("license:"):
                    self.license = tag.split(":", 1)[1]
                    break

    def count(self) -> int:
        url = (
            f"{self.BASE_URL}/rows?dataset={urllib.parse.quote(self.dataset, safe='')}"
            f"&config=default&split={urllib.parse.quote(self.split, safe='')}"
            "&offset=0&length=1"
        )
        payload = json.loads(self._get(url))
        total = payload.get("num_rows_total") if isinstance(payload, Mapping) else None
        if not isinstance(total, int) or total <= 0:
            raise BankBuildError(
                f"could not determine how many rows {self.dataset} has for split "
                f"{self.split!r}"
            )
        return total

    def rows(self, offset: int, length: int) -> Sequence[Mapping[str, Any]]:
        url = (
            f"{self.BASE_URL}/rows?dataset={urllib.parse.quote(self.dataset, safe='')}"
            f"&config=default&split={urllib.parse.quote(self.split, safe='')}"
            f"&offset={int(offset)}&length={int(length)}"
        )
        payload = json.loads(self._get(url))
        entries = payload.get("rows") if isinstance(payload, Mapping) else None
        if not isinstance(entries, Sequence):
            raise BankBuildError(f"the dataset server returned no rows for {url}")

        rows: list[Mapping[str, Any]] = []
        for entry in entries:
            row = entry.get("row") if isinstance(entry, Mapping) else None
            if isinstance(row, Mapping):
                rows.append(row)
        if self._page_delay:
            time.sleep(self._page_delay)
        return rows

    def download(self, url: str) -> bytes:
        return self._get(url)


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A corpus row that passed the filters, ready to become a bank voice."""

    row: Mapping[str, Any]
    row_index: int
    speaker: str
    gender: str
    speech_seconds: float
    clip_id: str

    def sort_key(self, target: float) -> tuple[float, int, str]:
        """Rank candidates: closest to target, most validated, stable id."""

        votes = _row_number(self.row, "up_votes") or 0.0
        return (abs(self.speech_seconds - target), -votes, self.clip_id)


def _candidate(row: Mapping[str, Any], row_index: int) -> _Candidate:
    """Return the :class:`_Candidate` view of a row that already passed filters."""

    _, speech, _, _ = _speech_bounds(row)
    if speech is None:
        raise BankBuildError(f"row {row_index} passed the filters without speech bounds")
    gender = _row_text(row, "gender")
    return _Candidate(
        row=row,
        row_index=row_index,
        speaker=_row_text(row, "speaker_id"),
        gender=gender if gender in _BANK_GENDERS else "unknown",
        speech_seconds=speech,
        clip_id=_row_text(row, "clip_id") or f"row-{row_index}",
    )


def _audio_source(row: Mapping[str, Any]) -> tuple[bytes | None, str | None]:
    """Return the inline bytes or the URL of a row's recording."""

    audio = row.get("audio")
    if isinstance(audio, Mapping):
        entries: Sequence[Any] = [audio]
    elif isinstance(audio, Sequence) and not isinstance(audio, (str, bytes)):
        entries = audio
    else:
        entries = []

    for entry in entries:
        if not isinstance(entry, Mapping):
            continue
        inline = entry.get("bytes")
        if isinstance(inline, (bytes, bytearray)) and inline:
            return bytes(inline), None
        src = entry.get("src")
        if isinstance(src, str) and src:
            return None, src
    return None, None


def _decode_audio(payload: bytes, *, label: str) -> tuple[np.ndarray, int]:
    """Return ``payload`` as mono ``float32`` samples and its rate."""

    try:
        data, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as exc:
        raise BankBuildError(f"could not decode the recording for {label}: {exc}") from exc
    if data.size == 0:
        raise BankBuildError(f"the recording for {label} contains no audio")
    return np.ascontiguousarray(data.mean(axis=1, dtype=np.float32)), int(rate)


def _voice_id(gender: str, speaker: str, taken: set[str]) -> str:
    """Return a readable, stable, collision-free id for a bank voice."""

    prefix = {"male": "m", "female": "f"}.get(gender, "x")
    slug = re.sub(r"[^0-9a-zA-Z]+", "", speaker)[-10:] or "speaker"
    base = f"am-{prefix}-{slug}"

    candidate = base
    counter = 2
    while candidate in taken:
        candidate = f"{base}-{counter}"
        counter += 1
    return candidate


def _balance_unmet(best: Mapping[str, _Candidate], balance: bool) -> bool:
    """Return whether a bank would still be one-sided with the speakers found so far."""

    if not balance:
        return False
    genders = {candidate.gender for candidate in best.values()}
    return not {"male", "female"} <= genders


def _select_cast(
    candidates: Iterable[_Candidate], *, voices: int, target: float, balance: bool
) -> list[_Candidate]:
    """Return the speakers to include, alternating genders when asked to."""

    ranked = sorted(candidates, key=lambda item: (item.sort_key(target), item.speaker))
    if not balance:
        return ranked[:voices]

    by_gender: dict[str, list[_Candidate]] = {"male": [], "female": [], "unknown": []}
    for candidate in ranked:
        by_gender.setdefault(candidate.gender, []).append(candidate)

    # Alternating genders keeps a cast from being all one voice, which is what a
    # naive "closest to target" cut produces when one gender dominates the corpus.
    # Any gender the corpus reports beyond the three expected still gets served.
    order = ["male", "female", "unknown"]
    order += [gender for gender in by_gender if gender not in order]

    chosen: list[_Candidate] = []
    while len(chosen) < voices and any(by_gender.get(gender) for gender in order):
        for gender in order:
            if len(chosen) >= voices:
                break
            pool = by_gender.get(gender)
            if pool:
                chosen.append(pool.pop(0))
    return chosen


def _write_attribution(bank_directory: Path, bank: VoiceBank) -> Path:
    """Write the credit the corpus licence requires and return its path."""

    path = bank_directory / BANK_ATTRIBUTION_FILENAME
    lines = [
        "# Voice bank attribution",
        "",
        f"This voice bank was built from **{bank.dataset}** (split `{bank.split}`, "
        f"revision `{bank.revision or 'unknown'}`), licensed "
        f"**{bank.license or 'see the dataset card'}**.",
        "",
        "These are human recordings published for speech research. They are used here "
        "as pronunciation references for a speech synthesizer, not as likenesses of "
        "any character, and no speaker is identified by this project.",
        "",
        f"- Dataset: https://huggingface.co/datasets/{bank.dataset}",
        f"- Licence: https://creativecommons.org/licenses/by/4.0/",
        "",
        "Rebuild the bank (and this file) with:",
        "",
        "    python scripts/build_voice_bank.py",
        "",
    ]
    try:
        path.write_text("\n".join(lines), encoding="utf-8")
    except OSError as exc:
        raise BankBuildError(f"could not write {path}: {exc}") from exc
    return path


def build_voice_bank(
    *,
    destination: str | Path | None = None,
    settings: Settings | None = None,
    voices: int = 12,
    filters: BankFilters | None = None,
    source: RowSource | None = None,
    balance_genders: bool = True,
    max_pages: int | None = None,
    report: Callable[[str], None] | None = None,
) -> VoiceBank:
    """Build a bank from a corpus and write it to ``destination``.

    The whole split is scanned once, so each speaker is represented by their best
    qualifying clip rather than by the first one encountered, and every rejection
    reason is counted and reported. A bank is a small, curated artifact: twelve
    voices take well under a minute to fetch and a couple of megabytes on disk.

    Parameters
    ----------
    destination:
        Directory to write the manifest, clips and attribution to. Defaults to
        ``settings.voice_bank_dir``.
    voices:
        How many voices to keep. When the corpus has more speakers than this, the
        closest clips to the target duration win, alternating between genders.
    filters:
        Row acceptance rules. Defaults to :class:`BankFilters`.
    source:
        Row provider; defaults to :class:`HuggingFaceRowSource` for
        :data:`DEFAULT_BANK_DATASET`. Tests inject a fake.
    max_pages:
        Stop after this many pages. ``None`` scans the whole split.
    report:
        Receives one line per notable event.

    Raises
    ------
    BankBuildError
        The corpus could not be read, no row passed the filters, or the bank
        could not be written.
    """

    if voices < 1:
        raise BankBuildError(f"voices must be at least 1, got {voices}")

    resolved = settings if settings is not None else get_settings()
    directory = Path(destination) if destination is not None else Path(resolved.voice_bank_dir)
    rules = filters if filters is not None else BankFilters()
    rows = source if source is not None else HuggingFaceRowSource(
        token=resolved.huggingface_token
    )
    say = report if report is not None else (lambda _message: None)

    total = rows.count()
    say(
        f"scanning {total} row(s) of {rows.dataset} [{rows.split}]"
        f"{' at ' + rows.revision if rows.revision else ''}"
    )

    best: dict[str, _Candidate] = {}
    rejected: dict[str, int] = {}
    accepted = 0
    offset = 0
    pages = 0

    while offset < total:
        page = rows.rows(offset, PAGE_SIZE)
        if not page:
            break
        for index, row in enumerate(page):
            reason = rules.rejection(row)
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            accepted += 1
            candidate = _candidate(row, offset + index)

            # One voice per source speaker: two clips of the same person would be
            # the same character twice, and a bank exists to give a cast variety.
            previous = best.get(candidate.speaker)
            if previous is None or candidate.sort_key(
                rules.target_speech_seconds
            ) < previous.sort_key(rules.target_speech_seconds):
                best[candidate.speaker] = candidate

        offset += len(page)
        pages += 1
        if max_pages is not None and pages >= max_pages:
            # A page limit exists to make a build quick, not to make it silently
            # one-sided. This corpus stores each speaker's clips together, so a short
            # scan can hold only one gender - and a bank like that casts every
            # character, women included, in a man's voice. So the scan continues until
            # both genders are represented, or until the split runs out.
            if not _balance_unmet(best, balance_genders):
                break
            if pages == max_pages:
                say(
                    "  only one gender found in the first "
                    f"{max_pages} page(s); scanning further to keep the cast mixed"
                )

    say(f"accepted {accepted} row(s) from {len(best)} distinct speaker(s)")
    for reason, count in sorted(rejected.items(), key=lambda item: -item[1]):
        say(f"  rejected {count:>5} row(s): {reason}")

    if not best:
        raise BankBuildError(
            "no row in the corpus passed the filters; loosen BankFilters or choose "
            "another corpus"
        )

    if balance_genders and _balance_unmet(best, balance_genders):
        available = sorted({candidate.gender for candidate in best.values()})
        say(
            "  only "
            + " and ".join(available)
            + " voices were found in the whole corpus, so every character will be "
            "cast in one of them"
        )

    cast = _select_cast(
        best.values(),
        voices=voices,
        target=rules.target_speech_seconds,
        balance=balance_genders,
    )
    say(f"selecting {len(cast)} voice(s)")

    clip_directory = directory / BANK_CLIP_DIRNAME
    try:
        clip_directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise BankBuildError(f"could not create {clip_directory}: {exc}") from exc

    built: list[BankVoice] = []
    taken: set[str] = set()
    for candidate in cast:
        label = f"{candidate.speaker}/{candidate.clip_id}"
        inline, url = _audio_source(candidate.row)
        if inline is None and url is None:
            say(f"  skipped {label}: no audio")
            continue
        payload = inline if inline is not None else rows.download(url or "")

        samples, rate = _decode_audio(payload, label=label)
        window = rules.window(candidate.row)
        if window is None:
            say(f"  skipped {label}: no keep window")
            continue

        start_index = max(0, int(round(window[0] * rate)))
        end_index = min(samples.size, int(round(window[1] * rate)))
        if end_index <= start_index:
            say(f"  skipped {label}: keep window is empty")
            continue

        canonical = _canonical_reference(samples[start_index:end_index], rate)
        duration = canonical.size / float(REFERENCE_SAMPLE_RATE)
        if not MIN_REFERENCE_DURATION <= duration <= MAX_REFERENCE_DURATION:
            say(f"  skipped {label}: kept clip is {duration:.2f}s")
            continue

        voice_id = _voice_id(candidate.gender, candidate.speaker, taken)
        taken.add(voice_id)
        path = clip_directory / f"{voice_id}.wav"
        try:
            sf.write(str(path), canonical, REFERENCE_SAMPLE_RATE, subtype="PCM_16")
        except (OSError, RuntimeError) as exc:
            raise BankBuildError(f"could not write the clip {path}: {exc}") from exc

        pitch_hz, _ = estimate_vocal_presentation(path)
        text = _normalise_text(_row_text(candidate.row, "sentence", "text"))

        built.append(
            BankVoice(
                voice_id=voice_id,
                audio=path,
                text=text,
                duration=duration,
                gender=candidate.gender,
                presentation=_GENDER_TO_PRESENTATION[candidate.gender],
                pitch_hz=pitch_hz,
                age_band=_row_text(candidate.row, "age_band"),
                region=_row_text(candidate.row, "region"),
                source_speaker=candidate.speaker,
                source_clip_id=candidate.clip_id,
                source_row=candidate.row_index,
            )
        )
        say(
            f"  {voice_id}: {candidate.gender}, {duration:.2f}s, "
            f"{pitch_hz and f'{pitch_hz:.0f} Hz' or 'pitch unmeasured'}, "
            f"{text[:40]!r}"
        )

    if not built:
        raise BankBuildError("every selected recording failed to convert")

    bank = VoiceBank(
        directory=directory,
        voices=tuple(built),
        dataset=rows.dataset,
        split=rows.split,
        revision=rows.revision,
        license=rows.license,
    )

    manifest = directory / BANK_MANIFEST_FILENAME
    payload = bank.to_dict()
    payload["filters"] = asdict(rules)
    try:
        manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    except OSError as exc:
        raise BankBuildError(f"could not write {manifest}: {exc}") from exc

    _write_attribution(directory, bank)
    say(f"wrote {len(bank)} voice(s) to {manifest}")
    return bank


__all__ = [
    "BANK_ATTRIBUTION_FILENAME",
    "BANK_CLIP_DIRNAME",
    "BANK_MANIFEST_FILENAME",
    "BANK_MANIFEST_VERSION",
    "BANK_REFERENCE_FILENAME",
    "BankBuildError",
    "BankFilters",
    "BankVoice",
    "CastingError",
    "DEFAULT_BANK_DATASET",
    "DEFAULT_BANK_SPLIT",
    "HuggingFaceRowSource",
    "InvalidBankError",
    "MissingBankError",
    "RowSource",
    "VoiceBank",
    "VoiceBankError",
    "VoiceRequest",
    "apply_voice_bank",
    "assign_bank_voices",
    "build_voice_bank",
    "load_voice_bank",
]
