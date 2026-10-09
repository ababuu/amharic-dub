"""Long-form dialogue context: scenes, character voices, and a duration budget.

The adaptation stage sees a film as a list of lines. Three things it needs are not
in that list, and they are the three things a film has that a short clip does not:

* **Scene structure.** A batch of ten consecutive lines can straddle a cut, so the
  model is told where the scene boundaries are and who is in the scene, rather than
  inferring a new location from the dialogue alone.
* **Characters.** A film runs 90-180 minutes, and consistency has to survive all of
  it: names, address forms, honorifics, register, relationships. That cannot be an
  LLM's recall over a ten-line window, so it is *state* - a persistent
  :class:`CharacterBible` that is handed to every request.
* **A time budget.** Amharic words are longer than English ones, so a faithful
  translation of an English line routinely needs more time than the original took.
  The budget is expressed in **syllables**, because one Fidel character is one
  syllable and Amharic therefore has a syllable count with no G2P model - the same
  reason :mod:`app.pipeline.qc` measures rate in syllables.

Why a budget rather than more stretching
---------------------------------------
The one large human study of professional dubbing found the audience complaint is
an unnatural *speaking rate* - "too slow, too fast, or too uneven" - and that
*shorter* translation output enables better synchronisation. So the line is asked
to fit before it is synthesized, and only a small final trim is left to
:mod:`app.pipeline.timing`. Time-stretching is what you do when the budget was
ignored, not the mechanism that makes a line fit.

Deliberate non-goals
--------------------
* **No models.** Everything here is arithmetic over timings and text.
* **No invention.** A character that is not in the bible is not guessed at; a
  scene is derived from the line timings, not from a model's idea of a scene.
* **No failing on a missing bible.** A run without one still works; it just has no
  consistency state to offer, which the prompt says rather than papers over.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from app.pipeline.amharic_text import count_syllables


class TimedLine(Protocol):
    """Anything that knows who speaks and when.

    Both :class:`~app.pipeline.transcription.TranscriptSegment` (before adaptation)
    and :class:`~app.pipeline.translation.AdaptedDialogue` (after it) satisfy this,
    which is what lets the same scene segmentation run on either side of the
    adaptation stage.
    """

    speaker_id: str
    start: float
    end: float


#: A gap this long between one line's end and the next line's start is treated as a
#: scene boundary. Sized to survive ordinary turn-taking pauses - a beat of silence
#: between two lines of the same conversation is not a new scene - while still
#: cutting at the long silences a film uses between locations.
DEFAULT_SCENE_GAP_SECONDS = 4.0

#: Longest scene that may be assembled by merging across short gaps. A single
#: unbroken stretch of dialogue longer than this is cut anyway, so one long
#: conversation cannot become a "scene" that covers half the film.
DEFAULT_SCENE_MAX_SECONDS = 90.0

#: Syllables a character of spoken Amharic delivers per second, used to turn an
#: original line's duration into the budget its Amharic replacement must respect.
#:
#: This is an **initial prior, not a measurement**, and it is the one number in this
#: module that wants calibrating: run a film, read the `syllables_per_second`
#: figures the qc block reports for the lines that did fit, and set this to their
#: median. 4.0 is the general speech rate the literature uses for a
#: syllable-timed language and sits inside the plausible band the qc module checks,
#: so it is a safe starting point rather than a claim about Amharic specifically.
DEFAULT_SYLLABLES_PER_SECOND = 4.0

#: How far over budget a line may be before it is sent back to be rewritten. Ten
#: percent is inside the band `timing.py` can absorb without an audible stretch, so
#: a line inside it is left alone rather than re-asked for.
DEFAULT_BUDGET_TOLERANCE = 0.10

#: Longest syllable budget a line may be given, whatever its window. Guards against
#: an unusually long original window turning into an invitation to pad a line.
MAXIMUM_SYLLABLE_BUDGET = 240


class DialogueContextError(RuntimeError):
    """Base class for every error raised by this module."""


class InvalidSceneError(DialogueContextError, ValueError):
    """A scene is malformed."""


class InvalidCharacterError(DialogueContextError, ValueError):
    """A character entry is malformed."""


class InvalidBudgetError(DialogueContextError, ValueError):
    """A syllable budget cannot be computed from the values given."""


@dataclass(frozen=True, slots=True)
class Scene:
    """A stretch of the film that plays as one continuous situation.

    ``line_indexes`` are positions in the full dialogue sequence, so a caller can
    slice a batch without carrying the lines themselves.
    """

    index: int
    start: float
    end: float
    line_indexes: tuple[int, ...]
    speaker_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise InvalidSceneError(f"index must be a non-negative integer, got {self.index!r}")

        start = _seconds("start", self.start)
        end = _seconds("end", self.end)
        if end < start:
            raise InvalidSceneError(f"end ({end}) must not precede start ({start})")

        indexes = tuple(self.line_indexes)
        if not indexes:
            raise InvalidSceneError("a scene needs at least one line")
        for value in indexes:
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InvalidSceneError(
                    f"line indexes must be non-negative integers, got {value!r}"
                )

        speakers = tuple(dict.fromkeys(str(speaker).strip() for speaker in self.speaker_ids))
        if any(not speaker for speaker in speakers):
            raise InvalidSceneError("speaker ids must be non-empty strings")

        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "line_indexes", indexes)
        object.__setattr__(self, "speaker_ids", tuple(sorted(speakers)))

    @property
    def duration(self) -> float:
        """Length of the scene in seconds."""

        return self.end - self.start

    @property
    def line_count(self) -> int:
        """How many lines the scene holds."""

        return len(self.line_indexes)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the scene."""

        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "lines": self.line_count,
            "speakers": list(self.speaker_ids),
        }


def segment_scenes(
    dialogue: Sequence[TimedLine],
    *,
    gap_seconds: float = DEFAULT_SCENE_GAP_SECONDS,
    max_seconds: float = DEFAULT_SCENE_MAX_SECONDS,
) -> tuple[Scene, ...]:
    """Cut ``dialogue`` into scenes on the silences between its lines.

    A new scene starts when the silence before a line is at least ``gap_seconds``,
    or when the scene already holds ``max_seconds`` of material. Both rules are
    deterministic and derived only from the line timings, so the same film always
    segments the same way and the segmentation is auditable.
    """

    if not math.isfinite(gap_seconds) or gap_seconds < 0:
        raise InvalidSceneError(f"gap_seconds must be a finite value >= 0, got {gap_seconds!r}")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise InvalidSceneError(f"max_seconds must be a finite value > 0, got {max_seconds!r}")

    scenes: list[Scene] = []
    current: list[tuple[int, TimedLine]] = []

    def _flush() -> None:
        if not current:
            return
        indexes = tuple(index for index, _ in current)
        speakers: list[str] = []
        for _, line in current:
            if line.speaker_id not in speakers:
                speakers.append(line.speaker_id)
        scenes.append(
            Scene(
                index=len(scenes),
                start=current[0][1].start,
                end=max(line.end for _, line in current),
                line_indexes=indexes,
                speaker_ids=tuple(speakers),
            )
        )
        current.clear()

    previous_end: float | None = None
    for index, line in enumerate(dialogue):
        if current:
            silence = line.start - (previous_end if previous_end is not None else line.start)
            would_run = max(line.end, current[0][1].end) - current[0][1].start
            if silence >= gap_seconds or would_run > max_seconds:
                _flush()
        current.append((index, line))
        previous_end = line.end

    _flush()
    return tuple(scenes)


@dataclass(frozen=True, slots=True)
class Character:
    """One character's persistent identity, as the adaptation prompt sees it.

    This is deliberately *state* rather than a guess: the same prayer - the name the
    dub uses, how they address others, how formal they are - is handed to every
    request, so it cannot drift a hundred lines later.
    """

    speaker_id: str
    name: str
    aliases: tuple[str, ...] = ()
    gender: str = ""
    age: str = ""
    register: str = ""
    relationships: Mapping[str, str] = field(default_factory=dict)
    #: Canonical Amharic spelling of names and terms this character uses, so a
    #: recurring proper noun is spelled the same way every time it appears.
    terms: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        for label, value in (("speaker_id", self.speaker_id), ("name", self.name)):
            if not isinstance(value, str) or not value.strip():
                raise InvalidCharacterError(f"{label} must be a non-empty string")

        aliases = tuple(
            dict.fromkeys(alias.strip() for alias in self.aliases if alias.strip())
        )
        object.__setattr__(self, "speaker_id", self.speaker_id.strip())
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "relationships", dict(self.relationships))
        object.__setattr__(self, "terms", dict(self.terms))

    def as_prompt_block(self) -> str:
        """Render this character as the lines a prompt can carry."""

        parts = [f"- {self.speaker_id} is {self.name}"]
        if self.aliases:
            parts.append(f" (also called {', '.join(self.aliases)})")
        descriptors = [
            value.strip()
            for value in (self.gender, self.age, self.register)
            if isinstance(value, str) and value.strip()
        ]
        if descriptors:
            parts.append(f", {', '.join(descriptors)}")
        parts.append(".")
        if self.relationships:
            parts.append(
                " Relationships: "
                + "; ".join(
                    f"{other} - {bond}" for other, bond in sorted(self.relationships.items())
                )
                + "."
            )
        if self.terms:
            parts.append(
                " Always spelled: "
                + "; ".join(
                    f"{source} -> {target}"
                    for source, target in sorted(self.terms.items())
                )
                + "."
            )
        if self.notes.strip():
            parts.append(f" {self.notes.strip()}")
        return "".join(parts)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the character."""

        return {
            "speaker_id": self.speaker_id,
            "name": self.name,
            "aliases": list(self.aliases),
            "gender": self.gender,
            "age": self.age,
            "register": self.register,
            "relationships": dict(self.relationships),
            "terms": dict(self.terms),
            "notes": self.notes,
        }


@dataclass(frozen=True, slots=True)
class CharacterBible:
    """Every character's persistent identity, keyed by diarized speaker id.

    The speaker ids are diarization's own labels, so a bible is only meaningful
    alongside the run that produced it - which is exactly why it is written to disk
    and reused: a second pass over the same film should not have to re-derive who
    everyone is.
    """

    characters: Mapping[str, Character] = field(default_factory=dict)

    def __post_init__(self) -> None:
        characters = dict(self.characters)
        for speaker_id, character in characters.items():
            if not isinstance(speaker_id, str) or not speaker_id.strip():
                raise InvalidCharacterError(
                    f"character keys must be non-empty speaker ids, got {speaker_id!r}"
                )
            if not isinstance(character, Character):
                raise InvalidCharacterError(
                    f"{speaker_id} must map to a Character, got "
                    f"{type(character).__name__}"
                )
            if character.speaker_id != speaker_id:
                raise InvalidCharacterError(
                    f"the entry for {speaker_id} names itself "
                    f"{character.speaker_id!r}"
                )
        object.__setattr__(self, "characters", characters)

    def __len__(self) -> int:
        return len(self.characters)

    def __contains__(self, speaker_id: object) -> bool:
        return speaker_id in self.characters

    def get(self, speaker_id: str) -> Character | None:
        """Return ``speaker_id``'s character, or ``None`` when it is not known."""

        return self.characters.get(speaker_id)

    def name_of(self, speaker_id: str) -> str:
        """Return the name used for ``speaker_id``, falling back to the label."""

        character = self.characters.get(speaker_id)
        return character.name if character is not None else speaker_id

    def as_prompt_block(self, speaker_ids: Iterable[str] = ()) -> str:
        """Render the characters of ``speaker_ids`` as prompt lines.

        Only characters that are in the scene are rendered: a casting list of forty
        names per request is noise, and costs context that the lines themselves
        need. An unknown speaker is skipped rather than described as unknown.
        """

        blocks = [
            character.as_prompt_block()
            for character in (
                self.characters.get(speaker_id) for speaker_id in speaker_ids
            )
            if character is not None
        ]
        return "\n".join(blocks)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the bible."""

        return {
            speaker_id: character.as_dict()
            for speaker_id, character in sorted(self.characters.items())
        }

    def save(self, path: str | Path) -> Path:
        """Write the bible as JSON, so a second pass can reuse it."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.as_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "CharacterBible":
        """Read a bible written by :meth:`save`.

        A missing file is an empty bible rather than an error: a first run has no
        state yet, and that is the normal case rather than a failure.
        """

        source = Path(path)
        if not source.is_file():
            return cls()

        payload = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise InvalidCharacterError(
                f"{source} must hold a JSON object of characters, got "
                f"{type(payload).__name__}"
            )

        characters: dict[str, Character] = {}
        for speaker_id, entry in payload.items():
            if not isinstance(entry, dict):
                raise InvalidCharacterError(
                    f"the entry for {speaker_id!r} must be a JSON object"
                )
            characters[speaker_id] = Character(
                speaker_id=speaker_id,
                name=str(entry.get("name", "") or speaker_id),
                aliases=tuple(entry.get("aliases", ()) or ()),
                gender=str(entry.get("gender", "") or ""),
                age=str(entry.get("age", "") or ""),
                register=str(entry.get("register", "") or ""),
                relationships=dict(entry.get("relationships", {}) or {}),
                terms=dict(entry.get("terms", {}) or {}),
                notes=str(entry.get("notes", "") or ""),
            )
        return cls(characters=characters)


@dataclass(frozen=True, slots=True)
class SyllableBudget:
    """How many syllables a line may use if it is to fit its original window."""

    window: float
    syllables: int
    rate: float

    def __post_init__(self) -> None:
        window = _seconds("window", self.window)
        if window <= 0:
            raise InvalidBudgetError(f"window must be positive, got {window}")
        if isinstance(self.syllables, bool) or not isinstance(self.syllables, int):
            raise InvalidBudgetError(
                f"syllables must be an integer, got {self.syllables!r}"
            )
        if self.syllables < 1:
            raise InvalidBudgetError(f"syllables must be at least 1, got {self.syllables}")
        if not math.isfinite(self.rate) or self.rate <= 0:
            raise InvalidBudgetError(f"rate must be a positive number, got {self.rate!r}")

        object.__setattr__(self, "window", window)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the budget."""

        return {
            "window": round(self.window, 3),
            "syllables": self.syllables,
            "rate": self.rate,
        }

    def verdict(self, amharic: str, *, tolerance: float = DEFAULT_BUDGET_TOLERANCE) -> "BudgetVerdict":
        """Judge ``amharic`` against this budget."""

        return BudgetVerdict(
            budget=self,
            used=count_syllables(amharic),
            tolerance=tolerance,
        )


def syllable_budget(
    window: float,
    *,
    rate: float = DEFAULT_SYLLABLES_PER_SECOND,
    maximum: int = MAXIMUM_SYLLABLE_BUDGET,
) -> SyllableBudget:
    """Return the syllable budget for a ``window`` of ``window`` seconds."""

    if not math.isfinite(rate) or rate <= 0:
        raise InvalidBudgetError(f"rate must be a positive number, got {rate!r}")
    if not math.isfinite(window) or window <= 0:
        raise InvalidBudgetError(f"window must be a positive number, got {window!r}")
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise InvalidBudgetError(f"maximum must be a positive integer, got {maximum!r}")

    syllables = max(1, min(maximum, round(window * rate)))
    return SyllableBudget(window=window, syllables=syllables, rate=rate)


#: How far one line may leave the film's own delivery speed. Narrow on purpose: the
#: film-wide rate absorbs the systematic difference between Amharic and the English it
#: replaces, so this is only for the lines that are individually unusual.
DEFAULT_LOCAL_TEMPO_MIN = 0.9
DEFAULT_LOCAL_TEMPO_MAX = 1.1

#: How far a line's estimated length has to be from its window before it is worth
#: changing anything, as a fraction. Below this the change would be inaudible, and
#: asking for it costs a different generation.
DEFAULT_PACING_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class PacingPlan:
    """How fast a whole film is delivered, and how far one line may leave that.

    Two levels, rather than one rule applied per line. Amharic is systematically longer
    or shorter than the English it replaces, and that bias belongs to the *film* rather
    than to any line: paying it once, as a single delivery speed, keeps the dialogue
    even. Letting every line find its own speed to fit its own window - which is what a
    single per-line rule amounts to - makes the delivery wander from line to line, and
    that wander is heard as unnatural pacing even when every line individually fits.

    Which is not a detail of arithmetic: the two-level form and the single-level form
    give the *same* rate for a line that needs no local correction. What differs is the
    clamping. Here the film rate is bounded once and each line only departs from it by
    ``local_minimum``..``local_maximum``, so the usual line barely moves.
    """

    film_rate: float = 1.0
    #: The band the *total* rate must stay inside, so one policy governs both this
    #: request and the stretch the timing stage may apply afterwards.
    minimum: float = 0.8
    maximum: float = 1.25
    local_minimum: float = DEFAULT_LOCAL_TEMPO_MIN
    local_maximum: float = DEFAULT_LOCAL_TEMPO_MAX
    tolerance: float = DEFAULT_PACING_TOLERANCE
    syllables_per_second: float = DEFAULT_SYLLABLES_PER_SECOND

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the plan."""

        return {
            "film_rate": round(self.film_rate, 4),
            "minimum": self.minimum,
            "maximum": self.maximum,
            "local_minimum": self.local_minimum,
            "local_maximum": self.local_maximum,
            "syllables_per_second": self.syllables_per_second,
        }

    def rate_for(self, *, seconds: float, syllables: int) -> float | None:
        """Return the rate to deliver one line at, or ``None`` to leave it as it comes.

        ``None`` means "ask for nothing", which is the right answer for a line whose
        estimate already lands within :attr:`tolerance` of its window: a needless request
        would spend a different generation on a difference nobody can hear.
        """

        if isinstance(syllables, bool) or not isinstance(syllables, int):
            return None
        if syllables < 1 or not math.isfinite(seconds) or seconds <= 0:
            return None

        natural = syllables / self.syllables_per_second
        if not math.isfinite(natural) or natural <= 0:
            return None

        # The rate that would make this line exactly fill its window...
        wanted = natural / seconds
        if not math.isfinite(wanted) or wanted <= 0:
            return None

        # ...reached from the film's own speed, by a correction small enough that the
        # delivery stays even. Bounded at both ends: by the local band, so one line
        # cannot wander off alone, and by the global band, because when the film rate is
        # already at its limit there is no headroom left for a local correction to use -
        # without that second clamp a typical line would be asked for 1.375, past the
        # limit the timing stage is allowed to stretch to.
        local = min(max(wanted / self.film_rate, self.local_minimum), self.local_maximum)
        rate = min(max(self.film_rate * local, self.minimum), self.maximum)

        if abs(rate - 1.0) <= self.tolerance:
            # Nothing worth asking for.
            return None
        return rate


def plan_pacing(
    measured: Iterable[tuple[float, int]],
    *,
    minimum: float,
    maximum: float,
    syllables_per_second: float = DEFAULT_SYLLABLES_PER_SECOND,
    tolerance: float = DEFAULT_PACING_TOLERANCE,
) -> PacingPlan:
    """Return the delivery plan for a whole film.

    ``measured`` is one ``(window_seconds, syllables)`` pair per line. The film rate is
    the total syllables divided by the total window - the single speed at which the
    film's dialogue, taken as a whole, exactly fills the time it has - clamped to
    ``minimum``..``maximum``. Those bounds are the same ones the timing stage stretches
    within, so one policy governs how far a performance may be pushed anywhere.

    An empty or unusable input returns the natural pace, which is the honest answer: with
    nothing to measure there is no bias to absorb.
    """

    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum <= 0:
        raise InvalidBudgetError(
            f"pacing bounds must be positive numbers, got {minimum!r}..{maximum!r}"
        )
    if maximum < minimum:
        raise InvalidBudgetError(
            f"the pacing maximum ({maximum}) must not be below the minimum ({minimum})"
        )
    if not math.isfinite(syllables_per_second) or syllables_per_second <= 0:
        raise InvalidBudgetError(
            f"rate must be a positive number, got {syllables_per_second!r}"
        )

    total_window = 0.0
    total_natural = 0.0
    for seconds, syllables in measured:
        if isinstance(syllables, bool) or not isinstance(syllables, int):
            continue
        if syllables < 1 or not math.isfinite(seconds) or seconds <= 0:
            continue
        total_window += seconds
        total_natural += syllables / syllables_per_second

    if total_window <= 0 or total_natural <= 0:
        return PacingPlan(
            minimum=minimum,
            maximum=maximum,
            syllables_per_second=syllables_per_second,
            tolerance=tolerance,
        )

    film_rate = total_natural / total_window
    return PacingPlan(
        film_rate=min(max(film_rate, minimum), maximum),
        minimum=minimum,
        maximum=maximum,
        local_minimum=DEFAULT_LOCAL_TEMPO_MIN,
        local_maximum=DEFAULT_LOCAL_TEMPO_MAX,
        tolerance=tolerance,
        syllables_per_second=syllables_per_second,
    )


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    """Whether one line's Amharic fits the time its original took."""

    budget: SyllableBudget
    used: int
    tolerance: float

    @property
    def ratio(self) -> float:
        """Syllables used as a fraction of the budget."""

        return self.used / self.budget.syllables

    @property
    def within_tolerance(self) -> bool:
        """``True`` when the line is close enough to be left alone."""

        return self.ratio <= 1.0 + self.tolerance

    @property
    def overshoot(self) -> int:
        """How many syllables over budget the line is, or ``0`` when it is not."""

        return max(0, self.used - self.budget.syllables)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe view of the verdict."""

        return {
            "syllables": self.used,
            "budget": self.budget.syllables,
            "window": round(self.budget.window, 3),
            "ratio": round(self.ratio, 4),
            "within_tolerance": self.within_tolerance,
            "overshoot": self.overshoot,
        }

    def describe(self) -> str:
        """Return the instruction a re-prompt carries for this line."""

        return (
            f"{self.used} syllables for a {self.budget.window:.1f}s window, which "
            f"allows about {self.budget.syllables} at "
            f"{self.budget.rate:g} syllables per second - cut at least "
            f"{self.overshoot} syllable(s) without losing meaning."
        )


def _seconds(name: str, value: object) -> float:
    """Return ``value`` as a finite ``float`` number of seconds."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidBudgetError(
            f"{name} must be a number of seconds, got {type(value).__name__}"
        )

    seconds = float(value)
    if not math.isfinite(seconds):
        raise InvalidBudgetError(f"{name} must be finite, got {value!r}")
    return seconds


__all__ = [
    "DEFAULT_BUDGET_TOLERANCE",
    "DEFAULT_LOCAL_TEMPO_MAX",
    "DEFAULT_LOCAL_TEMPO_MIN",
    "DEFAULT_PACING_TOLERANCE",
    "DEFAULT_SCENE_GAP_SECONDS",
    "DEFAULT_SCENE_MAX_SECONDS",
    "DEFAULT_SYLLABLES_PER_SECOND",
    "MAXIMUM_SYLLABLE_BUDGET",
    "BudgetVerdict",
    "Character",
    "CharacterBible",
    "DialogueContextError",
    "InvalidBudgetError",
    "InvalidCharacterError",
    "InvalidSceneError",
    "PacingPlan",
    "plan_pacing",
    "Scene",
    "SyllableBudget",
    "TimedLine",
    "segment_scenes",
    "syllable_budget",
]
