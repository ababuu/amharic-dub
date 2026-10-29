"""Dubbing from an authored dialogue plan, with voices cast from the film itself.

A hand-made plan (``<video>.plan.json`` in ``data/input``) already contains what
three stages would otherwise compute - the sentences, their speakers, their
timings, and the Amharic - so a run needs neither transcription output nor a
translation request. What it does not contain is *which measured voice belongs to
each character*, and that cannot be authored by hand: it is a property of the
audio. This module supplies it.

How a character is tied to a voice
----------------------------------
Every plan line has a start and an end, and the diarization already says which
voice was speaking over that span. For each character the overlap with every
diarized speaker is accumulated; the speaker that carries most of that
character's lines is the candidate. The mapping is accepted only when it is both
dominant and unambiguous, so a scene where two people talk over each other leaves
the character unassigned and reported rather than silently given the other
actor's voice.

Timeline
--------
A plan is usually written against the **episode**, while the file being dubbed may
be a clip cut from it. The offset is measured, not assumed: each plan line is
matched to the transcribed lines by its own words, and the median disagreement is
applied as a constant shift. Lines that do not fall inside the audio afterwards
belong to another part of the episode and are reported, not voiced.

Deliberate non-goals
--------------------
* **No guessing.** A character with no measured voice is dropped with a reason.
* **No translation.** The plan carries the Amharic; asking a model for it again
  would only produce a different line.
* **No model calls at all.** Identification is arithmetic over timings and text.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.pipeline.transcription import TranscriptSegment
from app.pipeline.translation import AdaptedDialogue

#: File-name suffixes searched for a plan, in priority order.
PLAN_SUFFIXES = (".plan.json", ".dialogue.json")

#: Widest constant shift applied to a plan, in seconds. A plan written against a
#: two-hour episode and a two-minute clip differ by at most the episode length.
MAX_PLAN_OFFSET_SECONDS = 7_200.0

#: Share of a line's words that must appear in the transcript before it counts as
#: evidence for the offset.
MIN_MATCH_FRACTION = 0.45

#: Smallest measured shift that is applied. A plan the operator has already aligned
#: to the video must not be nudged by the sub-second jitter of a text match, while a
#: plan written against the whole episode differs by whole minutes.
MIN_PLAN_OFFSET_SECONDS = 1.5

#: Matched lines needed before a measured shift is trusted at all.
MIN_OFFSET_VOTES = 3

#: Share of a character's measured time one diarized voice must carry before it is
#: accepted as that character's voice.
MIN_CAST_SHARE = 0.50

#: That voice must also beat the runner-up by this factor, so a character split
#: evenly between two voices is left unassigned.
MIN_CAST_MARGIN = 1.60

#: Least measured time a character needs before its mapping means anything.
MIN_CAST_SECONDS = 0.60

#: A line that overlaps simultaneous speech carries two voices, so it is not
#: evidence for either one.
MAX_AMBIGUOUS_OVERLAP_SHARE = 0.25

#: Keys a plan may use for each field.
_CHARACTER_KEYS = ("name", "speaker", "character")
_TEXT_KEYS = ("amharic", "am", "target_text")
_SOURCE_KEYS = ("source_text", "source", "text", "english")
_TIME_KEYS = ("start", "start_time", "from")
_END_KEYS = ("end", "end_time", "to")

_TIMECODE = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{1,2})[,.](\d{1,3})\s*$")
_SECONDS = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s?\s*$")
_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_BRACKETED = re.compile(r"\[[^\]]*\]")


class PlanError(ValueError):
    """A plan file is unreadable or does not describe usable dialogue."""


def _tokens(text: str) -> tuple[str, ...]:
    cleaned = _BRACKETED.sub(" ", text or "")
    cleaned = cleaned.replace("\ufffd", "").replace("\u2019", "'").casefold()
    return tuple(token.replace("'", "") for token in _TOKEN.findall(cleaned))


def parse_time(value: object, *, field_name: str) -> float:
    """Return ``value`` as seconds, accepting ``HH:MM:SS,mmm``, ``MM:SS`` or a number."""

    if isinstance(value, bool):
        raise PlanError(f"{field_name} must be a time, got {value!r}")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise PlanError(f"{field_name} must be a non-negative time, got {value!r}")
        return number
    text = str(value or "").strip()
    match = _TIMECODE.match(text)
    if match is not None:
        hours = int(match.group(1) or 0)
        minutes = int(match.group(2))
        seconds = int(match.group(3))
        millis = int(match.group(4).ljust(3, "0")[:3])
        return hours * 3600 + minutes * 60 + seconds + millis / 1000.0
    match = _SECONDS.match(text)
    if match is not None:
        return float(match.group(1))
    raise PlanError(f"{field_name} is not a timecode: {value!r}")


def _first(item: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in item and item[key] not in (None, ""):
            return item[key]
    return None


def _clamp(value: object, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(max(number, 0.0), 1.0)


@dataclass(frozen=True, slots=True)
class PlanCharacter:
    """A character the plan expects to be heard."""

    name: str
    aliases: tuple[str, ...] = ()
    gender: str = "unknown"

    def __post_init__(self) -> None:
        name = str(self.name or "").strip()
        if not name:
            raise PlanError("a plan character needs a name")
        aliases = tuple(
            dict.fromkeys(
                str(alias).strip() for alias in self.aliases if str(alias).strip()
            )
        )
        gender = str(self.gender or "").strip().lower()
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "aliases", aliases)
        object.__setattr__(self, "gender", gender or "unknown")


@dataclass(frozen=True, slots=True)
class PlanCue:
    """One authored line: who says it, when, and the Amharic already written for it."""

    index: int
    cue: int
    speaker: str
    start: float
    end: float
    source_text: str
    amharic: str
    emotion: str = "neutral"
    intensity: float = 0.5
    delivery: str = "neutral"
    pause_before: float = 0.0
    pause_after: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def shifted(self, offset: float) -> "PlanCue":
        """Return this cue moved by ``offset`` seconds."""

        if not offset:
            return self
        return PlanCue(
            index=self.index,
            cue=self.cue,
            speaker=self.speaker,
            start=max(0.0, self.start + offset),
            end=max(0.0, self.end + offset),
            source_text=self.source_text,
            amharic=self.amharic,
            emotion=self.emotion,
            intensity=self.intensity,
            delivery=self.delivery,
            pause_before=self.pause_before,
            pause_after=self.pause_after,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "cue": self.cue,
            "speaker": self.speaker,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "source_text": self.source_text,
            "amharic": self.amharic,
        }


@dataclass(frozen=True, slots=True)
class DialoguePlan:
    """An authored dialogue file, validated."""

    characters: tuple[PlanCharacter, ...] = ()
    cues: tuple[PlanCue, ...] = ()
    source: Path | None = None
    title: str = ""

    def __len__(self) -> int:
        return len(self.cues)

    def __bool__(self) -> bool:
        return bool(self.cues)

    @property
    def speakers(self) -> tuple[str, ...]:
        """Character names that actually speak, in first-appearance order."""

        return tuple(dict.fromkeys(cue.speaker for cue in self.cues))

    def character(self, name: str) -> PlanCharacter | None:
        word = name.strip().casefold()
        for character in self.characters:
            if word == character.name.casefold():
                return character
            if word in {alias.casefold() for alias in character.aliases}:
                return character
        return None

    def gender_of(self, name: str) -> str:
        character = self.character(name)
        return character.gender if character is not None else "unknown"

    def summary(self) -> dict[str, Any]:
        return {
            "source": None if self.source is None else str(self.source),
            "title": self.title,
            "cues": len(self.cues),
            "characters": [character.name for character in self.characters],
            "speakers": list(self.speakers),
        }


def _cue_from(item: Mapping[str, Any], index: int) -> PlanCue:
    speaker = _first(item, _CHARACTER_KEYS)
    amharic = _first(item, _TEXT_KEYS)
    start = _first(item, _TIME_KEYS)
    end = _first(item, _END_KEYS)
    if speaker is None:
        raise PlanError(f"dialogue[{index}] has no speaker")
    if amharic is None:
        raise PlanError(f"dialogue[{index}] has no Amharic text")
    if start is None or end is None:
        raise PlanError(f"dialogue[{index}] needs a start and an end")
    start_seconds = parse_time(start, field_name=f"dialogue[{index}].start")
    end_seconds = parse_time(end, field_name=f"dialogue[{index}].end")
    if end_seconds <= start_seconds:
        raise PlanError(
            f"dialogue[{index}] ends ({end_seconds}) at or before it starts "
            f"({start_seconds})"
        )
    source_text = _first(item, _SOURCE_KEYS)
    try:
        cue_value = int(item.get("cue", index + 1))
    except (TypeError, ValueError):
        cue_value = index + 1
    try:
        pause_before = max(0.0, float(item.get("pause_before", 0.0) or 0.0))
        pause_after = max(0.0, float(item.get("pause_after", 0.0) or 0.0))
    except (TypeError, ValueError) as exc:
        raise PlanError(f"dialogue[{index}] has an unusable pause value: {exc}") from exc
    return PlanCue(
        index=index,
        cue=cue_value,
        speaker=str(speaker).strip(),
        start=start_seconds,
        end=end_seconds,
        source_text=str(source_text or "").strip(),
        amharic=str(amharic).strip(),
        emotion=str(item.get("emotion", "neutral") or "neutral").strip(),
        intensity=_clamp(item.get("intensity"), 0.5),
        delivery=str(item.get("delivery", "neutral") or "neutral").strip(),
        pause_before=pause_before,
        pause_after=pause_after,
    )


def plan_from_payload(payload: object, *, source: Path | None = None) -> DialoguePlan:
    """Build a validated plan from parsed JSON."""

    if isinstance(payload, list):
        items: list[Any] = payload
        raw_characters: list[Any] = []
        title = ""
    elif isinstance(payload, Mapping):
        items = payload.get("dialogue") or payload.get("lines") or []
        raw_characters = payload.get("characters") or []
        title = str(payload.get("title", "") or "")
    else:
        raise PlanError(f"a plan must be a JSON object, got {type(payload).__name__}")

    if not isinstance(items, list) or not items:
        raise PlanError("a plan needs a non-empty dialogue list")

    characters: list[PlanCharacter] = []
    for index, item in enumerate(
        raw_characters if isinstance(raw_characters, list) else []
    ):
        if isinstance(item, str):
            characters.append(PlanCharacter(name=item))
            continue
        if not isinstance(item, Mapping):
            raise PlanError(f"characters[{index}] must be an object or a name")
        name = _first(item, _CHARACTER_KEYS)
        if name is None:
            raise PlanError(f"characters[{index}] has no name")
        aliases = item.get("aliases") or ()
        if isinstance(aliases, str):
            aliases = (aliases,)
        characters.append(
            PlanCharacter(
                name=str(name),
                aliases=tuple(aliases),
                gender=str(item.get("gender", "unknown") or "unknown"),
            )
        )

    cues: list[PlanCue] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise PlanError(f"dialogue[{index}] must be an object")
        cues.append(_cue_from(item, index))

    cues.sort(key=lambda cue: (cue.start, cue.end, cue.speaker))
    known = {character.name.casefold() for character in characters}
    for cue in cues:
        if cue.speaker.casefold() not in known:
            characters.append(PlanCharacter(name=cue.speaker))
            known.add(cue.speaker.casefold())
    return DialoguePlan(
        characters=tuple(characters), cues=tuple(cues), source=source, title=title
    )


def load_plan(path: str | Path) -> DialoguePlan:
    """Read and validate a plan file."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise PlanError(f"could not read the plan {source}: {exc}") from exc
    except ValueError as exc:
        raise PlanError(f"{source} is not valid JSON: {exc}") from exc
    return plan_from_payload(payload, source=source)


def discover_plan(input_dir: str | Path, source: str | Path) -> Path | None:
    """Return a plan named after ``source``, or the only plan in the directory."""

    directory = Path(input_dir)
    if not directory.is_dir():
        return None
    stem = Path(source).stem
    for suffix in PLAN_SUFFIXES:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
    matches = [
        path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.name.casefold().endswith(PLAN_SUFFIXES)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def retime_plan(
    plan: DialoguePlan,
    transcript: Sequence[TranscriptSegment],
    *,
    threshold: float = MIN_MATCH_FRACTION,
    max_skip: int = 25,
    fallback: bool = False,
    audio_duration: float | None = None,
) -> tuple[tuple[PlanCue, ...], int]:
    """Re-time each plan line from the transcript's own word timings.

    The plan is the source of truth for *which lines exist*; the video is the
    source of truth for *when they are spoken*. Each line is given the span of the
    transcribed speech that says its words, which is what keeps a line on its actor
    even when the plan was written against a different cut.

    ``fallback`` keeps a line whose words the transcript never recognised, placed
    between the lines either side of it rather than dropped: the plan says the line
    exists, so losing it would edit the dialogue. Returns the re-timed lines in
    order, and how many had no direct match.
    """

    if not plan.cues:
        return (), 0
    if not transcript:
        return (), len(plan.cues)

    line_tokens = [_tokens(line.text) for line in transcript]
    placed: list[tuple[PlanCue, float, float] | None] = []
    unmatched = 0
    cursor = 0

    for position, cue in enumerate(plan.cues):
        cue_tokens = _tokens(cue.source_text or cue.amharic)
        following_tokens = (
            _tokens(
                plan.cues[position + 1].source_text
                or plan.cues[position + 1].amharic
            )
            if position + 1 < len(plan.cues)
            else ()
        )
        first: int | None = None
        if len(set(cue_tokens)) >= 3:
            for offset in range(0, max_skip + 1):
                index = cursor + offset
                if index >= len(line_tokens):
                    break
                tokens = line_tokens[index]
                if tokens and _containment(cue_tokens, tokens) >= threshold:
                    first = index
                    break
        if first is None:
            placed.append(None)
            unmatched += 1
            continue

        last = first
        while last + 1 < len(line_tokens):
            tokens = line_tokens[last + 1]
            if not tokens or _containment(cue_tokens, tokens) < threshold:
                break
            # A transcriber line can contain the end of one plan line and the start
            # of the next. It belongs to whichever it matches better, so a long line
            # cannot swallow the beginning of the line after it.
            if following_tokens and _containment(
                following_tokens, tokens
            ) > _containment(cue_tokens, tokens):
                break
            last += 1
        cursor = last + 1
        placed.append((cue, transcript[first].start, transcript[last].end))

    if not fallback:
        cues = [
            _with_span(cue, start, end)
            for entry in placed
            if entry is not None
            for cue, start, end in (entry,)
        ]
        return tuple(cues), unmatched

    limit = None if audio_duration is None else audio_duration + 0.25
    anchors = [entry for entry in placed if entry is not None]
    cues: list[PlanCue] = []
    for index, entry in enumerate(placed):
        if entry is not None:
            cue, start, end = entry
        else:
            authored = plan.cues[index]
            start = _interpolated_start(plan.cues, index, anchors)
            if cues:
                # Never place a guessed line inside the neighbour that is already
                # speaking: the estimate is only allowed to move it later.
                start = max(start, cues[-1].end)
            end = start + authored.duration
            # ...and never over the matched line that follows it either.
            following = [
                item for item in anchors if item[0].index > authored.index
            ]
            if following and following[0][1] > start:
                end = min(end, following[0][1])
            cue = authored
        if limit is not None and start >= limit:
            continue
        if limit is not None:
            end = min(end, limit)
        if end <= start:
            continue
        cues.append(_with_span(cue, start, end))
    cues.sort(key=lambda item: (item.start, item.end))
    return tuple(cues), unmatched


def _interpolated_start(
    cues: Sequence[PlanCue],
    index: int,
    anchors: Sequence[tuple[PlanCue, float, float]],
) -> float:
    """Return a start time for an unmatched line, between its matched neighbours."""

    before = [entry for entry in anchors if entry[0].index < cues[index].index]
    after = [entry for entry in anchors if entry[0].index > cues[index].index]
    target = cues[index].start
    previous = before[-1] if before else None
    following = after[0] if after else None

    if previous is not None and following is not None:
        span = following[0].start - previous[0].start
        if span > 0:
            ratio = min(max((target - previous[0].start) / span, 0.0), 1.0)
            return previous[1] + ratio * (following[1] - previous[1])
    if previous is not None:
        return previous[2] + max(0.0, target - previous[0].end)
    if following is not None:
        return max(0.0, following[1] - max(0.0, following[0].start - target))
    return max(0.0, target)


def _with_span(cue: PlanCue, start: float, end: float) -> PlanCue:
    """Return ``cue`` with a new span, keeping everything the plan supplied."""

    if start == cue.start and end == cue.end:
        return cue
    return PlanCue(
        index=cue.index,
        cue=cue.cue,
        speaker=cue.speaker,
        start=start,
        end=end,
        source_text=cue.source_text,
        amharic=cue.amharic,
        emotion=cue.emotion,
        intensity=cue.intensity,
        delivery=cue.delivery,
        pause_before=cue.pause_before,
        pause_after=cue.pause_after,
    )


@dataclass(frozen=True, slots=True)
class PlanScope:
    """The plan lines that belong to this audio file, on its own timeline."""

    cues: tuple[PlanCue, ...] = ()
    offset: float = 0.0
    dropped_outside: int = 0
    dropped_unmatched: int = 0
    matched: int = 0
    interpolated: int = 0
    method: str = ""

    def __len__(self) -> int:
        return len(self.cues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "offset_seconds": round(self.offset, 3),
            "kept": len(self.cues),
            "matched": self.matched,
            "placed_between_neighbours": self.interpolated,
            "dropped_outside_audio": self.dropped_outside,
            "dropped_unmatched": self.dropped_unmatched,
            "method": self.method,
        }


def _containment(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Return how completely the shorter side's words appear in the longer one."""

    if not left or not right:
        return 0.0
    shared = len(set(left) & set(right))
    return shared / min(len(set(left)), len(set(right)))


def _best_offset(
    plan: DialoguePlan, transcript: Sequence[TranscriptSegment]
) -> tuple[float, int]:
    """Return the median shift between the plan and the transcript, and its support."""

    line_tokens = [_tokens(line.text) for line in transcript]
    votes: list[tuple[float, float]] = []
    for cue in plan.cues:
        cue_tokens = _tokens(cue.source_text or cue.amharic)
        if len(set(cue_tokens)) < 3:
            continue
        best: tuple[float, float] | None = None
        for index, tokens in enumerate(line_tokens):
            if not tokens or len(set(cue_tokens) & set(tokens)) < 3:
                continue
            # A plan cue often covers several transcriber fragments, and a fragment
            # covers part of a cue, so the comparison is containment rather than
            # whole-line similarity.
            fraction = _containment(cue_tokens, tokens)
            if fraction < MIN_MATCH_FRACTION:
                continue
            offset = transcript[index].start - cue.start
            if abs(offset) > MAX_PLAN_OFFSET_SECONDS:
                continue
            if best is None or fraction > best[0]:
                best = (fraction, offset)
        if best is not None:
            votes.append((best[0], best[1]))
    if not votes:
        return 0.0, 0

    median = statistics.median(offset for _, offset in votes)
    agreeing = [
        (weight, offset) for weight, offset in votes if abs(offset - median) <= 2.0
    ]
    if len(agreeing) < MIN_OFFSET_VOTES or len(agreeing) * 2 < len(votes):
        return 0.0, 0
    total = sum(weight for weight, _ in agreeing)
    refined = sum(weight * offset for weight, offset in agreeing) / total
    if abs(refined) < MIN_PLAN_OFFSET_SECONDS:
        # Already on this audio's timeline; a sub-second nudge would only move a
        # correctly authored plan off its own cues.
        return 0.0, len(agreeing)
    return float(refined), len(agreeing)


#: Share of plan lines whose authored timing must actually contain their own words
#: before those timings are trusted. Below this the plan is on a different grid from
#: the audio - an episode subtitle grid, or a hand alignment that drifted - and the
#: lines are re-timed from the transcript instead of being voiced at the wrong times.
MIN_AUTHORED_FIT = 0.60

#: Share of plan lines that must be *found* in the transcript before the transcript's
#: timings are trusted over the plan's. Below this there are too few anchors to
#: interpolate between: on a real episode 70 of 571 lines matched, 501 were then
#: invented, and every later stage - casting, windows, tempo - inherited the error.
#: A plan nobody could match to the audio is a plan whose own times are the better
#: evidence, so it is used as written and the problem is reported instead.
MIN_RETIME_MATCH_FRACTION = 0.50


def _authored_fit(
    cues: Sequence[PlanCue],
    transcript: Sequence[TranscriptSegment],
    *,
    threshold: float = MIN_MATCH_FRACTION,
) -> float:
    """Return the share of lines whose authored span contains their own words."""

    if not cues or not transcript:
        return 1.0
    matched = 0
    for cue in cues:
        cue_tokens = _tokens(cue.source_text or cue.amharic)
        if len(set(cue_tokens)) < 3:
            matched += 1
            continue
        for line in transcript:
            overlap = min(cue.end, line.end) - max(cue.start, line.start)
            if overlap <= 0 or overlap < 0.25 * cue.duration:
                continue
            if _containment(cue_tokens, _tokens(line.text)) >= threshold:
                matched += 1
                break
    return matched / len(cues)


def _inside_audio(cues: Sequence[PlanCue], audio_duration: float | None) -> tuple[PlanCue, ...]:
    """Return the cues that fall inside ``audio_duration``, in order."""

    if audio_duration is None:
        return tuple(cues)
    limit = audio_duration + 0.25
    return tuple(cue for cue in cues if cue.start < limit and cue.end <= limit)


def scope_plan(
    plan: DialoguePlan,
    *,
    transcript: Sequence[TranscriptSegment] = (),
    audio_duration: float | None = None,
    retime: bool = True,
    min_match_fraction: float = MIN_MATCH_FRACTION,
) -> PlanScope:
    """Put the plan onto this audio file's timeline and drop what is not in it.

    The plan is the source of truth for the dialogue - its lines, speakers and
    Amharic - and the audio is the source of truth for timing. So by default every
    line is given the span of the transcribed speech that says its words, which is
    what makes the cast correct: casting asks "who was speaking then", and the answer
    is only meaningful on the audio's own clock.

    * ``retime=True`` (default) re-times from the transcript. A line the transcript
      never recognised is placed between its matched neighbours rather than dropped,
      because the plan says it exists.
    * ``retime=False`` uses the authored timings as written. It is the escape hatch
      for a plan whose timing was already measured against this exact video, and the
      measured whole-plan shift is still applied when one is clearly present.
    """

    if not plan.cues:
        return PlanScope(method="empty plan")

    if retime and transcript:
        cues, unmatched = retime_plan(
            plan,
            transcript,
            threshold=min_match_fraction,
            fallback=True,
            audio_duration=audio_duration,
        )
        placed_between = max(0, len(cues) - (len(plan.cues) - unmatched))
        limit = None if audio_duration is None else audio_duration + 0.25
        dropped_outside = 0
        if limit is not None:
            dropped_outside = len(plan.cues) - len(cues) - (unmatched if unmatched else 0)
        if not cues:
            return PlanScope(
                method="no plan line could be placed in this audio",
                dropped_unmatched=unmatched,
            )
        matched = len(cues) - placed_between
        if matched < MIN_RETIME_MATCH_FRACTION * len(plan.cues):
            # Too few anchors to interpolate from. Voicing 501 invented timings is
            # what put the previous episode 3 minutes out of step, so the plan's own
            # times are kept - they are the only evidence left - and the caller is told
            # the match failed rather than being handed a silent approximation.
            offset, votes = _best_offset(plan, transcript)
            shifted = [cue.shifted(offset) for cue in plan.cues]
            kept = _inside_audio(shifted, audio_duration)
            detail = (
                f"only {matched} of {len(plan.cues)} line(s) were found in the "
                "transcript, which is too few to re-time from; the plan's own "
                "timings are used"
            )
            if offset:
                detail += f", shifted {offset:+.2f}s by {votes} matched line(s)"
            detail += ". Check that the plan belongs to this video"
            return PlanScope(
                cues=kept,
                offset=offset,
                matched=votes,
                dropped_outside=max(0, len(shifted) - len(kept)),
                method=detail,
            )
        detail = f"{matched} matched to the transcript"
        if placed_between:
            detail += f", {placed_between} placed between their neighbours"
        return PlanScope(
            cues=cues,
            matched=matched,
            interpolated=placed_between,
            dropped_unmatched=unmatched,
            dropped_outside=max(0, dropped_outside),
            method=f"timing from the transcript ({detail})",
        )

    offset = 0.0
    matched = 0
    method = "as authored"
    if transcript:
        offset, matched = _best_offset(plan, transcript)
        if offset:
            method = f"shifted {offset:+.2f}s by {matched} matched line(s)"

    shifted = [cue.shifted(offset) for cue in plan.cues]
    if audio_duration is None:
        return PlanScope(
            cues=tuple(shifted), offset=offset, matched=matched, method=method
        )

    limit = audio_duration + 0.25
    transcript_tokens = [_tokens(line.text) for line in transcript]
    kept: list[PlanCue] = []
    outside = 0
    unmatched = 0
    for cue in shifted:
        if cue.start >= limit or cue.end > limit:
            outside += 1
            continue
        if transcript_tokens:
            cue_tokens = _tokens(cue.source_text or cue.amharic)
            if len(set(cue_tokens)) >= 3:
                best = max(
                    (
                        _containment(cue_tokens, tokens)
                        for tokens in transcript_tokens
                        if tokens
                    ),
                    default=0.0,
                )
                if best < min_match_fraction:
                    unmatched += 1
                    continue
        kept.append(cue)

    if not kept:
        method += "; no plan line falls inside this audio"
    elif transcript:
        fit = _authored_fit(tuple(kept), transcript)
        if fit < MIN_AUTHORED_FIT:
            method += (
                f"; WARNING the authored timings match only {fit:.0%} of the lines, "
                "so drop --plan-keep-timing and let the video set the timing"
            )
    return PlanScope(
        cues=tuple(kept),
        offset=offset,
        dropped_outside=outside,
        dropped_unmatched=unmatched,
        matched=matched,
        method=method,
    )


#: Prefix for a character who has no measured voice in the film and is instead
#: voiced from the native-Amharic bank. The id is synthetic but stable, so a profile
#: and its clip directory are reused between runs.
BANK_SPEAKER_PREFIX = "CAST_"


#: Prefix for a character the plan names but the diarization could not tie to one
#: measured voice. Their own authored windows are still the best evidence available,
#: so the reference is cut from those under this stable id rather than immediately
#: borrowing a bank voice.
FILM_SPEAKER_PREFIX = "FILM_"


def bank_speaker_id(character: str) -> str:
    """Return the synthetic speaker id used for a bank-cast character."""

    slug = re.sub(r"[^A-Za-z0-9]+", "_", character).strip("_").upper()
    return f"{BANK_SPEAKER_PREFIX}{slug or 'CHARACTER'}"


def film_speaker_id(character: str) -> str:
    """Return the stable id for a character voiced from their own plan windows."""

    slug = re.sub(r"[^A-Za-z0-9]+", "_", character).strip("_").upper()
    return f"{FILM_SPEAKER_PREFIX}{slug or 'CHARACTER'}"


@dataclass(frozen=True, slots=True)
class CharacterCast:
    """One plan character and the measured voice it was given."""

    character: str
    speaker_id: str | None
    gender: str
    speaker_gender: str
    share: float
    seconds: float
    runner_up: str | None
    reason: str
    #: ``film`` when the voice was measured in this audio, ``voice_bank`` when the
    #: character had no measured voice and the native-Amharic bank supplied one.
    source: str = "film"

    def as_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "speaker_id": self.speaker_id,
            "source": self.source,
            "gender": self.gender,
            "speaker_gender": self.speaker_gender,
            "share": round(self.share, 3),
            "seconds": round(self.seconds, 2),
            "runner_up": self.runner_up,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class PlanCasting:
    """The character-to-voice map, and what could not be decided."""

    casts: tuple[CharacterCast, ...] = ()
    skipped: str = ""

    def __len__(self) -> int:
        return len(self.casts)

    @property
    def mapping(self) -> dict[str, str]:
        return {
            cast.character: cast.speaker_id
            for cast in self.casts
            if cast.speaker_id is not None
        }

    @property
    def unassigned(self) -> tuple[CharacterCast, ...]:
        return tuple(cast for cast in self.casts if cast.speaker_id is None)

    @property
    def bank_casts(self) -> tuple[CharacterCast, ...]:
        """Characters voiced from the bank because the film gave them no voice."""

        return tuple(cast for cast in self.casts if cast.source == "voice_bank")

    def as_dict(self) -> dict[str, Any]:
        return {
            "cast": [cast.as_dict() for cast in self.casts],
            "unassigned": [cast.character for cast in self.unassigned],
            "mapping": self.mapping,
            "skipped": self.skipped,
        }


def _presentation(value: object) -> str:
    prop = getattr(value, "presentation", None)
    if isinstance(prop, str) and prop:
        return prop
    median = getattr(value, "median_hz", value)
    if isinstance(median, bool) or not isinstance(median, (int, float)):
        return "unknown"
    number = float(median)
    if number <= 150.0:
        return "masculine"
    if number >= 175.0:
        return "feminine"
    return "uncertain"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_film_identities(casting: PlanCasting) -> PlanCasting:
    """Give every unmapped character an identity cut from their own plan windows.

    The plan states who speaks and when, so a character the diarization could not tie
    to one measured voice is still measurable directly: their own lines are where they
    spoke. Cutting the reference from those windows is the actor's voice - the thing a
    dub is supposed to preserve - while dropping straight to a bank voice makes the
    character somebody else entirely.

    The reference is only cut if it is clean; :mod:`app.pipeline.voice_profiles`
    decides that and reports what it refused, and only then does the bank apply.
    """

    if not casting.unassigned:
        return casting
    resolved: list[CharacterCast] = []
    for cast in casting.casts:
        if cast.speaker_id is not None:
            resolved.append(cast)
            continue
        resolved.append(
            CharacterCast(
                character=cast.character,
                speaker_id=film_speaker_id(cast.character),
                gender=cast.gender,
                speaker_gender=cast.speaker_gender,
                share=cast.share,
                seconds=cast.seconds,
                runner_up=cast.runner_up,
                reason=(
                    f"{cast.reason}; the reference is cut from this character's own "
                    "authored lines, and the voice bank is used only if they supply no "
                    "clean window"
                ),
                source="film",
            )
        )
    return PlanCasting(casts=tuple(resolved), skipped=casting.skipped)


def assign_bank_voices(casting: PlanCasting) -> PlanCasting:
    """Give a synthetic speaker id to every character the film could not voice.

    A character who never speaks alone in this footage has no measurable voice, but
    the plan still has lines for them. Dropping those lines would edit the dialogue,
    so each gets a stable synthetic id instead and the native-Amharic bank supplies
    the voice - matched by the gender the plan states for them.
    """

    if not casting.unassigned:
        return casting
    resolved: list[CharacterCast] = []
    for cast in casting.casts:
        if cast.speaker_id is not None:
            resolved.append(cast)
            continue
        resolved.append(
            CharacterCast(
                character=cast.character,
                speaker_id=bank_speaker_id(cast.character),
                gender=cast.gender,
                speaker_gender=cast.gender,
                share=cast.share,
                seconds=cast.seconds,
                runner_up=cast.runner_up,
                reason=f"{cast.reason}; voiced from the voice bank",
                source="voice_bank",
            )
        )
    return PlanCasting(casts=tuple(resolved), skipped=casting.skipped)


def cast_plan(
    plan: DialoguePlan,
    cues: Sequence[PlanCue],
    *,
    turns: Iterable[object] = (),
    registers: Mapping[str, object] | None = None,
    crosstalk: Iterable[object] = (),
    verified: Iterable[object] = (),
) -> PlanCasting:
    """Tie each plan character to the diarized voice that carries its lines.

    The evidence is the diarization itself: a plan line has exact start and end
    times, so the turn that covers it is a measurement of who spoke it. A line
    inside simultaneous speech is left out, because both voices are present and it
    cannot distinguish them.
    """

    if not cues:
        return PlanCasting(skipped="no plan lines in this audio")
    turn_list = [turn for turn in turns if getattr(turn, "speaker_id", None)]
    if not turn_list:
        return PlanCasting(skipped="no diarized turns to cast against")
    region_list = list(crosstalk)
    verified_list = list(verified)

    per_character: dict[str, Counter[str]] = {}
    per_character_seconds: dict[str, float] = {}

    for cue in cues:
        ambiguous = False
        for region in region_list:
            shared = _overlap(
                cue.start,
                cue.end,
                float(getattr(region, "start", 0.0)),
                float(getattr(region, "end", 0.0)),
            )
            if cue.duration > 0 and shared / cue.duration > MAX_AMBIGUOUS_OVERLAP_SHARE:
                ambiguous = True
                break
        if ambiguous:
            continue

        local: Counter[str] = Counter()
        measured = 0.0
        for turn in turn_list:
            shared = _overlap(
                cue.start,
                cue.end,
                float(getattr(turn, "start", 0.0)),
                float(getattr(turn, "end", 0.0)),
            )
            if shared <= 0.0:
                continue
            local[str(turn.speaker_id)] += shared
            measured += shared
        if measured <= 0.0:
            continue

        bucket = per_character.setdefault(cue.speaker, Counter())
        for speaker_id, seconds in local.items():
            bucket[speaker_id] += seconds
        per_character_seconds[cue.speaker] = (
            per_character_seconds.get(cue.speaker, 0.0) + measured
        )
        # A turn the embeddings settled at this line's own timescale is the
        # strongest evidence available, so it is counted twice.
        for turn in verified_list:
            speaker_id = str(getattr(turn, "speaker_id", ""))
            if speaker_id not in local:
                continue
            shared = _overlap(
                cue.start,
                cue.end,
                float(getattr(turn, "start", 0.0)),
                float(getattr(turn, "end", 0.0)),
            )
            if cue.duration > 0 and shared / cue.duration >= 0.8:
                bucket[speaker_id] += shared

    register_values = dict(registers or {})
    casts: list[CharacterCast] = []
    for character in plan.speakers:
        bucket = per_character.get(character)
        gender = plan.gender_of(character)
        if not bucket:
            casts.append(
                CharacterCast(
                    character=character,
                    speaker_id=None,
                    gender=gender,
                    speaker_gender="unknown",
                    share=0.0,
                    seconds=0.0,
                    runner_up=None,
                    reason=(
                        "no line of this character overlaps a single measured voice "
                        "(not in this footage, or always inside crosstalk)"
                    ),
                )
            )
            continue

        # The plan states each character's gender, so a voice that contradicts it is
        # not a candidate at all: on a full episode two characters can share a
        # register, and the authored gender is the stronger evidence about who is who.
        if gender in ("masculine", "feminine"):
            matching = {
                speaker_id: seconds
                for speaker_id, seconds in bucket.items()
                if speaker_id not in register_values
                or _presentation(register_values[speaker_id]) in (gender, "unknown", "uncertain")
            }
            if matching:
                bucket = Counter(matching)
            else:
                casts.append(
                    CharacterCast(
                        character=character,
                        speaker_id=None,
                        gender=gender,
                        speaker_gender=_presentation(
                            register_values[bucket.most_common(1)[0][0]]
                        )
                        if bucket.most_common(1)[0][0] in register_values
                        else "unknown",
                        share=0.0,
                        seconds=bucket.most_common(1)[0][1],
                        runner_up=None,
                        reason=(
                            f"the plan says {gender}, and none of the voices carrying "
                            "this character measures that way"
                        ),
                    )
                )
                continue

        total = sum(bucket.values())
        ranked = bucket.most_common()
        best_id, best_seconds = ranked[0]
        runner_up = ranked[1][0] if len(ranked) > 1 else None
        runner_seconds = ranked[1][1] if len(ranked) > 1 else 0.0
        share = best_seconds / total if total else 0.0
        speaker_gender = (
            _presentation(register_values[best_id])
            if best_id in register_values
            else "unknown"
        )

        problem = ""
        if per_character_seconds.get(character, 0.0) < MIN_CAST_SECONDS:
            problem = "too little measured speech to identify this character's voice"
        elif share < MIN_CAST_SHARE:
            problem = (
                f"no single voice carries this character ({share:.0%} at most); "
                "left unassigned rather than guessed"
            )
        elif runner_seconds and best_seconds < runner_seconds * MIN_CAST_MARGIN:
            problem = (
                f"{best_id} and {runner_up} both explain this character "
                f"({best_seconds:.1f}s vs {runner_seconds:.1f}s); left unassigned"
            )
        else:
            contradiction = (
                gender in ("masculine", "feminine")
                and speaker_gender in ("masculine", "feminine")
                and gender != speaker_gender
            )
            if contradiction and share < 0.85:
                problem = (
                    f"{best_id} measures {speaker_gender}, the plan says {gender}; "
                    "left unassigned"
                )
            elif contradiction:
                problem = (
                    f"{best_id} measures {speaker_gender}, the plan says {gender}; "
                    "kept on the timing evidence, verify by ear"
                )

        if problem and "kept on the timing evidence" not in problem:
            casts.append(
                CharacterCast(
                    character=character,
                    speaker_id=None,
                    gender=gender,
                    speaker_gender=speaker_gender,
                    share=share,
                    seconds=best_seconds,
                    runner_up=runner_up,
                    reason=problem,
                )
            )
            continue

        casts.append(
            CharacterCast(
                character=character,
                speaker_id=best_id,
                gender=gender,
                speaker_gender=speaker_gender,
                share=share,
                seconds=best_seconds,
                runner_up=runner_up,
                reason=problem or f"{share:.0%} of this character's lines are {best_id}",
            )
        )

    # Two characters cannot share one voice. When they do, the weaker claim is
    # dropped so the other keeps a voice and the run can say what happened.
    claims = Counter(cast.speaker_id for cast in casts if cast.speaker_id is not None)
    duplicated = {speaker for speaker, count in claims.items() if count > 1}
    if duplicated:
        strongest: dict[str, CharacterCast] = {}
        for cast in casts:
            if cast.speaker_id in duplicated:
                current = strongest.get(cast.speaker_id)  # type: ignore[index]
                if current is None or cast.seconds > current.seconds:
                    strongest[cast.speaker_id] = cast  # type: ignore[index]
        resolved: list[CharacterCast] = []
        for cast in casts:
            if (
                cast.speaker_id in duplicated
                and strongest.get(cast.speaker_id) is not cast
            ):
                resolved.append(
                    CharacterCast(
                        character=cast.character,
                        speaker_id=None,
                        gender=cast.gender,
                        speaker_gender=cast.speaker_gender,
                        share=cast.share,
                        seconds=cast.seconds,
                        runner_up=cast.runner_up,
                        reason=(
                            f"{cast.speaker_id} is already the voice of "
                            f"{strongest[cast.speaker_id].character}"
                        ),
                    )
                )
            else:
                resolved.append(cast)
        casts = resolved

    return PlanCasting(casts=tuple(casts))


#: Longest extra silence a plan line may use beyond its own spoken span, in
#: seconds. The plan's Amharic was written to be spoken, not to be squeezed into the
#: English actor's window, so a line may run into the silence that follows it - but
#: only a little of it, or a line would drift far from the face saying it.
MAX_WINDOW_GROWTH_SECONDS = 3.0

#: Silence kept before the next line so two voices never sound at once.
MIN_WINDOW_GAP_SECONDS = 0.12


def plan_dialogue(
    plan: DialoguePlan,
    scope: PlanScope,
    casting: PlanCasting,
    *,
    min_gap: float = MIN_WINDOW_GAP_SECONDS,
    max_growth: float = MAX_WINDOW_GROWTH_SECONDS,
) -> tuple[AdaptedDialogue, ...]:
    """Return the plan's lines as the dialogue stage's own type.

    Only characters that were cast are returned. A line whose character has no
    measured voice is dropped here rather than given somebody else's.

    Each line's ``end`` is its window for synthesis, and it is widened into the
    silence before the next line. The plan's Amharic is written to be *spoken*, so a
    faithful rendering is often longer than the English actor's window; without the
    spare room the timing stage would stretch it - or give up and report it unfitted.
    The line still begins where the actor began, and it still stops before the next
    character starts, which is what keeps two voices from colliding.
    """

    mapping = casting.mapping
    voiced = [cue for cue in scope.cues if mapping.get(cue.speaker)]
    lines: list[AdaptedDialogue] = []
    for position, cue in enumerate(voiced):
        speaker_id = mapping.get(cue.speaker)
        if speaker_id is None or not cue.amharic.strip():
            continue
        end = cue.end
        if position + 1 < len(voiced):
            next_start = voiced[position + 1].start
            room = next_start - min_gap
            if room > end:
                end = min(end + max_growth, room)
        if end <= cue.start:
            end = cue.end
        lines.append(
            AdaptedDialogue(
                speaker_id=speaker_id,
                start=cue.start,
                end=end,
                source_text=cue.source_text or cue.amharic,
                amharic=cue.amharic,
                emotion=cue.emotion or "neutral",
                intensity=cue.intensity,
                delivery=cue.delivery or "neutral",
                pause_before=cue.pause_before,
                pause_after=cue.pause_after,
            )
        )
    return tuple(lines)


def dropped_cues(scope: PlanScope, casting: PlanCasting) -> int:
    """Return how many plan lines in scope have no cast voice."""

    mapping = casting.mapping
    return sum(
        1
        for cue in scope.cues
        if mapping.get(cue.speaker) is None or not cue.amharic.strip()
    )


@dataclass(frozen=True, slots=True)
class PlanReport:
    """Everything a run should record about the plan it used."""

    plan: DialoguePlan = field(default_factory=DialoguePlan)
    scope: PlanScope = field(default_factory=PlanScope)
    casting: PlanCasting = field(default_factory=PlanCasting)

    def as_dict(self) -> dict[str, Any]:
        return {
            "file": None if self.plan.source is None else str(self.plan.source),
            "authored_cues": len(self.plan.cues),
            "timeline": self.scope.as_dict(),
            "casting": self.casting.as_dict(),
            "dropped_no_voice": dropped_cues(self.scope, self.casting),
            "voiced": len(plan_dialogue(self.plan, self.scope, self.casting)),
        }


__all__ = [
    "MAX_WINDOW_GROWTH_SECONDS",
    "MIN_WINDOW_GAP_SECONDS",
    "CharacterCast",
    "DialoguePlan",
    "PlanCasting",
    "PlanCharacter",
    "PlanCue",
    "PlanError",
    "PlanReport",
    "PlanScope",
    "assign_bank_voices",
    "assign_film_identities",
    "bank_speaker_id",
    "cast_plan",
    "discover_plan",
    "dropped_cues",
    "film_speaker_id",
    "load_plan",
    "parse_time",
    "plan_dialogue",
    "plan_from_payload",
    "retime_plan",
    "scope_plan",
]
