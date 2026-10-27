"""Reconcile transcribed lines with a sourced, speaker-labelled episode script.

Diarization is good at finding *where* speech is and *how many* voices there are.
It is much weaker at splitting one long turn when a second character interjects,
and at naming a 0.4-second fragment. That is the failure this module exists for:
on the real run, a single verified Skyler turn spans 18-37 seconds, so Walt's
"That is not the truth." and his repeated "I don't want to hear about the police."
inherited Skyler's voice.

The strongest cheap evidence for "who said this line" is the episode script. It is
already what a human dubber works from, it is stable across runs, and it can be
cached. This module aligns the ASR words to the script words, maps script
characters to the diarized voices using the researched guide and the measured
registers, and then moves a label only when the script match is strong enough.

Deliberate safeguards
---------------------
* **Script is supporting evidence, not audio truth.** A line whose label was
  confirmed by speaker embeddings *at that line's own timescale* is never moved.
  A long verified turn does not verify every interjection inside it.
* **No mapping, no change.** If the script names a character who cannot be tied to
  a diarized voice, the line is preserved for QC.
* **Unmatched lines stay as they are** and are reported. Dropping dialogue is worse
  than flagging it.
* **Everything is deterministic.** The model's only job was to turn a fetched page
  into a list of `(speaker, text)` lines; the alignment and the decision are local
  and testable.
"""

from __future__ import annotations

import difflib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.pipeline.dialogue_context import CharacterGuide
from app.pipeline.register import presentation_of
from app.pipeline.transcription import TranscriptSegment

#: Fraction of a line's words that must align to the script before its speaker is
#: considered proven. ASR drops and invents a word or two; half the words is a real
#: match, a quarter is a coincidence.
MIN_SCRIPT_MATCH_FRACTION = 0.50

#: Share of the matched words that must agree on one script speaker. Below this the
#: line straddles two script lines and is not safe to move.
MIN_SCRIPT_SPEAKER_PURITY = 0.70

#: Minimum number of matched words for a first-pass match. A one-word line is
#: matched in the second pass against the script lines between its neighbours.
MIN_SCRIPT_MATCHED_WORDS = 2

#: Similarity needed when a low-confidence line is matched against the script lines
#: between its matched neighbours.
MIN_SCRIPT_LINE_SIMILARITY = 0.62

#: A script mapping is only trusted when the character's pitch agrees with the
#: mapped voice within this ratio, or when the gender leaves exactly one candidate.
MAPPING_PITCH_RATIO = 1.60

#: A line is only "embedding verified" when the verified turn is about as long as
#: the line itself. A 19-second turn does not verify an interjection inside it.
VERIFIED_OVERLAP_FRACTION = 0.90
VERIFIED_DURATION_TOLERANCE = 0.50

#: Duplicate dialogue at a different time is usually separation bleed. It is
#: reported, never silently dropped.
DUPLICATE_TEXT_RATIO = 0.85


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_BRACKETED = re.compile(r"\[[^\]]*\]")
#: Words too common to identify a script line on their own. A one-word ASR line
#: ("No.", "Yes.") is left to the neighbours rather than matched to whichever
#: script line happens to contain it.
_COMMON_TOKENS = frozenset(
    """
    a an the and or but if of to in on at for with from by as is are was were am
    be been being do does did have has had i you he she it we they me him her us them
    my your his its our their this that these those not no yes oh well so then
    what who whom whose where when why how there here
    """.split()
)

_CONTRACTIONS = {
    "don't": "dont",
    "doesn't": "doesnt",
    "didn't": "didnt",
    "can't": "cant",
    "couldn't": "couldnt",
    "won't": "wont",
    "wouldn't": "wouldnt",
    "shouldn't": "shouldnt",
    "isn't": "isnt",
    "aren't": "arent",
    "wasn't": "wasnt",
    "weren't": "werent",
    "haven't": "havent",
    "hasn't": "hasnt",
    "hadn't": "hadnt",
    "i'm": "im",
    "i've": "ive",
    "i'll": "ill",
    "i'd": "id",
    "you're": "youre",
    "you've": "youve",
    "you'll": "youll",
    "we're": "were",
    "we've": "weve",
    "they're": "theyre",
    "he's": "hes",
    "she's": "shes",
    "it's": "its",
    "that's": "thats",
    "there's": "theres",
    "what's": "whats",
    "who's": "whos",
    "let's": "lets",
}


def _tokens(text: str) -> tuple[str, ...]:
    """Return comparable words, with stage directions and contractions normalised."""

    cleaned = _BRACKETED.sub(" ", text or "")
    # A fetched transcript sometimes carries a replacement character where an
    # apostrophe should be ("You�re"); removing it keeps "youre" comparable.
    cleaned = cleaned.replace("\ufffd", "").replace("\u2019", "'").casefold()
    words: list[str] = []
    for token in _TOKEN_RE.findall(cleaned):
        words.append(_CONTRACTIONS.get(token, token.replace("'", "")))
    return tuple(words)


@dataclass(frozen=True, slots=True)
class ScriptLine:
    """One spoken line from a sourced episode script."""

    speaker: str
    text: str

    def __post_init__(self) -> None:
        if not isinstance(self.speaker, str) or not self.speaker.strip():
            raise ValueError("a script line needs a speaker")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("a script line needs text")
        object.__setattr__(self, "speaker", self.speaker.strip())
        object.__setattr__(self, "text", self.text.strip())

    def as_dict(self) -> dict[str, str]:
        return {"speaker": self.speaker, "text": self.text}

    @classmethod
    def from_dict(cls, payload: object) -> "ScriptLine":
        if not isinstance(payload, Mapping):
            raise ValueError(f"a script line must be an object, got {type(payload).__name__}")
        return cls(
            speaker=str(payload.get("speaker", "") or ""),
            text=str(payload.get("text", "") or ""),
        )


@dataclass(frozen=True, slots=True)
class EpisodeScript:
    """Speaker-labelled dialogue for one episode, with the pages it came from."""

    title: str = ""
    season: int | None = None
    episode: int | None = None
    lines: tuple[ScriptLine, ...] = ()
    sources: tuple[str, ...] = ()
    extracted_at: str = ""

    def __post_init__(self) -> None:
        for label, value in (("season", self.season), ("episode", self.episode)):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{label} must be a non-negative integer or None")
        lines = tuple(self.lines)
        for line in lines:
            if not isinstance(line, ScriptLine):
                raise ValueError(
                    f"script lines must be ScriptLine, got {type(line).__name__}"
                )
        sources = tuple(
            dict.fromkeys(source.strip() for source in self.sources if str(source).strip())
        )
        object.__setattr__(self, "title", self.title.strip())
        object.__setattr__(self, "lines", lines)
        object.__setattr__(self, "sources", sources)

    def __len__(self) -> int:
        return len(self.lines)

    def __bool__(self) -> bool:
        return bool(self.lines)

    @property
    def label(self) -> str:
        if self.season is None and self.episode is None:
            return self.title
        parts = []
        if self.season is not None:
            parts.append(f"season {self.season}")
        if self.episode is not None:
            parts.append(f"episode {self.episode}")
        return f"{self.title} ({', '.join(parts)})" if self.title else ", ".join(parts)

    def summary(self) -> dict[str, Any]:
        """A small manifest view; the full script stays in its cache file."""

        return {
            "title": self.title,
            "season": self.season,
            "episode": self.episode,
            "lines": len(self.lines),
            "speakers": sorted({line.speaker for line in self.lines}),
            "sources": list(self.sources),
            "extracted_at": self.extracted_at,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "season": self.season,
            "episode": self.episode,
            "lines": [line.as_dict() for line in self.lines],
            "sources": list(self.sources),
            "extracted_at": self.extracted_at,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "EpisodeScript":
        if not isinstance(payload, Mapping):
            raise ValueError("an episode script must be a JSON object")
        lines = payload.get("lines", ()) or ()
        if not isinstance(lines, (list, tuple)):
            raise ValueError("script lines must be a list")
        return cls(
            title=str(payload.get("title", "") or ""),
            season=payload.get("season"),
            episode=payload.get("episode"),
            lines=tuple(ScriptLine.from_dict(item) for item in lines),
            sources=tuple(payload.get("sources", ()) or ()),
            extracted_at=str(payload.get("extracted_at", "") or ""),
        )

    def save(self, path: str | Path) -> Path:
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
    def load(cls, path: str | Path) -> "EpisodeScript | None":
        source = Path(path)
        if not source.is_file():
            return None
        try:
            script = cls.from_dict(json.loads(source.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None
        return script if script.lines else None


@dataclass(frozen=True, slots=True)
class ScriptMapping:
    """One script character tied to one diarized voice."""

    character: str
    speaker_id: str | None
    gender: str
    median_hz: float | None
    matched_lines: int
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "character": self.character,
            "speaker_id": self.speaker_id,
            "gender": self.gender,
            "median_hz": None if self.median_hz is None else round(self.median_hz, 1),
            "matched_lines": self.matched_lines,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ScriptChange:
    """A line whose speaker the script corrected."""

    index: int
    start: float
    end: float
    from_speaker: str
    to_speaker: str
    character: str
    script_line: int
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "from": self.from_speaker,
            "to": self.to_speaker,
            "character": self.character,
            "script_line": self.script_line + 1,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ScriptUncertain:
    """A line the script could not move safely, preserved for QC."""

    index: int
    start: float
    end: float
    from_speaker: str
    proposed_speaker: str
    character: str
    confidence: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "from": self.from_speaker,
            "proposed": self.proposed_speaker,
            "character": self.character,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ScriptReconciliation:
    """What the script alignment changed and what it left for a human."""

    applied: tuple[ScriptChange, ...] = ()
    uncertain: tuple[ScriptUncertain, ...] = ()
    mapping: tuple[ScriptMapping, ...] = ()
    matched_lines: int = 0
    skipped: str = ""

    def __bool__(self) -> bool:
        return bool(self.applied or self.uncertain)

    def as_dict(self) -> dict[str, Any]:
        return {
            "applied": [change.as_dict() for change in self.applied],
            "uncertain": [item.as_dict() for item in self.uncertain],
            "mapping": [item.as_dict() for item in self.mapping],
            "matched_lines": self.matched_lines,
            "skipped": self.skipped,
        }


@dataclass
class _LineMatch:
    index: int
    token_count: int
    matched: int = 0
    speakers: Counter[str] = field(default_factory=Counter)
    script_indexes: Counter[int] = field(default_factory=Counter)

    @property
    def fraction(self) -> float:
        return self.matched / self.token_count if self.token_count else 0.0

    @property
    def purity(self) -> float:
        return (max(self.speakers.values()) / self.matched) if self.matched else 0.0

    @property
    def character(self) -> str:
        return self.speakers.most_common(1)[0][0] if self.speakers else ""

    @property
    def script_index(self) -> int | None:
        return self.script_indexes.most_common(1)[0][0] if self.script_indexes else None

    @property
    def confidence(self) -> float:
        return self.fraction * self.purity

    @property
    def strong(self) -> bool:
        return (
            self.matched >= MIN_SCRIPT_MATCHED_WORDS
            and self.fraction >= MIN_SCRIPT_MATCH_FRACTION
            and self.purity >= MIN_SCRIPT_SPEAKER_PURITY
        )


@dataclass(frozen=True, slots=True)
class _Assignment:
    character: str
    script_index: int | None
    confidence: float


def _flatten_script(
    script: EpisodeScript,
) -> tuple[list[str], list[str], list[int]]:
    tokens: list[str] = []
    speakers: list[str] = []
    line_indexes: list[int] = []
    for index, line in enumerate(script.lines):
        words = _tokens(line.text)
        tokens.extend(words)
        speakers.extend([line.speaker] * len(words))
        line_indexes.extend([index] * len(words))
    return tokens, speakers, line_indexes


def _flatten_transcript(
    lines: Sequence[TranscriptSegment],
) -> tuple[list[str], list[int], list[int]]:
    tokens: list[str] = []
    line_indexes: list[int] = []
    token_counts: list[int] = []
    for index, line in enumerate(lines):
        words = _tokens(line.text)
        tokens.extend(words)
        line_indexes.extend([index] * len(words))
        token_counts.append(len(words))
    return tokens, line_indexes, token_counts


def _word_matches(
    script: EpisodeScript, lines: Sequence[TranscriptSegment]
) -> list[_LineMatch]:
    script_tokens, script_speakers, script_lines = _flatten_script(script)
    transcript_tokens, transcript_lines, token_counts = _flatten_transcript(lines)
    matches = [
        _LineMatch(index=index, token_count=count)
        for index, count in enumerate(token_counts)
    ]
    if not script_tokens or not transcript_tokens:
        return matches

    matcher = difflib.SequenceMatcher(
        None, script_tokens, transcript_tokens, autojunk=False
    )
    for a, b, size in matcher.get_matching_blocks():
        for offset in range(size):
            script_position = a + offset
            transcript_position = b + offset
            line_index = transcript_lines[transcript_position]
            match = matches[line_index]
            match.matched += 1
            match.speakers[script_speakers[script_position]] += 1
            match.script_indexes[script_lines[script_position]] += 1
    return matches


def _script_line_similarity(line_words: Sequence[str], script_words: Sequence[str]) -> float:
    if not line_words or not script_words:
        return 0.0
    matcher = difflib.SequenceMatcher(None, line_words, script_words, autojunk=False)
    ratio = matcher.ratio()
    matched = sum(block.size for block in matcher.get_matching_blocks())
    containment = matched / len(line_words)
    return max(ratio, containment * 0.95)


def _match_between_anchors(
    line_words: Sequence[str],
    script: EpisodeScript,
    *,
    low: int,
    high: int,
) -> tuple[int, float] | None:
    if high < low:
        low, high = high, low
    candidates = range(max(0, low), min(len(script.lines), high + 1))
    scored: list[tuple[float, int]] = []
    for index in candidates:
        score = _script_line_similarity(line_words, _tokens(script.lines[index].text))
        if score > 0:
            scored.append((score, index))
    if not scored:
        return None
    scored.sort(reverse=True)
    best_score, best_index = scored[0]
    if len(scored) > 1:
        margin = best_score - scored[1][0]
    else:
        margin = best_score
    if len(line_words) == 1:
        # "No." matches a hundred lines. Only a distinctive word that appears in
        # exactly one script line between the anchors is evidence.
        word = line_words[0]
        if word in _COMMON_TOKENS:
            return None
        exact = [
            index
            for index in candidates
            if word in set(_tokens(script.lines[index].text))
        ]
        if len(exact) == 1:
            return exact[0], 0.95
        return None
    if len(line_words) == 2:
        if all(word in _COMMON_TOKENS for word in line_words):
            return None
        if best_score < 0.999 or margin < 0.05 or len(scored) > 1:
            return None
        return best_index, best_score
    if best_score < MIN_SCRIPT_LINE_SIMILARITY:
        return None
    if margin < 0.06 and best_score < 0.9:
        return None
    return best_index, best_score


def _assign_lines(
    script: EpisodeScript, lines: Sequence[TranscriptSegment]
) -> list[_Assignment | None]:
    matches = _word_matches(script, lines)
    assignments: list[_Assignment | None] = [None] * len(lines)
    for index, match in enumerate(matches):
        if match.strong:
            assignments[index] = _Assignment(
                character=match.character,
                script_index=match.script_index,
                confidence=match.confidence,
            )

    anchors = [
        (index, assignment.script_index)
        for index, assignment in enumerate(assignments)
        if assignment is not None and assignment.script_index is not None
    ]
    script_last = max(len(script.lines) - 1, 0)
    for index, line in enumerate(lines):
        if assignments[index] is not None:
            continue
        words = _tokens(line.text)
        if not words:
            continue
        previous = [item for item in anchors if item[0] < index]
        following = [item for item in anchors if item[0] > index]
        low = previous[-1][1] if previous else 0
        high = following[0][1] if following else script_last
        # A partial word match already narrows the region; widen it a little for
        # an ASR line that dropped its distinctive words.
        voted = matches[index].script_index
        if voted is not None:
            low = min(low, max(0, voted - 3))
            high = max(high, min(script_last, voted + 3))
        best = _match_between_anchors(words, script, low=low - 2, high=high + 2)
        if best is None:
            continue
        script_index, score = best
        assignments[index] = _Assignment(
            character=script.lines[script_index].speaker,
            script_index=script_index,
            confidence=score * (0.95 if previous or following else 0.8),
        )
    return assignments


def _presentation(value: object) -> str:
    prop = getattr(value, "presentation", None)
    if isinstance(prop, str) and prop:
        return prop
    median = getattr(value, "median_hz", value)
    if isinstance(median, bool) or not isinstance(median, (int, float)):
        return "unknown"
    return presentation_of(float(median))


def _median_of(value: object) -> float | None:
    median = getattr(value, "median_hz", value)
    if isinstance(median, bool) or not isinstance(median, (int, float)):
        return None
    number = float(median)
    return number if math.isfinite(number) else None


def _character_gender(guide: CharacterGuide | None, name: str) -> str:
    if guide is None:
        return "unknown"
    word = name.strip().casefold()
    exact: list[str] = []
    prefixed: list[str] = []
    for character in guide.characters:
        names = {
            character.name.casefold(),
            *(alias.casefold() for alias in character.aliases),
        }
        if word in names:
            exact.append(character.gender)
            continue
        if len(word) >= 3 and any(
            other.startswith(word) or word.startswith(other)
            for other in names
            if len(other) >= 3
        ):
            prefixed.append(character.gender)
    if len(set(exact)) == 1:
        return exact[0]
    # "Walt" for "Walter White" is common in transcripts. Only accept a shortened
    # name when it points at exactly one researched character.
    if not exact and len(set(prefixed)) == 1:
        return prefixed[0]
    return "unknown"


def _median_pitch(
    indexes: Sequence[int],
    lines: Sequence[TranscriptSegment],
    pitches: Sequence[float | None] | None,
) -> float | None:
    if pitches is None:
        return None
    values = [
        pitch
        for index in indexes
        if (pitch := pitches[index]) is not None
        and pitch > 0.0
        and lines[index].duration >= 0.4
    ]
    if not values:
        return None
    return float(statistics.median(values))


def _map_characters(
    assignments: Sequence[_Assignment | None],
    lines: Sequence[TranscriptSegment],
    *,
    script: EpisodeScript,
    guide: CharacterGuide | None,
    registers: Mapping[str, object] | None,
    pitches: Sequence[float | None] | None,
) -> tuple[tuple[ScriptMapping, ...], dict[str, str]]:
    usage: dict[str, list[int]] = {}
    for index, assignment in enumerate(assignments):
        if assignment is not None:
            usage.setdefault(assignment.character, []).append(index)
    if not usage:
        return (), {}

    register_values = dict(registers or {})
    presentations = {sid: _presentation(value) for sid, value in register_values.items()}
    medians = {
        sid: median
        for sid, value in register_values.items()
        if (median := _median_of(value)) is not None
    }

    mappings: list[ScriptMapping] = []
    mapped: dict[str, str] = {}
    for character, indexes in sorted(usage.items(), key=lambda item: (-len(item[1]), item[0])):
        gender = _character_gender(guide, character)
        median = _median_pitch(indexes, lines, pitches)
        candidate_ids = [
            sid
            for sid in presentations
            if gender in ("masculine", "feminine") and presentations[sid] == gender
        ]
        chosen: str | None = None
        confidence = 0.0
        reason = ""
        # Diarization is wrong on individual interjections, but across every line
        # the script attributes to one character it is usually right. A dominant
        # label that also agrees with the researched gender is stronger evidence
        # than pitch, and it works when several characters share a gender.
        labelled = Counter(lines[index].speaker_id for index in indexes)
        dominant, dominant_count = labelled.most_common(1)[0]
        dominant_share = dominant_count / len(indexes)
        dominant_pitch_agrees = (
            median is None
            or dominant not in medians
            or abs(math.log(median / medians[dominant])) <= math.log(MAPPING_PITCH_RATIO)
        )
        if (
            len(indexes) >= 2
            and dominant_share >= 0.60
            and dominant_pitch_agrees
            and (
                gender not in ("masculine", "feminine")
                or presentations.get(dominant) in (gender, "unknown", "uncertain")
            )
        ):
            chosen = dominant
            confidence = min(1.0, len(indexes) / 3.0) * 0.9 * dominant_share
            reason = (
                f"{dominant_share:.0%} of this character's {len(indexes)} matched "
                f"lines were already labelled {dominant}"
            )
        if chosen is None and len(candidate_ids) == 1:
            chosen = candidate_ids[0]
            confidence = min(1.0, len(indexes) / 3.0) * 0.95
            reason = f"the only {gender} voice among the diarized speakers"
        elif chosen is None and median is not None and medians:
            ranked = sorted(
                medians.items(),
                key=lambda item: abs(math.log(median / item[1])),
            )
            best_id, best_median = ranked[0]
            gap = (
                abs(math.log(median / ranked[1][1]))
                if len(ranked) > 1
                else float("inf")
            )
            if len(ranked) == 1 or gap >= 0.12:
                if (
                    gender not in ("masculine", "feminine")
                    or presentations.get(best_id) in (gender, "unknown", "uncertain")
                ):
                    chosen = best_id
                    confidence = min(1.0, len(indexes) / 4.0) * 0.8
                    reason = (
                        f"median pitch {median:.0f} Hz closest to {best_id} "
                        f"({best_median:.0f} Hz)"
                    )
        if chosen is not None and median is not None and chosen in medians:
            ratio = abs(math.log(median / medians[chosen]))
            if ratio > math.log(MAPPING_PITCH_RATIO):
                confidence *= 0.6
                reason += f"; pitch differs by {ratio:.2f} log units"
        if chosen is None:
            reason = (
                "several diarized voices fit this character and the line-level "
                "pitch did not separate them"
            )
        mappings.append(
            ScriptMapping(
                character=character,
                speaker_id=chosen,
                gender=gender,
                median_hz=median,
                matched_lines=len(indexes),
                confidence=confidence,
                reason=reason,
            )
        )
        if chosen is not None:
            mapped[character] = chosen

    # Two script characters cannot share one voice. If they do, neither mapping is
    # evidence and both are reported unresolved rather than arbitrarily kept.
    counts = Counter(mapped.values())
    conflicts = {speaker for speaker, count in counts.items() if count > 1}
    if conflicts:
        for character, speaker in list(mapped.items()):
            if speaker in conflicts:
                mapped.pop(character, None)
        mappings = [
            ScriptMapping(
                character=item.character,
                speaker_id=None,
                gender=item.gender,
                median_hz=item.median_hz,
                matched_lines=item.matched_lines,
                confidence=0.0,
                reason=(
                    item.reason + "; two characters resolved to the same voice"
                    if item.reason
                    else "two characters resolved to the same voice"
                ),
            )
            for item in mappings
        ]
    return tuple(mappings), mapped


def _line_verified(
    line: TranscriptSegment, verified: Sequence[object]
) -> bool:
    """``True`` only when a verified turn covers this line at its own timescale."""

    for turn in verified:
        start = float(getattr(turn, "start", 0.0))
        end = float(getattr(turn, "end", 0.0))
        speaker = str(getattr(turn, "speaker_id", ""))
        if speaker != line.speaker_id:
            continue
        overlap = min(end, line.end) - max(start, line.start)
        if overlap <= 0:
            continue
        if overlap / max(line.end - line.start, 1e-6) < VERIFIED_OVERLAP_FRACTION:
            continue
        if abs((end - start) - (line.end - line.start)) > VERIFIED_DURATION_TOLERANCE:
            continue
        return True
    return False


def _duplicate_flags(
    lines: Sequence[TranscriptSegment],
    assignments: Sequence[_Assignment | None],
) -> dict[int, tuple[int, float]]:
    """Return ``later index -> (earlier index, similarity)`` for likely bleed.

    A film can genuinely repeat a line - Walt says "I don't want to hear about the
    police" twice in this scene - so a repeat is only suspicious when the script
    cannot explain it as two separate script lines.
    """

    normalized = [_tokens(line.text) for line in lines]
    flags: dict[int, tuple[int, float]] = {}
    for later in range(1, len(lines)):
        if not normalized[later]:
            continue
        later_assignment = assignments[later]
        for earlier in range(later):
            if not normalized[earlier]:
                continue
            gap = lines[later].start - lines[earlier].end
            if gap < 2.0:
                continue
            earlier_assignment = assignments[earlier]
            if (
                later_assignment is not None
                and earlier_assignment is not None
                and later_assignment.script_index is not None
                and earlier_assignment.script_index is not None
                and later_assignment.script_index != earlier_assignment.script_index
            ):
                # Two distinct script lines use these words: a real repetition.
                continue
            ratio = difflib.SequenceMatcher(
                None, normalized[earlier], normalized[later], autojunk=False
            ).ratio()
            if ratio >= DUPLICATE_TEXT_RATIO:
                flags[later] = (earlier, ratio)
                break
    return flags


def reconcile_speakers(
    lines: Sequence[TranscriptSegment],
    *,
    script: EpisodeScript | None,
    guide: CharacterGuide | None = None,
    registers: Mapping[str, object] | None = None,
    pitches: Sequence[float | None] | None = None,
    verified: Sequence[object] = (),
    threshold: float = 0.55,
) -> tuple[tuple[TranscriptSegment, ...], ScriptReconciliation]:
    """Use a sourced script to correct speaker labels before sentence assembly.

    ``pitches`` is one median fundamental frequency per line, or ``None`` where the
    line was too short to measure. It is only needed to separate characters of the
    same gender; the guide's gender plus the measured registers resolves the common
    two-character scene without it.
    """

    ordered = tuple(lines)
    if not ordered:
        return ordered, ScriptReconciliation(skipped="no transcribed lines")
    if script is None or not script.lines:
        return ordered, ScriptReconciliation(skipped="no episode script available")

    assignments = _assign_lines(script, ordered)
    mapping, mapped = _map_characters(
        assignments,
        ordered,
        script=script,
        guide=guide,
        registers=registers,
        pitches=pitches,
    )
    applied: list[ScriptChange] = []
    uncertain: list[ScriptUncertain] = []
    repaired = list(ordered)
    matched_lines = sum(1 for item in assignments if item is not None)

    for index, (line, assignment) in enumerate(zip(ordered, assignments)):
        if assignment is None:
            uncertain.append(
                ScriptUncertain(
                    index=index,
                    start=line.start,
                    end=line.end,
                    from_speaker=line.speaker_id,
                    proposed_speaker=line.speaker_id,
                    character="",
                    confidence=0.0,
                    reason="no matching line in the episode script",
                )
            )
            continue
        if assignment.confidence < threshold:
            uncertain.append(
                ScriptUncertain(
                    index=index,
                    start=line.start,
                    end=line.end,
                    from_speaker=line.speaker_id,
                    proposed_speaker=mapped.get(assignment.character, line.speaker_id),
                    character=assignment.character,
                    confidence=assignment.confidence,
                    reason=(
                        f"script match {assignment.confidence:.2f} is below the "
                        f"{threshold:.2f} threshold"
                    ),
                )
            )
            continue
        target = mapped.get(assignment.character)
        if target is None:
            uncertain.append(
                ScriptUncertain(
                    index=index,
                    start=line.start,
                    end=line.end,
                    from_speaker=line.speaker_id,
                    proposed_speaker=line.speaker_id,
                    character=assignment.character,
                    confidence=assignment.confidence,
                    reason=(
                        f"script says {assignment.character}, but that character "
                        "could not be tied to a diarized voice"
                    ),
                )
            )
            continue
        if target == line.speaker_id:
            continue
        if _line_verified(line, verified):
            uncertain.append(
                ScriptUncertain(
                    index=index,
                    start=line.start,
                    end=line.end,
                    from_speaker=line.speaker_id,
                    proposed_speaker=target,
                    character=assignment.character,
                    confidence=assignment.confidence,
                    reason=(
                        "the label was confirmed by speaker embeddings at this "
                        "line's own timescale; audio evidence wins"
                    ),
                )
            )
            continue
        repaired[index] = TranscriptSegment(
            speaker_id=target, start=line.start, end=line.end, text=line.text
        )
        script_line = assignment.script_index if assignment.script_index is not None else 0
        applied.append(
            ScriptChange(
                index=index,
                start=line.start,
                end=line.end,
                from_speaker=line.speaker_id,
                to_speaker=target,
                character=assignment.character,
                script_line=script_line,
                confidence=assignment.confidence,
                reason=f"episode script line {script_line + 1}",
            )
        )

    for index, (earlier, similarity) in sorted(
        _duplicate_flags(ordered, assignments).items()
    ):
        line = ordered[index]
        uncertain.append(
            ScriptUncertain(
                index=index,
                start=line.start,
                end=line.end,
                from_speaker=line.speaker_id,
                proposed_speaker=repaired[index].speaker_id,
                character="",
                confidence=similarity,
                reason=(
                    f"possible separation bleed: nearly the same words as line "
                    f"{earlier} at {ordered[earlier].start:.2f}s"
                ),
            )
        )

    return tuple(repaired), ScriptReconciliation(
        applied=tuple(applied),
        uncertain=tuple(uncertain),
        mapping=mapping,
        matched_lines=matched_lines,
    )


__all__ = [
    "DUPLICATE_TEXT_RATIO",
    "EpisodeScript",
    "ScriptChange",
    "ScriptLine",
    "ScriptMapping",
    "ScriptReconciliation",
    "ScriptUncertain",
    "reconcile_speakers",
]
