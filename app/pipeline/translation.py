"""Dialogue adaptation into dubbing-ready spoken Amharic (**DeepSeek**).

This module is the only place in the pipeline that talks to the DeepSeek API. It
turns the transcribed source-language lines produced by
:mod:`app.pipeline.transcription` into Amharic dialogue that a voice actor or a
TTS model can actually perform::

    [TranscriptSegment, ...] -> adapt_dialogue() -> [AdaptedDialogue, ...]

This is *adaptation*, not translation. The Amharic has to sound like something a
native speaker would say out loud in that scene, and it has to fit roughly the
time the original performance occupied. Each line also carries the performance
metadata the later voice-profile and TTS stages need: emotion, intensity,
delivery, and the pauses around the line.

Who owns what
-------------
The application owns identity and timing. Every input line is given a stable id
(``dialogue_000001``, ...); the model is asked for those ids, the Amharic text and
the performance metadata, and nothing else. ``speaker_id``, ``start``, ``end`` and
``source_text`` are always copied from the original
:class:`~app.pipeline.transcription.TranscriptSegment`, so the model can never
change who said what, or when. The speaker label is sent purely as context, so
the model can follow who is answering whom and keep pronouns and honorifics
consistent.

Context batching
----------------
Consecutive lines are adapted together in small bounded batches so the model can
follow the conversation, but a whole movie is never sent in one request. If any
line of a batch is missing, duplicated or malformed, the call fails instead of
silently dropping dialogue.

Contract with the ``openai`` package (checked against 3.22.1)
------------------------------------------------------------
* ``OpenAI(api_key=..., base_url=...)`` builds the client for DeepSeek's
  OpenAI-compatible endpoint.
* ``client.chat.completions.create(model=..., messages=[...],
  response_format={"type": "json_object"}, extra_body=...)`` returns an object
  whose ``choices[0].message.content`` holds the JSON text.
* ``openai.AuthenticationError`` (which is an ``openai.APIError``) reports
  rejected credentials.

Deliberate non-goals
--------------------
* **No retries.** The project has no retry abstraction, so failures are reported
  immediately and deterministically rather than masked.
* **No caching.** Re-runs re-bill; caching belongs to the orchestrator, which will
  own the ``*.amharic.json`` manifest.
* **No network access at import time** (and none in the test suite).
* **No DeepSeek objects in the public API.** Callers only ever see
  :class:`AdaptedDialogue`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
from app.pipeline.amharic_text import latin_spans
from app.pipeline.dialogue_context import (
    DEFAULT_BUDGET_TOLERANCE,
    DEFAULT_SYLLABLES_PER_SECOND,
    BudgetVerdict,
    CharacterBible,
    Scene,
    SyllableBudget,
    segment_scenes,
    syllable_budget,
)
from app.pipeline.transcription import TranscriptSegment

try:  # Runtime dependency (pulls in httpx). Guarded so that this module stays
    # importable (and testable) without the API client installed.
    from openai import OpenAI
    from openai import AuthenticationError as OpenAIAuthenticationError
except ImportError:  # pragma: no cover - only without the API client installed
    OpenAI = None  # type: ignore[assignment]
    OpenAIAuthenticationError = None  # type: ignore[assignment]


#: Prefix and width of the application-generated dialogue ids, e.g.
#: ``dialogue_000001``. The model must echo these back unchanged.
DIALOGUE_ID_PREFIX = "dialogue_"
DIALOGUE_ID_DIGITS = 6

#: The reuseable system prompt for cinematic English -> Amharic adaptation.
SYSTEM_PROMPT = """\
You are a veteran dialogue adaptor and dubbing script writer for Ethiopian cinema.

You adapt movie dialogue into Amharic for a DUB, not for a subtitle track. Your
lines will be spoken aloud by voice actors and by a text-to-speech model, so they
must sound like something a native Amharic speaker would actually say in that
moment - not like written, literary or overly formal Amharic.

Non-negotiable rules
--------------------
1. Never translate literally. Work out the intent, then write the line as natural
   spoken Amharic.
2. Preserve meaning AND subtext. What a character implies, hides, hints at or
   deliberately avoids saying matters as much as the words themselves.
3. Preserve the character: personality, education, register, age and social
   standing, and their relationship with the person they are talking to. Keep
   address forms, pronouns and honorifics consistent across the scene.
4. Preserve humor, sarcasm, irony, teasing, anger, fear, affection, tenderness,
   embarrassment, awkwardness, hesitation, hesitation in speech, and profanity.
   If the original line is rude, crude or blunt, keep it rude, crude or blunt.
   Do not sanitize, soften or moralize the dialogue.
5. Do not make dialogue more formal or more polite than the original. Street talk
   stays street talk, and casual speech stays casual.
6. Do not add information that is not present, and do not explain a joke.
7. Do not cut important meaning just to make a line shorter. Trim only harmless
   filler, and only when timing requires it.
8. When a literal rendering would sound unnatural in Amharic, restructure the
   whole sentence. Rewriting freely is encouraged as long as the meaning, tone and
   intent survive.

Borrowed and everyday English vocabulary
----------------------------------------
Amharic speakers constantly use English-derived words in daily speech, and forcing
a dictionary Amharic equivalent often makes dialogue sound archaic, academic or
simply unlike anything a real person would say.

When an English word, expression, technical term, modern concept, or brand name is
normally used in its English-derived form in everyday spoken Amharic, keep that
word - do not translate it. This commonly applies to:

- everyday loanwords that have no natural Amharic equivalent in speech;
- technical, medical, legal, military, computing, business and sports terms;
- modern concepts, technology and slang that speakers express with the English word;
- brand names and product names, which are never translated;
- people's names, place names and organisations.

**Write every borrowed word in Fidel script.** This is not optional. Your output is
spoken by an Amharic voice that reads Fidel, so a word left in Roman letters is a
word that cannot be pronounced. Write it the way Amharic speakers actually write it
- spelled as it sounds to an Amharic reader, with the vowels and consonants the
Fidel characters stand for. Keep the English *word*; change only its script.

Examples of the form to produce (only the script changes, never the word):

- Walt -> ዋልት
- computer -> ኮምፒውተር
- telephone -> ቴሌፎን
- doctor -> ዶክተር
- okay -> ኦኬይ
- airport -> ኤርፖርት

Some words have more than one accepted spelling. Choose the one that best matches
how the word sounds when an Amharic speaker says it, and - because the same word
must be pronounced the same way every time - use that same spelling every time it
appears in the film.

If a word has no established Fidel spelling in Amharic speech, still write it in
Fidel the way it sounds to an Amharic reader. Never leave it in Latin.

Use linguistic and conversational judgement, not a mechanical rule. Do not drop an
English word into every line, and do not replace a perfectly natural Amharic word
with an English one just to sound modern. When the natural Amharic word genuinely is
what people say, use it.

The target is authentic spoken Amharic. Linguistic purity is NOT a goal; sounding
like a real person on screen IS the goal.

Timing
------
Each line comes with the approximate duration of the original performance, and -
when it is supplied - a `syllable_budget`: the number of Amharic syllables that
fit that window at a natural speaking pace. Amharic words are usually longer than
their English equivalents, so a faithful line often needs more time than the
original took; that is expected, and the budget is what tells you how much room
you actually have.

Treat the budget as a hard limit, not a suggestion. A line that goes over it will
either be rushed or stretched, and both sound wrong. If a faithful rendering does
not fit, say the same thing in fewer words - drop filler, use a shorter synonym or
restructure the sentence - rather than allowing the line to run long. Never pad a
line to fill its window, and never cut meaning to reach the budget.

Punctuation should support spoken delivery - use commas, dashes, ellipses and
question marks the way a performer would breathe and pause.

Performance metadata
--------------------
For every line, also decide how it must be performed:

- emotion: the dominant emotion, for example neutral, happy, sad, angry,
  frustrated, afraid, surprised, embarrassed, sarcastic, playful, romantic,
  vulnerable, excited or disappointed. These are examples, not a fixed list.
- intensity: how strong the emotion is, from 0.0 (almost flat) to 1.0 (a full
  outburst).
- delivery: a short direction for the performer or TTS model, for example "calm
  and conversational", "quietly frustrated", "nervous and hesitant", "sarcastic
  with restrained amusement", "soft and vulnerable", "angry but controlled".
- pause_before and pause_after: estimated silence in seconds before and after the
  line, based on the rhythm of the scene. Keep them realistic - normally between
  0.0 and about 1.5 - and never invent long gaps.

Output format
-------------
Reply with a single JSON object and nothing else: no markdown, no code fences, no
commentary, no explanation. It must have exactly this shape:

{"lines": [{"id": "<the id you were given>", "amharic": "...", "emotion": "...",
            "intensity": 0.0, "delivery": "...", "pause_before": 0.0,
            "pause_after": 0.0}]}

Return exactly one object per input line, in the same order, and copy each "id"
back unchanged. Never invent, rename, merge, split, reorder or omit ids, and never
add or drop lines. Do not return speaker names, start times or end times: the
application owns those.
"""


class TranslationError(RuntimeError):
    """Base class for every error raised by this module."""


class MissingApiKeyError(TranslationError):
    """No DeepSeek API key is configured."""


class ConfigurationError(TranslationError, ValueError):
    """A translation setting is unusable."""


class ApiClientError(TranslationError):
    """The DeepSeek client could not be constructed."""


class ApiRequestError(TranslationError):
    """The DeepSeek request failed (network or API error)."""


class ApiAuthenticationError(ApiRequestError):
    """DeepSeek rejected the configured credentials."""


class MalformedResponseError(TranslationError):
    """The model response was not the JSON object we asked for."""


class InvalidModelOutputError(TranslationError):
    """The model returned well-formed JSON with unusable content."""


class DialogueIdError(InvalidModelOutputError, ValueError):
    """Dialogue ids were missing, duplicated or unknown."""


class InvalidTranscriptSegmentsError(TranslationError, ValueError):
    """The supplied input is not an iterable of TranscriptSegment."""


class InvalidSegmentError(TranslationError, ValueError):
    """An adapted dialogue line has an empty field or an out-of-range value."""


def _finite_number(name: str, value: object) -> float:
    """Return ``value`` as a finite ``float``."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidSegmentError(
            f"{name} must be a number, got {type(value).__name__}"
        )

    number = float(value)
    if not math.isfinite(number):
        raise InvalidSegmentError(f"{name} must be finite, got {value!r}")
    return number


def _required_text(name: str, value: object) -> str:
    """Return ``value`` stripped, or fail when it is not usable text."""

    if not isinstance(value, str) or not value.strip():
        raise InvalidSegmentError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True, slots=True)
class AdaptedDialogue:
    """One dubbing-ready Amharic line plus its performance direction.

    Instances are immutable and always valid, and the timing and identity fields
    are always the application-owned values from the original
    :class:`~app.pipeline.transcription.TranscriptSegment`. Text fields are stored
    stripped, so stray whitespace from the model never reaches the TTS stage.
    """

    speaker_id: str
    start: float
    end: float
    source_text: str
    amharic: str
    emotion: str
    intensity: float
    delivery: str
    pause_before: float
    pause_after: float

    def __post_init__(self) -> None:
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise InvalidSegmentError("speaker_id must be a non-empty string")

        start = _finite_number("start", self.start)
        end = _finite_number("end", self.end)
        if start < 0:
            raise InvalidSegmentError(f"start must be >= 0 seconds, got {start}")
        if end <= start:
            raise InvalidSegmentError(
                f"end must be greater than start ({start} seconds), got {end}"
            )

        source_text = _required_text("source_text", self.source_text)
        amharic = _required_text("amharic", self.amharic)
        emotion = _required_text("emotion", self.emotion)
        delivery = _required_text("delivery", self.delivery)

        intensity = _finite_number("intensity", self.intensity)
        if not 0.0 <= intensity <= 1.0:
            raise InvalidSegmentError(
                f"intensity must be between 0.0 and 1.0, got {intensity}"
            )

        pause_before = _finite_number("pause_before", self.pause_before)
        pause_after = _finite_number("pause_after", self.pause_after)
        if pause_before < 0:
            raise InvalidSegmentError(f"pause_before must be >= 0 seconds, got {pause_before}")
        if pause_after < 0:
            raise InvalidSegmentError(f"pause_after must be >= 0 seconds, got {pause_after}")

        # Normalise so the declared types are guaranteed.
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "source_text", source_text)
        object.__setattr__(self, "amharic", amharic)
        object.__setattr__(self, "emotion", emotion)
        object.__setattr__(self, "delivery", delivery)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "pause_before", pause_before)
        object.__setattr__(self, "pause_after", pause_after)

    @property
    def duration(self) -> float:
        """Length of the original line in seconds."""

        return self.end - self.start


def _dialogue_id(position: int) -> str:
    """Return the stable id of the ``position``-th line (1-based)."""

    return f"{DIALOGUE_ID_PREFIX}{position:0{DIALOGUE_ID_DIGITS}d}"


def _validate_segments(segments: Iterable[TranscriptSegment]) -> list[TranscriptSegment]:
    """Return the transcript as a chronological, validated list."""

    if isinstance(segments, (str, bytes)) or not isinstance(segments, Iterable):
        raise InvalidTranscriptSegmentsError(
            "segments must be an iterable of TranscriptSegment objects, got "
            f"{type(segments).__name__}"
        )

    items = list(segments)
    for index, item in enumerate(items):
        if not isinstance(item, TranscriptSegment):
            raise InvalidTranscriptSegmentsError(
                f"segments[{index}] is {type(item).__name__}, expected a TranscriptSegment"
            )

    return sorted(items, key=lambda item: (item.start, item.end, item.speaker_id))


def _require_api_key(settings: Settings) -> str:
    """Return the configured DeepSeek API key, or explain what is missing."""

    api_key = (settings.deepseek_api_key or "").strip()
    if not api_key:
        raise MissingApiKeyError(
            "DEEPSEEK_API_KEY is not set; dialogue adaptation needs a DeepSeek API "
            "key (see .env.example)"
        )
    return api_key


def _resolve_batch_size(settings: Settings) -> int:
    """Return the validated number of lines sent per request."""

    size = settings.translation_batch_size
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ConfigurationError(
            f"translation_batch_size must be a positive integer, got {size!r}"
        )
    return size


def _build_client(settings: Settings, api_key: str) -> Any:
    """Create the DeepSeek (OpenAI-compatible) client."""

    if OpenAI is None:  # pragma: no cover - only without the API client installed
        raise ApiClientError(
            "the openai package is not installed; install the runtime dependencies "
            "before running dialogue adaptation"
        )

    try:
        # The key is passed straight to the client and never logged or embedded in
        # any message we raise.
        return OpenAI(api_key=api_key, base_url=settings.translation_base_url)
    except Exception as exc:
        raise ApiClientError(
            f"could not create the DeepSeek client for "
            f"{settings.translation_base_url!r}: {type(exc).__name__}: {exc}"
        ) from exc


def _batches(
    segments: list[TranscriptSegment], size: int
) -> Iterator[list[TranscriptSegment]]:
    """Yield consecutive batches of at most ``size`` lines."""

    for start in range(0, len(segments), size):
        yield segments[start : start + size]


def _scene_of(scenes: Sequence[Scene], position: int) -> Scene | None:
    """Return the scene holding the line at ``position`` (0-based), if any."""

    for scene in scenes:
        if position in scene.line_indexes:
            return scene
    return None


def _context_block(
    *,
    scene: Scene | None,
    scene_lines: int,
    bible: CharacterBible | None,
    speakers: Sequence[str],
) -> str | None:
    """Build the scene-and-characters preamble for one batch, or ``None``.

    The block is deliberately built from state the application owns - the scene
    boundaries it derived from the timings, and the character bible it was given -
    rather than from anything the model inferred, so a request cannot invent a
    location or a relationship.
    """

    parts: list[str] = []

    if scene is not None:
        parts.append(
            f"SCENE: scene {scene.index + 1} of the film, covering "
            f"{scene.start:.1f}s to {scene.end:.1f}s, with {scene_lines} line(s) of "
            f"dialogue in total. {len(scene.speaker_ids)} character(s) are in it."
        )
        if len(scene.speaker_ids) > 1:
            parts.append(
                " This batch is part of that scene, so the lines belong to the same "
                "situation and the same conversation."
            )
        else:
            parts.append(
                " Only one character speaks in this scene, so it is not a "
                "conversation."
            )

    if bible is not None:
        block = bible.as_prompt_block(speakers)
        if block:
            parts.append("\nCHARACTERS IN THIS SCENE:\n" + block)

    if not parts:
        return None
    return "\n".join(parts)


def _request_kwargs(
    settings: Settings,
    batch: list[TranscriptSegment],
    ids: list[str],
    *,
    context: str | None = None,
    budgets: Sequence[SyllableBudget] | None = None,
    rewrite: str | None = None,
) -> dict[str, Any]:
    """Build the ``chat.completions.create`` arguments for one batch.

    Each line carries the length of the window it has to fill and, when budgets are
    supplied, the number of syllables it may use: a faithful translation of an
    English line routinely needs more time than the original took, so the line is
    asked to fit *before* it is synthesized rather than stretched afterwards.
    """

    payload: dict[str, Any] = {
        "lines": [
            {
                "id": line_id,
                "speaker": segment.speaker_id,  # context only; never returned
                "duration": round(segment.duration, 3),
                "text": segment.text,
                **(
                    {"syllable_budget": budget.syllables}
                    if budgets is not None
                    else {}
                ),
            }
            for segment, line_id, budget in zip(
                batch, ids, budgets if budgets is not None else (None,) * len(batch)
            )
        ]
    }
    if context:
        payload["context"] = context
    if rewrite:
        payload["rewrite"] = rewrite

    kwargs: dict[str, Any] = {
        "model": settings.translation_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "response_format": {"type": "json_object"},
    }

    if settings.translation_disable_thinking:
        # DeepSeek's hybrid models expose a thinking toggle through the OpenAI
        # client's escape hatch. Set TRANSLATION_DISABLE_THINKING=false if the
        # live API rejects this body.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    return kwargs


def _as_request_error(exc: Exception, settings: Settings) -> TranslationError:
    """Translate a client exception into a translation-specific error."""

    status = getattr(exc, "status_code", None)
    rejected = status in (401, 403) or (
        OpenAIAuthenticationError is not None
        and isinstance(exc, OpenAIAuthenticationError)
    )
    if rejected:
        return ApiAuthenticationError(
            "DeepSeek rejected the configured credentials for model "
            f"{settings.translation_model!r}"
            + (f" (HTTP {status})" if status is not None else "")
            + ": check DEEPSEEK_API_KEY and that it is allowed to use this model"
        )

    return ApiRequestError(
        f"the DeepSeek request for model {settings.translation_model!r} failed: "
        f"{type(exc).__name__}: {exc}"
    )


def _response_content(response: Any) -> str:
    """Return the text content of a DeepSeek chat completion."""

    choices = getattr(response, "choices", None)
    if not choices:
        raise MalformedResponseError(
            "the DeepSeek response contains no choices "
            f"(got {type(response).__name__})"
        )

    message = getattr(choices[0], "message", None)
    if message is None:
        raise MalformedResponseError("the DeepSeek response choice has no message")

    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise MalformedResponseError(
            "the DeepSeek response contains no text content (is the model running "
            "in reasoning mode?)"
        )

    return content


def _parse_response(content: str, ids: list[str]) -> dict[str, Any]:
    """Parse and validate the model's JSON, returning ``{dialogue_id: line}``."""

    try:
        payload = json.loads(content)
    except json.JSONDecodeError as exc:
        raise MalformedResponseError(
            f"the model did not return valid JSON: {exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise MalformedResponseError(
            f"the model returned a JSON {type(payload).__name__}, expected an object"
        )

    lines = payload.get("lines")
    if not isinstance(lines, list):
        raise MalformedResponseError(
            "the model response has no 'lines' array of adapted dialogue"
        )

    expected = set(ids)
    found: dict[str, Any] = {}
    for index, line in enumerate(lines):
        if not isinstance(line, dict):
            raise InvalidModelOutputError(
                f"line {index} of the model response is a JSON "
                f"{type(line).__name__}, expected an object"
            )

        raw_id = line.get("id")
        if not isinstance(raw_id, str) or not raw_id.strip():
            raise DialogueIdError(
                f"line {index} of the model response has no usable 'id'"
            )

        line_id = raw_id.strip()
        if line_id not in expected:
            raise DialogueIdError(
                f"the model returned an unknown dialogue id {line_id!r}"
            )
        if line_id in found:
            raise DialogueIdError(
                f"the model returned dialogue id {line_id!r} more than once"
            )
        found[line_id] = line

    missing = [line_id for line_id in ids if line_id not in found]
    if missing:
        raise DialogueIdError(
            f"the model returned {len(found)} of {len(ids)} dialogue lines; missing "
            + ", ".join(missing[:5])
            + (" ..." if len(missing) > 5 else "")
        )

    return found


def _build(segment: TranscriptSegment, line_id: str, line: Any) -> AdaptedDialogue:
    """Combine the model's adaptation with the application-owned metadata.

    Identity and timing come from ``segment``, never from the model, so a model
    that echoes its own speaker or timestamps cannot influence the result. Any
    unacceptable model field is reported as invalid model output rather than as a
    plain validation error.
    """

    try:
        return AdaptedDialogue(
            speaker_id=segment.speaker_id,
            start=segment.start,
            end=segment.end,
            source_text=segment.text,
            amharic=line.get("amharic"),
            emotion=line.get("emotion"),
            intensity=line.get("intensity"),
            delivery=line.get("delivery"),
            pause_before=line.get("pause_before"),
            pause_after=line.get("pause_after"),
        )
    except InvalidSegmentError as exc:
        raise InvalidModelOutputError(
            f"the model returned an unusable adaptation for {line_id!r}: {exc}"
        ) from exc


def _adapt_batch(
    client: Any,
    settings: Settings,
    batch: list[TranscriptSegment],
    ids: list[str],
    *,
    context: str | None = None,
    budgets: Sequence[SyllableBudget] | None = None,
    rewrite: str | None = None,
) -> list[AdaptedDialogue]:
    """Adapt one batch of consecutive lines, in order.

    The ids are supplied by the caller rather than derived from a start position,
    because a retry sends a *subset* of a batch and each line must keep the id it
    was given the first time.
    """

    kwargs = _request_kwargs(
        settings, batch, ids, context=context, budgets=budgets, rewrite=rewrite
    )

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise _as_request_error(exc, settings) from exc

    lines = _parse_response(_response_content(response), ids)
    return [_build(segment, line_id, lines[line_id]) for segment, line_id in zip(batch, ids)]


def _reduce_overshooting_lines(
    client: Any,
    settings: Settings,
    batch: list[TranscriptSegment],
    first_position: int,
    adapted: list[AdaptedDialogue],
    budgets: Sequence[SyllableBudget],
    *,
    context: str | None,
    tolerance: float,
    enforce_fidel_loanwords: bool = True,
) -> tuple[list[AdaptedDialogue], tuple[BudgetVerdict, ...]]:
    """Ask once more for the lines that came back wrong, and say how.

    Two independent problems are handled in a single pass, so a batch is never
    re-asked twice:

    * **Too long for its window.** English and Amharic do not express the same idea in
      the same number of syllables, so a faithful adaptation routinely overshoots. The
      literature's answer - and the one that preserves performance - is to make the
      *text* shorter rather than the audio faster.
    * **Roman text left in the line.** A borrowed word is meant to stay, but it must be
      written in Fidel, because the voice that speaks it reads Fidel. A rule in the
      prompt is not a guarantee, so this is verified rather than trusted.

    Only the offending lines are resent, and only once: what is still wrong afterwards
    is reported in the verdicts rather than retried indefinitely.
    """

    verdicts = tuple(
        budget.verdict(line.amharic, tolerance=tolerance)
        for budget, line in zip(budgets, adapted)
    )
    roman = tuple(
        latin_spans(line.amharic) if enforce_fidel_loanwords else ()
        for line in adapted
    )

    over_budget = [
        index for index, verdict in enumerate(verdicts) if not verdict.within_tolerance
    ]
    with_roman = [index for index, spans in enumerate(roman) if spans]
    offenders = sorted(set(over_budget) | set(with_roman))
    if not offenders:
        return adapted, verdicts

    ids = [_dialogue_id(first_position + index) for index in offenders]
    instructions: list[str] = []
    for index in offenders:
        if index in over_budget:
            instructions.append(
                f"- {_dialogue_id(first_position + index)}: "
                f"{verdicts[index].describe()}"
            )
        if index in with_roman:
            instructions.append(
                f"- {_dialogue_id(first_position + index)}: it still contains the "
                f"Roman-script word(s) {', '.join(roman[index])}; write each of them "
                "in Fidel the way an Amharic speaker writes it, keeping the word "
                "itself unchanged."
            )

    retried = _adapt_batch(
        client,
        settings,
        [batch[index] for index in offenders],
        ids,
        context=context,
        budgets=[budgets[index] for index in offenders],
        rewrite=(
            "Rewrite each of these lines to fix the problem described against it, "
            "keeping the meaning, tone and character:\n" + "\n".join(instructions)
        ),
    )
    for slot, line in zip(offenders, retried):
        adapted[slot] = line

    final = tuple(
        budget.verdict(line.amharic, tolerance=tolerance)
        for budget, line in zip(budgets, adapted)
    )
    return adapted, final


def adapt_dialogue(
    segments: Iterable[TranscriptSegment],
    *,
    settings: Settings | None = None,
    scenes: Sequence[Scene] | None = None,
    bible: CharacterBible | None = None,
    syllables_per_second: float = DEFAULT_SYLLABLES_PER_SECOND,
    budget_tolerance: float = DEFAULT_BUDGET_TOLERANCE,
    enforce_budget: bool = True,
    enforce_fidel_loanwords: bool = True,
) -> list[AdaptedDialogue]:
    """Adapt transcribed dialogue into dubbing-ready Amharic.

    Parameters
    ----------
    segments:
        The lines returned by :func:`app.pipeline.transcription.transcribe`. They
        are processed in chronological order and each one keeps its speaker id
        and timestamps verbatim.
    settings:
        Project settings override; defaults to :func:`app.config.get_settings`.
        The API key, base URL, model, batch size and thinking toggle all come from
        here - nothing is hard-coded in this module.
    scenes:
        Scene boundaries from
        :func:`app.pipeline.dialogue_context.segment_scenes`. Derived from the
        transcript's own timings when not supplied, so the model is always told
        which lines share a situation. Passing ``()`` disables scene context.
    bible:
        The persistent :class:`~app.pipeline.dialogue_context.CharacterBible`, so
        names, address forms and register survive the whole film instead of being
        re-derived from a ten-line window.
    syllables_per_second, budget_tolerance:
        The rate a performer delivers Amharic at and how far over the resulting
        budget a line may be before it is sent back to be shortened. Both come
        from :mod:`app.pipeline.dialogue_context`.
    enforce_budget:
        When ``True`` (the default), a line that comes back too long for its window
        is re-asked once. Set ``False`` to send every line exactly once, which is
        cheaper and is the right choice for a comparison run.
    enforce_fidel_loanwords:
        When ``True`` (the default), a line that comes back with Roman-script text
        in it is re-asked once to have those words written in Fidel. Borrowed
        English words are *meant* to stay - an Amharic speaker says "ኮምፒውተር", not a
        dictionary equivalent - but the voice that speaks the result reads Fidel, so
        a word left in Roman letters cannot be pronounced. Sharing the same pass as
        the budget check, so a batch is never re-asked twice.

    Returns
    -------
    list[AdaptedDialogue]
        One entry per input line, in chronological order. An empty list means
        there was nothing to adapt (no API call is made).

    Raises
    ------
    MissingApiKeyError
        ``DEEPSEEK_API_KEY`` is not configured.
    InvalidTranscriptSegmentsError
        ``segments`` is not an iterable of ``TranscriptSegment``.
    ConfigurationError
        A translation setting (currently the batch size) is unusable.
    ApiClientError
        The DeepSeek client could not be constructed.
    ApiAuthenticationError, ApiRequestError
        DeepSeek rejected the credentials, or the request failed.
    MalformedResponseError
        The model did not return the requested JSON object.
    InvalidModelOutputError, DialogueIdError
        The JSON was well-formed but a line was invalid, unknown, duplicated or
        missing.
    """

    settings = settings if settings is not None else get_settings()

    transcript = _validate_segments(segments)
    if not transcript:
        # Nothing to adapt, so there is no reason to call the API at all.
        return []

    api_key = _require_api_key(settings)
    batch_size = _resolve_batch_size(settings)
    client = _build_client(settings, api_key)

    resolved_scenes = segment_scenes(transcript) if scenes is None else tuple(scenes)

    adapted: list[AdaptedDialogue] = []
    position = 1
    for batch in _batches(transcript, batch_size):
        first_position = position
        ids = [_dialogue_id(first_position + offset) for offset in range(len(batch))]
        budgets = [syllable_budget(line.duration, rate=syllables_per_second) for line in batch]

        scene = _scene_of(resolved_scenes, first_position - 1)
        context = _context_block(
            scene=scene,
            scene_lines=scene.line_count if scene is not None else len(batch),
            bible=bible,
            speakers=[line.speaker_id for line in batch],
        )

        produced = _adapt_batch(
            client, settings, batch, ids, context=context, budgets=budgets
        )
        if enforce_budget or enforce_fidel_loanwords:
            produced, _ = _reduce_overshooting_lines(
                client,
                settings,
                batch,
                first_position,
                produced,
                budgets,
                context=context,
                tolerance=budget_tolerance,
                enforce_fidel_loanwords=enforce_fidel_loanwords,
            )

        adapted.extend(produced)
        position += len(batch)

    return adapted


__all__ = [
    "DIALOGUE_ID_PREFIX",
    "SYSTEM_PROMPT",
    "AdaptedDialogue",
    "ApiAuthenticationError",
    "ApiClientError",
    "ApiRequestError",
    "ConfigurationError",
    "DialogueIdError",
    "InvalidModelOutputError",
    "InvalidSegmentError",
    "InvalidTranscriptSegmentsError",
    "MalformedResponseError",
    "MissingApiKeyError",
    "TranslationError",
    "adapt_dialogue",
]
