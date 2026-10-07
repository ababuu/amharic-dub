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
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from app.config import Settings, get_settings
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
normally used in its English-derived or Anglicized form in everyday spoken
Amharic, prefer the form speakers actually use. Keep the borrowed word when that
is what a native speaker would naturally say in that situation. This commonly
applies to:

- everyday loanwords that have no natural Amharic equivalent in speech;
- technical, medical, legal, military, computing, business and sports terms;
- modern concepts, technology and slang that speakers express with the English word;
- brand names and product names, which are never translated.

Use linguistic and conversational judgement, not a mechanical rule. Do not drop an
English word into every line, and do not replace a perfectly natural Amharic word
with English just to sound modern. When the natural Amharic word genuinely is what
people say, use it.

The target is authentic spoken Amharic. Linguistic purity is NOT a goal; sounding
like a real person on screen IS the goal.

Timing
------
Each line comes with the approximate duration of the original performance. Choose
wording that can be delivered comfortably inside that window, at the character's
pace, without rushing and without padding. Punctuation should support spoken
delivery - use commas, dashes, ellipses and question marks the way a performer
would breathe and pause.

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


def _request_kwargs(
    settings: Settings,
    batch: list[TranscriptSegment],
    ids: list[str],
) -> dict[str, Any]:
    """Build the ``chat.completions.create`` arguments for one batch."""

    payload = {
        "lines": [
            {
                "id": line_id,
                "speaker": segment.speaker_id,  # context only; never returned
                "duration": round(segment.duration, 3),
                "text": segment.text,
            }
            for segment, line_id in zip(batch, ids)
        ]
    }

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
    first_position: int,
) -> list[AdaptedDialogue]:
    """Adapt one batch of consecutive lines, in order."""

    ids = [_dialogue_id(first_position + offset) for offset in range(len(batch))]
    kwargs = _request_kwargs(settings, batch, ids)

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        raise _as_request_error(exc, settings) from exc

    lines = _parse_response(_response_content(response), ids)
    return [_build(segment, line_id, lines[line_id]) for segment, line_id in zip(batch, ids)]


def adapt_dialogue(
    segments: Iterable[TranscriptSegment],
    *,
    settings: Settings | None = None,
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

    adapted: list[AdaptedDialogue] = []
    position = 1
    for batch in _batches(transcript, batch_size):
        adapted.extend(_adapt_batch(client, settings, batch, position))
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
