"""Tests for :mod:`app.pipeline.translation`.

The DeepSeek/OpenAI client is replaced by an in-memory fake, so the tests never
make a network call, never need an API key, and never touch a model. The fake
mirrors the real client only where this module depends on it: ``OpenAI(...)``,
``client.chat.completions.create(**kwargs)`` and
``response.choices[0].message.content``.
"""

from __future__ import annotations

import dataclasses
import json
import types
from pathlib import Path

import pytest

from app.config import (
    DEFAULT_TRANSLATION_BASE_URL,
    DEFAULT_TRANSLATION_BATCH_SIZE,
    DEFAULT_TRANSLATION_MODEL,
    Settings,
)
from app.pipeline import translation
from app.pipeline.transcription import TranscriptSegment
from app.pipeline.translation import (
    AdaptedDialogue,
    ApiAuthenticationError,
    ApiClientError,
    ApiRequestError,
    ConfigurationError,
    DialogueIdError,
    InvalidModelOutputError,
    InvalidSegmentError,
    InvalidTranscriptSegmentsError,
    MalformedResponseError,
    MissingApiKeyError,
    TranslationError,
    adapt_dialogue,
)


class FakeMessage:
    def __init__(self, content: object) -> None:
        self.content = content


class FakeResponse:
    """Stand-in for a chat completion response."""

    def __init__(self, content: object) -> None:
        self.choices = [types.SimpleNamespace(message=FakeMessage(content))]


class FakeCompletions:
    def __init__(self) -> None:
        pass

    def create(self, **kwargs):
        FakeOpenAI.requests.append(kwargs)
        if FakeOpenAI.error is not None and (
            FakeOpenAI.fail_on is None or FakeOpenAI.fail_on == len(FakeOpenAI.requests)
        ):
            raise FakeOpenAI.error
        return FakeOpenAI._build_response(kwargs)


class FakeOpenAI:
    """Stand-in for ``openai.OpenAI`` recording every interaction."""

    # Recorded state, reset by :func:`_patch`.
    init_kwargs: list[dict[str, object]] = []
    requests: list[dict[str, object]] = []

    # Scripted behaviour, set by individual tests.
    error: Exception | None = None
    fail_on: int | None = None
    raw: str | None = None
    transform: object = None
    response: object = None

    def __init__(self, api_key=None, base_url=None) -> None:
        FakeOpenAI.init_kwargs.append({"api_key": api_key, "base_url": base_url})
        self.chat = types.SimpleNamespace(completions=FakeCompletions())

    @staticmethod
    def _default_lines(request: dict) -> list[dict[str, object]]:
        payload = json.loads(request["messages"][1]["content"])
        return [
            {
                "id": line["id"],
                "amharic": f"የአማርኛ መስመር {line['id']}",
                "emotion": "neutral",
                "intensity": 0.5,
                "delivery": "calm and conversational",
                "pause_before": 0.1,
                "pause_after": 0.2,
            }
            for line in payload["lines"]
        ]

    @staticmethod
    def _build_response(request: dict):
        if FakeOpenAI.response is not None:
            return FakeOpenAI.response
        if FakeOpenAI.raw is not None:
            return FakeResponse(FakeOpenAI.raw)

        body: object = {"lines": FakeOpenAI._default_lines(request)}
        if FakeOpenAI.transform is not None:
            body = FakeOpenAI.transform(body)
        return FakeResponse(json.dumps(body, ensure_ascii=False))


def _patch(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install the fake client and reset all recorded state."""

    FakeOpenAI.init_kwargs = []
    FakeOpenAI.requests = []
    FakeOpenAI.error = None
    FakeOpenAI.fail_on = None
    FakeOpenAI.raw = None
    FakeOpenAI.transform = None
    FakeOpenAI.response = None

    monkeypatch.setattr(translation, "OpenAI", FakeOpenAI)


def _settings(**overrides) -> Settings:
    """Build settings for a component that never touches the filesystem."""

    values: dict[str, object] = {
        "input_dir": Path("input"),
        "work_dir": Path("work"),
        "output_dir": Path("output"),
        "model_cache_dir": Path("models"),
        "deepseek_api_key": "test-key",
    }
    values.update(overrides)
    return Settings(**values)


def _segment(
    speaker_id: str = "SPEAKER_00",
    start: float = 12.43,
    end: float = 16.82,
    text: str = "I'm here",
) -> TranscriptSegment:
    return TranscriptSegment(speaker_id=speaker_id, start=start, end=end, text=text)


def _dialogue(**overrides) -> AdaptedDialogue:
    values: dict[str, object] = {
        "speaker_id": "SPEAKER_00",
        "start": 12.43,
        "end": 16.82,
        "source_text": "I'm here",
        "amharic": "እኔ እዚህ ነኝ",
        "emotion": "vulnerable",
        "intensity": 0.7,
        "delivery": "soft and sincere",
        "pause_before": 0.3,
        "pause_after": 0.8,
    }
    values.update(overrides)
    return AdaptedDialogue(**values)


def _sent_lines(request: dict) -> list[dict[str, object]]:
    return json.loads(request["messages"][1]["content"])["lines"]


# ---------------------------------------------------------------------------
# AdaptedDialogue validation
# ---------------------------------------------------------------------------


def test_adapted_dialogue_keeps_its_values():
    line = _dialogue()

    assert line.speaker_id == "SPEAKER_00"
    assert line.start == 12.43
    assert line.end == 16.82
    assert line.source_text == "I'm here"
    assert line.amharic == "እኔ እዚህ ነኝ"
    assert line.emotion == "vulnerable"
    assert line.intensity == 0.7
    assert line.delivery == "soft and sincere"
    assert line.pause_before == 0.3
    assert line.pause_after == 0.8


def test_adapted_dialogue_duration():
    assert _dialogue(start=10.0, end=13.5).duration == 3.5


def test_adapted_dialogue_normalises_numbers_to_float():
    line = _dialogue(start=1, end=3, intensity=1, pause_before=0, pause_after=0)

    assert isinstance(line.start, float)
    assert isinstance(line.end, float)
    assert isinstance(line.intensity, float)
    assert isinstance(line.pause_before, float)
    assert line.duration == 2.0


def test_adapted_dialogue_strips_its_text_fields():
    line = _dialogue(
        source_text="  hi  ",
        amharic="  ሰላም  ",
        emotion=" angry ",
        delivery=" calm ",
    )

    assert line.source_text == "hi"
    assert line.amharic == "ሰላም"
    assert line.emotion == "angry"
    assert line.delivery == "calm"


def test_adapted_dialogue_is_immutable():
    line = _dialogue()

    with pytest.raises(dataclasses.FrozenInstanceError):
        line.amharic = "changed"


def test_blank_speaker_id_is_rejected():
    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        _dialogue(speaker_id="   ")


def test_non_string_speaker_id_is_rejected():
    with pytest.raises(InvalidSegmentError, match="speaker_id"):
        _dialogue(speaker_id=None)


def test_zero_length_dialogue_is_rejected():
    with pytest.raises(InvalidSegmentError, match="greater than start"):
        _dialogue(start=4.0, end=4.0)


def test_negative_start_is_rejected():
    with pytest.raises(InvalidSegmentError, match="start must be"):
        _dialogue(start=-0.1)


def test_non_finite_timestamp_is_rejected():
    with pytest.raises(InvalidSegmentError, match="finite"):
        _dialogue(end=float("inf"))


def test_empty_source_text_is_rejected():
    with pytest.raises(InvalidSegmentError, match="source_text"):
        _dialogue(source_text="   ")


def test_empty_amharic_is_rejected():
    with pytest.raises(InvalidSegmentError, match="amharic"):
        _dialogue(amharic="")


def test_empty_emotion_is_rejected():
    with pytest.raises(InvalidSegmentError, match="emotion"):
        _dialogue(emotion="  ")


def test_empty_delivery_is_rejected():
    with pytest.raises(InvalidSegmentError, match="delivery"):
        _dialogue(delivery="")


def test_intensity_above_one_is_rejected():
    with pytest.raises(InvalidSegmentError, match="intensity"):
        _dialogue(intensity=1.01)


def test_negative_intensity_is_rejected():
    with pytest.raises(InvalidSegmentError, match="intensity"):
        _dialogue(intensity=-0.1)


def test_non_numeric_intensity_is_rejected():
    with pytest.raises(InvalidSegmentError, match="intensity"):
        _dialogue(intensity="loud")


def test_negative_pauses_are_rejected():
    with pytest.raises(InvalidSegmentError, match="pause_before"):
        _dialogue(pause_before=-0.1)

    with pytest.raises(InvalidSegmentError, match="pause_after"):
        _dialogue(pause_after=-1.0)


def test_non_numeric_pause_is_rejected():
    with pytest.raises(InvalidSegmentError, match="pause_before"):
        _dialogue(pause_before=None)


def test_translation_error_hierarchy():
    assert issubclass(InvalidSegmentError, ValueError)
    assert issubclass(InvalidSegmentError, TranslationError)
    assert issubclass(InvalidTranscriptSegmentsError, ValueError)
    assert issubclass(ConfigurationError, ValueError)
    assert issubclass(DialogueIdError, InvalidModelOutputError)
    assert issubclass(DialogueIdError, ValueError)
    assert issubclass(ApiAuthenticationError, ApiRequestError)
    assert issubclass(ApiAuthenticationError, TranslationError)


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_without_calling_the_api(monkeypatch):
    _patch(monkeypatch)

    assert adapt_dialogue([], settings=_settings()) == []
    assert FakeOpenAI.init_kwargs == []
    assert FakeOpenAI.requests == []


def test_non_transcript_segment_input_is_rejected(monkeypatch):
    _patch(monkeypatch)

    with pytest.raises(InvalidTranscriptSegmentsError, match="expected a TranscriptSegment"):
        adapt_dialogue([("SPEAKER_00", 0.0, 1.0)], settings=_settings())

    assert FakeOpenAI.requests == []


def test_non_iterable_input_is_rejected(monkeypatch):
    _patch(monkeypatch)

    with pytest.raises(InvalidTranscriptSegmentsError, match="iterable"):
        adapt_dialogue(42, settings=_settings())


# ---------------------------------------------------------------------------
# API key, client and model configuration
# ---------------------------------------------------------------------------


def test_missing_api_key_is_rejected(monkeypatch):
    _patch(monkeypatch)

    with pytest.raises(MissingApiKeyError, match="DEEPSEEK_API_KEY"):
        adapt_dialogue([_segment()], settings=_settings(deepseek_api_key=None))

    assert FakeOpenAI.requests == []


def test_blank_api_key_is_rejected(monkeypatch):
    _patch(monkeypatch)

    with pytest.raises(MissingApiKeyError, match="DEEPSEEK_API_KEY"):
        adapt_dialogue([_segment()], settings=_settings(deepseek_api_key="   "))


def test_client_uses_the_configured_base_url_and_key(monkeypatch):
    _patch(monkeypatch)

    adapt_dialogue(
        [_segment()],
        settings=_settings(
            deepseek_api_key="sk-test",
            translation_base_url=DEFAULT_TRANSLATION_BASE_URL,
        ),
    )

    assert FakeOpenAI.init_kwargs == [
        {"api_key": "sk-test", "base_url": "https://api.deepseek.com"}
    ]


def test_default_model_is_deepseek_flash(monkeypatch):
    _patch(monkeypatch)

    adapt_dialogue([_segment()], settings=_settings())

    assert DEFAULT_TRANSLATION_MODEL == "deepseek-flash"
    assert FakeOpenAI.requests[0]["model"] == "deepseek-flash"


def test_the_configured_model_is_used(monkeypatch):
    _patch(monkeypatch)

    adapt_dialogue([_segment()], settings=_settings(translation_model="deepseek-reasoner"))

    assert FakeOpenAI.requests[0]["model"] == "deepseek-reasoner"


def test_missing_openai_package_is_reported(monkeypatch):
    _patch(monkeypatch)
    monkeypatch.setattr(translation, "OpenAI", None)

    with pytest.raises(ApiClientError, match="not installed"):
        adapt_dialogue([_segment()], settings=_settings())


def test_batch_size_must_be_positive(monkeypatch):
    _patch(monkeypatch)

    with pytest.raises(ConfigurationError, match="positive integer"):
        adapt_dialogue([_segment()], settings=_settings(translation_batch_size=0))

    assert FakeOpenAI.requests == []


def test_batch_size_must_be_an_integer(monkeypatch):
    _patch(monkeypatch)

    with pytest.raises(ConfigurationError, match="positive integer"):
        adapt_dialogue([_segment()], settings=_settings(translation_batch_size="ten"))


# ---------------------------------------------------------------------------
# request construction
# ---------------------------------------------------------------------------


def test_request_contains_the_system_prompt(monkeypatch):
    _patch(monkeypatch)

    adapt_dialogue([_segment(text="Hello")], settings=_settings())

    messages = FakeOpenAI.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == translation.SYSTEM_PROMPT
    assert messages[1]["role"] == "user"


def test_user_message_carries_id_speaker_duration_and_text(monkeypatch):
    _patch(monkeypatch)

    adapt_dialogue(
        [_segment(speaker_id="SPEAKER_07", start=10.0, end=14.0, text="Line one")],
        settings=_settings(),
    )

    assert _sent_lines(FakeOpenAI.requests[0]) == [
        {
            "id": "dialogue_000001",
            "speaker": "SPEAKER_07",
            "duration": 4.0,
            "text": "Line one",
        }
    ]


def test_json_mode_is_requested(monkeypatch):
    _patch(monkeypatch)

    adapt_dialogue([_segment()], settings=_settings())

    assert FakeOpenAI.requests[0]["response_format"] == {"type": "json_object"}


def test_thinking_is_disabled_by_default_and_can_be_turned_off(monkeypatch):
    _patch(monkeypatch)
    adapt_dialogue([_segment()], settings=_settings())
    assert FakeOpenAI.requests[0]["extra_body"] == {"thinking": {"type": "disabled"}}

    _patch(monkeypatch)
    adapt_dialogue([_segment()], settings=_settings(translation_disable_thinking=False))
    assert "extra_body" not in FakeOpenAI.requests[0]


def test_system_prompt_demands_spoken_amharic_adaptation():
    prompt = translation.SYSTEM_PROMPT.lower()

    assert "amharic" in prompt
    assert "translate literally" in prompt
    assert "subtext" in prompt
    assert "humor" in prompt
    assert "sarcasm" in prompt
    assert "profanity" in prompt
    assert "duration" in prompt
    assert "emotion" in prompt
    assert "intensity" in prompt
    assert "delivery" in prompt
    assert "json" in prompt


def test_system_prompt_covers_borrowed_english_vocabulary():
    prompt = translation.SYSTEM_PROMPT.lower()

    assert "borrowed" in prompt
    assert "english-derived" in prompt
    assert "linguistic purity is not a goal" in prompt
    assert "brand name" in prompt
    # ...and that this must not be applied mechanically.
    assert "judgement, not a mechanical rule" in prompt


# ---------------------------------------------------------------------------
# batching and application-generated ids
# ---------------------------------------------------------------------------


def test_dialogue_is_sent_in_bounded_batches(monkeypatch):
    _patch(monkeypatch)
    segments = [
        _segment(start=float(index), end=float(index) + 1.0, text=f"line {index}")
        for index in range(1, 8)
    ]

    adapt_dialogue(segments, settings=_settings(translation_batch_size=3))

    assert [len(_sent_lines(request)) for request in FakeOpenAI.requests] == [3, 3, 1]


def test_a_whole_movie_is_never_sent_in_one_request(monkeypatch):
    _patch(monkeypatch)
    segments = [_segment(start=float(index), end=float(index) + 1.0) for index in range(100)]

    adapt_dialogue(segments, settings=_settings())

    assert DEFAULT_TRANSLATION_BATCH_SIZE == 10
    assert len(FakeOpenAI.requests) == 10
    assert all(len(_sent_lines(request)) == 10 for request in FakeOpenAI.requests)


def test_stable_ids_are_assigned_across_batches(monkeypatch):
    _patch(monkeypatch)
    segments = [_segment(start=float(index), end=float(index) + 1.0) for index in range(4)]

    adapt_dialogue(segments, settings=_settings(translation_batch_size=2))

    sent_ids = [line["id"] for request in FakeOpenAI.requests for line in _sent_lines(request)]
    assert sent_ids == [
        "dialogue_000001",
        "dialogue_000002",
        "dialogue_000003",
        "dialogue_000004",
    ]
    assert len(set(sent_ids)) == len(sent_ids)


def test_batch_context_includes_who_is_speaking(monkeypatch):
    _patch(monkeypatch)

    adapt_dialogue(
        [
            _segment(speaker_id="SPEAKER_00", start=0.0, end=1.0, text="Where were you?"),
            _segment(speaker_id="SPEAKER_01", start=1.0, end=2.0, text="At work."),
        ],
        settings=_settings(),
    )

    speakers = [line["speaker"] for line in _sent_lines(FakeOpenAI.requests[0])]
    assert speakers == ["SPEAKER_00", "SPEAKER_01"]


# ---------------------------------------------------------------------------
# the application owns identity and timing
# ---------------------------------------------------------------------------


def test_model_cannot_overwrite_speaker_ids(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        for line in body["lines"]:
            line["speaker_id"] = "EVIL_SPEAKER"
            line["speaker"] = "EVIL_SPEAKER"
        return body

    FakeOpenAI.transform = transform

    adapted = adapt_dialogue([_segment(speaker_id="SPEAKER_03")], settings=_settings())

    assert [line.speaker_id for line in adapted] == ["SPEAKER_03"]


def test_model_cannot_overwrite_timestamps(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        for line in body["lines"]:
            line["start"] = 999.0
            line["end"] = 1000.0
        return body

    FakeOpenAI.transform = transform

    adapted = adapt_dialogue([_segment(start=12.43, end=16.82)], settings=_settings())

    assert [(line.start, line.end) for line in adapted] == [(12.43, 16.82)]


def test_source_text_always_comes_from_the_transcript(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        for line in body["lines"]:
            line["source_text"] = "rewritten by the model"
            line["text"] = "rewritten by the model"
        return body

    FakeOpenAI.transform = transform

    adapted = adapt_dialogue([_segment(text="Original line")], settings=_settings())

    assert [line.source_text for line in adapted] == ["Original line"]


def test_output_preserves_chronological_order(monkeypatch):
    _patch(monkeypatch)
    segments = [
        _segment(speaker_id="SPEAKER_00", start=1.0, end=2.0, text="first"),
        _segment(speaker_id="SPEAKER_01", start=3.0, end=4.0, text="second"),
        _segment(speaker_id="SPEAKER_00", start=5.0, end=6.0, text="third"),
    ]

    adapted = adapt_dialogue(segments, settings=_settings())

    assert [line.source_text for line in adapted] == ["first", "second", "third"]
    assert [line.speaker_id for line in adapted] == [
        "SPEAKER_00",
        "SPEAKER_01",
        "SPEAKER_00",
    ]
    assert [line.start for line in adapted] == sorted(line.start for line in adapted)


def test_output_is_chronological_even_when_the_input_is_not(monkeypatch):
    _patch(monkeypatch)
    segments = [
        _segment(start=30.0, end=31.0, text="later"),
        _segment(start=1.0, end=2.0, text="earlier"),
    ]

    adapted = adapt_dialogue(segments, settings=_settings())

    assert [line.source_text for line in adapted] == ["earlier", "later"]


def test_one_adapted_line_is_returned_per_input_line(monkeypatch):
    _patch(monkeypatch)
    segments = [_segment(start=float(index), end=float(index) + 1.0) for index in range(7)]

    adapted = adapt_dialogue(segments, settings=_settings(translation_batch_size=3))

    assert len(adapted) == 7
    assert all(isinstance(line, AdaptedDialogue) for line in adapted)
    assert [line.source_text for line in adapted] == [segment.text for segment in segments]


def test_model_values_are_used_for_the_amharic_and_performance_fields(monkeypatch):
    _patch(monkeypatch)

    adapted = adapt_dialogue([_segment()], settings=_settings())

    assert "dialogue_000001" in adapted[0].amharic
    assert adapted[0].emotion == "neutral"
    assert adapted[0].intensity == 0.5
    assert adapted[0].delivery == "calm and conversational"
    assert adapted[0].pause_before == 0.1
    assert adapted[0].pause_after == 0.2


# ---------------------------------------------------------------------------
# response validation
# ---------------------------------------------------------------------------


def test_malformed_json_is_rejected(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.raw = "this is not json"

    with pytest.raises(MalformedResponseError, match="valid JSON"):
        adapt_dialogue([_segment()], settings=_settings())


def test_json_that_is_not_an_object_is_rejected(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.raw = json.dumps([{"id": "dialogue_000001"}])

    with pytest.raises(MalformedResponseError, match="expected an object"):
        adapt_dialogue([_segment()], settings=_settings())


def test_response_without_a_lines_array_is_rejected(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.raw = json.dumps({"adapted": []})

    with pytest.raises(MalformedResponseError, match="lines"):
        adapt_dialogue([_segment()], settings=_settings())


def test_line_that_is_not_an_object_is_rejected(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.raw = json.dumps({"lines": ["dialogue_000001"]})

    with pytest.raises(InvalidModelOutputError, match="expected an object"):
        adapt_dialogue([_segment()], settings=_settings())


def test_line_without_an_id_is_rejected(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.raw = json.dumps(
        {
            "lines": [
                {
                    "amharic": "ሰላም",
                    "emotion": "neutral",
                    "intensity": 0.5,
                    "delivery": "calm",
                    "pause_before": 0.0,
                    "pause_after": 0.0,
                }
            ]
        }
    )

    with pytest.raises(DialogueIdError, match="usable 'id'"):
        adapt_dialogue([_segment()], settings=_settings())


def test_missing_dialogue_line_is_rejected(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.transform = lambda body: {"lines": []}

    with pytest.raises(DialogueIdError, match="missing"):
        adapt_dialogue([_segment()], settings=_settings())


def test_duplicate_dialogue_id_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"] = body["lines"] + [dict(body["lines"][0])]
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(DialogueIdError, match="more than once"):
        adapt_dialogue([_segment()], settings=_settings())


def test_unknown_dialogue_id_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"][0]["id"] = "dialogue_999999"
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(DialogueIdError, match="unknown dialogue id"):
        adapt_dialogue([_segment()], settings=_settings())


def test_missing_required_field_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        del body["lines"][0]["amharic"]
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(InvalidModelOutputError, match="amharic"):
        adapt_dialogue([_segment()], settings=_settings())


def test_empty_amharic_from_the_model_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"][0]["amharic"] = "   "
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(InvalidModelOutputError, match="amharic"):
        adapt_dialogue([_segment()], settings=_settings())


def test_empty_delivery_from_the_model_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"][0]["delivery"] = ""
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(InvalidModelOutputError, match="delivery"):
        adapt_dialogue([_segment()], settings=_settings())


def test_invalid_intensity_from_the_model_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"][0]["intensity"] = 1.5
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(InvalidModelOutputError, match="intensity"):
        adapt_dialogue([_segment()], settings=_settings())


def test_non_numeric_intensity_from_the_model_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"][0]["intensity"] = "loud"
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(InvalidModelOutputError, match="intensity"):
        adapt_dialogue([_segment()], settings=_settings())


def test_negative_pause_from_the_model_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"][0]["pause_after"] = -2.0
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(InvalidModelOutputError, match="pause_after"):
        adapt_dialogue([_segment()], settings=_settings())


def test_non_numeric_pause_from_the_model_is_rejected(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"][0]["pause_before"] = "a while"
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(InvalidModelOutputError, match="pause_before"):
        adapt_dialogue([_segment()], settings=_settings())


def test_bad_model_output_names_the_offending_dialogue_id(monkeypatch):
    _patch(monkeypatch)

    def transform(body):
        body["lines"][0]["intensity"] = "loud"
        return body

    FakeOpenAI.transform = transform

    with pytest.raises(InvalidModelOutputError, match="dialogue_000001"):
        adapt_dialogue([_segment()], settings=_settings())


# ---------------------------------------------------------------------------
# API failures
# ---------------------------------------------------------------------------


def test_response_without_choices_is_rejected(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.response = types.SimpleNamespace(choices=[])

    with pytest.raises(MalformedResponseError, match="no choices"):
        adapt_dialogue([_segment()], settings=_settings())


def test_response_without_text_content_is_rejected(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.response = FakeResponse(None)

    with pytest.raises(MalformedResponseError, match="no text content"):
        adapt_dialogue([_segment()], settings=_settings())


def test_api_failure_is_converted_into_a_translation_error(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.error = RuntimeError("connection reset by peer")

    with pytest.raises(ApiRequestError, match="failed") as excinfo:
        adapt_dialogue([_segment()], settings=_settings())

    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_authentication_failure_is_reported(monkeypatch):
    _patch(monkeypatch)

    class FakeAuthError(Exception):
        status_code = 401

    FakeOpenAI.error = FakeAuthError("invalid api key")

    with pytest.raises(ApiAuthenticationError, match="rejected the configured credentials"):
        adapt_dialogue([_segment()], settings=_settings())


def test_forbidden_response_is_reported_as_authentication_failure(monkeypatch):
    _patch(monkeypatch)

    class FakeForbiddenError(Exception):
        status_code = 403

    FakeOpenAI.error = FakeForbiddenError("forbidden")

    with pytest.raises(ApiAuthenticationError, match="rejected the configured credentials"):
        adapt_dialogue([_segment()], settings=_settings())


def test_api_key_is_never_added_to_our_error_messages(monkeypatch):
    _patch(monkeypatch)

    class FakeAuthError(Exception):
        status_code = 401

    FakeOpenAI.error = FakeAuthError("invalid key sk-secret-value")

    with pytest.raises(ApiAuthenticationError) as excinfo:
        adapt_dialogue(
            [_segment()], settings=_settings(deepseek_api_key="sk-secret-value")
        )

    assert "sk-secret-value" not in str(excinfo.value)


def test_api_key_is_not_added_to_generic_request_errors(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.error = RuntimeError("boom")

    with pytest.raises(ApiRequestError) as excinfo:
        adapt_dialogue(
            [_segment()], settings=_settings(deepseek_api_key="sk-secret-value")
        )

    assert "sk-secret-value" not in str(excinfo.value)


def test_a_failing_batch_fails_the_whole_call(monkeypatch):
    _patch(monkeypatch)
    FakeOpenAI.error = RuntimeError("rate limited")
    FakeOpenAI.fail_on = 2  # the second of the four two-line batches
    segments = [_segment(start=float(index), end=float(index) + 1.0) for index in range(4)]

    with pytest.raises(ApiRequestError):
        adapt_dialogue(segments, settings=_settings(translation_batch_size=2))

    # The call stops at the failure instead of returning a partial adaptation.
    assert len(FakeOpenAI.requests) == 2
