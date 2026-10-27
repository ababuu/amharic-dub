"""Tests for :mod:`app.pipeline.context_ai`.

Every network call is replaced: ``DeepSeekTools`` receives an in-memory POST
function and the chat model is a fake with the two attributes this module uses.
The tests cover title discovery, the Search/Fetch contract, guide caching, and the
evidence gate that decides whether a speaker label may move.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.pipeline import context_ai
from app.pipeline.context_ai import (
    DeepSeekTools,
    FetchedPage,
    LineEvidence,
    ResearchUnavailable,
    SearchHit,
    TitleInfo,
    build_character_guide,
    collect_evidence,
    discover_title,
    review_speaker_attribution,
)
from app.pipeline.transcription import TranscriptSegment

SPEAKER_A = "SPEAKER_00"
SPEAKER_B = "SPEAKER_01"


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root / "input",
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "models",
        "dialogue_bible_path": root / "work" / "dialogue_bible.json",
        "character_guide_path": root / "work" / "character_guide.json",
        "episode_script_path": root / "work" / "episode_script.json",
        "deepseek_api_key": "test-key",
        "translation_backend": "openai",
        "translation_provider": "deepseek",
        "translation_model": "deepseek-v4-pro",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _title_file(root: Path, name: str, text: str) -> Path:
    directory = root / "input"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Title discovery
# ---------------------------------------------------------------------------


def test_title_is_read_from_a_discovered_txt_file(tmp_path: Path) -> None:
    _title_file(tmp_path, "anything-at-all.txt", "Breaking Bad\nSeason 4, Episode 6\n")

    title = discover_title(tmp_path / "input", source=tmp_path / "movie.mp4")

    assert title is not None
    assert title.title == "Breaking Bad"
    assert (title.season, title.episode) == (4, 6)
    assert title.label == "Breaking Bad (season 4, episode 6)"


def test_a_title_file_named_after_the_video_wins(tmp_path: Path) -> None:
    _title_file(tmp_path, "movie.txt", "The Matrix")
    _title_file(tmp_path, "notes.txt", "Some other film")

    title = discover_title(tmp_path / "input", source=tmp_path / "movie.mp4")

    assert title is not None
    assert title.title == "The Matrix"
    assert title.path is not None and title.path.name == "movie.txt"


def test_different_titles_are_reported_as_ambiguous(tmp_path: Path) -> None:
    _title_file(tmp_path, "one.txt", "Breaking Bad")
    _title_file(tmp_path, "two.txt", "Better Call Saul")

    title = discover_title(tmp_path / "input", source=tmp_path / "movie.mp4")

    assert title is not None
    assert not title
    assert len(title.ambiguous) == 2
    assert "different title" in title.reason


def test_a_missing_or_empty_title_is_not_used(tmp_path: Path) -> None:
    assert discover_title(tmp_path / "input") is None

    _title_file(tmp_path, "empty.txt", "   \n\n")

    title = discover_title(tmp_path / "input")
    assert title is not None
    assert not title
    assert "usable title" in title.reason


def test_a_transcript_file_is_not_mistaken_for_a_title(tmp_path: Path) -> None:
    _title_file(
        tmp_path,
        "script.txt",
        "Breaking Bad\n\nWalt: I am the danger.\nSkyler: Walt, please.\n"
        "Walt: A guy opens his door.\nSkyler: Stop.\n",
    )

    title = discover_title(tmp_path / "input")

    assert title is not None
    assert not title
    assert "usable title" in title.reason


def test_compact_season_episode_forms_are_parsed(tmp_path: Path) -> None:
    _title_file(tmp_path, "show.txt", "The Wire S03E07")

    title = discover_title(tmp_path / "input")
    assert title is not None and (title.season, title.episode) == (3, 7)


# ---------------------------------------------------------------------------
# Search and Fetch contract
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[tuple[str, dict, float]] = []

    def __call__(self, url: str, payload: dict, timeout: float) -> dict:
        self.calls.append((url, dict(payload), timeout))
        if isinstance(self.response, Exception):
            raise self.response
        return self.response  # type: ignore[return-value]


def test_search_parses_results_and_bounds_the_request() -> None:
    recorder = _Recorder(
        {
            "results": [
                {
                    "title": "Breaking Bad characters",
                    "url": "https://example.com/cast",
                    "summary": "A chemistry teacher.",
                    "rerank_score": 2.4,
                    "authority": 3,
                },
                {"title": "bad", "url": "not-a-url"},
            ]
        }
    )
    tools = DeepSeekTools("key", post=recorder, timeout=12.5)

    hits = tools.search("x" * 200, count=99)

    assert [hit.url for hit in hits] == ["https://example.com/cast"]
    assert hits[0].authority == 3
    assert hits[0].rerank_score == 2.4
    url, payload, timeout = recorder.calls[0]
    assert url.endswith("/v1/tools/search")
    assert len(payload["query"]) == 128
    assert payload["count"] == 50
    assert timeout == 12.5


def test_fetch_truncates_page_content() -> None:
    recorder = _Recorder({"title": "Cast", "content": "y" * 100})
    tools = DeepSeekTools("key", post=recorder, max_content_chars=10)

    page = tools.fetch("https://example.com/cast")

    assert page.title == "Cast"
    assert page.content == "y" * 10
    assert recorder.calls[0][0].endswith("/v1/tools/fetch")
    assert recorder.calls[0][1] == {"url": "https://example.com/cast"}


def test_a_failed_tool_request_becomes_research_unavailable() -> None:
    tools = DeepSeekTools("key", post=_Recorder(RuntimeError("offline")))

    with pytest.raises(ResearchUnavailable):
        tools.search("Breaking Bad characters")


# ---------------------------------------------------------------------------
# Character research and caching
# ---------------------------------------------------------------------------


class _FakeCompletions:
    def __init__(self, replies: list[object]) -> None:
        self.replies = list(replies)
        self.requests: list[dict] = []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeChat:
    def __init__(self, *replies: object) -> None:
        self.completions = _FakeCompletions(list(replies))
        self.chat = SimpleNamespace(completions=self.completions)


class _FakeTools:
    def __init__(self, hits: list[SearchHit], pages: list[FetchedPage]) -> None:
        self.hits = hits
        self.pages = pages
        self.searches: list[str] = []
        self.fetches: list[str] = []

    def search(self, query: str, *, count: int = 5) -> tuple[SearchHit, ...]:
        self.searches.append(query)
        return tuple(self.hits)

    def fetch(self, url: str) -> FetchedPage:
        self.fetches.append(url)
        for page in self.pages:
            if page.url == url:
                return page
        raise ResearchUnavailable("not found")


def _research_fixture(
    root: Path,
) -> tuple[TitleInfo, _FakeTools, _FakeChat]:
    path = _title_file(root, "breaking-bad.txt", "Breaking Bad S04E06")
    title = TitleInfo(title="Breaking Bad", path=path, season=4, episode=6)
    hits = [
        SearchHit("Walt", "https://example.com/walt", authority=3, rerank_score=2.0),
        SearchHit("Skyler", "https://example.com/skyler", authority=2, rerank_score=1.8),
    ]
    pages = [
        FetchedPage("https://example.com/walt", "Walt", "Walter White is a teacher."),
        FetchedPage("https://example.com/skyler", "Skyler", "Skyler is his wife."),
    ]
    reply = {
        "characters": [
            {
                "name": "Walter White",
                "aliases": ["Walt"],
                "gender": "masculine",
                "relationships": {"Skyler": "wife"},
                "notes": "Chemistry teacher.",
                "sources": ["S1"],
                "confidence": 0.92,
            }
        ]
    }
    return title, _FakeTools(hits, pages), _FakeChat(reply)


def test_research_builds_a_sourced_guide_and_caches_it(tmp_path: Path) -> None:
    title, tools, chat = _research_fixture(tmp_path)
    settings = _settings(tmp_path)

    outcome = build_character_guide(
        title, settings=settings, tools=tools, chat_client=chat
    )

    assert outcome.guide is not None
    character = outcome.guide.characters[0]
    assert character.name == "Walter White"
    assert character.gender == "masculine"
    assert character.sources == ("https://example.com/walt",)
    assert outcome.guide.sources == (
        "https://example.com/walt",
        "https://example.com/skyler",
    )
    assert settings.character_guide_path.is_file()
    assert "Breaking Bad" in chat.completions.requests[0]["messages"][1]["content"]
    assert len(tools.fetches) == 2


def test_a_cached_guide_is_reused_without_network_calls(tmp_path: Path) -> None:
    title, tools, chat = _research_fixture(tmp_path)
    settings = _settings(tmp_path)
    build_character_guide(title, settings=settings, tools=tools, chat_client=chat)

    unused_tools = _FakeTools([], [])
    unused_chat = _FakeChat(RuntimeError("must not be called"))
    outcome = build_character_guide(
        title, settings=settings, tools=unused_tools, chat_client=unused_chat
    )

    assert outcome.cached is True
    assert outcome.guide is not None
    assert unused_tools.searches == []
    assert unused_tools.fetches == []
    assert unused_chat.completions.requests == []


def test_a_character_without_a_source_is_dropped(tmp_path: Path) -> None:
    title, tools, _ = _research_fixture(tmp_path)
    chat = _FakeChat(
        {
            "characters": [
                {
                    "name": "Invented Person",
                    "gender": "feminine",
                    "sources": ["S9"],
                    "confidence": 0.9,
                }
            ]
        }
    )
    settings = _settings(tmp_path)

    outcome = build_character_guide(
        title, settings=settings, tools=tools, chat_client=chat
    )

    assert outcome.guide is None
    assert any("did not establish" in note for note in outcome.notes)


def test_research_without_a_key_does_nothing(tmp_path: Path) -> None:
    title, tools, chat = _research_fixture(tmp_path)
    settings = _settings(tmp_path, deepseek_api_key=None)

    outcome = build_character_guide(
        title, settings=settings, tools=tools, chat_client=chat
    )

    assert outcome.guide is None
    assert any("DEEPSEEK_API_KEY" in note for note in outcome.notes)
    assert tools.searches == []


def test_an_ambiguous_title_is_not_researched(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    outcome = build_character_guide(
        TitleInfo(title="", reason="ambiguous"), settings=settings
    )

    assert outcome.guide is None
    assert outcome.notes == ("ambiguous",)


# ---------------------------------------------------------------------------
# Speaker-attribution review
# ---------------------------------------------------------------------------


def _line(speaker: str = SPEAKER_A, text: str = "Hello.") -> TranscriptSegment:
    return TranscriptSegment(speaker_id=speaker, start=0.0, end=1.0, text=text)


def _evidence(**overrides: object) -> LineEvidence:
    values: dict[str, object] = {
        "index": 0,
        "speaker_id": SPEAKER_A,
        "start": 0.0,
        "end": 1.0,
        "text": "Hello.",
        "diarization_speakers": (SPEAKER_B,),
        "crosstalk_speakers": (),
        "crosstalk_seconds": 0.0,
        "verified": False,
        "speaker_median_hz": 120.0,
        "register_medians": {SPEAKER_A: 120.0, SPEAKER_B: 200.0},
    }
    values.update(overrides)
    return LineEvidence(**values)  # type: ignore[arg-type]


def _correction(
    *,
    speaker: str = SPEAKER_B,
    confidence: float = 0.9,
    evidence: list[str] | None = None,
) -> dict:
    return {
        "corrections": [
            {
                "index": 0,
                "speaker": speaker,
                "confidence": confidence,
                "evidence": evidence if evidence is not None else ["diarization"],
                "reason": "the line sits inside the other character's diarization turn",
            }
        ]
    }


def test_evidence_marks_crosstalk_and_embedded_labels() -> None:
    turns = [SimpleNamespace(start=0.0, end=2.0, speaker_id=SPEAKER_B)]
    verified = [SimpleNamespace(start=0.0, end=1.0, speaker_id=SPEAKER_A)]
    crosstalk = [SimpleNamespace(start=0.5, end=1.5, speaker_ids=(SPEAKER_A, SPEAKER_B))]

    evidence = collect_evidence(
        [_line()],
        turns=turns,
        verified=verified,
        crosstalk=crosstalk,
        registers={SPEAKER_A: SimpleNamespace(median_hz=120.0)},
    )

    assert evidence[0].verified is True
    assert evidence[0].crosstalk_speakers == (SPEAKER_A, SPEAKER_B)
    assert evidence[0].diarization_speakers == (SPEAKER_B,)
    assert evidence[0].speaker_median_hz == 120.0
    # A label the embeddings confirmed is never a review candidate.
    assert evidence[0].candidate is False

    unverified = collect_evidence(
        [_line()], turns=turns, crosstalk=crosstalk
    )
    assert unverified[0].candidate is True


def test_a_brief_overlap_does_not_make_a_line_a_candidate() -> None:
    """A few milliseconds at the edge of a turn is not an interruption."""

    turns = [SimpleNamespace(start=0.0, end=1.0, speaker_id=SPEAKER_A)]
    crosstalk = [
        SimpleNamespace(start=0.0, end=0.05, speaker_ids=(SPEAKER_A, SPEAKER_B))
    ]

    evidence = collect_evidence([_line()], turns=turns, crosstalk=crosstalk)

    assert evidence[0].crosstalk_seconds == pytest.approx(0.05)
    assert evidence[0].candidate is False


def test_overlap_evidence_must_cover_the_line(tmp_path: Path) -> None:
    chat = _FakeChat(
        _correction(evidence=["overlap"]),
    )
    settings = _settings(tmp_path)
    brushed = _evidence(
        crosstalk_speakers=(SPEAKER_A, SPEAKER_B),
        crosstalk_seconds=0.02,
        diarization_speakers=(SPEAKER_B,),
    )

    lines, review = review_speaker_attribution(
        [_line()], settings=settings, evidence=[brushed], chat_client=chat
    )

    assert lines[0].speaker_id == SPEAKER_A
    assert review.applied == ()
    assert "no diarization or overlap evidence" in review.uncertain[0].reason


def test_a_line_inside_crosstalk_is_flagged_not_changed(tmp_path: Path) -> None:
    """Both voices in the same instant cannot prove which one a line belongs to."""

    chat = _FakeChat(_correction())
    settings = _settings(tmp_path)
    overlapping = _evidence(
        crosstalk_speakers=(SPEAKER_A, SPEAKER_B), crosstalk_seconds=0.6
    )

    lines, review = review_speaker_attribution(
        [_line()], settings=settings, evidence=[overlapping], chat_client=chat
    )

    assert lines[0].speaker_id == SPEAKER_A
    assert review.applied == ()
    assert "overlapping speech" in review.uncertain[0].reason


def test_a_diarization_backed_correction_is_applied(tmp_path: Path) -> None:
    chat = _FakeChat(_correction())
    settings = _settings(tmp_path)

    lines, review = review_speaker_attribution(
        [_line()],
        settings=settings,
        evidence=[_evidence()],
        chat_client=chat,
    )

    assert lines[0].speaker_id == SPEAKER_B
    assert len(review.applied) == 1
    assert review.applied[0].evidence == ("diarization",)
    payload = json.loads(chat.completions.requests[0]["messages"][1]["content"])
    assert payload["lines"][0]["review"] is True
    assert payload["lines"][0]["diarization"] == [SPEAKER_B]


def test_context_only_proposal_is_left_for_qc(tmp_path: Path) -> None:
    chat = _FakeChat(_correction(evidence=["context"]))
    settings = _settings(tmp_path)

    lines, review = review_speaker_attribution(
        [_line()],
        settings=settings,
        evidence=[_evidence()],
        chat_client=chat,
    )

    assert lines[0].speaker_id == SPEAKER_A
    assert review.applied == ()
    assert len(review.uncertain) == 1
    assert "no diarization or overlap evidence" in review.uncertain[0].reason


def test_low_confidence_correction_is_not_applied(tmp_path: Path) -> None:
    chat = _FakeChat(_correction(confidence=0.4))
    settings = _settings(tmp_path)

    lines, review = review_speaker_attribution(
        [_line()],
        settings=settings,
        evidence=[_evidence()],
        chat_client=chat,
    )

    assert lines[0].speaker_id == SPEAKER_A
    assert review.applied == ()
    assert "below the" in review.uncertain[0].reason


def test_a_verified_label_is_never_sent_for_review(tmp_path: Path) -> None:
    chat = _FakeChat(RuntimeError("must not be called"))
    settings = _settings(tmp_path)

    lines, review = review_speaker_attribution(
        [_line()],
        settings=settings,
        evidence=[_evidence(verified=True)],
        chat_client=chat,
    )

    assert lines[0].speaker_id == SPEAKER_A
    assert review.applied == ()
    assert chat.completions.requests == []


def test_no_conflicting_evidence_makes_no_request(tmp_path: Path) -> None:
    chat = _FakeChat(RuntimeError("must not be called"))
    settings = _settings(tmp_path)

    lines, review = review_speaker_attribution(
        [_line()],
        settings=settings,
        evidence=[_evidence(diarization_speakers=(SPEAKER_A,))],
        chat_client=chat,
    )

    assert lines == (_line(),)
    assert review.applied == ()
    assert chat.completions.requests == []


def test_a_malformed_review_response_does_not_change_labels(tmp_path: Path) -> None:
    chat = _FakeChat("this is not JSON")
    settings = _settings(tmp_path)

    lines, review = review_speaker_attribution(
        [_line()],
        settings=settings,
        evidence=[_evidence()],
        chat_client=chat,
    )

    assert lines[0].speaker_id == SPEAKER_A
    assert review.applied == ()
    assert review.uncertain == ()

# ---------------------------------------------------------------------------
# Episode script research
# ---------------------------------------------------------------------------


def _dialogue_reply(count: int = 12) -> dict:
    speakers = ["Walt", "Skyler"]
    return {
        "lines": [
            {
                "speaker": speakers[index % 2],
                "text": f"Line number {index} about the family and the danger.",
            }
            for index in range(count)
        ]
    }


def _script_fixture(root: Path):
    from app.pipeline.dialogue_context import CharacterGuide, GuideCharacter

    path = _title_file(root, "breaking-bad.txt", "Breaking Bad S04E06")
    title = TitleInfo(title="Breaking Bad S04E06", path=path, season=4, episode=6)
    hits = [SearchHit("Transcript", "https://example.com/transcript", authority=2)]
    pages = [
        FetchedPage(
            "https://example.com/transcript",
            "Transcript",
            "\n".join(
                f"[Walt] Something spoken at length number {index}."
                for index in range(40)
            ),
        )
    ]
    guide = CharacterGuide(
        title="Breaking Bad S04E06",
        season=4,
        episode=6,
        characters=(
            GuideCharacter(name="Walt", gender="masculine", sources=("https://example.com",)),
            GuideCharacter(name="Skyler", gender="feminine", sources=("https://example.com",)),
        ),
    )
    return title, _FakeTools(hits, pages), guide


def test_episode_script_is_extracted_and_cached(tmp_path: Path) -> None:
    from app.pipeline.context_ai import build_episode_script

    title, tools, guide = _script_fixture(tmp_path)
    chat = _FakeChat(_dialogue_reply())
    settings = _settings(tmp_path)

    script = build_episode_script(
        title, settings=settings, guide=guide, tools=tools, chat_client=chat
    )

    assert script is not None
    assert len(script) == 12
    assert {line.speaker for line in script.lines} == {"Walt", "Skyler"}
    assert script.sources == ("https://example.com/transcript",)
    assert settings.episode_script_path.is_file()
    payload = json.loads(chat.completions.requests[0]["messages"][1]["content"])
    assert payload["title"] == title.label
    assert "Walt" in payload["known_characters"]


def test_a_cached_script_is_reused_without_network_calls(tmp_path: Path) -> None:
    from app.pipeline.context_ai import build_episode_script

    title, tools, guide = _script_fixture(tmp_path)
    settings = _settings(tmp_path)
    build_episode_script(
        title,
        settings=settings,
        guide=guide,
        tools=tools,
        chat_client=_FakeChat(_dialogue_reply()),
    )

    unused_tools = _FakeTools([], [])
    unused_chat = _FakeChat(RuntimeError("must not be called"))
    cached = build_episode_script(
        title,
        settings=settings,
        guide=guide,
        tools=unused_tools,
        chat_client=unused_chat,
    )

    assert cached is not None and len(cached) == 12
    assert unused_tools.searches == []
    assert unused_chat.completions.requests == []


def test_script_extraction_rejects_stage_directions_and_thin_pages(tmp_path: Path) -> None:
    from app.pipeline.context_ai import build_episode_script

    title, tools, guide = _script_fixture(tmp_path)
    chat = _FakeChat(
        {
            "lines": [
                {"speaker": "[gunshot]", "text": "bang"},
                {"speaker": "Walt", "text": "[door slams]"},
                {"speaker": "Walt", "text": "One."},
            ]
        }
    )

    script = build_episode_script(
        title, settings=_settings(tmp_path), guide=guide, tools=tools, chat_client=chat
    )

    assert script is None


def test_script_extraction_requires_two_speakers(tmp_path: Path) -> None:
    from app.pipeline.context_ai import build_episode_script

    title, tools, guide = _script_fixture(tmp_path)
    chat = _FakeChat(
        {
            "lines": [
                {"speaker": "Walt", "text": f"Only one voice says line {index}."}
                for index in range(12)
            ]
        }
    )

    script = build_episode_script(
        title, settings=_settings(tmp_path), guide=guide, tools=tools, chat_client=chat
    )

    assert script is None


def test_episode_script_research_needs_a_key(tmp_path: Path) -> None:
    from app.pipeline.context_ai import build_episode_script

    title, tools, guide = _script_fixture(tmp_path)
    chat = _FakeChat(_dialogue_reply())

    script = build_episode_script(
        title,
        settings=_settings(tmp_path, deepseek_api_key=None),
        guide=guide,
        tools=tools,
        chat_client=chat,
    )

    assert script is None
    assert tools.searches == []

def test_a_wrong_episode_script_is_rejected_by_phrase_overlap(tmp_path: Path) -> None:
    from app.pipeline.context_ai import build_episode_script

    title, tools, guide = _script_fixture(tmp_path)
    chat = _FakeChat(
        {
            "lines": [
                {
                    "speaker": "Walt" if index % 2 else "Skyler",
                    "text": f"An unrelated scene sentence number {index} about something else.",
                }
                for index in range(12)
            ]
        }
    )

    script = build_episode_script(
        title,
        settings=_settings(tmp_path),
        guide=guide,
        tools=tools,
        chat_client=chat,
        transcript=["I am not in danger, Skyler.", "I am the danger."],
    )

    assert script is None
    assert not _settings(tmp_path).episode_script_path.exists()
