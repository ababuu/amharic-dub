"""Tests for operator-supplied scripts, subtitles and cast files.

These are the free path: local files must be consumed without a Search, Fetch or
LLM extraction request.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings
from app.pipeline import context_ai
from app.pipeline.context_ai import (
    ResearchUnavailable,
    discover_title,
    episode_script_for_run,
    local_cast_for_run,
    local_script_for_run,
)
from app.pipeline.dialogue_context import CharacterGuide


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root / "input",
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "models",
        "dialogue_bible_path": root / "work" / "dialogue_bible.json",
        "character_guide_path": root / "work" / "character_guide.json",
        "episode_script_path": root / "work" / "episode_script.json",
        "deepseek_api_key": None,
        "translation_backend": "openai",
        "translation_provider": "deepseek",
        "translation_model": "deepseek-v4-pro",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _write(root: Path, name: str, text: str) -> Path:
    directory = root / "input"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def _source(root: Path) -> Path:
    path = root / "test.mp4"
    path.write_bytes(b"stand-in")
    return path


class _ExplodingTools:
    def __init__(self) -> None:
        self.searches: list[str] = []
        self.fetches: list[str] = []

    def search(self, query: str, *, count: int = 5):
        self.searches.append(query)
        raise ResearchUnavailable("must not be called")

    def fetch(self, url: str):
        self.fetches.append(url)
        raise ResearchUnavailable("must not be called")


class _ExplodingChat:
    def __init__(self) -> None:
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **_kwargs):
        raise AssertionError("the model must not be called for local sources")


def test_title_discovery_ignores_a_local_script(tmp_path: Path) -> None:
    _write(tmp_path, "test.title.txt", "Breaking Bad S04E06\n")
    _write(
        tmp_path,
        "test.script.txt",
        "\n".join(
            f"{'Walt' if index % 2 else 'Skyler'}: Script line number {index}."
            for index in range(20)
        ),
    )

    title = discover_title(tmp_path / "input", source=_source(tmp_path))

    assert title is not None
    assert title.title == "Breaking Bad S04E06"
    assert (title.season, title.episode) == (4, 6)


def test_local_labeled_txt_is_parsed_without_network(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.script.txt",
        "Walt: I don't want to hear about the police.\n"
        "Skyler: I do not say that lightly.\n"
        "Walt: I don't want to hear about the police.\n"
        "Skyler: You're not some hardened criminal, Walt.\n",
    )
    settings = _settings(tmp_path)

    script = local_script_for_run(_source(tmp_path), settings=settings)

    assert script is not None
    assert len(script) == 4
    assert {line.speaker for line in script.lines} == {"Walt", "Skyler"}
    assert script.sources == (str(tmp_path / "input" / "test.script.txt"),)


def test_local_script_wins_over_web_research(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.script.txt",
        "Walt: I am the danger.\nSkyler: Walt, please.\nWalt: No.\n",
    )
    settings = _settings(tmp_path)
    tools = _ExplodingTools()

    script = episode_script_for_run(
        _source(tmp_path),
        settings=settings,
        tools=tools,
        chat_client=_ExplodingChat(),
    )

    assert script is not None and len(script) == 3
    assert tools.searches == []
    assert tools.fetches == []


def test_local_srt_with_speaker_names_is_parsed(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.srt",
        "1\n00:00:01,000 --> 00:00:02,000\nWalt: I don't want to hear about the police.\n\n"
        "2\n00:00:02,100 --> 00:00:03,000\n<v Skyler>I do not say that lightly.\n\n"
        "3\n00:00:03,100 --> 00:00:04,000\nWalt: I don't want to hear about the police.\n",
    )

    script = local_script_for_run(_source(tmp_path), settings=_settings(tmp_path))

    assert script is not None
    assert [line.speaker for line in script.lines] == ["Walt", "Skyler", "Walt"]
    assert script.lines[1].text == "I do not say that lightly."


def test_unlabelled_srt_is_rejected(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.srt",
        "1\n00:00:01,000 --> 00:00:02,000\nI am the danger.\n\n"
        "2\n00:00:03,000 --> 00:00:04,000\nI am the one who knocks.\n",
    )
    messages: list[str] = []

    script = local_script_for_run(
        _source(tmp_path), settings=_settings(tmp_path), report=messages.append
    )

    assert script is None
    assert any("labelled line" in message for message in messages)


def test_unlabelled_lines_carry_the_previous_speaker(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.script.txt",
        "Walt: I am the danger.\nA guy opens his door and gets shot.\n"
        "Skyler: Walt, please.\nNo, I am the one who knocks.\n",
    )

    script = local_script_for_run(_source(tmp_path), settings=_settings(tmp_path))

    assert script is not None
    assert [line.speaker for line in script.lines] == [
        "Walt",
        "Walt",
        "Skyler",
        "Skyler",
    ]


def test_local_cast_json_skips_character_research(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.cast.json",
        json.dumps(
            {
                "characters": [
                    {"name": "Walt", "aliases": ["Walter White"], "gender": "masculine"},
                    {"name": "Skyler", "gender": "feminine"},
                ]
            }
        ),
    )
    settings = _settings(tmp_path)
    tools = _ExplodingTools()

    guide = context_ai.character_guide_for_run(
        _source(tmp_path),
        settings=settings,
        tools=tools,
        chat_client=_ExplodingChat(),
    )

    assert guide is not None
    assert {character.name: character.gender for character in guide.characters} == {
        "Walt": "masculine",
        "Skyler": "feminine",
    }
    assert tools.searches == []


def test_local_script_supplies_names_when_no_cast_exists(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.script.txt",
        "Walt: I am the danger.\nSkyler: Walt, please.\nWalt: No.\n",
    )
    settings = _settings(tmp_path)

    guide = context_ai.character_guide_for_run(
        _source(tmp_path),
        settings=settings,
        tools=_ExplodingTools(),
        chat_client=_ExplodingChat(),
    )

    assert guide is not None
    assert {character.name for character in guide.characters} == {"Walt", "Skyler"}
    assert all(character.gender == "unknown" for character in guide.characters)


def test_local_cast_is_available_without_an_api_key(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.cast.json",
        json.dumps({"characters": [{"name": "Walt", "gender": "masculine"}]}),
    )

    guide = local_cast_for_run(
        _source(tmp_path), settings=_settings(tmp_path, deepseek_api_key=None)
    )

    assert isinstance(guide, CharacterGuide)
    assert len(guide) == 1

def test_an_unusable_script_txt_falls_back_to_a_labelled_srt(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "test.script.txt",
        "This is plain prose with no speaker names at all.\n"
        "It should not be accepted as a script.\n",
    )
    _write(
        tmp_path,
        "test.srt",
        "1\n00:00:01,000 --> 00:00:02,000\nWalt: I am the danger.\n\n"
        "2\n00:00:02,100 --> 00:00:03,000\nSkyler: Walt, please.\n\n"
        "3\n00:00:03,100 --> 00:00:04,000\nWalt: No.\n",
    )

    script = local_script_for_run(_source(tmp_path), settings=_settings(tmp_path))

    assert script is not None
    assert {line.speaker for line in script.lines} == {"Walt", "Skyler"}
    assert script.sources[0].endswith("test.srt")
