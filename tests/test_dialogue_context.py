"""Tests for :mod:`app.pipeline.dialogue_context`.

Everything here is arithmetic over line timings and text, so no model is involved
and the whole module is exercised directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.pipeline.dialogue_context import (
    DEFAULT_SYLLABLES_PER_SECOND,
    MAXIMUM_SYLLABLE_BUDGET,
    Character,
    CharacterBible,
    InvalidBudgetError,
    InvalidCharacterError,
    InvalidSceneError,
    SyllableBudget,
    segment_scenes,
    syllable_budget,
)
from app.pipeline.translation import AdaptedDialogue

SPEAKER_A = "SPEAKER_00"
SPEAKER_B = "SPEAKER_01"

#: Three syllables.
THREE = "ሰላም"
#: Ten syllables.
TEN = "ሰላም እንደምን ነህ"


def _line(
    start: float,
    end: float,
    *,
    speaker_id: str = SPEAKER_A,
    amharic: str = THREE,
) -> AdaptedDialogue:
    return AdaptedDialogue(
        speaker_id=speaker_id,
        start=start,
        end=end,
        source_text="Hello.",
        amharic=amharic,
        emotion="neutral",
        intensity=0.5,
        delivery="calm",
        pause_before=0.0,
        pause_after=0.0,
    )


def _character(speaker_id: str = SPEAKER_A, name: str = "Selam", **overrides: object) -> Character:
    values: dict[str, object] = {"speaker_id": speaker_id, "name": name}
    values.update(overrides)
    return Character(**values)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# scene segmentation
# ---------------------------------------------------------------------------


def test_a_short_pause_does_not_start_a_new_scene() -> None:
    """A beat of silence between two lines is turn-taking, not a cut."""

    scenes = segment_scenes(
        [_line(0.0, 2.0), _line(3.0, 5.0), _line(6.0, 8.0)], gap_seconds=4.0
    )

    assert len(scenes) == 1
    assert scenes[0].line_indexes == (0, 1, 2)


def test_a_long_silence_starts_a_new_scene() -> None:
    scenes = segment_scenes(
        [_line(0.0, 2.0), _line(30.0, 32.0), _line(31.0, 33.0)],
        gap_seconds=4.0,
    )

    assert len(scenes) == 2
    assert scenes[0].line_indexes == (0,)
    assert scenes[1].line_indexes == (1, 2)
    assert [scene.index for scene in scenes] == [0, 1]


def test_a_scene_is_cut_when_it_runs_too_long() -> None:
    """One unbroken conversation must not become a scene covering half the film."""

    lines = [_line(float(second), float(second) + 1.0) for second in range(20)]

    scenes = segment_scenes(lines, gap_seconds=4.0, max_seconds=5.0)

    assert len(scenes) > 1
    assert all(scene.duration <= 5.0 for scene in scenes)
    # Every line lands in exactly one scene, in order.
    assert [index for scene in scenes for index in scene.line_indexes] == list(range(20))


def test_scene_participants_are_the_speakers_present() -> None:
    scenes = segment_scenes(
        [
            _line(0.0, 1.0, speaker_id=SPEAKER_A),
            _line(1.0, 2.0, speaker_id=SPEAKER_B),
            _line(2.0, 3.0, speaker_id=SPEAKER_A),
        ]
    )

    assert scenes[0].speaker_ids == (SPEAKER_A, SPEAKER_B)
    assert scenes[0].line_count == 3
    assert scenes[0].start == 0.0
    assert scenes[0].end == 3.0
    assert scenes[0].duration == 3.0


def test_an_empty_dialogue_makes_no_scenes() -> None:
    assert segment_scenes([]) == ()


def test_segmentation_is_deterministic() -> None:
    lines = [_line(0.0, 1.0), _line(9.0, 10.0), _line(10.0, 11.0)]

    assert segment_scenes(lines) == segment_scenes(lines)


def test_scene_segmentation_rejects_impossible_settings() -> None:
    with pytest.raises(InvalidSceneError, match="gap_seconds"):
        segment_scenes([_line(0.0, 1.0)], gap_seconds=-1.0)
    with pytest.raises(InvalidSceneError, match="max_seconds"):
        segment_scenes([_line(0.0, 1.0)], max_seconds=0)


def test_scene_as_dict_is_json_safe() -> None:
    (scene,) = segment_scenes([_line(0.0, 1.0)])

    assert json.loads(json.dumps(scene.as_dict())) == scene.as_dict()


# ---------------------------------------------------------------------------
# characters and the bible
# ---------------------------------------------------------------------------


def test_a_character_renders_its_prompt_block() -> None:
    block = _character(
        name="Selam",
        aliases=("Selamitu",),
        gender="female",
        age="mid thirties",
        register="informal with friends, formal with officials",
        relationships={"SPEAKER_01": "her younger brother"},
        terms={"Addis": "አዲስ"},
        notes="Never swears.",
    ).as_prompt_block()

    assert block.startswith(f"- {SPEAKER_A} is Selam (also called Selamitu)")
    assert "female, mid thirties" in block
    assert "her younger brother" in block
    assert "Addis -> አዲስ" in block
    assert block.endswith("Never swears.")


def test_a_minimal_character_still_renders() -> None:
    assert _character().as_prompt_block() == f"- {SPEAKER_A} is Selam."


def test_a_character_needs_an_id_and_a_name() -> None:
    with pytest.raises(InvalidCharacterError, match="speaker_id"):
        Character(speaker_id="  ", name="Selam")
    with pytest.raises(InvalidCharacterError, match="name"):
        Character(speaker_id=SPEAKER_A, name="")


def test_character_aliases_are_deduplicated_and_trimmed() -> None:
    character = _character(aliases=(" Sel ", "Sel", ""), )

    assert character.aliases == ("Sel",)


def test_the_bible_only_renders_the_characters_in_the_scene() -> None:
    """A casting list of forty names per request is noise."""

    bible = CharacterBible(
        {
            SPEAKER_A: _character(name="Selam"),
            SPEAKER_B: _character(SPEAKER_B, name="Dawit"),
        }
    )

    block = bible.as_prompt_block([SPEAKER_A])

    assert "Selam" in block
    assert "Dawit" not in block


def test_an_unknown_speaker_is_skipped_not_invented() -> None:
    bible = CharacterBible({SPEAKER_A: _character()})

    assert bible.as_prompt_block(["SPEAKER_99"]) == ""
    assert bible.get("SPEAKER_99") is None
    # A missing name falls back to the diarization label, which is what it is.
    assert bible.name_of("SPEAKER_99") == "SPEAKER_99"


def test_a_bible_rejects_a_mismatched_entry() -> None:
    with pytest.raises(InvalidCharacterError, match="names itself"):
        CharacterBible({SPEAKER_A: _character(SPEAKER_B)})

    with pytest.raises(InvalidCharacterError, match="non-empty speaker ids"):
        CharacterBible({"  ": _character()})


def test_a_bible_round_trips_through_disk(tmp_path: Path) -> None:
    bible = CharacterBible(
        {
            SPEAKER_A: _character(
                name="Selam",
                aliases=("Selamitu",),
                register="informal",
                relationships={SPEAKER_B: "her brother"},
                terms={"Addis": "አዲስ"},
                notes="Never swears.",
            )
        }
    )

    path = bible.save(tmp_path / "voices" / "bible.json")

    assert path.is_file()
    assert CharacterBible.load(path) == bible


def test_a_missing_bible_is_empty_rather_than_an_error(tmp_path: Path) -> None:
    """A first run has no state yet, which is normal, not a failure."""

    bible = CharacterBible.load(tmp_path / "not-written-yet.json")

    assert len(bible) == 0
    assert bible.as_prompt_block([SPEAKER_A]) == ""


def test_a_malformed_bible_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "bible.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(InvalidCharacterError, match="JSON object"):
        CharacterBible.load(path)


def test_a_bible_entry_without_a_name_falls_back_to_its_label(tmp_path: Path) -> None:
    path = tmp_path / "bible.json"
    path.write_text(json.dumps({SPEAKER_A: {"register": "formal"}}), encoding="utf-8")

    bible = CharacterBible.load(path)

    assert bible.name_of(SPEAKER_A) == SPEAKER_A
    assert bible.get(SPEAKER_A).register == "formal"  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# the syllable budget
# ---------------------------------------------------------------------------


def test_a_budget_scales_with_the_window() -> None:
    assert syllable_budget(2.0, rate=4.0).syllables == 8
    assert syllable_budget(0.5, rate=4.0).syllables == 2


def test_a_budget_is_never_zero() -> None:
    """Even a very short window has to allow a syllable."""

    assert syllable_budget(0.01, rate=4.0).syllables == 1


def test_a_budget_is_capped() -> None:
    """A long original window must not become an invitation to pad a line."""

    assert syllable_budget(600.0, rate=4.0).syllables == MAXIMUM_SYLLABLE_BUDGET
    assert syllable_budget(600.0, rate=4.0, maximum=20).syllables == 20


def test_the_default_rate_is_the_documented_prior() -> None:
    assert DEFAULT_SYLLABLES_PER_SECOND == 4.0
    assert syllable_budget(1.0).rate == DEFAULT_SYLLABLES_PER_SECOND


def test_a_budget_rejects_impossible_inputs() -> None:
    with pytest.raises(InvalidBudgetError, match="window"):
        syllable_budget(0.0)
    with pytest.raises(InvalidBudgetError, match="rate"):
        syllable_budget(1.0, rate=0.0)


def test_a_line_inside_its_budget_is_accepted() -> None:
    verdict = syllable_budget(2.0, rate=4.0).verdict(THREE)

    assert verdict.used == 3
    assert verdict.budget.syllables == 8
    assert verdict.within_tolerance is True
    assert verdict.overshoot == 0


def test_a_line_inside_the_tolerance_band_is_left_alone() -> None:
    """Ten percent over is inside what timing can absorb without an audible stretch."""

    verdict = syllable_budget(1.0, rate=4.0).verdict("ሰላም እ")  # 4 syllables, budget 4

    assert verdict.within_tolerance is True


def test_an_overlong_line_is_reported_with_how_much_to_cut() -> None:
    verdict = syllable_budget(1.0, rate=4.0).verdict(TEN)

    assert verdict.used == 10
    assert verdict.budget.syllables == 4
    assert verdict.within_tolerance is False
    assert verdict.overshoot == 6
    assert "cut at least 6 syllable(s)" in verdict.describe()
    assert "4" in verdict.describe()


def test_exactly_at_budget_is_within_tolerance_even_with_none_allowed() -> None:
    verdict = syllable_budget(1.0, rate=4.0).verdict("ሰላም እ", tolerance=0.0)

    assert verdict.used == 4
    assert verdict.within_tolerance is True


def test_a_tighter_tolerance_is_honoured() -> None:
    """Ten syllables against a four-syllable budget is over however generous."""

    verdict = syllable_budget(1.0, rate=4.0).verdict(TEN, tolerance=0.0)

    assert verdict.within_tolerance is False


def test_a_budget_verdict_is_json_safe() -> None:
    verdict = syllable_budget(1.0, rate=4.0).verdict(TEN)

    assert json.loads(json.dumps(verdict.as_dict())) == verdict.as_dict()
    assert json.loads(json.dumps(verdict.budget.as_dict())) == verdict.budget.as_dict()


def test_a_budget_needs_a_positive_window() -> None:
    with pytest.raises(InvalidBudgetError, match="window must be positive"):
        SyllableBudget(window=0.0, syllables=1, rate=4.0)
