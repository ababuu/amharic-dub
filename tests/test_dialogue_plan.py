"""Tests for :mod:`app.pipeline.dialogue_plan`.

The fixture is the shape of a real authored plan: 26 cues for a scene, written
against the episode's subtitle grid, and a two-minute clip cut from that episode.
Nothing here touches a model, a GPU or the network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.pipeline.dialogue_plan import (
    PlanError,
    cast_plan,
    discover_plan,
    dropped_cues,
    load_plan,
    parse_time,
    plan_dialogue,
    plan_from_payload,
    retime_plan,
    scope_plan,
)
from app.pipeline.transcription import TranscriptSegment

SKYLER = "SPEAKER_00"
WALTER = "SPEAKER_01"

PAYLOAD = {
    "characters": [
        {"name": "Walter", "aliases": ["Walt"]},
        {"name": "Skyler"},
        {"name": "Narrator", "aliases": ["Previously on"]},
    ],
    "dialogue": [
        {
            "cue": 68,
            "speaker": "Skyler",
            "start": "00:03:30,000",
            "end": "00:03:32,000",
            "source_text": "Walt. I've said it before. If you are in danger, we go to the police.",
            "amharic": "ዋልት። ቀደም ብዬ ነግሬሃለሁ።",
            "emotion": "serious",
            "intensity": 0.6,
            "delivery": "firm",
        },
        {
            "cue": 69,
            "speaker": "Walter",
            "start": "00:03:33,000",
            "end": "00:03:35,000",
            "source_text": "I don't want to hear about the police.",
            "amharic": "ስለ ፖሊስ ማዳመጥ አልፈልግም።",
            "emotion": "dismissive",
            "intensity": 0.6,
            "delivery": "firm",
        },
        {
            "cue": 70,
            "speaker": "Skyler",
            "start": "00:03:36,000",
            "end": "00:03:38,000",
            "source_text": "You're not some hardened criminal, Walt.",
            "amharic": "አንተ የሰለጠነ ወንጀለኛ አይደለህም፣ ዋልት።",
            "emotion": "pleading",
            "intensity": 0.5,
            "delivery": "gentle",
        },
        {
            "cue": 71,
            "speaker": "Walter",
            "start": "00:03:39,000",
            "end": "00:03:41,000",
            "source_text": "That is not the truth.",
            "amharic": "ያ እውነት አይደለም።",
            "emotion": "angry",
            "intensity": 0.7,
            "delivery": "firm",
        },
    ],
}


def _plan(**overrides):
    return plan_from_payload(overrides or PAYLOAD)


def _transcript() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(SKYLER, 7.52, 8.24, "I've said it before."),
        TranscriptSegment(SKYLER, 8.91, 10.51, "if you are in danger."),
        TranscriptSegment(SKYLER, 11.19, 11.93, "We go to the police."),
        TranscriptSegment(WALTER, 12.82, 14.38, "I don't want to hear about the police."),
        TranscriptSegment(SKYLER, 14.14, 16.56, "You're not some hardened criminal, Walt."),
        TranscriptSegment(WALTER, 16.70, 18.10, "That is not the truth."),
    ]


def _turns() -> list[SimpleNamespace]:
    return [
        SimpleNamespace(start=7.52, end=11.93, speaker_id=SKYLER),
        SimpleNamespace(start=12.82, end=14.38, speaker_id=WALTER),
        SimpleNamespace(start=14.14, end=16.56, speaker_id=SKYLER),
        SimpleNamespace(start=16.70, end=18.10, speaker_id=WALTER),
    ]


def _registers() -> dict[str, object]:
    return {
        SKYLER: SimpleNamespace(median_hz=200.0, presentation="feminine"),
        WALTER: SimpleNamespace(median_hz=120.0, presentation="masculine"),
    }


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_timecodes_are_parsed():
    assert parse_time("00:03:30,000", field_name="t") == 210.0
    assert parse_time("00:03:30.500", field_name="t") == 210.5
    assert parse_time("01:02:03,004", field_name="t") == 3723.004
    assert parse_time("12.5", field_name="t") == 12.5
    assert parse_time(9, field_name="t") == 9.0


def test_an_unparseable_time_is_rejected():
    with pytest.raises(PlanError):
        parse_time("half past three", field_name="t")


def test_a_plan_is_parsed_in_order_with_its_metadata():
    plan = _plan()

    assert len(plan) == 4
    assert plan.speakers == ("Skyler", "Walter")
    assert plan.gender_of("Walter") == "unknown"
    assert plan.cues[0].start == 210.0
    assert plan.cues[0].amharic.startswith("ዋልት")
    assert plan.cues[0].emotion == "serious"
    assert plan.gender_of("Nobody") == "unknown"


def test_a_character_named_only_in_the_dialogue_is_still_known():
    plan = plan_from_payload(
        {
            "dialogue": [
                {
                    "speaker": "Unlisted",
                    "start": 1.0,
                    "end": 2.0,
                    "amharic": "ሰላም።",
                }
            ]
        }
    )

    assert [character.name for character in plan.characters] == ["Unlisted"]


def test_missing_amharic_or_speaker_is_rejected():
    with pytest.raises(PlanError):
        plan_from_payload(
            {"dialogue": [{"speaker": "Walt", "start": 1.0, "end": 2.0}]}
        )
    with pytest.raises(PlanError):
        plan_from_payload(
            {
                "dialogue": [
                    {"speaker": "Walt", "start": 2.0, "end": 1.0, "amharic": "x"}
                ]
            }
        )


def test_an_empty_plan_is_rejected():
    with pytest.raises(PlanError):
        plan_from_payload({"dialogue": []})


def test_a_plan_is_discovered_from_the_video_stem(tmp_path: Path) -> None:
    source = tmp_path / "test.mp4"
    source.write_bytes(b"stand-in")
    named = tmp_path / "test.plan.json"
    named.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    (tmp_path / "other.plan.json").write_text(json.dumps(PAYLOAD), encoding="utf-8")

    assert discover_plan(tmp_path, source) == named


def test_a_cast_file_is_not_mistaken_for_a_plan(tmp_path: Path) -> None:
    (tmp_path / "test.cast.json").write_text(
        json.dumps({"characters": [{"name": "Walt"}]}), encoding="utf-8"
    )

    assert discover_plan(tmp_path, tmp_path / "test.mp4") is None


def test_the_sample_file_shape_loads(tmp_path: Path) -> None:
    path = tmp_path / "sample.plan.json"
    path.write_text(json.dumps(PAYLOAD), encoding="utf-8")

    plan = load_plan(path)

    assert plan.source == path
    assert len(plan) == 4


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


def test_authored_timing_is_used_when_it_is_explicitly_kept():
    plan = plan_from_payload(
        {
            "dialogue": [
                {
                    "speaker": "Skyler",
                    "start": "00:00:07,520",
                    "end": "00:00:11,930",
                    "source_text": (
                        "I've said it before. If you are in danger, we go to the police."
                    ),
                    "amharic": "x",
                }
            ]
        }
    )

    scope = scope_plan(plan, transcript=_transcript(), audio_duration=131.9)

    assert scope.offset == 0.0
    assert len(scope) == 1
    assert scope.cues[0].start == pytest.approx(7.52)


def test_lines_outside_the_audio_are_dropped_and_counted():
    plan = _plan()

    scope = scope_plan(plan, audio_duration=60.0)

    assert len(scope) == 0
    assert scope.dropped_outside == 4
    assert "no plan line falls inside this audio" in scope.method


def test_keeping_authored_timing_still_measures_a_whole_plan_shift():
    plan = _plan()

    scope = scope_plan(
        plan, transcript=_transcript(), audio_duration=131.9, retime=False
    )

    # The transcript says the same words about 202s earlier, so the plan is moved
    # onto the clip rather than being thrown away.
    assert scope.offset == pytest.approx(-202.48, abs=1.0)
    assert len(scope) == 4
    assert scope.dropped_outside == 0


def test_re_timing_gives_each_line_the_span_of_its_words():
    plan = _plan()

    scope = scope_plan(plan, transcript=_transcript(), audio_duration=131.9)

    assert scope.method.startswith("timing from the transcript")
    assert len(scope) == 4
    # The first cue's words are three transcribed fragments, so its span covers them.
    assert scope.cues[0].start == pytest.approx(7.52)
    assert scope.cues[0].end == pytest.approx(11.93)
    assert scope.cues[1].start == pytest.approx(12.82)
    assert scope.cues[3].end == pytest.approx(18.10)


def test_a_line_the_transcript_missed_is_kept_between_its_neighbours():
    """The plan says the line exists, so it is not silently dropped."""

    plan = plan_from_payload(
        {
            "dialogue": [
                {
                    "speaker": "Skyler",
                    "start": 7.0,
                    "end": 12.0,
                    "source_text": (
                        "I've said it before. If you are in danger, we go to the police."
                    ),
                    "amharic": "a",
                },
                {
                    "speaker": "Skyler",
                    "start": 12.0,
                    "end": 14.0,
                    "source_text": "Something the recogniser never caught at all.",
                    "amharic": "b",
                },
                {
                    "speaker": "Walter",
                    "start": 14.0,
                    "end": 18.2,
                    "source_text": "I don't want to hear about the police.",
                    "amharic": "c",
                },
            ]
        }
    )

    scope = scope_plan(plan, transcript=_transcript(), audio_duration=131.9)

    assert len(scope) == 3
    assert scope.interpolated == 1
    kept = scope.cues[1]
    assert kept.amharic == "b"
    # Between the matched neighbours, in plan order, and never before the first.
    assert scope.cues[0].start <= kept.start <= scope.cues[2].start
    assert kept.end > kept.start


def test_re_timing_drops_cues_with_no_matching_speech():
    plan = plan_from_payload(
        {
            "dialogue": [
                {
                    "speaker": "Skyler",
                    "start": 0.0,
                    "end": 1.0,
                    "source_text": "Completely unrelated dialogue about submarines.",
                    "amharic": "x",
                },
                {
                    "speaker": "Walter",
                    "start": 2.0,
                    "end": 3.0,
                    "source_text": "That is not the truth.",
                    "amharic": "y",
                },
            ]
        }
    )

    cues, unmatched = retime_plan(plan, _transcript())

    assert unmatched == 1
    assert [cue.speaker for cue in cues] == ["Walter"]


# ---------------------------------------------------------------------------
# Casting
# ---------------------------------------------------------------------------


def test_each_character_is_cast_to_the_voice_that_carries_its_lines():
    plan = _plan()
    scope = scope_plan(plan, transcript=_transcript(), retime=True)

    casting = cast_plan(
        plan, scope.cues, turns=_turns(), registers=_registers()
    )

    mapping = casting.mapping
    assert mapping == {"Skyler": SKYLER, "Walter": WALTER}
    assert casting.unassigned == ()
    assert all(cast.speaker_gender for cast in casting.casts)


def test_a_character_with_no_measured_voice_is_left_unassigned():
    plan = plan_from_payload(
        {
            "characters": [{"name": "Narrator"}],
            "dialogue": [
                {
                    "speaker": "Narrator",
                    "start": 200.0,
                    "end": 202.0,
                    "source_text": "Previously on Breaking Bad.",
                    "amharic": "x",
                }
            ],
        }
    )

    casting = cast_plan(plan, plan.cues, turns=_turns(), registers=_registers())

    narrator = casting.casts[0]
    assert narrator.speaker_id is None
    assert "no line of this character" in narrator.reason


def test_a_line_inside_crosstalk_is_not_casting_evidence():
    plan = plan_from_payload(
        {
            "dialogue": [
                {
                    "speaker": "Skyler",
                    "start": 10.0,
                    "end": 12.0,
                    "amharic": "x",
                }
            ]
        }
    )
    crosstalk = [SimpleNamespace(start=9.5, end=12.5)]

    casting = cast_plan(
        plan,
        plan.cues,
        turns=_turns(),
        registers=_registers(),
        crosstalk=crosstalk,
    )

    assert casting.mapping == {}
    assert "crosstalk" in casting.casts[0].reason


def test_two_characters_never_share_one_voice():
    plan = plan_from_payload(
        {
            "dialogue": [
                {"speaker": "A", "start": 7.5, "end": 11.9, "amharic": "x"},
                {"speaker": "B", "start": 7.6, "end": 11.8, "amharic": "y"},
            ]
        }
    )

    casting = cast_plan(
        plan, plan.cues, turns=_turns(), registers=_registers()
    )

    assert len(casting.mapping) == 1


def test_the_authored_gender_decides_which_voice_a_character_gets():
    plan = plan_from_payload(
        {
            "characters": [{"name": "Skyler", "gender": "masculine"}],
            "dialogue": [
                {"speaker": "Skyler", "start": 7.5, "end": 11.9, "amharic": "x"}
            ],
        }
    )

    casting = cast_plan(plan, plan.cues, turns=_turns(), registers=_registers())

    # The only voice carrying this line measures feminine, so it contradicts the
    # gender the plan states and is not used.
    assert casting.mapping == {}
    assert "the plan says masculine" in casting.casts[0].reason


def test_the_authored_gender_rejects_the_other_register_even_when_it_dominates():
    """Two characters can share a register; the authored gender breaks the tie."""

    plan = plan_from_payload(
        {
            "characters": [{"name": "Skyler", "gender": "masculine"}],
            "dialogue": [
                {"speaker": "Skyler", "start": 7.5, "end": 11.9, "amharic": "x"},
                {"speaker": "Skyler", "start": 12.9, "end": 14.3, "amharic": "y"},
            ],
        }
    )
    turns = [
        SimpleNamespace(start=7.52, end=11.93, speaker_id=SKYLER),
        SimpleNamespace(start=12.82, end=14.38, speaker_id=WALTER),
    ]

    casting = cast_plan(plan, plan.cues, turns=turns, registers=_registers())

    assert casting.mapping == {"Skyler": WALTER}


def test_no_diarized_turns_means_no_casting():
    plan = _plan()

    casting = cast_plan(plan, plan.cues, turns=())

    assert casting.mapping == {}
    assert "no diarized turns" in casting.skipped


# ---------------------------------------------------------------------------
# Dialogue
# ---------------------------------------------------------------------------


def test_the_plan_becomes_dialogue_lines_with_the_cast_voices():
    plan = _plan()
    scope = scope_plan(plan, transcript=_transcript(), retime=True)
    casting = cast_plan(plan, scope.cues, turns=_turns(), registers=_registers())

    dialogue = plan_dialogue(plan, scope, casting)

    assert len(dialogue) == 4
    assert dialogue[0].speaker_id == SKYLER
    assert dialogue[1].speaker_id == WALTER
    assert dialogue[0].amharic.startswith("ዋልት")
    assert dialogue[1].emotion == "dismissive"
    assert dialogue[0].intensity == pytest.approx(0.6)
    assert all(line.start < line.end for line in dialogue)


def test_a_line_whose_character_has_no_voice_is_not_voiced():
    plan = plan_from_payload(
        {
            "characters": [{"name": "Narrator"}],
            "dialogue": [
                {"speaker": "Narrator", "start": 1.0, "end": 2.0, "amharic": "x"}
            ],
        }
    )
    scope = scope_plan(plan)
    casting = cast_plan(plan, scope.cues, turns=_turns())

    assert plan_dialogue(plan, scope, casting) == ()
    assert dropped_cues(scope, casting) == 1

def test_authored_timings_on_the_wrong_grid_are_ignored_by_default():
    """A plan 2s early would cast every line to whoever was talking then."""

    plan = plan_from_payload(
        {
            "dialogue": [
                {
                    "speaker": "Skyler",
                    "start": 5.88,
                    "end": 7.88,
                    "source_text": (
                        "I've said it before. If you are in danger, we go to the police."
                    ),
                    "amharic": "x",
                },
                {
                    "speaker": "Walter",
                    "start": 10.0,
                    "end": 12.0,
                    "source_text": "I don't want to hear about the police.",
                    "amharic": "y",
                },
                {
                    "speaker": "Skyler",
                    "start": 12.5,
                    "end": 14.5,
                    "source_text": "You're not some hardened criminal, Walt.",
                    "amharic": "z",
                },
            ]
        }
    )

    scope = scope_plan(plan, transcript=_transcript(), audio_duration=131.9)

    assert scope.method.startswith("timing from the transcript")
    assert len(scope) == 3
    # The words, not the authored grid, decided where each line sits.
    assert scope.cues[0].start == pytest.approx(7.52)
    assert scope.cues[1].start == pytest.approx(12.82)
    assert scope.cues[2].start == pytest.approx(14.14)

def _casting(plan) -> "PlanCasting":
    from app.pipeline.dialogue_plan import CharacterCast, PlanCasting

    return PlanCasting(
        casts=tuple(
            CharacterCast(
                character=name,
                speaker_id=SKYLER if name == "Skyler" else WALTER,
                gender="unknown",
                speaker_gender="unknown",
                share=1.0,
                seconds=1.0,
                runner_up=None,
                reason="fixture",
            )
            for name in plan.speakers
        )
    )


def test_a_plan_line_may_use_the_silence_before_the_next_line():
    plan = plan_from_payload(
        {
            "dialogue": [
                {
                    "speaker": "Skyler",
                    "start": 10.0,
                    "end": 11.0,
                    "amharic": "a",
                },
                {
                    "speaker": "Walter",
                    "start": 14.0,
                    "end": 15.0,
                    "amharic": "b",
                },
            ]
        }
    )
    scope = scope_plan(plan)
    casting = _casting(plan)

    dialogue = plan_dialogue(plan, scope, casting, min_gap=0.12)

    # The first line keeps its start and gains room for Amharic that needs longer;
    # it still stops before the other character speaks.
    assert dialogue[0].start == pytest.approx(10.0)
    assert dialogue[0].end == pytest.approx(13.88)
    assert dialogue[1].start == pytest.approx(14.0)
    # A line with nowhere to grow is left exactly as it was.
    assert dialogue[1].end == pytest.approx(15.0)


def test_window_growth_is_capped_so_a_line_does_not_drift_to_its_neighbour():
    plan = plan_from_payload(
        {
            "dialogue": [
                {"speaker": "Skyler", "start": 10.0, "end": 11.0, "amharic": "a"},
                {"speaker": "Walter", "start": 40.0, "end": 41.0, "amharic": "b"},
            ]
        }
    )
    scope = scope_plan(plan)
    casting = _casting(plan)

    dialogue = plan_dialogue(plan, scope, casting, min_gap=0.12, max_growth=3.0)

    assert dialogue[0].end == pytest.approx(14.0)

def test_an_unvoiced_plan_character_is_cast_from_the_bank_by_its_gender():
    from app.pipeline.dialogue_plan import bank_speaker_id, assign_bank_voices

    plan = plan_from_payload(
        {
            "characters": [{"name": "Gus", "gender": "masculine"}],
            "dialogue": [
                {"speaker": "Gus", "start": 200.0, "end": 202.0, "amharic": "x"}
            ],
        }
    )
    casting = cast_plan(plan, plan.cues, turns=_turns(), registers=_registers())

    assert casting.mapping == {}
    bank_cast = assign_bank_voices(casting)

    assert bank_cast.mapping == {"Gus": bank_speaker_id("Gus")}
    assert bank_cast.bank_casts[0].gender == "masculine"
    assert "voice bank" in bank_cast.bank_casts[0].reason


def test_the_bank_never_takes_a_voice_the_film_supplied():
    """The actor is the identity: a character with usable film audio keeps it.

    The bank exists only for characters the film cannot voice - never as a
    replacement, and never as an ambiguity tie-breaker.
    """

    from app.pipeline.dialogue_plan import assign_bank_voices

    plan = plan_from_payload(
        {
            "characters": [
                {"name": "Walter", "gender": "masculine"},
                {"name": "Gus", "gender": "masculine"},
            ],
            "dialogue": [
                {"speaker": "Walter", "start": 10.0, "end": 14.0, "amharic": "x"},
                {"speaker": "Gus", "start": 200.0, "end": 202.0, "amharic": "y"},
            ],
        }
    )
    casting = cast_plan(plan, plan.cues, turns=_turns(), registers=_registers())
    assert casting.mapping["Walter"] is not None

    bank_cast = assign_bank_voices(casting)

    assert bank_cast.mapping["Walter"] == casting.mapping["Walter"]
    assert [cast.character for cast in bank_cast.bank_casts] == ["Gus"]
    assert all(cast.source == "film" for cast in bank_cast.casts
               if cast.character == "Walter")


def test_a_bank_cast_character_still_voices_its_lines():
    from app.pipeline.dialogue_plan import assign_bank_voices, bank_speaker_id

    plan = plan_from_payload(
        {
            "characters": [{"name": "Gus", "gender": "masculine"}],
            "dialogue": [
                {"speaker": "Gus", "start": 10.0, "end": 12.0, "amharic": "ሰላም።"}
            ],
        }
    )
    scope = scope_plan(plan)
    casting = assign_bank_voices(
        cast_plan(plan, scope.cues, turns=_turns(), registers=_registers())
    )

    dialogue = plan_dialogue(plan, scope, casting)

    assert len(dialogue) == 1
    assert dialogue[0].speaker_id == bank_speaker_id("Gus")
    assert dropped_cues(scope, casting) == 0
