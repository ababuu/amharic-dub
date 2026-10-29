"""Tests for the delivery report and the plan-first corrections it measures.

The previous full-episode run failed in ways no stage reported: 380 of 571 lines were
dragged to the tempo floor, two thirds of the timeline was silent, and 501 of 595 plan
lines were given invented times. Every one of those is arithmetic on the aligned clips,
so every one of them is now measured and stated.
"""

from __future__ import annotations

import types
from pathlib import Path
from typing import Any

import pytest

from app.pipeline import dialogue_plan, timing
from app.pipeline.orchestrator import _report_timing_health, timing_health


def _clip(
    index: int,
    *,
    start: float,
    window: float,
    speech: float,
    tempo: float = 1.0,
    overrun: float = 0.0,
) -> Any:
    """Return a stand-in carrying only the fields the health report reads.

    ``AlignedClip`` validates that it wraps a real ``TtsClip``, which this measurement
    does not need; the report is arithmetic over an aligned line's own numbers.
    """

    return types.SimpleNamespace(
        index=index,
        start=start,
        tempo=tempo,
        required_tempo=tempo,
        speech_duration=speech,
        original_window=window,
        overrun=overrun,
        fits=abs(speech - window) <= 0.03,
    )


# ---------------------------------------------------------------------------
# The delivery report
# ---------------------------------------------------------------------------


def test_a_healthy_run_reports_natural_pace_and_high_occupancy() -> None:
    clips = [_clip(i, start=i * 3.0, window=2.5, speech=2.5) for i in range(10)]

    health = timing_health(clips)
    messages: list[str] = []
    _report_timing_health(clips, messages.append)

    assert health["slowed"] == 0
    assert health["at_natural_pace"] == 10
    assert health["occupancy"] > 0.8
    assert any("natural pace" in message for message in messages)
    assert not any("WARNING" in message for message in messages)


def test_lines_dragged_to_the_floor_are_called_out() -> None:
    """The exact failure: most lines slowed, which no stage treats as an error."""

    clips = [_clip(i, start=i * 6.0, window=6.0, speech=4.8, tempo=0.8) for i in range(10)]

    health = timing_health(clips)
    messages: list[str] = []
    _report_timing_health(clips, messages.append)

    assert health["slowed"] == 10
    assert health["median_tempo"] == pytest.approx(0.8)
    assert any("slower than the actor performed them" in message for message in messages)


def test_a_mostly_silent_timeline_is_called_out() -> None:
    clips = [_clip(i, start=i * 10.0, window=1.0, speech=1.0) for i in range(60)]

    health = timing_health(clips)
    messages: list[str] = []
    _report_timing_health(clips, messages.append)

    assert health["occupancy"] < 0.2
    assert any("mostly silence" in message for message in messages)


def test_an_empty_alignment_reports_nothing() -> None:
    assert timing_health([]) == {"lines": 0}


# ---------------------------------------------------------------------------
# Room is not a gap
# ---------------------------------------------------------------------------


def test_the_room_a_line_may_borrow_is_bounded() -> None:
    """A gap to the next line is not room.

    An interpolated line can be a minute from its neighbours; believing that gap
    produced ``available: 140.4s`` against a 2.35s line, a required tempo of 0.017,
    and a dub that drifted away from the picture.
    """

    assert timing.MAX_ROOM_GROWTH_SECONDS == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Plan-first identity
# ---------------------------------------------------------------------------


def test_a_character_the_diarization_could_not_map_is_voiced_from_the_film() -> None:
    """Walter was handed a bank voice because casting could not see his lines.

    His own authored windows are where he spoke, so the reference is cut from those -
    which keeps the actor's voice - and the bank stays the fallback for a character
    whose windows hold no clean speech at all.
    """

    casting = dialogue_plan.PlanCasting(
        casts=(
            dialogue_plan.CharacterCast(
                character="Walter",
                speaker_id=None,
                gender="masculine",
                speaker_gender="feminine",
                share=0.31,
                seconds=75.5,
                runner_up="SPEAKER_03",
                reason="no single voice carries this character",
            ),
        )
    )

    resolved = dialogue_plan.assign_film_identities(casting)

    (cast,) = resolved.casts
    assert cast.speaker_id == dialogue_plan.film_speaker_id("Walter")
    assert cast.speaker_id.startswith("FILM_")
    assert cast.source == "film"
    assert "own authored lines" in cast.reason


def test_a_character_the_diarization_did_map_keeps_its_measured_voice() -> None:
    casting = dialogue_plan.PlanCasting(
        casts=(
            dialogue_plan.CharacterCast(
                character="Skyler",
                speaker_id="SPEAKER_14",
                gender="feminine",
                speaker_gender="feminine",
                share=0.72,
                seconds=12.8,
                runner_up="SPEAKER_02",
                reason="72% of this character's lines are SPEAKER_14",
            ),
        )
    )

    assert dialogue_plan.assign_film_identities(casting) is casting


def test_the_film_identity_is_stable_across_runs_and_names() -> None:
    assert dialogue_plan.film_speaker_id("Walter Jr.") == "FILM_WALTER_JR"
    assert dialogue_plan.film_speaker_id("Hank (TV)") == "FILM_HANK_TV"
    assert dialogue_plan.film_speaker_id("  ") == "FILM_CHARACTER"


def test_a_film_identity_never_collides_with_a_bank_identity() -> None:
    for name in ("Walter", "Skyler", "Hank (TV)", "Bully's Friends"):
        assert dialogue_plan.film_speaker_id(name) != dialogue_plan.bank_speaker_id(name)
