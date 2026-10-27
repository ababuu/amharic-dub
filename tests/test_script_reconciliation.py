"""Tests for :mod:`app.pipeline.script_reconciliation`.

The fixture is the real shape of the failure: a long verified Skyler turn that
contains two short Walt lines. The script has to recover those two lines without
moving anything the audio genuinely proved.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.pipeline.dialogue_context import CharacterGuide, GuideCharacter
from app.pipeline.script_reconciliation import (
    EpisodeScript,
    ScriptLine,
    reconcile_speakers,
)
from app.pipeline.transcription import TranscriptSegment

SKYLER = "SPEAKER_00"
WALT = "SPEAKER_01"

SCRIPT_LINES = (
    ("Skyler", "Walt. I've said it before. If you are in danger, we go to the police."),
    ("Walt", "Oh, no. I don't want to hear about the police."),
    (
        "Skyler",
        "I do not say that lightly. I know what it could do to this family. But if "
        "it's the only real choice we have... If it's either that or your getting "
        "shot when you open your front door...",
    ),
    ("Walt", "I don't want to hear about the police."),
    (
        "Skyler",
        "You're not some hardened criminal, Walt. You're in over your head. That's "
        "what we tell them. That's the truth.",
    ),
    ("Walt", "That is not the truth."),
    ("Skyler", "Of course it is. A schoolteacher, cancer, desperate for money?"),
    ("Walt", "Okay, we're done here."),
    (
        "Skyler",
        "Roped into working for... Unable to even quit? You told me that yourself, "
        "Walt. Jesus, what was I thinking? Walt, please. Let's both of us stop trying "
        "to justify this whole thing and admit you're in danger.",
    ),
    (
        "Walt",
        "Who are you talking to right now? Who is it you think you see? Do you know "
        "how much I make a year? I mean, even if I told you, you wouldn't believe it. "
        "Do you know what would happen if I suddenly decided to stop going in to work? "
        "A business big enough that it could be listed on the NASDAQ goes belly-up, "
        "disappears. It ceases to exist without me. No. You clearly don't know who "
        "you're talking to, so let me clue you in. I am not in danger, Skyler. I am "
        "the danger. A guy opens his door and gets shot, and you think that of me? "
        "No. I am the one who knocks.",
    ),
)


def _script() -> EpisodeScript:
    return EpisodeScript(
        title="Breaking Bad",
        season=4,
        episode=6,
        lines=tuple(ScriptLine(speaker, text) for speaker, text in SCRIPT_LINES),
        sources=("https://example.com/transcript",),
    )


def _guide() -> CharacterGuide:
    return CharacterGuide(
        title="Breaking Bad",
        season=4,
        episode=6,
        characters=(
            GuideCharacter(
                name="Walt", gender="masculine", sources=("https://example.com",)
            ),
            GuideCharacter(
                name="Skyler", gender="feminine", sources=("https://example.com",)
            ),
        ),
    )


def _registers() -> dict[str, object]:
    return {
        SKYLER: SimpleNamespace(median_hz=200.0, presentation="feminine"),
        WALT: SimpleNamespace(median_hz=120.0, presentation="masculine"),
    }


def _lines() -> list[TranscriptSegment]:
    return [
        TranscriptSegment(SKYLER, 7.52, 8.24, "I've said it before."),
        TranscriptSegment(SKYLER, 8.91, 10.51, "if you are in danger."),
        TranscriptSegment(SKYLER, 11.19, 11.93, "We go to the police."),
        TranscriptSegment(WALT, 12.82, 13.86, "I don't want to hear about"),
        TranscriptSegment(WALT, 14.14, 14.38, "the police."),
        TranscriptSegment(SKYLER, 14.14, 16.56, "I do not say that lightly."),
        TranscriptSegment(
            SKYLER,
            18.00,
            22.56,
            "I know what it could do to this family, but if it's the only real choice we have,",
        ),
        TranscriptSegment(
            SKYLER,
            22.64,
            26.86,
            "if it's either that or you getting shot when you open your front door...",
        ),
        # The bleed repeat, mislabelled as Skyler.
        TranscriptSegment(
            SKYLER, 26.86, 28.06, "I don't want to hear about the police."
        ),
        TranscriptSegment(SKYLER, 28.06, 30.46, "You're not some hardened criminal, Walt."),
        TranscriptSegment(SKYLER, 30.58, 32.44, "You are in over your head."),
        TranscriptSegment(SKYLER, 32.48, 33.66, "That's what we tell them."),
        TranscriptSegment(SKYLER, 33.66, 34.60, "That's the truth."),
        # Walt's interjection, inside the long verified Skyler turn.
        TranscriptSegment(SKYLER, 34.68, 35.38, "No, it's not the truth."),
        TranscriptSegment(SKYLER, 35.54, 36.56, "Of course it is."),
        TranscriptSegment(SKYLER, 37.85, 40.61, "A school teacher? Cancer? Desperate for money?"),
        TranscriptSegment(WALT, 40.82, 41.10, "We're done."),
        TranscriptSegment(SKYLER, 41.21, 43.49, "roped into working for, unable to even quit?"),
    ]


def test_script_recovers_interjections_inside_a_long_verified_turn() -> None:
    long_turn = SimpleNamespace(start=18.00, end=37.00, speaker_id=SKYLER)

    lines, review = reconcile_speakers(
        _lines(),
        script=_script(),
        guide=_guide(),
        registers=_registers(),
        verified=(long_turn,),
    )

    assert lines[8].speaker_id == WALT
    assert lines[13].speaker_id == WALT
    assert {change.index for change in review.applied} == {8, 13}
    mapping = {item.character: item.speaker_id for item in review.mapping}
    assert mapping == {"Walt": WALT, "Skyler": SKYLER}
    assert review.matched_lines >= 15
    # Walt genuinely repeats this line in the script, so it is not bleed.
    assert not any("bleed" in item.reason for item in review.uncertain)


def test_an_unmatched_interjection_stays_put_and_is_reported() -> None:
    lines = [
        TranscriptSegment(WALT, 27.82, 28.22, "at least"),
        TranscriptSegment(SKYLER, 28.06, 30.46, "You're not some hardened criminal, Walt."),
    ]

    repaired, review = reconcile_speakers(
        lines, script=_script(), guide=_guide(), registers=_registers()
    )

    assert repaired[0].speaker_id == WALT
    assert any(item.index == 0 for item in review.uncertain)


def test_a_repeat_the_script_cannot_explain_is_flagged_as_bleed() -> None:
    script = EpisodeScript(
        title="Breaking Bad",
        lines=(
            ScriptLine("Skyler", "We go to the police."),
            ScriptLine("Walt", "I don't want to hear about the police."),
            ScriptLine("Skyler", "You're not some hardened criminal, Walt."),
        ),
    )
    lines = [
        TranscriptSegment(WALT, 0.0, 2.0, "I don't want to hear about the police."),
        TranscriptSegment(WALT, 10.0, 12.0, "I don't want to hear about the police."),
        TranscriptSegment(SKYLER, 13.0, 15.0, "You're not some hardened criminal, Walt."),
    ]

    _, review = reconcile_speakers(
        lines, script=script, guide=_guide(), registers=_registers()
    )

    assert any("bleed" in item.reason for item in review.uncertain)


def test_a_line_verified_at_its_own_timescale_is_not_moved() -> None:
    line = TranscriptSegment(SKYLER, 92.69, 94.17, "I am the danger.")
    script = EpisodeScript(
        title="Breaking Bad",
        lines=(ScriptLine("Walt", "I am the danger."),),
        sources=("https://example.com/transcript",),
    )
    verified = (SimpleNamespace(start=92.69, end=94.17, speaker_id=SKYLER),)

    repaired, review = reconcile_speakers(
        [line], script=script, guide=_guide(), registers=_registers(), verified=verified
    )

    assert repaired[0].speaker_id == SKYLER
    assert review.applied == ()
    assert "confirmed by speaker embeddings" in review.uncertain[0].reason


def test_a_character_that_cannot_be_mapped_is_left_for_qc() -> None:
    line = TranscriptSegment(WALT, 0.0, 1.0, "Yeah, science.")
    script = EpisodeScript(
        title="Breaking Bad",
        lines=(ScriptLine("Jesse", "Yeah, science."),),
    )

    repaired, review = reconcile_speakers(
        [line], script=script, guide=_guide(), registers=_registers()
    )

    assert repaired[0].speaker_id == WALT
    assert review.applied == ()
    assert "could not be tied to a diarized voice" in review.uncertain[0].reason


def test_a_missing_script_is_reported_not_guessed() -> None:
    lines = _lines()

    repaired, review = reconcile_speakers(lines, script=None, guide=_guide())

    assert repaired == tuple(lines)
    assert review.applied == ()
    assert "no episode script" in review.skipped


def test_script_round_trips_through_the_cache(tmp_path: Path) -> None:
    path = _script().save(tmp_path / "episode_script.json")

    loaded = EpisodeScript.load(path)

    assert loaded is not None
    assert len(loaded) == len(SCRIPT_LINES)
    assert loaded.lines[0].speaker == "Skyler"
    assert loaded.sources == ("https://example.com/transcript",)
