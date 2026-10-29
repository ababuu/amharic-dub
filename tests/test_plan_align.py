"""Tests for aligning an authored plan to the audio without transcribing it.

The point of this module is that a plan's timings can be checked against the dialogue
stem directly - the plan is a speech mask and the stem is where the speech is - so the
answer costs an FFT instead of a diarization and a transcription pass. Every test here
is a synthetic signal with a known offset, so the correct answer is known in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.pipeline import plan_align
from app.pipeline.dialogue_plan import PlanCue

RATE = 48_000
HOP = 0.05


def _cue(index: int, start: float, end: float, speaker: str = "Walter") -> PlanCue:
    return PlanCue(
        index=index,
        cue=str(index),
        speaker=speaker,
        start=start,
        end=end,
        source_text="line",
        amharic="መስመር",
    )


def _speech_then_silence(
    spans: list[tuple[float, float]], *, duration: float, rate: int = RATE
) -> np.ndarray:
    """Return noise inside ``spans`` and near-silence outside them."""

    samples = np.zeros(int(round(duration * rate)), dtype=np.float64)
    rng = np.random.default_rng(7)
    for start, end in spans:
        first = int(round(start * rate))
        last = int(round(end * rate))
        samples[first:last] = rng.normal(0.0, 0.12, last - first)
    # A little room tone everywhere, so the envelope has a floor to sit above.
    samples += rng.normal(0.0, 0.0005, samples.size)
    return samples


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_the_envelope_is_one_when_speech_is_present_and_zero_when_it_is_not() -> None:
    audio = _speech_then_silence([(2.0, 4.0)], duration=10.0)

    envelope, hop = plan_align.speech_envelope(audio, sample_rate=RATE)

    assert hop == pytest.approx(HOP)
    inside = envelope[int(2.5 / hop) : int(3.5 / hop)]
    outside = envelope[int(6.0 / hop) : int(9.0 / hop)]
    assert inside.mean() > 0.5
    assert outside.mean() < 0.05


def test_an_empty_stem_is_reported_rather_than_scored() -> None:
    with pytest.raises(plan_align.PlanAlignError, match="empty"):
        plan_align.speech_envelope(np.zeros(0), sample_rate=RATE)


def test_an_in_memory_envelope_needs_a_rate() -> None:
    with pytest.raises(plan_align.PlanAlignError, match="sample rate"):
        plan_align.speech_envelope(np.zeros(4800))


# ---------------------------------------------------------------------------
# Finding the offset
# ---------------------------------------------------------------------------


def test_a_plan_that_already_fits_is_not_moved() -> None:
    spans = [(2.0, 3.0), (5.0, 6.5), (9.0, 10.0)]
    audio = _speech_then_silence(spans, duration=14.0)
    cues = [_cue(i, start, end) for i, (start, end) in enumerate(spans)]

    envelope, hop = plan_align.speech_envelope(audio, sample_rate=RATE)
    found = plan_align.estimate_offset(envelope, hop, cues)

    assert found.applied is False
    assert found.offset == 0.0
    assert "already sits" in found.reason


def test_a_late_plan_is_shifted_back_onto_the_speech() -> None:
    """The case the transcript used to catch, at no model cost."""

    spans = [(2.0, 3.0), (5.0, 6.5), (9.0, 10.0), (12.0, 13.0)]
    audio = _speech_then_silence(spans, duration=20.0)
    # The plan is authored 7 seconds late.
    cues = [_cue(i, start + 7.0, end + 7.0) for i, (start, end) in enumerate(spans)]

    envelope, hop = plan_align.speech_envelope(audio, sample_rate=RATE)
    found = plan_align.estimate_offset(envelope, hop, cues)

    assert found.applied is True
    assert found.offset == pytest.approx(-7.0, abs=0.11)
    assert found.score > found.baseline


def test_an_early_plan_is_shifted_forward() -> None:
    spans = [(6.0, 7.0), (9.0, 10.0), (13.0, 14.0)]
    audio = _speech_then_silence(spans, duration=20.0)
    cues = [_cue(i, start - 4.0, end - 4.0) for i, (start, end) in enumerate(spans)]

    envelope, hop = plan_align.speech_envelope(audio, sample_rate=RATE)
    found = plan_align.estimate_offset(envelope, hop, cues)

    assert found.applied is True
    assert found.offset == pytest.approx(4.0, abs=0.11)


def test_a_plan_carrying_no_timing_information_is_refused() -> None:
    """A flat profile means nothing can be concluded, so nothing may be done.

    This is the guard against the failure that started all this: inventing evidence
    rather than admitting there is none. A plan whose mask covers the whole audio has
    the same score at every lag, so there is no shift to prefer.
    """

    audio = _speech_then_silence([(2.0, 4.0)], duration=12.0)
    # One cue covering everything: the mask is all ones, so every lag scores alike.
    cues = [_cue(0, 0.0, 12.0)]

    envelope, hop = plan_align.speech_envelope(audio, sample_rate=RATE)
    found = plan_align.estimate_offset(envelope, hop, cues)

    assert found.applied is False
    assert found.offset == 0.0
    assert "too weak" in found.reason or "already sits" in found.reason


def test_an_empty_plan_is_not_an_error() -> None:
    envelope, hop = plan_align.speech_envelope(
        _speech_then_silence([(1.0, 2.0)], duration=6.0), sample_rate=RATE
    )

    found = plan_align.estimate_offset(envelope, hop, [])

    assert found.applied is False
    assert "no cues" in found.reason


def test_the_offset_search_is_bounded() -> None:
    """A limit must be honoured even when the true shift is outside it."""

    spans = [(2.0, 3.0), (5.0, 6.0), (8.0, 9.0)]
    audio = _speech_then_silence(spans, duration=12.0)
    cues = [_cue(i, start + 4.0, end + 4.0) for i, (start, end) in enumerate(spans)]

    envelope, hop = plan_align.speech_envelope(audio, sample_rate=RATE)
    found = plan_align.estimate_offset(
        envelope, hop, cues, max_offset_seconds=1.0
    )

    assert abs(found.offset) <= 1.0 + hop


# ---------------------------------------------------------------------------
# Snapping one line onto its own speech
# ---------------------------------------------------------------------------


def test_a_silent_cue_is_moved_onto_its_own_speech() -> None:
    audio = _speech_then_silence([(2.0, 3.0), (8.0, 9.5), (12.0, 13.0)], duration=16.0)
    # The middle line is authored in the silence just before its own speech.
    cues = [_cue(0, 2.0, 3.0), _cue(1, 6.5, 8.0), _cue(2, 12.0, 13.0)]

    envelope, hop = plan_align.speech_envelope(audio, sample_rate=RATE)
    snapped, moved = plan_align.snap_cues(cues, envelope, hop)

    assert moved == 1
    assert snapped[1].start == pytest.approx(8.0, abs=hop * 2)
    # The lines that were already on their speech are untouched.
    assert snapped[0].start == cues[0].start
    assert snapped[2].start == cues[2].start


def test_a_cue_is_never_snapped_onto_another_cue() -> None:
    """Two characters sharing one window is what the mixer cannot undo."""

    audio = _speech_then_silence([(2.0, 3.0), (5.0, 6.0)], duration=10.0)
    cues = [_cue(0, 2.0, 3.0), _cue(1, 4.6, 5.4)]

    envelope, hop = plan_align.speech_envelope(audio, sample_rate=RATE)
    snapped, _moved = plan_align.snap_cues(cues, envelope, hop, radius_seconds=1.5)

    for left, right in zip(snapped, snapped[1:]):
        assert left.end <= right.start + 1e-6


def test_alignment_reports_what_it_did() -> None:
    spans = [(3.0, 4.0), (6.0, 7.5), (10.0, 11.0)]
    audio = _speech_then_silence(spans, duration=16.0)
    cues = [_cue(i, start + 5.0, end + 5.0) for i, (start, end) in enumerate(spans)]

    aligned, report = plan_align.align_plan(cues, audio, sample_rate=RATE)

    assert report.applied is True
    assert report.offset == pytest.approx(-5.0, abs=0.11)
    assert report.placed > 0.5
    payload = report.as_dict()
    assert set(payload) == {
        "offset_seconds",
        "applied",
        "score",
        "baseline",
        "snapped",
        "placed",
        "reason",
    }
    # The line text and speakers are never touched by alignment.
    assert [cue.amharic for cue in aligned] == [cue.amharic for cue in cues]
    assert [cue.speaker for cue in aligned] == [cue.speaker for cue in cues]


def test_words_are_never_changed_by_alignment() -> None:
    audio = _speech_then_silence([(1.0, 2.0)], duration=8.0)
    cues = [_cue(0, 9.0, 10.0)]

    aligned, _report = plan_align.align_plan(cues, audio, sample_rate=RATE)

    (cue,) = aligned
    assert cue.source_text == cues[0].source_text
    assert cue.amharic == cues[0].amharic
    assert cue.index == cues[0].index
