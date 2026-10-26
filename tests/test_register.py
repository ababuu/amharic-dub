"""Tests for measuring who is speaking from the voice itself.

The defect these cover is specific: on a real two-person scene, diarization's
clusters were right (200 Hz and 120 Hz apart) but seven of forty-one turns sat on
the wrong side of that gap, and the character's reference recording was cut from
one of them. Walter got a female voice for the whole film because of it.

So the properties under test are: a register is measured from every turn rather
than one window; only a decisive contradiction moves a turn; and the material is
reported as unusable when it cannot support an automatic cast.
"""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from app.pipeline import register
from app.pipeline.transcription import TranscriptSegment

RATE = 24_000


def _voiced(seconds: float, hz: float, rate: int = RATE) -> np.ndarray:
    """Return a periodic signal with a clear fundamental at ``hz``."""

    time = np.arange(int(round(seconds * rate))) / rate
    # A few harmonics, so autocorrelation sees a real periodic signal rather than a
    # pure sine (which some estimators treat as ambiguous).
    signal = np.sin(2 * np.pi * hz * time)
    signal += 0.5 * np.sin(2 * np.pi * 2 * hz * time)
    signal += 0.25 * np.sin(2 * np.pi * 3 * hz * time)
    return (0.3 * signal).astype(np.float32)


def _dialogue(path, turns: list[tuple[str, float, float, float]]) -> list[TranscriptSegment]:
    """Write a stem containing one voice per turn and return matching segments.

    ``turns`` is ``(speaker_id, start, end, hz)``.
    """

    total = max(end for _sid, _start, end, _hz in turns) + 1.0
    samples = np.zeros(int(round(total * RATE)), dtype=np.float32)
    for _speaker_id, start, end, hz in turns:
        first, last = int(round(start * RATE)), int(round(end * RATE))
        tone = _voiced((last - first) / RATE, hz)
        samples[first:last] = tone[: last - first]
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, RATE, subtype="PCM_16")
    return [
        TranscriptSegment(speaker_id, start, end, "line")
        for speaker_id, start, end, _hz in turns
    ]


def test_presentation_reports_the_register_it_measures() -> None:
    assert register.presentation_of(None) == "unknown"
    assert register.presentation_of(120.0) == "masculine"
    assert register.presentation_of(200.0) == "feminine"
    # Between the bands is a real answer, not a guess.
    assert register.presentation_of(170.0) == "uncertain"


def test_register_aggregates_every_turn_not_one_window(tmp_path) -> None:
    """One mislabelled turn must not decide a character's voice."""

    segments = _dialogue(
        tmp_path / "speech.wav",
        [
            ("SPEAKER_00", 0.0, 1.0, 120.0),
            ("SPEAKER_00", 1.2, 2.2, 118.0),
            # A female-sounding turn inside the male cluster: exactly the defect.
            ("SPEAKER_00", 2.4, 3.4, 240.0),
            ("SPEAKER_00", 3.6, 4.6, 122.0),
        ],
    )

    _fixed, report = register.repair_attribution(segments, tmp_path / "speech.wav")

    measured = report.after["SPEAKER_00"]
    assert measured.median_hz == pytest.approx(120.0, abs=12.0)
    assert measured.presentation == "masculine"
    assert measured.measured_turns == 4


def test_a_turn_that_contradicts_its_cluster_is_reassigned(tmp_path) -> None:
    segments = _dialogue(
        tmp_path / "speech.wav",
        [
            ("SPEAKER_00", 0.0, 1.6, 120.0),
            ("SPEAKER_00", 1.7, 3.3, 118.0),
            ("SPEAKER_00", 3.4, 5.0, 122.0),
            ("SPEAKER_01", 5.1, 6.7, 205.0),
            ("SPEAKER_01", 6.8, 8.4, 210.0),
            # A male turn labelled with the female speaker.
            ("SPEAKER_01", 8.5, 10.1, 119.0),
            ("SPEAKER_01", 10.2, 11.8, 208.0),
        ],
    )

    fixed, report = register.repair_attribution(segments, tmp_path / "speech.wav")

    moved = [segment for segment in fixed if segment.speaker_id == "SPEAKER_00"]
    # The three original male turns plus the male one that was labelled female.
    assert len(moved) == 4
    assert any(segment.start == pytest.approx(8.5) for segment in moved)
    assert len(report.corrections) == 1
    correction = report.corrections[0]
    assert (correction.from_speaker, correction.to_speaker) == ("SPEAKER_01", "SPEAKER_00")
    assert report.after["SPEAKER_00"].presentation == "masculine"
    assert report.after["SPEAKER_01"].presentation == "feminine"


def test_a_short_turn_is_never_moved_by_pitch_alone(tmp_path) -> None:
    """A third of a second of audio is not a measurement of a voice.

    Autocorrelation octave-errors on a fragment, and a fragment cut from inside one
    person's sentence is not an utterance at all. On a real film this moved five
    pieces of a single monologue into the other character's voice.
    """

    segments = _dialogue(
        tmp_path / "speech.wav",
        [
            ("SPEAKER_00", 0.0, 1.6, 120.0),
            ("SPEAKER_00", 1.7, 3.3, 118.0),
            ("SPEAKER_01", 3.4, 5.0, 210.0),
            ("SPEAKER_01", 5.1, 6.7, 208.0),
            # Short, and its own pitch sits in the male register - but far too short
            # for that measurement to outrank the cluster it was diarized into.
            ("SPEAKER_01", 6.8, 7.2, 110.0),
            ("SPEAKER_01", 7.3, 8.9, 206.0),
        ],
    )

    fixed, report = register.repair_attribution(segments, tmp_path / "speech.wav")

    short = [segment for segment in fixed if segment.start == pytest.approx(6.8)]
    assert short[0].speaker_id == "SPEAKER_01"
    assert report.corrections == () or all(
        not (correction.start == pytest.approx(6.8))
        for correction in report.corrections
    )


def test_two_voices_too_alike_are_left_alone_and_reported(tmp_path) -> None:
    """A coin flip is not an improvement: equal registers must not be 'repaired'."""

    segments = _dialogue(
        tmp_path / "speech.wav",
        [
            ("SPEAKER_00", 0.0, 1.0, 130.0),
            ("SPEAKER_00", 1.1, 2.1, 132.0),
            ("SPEAKER_01", 2.2, 3.2, 128.0),
            ("SPEAKER_01", 3.3, 4.3, 131.0),
        ],
    )

    fixed, report = register.repair_attribution(segments, tmp_path / "speech.wav")

    assert [segment.speaker_id for segment in fixed] == [
        segment.speaker_id for segment in segments
    ]
    assert report.corrections == ()
    assert register.disagreement(report) is not None
    assert "too close" in (register.disagreement(report) or "")


def test_a_turn_between_two_clusters_is_not_moved(tmp_path) -> None:
    """Halfway is not evidence; only a decisive contradiction moves a turn."""

    segments = _dialogue(
        tmp_path / "speech.wav",
        [
            ("SPEAKER_00", 0.0, 1.0, 118.0),
            ("SPEAKER_00", 1.1, 2.1, 120.0),
            ("SPEAKER_00", 2.2, 3.2, 122.0),
            ("SPEAKER_01", 3.3, 4.3, 210.0),
            ("SPEAKER_01", 4.4, 5.4, 208.0),
            ("SPEAKER_01", 5.5, 6.5, 212.0),
            # 160 Hz: inside neither band, closer to nothing in particular.
            ("SPEAKER_01", 6.6, 7.6, 160.0),
        ],
    )

    fixed, _report = register.repair_attribution(segments, tmp_path / "speech.wav")

    assert fixed[6].speaker_id == "SPEAKER_01"


def test_an_unreadable_stem_is_an_error_not_a_silent_pass(tmp_path) -> None:
    """Losing this stage silently is how a character gets the wrong voice."""

    segments = [TranscriptSegment("SPEAKER_00", 0.0, 1.0, "line")]
    with pytest.raises(register.AudioReadError):
        register.repair_attribution(segments, tmp_path / "missing.wav")


def test_segments_must_be_transcript_segments(tmp_path) -> None:
    with pytest.raises(register.InvalidSegmentsError):
        register.repair_attribution(["not a segment"], tmp_path / "speech.wav")  # type: ignore[list-item]


def test_contiguous_turns_join_one_speaker_but_never_two(tmp_path) -> None:
    segments = [
        TranscriptSegment("SPEAKER_00", 0.0, 1.0, "a"),
        TranscriptSegment("SPEAKER_00", 1.2, 2.0, "b"),
        TranscriptSegment("SPEAKER_01", 2.1, 3.0, "c"),
        TranscriptSegment("SPEAKER_00", 5.0, 6.0, "d"),
    ]

    assert register.contiguous_turns(segments) == [
        ("SPEAKER_00", 0.0, 2.0),
        ("SPEAKER_01", 2.1, 3.0),
        ("SPEAKER_00", 5.0, 6.0),
    ]


def test_too_little_speech_is_reported_rather_than_guessed(tmp_path) -> None:
    segments = _dialogue(
        tmp_path / "speech.wav",
        [
            ("SPEAKER_00", 0.0, 0.15, 120.0),
            ("SPEAKER_01", 0.2, 0.35, 210.0),
        ],
    )

    _fixed, report = register.repair_attribution(segments, tmp_path / "speech.wav")

    assert register.disagreement(report) is not None


def test_a_confident_two_voice_scene_is_trusted(tmp_path) -> None:
    segments = _dialogue(
        tmp_path / "speech.wav",
        [
            ("SPEAKER_00", 0.0, 1.0, 120.0),
            ("SPEAKER_00", 1.1, 2.1, 118.0),
            ("SPEAKER_00", 2.2, 3.2, 122.0),
            ("SPEAKER_01", 3.3, 4.3, 205.0),
            ("SPEAKER_01", 4.4, 5.4, 210.0),
            ("SPEAKER_01", 5.5, 6.5, 208.0),
        ],
    )

    _fixed, report = register.repair_attribution(segments, tmp_path / "speech.wav")

    assert register.disagreement(report) is None
    assert report.after["SPEAKER_00"].presentation == "masculine"
    assert report.after["SPEAKER_01"].presentation == "feminine"
    assert report.reassigned_fraction == 0.0


# ---------------------------------------------------------------------------
# Short fragments take the speaker of the sentence they sit in
# ---------------------------------------------------------------------------


def _segments(rows: list[tuple[str, float, float]]) -> list[TranscriptSegment]:
    return [
        TranscriptSegment(speaker_id, start, end, "line")
        for speaker_id, start, end in rows
    ]


def test_a_fragment_inside_one_sentence_takes_that_speakers_label() -> None:
    """The defect, in miniature - taken from the real clip.

    One character talks through this stretch. Diarization labels a fragment of it as
    the other character, so the dub puts the other clone inside a sentence.
    """

    rows = [
        ("SPEAKER_01", 59.08, 60.30),
        ("SPEAKER_00", 63.72, 64.08),  # wrong: continues SPEAKER_01
        ("SPEAKER_01", 64.44, 65.12),
    ]

    fixed, corrections = register.smooth_short_turns(_segments(rows))

    assert [segment.speaker_id for segment in fixed] == ["SPEAKER_01"] * 3
    assert [correction.to_speaker for correction in corrections] == ["SPEAKER_01"]
    assert corrections[0].duration == pytest.approx(0.36)


def test_a_fragment_talked_over_keeps_its_label() -> None:
    """A real interjection is normally simultaneous, so overlap protects the label."""

    rows = [
        ("SPEAKER_01", 5.0, 7.0),
        ("SPEAKER_00", 7.1, 8.4),
        ("SPEAKER_01", 8.5, 10.5),
    ]

    fixed, corrections = register.smooth_short_turns(
        _segments(rows), overlaps=[(7.0, 8.6)]
    )

    assert fixed[1].speaker_id == "SPEAKER_00"
    assert corrections == ()


def test_a_short_reply_with_a_beat_either_side_keeps_its_label() -> None:
    """Not every short turn is a fragment: a separated reply is its own utterance."""

    rows = [
        ("SPEAKER_01", 5.0, 7.0),
        ("SPEAKER_00", 8.5, 9.0),  # 1.5s after one, 1.5s before the other
        ("SPEAKER_01", 10.5, 12.5),
    ]

    fixed, corrections = register.smooth_short_turns(_segments(rows))

    assert fixed[1].speaker_id == "SPEAKER_00"
    assert corrections == ()


def test_a_long_turn_is_left_for_the_register_to_judge() -> None:
    """Above the fragment threshold its own audio is worth measuring."""

    rows = [
        ("SPEAKER_01", 5.0, 7.0),
        ("SPEAKER_00", 7.1, 9.5),
        ("SPEAKER_01", 9.6, 11.6),
    ]

    fixed, corrections = register.smooth_short_turns(_segments(rows))

    assert fixed[1].speaker_id == "SPEAKER_00"
    assert corrections == ()


def test_a_run_of_fragments_cannot_relabel_itself() -> None:
    """Relabelling needs an anchor: one neighbour long enough to be trusted."""

    rows = [
        ("SPEAKER_01", 5.0, 6.0),
        ("SPEAKER_00", 6.1, 7.0),
        ("SPEAKER_00", 7.1, 8.0),
        ("SPEAKER_01", 8.1, 9.1),
    ]

    fixed, corrections = register.smooth_short_turns(_segments(rows))

    # Every turn in the middle is short, so none of them is evidence for the others.
    assert [segment.speaker_id for segment in fixed] == [
        "SPEAKER_01",
        "SPEAKER_00",
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert corrections == ()


def test_smoothing_leaves_a_normal_exchange_alone() -> None:
    """Nothing to do when the labels already agree with the shape of the dialogue."""

    rows = [
        ("SPEAKER_00", 0.0, 2.0),
        ("SPEAKER_01", 2.2, 4.0),
        ("SPEAKER_00", 4.2, 6.0),
    ]

    fixed, corrections = register.smooth_short_turns(_segments(rows))

    assert fixed == _segments(rows)
    assert corrections == ()


def test_pitch_does_not_overturn_a_label_the_embeddings_confirmed(tmp_path) -> None:
    """Rank the evidence: an embedding compares the voice, a pitch measures one axis.

    A trained actor's hushed line and shouted line can sit an octave apart. On a real
    film this is exactly how fragments of one character ended up voiced by the other
    character's clone: a long turn the diarization model had *confirmed* was moved by
    its own pitch median.
    """

    segments = _dialogue(
        tmp_path / "speech.wav",
        [
            ("SPEAKER_00", 0.0, 1.6, 120.0),
            ("SPEAKER_00", 1.7, 3.3, 118.0),
            ("SPEAKER_01", 3.4, 5.0, 210.0),
            ("SPEAKER_01", 5.1, 6.7, 208.0),
            # A long male-sounding turn labelled female - and confirmed as female.
            ("SPEAKER_01", 6.8, 8.8, 119.0),
            ("SPEAKER_01", 8.9, 10.5, 206.0),
        ],
    )

    confirmed = [(6.8, 8.8)]
    _fixed, report = register.repair_attribution(
        segments, tmp_path / "speech.wav", trusted=confirmed
    )

    assert report.corrections == ()

    # Without the confirmation the same turn is moved, because pitch alone says male.
    _fixed, report = register.repair_attribution(segments, tmp_path / "speech.wav")
    assert [correction.start for correction in report.corrections] == [
        pytest.approx(6.8)
    ]


def test_registers_are_measured_before_any_transcript_exists(tmp_path) -> None:
    """The transcript stage needs them to tell who is audible in a mixture.

    A crosstalk can only be heard by decoding a mixture, and the decode returns
    whichever voice dominates it - so the check has to be available *before* the
    transcript exists, which is why it is measured from the diarized turns.
    """

    turns = [
        ("SPEAKER_00", 0.0, 1.8, 120.0),
        ("SPEAKER_00", 1.9, 3.7, 118.0),
        ("SPEAKER_00", 3.8, 5.6, 122.0),
        ("SPEAKER_01", 5.7, 7.5, 205.0),
        ("SPEAKER_01", 7.6, 9.4, 210.0),
        ("SPEAKER_01", 9.5, 11.3, 208.0),
    ]
    stem = tmp_path / "speech.wav"
    _dialogue(stem, turns)

    from app.pipeline.diarization import SpeakerSegment

    measured = register.measure_registers(
        [SpeakerSegment(s, a, b) for s, a, b, _hz in turns], stem
    )

    assert measured["SPEAKER_00"] == pytest.approx(120.0, abs=12.0)
    assert measured["SPEAKER_01"] == pytest.approx(208.0, abs=15.0)


def test_a_short_turn_does_not_define_a_register(tmp_path) -> None:
    """Same floor as the repair pass: a fragment is not a measurement."""

    turns = [
        ("SPEAKER_00", 0.0, 1.8, 120.0),
        ("SPEAKER_00", 1.9, 3.7, 118.0),
        ("SPEAKER_01", 3.8, 4.1, 260.0),
    ]
    stem = tmp_path / "speech.wav"
    _dialogue(stem, turns)

    from app.pipeline.diarization import SpeakerSegment

    measured = register.measure_registers(
        [SpeakerSegment(s, a, b) for s, a, b, _hz in turns], stem
    )

    assert "SPEAKER_01" not in measured
