"""Tests for :mod:`app.pipeline.qc`.

No model is imported, downloaded or run here: the pronunciation measurement takes
its transcriber as an argument, so a lambda stands in for an Amharic ASR model and
the rest of the report is arithmetic over metadata the stages already produced.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline import qc
from app.pipeline.diarization import CrosstalkRegion
from app.pipeline.qc import (
    CLOSE_FIT_RATIO,
    InvalidMeasurementError,
    InvalidTextError,
    character_error_rate,
    count_syllables,
    has_latin,
    has_pronounceable_text,
    latin_spans,
    measure_pronunciation,
    normalise_for_comparison,
)
from app.pipeline.timing import AlignedClip
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import PerformanceControls, TtsClip

SPEAKER = "SPEAKER_00"

#: "hello, how are you" - ten syllables.
AMHARIC = "ሰላም እንደምን ነህ"


def _dialogue(
    *,
    start: float = 0.0,
    end: float = 1.0,
    amharic: str = AMHARIC,
    speaker_id: str = SPEAKER,
) -> AdaptedDialogue:
    return AdaptedDialogue(
        speaker_id=speaker_id,
        start=start,
        end=end,
        source_text="Hello, how are you?",
        amharic=amharic,
        emotion="neutral",
        intensity=0.5,
        delivery="calm and conversational",
        pause_before=0.0,
        pause_after=0.0,
    )


def _clip(root: Path, *, index: int = 0, dialogue: AdaptedDialogue | None = None) -> TtsClip:
    line = dialogue if dialogue is not None else _dialogue()
    path = root / f"{SPEAKER}_{index}.wav"
    path.write_bytes(b"")
    return TtsClip(
        index=index,
        dialogue=line,
        performance=PerformanceControls.from_dialogue(line, seed=index + 1),
        audio_path=path,
        take_path=path,
        performance_reference_path=path,
        voice_reference_path=path,
        sample_rate=24_000,
        speech_duration=1.0,
        rendered_pause_before=0.0,
        rendered_pause_after=0.0,
        performance_engine="chatterbox-amharic",
        style_engine="seed-vc-v2",
    )


def _aligned(
    root: Path,
    *,
    index: int = 0,
    window: float = 2.0,
    speech: float = 2.0,
    tempo: float = 1.0,
    required_tempo: float = 1.0,
    dialogue: AdaptedDialogue | None = None,
    notes: tuple[str, ...] = (),
) -> AlignedClip:
    line = dialogue if dialogue is not None else _dialogue(end=window)
    return AlignedClip(
        index=index,
        clip=_clip(root, index=index, dialogue=line),
        audio_path=root / f"aligned_{index}.wav",
        sample_rate=24_000,
        start=line.start,
        tempo=tempo,
        required_tempo=required_tempo,
        speech_duration=speech,
        original_window=window,
        rendered_pause_before=0.0,
        rendered_pause_after=0.0,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# syllables
# ---------------------------------------------------------------------------


def test_one_fidel_character_is_one_syllable() -> None:
    assert count_syllables("ሰላም") == 3
    assert count_syllables(AMHARIC) == 10
    assert count_syllables("") == 0


def test_punctuation_and_digits_are_not_syllables() -> None:
    """Ethiopic punctuation and numerals sit inside the block but are not spoken."""

    assert count_syllables("ሰላም።") == 3  # U+1362 full stop
    assert count_syllables("ሰላም፣") == 3  # U+1361 comma
    assert count_syllables("፩፪፫") == 0  # U+1369.. Ethiopic digits
    assert count_syllables("ሰላም 123 abc!") == 3


def test_combining_marks_are_not_counted_as_syllables() -> None:
    """U+135D..U+135F modify a syllable; they do not add one."""

    assert count_syllables("ሰ\u135dላም") == 3


def test_syllable_counting_rejects_non_text() -> None:
    with pytest.raises(InvalidTextError, match="needs text"):
        count_syllables(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Roman script
# ---------------------------------------------------------------------------


def test_latin_text_is_detected() -> None:
    assert has_latin("Walt") is True
    assert has_latin("ዋልት ኮምፒውተር") is False
    assert has_latin("ዋልት ገዛ iPhone") is True


def test_latin_spans_are_extracted_in_order() -> None:
    assert latin_spans("Walt ኮምፒውተር ገዛ") == ("Walt",)
    assert latin_spans("ዋልት iPhone እና Coca-Cola አየ") == ("iPhone", "Coca-Cola")


def test_latin_spans_keep_digits_and_internal_punctuation() -> None:
    """Real names and product names contain them."""

    assert latin_spans("iPhone15 ነው") == ("iPhone15",)
    assert latin_spans("O'Brien መጣ") == ("O'Brien",)
    assert latin_spans("Wi-Fi የለም") == ("Wi-Fi",)


def test_whitespace_ends_a_span() -> None:
    """A separate number is not part of the word, and a bare number is not a word."""

    assert latin_spans("iPhone 15 ነው") == ("iPhone",)
    assert latin_spans("15 ነው") == ()


def test_latin_spans_are_deduplicated() -> None:
    assert latin_spans("Walt እና Walt") == ("Walt",)


def test_latin_spans_of_fidel_text_are_empty() -> None:
    assert latin_spans("ዋልት ኮምፒውተር ገዛ") == ()
    assert latin_spans("") == ()


def test_a_trailing_separator_is_not_part_of_a_span() -> None:
    assert latin_spans("Walt።") == ("Walt",)
    assert latin_spans("Walt-") == ("Walt",)


def test_a_borrowed_word_counts_as_pronounceable() -> None:
    """An English-derived word is a word; only punctuation has nothing to say."""

    assert has_pronounceable_text("Walt") is True
    assert has_pronounceable_text("ዋልት") is True
    assert has_pronounceable_text("ዋልት iPhone") is True
    assert has_pronounceable_text("።፣?!") is False
    assert has_pronounceable_text("   ") is False
    assert has_pronounceable_text("") is False


def test_a_latin_check_rejects_non_text() -> None:
    with pytest.raises(InvalidTextError, match="needs text"):
        has_latin(None)  # type: ignore[arg-type]
    with pytest.raises(InvalidTextError, match="needs text"):
        latin_spans(None)  # type: ignore[arg-type]
    with pytest.raises(InvalidTextError, match="needs text"):
        has_pronounceable_text(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# comparison normalisation and CER
# ---------------------------------------------------------------------------


def test_homophones_are_folded_so_spelling_is_not_scored() -> None:
    assert normalise_for_comparison("ሠላም") == normalise_for_comparison("ሰላም")
    assert normalise_for_comparison("አለ") == normalise_for_comparison("ዐለ")
    assert normalise_for_comparison("ጸሀይ") == normalise_for_comparison("ፀሐይ")


def test_normalisation_drops_everything_unspoken() -> None:
    assert normalise_for_comparison("ሰላም፣ ልዑል! ok 42") == "ሰላምልዑል"


def test_an_identical_transcript_has_no_error() -> None:
    assert character_error_rate(AMHARIC, AMHARIC) == 0.0


def test_a_homophone_difference_is_not_an_error() -> None:
    assert character_error_rate("ሰላም", "ሠላም") == 0.0


def test_one_missing_syllable_costs_one_over_the_length() -> None:
    assert character_error_rate("ሰላም", "ሰላ") == pytest.approx(1 / 3)


def test_a_silent_transcriber_scores_the_worst_rate() -> None:
    assert character_error_rate("ሰላም", "") == 1.0
    assert character_error_rate("ሰላም", "፣።") == 1.0


def test_a_wrong_transcript_is_penalised_proportionally() -> None:
    assert character_error_rate("ሰላም", "ሰላምሰላም") == pytest.approx(1.0)
    assert character_error_rate("ሰላም", "ላም") == pytest.approx(1 / 3)


def test_an_error_rate_needs_expected_text() -> None:
    with pytest.raises(InvalidMeasurementError, match="at least one syllable"):
        character_error_rate("", "ሰላም")
    with pytest.raises(InvalidMeasurementError, match="at least one syllable"):
        character_error_rate("hello", "ሰላም")


def test_latin_text_is_not_mistaken_for_amharic() -> None:
    """A transcriber that returned English must not look like a good transcript."""

    assert character_error_rate(AMHARIC, "hello how are you") == 1.0


# ---------------------------------------------------------------------------
# duration fit
# ---------------------------------------------------------------------------


def test_an_exact_fit_is_a_fitted_line(tmp_path: Path) -> None:
    (metrics,) = qc.measure_lines([_aligned(tmp_path)])

    assert metrics.fits is True
    assert metrics.close_fit is True
    assert metrics.residual == 0.0
    assert metrics.residual_ratio == 0.0
    assert metrics.tempo == 1.0


def test_a_line_inside_the_close_fit_tolerance_counts_as_close(tmp_path: Path) -> None:
    (metrics,) = qc.measure_lines([_aligned(tmp_path, window=2.0, speech=2.1)])

    assert metrics.fits is False  # not exact
    assert metrics.close_fit is True  # but within ten percent
    assert metrics.residual == pytest.approx(0.1)
    assert metrics.residual_ratio == pytest.approx(0.05)


def test_a_line_outside_the_tolerance_is_reported_as_not_close(tmp_path: Path) -> None:
    (metrics,) = qc.measure_lines([_aligned(tmp_path, window=2.0, speech=3.0)])

    assert metrics.close_fit is False
    assert metrics.residual_ratio == pytest.approx(0.5)


def test_a_line_that_underruns_is_measured_in_the_same_way(tmp_path: Path) -> None:
    (metrics,) = qc.measure_lines([_aligned(tmp_path, window=2.0, speech=1.0)])

    assert metrics.residual == pytest.approx(-1.0)
    assert metrics.residual_ratio == pytest.approx(-0.5)


def test_the_summary_reports_the_distribution_not_an_average(tmp_path: Path) -> None:
    """One very bad line must be visible, not averaged away by good ones."""

    lines = qc.measure_lines(
        [
            _aligned(tmp_path, index=0, window=2.0, speech=2.0),
            _aligned(tmp_path, index=1, window=2.0, speech=2.0),
            _aligned(tmp_path, index=2, window=2.0, speech=6.0),
        ]
    )

    report = qc.summarise_duration(lines)

    assert report.lines == 3
    assert report.fitted == 2
    assert report.close_fits == 2
    assert report.unfitted == 1
    assert report.close_fit_ratio == pytest.approx(2 / 3)
    assert report.worst_residual == pytest.approx(4.0)
    assert report.mean_absolute_residual == pytest.approx(4 / 3)


def test_tempo_extremes_are_reported(tmp_path: Path) -> None:
    lines = qc.measure_lines(
        [
            _aligned(tmp_path, index=0, tempo=0.8, required_tempo=0.8),
            _aligned(tmp_path, index=1, tempo=1.25, required_tempo=1.25),
        ]
    )

    report = qc.summarise_duration(lines)

    assert report.stretched == 2
    assert report.slowest_tempo == pytest.approx(0.8)
    assert report.fastest_tempo == pytest.approx(1.25)
    assert report.mean_tempo == pytest.approx(1.025)


def test_delivery_rate_is_measured_per_line(tmp_path: Path) -> None:
    """Ten syllables delivered in two seconds is five syllables per second."""

    (metrics,) = qc.measure_lines([_aligned(tmp_path, speech=2.0)])

    assert metrics.syllables == 10
    assert metrics.syllables_per_second == pytest.approx(5.0)
    assert metrics.plausible_rate is True


def test_an_impossible_delivery_rate_is_flagged(tmp_path: Path) -> None:
    """Ten syllables in half a second is not something a performer can deliver."""

    (metrics,) = qc.measure_lines([_aligned(tmp_path, window=0.5, speech=0.5)])

    assert metrics.syllables_per_second == pytest.approx(20.0)
    assert metrics.plausible_rate is False


def test_a_drawn_out_delivery_is_flagged(tmp_path: Path) -> None:
    (metrics,) = qc.measure_lines([_aligned(tmp_path, window=20.0, speech=20.0)])

    assert metrics.syllables_per_second == pytest.approx(0.5)
    assert metrics.plausible_rate is False


def test_implausible_lines_are_listed(tmp_path: Path) -> None:
    report = qc.build_qc_report(
        [
            _aligned(tmp_path, index=0, window=2.0, speech=2.0),
            _aligned(tmp_path, index=1, window=0.4, speech=0.4),
        ]
    )

    assert report.duration.implausible_rate == 1
    assert [line.index for line in report.implausible_lines] == [1]


def test_worst_lines_are_ordered_by_absolute_residual(tmp_path: Path) -> None:
    report = qc.build_qc_report(
        [
            _aligned(tmp_path, index=0, window=2.0, speech=2.0),
            _aligned(tmp_path, index=1, window=2.0, speech=1.0),
            _aligned(tmp_path, index=2, window=2.0, speech=5.0),
        ]
    )

    assert [line.index for line in report.worst_lines] == [2, 1, 0]


def test_an_empty_alignment_reports_zeroes_rather_than_failing() -> None:
    report = qc.build_qc_report([])

    assert report.duration.lines == 0
    assert report.duration.close_fit_ratio == 0.0
    assert report.lines == ()
    assert report.pronunciation is None


def test_a_line_at_the_start_of_the_film_keeps_its_notes(tmp_path: Path) -> None:
    (metrics,) = qc.measure_lines(
        [_aligned(tmp_path, notes=("the leading pause was trimmed",))]
    )

    assert metrics.notes == ("the leading pause was trimmed",)


# ---------------------------------------------------------------------------
# crosstalk and the assembled report
# ---------------------------------------------------------------------------


def test_crosstalk_is_carried_into_the_report(tmp_path: Path) -> None:
    report = qc.build_qc_report(
        [_aligned(tmp_path)],
        crosstalk=[
            CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), start=0.0, end=1.5),
            CrosstalkRegion(("SPEAKER_01", "SPEAKER_02"), start=4.0, end=4.5),
        ],
    )

    assert report.crosstalk_regions == 2
    assert report.crosstalk_seconds == pytest.approx(2.0)


def test_a_line_inside_simultaneous_speech_is_named(tmp_path: Path) -> None:
    """Attribution is temporal, so an overlapped line is a claim nobody checked.

    Each diarized turn is transcribed on its own and everything it contains is credited
    to that turn's speaker. Where two turns overlap, the exclusive view has already
    arbitrated, and nothing compares that decision with the actual voice. The run should
    say which lines are affected rather than presenting them as certain.
    """

    line = _aligned(tmp_path, index=3, dialogue=_dialogue(start=5.0, end=6.0))
    region = CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 5.2, 5.8)

    report = qc.build_qc_report([line], crosstalk=[region])

    assert report.attribution_uncertain == (3,)
    assert "cannot be verified" in report.summary()
    assert report.as_dict()["crosstalk"]["attribution_uncertain"] == [3]


def test_a_line_outside_simultaneous_speech_is_not_named(tmp_path: Path) -> None:
    line = _aligned(tmp_path, dialogue=_dialogue(start=5.0, end=6.0))
    region = CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 9.0, 9.5)

    report = qc.build_qc_report([line], crosstalk=[region])

    assert report.attribution_uncertain == ()
    assert "cannot be verified" not in report.summary()


def test_uncertain_attribution_is_reported_not_acted_on(tmp_path: Path) -> None:
    """The line is still voiced: a hole in the dialogue is worse than a doubtful voice."""

    line = _aligned(tmp_path, index=7, dialogue=_dialogue(start=5.0, end=6.0))
    region = CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), 5.0, 6.0)

    report = qc.build_qc_report([line], crosstalk=[region])

    assert report.duration.lines == 1
    assert report.lines[0].index == 7


def test_the_report_is_json_safe(tmp_path: Path) -> None:
    import json

    report = qc.build_qc_report([_aligned(tmp_path)], crosstalk=[])
    payload = report.as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["duration"]["lines"] == 1
    assert payload["crosstalk"] == {
        "regions": 0,
        "seconds": 0.0,
        "attribution_uncertain": [],
    }
    assert payload["lines"][0]["index"] == 0
    assert payload["pronunciation"] is None
    assert payload["duration"]["close_fit_tolerance"] == CLOSE_FIT_RATIO


def test_the_summary_says_what_was_and_was_not_measured(tmp_path: Path) -> None:
    summary = qc.build_qc_report([_aligned(tmp_path)]).summary()

    assert "1/1 line(s)" in summary
    assert "pronunciation not measured" in summary


# ---------------------------------------------------------------------------
# pronunciation (injected transcriber)
# ---------------------------------------------------------------------------


def test_pronunciation_is_measured_against_what_was_synthesized(tmp_path: Path) -> None:
    line = _dialogue()
    clips = [_clip(tmp_path, index=0, dialogue=line)]

    report = measure_pronunciation(
        [line], clips, transcribe=lambda path: "ሰላም እንደምን ነህ", model="fake-asr"
    )

    assert report.measured == 1
    assert report.unmeasured == 0
    assert report.mean_cer == 0.0
    assert report.model == "fake-asr"


def test_the_worst_line_is_identified(tmp_path: Path) -> None:
    lines = [_dialogue(), _dialogue(amharic="ሰላም እንደምን ነህ")]
    clips = [_clip(tmp_path, index=0), _clip(tmp_path, index=1)]

    def _transcribe(path: Path) -> str:
        return "ሰላም እንደምን ነህ" if path.name.endswith("_0.wav") else ""

    report = measure_pronunciation(lines, clips, transcribe=_transcribe)

    assert report.measured == 1
    assert report.unmeasured == 1
    assert report.worst_index == 0


def test_a_failing_transcriber_is_unmeasured_rather_than_an_error(tmp_path: Path) -> None:
    """A measurement must never fail a run."""

    def _boom(path: Path) -> str:
        raise RuntimeError("the ASR model is not installed")

    report = measure_pronunciation([_dialogue()], [_clip(tmp_path)], transcribe=_boom)

    assert report.measured == 0
    assert report.unmeasured == 1
    assert report.mean_cer == 0.0
    assert report.worst_index is None


def test_a_clip_without_a_matching_line_is_unmeasured(tmp_path: Path) -> None:
    report = measure_pronunciation(
        [_dialogue()],
        [_clip(tmp_path, index=0), _clip(tmp_path, index=7)],
        transcribe=lambda path: AMHARIC,
    )

    assert report.measured == 1
    assert report.unmeasured == 1


def test_a_transcriber_that_returns_the_wrong_type_is_rejected(tmp_path: Path) -> None:
    """A broken measurement is a bug, not a quiet zero."""

    with pytest.raises(InvalidMeasurementError, match="expected the transcript"):
        measure_pronunciation(
            [_dialogue()],
            [_clip(tmp_path)],
            transcribe=lambda path: 42,  # type: ignore[arg-type,return-value]
        )


def test_pronunciation_appears_in_the_assembled_report(tmp_path: Path) -> None:
    line = _dialogue()
    report = qc.build_qc_report(
        [_aligned(tmp_path, dialogue=line)],
        dialogue=[line],
        clips=[_clip(tmp_path, dialogue=line)],
        transcribe=lambda path: AMHARIC,
        transcription_model="fake-asr",
    )

    assert report.pronunciation is not None
    assert report.pronunciation.measured == 1
    assert "pronunciation CER 0.000" in report.summary()
    assert report.as_dict()["pronunciation"]["model"] == "fake-asr"
