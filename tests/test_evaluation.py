"""Tests for :mod:`app.pipeline.evaluation`.

The measurements that need a model - speaker embeddings and the transcriber - are
injected as plain functions here, so nothing is downloaded and the expected answer
is known in advance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.pipeline import evaluation
from app.pipeline.diarization import CrosstalkRegion
from app.pipeline.evaluation import (
    GOLDEN_CATEGORIES,
    BaselineComparison,
    EvaluationError,
    InvalidMeasurementError,
    compare_to_baseline,
    cosine_similarity,
    load_baseline,
    measure_coverage,
    measure_performance_preservation,
    measure_speaker_identity,
    save_baseline,
)
from app.pipeline.timing import AlignedClip
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import PerformanceControls, TtsClip

SPEAKER = "SPEAKER_00"
OTHER = "SPEAKER_01"
RATE = 24_000


def _dialogue(
    *,
    speaker_id: str = SPEAKER,
    start: float = 0.0,
    end: float = 2.0,
    amharic: str = "ሰላም እንደምን ነህ",
    emotion: str = "neutral",
    delivery: str = "calm and conversational",
    intensity: float = 0.5,
) -> AdaptedDialogue:
    return AdaptedDialogue(
        speaker_id=speaker_id,
        start=start,
        end=end,
        source_text="Hello, how are you?",
        amharic=amharic,
        emotion=emotion,
        intensity=intensity,
        delivery=delivery,
        pause_before=0.0,
        pause_after=0.0,
    )


def _write(path: Path, samples: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, RATE, format="WAV", subtype="PCM_16")
    return path


def _tone(hz: float, seconds: float = 1.0) -> np.ndarray:
    times = np.arange(int(seconds * RATE), dtype=np.float64) / RATE
    return (0.4 * np.sin(2.0 * np.pi * hz * times)).astype(np.float32)


def _sweep(low: float, high: float, seconds: float = 1.0) -> np.ndarray:
    """A tone whose pitch moves, so it has a wide range by construction."""

    times = np.arange(int(seconds * RATE), dtype=np.float64) / RATE
    centre = (low + high) / 2.0
    depth = (high - low) / 2.0
    frequency = centre + depth * np.sin(2.0 * np.pi * 0.5 * times)
    return (0.4 * np.sin(2.0 * np.pi * np.cumsum(frequency) / RATE)).astype(np.float32)


def _clip(
    root: Path,
    *,
    index: int = 0,
    speaker_id: str = SPEAKER,
    dialogue: AdaptedDialogue | None = None,
    pitch: float = 140.0,
    reference_pitch: float | None = None,
) -> TtsClip:
    line = dialogue if dialogue is not None else _dialogue(speaker_id=speaker_id)
    audio = _write(root / f"clip_{speaker_id}_{index}.wav", _tone(pitch))
    reference = _write(
        root / f"ref_{speaker_id}_{index}.wav",
        _tone(reference_pitch if reference_pitch is not None else pitch),
    )
    return TtsClip(
        index=index,
        dialogue=line,
        performance=PerformanceControls.from_dialogue(line, seed=index + 1),
        audio_path=audio,
        take_path=audio,
        performance_reference_path=reference,
        voice_reference_path=reference,
        sample_rate=RATE,
        speech_duration=1.0,
        rendered_pause_before=0.0,
        rendered_pause_after=0.0,
        performance_engine="chatterbox-amharic",
        style_engine="seed-vc-v2",
    )


# ---------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------


def test_coverage_reports_the_categories_that_are_missing() -> None:
    """An easy set is not evidence, so what is absent is reported first."""

    report = measure_coverage([_dialogue()])

    assert report.lines == 1
    assert report.representative is False
    assert "shout" in report.missing
    assert "overlapping_speech" in report.missing
    assert "missing" in report.summary()


def test_a_representative_set_covers_every_category() -> None:
    lines = [
        _dialogue(end=0.8, intensity=0.1),  # short, low intensity
        _dialogue(end=12.0, intensity=0.95, delivery="shouting in rage"),  # long, loud
        _dialogue(delivery="whispering so as not to be heard", intensity=0.2),
        _dialogue(amharic="ሰላም iPhone ነው"),  # code-switching
    ]
    lines += [
        _dialogue(start=2.0 * index, end=2.0 * index + 1.0, speaker_id=f"SPEAKER_{index:02d}")
        for index in range(1, 6)
    ]
    lines += [
        _dialogue(
            speaker_id=SPEAKER,
            start=float(20 + index),
            end=float(20 + index) + 1.0,
        )
        for index in range(6)
    ]
    # Rapid turn taking between two speakers.
    lines.append(_dialogue(speaker_id=SPEAKER, start=40.0, end=41.0))
    lines.append(_dialogue(speaker_id=OTHER, start=41.05, end=42.0))

    report = measure_coverage(
        lines, crosstalk=[CrosstalkRegion((SPEAKER, OTHER), start=40.5, end=41.0)]
    )

    assert report.missing == ()
    assert report.representative is True
    assert report.covered_ratio == 1.0
    assert "all" in report.summary()


def test_each_category_is_detected_on_its_own() -> None:
    assert measure_coverage([_dialogue(end=0.5)]).counts["short_line"] == 1
    assert measure_coverage([_dialogue(end=9.0)]).counts["long_line"] == 1
    assert measure_coverage([_dialogue(intensity=0.1)]).counts["low_intensity"] == 1
    assert measure_coverage([_dialogue(intensity=0.9)]).counts["high_intensity"] == 1
    assert measure_coverage([_dialogue(amharic="ሰላም wifi")]).counts["code_switching"] == 1
    assert (
        measure_coverage([_dialogue(delivery="hushed and shaking")]).counts["whisper"] == 1
    )
    assert measure_coverage([_dialogue(emotion="screaming")]).counts["shout"] == 1


def test_crowded_and_recurring_categories_count_speakers() -> None:
    crowded = [
        _dialogue(speaker_id=f"SPEAKER_{index:02d}") for index in range(3)
    ]
    report = measure_coverage(crowded)

    assert report.counts["crowded_scene"] == 3
    assert report.counts["recurring_speaker"] == 0

    recurring = [_dialogue(speaker_id=SPEAKER) for _ in range(5)]
    assert measure_coverage(recurring).counts["recurring_speaker"] == 5


def test_rapid_turn_taking_needs_two_speakers() -> None:
    same = [_dialogue(end=1.0), _dialogue(start=1.05, end=2.0)]
    assert measure_coverage(same).counts["rapid_turn_taking"] == 0

    different = [
        _dialogue(speaker_id=SPEAKER, end=1.0),
        _dialogue(speaker_id=OTHER, start=1.05, end=2.0),
    ]
    assert measure_coverage(different).counts["rapid_turn_taking"] == 1

    slow = [
        _dialogue(speaker_id=SPEAKER, end=1.0),
        _dialogue(speaker_id=OTHER, start=5.0, end=6.0),
    ]
    assert measure_coverage(slow).counts["rapid_turn_taking"] == 0


def test_an_empty_dialogue_covers_nothing() -> None:
    report = measure_coverage([])

    assert report.lines == 0
    assert report.missing == GOLDEN_CATEGORIES
    assert report.representative is False


def test_a_coverage_report_is_json_safe() -> None:
    payload = measure_coverage([_dialogue()]).as_dict()

    assert json.loads(json.dumps(payload)) == payload
    assert payload["lines"] == 1


# ---------------------------------------------------------------------------
# embeddings and similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_is_one_for_identical_vectors() -> None:
    assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors_is_zero() -> None:
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_ignores_magnitude() -> None:
    assert cosine_similarity([1.0, 1.0], [100.0, 100.0]) == pytest.approx(1.0)


def test_cosine_similarity_rejects_mismatched_widths() -> None:
    with pytest.raises(InvalidMeasurementError, match="same width"):
        cosine_similarity([1.0], [1.0, 2.0])


def test_cosine_similarity_of_a_zero_vector_is_zero() -> None:
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert cosine_similarity([], []) == 0.0


def test_a_consistent_character_is_measured_as_consistent(tmp_path: Path) -> None:
    clips = [_clip(tmp_path, index=index) for index in range(3)]

    # Every clip embeds to the same direction as its reference, so similarity is 1.
    report = measure_speaker_identity(clips, embed=lambda path: [1.0, 0.0, 0.5])

    assert report.measured == 3
    assert report.unmeasured == 0
    (character,) = report.characters
    assert character.speaker_id == SPEAKER
    assert character.mean_similarity == pytest.approx(1.0)
    assert character.consistent is True
    assert report.all_consistent is True
    assert "character(s)" in report.summary()


def test_a_drifting_character_is_reported_by_its_worst_line(tmp_path: Path) -> None:
    """An average would hide the one line where the character sounds wrong."""

    clips = [_clip(tmp_path, index=index) for index in range(3)]

    def embed(path: Path):
        # The middle clip points elsewhere; the other two match their reference.
        return [0.0, 1.0] if "clip" in path.name and "_1." in path.name else [1.0, 0.0]

    report = measure_speaker_identity(clips, embed=embed)

    (character,) = report.characters
    assert character.samples == 3
    assert character.minimum_similarity == pytest.approx(0.0)
    assert character.spread > 0.0
    assert character.consistent is False
    assert report.all_consistent is False


def test_the_identity_reference_prefers_the_profile(tmp_path: Path) -> None:
    """The contract is "sound like this character", so the profile's reference wins."""

    from app.pipeline.voice_profiles import VoiceProfile

    clip = _clip(tmp_path)
    profile_reference = _write(tmp_path / "profile.wav", _tone(200.0))
    profile = VoiceProfile(
        speaker_id=SPEAKER,
        reference_audio=profile_reference,
        reference_start=0.0,
        reference_end=1.0,
        reference_text="ሰላም",
        clone_prompt_path=None,
    )
    seen: list[str] = []

    def embed(path: Path):
        seen.append(Path(path).name)
        return [1.0, 0.0]

    measure_speaker_identity([clip], profiles={SPEAKER: profile}, embed=embed)

    assert "profile.wav" in seen
    assert clip.voice_reference_path.name not in seen


def test_a_clip_without_a_profile_is_unmeasured(tmp_path: Path) -> None:
    report = measure_speaker_identity(
        [_clip(tmp_path)], profiles={}, embed=lambda path: [1.0]
    )

    assert report.measured == 0
    assert report.unmeasured == 1
    assert report.summary() == "voice identity: not measured"


def test_a_failing_embedder_is_unmeasured_rather_than_an_error(tmp_path: Path) -> None:
    def _boom(path: Path):
        raise RuntimeError("the embedding model is not installed")

    report = measure_speaker_identity([_clip(tmp_path)], embed=_boom)

    assert report.measured == 0
    assert report.unmeasured == 1


def test_a_multi_character_report_weights_the_overall_mean(tmp_path: Path) -> None:
    clips = [
        _clip(tmp_path, index=0, speaker_id=SPEAKER),
        _clip(tmp_path, index=1, speaker_id=SPEAKER),
        _clip(tmp_path, index=2, speaker_id=OTHER),
    ]
    values = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}

    def embed(path: Path):
        name = Path(path).name
        for speaker_id, vector in values.items():
            if speaker_id in name:
                return vector
        return [1.0, 0.0]

    report = measure_speaker_identity(clips, embed=embed)

    assert len(report.characters) == 2
    assert report.worst is not None
    assert report.overall_mean_similarity == pytest.approx(1.0)
    assert json.loads(json.dumps(report.as_dict())) == report.as_dict()


# ---------------------------------------------------------------------------
# performance preservation
# ---------------------------------------------------------------------------


def test_a_take_that_keeps_the_pitch_is_reported_as_preserved(tmp_path: Path) -> None:
    clip = _clip(tmp_path, pitch=150.0, reference_pitch=150.0)

    report = measure_performance_preservation([clip])

    assert report.measured == 1
    assert report.unusable == 0
    assert report.preserved_ratio == pytest.approx(1.0)
    assert report.mean_absolute_pitch_shift_semitones == pytest.approx(0.0, abs=0.2)
    # A monotone pair has no range to be a fraction of, so the ratio is undefined
    # rather than zero - and the line still counts as preserved, which it is.
    assert report.mean_pitch_range_ratio is None
    assert "preserved" in report.summary()


def test_an_expressive_take_that_keeps_its_range_is_preserved(tmp_path: Path) -> None:
    clip = _clip(tmp_path, pitch=150.0)
    sweep = _sweep(90.0, 180.0)
    _write(clip.performance_reference_path, sweep)
    _write(clip.audio_path, sweep)

    report = measure_performance_preservation([clip])

    assert report.preserved_ratio == pytest.approx(1.0)
    assert report.mean_pitch_range_ratio == pytest.approx(1.0, abs=0.05)
    assert report.worst is not None


def test_a_flattened_take_is_reported_as_a_lost_performance(tmp_path: Path) -> None:
    """The failure this measurement exists for: delivery flattened to a neutral read."""

    clip = _clip(tmp_path, pitch=135.0)
    _write(clip.performance_reference_path, _sweep(90.0, 180.0))

    report = measure_performance_preservation([clip])

    assert report.preserved_ratio == pytest.approx(0.0)
    ratio = report.mean_pitch_range_ratio
    assert ratio is not None and ratio < 0.5
    assert report.worst is not None


def test_a_take_that_moved_an_octave_is_reported_as_not_preserved(tmp_path: Path) -> None:
    clip = _clip(tmp_path, pitch=300.0, reference_pitch=150.0)

    report = measure_performance_preservation([clip])

    assert report.preserved_ratio == 0.0
    shift = report.mean_absolute_pitch_shift_semitones
    assert shift is not None and shift == pytest.approx(12.0, abs=0.5)


def test_an_unmeasured_line_does_not_score(tmp_path: Path) -> None:
    clip = _clip(tmp_path, index=0)
    # Replace the generated audio with something that cannot be read.
    (clip.audio_path).write_bytes(b"not audio")

    report = measure_performance_preservation([clip])

    assert report.measured == 0
    assert report.unmeasured == 1
    assert report.summary() == "performance: not measured"


def test_a_whispered_take_is_reported_as_unusable(tmp_path: Path) -> None:
    """No pitch means no comparison, which is said rather than faked."""

    rng = np.random.default_rng(3)
    clip = _clip(tmp_path, pitch=150.0)
    _write(clip.audio_path, (0.3 * rng.standard_normal(RATE)).astype(np.float32))

    report = measure_performance_preservation([clip])

    assert report.measured == 1
    assert report.unusable == 1
    assert report.usable_comparisons == ()
    assert report.preserved_ratio == 0.0


def test_a_preservation_report_is_json_safe(tmp_path: Path) -> None:
    report = measure_performance_preservation([_clip(tmp_path)])

    payload = report.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["lines"][0]["label"].startswith(SPEAKER)


# ---------------------------------------------------------------------------
# baseline comparison
# ---------------------------------------------------------------------------


def _aligned(
    tmp_path: Path,
    *,
    index: int = 0,
    window: float = 2.0,
    speech: float = 2.0,
    dialogue: AdaptedDialogue | None = None,
) -> AlignedClip:
    line = dialogue if dialogue is not None else _dialogue(end=window)
    return AlignedClip(
        index=index,
        clip=_clip(tmp_path, index=index, dialogue=line),
        audio_path=tmp_path / f"aligned_{index}.wav",
        sample_rate=RATE,
        start=line.start,
        tempo=1.0,
        required_tempo=1.0,
        speech_duration=speech,
        original_window=window,
        rendered_pause_before=0.0,
        rendered_pause_after=0.0,
    )


def test_an_unchanged_metric_is_not_a_regression() -> None:
    payload = {"duration": {"close_fit_ratio": 0.5, "unfitted": 2}}
    comparison = compare_to_baseline(payload, payload)

    assert comparison.ok is True
    assert comparison.regressions == ()
    assert comparison.improvements == ()
    assert len(comparison.changes) == 2
    assert "no regression" in comparison.summary()


def test_a_drop_in_close_fit_ratio_is_a_regression() -> None:
    current = {"duration": {"close_fit_ratio": 0.40}}
    baseline = {"duration": {"close_fit_ratio": 0.50}}

    comparison = compare_to_baseline(current, baseline)

    assert comparison.ok is False
    (regression,) = comparison.regressions
    assert regression.path == "duration.close_fit_ratio"
    assert regression.direction == "higher"
    assert regression.relative == pytest.approx(-0.2)
    assert "REGRESSION" in comparison.summary()


def test_a_rise_in_unfitted_lines_is_a_regression() -> None:
    comparison = compare_to_baseline(
        {"duration": {"unfitted": 10}}, {"duration": {"unfitted": 5}}
    )

    assert comparison.regressions
    assert comparison.regressions[0].direction == "lower"


def test_an_improvement_is_reported_as_such() -> None:
    comparison = compare_to_baseline(
        {"duration": {"close_fit_ratio": 0.7}}, {"duration": {"close_fit_ratio": 0.5}}
    )

    assert comparison.ok is True
    assert len(comparison.improvements) == 1
    assert "1 improved" in comparison.summary()


def test_a_small_move_is_inside_the_tolerance() -> None:
    comparison = compare_to_baseline(
        {"duration": {"mean_absolute_residual": 0.101}},
        {"duration": {"mean_absolute_residual": 0.100}},
    )

    assert comparison.ok is True
    assert comparison.regressions == ()
    assert comparison.improvements == ()
    assert comparison.changes[0].changed is False


def test_identity_has_a_tighter_tolerance_than_the_default() -> None:
    """Voice identity is a quality that must not drift quietly."""

    comparison = compare_to_baseline(
        {"identity": {"overall_mean_similarity": 0.80}},
        {"identity": {"overall_mean_similarity": 0.85}},
    )

    assert comparison.regressions
    assert comparison.regressions[0].tolerance < evaluation.DEFAULT_TOLERANCE


def test_a_directionless_metric_is_reported_as_changed_not_regressed() -> None:
    comparison = compare_to_baseline(
        {"lines": [{"x": 1}], "elapsed_seconds": 12.0}, {"elapsed_seconds": 10.0}
    )

    (change,) = comparison.changes
    assert change.direction == "neutral"
    assert change.regression is False
    assert change.changed is True
    assert comparison.ok is True


def test_metrics_present_on_one_side_only_are_skipped() -> None:
    """A new measurement is not a regression just because it was not there before."""

    comparison = compare_to_baseline(
        {"duration": {"lines": 10}, "identity": {"measured": 3}},
        {"duration": {"lines": 10}},
    )

    assert [change.path for change in comparison.changes] == ["duration.lines"]
    assert comparison.ok is True


def test_booleans_and_non_numbers_are_not_compared() -> None:
    comparison = compare_to_baseline(
        {"coverage": {"representative": False}, "seed": "abc"},
        {"coverage": {"representative": True}, "seed": "abc"},
    )

    assert comparison.changes == ()
    assert comparison.summary() == "baseline: no comparable metrics"


def test_per_line_detail_is_not_compared() -> None:
    """A line added or removed would shift every index, so only aggregates count."""

    comparison = compare_to_baseline(
        {"lines": [{"residual": 1.0}, {"residual": 2.0}]},
        {"lines": [{"residual": 1.0}]},
    )

    assert comparison.changes == ()


def test_a_baseline_round_trips_through_disk(tmp_path: Path) -> None:
    payload = {"duration": {"close_fit_ratio": 0.5}, "coverage": {"representative": True}}
    path = save_baseline(payload, tmp_path / "eval" / "baseline.json")

    assert path.is_file()
    assert load_baseline(path) == payload


def test_an_evaluation_report_can_be_saved_as_a_baseline(tmp_path: Path) -> None:
    from app.pipeline.evaluation import build_evaluation_report

    report = build_evaluation_report([_aligned(tmp_path)], dialogue=[_dialogue()])
    path = save_baseline(report, tmp_path / "baseline.json")

    stored = load_baseline(path)
    assert stored["qc"]["duration"]["lines"] == 1
    assert "coverage" in stored


def test_a_missing_baseline_is_reported(tmp_path: Path) -> None:
    """Silently comparing against nothing is the false confidence to avoid."""

    with pytest.raises(EvaluationError, match="no baseline"):
        load_baseline(tmp_path / "absent.json")


def test_a_malformed_baseline_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "baseline.json"
    path.write_text("[1, 2]", encoding="utf-8")

    with pytest.raises(EvaluationError, match="JSON object"):
        load_baseline(path)


def test_a_comparison_is_json_safe() -> None:
    comparison = compare_to_baseline(
        {"duration": {"close_fit_ratio": 0.4}}, {"duration": {"close_fit_ratio": 0.5}}
    )

    payload = comparison.as_dict()
    assert json.loads(json.dumps(payload)) == payload
    assert payload["ok"] is False
    assert payload["regressions"][0]["verdict"] == "regression"


# ---------------------------------------------------------------------------
# the assembled report
# ---------------------------------------------------------------------------


def test_the_report_assembles_what_was_supplied(tmp_path: Path) -> None:
    line = _dialogue()
    report = evaluation.build_evaluation_report(
        [_aligned(tmp_path, dialogue=line)],
        dialogue=[line],
        clips=[_clip(tmp_path, dialogue=line)],
        crosstalk=[CrosstalkRegion((SPEAKER, OTHER), start=0.0, end=0.5)],
        embed=lambda path: [1.0, 0.0],
        embedder="fake-embed",
        measure_performance=True,
    )

    assert report.qc.duration.lines == 1
    assert report.coverage.counts["overlapping_speech"] == 1
    assert report.identity is not None and report.identity.measured == 1
    assert report.performance is not None and report.performance.measured == 1

    summary = report.summary()
    assert "line(s) within" in summary
    assert "coverage:" in summary
    assert "performance:" in summary
    assert "voice identity:" in summary

    payload = report.as_dict()
    assert json.loads(json.dumps(payload)) == payload


def test_the_report_omits_measurements_nobody_supplied(tmp_path: Path) -> None:
    report = evaluation.build_evaluation_report([_aligned(tmp_path)])

    assert report.identity is None
    assert report.performance is None
    assert "voice identity" not in report.summary()
    assert "performance:" not in report.summary()


def test_the_report_can_be_compared_against_itself(tmp_path: Path) -> None:
    """The end-to-end path: measure, save, re-measure, compare."""

    report = evaluation.build_evaluation_report(
        [_aligned(tmp_path)], dialogue=[_dialogue()]
    )
    baseline = report.as_dict()

    comparison = compare_to_baseline(report.as_dict(), baseline)
    assert isinstance(comparison, BaselineComparison)
    assert comparison.ok is True
