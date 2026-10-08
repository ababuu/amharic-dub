"""Tests for :mod:`app.pipeline.orchestrator`.

Every stage is replaced by a recording fake, so this suite runs no model, makes no
network call and touches no GPU. What it verifies is the orchestration itself: the
stage order, that each stage receives the previous stage's output, the ``max_lines``
cost control, the manifest, and that a failure names the stage it came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import Settings
from app.pipeline import orchestrator
from app.pipeline.diarization import CrosstalkRegion, SpeakerSegment
from app.pipeline.mixing import MixResult
from app.pipeline.orchestrator import (
    MANIFEST_FILENAME,
    STAGES_DIRNAME,
    STAGE_ORDER,
    EmptyStageError,
    MissingSourceError,
    StageError,
    run_pipeline,
)
from app.pipeline.timing import AlignedClip
from app.pipeline.transcription import TranscriptSegment
from app.pipeline.translation import AdaptedDialogue
from app.pipeline.tts import PerformanceControls, TtsClip
from app.pipeline.video import MuxResult
from app.pipeline.voice_profiles import VoiceProfile

SPEAKER = "SPEAKER_00"
RATE = 48_000


def _settings(root: Path, **overrides: object) -> Settings:
    """Return settings pointing every directory at ``root``."""

    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "cache",
        "voice_profile_dir": root / "voices",
        "seed_vc_repo_path": root / "seed-vc",
        "dialogue_bible_path": root / "work" / "dialogue_bible.json",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _turn(start: float = 1.0, end: float = 4.0, speaker_id: str = SPEAKER) -> SpeakerSegment:
    return SpeakerSegment(speaker_id=speaker_id, start=start, end=end)


def _line(start: float = 1.0, end: float = 4.0, text: str = "I love you.") -> TranscriptSegment:
    return TranscriptSegment(speaker_id=SPEAKER, start=start, end=end, text=text)


def _dialogue(start: float = 1.0, end: float = 4.0, amharic: str = "እወድሃለሁ።") -> AdaptedDialogue:
    return AdaptedDialogue(
        speaker_id=SPEAKER,
        start=start,
        end=end,
        source_text="I love you.",
        amharic=amharic,
        emotion="romantic",
        intensity=0.7,
        delivery="soft",
        pause_before=0.1,
        pause_after=0.2,
    )


def _profile(root: Path) -> VoiceProfile:
    return VoiceProfile(
        speaker_id=SPEAKER,
        reference_audio=root / "voices" / SPEAKER / "reference.wav",
        reference_start=0.0,
        reference_end=5.0,
        quality_score=0.9,
        selection_reason="cleanest turn",
    )


def _clip(root: Path, index: int, dialogue: AdaptedDialogue) -> TtsClip:
    return TtsClip(
        index=index,
        dialogue=dialogue,
        performance=PerformanceControls.from_dialogue(dialogue, seed=index),
        audio_path=root / "tts" / "clips" / f"{index}.wav",
        take_path=root / "tts" / "chatterbox" / f"{index}.wav",
        performance_reference_path=root / "tts" / "performance" / f"{index}.wav",
        voice_reference_path=root / "voices" / SPEAKER / "reference.wav",
        sample_rate=24_000,
        speech_duration=2.5,
        rendered_pause_before=0.1,
        rendered_pause_after=0.2,
        performance_engine="chatterbox-amharic",
        style_engine="seed-vc-v2",
    )


def _aligned(root: Path, clip: TtsClip) -> AlignedClip:
    """Return the aligned form of ``clip``: fitted exactly, so it is a normal result."""

    return AlignedClip(
        index=clip.index,
        clip=clip,
        audio_path=root / "timing" / "aligned" / f"{clip.index}.wav",
        sample_rate=RATE,
        start=clip.start,
        tempo=1.0,
        required_tempo=1.0,
        speech_duration=clip.dialogue.duration,
        original_window=clip.dialogue.duration,
        rendered_pause_before=0.1,
        rendered_pause_after=0.2,
    )


def _mix(root: Path) -> MixResult:
    return MixResult(
        dialogue_path=root / "mix" / "dialogue_amharic.wav",
        mixed_path=root / "mix" / "mix_amharic.wav",
        sample_rate=RATE,
        channels=2,
        duration=10.0,
        dialogue_gain_db=0.0,
        duck_db=6.0,
        peak=0.8,
        peak_gain_db=0.0,
    )


def _dub(root: Path) -> MuxResult:
    return MuxResult(
        path=root / "movie_amharic.mp4",
        video_codec="h264",
        audio_codec="aac",
        audio_sample_rate=RATE,
        audio_channels=2,
        video_streams=1,
        audio_streams=1,
        duration=10.0,
    )


class Stages:
    """Recording fakes for every stage the orchestrator calls."""

    def __init__(
        self,
        root: Path,
        *,
        turns: list[SpeakerSegment] | None = None,
        crosstalk: list[CrosstalkRegion] | None = None,
        lines: list[TranscriptSegment] | None = None,
        dialogue: list[AdaptedDialogue] | None = None,
        profiles: dict[str, VoiceProfile] | None = None,
        failures: dict[str, Exception] | None = None,
    ) -> None:
        self.root = root
        self.turns = [_turn()] if turns is None else turns
        self.crosstalk = [] if crosstalk is None else crosstalk
        self.lines = [_line()] if lines is None else lines
        self.dialogue = [_dialogue()] if dialogue is None else dialogue
        self.profiles = {SPEAKER: _profile(root)} if profiles is None else profiles
        self.failures = failures or {}
        self.received: dict[str, tuple[object, ...]] = {}
        self.order: list[str] = []
        self.track = root / "stages" / "movie_mix.wav"

    def _record(self, name: str, *args: object) -> None:
        self.order.append(name)
        self.received[name] = args
        failure = self.failures.get(name)
        if failure is not None:
            raise failure

    def extract_audio(self, source: Path, destination: Path, *, settings: Settings) -> Path:
        self._record("extract", source, destination, settings)
        return self.track

    def separate_stems(self, track: Path, *, output_dir: Path, settings: Settings):  # noqa: ANN201
        from app.pipeline.separation import StemPaths

        self._record("separation", track, output_dir, settings)
        return StemPaths(
            speech=output_dir / "movie_speech.wav",
            music=output_dir / "movie_music.wav",
            effects=output_dir / "movie_effects.wav",
        )

    def diarize_detailed(self, speech: Path, *, settings: Settings):  # noqa: ANN201
        from app.pipeline.diarization import DiarizationResult

        self._record("diarization", speech, settings)
        return DiarizationResult(turns=tuple(self.turns), crosstalk=tuple(self.crosstalk))

    def transcribe(
        self, speech: Path, turns: object, *, settings: Settings
    ) -> list[TranscriptSegment]:
        self._record("transcription", speech, turns, settings)
        return list(self.lines)

    def adapt_dialogue(
        self, lines: object, *, settings: Settings, bible: object = None
    ) -> list[AdaptedDialogue]:
        self._record("translation", lines, settings, bible)
        return list(self.dialogue)

    def build_voice_profiles(
        self,
        turns: object,
        speech: Path,
        *,
        transcript: object,
        settings: Settings,
    ) -> dict[str, VoiceProfile]:
        self._record("voice_profiles", turns, speech, transcript, settings)
        return dict(self.profiles)

    def synthesize_dialogue(
        self,
        dialogue: object,
        speech: Path,
        profiles: object,
        *,
        settings: Settings,
    ) -> list[TtsClip]:
        self._record("tts", dialogue, speech, profiles, settings)
        spoken = list(dialogue)  # type: ignore[arg-type]
        return [_clip(self.root, index, line) for index, line in enumerate(spoken, start=1)]

    def align_dialogue(
        self, clips: object, *, output_dir: Path, settings: Settings
    ) -> list[AlignedClip]:
        self._record("timing", clips, output_dir, settings)
        return [_aligned(self.root, clip) for clip in clips]  # type: ignore[union-attr]

    def mix_track(
        self,
        aligned: object,
        music: Path,
        effects: Path,
        *,
        output_dir: Path,
        settings: Settings,
    ) -> MixResult:
        self._record("mixing", aligned, music, effects, output_dir, settings)
        return _mix(self.root)

    def mux_dub(
        self,
        source: Path,
        audio: Path,
        destination: Path,
        *,
        settings: Settings,
    ) -> MuxResult:
        self._record("mux", source, audio, destination, settings)
        return _dub(self.root)


def _wire(monkeypatch: pytest.MonkeyPatch, fake: Stages) -> None:
    """Point every stage the orchestrator calls at ``fake``."""

    monkeypatch.setattr(orchestrator.video, "extract_audio", fake.extract_audio)
    monkeypatch.setattr(orchestrator.separation, "separate_stems", fake.separate_stems)
    monkeypatch.setattr(
        orchestrator.diarization, "diarize_detailed", fake.diarize_detailed
    )
    monkeypatch.setattr(orchestrator.transcription, "transcribe", fake.transcribe)
    monkeypatch.setattr(orchestrator.translation, "adapt_dialogue", fake.adapt_dialogue)
    monkeypatch.setattr(
        orchestrator.voice_profiles, "build_voice_profiles", fake.build_voice_profiles
    )
    monkeypatch.setattr(orchestrator.tts, "synthesize_dialogue", fake.synthesize_dialogue)
    monkeypatch.setattr(orchestrator.timing, "align_dialogue", fake.align_dialogue)
    monkeypatch.setattr(orchestrator.mixing, "mix_track", fake.mix_track)
    monkeypatch.setattr(orchestrator.video, "mux_dub", fake.mux_dub)


@pytest.fixture
def stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Stages:
    """Wire the orchestrator to recording fakes and return them."""

    fake = Stages(tmp_path)
    _wire(monkeypatch, fake)
    return fake


def _source(root: Path) -> Path:
    path = root / "movie.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"stand-in")
    return path


def test_stage_order_is_the_documented_order() -> None:
    assert STAGE_ORDER == (
        "extract",
        "separation",
        "diarization",
        "transcription",
        "translation",
        "voice_profiles",
        "tts",
        "timing",
        "mixing",
        "mux",
    )


def test_every_stage_runs_once_in_order(
    tmp_path: Path, stages: Stages
) -> None:
    run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    assert stages.order == list(STAGE_ORDER)


def test_each_stage_receives_the_previous_stage_output(
    tmp_path: Path, stages: Stages
) -> None:
    run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    separation_stems = stages.received["separation"][0]
    assert separation_stems == stages.track
    assert stages.received["diarization"][0] == stages.received["separation"][1] / "movie_speech.wav"
    assert stages.received["transcription"][0] == stages.received["diarization"][0]
    assert stages.received["translation"][0] == tuple(stages.lines)
    assert stages.received["tts"][0] == tuple(stages.dialogue)
    placed = stages.received["timing"][0]
    assert len(placed) == len(stages.dialogue)  # type: ignore[arg-type]
    assert all(isinstance(clip, TtsClip) for clip in placed)  # type: ignore[union-attr]
    assert stages.received["mixing"][0] == tuple(_aligned(tmp_path, clip) for clip in placed)  # type: ignore[arg-type]
    assert stages.received["mixing"][1] == stages.received["separation"][1] / "movie_music.wav"
    assert stages.received["mixing"][2] == stages.received["separation"][1] / "movie_effects.wav"
    assert stages.received["mux"][0] == stages.received["extract"][0]
    assert stages.received["mux"][1] == _mix(tmp_path).mixed_path


def test_stems_and_track_land_in_the_run_directory(
    tmp_path: Path, stages: Stages
) -> None:
    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    assert result.output_dir == tmp_path / "out" / "movie"
    destination = stages.received["extract"][1]
    assert destination == result.output_dir / STAGES_DIRNAME / "movie_mix.wav"
    assert stages.received["separation"][1] == result.output_dir / STAGES_DIRNAME


def test_result_exposes_every_stage_output(
    tmp_path: Path, stages: Stages
) -> None:
    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    assert result.track == stages.track
    assert result.turns == tuple(stages.turns)
    assert result.lines == tuple(stages.lines)
    assert result.dialogue == tuple(stages.dialogue)
    assert result.profiles == stages.profiles
    assert len(result.clips) == len(stages.dialogue)
    assert len(result.alignment) == len(stages.dialogue)
    assert result.mix == _mix(tmp_path)
    assert result.dub == _dub(tmp_path)
    assert result.final_video == _dub(tmp_path).path
    assert not result.partial
    assert set(result.stage_seconds) == set(STAGE_ORDER)


def test_the_run_ends_with_the_dubbed_video(tmp_path: Path, stages: Stages) -> None:
    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    assert stages.order[-1] == "mux"
    assert result.dub.audio_codec == "aac"
    assert result.dub.path == _dub(tmp_path).path


def test_max_lines_shortens_only_the_synthesis(
    tmp_path: Path, stages: Stages
) -> None:
    stages.dialogue = [_dialogue(start=float(i), end=float(i) + 2.0) for i in range(4)]

    result = run_pipeline(
        _source(tmp_path), settings=_settings(tmp_path), max_lines=2
    )

    assert result.dialogue == tuple(stages.dialogue)
    assert len(stages.received["tts"][0]) == 2  # type: ignore[arg-type]
    assert len(result.clips) == 2
    assert stages.received["translation"][0] == tuple(stages.lines)  # type: ignore[arg-type]
    # A shortened run still produces a deliverable, but it is a partial dub.
    assert result.partial
    assert result.synthesized_lines == 2
    assert stages.order[-1] == "mux"


def test_max_lines_larger_than_the_dialogue_changes_nothing(
    tmp_path: Path, stages: Stages
) -> None:
    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path), max_lines=99)

    assert stages.received["tts"][0] == tuple(stages.dialogue)
    assert not result.partial


def test_manifest_records_the_run(tmp_path: Path, stages: Stages) -> None:
    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.manifest_path == result.output_dir / MANIFEST_FILENAME
    assert payload["run"]["final_video"] == result.to_dict()["final_video"]
    assert payload["run"]["synthesized_lines"] == len(result.clips)
    assert payload["run"]["partial"] is False
    assert payload["run"]["diarization"] == {
        "turns": 1,
        "speakers": [SPEAKER],
        "crosstalk": {"regions": 0, "seconds": 0.0, "detail": []},
    }
    assert payload["run"]["tts"]["clips"][0]["amharic"] == stages.dialogue[0].amharic
    assert payload["run"]["voice_profiles"][SPEAKER]["reference_duration"] == 5.0
    assert payload["run"]["timing"]["lines"] == len(result.alignment)
    assert payload["run"]["timing"]["unfitted"] == 0
    assert payload["run"]["mixing"]["duck_db"] == 6.0
    assert payload["run"]["mux"]["audio_codec"] == "aac"
    assert payload["settings"]["deepseek_api_key_set"] is False
    assert payload["notes"]
    # Every run measures itself, so a regression is visible without re-listening.
    assert payload["qc"]["duration"]["lines"] == len(result.alignment)
    assert payload["qc"]["duration"]["close_fits"] == len(result.alignment)
    assert payload["qc"]["pronunciation"] is None
    assert payload["qc"]["crosstalk"] == {"regions": 0, "seconds": 0.0}
    # Coverage says what the run did *not* exercise, so a good score cannot be
    # mistaken for good material.
    assert payload["coverage"]["lines"] == len(result.dialogue)
    assert payload["coverage"]["representative"] is False
    assert "long_line" in payload["coverage"]["missing"]
    # Seed-VC is a checkout, not a pinned dependency, so a run records what it used.
    assert payload["provenance"]["seed_vc_convert_style"] is False
    assert payload["provenance"]["seed_vc_revision"] is None
    assert payload["provenance"]["seed_vc_diffusion_steps"] == 30


def test_crosstalk_is_reported_in_the_manifest(tmp_path: Path, monkeypatch) -> None:
    """Simultaneous speech is invisible to exclusive attribution, so it is reported."""

    fake = Stages(
        tmp_path,
        crosstalk=[CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), start=1.0, end=2.5)],
    )
    _wire(monkeypatch, fake)

    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    crosstalk = json.loads(result.manifest_path.read_text(encoding="utf-8"))["run"][
        "diarization"
    ]["crosstalk"]
    assert crosstalk["regions"] == 1
    assert crosstalk["seconds"] == 1.5
    assert crosstalk["detail"] == [
        {"start": 1.0, "end": 2.5, "speakers": ["SPEAKER_00", "SPEAKER_01"]}
    ]
    assert result.crosstalk == tuple(fake.crosstalk)

    crosstalk = json.loads(result.manifest_path.read_text(encoding="utf-8"))["run"][
        "diarization"
    ]["crosstalk"]
    assert crosstalk["regions"] == 1
    assert crosstalk["seconds"] == 1.5
    assert crosstalk["detail"] == [
        {"start": 1.0, "end": 2.5, "speakers": ["SPEAKER_00", "SPEAKER_01"]}
    ]


def test_the_dialogue_bible_is_handed_to_adaptation(
    tmp_path: Path, stages: Stages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Consistency state must reach the prompt, not just exist on disk."""

    from app.pipeline.dialogue_context import Character, CharacterBible

    settings = _settings(tmp_path)
    CharacterBible(
        {SPEAKER: Character(speaker_id=SPEAKER, name="Selam", register="informal")}
    ).save(settings.dialogue_bible_path)
    monkeypatch.setattr(orchestrator, "get_settings", lambda: settings)

    run_pipeline(_source(tmp_path), settings=settings)

    bible = stages.received["translation"][2]
    assert isinstance(bible, CharacterBible)
    assert bible.name_of(SPEAKER) == "Selam"


def test_an_absent_bible_is_empty_rather_than_fatal(
    tmp_path: Path, stages: Stages
) -> None:
    """A first run has no consistency state yet, which is the normal case."""

    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    bible = stages.received["translation"][2]
    assert len(bible) == 0
    assert result.dialogue


def test_manifest_truncation_is_recorded(
    tmp_path: Path, stages: Stages
) -> None:
    stages.dialogue = [_dialogue(start=float(i), end=float(i) + 2.0) for i in range(3)]

    result = run_pipeline(
        _source(tmp_path), settings=_settings(tmp_path), max_lines=1
    )

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["run"]["synthesized_lines"] == 1
    assert payload["run"]["partial"] is True
    assert payload["run"]["translation"]["lines"] == 3
    assert len(payload["run"]["tts"]["clips"]) == 1


def test_manifest_is_written_atomically(tmp_path: Path, stages: Stages) -> None:
    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    leftovers = list(result.output_dir.glob("*.tmp"))
    assert leftovers == []


def test_report_summarizes_the_run(tmp_path: Path, stages: Stages) -> None:
    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path), progress=lambda _: None)

    report = result.report()
    assert "diarization" in report
    assert "1 clip(s)" in report
    assert "timing" in report
    assert "0 not fitted" in report
    assert "mixing" in report
    assert _dub(tmp_path).path.name in report
    assert "PARTIAL DUB" not in report


def test_report_flags_a_partial_dub(tmp_path: Path, stages: Stages) -> None:
    stages.dialogue = [_dialogue(start=float(i), end=float(i) + 2.0) for i in range(3)]

    result = run_pipeline(
        _source(tmp_path), settings=_settings(tmp_path), max_lines=1, progress=lambda _: None
    )

    assert "PARTIAL DUB" in result.report()


def test_progress_receives_one_line_per_stage(
    tmp_path: Path, stages: Stages
) -> None:
    messages: list[str] = []

    run_pipeline(
        _source(tmp_path), settings=_settings(tmp_path), progress=messages.append
    )

    reported = [message.strip() for message in messages]
    for name in STAGE_ORDER:
        assert f"-> {name}" in reported


# ---------------------------------------------------------------------------
# Stopping conditions
# ---------------------------------------------------------------------------


def test_missing_source_is_reported_before_anything_runs(
    tmp_path: Path, stages: Stages
) -> None:
    with pytest.raises(MissingSourceError):
        run_pipeline(tmp_path / "nope.mp4", settings=_settings(tmp_path))

    assert stages.order == []


def test_empty_diarization_stops_the_run(tmp_path: Path, stages: Stages) -> None:
    stages.turns = []

    with pytest.raises(EmptyStageError) as info:
        run_pipeline(_source(tmp_path), settings=_settings(tmp_path), progress=lambda _: None)

    assert "diarization" in str(info.value)
    assert stages.order == ["extract", "separation", "diarization"]


def test_empty_transcript_stops_the_run(tmp_path: Path, stages: Stages) -> None:
    stages.lines = []

    with pytest.raises(EmptyStageError) as info:
        run_pipeline(_source(tmp_path), settings=_settings(tmp_path), progress=lambda _: None)

    assert "transcription" in str(info.value)
    assert "tts" not in stages.order


def test_empty_adaptation_stops_the_run(tmp_path: Path, stages: Stages) -> None:
    stages.dialogue = []

    with pytest.raises(EmptyStageError) as info:
        run_pipeline(_source(tmp_path), settings=_settings(tmp_path), progress=lambda _: None)

    assert "adaptation" in str(info.value)
    assert "voice_profiles" not in stages.order


def test_no_voice_profiles_stops_the_run(tmp_path: Path, stages: Stages) -> None:
    stages.profiles = {}

    with pytest.raises(EmptyStageError) as info:
        run_pipeline(_source(tmp_path), settings=_settings(tmp_path), progress=lambda _: None)

    assert "voice profile" in str(info.value)
    assert "tts" not in stages.order


def test_negative_max_lines_is_rejected(tmp_path: Path, stages: Stages) -> None:
    with pytest.raises(orchestrator.OrchestrationError):
        run_pipeline(
            _source(tmp_path), settings=_settings(tmp_path), max_lines=-1, progress=lambda _: None
        )

    assert "tts" not in stages.order


# ---------------------------------------------------------------------------
# Failures
# ---------------------------------------------------------------------------


def test_stage_failure_names_the_stage_and_keeps_the_cause(
    tmp_path: Path, stages: Stages
) -> None:
    original = ValueError("boom")
    stages.failures = {"transcription": original}

    with pytest.raises(StageError) as info:
        run_pipeline(_source(tmp_path), settings=_settings(tmp_path), progress=lambda _: None)

    assert "transcription" in str(info.value)
    assert info.value.__cause__ is original
    assert "translation" not in stages.order


def test_cli_returns_nonzero_and_reports_when_a_stage_fails(
    tmp_path: Path, stages: Stages, monkeypatch: pytest.MonkeyPatch
) -> None:
    stages.failures = {"separation": RuntimeError("no weights")}
    # The CLI resolves settings itself, so pin it at the temporary tree instead of
    # letting it create directories inside the project.
    monkeypatch.setattr(orchestrator, "get_settings", lambda: _settings(tmp_path))

    code = orchestrator.main(
        [str(_source(tmp_path)), "--out", str(tmp_path / "run")]
    )

    assert code == 1


def test_cli_returns_zero_on_a_complete_run(
    tmp_path: Path, stages: Stages, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(orchestrator, "get_settings", lambda: _settings(tmp_path))

    code = orchestrator.main([str(_source(tmp_path)), "--out", str(tmp_path / "run")])

    assert code == 0
    assert (tmp_path / "run" / MANIFEST_FILENAME).is_file()
