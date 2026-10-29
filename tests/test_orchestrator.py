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

import numpy as np

from app.config import DEFAULT_TRANSLATION_GEMINI_MODEL, Settings
from app.pipeline import orchestrator, target_extraction
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
from app.pipeline.tts import PerformanceControls, SkippedLine, TtsClip
from app.pipeline.video import MuxResult
from app.pipeline.voice_profiles import NoUsableReferenceError, VoiceProfile

SPEAKER = "SPEAKER_00"
SECOND_SPEAKER = "SPEAKER_01"
RATE = 48_000


# ---------------------------------------------------------------------------
# The separation cache
# ---------------------------------------------------------------------------
# Separation is the longest stage by far and a rented pod's container disk is gone
# with the pod, so the stems are cached on Drive and restored instead of recomputed.


def _stem_track(tmp_path: Path) -> Path:
    """Write a track whose stems the cache path will recognise."""

    import soundfile as sf

    track = tmp_path / "stages" / "S01E01_mix.wav"
    track.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        str(track),
        np.zeros((RATE, 1), dtype=np.float32),
        RATE,
        subtype="FLOAT",
    )
    return track


def test_stems_already_in_the_run_are_not_separated_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import soundfile as sf

    track = _stem_track(tmp_path)
    stages = track.parent
    for name in ("S01E01_mix_speech.wav", "S01E01_mix_music.wav"):
        sf.write(str(stages / name), np.zeros((RATE, 1), dtype=np.float32), RATE)
    called: list[str] = []
    monkeypatch.setattr(
        orchestrator.separation,
        "separate_stems",
        lambda *a, **k: called.append("separated"),
    )

    stems = orchestrator._obtain_stems(track, stages, _settings(tmp_path), {}, print)

    assert called == []
    assert stems.speech.name == "S01E01_mix_speech.wav"


def test_cached_stems_are_restored_instead_of_separating(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The pod that runs second has no stems on disk, only the cache on Drive."""

    import soundfile as sf

    track = _stem_track(tmp_path)
    stages = track.parent
    settings = _settings(
        tmp_path, drive_remote="gdrive", drive_stem_folder="amharic-dub-cache/stems"
    )
    monkeypatch.setattr(orchestrator.rclone, "drive_ready", lambda _s: True)
    restored_calls: list[str] = []

    def restore(_settings_, *, folder, stem, destination, names, report=None):
        restored_calls.append(folder)
        for name in ("S01E01_mix_speech.wav", "S01E01_mix_music.wav"):
            sf.write(
                str(Path(destination) / name),
                np.zeros((RATE, 1), dtype=np.float32),
                RATE,
            )
        return [Path(destination) / "S01E01_mix_speech.wav"]

    monkeypatch.setattr(orchestrator.rclone, "restore_stems", restore)
    monkeypatch.setattr(
        orchestrator.separation,
        "separate_stems",
        lambda *a, **k: pytest.fail("separated again despite a cached copy"),
    )

    stems = orchestrator._obtain_stems(track, stages, settings, {}, print)

    assert restored_calls == ["amharic-dub-cache/stems"]
    assert stems.speech.exists()


def test_a_fresh_separation_is_cached_for_the_next_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import soundfile as sf

    track = _stem_track(tmp_path)
    stages = track.parent
    settings = _settings(
        tmp_path, drive_remote="gdrive", drive_stem_folder="amharic-dub-cache/stems"
    )
    monkeypatch.setattr(orchestrator.rclone, "drive_ready", lambda _s: True)
    monkeypatch.setattr(
        orchestrator.rclone, "restore_stems", lambda *a, **k: []
    )
    cached: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        orchestrator.rclone,
        "cache_stems",
        lambda _s, *, folder, stem, files, report=None: cached.append(
            tuple(Path(path).name for path in files)
        ),
    )

    def fake_separate(track_path, *, output_dir, settings, on_progress=None):
        from app.pipeline.separation import StemPaths

        for name in ("_speech.wav", "_music.wav", "_effects.wav"):
            (Path(output_dir) / f"{Path(track_path).stem}{name}").write_bytes(b"x")
        return StemPaths(
            speech=Path(output_dir) / "S01E01_mix_speech.wav",
            music=Path(output_dir) / "S01E01_mix_music.wav",
            effects=Path(output_dir) / "S01E01_mix_effects.wav",
        )

    monkeypatch.setattr(orchestrator.separation, "separate_stems", fake_separate)

    orchestrator._obtain_stems(track, stages, settings, {}, print)

    assert cached, "the freshly separated stems were not offered to the cache"
    assert "S01E01_mix_speech.wav" in cached[0]


def test_a_failed_cache_upload_does_not_fail_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An hour of separation must not be lost because the upload failed."""

    track = _stem_track(tmp_path)
    stages = track.parent
    settings = _settings(
        tmp_path, drive_remote="gdrive", drive_stem_folder="amharic-dub-cache/stems"
    )
    monkeypatch.setattr(orchestrator.rclone, "drive_ready", lambda _s: True)
    monkeypatch.setattr(orchestrator.rclone, "restore_stems", lambda *a, **k: [])

    def boom(*_args, **_kwargs):
        raise orchestrator.rclone.TransferError("quota exceeded")

    monkeypatch.setattr(orchestrator.rclone, "cache_stems", boom)

    def fake_separate(track_path, *, output_dir, settings, on_progress=None):
        from app.pipeline.separation import StemPaths

        return StemPaths(
            speech=Path(output_dir) / "S01E01_mix_speech.wav",
            music=Path(output_dir) / "S01E01_mix_music.wav",
            effects=Path(output_dir) / "S01E01_mix_effects.wav",
        )

    monkeypatch.setattr(orchestrator.separation, "separate_stems", fake_separate)
    messages: list[str] = []

    stems = orchestrator._obtain_stems(track, stages, settings, {}, messages.append)

    assert stems.speech.name == "S01E01_mix_speech.wav"
    assert any("not cached" in message for message in messages)


def test_no_stem_cache_skips_the_remote_entirely(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    track = _stem_track(tmp_path)
    stages = track.parent
    settings = _settings(
        tmp_path, drive_remote="gdrive", drive_stem_folder="amharic-dub-cache/stems"
    )
    monkeypatch.setattr(orchestrator.rclone, "drive_ready", lambda _s: True)
    monkeypatch.setattr(
        orchestrator.rclone,
        "restore_stems",
        lambda *a, **k: pytest.fail("the cache was consulted with --no-stem-cache"),
    )
    monkeypatch.setattr(
        orchestrator.separation,
        "separate_stems",
        lambda track_path, *, output_dir, settings, on_progress=None: (
            orchestrator.separation.StemPaths(
                speech=Path(output_dir) / "S01E01_mix_speech.wav",
                music=Path(output_dir) / "S01E01_mix_music.wav",
                effects=Path(output_dir) / "S01E01_mix_effects.wav",
            )
        ),
    )

    orchestrator._obtain_stems(
        track, stages, settings, {}, print, cache_on_drive=False
    )


def _settings(root: Path, **overrides: object) -> Settings:
    """Return settings pointing every directory at ``root``."""

    values: dict[str, object] = {
        "input_dir": root,
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "cache",
        "voice_profile_dir": root / "voices",
        # Pointed at a path that does not exist so a test never picks up the real
        # native-speaker samples in `data/voice-samples`: those switch the run to a
        # Whisper transcription of each sample, which is not what these tests exercise.
        "voice_sample_dir": root / "voice-samples",
        # Same reason: a bank built into `data/voice-bank` would make the fallback
        # available to every run in this file, so the default here is a path with no
        # manifest.
        "voice_bank_dir": root / "voice-bank",
        "dialogue_bible_path": root / "work" / "dialogue_bible.json",
        "character_guide_path": root / "work" / "character_guide.json",
        "episode_script_path": root / "work" / "episode_script.json",
        # These tests exercise the per-character path, which needs voice profiles,
        # so they pin the engine that uses them rather than inheriting the default.
        "tts_engine": "omnivoice",
        "omnivoice_model": "k2-fsa/OmniVoice",
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


def _profile(root: Path, speaker_id: str = SPEAKER) -> VoiceProfile:
    return VoiceProfile(
        speaker_id=speaker_id,
        reference_audio=root / "voices" / speaker_id / "reference.wav",
        reference_start=0.0,
        reference_end=5.0,
        quality_score=0.9,
        selection_reason="cleanest turn",
        estimated_vocal_presentation="masculine",
        estimated_pitch_hz=120.0,
        source_vocal_presentation="masculine",
        source_pitch_hz=120.0,
    )


def _clip(root: Path, index: int, dialogue: AdaptedDialogue) -> TtsClip:
    return TtsClip(
        index=index,
        dialogue=dialogue,
        performance=PerformanceControls.from_dialogue(dialogue, seed=index),
        audio_path=root / "tts" / "clips" / f"{index}.wav",
        take_path=root / "tts" / "takes" / f"{index}.wav",
        performance_reference_path=None,
        voice_reference_path=root / "voices" / SPEAKER / "reference.wav",
        sample_rate=24_000,
        speech_duration=2.5,
        rendered_pause_before=0.1,
        rendered_pause_after=0.2,
        performance_engine="omnivoice",
        style_engine="none",
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
        skipped: list[object] | None = None,
        lines: list[TranscriptSegment] | None = None,
        dialogue: list[AdaptedDialogue] | None = None,
        profiles: dict[str, VoiceProfile] | None = None,
        unusable: dict[str, Exception] | None = None,
        failures: dict[str, Exception] | None = None,
    ) -> None:
        self.root = root
        self.turns = [_turn()] if turns is None else turns
        self.crosstalk = [] if crosstalk is None else crosstalk
        self.skipped = [] if skipped is None else skipped
        self.lines = [_line()] if lines is None else lines
        self.dialogue = [_dialogue()] if dialogue is None else dialogue
        self.profiles = {SPEAKER: _profile(root)} if profiles is None else profiles
        self.unusable = {} if unusable is None else unusable
        self.failures = failures or {}
        self.received: dict[str, tuple[object, ...]] = {}
        self.order: list[str] = []
        self.seen_crosstalk: object = ()
        self.seen_overlap_turns: object = ()
        self.seen_registers_before_transcription: object = None
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

    def separate_stems(
        self,
        track: Path,
        *,
        output_dir: Path,
        settings: Settings,
        on_progress: object = None,
    ):  # noqa: ANN201
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
        self,
        speech: Path,
        turns: object,
        *,
        settings: Settings,
        crosstalk: object = (),
        overlap_turns: object = (),
        registers: object = None,
    ) -> list[TranscriptSegment]:
        self._record("transcription", speech, turns, settings)
        self.seen_overlap_turns = overlap_turns
        self.seen_registers_before_transcription = registers
        return list(self.lines)

    def adapt_dialogue(
        self,
        lines: object,
        *,
        settings: Settings,
        bible: object = None,
        enforce_budget: bool = True,
        enforce_fidel_loanwords: bool = True,
        maximum_reduction_attempts: int = 3,
        speaker_genders: object = None,
        on_progress: object = None,
    ) -> list[AdaptedDialogue]:
        self._record("translation", lines, settings, bible)
        self.seen_speaker_genders = speaker_genders
        return list(self.dialogue)

    def build_voice_profiles(
        self,
        turns: object,
        speech: Path,
        *,
        transcript: object,
        crosstalk: object = (),
        settings: Settings,
        require_reference_transcript: bool = False,
        allow_clean_fallback: bool = False,
        on_unusable: object = None,
        registers: object = None,
    ) -> dict[str, VoiceProfile]:
        self._record(
            "voice_profiles",
            turns,
            speech,
            transcript,
            settings,
            require_reference_transcript,
            allow_clean_fallback,
        )
        self.seen_crosstalk = crosstalk
        self.seen_registers = registers
        for speaker_id, error in self.unusable.items():
            if on_unusable is None:
                raise error
            on_unusable(speaker_id, error)  # type: ignore[operator]
        return {
            speaker_id: profile
            for speaker_id, profile in self.profiles.items()
            if speaker_id not in self.unusable
        }

    def synthesize_dialogue_detailed(
        self,
        dialogue: object,
        speech: Path,
        profiles: object,
        *,
        settings: Settings,
    ):  # noqa: ANN201
        from app.pipeline.tts import SynthesisResult

        self._record("tts", dialogue, speech, profiles, settings)
        spoken = list(dialogue)  # type: ignore[arg-type]
        clips = [_clip(self.root, index, line) for index, line in enumerate(spoken, start=1)]
        return SynthesisResult(clips=tuple(clips), skipped=tuple(self.skipped))

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

    def transcribe_detailed(*args, report=None, **kwargs):
        # The orchestrator calls the detailed form so it can record the
        # separation audit; the fake only implements the line-returning form.
        return orchestrator.transcription.TranscriptionResult(
            lines=tuple(fake.transcribe(*args, **kwargs))
        )

    monkeypatch.setattr(
        orchestrator.transcription, "transcribe_detailed", transcribe_detailed
    )
    monkeypatch.setattr(orchestrator.translation, "adapt_dialogue", fake.adapt_dialogue)
    monkeypatch.setattr(
        orchestrator.voice_profiles, "build_voice_profiles", fake.build_voice_profiles
    )
    monkeypatch.setattr(
        orchestrator.tts, "synthesize_dialogue_detailed", fake.synthesize_dialogue_detailed
    )
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


def test_omnivoice_profiles_require_clean_matching_actor_transcripts(
    tmp_path: Path, stages: Stages
) -> None:
    run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    (
        _turns,
        _speech,
        _transcript,
        _passed,
        require_transcript,
        clean_fallback,
    ) = stages.received["voice_profiles"]
    # The actor stays the identity, so the prompt must be one voice and its text
    # must describe its audio; an undescribed clean window is the only step down.
    assert require_transcript is True
    assert clean_fallback is True


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
    # What reaches adaptation is the sentence, not the raw transcriber segment.
    units = stages.received["translation"][0]
    assert isinstance(units, tuple)  # type: ignore[unreachable]
    assert [unit.text for unit in units] == ["I love you."]  # type: ignore[union-attr]
    assert [(unit.start, unit.end) for unit in units] == [(1.0, 4.0)]  # type: ignore[union-attr]
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
    assert len(stages.received["translation"][0]) == 1  # type: ignore[arg-type]
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


def test_the_diarizations_crosstalk_reaches_the_voice_profiles(
    monkeypatch, tmp_path: Path
) -> None:
    """The reference builder must see simultaneous speech, not just exclusive turns.

    The overlap term it used could never fire on real overlap, because the turns it
    compares are exclusive by construction - so a window containing both characters
    still scored "0.0s overlapped by other speakers".
    """

    region = CrosstalkRegion((SPEAKER, "SPEAKER_01"), 11.0, 12.0)
    fake = Stages(tmp_path, crosstalk=[region])
    _wire(monkeypatch, fake)

    run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    assert list(fake.seen_crosstalk) == [region]


def test_registers_are_measured_before_the_transcript_is_written(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The crosstalk gate needs each character's pitch before any text exists.

    A mixture has to be decoded to be heard at all, and the decode returns whichever
    voice dominates it - so a recovered line can only be trusted if it measures as the
    character it is about to be credited to. That means the register has to be known
    *before* the transcript stage runs, not afterwards where it is used to repair
    attribution.
    """

    fake = Stages(tmp_path)
    _wire(monkeypatch, fake)
    measured = {SPEAKER: 118.0, "SPEAKER_01": 205.0}
    monkeypatch.setattr(
        orchestrator.register,
        "measure_registers",
        lambda spans, audio: dict(measured),
    )

    run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    assert fake.seen_registers_before_transcription == measured


def test_a_stem_that_cannot_be_measured_still_yields_its_dialogue(
    stages: Stages, tmp_path: Path
) -> None:
    """An unmeasurable stem must not cost the run its dialogue.

    The measurement is evidence, not a prerequisite: without it the recovery falls back
    to its structural checks, and the run continues with the text it already has.
    """

    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    assert stages.seen_registers_before_transcription == {}
    assert result.synthesized_lines == len(stages.dialogue)


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
        # The turns themselves: a wrong line must be traceable to the turn that
        # produced it without re-running the model.
        "detail": [{"start": 1.0, "end": 4.0, "speaker_id": SPEAKER}],
        # Present even when empty: a run that corrected nothing has to be
        # distinguishable from a run whose diarization could not verify attribution.
        "verified": [],
        "speaker_corrections": [],
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
    assert payload["qc"]["crosstalk"] == {
        "regions": 0,
        "seconds": 0.0,
        "attribution_uncertain": [],
    }
    # Coverage says what the run did *not* exercise, so a good score cannot be
    # mistaken for good material.
    assert payload["coverage"]["lines"] == len(result.dialogue)
    assert payload["coverage"]["representative"] is False
    assert "long_line" in payload["coverage"]["missing"]
    # Which engine spoke the dub travels with it.
    assert payload["provenance"]["tts_engine"] == "omnivoice"
    # Which model adapted the dialogue travels with the dub, because the terminal that
    # printed it does not survive the Pod.
    assert payload["provenance"]["translation_backend"] == "openai"
    assert payload["provenance"]["translation_provider"] == "gemini"
    assert payload["provenance"]["translation_model"] == DEFAULT_TRANSLATION_GEMINI_MODEL
    # What was *sent*, not what was configured: `off` cannot be honoured by a Gemini 3
    # model, so the manifest records the lowest level that was actually asked for
    # rather than echoing the request.
    recorded = payload["provenance"]["translation_thinking"]
    assert recorded.startswith("off requested, sent reasoning_effort=low")
    assert "cannot disable reasoning" in recorded


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


def test_the_target_extraction_audit_reaches_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run has to be able to say which regions were split and which were not."""

    fake = Stages(tmp_path)
    _wire(monkeypatch, fake)
    region = CrosstalkRegion(("SPEAKER_00", "SPEAKER_01"), start=1.0, end=2.5)
    report = target_extraction.ExtractionReport(
        backend="fake",
        model="fake/separator",
        available=True,
        regions=(
            target_extraction.RegionExtraction(
                region=region,
                tracks=(
                    target_extraction.ExtractedTrack(
                        speaker_id="SPEAKER_01",
                        start=region.start,
                        end=region.end,
                        samples=np.zeros(int(1.5 * target_extraction.SAMPLE_RATE), np.float32),
                        measured_hz=120.0,
                    ),
                ),
                refusals=(target_extraction.Refusal("SPEAKER_00", "ambiguous"),),
            ),
        ),
    )
    monkeypatch.setattr(
        orchestrator.transcription,
        "transcribe_detailed",
        lambda *args, **kwargs: orchestrator.transcription.TranscriptionResult(
            lines=tuple(fake.lines),
            separation=report,
            separated_lines=1,
            legacy_lines=0,
            refused_recoveries=1,
        ),
    )

    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))["run"][
        "transcription"
    ]
    assert payload["crosstalk_separation"]["regions_separated"] == 1
    assert payload["crosstalk_separation"]["tracks"] == 1
    assert payload["crosstalk_separation"]["refused_tracks"] == 1
    assert payload["crosstalk_separation"]["detail"][0]["refusals"][0] == {
        "speaker_id": "SPEAKER_00",
        "reason": "ambiguous",
        "measured_hz": None,
    }
    assert payload["separated_lines"] == 1
    assert payload["legacy_separated_lines"] == 0
    assert payload["refused_recoveries"] == 1
    assert "1/1 crosstalk region(s) separated" in result.report()


def test_the_measured_voice_gender_reaches_adaptation(
    tmp_path: Path, stages: Stages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Amharic marks gender on the verb, so the prompt gets it from the film."""

    monkeypatch.setattr(
        orchestrator.register,
        "measure_registers",
        lambda turns, audio: {SPEAKER: 200.0},
    )

    run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    assert stages.seen_speaker_genders == {SPEAKER: "feminine"}


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


def test_nllb_adaptation_says_what_it_cannot_do(
    tmp_path: Path, stages: Stages
) -> None:
    """A translation backend has no use for the context, so the run says so.

    NLLB cannot be handed a scene, a character or a request to shorten a line. The
    orchestrator still derives the bible and passes it, because the call is the same
    either way, so the report has to be explicit rather than implying work happened.
    """

    from app.pipeline.dialogue_context import Character, CharacterBible

    settings = _settings(tmp_path, translation_backend="nllb")
    CharacterBible(
        {SPEAKER: Character(speaker_id=SPEAKER, name="Selam", register="informal")}
    ).save(settings.dialogue_bible_path)
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    reported = " ".join(messages)
    assert "NLLB translates line by line" in reported
    assert "dialogue bible" not in reported


def test_the_bible_is_reported_as_applied_only_when_it_is(
    tmp_path: Path, stages: Stages
) -> None:
    """The instruction-following backend is the one that reads the bible."""

    from app.pipeline.dialogue_context import Character, CharacterBible

    settings = _settings(tmp_path, translation_backend="openai")
    CharacterBible(
        {SPEAKER: Character(speaker_id=SPEAKER, name="Selam", register="informal")}
    ).save(settings.dialogue_bible_path)
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    reported = " ".join(messages)
    assert "dialogue bible: 1 character(s) applied" in reported
    assert "NLLB translates line by line" not in reported


def test_voice_sample_files_do_not_replace_film_actor_references(
    tmp_path: Path, stages: Stages
) -> None:
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "native.wav").write_bytes(b"sample")
    settings = _settings(tmp_path, voice_sample_dir=samples)
    messages: list[str] = []

    result = run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    reported = " ".join(messages)
    assert "voice samples" not in reported
    # The line names the film-actor path so a run is auditable for which of the two
    # reference sources produced the dub.
    assert (
        "voice references: 1 character(s) cloned from the film's own dialogue"
        in reported
    )
    assert "voice_samples" not in result.stage_seconds
    (_dialogue, _speech, received, passed_settings) = stages.received["tts"]
    (profile,) = received.values()
    assert profile.reference_audio == stages.profiles[SPEAKER].reference_audio
    assert profile.reference_text == stages.profiles[SPEAKER].reference_text


def test_without_samples_the_films_own_references_are_used(
    tmp_path: Path, stages: Stages
) -> None:
    """The mechanism is additive: no recordings, no change in behaviour."""

    settings = _settings(tmp_path, voice_sample_dir=tmp_path / "empty")
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    assert "voice samples" not in " ".join(messages)
    (_dialogue, _speech, received, passed_settings) = stages.received["tts"]
    (profile,) = received.values()
    assert profile.reference_audio == stages.profiles["SPEAKER_00"].reference_audio
    assert profile.reference_text != "ሰላም"


def _write_local_samples(directory: Path) -> Path:
    """Write one compliant recording/transcript pair, in the operator's convention."""

    import numpy as np
    import soundfile as sf

    directory.mkdir(parents=True, exist_ok=True)
    rate = 48_000
    seconds = 6.0
    time = np.arange(int(seconds * rate)) / rate
    sf.write(
        str(directory / "male-01.wav"),
        (0.3 * np.sin(2 * np.pi * 120.0 * time)).astype(np.float32),
        rate,
    )
    word = "\u123b\u122d"
    (directory / "male-01.txt").write_text(
        " ".join([word] * 16) + "\u1362", encoding="utf-8"
    )
    return directory


def _write_bank(root: Path) -> Path:
    """Build a one-voice bank so the fallback path runs without the network."""

    import numpy as np
    import soundfile as sf

    from app.pipeline.voice_bank import (
        BANK_MANIFEST_FILENAME,
        BankVoice,
        VoiceBank,
    )

    clip = root / "bank" / "clips" / "am-m-test.wav"
    clip.parent.mkdir(parents=True, exist_ok=True)
    rate = 24_000
    seconds = 5.0
    time = np.arange(int(seconds * rate)) / rate
    sf.write(
        str(clip),
        (0.2 * np.sin(2 * np.pi * 120.0 * time)).astype(np.float32),
        rate,
        subtype="PCM_16",
    )

    voice = BankVoice(
        voice_id="am-m-test",
        audio=clip,
        text="ዛሬ ጠዋት አዲስ አበባ ውስጥ በጣም ቀዝቃዛ አየር ነበር",
        duration=seconds,
        gender="male",
        presentation="masculine",
        pitch_hz=118.0,
    )
    bank = VoiceBank(directory=root / "bank", voices=(voice,), dataset="fake/corpus")
    (root / "bank" / BANK_MANIFEST_FILENAME).write_text(
        json.dumps(bank.to_dict()), encoding="utf-8"
    )
    return root / "bank"


def test_a_bank_never_replaces_a_usable_actor_reference(
    tmp_path: Path, stages: Stages
) -> None:
    """The actor is the voice identity; a bank exists only for a broken window."""

    bank_dir = _write_bank(tmp_path)
    settings = _settings(tmp_path, voice_bank_dir=bank_dir)
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    reported = " ".join(messages)
    assert "voice bank: available at" in reported
    assert "every character kept the actor's own reference" in reported
    assert "cast:" not in reported

    (_turns, _speech, _transcript, _passed, require_transcript, clean_fallback) = (
        stages.received["voice_profiles"]
    )
    # The bank is not allowed to relax the reference: the actor's window still has
    # to be one voice whose transcript describes its audio.
    assert require_transcript is True
    assert clean_fallback is True

    for profile in stages.received["tts"][2].values():
        assert profile.reference_source == "film"
        assert profile.reference_voice_id is None
        assert profile.reference_audio == stages.profiles[SPEAKER].reference_audio


def test_only_a_character_without_a_usable_film_reference_is_cast(
    tmp_path: Path, stages: Stages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One unusable actor window must not cost the rest of the cast their voices."""

    from app.pipeline.register import AttributionReport, SpeakerRegister

    def registers(speaker_id: str, pitch: float) -> SpeakerRegister:
        return SpeakerRegister(
            speaker_id=speaker_id,
            median_hz=pitch,
            low_hz=pitch - 20.0,
            high_hz=pitch + 20.0,
            turns=3,
            measured_turns=3,
            speech_seconds=4.0,
        )

    measured = {
        SPEAKER: registers(SPEAKER, 118.0),
        SECOND_SPEAKER: registers(SECOND_SPEAKER, 212.0),
    }
    monkeypatch.setattr(
        orchestrator.register,
        "repair_attribution",
        lambda lines, audio, **kwargs: (
            list(lines),
            AttributionReport(before=measured, after=measured, corrections=()),
        ),
    )

    bank_dir = _write_bank(tmp_path)
    settings = _settings(tmp_path, voice_bank_dir=bank_dir)
    stages.profiles[SECOND_SPEAKER] = _profile(tmp_path, SECOND_SPEAKER)
    stages.unusable[SECOND_SPEAKER] = NoUsableReferenceError(
        "only simultaneous speech was available"
    )
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    reported = " ".join(messages)
    assert "1 of 2 character(s) cast" in reported
    assert f"cast: {SECOND_SPEAKER} -> am-m-test" in reported
    assert "only simultaneous speech was available" in reported

    received = stages.received["tts"][2]
    assert received[SPEAKER].reference_source == "film"
    assert received[SECOND_SPEAKER].reference_source == "voice_bank"
    assert received[SECOND_SPEAKER].reference_voice_id == "am-m-test"
    # The actor's measured register survives the swap, so the fallback voice is
    # chosen in the same register and stays auditable.
    assert received[SECOND_SPEAKER].source_pitch_hz == pytest.approx(212.0)
    assert received[SECOND_SPEAKER].source_vocal_presentation == "feminine"
    # The manifest records why the film could not serve, not only that it did not.
    assert (
        "film reference unusable: only simultaneous speech was available"
        in received[SECOND_SPEAKER].selection_reason
    )
    # The only bank voice is male while the film measured a feminine register, so
    # the swap is audible and the run has to say so.
    assert (
        f"WARNING: {SECOND_SPEAKER} measured feminine in the film but the bank "
        "supplied a masculine voice"
    ) in reported


def test_a_bad_operator_pin_fails_as_a_voice_profile_stage_error(
    tmp_path: Path, stages: Stages
) -> None:
    """A typo in VOICE_BANK_MAP must name the stage, not escape as a raw error."""

    bank_dir = _write_bank(tmp_path)
    settings = _settings(
        tmp_path,
        voice_bank_dir=bank_dir,
        voice_bank_voice_map="SPEAKER_09=am-m-test",
    )
    stages.unusable[SPEAKER] = NoUsableReferenceError("no clean window")

    with pytest.raises(StageError) as failure:
        run_pipeline(_source(tmp_path), settings=settings)

    assert "while casting" in str(failure.value)
    assert "SPEAKER_09" in str(failure.value)


def test_a_character_with_no_film_profile_at_all_is_still_cast(
    tmp_path: Path, stages: Stages
) -> None:
    """A speaker with no measurable window is exactly who the bank is for."""

    bank_dir = _write_bank(tmp_path)
    settings = _settings(tmp_path, voice_bank_dir=bank_dir)
    stages.profiles = {}
    stages.unusable[SPEAKER] = NoUsableReferenceError("no speech at all")
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    assert "1 of 1 character(s) cast" in " ".join(messages)
    (profile,) = stages.received["tts"][2].values()
    assert profile.reference_source == "voice_bank"
    assert profile.reference_voice_id == "am-m-test"


def test_an_unusable_reference_without_a_bank_fails_with_guidance(
    tmp_path: Path, stages: Stages
) -> None:
    """Cloning a two-voice window is worse than failing: say so and say how to fix it."""

    settings = _settings(tmp_path, voice_bank_dir=tmp_path / "no-bank")
    stages.unusable[SPEAKER] = NoUsableReferenceError(
        "only simultaneous speech was available"
    )

    with pytest.raises(StageError) as failure:
        run_pipeline(_source(tmp_path), settings=settings)

    message = str(failure.value)
    assert "no usable actor reference" in message
    assert "only simultaneous speech was available" in message
    assert "build_voice_bank.py" in message


def test_a_bank_is_reported_when_the_engine_cannot_use_it(
    tmp_path: Path, stages: Stages
) -> None:
    """A single-voice engine ignores references, so say so instead of dropping it."""

    bank_dir = _write_bank(tmp_path)
    settings = _settings(
        tmp_path, voice_bank_dir=bank_dir, tts_engine="mms"
    )
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    reported = " ".join(messages)
    assert "voice profiles: skipped" in reported
    assert "voice bank: not used" in reported


def test_no_bank_keeps_the_films_own_references(tmp_path: Path, stages: Stages) -> None:
    """No bank and usable actor audio is the normal, unchanged run."""

    settings = _settings(tmp_path, voice_bank_dir=tmp_path / "no-bank")
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    reported = " ".join(messages)
    # The absence is stated, with where it looked: the bank is a build artifact, so a
    # fallback that silently did nothing is the likeliest way to think it ran.
    assert "voice bank: none at" in reported
    assert "fallback only" in reported
    (_turns, _speech, _transcript, _passed, require_transcript, clean_fallback) = (
        stages.received["voice_profiles"]
    )
    # Without a bank the film reference is still the identity, so it still has to be
    # transcribed exactly.
    assert require_transcript is True
    assert clean_fallback is True

    (_dialogue, _speech_stem, received, _passed) = stages.received["tts"]
    (profile,) = received.values()
    assert profile.reference_source == "film"
    assert profile.reference_voice_id is None


def test_the_adaptation_report_names_the_model_and_what_it_was_asked_for(
    tmp_path: Path, stages: Stages
) -> None:
    """Two runs are compared on these words, so a bare model name is not enough.

    The provider decides whether the requested thinking level was available at all, so
    the report states what was actually sent rather than what was configured.
    """

    settings = _settings(
        tmp_path,
        translation_backend="openai",
        translation_provider="gemini",
        translation_model="gemini-3.8-flash",
        translation_thinking="off",
        gemini_api_key="test-key",
    )
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=settings, progress=messages.append)

    reported = " ".join(messages)
    # ``off`` cannot be honoured by a Gemini 3 model, and the report says so.
    assert "adaptation: gemini gemini-3.8-flash" in reported
    assert "cannot disable reasoning" in reported


def test_a_line_that_cannot_be_voiced_is_recorded_in_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hole in the dub has to be auditable, not just missing."""

    from app.pipeline.tts import SkippedLine

    fake = Stages(
        tmp_path,
        skipped=[
            SkippedLine(
                index=0,
                speaker_id=SPEAKER,
                start=5.836,
                end=6.056,
                amharic="ዋልት።",
                reason="cannot be dubbed: the original window is 0.220s",
            )
        ],
    )
    _wire(monkeypatch, fake)

    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    skipped = payload["run"]["tts"]["skipped"]
    assert len(skipped) == 1
    assert skipped[0]["amharic"] == "ዋልት።"
    assert skipped[0]["duration"] == 0.22
    assert "cannot be dubbed" in skipped[0]["reason"]
    assert result.skipped == tuple(fake.skipped)


def test_an_incomplete_dub_says_so_in_the_manifest_and_the_exit_code(
    monkeypatch, tmp_path: Path
) -> None:
    """A missing line must not be discoverable only by listening to the film.

    The exit code is the part that matters: a caller that only checks it would
    otherwise be able to ship a dub with dialogue missing from it.
    """

    fake = Stages(
        tmp_path,
        skipped=[
            SkippedLine(
                index=0,
                speaker_id=SPEAKER,
                start=5.836,
                end=6.056,
                amharic="።፣?!",
                reason="cannot be dubbed: the Amharic line has nothing to pronounce",
            )
        ],
    )
    _wire(monkeypatch, fake)

    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert payload["notes"][0].startswith("INCOMPLETE DUB")
    assert "PARTIAL DUB" in result.report()

    monkeypatch.setattr(
        orchestrator, "run_pipeline", lambda *a, **k: result, raising=False
    )
    assert orchestrator.main([str(_source(tmp_path))]) == 2


def test_a_complete_dub_exits_zero(monkeypatch, tmp_path: Path, stages: Stages) -> None:
    _wire(monkeypatch, stages)

    result = run_pipeline(_source(tmp_path), settings=_settings(tmp_path))

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert not any(note.startswith("INCOMPLETE DUB") for note in payload["notes"])

    monkeypatch.setattr(
        orchestrator, "run_pipeline", lambda *a, **k: result, raising=False
    )
    assert orchestrator.main([str(_source(tmp_path))]) == 0


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
    assert (
        "adaptation requests: one per batch; no rewrites, retries, or failover"
        in reported
    )


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


def test_cast_from_bank_builds_a_missing_bank_before_the_run(
    tmp_path: Path, stages: Stages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh checkout has no bank, so the flag has to be able to create one.

    The bank is a build artifact - deliberately not committed - which makes "I asked for
    casting and got the actor's voice" the easy mistake on a new pod. This is the flag
    that removes the extra manual step; the network is stubbed by the fake source.
    """

    from app.pipeline import voice_bank as vb

    monkeypatch.setattr(orchestrator, "get_settings", lambda: _settings(tmp_path))

    built: list[Path] = []

    def fake_build(*, destination, settings, **kwargs):
        built.append(Path(destination))
        Path(destination).mkdir(parents=True, exist_ok=True)
        clip = Path(destination) / "clips" / "am-m-test.wav"
        clip.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        import soundfile as sf

        rate = 24_000
        seconds = 5.0
        time = np.arange(int(seconds * rate)) / rate
        sf.write(
            str(clip),
            (0.2 * np.sin(2 * np.pi * 120.0 * time)).astype(np.float32),
            rate,
            subtype="PCM_16",
        )
        voice = vb.BankVoice(
            voice_id="am-m-test",
            audio=clip,
            text="ዛሬ ጠዋት አዲስ አበባ ውስጥ በጣም ቀዝቃዛ አየር ነበር",
            duration=seconds,
            gender="male",
            presentation="masculine",
            pitch_hz=118.0,
        )
        bank = vb.VoiceBank(directory=Path(destination), voices=(voice,))
        (Path(destination) / vb.BANK_MANIFEST_FILENAME).write_text(
            json.dumps(bank.to_dict()), encoding="utf-8"
        )
        return bank

    monkeypatch.setattr(orchestrator.voice_bank, "build_voice_bank", fake_build)
    # The flag makes a bank *available*; it is used only for a character whose own
    # film audio cannot supply a reference, so this is that character.
    stages.unusable[SPEAKER] = NoUsableReferenceError("no clean window")

    code = orchestrator.main(
        [str(_source(tmp_path)), "--out", str(tmp_path / "run"), "--cast-from-bank"]
    )

    assert code == 0
    assert built == [tmp_path / "voice-bank"]
    (_dialogue, _speech, received, _passed) = stages.received["tts"]
    (profile,) = received.values()
    assert profile.reference_source == "voice_bank"


def test_cast_from_bank_prefers_the_operators_own_samples(
    tmp_path: Path, stages: Stages, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Supplied recordings beat a corpus download, which would replace them."""

    from app.pipeline import voice_bank as vb

    samples = tmp_path / "voice-samples"
    _write_local_samples(samples)
    settings = _settings(tmp_path, voice_sample_dir=samples)
    monkeypatch.setattr(orchestrator, "get_settings", lambda: settings)

    def no_corpus(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the corpus must not be used when samples are supplied")

    monkeypatch.setattr(orchestrator.voice_bank, "build_voice_bank", no_corpus)
    stages.unusable[SPEAKER] = NoUsableReferenceError("no clean window")

    code = orchestrator.main(
        [str(_source(tmp_path)), "--out", str(tmp_path / "run"), "--cast-from-bank"]
    )

    assert code == 0
    assert (tmp_path / "voice-bank" / vb.BANK_MANIFEST_FILENAME).is_file()
    (_dialogue, _speech, received, _passed) = stages.received["tts"]
    (profile,) = received.values()
    assert profile.reference_source == "voice_bank"


def test_cast_from_bank_reuses_an_existing_bank(tmp_path: Path, stages: Stages, monkeypatch: pytest.MonkeyPatch) -> None:
    """An already-built bank is not rebuilt: the build is network work, not a no-op."""

    bank_dir = _write_bank(tmp_path)
    monkeypatch.setattr(
        orchestrator, "get_settings",
        lambda: _settings(tmp_path, voice_bank_dir=bank_dir),
    )

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("an existing bank must not be rebuilt")

    monkeypatch.setattr(orchestrator.voice_bank, "build_voice_bank", explode)
    stages.unusable[SPEAKER] = NoUsableReferenceError("no clean window")

    code = orchestrator.main(
        [str(_source(tmp_path)), "--out", str(tmp_path / "run"), "--cast-from-bank"]
    )

    assert code == 0
    (_dialogue, _speech, received, _passed) = stages.received["tts"]
    (profile,) = received.values()
    assert profile.reference_voice_id == "am-m-test"


def test_one_spoken_sentence_reaches_adaptation_as_one_unit(
    tmp_path: Path, stages: Stages
) -> None:
    """A pause must not turn one thought into two translation and speech units.

    The transcriber segments on pauses, so a single sentence routinely arrives as
    several segments. Adapting and speaking each one separately is what produces
    several utterances with several final intonations.
    """

    stages.lines = [
        _line(start=1.0, end=2.0, text="It is the world"),
        _line(start=2.2, end=4.0, text="that has been pulled over your eyes."),
    ]
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=_settings(tmp_path), progress=messages.append)

    units = stages.received["translation"][0]
    assert len(units) == 1  # type: ignore[arg-type]
    (sentence,) = units  # type: ignore[union-attr]
    assert sentence.text == "It is the world that has been pulled over your eyes."
    assert len(sentence.parts) == 2
    assert "assembled into 1 sentence" in " ".join(messages)


def test_a_sentence_report_is_not_printed_when_nothing_was_joined(
    tmp_path: Path, stages: Stages
) -> None:
    messages: list[str] = []

    run_pipeline(_source(tmp_path), settings=_settings(tmp_path), progress=messages.append)

    assert "assembled into" not in " ".join(messages)

# ---------------------------------------------------------------------------
# Authored dialogue plan
# ---------------------------------------------------------------------------


def _plan_file(root: Path, *, amharic: str = "እወድሃለሁ።") -> Path:
    path = root / "movie.plan.json"
    path.write_text(
        json.dumps(
            {
                "characters": [{"name": "Walter"}, {"name": "Skyler"}],
                "dialogue": [
                    {
                        "cue": 1,
                        "speaker": "Walter",
                        "start": 1.0,
                        "end": 4.0,
                        "source_text": "I love you.",
                        "amharic": amharic,
                        "emotion": "tender",
                        "intensity": 0.7,
                        "delivery": "soft",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_a_plan_skips_translation_and_casts_the_character(
    tmp_path: Path, stages: Stages
) -> None:
    _plan_file(tmp_path)

    result = run_pipeline(
        _source(tmp_path),
        settings=_settings(tmp_path),
        progress=lambda _message: None,
    )

    # The Amharic came from the plan, so no adaptation request was made at all.
    assert "translation" not in stages.order
    assert result.dialogue[0].amharic == "እወድሃለሁ።"
    assert result.dialogue[0].emotion == "tender"
    # The character was tied to the measured voice that carries its line.
    assert result.plan is not None
    assert result.plan.casting.mapping == {"Walter": SPEAKER}
    assert result.dialogue[0].speaker_id == SPEAKER


def test_a_plan_is_recorded_in_the_manifest(tmp_path: Path, stages: Stages) -> None:
    _plan_file(tmp_path)

    result = run_pipeline(
        _source(tmp_path),
        settings=_settings(tmp_path),
        progress=lambda _message: None,
    )

    payload = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    plan = payload["run"]["plan"]
    assert plan["authored_cues"] == 1
    assert plan["voiced"] == 1
    assert plan["casting"]["mapping"] == {"Walter": SPEAKER}
    assert payload["run"]["translation"]["lines"] == 1
