"""Tests for the environment-driven configuration system.

These tests never touch real credentials and never require a GPU, FFmpeg, or any
model weights - the configuration layer must work on a bare machine.
"""

from __future__ import annotations

import pytest

from app.config import (
    DEFAULT_DIALOGUE_BIBLE_FILENAME,
    DEFAULT_DIARIZATION_MODEL,
    DEFAULT_MIX_DIALOGUE_GAIN_DB,
    DEFAULT_MIX_DUCK_DB,
    DEFAULT_SEED_VC_DIFFUSION_STEPS,
    DEFAULT_SEED_VC_REPO_NAME,
    DEFAULT_TIMING_MAX_TEMPO,
    DEFAULT_TIMING_MIN_LINE_GAP,
    DEFAULT_TRANSLATION_LENGTH_PENALTY,
    DEFAULT_TRANSLATION_SHORTEN_PENALTY,
    DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND,
    DEFAULT_TIMING_MIN_TEMPO,
    DEFAULT_TRANSCRIPTION_COMPUTE_TYPE,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_TRANSLATION_BASE_URL,
    DEFAULT_TRANSLATION_BATCH_SIZE,
    DEFAULT_TRANSLATION_MODEL,
    DEFAULT_TTS_MIN_LINE_SECONDS,
    DEFAULT_TTS_MAX_PAUSE_SECONDS,
    DEFAULT_TTS_MODEL,
    DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION,
    DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION,
    DEFAULT_VOICE_REFERENCE_MAX_DURATION,
    DEFAULT_VOICE_REFERENCE_MIN_DURATION,
    DEFAULT_VOICE_REFERENCE_TARGET_DURATION,
    Settings,
    get_settings,
)

_ENV_VARS = (
    "INPUT_DIR",
    "WORK_DIR",
    "OUTPUT_DIR",
    "MODEL_CACHE_DIR",
    "DEEPSEEK_API_KEY",
    "HUGGINGFACE_TOKEN",
    "DIARIZATION_MODEL",
    "DIARIZATION_MIN_SPEAKERS",
    "DIARIZATION_MAX_SPEAKERS",
    "TRANSCRIPTION_MODEL",
    "TRANSCRIPTION_COMPUTE_TYPE",
    "TRANSCRIPTION_LANGUAGE",
    "TRANSLATION_BACKEND",
    "TRANSLATION_MODEL",
    "TRANSLATION_BASE_URL",
    "TRANSLATION_BATCH_SIZE",
    "TRANSLATION_DISABLE_THINKING",
    "TRANSLATION_NUM_BEAMS",
    "TRANSLATION_MAX_NEW_TOKENS",
    "TRANSLATION_ENFORCE_BUDGET",
    "TRANSLATION_ENFORCE_FIDEL_LOANWORDS",
    "VOICE_PROFILE_DIR",
    "VOICE_REFERENCE_MIN_DURATION",
    "VOICE_REFERENCE_TARGET_DURATION",
    "VOICE_REFERENCE_MAX_DURATION",
    "DIALOGUE_BIBLE_PATH",
    "TTS_MODEL",
    "TTS_ENGINE",
    "CHATTERBOX_MODEL",
    "MMS_SAMPLE_RATE",
    "MMS_SEED",
    "MMS_SPEAKING_RATE",
    "SEED_VC_REPO_PATH",
    "SEED_VC_DIFFUSION_STEPS",
    "SEED_VC_CONVERT_STYLE",
    "TTS_PERFORMANCE_REFERENCE_MIN_DURATION",
    "TTS_PERFORMANCE_REFERENCE_MAX_DURATION",
    "TTS_MIN_LINE_SECONDS",
    "TTS_CONTINUE_ON_FAILURE",
    "TTS_MAX_PAUSE_SECONDS",
    "TIMING_MIN_TEMPO",
    "TIMING_MAX_TEMPO",
    "MIX_DIALOGUE_GAIN_DB",
    "MIX_DUCK_DB",
    "DEVICE",
    "LOG_LEVEL",
)


@pytest.fixture(autouse=True)
def _isolate_from_dotenv(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of a developer's local ``.env`` file."""

    monkeypatch.setattr("app.config.load_env_file", lambda *args, **kwargs: None)


@pytest.fixture
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults_are_project_relative(_clean_env: None) -> None:
    settings = Settings.from_env()

    assert settings.input_dir.name == "input"
    assert settings.work_dir.name == "working"
    assert settings.output_dir.name == "output"
    assert settings.model_cache_dir.name == "models_cache"
    assert settings.device == "cuda"
    assert settings.log_level == "INFO"


def test_missing_credentials_do_not_break_construction(_clean_env: None) -> None:
    settings = Settings.from_env()

    assert settings.deepseek_api_key is None
    assert settings.huggingface_token is None
    assert settings.has_deepseek_credentials is False
    assert settings.has_huggingface_credentials is False
    # The default backend is NLLB, which reads no credential, so only the token the
    # gated diarization pipeline needs is reported.
    assert settings.translation_backend == "nllb"
    assert settings.missing_credentials() == ["HUGGINGFACE_TOKEN"]


def test_missing_credentials_includes_the_key_the_backend_reads(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TRANSLATION_BACKEND", "openai")

    settings = Settings.from_env()

    assert settings.missing_credentials() == [
        "DEEPSEEK_API_KEY",
        "HUGGINGFACE_TOKEN",
    ]


def test_environment_variables_override_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "in"))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_test_token")
    monkeypatch.setenv("DEVICE", "cpu")
    monkeypatch.setenv("LOG_LEVEL", "debug")

    settings = Settings.from_env()

    assert settings.input_dir == tmp_path / "in"
    assert settings.work_dir == tmp_path / "work"
    assert settings.output_dir == tmp_path / "out"
    assert settings.model_cache_dir == tmp_path / "models"
    assert settings.device == "cpu"
    assert settings.log_level == "DEBUG"
    assert settings.missing_credentials() == []


def test_ensure_directories_creates_all_paths(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setenv("WORK_DIR", str(tmp_path / "working"))
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "models_cache"))
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    settings = Settings.from_env()
    settings.ensure_directories()

    assert settings.input_dir.is_dir()
    assert settings.work_dir.is_dir()
    assert settings.output_dir.is_dir()
    assert settings.model_cache_dir.is_dir()


def test_as_dict_never_exposes_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "super-secret-value")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "hf_secret_value")

    payload = Settings.from_env().as_dict()

    assert payload["deepseek_api_key_set"] is True
    assert payload["huggingface_token_set"] is True
    assert "super-secret-value" not in str(payload)
    assert "hf_secret_value" not in str(payload)


def test_diarization_model_is_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    assert Settings.from_env().diarization_model == DEFAULT_DIARIZATION_MODEL

    monkeypatch.setenv("DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1")

    settings = Settings.from_env()
    assert settings.diarization_model == "pyannote/speaker-diarization-3.1"
    assert settings.as_dict()["diarization_model"] == "pyannote/speaker-diarization-3.1"


def test_diarization_speaker_counts_are_optional_and_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.diarization_min_speakers is None
    assert defaults.diarization_max_speakers is None

    monkeypatch.setenv("DIARIZATION_MIN_SPEAKERS", "4")
    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "12")

    settings = Settings.from_env()
    assert settings.diarization_min_speakers == 4
    assert settings.diarization_max_speakers == 12
    assert settings.as_dict()["diarization_min_speakers"] == 4
    assert settings.as_dict()["diarization_max_speakers"] == 12


def test_a_malformed_speaker_count_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_env()

    monkeypatch.setenv("DIARIZATION_MAX_SPEAKERS", "several")
    with pytest.raises(ValueError, match="integer"):
        Settings.from_env()


def test_transcription_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.transcription_model == DEFAULT_TRANSCRIPTION_MODEL
    assert defaults.transcription_compute_type == DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
    assert defaults.transcription_language is None

    monkeypatch.setenv("TRANSCRIPTION_MODEL", "medium")
    monkeypatch.setenv("TRANSCRIPTION_COMPUTE_TYPE", "int8")
    monkeypatch.setenv("TRANSCRIPTION_LANGUAGE", "AM")

    settings = Settings.from_env()
    assert settings.transcription_model == "medium"
    assert settings.transcription_compute_type == "int8"
    assert settings.transcription_language == "am"  # normalised to lower case

    payload = settings.as_dict()
    assert payload["transcription_model"] == "medium"
    assert payload["transcription_compute_type"] == "int8"
    assert payload["transcription_language"] == "am"


def test_translation_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.translation_model == DEFAULT_TRANSLATION_MODEL
    assert defaults.translation_base_url == DEFAULT_TRANSLATION_BASE_URL
    assert defaults.translation_batch_size == DEFAULT_TRANSLATION_BATCH_SIZE
    assert defaults.translation_disable_thinking is True

    monkeypatch.setenv("TRANSLATION_MODEL", "deepseek-reasoner")
    monkeypatch.setenv("TRANSLATION_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("TRANSLATION_BATCH_SIZE", "4")
    monkeypatch.setenv("TRANSLATION_DISABLE_THINKING", "false")

    settings = Settings.from_env()
    assert settings.translation_model == "deepseek-reasoner"
    assert settings.translation_base_url == "https://example.test/v1"
    assert settings.translation_batch_size == 4
    assert settings.translation_disable_thinking is False

    payload = settings.as_dict()
    assert payload["translation_model"] == "deepseek-reasoner"
    assert payload["translation_base_url"] == "https://example.test/v1"
    assert payload["translation_batch_size"] == 4
    assert payload["translation_disable_thinking"] is False


def test_malformed_integer_setting_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TRANSLATION_BATCH_SIZE", "0")
    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_env()

    monkeypatch.setenv("TRANSLATION_BATCH_SIZE", "lots")
    with pytest.raises(ValueError, match="integer"):
        Settings.from_env()


def test_voice_profile_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.voice_profile_dir == defaults.work_dir / "voices"
    assert defaults.voice_reference_min_duration == DEFAULT_VOICE_REFERENCE_MIN_DURATION
    assert (
        defaults.voice_reference_target_duration == DEFAULT_VOICE_REFERENCE_TARGET_DURATION
    )
    assert defaults.voice_reference_max_duration == DEFAULT_VOICE_REFERENCE_MAX_DURATION

    monkeypatch.setenv("VOICE_PROFILE_DIR", "custom/voices")
    monkeypatch.setenv("VOICE_REFERENCE_MIN_DURATION", "2.5")
    monkeypatch.setenv("VOICE_REFERENCE_TARGET_DURATION", "8")
    monkeypatch.setenv("VOICE_REFERENCE_MAX_DURATION", "12.5")

    settings = Settings.from_env()
    assert settings.voice_profile_dir.name == "voices"
    assert settings.voice_profile_dir.parent.name == "custom"
    assert settings.voice_reference_min_duration == 2.5
    assert settings.voice_reference_target_duration == 8.0
    assert settings.voice_reference_max_duration == 12.5

    payload = settings.as_dict()
    assert payload["voice_reference_min_duration"] == 2.5
    assert payload["voice_reference_target_duration"] == 8.0
    assert payload["voice_reference_max_duration"] == 12.5
    assert str(payload["voice_profile_dir"]).endswith("voices")


def test_voice_profile_dir_follows_a_custom_work_dir(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("WORK_DIR", "custom-work")

    settings = Settings.from_env()

    assert settings.voice_profile_dir == settings.work_dir / "voices"


def test_the_dialogue_bible_path_follows_the_work_directory(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """Consistency state is per-film working data, so it lives with the work dir."""

    defaults = Settings.from_env()
    assert defaults.dialogue_bible_path == (
        defaults.work_dir / DEFAULT_DIALOGUE_BIBLE_FILENAME
    )

    monkeypatch.setenv("WORK_DIR", "scratch")
    monkeypatch.setenv("DIALOGUE_BIBLE_PATH", "scratch/bible.json")

    settings = Settings.from_env()
    assert settings.dialogue_bible_path.name == "bible.json"
    assert settings.as_dict()["dialogue_bible_path"].endswith("bible.json")


def test_tts_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    defaults = Settings.from_env()
    assert defaults.tts_model == DEFAULT_TTS_MODEL
    assert defaults.seed_vc_repo_path == (
        defaults.model_cache_dir / DEFAULT_SEED_VC_REPO_NAME
    )
    assert defaults.seed_vc_diffusion_steps == DEFAULT_SEED_VC_DIFFUSION_STEPS
    # Timbre-only by default: Seed-VC's style branch would re-impose the reference's
    # (English) accent and overwrite the take's performance.
    assert defaults.seed_vc_convert_style is False
    assert (
        defaults.tts_performance_reference_min_duration
        == DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION
    )
    assert (
        defaults.tts_performance_reference_max_duration
        == DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION
    )
    assert defaults.tts_max_pause_seconds == DEFAULT_TTS_MAX_PAUSE_SECONDS
    assert defaults.tts_min_line_seconds == DEFAULT_TTS_MIN_LINE_SECONDS
    # A failure should be visible, so absorbing one is opt-in.
    assert defaults.tts_continue_on_failure is False

    monkeypatch.setenv("TTS_MODEL", "someone/other-adapter")
    monkeypatch.setenv("SEED_VC_REPO_PATH", "vendor/seed-vc")
    monkeypatch.setenv("SEED_VC_DIFFUSION_STEPS", "12")
    monkeypatch.setenv("SEED_VC_CONVERT_STYLE", "on")
    monkeypatch.setenv("TTS_PERFORMANCE_REFERENCE_MIN_DURATION", "4")
    monkeypatch.setenv("TTS_PERFORMANCE_REFERENCE_MAX_DURATION", "8.5")
    monkeypatch.setenv("TTS_MAX_PAUSE_SECONDS", "1.5")
    monkeypatch.setenv("TTS_MIN_LINE_SECONDS", "0.45")
    monkeypatch.setenv("TTS_CONTINUE_ON_FAILURE", "true")

    settings = Settings.from_env()
    assert settings.tts_model == "someone/other-adapter"
    assert settings.seed_vc_repo_path.name == "seed-vc"
    assert settings.seed_vc_repo_path.parent.name == "vendor"
    assert settings.seed_vc_diffusion_steps == 12
    assert settings.seed_vc_convert_style is True
    assert settings.tts_performance_reference_min_duration == 4.0
    assert settings.tts_performance_reference_max_duration == 8.5
    assert settings.tts_max_pause_seconds == 1.5
    assert settings.tts_min_line_seconds == 0.45
    assert settings.tts_continue_on_failure is True

    payload = settings.as_dict()
    assert payload["tts_model"] == "someone/other-adapter"
    assert payload["seed_vc_diffusion_steps"] == 12
    assert payload["seed_vc_convert_style"] is True
    assert payload["tts_performance_reference_min_duration"] == 4.0
    assert payload["tts_performance_reference_max_duration"] == 8.5
    assert payload["tts_max_pause_seconds"] == 1.5
    assert payload["tts_min_line_seconds"] == 0.45
    assert payload["tts_continue_on_failure"] is True
    assert str(payload["seed_vc_repo_path"]).endswith("seed-vc")


def test_seed_vc_repo_path_follows_a_custom_model_cache(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("MODEL_CACHE_DIR", "custom-cache")

    settings = Settings.from_env()

    assert settings.seed_vc_repo_path == settings.model_cache_dir / "seed-vc"


def test_malformed_seed_vc_diffusion_steps_are_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("SEED_VC_DIFFUSION_STEPS", "0")
    with pytest.raises(ValueError, match="positive integer"):
        Settings.from_env()

    monkeypatch.setenv("SEED_VC_DIFFUSION_STEPS", "many")
    with pytest.raises(ValueError, match="integer"):
        Settings.from_env()


def test_malformed_float_setting_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("VOICE_REFERENCE_TARGET_DURATION", "longish")
    with pytest.raises(ValueError, match="must be a number"):
        Settings.from_env()

    monkeypatch.setenv("VOICE_REFERENCE_TARGET_DURATION", "0")
    with pytest.raises(ValueError, match="positive number"):
        Settings.from_env()

    monkeypatch.setenv("VOICE_REFERENCE_TARGET_DURATION", "inf")
    with pytest.raises(ValueError, match="finite"):
        Settings.from_env()


def test_malformed_boolean_setting_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TRANSLATION_DISABLE_THINKING", "maybe")
    with pytest.raises(ValueError, match="boolean flag"):
        Settings.from_env()


def test_timing_settings_are_configurable(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TIMING_MIN_TEMPO", "0.9")
    monkeypatch.setenv("TIMING_MAX_TEMPO", "1.1")

    settings = Settings.from_env()

    assert settings.timing_min_tempo == 0.9
    assert settings.timing_max_tempo == 1.1
    payload = settings.as_dict()
    assert payload["timing_min_tempo"] == 0.9
    assert payload["timing_max_tempo"] == 1.1


def test_timing_defaults_allow_both_directions(
    _clean_env: None,
) -> None:
    settings = Settings.from_env()

    assert settings.timing_min_tempo == DEFAULT_TIMING_MIN_TEMPO
    assert settings.timing_max_tempo == DEFAULT_TIMING_MAX_TEMPO
    assert settings.timing_min_tempo < 1.0 < settings.timing_max_tempo


def test_a_malformed_tempo_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("TIMING_MAX_TEMPO", "fast")
    with pytest.raises(ValueError, match="must be a number"):
        Settings.from_env()

    monkeypatch.setenv("TIMING_MAX_TEMPO", "0")
    with pytest.raises(ValueError, match="positive number"):
        Settings.from_env()


def test_mix_settings_accept_signed_levels(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    """A level has to be able to be zero or negative, unlike a duration."""

    monkeypatch.setenv("MIX_DIALOGUE_GAIN_DB", "-3.5")
    monkeypatch.setenv("MIX_DUCK_DB", "0")

    settings = Settings.from_env()

    assert settings.mix_dialogue_gain_db == -3.5
    assert settings.mix_duck_db == 0.0
    payload = settings.as_dict()
    assert payload["mix_dialogue_gain_db"] == -3.5
    assert payload["mix_duck_db"] == 0.0


def test_mix_defaults_are_sane(_clean_env: None) -> None:
    settings = Settings.from_env()

    assert settings.mix_dialogue_gain_db == DEFAULT_MIX_DIALOGUE_GAIN_DB
    assert settings.mix_duck_db == DEFAULT_MIX_DUCK_DB
    assert settings.mix_duck_db > 0.0


def test_a_malformed_mix_level_is_rejected(
    monkeypatch: pytest.MonkeyPatch, _clean_env: None
) -> None:
    monkeypatch.setenv("MIX_DUCK_DB", "quiet please")
    with pytest.raises(ValueError, match="must be a number"):
        Settings.from_env()

    monkeypatch.setenv("MIX_DUCK_DB", "inf")
    with pytest.raises(ValueError, match="finite"):
        Settings.from_env()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_the_syllable_prior_matches_the_adaptation_prompt() -> None:
    """Two copies of one prior, so they must not drift apart."""

    from app.pipeline.dialogue_context import DEFAULT_SYLLABLES_PER_SECOND

    assert (
        DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND == DEFAULT_SYLLABLES_PER_SECOND
    )


def test_the_default_length_penalty_leaves_the_model_alone() -> None:
    """Only a line that will not fit should be searched for brevity."""

    assert DEFAULT_TRANSLATION_LENGTH_PENALTY == 1.0
    assert 0.0 < DEFAULT_TRANSLATION_SHORTEN_PENALTY < 1.0


def test_the_line_gap_is_positive_and_small() -> None:
    """It exists to keep two voices apart without giving up usable silence."""

    assert 0.0 < DEFAULT_TIMING_MIN_LINE_GAP <= 0.5
