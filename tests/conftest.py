"""Shared fixtures for the test suite.

Both fixtures exist because :class:`app.config.Settings` reads the process environment
*and* the developer's ``.env`` file. A test that builds settings from the environment has
to neutralise both, or it passes or fails depending on whose machine it runs on. They
live in ``conftest.py`` rather than in one test module so every test file that touches
configuration gets the same isolation.
"""

from __future__ import annotations

import pytest

_ENV_VARS = (
    "INPUT_DIR",
    "WORK_DIR",
    "OUTPUT_DIR",
    "MODEL_CACHE_DIR",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "TRANSLATION_API_KEY",
    "HUGGINGFACE_TOKEN",
    "DIARIZATION_MODEL",
    "DIARIZATION_MIN_SPEAKERS",
    "DIARIZATION_MAX_SPEAKERS",
    "TRANSCRIPTION_MODEL",
    "TRANSCRIPTION_COMPUTE_TYPE",
    "TRANSCRIPTION_LANGUAGE",
    "TRANSLATION_BACKEND",
    "TRANSLATION_PROVIDER",
    "TRANSLATION_MODEL",
    "TRANSLATION_BASE_URL",
    "TRANSLATION_BATCH_SIZE",
    "TRANSLATION_THINKING",
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
    """Remove every variable the application reads, so defaults are the only input."""

    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
