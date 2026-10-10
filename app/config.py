"""Central configuration for the Amharic dubbing pipeline.

All runtime configuration is read from environment variables so that the exact
same code runs locally and on the RunPod GPU worker without any changes.

Design rules
------------
* A local ``.env`` file is loaded when present, but it never overrides variables
  that are already defined in the real environment. This lets the RunPod worker
  environment inject values that win over a developer's local file.
* Secrets (``DEEPSEEK_API_KEY``, ``HUGGINGFACE_TOKEN``) are optional. The
  project must stay runnable without them so that the scaffold, the health
  check, and the tests work before any credentials exist.
* Secrets are *never* hard-coded and are *never* logged or printed.

Environment variables
---------------------
``DEEPSEEK_API_KEY``  API key for the DeepSeek dialogue adaptation/translation.
``HUGGINGFACE_TOKEN`` Token used to download gated Hugging Face models
                      (e.g. pyannote speaker diarization).
``INPUT_DIR``         Directory holding source videos. Defaults to ``data/input``.
``WORK_DIR``          Scratch directory for intermediate artifacts.
``OUTPUT_DIR``        Directory for final dubbed videos and manifests.
``MODEL_CACHE_DIR``   Where model weights are downloaded at runtime.
``DIARIZATION_MODEL`` Hugging Face pipeline id used for speaker diarization.
``TRANSCRIPTION_MODEL``
                      faster-whisper model used for transcription.
``TRANSCRIPTION_COMPUTE_TYPE``
                      CTranslate2 compute type (e.g. ``float16`` on GPU).
``TRANSCRIPTION_LANGUAGE``
                      Source language code; unset means detect automatically.
``TRANSLATION_MODEL`` DeepSeek model used for dialogue adaptation.
``TRANSLATION_BASE_URL``
                      DeepSeek API base URL (OpenAI-compatible endpoint).
``TRANSLATION_BATCH_SIZE``
                      Consecutive dialogue lines adapted in one request.
``TRANSLATION_DISABLE_THINKING``
                      Set ``false`` if the API rejects the thinking toggle.
``TRANSLATION_ENFORCE_BUDGET`` / ``TRANSLATION_ENFORCE_FIDEL_LOANWORDS``
                      Whether a line that comes back over its syllable budget, or
                      with a borrowed word left in Roman script, is sent back once
                      to be fixed. Both on by default.
``VOICE_PROFILE_DIR`` Directory holding per-speaker voice profiles.
``DIALOGUE_BIBLE_PATH``
                      Persistent character/consistency state (names, address
                      forms, register, recurring spellings) handed to dialogue
                      adaptation. Absent means an empty bible.
``VOICE_REFERENCE_MIN_DURATION`` / ``..._TARGET_DURATION`` / ``..._MAX_DURATION``
                      Preferred length of a voice-cloning reference, in seconds.
``TTS_MODEL``         Amharic speech adapter used by the TTS stage.
``SEED_VC_REPO_PATH`` Seed-VC checkout used for the identity-conversion step.
``SEED_VC_DIFFUSION_STEPS``
                      Diffusion steps of the Seed-VC V2 converter.
``SEED_VC_CONVERT_STYLE``
                      Whether Seed-VC V2 converts the *reference's* accent and
                      style as well as its timbre. Off by default: see
                      :mod:`app.pipeline.tts` for why that is the correct mode
                      for a voice-matched dub.
``DIARIZATION_MIN_SPEAKERS`` / ``DIARIZATION_MAX_SPEAKERS``
                      Optional bounds on how many speakers diarization may find.
                      A cast list turns these into the strongest single guard
                      against a character splitting into several clusters.
``TTS_PERFORMANCE_REFERENCE_MIN_DURATION`` / ``..._MAX_DURATION``
                      Length of the original-performance prompt handed to
                      Chatterbox, in seconds.
``TTS_MAX_PAUSE_SECONDS``
                      Longest pause rendered around a synthesized line.
``TTS_MIN_LINE_SECONDS``
                      Shortest original window that can be dubbed; shorter lines
                      are skipped and reported rather than sent to the engine.
``TTS_CONTINUE_ON_FAILURE``
                      Skip and report a line an engine *fails* on instead of ending
                      the run. Off by default, because a failure should be visible.
``DEVICE``            Compute device hint, ``cuda`` by default.
``LOG_LEVEL``         Logging verbosity, ``INFO`` by default.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:  # ``python-dotenv`` is part of the base requirements.
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps the module importable without it
    load_dotenv = None  # type: ignore[assignment]


#: Repository root (the directory that contains ``app/``).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "input"
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "working"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "output"
DEFAULT_MODEL_CACHE_DIR = PROJECT_ROOT / "models_cache"

#: pyannote Community-1 speaker diarization pipeline on the Hugging Face Hub.
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-community-1"

#: Optional bounds on the number of speakers diarization may report, or ``None``
#: to let the model decide. Supplying them is the strongest single guard against
#: a character splitting into several clusters or merging with another, so a cast
#: count - or a generous range around one - is worth configuring for a film.
DEFAULT_DIARIZATION_MIN_SPEAKERS: Optional[int] = None
DEFAULT_DIARIZATION_MAX_SPEAKERS: Optional[int] = None

#: faster-whisper model used to transcribe the dialogue stem.
DEFAULT_TRANSCRIPTION_MODEL = "large-v3"

#: CTranslate2 compute type. ``float16`` targets the A40 GPU; CPU runs need
#: ``int8`` or ``float32``, because CTranslate2 does not support fp16 on CPU.
DEFAULT_TRANSCRIPTION_COMPUTE_TYPE = "float16"

#: Whether a run round-trips its own dialogue through an Amharic ASR model to measure
#: pronunciation. Off by default: it loads a second model and transcribes every clip.
#: It is the only automated check that can see content the speech model invented - a
#: stray word at the start of a line, a fragment of the English prompt the voice was
#: cloned from, a mispronunciation - because none of that is the film's original audio,
#: so no comparison with the original can find it.
DEFAULT_QC_PRONUNCIATION = False

#: Which translation backend :mod:`app.pipeline.translation` uses.
#:
#: ``"openai"`` (the default) talks to any OpenAI-compatible endpoint - DeepSeek by
#: default - and is what provides *adaptation* rather than translation: scene and
#: character context, a syllable budget that a line can be sent back to meet, the
#: borrowed-word policy, and per-line performance metadata.
#:
#: ``"nllb"`` translates locally with Meta's NLLB-200 and needs no key. It is kept
#: because it is genuinely the stronger *translator* on measured Amharic - it beats every
#: open instruction model on this direction - but it cannot be instructed, and that turns
#: out to matter more for dubbing than translation quality does: NLLB cannot be told to
#: be brief, to keep a borrowed word in Fidel, or to match a character. Measured on a real
#: run it produced Amharic needing **1.66x** the time available, with no mechanism to
#: shorten it, which forced the timing stage to cut 31 of 38 lines short.
DEFAULT_TRANSLATION_BACKEND = "openai"

#: NLLB-200 at 3.3B parameters.
#:
#: Chosen over the distilled 1.3B on measured Amharic, not on size. On FLORES-200
#: (chrF++, eng->amh) 3.3B scores 37.9 against 37.2 for both the 1.3B and 600M
#: checkpoints, and on AFRIDOC-MT's document-level metric the gap is wider: 52.2
#: against 49.3. Document-level context is the part that matters for dialogue, and
#: NLLB is the only model family with published Amharic numbers that beat every open
#: instruction-following model tried on it (AFRIDOC-MT measured Gemma-2-9B at 6.5
#: d-chrF on this direction, and AfriScience-MT put the best open LLM about 27 points
#: of SSA-COMET behind NLLB).
#:
#: It costs ~6.6 GB in bf16, which the target 48 GB card carries without displacing
#: anything else. Amharic (``amh_Ethi``) has real parallel data behind it in NLLB-200.
#: The weights are CC-BY-NC-4.0.
DEFAULT_TRANSLATION_MODEL = "facebook/nllb-200-3.3B"

#: The model the instruction-following backend asks for by default. Measured against the
#: test film's real 42-line transcript, ``deepseek-v4-pro`` produced 426 syllables where
#: ``deepseek-flash`` produced 462 and the previous prompt produced 486, and it got the
#: short lines right where flash did not ("We're done." -> ጨረስን rather than ጨረቃ, "moon").
#: Fitting a dub is a hard constraint-following job, and the stronger model is the one
#: that honours it. A local OpenAI-compatible server needs whatever it serves instead,
#: set through ``TRANSLATION_MODEL``.
DEFAULT_TRANSLATION_OPENAI_MODEL = "deepseek-v4-pro"

#: Beam search width. ``1`` is greedy, which is deterministic - the same line always
#: translates the same way, so two runs can be compared against one another. Raise it
#: when quality matters more than reproducibility, at a proportional cost in time.
DEFAULT_TRANSLATION_NUM_BEAMS = 1

#: Longest output NLLB may generate for one chunk, in tokens. 512 is comfortably
#: above any spoken line; it exists to bound a runaway generation rather than to
#: shape it.
DEFAULT_TRANSLATION_MAX_NEW_TOKENS = 512

#: NLLB's decoding preference over the *length* of its output. ``1.0`` is the model's own
#: balance; below it the search favours shorter renderings, above it longer ones.
#:
#: This matters more here than in ordinary translation. Amharic does not express the same
#: idea in the same number of syllables as English, so a faithful translation of film
#: dialogue routinely needs about twice the time the English line occupied - measured on a
#: real run, the delivered speech came to 117s against 63s of original window. A dub can
#: recover about a third of that from the silence between lines, and the rest has to come
#: out of the text. NLLB cannot be *told* to be brief, but it can be *searched* for
#: brevity, and this is that control. Left at the model's own default; see
#: :data:`DEFAULT_TRANSLATION_SHORTEN_PENALTY` for the second pass.
DEFAULT_TRANSLATION_LENGTH_PENALTY = 1.0

#: The length penalty a line is re-translated with when it will not fit the time it has.
#: Below 1.0, because the only thing wrong with the first attempt is that it is too long.
#: The shorter rendering is used only if it is actually shorter and still non-empty, and
#: the choice is reported, so this trades a little literalness for a line that can be
#: spoken in the time available.
DEFAULT_TRANSLATION_SHORTEN_PENALTY = 0.6

#: Syllables per second an Amharic performer delivers, used to turn a span of time into a
#: syllable budget. Measured at about 4.6 syllables/second on real synthesized output, and
#: kept a little slower than that on purpose: asking a line to fit when it comfortably
#: can is harmless, while asking one to fit when it cannot leads to it being cut.
#: ``app.pipeline.dialogue_context.DEFAULT_SYLLABLES_PER_SECOND`` holds the same prior
#: for the adaptation prompt, and a test keeps the two in step.
DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND = 4.0

#: How many times a line may be sent back to be shortened, including the first attempt.
#: More than one is needed: a single rewrite left every over-long line still over at
#: roughly 1.6-1.8x its budget, because a model asked to fit a small window returns
#: something *closer* rather than something inside it. Bounded, because an unbounded loop
#: would eventually fit the window by losing the meaning.
DEFAULT_TRANSLATION_MAX_REDUCTION_ATTEMPTS = 3

#: DeepSeek API endpoint. The official endpoint is OpenAI-compatible.
DEFAULT_TRANSLATION_BASE_URL = "https://api.deepseek.com"

#: How many consecutive dialogue lines are adapted in a single API request.
#: Small enough to stay well inside the context window and keep a rejection
#: cheap, large enough for the model to follow who is answering whom.
DEFAULT_TRANSLATION_BATCH_SIZE = 10

#: Whether a line that comes back too long for its window, or with a borrowed word
#: left in Roman script, is sent back once to be fixed. Both are on by default: the
#: first is how a line is made to fit without being stretched, and the second is how
#: a word gets into a script the Amharic voice can actually read. Turn them off to
#: send every line exactly once, which is cheaper and is what a comparison run wants.
DEFAULT_TRANSLATION_ENFORCE_BUDGET = True
DEFAULT_TRANSLATION_ENFORCE_FIDEL_LOANWORDS = True

#: Directory holding the per-speaker voice profiles consumed by the TTS stage.
DEFAULT_VOICE_PROFILE_DIR = DEFAULT_WORK_DIR / "voices"

#: Persistent consistency state for dialogue adaptation: who the characters are,
#: how they address one another, and how their recurring names and terms are
#: spelled. Kept beside the voice profiles because both are per-film state that
#: outlives a single stage - unlike an artifact, it is meant to be edited by hand
#: between runs and reused, which is what stops a character drifting over 90-180
#: minutes. A missing file is an empty bible, not an error.
DEFAULT_DIALOGUE_BIBLE_FILENAME = "dialogue_bible.json"
DEFAULT_DIALOGUE_BIBLE_PATH = DEFAULT_WORK_DIR / DEFAULT_DIALOGUE_BIBLE_FILENAME

#: Preferred length of a voice-cloning reference, in seconds. The target is the
#: window the selector aims for, the minimum rejects snippets too short to clone
#: from, and the maximum caps how much audio a single profile keeps around.
DEFAULT_VOICE_REFERENCE_MIN_DURATION = 3.0
DEFAULT_VOICE_REFERENCE_TARGET_DURATION = 10.0
DEFAULT_VOICE_REFERENCE_MAX_DURATION = 15.0

#: Amharic speech engine consumed by :mod:`app.pipeline.tts`.
#:
#: ``"omnivoice"`` is k2-fsa's OmniVoice: a zero-shot model covering hundreds of
#: languages that clones a voice from a short reference and accepts a per-line rate.
#: Cloning from each character's own audio is what keeps one voice per character for a
#: whole film, and the rate is what lets a line be asked for at the length its window
#: allows instead of being stretched into it.
#: ``"chatterbox"`` is the Chatterbox Multilingual v3 + Amharic LoRA pair, which does
#: per-character voices by prompting with the original actor's audio and converting the
#: timbre afterwards. It has the strongest *published* Amharic numbers of the three
#: (a held-out character error rate of 0.095 and speaker similarity of 0.860), so it is
#: kept as the measured alternative to OmniVoice's unmeasured one.
#: ``"mms"`` is Meta's MMS-TTS Amharic: a VITS model trained on Amharic alone. It is a
#: **single-speaker** model - one voice for the whole film, with no cloning and no
#: per-character identity - and it needs its text romanised first (see the module).
DEFAULT_TTS_ENGINE = "omnivoice"

#: The Amharic TTS checkpoint. Which model this id should name depends on
#: ``TTS_ENGINE``: an MMS-TTS checkpoint for ``mms``. The ``chatterbox`` and
#: ``omnivoice`` engines name their models through their own settings.
DEFAULT_TTS_MODEL = "facebook/mms-tts-amh"

#: The Chatterbox Amharic adapter, used when ``TTS_ENGINE=chatterbox``. It is a LoRA
#: delta plus a Fidel tokenizer applied on top of Chatterbox Multilingual v3 at
#: runtime, so only the adapter is named here.
DEFAULT_CHATTERBOX_MODEL = "gabar-tech/chatterbox-amharic"

#: The OmniVoice checkpoint used when ``TTS_ENGINE=omnivoice``.
#:
#: ``k2-fsa/OmniVoice`` is the base model: hundreds of languages, Amharic among them,
#: zero-shot cloning from a short reference. It is the safer default of the two
#: OmniVoice options because it is a maintained release with a paper and a package
#: behind it. What is *not* established is how well its Amharic performs: its language
#: list gives Amharic about 12.8 hours of the data it was trained on, and no Amharic
#: evaluation has been published for it. Treat any Amharic claim here as unmeasured.
#:
#: ``african-low-resource/omnivoice-amharic`` (also published as
#: ``Lab-et/omnivoice-amharic``, same weights) is Amharic-only and trained on far more
#: Amharic audio, so it is worth an A/B - but its model card reports every evaluation
#: metric as "TBD", ships no samples, and names no datasets, so its quality is
#: unverified in both directions. It is not the default for that reason.
DEFAULT_OMNIVOICE_MODEL = "k2-fsa/OmniVoice"

#: Diffusion steps for OmniVoice. The model's own default is 32; 16 is the documented
#: faster setting. Quality is traded for speed here, so the default is left alone.
DEFAULT_OMNIVOICE_STEPS = 32

#: Classifier-free guidance scale. The model's default is 2.0.
DEFAULT_OMNIVOICE_GUIDANCE_SCALE = 2.0

#: Precision for the OmniVoice weights. ``float16`` halves the footprint; the model
#: card documents it, so it is what a run uses.
DEFAULT_OMNIVOICE_TORCH_DTYPE = "float16"

#: Sample rate MMS-TTS produces, in Hz. VITS models are trained per language and this
#: one is trained at 16 kHz, so the pipeline resamples its output rather than
#: pretending it is the 24 kHz Chatterbox produces.
DEFAULT_MMS_SAMPLE_RATE = 16_000

#: Fixed seed for the MMS duration predictor. VITS samples its rhythm, so the same
#: line would otherwise come out a slightly different length every run - which would
#: make a dub unreproducible and a baseline meaningless.
DEFAULT_MMS_SEED = 0

#: Speaking rate asked of the MMS duration predictor. ``1.0`` is the model's natural
#: pace, and the value reaches the predictor through the forward call that reads it.
#: It sets one delivery speed for the whole film rather than fitting each line: a line
#: is still fitted to its own window by ``timing.py``.
DEFAULT_MMS_SPEAKING_RATE = 1.0

#: Name of the Seed-VC checkout inside ``MODEL_CACHE_DIR``. Seed-VC is not
#: published as a package, so the identity-conversion step of
#: :mod:`app.pipeline.tts` runs it from a checkout of its repository.
DEFAULT_SEED_VC_REPO_NAME = "seed-vc"

#: Seed-VC V2 diffusion steps. The V2 inference script's own default is 30;
#: fewer steps trade quality for speed.
DEFAULT_SEED_VC_DIFFUSION_STEPS = 30

#: Whether Seed-VC V2 converts the *reference's* accent and style along with its
#: timbre. Left **off**, which is the only mode that preserves the performance of
#: the take: Seed-VC V2's style branch conditions its autoregressive stage on the
#: reference's acoustic tokens and content indices, so switching it on speaks the
#: source content in the *reference's* style. With a character reference taken
#: from the original actor's English audio, that re-imposes an English accent on
#: the Amharic and discards the performance the Chatterbox prompt carried over.
DEFAULT_SEED_VC_CONVERT_STYLE = False

#: Preferred length of the original-performance prompt handed to Chatterbox, in
#: seconds. The Amharic adapter clones from roughly ten seconds of audio, so a
#: short line is padded with its own surrounding dialogue and a long monologue
#: is trimmed, both around the line that is being synthesized.
DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION = 6.0
DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION = 12.0

#: Longest pause rendered before and after a synthesized line, in seconds. The
#: dialogue model estimates the pauses of a scene and normally stays below 1.5
#: seconds; this bound keeps a wildly wrong estimate from becoming a hole in the
#: dub.
DEFAULT_TTS_MAX_PAUSE_SECONDS = 2.0

#: Shortest original window that can be dubbed, in seconds. Below this the original
#: is a fragment - a breath, a click, a sliver of a mis-diarized turn - rather than a
#: spoken line: there is no room for a word in the time it occupied, any Amharic
#: written for it is unintelligible, and the speech engine can fail on it outright.
#: Chatterbox does, with an empty mel spectrogram that trips a convolution inside
#: its vocoder, so such a line must be skipped and reported rather than attempted.
DEFAULT_TTS_MIN_LINE_SECONDS = 0.30

#: Whether a line an engine *fails* on is recorded and skipped instead of ending the
#: run. Off by default, because hiding a failure is worse than stopping: a stage error
#: names the line, and a dub with a hole in it is only acceptable when it was asked
#: for. Set it on a long run where finishing matters more than a perfect first pass.
DEFAULT_TTS_CONTINUE_ON_FAILURE = False

#: Whether the TTS stage asks a cloning engine for a per-line length before the timing
#: stage falls back to stretching. Asking a model for a duration changes how the line is
#: *spoken*; time-stretching changes audio that has already been spoken, which is why
#: the first is worth doing whenever an engine can honour it. Exposed as a flag so a
#: comparison run can switch it off and attribute a difference to it.
DEFAULT_TTS_REQUEST_RATE = True

#: How far a line's delivery may be time-stretched to fit the window of the
#: original line, as a tempo factor. A factor below 1 slows the line down, above
#: 1 speeds it up, and both are pitch-preserving. The band is deliberately narrow:
#: a line that needs more than this is reported as not fitting rather than being
#: mangled to fit. FFmpeg's ``atempo`` filter only accepts factors between 0.5 and
#: 2.0, which bounds any configuration here.
#:
#: The ceiling is 1.45 rather than 1.25 because it was measured, not chosen. Replaying
#: the failed run's own line timings through the placement policy below, with the
#: Amharic the adaptation stage produced for this project, the worst line landed after
#: the actor spoke by:
#:
#: * 1.25 - 9.24s (the old default, with strict separation)
#: * 1.35 - 6.13s
#: * 1.45 - 4.96s
#: * 1.55 - 3.98s
#:
#: Below 1.25 the lines have nowhere to go and the whole dub falls behind the picture;
#: past 1.45 the delivery starts to sound hurried for what is left to gain. This is
#: deliberately a *band*, not a target: a line that fits is left at its natural pace,
#: and only the lines that cannot fit are delivered faster. FFmpeg's ``atempo`` filter
#: only accepts factors between 0.5 and 2.0, which bounds any configuration here.
DEFAULT_TIMING_MIN_TEMPO = 0.80
DEFAULT_TIMING_MAX_TEMPO = 1.45

#: How far a line may run into the next one before the next one is moved instead, in
#: seconds.
#:
#: Two voices a fraction of a second apart is what a conversation already sounds like -
#: the listener hears a natural hand-over (and the window includes the next line's own
#: leading pause, so the *speech* overlap is shorter still). Two voices *seconds* apart
#: is unintelligible, and moving a line that far breaks lip-sync, so the overrun is paid
#: first in a small overlap and only then in position. Measured with the real timing
#: stage on the failed run's own numbers, the worst line's drift falls from 6.13s to
#: 4.13s when this goes from 0.30 to 0.45.
DEFAULT_TIMING_MAX_OVERLAP_SECONDS = 0.45

#: Silence kept between one dubbed line and the next, in seconds.
#:
#: A line may run past its own original window into the silence that follows it, which is
#: what keeps a language longer than English from being squashed, but it must stop this
#: far short of the next line. Below the guard, two voices are audible at once and the
#: dub stops being intelligible; the guard is small because the silence between film
#: lines is usually short and worth using.
DEFAULT_TIMING_MIN_LINE_GAP = 0.12

#: Whether a line that will not fit its available time is cut short, with a fade, so that
#: it never overlaps the next line.
#:
#: **Off by default, deliberately.** Cutting is the one thing that damages the
#: performance: the listener hears a word clipped off the end, and on a real run where
#: the text was too long it happened to 31 of 38 lines, which sounds broken rather than
#: fast. The pipeline's answer to a line that runs long is to make the *text* fit, and
#: that is settled at the adaptation stage - see
#: :data:`DEFAULT_TRANSLATION_SHORTEN_PENALTY`. When the text fits, this never fires and
#: the setting is irrelevant. When it does not, the run says so instead: a line that
#: overruns is reported, and is heard as a brief overlap rather than as a missing word.
#: Set it true to trade a clipped word for a guaranteed gap between lines.
DEFAULT_TIMING_TRIM_TO_FIT = False

#: Level of the dubbed dialogue in the final mix, in dB relative to the clips the
#: TTS stage produced. Never a boost by default: the dialogue was generated at a
#: consistent level, so raising it here would only invite clipping.
DEFAULT_MIX_DIALOGUE_GAIN_DB = 0.0

#: How far the music and effects are ducked while dialogue is playing, in dB. The
#: reduction is smoothed, so it follows the dialogue instead of gating it, and it
#: is applied to the bed only - never to the dialogue itself.
DEFAULT_MIX_DUCK_DB = 6.0


def load_env_file(path: Optional[Path] = None) -> None:
    """Load a ``.env`` file without overriding existing environment variables.

    Missing files and a missing ``python-dotenv`` installation are both ignored
    on purpose: the project has to start even when nothing is configured.
    """

    if load_dotenv is None:
        return

    env_path = Path(path) if path is not None else PROJECT_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)


def _translation_model_default() -> str:
    """Return the model to use when ``TRANSLATION_MODEL`` is not set.

    The two backends name completely different things: a local NLLB *checkpoint path* or
    the model a *served* endpoint answers to. A single default would silently ask DeepSeek
    for a model called ``facebook/nllb-200-3.3B`` - which fails - so the default follows
    the configured backend.
    """

    backend = (
        _read_env("TRANSLATION_BACKEND", DEFAULT_TRANSLATION_BACKEND)
        or DEFAULT_TRANSLATION_BACKEND
    ).strip().lower()
    if backend == "nllb":
        return DEFAULT_TRANSLATION_MODEL
    return DEFAULT_TRANSLATION_OPENAI_MODEL


def _read_env(name: str, default: Optional[str] = None) -> Optional[str]:
    """Return a trimmed environment value, falling back to ``default``."""

    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _read_path(name: str, default: Path) -> Path:
    """Return an environment value as a :class:`~pathlib.Path`.

    Relative paths are resolved against the project root so that behaviour does
    not depend on the current working directory.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_int(name: str, default: int) -> int:
    """Return an environment value as a positive integer, or ``default``.

    A malformed value is a configuration mistake and fails loudly rather than
    being silently replaced by the default.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def _read_nonnegative_int(name: str, default: int) -> int:
    """Return an environment value as a non-negative integer, or ``default``.

    Zero is a legitimate value here where :func:`_read_int` would reject it: a seed
    of zero is an ordinary seed.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 0:
        raise ValueError(f"{name} must not be negative, got {value}")
    return value


def _read_optional_int(name: str, default: Optional[int]) -> Optional[int]:
    """Return an environment value as an optional positive integer.

    Unlike :func:`_read_int`, an unset variable is a legitimate value here: the
    speaker-count bounds of diarization are optional, and "not configured" has to
    stay distinguishable from any number.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc

    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


#: Values accepted (case-insensitively) for a boolean environment flag.
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_FALSE_VALUES = frozenset({"0", "false", "no", "off"})


def _read_bool(name: str, default: bool) -> bool:
    """Return an environment value as a boolean, or ``default``."""

    raw = _read_env(name)
    if raw is None:
        return default

    lowered = raw.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean flag, got {raw!r}")


def _read_float(name: str, default: float) -> float:
    """Return an environment value as a finite positive float, or ``default``."""

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    if value <= 0.0:
        raise ValueError(f"{name} must be a positive number, got {value}")
    return value


def _read_signed_float(name: str, default: float) -> float:
    """Return an environment value as a finite float of either sign, or ``default``.

    Levels are naturally signed - a gain of ``0`` means "leave it alone" and a
    negative one attenuates - so they cannot use :func:`_read_float`.
    """

    raw = _read_env(name)
    if raw is None:
        return default

    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc

    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {raw!r}")
    return value


@dataclass(frozen=True)
class Settings:
    """Resolved, immutable runtime settings for the dubbing pipeline."""

    input_dir: Path
    work_dir: Path
    output_dir: Path
    model_cache_dir: Path
    deepseek_api_key: Optional[str] = None
    huggingface_token: Optional[str] = None
    device: str = "cuda"
    log_level: str = "INFO"
    #: Hugging Face pipeline id used by :mod:`app.pipeline.diarization`.
    diarization_model: str = DEFAULT_DIARIZATION_MODEL
    #: Optional speaker-count bounds for diarization; ``None`` lets it decide.
    diarization_min_speakers: Optional[int] = DEFAULT_DIARIZATION_MIN_SPEAKERS
    diarization_max_speakers: Optional[int] = DEFAULT_DIARIZATION_MAX_SPEAKERS
    #: faster-whisper model and CTranslate2 compute type for transcription.
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    transcription_compute_type: str = DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
    #: Round-trip every delivered clip through an Amharic ASR model and report the
    #: character error rate against the text that was synthesized. Off by default
    #: because it loads a second model and transcribes every clip.
    qc_pronunciation: bool = DEFAULT_QC_PRONUNCIATION
    #: Source language code for transcription; ``None`` detects it automatically.
    transcription_language: Optional[str] = None
    #: DeepSeek dialogue adaptation settings for :mod:`app.pipeline.translation`.
    #: Dialogue adaptation / translation settings for
    #: :mod:`app.pipeline.translation`.
    translation_backend: str = DEFAULT_TRANSLATION_BACKEND
    translation_model: str = DEFAULT_TRANSLATION_MODEL
    translation_base_url: str = DEFAULT_TRANSLATION_BASE_URL
    translation_batch_size: int = DEFAULT_TRANSLATION_BATCH_SIZE
    translation_disable_thinking: bool = True
    translation_num_beams: int = DEFAULT_TRANSLATION_NUM_BEAMS
    translation_max_new_tokens: int = DEFAULT_TRANSLATION_MAX_NEW_TOKENS
    translation_length_penalty: float = DEFAULT_TRANSLATION_LENGTH_PENALTY
    translation_shorten_penalty: float = DEFAULT_TRANSLATION_SHORTEN_PENALTY
    translation_syllables_per_second: float = DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND
    translation_max_reduction_attempts: int = (
        DEFAULT_TRANSLATION_MAX_REDUCTION_ATTEMPTS
    )
    translation_enforce_budget: bool = DEFAULT_TRANSLATION_ENFORCE_BUDGET
    translation_enforce_fidel_loanwords: bool = (
        DEFAULT_TRANSLATION_ENFORCE_FIDEL_LOANWORDS
    )
    #: Voice-profile settings for :mod:`app.pipeline.voice_profiles`.
    voice_profile_dir: Path = DEFAULT_VOICE_PROFILE_DIR
    voice_reference_min_duration: float = DEFAULT_VOICE_REFERENCE_MIN_DURATION
    voice_reference_target_duration: float = DEFAULT_VOICE_REFERENCE_TARGET_DURATION
    voice_reference_max_duration: float = DEFAULT_VOICE_REFERENCE_MAX_DURATION
    #: Persistent character/consistency state handed to dialogue adaptation.
    dialogue_bible_path: Path = DEFAULT_DIALOGUE_BIBLE_PATH
    #: TTS settings for :mod:`app.pipeline.tts`: the Amharic speech adapter, the
    #: Seed-VC checkout that converts identity, and the performance-prompt and
    #: pause bounds of one synthesized line.
    tts_model: str = DEFAULT_TTS_MODEL
    tts_engine: str = DEFAULT_TTS_ENGINE
    chatterbox_model: str = DEFAULT_CHATTERBOX_MODEL
    omnivoice_model: str = DEFAULT_OMNIVOICE_MODEL
    omnivoice_steps: int = DEFAULT_OMNIVOICE_STEPS
    omnivoice_guidance_scale: float = DEFAULT_OMNIVOICE_GUIDANCE_SCALE
    omnivoice_torch_dtype: str = DEFAULT_OMNIVOICE_TORCH_DTYPE
    mms_sample_rate: int = DEFAULT_MMS_SAMPLE_RATE
    mms_seed: int = DEFAULT_MMS_SEED
    mms_speaking_rate: float = DEFAULT_MMS_SPEAKING_RATE
    seed_vc_repo_path: Path = DEFAULT_MODEL_CACHE_DIR / DEFAULT_SEED_VC_REPO_NAME
    seed_vc_diffusion_steps: int = DEFAULT_SEED_VC_DIFFUSION_STEPS
    seed_vc_convert_style: bool = DEFAULT_SEED_VC_CONVERT_STYLE
    tts_performance_reference_min_duration: float = (
        DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION
    )
    tts_performance_reference_max_duration: float = (
        DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION
    )
    tts_max_pause_seconds: float = DEFAULT_TTS_MAX_PAUSE_SECONDS
    #: Shortest original window that can be dubbed at all; shorter lines are skipped
    #: and reported instead of being sent to the engine.
    tts_min_line_seconds: float = DEFAULT_TTS_MIN_LINE_SECONDS
    #: Whether a line an engine fails on is skipped and reported, rather than ending
    #: the run. Off by default: a failure should be visible, not absorbed.
    tts_continue_on_failure: bool = DEFAULT_TTS_CONTINUE_ON_FAILURE
    tts_request_rate: bool = DEFAULT_TTS_REQUEST_RATE
    #: Timing settings for :mod:`app.pipeline.timing`: the tempo band a line may
    #: be stretched within to fit the time it has, and the silence kept before
    #: the next line so two voices are never heard at once.
    timing_min_tempo: float = DEFAULT_TIMING_MIN_TEMPO
    timing_max_tempo: float = DEFAULT_TIMING_MAX_TEMPO
    timing_max_overlap_seconds: float = DEFAULT_TIMING_MAX_OVERLAP_SECONDS
    timing_min_line_gap: float = DEFAULT_TIMING_MIN_LINE_GAP
    timing_trim_to_fit: bool = DEFAULT_TIMING_TRIM_TO_FIT
    #: Mix settings for :mod:`app.pipeline.mixing`: the dialogue level and how far
    #: the music and effects are ducked under it.
    mix_dialogue_gain_db: float = DEFAULT_MIX_DIALOGUE_GAIN_DB
    mix_duck_db: float = DEFAULT_MIX_DUCK_DB

    # -- construction -------------------------------------------------------
    @classmethod
    def from_env(cls) -> "Settings":
        """Build a :class:`Settings` instance from the environment."""

        load_env_file()
        work_dir = _read_path("WORK_DIR", DEFAULT_WORK_DIR)
        model_cache_dir = _read_path("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)
        return cls(
            input_dir=_read_path("INPUT_DIR", DEFAULT_INPUT_DIR),
            work_dir=work_dir,
            output_dir=_read_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
            model_cache_dir=model_cache_dir,
            deepseek_api_key=_read_env("DEEPSEEK_API_KEY"),
            huggingface_token=_read_env("HUGGINGFACE_TOKEN"),
            device=(_read_env("DEVICE", "cuda") or "cuda").lower(),
            log_level=(_read_env("LOG_LEVEL", "INFO") or "INFO").upper(),
            diarization_model=(
                _read_env("DIARIZATION_MODEL", DEFAULT_DIARIZATION_MODEL)
                or DEFAULT_DIARIZATION_MODEL
            ),
            diarization_min_speakers=_read_optional_int(
                "DIARIZATION_MIN_SPEAKERS", DEFAULT_DIARIZATION_MIN_SPEAKERS
            ),
            diarization_max_speakers=_read_optional_int(
                "DIARIZATION_MAX_SPEAKERS", DEFAULT_DIARIZATION_MAX_SPEAKERS
            ),
            transcription_model=(
                _read_env("TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL)
                or DEFAULT_TRANSCRIPTION_MODEL
            ),
            transcription_compute_type=(
                _read_env("TRANSCRIPTION_COMPUTE_TYPE", DEFAULT_TRANSCRIPTION_COMPUTE_TYPE)
                or DEFAULT_TRANSCRIPTION_COMPUTE_TYPE
            ),
            qc_pronunciation=_read_bool(
                "QC_PRONUNCIATION", DEFAULT_QC_PRONUNCIATION
            ),
            transcription_language=(
                (_read_env("TRANSCRIPTION_LANGUAGE") or "").lower() or None
            ),
            translation_backend=(
                _read_env("TRANSLATION_BACKEND", DEFAULT_TRANSLATION_BACKEND)
                or DEFAULT_TRANSLATION_BACKEND
            ).lower(),
            translation_model=(
                _read_env("TRANSLATION_MODEL") or _translation_model_default()
            ),
            translation_base_url=(
                _read_env("TRANSLATION_BASE_URL", DEFAULT_TRANSLATION_BASE_URL)
                or DEFAULT_TRANSLATION_BASE_URL
            ),
            translation_batch_size=_read_int(
                "TRANSLATION_BATCH_SIZE", DEFAULT_TRANSLATION_BATCH_SIZE
            ),
            translation_disable_thinking=_read_bool("TRANSLATION_DISABLE_THINKING", True),
            translation_num_beams=_read_int(
                "TRANSLATION_NUM_BEAMS", DEFAULT_TRANSLATION_NUM_BEAMS
            ),
            translation_max_new_tokens=_read_int(
                "TRANSLATION_MAX_NEW_TOKENS", DEFAULT_TRANSLATION_MAX_NEW_TOKENS
            ),
            translation_length_penalty=_read_float(
                "TRANSLATION_LENGTH_PENALTY", DEFAULT_TRANSLATION_LENGTH_PENALTY
            ),
            translation_shorten_penalty=_read_float(
                "TRANSLATION_SHORTEN_PENALTY", DEFAULT_TRANSLATION_SHORTEN_PENALTY
            ),
            translation_syllables_per_second=_read_float(
                "TRANSLATION_SYLLABLES_PER_SECOND",
                DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND,
            ),
            translation_max_reduction_attempts=_read_int(
                "TRANSLATION_MAX_REDUCTION_ATTEMPTS",
                DEFAULT_TRANSLATION_MAX_REDUCTION_ATTEMPTS,
            ),
            translation_enforce_budget=_read_bool(
                "TRANSLATION_ENFORCE_BUDGET", DEFAULT_TRANSLATION_ENFORCE_BUDGET
            ),
            translation_enforce_fidel_loanwords=_read_bool(
                "TRANSLATION_ENFORCE_FIDEL_LOANWORDS",
                DEFAULT_TRANSLATION_ENFORCE_FIDEL_LOANWORDS,
            ),
            voice_profile_dir=_read_path("VOICE_PROFILE_DIR", work_dir / "voices"),
            voice_reference_min_duration=_read_float(
                "VOICE_REFERENCE_MIN_DURATION", DEFAULT_VOICE_REFERENCE_MIN_DURATION
            ),
            voice_reference_target_duration=_read_float(
                "VOICE_REFERENCE_TARGET_DURATION", DEFAULT_VOICE_REFERENCE_TARGET_DURATION
            ),
            voice_reference_max_duration=_read_float(
                "VOICE_REFERENCE_MAX_DURATION", DEFAULT_VOICE_REFERENCE_MAX_DURATION
            ),
            dialogue_bible_path=_read_path(
                "DIALOGUE_BIBLE_PATH", work_dir / DEFAULT_DIALOGUE_BIBLE_FILENAME
            ),
            tts_model=(_read_env("TTS_MODEL", DEFAULT_TTS_MODEL) or DEFAULT_TTS_MODEL),
            tts_engine=(
                _read_env("TTS_ENGINE", DEFAULT_TTS_ENGINE) or DEFAULT_TTS_ENGINE
            ).lower(),
            chatterbox_model=(
                _read_env("CHATTERBOX_MODEL", DEFAULT_CHATTERBOX_MODEL)
                or DEFAULT_CHATTERBOX_MODEL
            ),
            omnivoice_model=(
                _read_env("OMNIVOICE_MODEL", DEFAULT_OMNIVOICE_MODEL)
                or DEFAULT_OMNIVOICE_MODEL
            ),
            omnivoice_steps=_read_int("OMNIVOICE_STEPS", DEFAULT_OMNIVOICE_STEPS),
            omnivoice_guidance_scale=_read_float(
                "OMNIVOICE_GUIDANCE_SCALE", DEFAULT_OMNIVOICE_GUIDANCE_SCALE
            ),
            omnivoice_torch_dtype=(
                _read_env("OMNIVOICE_TORCH_DTYPE", DEFAULT_OMNIVOICE_TORCH_DTYPE)
                or DEFAULT_OMNIVOICE_TORCH_DTYPE
            ),
            mms_sample_rate=_read_int("MMS_SAMPLE_RATE", DEFAULT_MMS_SAMPLE_RATE),
            mms_seed=_read_nonnegative_int("MMS_SEED", DEFAULT_MMS_SEED),
            mms_speaking_rate=_read_float(
                "MMS_SPEAKING_RATE", DEFAULT_MMS_SPEAKING_RATE
            ),
            seed_vc_repo_path=_read_path(
                "SEED_VC_REPO_PATH", model_cache_dir / DEFAULT_SEED_VC_REPO_NAME
            ),
            seed_vc_diffusion_steps=_read_int(
                "SEED_VC_DIFFUSION_STEPS", DEFAULT_SEED_VC_DIFFUSION_STEPS
            ),
            seed_vc_convert_style=_read_bool(
                "SEED_VC_CONVERT_STYLE", DEFAULT_SEED_VC_CONVERT_STYLE
            ),
            tts_performance_reference_min_duration=_read_float(
                "TTS_PERFORMANCE_REFERENCE_MIN_DURATION",
                DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION,
            ),
            tts_performance_reference_max_duration=_read_float(
                "TTS_PERFORMANCE_REFERENCE_MAX_DURATION",
                DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION,
            ),
            tts_max_pause_seconds=_read_float(
                "TTS_MAX_PAUSE_SECONDS", DEFAULT_TTS_MAX_PAUSE_SECONDS
            ),
            tts_min_line_seconds=_read_float(
                "TTS_MIN_LINE_SECONDS", DEFAULT_TTS_MIN_LINE_SECONDS
            ),
            tts_continue_on_failure=_read_bool(
                "TTS_CONTINUE_ON_FAILURE", DEFAULT_TTS_CONTINUE_ON_FAILURE
            ),
            tts_request_rate=_read_bool(
                "TTS_REQUEST_RATE", DEFAULT_TTS_REQUEST_RATE
            ),
            timing_min_tempo=_read_float("TIMING_MIN_TEMPO", DEFAULT_TIMING_MIN_TEMPO),
            timing_max_tempo=_read_float("TIMING_MAX_TEMPO", DEFAULT_TIMING_MAX_TEMPO),
            timing_max_overlap_seconds=_read_float(
                "TIMING_MAX_OVERLAP_SECONDS", DEFAULT_TIMING_MAX_OVERLAP_SECONDS
            ),
            timing_min_line_gap=_read_float(
                "TIMING_MIN_LINE_GAP", DEFAULT_TIMING_MIN_LINE_GAP
            ),
            timing_trim_to_fit=_read_bool(
                "TIMING_TRIM_TO_FIT", DEFAULT_TIMING_TRIM_TO_FIT
            ),
            mix_dialogue_gain_db=_read_signed_float(
                "MIX_DIALOGUE_GAIN_DB", DEFAULT_MIX_DIALOGUE_GAIN_DB
            ),
            mix_duck_db=_read_signed_float("MIX_DUCK_DB", DEFAULT_MIX_DUCK_DB),
        )

    # -- helpers ------------------------------------------------------------
    @property
    def has_deepseek_credentials(self) -> bool:
        """``True`` when a DeepSeek API key is configured."""

        return bool(self.deepseek_api_key)

    @property
    def has_huggingface_credentials(self) -> bool:
        """``True`` when a Hugging Face token is configured."""

        return bool(self.huggingface_token)

    def missing_credentials(self) -> list[str]:
        """Return the names of the credentials the *configured* run still needs.

        The Hugging Face token is always needed, because the diarization pipeline it
        authenticates is gated. The DeepSeek key belongs to the instruction-following
        translation backend alone: under ``TRANSLATION_BACKEND=nllb`` the run never
        reads it, so reporting it as missing would hold up a session that is ready to
        go.
        """

        missing: list[str] = []
        if self.translation_backend == "openai" and not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not self.huggingface_token:
            missing.append("HUGGINGFACE_TOKEN")
        return missing

    def ensure_directories(self) -> None:
        """Create the input/working/output/model-cache directories if needed."""

        for directory in (
            self.input_dir,
            self.work_dir,
            self.output_dir,
            self.model_cache_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def as_dict(self) -> dict[str, object]:
        """Return a redacted, JSON-safe view of the settings (never secrets)."""

        return {
            "input_dir": str(self.input_dir),
            "work_dir": str(self.work_dir),
            "output_dir": str(self.output_dir),
            "model_cache_dir": str(self.model_cache_dir),
            "deepseek_api_key_set": self.has_deepseek_credentials,
            "huggingface_token_set": self.has_huggingface_credentials,
            "device": self.device,
            "log_level": self.log_level,
            "diarization_model": self.diarization_model,
            "diarization_min_speakers": self.diarization_min_speakers,
            "diarization_max_speakers": self.diarization_max_speakers,
            "transcription_model": self.transcription_model,
            "transcription_compute_type": self.transcription_compute_type,
            "qc_pronunciation": self.qc_pronunciation,
            "transcription_language": self.transcription_language,
            "translation_backend": self.translation_backend,
            "translation_model": self.translation_model,
            "translation_base_url": self.translation_base_url,
            "translation_batch_size": self.translation_batch_size,
            "translation_disable_thinking": self.translation_disable_thinking,
            "translation_num_beams": self.translation_num_beams,
            "translation_max_new_tokens": self.translation_max_new_tokens,
            "translation_length_penalty": self.translation_length_penalty,
            "translation_shorten_penalty": self.translation_shorten_penalty,
            "translation_syllables_per_second": self.translation_syllables_per_second,
            "translation_max_reduction_attempts": (
                self.translation_max_reduction_attempts
            ),
            "translation_enforce_budget": self.translation_enforce_budget,
            "translation_enforce_fidel_loanwords": (
                self.translation_enforce_fidel_loanwords
            ),
            "voice_profile_dir": str(self.voice_profile_dir),
            "voice_reference_min_duration": self.voice_reference_min_duration,
            "voice_reference_target_duration": self.voice_reference_target_duration,
            "voice_reference_max_duration": self.voice_reference_max_duration,
            "dialogue_bible_path": str(self.dialogue_bible_path),
            "tts_model": self.tts_model,
            "tts_engine": self.tts_engine,
            "chatterbox_model": self.chatterbox_model,
            "omnivoice_model": self.omnivoice_model,
            "omnivoice_steps": self.omnivoice_steps,
            "omnivoice_guidance_scale": self.omnivoice_guidance_scale,
            "omnivoice_torch_dtype": self.omnivoice_torch_dtype,
            "mms_sample_rate": self.mms_sample_rate,
            "mms_seed": self.mms_seed,
            "mms_speaking_rate": self.mms_speaking_rate,
            "seed_vc_repo_path": str(self.seed_vc_repo_path),
            "seed_vc_diffusion_steps": self.seed_vc_diffusion_steps,
            "seed_vc_convert_style": self.seed_vc_convert_style,
            "tts_performance_reference_min_duration": (
                self.tts_performance_reference_min_duration
            ),
            "tts_performance_reference_max_duration": (
                self.tts_performance_reference_max_duration
            ),
            "tts_max_pause_seconds": self.tts_max_pause_seconds,
            "tts_min_line_seconds": self.tts_min_line_seconds,
            "tts_continue_on_failure": self.tts_continue_on_failure,
            "tts_request_rate": self.tts_request_rate,
            "timing_min_tempo": self.timing_min_tempo,
            "timing_max_tempo": self.timing_max_tempo,
            "timing_max_overlap_seconds": self.timing_max_overlap_seconds,
            "timing_min_line_gap": self.timing_min_line_gap,
            "timing_trim_to_fit": self.timing_trim_to_fit,
            "mix_dialogue_gain_db": self.mix_dialogue_gain_db,
            "mix_duck_db": self.mix_duck_db,
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide, cached :class:`Settings` instance."""

    return Settings.from_env()


__all__ = [
    "DEFAULT_CHATTERBOX_MODEL",
    "DEFAULT_OMNIVOICE_GUIDANCE_SCALE",
    "DEFAULT_OMNIVOICE_MODEL",
    "DEFAULT_OMNIVOICE_STEPS",
    "DEFAULT_OMNIVOICE_TORCH_DTYPE",
    "DEFAULT_DIALOGUE_BIBLE_FILENAME",
    "DEFAULT_DIALOGUE_BIBLE_PATH",
    "DEFAULT_DIARIZATION_MAX_SPEAKERS",
    "DEFAULT_DIARIZATION_MIN_SPEAKERS",
    "DEFAULT_DIARIZATION_MODEL",
    "DEFAULT_MIX_DIALOGUE_GAIN_DB",
    "DEFAULT_MIX_DUCK_DB",
    "DEFAULT_MMS_SAMPLE_RATE",
    "DEFAULT_MMS_SEED",
    "DEFAULT_MMS_SPEAKING_RATE",
    "DEFAULT_SEED_VC_CONVERT_STYLE",
    "DEFAULT_SEED_VC_DIFFUSION_STEPS",
    "DEFAULT_SEED_VC_REPO_NAME",
    "DEFAULT_TRANSCRIPTION_COMPUTE_TYPE",
    "DEFAULT_QC_PRONUNCIATION",
    "DEFAULT_TRANSCRIPTION_MODEL",
    "DEFAULT_TRANSLATION_BACKEND",
    "DEFAULT_TRANSLATION_MAX_NEW_TOKENS",
    "DEFAULT_TRANSLATION_NUM_BEAMS",
    "DEFAULT_TRANSLATION_BASE_URL",
    "DEFAULT_TRANSLATION_BATCH_SIZE",
    "DEFAULT_TRANSLATION_ENFORCE_BUDGET",
    "DEFAULT_TRANSLATION_ENFORCE_FIDEL_LOANWORDS",
    "DEFAULT_TRANSLATION_LENGTH_PENALTY",
    "DEFAULT_TRANSLATION_MODEL",
    "DEFAULT_TRANSLATION_MAX_REDUCTION_ATTEMPTS",
    "DEFAULT_TRANSLATION_OPENAI_MODEL",
    "DEFAULT_TRANSLATION_SHORTEN_PENALTY",
    "DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND",
    "DEFAULT_TTS_CONTINUE_ON_FAILURE",
    "DEFAULT_TTS_MAX_PAUSE_SECONDS",
    "DEFAULT_TTS_MIN_LINE_SECONDS",
    "DEFAULT_TTS_REQUEST_RATE",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_TTS_PERFORMANCE_REFERENCE_MAX_DURATION",
    "DEFAULT_TTS_PERFORMANCE_REFERENCE_MIN_DURATION",
    "DEFAULT_TIMING_MAX_OVERLAP_SECONDS",
    "DEFAULT_TIMING_MAX_TEMPO",
    "DEFAULT_TIMING_MIN_LINE_GAP",
    "DEFAULT_TIMING_TRIM_TO_FIT",
    "DEFAULT_TIMING_MIN_TEMPO",
    "DEFAULT_VOICE_PROFILE_DIR",
    "DEFAULT_VOICE_REFERENCE_MAX_DURATION",
    "DEFAULT_VOICE_REFERENCE_MIN_DURATION",
    "DEFAULT_VOICE_REFERENCE_TARGET_DURATION",
    "PROJECT_ROOT",
    "Settings",
    "get_settings",
    "load_env_file",
]
