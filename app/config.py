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
``CROSSTALK_SEPARATION``
                      Set to ``1`` to separate a crosstalk region into one track per
                      character before transcribing it. Off by default; unavailable
                      models/tokens fall back to refusing the recovery.
``CROSSTALK_SEPARATION_BACKEND``
                      ``pyannote`` (default) or ``speechbrain``.
``CROSSTALK_SEPARATION_MODEL``
                      Checkpoint id; defaults to the backend's own model.
``CROSSTALK_SEPARATION_CONTEXT_SECONDS``
                      Audio kept on each side of the region when the model runs.
``TRANSLATION_BACKEND``
                      ``nllb`` for local translation, or ``openai`` (the default) for
                      an instruction-following model served over an OpenAI-compatible
                      endpoint.
``TRANSLATION_PROVIDER``
                      Which vendor the served backend talks to: ``gemini`` (the
                      default), ``deepseek``, or ``other`` for a server you name
                      yourself. Decides the endpoint, the model, the key variable and
                      how reasoning is requested.
``TRANSLATION_MODEL`` The model the served endpoint answers to; follows the provider.
``TRANSLATION_BASE_URL``
                      Endpoint base URL; follows the provider, required for ``other``.
``TRANSLATION_BATCH_SIZE``
                      Consecutive dialogue lines adapted in one request.
``TRANSLATION_THINKING``
                      ``off``/``low``/``medium``/``high``, mapped onto whatever the
                      provider understands. Reported as what was actually sent.
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
``VOICE_BANK_DIR``    A built native-Amharic voice bank, used as the fallback for a
                      character whose film audio cannot supply a clean reference.
                      The actor's own reference stays the identity for everyone
                      else. Absent means no fallback is available.
``VOICE_BANK_MAP``    Optional ``SPEAKER_00=voice-id`` list pinning specific bank
                      voices to specific fallback-cast characters.
``TTS_MODEL``         Amharic speech adapter used by the TTS stage.``DIARIZATION_MIN_SPEAKERS`` / ``DIARIZATION_MAX_SPEAKERS``
                      Optional bounds on how many speakers diarization may find.
                      A cast list turns these into the strongest single guard
                      against a character splitting into several clusters.
``TTS_MAX_PAUSE_SECONDS``
                      Longest pause rendered around a synthesized line.
``TTS_MIN_LINE_SECONDS``
                      Shortest original window that can be dubbed; shorter lines
                      are skipped and reported rather than sent to the engine.
``TTS_CONTINUE_ON_FAILURE``
                      Skip and report a line an engine *fails* on instead of ending
                      the run. Off by default, because a failure should be visible.
``DEVICE``            Compute device hint, ``cuda`` by default.
``ALLOW_TORCH_COMPILE``
                      Set to ``1`` to re-enable TorchInductor. Compilation is off
                      by default because a pod whose PyTorch CUDA build predates
                      its GPU aborts at NVRTC compile time.
``LOG_LEVEL``         Logging verbosity, ``INFO`` by default.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
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

#: Whether a crosstalk region is separated into one track per character before it is
#: transcribed. Off by default: it loads another model and only helps on films where
#: characters talk over each other. When it is off - or when its model, checkpoint or
#: token is unavailable - the pipeline keeps its previous behaviour and refuses a
#: recovery it cannot attribute, rather than failing the run.
DEFAULT_CROSSTALK_SEPARATION = False

#: ``pyannote`` uses pyannote.audio's SpeechSeparation pipeline; ``speechbrain`` uses
#: SepFormer. Both are blind separators; the per-character association is made from
#: the registers the pipeline already measured.
DEFAULT_CROSSTALK_SEPARATION_BACKEND = "pyannote"

#: Which model family separates the film into dialogue and a music/effects bed.
#:
#: ``roformer`` is BS-RoFormer, the most accurate of the two and the default; it costs
#: roughly 1.3x real time on a small workstation GPU, so an hour and a quarter for a
#: feature-length episode. ``demucs`` is htdemucs, a lighter four-stem model: usually
#: faster, somewhat less clean at pulling a voice out of a music bed.
DEFAULT_SEPARATION_BACKEND = "roformer"
SEPARATION_BACKENDS = ("roformer", "demucs")

#: htdemucs checkpoint used by the ``demucs`` backend.
DEFAULT_SEPARATION_DEMUCS_MODEL = "htdemucs"

#: Whether the separator runs its forward passes in half precision. Halves the memory
#: traffic and is typically 1.5-2x faster, but a model that is unstable in fp16 returns
#: a quiet or hollow stem instead of failing, so it stays off until it has been
#: measured on the hardware in use.
DEFAULT_SEPARATION_AUTOCAST = False

#: Gated PixIT checkpoint for the ``pyannote`` backend. The token's account must have
#: accepted the model's conditions on Hugging Face.
DEFAULT_CROSSTALK_SEPARATION_MODEL = "pyannote/speech-separation-ami-1.0"

#: Alternative, ungated checkpoint for the ``speechbrain`` backend.
SPEECHBRAIN_SEPARATION_MODEL = "speechbrain/sepformer-whamr16k"

#: Seconds of audio kept on each side of a crosstalk region for the separator. A
#: model cannot learn two voices from 0.2 s of overlap; the tracks are cropped back
#: to the region afterwards.
DEFAULT_CROSSTALK_SEPARATION_CONTEXT_SECONDS = 1.0


def default_crosstalk_separation_model(backend: str) -> str:
    """Return the checkpoint a separation backend uses when none is configured."""

    if (backend or "").lower() == "speechbrain":
        return SPEECHBRAIN_SEPARATION_MODEL
    return DEFAULT_CROSSTALK_SEPARATION_MODEL

#: Which translation backend :mod:`app.pipeline.translation` uses.
#:
#: ``"openai"`` (the default) talks to any OpenAI-compatible endpoint - Gemini by
#: default, DeepSeek or a server you run yourself if configured - and is what provides
#: *adaptation* rather than translation: scene and character context, a syllable budget
#: that a line can be sent back to meet, the
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

#: Which vendor serves the instruction-following translation model. Both speak the
#: OpenAI chat-completions protocol, so the pipeline code is the same; the provider
#: decides three things that genuinely differ between them - the environment variable
#: the API key is read from, the default endpoint and model, and how "think harder"
#: is requested.
DEFAULT_TRANSLATION_PROVIDER = "gemini"

#: The model the Gemini provider asks for by default.
#:
#: ``gemini-3.8-flash`` is the current Flash model, and is free of charge on the free
#: tier. It is *not* an arbitrary pick: ``gemini-3.7-flash`` - the id this project
#: first reached for - is deprecated, and Google now routes requests for it to
#: ``gemini-3.8-flash`` automatically, so naming 3.8 explicitly is naming what
#: actually answers. Thinking is on by default at level ``medium``.
#:
#: Amharic evidence is thin and is stated as such in the README: there is no
#: Google-published Amharic MT benchmark. What justifies an instruction-following
#: model here is not raw translation quality but *instructability* - the adaptation
#: stage has to hit a syllable budget, keep a borrowed word in Fidel, and hold a
#: character consistent, none of which a sentence-level MT model can be asked for.
DEFAULT_TRANSLATION_GEMINI_MODEL = "gemini-3.8-flash"

#: Gemini's OpenAI-compatible endpoint. Documented by Google as beta; switching back
#: to another provider is a ``TRANSLATION_PROVIDER`` change and nothing else.
DEFAULT_TRANSLATION_GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)

#: How hard the served model is asked to think, one of ``off``, ``low``, ``medium``,
#: ``high``. The adaptation stage is the one place in this pipeline where reasoning
#: pays: it must read a scene, keep a character consistent, and hit a syllable count
#: that is often far below a literal rendering. ``off`` minimizes latency and cost;
#: ``medium`` or ``high`` may help with difficult dialogue.
#:
#: ``off`` is honest about being a request rather than a guarantee: Gemini 3 models
#: cannot disable reasoning, so on that provider it selects the lowest level available.
DEFAULT_TRANSLATION_THINKING = "off"

#: Retained for compatibility with existing environment files. Requests do not fail
#: over: a missing Gemini key selects DeepSeek before the first request, while any
#: request failure is surfaced immediately.
DEFAULT_TRANSLATION_FALLBACK_PROVIDER = "deepseek"

#: The model the other instruction-following provider asks for by default. Measured
#: against the test film's real 42-line transcript, ``deepseek-v4-pro`` produced 426
#: syllables where ``deepseek-flash`` produced 462 and the older prompt produced 486,
#: and it got the short lines right where flash did not ("We're done." -> ጨረስን rather
#: than ጨረቃ, "moon"). Kept so the two providers can be compared on the same film.
DEFAULT_TRANSLATION_DEEPSEEK_MODEL = "deepseek-v4-pro"

#: DeepSeek's endpoint, for ``TRANSLATION_PROVIDER=deepseek``.
DEFAULT_TRANSLATION_DEEPSEEK_BASE_URL = "https://api.deepseek.com"

#: Kept as the name of the *served instruction-following backend's* default model, which
#: is what older code and documentation call it. New code should read
#: :data:`DEFAULT_TRANSLATION_GEMINI_MODEL` or :data:`DEFAULT_TRANSLATION_DEEPSEEK_MODEL`
#: according to the provider.
DEFAULT_TRANSLATION_OPENAI_MODEL = DEFAULT_TRANSLATION_GEMINI_MODEL

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
#: brevity. Left at the model's own default.
DEFAULT_TRANSLATION_LENGTH_PENALTY = 1.0

#: Legacy setting retained for compatibility. NLLB translation is single-pass and does
#: not perform a second inference to shorten an over-budget line.
DEFAULT_TRANSLATION_SHORTEN_PENALTY = 0.6

#: Syllables per second an Amharic performer delivers, used to turn a span of time into a
#: syllable budget. Measured at about 4.6 syllables/second on real synthesized output, and
#: kept a little slower than that on purpose: asking a line to fit when it comfortably
#: can is harmless, while asking one to fit when it cannot leads to it being cut.
#: ``app.pipeline.dialogue_context.DEFAULT_SYLLABLES_PER_SECOND`` holds the same prior
#: for the adaptation prompt, and a test keeps the two in step.
DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND = 4.0

#: Maximum number of rewrite requests for lines that miss the syllable budget or
#: contain Roman-script text. Zero keeps each batch to one hosted model call, avoiding
#: a second billable request; set to one to re-ask for targeted quality corrections.
DEFAULT_TRANSLATION_MAX_REDUCTION_ATTEMPTS = 0

#: The endpoint the *served* instruction-following backend talks to by default.
#: Kept under its original name for callers that already read it; the value follows the
#: default provider.
DEFAULT_TRANSLATION_BASE_URL = DEFAULT_TRANSLATION_GEMINI_BASE_URL

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

#: Cached, sourced character research for the film being dubbed. Kept separate from
#: the operator's bible: research is evidence, not a hand-written casting decision.
DEFAULT_CHARACTER_GUIDE_FILENAME = "character_guide.json"
DEFAULT_CHARACTER_GUIDE_PATH = DEFAULT_WORK_DIR / DEFAULT_CHARACTER_GUIDE_FILENAME

#: Cached, speaker-labelled episode script used to reconcile ASR lines with who
#: actually says them. Kept beside the guide and reused for the same episode.
DEFAULT_EPISODE_SCRIPT_FILENAME = "episode_script.json"
DEFAULT_EPISODE_SCRIPT_PATH = DEFAULT_WORK_DIR / DEFAULT_EPISODE_SCRIPT_FILENAME

#: Whether the context-aware AI layer runs at all. It needs a DeepSeek key and
#: degrades to no-ops when one is absent, so this is an escape hatch rather than a
#: hard dependency.
DEFAULT_CONTEXT_AI_ENABLED = True

#: Minimum model-reported confidence before a speaker correction is applied. The
#: evidence gate is checked as well; confidence alone never changes a label.
DEFAULT_CONTEXT_AI_CONFIDENCE_THRESHOLD = 0.75

#: Per-request timeout for character research, in seconds. Bounded so an unavailable
#: search or fetch endpoint cannot stall a run.
DEFAULT_CONTEXT_AI_TIMEOUT_SECONDS = 20.0

#: Timeout for script extraction, in seconds. A full episode transcript is a much
#: larger request than a character lookup, so it gets its own, longer bound.
DEFAULT_CONTEXT_AI_SCRIPT_TIMEOUT_SECONDS = 120.0

#: DeepSeek's documented Search and Fetch endpoints, relative to its base URL.
DEFAULT_DEEPSEEK_SEARCH_PATH = "/v1/tools/search"
DEFAULT_DEEPSEEK_FETCH_PATH = "/v1/tools/fetch"

#: Google Drive transfer through rclone. The episode is far too large to live in git,
#: so it comes down from Drive and the finished dub goes back up. ``rclone`` is the
#: only mover: it resumes, verifies and handles Drive's own download handshakes.
#: The remote name here must match a remote in the rclone config file.
DEFAULT_DRIVE_REMOTE = "gdrive"
#: Folder inside that remote holding source episodes, and one for finished dubs.
DEFAULT_DRIVE_SOURCE_FOLDER = ""
DEFAULT_DRIVE_OUTPUT_FOLDER = "amharic-dubs"
#: Where the rclone config lives on the worker. Kept outside the repository because
#: it holds an OAuth refresh token, and copied there once during setup.
DEFAULT_RCLONE_CONFIG_PATH = Path("/workspace/rclone.conf")
DEFAULT_RCLONE_BIN = "rclone"

#: Preferred length of a voice-cloning reference, in seconds. A short, clean
#: prompt limits the amount of reference dialogue OmniVoice can echo; longer
#: references remain available when no suitable shorter interval exists.
DEFAULT_VOICE_REFERENCE_MIN_DURATION = 4.0
DEFAULT_VOICE_REFERENCE_TARGET_DURATION = 5.0
DEFAULT_VOICE_REFERENCE_MAX_DURATION = 10.0

#: Legacy sample-voice configuration retained so existing environment files continue
#: to parse. The production pipeline no longer loads sample recordings; OmniVoice uses
#: each character's own clean, transcript-matched actor reference instead.
DEFAULT_VOICE_SAMPLE_DIR = PROJECT_ROOT / "data" / "voice-samples"

#: Legacy sample mapping, currently unused by the production pipeline.
DEFAULT_VOICE_SAMPLE_MAP = ""

#: Directory holding a built native-Amharic voice bank: a manifest of curated
#: human recordings plus their canonical clips. It is a **fallback**, not a casting
#: mode: a character keeps the actor's own reference unless the film cannot supply
#: one that is safe to clone (too little clean speech, only simultaneous speech, or
#: no window that is one voice). Those characters are given a native speaker's
#: reference instead, which removes the cross-lingual accent at its source and gives
#: up the actor's identity for that character only. It is opt-in: no manifest, no
#: fallback.
DEFAULT_VOICE_BANK_DIR = PROJECT_ROOT / "data" / "voice-bank"

#: Optional ``SPEAKER_00=voice-id`` list pinning bank voices to characters. It only
#: applies to characters the bank is actually used for - the ones the film could not
#: serve - and lets someone who has listened choose a particular voice for one of
#: them instead of accepting the automatic, pitch-matched pick.
DEFAULT_VOICE_BANK_VOICE_MAP = ""

#: Amharic speech engine consumed by :mod:`app.pipeline.tts`.
#:
#: ``"omnivoice"`` is k2-fsa's OmniVoice: a zero-shot model covering hundreds of
#: languages that clones a voice from a short reference and accepts a per-line rate.
#: Cloning from each character's own audio is what keeps one voice per character for a
#: whole film, and the rate is what lets a line be asked for at the length its window
#: allows instead of being stretched into it.
#: ``"mms"`` is Meta's MMS-TTS Amharic: a VITS model trained on Amharic alone. It is a
#: **single-speaker** model - one voice for the whole film, with no cloning and no
#: per-character identity - and it needs its text romanised first (see the module).
DEFAULT_TTS_ENGINE = "omnivoice"

#: The Amharic TTS checkpoint. Which model this id should name depends on
#: ``TTS_ENGINE``: an MMS-TTS checkpoint for ``mms``. The ``omnivoice`` engine names
#: its model through its own setting.
DEFAULT_TTS_MODEL = "facebook/mms-tts-amh"

#: The OmniVoice checkpoint used when ``TTS_ENGINE=omnivoice``.
#:
#: Amharic-tuned OmniVoice. The Lab-et page documents Amharic voice cloning; its
#: example loads the same weights through the canonical african-low-resource id.
#: Its published objective and human-evaluation metrics are still TBD, so quality
#: must be judged from generated output rather than assumed from the fine-tuning.
DEFAULT_OMNIVOICE_MODEL = "african-low-resource/omnivoice-amharic"

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
#: pretending it is at a rate the model never produced.
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

#: Longest pause rendered before and after a synthesized line, in seconds. The
#: dialogue model estimates the pauses of a scene and normally stays below 1.5
#: seconds; this bound keeps a wildly wrong estimate from becoming a hole in the
#: dub.
DEFAULT_TTS_MAX_PAUSE_SECONDS = 2.0

#: Shortest original window that can be dubbed, in seconds. Below this the original
#: is a fragment - a breath, a click, a sliver of a mis-diarized turn - rather than a
#: spoken line: there is no room for a word in the time it occupied, any Amharic
#: written for it is unintelligible, and the speech engine can fail on it outright.
#: A vocoder can be driven to an empty mel spectrogram by input that short, which
#: trips a convolution inside it, so such a line must be skipped and reported rather
#: than attempted.
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
#:
#: The value is a ceiling in two places, not one: it also caps an overlap the film
#: itself had, so a source crosstalk region is preserved up to this much and no
#: further. Below it, the source overlap is kept as it was.
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
#: fast. The pipeline asks for concise text in its first translation pass. When the text
#: fits, this never fires and the setting is irrelevant. When it does not, the run says
#: so instead: a line that
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


#: Whether :func:`apply_torch_compile_policy` installed ``TORCH_COMPILE_DISABLE``
#: itself, so a configuration that opts compilation back in can undo exactly that.
_compile_guard_installed = False


def apply_torch_compile_policy() -> None:
    """Keep TorchInductor off unless ``ALLOW_TORCH_COMPILE=1``.

    Inductor compiles kernels for the visible GPU's compute capability through
    NVRTC. On a pod whose PyTorch CUDA build predates the GPU, that compilation
    aborts with ``nvrtc: error: invalid value for --gpu-architecture (-arch)``
    even though eager CUDA inference itself works. No stage of this pipeline
    needs compilation, so it stays off; set ``ALLOW_TORCH_COMPILE=1`` to opt in.
    """

    global _compile_guard_installed

    if _read_env("ALLOW_TORCH_COMPILE", "0") == "1":
        if _compile_guard_installed:
            os.environ.pop("TORCH_COMPILE_DISABLE", None)
            _compile_guard_installed = False
        return

    if os.environ.get("TORCH_COMPILE_DISABLE") == "1":
        return

    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    _compile_guard_installed = True


def _translation_model_default() -> str:
    """Return the model to use when ``TRANSLATION_MODEL`` is not set.

    The two backends name completely different things: a local NLLB *checkpoint path* or
    the model a *served* endpoint answers to. A single default would silently ask Gemini
    for a model called ``facebook/nllb-200-3.3B`` - which fails - so the default follows
    the configured backend, and within the served backend it follows the provider.
    """

    backend = (
        _read_env("TRANSLATION_BACKEND", DEFAULT_TRANSLATION_BACKEND)
        or DEFAULT_TRANSLATION_BACKEND
    ).strip().lower()
    if backend == "nllb":
        return DEFAULT_TRANSLATION_MODEL
    return translation_provider_defaults(
        _read_env("TRANSLATION_PROVIDER", DEFAULT_TRANSLATION_PROVIDER)
        or DEFAULT_TRANSLATION_PROVIDER
    ).model


def _translation_base_url_default() -> str:
    """Return the endpoint to use when ``TRANSLATION_BASE_URL`` is not set.

    Follows the provider for the same reason the model does: the endpoint *is* the
    provider on an OpenAI-compatible backend, and a single default would quietly send
    Gemini's key to DeepSeek's host. Meaningless for the NLLB backend, which is local
    and talks to no endpoint at all.
    """

    return translation_provider_defaults(
        _read_env("TRANSLATION_PROVIDER", DEFAULT_TRANSLATION_PROVIDER)
        or DEFAULT_TRANSLATION_PROVIDER
    ).base_url


@dataclass(frozen=True)
class TranslationProvider:
    """What one served-model vendor decides, and nothing else.

    Keeping this to three fields is deliberate. Everything else about a request - the
    prompt, the JSON protocol, the batching, the syllable budgets, the rewrite passes -
    is the pipeline's and is identical for every provider, which is why adding a vendor
    does not touch the pipeline.
    """

    name: str
    base_url: str
    model: str
    api_key_env: str
    #: ``True`` when the vendor understands ``reasoning_effort``.
    supports_reasoning_effort: bool
    #: ``True`` when the vendor understands the ``thinking`` request body this project
    #: used for DeepSeek's hybrid models.
    supports_thinking_switch: bool


#: The vendors this project knows how to talk to. A local or self-hosted
#: OpenAI-compatible server is not in this table: name it with ``TRANSLATION_PROVIDER=other``
#: and set ``TRANSLATION_BASE_URL``, ``TRANSLATION_MODEL`` and its key yourself.
TRANSLATION_PROVIDERS: dict[str, TranslationProvider] = {
    "gemini": TranslationProvider(
        name="gemini",
        base_url=DEFAULT_TRANSLATION_GEMINI_BASE_URL,
        model=DEFAULT_TRANSLATION_GEMINI_MODEL,
        api_key_env="GEMINI_API_KEY",
        supports_reasoning_effort=True,
        supports_thinking_switch=False,
    ),
    "deepseek": TranslationProvider(
        name="deepseek",
        base_url=DEFAULT_TRANSLATION_DEEPSEEK_BASE_URL,
        model=DEFAULT_TRANSLATION_DEEPSEEK_MODEL,
        api_key_env="DEEPSEEK_API_KEY",
        supports_reasoning_effort=False,
        supports_thinking_switch=True,
    ),
    "other": TranslationProvider(
        name="other",
        # Deliberately blank: there is no sensible guess for a server this project has
        # never seen, and a *wrong* guess would send a film's requests to somebody
        # else's host. Empty values are refused at start-up with a named variable.
        base_url="",
        model="",
        api_key_env="TRANSLATION_API_KEY",
        supports_reasoning_effort=False,
        supports_thinking_switch=False,
    ),
}

#: Accepted values of ``TRANSLATION_THINKING``, weakest first.
TRANSLATION_THINKING_LEVELS: tuple[str, ...] = ("off", "low", "medium", "high")


def translation_provider_defaults(name: str) -> TranslationProvider:
    """Return the provider named by ``name``, or the default if it is unknown."""

    key = (name or "").strip().lower()
    return TRANSLATION_PROVIDERS.get(key, TRANSLATION_PROVIDERS[DEFAULT_TRANSLATION_PROVIDER])


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

    Zero is a legitimate value for settings that use it to disable an operation,
    such as translation rewrites, and for ordinary random seeds.
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
    #: API keys for the served translation providers. Only the configured provider's
    #: key is ever read; the other stays ``None`` and unused.
    gemini_api_key: Optional[str] = None
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
    #: Target-speaker extraction for crosstalk regions. See
    #: :mod:`app.pipeline.target_extraction`. Off by default; an unavailable model
    #: or token degrades to the previous refuse-if-unprovable behaviour.
    crosstalk_separation: bool = DEFAULT_CROSSTALK_SEPARATION
    crosstalk_separation_backend: str = DEFAULT_CROSSTALK_SEPARATION_BACKEND
    crosstalk_separation_model: str = DEFAULT_CROSSTALK_SEPARATION_MODEL
    crosstalk_separation_context_seconds: float = (
        DEFAULT_CROSSTALK_SEPARATION_CONTEXT_SECONDS
    )
    #: How the film is split into dialogue and a music/effects bed. See
    #: :mod:`app.pipeline.separation`. ``roformer`` is the accurate default;
    #: ``demucs`` is the lighter alternative.
    separation_backend: str = DEFAULT_SEPARATION_BACKEND
    separation_demucs_model: str = DEFAULT_SEPARATION_DEMUCS_MODEL
    separation_autocast: bool = DEFAULT_SEPARATION_AUTOCAST
    #: Round-trip every delivered clip through an Amharic ASR model and report the
    #: character error rate against the text that was synthesized. Off by default
    #: because it loads a second model and transcribes every clip.
    qc_pronunciation: bool = DEFAULT_QC_PRONUNCIATION
    #: Source language code for transcription; ``None`` detects it automatically.
    transcription_language: Optional[str] = None
    #: Dialogue adaptation / translation settings for
    #: :mod:`app.pipeline.translation`.
    translation_backend: str = DEFAULT_TRANSLATION_BACKEND
    #: Which vendor serves the instruction-following model. Decides the default endpoint,
    #: the default model, the environment variable the key is read from, and how thinking
    #: is requested - and nothing else.
    translation_provider: str = DEFAULT_TRANSLATION_PROVIDER
    translation_model: str = DEFAULT_TRANSLATION_GEMINI_MODEL
    translation_base_url: str = DEFAULT_TRANSLATION_GEMINI_BASE_URL
    translation_batch_size: int = DEFAULT_TRANSLATION_BATCH_SIZE
    #: ``off``/``low``/``medium``/``high``. See :data:`DEFAULT_TRANSLATION_THINKING`.
    translation_thinking: str = DEFAULT_TRANSLATION_THINKING
    #: Legacy setting retained for environment compatibility; request failures never
    #: trigger provider failover.
    translation_fallback_provider: str = DEFAULT_TRANSLATION_FALLBACK_PROVIDER
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
    #: Operator recordings used to build the fallback bank (``--cast-from-bank``), and
    #: an optional ``SPEAKER=file`` list. Never used as a cloning reference directly.
    voice_sample_dir: Path = DEFAULT_VOICE_SAMPLE_DIR
    voice_sample_map: str = DEFAULT_VOICE_SAMPLE_MAP
    #: Built native-Amharic voice bank, and an optional ``SPEAKER=voice-id`` list
    #: pinning specific voices. A manifest in this directory makes the fallback
    #: available; it is used only for a character the film cannot supply a reference
    #: for.
    voice_bank_dir: Path = DEFAULT_VOICE_BANK_DIR
    voice_bank_voice_map: str = DEFAULT_VOICE_BANK_VOICE_MAP
    #: Persistent character/consistency state handed to dialogue adaptation.
    dialogue_bible_path: Path = DEFAULT_DIALOGUE_BIBLE_PATH
    #: Cached character research and the settings that bound it. See
    #: :mod:`app.pipeline.context_ai`.
    character_guide_path: Path = DEFAULT_CHARACTER_GUIDE_PATH
    episode_script_path: Path = DEFAULT_EPISODE_SCRIPT_PATH
    context_ai_enabled: bool = DEFAULT_CONTEXT_AI_ENABLED
    context_ai_confidence_threshold: float = DEFAULT_CONTEXT_AI_CONFIDENCE_THRESHOLD
    context_ai_timeout_seconds: float = DEFAULT_CONTEXT_AI_TIMEOUT_SECONDS
    context_ai_script_timeout_seconds: float = (
        DEFAULT_CONTEXT_AI_SCRIPT_TIMEOUT_SECONDS
    )
    #: Google Drive transfer settings, used by ``scripts/drive_sync.py``.
    drive_remote: str = DEFAULT_DRIVE_REMOTE
    drive_source_folder: str = DEFAULT_DRIVE_SOURCE_FOLDER
    drive_output_folder: str = DEFAULT_DRIVE_OUTPUT_FOLDER
    rclone_config: Path = DEFAULT_RCLONE_CONFIG_PATH
    rclone_bin: str = DEFAULT_RCLONE_BIN
    #: TTS settings for :mod:`app.pipeline.tts`: the Amharic speech adapter and the
    #: pause bounds of one synthesized line.
    tts_model: str = DEFAULT_TTS_MODEL
    tts_engine: str = DEFAULT_TTS_ENGINE
    omnivoice_model: str = DEFAULT_OMNIVOICE_MODEL
    omnivoice_steps: int = DEFAULT_OMNIVOICE_STEPS
    omnivoice_guidance_scale: float = DEFAULT_OMNIVOICE_GUIDANCE_SCALE
    omnivoice_torch_dtype: str = DEFAULT_OMNIVOICE_TORCH_DTYPE
    mms_sample_rate: int = DEFAULT_MMS_SAMPLE_RATE
    mms_seed: int = DEFAULT_MMS_SEED
    mms_speaking_rate: float = DEFAULT_MMS_SPEAKING_RATE
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
        apply_torch_compile_policy()
        work_dir = _read_path("WORK_DIR", DEFAULT_WORK_DIR)
        model_cache_dir = _read_path("MODEL_CACHE_DIR", DEFAULT_MODEL_CACHE_DIR)
        crosstalk_separation_backend = (
            _read_env(
                "CROSSTALK_SEPARATION_BACKEND",
                DEFAULT_CROSSTALK_SEPARATION_BACKEND,
            )
            or DEFAULT_CROSSTALK_SEPARATION_BACKEND
        ).lower()
        return cls(
            input_dir=_read_path("INPUT_DIR", DEFAULT_INPUT_DIR),
            work_dir=work_dir,
            output_dir=_read_path("OUTPUT_DIR", DEFAULT_OUTPUT_DIR),
            model_cache_dir=model_cache_dir,
            gemini_api_key=_read_env("GEMINI_API_KEY"),
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
            crosstalk_separation=_read_bool(
                "CROSSTALK_SEPARATION", DEFAULT_CROSSTALK_SEPARATION
            ),
            crosstalk_separation_backend=crosstalk_separation_backend,
            crosstalk_separation_model=(
                _read_env("CROSSTALK_SEPARATION_MODEL")
                or default_crosstalk_separation_model(crosstalk_separation_backend)
            ),
            crosstalk_separation_context_seconds=_read_float(
                "CROSSTALK_SEPARATION_CONTEXT_SECONDS",
                DEFAULT_CROSSTALK_SEPARATION_CONTEXT_SECONDS,
            ),
            separation_backend=(
                _read_env("SEPARATION_BACKEND", DEFAULT_SEPARATION_BACKEND)
                or DEFAULT_SEPARATION_BACKEND
            ).strip().lower(),
            separation_demucs_model=(
                _read_env("SEPARATION_DEMUCS_MODEL", DEFAULT_SEPARATION_DEMUCS_MODEL)
                or DEFAULT_SEPARATION_DEMUCS_MODEL
            ),
            separation_autocast=_read_bool(
                "SEPARATION_AUTOCAST", DEFAULT_SEPARATION_AUTOCAST
            ),
            transcription_language=(
                (_read_env("TRANSCRIPTION_LANGUAGE") or "").lower() or None
            ),
            translation_backend=(
                _read_env("TRANSLATION_BACKEND", DEFAULT_TRANSLATION_BACKEND)
                or DEFAULT_TRANSLATION_BACKEND
            ).lower(),
            translation_provider=(
                _read_env("TRANSLATION_PROVIDER", DEFAULT_TRANSLATION_PROVIDER)
                or DEFAULT_TRANSLATION_PROVIDER
            ).lower(),
            translation_model=(
                _read_env("TRANSLATION_MODEL") or _translation_model_default()
            ),
            translation_base_url=(
                _read_env("TRANSLATION_BASE_URL") or _translation_base_url_default()
            ),
            translation_batch_size=_read_int(
                "TRANSLATION_BATCH_SIZE", DEFAULT_TRANSLATION_BATCH_SIZE
            ),
            translation_thinking=(
                _read_env("TRANSLATION_THINKING", DEFAULT_TRANSLATION_THINKING)
                or DEFAULT_TRANSLATION_THINKING
            ).lower(),
            translation_fallback_provider=(
                _read_env(
                    "TRANSLATION_FALLBACK_PROVIDER",
                    DEFAULT_TRANSLATION_FALLBACK_PROVIDER,
                )
                or DEFAULT_TRANSLATION_FALLBACK_PROVIDER
            ).lower(),
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
            translation_max_reduction_attempts=_read_nonnegative_int(
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
            voice_sample_dir=_read_path("VOICE_SAMPLE_DIR", DEFAULT_VOICE_SAMPLE_DIR),
            voice_sample_map=(
                _read_env("VOICE_SAMPLE_MAP", DEFAULT_VOICE_SAMPLE_MAP)
                or DEFAULT_VOICE_SAMPLE_MAP
            ),
            voice_bank_dir=_read_path("VOICE_BANK_DIR", DEFAULT_VOICE_BANK_DIR),
            voice_bank_voice_map=(
                _read_env("VOICE_BANK_MAP", DEFAULT_VOICE_BANK_VOICE_MAP)
                or DEFAULT_VOICE_BANK_VOICE_MAP
            ),
            dialogue_bible_path=_read_path(
                "DIALOGUE_BIBLE_PATH", work_dir / DEFAULT_DIALOGUE_BIBLE_FILENAME
            ),
            character_guide_path=_read_path(
                "CHARACTER_GUIDE_PATH", work_dir / DEFAULT_CHARACTER_GUIDE_FILENAME
            ),
            episode_script_path=_read_path(
                "EPISODE_SCRIPT_PATH", work_dir / DEFAULT_EPISODE_SCRIPT_FILENAME
            ),
            context_ai_enabled=_read_bool(
                "CONTEXT_AI_ENABLED", DEFAULT_CONTEXT_AI_ENABLED
            ),
            context_ai_confidence_threshold=_read_float(
                "CONTEXT_AI_CONFIDENCE_THRESHOLD",
                DEFAULT_CONTEXT_AI_CONFIDENCE_THRESHOLD,
            ),
            context_ai_timeout_seconds=_read_float(
                "CONTEXT_AI_TIMEOUT_SECONDS", DEFAULT_CONTEXT_AI_TIMEOUT_SECONDS
            ),
            context_ai_script_timeout_seconds=_read_float(
                "CONTEXT_AI_SCRIPT_TIMEOUT_SECONDS",
                DEFAULT_CONTEXT_AI_SCRIPT_TIMEOUT_SECONDS,
            ),
            drive_remote=(
                _read_env("DRIVE_REMOTE", DEFAULT_DRIVE_REMOTE) or DEFAULT_DRIVE_REMOTE
            ),
            drive_source_folder=(
                _read_env("DRIVE_SOURCE_FOLDER", DEFAULT_DRIVE_SOURCE_FOLDER)
                or DEFAULT_DRIVE_SOURCE_FOLDER
            ),
            drive_output_folder=(
                _read_env("DRIVE_OUTPUT_FOLDER", DEFAULT_DRIVE_OUTPUT_FOLDER)
                or DEFAULT_DRIVE_OUTPUT_FOLDER
            ),
            rclone_config=_read_path("RCLONE_CONFIG", DEFAULT_RCLONE_CONFIG_PATH),
            rclone_bin=(
                _read_env("RCLONE_BIN", DEFAULT_RCLONE_BIN) or DEFAULT_RCLONE_BIN
            ),
            tts_model=(_read_env("TTS_MODEL", DEFAULT_TTS_MODEL) or DEFAULT_TTS_MODEL),
            tts_engine=(
                _read_env("TTS_ENGINE", DEFAULT_TTS_ENGINE) or DEFAULT_TTS_ENGINE
            ).lower(),
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
    def translation_provider_defaults(self) -> TranslationProvider:
        """Return the provider record this run is configured to talk to."""

        return translation_provider_defaults(self.translation_provider)

    def with_available_translation_provider(self) -> "Settings":
        """Use DeepSeek when Gemini is selected but only a DeepSeek key is present."""

        if (
            self.translation_backend != "openai"
            or self.translation_provider != "gemini"
            or self.translation_api_key
            or not (self.deepseek_api_key or "").strip()
        ):
            return self

        provider = translation_provider_defaults("deepseek")
        return replace(
            self,
            translation_provider=provider.name,
            translation_model=provider.model,
            translation_base_url=provider.base_url,
        )

    @property
    def translation_api_key(self) -> str:
        """Return the API key for the configured provider, or an empty string.

        One accessor rather than a branch at every call site: the provider is what says
        which variable holds the key, and the rest of the pipeline should not have to
        know which vendor it is talking to.
        """

        env = self.translation_provider_defaults.api_key_env
        if env == "GEMINI_API_KEY":
            return (self.gemini_api_key or "").strip()
        if env == "DEEPSEEK_API_KEY":
            return (self.deepseek_api_key or "").strip()
        # ``other`` names its own variable, so that a self-hosted server with its own
        # key convention needs no code change.
        return (_read_env(env) or "").strip()

    @property
    def translation_api_key_env(self) -> str:
        """Return the environment variable name the provider needs."""

        return self.translation_provider_defaults.api_key_env

    @property
    def has_deepseek_credentials(self) -> bool:
        """``True`` when a DeepSeek API key is configured."""

        return bool(self.deepseek_api_key)

    @property
    def has_gemini_credentials(self) -> bool:
        """``True`` when a Gemini API key is configured."""

        return bool(self.gemini_api_key)

    @property
    def has_huggingface_credentials(self) -> bool:
        """``True`` when a Hugging Face token is configured."""

        return bool(self.huggingface_token)

    def missing_credentials(self) -> list[str]:
        """Return the names of the credentials the *configured* run still needs.

        The Hugging Face token is always needed, because the diarization pipeline it
        authenticates is gated. The translation key belongs to the instruction-following
        backend alone, and *which* variable holds it follows the provider: under
        ``TRANSLATION_BACKEND=nllb`` the run never reads either, so reporting a key as
        missing would hold up a session that is ready to go.
        """

        missing: list[str] = []
        if self.translation_backend == "openai" and not self.translation_api_key:
            missing.append(self.translation_api_key_env)
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
            "gemini_api_key_set": self.has_gemini_credentials,
            "huggingface_token_set": self.has_huggingface_credentials,
            "device": self.device,
            "log_level": self.log_level,
            "diarization_model": self.diarization_model,
            "diarization_min_speakers": self.diarization_min_speakers,
            "diarization_max_speakers": self.diarization_max_speakers,
            "transcription_model": self.transcription_model,
            "transcription_compute_type": self.transcription_compute_type,
            "qc_pronunciation": self.qc_pronunciation,
            "crosstalk_separation": self.crosstalk_separation,
            "crosstalk_separation_backend": self.crosstalk_separation_backend,
            "crosstalk_separation_model": self.crosstalk_separation_model,
            "crosstalk_separation_context_seconds": (
                self.crosstalk_separation_context_seconds
            ),
            "separation_backend": self.separation_backend,
            "separation_demucs_model": self.separation_demucs_model,
            "separation_autocast": self.separation_autocast,
            "transcription_language": self.transcription_language,
            "translation_backend": self.translation_backend,
            "translation_provider": self.translation_provider,
            "translation_api_key_env": self.translation_api_key_env,
            "translation_model": self.translation_model,
            "translation_base_url": self.translation_base_url,
            "translation_thinking": self.translation_thinking,
            "translation_fallback_provider": self.translation_fallback_provider,
            "translation_batch_size": self.translation_batch_size,
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
            "voice_sample_dir": str(self.voice_sample_dir),
            "voice_sample_map": self.voice_sample_map,
            "voice_bank_dir": str(self.voice_bank_dir),
            "voice_bank_voice_map": self.voice_bank_voice_map,
            "dialogue_bible_path": str(self.dialogue_bible_path),
            "character_guide_path": str(self.character_guide_path),
            "episode_script_path": str(self.episode_script_path),
            "context_ai_enabled": self.context_ai_enabled,
            "context_ai_confidence_threshold": self.context_ai_confidence_threshold,
            "context_ai_timeout_seconds": self.context_ai_timeout_seconds,
            "context_ai_script_timeout_seconds": (
                self.context_ai_script_timeout_seconds
            ),
            "drive_remote": self.drive_remote,
            "drive_source_folder": self.drive_source_folder,
            "drive_output_folder": self.drive_output_folder,
            "rclone_config": str(self.rclone_config),
            "rclone_bin": self.rclone_bin,
            "tts_model": self.tts_model,
            "tts_engine": self.tts_engine,
            "omnivoice_model": self.omnivoice_model,
            "omnivoice_steps": self.omnivoice_steps,
            "omnivoice_guidance_scale": self.omnivoice_guidance_scale,
            "omnivoice_torch_dtype": self.omnivoice_torch_dtype,
            "mms_sample_rate": self.mms_sample_rate,
            "mms_seed": self.mms_seed,
            "mms_speaking_rate": self.mms_speaking_rate,
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

    return Settings.from_env().with_available_translation_provider()


__all__ = [
    "DEFAULT_OMNIVOICE_GUIDANCE_SCALE",
    "DEFAULT_OMNIVOICE_MODEL",
    "DEFAULT_OMNIVOICE_STEPS",
    "DEFAULT_OMNIVOICE_TORCH_DTYPE",
    "DEFAULT_DIALOGUE_BIBLE_FILENAME",
    "DEFAULT_DIALOGUE_BIBLE_PATH",
    "DEFAULT_CHARACTER_GUIDE_FILENAME",
    "DEFAULT_CHARACTER_GUIDE_PATH",
    "DEFAULT_EPISODE_SCRIPT_FILENAME",
    "DEFAULT_EPISODE_SCRIPT_PATH",
    "DEFAULT_CONTEXT_AI_SCRIPT_TIMEOUT_SECONDS",
    "DEFAULT_CONTEXT_AI_ENABLED",
    "DEFAULT_CONTEXT_AI_CONFIDENCE_THRESHOLD",
    "DEFAULT_CONTEXT_AI_TIMEOUT_SECONDS",
    "DEFAULT_DEEPSEEK_SEARCH_PATH",
    "DEFAULT_DEEPSEEK_FETCH_PATH",
    "DEFAULT_DRIVE_REMOTE",
    "DEFAULT_DRIVE_SOURCE_FOLDER",
    "DEFAULT_DRIVE_OUTPUT_FOLDER",
    "DEFAULT_RCLONE_CONFIG_PATH",
    "DEFAULT_RCLONE_BIN",
    "DEFAULT_DIARIZATION_MAX_SPEAKERS",
    "DEFAULT_DIARIZATION_MIN_SPEAKERS",
    "DEFAULT_DIARIZATION_MODEL",
    "DEFAULT_MIX_DIALOGUE_GAIN_DB",
    "DEFAULT_MIX_DUCK_DB",
    "DEFAULT_MMS_SAMPLE_RATE",
    "DEFAULT_MMS_SEED",
    "DEFAULT_MMS_SPEAKING_RATE",
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
    "DEFAULT_TRANSLATION_PROVIDER",
    "DEFAULT_TRANSLATION_GEMINI_BASE_URL",
    "DEFAULT_TRANSLATION_GEMINI_MODEL",
    "DEFAULT_TRANSLATION_DEEPSEEK_BASE_URL",
    "DEFAULT_TRANSLATION_DEEPSEEK_MODEL",
    "DEFAULT_TRANSLATION_THINKING",
    "DEFAULT_TRANSLATION_FALLBACK_PROVIDER",
    "TRANSLATION_PROVIDERS",
    "TRANSLATION_THINKING_LEVELS",
    "TranslationProvider",
    "translation_provider_defaults",
    "DEFAULT_TRANSLATION_MAX_REDUCTION_ATTEMPTS",
    "DEFAULT_TRANSLATION_OPENAI_MODEL",
    "DEFAULT_TRANSLATION_SHORTEN_PENALTY",
    "DEFAULT_TRANSLATION_SYLLABLES_PER_SECOND",
    "DEFAULT_TTS_CONTINUE_ON_FAILURE",
    "DEFAULT_TTS_MAX_PAUSE_SECONDS",
    "DEFAULT_TTS_MIN_LINE_SECONDS",
    "DEFAULT_TTS_REQUEST_RATE",
    "DEFAULT_TTS_MODEL",
    "DEFAULT_TIMING_MAX_OVERLAP_SECONDS",
    "DEFAULT_TIMING_MAX_TEMPO",
    "DEFAULT_TIMING_MIN_LINE_GAP",
    "DEFAULT_TIMING_TRIM_TO_FIT",
    "DEFAULT_TIMING_MIN_TEMPO",
    "DEFAULT_VOICE_PROFILE_DIR",
    "DEFAULT_VOICE_REFERENCE_MAX_DURATION",
    "DEFAULT_VOICE_SAMPLE_DIR",
    "DEFAULT_VOICE_SAMPLE_MAP",
    "DEFAULT_VOICE_BANK_DIR",
    "DEFAULT_VOICE_BANK_VOICE_MAP",
    "DEFAULT_VOICE_REFERENCE_MIN_DURATION",
    "DEFAULT_VOICE_REFERENCE_TARGET_DURATION",
    "PROJECT_ROOT",
    "Settings",
    "apply_torch_compile_policy",
    "get_settings",
    "load_env_file",
]


# Applied at import so the guard is in place before any stage imports torch,
# which is what lets it cover a dependency that compiles during its own import.
apply_torch_compile_policy()
