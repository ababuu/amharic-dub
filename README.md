# Amharic Dubbing Pipeline

An open-source pipeline for dubbing movies into **Amharic** while preserving the
original music and sound effects. The soundtrack is split into dialogue, music,
and effects; the dialogue is transcribed, translated/adapted, re-voiced with
cloned voices, re-timed, and mixed back with the untouched music and effects.

> **Status: the pipeline is complete end to end and produces a dubbed MP4.** The
> soundtrack is split into dialogue, music and effects; the dialogue is
> diarized, transcribed, adapted into spoken Amharic, voiced per character, fitted
> to the original timings, re-mixed under the untouched music and effects, and
> muxed back into the video with the picture copied rather than re-encoded. A
> single command drives all of it. See [Roadmap](#roadmap) for what is
> deliberately left for later.

---

## Planned pipeline

```mermaid
flowchart LR
    V[Source video] --> SEP[Source separation<br/>BandIt v2 Multi]
    SEP -->|dialogue| DIA[Speaker diarization<br/>pyannote Community-1]
    DIA --> ASR[Transcription<br/>faster-whisper large-v3]
    ASR --> TR[Adapt + translate to Amharic<br/>scene + character bible + syllable budget]
    TR --> VP[Voice profiles<br/>best clean reference per speaker]
    VP --> TTS[Speech synthesis + voice adaptation<br/>Chatterbox Amharic → Seed-VC V2]
    TTS --> TIM[Timing alignment]
    TIM --> MIX[Mixing]
    SEP -->|music + effects| MIX
    MIX --> QC[Quality control<br/>fit, rate, crosstalk]
    QC --> OUT[Dubbed video<br/>Bunny Stream]
```

## Tech stack

| Concern              | Choice                                  |
| -------------------- | --------------------------------------- |
| Language             | Python 3.12                             |
| Runtime              | RunPod A40 worker, NVIDIA CUDA 12.8     |
| Framework            | PyTorch 2.8 + torchaudio 2.8            |
| Media                | FFmpeg                                  |
| Source separation    | BandIt v2 Multi                         |
| Diarization          | pyannote Community-1                    |
| Transcription        | faster-whisper large-v3                 |
| Translation/adaptation | DeepSeek V4.1 Flash API (any OpenAI-compatible endpoint, local or hosted) |
| Speech synthesis     | Chatterbox Multilingual v3 + gabar-tech Amharic adapter |
| Voice conversion     | Seed-VC V2 (timbre-only, `convert_style=false`) |
| Quality control      | `qc.py` (model-free; pronunciation measurement is injected) |
| Delivery (later)     | Bunny Stream                            |

Target GPU: **NVIDIA RTX A40 (48 GB VRAM)**.

## Runtime environment

The pipeline runs on a RunPod A40 pod created from the **official RunPod PyTorch
image**. That image already provides the heavy parts of the stack:

| Provided by the worker image | Version   |
| ---------------------------- | --------- |
| Python                       | 3.12      |
| PyTorch                      | 2.8       |
| torchaudio                   | 2.8       |
| CUDA                         | 12.8      |
| FFmpeg                       | on `PATH` |

Because of that, this repository deliberately contains **no container or
deployment artifacts** (no Dockerfile, Compose file, or lockfile), and it does
not install or pin a CUDA/PyTorch stack of its own. Only the application's own
dependencies are installed, from `requirements.txt` - see
[Quickstart](#quickstart---runpod-a40-worker) for why that is done through
`scripts/install_dependencies.sh` rather than a plain `pip install`.

## Project structure

```
amharic-dub/
  app/
    __init__.py
    config.py                 # environment-driven configuration (implemented)
    pipeline/                 # one module per stage
      __init__.py
      separation.py           # BandIt v2 Multi separation (implemented)
      diarization.py          # pyannote Community-1 diarization (implemented)
      transcription.py        # faster-whisper large-v3 transcription (implemented)
      translation.py          # DeepSeek Amharic dialogue adaptation (implemented)
      voice_profiles.py       # per-speaker voice references + clone-prompt cache (implemented)
      dialogue_context.py     # scenes, character bible, syllable budget (implemented)
      amharic_text.py         # Ethiopic syllables + homophone folding (implemented)
      tts.py                  # Chatterbox Amharic + Seed-VC V2 synthesis (implemented)
      timing.py               # pitch-preserving fit to the original timings (implemented)
      mixing.py               # dialogue over ducked music + effects (implemented)
      qc.py                   # measures each run: fit, rate, crosstalk (implemented)
      prosody.py              # model-free pitch tracking for performance checks (implemented)
      evaluation.py           # coverage, identity, performance, baselines (implemented)
      video.py                # FFmpeg extraction + mux into a dubbed MP4 (implemented)
      orchestrator.py         # runs every stage and produces the deliverable (implemented)
    models/                   # shared model loading/caching (placeholder)
    audio/                    # audio IO + DSP helpers (placeholder)
    utils/                    # logging, subprocess, manifests (placeholder)
  tests/                      # pytest suite
  scripts/
    check_environment.py      # runtime health check; --full is the session pre-flight
    install_dependencies.sh   # worker install, including the Chatterbox --no-deps step
    validate_models.py        # loads each model on its own and reports the result
    test_gpu.py               # end-to-end run of the implemented stages
  data/
    input/                    # source videos (not committed)
    working/                  # intermediate artifacts (not committed), including
                              #   voices/<SPEAKER_ID>/{reference.wav,voice_clone.pt,profile.json}
                              #   tts/{performance,chatterbox,converted,clips}/*.wav
    output/                   # final dubbed videos (not committed)
  .env.example
  .gitignore
  README.md
  pyproject.toml
  requirements.txt
```

## Configuration

All configuration comes from environment variables. A local `.env` file is
loaded when present but never overrides values already set in the real
environment, so values configured on the RunPod pod always win.

| Variable            | Purpose                                             | Default            |
| ------------------- | --------------------------------------------------- | ------------------ |
| `DEEPSEEK_API_KEY`  | DeepSeek dialogue adaptation/translation            | *(unset)*          |
| `HUGGINGFACE_TOKEN` | Download gated model weights (e.g. pyannote)        | *(unset)*          |
| `INPUT_DIR`         | Source videos                                       | `./data/input`     |
| `WORK_DIR`          | Intermediate artifacts                              | `./data/working`   |
| `OUTPUT_DIR`        | Final dubbed videos                                 | `./data/output`    |
| `MODEL_CACHE_DIR`   | Runtime model weight downloads                      | `./models_cache`   |
| `DIARIZATION_MODEL` | Hugging Face pipeline id for diarization            | `pyannote/speaker-diarization-community-1` |
| `DIARIZATION_MIN_SPEAKERS` | Fewest speakers diarization may report; unset lets it decide | *(unset)* |
| `DIARIZATION_MAX_SPEAKERS` | Most speakers diarization may report; unset lets it decide | *(unset)* |
| `TRANSCRIPTION_MODEL` | faster-whisper model for transcription            | `large-v3`         |
| `TRANSCRIPTION_COMPUTE_TYPE` | CTranslate2 compute type (`float16` on GPU) | `float16`          |
| `TRANSCRIPTION_LANGUAGE` | Source language code; unset detects it        | *(unset → detect)* |
| `TRANSLATION_BASE_URL` | DeepSeek API endpoint (OpenAI-compatible)        | `https://api.deepseek.com` |
| `TRANSLATION_MODEL` | DeepSeek model for dialogue adaptation            | `deepseek-flash`   |
| `TRANSLATION_BATCH_SIZE` | Dialogue lines adapted per request           | `10`               |
| `TRANSLATION_DISABLE_THINKING` | Turn off reasoning/thinking mode       | `true`             |
| `VOICE_PROFILE_DIR` | Per-speaker voice profiles directory                | `$WORK_DIR/voices` |
| `VOICE_REFERENCE_MIN_DURATION` | Shortest usable voice-cloning reference  | `3.0`              |
| `VOICE_REFERENCE_TARGET_DURATION` | Preferred reference length            | `10.0`             |
| `VOICE_REFERENCE_MAX_DURATION` | Longest reference kept (a longer continuous turn is scanned with a sliding window) | `15.0` |
| `DIALOGUE_BIBLE_PATH` | Persistent character/consistency state for adaptation (see [Cinematic dialogue adaptation](#cinematic-dialogue-adaptation-dialogue_contextpy)) | `$WORK_DIR/dialogue_bible.json` |
| `TTS_MODEL` | Amharic speech adapter used by the TTS stage   | `gabar-tech/chatterbox-amharic` |
| `SEED_VC_REPO_PATH` | Seed-VC checkout for the identity-conversion step | `$MODEL_CACHE_DIR/seed-vc` |
| `SEED_VC_DIFFUSION_STEPS` | Diffusion steps of the Seed-VC V2 converter | `30` |
| `SEED_VC_CONVERT_STYLE` | Also convert the *reference's* accent and style. **Keep this off** - see [Speech synthesis](#speech-synthesis-ttspy) | `false` |
| `TTS_PERFORMANCE_REFERENCE_MIN_DURATION` / `..._MAX_DURATION` | Length of the original-performance prompt handed to Chatterbox | `6.0` / `12.0` |
| `TTS_MAX_PAUSE_SECONDS` | Longest pause rendered around a synthesized line | `2.0` |
| `TTS_MIN_LINE_SECONDS` | Shortest original window that can be dubbed; shorter lines are skipped and reported | `0.30` |
| `TTS_CONTINUE_ON_FAILURE` | Skip and report a line an engine fails on instead of ending the run | `false` |
| `TIMING_MIN_TEMPO` | Slowest a line may be stretched to fit its window | `0.80` |
| `TIMING_MAX_TEMPO` | Fastest a line may be stretched to fit its window | `1.25` |
| `MIX_DIALOGUE_GAIN_DB` | Dialogue level in the final mix (signed dB)     | `0.0`              |
| `MIX_DUCK_DB`       | How far music/effects are ducked under dialogue    | `6.0`              |
| `DEVICE`            | `cuda` on a GPU worker, `cpu` for CPU-only checks   | `cuda`             |
| `LOG_LEVEL`         | `DEBUG` / `INFO` / `WARNING` / `ERROR`              | `INFO`             |

**Secrets are never hard-coded.** The project runs without any API keys, so you
can test and run the health check before configuring credentials.

```bash
cp .env.example .env   # then edit .env and fill in real values
```

### Which variables you actually have to set

`.env.example` and the table above are the complete catalogue of knobs, listing
every setting with the value it already has. **Nothing has to be copied into
`.env`**: the credentials have no fallback, and every other variable has a working
default resolved against the project root.

| | Variables | Why |
| --- | --- | --- |
| Required | `HUGGINGFACE_TOKEN` | No fallback value; diarization cannot run without it |
| Required only for the hosted API | `DEEPSEEK_API_KEY` | Needed when `TRANSLATION_BASE_URL` points at the hosted DeepSeek API. A local OpenAI-compatible server needs no key |
| Required on a Pod | `HF_HOME` | Defaults to the container's `~/.cache/huggingface`, which is lost when the Pod stops |
| Worth setting | `MODEL_CACHE_DIR`, `SEED_VC_REPO_PATH`, `DIALOGUE_BIBLE_PATH` | The first two default under the project root, so they follow the repository onto the volume. The bible is per-film consistency state worth keeping between runs |
| Everything else | e.g. `TRANSCRIPTION_MODEL`, `TIMING_MAX_TEMPO`, `MIX_DUCK_DB` | Set only to change behaviour |

**Running without a commercial API.** Adaptation talks to any OpenAI-compatible
endpoint, so a self-hosted server replaces the hosted one by configuration alone -
no code change:

```
TRANSLATION_BASE_URL=http://localhost:8000/v1   # vLLM, Ollama, llama.cpp server
TRANSLATION_MODEL=<the served model name>
```

Which local model adapts English dialogue into performable Amharic best is an open
question: no source publishes credible English-to-Amharic *dubbing* quality
figures, and there is no Amharic dialogue-MT research at all. Treat the choice as a
bake-off over a few hundred representative lines, scored against the hosted
baseline on the `qc` block plus a native-speaker read - not as a model swap that
can be assumed to work.

Values are resolved in three layers, in this order: a real environment variable,
then `.env`, then the built-in default. `.env` never overrides a real environment
variable, so configuration made on the Pod always wins over a checked-out file.

To see what is actually in effect - including the path every default resolved to -
without exposing secrets:

```bash
python -c "from app.config import get_settings; print(get_settings().as_dict())"
```

The two credentials appear only as the booleans `deepseek_api_key_set` and
`huggingface_token_set`; their values are never printed or logged.

## Quickstart - RunPod A40 worker

The pipeline runs on a GPU worker; the repository is Git-based, so the runbook is
create, configure, install, validate, run, collect. **Only the first three steps
are setup** - after that a run is a single command, and the last step gets you the
file.

**1. Create the Pod.** From the official RunPod PyTorch image
(`runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`, which provides Python 3.12,
PyTorch 2.8, CUDA 12.8 and FFmpeg) on an A40, with a **network volume mounted at
`/workspace`**. The volume is what you actually want: without one, every session
re-downloads many gigabytes of weights before doing any work.

**2. Configure it through Pod environment variables, and clone both
repositories.** Set these in the Pod's **Environment Variables** section (one
`KEY=VALUE` per line, or a JSON object through the CLI) rather than exporting them
in a shell later. A Pod environment variable exists before Python starts, which is
what `HF_HOME` needs, and it survives reconnects:

```
DEEPSEEK_API_KEY=...                 # required - dialogue adaptation
HUGGINGFACE_TOKEN=hf_...             # required - gated pyannote Community-1 weights
HF_HOME=/workspace/models_cache      # required on a Pod - see below
MODEL_CACHE_DIR=/workspace/models_cache    # optional; defaults under the repo
SEED_VC_REPO_PATH=/workspace/seed-vc       # optional; defaults under the repo
```

Create the Hugging Face token with the **Read** role, or as a fine-grained token
with read access to the gated repository. The pipeline only ever downloads from the
Hub, so a write token would be unnecessary and riskier if it leaked. Access to a
gated repository is granted to your **account** by accepting its licence, not by
the token's role - so a Read token works, as long as the account has been approved
on the
[pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
model page.

`HF_HOME` is the one variable that has to be set even though the project also
works without it: its default is the container's `~/.cache/huggingface`, which is
lost when the Pod stops. It is frozen when `huggingface_hub` is first imported, so
it has to be a real environment variable - exporting it in a shell works but must
be repeated every session, and putting it in `.env` does not work at all.
`MODEL_CACHE_DIR` and `SEED_VC_REPO_PATH` are listed for completeness: their
defaults already resolve under the project root, so if you cloned onto the volume
they are on the volume too. Set them only to put the weights somewhere else, such
as a cache shared between checkouts. The other four variables can also go in a
`.env` in the project root, which `app.config` loads while never letting it
override a real environment variable; `.env` is git-ignored, so it is never
committed. Changing a running Pod's variables requires restarting it.

Then clone both repositories onto the volume:

```bash
git clone <your fork> /workspace/amharic-dub
git clone https://github.com/Plachtaa/seed-vc /workspace/seed-vc
cd /workspace/amharic-dub
```

**3. Install.** Do not install with a plain `pip install -r requirements.txt`: the
Chatterbox entry pins `torch==2.6.0`, `torchaudio==2.6.0` and `numpy<2`, and
resolving it would replace the CUDA-matched PyTorch 2.8 the image provides and
downgrade NumPy below what pyannote needs. `scripts/install_dependencies.sh`
installs everything except Chatterbox first, then Chatterbox with `--no-deps`, and
finally imports every runtime module to prove the result is usable:

```bash
bash scripts/install_dependencies.sh
```

Everything Chatterbox's code actually imports is declared in `requirements.txt`
instead, which is why the manifest lists `s3tokenizer`, `conformer`, `diffusers`
and the Perth watermarker alongside `peft` and `safetensors`. `pip check` will
still report Chatterbox's metadata as unsatisfied: that conflict is known and
intentional, and nothing should be downgraded because of it.

**4. Pre-flight, then validate the models stage by stage.**

```bash
python scripts/check_environment.py --full
python scripts/validate_models.py
```

`scripts/check_environment.py` reports the Python version, the PyTorch version,
CUDA availability, GPU name, available VRAM, and FFmpeg availability. It needs no
credentials and downloads nothing, so it is safe to run first. With `--full` it
becomes the session pre-flight: it also checks both credentials (flagging a value
that is still the `.env.example` placeholder), the Seed-VC checkout and its
`configs/v2/vc_wrapper.yaml`, and every module the stages import at run time.

`scripts/validate_models.py` loads each model **on its own** - BandIt, pyannote,
faster-whisper, the DeepSeek client, the Chatterbox adapter, the Seed-VC wrapper -
reports the result for each, and releases it before the next one, so a failure
names exactly one component instead of surfacing halfway through a run after other
weights have already been downloaded. `--only chatterbox` (or any comma-separated
subset) re-checks a single component. Because every step but the runtime report
downloads real weights, the script refuses to run on a machine where torch reports
no CUDA device unless `--allow-cpu` is passed, so a development box cannot quietly
fill its disk with checkpoints.

**5. Run it.** On a small test clip first, then the film:

```bash
python scripts/test_gpu.py --video data/input/test.mp4   # per-stage report
python -m app.pipeline.orchestrator data/input/test.mp4  # the deliverable
python -m app.pipeline.orchestrator data/input/movie.mp4 # the film
```

`test_gpu.py` stops after speech synthesis on purpose - it validates the GPU stages
and prints what each produced - while the orchestrator continues through timing,
mixing and muxing and writes the dubbed MP4. The orchestrator reuses the clips the
runner already wrote, so running both costs one round of synthesis, not two. For a
first look at a long film, `--max-lines 5` voices only the first few lines and
produces a partial dub. For a long film, run it under `tmux` (or `nohup` with a log
file) so a dropped connection does not kill the run.

**6. Take the result off the Pod.** The deliverable is at
`<OUTPUT_DIR>/<source stem>/<source stem>_amharic.mp4`, which with the defaults and
the repository on the volume is:

```
/workspace/amharic-dub/data/output/movie/movie_amharic.mp4
```

Three ways to get it, easiest first:

* **JupyterLab** - on the Pods page, **Connect** → **HTTP Services** → **Jupyter
  Lab**. Its file browser sees the whole container, so navigate to the path above
  and download the file. No SSH setup needed.
* **`scp`** - get the address from **Connect** → **SSH** (or
  `runpodctl ssh info <pod-id>`) and pull the file, along with the manifest, which
  records the per-line stretch factors, overlaps and mix peak for QA:

  ```bash
  scp -P <ssh-port> root@<pod-ip>:/workspace/amharic-dub/data/output/movie/movie_amharic.mp4 .
  scp -P <ssh-port> root@<pod-ip>:/workspace/amharic-dub/data/output/movie/manifest.json .
  ```

* **In the browser** - serve the directory from the Pod, expose the port as HTTP,
  and open it through the Pod's proxy:

  ```bash
  cd /workspace/amharic-dub/data/output/movie && python -m http.server 8000
  ```

Keeping the repository (and so the output) on the volume is what makes the result
survive a stop, and findable from a later session. Anything written to the
container disk is lost when the Pod stops.

Model weights (BandIt, pyannote, faster-whisper, Chatterbox, Seed-VC) are **not**
part of this repository. They are downloaded at runtime into `MODEL_CACHE_DIR`;
point that variable at the Pod's persistent volume so the weights survive
restarts. `app.pipeline.tts` snapshots the Chatterbox Amharic adapter and its
pinned base model into `MODEL_CACHE_DIR` itself; Seed-VC fetches its own
checkpoints and vocoder through Hugging Face, which takes no cache argument, so
`HF_HOME` has to be set to the same directory as well (step 2).

The pyannote Community-1 diarization checkpoint is a **gated** Hugging Face
model, so `HUGGINGFACE_TOKEN` is required before running that stage. The token's
account must first accept the conditions on the
[model page](https://huggingface.co/pyannote/speaker-diarization-community-1).

Transcription uses faster-whisper `large-v3`, which is **not** gated and is
downloaded into `MODEL_CACHE_DIR` on first run. It defaults to `float16`, which
targets the A40 GPU; a CPU-only run needs
`TRANSCRIPTION_COMPUTE_TYPE=int8` because CTranslate2 does not support fp16 on
CPU. The pipeline never falls back from GPU to CPU on its own.

Dialogue adaptation calls the DeepSeek API, so `DEEPSEEK_API_KEY` is required
before running that stage. It is the only stage that needs network access to a
third party; every model that runs locally has its weights cached under
`MODEL_CACHE_DIR`.

Voice profiles are built from the **dialogue stem** produced by separation, never
from the full movie mix, so each reference is free of music and effects. For every
diarized speaker the stage scores that speaker's own continuous turns and keeps
the cleanest 3-10 second window, then cuts it to mono 24 kHz PCM WAV with FFmpeg
under `<VOICE_PROFILE_DIR>/<SPEAKER_ID>/reference.wav`. FFmpeg must therefore be
on `PATH`. Emotion, intensity and delivery are **not** stored in a profile: they
change per line and belong to `AdaptedDialogue`.

Candidate windows are ranked on duration, speech presence, dynamic range,
loudness, overlap with other speakers and clipping. Speech presence is an energy
VAD: a frame counts as speech only when it rises above the *local* noise floor, so
a loud music bed or steady bleed - which stays close to its own floor whatever its
level - is rejected rather than outranking quieter dialogue. When a transcript is
available it also contributes a modest orthographic phonetic-variety term; without
one the term is dropped instead of being guessed.

Speaker ids are kept exactly as diarization reported them, but only ever used as a
single sanitised directory component (`speaker_directory_name`), so an unusual or
hostile label cannot write outside `VOICE_PROFILE_DIR`.

Voice-clone prompts are cached next to each reference as `voice_clone.pt`. Encoding
is the expensive step, so an existing prompt is always reused and never
re-encoded; pass a `clone_encoder` callable to `build_voice_profiles` if your setup
needs a serialized prompt. The TTS stage does not need one: the Amharic adapter
takes reference audio directly, so `clone_prompt_path` may stay `None`.

### Speech synthesis (`tts.py`)

Two engines run per line, in this order:

1. **Chatterbox Amharic** (`gabar-tech/chatterbox-amharic`, a LoRA adapter plus a
   Fidel tokenizer on Chatterbox Multilingual v3) speaks the Amharic text. It is
   loaded through the loader the adapter repository ships
   (`amharic_tts.py` → `load_amharic_tts`), so training and inference see the
   same text front-end - including the Amharic normalization and sentence
   splitting. The adapter's loader downloads the base checkpoint from
   `ResembleAI/chatterbox` at the revision the *adapter* pins, which is not the
   revision `requirements.txt` installs the code from; the loader validates the
   checkpoint against the installed build and raises if the two disagree, so a
   mismatch is reported rather than producing a wrong voice. Note also that
   `chatterbox-tts` is installed with `--no-deps` (see
   [Quickstart](#quickstart---runpod-a40-worker)): the packages its code needs are
   declared in `requirements.txt` instead.
2. **Seed-VC V2** converts that take into the character's identity in its
   **timbre-only mode** (`convert_style=False`, the default), which replaces the
   voice without touching the delivery. The style-converting mode must not be used
   here: Seed-VC V2's style branch conditions its autoregressive stage on the
   *reference's* acoustic tokens and content indices, so it speaks the source
   content in the reference's accent and emotion. The character reference is the
   original actor's English audio, so switching that mode on would re-impose an
   English accent on the Amharic and overwrite the very performance this stage
   exists to preserve. `SEED_VC_CONVERT_STYLE=true` remains available for a single
   controlled comparison, and nothing else uses it. Seed-VC is not a package, so
   clone it and point
   `SEED_VC_REPO_PATH` at the checkout:
   `git clone https://github.com/Plachtaa/seed-vc <SEED_VC_REPO_PATH>`. Upstream
   has been read-only since April 2025, so **pin a known-good commit**
   (`git -C <SEED_VC_REPO_PATH> checkout <commit>`): the checkout is the one
   engine here that is code rather than a pinned dependency, and the run's
   manifest records `provenance.seed_vc_revision`, so a re-clone that silently
   moves the engine is visible afterwards. Its V2
   converter is called as its own `inference_v2.py` calls it: a `torch.device`
   (it reads `device.type`) and `stream_output=True`, whose generator yields
   `(mp3_bytes, full_audio)` with the completed `(sample_rate, samples)` only on
   the final chunk.

   Note that `length_adjust`, which the V2 converter also accepts, only has an
   effect in the style-converting branch it is not using - in timbre-only mode the
   converted take follows the length of the take it was given. Fitting the line to
   its window stays where it already was, in `timing.py`.

The checkout was written against `huggingface_hub` 0.x (its own requirements ask
only for `>=0.28.1`), while this project runs the 1.x line: `transformers` 5.x,
which Chatterbox needs, requires `huggingface-hub>=1.3.0`. Two arguments were
removed in that major release and Seed-VC's BigVGAN vocoder still uses both:
`ModelHubMixin.from_pretrained` no longer passes `proxies` and `resume_download`
to `_from_pretrained`, which declares them as required, and `hf_hub_download` no
longer accepts them either.

Pinning hub down is not an option, because it would break the engine that works, so
`app.pipeline.tts` adapts the checkout at load time. Note that supplying the two
arguments from the config does **not** work: hub's argument validator pops both
names out of the keyword arguments before the call happens, which is how it retires
them silently, so anything passed from above the method is discarded. They are
therefore defaulted *inside* `_from_pretrained`, by wrapping it as a classmethod,
and the vocoder module's own `hf_hub_download` drops them. Both only ever selected
defaults hub now applies itself, and the adaptation touches nothing else in the
process. It is the only incompatible call site in the checkout.

Note that Seed-VC's `hf_utils.py` writes its checkpoints to `./checkpoints`
relative to the current directory, so they land next to wherever a run is started
rather than under `MODEL_CACHE_DIR`. Run the pipeline from the repository root to
keep them in one predictable place.

The two references come from two different places on purpose. Chatterbox's prompt
decides *how* the line is performed, so it is the original actor's own audio for
that line, cut from the clean BandIt speech stem and centred on the line (padded
when the line is short, trimmed when it is long, bounded by
`TTS_PERFORMANCE_REFERENCE_MIN_DURATION` / `..._MAX_DURATION`).
`VoiceProfile.reference_audio` is used only as the Seed-VC target identity, which
is what makes a character sound like themselves across a film. A `VoiceProfile`
is never modified by this stage.

**Lines that cannot be dubbed are skipped and reported.** A window shorter than
`TTS_MIN_LINE_SECONDS` (0.30s by default) is a fragment, not a spoken line: there is
no room for a word in the time it occupied, so any Amharic written for it is
unintelligible - and Chatterbox can fail outright on it, with an empty mel
spectrogram that trips a convolution inside its vocoder. Such a line, and a line
whose Amharic has nothing to pronounce, is skipped *before* the engines are called,
the lines around it are unaffected, and both the line and the reason are recorded in
the manifest under `run.tts.skipped`. This is input validation, not failure
tolerance: an engine that actually *fails* on a line still stops the run with that
line named, unless `TTS_CONTINUE_ON_FAILURE=true` asks for a reported hole instead.

`AdaptedDialogue`'s performance metadata is mapped, not discarded. `intensity`
moves Chatterbox's `exaggeration` and `temperature` predictably; `emotion` and
`delivery` add a small bias from a cue table; `pause_before` and `pause_after` are
rendered as silence around the converted line, capped at `TTS_MAX_PAUSE_SECONDS`.
Every adjustment is bounded around the adapter's own validated defaults, is
recorded in `PerformanceControls` (with the matched cue terms and the notes that
explain each change), and an unknown emotion or delivery direction changes nothing
and says so rather than inventing a mapping.

Artifacts are deterministic and reusable, one file per line per step, under
`<WORK_DIR>/tts/`:

```
tts/
  performance/  <SPEAKER_ID>_<digest>.wav   # the original performance prompt
  chatterbox/   <SPEAKER_ID>_<digest>.wav   # the Amharic take
  converted/    <SPEAKER_ID>_<digest>.wav   # the same take in the character's voice
  clips/        <SPEAKER_ID>_<digest>.wav   # converted audio + rendered pauses
```

The digest covers the line's timing, text and performance metadata (plus a recipe
version), so the same input always produces the same paths, a changed line never
silently reuses a stale take, and re-running a stage skips work that is already
on disk.

Durations are reported, not fitted: `TtsClip` carries `speech_duration`, the
rendered pauses, the original window and `duration`, and never time-stretches
anything. Making each line fit its original window is `timing.py`'s job. Both
engines are lazy and cached per process (keyed by device/model and by
device/checkout/diffusion steps), so no model is loaded per dialogue line. The
Chatterbox adapter, its pinned base model and the loader file are snapshotted
into `MODEL_CACHE_DIR`; Seed-VC's own downloads follow `HF_HOME` (see
[Quickstart](#quickstart---runpod-a40-worker)).

### Timing, mixing and muxing (`timing.py`, `mixing.py`, `video.py`)

Alignment fits each line to the window of the line it replaces. Only the **speech**
is time-stretched, with FFmpeg's pitch-preserving `atempo`; the rendered pauses
keep their length, and a line's leading pause shifts the file rather than the line,
so the speech still lands on its original start. A line that would need more than
the `TIMING_MIN_TEMPO`/`TIMING_MAX_TEMPO` band is clamped and **reported** as not
fitting - the run records by how much, in the manifest - because mangling a
performance to fit a window is worse than being 0.4 s long. `atempo` accepts
0.5-2.0, so a wider band is rejected rather than passed through.

Mixing places every aligned line at its own timestamp into a continuous dialogue
stem, then sums that with the **music and effects only**. The original English
dialogue is never used: separation already removed it, and mixing the full
original mix back in would reintroduce the language the pipeline exists to
replace. Overlaps are kept - the original performances overlapped too, and moving
a line would break its sync - and every overlap is reported with its speaker pair
and duration. The bed is ducked under the dialogue by `MIX_DUCK_DB` with a ramped
reduction, so it never gates or clicks, and only the bed is reduced. The mix is
then held under a -1 dBFS ceiling; if that needs a global scale-down, the
reduction and the peak that caused it are reported.

Loudness normalization (EBU R128) is deliberately **not** applied: the mix is
placed at a defined peak with defined dialogue and bed levels so it stays
reproducible and auditable, and a programme-loudness pass belongs with the encode
of the deliverable rather than with the mix.

Muxing copies every video stream and encodes exactly one audio track: AAC at
48 kHz stereo, tagged `language=amh`. The source's own audio streams are **not**
mapped, which is what removes the English dialogue. Both inputs and the result are
inspected with `ffprobe` rather than trusted - the video codec, the audio codec,
rate and channel count, the stream counts and the duration are all verified, and
any mismatch is reported rather than assumed away. This is the check missing from
an earlier attempt that muxed a 48 kHz track into a video whose audio was around
44.1 kHz and produced decode errors.

### Running the whole pipeline (`orchestrator.py`)

One command runs every stage in order and writes the deliverable:

```bash
python -m app.pipeline.orchestrator data/input/movie.mp4
python -m app.pipeline.orchestrator movie.mp4 --out runs/movie --max-lines 5
```

The extracted track and the three stems go under `<run dir>/stages/`, the manifest
is `<run dir>/manifest.json`, the mix artifacts are under `<run dir>/mix/`, and
the dubbed video is `<run dir>/<source stem>_amharic.mp4`. The run directory
defaults to `<OUTPUT_DIR>/<source stem>`. Voice profiles and the TTS artifacts
stay in their configured `WORK_DIR` locations rather than moving per run, because
both are content-addressed and are meant to be reused: re-running re-synthesizes
only the lines whose text, timing or performance actually changed.

`--max-lines N` synthesizes only the first `N` adapted lines. It is a cost control
for the first run of a long film, and it produces a **partial dub**: every earlier
stage still runs in full, the manifest records `partial: true`, and the report
says so.

The manifest records what each stage produced, including per-line stretch factors
and fit residuals, the overlaps the mix found, the peak and any scale-down, and
what was verified about the delivered file - so a run can be audited without
re-listening to it.

`scripts/test_gpu.py` is the same chain as a reporting runner. It stops after
speech synthesis on purpose: it exists to validate the GPU stages on a new
machine, and the stages after it need no GPU. Run the orchestrator afterwards to
get the MP4 - it reuses the clips the runner already wrote.

### Cinematic dialogue adaptation (`dialogue_context.py`, `translation.py`)

Adaptation is where the largest quality gain lives, so the stage is given three
things a ten-line window cannot provide:

* **Scene structure.** `segment_scenes` cuts the transcript on the silences
  between lines (and on a maximum scene length), so the model is told which lines
  share a situation, who is in the scene, and whether it is a conversation at all.
  The boundaries come from the timings, so the same film always segments the same
  way.
* **A character bible.** `CharacterBible` is persistent state - names, aliases,
  relationships, register, and the canonical Amharic spelling of recurring names
  and terms - keyed by diarized speaker id and written to
  `DIALOGUE_BIBLE_PATH`. Consistency across 90-180 minutes then rests on a file a
  human can edit between runs rather than on an LLM's recall. A missing file is an
  empty bible, not an error, so a first run just works.
* **A duration budget.** Amharic words are longer than English ones, so a faithful
  line often needs more time than the original took. Each line is given the number
  of **syllables** its window allows - one Fidel character is one syllable, so this
  needs no G2P model - and a line that comes back over budget is **sent back once
  with the number of syllables to cut**. Shortening the text is the fix the
  literature supports; `timing.py` is left to do only a small final trim.

```python
# What the model receives for one line, alongside the system prompt.
{"id": "dialogue_000042", "speaker": "SPEAKER_03", "duration": 2.5,
 "text": "You have to listen to me.", "syllable_budget": 10}
```

The syllable rate behind the budget is `DEFAULT_SYLLABLES_PER_SECOND` (4.0), which
is documented as an **initial prior rather than a measurement**. Calibrate it from
a real run: read the `syllables_per_second` figures the `qc` block reports for the
lines that *did* fit, and set the constant to their median.

`enforce_budget=False` sends every line exactly once, which is cheaper and is what
a comparison run wants.

### Quality control (`qc.py`)

Every run measures itself. `qc.py` builds a report from the metadata the stages
already produced, so it needs no model, no GPU and no audio decoding, and it is
written into the manifest as the `qc` block and printed as the run's last line:

```
qc              19/21 line(s) within 10% of their window, 2 not fitted, 1 at an
                implausible rate, 3 crosstalk region(s) (2.41s)
```

What it reports, and why each figure is there:

* **Duration fit as a distribution, not an average.** The largest human study of
  professional dubbing found the audience complaint is an unnatural speaking rate
  - "too slow, too fast, or too uneven" - so a mean would hide exactly the tail
  that matters. The block carries the fitted/close/unfitted counts, the
  close-fit ratio, the mean and worst residual, and the tempo extremes, plus a
  per-line breakdown.
* **Speaking rate in syllables per second.** One Fidel character is one syllable,
  which is what the script encodes, so this needs no grapheme-to-phoneme model
  (none exists for Amharic). A rate outside the plausible band marks a line no
  performer could deliver, however well it "fits".
* **Crosstalk.** Simultaneous speech is what the exclusive diarization cannot
  describe, so it is reported as a known uncertainty.
* **Pronunciation, when a transcriber is supplied.** A round trip through an
  Amharic ASR model gives a character error rate against the text that was
  synthesized - the only automated check that the dub is *intelligible*. It needs
  a model, so `build_qc_report(..., transcribe=...)` takes it as an argument and a
  run leaves it unmeasured rather than pretending. Homophone families (ሀ/ሐ/ኀ, ሰ/ሠ,
  አ/ዐ, ጸ/ፀ) are folded before comparison, because a difference between them is a
  spelling choice rather than a pronunciation error.

Nothing in the report fails a run: it states what was measured, and deciding what
is good enough stays a project decision.

### Evaluation and baselines (`prosody.py`, `evaluation.py`)

`qc.py` measures one run. Two questions decide whether the system is *getting
better*, and `evaluation.py` answers them:

**Is the evaluation material representative?** A run can score well simply because
it was easy. Every run therefore reports coverage of the cases that break the
pipeline in different ways - whispers, shouts, one-word lines, monologues, rapid
turn-taking, overlapping speech, low and high intensity, English code-switching,
recurring characters, crowded scenes - and names what is **missing**:

```
coverage        4/11 categories; missing whisper, shout, long_line, overlapping_speech, ...
```

The point is that a good score on non-representative material is not evidence.

**Did a change make it worse?** A report can be saved as a baseline and later runs
compared against it, metric by metric, with a tolerance per metric and a direction
per metric (more `close_fit_ratio` is better; more `unfitted` is worse). Identity
and delivery carry tighter tolerances than the rest, because they are the qualities
least able to absorb drift:

```bash
# save a baseline, then compare a later run against it
python -c "from app.pipeline import evaluation as e; e.save_baseline(load_json(), 'baseline.json')"
```

The two qualities the project is judged on most are measured directly, both by
injected callables so that nothing is downloaded by a run:

* **Consistent character voices** - `measure_speaker_identity` compares each
  generated line with its character's own reference and reports the mean, the
  *worst* line and the spread per character. The worst line matters more than the
  mean: a character who sounds right four times and like somebody else once is the
  failure an audience notices, and an average hides it.
* **Preservation of the actor's performance** - `prosody.py` measures the pitch
  centre, pitch range and periodicity of the generated take against the original
  actor's prompt audio, with a model-free autocorrelation tracker. `range_ratio`
  near `1.0` means the delivery survived; well below it means the take was
  flattened toward a neutral read, which is what losing the performance sounds like
  numerically.

`prosody.py` documents its own limitations (an above-range pitch is reported as a
sub-multiple; a *sustained musical tone* is indistinguishable from a sustained
vowel by these means; whispers and creaky voice are reported as unvoiced rather than
guessed at). Its numbers are meaningful as **relative** measurements - both sides of
a comparison go through the same tracker, so systematic bias largely cancels - and
should not be quoted as absolute pitches.

### Configuration-only check (no GPU or API keys required)

The configuration layer and its tests run on any machine with Python 3.12:

```bash
pip install python-dotenv pytest
pytest tests/test_config.py
```

## Roadmap

- [x] Project scaffold, configuration, health check
- [x] `separation`: BandIt v2 Multi integration
- [x] `diarization`: pyannote Community-1 integration
- [x] `transcription`: faster-whisper large-v3 integration
- [x] `translation`: DeepSeek Amharic dialogue adaptation
- [x] `voice_profiles`: per-speaker voice references and clone-prompt cache
- [x] `tts`: Chatterbox Amharic synthesis + Seed-VC V2 voice adaptation
- [x] `timing`: pitch-preserving fit to the original windows
- [x] `mixing`: dialogue over ducked music and effects
- [x] `video`: FFmpeg extraction and mux into a dubbed MP4
- [x] Orchestrator: one command from a video to the deliverable
- [x] `qc`: per-run quality measurement (fit, delivery rate, crosstalk)
- [x] `dialogue_context`: scenes, character bible, syllable budget
- [x] Duration-aware adaptation: lines shortened to fit before synthesis
- [x] `diarization`: crosstalk from the overlap-aware view, speaker-count hints
- [x] `prosody` + `evaluation`: coverage, identity consistency, performance
      preservation, and baseline comparison
- [ ] Representative evaluation clips: record a scored baseline on real material
- [ ] Per-stage resume, so a long run continues instead of restarting
- [ ] Mix realism: ambience continuity, dialogue EQ/reverb match, EBU R128
- [ ] Character-name reconciliation above the diarization clusters
- [ ] Full-film soak test (separation seams, cluster drift, cost)
- [ ] Bunny Stream upload/streaming integration
- [ ] (Later) production API + database
- [ ] (Optional) 5.1 output; lip regeneration for close-ups

## Contributing

Contributions are welcome. Each pipeline stage lives in its own module under
`app/pipeline/` with a docstring describing its intended interface; please keep
that structure and add tests alongside new code.

## Priorities

The project is judged on, in order:

1. natural cinematic Amharic dialogue
2. consistent character voices across the whole film
3. preservation of the actor's emotion and performance
4. high-quality speech separation
5. realistic music and effects reconstruction
6. natural timing and pacing
7. robustness over 90-180+ minute films

Those are qualities, not features, so they are *measured* rather than asserted:
see [Evaluation and baselines](#evaluation-and-baselines-prosodypy-evaluationpy).
Before lower-value infrastructure, the priority is proving those qualities on
representative material and then on a full-length soak test.

## Licensing

This project is intended to remain open source for personal, non-commercial use.
Licences of the components are therefore documented here for transparency but do
**not** drive model selection - the best available component wins on technical
merit. What is currently in use, so the obligations are known rather than assumed:

| Component | Licence | Note |
| --- | --- | --- |
| BandIt v2 (`v2-multi`) | code Apache-2.0, weights **CC-BY-SA-4.0** | share-alike on derivatives; the original BandIt's weights are CC-BY-NC-4.0 and are *not* used |
| pyannote Community-1 | **CC-BY-4.0**, gated | requires accepting the model card; the HF token must have read access |
| faster-whisper `large-v3` | MIT | not gated |
| Chatterbox Multilingual v3 | MIT | base model |
| `gabar-tech/chatterbox-amharic` adapter | **CC-BY-SA-4.0** | share-alike propagates from WaxalNLP |
| Seed-VC V2 | **GPL-3.0**, *archived* | read-only upstream since April 2025; pin the commit (`provenance.seed_vc_revision`) |
| DeepSeek API | proprietary service | the adaptation baseline; swappable for a local OpenAI-compatible server |

Cloning real performers' voices carries likeness and publicity considerations that
a software licence does not address. That is worth stating plainly for a
voice-matched dub, and it is the one legal question here that no licence table
answers.

Two decisions taken deliberately, for now:

* **Stereo 2.0 is the deliverable.** 5.1 is a possible later output format and does
  not drive the architecture.
* **DeepSeek remains the adaptation baseline.** It is not replaced merely for being
  a hosted service; the stage is improved through context, prompting, timing budgets
  and Amharic-specific processing, and a different model replaces it only if
  measurement shows a materially better Amharic result.

## License

Open source. A license file will be added before the first release.
