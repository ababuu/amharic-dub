"""Orchestration layer for the Amharic dubbing pipeline.

The pipeline is a linear sequence of independent stages. Each stage lives in its
own module and communicates only through files in ``WORK_DIR`` plus a small
manifest, so stages can be re-run, cached, and debugged in isolation.

Planned stage order
-------------------
1. ``video``         - extract audio from the source video, later mux the dub back.
2. ``separation``    - split the soundtrack into dialogue / music / effects stems.
3. ``diarization``   - identify who spoke when.
4. ``transcription`` - transcribe the dialogue stem with timestamps.
5. ``translation``   - adapt and translate the dialogue into Amharic.
6. ``voice_profiles``- build a per-character Amharic voice profile.
7. ``tts``           - synthesize Amharic dialogue with cloned voices.
8. ``timing``        - align synthesized speech to the original timings.
9. ``mixing``        - combine dubbed dialogue with the music/effects stems.

TODO: implement a thin orchestrator (e.g. ``run_pipeline(source)``) that chains
the stages above, emits progress, and resumes from cached artifacts. Do not add
that orchestrator until at least one stage is implemented.
"""

__all__: list[str] = []
