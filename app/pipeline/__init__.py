"""Orchestration layer for the Amharic dubbing pipeline.

The pipeline is a linear sequence of independent stages. Each stage lives in its
own module and communicates only through files in ``WORK_DIR`` plus a small
manifest, so stages can be re-run, cached, and debugged in isolation.

Stage order
-----------
1. ``video``         - extract the audio from the source video; mux the dub back later.
2. ``separation``    - split the soundtrack into dialogue / music / effects stems.
3. ``diarization``   - identify who spoke when.
4. ``transcription`` - transcribe the dialogue stem with timestamps, then assemble
   the pause-delimited segments into complete sentences. A crosstalk region where
   the exclusive diarization dropped a speaker may first be split into per-speaker
   tracks by ``target_extraction``, so both characters' words can be transcribed.
5. ``translation``   - adapt and translate the sentences into Amharic.
6. ``voice_profiles``- build a per-character Amharic voice profile.
7. ``tts``           - synthesize Amharic dialogue with cloned voices.
8. ``timing``        - align synthesized speech to the original timings.
9. ``mixing``        - combine dubbed dialogue with the music/effects stems.

Steps 1-9 are implemented and are chained by
:func:`app.pipeline.orchestrator.run_pipeline`, which is the single entry point for
a run:

    python -m app.pipeline.orchestrator data/input/movie.mp4

The deliverable is a dubbed MP4: the original picture, copied rather than
re-encoded, with the Amharic dialogue track in place of the English one.
"""

__all__: list[str] = []
