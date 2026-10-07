"""Amharic movie dubbing pipeline.

This package holds the source code for an open-source pipeline that dubs movies
into Amharic while preserving the original music and sound effects:

    source video
        -> audio source separation (dialogue / music / effects)
        -> speaker diarization
        -> speech transcription
        -> dialogue adaptation & translation (Amharic)
        -> voice cloning / text-to-speech
        -> timing alignment
        -> re-mixing
        -> dubbed video

Currently this package only contains the project scaffold and the configuration
system. The individual pipeline stages are placeholders that document the
intended design; none of them are implemented yet.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
