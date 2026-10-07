"""Shared model loading, caching, and device management.

Planned responsibility
----------------------
Provide one place that knows how to load each heavyweight model used by the
pipeline (BandIt v2 Multi, pyannote Community-1, faster-whisper large-v3,
Chatterbox Multilingual v3 with the Amharic adapter, Seed-VC V2) so the pipeline
stages stay thin.

Design rules
------------
* Weights are downloaded at runtime into ``MODEL_CACHE_DIR`` (see
  :mod:`app.config`); they are never committed to the repository.
* Models are loaded lazily and cached per process to avoid re-loading a
  multi-gigabyte checkpoint between stages.
* GPU/CPU selection is centralized here.

TODO: add per-model loader wrappers, a lazy cache, and a device resolver.
      No model-loading code yet - installing weights/libraries is intentionally
      deferred until the corresponding stage is implemented.
"""

__all__: list[str] = []
