"""Final audio re-mixing.

Planned responsibility
----------------------
Rebuild the movie's soundtrack by placing the time-aligned Amharic dialogue on
top of the original ``music`` and ``effects`` stems, then normalize loudness so
the result matches typical streaming targets.

Interfaces (to be implemented)
------------------------------
* Input : aligned dialogue clips + music/effects stems.
* Output: a single mixed audio track in ``WORK_DIR`` for :mod:`app.pipeline.video`.

TODO: duck the music/effects slightly under dialogue instead of hard-cutting.
TODO: apply loudness normalization (e.g. EBU R128) and dither on export.
TODO: keep the original channel count and sample rate.
TODO: expose a mixing function. No stub implementation yet.
"""

__all__: list[str] = []
