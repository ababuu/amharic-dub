"""Audio input/output and DSP helpers.

Planned responsibility
----------------------
Small, dependency-light building blocks shared by the pipeline stages:

* probe/read/write audio and inspect sample rate, channels, and duration;
* resample and convert channel layouts consistently;
* loudness measurement and normalization;
* pitch-preserving time-stretching used by :mod:`app.pipeline.timing`.

TODO: prefer thin FFmpeg wrappers over a heavy audio framework.
TODO: keep every helper pure and unit-testable (path in -> path/metadata out).
No concrete helpers yet.
"""

__all__: list[str] = []
