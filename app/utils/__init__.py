"""General-purpose utilities shared across the project.

Planned responsibility
----------------------
Small helpers that have nothing to do with AI models:

* logging setup that honours ``LOG_LEVEL`` from :mod:`app.config`;
* subprocess wrappers (with timeouts and captured output) for FFmpeg and friends;
* safe JSON manifest read/write helpers so stages can cache their artifacts;
* path/time formatting helpers.

TODO: implement logging setup and the manifest helpers first, since nearly every
      pipeline stage depends on them. No implementations yet.
"""

__all__: list[str] = []
