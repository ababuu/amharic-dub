"""Video input/output handling with FFmpeg.

Planned responsibility
----------------------
Two ends of the pipeline:

* **Demux/extract** - pull the audio track (and any needed metadata such as
  duration, frame rate, and subtitle tracks) out of the source video.
* **Mux/export** - put the final mixed Amharic audio back into the video while
  copying the original video stream untouched, producing the deliverable that
  will later be uploaded to Bunny Stream.

Interfaces (to be implemented)
------------------------------
* Input : a source video in ``INPUT_DIR`` (``.mp4`` / ``.mkv`` / ...).
* Output: extracted audio in ``WORK_DIR`` and the final dubbed video in
  ``OUTPUT_DIR``.

TODO: wrap FFmpeg via ``subprocess`` with explicit, tested argument lists.
TODO: support audio-stream selection and preserve the original video codec.
TODO: expose extract/mux functions. No stub implementation yet.
"""

__all__: list[str] = []
