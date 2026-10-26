#!/usr/bin/env python3
"""Replay crosstalk target extraction on a completed run's dialogue stem.

This is the measuring instrument for :mod:`app.pipeline.target_extraction`. It
reads the crosstalk regions and exclusive lines from a run manifest, asks the
configured separator for one track per region speaker, and - with ``--asr`` -
decodes the mixture and each track so the words and their measured F0 can be
compared side by side.

Example (the reference clip in this repository)::

    python scripts/replay_crosstalk.py data/output/test \
        --registers SPEAKER_00=200.0,SPEAKER_01=120.3 --asr

The per-region report is also written to the run's manifest by a normal run, in
``run.transcription.crosstalk_separation``. The registers can come from the
manifest's ``run.speakers.after`` block when dialogue adaptation measured them;
pass ``--registers`` to override or when that block is absent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings  # noqa: E402
from app.pipeline import target_extraction, transcription  # noqa: E402
from app.pipeline.diarization import CrosstalkRegion  # noqa: E402
from app.pipeline.prosody import estimate_f0  # noqa: E402

RATE = target_extraction.SAMPLE_RATE


def _parse_registers(raw: str | None, manifest: dict) -> dict[str, float]:
    """Return ``SPEAKER=n`` pairs from the flag, or from the manifest's audit."""

    if raw:
        registers: dict[str, float] = {}
        for item in raw.split(","):
            name, _, value = item.partition("=")
            if not name.strip() or not value.strip():
                raise SystemExit(f"bad --registers entry {item!r}; use SPEAKER=Hz")
            registers[name.strip()] = float(value)
        return registers

    after = (
        manifest.get("run", {}).get("speakers", {}).get("after", {})
        if isinstance(manifest.get("run"), dict)
        else {}
    )
    registers = {
        speaker: float(values["median_hz"])
        for speaker, values in after.items()
        if isinstance(values, dict) and values.get("median_hz")
    }
    if not registers:
        raise SystemExit(
            "no measured registers in the manifest; pass --registers "
            "SPEAKER_00=200.0,SPEAKER_01=120.3"
        )
    return registers


def _load_mono16(path: Path) -> np.ndarray:
    data, rate = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if rate != RATE:
        divisor = math.gcd(rate, RATE)
        mono = resample_poly(mono, RATE // divisor, rate // divisor).astype(np.float32)
    return np.ascontiguousarray(mono)


def _measure(samples: np.ndarray) -> str:
    profile = estimate_f0(samples, RATE)
    if profile.median_hz is None:
        return "unmeasured"
    return f"{profile.median_hz:.1f} Hz (voiced {profile.voiced_frames}/{profile.speech_frames})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="directory holding manifest.json")
    parser.add_argument("--registers", help="SPEAKER=Hz list, e.g. SPEAKER_00=200,SPEAKER_01=120")
    parser.add_argument("--backend", help="override CROSSTALK_SEPARATION_BACKEND")
    parser.add_argument("--model", help="override CROSSTALK_SEPARATION_MODEL")
    parser.add_argument("--context", type=float, default=None, help="seconds of context")
    parser.add_argument("--asr", action="store_true", help="decode the mixture and tracks")
    parser.add_argument("--asr-model", default="tiny.en", help="faster-whisper model")
    args = parser.parse_args(argv)

    manifest_path = args.run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run = manifest.get("run", {})
    stems = run.get("stems", {})
    speech = Path(stems.get("speech", ""))
    if not speech.is_absolute():
        speech = PROJECT_ROOT / speech
    if not speech.is_file():
        raise SystemExit(f"dialogue stem not found: {speech}")

    regions = [
        CrosstalkRegion(item["speakers"], item["start"], item["end"])
        for item in run.get("diarization", {}).get("crosstalk", {}).get("detail", [])
    ]
    lines = run.get("transcription", {}).get("segments", [])
    registers = _parse_registers(args.registers, manifest)

    audio = _load_mono16(speech)
    requests: list[target_extraction.ExtractionRequest] = []
    for region in regions:
        present = {
            line["speaker_id"]
            for line in lines
            if line["start"] < region.end and line["end"] > region.start
        }
        missing = [speaker for speaker in region.speaker_ids if speaker not in present]
        window = audio[int(region.start * RATE) : int(region.end * RATE)]
        print(
            f"{region.start:7.2f}-{region.end:7.2f}s  present={sorted(present)} "
            f"missing={missing}  mixture={_measure(window)}"
        )
        if missing and region.duration >= transcription.MIN_RECOVERED_OVERLAP_SECONDS:
            requests.append(
                target_extraction.ExtractionRequest(region, region.speaker_ids)
            )

    settings = get_settings()
    changes = {"crosstalk_separation": True}
    if args.backend:
        changes["crosstalk_separation_backend"] = args.backend
    if args.model:
        changes["crosstalk_separation_model"] = args.model
    if args.context is not None:
        changes["crosstalk_separation_context_seconds"] = args.context
    settings = replace(settings, **changes)

    separator, reason = target_extraction.build_separator(settings)
    if separator is None:
        print(f"separation unavailable: {reason}")
        return 2

    report = target_extraction.extract_tracks(
        audio,
        requests,
        separator=separator,
        registers=registers,
        context_seconds=settings.crosstalk_separation_context_seconds,
    )
    print(f"\n{report.summary()}")
    for region in report.regions:
        print(f"  {region.describe()}")

    if args.asr:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            print("\nfaster-whisper is not installed; skipping --asr")
            return 0
        model = WhisperModel(args.asr_model, device="cpu", compute_type="int8")

        def decode(samples: np.ndarray) -> str:
            segments, _info = model.transcribe(
                samples, language="en", task="transcribe", beam_size=5
            )
            return " ".join(segment.text.strip() for segment in segments).strip()

        print("\nwhat each waveform says:")
        for item in report.regions:
            start, end = item.region.start, item.region.end
            kept = [
                line["text"]
                for line in lines
                if line["start"] < end and line["end"] > start
            ]
            print(f"\n{start:.2f}-{end:.2f}s kept={kept}")
            print(f"  mixture: {decode(audio[int(start * RATE) : int(end * RATE)])!r}")
            for track in item.tracks:
                print(
                    f"  {track.speaker_id} ({track.measured_hz:.1f} Hz): "
                    f"{decode(track.samples)!r}"
                )
            for refusal in item.refusals:
                print(f"  {refusal.speaker_id}: refused - {refusal.reason}")

    print("\n" + json.dumps(report.as_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
