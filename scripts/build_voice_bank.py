#!/usr/bin/env python3
"""Build a native-Amharic voice bank for casting mode.

The pipeline clones a character from a reference recording. When the film is
English, that reference carries an English accent into the Amharic line - the
model documentation says so explicitly - and the only fix is a reference in the
target language. This script builds one: it filters a human-recorded Amharic
corpus for clips that are clean, long enough to clone from, exactly transcribed
and free of Latin script or numerals, then writes them as canonical 24 kHz
references plus a manifest.

Typical use::

    python scripts/build_voice_bank.py                 # 12 voices -> data/voice-bank
    python scripts/build_voice_bank.py --voices 8 --out runs/bank-a

Point the pipeline at the result with ``VOICE_BANK_DIR``. A manifest switches
casting on for every run that sees it; deleting it switches casting off again,
and no previous run is changed either way.

The corpus is scanned through Hugging Face's dataset server, so the script
downloads metadata plus a handful of clips rather than the multi-gigabyte
parquet release. It needs no extra dependency and never imports a model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The summary quotes Amharic transcripts, and a Windows console still defaults to a
# legacy code page that cannot encode them. Reconfiguring is not cosmetic: without it
# the build dies *after* the corpus scan, having already done the work.
for stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

from app.config import get_settings  # noqa: E402
from app.pipeline.voice_bank import (  # noqa: E402
    DEFAULT_BANK_DATASET,
    DEFAULT_BANK_SPLIT,
    BankBuildError,
    BankFilters,
    HuggingFaceRowSource,
    build_voice_bank,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a native-Amharic voice bank from a public corpus.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="directory to write the manifest and clips to (default: VOICE_BANK_DIR)",
    )
    parser.add_argument(
        "--voices", type=int, default=12, help="how many voices to keep"
    )
    parser.add_argument(
        "--dataset", default=DEFAULT_BANK_DATASET, help="Hugging Face dataset id"
    )
    parser.add_argument("--split", default=DEFAULT_BANK_SPLIT, help="dataset split")
    parser.add_argument(
        "--target",
        type=float,
        default=5.0,
        help="preferred speech length in seconds (clips nearest this win)",
    )
    parser.add_argument(
        "--min-speech", type=float, default=4.0, help="shortest speech span, seconds"
    )
    parser.add_argument(
        "--max-speech", type=float, default=9.0, help="longest speech span, seconds"
    )
    parser.add_argument(
        "--min-lufs",
        type=float,
        default=-34.0,
        help="reject recordings quieter than this",
    )
    parser.add_argument(
        "--max-lufs",
        type=float,
        default=-14.0,
        help="reject recordings louder than this",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="scan only this many pages of 100 rows (for a quick smoke build)",
    )
    parser.add_argument(
        "--no-gender-balance",
        action="store_true",
        help="take the best clips overall instead of alternating genders",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="print only the final summary"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()

    filters = BankFilters(
        target_speech_seconds=args.target,
        min_speech_seconds=args.min_speech,
        max_speech_seconds=args.max_speech,
        min_lufs=args.min_lufs,
        max_lufs=args.max_lufs,
    )

    def report(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)

    try:
        bank = build_voice_bank(
            destination=args.out,
            settings=settings,
            voices=args.voices,
            filters=filters,
            source=HuggingFaceRowSource(
                args.dataset, split=args.split, token=settings.huggingface_token
            ),
            balance_genders=not args.no_gender_balance,
            max_pages=args.max_pages,
            report=report,
        )
    except BankBuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print()
    print(f"voice bank: {Path(bank.directory)}")
    print(f"  dataset : {bank.dataset} [{bank.split}] {bank.revision[:12]}")
    print(f"  licence : {bank.license or 'see the dataset card'}")
    print(f"  voices  : {len(bank)}")

    for voice in bank:
        pitch = f"{voice.pitch_hz:.0f} Hz" if voice.pitch_hz else "unmeasured"
        print(
            f"    {voice.voice_id:<22} {voice.gender:<6} {voice.duration:4.2f}s "
            f"{pitch:>10}  {voice.text[:48]}"
        )

    print()
    print("Set VOICE_BANK_DIR to this directory to dub with these voices.")
    print("Listen to a clip before trusting the casting: no filter hears a microphone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
