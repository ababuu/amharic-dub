#!/usr/bin/env python3
"""Move episodes and finished dubs between Google Drive and the worker with rclone.

A thin command line over :mod:`app.utils.rclone`, which the orchestrator also uses.
The transfer itself - the rclone command, the credential source, the check that what
arrived is a video - lives in the app layer so both entry points behave the same.

Setup is described in ``config/rclone.example.conf``. In short: either copy an rclone
config to ``RCLONE_CONFIG``, or set ``RCLONE_CONFIG_<REMOTE>_TYPE/_SCOPE/_TOKEN`` as
pod environment variables. ``status`` says which one is in use.

Usage
-----
    python scripts/drive_sync.py status
    python scripts/drive_sync.py ls
    python scripts/drive_sync.py pull --name "S04E06.mkv"
    python scripts/drive_sync.py pull --all
    python scripts/drive_sync.py push --run data/output/episode
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings  # noqa: E402
from app.utils import rclone  # noqa: E402

#: Backwards-compatible name: this module raised ``SyncError`` before the transfer
#: logic moved into the app layer.
SyncError = rclone.TransferError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/drive_sync.py",
        description=(
            "Pull an episode from Google Drive and push a finished dub back, through "
            "rclone. Setup is described in config/rclone.example.conf."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("status", "check the remote and its quota"),
        ("ls", "list the source folder on Drive"),
    ):
        sub = subparsers.add_parser(name, help=help_text)
        sub.add_argument("--dry-run", action="store_true")
        sub.add_argument("--config", type=Path, default=None)

    pull = subparsers.add_parser("pull", help="download an episode into INPUT_DIR")
    pull.add_argument("--name", default="", help="file name inside the source folder")
    pull.add_argument(
        "--all", action="store_true", help="copy everything in the source folder"
    )
    pull.add_argument("--dry-run", action="store_true")
    pull.add_argument(
        "--resync",
        action="store_true",
        help="re-copy even when rclone thinks the local file is current",
    )
    pull.add_argument(
        "--no-acknowledge",
        action="store_true",
        help=(
            "do not send --drive-acknowledge-abuse. That flag is what accepts Drive's "
            "warning page for a large file it could not scan; without it the transfer "
            "saves a web page instead of the video."
        ),
    )
    pull.add_argument("--config", type=Path, default=None)

    push = subparsers.add_parser("push", help="upload a finished run to Drive")
    push.add_argument("--run", type=Path, required=True, help="the run directory")
    push.add_argument("--name", default="", help="folder name to create on Drive")
    push.add_argument("--dry-run", action="store_true")
    push.add_argument("--config", type=Path, default=None)

    args = parser.parse_args(argv)
    settings = rclone.with_config(get_settings(), getattr(args, "config", None))

    try:
        if args.command == "status":
            return _status(settings, args.dry_run)
        if args.command == "ls":
            return _ls(settings, args.dry_run)
        if args.command == "pull":
            return _pull(settings, args)
        return _push(settings, args)
    except rclone.TransferError as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1


def _status(settings: Settings, dry_run: bool) -> int:
    print(f"remote          {settings.drive_remote}")
    print(f"source folder   {settings.drive_source_folder or '(root)'}")
    print(f"output folder   {settings.drive_output_folder}")
    print(f"credentials     {rclone.describe(settings)}")
    print(f"rclone binary   {settings.rclone_bin}")
    print(f"local input     {settings.input_dir}")
    command = rclone.base_command(settings) + [
        "about",
        rclone.remote_path(settings, settings.drive_source_folder),
    ]
    code = rclone.run_command(command, dry_run=dry_run)
    if code != 0 and not dry_run:
        print(
            "\nrclone could not read the remote. Either the credentials are not on "
            "this machine, or the refresh token was revoked - re-authorise and either "
            "copy the config across or update the token variable.",
            file=sys.stderr,
        )
    return code


def _ls(settings: Settings, dry_run: bool) -> int:
    command = rclone.base_command(settings) + [
        "lsjson",
        rclone.remote_path(settings, settings.drive_source_folder),
        "--no-modtime",
        "--files-only",
    ]
    return rclone.run_command(command, dry_run=dry_run)


def _pull(settings: Settings, args: argparse.Namespace) -> int:
    name = "" if args.all else (args.name or "")
    if not args.all and not name:
        raise rclone.TransferError("name the file to pull (--name NAME) or use --all")
    command = rclone.pull_command(
        settings,
        name=name,
        acknowledge_abuse=not args.no_acknowledge,
        resync=args.resync,
    )
    code = rclone.run_command(command, dry_run=args.dry_run)
    if code != 0 or args.dry_run:
        return code
    episode = rclone.verify_downloaded_episode(settings, name)
    if episode is None:
        raise rclone.TransferError(
            "the transfer finished but no video appeared in " + str(settings.input_dir)
        )
    print(f"\npulled {episode} ({episode.stat().st_size / 1e6:.0f} MB)")
    print(
        "next: python -m app.pipeline.orchestrator "
        f"{episode} --cast-from-bank --out data/output/{episode.stem}"
    )
    return code


def _push(settings: Settings, args: argparse.Namespace) -> int:
    rclone.push_run(settings, args.run, name=args.name, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
