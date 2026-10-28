"""Drive transfers through rclone, for the orchestrator and for scripts.

The episode is far too large for git and the finished dub is not far behind, so both
live in Google Drive. This module is the only mover: it builds the rclone commands,
works out where the credential came from, and checks that what arrived is really a
video.

Configuration is either a config file or the pod's environment
--------------------------------------------------------------
rclone reads a remote's settings from either an rclone config file or from
``RCLONE_CONFIG_<NAME>_<KEY>`` variables. Both are supported, and the environment form
is what makes a recreated pod need no manual step at all. A config *path* that does
not exist is refused here rather than passed to rclone, because rclone then fails
before it reads the variables that would have worked.

Why rclone rather than a plain downloader
-----------------------------------------
Resumable transfers, verification, progress, and Drive's own handshakes (the "can't
scan a file over 100 MB" confirmation, and the acknowledgement for a file Drive has
flagged) are all already correct in rclone. :mod:`app.utils.drive` keeps a one-shot
link fetch for the case where no remote is configured yet.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

from app.config import Settings
from app.utils import drive

#: Extensions treated as an episode when a pull is verified.
EPISODE_SUFFIXES = (".mp4", ".mkv", ".mov", ".webm", ".m4v")

#: Prefix rclone reads remote settings from, e.g. ``RCLONE_CONFIG_GDRIVE_TYPE``.
ENV_CONFIG_PREFIX = "RCLONE_CONFIG_"

#: A Windows drive path (``C:\\video.mp4``) looks like an rclone remote to a naive
#: test, so it is excluded explicitly.
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


class TransferError(RuntimeError):
    """A transfer could not be completed."""


def env_remote_configured(settings: Settings) -> bool:
    """``True`` when the remote is described by ``RCLONE_CONFIG_<REMOTE>_*`` variables."""

    remote = (settings.drive_remote or "").strip().rstrip(":").upper()
    if not remote:
        return False
    prefix = f"{ENV_CONFIG_PREFIX}{remote}_"
    return any(name.startswith(prefix) for name in os.environ)


def config_source(settings: Settings) -> tuple[str, str]:
    """Return ``(kind, detail)``: where the remote's credentials come from."""

    path = Path(settings.rclone_config)
    if path.is_file():
        return "file", str(path)
    if env_remote_configured(settings):
        return "environment", f"{ENV_CONFIG_PREFIX}{settings.drive_remote.upper()}_*"
    return "none", str(path)


def drive_ready(settings: Settings) -> bool:
    """``True`` when a transfer could be attempted without further setup."""

    if not (settings.drive_remote or "").strip():
        return False
    return config_source(settings)[0] != "none"


def looks_like_remote_path(value: str) -> bool:
    """``True`` for an rclone path such as ``gdrive:folder/file.mkv``."""

    text = (value or "").strip()
    if not text or ":" not in text:
        return False
    if text.lower().startswith(("http://", "https://")):
        return False
    if _WINDOWS_DRIVE.match(text):
        return False
    head, _, _rest = text.partition(":")
    # A remote name is one token: no separators and no spaces. Everything after the
    # colon is the path inside it, with or without a leading slash.
    if not head or any(character in head for character in "\\/ "):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.\-]+", head))


def remote_path(settings: Settings, folder: str, name: str = "") -> str:
    """Return ``remote:folder/name`` with the empty cases handled."""

    remote = (settings.drive_remote or "").strip()
    if not remote:
        raise TransferError(
            "DRIVE_REMOTE is empty; set it to the rclone remote name from your "
            "config or environment variables (for example DRIVE_REMOTE=gdrive)"
        )
    if ":" not in remote:
        remote = f"{remote}:"
    parts = [part.strip("/") for part in (folder or "", name or "") if part.strip()]
    return f"{remote}{'/'.join(parts)}" if parts else remote


def base_command(settings: Settings) -> list[str]:
    """Return the rclone invocation, checking that it and its credentials exist."""

    binary = settings.rclone_bin or "rclone"
    if shutil.which(binary) is None and not Path(binary).is_file():
        raise TransferError(
            f"rclone was not found ({binary!r}). Install it on the worker, for "
            "example: curl https://rclone.org/install.sh | sudo bash"
        )
    command = [binary]
    kind, detail = config_source(settings)
    if kind == "file":
        command += ["--config", detail]
    elif kind == "none":
        raise TransferError(
            "the Drive remote is not configured: there is no rclone config at "
            f"{detail} and no {ENV_CONFIG_PREFIX}* variables in the environment. "
            "Either copy an rclone config to that path, or set "
            f"{ENV_CONFIG_PREFIX}{settings.drive_remote.upper()}_TYPE (and _SCOPE, "
            "_TOKEN) as pod environment variables - see config/rclone.example.conf."
        )
    elif os.environ.get("RCLONE_CONFIG"):
        raise TransferError(
            "RCLONE_CONFIG points at a file that does not exist, which makes rclone "
            "fail before it reads the environment variables describing the remote. "
            "Unset RCLONE_CONFIG when configuring the remote through "
            f"{ENV_CONFIG_PREFIX}* variables, or create that file."
        )
    # Environment-configured: rclone reads RCLONE_CONFIG_<REMOTE>_* itself, so no
    # --config argument belongs here.
    return command


def pull_command(
    settings: Settings,
    *,
    name: str = "",
    destination: str | Path | None = None,
    acknowledge_abuse: bool = True,
    resync: bool = False,
) -> list[str]:
    """Return the ``rclone copy`` that brings an episode down from Drive.

    ``acknowledge_abuse`` sends ``--drive-acknowledge-abuse``. Drive serves a warning
    page instead of the file when it has flagged it - which happens to large video
    files it cannot scan - and this is rclone's documented switch for accepting that
    page on a file you own. Without it the transfer quietly saves an HTML file.
    """

    target = Path(destination) if destination is not None else Path(settings.input_dir)
    source = remote_path(settings, settings.drive_source_folder, name)
    command = base_command(settings) + [
        "copy",
        source,
        str(target),
        "--progress",
        "--stats-one-line",
        "--stats",
        "15s",
        "--create-empty-src-dirs",
        "--drive-chunk-size",
        "64M",
    ]
    if acknowledge_abuse:
        command.append("--drive-acknowledge-abuse")
    if resync:
        command.append("--ignore-times")
    return command


def fetch_command(
    settings: Settings,
    source: str,
    *,
    destination: str | Path | None = None,
    acknowledge_abuse: bool = True,
) -> list[str]:
    """Return the ``rclone copyto`` that fetches one named remote file."""

    target = Path(destination) if destination is not None else Path(settings.input_dir)
    command = base_command(settings) + [
        "copyto",
        source,
        str(target / Path(source).name),
        "--progress",
        "--stats-one-line",
        "--stats",
        "15s",
        "--drive-chunk-size",
        "64M",
    ]
    if acknowledge_abuse:
        command.append("--drive-acknowledge-abuse")
    return command


def push_command(
    settings: Settings,
    *,
    run: str | Path,
    name: str = "",
    dry_run: bool = False,
) -> list[str]:
    """Return the ``rclone copy`` that sends a finished run back to Drive."""

    run_path = Path(run)
    if not run_path.is_dir():
        raise TransferError(f"nothing to push: {run_path} is not a directory")
    folder = name or run_path.name
    destination = remote_path(settings, settings.drive_output_folder, folder)
    command = base_command(settings) + [
        "copy",
        str(run_path),
        destination,
        "--progress",
        "--stats-one-line",
        "--stats",
        "15s",
        # The dub is the deliverable. The timing and mix intermediates under a run
        # directory are not worth the upload, so only what a reviewer needs goes up.
        "--include",
        "*.mp4",
        "--include",
        "manifest.json",
    ]
    if dry_run:
        command.append("--dry-run")
    return command


def run_command(
    command: Sequence[str],
    *,
    dry_run: bool = False,
    report: Callable[[str], None] | None = None,
) -> int:
    """Run ``command`` to completion, reporting it first."""

    say = report if report is not None else (lambda message: print(message, flush=True))
    say("$ " + " ".join(command))
    if dry_run:
        return 0
    # rclone draws its own progress; letting it inherit stdout keeps the transfer
    # visible instead of buffered until the end.
    return subprocess.run(list(command), check=False).returncode


def verify_downloaded_episode(settings: Settings, name: str = "") -> Path | None:
    """Return the pulled episode, after checking it really is a video."""

    directory = Path(settings.input_dir)
    candidate = directory / name if name else None
    if candidate is None or not candidate.is_file():
        matches = sorted(
            path
            for path in directory.glob("*")
            if path.is_file() and path.suffix.lower() in EPISODE_SUFFIXES
        )
        candidate = matches[0] if len(matches) == 1 else None
    if candidate is None:
        return None
    if candidate.stat().st_size < drive.MIN_DOWNLOAD_BYTES:
        raise TransferError(
            f"{candidate} is only {candidate.stat().st_size} bytes; Drive returned a "
            "page instead of the file (keep --drive-acknowledge-abuse on, and check "
            "that the file is shared with the account the remote belongs to)"
        )
    return candidate


def pull_episode(
    settings: Settings,
    *,
    name: str = "",
    report: Callable[[str], None] | None = None,
) -> Path:
    """Bring ``name`` from the source folder into ``INPUT_DIR`` and verify it."""

    say = report if report is not None else (lambda message: print(message, flush=True))
    source = remote_path(settings, settings.drive_source_folder, name)
    say(f"  drive: pulling {source}")
    code = run_command(pull_command(settings, name=name), report=say)
    if code != 0:
        raise TransferError(f"rclone exited {code} while pulling {source}")
    episode = verify_downloaded_episode(settings, name)
    if episode is None:
        raise TransferError(
            "the transfer finished but no video appeared in "
            f"{settings.input_dir}; check DRIVE_SOURCE_FOLDER and the file name"
        )
    say(f"  drive: {episode.name} ({episode.stat().st_size / 1e6:.0f} MB)")
    return episode


def fetch_remote_file(
    settings: Settings,
    source: str,
    *,
    report: Callable[[str], None] | None = None,
) -> Path:
    """Fetch one ``remote:path/file`` into ``INPUT_DIR`` and verify it."""

    say = report if report is not None else (lambda message: print(message, flush=True))
    say(f"  drive: fetching {source}")
    code = run_command(fetch_command(settings, source), report=say)
    if code != 0:
        raise TransferError(f"rclone exited {code} while fetching {source}")
    episode = verify_downloaded_episode(settings, Path(source).name)
    if episode is None:
        raise TransferError(
            f"the transfer finished but {Path(source).name} did not appear in "
            f"{settings.input_dir}"
        )
    say(f"  drive: {episode.name} ({episode.stat().st_size / 1e6:.0f} MB)")
    return episode


def push_run(
    settings: Settings,
    run: str | Path,
    *,
    name: str = "",
    report: Callable[[str], None] | None = None,
    dry_run: bool = False,
) -> str:
    """Upload a finished run's dub and manifest, returning the remote location."""

    say = report if report is not None else (lambda message: print(message, flush=True))
    folder = name or Path(run).name
    destination = remote_path(settings, settings.drive_output_folder, folder)
    code = run_command(
        push_command(settings, run=run, name=name, dry_run=dry_run),
        dry_run=dry_run,
        report=say,
    )
    if code != 0:
        raise TransferError(f"rclone exited {code} while pushing to {destination}")
    say(f"  drive: {destination}")
    return destination


def with_config(settings: Settings, path: str | Path | None) -> Settings:
    """Return ``settings`` with a different rclone config path, when one is given."""

    if path is None:
        return settings
    return replace(settings, rclone_config=Path(path))


def describe(settings: Settings) -> str:
    """Return a one-line statement of where the credentials come from."""

    kind, detail = config_source(settings)
    label = {
        "file": "config file",
        "environment": "environment variables",
        "none": "not configured",
    }[kind]
    return f"{label}: {detail}"


__all__ = [
    "ENV_CONFIG_PREFIX",
    "EPISODE_SUFFIXES",
    "TransferError",
    "base_command",
    "config_source",
    "describe",
    "drive_ready",
    "env_remote_configured",
    "fetch_command",
    "fetch_remote_file",
    "looks_like_remote_path",
    "pull_command",
    "pull_episode",
    "push_command",
    "push_run",
    "remote_path",
    "run_command",
    "verify_downloaded_episode",
    "with_config",
]
