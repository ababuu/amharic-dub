"""Tests for :mod:`app.utils.rclone` and the ``drive_sync`` command line.

rclone is never executed: the tests assert the command it would run, which is where
the mistakes are (a missing config path, a missing acknowledgement flag, pushing the
audio intermediates instead of the dub), and cover both ways of describing the
remote - a config file and the pod's environment variables.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from app.config import Settings
from app.utils import drive, rclone
from scripts import drive_sync


def _settings(root: Path, **overrides: object) -> Settings:
    values: dict[str, object] = {
        "input_dir": root / "input",
        "work_dir": root / "work",
        "output_dir": root / "out",
        "model_cache_dir": root / "models",
        "drive_remote": "gdrive",
        "drive_source_folder": "amharic-dub/input",
        "drive_output_folder": "amharic-dub/output",
        "rclone_config": root / "rclone.conf",
        "rclone_bin": "rclone",
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def _with_config(settings: Settings, root: Path) -> Settings:
    root.mkdir(parents=True, exist_ok=True)
    (root / "rclone.conf").write_text("[gdrive]\ntype = drive\n", encoding="utf-8")
    return settings


@pytest.fixture(autouse=True)
def _rclone_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend rclone is on PATH, so the tests describe commands rather than a host."""

    monkeypatch.setattr(rclone.shutil, "which", lambda name: f"/usr/local/bin/{name}")


@pytest.fixture(autouse=True)
def _no_env_remote(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's own rclone variables out of the results."""

    for name in list(__import__("os").environ):
        if name.startswith(rclone.ENV_CONFIG_PREFIX) or name == "RCLONE_CONFIG":
            monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Addressing and credential source
# ---------------------------------------------------------------------------


def test_the_remote_path_handles_empty_folders():
    settings = _settings(Path("."))

    assert rclone.remote_path(settings, "a/b", "c.mkv") == "gdrive:a/b/c.mkv"
    assert rclone.remote_path(settings, "", "") == "gdrive:"
    assert rclone.remote_path(settings, "/a/", "/b/") == "gdrive:a/b"


def test_a_remote_without_a_colon_is_still_a_remote():
    settings = _settings(Path("."), drive_remote="my-remote")

    assert rclone.remote_path(settings, "", "") == "my-remote:"


def test_an_empty_remote_is_reported_clearly(tmp_path: Path):
    settings = _settings(tmp_path, drive_remote="  ")

    with pytest.raises(rclone.TransferError, match="DRIVE_REMOTE is empty"):
        rclone.pull_command(settings, name="x.mkv")


def test_remote_paths_are_told_apart_from_local_paths():
    assert rclone.looks_like_remote_path("gdrive:amharic-dub/input/x.mkv")
    assert rclone.looks_like_remote_path("gdrive:")
    assert not rclone.looks_like_remote_path("data/input/x.mp4")
    assert not rclone.looks_like_remote_path(r"C:\videos\x.mp4")
    assert not rclone.looks_like_remote_path("https://drive.google.com/file/d/abc/view")
    assert not rclone.looks_like_remote_path("")


def test_a_config_file_is_preferred_when_it_exists(tmp_path: Path):
    settings = _with_config(_settings(tmp_path), tmp_path)

    kind, detail = rclone.config_source(settings)
    command = rclone.pull_command(settings, name="x.mkv")

    assert kind == "file"
    assert detail == str(tmp_path / "rclone.conf")
    assert "--config" in command


def test_a_missing_config_with_no_environment_config_is_reported(tmp_path: Path):
    settings = _settings(tmp_path)

    with pytest.raises(rclone.TransferError, match="not configured"):
        rclone.pull_command(settings, name="x.mkv")


def test_the_remote_can_be_configured_from_environment_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No config file: the pod's environment variables describe the remote."""

    monkeypatch.setenv("RCLONE_CONFIG_GDRIVE_TYPE", "drive")
    monkeypatch.setenv("RCLONE_CONFIG_GDRIVE_SCOPE", "drive")
    monkeypatch.setenv("RCLONE_CONFIG_GDRIVE_TOKEN", '{"refresh_token":"x"}')
    settings = _settings(tmp_path)

    command = rclone.pull_command(settings, name="S04E06.mkv")

    assert rclone.config_source(settings)[0] == "environment"
    assert rclone.drive_ready(settings) is True
    # rclone reads RCLONE_CONFIG_* itself; a --config argument would override it.
    assert "--config" not in command
    assert "gdrive:amharic-dub/input/S04E06.mkv" in command


def test_a_dangling_config_path_is_reported_instead_of_being_passed_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """RCLONE_CONFIG pointing at nothing makes rclone fail before it reads the vars."""

    monkeypatch.setenv("RCLONE_CONFIG_GDRIVE_TYPE", "drive")
    monkeypatch.setenv("RCLONE_CONFIG", str(tmp_path / "missing.conf"))
    settings = _settings(tmp_path)

    with pytest.raises(rclone.TransferError, match="RCLONE_CONFIG points at a file"):
        rclone.pull_command(settings, name="x.mkv")


def test_a_remote_named_differently_uses_its_own_variables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setenv("RCLONE_CONFIG_WORKDRIVE_TYPE", "drive")
    settings = _settings(tmp_path, drive_remote="workdrive")

    assert rclone.env_remote_configured(settings) is True
    assert rclone.env_remote_configured(
        _settings(tmp_path, drive_remote="gdrive")
    ) is False


def test_nothing_configured_means_not_ready(tmp_path: Path):
    assert rclone.drive_ready(_settings(tmp_path)) is False
    assert rclone.drive_ready(_settings(tmp_path, drive_remote="")) is False


def test_describe_names_the_credential_source(tmp_path: Path):
    settings = _with_config(_settings(tmp_path / "configured"), tmp_path / "configured")

    assert "config file" in rclone.describe(settings)
    assert "not configured" in rclone.describe(_settings(tmp_path / "bare"))


# ---------------------------------------------------------------------------
# A damaged token is named, not passed through
# ---------------------------------------------------------------------------


def test_a_stripped_environment_token_is_named_instead_of_being_passed_on(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The reported failure: the value lost its leading '{', and rclone's own message
    ("invalid character 'a' looking for beginning of object key string") names neither
    the variable nor the fix."""

    monkeypatch.setenv("RCLONE_CONFIG_GDRIVE_TYPE", "drive")
    monkeypatch.setenv(
        "RCLONE_CONFIG_GDRIVE_TOKEN", 'access_token":"a","refresh_token":"r"}'
    )
    settings = _settings(tmp_path)

    with pytest.raises(rclone.TransferError) as caught:
        rclone.pull_command(settings, name="S01E01.mp4")

    message = str(caught.value)
    assert "RCLONE_CONFIG_GDRIVE_TOKEN" in message
    assert "not valid JSON" in message
    assert "single quotes" in message


def test_a_correctly_quoted_token_is_used_as_is(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The fix for that failure is the quoting, so a sound token must just work."""

    monkeypatch.setenv("RCLONE_CONFIG_GDRIVE_TYPE", "drive")
    monkeypatch.setenv(
        "RCLONE_CONFIG_GDRIVE_TOKEN",
        '{"access_token":"a","token_type":"Bearer","refresh_token":"r"}',
    )
    settings = _settings(tmp_path)

    assert rclone.malformed_env_token(settings) is None
    command = rclone.pull_command(settings, name="S01E01.mp4")
    assert "--config" not in command



# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def test_pull_acknowledges_drives_warning_page(tmp_path: Path):
    settings = _with_config(_settings(tmp_path), tmp_path)

    command = rclone.pull_command(settings, name="S04E06.mkv")

    assert "--drive-acknowledge-abuse" in command
    assert "gdrive:amharic-dub/input/S04E06.mkv" in command
    assert str(tmp_path / "input") in command
    assert command[0] == "rclone"


def test_pull_can_skip_the_acknowledgement(tmp_path: Path):
    settings = _with_config(_settings(tmp_path), tmp_path)

    command = rclone.pull_command(settings, name="x.mkv", acknowledge_abuse=False)

    assert "--drive-acknowledge-abuse" not in command


def test_pull_copies_the_whole_folder_when_no_name_is_given(tmp_path: Path):
    settings = _with_config(_settings(tmp_path), tmp_path)

    command = rclone.pull_command(settings)

    assert "copy" in command
    assert "gdrive:amharic-dub/input" in command


def test_fetching_one_remote_file_uses_copyto(tmp_path: Path):
    settings = _with_config(_settings(tmp_path), tmp_path)

    command = rclone.fetch_command(settings, "gdrive:amharic-dub/input/S04E06.mkv")

    assert "copyto" in command
    source = command[command.index("copyto") + 1]
    assert source == "gdrive:amharic-dub/input/S04E06.mkv"
    assert str(tmp_path / "input" / "S04E06.mkv") in command
    assert "--drive-acknowledge-abuse" in command


def test_push_uploads_only_the_dub_and_the_manifest(tmp_path: Path):
    run = tmp_path / "out" / "episode"
    run.mkdir(parents=True)
    (run / "episode_amharic.mp4").write_bytes(b"x")
    settings = _with_config(_settings(tmp_path), tmp_path)

    command = rclone.push_command(settings, run=run)

    assert "gdrive:amharic-dub/output/episode" in command
    assert "--include" in command
    assert "*.mp4" in command
    assert "manifest.json" in command


def test_push_names_the_remote_folder_after_the_run(tmp_path: Path):
    run = tmp_path / "out" / "S04E06"
    run.mkdir(parents=True)
    settings = _with_config(_settings(tmp_path), tmp_path)

    command = rclone.push_command(settings, run=run, name="final")

    assert "gdrive:amharic-dub/output/final" in command


def test_push_refuses_a_directory_that_does_not_exist(tmp_path: Path):
    settings = _with_config(_settings(tmp_path), tmp_path)

    with pytest.raises(rclone.TransferError):
        rclone.push_command(settings, run=tmp_path / "nope")


def test_a_missing_rclone_is_reported_clearly(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(rclone.shutil, "which", lambda name: None)
    settings = _settings(tmp_path, rclone_bin="definitely-not-rclone")

    with pytest.raises(rclone.TransferError, match="rclone was not found"):
        rclone.pull_command(settings, name="x.mkv")


# ---------------------------------------------------------------------------
# The separation cache
# ---------------------------------------------------------------------------
# A rented pod's container disk dies with the pod, and separation costs over an hour
# per episode, so the stems are kept on Drive and dropped back into the run.


def _separator_settings(root: Path) -> Settings:
    configured = _with_config(_settings(root), root)
    return dataclasses.replace(configured, drive_stem_folder="amharic-dub-cache/stems")


def _record_remote(monkeypatch: pytest.MonkeyPatch, sizes: dict[str, int]) -> None:
    """Make the remote listing return ``sizes`` without running rclone."""

    monkeypatch.setattr(rclone, "remote_sizes", lambda _settings, _folder: dict(sizes))


def test_a_stem_already_cached_is_not_uploaded_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _separator_settings(tmp_path)
    stem = tmp_path / "S01E01_speech.wav"
    stem.write_bytes(b"1234")
    _record_remote(monkeypatch, {"S01E01_speech.wav": 4})
    commands: list[list[str]] = []
    monkeypatch.setattr(
        rclone, "run_command", lambda command, **_kw: commands.append(command) or 0
    )

    uploaded = rclone.cache_stems(
        settings, folder=settings.drive_stem_folder, stem="S01E01", files=[stem]
    )

    assert uploaded == []
    assert commands == []


def test_a_stem_missing_from_the_cache_is_uploaded_with_copyto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _separator_settings(tmp_path)
    stem = tmp_path / "S01E01_speech.wav"
    stem.write_bytes(b"1234")
    _record_remote(monkeypatch, {})
    commands: dict[str, list[str]] = {}

    def record(command, **_kwargs):
        commands[Path(command[-1]).name if command[-1].startswith("/") else command[1]] = (
            command
        )
        return 0

    monkeypatch.setattr(rclone, "run_command", record)

    uploaded = rclone.cache_stems(
        settings, folder=settings.drive_stem_folder, stem="S01E01", files=[stem]
    )

    assert uploaded == ["S01E01_speech.wav"]
    (command,) = commands.values()
    assert "copyto" in command
    assert "gdrive:amharic-dub-cache/stems/S01E01_speech.wav" in command


def test_a_failed_cache_upload_names_the_consequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _separator_settings(tmp_path)
    stem = tmp_path / "S01E01_speech.wav"
    stem.write_bytes(b"1234")
    _record_remote(monkeypatch, {})
    monkeypatch.setattr(rclone, "run_command", lambda command, **_kw: 1)

    with pytest.raises(rclone.TransferError, match="separate again"):
        rclone.cache_stems(
            settings, folder=settings.drive_stem_folder, stem="S01E01", files=[stem]
        )


def test_restoring_copies_only_the_files_the_cache_has(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _separator_settings(tmp_path)
    target = tmp_path / "stages"
    _record_remote(
        monkeypatch,
        {"S01E01_speech.wav": 10, "S01E01_music.wav": 10},
    )
    copied: list[str] = []

    def record(command, **_kwargs):
        copied.append(command[1])
        Path(command[2]).write_bytes(b"x")
        return 0

    monkeypatch.setattr(rclone, "run_command", record)

    restored = rclone.restore_stems(
        settings,
        folder=settings.drive_stem_folder,
        stem="S01E01",
        destination=target,
        names=("S01E01_speech.wav", "S01E01_music.wav", "S01E01_effects.wav"),
    )

    assert [path.name for path in restored] == ["S01E01_speech.wav", "S01E01_music.wav"]
    # The effects stem is regenerated locally, so it is never downloaded.
    assert all("effects" not in source for source in copied)


def test_restoring_nothing_cached_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    settings = _separator_settings(tmp_path)
    _record_remote(monkeypatch, {})

    assert (
        rclone.restore_stems(
            settings,
            folder=settings.drive_stem_folder,
            stem="S01E01",
            destination=tmp_path / "stages",
            names=("S01E01_speech.wav",),
        )
        == []
    )


def test_an_unreachable_remote_lists_no_sizes_instead_of_failing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A failed listing must cost the cache, not the run."""

    settings = _separator_settings(tmp_path)

    class Failing:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(rclone.subprocess, "run", lambda *a, **k: Failing())

    assert rclone.remote_sizes(settings, "amharic-dub-cache/stems") == {}


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def test_a_pulled_file_that_is_too_small_is_refused(tmp_path: Path):
    settings = _settings(tmp_path)
    (tmp_path / "input").mkdir()
    (tmp_path / "input" / "episode.mp4").write_bytes(b"<html></html>")

    with pytest.raises(rclone.TransferError, match="page instead of the file"):
        rclone.verify_downloaded_episode(settings, "episode.mp4")


def test_a_pulled_episode_is_found(tmp_path: Path):
    settings = _settings(tmp_path)
    (tmp_path / "input").mkdir()
    episode = tmp_path / "input" / "S04E06.mp4"
    episode.write_bytes(b"\x00" * (drive.MIN_DOWNLOAD_BYTES + 1))

    assert rclone.verify_downloaded_episode(settings, "S04E06.mp4") == episode


# ---------------------------------------------------------------------------
# The command line over it
# ---------------------------------------------------------------------------


def test_pull_without_a_name_or_all_is_refused(tmp_path: Path, monkeypatch, capsys):
    settings = _with_config(_settings(tmp_path), tmp_path)
    monkeypatch.setattr(drive_sync, "get_settings", lambda: settings)

    code = drive_sync.main(["pull"])

    assert code == 1
    assert "name the file to pull" in capsys.readouterr().err


def test_its_dry_run_prints_the_command_and_runs_nothing(tmp_path: Path, monkeypatch, capsys):
    settings = _with_config(_settings(tmp_path), tmp_path)
    monkeypatch.setattr(drive_sync, "get_settings", lambda: settings)
    calls: list[list[str]] = []
    monkeypatch.setattr(
        rclone.subprocess,
        "run",
        lambda command, check=False: calls.append(command),
    )

    code = drive_sync.main(["pull", "--name", "S04E06.mkv", "--dry-run"])

    assert code == 0
    assert calls == []
    assert " copy " in capsys.readouterr().out


def test_its_status_reports_the_credential_source(tmp_path: Path, monkeypatch, capsys):
    settings = _with_config(_settings(tmp_path), tmp_path)
    monkeypatch.setattr(drive_sync, "get_settings", lambda: settings)

    code = drive_sync.main(["status", "--dry-run"])

    out = capsys.readouterr().out
    assert code == 0
    assert "credentials     config file" in out
    assert "about" in out


def test_its_config_flag_overrides_the_setting(tmp_path: Path, monkeypatch, capsys):
    settings = _settings(tmp_path)
    monkeypatch.setattr(drive_sync, "get_settings", lambda: settings)
    other = tmp_path / "elsewhere.conf"
    other.write_text("[gdrive]\ntype = drive\n", encoding="utf-8")

    code = drive_sync.main(["status", "--dry-run", "--config", str(other)])

    assert code == 0
    assert str(other) in capsys.readouterr().out
