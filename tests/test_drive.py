"""Tests for :mod:`app.utils.drive`.

No network: the opener is replaced with one that serves scripted responses, which is
the only way to cover the confirmation handshake deterministically. The bytes used
carry a real MP4 signature so the container check is exercised for real.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.utils import drive

FILE_ID = "1AbCdEfGhIjKlMnOpQrStUvWxYz012345"
SHARE_LINK = f"https://drive.google.com/file/d/{FILE_ID}/view?usp=sharing"

#: A believable MP4 head: the "ftyp" box, then padding to a plausible size.
MP4_HEAD = b"\x00\x00\x00\x18ftypmp42" + b"\x00" * (drive.MIN_DOWNLOAD_BYTES + 16)

CONFIRM_PAGE = """<!DOCTYPE html><html><body>
<p>Google Drive can't scan this file for viruses.</p>
<form id="download-form" action="https://drive.usercontent.google.com/download">
  <input type="hidden" name="id" value="{file_id}">
  <input type="hidden" name="export" value="download">
  <input type="hidden" name="confirm" value="t">
  <input type="hidden" name="uuid" value="1234-abcd">
</form></body></html>"""


class _Headers(dict):
    def get(self, key, default=None):
        for name, value in self.items():
            if name.lower() == key.lower():
                return value
        return default


class _Response:
    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = body
        self._offset = 0
        self.headers = _Headers({"Content-Type": content_type})

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        pass


class _Opener:
    def __init__(self, responses: list[_Response]) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    def open(self, request, timeout=None):
        self.urls.append(request.full_url)
        if not self.responses:
            raise AssertionError("more requests than scripted responses")
        return self.responses.pop(0)


def _install(monkeypatch: pytest.MonkeyPatch, *responses: _Response) -> _Opener:
    opener = _Opener(list(responses))
    monkeypatch.setattr(drive, "_opener", lambda: opener)
    return opener


def test_a_share_link_yields_its_file_id():
    assert drive.parse_file_id(SHARE_LINK) == FILE_ID
    assert drive.parse_file_id(f"https://drive.google.com/open?id={FILE_ID}") == FILE_ID
    assert (
        drive.parse_file_id(f"https://drive.google.com/uc?id={FILE_ID}&export=download")
        == FILE_ID
    )
    assert drive.parse_file_id(FILE_ID) == FILE_ID


def test_a_folder_link_is_not_a_file_link():
    with pytest.raises(drive.DriveError):
        drive.parse_file_id("https://drive.google.com/drive/folders/abc")


def test_the_confirmation_page_is_answered_and_the_file_saved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    opener = _install(
        monkeypatch,
        _Response(CONFIRM_PAGE.format(file_id=FILE_ID).encode(), "text/html"),
        _Response(MP4_HEAD, "video/mp4"),
    )
    target = tmp_path / "episode.mp4"

    path = drive.download_file(SHARE_LINK, target)

    assert path == target
    assert target.read_bytes() == MP4_HEAD
    assert len(opener.urls) == 2
    assert "drive.usercontent.google.com/download" in opener.urls[1]
    assert f"id={FILE_ID}" in opener.urls[1]
    assert "confirm=t" in opener.urls[1]
    assert "uuid=1234-abcd" in opener.urls[1]


def test_a_direct_file_response_needs_no_handshake(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    opener = _install(monkeypatch, _Response(MP4_HEAD, "video/mp4"))

    path = drive.download_file(SHARE_LINK, tmp_path / "episode.mp4")

    assert path.stat().st_size == len(MP4_HEAD)
    assert len(opener.urls) == 1


def test_a_sign_in_page_says_the_link_is_not_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    page = b'<html><a href="https://accounts.google.com/signin">Sign in</a></html>'
    _install(monkeypatch, _Response(page, "text/html"))

    with pytest.raises(drive.DriveError, match="not shared publicly"):
        drive.download_file(SHARE_LINK, tmp_path / "episode.mp4")


def test_a_quota_refusal_is_reported_rather_than_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    page = b"<html>Too many users have viewed or downloaded this file recently.</html>"
    _install(monkeypatch, _Response(page, "text/html"))

    with pytest.raises(drive.DriveError, match="too many times"):
        drive.download_file(SHARE_LINK, tmp_path / "episode.mp4")


def test_an_html_page_is_never_saved_as_a_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install(monkeypatch, _Response(b"<html>something else</html>", "text/html"))
    target = tmp_path / "episode.mp4"

    with pytest.raises(drive.DriveError):
        drive.download_file(SHARE_LINK, target)

    assert not target.exists()
    assert not target.with_suffix(".mp4.part").exists()


def test_a_file_that_is_too_small_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install(monkeypatch, _Response(MP4_HEAD[:2048], "video/mp4"))
    target = tmp_path / "episode.mp4"

    with pytest.raises(drive.DriveError, match="not a video"):
        drive.download_file(SHARE_LINK, target)

    assert not target.exists()


def test_bytes_that_are_not_a_container_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install(monkeypatch, _Response(b"x" * (drive.MIN_DOWNLOAD_BYTES + 1), "video/mp4"))

    with pytest.raises(drive.DriveError, match="container"):
        drive.download_file(SHARE_LINK, tmp_path / "episode.mp4")


def test_an_already_downloaded_file_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    target = tmp_path / "episode.mp4"
    target.write_bytes(MP4_HEAD)
    opener = _install(monkeypatch)

    path = drive.download_file(SHARE_LINK, target)

    assert path == target
    assert opener.urls == []


def test_filenames_are_made_safe():
    assert drive.safe_filename("Breaking Bad S04E06.mkv") == "Breaking_Bad_S04E06.mkv"
    assert drive.safe_filename("../../etc/passwd") == "etc_passwd.mp4"
    assert drive.safe_filename("") == "episode.mp4"
    assert drive.safe_filename("episode") == "episode.mp4"
