"""Download a Google Drive share link onto the worker.

The repository holds the code and the plan; an episode is tens of gigabytes and
belongs on the volume, not in git. This module is how it gets there: give it the
same link a browser would use, and it writes the file.

Why this is not simply a URL fetch
----------------------------------
For a file over roughly 100 MB, Google Drive does not answer the download URL with
the file. It answers with a small HTML page - the "Google Drive can't scan this file
for viruses" confirmation - carrying a form token. The file is only served once that
token is sent back. A plain downloader therefore saves the HTML page and calls it an
episode, which is a confusing failure much later in a run.

So the sequence is: ask for the file; if the answer is that page, read its token and
ask again with it. Cookies are kept between the two requests because Drive ties the
token to the session that received it.

Deliberate non-goals
--------------------
* **No auth dance.** The link has to be shared ("anyone with the link"). A private
  file returns the sign-in page, which is reported as such rather than retried.
* **No silent partial file.** Bytes go to a ``.part`` file and are renamed only after
  the size and the first bytes are checked.
* **No extra dependency.** ``urllib`` only, so this runs on a bare worker.
"""

from __future__ import annotations

import http.cookiejar
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

#: A browser user agent. Drive serves a different page to obvious clients.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

#: Largest HTML body read while looking for the confirmation token, in bytes. The
#: page is a few kilobytes; anything much larger is not that page.
MAX_CONFIRM_PAGE_BYTES = 512 * 1024

#: Smallest file worth accepting as an episode, in bytes. A tiny file is an error
#: page that happened to have a binary content type.
MIN_DOWNLOAD_BYTES = 1024 * 1024

#: Container signatures checked in the first bytes, so a saved HTML page cannot pass
#: for a video. ``offset`` is where the signature starts.
_CONTAINER_SIGNATURES: tuple[tuple[int, bytes], ...] = (
    (4, b"ftyp"),  # MP4 / MOV
    (0, b"\x1a\x45\xdf\xa3"),  # Matroska / WebM
    (0, b"RIFF"),  # AVI
    (0, b"FLV"),
    (0, b"\x00\x00\x01\xba"),  # MPEG program stream
    (0, b"\x47"),  # MPEG transport stream
)

_ID_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/file/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"/d/([A-Za-z0-9_-]{10,})"),
    re.compile(r"[?&]id=([A-Za-z0-9_-]{10,})"),
    re.compile(r"/uc/([A-Za-z0-9_-]{10,})"),
    re.compile(r"^([A-Za-z0-9_-]{20,})$"),
)

_CONFIRM_TOKEN = re.compile(r'name="confirm"\s+value="([^"]+)"', re.IGNORECASE)
_CONFIRM_UUID = re.compile(r'name="uuid"\s+value="([^"]+)"', re.IGNORECASE)
_DOWNLOAD_ACTION = re.compile(
    r'action="(https://drive\.usercontent\.google\.com/download[^"]*)"', re.IGNORECASE
)
_BLOCKED_PAGE_MARKERS = (
    "accounts.google.com/signin",
    "you need access",
    "request access",
    "permission denied",
)
_QUOTA_MARKERS = (
    "too many users have viewed or downloaded this file",
    "download quota",
)


class DriveError(RuntimeError):
    """A Google Drive link could not be turned into a file."""


def parse_file_id(link: str) -> str:
    """Return the Drive file id from a share link, a download URL or the bare id."""

    text = (link or "").strip()
    if not text:
        raise DriveError("no Google Drive link given")
    for pattern in _ID_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return match.group(1)
    raise DriveError(
        f"could not find a Google Drive file id in {text!r}; expected something like "
        "https://drive.google.com/file/d/<id>/view"
    )


def _opener() -> urllib.request.OpenerDirector:
    jar = http.cookiejar.CookieJar()
    return urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))


def _get(
    opener: urllib.request.OpenerDirector, url: str, *, timeout: float
) -> urllib.request.addinfourl:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    return opener.open(request, timeout=timeout)


def _content_type(response: object) -> str:
    headers = getattr(response, "headers", None)
    if headers is None:
        return ""
    return (headers.get("Content-Type", "") or "").split(";")[0].strip().lower()


def _looks_like_html(response: object) -> bool:
    return _content_type(response) in ("text/html", "application/xhtml+xml")


def _confirmation_url(html: str, file_id: str) -> str | None:
    """Return the URL that serves the file once the confirmation is answered."""

    lower = html.casefold()
    for marker in _QUOTA_MARKERS:
        if marker in lower:
            raise DriveError(
                "Google Drive is refusing this download because the file has been "
                "downloaded too many times recently; try again later, or copy it into "
                "your own Drive and share that copy"
            )
    for marker in _BLOCKED_PAGE_MARKERS:
        if marker in lower:
            raise DriveError(
                "Google Drive asked for a sign-in, so the link is not shared publicly; "
                "set the file to 'Anyone with the link' and try again"
            )

    parts = [f"id={file_id}", "export=download", "confirm=t"]
    uuid = _CONFIRM_UUID.search(html)
    if uuid is not None:
        parts.append(f"uuid={uuid.group(1)}")
    token = _CONFIRM_TOKEN.search(html)
    if token is not None and token.group(1) not in ("t", "true", ""):
        parts[-1] = f"confirm={urllib.parse.quote(token.group(1))}"
    action = _DOWNLOAD_ACTION.search(html)
    if action is not None:
        base = action.group(1)
        separator = "&" if "?" in base else "?"
        return f"{base}{separator}{'&'.join(parts)}"
    if token is None and uuid is None and action is None:
        return None
    return f"https://drive.usercontent.google.com/download?{'&'.join(parts)}"


def _looks_like_media(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return False
    return any(
        head[offset : offset + len(magic)] == magic
        for offset, magic in _CONTAINER_SIGNATURES
    )


def _stream(
    response: object,
    destination: Path,
    *,
    progress: Callable[[int, int], None] | None,
) -> int:
    length_header = getattr(response, "headers", None)
    total = 0
    if length_header is not None:
        try:
            total = int(length_header.get("Content-Length", "0") or 0)
        except (TypeError, ValueError):
            total = 0

    partial = destination.with_suffix(destination.suffix + ".part")
    written = 0
    with partial.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            handle.write(chunk)
            written += len(chunk)
            if progress is not None:
                progress(written, total)
    partial.replace(destination)
    return written


def download_file(
    link: str,
    destination: str | Path,
    *,
    timeout: float = 60.0,
    progress: Callable[[int, int], None] | None = None,
    overwrite: bool = False,
) -> Path:
    """Download the Drive file ``link`` points at, and return where it was written."""

    target = Path(destination)
    if target.exists() and target.stat().st_size >= MIN_DOWNLOAD_BYTES and not overwrite:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    file_id = parse_file_id(link)
    opener = _opener()
    first = f"https://drive.google.com/uc?export=download&id={file_id}"

    try:
        response = _get(opener, first, timeout=timeout)
    except urllib.error.HTTPError as exc:
        raise DriveError(f"Google Drive returned HTTP {exc.code} for file {file_id}") from exc
    except urllib.error.URLError as exc:
        raise DriveError(f"could not reach Google Drive: {exc.reason}") from exc

    if _looks_like_html(response):
        html = response.read(MAX_CONFIRM_PAGE_BYTES).decode("utf-8", errors="replace")
        response.close()
        # A virus-scan warning page for a big file is expected; a sign-in or a quota
        # page is not, and _confirmation_url raises for those.
        follow = _confirmation_url(html, file_id)
        if follow is None:
            raise DriveError(
                "Google Drive answered with a web page instead of the file, and it "
                "carries no download confirmation; the link is probably not a file "
                "link (open the file in Drive and use Share -> Copy link)"
            )
        try:
            response = _get(opener, follow, timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise DriveError(
                f"Google Drive returned HTTP {exc.code} after the download confirmation"
            ) from exc
        if _looks_like_html(response):
            page = response.read(MAX_CONFIRM_PAGE_BYTES).decode("utf-8", errors="replace")
            response.close()
            _confirmation_url(page, file_id)  # raises with the real reason
            raise DriveError(
                "Google Drive kept returning a web page instead of the file; open the "
                "link in a browser to see what it says"
            )

    if _looks_like_html(response):
        response.close()
        raise DriveError("Google Drive returned a web page instead of the file")

    try:
        written = _stream(response, target, progress=progress)
    finally:
        response.close()

    if written < MIN_DOWNLOAD_BYTES:
        target.unlink(missing_ok=True)
        raise DriveError(
            f"the download was only {written} bytes, which is not a video; the link "
            "may point at a folder or an empty file"
        )
    if not _looks_like_media(target):
        raise DriveError(
            f"{target} does not start like a video container (mp4/mov/mkv/avi); if "
            "Drive served an error page this is where it shows up"
        )
    return target


def safe_filename(name: str, *, default: str = "episode.mp4") -> str:
    """Return ``name`` as a filename that is safe on the worker."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (name or "").strip()).strip("._")
    if not cleaned:
        return default
    if "." not in cleaned:
        cleaned = f"{cleaned}.mp4"
    return cleaned


def filename_from_headers(response: object) -> str | None:
    """Return the filename Drive states in the headers, when it states one."""

    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    disposition = headers.get("Content-Disposition", "") or ""
    match = re.search(r"filename\*?=(?:UTF-8'')?\"?([^\";]+)\"?", disposition)
    if match is None:
        return None
    from urllib.parse import unquote

    return os.path.basename(unquote(match.group(1)))


__all__ = [
    "DriveError",
    "MIN_DOWNLOAD_BYTES",
    "download_file",
    "filename_from_headers",
    "parse_file_id",
    "safe_filename",
]
