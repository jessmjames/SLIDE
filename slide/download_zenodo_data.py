#!/usr/bin/env python3
"""Download and extract SLIDE raw and processed data from Zenodo.

The default targets the published SLIDE dataset at Zenodo record 21282413.
Draft depositions can still be accessed with ``--draft`` and an access token
provided through ``--access-token`` or ``ZENODO_ACCESS_TOKEN``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath


DEFAULT_ZENODO_ID = "21282413"
DEFAULT_ARCHIVES: tuple[str, ...] = ("raw_data.zip", "processed_data.zip")
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
ZENODO_API_ROOT = "https://zenodo.org/api"
ZENODO_RECORD_ROOT = "https://zenodo.org/records"


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
    - argparse.Namespace
        Parsed downloader options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Download raw_data.zip and processed_data.zip from Zenodo and "
            "extract them into raw_data/ and processed_data/."
        )
    )
    parser.add_argument(
        "--record-id",
        default=DEFAULT_ZENODO_ID,
        help="Zenodo record/deposition ID. Defaults to the published SLIDE dataset.",
    )
    parser.add_argument(
        "--access-token",
        default=os.getenv("ZENODO_ACCESS_TOKEN"),
        help=(
            "Zenodo access token. Defaults to ZENODO_ACCESS_TOKEN and is only "
            "required for draft/private deposition downloads."
        ),
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--published",
        dest="published",
        action="store_true",
        default=True,
        help="Download from the public records endpoint (default).",
    )
    mode_group.add_argument(
        "--draft",
        dest="published",
        action="store_false",
        help="Download from the private draft API; requires an access token.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite existing files without asking.",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help=(
            "Base folder into which raw_data and processed_data are placed. "
            "Defaults to the current working directory."
        ),
    )
    parser.add_argument(
        "--archive",
        action="append",
        choices=DEFAULT_ARCHIVES,
        help=(
            "Archive to download. May be supplied more than once. Defaults to "
            "raw_data.zip and processed_data.zip."
        ),
    )
    return parser.parse_args()


def url_with_token(url: str, access_token: str | None) -> str:
    """Append a Zenodo access token to a URL when one is available.

    Parameters:
    - url: str
        URL to update.
    - access_token: str | None
        Zenodo access token.

    Returns:
    - str
        URL with ``access_token`` query parameter when a token is supplied.
    """
    if not access_token:
        return url
    parsed = urllib.parse.urlparse(url)
    query = dict(urllib.parse.parse_qsl(parsed.query))
    query["access_token"] = access_token
    return urllib.parse.urlunparse(
        parsed._replace(query=urllib.parse.urlencode(query))
    )


def request_json(url: str, access_token: str | None) -> object:
    """Fetch JSON from Zenodo.

    Parameters:
    - url: str
        API URL to fetch.
    - access_token: str | None
        Zenodo access token.

    Returns:
    - object
        Decoded JSON payload.
    """
    request = urllib.request.Request(
        url_with_token(url, access_token),
        headers={"User-Agent": "SLIDE-data-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Zenodo returned HTTP {exc.code} for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Zenodo at {url}: {exc.reason}") from exc


def draft_file_download_urls(record_id: str, access_token: str | None) -> dict[str, str]:
    """Return draft-deposition file download URLs keyed by archive filename.

    Parameters:
    - record_id: str
        Zenodo draft deposition ID.
    - access_token: str | None
        Zenodo access token.

    Returns:
    - dict[str, str]
        Mapping from filename to authenticated download URL.
    """
    if not access_token:
        raise RuntimeError(
            "Draft Zenodo downloads require --access-token or ZENODO_ACCESS_TOKEN."
        )
    payload = request_json(
        f"{ZENODO_API_ROOT}/deposit/depositions/{record_id}/files",
        access_token,
    )
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Zenodo draft files response.")
    urls: dict[str, str] = {}
    for item in payload:
        if not isinstance(item, Mapping):
            continue
        filename = item.get("filename") or item.get("key")
        links = item.get("links")
        if not isinstance(filename, str) or not isinstance(links, Mapping):
            continue
        download_url = links.get("download")
        if isinstance(download_url, str):
            urls[filename] = url_with_token(download_url, access_token)
    return urls


def published_file_download_urls(record_id: str, access_token: str | None) -> dict[str, str]:
    """Return public-record file download URLs keyed by archive filename.

    Parameters:
    - record_id: str
        Published Zenodo record ID.
    - access_token: str | None
        Optional access token.

    Returns:
    - dict[str, str]
        Mapping from filename to download URL.
    """
    urls = {
        archive_name: f"{ZENODO_RECORD_ROOT}/{record_id}/files/{archive_name}?download=1"
        for archive_name in DEFAULT_ARCHIVES
    }
    return {name: url_with_token(url, access_token) for name, url in urls.items()}


def resolve_download_urls(
    record_id: str,
    access_token: str | None,
    published: bool,
) -> dict[str, str]:
    """Resolve archive download URLs for the requested Zenodo mode.

    Parameters:
    - record_id: str
        Zenodo record or draft deposition ID.
    - access_token: str | None
        Optional Zenodo access token.
    - published: bool
        Whether to use public-record URL construction instead of draft API lookup.

    Returns:
    - dict[str, str]
        Mapping from archive filename to download URL.
    """
    if published:
        return published_file_download_urls(record_id, access_token)
    return draft_file_download_urls(record_id, access_token)


def download_file(url: str, destination: Path) -> None:
    """Download a URL to a local file using streaming chunks.

    Parameters:
    - url: str
        Download URL.
    - destination: Path
        Local output path.

    Returns:
    - None
        The downloaded file is written to ``destination``.
    """
    print(f"Downloading {destination.name}...")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SLIDE-data-downloader/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            total_header = response.headers.get("Content-Length")
            total_size = int(total_header) if total_header else None
            downloaded = 0
            with destination.open("wb") as output_file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    output_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        percentage = 100.0 * downloaded / total_size
                        print(
                            f"\r  {downloaded / 1024**2:.1f} MiB / "
                            f"{total_size / 1024**2:.1f} MiB ({percentage:.1f}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        print(
                            f"\r  {downloaded / 1024**2:.1f} MiB",
                            end="",
                            flush=True,
                        )
            print()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Zenodo returned HTTP {exc.code} while downloading {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not download {url}: {exc.reason}") from exc


def confirm_overwrite(path: Path) -> bool:
    """Ask whether an existing file should be overwritten.

    Parameters:
    - path: Path
        Existing file path.

    Returns:
    - bool
        True when the user confirms overwrite.
    """
    while True:
        try:
            response = input(f"File already exists:\n  {path}\nOverwrite it? [y/N]: ")
        except EOFError:
            return False
        normalized = response.strip().lower()
        if normalized in {"y", "yes"}:
            return True
        if normalized in {"", "n", "no"}:
            return False
        print("Please enter 'y' or 'n'.")


def safe_relative_path(zip_member_name: str) -> Path:
    """Convert a ZIP member name into a safe relative path.

    Parameters:
    - zip_member_name: str
        Member name from the ZIP archive.

    Returns:
    - Path
        Safe relative path.
    """
    member_path = PurePosixPath(zip_member_name)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise RuntimeError(f"Unsafe path found in ZIP archive: {zip_member_name!r}")
    return Path(*member_path.parts)


def remove_common_top_level_folder(member_path: Path, expected_folder_name: str) -> Path:
    """Remove redundant archive top-level folder when present.

    Parameters:
    - member_path: Path
        Safe archive member path.
    - expected_folder_name: str
        Destination folder name, such as ``raw_data``.

    Returns:
    - Path
        Member path relative to the target destination folder.
    """
    parts = member_path.parts
    if parts and parts[0] == expected_folder_name:
        return Path(*parts[1:])
    return member_path


def extract_archive(archive_path: Path, destination_dir: Path, force: bool) -> tuple[int, int]:
    """Extract one ZIP archive into a destination directory.

    Parameters:
    - archive_path: Path
        ZIP archive to extract.
    - destination_dir: Path
        Target directory.
    - force: bool
        Whether to overwrite files without prompting.

    Returns:
    - tuple[int, int]
        Number of extracted files and skipped files.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    skipped = 0
    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"Downloaded file is not a valid ZIP archive: {archive_path}") from exc
    with archive:
        for member in archive.infolist():
            relative_path = remove_common_top_level_folder(
                safe_relative_path(member.filename),
                destination_dir.name,
            )
            if not relative_path.parts:
                continue
            output_path = destination_dir / relative_path
            if member.is_dir():
                output_path.mkdir(parents=True, exist_ok=True)
                continue
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not force:
                if not confirm_overwrite(output_path):
                    print(f"Skipping {output_path}")
                    skipped += 1
                    continue
            print(f"Extracting {output_path}")
            with archive.open(member, "r") as source:
                with output_path.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
            extracted += 1
    return extracted, skipped


def destination_for_archive(base_dir: Path, archive_name: str) -> Path:
    """Return extraction destination for a known archive.

    Parameters:
    - base_dir: Path
        Base repository or working directory.
    - archive_name: str
        Archive filename.

    Returns:
    - Path
        Destination directory under ``base_dir``.
    """
    if archive_name == "raw_data.zip":
        return base_dir / "raw_data"
    if archive_name == "processed_data.zip":
        return base_dir / "processed_data"
    raise RuntimeError(f"Unsupported archive name: {archive_name}")


def main() -> int:
    """Run the downloader command.

    Returns:
    - int
        Process exit code.
    """
    args = parse_arguments()
    base_dir = args.base_dir.expanduser().resolve()
    archive_names = tuple(args.archive) if args.archive else DEFAULT_ARCHIVES
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base folder: {base_dir}")
    print(f"Zenodo ID: {args.record_id}")
    print(f"Zenodo mode: {'published record' if args.published else 'draft deposition'}")
    print(f"Overwrite without prompting: {'yes' if args.force else 'no'}")

    try:
        download_urls = resolve_download_urls(
            record_id=str(args.record_id),
            access_token=args.access_token,
            published=bool(args.published),
        )
        missing_archives = [name for name in archive_names if name not in download_urls]
        if missing_archives:
            available = ", ".join(sorted(download_urls)) or "none"
            raise RuntimeError(
                "Requested archive(s) absent from Zenodo response: "
                f"{', '.join(missing_archives)}. Available: {available}"
            )

        total_extracted = 0
        total_skipped = 0
        with tempfile.TemporaryDirectory(prefix="slide_zenodo_download_") as temporary_directory:
            temporary_dir = Path(temporary_directory)
            for archive_name in archive_names:
                archive_path = temporary_dir / archive_name
                download_file(download_urls[archive_name], archive_path)
                extracted, skipped = extract_archive(
                    archive_path=archive_path,
                    destination_dir=destination_for_archive(base_dir, archive_name),
                    force=bool(args.force),
                )
                total_extracted += extracted
                total_skipped += skipped
                print(
                    f"Finished {archive_name}: {extracted} file(s) extracted, "
                    f"{skipped} file(s) skipped.\n"
                )
    except KeyboardInterrupt:
        print("\nCancelled by user.", file=sys.stderr)
        return 130
    except RuntimeError as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"\nFilesystem error: {exc}", file=sys.stderr)
        return 1

    print(f"Done. {total_extracted} file(s) extracted and {total_skipped} file(s) skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
