#!/usr/bin/env python3
"""
Download and extract the SLIDE raw and processed datasets from Zenodo.

By default:
- Files are extracted into ./raw_data and ./processed_data.
- Existing files trigger an interactive overwrite prompt.

Use --force to overwrite all existing files without prompting:

    python download_data.py --force
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath


# Replace this with the numerical ID of the published Zenodo record.
ZENODO_RECORD_ID = "12345678"

ZENODO_BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files"

DATASETS = {
    "raw_data.zip": {
        "url": f"{ZENODO_BASE_URL}/raw_data.zip?download=1",
        "destination": "raw_data",
    },
    "processed_data.zip": {
        "url": f"{ZENODO_BASE_URL}/processed_data.zip?download=1",
        "destination": "processed_data",
    },
}

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download raw_data.zip and processed_data.zip from Zenodo and "
            "extract them into the current base folder."
        )
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
    return parser.parse_args()


def download_file(url: str, destination: Path) -> None:
    """Download a URL to a local file using streaming chunks."""
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
                            f"{total_size / 1024**2:.1f} MiB "
                            f"({percentage:.1f}%)",
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
        raise RuntimeError(
            f"Zenodo returned HTTP {exc.code} while downloading:\n{url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not download {url}: {exc.reason}"
        ) from exc


def confirm_overwrite(path: Path) -> bool:
    """Ask whether an existing file should be overwritten."""
    while True:
        try:
            response = input(
                f"File already exists:\n  {path}\nOverwrite it? [y/N]: "
            ).strip().lower()
        except EOFError:
            return False

        if response in {"y", "yes"}:
            return True
        if response in {"", "n", "no"}:
            return False

        print("Please enter 'y' or 'n'.")


def safe_relative_path(zip_member_name: str) -> Path:
    """
    Convert a ZIP member name into a safe relative path.

    Absolute paths and '..' traversal components are rejected.
    """
    member_path = PurePosixPath(zip_member_name)

    if member_path.is_absolute() or ".." in member_path.parts:
        raise RuntimeError(
            f"Unsafe path found in ZIP archive: {zip_member_name!r}"
        )

    return Path(*member_path.parts)


def remove_common_top_level_folder(
    member_path: Path,
    expected_folder_name: str,
) -> Path:
    """
    Avoid producing raw_data/raw_data/... if the ZIP already contains a
    top-level raw_data folder, and similarly for processed_data.
    """
    parts = member_path.parts

    if parts and parts[0] == expected_folder_name:
        return Path(*parts[1:])

    return member_path


def extract_archive(
    archive_path: Path,
    destination_dir: Path,
    force: bool,
) -> tuple[int, int]:
    """
    Extract an archive into destination_dir.

    Returns:
        (number_extracted, number_skipped)
    """
    destination_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0

    try:
        archive = zipfile.ZipFile(archive_path)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(
            f"Downloaded file is not a valid ZIP archive: {archive_path}"
        ) from exc

    with archive:
        for member in archive.infolist():
            relative_path = safe_relative_path(member.filename)
            relative_path = remove_common_top_level_folder(
                relative_path,
                destination_dir.name,
            )

            # This can occur for the top-level raw_data/ directory itself.
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


def main() -> int:
    args = parse_arguments()
    base_dir = args.base_dir.expanduser().resolve()

    if ZENODO_RECORD_ID == "12345678":
        print(
            "Error: replace ZENODO_RECORD_ID in the script with the "
            "published Zenodo record ID.",
            file=sys.stderr,
        )
        return 1

    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base folder: {base_dir}")
    print(f"Overwrite without prompting: {'yes' if args.force else 'no'}")

    total_extracted = 0
    total_skipped = 0

    try:
        with tempfile.TemporaryDirectory(
            prefix="slide_zenodo_download_"
        ) as temporary_directory:
            temporary_dir = Path(temporary_directory)

            for archive_name, dataset in DATASETS.items():
                archive_path = temporary_dir / archive_name
                destination_dir = base_dir / dataset["destination"]

                download_file(dataset["url"], archive_path)

                extracted, skipped = extract_archive(
                    archive_path=archive_path,
                    destination_dir=destination_dir,
                    force=args.force,
                )

                total_extracted += extracted
                total_skipped += skipped

                print(
                    f"Finished {archive_name}: "
                    f"{extracted} file(s) extracted, "
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

    print(
        f"Done. {total_extracted} file(s) extracted and "
        f"{total_skipped} file(s) skipped."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
