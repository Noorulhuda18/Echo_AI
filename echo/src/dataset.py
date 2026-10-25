"""Dataset discovery and safe file management.

Genre labels come ONLY from the folder you put files in:

    data/raw/classical/*.mid   -> genre "classical"
    data/raw/jazz/*.mid        -> genre "jazz"
    data/raw/*.mid             -> genre "general" (unlabelled)

ECHO never guesses a genre from a file name. If you put a file in the
"jazz" folder, you are the one confirming it is jazz.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import config
from src.utils import (
    DatasetError,
    InvalidInputError,
    ensure_writable_dir,
    get_logger,
    slugify,
    unique_path,
)

log = get_logger("dataset")


@dataclass
class DatasetScan:
    """Result of looking at a folder (no MIDI parsing happens here)."""

    genre: str
    files: list[Path] = field(default_factory=list)  # unique MIDI files
    hashes: dict[Path, str] = field(default_factory=dict)
    duplicates: list[tuple[Path, Path]] = field(default_factory=list)  # (copy, original)
    unsupported: list[Path] = field(default_factory=list)

    @property
    def discovered(self) -> int:
        """All MIDI files found, including duplicates."""
        return len(self.files) + len(self.duplicates)


def file_sha1(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def list_genres(raw_dir: Path) -> dict[str, int]:
    """Map each genre folder to its number of MIDI files.

    Files placed directly in ``raw_dir`` are reported as ``general``.
    """
    genres: dict[str, int] = {}
    if not raw_dir.is_dir():
        return genres
    top_level = [
        p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() in config.MIDI_EXTENSIONS
    ]
    if top_level:
        genres[config.UNLABELLED_GENRE] = len(top_level)
    for folder in sorted(p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        count = sum(
            1 for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in config.MIDI_EXTENSIONS
        )
        genres[folder.name] = count
    return genres


def _candidate_files(raw_dir: Path, genre: str) -> list[Path]:
    if genre == config.ALL_GENRES:
        return sorted(p for p in raw_dir.rglob("*") if p.is_file())
    if genre == config.UNLABELLED_GENRE:
        return sorted(p for p in raw_dir.iterdir() if p.is_file())
    folder = raw_dir / genre
    if not folder.is_dir():
        raise DatasetError(
            f"There is no folder named '{genre}' inside {raw_dir}. "
            f"Create it and copy your .mid files into it, or choose another genre."
        )
    return sorted(p for p in folder.rglob("*") if p.is_file())


def scan_dataset(raw_dir: Path, genre: str = config.ALL_GENRES) -> DatasetScan:
    """Find MIDI files for ``genre``; separate duplicates and unsupported files."""
    if not raw_dir.is_dir():
        raise DatasetError(
            f"The dataset folder {raw_dir} does not exist. "
            "Create it and add .mid files (see README, 'Dataset setup')."
        )
    scan = DatasetScan(genre=genre)
    seen: dict[str, Path] = {}
    for path in _candidate_files(raw_dir, genre):
        suffix = path.suffix.lower()
        if path.name.startswith("."):
            continue
        if suffix in config.IGNORED_EXTENSIONS:
            continue
        if suffix not in config.MIDI_EXTENSIONS:
            scan.unsupported.append(path)
            continue
        try:
            digest = file_sha1(path)
        except OSError as exc:
            log.warning("Cannot read %s: %s", path, exc)
            scan.unsupported.append(path)
            continue
        if digest in seen:
            scan.duplicates.append((path, seen[digest]))
            continue
        seen[digest] = path
        scan.files.append(path)
        scan.hashes[path] = digest
    return scan


def save_uploaded_files(
    raw_dir: Path,
    genre: str,
    uploads: list[tuple[str, bytes]],
    max_mb: int = config.MAX_UPLOAD_MB,
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Store uploaded MIDI files under ``raw_dir/<genre>/``.

    Returns ``(saved_paths, rejected)`` where ``rejected`` is ``(name, reason)``.
    Existing files are never overwritten. Content is checked for the MIDI
    header (``MThd``) so a renamed .mp3 is rejected.
    """
    genre_slug = slugify(genre, default="")
    if not genre_slug:
        raise InvalidInputError("Please enter a genre/folder name, for example 'classical'.")
    target = ensure_writable_dir(raw_dir / genre_slug)
    saved: list[Path] = []
    rejected: list[tuple[str, str]] = []
    for original_name, data in uploads:
        name = Path(original_name).name  # drop any folder parts
        suffix = Path(name).suffix.lower()
        if suffix not in config.MIDI_EXTENSIONS:
            rejected.append((name, "not a .mid/.midi file"))
            continue
        if len(data) > max_mb * 1024 * 1024:
            rejected.append((name, f"larger than {max_mb} MB"))
            continue
        if not data.startswith(b"MThd"):
            rejected.append((name, "file content is not a MIDI file"))
            continue
        destination = unique_path(target, slugify(Path(name).stem, 80, "midi"), suffix)
        destination.write_bytes(data)
        saved.append(destination)
    return saved, rejected


def import_folder(
    source: Path, raw_dir: Path, genre: str | None
) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Copy MIDI files from a folder on disk into the dataset.

    ``genre=None`` puts them in the unlabelled top level of ``data/raw``.
    Returns ``(copied, skipped)``.
    """
    if not source.is_dir():
        raise DatasetError(f"The folder '{source}' does not exist.")
    if genre:
        slug = slugify(genre, default="")
        if not slug:
            raise InvalidInputError("The genre name must contain letters or digits.")
        target = ensure_writable_dir(raw_dir / slug)
    else:
        target = ensure_writable_dir(raw_dir)
    copied: list[Path] = []
    skipped: list[tuple[Path, str]] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in config.MIDI_EXTENSIONS:
            continue
        try:
            with open(path, "rb") as handle:
                if handle.read(4) != b"MThd":
                    skipped.append((path, "not a MIDI file"))
                    continue
            destination = unique_path(target, slugify(path.stem, 80, "midi"), path.suffix.lower())
            shutil.copy2(path, destination)
            copied.append(destination)
        except OSError as exc:
            skipped.append((path, str(exc)))
    return copied, skipped


# Documented, freely available sources. ECHO does NOT bundle any of them.
DATASET_SOURCES = [
    {
        "name": "MAESTRO v3.0.0 (MIDI only)",
        "genre": "classical (solo piano)",
        "size": "about 56 MB download",
        "license": "CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike)",
        "url": "https://magenta.tensorflow.org/datasets/maestro",
        "download": "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip",
        "attribution": (
            "Hawthorne et al., 'Enabling Factorized Piano Music Modeling and Generation "
            "with the MAESTRO Dataset', ICLR 2019. Mention version v3.0.0."
        ),
        "automated": True,
    },
    {
        "name": "Weimar Jazz Database (MIDI zip of 456 solo transcriptions)",
        "genre": "jazz (monophonic solos)",
        "size": "see download page",
        "license": "Open Database License (ODbL) v1.0 - read its terms before redistributing",
        "url": "https://jazzomat.hfm-weimar.de/download/download.html",
        "download": None,
        "attribution": "Credit the Jazzomat Research Project, HfM Weimar, as the source.",
        "automated": False,
    },
    {
        "name": "Your own MIDI files",
        "genre": "any",
        "size": "any",
        "license": "You are responsible for having the right to use them",
        "url": None,
        "download": None,
        "attribution": "n/a",
        "automated": False,
    },
]
