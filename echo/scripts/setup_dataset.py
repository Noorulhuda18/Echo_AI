"""Dataset setup helper for ECHO.

    python scripts/setup_dataset.py sources
    python scripts/setup_dataset.py status
    python scripts/setup_dataset.py import --from "C:\\Music\\MyMidi" --genre jazz --confirm-genre
    python scripts/setup_dataset.py maestro --max-files 20

ECHO ships with NO dataset. The 'maestro' command downloads the official MIDI-only
MAESTRO v3.0.0 archive (about 56 MB, CC BY-NC-SA 4.0) from Google's servers on YOUR
machine, extracts a spread of pieces into data/raw/classical/ and writes an
ATTRIBUTION.md next to them. It prints exactly what it did; nothing is claimed in advance.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.dataset import DATASET_SOURCES, import_folder, list_genres  # noqa: E402
from src.utils import EchoError, ensure_writable_dir, slugify  # noqa: E402

MAESTRO = next(s for s in DATASET_SOURCES if s["name"].startswith("MAESTRO"))
ATTRIBUTION = """# Attribution - MAESTRO v3.0.0 (MIDI only)

These MIDI files come from the MAESTRO dataset, made available by Google LLC under the
Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0) licence.
https://magenta.tensorflow.org/datasets/maestro

Cite: Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang,
Sander Dieleman, Erich Elsen, Jesse Engel, and Douglas Eck. "Enabling Factorized Piano Music
Modeling and Generation with the MAESTRO Dataset." International Conference on Learning
Representations, 2019. (Version used: v3.0.0)

Non-commercial use only; derived material must be shared under the same licence.
Do not commit these files to a public repository without keeping this notice.
"""


def cmd_sources(_: argparse.Namespace) -> int:
    for s in DATASET_SOURCES:
        print(f"- {s['name']}\n    genre:   {s['genre']}\n    size:    {s['size']}\n    licence: {s['license']}")
        if s["url"]:
            print(f"    page:    {s['url']}")
        print(f"    credit:  {s['attribution']}\n")
    print("Jazz: download the Weimar Jazz Database MIDI zip from its page, unzip it, then run:")
    print('  python scripts/setup_dataset.py import --from "<unzipped folder>" --genre jazz --confirm-genre')
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    genres = list_genres(config.DEFAULT_PATHS.raw)
    if not genres:
        print("data/raw is empty.")
        return 1
    for name, count in genres.items():
        print(f"  {name}: {count} MIDI file(s)")
    return 0


def cmd_import(args: argparse.Namespace) -> int:
    genre = None
    if args.genre:
        if not args.confirm_genre:
            print(
                "[error] ECHO does not guess genres. Add --confirm-genre to state that every file in "
                f"'{args.source}' really belongs to the genre '{args.genre}'."
            )
            return 2
        genre = args.genre
    try:
        copied, skipped = import_folder(Path(args.source), config.DEFAULT_PATHS.raw, genre)
    except EchoError as exc:
        print(f"[error] {exc}")
        return 1
    print(f"[done] copied {len(copied)} file(s) into data/raw/{slugify(genre) if genre else ''}")
    for path, reason in skipped:
        print(f"[skipped] {path.name}: {reason}")
    return 0 if copied else 1


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "ECHO-dataset-setup/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, open(destination, "wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        while block := response.read(1 << 20):
            out.write(block)
            done += len(block)
            if total:
                print(f"\r[download] {done / 1e6:.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def cmd_maestro(args: argparse.Namespace) -> int:
    if not args.yes:
        print(
            "This downloads about 56 MB from storage.googleapis.com and extracts up to "
            f"{args.max_files} MIDI files into data/raw/classical/.\n"
            "The data is licensed CC BY-NC-SA 4.0 (non-commercial). Re-run with --yes to accept."
        )
        return 2
    target = ensure_writable_dir(config.DEFAULT_PATHS.raw / "classical")
    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "maestro-midi.zip"
        try:
            _download(MAESTRO["download"], archive)
        except OSError as exc:
            print(f"[error] Download failed: {exc}\nDownload the zip manually from {MAESTRO['url']} "
                  "and use the 'import' command on the unzipped folder.")
            return 1
        try:
            with zipfile.ZipFile(archive) as zf:
                members = sorted(
                    n for n in zf.namelist()
                    if n.lower().endswith((".mid", ".midi")) and not n.endswith("/")
                )
                if not members:
                    print("[error] The archive contains no MIDI files.")
                    return 1
                count = min(args.max_files, len(members))
                step = len(members) / count
                chosen = [members[int(i * step)] for i in range(count)]  # evenly spread
                for name in chosen:
                    safe_name = "maestro_" + slugify(Path(name).stem, 80, "piece") + ".midi"
                    destination = target / safe_name  # basename only: no path traversal
                    with zf.open(name) as src, open(destination, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        except zipfile.BadZipFile:
            print("[error] The downloaded file is not a valid zip archive.")
            return 1
    (target / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")
    print(f"[done] extracted {len(chosen)} of {len(members)} MIDI files into {target}")
    print("[done] wrote ATTRIBUTION.md (keep it with the files).")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("sources", help="list free MIDI sources and their licences").set_defaults(func=cmd_sources)
    sub.add_parser("status", help="count MIDI files per genre in data/raw").set_defaults(func=cmd_status)
    p_import = sub.add_parser("import", help="copy MIDI files from a folder into data/raw")
    p_import.add_argument("--from", dest="source", required=True, help="folder containing .mid/.midi files")
    p_import.add_argument("--genre", help="genre folder to create, e.g. classical")
    p_import.add_argument("--confirm-genre", action="store_true", help="confirm the genre label is correct for ALL files")
    p_import.set_defaults(func=cmd_import)
    p_maestro = sub.add_parser("maestro", help="download part of MAESTRO v3.0.0 (classical piano)")
    p_maestro.add_argument("--max-files", type=int, default=20, help="how many pieces to extract (default 20)")
    p_maestro.add_argument("--yes", action="store_true", help="accept the download and the licence terms")
    p_maestro.set_defaults(func=cmd_maestro)
    args = parser.parse_args(argv)
    config.DEFAULT_PATHS.ensure()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
