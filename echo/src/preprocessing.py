"""MIDI -> sequences of musical events -> model-ready windows.

Representation (beginner-friendly):
    event = (pitch, duration_index)
    * pitch is a MIDI note number (21-108) or REST_PITCH (-1) for silence
    * duration_index points into config.DURATION_CLASSES (lengths in beats)

Pipeline:
    MIDI file --load_midi_events--> list of events per song
              --Vocabulary-------> integer ids
              --split_songs------> train / validation segments (no shared events)
              --WindowDataset----> (X, y) batches: 'last N events' -> 'next event'

Only ``load_midi_events`` needs the pretty_midi package. Everything else uses
NumPy only, so the logic can be unit-tested without MIDI libraries.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

import config
from src.dataset import DatasetScan, scan_dataset
from src.utils import (
    DatasetError,
    InsufficientDataError,
    MidiLoadError,
    get_logger,
    require_module,
    slugify,
    utc_iso,
)

log = get_logger("preprocessing")

Event = tuple[int, int]  # (pitch or REST_PITCH, duration_index)
ProgressFn = Callable[[dict], None]


# --------------------------------------------------------------------------- #
# Quantisation
# --------------------------------------------------------------------------- #


def quantize_duration(beats: float) -> int:
    """Index of the duration class closest to ``beats`` (ties go to the shorter)."""
    best_index = 0
    best_key = (float("inf"), float("inf"))
    for index, length in enumerate(config.DURATION_CLASSES):
        key = (abs(length - beats), length)
        if key < best_key:
            best_key, best_index = key, index
    return best_index


def rest_events(gap_beats: float) -> list[Event]:
    """Break a silence into rest events using the available duration classes."""
    remaining = min(gap_beats, config.MAX_REST_BEATS)
    events: list[Event] = []
    smallest = config.DURATION_CLASSES[0]
    while remaining >= smallest / 2 and len(events) < 8:
        chosen = 0
        for index, length in enumerate(config.DURATION_CLASSES):
            if length <= remaining + 1e-9:
                chosen = index
        events.append((config.REST_PITCH, chosen))
        remaining -= config.DURATION_CLASSES[chosen]
    return events


def notes_to_events(notes: Iterable[tuple[float, float, int]]) -> list[Event]:
    """Turn ``(start_beat, end_beat, pitch)`` notes into one melodic line.

    A "skyline" is used: when several notes start together (a chord, or
    left hand + right hand) only the highest pitch is kept. Onsets are snapped
    to a 16th-note grid, note lengths are quantised, and gaps become rests.
    """
    grid = config.GRID_BEATS
    best: dict[int, tuple[int, float]] = {}
    for start, end, pitch in notes:
        slot = int(round(start / grid))
        current = best.get(slot)
        if current is None or pitch > current[0]:
            best[slot] = (int(pitch), float(end))
    slots = sorted(best)
    events: list[Event] = []
    for i, slot in enumerate(slots):
        pitch, end = best[slot]
        onset = slot * grid
        length = max(end - onset, grid)
        next_onset = slots[i + 1] * grid if i + 1 < len(slots) else None
        if next_onset is not None:
            length = min(length, next_onset - onset)
        duration_index = quantize_duration(length)
        events.append((pitch, duration_index))
        if next_onset is not None:
            gap = (next_onset - onset) - config.DURATION_CLASSES[duration_index]
            events.extend(rest_events(gap))
    return events


# --------------------------------------------------------------------------- #
# MIDI reading (needs pretty_midi)
# --------------------------------------------------------------------------- #


def load_midi_events(path: Path, max_events: int = config.MAX_EVENTS_PER_SONG) -> list[Event]:
    """Read one MIDI file and return its melodic events.

    Raises ``MidiLoadError`` with a short reason for corrupt files, files with
    no pitched notes, or files with too few notes to learn from.
    """
    pretty_midi = require_module("pretty_midi")
    path = Path(path)
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:  # pretty_midi/mido raise many different error types
        raise MidiLoadError(f"{path.name}: corrupt or unreadable MIDI ({type(exc).__name__})") from exc

    try:
        resolution = float(midi.resolution)
        notes: list[tuple[float, float, int]] = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                if not config.MIN_PITCH <= note.pitch <= config.MAX_PITCH:
                    continue
                start = midi.time_to_tick(note.start) / resolution
                end = midi.time_to_tick(note.end) / resolution
                notes.append((start, end, int(note.pitch)))
    except Exception as exc:
        raise MidiLoadError(f"{path.name}: could not read note timing ({type(exc).__name__})") from exc

    if not notes:
        raise MidiLoadError(f"{path.name}: no usable pitched notes (drums-only or empty)")
    if len(notes) < config.MIN_NOTES_PER_FILE:
        raise MidiLoadError(
            f"{path.name}: only {len(notes)} usable notes (need at least {config.MIN_NOTES_PER_FILE})"
        )
    events = notes_to_events(notes)[:max_events]
    if sum(1 for pitch, _ in events if pitch != config.REST_PITCH) < config.MIN_NOTES_PER_FILE:
        raise MidiLoadError(f"{path.name}: too few distinct note onsets after quantisation")
    return events


# --------------------------------------------------------------------------- #
# Vocabulary (encode / decode)
# --------------------------------------------------------------------------- #


class Vocabulary:
    """Bidirectional mapping between events and integer ids."""

    def __init__(self, events: Sequence[Event]):
        self._events: list[Event] = [(int(p), int(d)) for p, d in events]
        self._index: dict[Event, int] = {e: i for i, e in enumerate(self._events)}
        if len(self._index) != len(self._events):
            raise ValueError("Vocabulary events must be unique")

    @classmethod
    def build(cls, songs: Iterable[Sequence[Event]], min_count: int = 1) -> "Vocabulary":
        counts: Counter[Event] = Counter()
        for song in songs:
            counts.update((int(p), int(d)) for p, d in song)
        kept = sorted(e for e, c in counts.items() if c >= min_count)
        return cls(kept)

    def __len__(self) -> int:
        return len(self._events)

    def __contains__(self, event: Event) -> bool:
        return event in self._index

    def encode(self, event: Event) -> int:
        try:
            return self._index[(int(event[0]), int(event[1]))]
        except KeyError as exc:
            raise KeyError(f"Event {event} is not in the vocabulary") from exc

    def decode(self, index: int) -> Event:
        if not 0 <= index < len(self._events):
            raise IndexError(f"Event id {index} is outside the vocabulary (size {len(self)})")
        return self._events[index]

    def encode_song(self, song: Sequence[Event]) -> np.ndarray:
        """Encode a song, silently skipping events that are not in the vocabulary."""
        ids = [self._index[e] for e in ((int(p), int(d)) for p, d in song) if e in self._index]
        return np.asarray(ids, dtype=np.int32)

    def to_dict(self) -> dict:
        return {"events": [[p, d] for p, d in self._events]}

    @classmethod
    def from_dict(cls, data: dict) -> "Vocabulary":
        return cls([(int(p), int(d)) for p, d in data["events"]])

    def fingerprint(self) -> str:
        """Short hash that changes whenever the encoding changes."""
        payload = json.dumps(
            {"events": self._events, "durations": config.DURATION_CLASSES}, sort_keys=True
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Splitting and windowing
# --------------------------------------------------------------------------- #


def split_songs(
    songs: Sequence[np.ndarray], val_fraction: float, seed: int
) -> tuple[list[np.ndarray], list[np.ndarray], str]:
    """Split encoded songs into training and validation segments.

    * 5 or more songs: whole songs go to either side ("by_file"). Validation
      then measures generalisation to unseen pieces.
    * fewer songs: each song is cut once, the first part trains and the last
      part validates ("by_time"). The two parts are separate arrays, so no
      event (and therefore no window) appears on both sides.
    """
    songs = [np.asarray(s, dtype=np.int32) for s in songs if len(s) > 1]
    if len(songs) >= 5:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(songs))
        n_val = max(1, int(round(len(songs) * val_fraction)))
        val_ids = set(order[:n_val].tolist())
        train = [s for i, s in enumerate(songs) if i not in val_ids]
        val = [s for i, s in enumerate(songs) if i in val_ids]
        return train, val, "by_file"
    train, val = [], []
    for song in songs:
        cut = int(len(song) * (1.0 - val_fraction))
        train.append(song[:cut])
        val.append(song[cut:])
    return train, val, "by_time"


class WindowDataset:
    """Serves ``(X, y)`` batches without storing every window in memory.

    Only an index of ``(song, start)`` pairs is kept; the actual windows are cut
    out of the song arrays when a batch is requested. Each window is
    ``seq_len`` consecutive event ids and the target is the id that follows.
    """

    def __init__(
        self,
        segments: Sequence[np.ndarray],
        seq_len: int,
        batch_size: int,
        stride: int = 1,
        shuffle: bool = False,
        seed: int = 0,
    ):
        if seq_len < 1 or batch_size < 1 or stride < 1:
            raise ValueError("seq_len, batch_size and stride must be positive")
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.segments = [np.asarray(s, dtype=np.int32) for s in segments if len(s) > seq_len]
        parts = []
        for song_id, segment in enumerate(self.segments):
            starts = np.arange(0, len(segment) - seq_len, stride, dtype=np.int64)
            parts.append(np.stack([np.full_like(starts, song_id), starts], axis=1))
        self.index = np.concatenate(parts) if parts else np.zeros((0, 2), dtype=np.int64)
        self._rng = np.random.default_rng(seed)
        if shuffle:
            self._rng.shuffle(self.index)

    @property
    def num_windows(self) -> int:
        return int(len(self.index))

    @property
    def num_batches(self) -> int:
        return int(np.ceil(self.num_windows / self.batch_size)) if self.num_windows else 0

    def batch(self, batch_number: int) -> tuple[np.ndarray, np.ndarray]:
        if not 0 <= batch_number < self.num_batches:
            raise IndexError(f"batch {batch_number} out of range (0..{self.num_batches - 1})")
        rows = self.index[batch_number * self.batch_size : (batch_number + 1) * self.batch_size]
        x = np.empty((len(rows), self.seq_len), dtype=np.int32)
        y = np.empty((len(rows),), dtype=np.int32)
        for i, (song_id, start) in enumerate(rows):
            segment = self.segments[int(song_id)]
            x[i] = segment[start : start + self.seq_len]
            y[i] = segment[start + self.seq_len]
        return x, y

    def on_epoch_end(self) -> None:
        if self.shuffle:
            self._rng.shuffle(self.index)

    def all_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Every window at once (use only for small data, e.g. tests)."""
        if not self.num_batches:
            return np.zeros((0, self.seq_len), np.int32), np.zeros((0,), np.int32)
        xs, ys = zip(*(self.batch(i) for i in range(self.num_batches)))
        return np.concatenate(xs), np.concatenate(ys)


def validate_windows(x: np.ndarray, y: np.ndarray, vocab_size: int) -> None:
    """Sanity-check model inputs; raises ValueError describing the first problem."""
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
        raise ValueError(f"Bad shapes: X{x.shape}, y{y.shape}")
    if len(x) == 0:
        raise ValueError("No windows to validate")
    if x.min() < 0 or x.max() >= vocab_size or y.min() < 0 or y.max() >= vocab_size:
        raise ValueError("Event ids fall outside the vocabulary range")


def pick_seed_windows(songs: Sequence[np.ndarray], seq_len: int, limit: int = 32) -> list[list[int]]:
    """Choose real opening phrases to start generation from.

    Generation needs ``seq_len`` events of context. Using genuine passages from
    the training music (song openings first) gives the model a valid start.
    """
    windows: list[list[int]] = []
    usable = [s for s in songs if len(s) > seq_len]
    for song in usable:
        windows.append(song[:seq_len].tolist())
        if len(windows) >= limit:
            return windows
    if usable:
        position = 0
        while len(windows) < limit:
            song = usable[position % len(usable)]
            span = len(song) - seq_len
            offset = (position * 7919) % span if span > 0 else 0  # deterministic spread
            windows.append(song[offset : offset + seq_len].tolist())
            position += 1
            if position > limit * 4:
                break
    return windows


# --------------------------------------------------------------------------- #
# Reports and the full pre-processing pipeline
# --------------------------------------------------------------------------- #


@dataclass
class DatasetReport:
    genre: str
    discovered: int = 0
    processed: int = 0
    duplicates: int = 0
    skipped: list[dict] = field(default_factory=list)  # {"file": ..., "reason": ...}
    unsupported: list[str] = field(default_factory=list)
    total_events: int = 0
    note_events: int = 0
    rest_events: int = 0
    pitch_counts: dict[str, int] = field(default_factory=dict)
    duration_counts: dict[str, int] = field(default_factory=dict)
    events_per_file: dict[str, int] = field(default_factory=dict)
    vocab_size: int = 0
    fingerprint: str = ""
    created: str = ""

    @property
    def skipped_count(self) -> int:
        return len(self.skipped)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "DatasetReport":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class ProcessedDataset:
    vocab: Vocabulary
    songs: list[np.ndarray]
    song_names: list[str]
    report: DatasetReport


def _fingerprint(scan: DatasetScan, cfg: config.ModelConfig) -> str:
    payload = json.dumps(
        {
            "files": sorted(scan.hashes.values()),
            "min_event_count": cfg.min_event_count,
            "durations": config.DURATION_CLASSES,
            "grid": config.GRID_BEATS,
            "max_events": config.MAX_EVENTS_PER_SONG,
            "pitch": [config.MIN_PITCH, config.MAX_PITCH],
            "version": 1,
        },
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def preprocess_dataset(
    raw_dir: Path,
    genre: str,
    cfg: config.ModelConfig,
    loader: Callable[[Path], list[Event]] = load_midi_events,
    progress: ProgressFn | None = None,
) -> ProcessedDataset:
    """Scan, parse and encode every MIDI file for ``genre``.

    Corrupt files are skipped and reported, never fatal. Raises
    ``DatasetError`` when nothing at all could be processed.
    """
    scan = scan_dataset(raw_dir, genre)
    report = DatasetReport(
        genre=genre,
        discovered=scan.discovered,
        duplicates=len(scan.duplicates),
        unsupported=[p.name for p in scan.unsupported],
        created=utc_iso(),
    )
    if scan.discovered == 0:
        raise DatasetError(
            f"No MIDI files were found for '{genre}' in {raw_dir}. Add .mid or .midi files "
            "(see README, 'Dataset setup') and try again."
        )

    raw_songs: list[list[Event]] = []
    names: list[str] = []
    total = len(scan.files)
    for number, path in enumerate(scan.files, start=1):
        try:
            events = loader(path)
        except MidiLoadError as exc:
            report.skipped.append({"file": path.name, "reason": str(exc)})
            log.warning("Skipped %s: %s", path, exc)
        except Exception as exc:  # defensive: never let one file stop the run
            report.skipped.append({"file": path.name, "reason": f"unexpected error: {exc}"})
            log.exception("Unexpected error while reading %s", path)
        else:
            raw_songs.append(events)
            names.append(path.name)
        if progress:
            progress({"type": "preprocess", "done": number, "total": total, "file": path.name})

    if not raw_songs:
        raise DatasetError(
            f"All {scan.discovered} MIDI files were skipped (corrupt, empty or too short). "
            "See the skipped-files list for each reason."
        )

    vocab = Vocabulary.build(raw_songs, min_count=cfg.min_event_count)
    if len(vocab) < 2:
        raise InsufficientDataError(
            "The music contains fewer than 2 distinct events, so there is nothing to learn. "
            "Add more varied MIDI files."
        )
    encoded = [vocab.encode_song(song) for song in raw_songs]

    pitch_counter: Counter[int] = Counter()
    duration_counter: Counter[int] = Counter()
    for song in raw_songs:
        for pitch, duration_index in song:
            duration_counter[duration_index] += 1
            if pitch != config.REST_PITCH:
                pitch_counter[pitch] += 1
    report.processed = len(raw_songs)
    report.total_events = int(sum(len(s) for s in raw_songs))
    report.note_events = int(sum(pitch_counter.values()))
    report.rest_events = report.total_events - report.note_events
    report.pitch_counts = {str(k): int(v) for k, v in sorted(pitch_counter.items())}
    report.duration_counts = {
        str(config.DURATION_CLASSES[k]): int(v) for k, v in sorted(duration_counter.items())
    }
    report.events_per_file = {n: len(s) for n, s in zip(names, raw_songs)}
    report.vocab_size = len(vocab)
    report.fingerprint = _fingerprint(scan, cfg)
    return ProcessedDataset(vocab=vocab, songs=encoded, song_names=names, report=report)


# --------------------------------------------------------------------------- #
# Saving / reusing processed data
# --------------------------------------------------------------------------- #


def processed_dir(paths: config.Paths, genre: str) -> Path:
    return paths.processed / slugify(genre)


def save_processed(data: ProcessedDataset, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    arrays = {f"song_{i}": s for i, s in enumerate(data.songs)}
    np.savez_compressed(folder / "events.npz", **arrays)
    (folder / "vocab.json").write_text(json.dumps(data.vocab.to_dict()), encoding="utf-8")
    meta = {"song_names": data.song_names, "report": data.report.to_dict()}
    (folder / "report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def load_processed(folder: Path) -> ProcessedDataset:
    try:
        meta = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        vocab = Vocabulary.from_dict(json.loads((folder / "vocab.json").read_text(encoding="utf-8")))
        with np.load(folder / "events.npz") as archive:
            songs = [archive[f"song_{i}"] for i in range(len(meta["song_names"]))]
    except (OSError, KeyError, ValueError) as exc:
        raise DatasetError(f"Saved processed data in {folder} is unreadable: {exc}") from exc
    return ProcessedDataset(
        vocab=vocab,
        songs=songs,
        song_names=list(meta["song_names"]),
        report=DatasetReport.from_dict(meta["report"]),
    )


def load_report(folder: Path) -> DatasetReport | None:
    """Return the saved report, or None if the genre was never processed."""
    try:
        meta = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        return DatasetReport.from_dict(meta["report"])
    except (OSError, KeyError, ValueError):
        return None


def get_or_build_processed(
    paths: config.Paths,
    genre: str,
    cfg: config.ModelConfig,
    loader: Callable[[Path], list[Event]] = load_midi_events,
    progress: ProgressFn | None = None,
    force: bool = False,
) -> tuple[ProcessedDataset, bool]:
    """Reuse saved processed data when the files and settings are unchanged.

    Returns ``(dataset, reused)``.
    """
    folder = processed_dir(paths, genre)
    if not force:
        scan = scan_dataset(paths.raw, genre)
        saved = load_report(folder)
        if saved and scan.discovered and saved.fingerprint == _fingerprint(scan, cfg):
            try:
                return load_processed(folder), True
            except DatasetError:
                log.warning("Saved processed data was unreadable; rebuilding")
    data = preprocess_dataset(paths.raw, genre, cfg, loader=loader, progress=progress)
    save_processed(data, folder)
    return data, False
