"""Convert generated events into a valid, verified MIDI file."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import config
from src.preprocessing import Event
from src.utils import (
    MidiExportError,
    InvalidInputError,
    ensure_writable_dir,
    get_logger,
    require_module,
    slugify,
    unique_path,
    utc_stamp,
)

log = get_logger("midi_export")

DEFAULT_VELOCITY = 84


def events_to_pretty_midi(
    events: Sequence[Event],
    tempo_bpm: float,
    program: int = 0,
    velocity: int = DEFAULT_VELOCITY,
):
    """Build a ``pretty_midi.PrettyMIDI`` object from events.

    Durations are in beats, so seconds = beats * 60 / tempo. Each note sounds for
    95% of its length so that repeated pitches get a clean note-off before the
    next note-on. Rests simply advance time.
    """
    pretty_midi = require_module("pretty_midi")
    if not config.MIN_BPM <= tempo_bpm <= config.MAX_BPM:
        raise InvalidInputError(f"Tempo must be between {config.MIN_BPM} and {config.MAX_BPM} BPM.")
    if not 0 <= program <= 127:
        raise InvalidInputError("MIDI program must be between 0 and 127.")
    if not events:
        raise MidiExportError("There are no events to export.")

    seconds_per_beat = 60.0 / float(tempo_bpm)
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    instrument = pretty_midi.Instrument(program=int(program), name="ECHO")
    cursor = 0.0
    for pitch, duration_index in events:
        if not 0 <= duration_index < len(config.DURATION_CLASSES):
            raise MidiExportError(f"Invalid duration index {duration_index}.")
        length = config.DURATION_CLASSES[duration_index] * seconds_per_beat
        if pitch != config.REST_PITCH:
            if not 0 <= pitch <= 127:
                raise MidiExportError(f"Invalid MIDI pitch {pitch}.")
            instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(velocity),
                    pitch=int(pitch),
                    start=cursor,
                    end=cursor + length * config.NOTE_ARTICULATION,
                )
            )
        cursor += length
    if not instrument.notes:
        raise MidiExportError("The sequence contains no notes (only rests).")
    midi.instruments.append(instrument)
    return midi


def write_midi(
    events: Sequence[Event],
    tempo_bpm: float,
    out_dir: Path,
    model_name: str = "model",
    seed: int | None = None,
    program: int = 0,
) -> Path:
    """Write events to ``out_dir`` as a new .mid file and verify it by re-reading it."""
    pretty_midi = require_module("pretty_midi")
    midi = events_to_pretty_midi(events, tempo_bpm, program)
    folder = ensure_writable_dir(Path(out_dir))
    stem = f"echo_{slugify(model_name)}_{int(round(tempo_bpm))}bpm_{utc_stamp()}"
    if seed is not None:
        stem += f"_s{seed}"
    path = unique_path(folder, stem, ".mid")
    try:
        midi.write(str(path))
    except Exception as exc:
        raise MidiExportError(f"Could not write the MIDI file: {exc}") from exc

    expected_notes = sum(1 for pitch, _ in events if pitch != config.REST_PITCH)
    try:
        reopened = pretty_midi.PrettyMIDI(str(path))
        actual_notes = sum(len(i.notes) for i in reopened.instruments)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise MidiExportError(f"The written MIDI file could not be re-opened: {exc}") from exc
    if actual_notes != expected_notes or reopened.get_end_time() <= 0:
        path.unlink(missing_ok=True)
        raise MidiExportError(
            f"MIDI verification failed: expected {expected_notes} notes, found {actual_notes}."
        )
    log.info("Wrote %s (%d notes)", path, actual_notes)
    return path
