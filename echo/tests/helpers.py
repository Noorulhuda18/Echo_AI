"""Helpers that build tiny synthetic MIDI files for tests and the smoke test.

These files exist only to exercise the pipeline (they are NOT music data and say
nothing about how good the generated music would be). Each one repeats a short
rhythm pattern over a random-walk melody in C major so a small LSTM has a
pattern to learn.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCALE = [0, 2, 4, 5, 7, 9, 11, 12]
RHYTHM = [1.0, 0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5]  # beats, repeated


def make_midi_file(path: Path, seed: int = 0, n_notes: int = 320, tempo: float = 120.0,
                   root: int = 60, program: int = 0) -> Path:
    import pretty_midi

    rng = np.random.default_rng(seed)
    midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    instrument = pretty_midi.Instrument(program=program)
    seconds_per_beat = 60.0 / tempo
    position = 3
    cursor = 0.0
    for i in range(n_notes):
        position = int(np.clip(position + rng.choice([-2, -1, -1, 0, 1, 1, 2]), 0, len(SCALE) - 1))
        length = RHYTHM[i % len(RHYTHM)] * seconds_per_beat
        instrument.notes.append(
            pretty_midi.Note(velocity=80, pitch=root + SCALE[position], start=cursor, end=cursor + length * 0.9)
        )
        cursor += length
        if i % 16 == 15:  # an occasional rest
            cursor += 0.5 * seconds_per_beat
    midi.instruments.append(instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def make_drum_only_midi(path: Path) -> Path:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    drums = pretty_midi.Instrument(program=0, is_drum=True)
    for i in range(40):
        drums.notes.append(pretty_midi.Note(velocity=90, pitch=36, start=i * 0.25, end=i * 0.25 + 0.1))
    midi.instruments.append(drums)
    midi.write(str(path))
    return path


def make_short_midi(path: Path, n_notes: int = 4) -> Path:
    import pretty_midi

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for i in range(n_notes):
        instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60 + i, start=i * 0.5, end=i * 0.5 + 0.4))
    midi.instruments.append(instrument)
    midi.write(str(path))
    return path
