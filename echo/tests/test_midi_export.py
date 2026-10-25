"""MIDI export: timing, tempo, validation, naming and failure handling."""

from __future__ import annotations

import pytest

pretty_midi = pytest.importorskip("pretty_midi")

import config  # noqa: E402
from src import midi_export as me  # noqa: E402
from src.utils import InvalidInputError, MidiExportError  # noqa: E402

R = config.REST_PITCH
D = config.DURATION_CLASSES
EVENTS = [(60, D.index(1.0)), (R, D.index(0.5)), (64, D.index(0.5)), (67, D.index(2.0))]


def test_notes_have_correct_pitch_start_and_end():
    midi = me.events_to_pretty_midi(EVENTS, tempo_bpm=120)  # 0.5 s per beat
    notes = midi.instruments[0].notes
    assert [n.pitch for n in notes] == [60, 64, 67]
    assert notes[0].start == pytest.approx(0.0)
    assert notes[0].end == pytest.approx(0.5 * config.NOTE_ARTICULATION)
    assert notes[1].start == pytest.approx(0.5 + 0.25)  # after note (0.5 s) and rest (0.25 s)
    assert notes[2].start == pytest.approx(0.5 + 0.25 + 0.25)
    assert notes[2].end - notes[2].start == pytest.approx(1.0 * config.NOTE_ARTICULATION)


def test_tempo_scales_timing():
    slow = me.events_to_pretty_midi(EVENTS, 60).get_end_time()
    fast = me.events_to_pretty_midi(EVENTS, 120).get_end_time()
    assert slow == pytest.approx(2 * fast)


def test_write_midi_creates_valid_reopenable_file(tmp_path):
    path = me.write_midi(EVENTS, 100, tmp_path / "out", model_name="Jazz Model!", seed=9, program=11)
    assert path.suffix == ".mid" and path.parent == tmp_path / "out"
    assert path.read_bytes().startswith(b"MThd")
    assert "jazz-model" in path.name and "100bpm" in path.name and path.name.endswith("_s9.mid")
    reopened = pretty_midi.PrettyMIDI(str(path))
    assert len(reopened.instruments[0].notes) == 3
    assert reopened.instruments[0].program == 11
    assert reopened.estimate_tempo() > 0
    tempo_changes = reopened.get_tempo_changes()[1]
    assert tempo_changes[0] == pytest.approx(100, abs=0.5)


def test_note_on_off_pairs_are_balanced(tmp_path):
    import mido

    path = me.write_midi(EVENTS * 10, 90, tmp_path)
    mid = mido.MidiFile(str(path))
    on = off = 0
    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                on += 1
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                off += 1
    assert on == off == 30


def test_files_are_never_overwritten(tmp_path):
    first = me.write_midi(EVENTS, 100, tmp_path, seed=1)
    second = me.write_midi(EVENTS, 100, tmp_path, seed=1)
    assert first != second and first.exists() and second.exists()


@pytest.mark.parametrize(
    "events, tempo, program, error",
    [
        ([], 100, 0, MidiExportError),
        ([(R, 1), (R, 2)], 100, 0, MidiExportError),
        ([(60, 99)], 100, 0, MidiExportError),
        ([(300, 1)], 100, 0, MidiExportError),
        (EVENTS, 5, 0, InvalidInputError),
        (EVENTS, 100, 200, InvalidInputError),
    ],
)
def test_invalid_input_is_rejected(tmp_path, events, tempo, program, error):
    with pytest.raises(error):
        me.write_midi(events, tempo, tmp_path, program=program)
    assert list(tmp_path.glob("*.mid")) == []


def test_unwritable_output_folder_gives_clear_error(tmp_path):
    blocker = tmp_path / "file.txt"
    blocker.write_text("i am a file, not a folder")
    with pytest.raises(Exception, match="Cannot write to the folder"):
        me.write_midi(EVENTS, 100, blocker / "sub")
