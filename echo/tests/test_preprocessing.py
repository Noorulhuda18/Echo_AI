"""Preprocessing, encoding, windowing, dataset handling and MIDI loading."""

from __future__ import annotations

import numpy as np
import pytest

import config
from src import dataset, preprocessing as pp
from src.utils import DatasetError, InvalidInputError, MidiLoadError

R = config.REST_PITCH
D = config.DURATION_CLASSES
FAKE_MIDI = b"MThd\x00\x00\x00\x06\x00\x01\x00\x01\x01\xe0"  # header only: enough for scanning


# ----------------------------------------------------------------------------
# Quantisation
# ----------------------------------------------------------------------------
def test_quantize_duration_picks_nearest_class():
    assert D[pp.quantize_duration(0.26)] == 0.25
    assert D[pp.quantize_duration(1.0)] == 1.0
    assert D[pp.quantize_duration(1.4)] == 1.5
    assert D[pp.quantize_duration(100)] == 4.0
    assert D[pp.quantize_duration(0.0)] == 0.25


def test_quantize_duration_tie_goes_to_shorter():
    assert D[pp.quantize_duration(1.25)] == 1.0  # exactly between 1.0 and 1.5


def test_rest_events_cover_the_gap():
    events = pp.rest_events(1.0)
    assert events == [(R, D.index(1.0))]
    total = sum(D[d] for _, d in pp.rest_events(5.5))
    assert abs(total - 5.5) <= 0.25
    assert pp.rest_events(0.0) == []
    assert sum(D[d] for _, d in pp.rest_events(500)) <= config.MAX_REST_BEATS + 1e-9


def test_notes_to_events_skyline_and_rest():
    notes = [(0.0, 1.0, 60), (0.0, 1.0, 64), (2.0, 3.0, 67)]
    assert pp.notes_to_events(notes) == [(64, D.index(1.0)), (R, D.index(1.0)), (67, D.index(1.0))]


def test_notes_to_events_cuts_note_at_next_onset():
    # a note held for 4 beats, but the next note starts after 1 beat -> length 1
    events = pp.notes_to_events([(0.0, 4.0, 60), (1.0, 2.0, 62)])
    assert events[0] == (60, D.index(1.0))
    assert all(r[0] != R for r in events[:1])


# ----------------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------------
def test_vocabulary_round_trip():
    songs = [[(60, 0), (62, 1), (R, 2), (60, 0)], [(64, 3)]]
    vocab = pp.Vocabulary.build(songs)
    assert len(vocab) == 4
    for song in songs:
        for event in song:
            assert vocab.decode(vocab.encode(event)) == event
    restored = pp.Vocabulary.from_dict(vocab.to_dict())
    assert restored.fingerprint() == vocab.fingerprint()
    assert [restored.decode(i) for i in range(len(restored))] == [vocab.decode(i) for i in range(len(vocab))]


def test_vocabulary_rejects_unknown_and_bad_ids():
    vocab = pp.Vocabulary.build([[(60, 0), (62, 1)]])
    with pytest.raises(KeyError):
        vocab.encode((99, 0))
    with pytest.raises(IndexError):
        vocab.decode(5)
    with pytest.raises(IndexError):
        vocab.decode(-1)


def test_vocabulary_min_count_and_encode_song_skips_rare():
    songs = [[(60, 0), (60, 0), (61, 0)]]
    vocab = pp.Vocabulary.build(songs, min_count=2)
    assert len(vocab) == 1
    assert pp.Vocabulary.encode_song(vocab, songs[0]).tolist() == [0, 0]


def test_fingerprint_changes_with_vocabulary():
    a = pp.Vocabulary.build([[(60, 0), (62, 0)]])
    b = pp.Vocabulary.build([[(60, 0), (63, 0)]])
    assert a.fingerprint() != b.fingerprint()


# ----------------------------------------------------------------------------
# Splitting and windows
# ----------------------------------------------------------------------------
def test_split_by_file_keeps_songs_whole_and_disjoint():
    songs = [np.full(50, i, dtype=np.int32) for i in range(10)]
    train, val, strategy = pp.split_songs(songs, 0.2, seed=1)
    assert strategy == "by_file"
    assert len(train) + len(val) == 10 and len(val) == 2
    assert not ({int(s[0]) for s in train} & {int(s[0]) for s in val})


def test_split_by_time_has_no_shared_events():
    songs = [np.arange(100, dtype=np.int32), np.arange(100, 200, dtype=np.int32)]
    train, val, strategy = pp.split_songs(songs, 0.2, seed=1)
    assert strategy == "by_time"
    for t, v in zip(train, val):
        assert len(t) + len(v) == 100
        assert not (set(t.tolist()) & set(v.tolist()))
        assert t.max() < v.min()


def test_window_dataset_inputs_and_targets():
    segment = np.arange(10, dtype=np.int32)
    ds = pp.WindowDataset([segment], seq_len=3, batch_size=4, stride=1)
    assert ds.num_windows == 7 and ds.num_batches == 2
    x, y = ds.batch(0)
    assert x[0].tolist() == [0, 1, 2] and y[0] == 3
    x_all, y_all = ds.all_arrays()
    assert x_all.shape == (7, 3) and y_all[-1] == 9
    assert (x_all[:, -1] + 1 == y_all).all()  # target always follows its window


def test_window_dataset_stride_and_short_segments():
    ds = pp.WindowDataset([np.arange(20), np.arange(3)], seq_len=4, batch_size=8, stride=4)
    assert ds.num_windows == 4  # the 3-event song is skipped
    empty = pp.WindowDataset([np.arange(3)], seq_len=4, batch_size=8)
    assert empty.num_windows == 0 and empty.num_batches == 0
    with pytest.raises(IndexError):
        empty.batch(0)


def test_window_dataset_shuffle_is_reproducible_and_complete():
    seg = np.arange(60, dtype=np.int32)
    a = pp.WindowDataset([seg], 5, 8, shuffle=True, seed=3).all_arrays()[1]
    b = pp.WindowDataset([seg], 5, 8, shuffle=True, seed=3).all_arrays()[1]
    assert a.tolist() == b.tolist()
    assert sorted(a.tolist()) == list(range(5, 60))


def test_validate_windows_rejects_bad_ids():
    x = np.zeros((4, 3), dtype=np.int32)
    pp.validate_windows(x, np.zeros(4, dtype=np.int32), vocab_size=2)
    with pytest.raises(ValueError):
        pp.validate_windows(x, np.full(4, 5, dtype=np.int32), vocab_size=2)
    with pytest.raises(ValueError):
        pp.validate_windows(x[:0], np.zeros(0, dtype=np.int32), vocab_size=2)


def test_pick_seed_windows_have_right_length():
    songs = [np.arange(100, dtype=np.int32), np.arange(50, dtype=np.int32)]
    seeds = pp.pick_seed_windows(songs, seq_len=8, limit=6)
    assert len(seeds) == 6 and all(len(s) == 8 for s in seeds)
    assert seeds[0] == list(range(8))
    assert pp.pick_seed_windows([np.arange(3)], seq_len=8) == []


# ----------------------------------------------------------------------------
# Dataset handling (uses a stub loader, so no MIDI library is needed)
# ----------------------------------------------------------------------------
def _stub_loader(path):
    if "bad" in path.name:
        raise MidiLoadError(f"{path.name}: corrupt or unreadable MIDI (stub)")
    base = [(60 + (i % 7), i % 4) for i in range(60)]
    return base if "a" in path.name else list(reversed(base))


def _write(folder, name, data=FAKE_MIDI):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_bytes(data)
    return path


def test_empty_dataset_folder_raises_clear_error(paths):
    with pytest.raises(DatasetError, match="No MIDI files"):
        pp.preprocess_dataset(paths.raw, config.ALL_GENRES, config.ModelConfig(), loader=_stub_loader)


def test_missing_genre_folder_raises(paths):
    with pytest.raises(DatasetError, match="no folder named"):
        dataset.scan_dataset(paths.raw, "jazz")


def test_corrupt_duplicate_and_unsupported_files_are_reported(paths):
    folder = paths.raw / "classical"
    _write(folder, "a1.mid", FAKE_MIDI + b"1")
    _write(folder, "b2.mid", FAKE_MIDI + b"2")
    _write(folder, "bad.mid", FAKE_MIDI + b"3")
    _write(folder, "copy_of_a1.mid", FAKE_MIDI + b"1")  # byte-identical duplicate
    _write(folder, "notes.mp3", b"ID3")
    _write(folder, "README.txt", b"ignored")
    cfg = config.ModelConfig(min_event_count=1)
    data = pp.preprocess_dataset(paths.raw, "classical", cfg, loader=_stub_loader)
    r = data.report
    assert r.discovered == 4 and r.duplicates == 1
    assert r.processed == 2 and r.skipped_count == 1
    assert r.skipped[0]["file"] == "bad.mid" and "corrupt" in r.skipped[0]["reason"]
    assert r.unsupported == ["notes.mp3"]
    assert r.total_events == 120 and r.vocab_size == len(data.vocab)
    assert r.note_events == 120 and r.rest_events == 0


def test_all_files_corrupt_raises(paths):
    _write(paths.raw / "jazz", "bad1.mid", FAKE_MIDI + b"1")
    with pytest.raises(DatasetError, match="All 1 MIDI files were skipped"):
        pp.preprocess_dataset(paths.raw, "jazz", config.ModelConfig(), loader=_stub_loader)


def test_genre_comes_from_folder_not_filename(paths):
    _write(paths.raw / "classical", "jazz_standard.mid", FAKE_MIDI + b"1")
    _write(paths.raw, "top_level.mid", FAKE_MIDI + b"2")
    assert dataset.list_genres(paths.raw) == {"general": 1, "classical": 1}
    assert [p.name for p in dataset.scan_dataset(paths.raw, "classical").files] == ["jazz_standard.mid"]
    assert dataset.scan_dataset(paths.raw, config.ALL_GENRES).discovered == 2


def test_processed_data_is_saved_and_reused(paths):
    _write(paths.raw / "g", "a1.mid", FAKE_MIDI + b"1")
    _write(paths.raw / "g", "a2.mid", FAKE_MIDI + b"2")
    cfg = config.ModelConfig(min_event_count=1)
    first, reused_first = pp.get_or_build_processed(paths, "g", cfg, loader=_stub_loader)
    second, reused_second = pp.get_or_build_processed(paths, "g", cfg, loader=_stub_loader)
    assert (reused_first, reused_second) == (False, True)
    assert second.vocab.fingerprint() == first.vocab.fingerprint()
    assert [s.tolist() for s in second.songs] == [s.tolist() for s in first.songs]
    # adding a file invalidates the saved data
    _write(paths.raw / "g", "a3.mid", FAKE_MIDI + b"3")
    _, reused_third = pp.get_or_build_processed(paths, "g", cfg, loader=_stub_loader)
    assert reused_third is False


def test_upload_rejects_non_midi_and_never_overwrites(paths):
    saved, rejected = dataset.save_uploaded_files(
        paths.raw, "Classical", [("one.mid", FAKE_MIDI), ("fake.mid", b"RIFF...."), ("song.mp3", b"x")]
    )
    assert [p.name for p in saved] == ["one.mid"]
    assert {name for name, _ in rejected} == {"fake.mid", "song.mp3"}
    again, _ = dataset.save_uploaded_files(paths.raw, "classical", [("one.mid", FAKE_MIDI + b"x")])
    assert again[0].name == "one-2.mid" and (paths.raw / "classical" / "one.mid").exists()


def test_upload_needs_genre_and_strips_paths(paths):
    with pytest.raises(InvalidInputError):
        dataset.save_uploaded_files(paths.raw, "  !!  ", [("a.mid", FAKE_MIDI)])
    saved, _ = dataset.save_uploaded_files(paths.raw, "jazz", [("../../evil.mid", FAKE_MIDI)])
    assert saved[0].parent == paths.raw / "jazz"


def test_import_folder_copies_only_real_midi(paths, tmp_path):
    source = tmp_path / "source"
    _write(source, "a.mid")
    _write(source / "sub", "b.midi")
    _write(source, "c.mid", b"not midi")
    _write(source, "d.txt", b"hello")
    copied, skipped = dataset.import_folder(source, paths.raw, "jazz")
    assert len(copied) == 2 and len(skipped) == 1
    with pytest.raises(DatasetError):
        dataset.import_folder(tmp_path / "missing", paths.raw, "jazz")


# ----------------------------------------------------------------------------
# Real MIDI files (need pretty_midi; skipped automatically if it is missing)
# ----------------------------------------------------------------------------
def test_load_valid_midi_returns_events(tmp_path):
    pytest.importorskip("pretty_midi")
    from tests.helpers import make_midi_file

    path = make_midi_file(tmp_path / "song.mid", seed=1, n_notes=64)
    events = pp.load_midi_events(path)
    assert len(events) >= 64
    assert all(p == R or config.MIN_PITCH <= p <= config.MAX_PITCH for p, _ in events)
    assert all(0 <= d < len(D) for _, d in events)
    assert any(p == R for p, _ in events)  # the helper inserts rests


def test_load_corrupt_midi_raises(tmp_path):
    pytest.importorskip("pretty_midi")
    bad = tmp_path / "broken.mid"
    bad.write_bytes(b"this is definitely not a midi file")
    with pytest.raises(MidiLoadError):
        pp.load_midi_events(bad)


def test_load_drum_only_and_too_short_midi_raise(tmp_path):
    pytest.importorskip("pretty_midi")
    from tests.helpers import make_drum_only_midi, make_short_midi

    with pytest.raises(MidiLoadError, match="no usable"):
        pp.load_midi_events(make_drum_only_midi(tmp_path / "drums.mid"))
    with pytest.raises(MidiLoadError, match="only 4"):
        pp.load_midi_events(make_short_midi(tmp_path / "short.mid"))


def test_load_midi_respects_max_events(tmp_path):
    pytest.importorskip("pretty_midi")
    from tests.helpers import make_midi_file

    path = make_midi_file(tmp_path / "long.mid", n_notes=200)
    assert len(pp.load_midi_events(path, max_events=50)) == 50
