"""Sampling, validation, model discovery, and (when TensorFlow is installed) a real
train -> generate round trip on tiny synthetic data."""

from __future__ import annotations

import json

import numpy as np
import pytest

import config
from src import generate as gen
from src.utils import (
    GenerationError,
    IncompatibleModelError,
    InsufficientDataError,
    InvalidInputError,
    ModelNotFoundError,
)

R = config.REST_PITCH


# ----------------------------------------------------------------------------
# Pure logic (no TensorFlow needed)
# ----------------------------------------------------------------------------
def test_sample_next_low_temperature_prefers_most_likely():
    probs = np.array([0.1, 0.7, 0.2])
    rng = np.random.default_rng(0)
    picks = [gen.sample_next(probs, 0.05, 1.0, rng) for _ in range(50)]
    assert set(picks) == {1}


def test_sample_next_top_p_excludes_unlikely_events():
    probs = np.array([0.6, 0.3, 0.05, 0.05])
    rng = np.random.default_rng(1)
    picks = {gen.sample_next(probs, 1.0, 0.85, rng) for _ in range(300)}
    assert picks <= {0, 1}


def test_sample_next_high_temperature_explores():
    probs = np.array([0.9, 0.05, 0.05])
    rng = np.random.default_rng(2)
    picks = {gen.sample_next(probs, 1.8, 1.0, rng) for _ in range(300)}
    assert len(picks) == 3


@pytest.mark.parametrize("bad", [np.array([np.nan, 1.0]), np.array([0.0, 0.0])])
def test_sample_next_rejects_invalid_probabilities(bad):
    with pytest.raises(GenerationError):
        gen.sample_next(bad, 1.0, 1.0, np.random.default_rng(0))


def test_validate_events_accepts_good_and_rejects_bad():
    gen.validate_events([(60, 0), (R, 1), (72, 3)])
    for bad in ([], [(R, 1), (R, 2)], [(200, 0)], [(60, 99)], [(5, 0)]):
        with pytest.raises(GenerationError):
            gen.validate_events(bad)


def test_check_generation_params_limits():
    gen.check_generation_params(30, 100, 1.0)
    for args in ((1, 100, 1.0), (30, 5, 1.0), (30, 100, 9.0), (999, 100, 1.0)):
        with pytest.raises(InvalidInputError):
            gen.check_generation_params(*args)


def test_missing_model_gives_guidance(paths):
    with pytest.raises(ModelNotFoundError, match="Train Model"):
        gen.load_bundle("jazz", paths)
    with pytest.raises(ModelNotFoundError):
        gen.generate_music("jazz", 10, 100, 1.0, paths=paths)
    assert gen.list_trained_models(paths) == []
    assert gen.model_signature("jazz", paths) == "missing"


def test_unfinished_models_are_not_listed(paths):
    half = paths.models / "half"
    half.mkdir()
    (half / "metadata.json").write_text(json.dumps({"status": "training"}))
    (half / "model.keras").write_bytes(b"x")
    no_model = paths.models / "nomodel"
    no_model.mkdir()
    (no_model / "metadata.json").write_text(json.dumps({"status": "completed"}))
    hidden = paths.models / ".staging-x"
    hidden.mkdir()
    (hidden / "metadata.json").write_text(json.dumps({"status": "completed"}))
    (hidden / "model.keras").write_bytes(b"x")
    assert gen.list_trained_models(paths) == []


def test_completed_model_is_listed(paths):
    done = paths.models / "classical"
    done.mkdir()
    (done / "model.keras").write_bytes(b"x")
    (done / "metadata.json").write_text(
        json.dumps({"status": "completed", "genre": "classical", "vocab_size": 9,
                    "metrics": {"best_val_loss": 1.5}, "dataset": {"files_processed": 4}})
    )
    infos = gen.list_trained_models(paths)
    assert [(i.name, i.vocab_size, i.files_processed) for i in infos] == [("classical", 9, 4)]


def test_vocabulary_mismatch_is_detected_before_loading_tensorflow(paths):
    folder = paths.models / "g"
    folder.mkdir()
    (folder / "model.keras").write_bytes(b"x")
    (folder / "vocab.json").write_text(json.dumps({"events": [[60, 0], [62, 1]]}))
    (folder / "seeds.json").write_text(json.dumps([[0, 1]]))
    (folder / "metadata.json").write_text(
        json.dumps({"status": "completed", "vocab_fingerprint": "does-not-match", "config": {"seq_len": 2}})
    )
    with pytest.raises(IncompatibleModelError, match="does not match"):
        gen.load_bundle("g", paths)


# ----------------------------------------------------------------------------
# Real training + generation on synthetic data (needs TensorFlow and pretty_midi)
# ----------------------------------------------------------------------------
TINY = dict(seq_len=8, embedding_dim=8, lstm_units=16, num_layers=1, epochs=2,
            batch_size=32, stride=1, patience=2, min_event_count=1, seed=7)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    pytest.importorskip("tensorflow")
    pytest.importorskip("pretty_midi")
    from src.train import train_model
    from tests.helpers import make_midi_file

    root = tmp_path_factory.mktemp("echo_trained")
    paths = config.Paths.from_root(root)
    paths.ensure()
    for i in range(6):
        make_midi_file(paths.raw / "synthetic" / f"song{i}.mid", seed=i, n_notes=120)
    result = train_model("synthetic", config.ModelConfig(**TINY), paths=paths,
                         min_train_windows=50, min_val_windows=5)
    return paths, result


def test_training_saves_complete_model_and_metadata(trained):
    paths, result = trained
    meta = result.metadata
    assert meta["status"] == "completed" and meta["metrics"]["epochs_run"] >= 1
    assert np.isfinite(meta["metrics"]["best_val_loss"])
    for name in ("model.keras", "vocab.json", "seeds.json", "metadata.json", "history.json"):
        assert (result.model_dir / name).is_file()
    assert [m.name for m in gen.list_trained_models(paths)] == ["synthetic"]
    assert not [p for p in paths.models.iterdir() if p.name.startswith(".")]  # staging cleaned up


def test_model_construction_and_inference_shapes():
    pytest.importorskip("tensorflow")
    from src.model import build_model

    cfg = config.ModelConfig(**TINY)
    model = build_model(12, cfg)
    out = model(np.zeros((3, cfg.seq_len), dtype="int32"), training=False).numpy()
    assert out.shape == (3, 12)
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-4)


def test_generation_with_trained_model_is_valid_and_seed_reproducible(trained):
    paths, _ = trained
    bundle = gen.load_bundle("synthetic", paths)
    events1, seed1 = gen.generate_events(bundle, target_beats=16, temperature=1.0, seed=123)
    events2, _ = gen.generate_events(bundle, target_beats=16, temperature=1.0, seed=123)
    assert events1 == events2 and seed1 == 123
    gen.validate_events(events1)
    assert gen.total_beats(events1) >= 16
    assert all(ev in bundle.vocab for ev in events1)


def test_generation_stops_at_hard_limit(trained):
    paths, _ = trained
    bundle = gen.load_bundle("synthetic", paths)
    events, _ = gen.generate_events(bundle, target_beats=10_000, seed=1, max_events=25)
    assert len(events) == 25


def test_generate_music_writes_midi_that_reopens(trained):
    pretty_midi = pytest.importorskip("pretty_midi")
    paths, _ = trained
    result = gen.generate_music("synthetic", duration_seconds=8, tempo_bpm=120, temperature=0.9,
                                seed=5, paths=paths)
    assert result.midi_path.is_file() and result.midi_path.parent == paths.midi_out
    reopened = pretty_midi.PrettyMIDI(str(result.midi_path))
    assert sum(len(i.notes) for i in reopened.instruments) == result.note_count > 0
    assert result.duration_seconds >= 8 - 1e-6


def test_tampered_vocabulary_is_rejected(trained):
    paths, result = trained
    vocab_path = result.model_dir / "vocab.json"
    original = vocab_path.read_text()
    try:
        data = json.loads(original)
        data["events"].append([108, 0])
        vocab_path.write_text(json.dumps(data))
        with pytest.raises(IncompatibleModelError):
            gen.load_bundle("synthetic", paths)
    finally:
        vocab_path.write_text(original)


def test_training_refuses_empty_and_tiny_datasets(tmp_path):
    pytest.importorskip("tensorflow")
    pytest.importorskip("pretty_midi")
    from src.train import train_model
    from src.utils import DatasetError
    from tests.helpers import make_midi_file

    paths = config.Paths.from_root(tmp_path)
    paths.ensure()
    with pytest.raises(DatasetError, match="No MIDI files"):
        train_model("all", config.ModelConfig(**TINY), paths=paths)
    make_midi_file(paths.raw / "tiny" / "one.mid", n_notes=40)
    with pytest.raises(InsufficientDataError):
        train_model("tiny", config.ModelConfig(**TINY), paths=paths)
    assert gen.list_trained_models(paths) == []  # nothing pretends to be trained
