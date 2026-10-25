"""Training workflow: preprocess -> split -> train -> save model + metadata.

A model only counts as "trained" when ``metadata.json`` with
``"status": "completed"`` exists next to ``model.keras``. That file is written
last, after training has genuinely finished with finite losses, so a crashed or
interrupted run can never look like a success.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

import config
from src.model import build_model
from src.preprocessing import (
    WindowDataset,
    get_or_build_processed,
    pick_seed_windows,
    split_songs,
    validate_windows,
)
from src.utils import (
    InsufficientDataError,
    InvalidInputError,
    TrainingError,
    check_python_version,
    get_logger,
    require_module,
    slugify,
    utc_iso,
    utc_stamp,
)

log = get_logger("train")

ProgressFn = Callable[[dict], None]

MODEL_FILE = "model.keras"
METADATA_FILE = "metadata.json"
VOCAB_FILE = "vocab.json"
SEEDS_FILE = "seeds.json"
HISTORY_FILE = "history.json"


@dataclass
class TrainingResult:
    model_name: str
    model_dir: Path
    metadata: dict
    history: dict[str, list[float]]


def model_dir_for(paths: config.Paths, genre: str) -> Path:
    return paths.models / slugify(genre)


def set_global_seed(seed: int) -> None:
    """Make runs repeatable where TensorFlow allows it."""
    tf = require_module("tensorflow")
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _make_keras_sequence(tf, window_ds: WindowDataset):
    """Wrap a WindowDataset so Keras can stream batches from it."""
    base = getattr(tf.keras.utils, "PyDataset", None) or tf.keras.utils.Sequence

    class WindowSequence(base):  # type: ignore[misc, valid-type]
        def __init__(self, dataset: WindowDataset):
            super().__init__()
            self.dataset = dataset

        def __len__(self) -> int:
            return self.dataset.num_batches

        def __getitem__(self, index: int):
            return self.dataset.batch(index)

        def on_epoch_end(self) -> None:
            self.dataset.on_epoch_end()

    return WindowSequence(window_ds)


def train_model(
    genre: str = config.ALL_GENRES,
    cfg: config.ModelConfig | None = None,
    paths: config.Paths = config.DEFAULT_PATHS,
    progress: ProgressFn | None = None,
    force_reprocess: bool = False,
    min_train_windows: int = config.MIN_TRAIN_WINDOWS,
    min_val_windows: int = config.MIN_VAL_WINDOWS,
) -> TrainingResult:
    """Train an LSTM on the MIDI files of ``genre`` and save it under ``models/<genre>/``.

    ``progress`` receives dictionaries:
      {"type": "status", "message": str}
      {"type": "preprocess", "done": int, "total": int, "file": str}
      {"type": "epoch", "epoch": int, "epochs": int, "loss": float, "val_loss": float, ...}

    Raises an ``EchoError`` subclass (never returns) if anything goes wrong.
    """
    cfg = cfg or config.ModelConfig()
    notify = progress or (lambda event: None)

    check_python_version(strict=False)
    try:
        cfg.validate()
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    tf = require_module("tensorflow")

    name = slugify(genre)
    notify({"type": "status", "message": f"Scanning and preprocessing MIDI files for '{name}'..."})
    data, reused = get_or_build_processed(
        paths, genre, cfg, progress=progress, force=force_reprocess
    )
    notify(
        {
            "type": "status",
            "message": (
                f"{'Reused saved' if reused else 'Built'} dataset: {data.report.processed} files, "
                f"{data.report.total_events:,} events, vocabulary of {len(data.vocab)}."
            ),
        }
    )

    train_segments, val_segments, strategy = split_songs(data.songs, cfg.val_fraction, cfg.seed)
    train_ds = WindowDataset(
        train_segments, cfg.seq_len, cfg.batch_size, stride=cfg.stride, shuffle=True, seed=cfg.seed
    )
    val_ds = WindowDataset(val_segments, cfg.seq_len, cfg.batch_size, stride=cfg.stride)
    if train_ds.num_windows < min_train_windows or val_ds.num_windows < min_val_windows:
        raise InsufficientDataError(
            f"Not enough music to train: {train_ds.num_windows} training windows and "
            f"{val_ds.num_windows} validation windows (need at least {min_train_windows} and "
            f"{min_val_windows}). Add more MIDI files, or reduce 'sequence length' / 'stride' "
            f"in the training settings."
        )
    for sample_ds in (train_ds, val_ds):
        x_check, y_check = sample_ds.batch(0)
        try:
            validate_windows(x_check, y_check, len(data.vocab))
        except ValueError as exc:
            raise TrainingError(f"Prepared training data failed validation: {exc}") from exc

    set_global_seed(cfg.seed)
    model = build_model(len(data.vocab), cfg)
    notify(
        {
            "type": "status",
            "message": (
                f"Training {model.count_params():,} parameters on {train_ds.num_windows:,} windows "
                f"(validating on {val_ds.num_windows:,}, split: {strategy})."
            ),
        }
    )

    staging = paths.models / f".staging-{name}-{utc_stamp()}"
    staging.mkdir(parents=True, exist_ok=True)
    started = time.time()

    class EpochReporter(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            notify(
                {
                    "type": "epoch",
                    "epoch": epoch + 1,
                    "epochs": cfg.epochs,
                    "loss": float(logs.get("loss", float("nan"))),
                    "val_loss": float(logs.get("val_loss", float("nan"))),
                    "accuracy": float(logs.get("accuracy", float("nan"))),
                    "val_accuracy": float(logs.get("val_accuracy", float("nan"))),
                }
            )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(staging / "best.keras"), monitor="val_loss", save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=cfg.patience, restore_best_weights=True
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        EpochReporter(),
    ]

    try:
        history = model.fit(
            _make_keras_sequence(tf, train_ds),
            validation_data=_make_keras_sequence(tf, val_ds),
            epochs=cfg.epochs,
            callbacks=callbacks,
            verbose=0,
        )
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        log.exception("Training crashed")
        raise TrainingError(f"Training failed: {type(exc).__name__}: {exc}") from exc

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    val_losses = hist.get("val_loss", [])
    if not val_losses or not all(math.isfinite(v) for v in val_losses):
        shutil.rmtree(staging, ignore_errors=True)
        raise TrainingError(
            "Training produced invalid (NaN/infinite) losses and was discarded. "
            "Try a lower learning rate or a smaller model."
        )

    best_epoch = int(np.argmin(val_losses)) + 1
    try:
        model.save(str(staging / MODEL_FILE))
        (staging / "best.keras").unlink(missing_ok=True)
        (staging / VOCAB_FILE).write_text(json.dumps(data.vocab.to_dict()), encoding="utf-8")
        seeds = pick_seed_windows(train_segments or data.songs, cfg.seq_len)
        (staging / SEEDS_FILE).write_text(json.dumps(seeds), encoding="utf-8")
        (staging / HISTORY_FILE).write_text(json.dumps(hist, indent=2), encoding="utf-8")
        metadata = {
            "status": "completed",
            "model_name": name,
            "genre": genre,
            "trained_at": utc_iso(),
            "echo_version": config.APP_VERSION,
            "tensorflow_version": tf.__version__,
            "model_file": MODEL_FILE,
            "config": cfg.to_dict(),
            "vocab_size": len(data.vocab),
            "vocab_fingerprint": data.vocab.fingerprint(),
            "duration_classes": list(config.DURATION_CLASSES),
            "dataset": {
                "files_discovered": data.report.discovered,
                "files_processed": data.report.processed,
                "files_skipped": data.report.skipped_count,
                "total_events": data.report.total_events,
                "split_strategy": strategy,
                "train_windows": train_ds.num_windows,
                "val_windows": val_ds.num_windows,
                "dataset_fingerprint": data.report.fingerprint,
            },
            "metrics": {
                "epochs_run": len(val_losses),
                "best_epoch": best_epoch,
                "best_val_loss": float(min(val_losses)),
                "final_train_loss": hist["loss"][-1],
                "best_val_accuracy": float(max(hist.get("val_accuracy", [float("nan")]))),
            },
            "training_seconds": round(time.time() - started, 1),
        }
        (staging / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        log.exception("Saving the trained model failed")
        raise TrainingError(f"Training finished but saving the model failed: {exc}") from exc

    final_dir = model_dir_for(paths, genre)
    backup = None
    try:
        if final_dir.exists():
            backup = final_dir.with_name(f".old-{final_dir.name}-{utc_stamp()}")
            final_dir.rename(backup)
        staging.rename(final_dir)
    except OSError as exc:
        if backup is not None and backup.exists() and not final_dir.exists():
            backup.rename(final_dir)  # restore the previous model
        raise TrainingError(
            f"The model was trained but could not be moved into {final_dir}: {exc}. "
            f"The files are still in {staging}."
        ) from exc
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)

    notify({"type": "status", "message": f"Training complete. Model saved to {final_dir}."})
    log.info("Trained model '%s': %s", name, metadata["metrics"])
    return TrainingResult(model_name=name, model_dir=final_dir, metadata=metadata, history=hist)
