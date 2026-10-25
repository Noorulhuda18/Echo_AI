"""Command-line training for ECHO.

Examples (Windows PowerShell, from the project folder, venv active):

    python scripts/train_model.py --list
    python scripts/train_model.py --genre classical --quick
    python scripts/train_model.py --genre all --epochs 20 --units 128

The result is saved to models/<genre>/ and appears in the Streamlit app.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.dataset import list_genres  # noqa: E402
from src.utils import EchoError, python_version_message, setup_logging  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the ECHO LSTM on MIDI files in data/raw/<genre>/.")
    p.add_argument("--genre", default=config.ALL_GENRES, help="folder name inside data/raw, or 'all' (default)")
    p.add_argument("--list", action="store_true", help="show the genre folders found in data/raw and exit")
    p.add_argument("--quick", action="store_true", help="tiny model, 8 epochs: for a first end-to-end test")
    d = config.ModelConfig()
    p.add_argument("--epochs", type=int, default=None, help=f"maximum epochs (default {d.epochs})")
    p.add_argument("--seq-len", type=int, default=None, help=f"context length in events (default {d.seq_len})")
    p.add_argument("--units", type=int, default=None, help=f"LSTM units (default {d.lstm_units})")
    p.add_argument("--layers", type=int, choices=(1, 2), default=None, help=f"LSTM layers (default {d.num_layers})")
    p.add_argument("--embedding", type=int, default=None, help=f"embedding size (default {d.embedding_dim})")
    p.add_argument("--dropout", type=float, default=None, help=f"dropout (default {d.dropout})")
    p.add_argument("--lr", type=float, default=None, help=f"learning rate (default {d.learning_rate})")
    p.add_argument("--batch-size", type=int, default=None, help=f"batch size (default {d.batch_size})")
    p.add_argument("--stride", type=int, default=None, help=f"window stride (default {d.stride})")
    p.add_argument("--patience", type=int, default=None, help=f"early-stopping patience (default {d.patience})")
    p.add_argument("--seed", type=int, default=None, help=f"random seed (default {d.seed})")
    p.add_argument("--reprocess", action="store_true", help="ignore saved processed data and re-read the MIDI files")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config.DEFAULT_PATHS.ensure()
    setup_logging()

    if args.list:
        genres = list_genres(config.DEFAULT_PATHS.raw)
        if not genres:
            print("data/raw is empty. Add .mid files (see README, 'Dataset setup').")
            return 1
        for name, count in genres.items():
            print(f"  {name}: {count} MIDI file(s)")
        print(f"  {config.ALL_GENRES}: every file above")
        return 0

    warning = python_version_message()
    if warning:
        print(f"[warning] {warning}")

    cfg = config.ModelConfig.from_dict((config.QUICK_PRESET if args.quick else config.ModelConfig()).to_dict())
    overrides = {
        "epochs": args.epochs, "seq_len": args.seq_len, "lstm_units": args.units,
        "num_layers": args.layers, "embedding_dim": args.embedding, "dropout": args.dropout,
        "learning_rate": args.lr, "batch_size": args.batch_size, "stride": args.stride,
        "patience": args.patience, "seed": args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    from src.train import train_model  # imported late so --list works without TensorFlow

    def show(event: dict) -> None:
        kind = event.get("type")
        if kind == "status":
            print(f"[info] {event['message']}")
        elif kind == "preprocess" and (event["done"] == event["total"] or event["done"] % 10 == 0):
            print(f"[info] read {event['done']}/{event['total']} MIDI files")
        elif kind == "epoch":
            print(
                f"[epoch {event['epoch']:>3}/{event['epochs']}] loss {event['loss']:.4f}  "
                f"val_loss {event['val_loss']:.4f}  val_acc {event['val_accuracy']:.3f}"
            )

    try:
        result = train_model(args.genre, cfg, progress=show, force_reprocess=args.reprocess)
    except EchoError as exc:
        print(f"[error] {exc}")
        return 1
    except KeyboardInterrupt:
        print("[error] Training interrupted. No model was saved.")
        return 130
    m = result.metadata["metrics"]
    print(
        f"[done] Saved '{result.model_name}' to {result.model_dir}\n"
        f"       best validation loss {m['best_val_loss']:.4f} at epoch {m['best_epoch']} "
        f"({m['epochs_run']} epochs, {result.metadata['training_seconds']} s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
