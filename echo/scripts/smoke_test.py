"""End-to-end smoke test for ECHO (about a minute on a laptop CPU).

    python scripts/smoke_test.py

What it does, in a throw-away temporary folder (your real data/, models/ and
outputs/ folders are never touched):

  1. writes a few small SYNTHETIC MIDI files,
  2. preprocesses them,
  3. trains a tiny LSTM for a few epochs,
  4. generates a melody and exports it as a MIDI file,
  5. re-opens that MIDI file to prove it is valid,
  6. renders WAV audio if FluidSynth and a SoundFont are installed.

The synthetic files are only plumbing for the test. They are NOT music data, and a
passing smoke test says nothing about how good ECHO sounds on real music. The
script prints exactly which steps ran and which were skipped, and exits with a
non-zero code if any step that should work failed.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from src.utils import EchoError, module_available  # noqa: E402

TINY = config.ModelConfig(
    seq_len=8, embedding_dim=8, lstm_units=16, num_layers=1, epochs=3,
    batch_size=32, stride=1, patience=3, min_event_count=1, seed=7,
)


def step(message: str) -> None:
    print(f"[smoke] {message}", flush=True)


def main() -> int:
    missing = [m for m in ("tensorflow", "pretty_midi", "numpy") if not module_available(m)]
    if missing:
        print(
            f"[smoke] FAILED: missing package(s): {', '.join(missing)}.\n"
            "        Activate your virtual environment and run: pip install -r requirements.txt"
        )
        return 1

    from src.audio_renderer import check_audio_status, render_wav, wav_duration_seconds
    from src.generate import generate_music, list_trained_models
    from src.train import train_model
    from tests.helpers import make_midi_file

    with tempfile.TemporaryDirectory(prefix="echo-smoke-") as tmp:
        paths = config.Paths.from_root(tmp)
        paths.ensure()
        step(f"working in a temporary folder: {tmp}")

        for i in range(6):
            make_midi_file(paths.raw / "synthetic" / f"song{i}.mid", seed=i, n_notes=120)
        step("1/6 wrote 6 synthetic MIDI files")

        try:
            result = train_model(
                "synthetic", TINY, paths=paths, min_train_windows=50, min_val_windows=5,
                progress=lambda e: e.get("type") == "epoch" and print(
                    f"        epoch {e['epoch']}/{e['epochs']}  loss {e['loss']:.3f}  val_loss {e['val_loss']:.3f}"
                ),
            )
        except EchoError as exc:
            print(f"[smoke] FAILED while training: {exc}")
            return 1
        metrics = result.metadata["metrics"]
        step(f"2/6 trained a tiny LSTM: best validation loss {metrics['best_val_loss']:.3f} "
             f"after {metrics['epochs_run']} epoch(s)")

        models = list_trained_models(paths)
        if [m.name for m in models] != ["synthetic"]:
            print("[smoke] FAILED: the trained model was not listed as completed.")
            return 1
        step("3/6 the saved model is discoverable")

        try:
            music = generate_music("synthetic", duration_seconds=8, tempo_bpm=120,
                                   temperature=1.0, seed=1, paths=paths)
        except EchoError as exc:
            print(f"[smoke] FAILED while generating: {exc}")
            return 1
        step(f"4/6 generated {music.note_count} notes ({music.duration_seconds:.1f} s) -> {music.midi_path.name}")

        import pretty_midi

        reopened = pretty_midi.PrettyMIDI(str(music.midi_path))
        notes = sum(len(i.notes) for i in reopened.instruments)
        if notes != music.note_count or notes == 0:
            print(f"[smoke] FAILED: MIDI file holds {notes} notes, expected {music.note_count}.")
            return 1
        step(f"5/6 the MIDI file re-opens correctly ({notes} notes)")

        audio = check_audio_status(paths.soundfonts)
        if audio.can_render:
            try:
                wav = render_wav(music.midi_path, paths.audio_out, audio)
            except EchoError as exc:
                print(f"[smoke] FAILED while rendering audio: {exc}")
                return 1
            step(f"6/6 rendered WAV audio, {wav_duration_seconds(wav):.1f} s")
        else:
            step("6/6 SKIPPED audio rendering: " + " ".join(audio.problems()))
            step("      (MIDI export works without it; see README, 'Audio setup')")

    step("PASSED. The pipeline works end to end on synthetic data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
