"""Central configuration for ECHO.

Everything that several modules must agree on lives here: folder locations,
the musical event representation, validation limits and the model settings.
Keeping them in one place is what prevents a model saved by ``train.py`` from
becoming unreadable by ``generate.py``.

This module imports only the standard library, so it is always safe to import.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Branding
# --------------------------------------------------------------------------- #
APP_NAME = "ECHO"
APP_TAGLINE = "Every note leaves an echo."
APP_SUBTITLE = "Intelligent Music Generation Studio"
APP_VERSION = "1.0.0"

# --------------------------------------------------------------------------- #
# Python compatibility
# --------------------------------------------------------------------------- #
MIN_PYTHON = (3, 10)
# TensorFlow wheels pinned in requirements.txt exist for Python 3.9 - 3.12.
MAX_TESTED_PYTHON = (3, 12)
RECOMMENDED_PYTHON = "3.11 or 3.12"

# --------------------------------------------------------------------------- #
# Folder layout
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Paths:
    """All folders used by the application.

    Functions accept a ``Paths`` object so tests can point everything at a
    temporary directory and never touch real user files.
    """

    root: Path
    raw: Path
    processed: Path
    models: Path
    midi_out: Path
    audio_out: Path
    soundfonts: Path
    logs: Path

    @classmethod
    def from_root(cls, root: Path | str) -> "Paths":
        root = Path(root)
        return cls(
            root=root,
            raw=root / "data" / "raw",
            processed=root / "data" / "processed",
            models=root / "models",
            midi_out=root / "outputs" / "midi",
            audio_out=root / "outputs" / "audio",
            soundfonts=root / "assets" / "soundfonts",
            logs=root / "logs",
        )

    def ensure(self) -> None:
        """Create every folder if missing (never deletes anything)."""
        for folder in (
            self.raw,
            self.processed,
            self.models,
            self.midi_out,
            self.audio_out,
            self.soundfonts,
            self.logs,
        ):
            folder.mkdir(parents=True, exist_ok=True)


DEFAULT_PATHS = Paths.from_root(ROOT_DIR)

# --------------------------------------------------------------------------- #
# Musical event representation
# --------------------------------------------------------------------------- #
# One "event" is a pair (pitch, duration_index):
#   * pitch            - MIDI note number 21..108, or REST_PITCH for a silence
#   * duration_index   - index into DURATION_CLASSES (lengths in quarter notes)
# Durations are measured in beats, not seconds, so the training data is tempo
# independent. The tempo is chosen later, when the MIDI file is written.
REST_PITCH = -1
MIN_PITCH = 21  # A0, lowest piano key
MAX_PITCH = 108  # C8, highest piano key
DURATION_CLASSES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0)
GRID_BEATS = 0.25  # onsets are snapped to a 16th-note grid
MAX_REST_BEATS = 8.0  # longer silences are shortened to this
NOTE_ARTICULATION = 0.95  # fraction of a duration that a note actually sounds

# --------------------------------------------------------------------------- #
# Dataset limits
# --------------------------------------------------------------------------- #
MIDI_EXTENSIONS = (".mid", ".midi")
IGNORED_EXTENSIONS = (".gitkeep", ".md", ".txt", ".json", ".csv", ".license")
MIN_NOTES_PER_FILE = 16
MAX_EVENTS_PER_SONG = 8000  # keeps one very long piece from dominating
MAX_UPLOAD_MB = 20
UNLABELLED_GENRE = "general"  # files placed directly in data/raw
ALL_GENRES = "all"  # every MIDI file under data/raw
MIN_TRAIN_WINDOWS = 200  # below this, training is refused
MIN_VAL_WINDOWS = 20

# --------------------------------------------------------------------------- #
# Generation limits
# --------------------------------------------------------------------------- #
MIN_BPM, MAX_BPM = 40, 220
MIN_DURATION_SECONDS, MAX_DURATION_SECONDS = 5, 180
MIN_TEMPERATURE, MAX_TEMPERATURE = 0.2, 1.8
MAX_GENERATED_EVENTS = 2000  # hard stop so generation can never loop forever
DEFAULT_TOP_P = 0.95

# General MIDI programs offered in the interface (name -> program number).
INSTRUMENTS: dict[str, int] = {
    "Grand Piano": 0,
    "Electric Piano": 4,
    "Vibraphone": 11,
    "Nylon Guitar": 24,
    "Violin": 40,
    "Flute": 73,
}

# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #
SAMPLE_RATE = 44100
RENDER_TIMEOUT_SECONDS = 180
ENV_FLUIDSYNTH = "ECHO_FLUIDSYNTH"  # optional: full path to fluidsynth(.exe)
ENV_SOUNDFONT = "ECHO_SOUNDFONT"  # optional: full path to a .sf2 / .sf3 file


# --------------------------------------------------------------------------- #
# Model configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Settings that define the LSTM and how it is trained.

    The defaults are deliberately small so that training is practical on a
    normal Windows laptop CPU.
    """

    seq_len: int = 32  # how many past events the model looks at
    embedding_dim: int = 48
    lstm_units: int = 128
    num_layers: int = 2  # 1 or 2 LSTM layers
    dropout: float = 0.3
    learning_rate: float = 0.002
    batch_size: int = 128
    epochs: int = 20
    patience: int = 4  # early-stopping patience (epochs)
    stride: int = 2  # distance between training windows
    val_fraction: float = 0.15
    min_event_count: int = 2  # rarer events are dropped from the vocabulary
    seed: int = 42

    def validate(self) -> None:
        problems: list[str] = []
        if not 4 <= self.seq_len <= 256:
            problems.append("seq_len must be between 4 and 256")
        if not 4 <= self.embedding_dim <= 512:
            problems.append("embedding_dim must be between 4 and 512")
        if not 8 <= self.lstm_units <= 1024:
            problems.append("lstm_units must be between 8 and 1024")
        if self.num_layers not in (1, 2):
            problems.append("num_layers must be 1 or 2")
        if not 0.0 <= self.dropout < 0.9:
            problems.append("dropout must be between 0.0 and 0.9")
        if not 1e-5 <= self.learning_rate <= 0.1:
            problems.append("learning_rate must be between 0.00001 and 0.1")
        if not 1 <= self.batch_size <= 4096:
            problems.append("batch_size must be between 1 and 4096")
        if not 1 <= self.epochs <= 500:
            problems.append("epochs must be between 1 and 500")
        if not 1 <= self.patience <= 100:
            problems.append("patience must be between 1 and 100")
        if not 1 <= self.stride <= self.seq_len:
            problems.append("stride must be between 1 and seq_len")
        if not 0.05 <= self.val_fraction <= 0.5:
            problems.append("val_fraction must be between 0.05 and 0.5")
        if self.min_event_count < 1:
            problems.append("min_event_count must be at least 1")
        if problems:
            raise ValueError("Invalid model configuration: " + "; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# A very small preset for a first end-to-end run on a slow machine.
QUICK_PRESET = ModelConfig(
    seq_len=24,
    embedding_dim=32,
    lstm_units=64,
    num_layers=1,
    epochs=8,
    batch_size=128,
    stride=2,
)
