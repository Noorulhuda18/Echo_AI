"""Music generation with a trained LSTM.

Algorithm (autoregressive sampling):
    1. Load the model and the vocabulary that was saved with it.
    2. Start from a real opening phrase of the training music (the "seed").
    3. Ask the model for a probability for every possible next event.
    4. Apply the creativity (temperature) and nucleus (top-p) filters, then
       sample one event.
    5. Append it, slide the context window forward by one event, repeat
       until the requested length is reached (or the hard safety limit).
    6. Validate the finished sequence before it is exported.

One model is trained per genre folder (``models/<genre>/``). ECHO does not
claim that a single model can switch style on demand.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import config
from src.midi_export import write_midi
from src.preprocessing import Event, Vocabulary
from src.train import (
    METADATA_FILE,
    MODEL_FILE,
    SEEDS_FILE,
    VOCAB_FILE,
)
from src.utils import (
    GenerationError,
    IncompatibleModelError,
    InvalidInputError,
    ModelNotFoundError,
    get_logger,
    require_module,
    slugify,
)

log = get_logger("generate")


@dataclass
class ModelInfo:
    name: str
    genre: str
    trained_at: str
    best_val_loss: float
    vocab_size: int
    files_processed: int
    metadata: dict = field(repr=False, default_factory=dict)


@dataclass
class ModelBundle:
    name: str
    model: object
    vocab: Vocabulary
    metadata: dict
    seeds: list[list[int]]

    @property
    def seq_len(self) -> int:
        return int(self.metadata["config"]["seq_len"])


@dataclass
class GenerationResult:
    events: list[Event]
    midi_path: Path
    seed: int
    tempo_bpm: int
    duration_beats: float
    duration_seconds: float
    note_count: int


# --------------------------------------------------------------------------- #
# Discovering and loading models
# --------------------------------------------------------------------------- #


def _read_metadata(folder: Path) -> dict | None:
    try:
        meta = json.loads((folder / METADATA_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if meta.get("status") != "completed" or not (folder / MODEL_FILE).is_file():
        return None
    return meta


def list_trained_models(paths: config.Paths = config.DEFAULT_PATHS) -> list[ModelInfo]:
    """All fully trained models. Half-finished or hidden folders are ignored."""
    infos: list[ModelInfo] = []
    if not paths.models.is_dir():
        return infos
    for folder in sorted(p for p in paths.models.iterdir() if p.is_dir() and not p.name.startswith(".")):
        meta = _read_metadata(folder)
        if meta is None:
            continue
        infos.append(
            ModelInfo(
                name=folder.name,
                genre=str(meta.get("genre", folder.name)),
                trained_at=str(meta.get("trained_at", "")),
                best_val_loss=float(meta.get("metrics", {}).get("best_val_loss", float("nan"))),
                vocab_size=int(meta.get("vocab_size", 0)),
                files_processed=int(meta.get("dataset", {}).get("files_processed", 0)),
                metadata=meta,
            )
        )
    return infos


def model_signature(name: str, paths: config.Paths = config.DEFAULT_PATHS) -> str:
    """Changes whenever the model is retrained; used as a cache key by the UI."""
    folder = paths.models / slugify(name)
    meta = _read_metadata(folder)
    if meta is None:
        return "missing"
    mtime = (folder / MODEL_FILE).stat().st_mtime_ns
    return f"{meta.get('trained_at')}|{meta.get('vocab_fingerprint')}|{mtime}"


def load_bundle(name: str, paths: config.Paths = config.DEFAULT_PATHS) -> ModelBundle:
    """Load a trained model with its vocabulary and verify they belong together."""
    slug = slugify(name)
    folder = paths.models / slug
    meta = _read_metadata(folder) if folder.is_dir() else None
    if meta is None:
        raise ModelNotFoundError(
            f"No trained model named '{slug}' was found. Open the 'Train Model' page "
            f"(or run scripts/train_model.py) to train one first."
        )
    try:
        vocab = Vocabulary.from_dict(json.loads((folder / VOCAB_FILE).read_text(encoding="utf-8")))
        seeds = json.loads((folder / SEEDS_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError) as exc:
        raise IncompatibleModelError(
            f"The saved vocabulary for '{slug}' is missing or unreadable ({exc}). Retrain the model."
        ) from exc

    if vocab.fingerprint() != meta.get("vocab_fingerprint"):
        raise IncompatibleModelError(
            f"The vocabulary for '{slug}' does not match the one used during training "
            "(the musical event encoding changed). Retrain the model."
        )

    tf = require_module("tensorflow")
    try:
        model = tf.keras.models.load_model(str(folder / MODEL_FILE), compile=False)
    except Exception as exc:
        log.exception("Could not load model %s", folder)
        raise IncompatibleModelError(
            f"The model file for '{slug}' could not be loaded ({type(exc).__name__}). "
            "It may have been saved by a different TensorFlow version - retrain it."
        ) from exc

    seq_len = int(meta["config"]["seq_len"])
    out_size = model.output_shape[-1]
    in_len = model.input_shape[1]
    if out_size != len(vocab) or in_len != seq_len:
        raise IncompatibleModelError(
            f"The model for '{slug}' expects {in_len} inputs and {out_size} outputs, but its "
            f"vocabulary has {len(vocab)} events and the sequence length is {seq_len}. Retrain it."
        )
    valid_seeds = [s for s in seeds if len(s) == seq_len and all(0 <= i < len(vocab) for i in s)]
    if not valid_seeds:
        raise IncompatibleModelError(f"The saved seed phrases for '{slug}' are invalid. Retrain it.")
    return ModelBundle(name=slug, model=model, vocab=vocab, metadata=meta, seeds=valid_seeds)


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #


def sample_next(
    probs: np.ndarray,
    temperature: float,
    top_p: float,
    rng: np.random.Generator,
) -> int:
    """Pick the next event id from a probability vector.

    * temperature < 1 : safer, more repetitive;  > 1 : bolder, more surprising
    * top_p keeps only the smallest group of events whose probability adds up to top_p
    """
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 1 or not np.all(np.isfinite(probs)) or probs.sum() <= 0:
        raise GenerationError("The model returned invalid probabilities.")
    logits = np.log(np.clip(probs, 1e-12, None)) / max(temperature, 1e-3)
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    if top_p < 1.0:
        order = np.argsort(weights)[::-1]
        cumulative = np.cumsum(weights[order])
        keep = order[: int(np.searchsorted(cumulative, top_p) + 1)]
        mask = np.zeros_like(weights)
        mask[keep] = weights[keep]
        weights = mask / mask.sum()
    return int(rng.choice(len(weights), p=weights))


def validate_events(events: list[Event]) -> None:
    """Raise GenerationError unless the sequence is musically usable."""
    if not events:
        raise GenerationError("Generation produced no events.")
    notes = 0
    for pitch, duration_index in events:
        if not 0 <= duration_index < len(config.DURATION_CLASSES):
            raise GenerationError(f"Invalid duration index {duration_index}.")
        if pitch == config.REST_PITCH:
            continue
        if not config.MIN_PITCH <= pitch <= config.MAX_PITCH:
            raise GenerationError(f"Invalid pitch {pitch}.")
        notes += 1
    if notes == 0:
        raise GenerationError("Generation produced only silence. Try a higher creativity value.")


def total_beats(events: list[Event]) -> float:
    return float(sum(config.DURATION_CLASSES[d] for _, d in events))


def check_generation_params(duration_seconds: float, tempo_bpm: float, temperature: float) -> None:
    problems = []
    if not config.MIN_BPM <= tempo_bpm <= config.MAX_BPM:
        problems.append(f"tempo must be between {config.MIN_BPM} and {config.MAX_BPM} BPM")
    if not config.MIN_DURATION_SECONDS <= duration_seconds <= config.MAX_DURATION_SECONDS:
        problems.append(
            f"duration must be between {config.MIN_DURATION_SECONDS} and "
            f"{config.MAX_DURATION_SECONDS} seconds"
        )
    if not config.MIN_TEMPERATURE <= temperature <= config.MAX_TEMPERATURE:
        problems.append(
            f"creativity must be between {config.MIN_TEMPERATURE} and {config.MAX_TEMPERATURE}"
        )
    if problems:
        raise InvalidInputError("Invalid settings: " + "; ".join(problems) + ".")


def generate_events(
    bundle: ModelBundle,
    target_beats: float,
    temperature: float = 1.0,
    top_p: float = config.DEFAULT_TOP_P,
    seed: int | None = None,
    max_events: int = config.MAX_GENERATED_EVENTS,
) -> tuple[list[Event], int]:
    """Generate events until ``target_beats`` of music exist. Returns ``(events, seed)``."""
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**31))
    rng = np.random.default_rng(seed)
    window = np.asarray(bundle.seeds[int(rng.integers(len(bundle.seeds)))], dtype=np.int32)
    events: list[Event] = []
    recent: list[int] = []
    beats = 0.0
    while beats < target_beats and len(events) < max_events:
        probs = bundle.model(window[None, :], training=False).numpy()[0]
        probs = np.asarray(probs, dtype=np.float64)
        # Escape hatch: if the model gets stuck repeating one event, forbid it once.
        if len(recent) >= 16 and len(set(recent[-16:])) == 1 and probs.sum() > 0:
            probs[recent[-1]] = 0.0
            probs = probs / probs.sum() if probs.sum() > 0 else probs
        next_id = sample_next(probs, temperature, top_p, rng)
        event = bundle.vocab.decode(next_id)
        events.append(event)
        recent.append(next_id)
        beats += config.DURATION_CLASSES[event[1]]
        window = np.append(window[1:], np.int32(next_id))
    validate_events(events)
    return events, seed


def generate_music(
    model_name: str,
    duration_seconds: float = 30,
    tempo_bpm: int = 100,
    temperature: float = 1.0,
    seed: int | None = None,
    program: int = 0,
    paths: config.Paths = config.DEFAULT_PATHS,
    bundle: ModelBundle | None = None,
) -> GenerationResult:
    """Generate a piece and save it as a MIDI file (never overwrites existing files)."""
    check_generation_params(duration_seconds, tempo_bpm, temperature)
    bundle = bundle or load_bundle(model_name, paths)
    target_beats = duration_seconds * tempo_bpm / 60.0
    started = time.time()
    events, used_seed = generate_events(bundle, target_beats, temperature, seed=seed)
    midi_path = write_midi(
        events,
        tempo_bpm=tempo_bpm,
        out_dir=paths.midi_out,
        model_name=bundle.name,
        seed=used_seed,
        program=program,
    )
    beats = total_beats(events)
    log.info(
        "Generated %d events (%.1f beats) from '%s' in %.1fs -> %s",
        len(events), beats, bundle.name, time.time() - started, midi_path,
    )
    return GenerationResult(
        events=events,
        midi_path=midi_path,
        seed=used_seed,
        tempo_bpm=int(tempo_bpm),
        duration_beats=beats,
        duration_seconds=beats * 60.0 / tempo_bpm,
        note_count=sum(1 for p, _ in events if p != config.REST_PITCH),
    )
