"""Shared helpers: error types, logging, dependency checks and file safety."""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from types import ModuleType

import config

# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class EchoError(Exception):
    """Base class. The message is written to be shown directly to the user."""


class DependencyError(EchoError):
    """A required Python package or Python version is missing/unsupported."""


class InvalidInputError(EchoError):
    """A user-supplied value is outside its allowed range."""


class DatasetError(EchoError):
    """The dataset is missing, empty or unusable."""


class MidiLoadError(DatasetError):
    """A single MIDI file could not be read or contains no usable notes."""


class InsufficientDataError(DatasetError):
    """There is not enough music to train a meaningful model."""


class TrainingError(EchoError):
    """Training started but did not finish successfully."""


class ModelNotFoundError(EchoError):
    """No completed trained model exists for the requested name."""


class IncompatibleModelError(EchoError):
    """The saved model and its vocabulary/metadata do not match."""


class GenerationError(EchoError):
    """A generated sequence was invalid or generation failed."""


class MidiExportError(EchoError):
    """A MIDI file could not be written or failed verification."""


class AudioRenderError(EchoError):
    """MIDI could not be rendered into audio."""


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOGGING_READY = False


def setup_logging(paths: config.Paths = config.DEFAULT_PATHS) -> None:
    """Send detailed diagnostics to ``logs/echo.log`` (idempotent)."""
    global _LOGGING_READY
    if _LOGGING_READY:
        return
    logger = logging.getLogger("echo")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        paths.logs.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = RotatingFileHandler(
            paths.logs / "echo.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8"
        )
    except OSError:
        handler = logging.StreamHandler()  # read-only folder: fall back to console
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    _LOGGING_READY = True


def get_logger(name: str = "echo") -> logging.Logger:
    setup_logging()
    return logging.getLogger(name if name.startswith("echo") else f"echo.{name}")


# --------------------------------------------------------------------------- #
# Environment checks
# --------------------------------------------------------------------------- #


def python_version_message() -> str | None:
    """Return a warning/error string if the Python version is a problem, else None."""
    current = sys.version_info[:2]
    if current < config.MIN_PYTHON:
        return (
            f"Python {current[0]}.{current[1]} is too old. "
            f"Install Python {config.RECOMMENDED_PYTHON}."
        )
    if current > config.MAX_TESTED_PYTHON:
        return (
            f"Python {current[0]}.{current[1]} is newer than the versions the pinned "
            f"TensorFlow release supports. Create a virtual environment with "
            f"Python {config.RECOMMENDED_PYTHON}."
        )
    return None


def check_python_version(strict: bool = True) -> None:
    """Raise DependencyError when Python is unsupported (``strict``)."""
    message = python_version_message()
    if message and strict:
        raise DependencyError(message)


_PIP_NAMES = {
    "tensorflow": "tensorflow==2.16.2",
    "pretty_midi": "pretty_midi==0.2.10",
    "streamlit": "streamlit",
    "matplotlib": "matplotlib",
    "pandas": "pandas",
    "numpy": "numpy",
}


def module_available(import_name: str) -> bool:
    """True if a package can be found, without importing it (fast)."""
    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ValueError):
        return False


def require_module(import_name: str) -> ModuleType:
    """Import a package or raise a DependencyError with the exact fix."""
    if import_name == "tensorflow":
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        return importlib.import_module(import_name)
    except ImportError as exc:
        pip_name = _PIP_NAMES.get(import_name, import_name)
        raise DependencyError(
            f"The package '{import_name}' is not installed or failed to load "
            f"({exc}). Activate your virtual environment and run: "
            f"pip install -r requirements.txt   (or: pip install {pip_name})"
        ) from exc


# --------------------------------------------------------------------------- #
# Files and names
# --------------------------------------------------------------------------- #


def slugify(text: str, max_length: int = 40, default: str = "general") -> str:
    """Make a safe lowercase folder/file name ('Jazz Piano!' -> 'jazz-piano')."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).strip().lower()).strip("-")
    slug = slug[:max_length].strip("-")
    return slug or default


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_writable_dir(folder: Path) -> Path:
    """Create ``folder`` and prove we can write to it, or raise a clear error."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".echo-write-test-"):
            pass
    except OSError as exc:
        raise EchoError(
            f"Cannot write to the folder '{folder}' ({exc.strerror or exc}). "
            "Check that it is not read-only, not inside a protected location, and "
            "that antivirus or OneDrive is not locking it."
        ) from exc
    return folder


def unique_path(folder: Path, stem: str, suffix: str) -> Path:
    """Return a path in ``folder`` that does not exist yet (never overwrites)."""
    candidate = folder / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = folder / f"{stem}-{counter}{suffix}"
        counter += 1
    return candidate


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"
