"""MIDI -> WAV (-> optional MP3) using a locally installed FluidSynth.

A MIDI file contains instructions ("play C4 for half a beat"), not sound.
To hear it, a synthesizer (FluidSynth) plays those instructions using a
SoundFont (.sf2/.sf3 file of instrument samples) and records the result as a
WAV file. This module finds those tools, runs them and checks the output.

Nothing here is uploaded anywhere and no paid service is used.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import config
from src.utils import AudioRenderError, ensure_writable_dir, get_logger, unique_path

log = get_logger("audio")

SOUNDFONT_SUFFIXES = (".sf2", ".sf3")

WINDOWS_FLUIDSYNTH_HINTS = (
    r"C:\tools\fluidsynth\bin\fluidsynth.exe",
    r"C:\fluidsynth\bin\fluidsynth.exe",
    r"C:\Program Files\fluidsynth\bin\fluidsynth.exe",
    r"C:\ProgramData\chocolatey\bin\fluidsynth.exe",
)

INSTALL_HELP = (
    "To enable audio playback:\n"
    "1. Download FluidSynth for Windows from https://github.com/FluidSynth/fluidsynth/releases "
    "(choose the x64 .zip), unzip it, and either add its 'bin' folder to your PATH or set the "
    f"environment variable {config.ENV_FLUIDSYNTH} to the full path of fluidsynth.exe.\n"
    "2. Download a free General MIDI SoundFont (for example GeneralUser GS or FluidR3 GM - "
    "check each licence on its download page) and put the .sf2 file in "
    f"'assets/soundfonts/' (or set {config.ENV_SOUNDFONT} to its full path).\n"
    "3. Restart the app. Your MIDI download keeps working without these."
)


@dataclass
class AudioStatus:
    fluidsynth: Path | None
    soundfont: Path | None
    ffmpeg: Path | None

    @property
    def can_render(self) -> bool:
        return self.fluidsynth is not None and self.soundfont is not None

    @property
    def can_make_mp3(self) -> bool:
        return self.ffmpeg is not None

    def problems(self) -> list[str]:
        issues = []
        if self.fluidsynth is None:
            issues.append("FluidSynth was not found.")
        if self.soundfont is None:
            issues.append("No SoundFont (.sf2/.sf3) was found.")
        return issues

    def summary(self) -> str:
        if self.can_render:
            return "Audio rendering is ready."
        return " ".join(self.problems()) + "\n\n" + INSTALL_HELP


def find_fluidsynth() -> Path | None:
    override = os.environ.get(config.ENV_FLUIDSYNTH)
    if override and Path(override).is_file():
        return Path(override)
    found = shutil.which("fluidsynth")
    if found:
        return Path(found)
    for hint in WINDOWS_FLUIDSYNTH_HINTS:
        if Path(hint).is_file():
            return Path(hint)
    return None


def find_soundfont(soundfont_dir: Path = config.DEFAULT_PATHS.soundfonts) -> Path | None:
    override = os.environ.get(config.ENV_SOUNDFONT)
    if override and Path(override).is_file():
        return Path(override)
    if soundfont_dir.is_dir():
        candidates = sorted(
            p for p in soundfont_dir.iterdir() if p.is_file() and p.suffix.lower() in SOUNDFONT_SUFFIXES
        )
        if candidates:
            return candidates[0]
    return None


def find_ffmpeg() -> Path | None:
    found = shutil.which("ffmpeg")
    return Path(found) if found else None


def check_audio_status(soundfont_dir: Path = config.DEFAULT_PATHS.soundfonts) -> AudioStatus:
    return AudioStatus(find_fluidsynth(), find_soundfont(soundfont_dir), find_ffmpeg())


def wav_duration_seconds(path: Path) -> float:
    """Length of a WAV file; raises AudioRenderError if it is not valid audio."""
    try:
        with wave.open(str(path), "rb") as handle:
            frames, rate = handle.getnframes(), handle.getframerate()
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioRenderError(f"'{Path(path).name}' is not a valid WAV file ({exc}).") from exc
    if frames <= 0 or rate <= 0:
        raise AudioRenderError("The rendered WAV file contains no audio.")
    return frames / float(rate)


def render_wav(
    midi_path: Path,
    out_dir: Path = config.DEFAULT_PATHS.audio_out,
    status: AudioStatus | None = None,
    sample_rate: int = config.SAMPLE_RATE,
    timeout: int = config.RENDER_TIMEOUT_SECONDS,
) -> Path:
    """Render a MIDI file into a WAV file next to the other generated audio."""
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise AudioRenderError(f"The MIDI file '{midi_path}' does not exist.")
    status = status or check_audio_status()
    if not status.can_render:
        raise AudioRenderError(status.summary())

    folder = ensure_writable_dir(Path(out_dir))
    wav_path = unique_path(folder, midi_path.stem, ".wav")
    command = [
        str(status.fluidsynth),
        "-ni",  # no shell, no MIDI driver: just render
        "-g", "1.0",
        "-r", str(sample_rate),
        "-T", "wav",
        "-F", str(wav_path),
        str(status.soundfont),
        str(midi_path),
    ]
    log.info("Rendering: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except subprocess.TimeoutExpired as exc:
        wav_path.unlink(missing_ok=True)
        raise AudioRenderError(f"FluidSynth did not finish within {timeout} seconds.") from exc
    except OSError as exc:
        raise AudioRenderError(
            f"FluidSynth could not be started ({exc}). Check the path: {status.fluidsynth}"
        ) from exc

    if completed.returncode != 0 or not wav_path.is_file():
        wav_path.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "").strip()[-400:]
        log.error("FluidSynth failed (%s): %s", completed.returncode, detail)
        raise AudioRenderError(
            f"FluidSynth failed (exit code {completed.returncode}). The SoundFont may be "
            f"corrupt or unsupported.\nDetails: {detail or 'no output'}"
        )
    try:
        wav_duration_seconds(wav_path)
    except AudioRenderError:
        wav_path.unlink(missing_ok=True)
        raise
    return wav_path


def wav_to_mp3(wav_path: Path, status: AudioStatus | None = None) -> Path:
    """Optional MP3 export using a locally installed ffmpeg."""
    wav_path = Path(wav_path)
    status = status or check_audio_status()
    if not status.can_make_mp3:
        raise AudioRenderError(
            "MP3 export needs ffmpeg, which was not found. Install it (for example "
            "'winget install Gyan.FFmpeg') and restart the app, or use the WAV download."
        )
    if not wav_path.is_file():
        raise AudioRenderError(f"The WAV file '{wav_path}' does not exist.")
    mp3_path = unique_path(wav_path.parent, wav_path.stem, ".mp3")
    command = [
        str(status.ffmpeg), "-y", "-loglevel", "error",
        "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-q:a", "2",
        str(mp3_path),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
    except (subprocess.TimeoutExpired, OSError) as exc:
        mp3_path.unlink(missing_ok=True)
        raise AudioRenderError(f"MP3 conversion failed: {exc}") from exc
    if completed.returncode != 0 or not mp3_path.is_file() or mp3_path.stat().st_size == 0:
        mp3_path.unlink(missing_ok=True)
        raise AudioRenderError(
            f"MP3 conversion failed: {(completed.stderr or '').strip()[-300:] or 'unknown error'}"
        )
    return mp3_path
