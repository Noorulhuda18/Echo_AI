"""Audio rendering. Real rendering is tested only when FluidSynth and a SoundFont exist;
everything else (detection, error messages, failure handling) always runs."""

from __future__ import annotations

import struct
import wave

import pytest

import config
from src import audio_renderer as ar
from src.utils import AudioRenderError


def _write_wav(path, seconds=0.2, rate=8000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return path


def _status(**kw):
    base = dict(fluidsynth=None, soundfont=None, ffmpeg=None)
    base.update(kw)
    return ar.AudioStatus(**base)


# ----------------------------------------------------------------------------
# Detection and messages
# ----------------------------------------------------------------------------
def test_status_reports_what_is_missing():
    status = _status()
    assert not status.can_render and not status.can_make_mp3
    assert status.problems() == ["FluidSynth was not found.", "No SoundFont (.sf2/.sf3) was found."]
    assert "github.com/FluidSynth" in status.summary() and "assets/soundfonts" in status.summary()
    only_fs = _status(fluidsynth="fs")
    assert only_fs.problems() == ["No SoundFont (.sf2/.sf3) was found."]


def test_find_soundfont_in_folder_and_by_env(tmp_path, monkeypatch):
    monkeypatch.delenv(config.ENV_SOUNDFONT, raising=False)
    assert ar.find_soundfont(tmp_path) is None
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "b.sf2").write_bytes(b"x")
    (tmp_path / "a.SF3").write_bytes(b"x")
    assert ar.find_soundfont(tmp_path).name == "a.SF3"
    custom = tmp_path / "custom.sf2"
    custom.write_bytes(b"x")
    monkeypatch.setenv(config.ENV_SOUNDFONT, str(custom))
    assert ar.find_soundfont(tmp_path) == custom


def test_find_fluidsynth_respects_env_override(tmp_path, monkeypatch):
    exe = tmp_path / "fluidsynth.exe"
    exe.write_bytes(b"x")
    monkeypatch.setenv(config.ENV_FLUIDSYNTH, str(exe))
    assert ar.find_fluidsynth() == exe
    monkeypatch.setenv(config.ENV_FLUIDSYNTH, str(tmp_path / "nope.exe"))
    monkeypatch.setattr(ar.shutil, "which", lambda _: None)
    assert ar.find_fluidsynth() is None or ar.find_fluidsynth().name == "fluidsynth.exe"


def test_wav_duration_and_invalid_wav(tmp_path):
    assert ar.wav_duration_seconds(_write_wav(tmp_path / "ok.wav", 0.5, 8000)) == pytest.approx(0.5)
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"this is not audio")
    with pytest.raises(AudioRenderError, match="not a valid WAV"):
        ar.wav_duration_seconds(bad)
    with pytest.raises(AudioRenderError, match="no audio"):
        ar.wav_duration_seconds(_write_wav(tmp_path / "empty.wav", 0.0))


# ----------------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------------
def test_render_without_synth_gives_actionable_error(tmp_path):
    midi = tmp_path / "a.mid"
    midi.write_bytes(b"MThd")
    with pytest.raises(AudioRenderError, match="FluidSynth was not found"):
        ar.render_wav(midi, tmp_path / "audio", status=_status())
    assert not (tmp_path / "audio").exists() or not list((tmp_path / "audio").glob("*.wav"))


def test_render_missing_midi(tmp_path):
    with pytest.raises(AudioRenderError, match="does not exist"):
        ar.render_wav(tmp_path / "nope.mid", tmp_path, status=_status())


def test_render_cleans_up_when_synth_fails(tmp_path):
    fake_synth = tmp_path / "fake_synth.py"
    fake_synth.write_text("import sys; sys.stderr.write('boom: bad soundfont'); sys.exit(3)")
    import sys

    wrapper = tmp_path / ("synth.bat" if sys.platform == "win32" else "synth.sh")
    if sys.platform == "win32":
        wrapper.write_text(f'@"{sys.executable}" "{fake_synth}" %*')
    else:
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake_synth}" "$@"\n')
        wrapper.chmod(0o755)
    sf = tmp_path / "x.sf2"
    sf.write_bytes(b"x")
    midi = tmp_path / "a.mid"
    midi.write_bytes(b"MThd")
    with pytest.raises(AudioRenderError, match=r"(?s)exit code 3.*boom"):  # message spans two lines
        ar.render_wav(midi, tmp_path / "audio", status=_status(fluidsynth=wrapper, soundfont=sf))
    assert list((tmp_path / "audio").glob("*.wav")) == []


def test_render_rejects_synth_output_that_is_not_audio(tmp_path):
    """A synth that exits 0 but writes junk must not be reported as success."""
    import sys

    fake_synth = tmp_path / "junk_synth.py"
    fake_synth.write_text(
        "import sys\nout = sys.argv[sys.argv.index('-F') + 1]\nopen(out, 'wb').write(b'not a wav')\n"
    )
    wrapper = tmp_path / ("junk.bat" if sys.platform == "win32" else "junk.sh")
    if sys.platform == "win32":
        wrapper.write_text(f'@"{sys.executable}" "{fake_synth}" %*')
    else:
        wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{fake_synth}" "$@"\n')
        wrapper.chmod(0o755)
    sf = tmp_path / "x.sf2"
    sf.write_bytes(b"x")
    midi = tmp_path / "a.mid"
    midi.write_bytes(b"MThd")
    with pytest.raises(AudioRenderError, match="not a valid WAV"):
        ar.render_wav(midi, tmp_path / "audio", status=_status(fluidsynth=wrapper, soundfont=sf))
    assert list((tmp_path / "audio").glob("*.wav")) == []


def test_mp3_requires_ffmpeg(tmp_path):
    wav = _write_wav(tmp_path / "a.wav")
    with pytest.raises(AudioRenderError, match="needs ffmpeg"):
        ar.wav_to_mp3(wav, status=_status())


def test_mp3_conversion_when_ffmpeg_available(tmp_path):
    status = ar.check_audio_status(tmp_path)
    if not status.can_make_mp3:
        pytest.skip("ffmpeg not installed")
    wav = _write_wav(tmp_path / "a.wav", 0.5, 22050)
    try:
        mp3 = ar.wav_to_mp3(wav, status)
    except AudioRenderError as exc:
        if "libmp3lame" in str(exc) or "Unknown encoder" in str(exc):
            pytest.skip("this ffmpeg build has no MP3 encoder")
        raise
    assert mp3.suffix == ".mp3" and mp3.stat().st_size > 0


# ----------------------------------------------------------------------------
# Real rendering (skipped unless FluidSynth AND a SoundFont are installed)
# ----------------------------------------------------------------------------
def test_real_render_produces_playable_wav(tmp_path):
    pytest.importorskip("pretty_midi")
    status = ar.check_audio_status()
    if not status.can_render:
        pytest.skip("FluidSynth and/or a SoundFont are not installed")
    from src.midi_export import write_midi

    events = [(60, 3), (62, 3), (64, 3), (65, 5)]
    midi = write_midi(events, 120, tmp_path / "midi")
    wav = ar.render_wav(midi, tmp_path / "audio", status)
    assert wav.suffix == ".wav" and ar.wav_duration_seconds(wav) > 1.0
