Demo Link :https://echoai-2rqdtu4fehtsjetazmdjog.streamlit.app/



# ECHO — Intelligent Music Generation Studio

> *Every note leaves an echo.*

ECHO is a local, beginner-friendly **AI music generation system**. An **LSTM neural network** (TensorFlow/Keras) learns the patterns in MIDI music that *you* supply, then composes new melodies one note at a time. You can train it, generate music, download the MIDI, and — with a free synthesizer installed — listen to a WAV recording, all from a Streamlit studio interface.

It runs entirely on your own computer, on the CPU. No GPU, no cloud, no paid API.

**Honest status.** ECHO ships with **no dataset and no pretrained model**. You add MIDI files, then train. The app never pretends a model is trained until training has really finished.

---

## Contents

1. [Problem and objectives](#problem-and-objectives)
2. [How it works](#how-it-works)
3. [Quick start (Windows + VS Code)](#quick-start-windows--vs-code)
4. [Dataset setup and licences](#dataset-setup-and-licences)
5. [Training](#training)
6. [Generating music, MIDI and audio](#generating-music-midi-and-audio)
7. [The interface](#the-interface)
8. [Project structure](#project-structure)
9. [Testing](#testing)
10. [Troubleshooting](#troubleshooting)
11. [Known limitations](#known-limitations)
12. [Future improvements](#future-improvements)
13. [Publish to GitHub](#publish-to-github)
14. [Acknowledgments](#acknowledgments)

---

## Problem and objectives

**Problem.** Composing music is hard, and most "AI music" demos hide how they work. This project shows a *complete, understandable* deep-learning pipeline for music, from raw MIDI files to audio you can play.

**Objectives**

- Read MIDI files and turn them into sequences a neural network can learn from.
- Train an LSTM to predict the next musical event.
- Generate new music from the trained model and export valid MIDI.
- Render MIDI into WAV audio with a local synthesizer, and play it in the browser.
- Keep everything inspectable, testable and honest about failures.

## How it works

```
MIDI files ─▶ preprocess ─▶ events + vocabulary ─▶ LSTM training ─▶ saved model
                                                                        │
 Streamlit player ◀─ WAV ◀─ FluidSynth ◀─ MIDI file ◀─ sampling ◀───────┘
```

**Musical events.** Each note becomes an *event*: `(pitch, length)`. Pitch is a MIDI note number (21–108) or a *rest*. Length is quantised to one of 8 values measured in **beats** (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4), so the training data does not depend on tempo — you choose the tempo when you generate. When several notes start together, ECHO keeps only the highest one (a "skyline"), producing one melodic line per piece.

**The LSTM.** An *LSTM* (Long Short-Term Memory) is a recurrent network that reads a sequence in order and keeps a memory of what came before.

1. **Input** — the last `seq_len` events (default 32), each an integer id.
2. **Embedding** — each id becomes a small learned vector.
3. **LSTM layer(s)** — 1 or 2 layers read the sequence and build a memory of the melody.
4. **Dropout** — randomly switches off units while training so the model does not just memorise songs.
5. **Dense + softmax** — a probability for *every* possible next event.

Training uses sparse categorical cross-entropy loss and the Adam optimiser, with early stopping and checkpointing. Training and validation loss are reported every epoch.

**Generation.** Start from a real opening phrase from the training music, ask the model for next-event probabilities, apply **creativity** (sampling temperature: lower = safer, higher = bolder) and nucleus (top-p) filtering, sample one event, slide the window forward, and repeat until the requested length. The result is validated before export. A hard event limit means generation can never loop forever.

**Genres.** One model is trained **per genre folder** (`models/<genre>/`). A single model is *not* claimed to switch style on demand. ECHO never guesses a genre from a file name — the folder you place files in is the label you confirm.

---

## Quick start (Windows + VS Code)

All commands are for **PowerShell** (VS Code: *Terminal → New Terminal*), run from the project folder.

**1. Check Python.** Use **3.11 or 3.12**, 64-bit. Python 3.13+ is not covered by the pinned TensorFlow.

```powershell
python --version
py -0p            # lists every installed Python, if you have several
```

If needed, install Python 3.12 from <https://www.python.org/downloads/> and tick *Add python.exe to PATH*.

**2. Create and activate a virtual environment.**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell refuses to run the script: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then try again. In VS Code, choose the `.venv` interpreter (*Ctrl+Shift+P → Python: Select Interpreter*).

**3. Install dependencies.**

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> The pinned versions in `requirements.txt` are the combination this project targets. They could **not** be test-installed in the environment where the code was written (the package index was unreachable there). If `pip` reports a conflict, see [Troubleshooting](#troubleshooting) and tell me which package.

**4. Add a dataset** — see [Dataset setup](#dataset-setup-and-licences).

**5. Verify the pipeline in about a minute** (uses throw-away synthetic MIDI in a temp folder; touches none of your files):

```powershell
python scripts/smoke_test.py
```

**6. Train** (a small first run, then the full default):

```powershell
python scripts/train_model.py --genre classical --quick
python scripts/train_model.py --genre classical
```

**7. Launch the app.**

```powershell
streamlit run app.py
```

**8. Generate.** Open **Generate Music**, pick your model, press **Generate Music**, then download the MIDI. **9. Listen** — set up audio first ([below](#audio-setup-optional-but-recommended)).

---

## Dataset setup and licences

ECHO **does not bundle any dataset**. `data/raw/` is empty until you fill it. The folder name is the genre:

```
data/raw/classical/*.mid     → genre "classical"
data/raw/jazz/*.mid          → genre "jazz"
data/raw/*.mid               → genre "general" (unlabelled)
```

Three ways to add music:

**A. Your own MIDI files** (you are responsible for having the right to use them)

```powershell
python scripts/setup_dataset.py import --from "C:\Music\MyMidi" --genre jazz --confirm-genre
```

`--confirm-genre` is required: it states that *every* file in that folder really belongs to the genre. Or use the **Upload MIDI files** panel on the Train Model page, or copy files into `data/raw/<genre>/` yourself.

**B. MAESTRO — classical solo piano (automated, runs on your machine)**

```powershell
python scripts/setup_dataset.py maestro --max-files 20 --yes
```

Downloads the official MIDI-only archive (about 56 MB) from Google's servers and extracts an evenly spread sample into `data/raw/classical/`, with an `ATTRIBUTION.md`. `--yes` means you accept the licence.

**C. Weimar Jazz Database — jazz solos (manual download)**

Download the MIDI ZIP of the 456 solo transcriptions from <https://jazzomat.hfm-weimar.de/download/download.html>, unzip it, then use the `import` command from option A with `--genre jazz --confirm-genre`. These MIDI files are *unquantised*; ECHO snaps them to a 16th-note grid.

| Source | Content | Licence | Attribution |
|---|---|---|---|
| [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) | Classical piano performances | CC BY-NC-SA 4.0 (**non-commercial**, share-alike) | Hawthorne et al., *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset*, ICLR 2019 |
| [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/download/download.html) | 456 jazz solo transcriptions | Open Database License (ODbL) v1.0 | Credit the Jazzomat Research Project, HfM Weimar |
| Your own files | Anything | Your responsibility | — |

Read each licence yourself before using or sharing the data. **Do not commit datasets to GitHub** — `.gitignore` already excludes `data/raw/`.

Check what ECHO sees:

```powershell
python scripts/setup_dataset.py status
```

**Preprocessing handles** empty folders, corrupt or truncated MIDI files, non-MIDI files, byte-identical duplicates, drum-only files, and files with too few notes. Each is *counted and reported* (discovered / processed / skipped, with reasons) rather than crashing. Processed data is cached in `data/processed/` and rebuilt automatically when the files or settings change.

## Training

Training is always an **explicit action**; opening the app never trains anything.

**Command line**

```powershell
python scripts/train_model.py --list                     # genre folders found
python scripts/train_model.py --genre classical --quick  # tiny model, 8 epochs: first test
python scripts/train_model.py --genre all --epochs 20 --units 128
```

**In the app:** *Train Model* → choose genre → settings (presets *Quick first run*, *Balanced*, *Custom*) → *Validate and preprocess* (optional) → *Start training*. A live loss chart updates each epoch.

**CPU-friendly defaults** (`config.ModelConfig`): sequence length 32, embedding 48, 2 LSTM layers × 128 units, dropout 0.3, batch 128, up to 20 epochs, early-stopping patience 4. The *Quick* preset is 1 layer × 64 units, 8 epochs. How long training takes, and how good the music sounds, depend on your CPU and on how much and how consistent your data is. Start with `--quick`.

**What is saved** in `models/<genre>/`: `model.keras`, `vocab.json`, `seeds.json`, `history.json`, and `metadata.json` (configuration, dataset fingerprint, split strategy, metrics). `metadata.json` with `"status": "completed"` is written **last**, so a crashed or interrupted run can never appear as a trained model. A previous model is replaced only after the new one is complete.

**Refused honestly:** empty datasets, too little music (fewer than 200 training or 20 validation windows), invalid settings, and NaN losses all stop with a clear message and save nothing.

**Validation split.** With 5+ pieces, whole pieces go to validation (measures generalisation to unseen music). With fewer, the end of each piece is held out — separate arrays, so no event is shared between train and validation.

## Generating music, MIDI and audio

**Settings:** model (genre), instrument (playback only), approximate length in seconds, tempo (BPM), creativity (temperature), optional fixed random seed — same seed + same settings recreates the same melody.

**MIDI export.** Events are converted to correct pitches, durations and note-on/note-off pairs at your chosen tempo, written to `outputs/midi/` with a meaningful name (`echo_<model>_<bpm>bpm_<timestamp>_s<seed>.mid`), then **re-opened and verified**. Files are never overwritten. Every result has a download button.

**A MIDI file is not audio.** It contains instructions ("play C4 for half a beat"), not sound. Audio needs a synthesizer plus a SoundFont.

### Audio setup (optional but recommended)

If these are missing, the app says what is missing and **the MIDI download still works**.

1. **FluidSynth.** Download the Windows x64 `.zip` from <https://github.com/FluidSynth/fluidsynth/releases>, unzip it (for example to `C:\tools\fluidsynth`), and either add its `bin` folder to your `PATH`, or set the path for the current session:
   ```powershell
   $env:ECHO_FLUIDSYNTH = "C:\tools\fluidsynth\bin\fluidsynth.exe"
   ```
   (Chocolatey users: `choco install fluidsynth`.)
2. **A General MIDI SoundFont** (`.sf2` / `.sf3`), for example *GeneralUser GS* or *FluidR3 GM* — check each licence on its download page. Put the file in `assets/soundfonts/`, or set `$env:ECHO_SOUNDFONT = "C:\path\to\font.sf2"`.
3. Restart the app. The sidebar badge changes to **Audio ready**.

ECHO then renders a **WAV** (`outputs/audio/`), plays it in the page, and offers a download. **MP3** is optional and needs a local `ffmpeg` (`winget install Gyan.FFmpeg`); no online converter is ever used.

## The interface

A midnight-plum studio with blush-to-lavender gradients, glass-like cards, a restrained waveform motif, and cherry red reserved for primary actions and errors. Brand type: italic display serif (Cormorant Garamond) for the wordmark, Manrope for everything else — loaded from Google Fonts when you are online, with safe fallbacks offline.

| Page | Purpose |
|---|---|
| **Home** | Introduction, the five-step process, live studio status (files, models, audio, Python) |
| **Generate Music** | Controls, progress, piano-roll preview, audio player, MIDI/WAV/MP3 downloads |
| **Train Model** | Upload or pick data, validate, configure, train with live loss chart, saved models |
| **Dataset Explorer** | File counts, skipped files with reasons, pitch and duration charts, source licences |
| **About Project** | Architecture, technologies, workflow, limitations, future ideas |

Design rules verified in `tests/test_ui_theme.py`: every text/background pair meets WCAG AA (4.5:1) and chart marks meet 3:1 on the card colour; motion is switched off for visitors who ask for reduced motion; all dynamic text is HTML-escaped.

Two interfaces are included: `streamlit run app.py` (the detailed studio: system-status sidebar, active-model card, KPI tiles, getting-started checklist, tabbed results) and `streamlit run app_classic.py` (the simpler original). Both use the same code underneath.

Palette: blush `#F3A6B8` → lavender `#B7A0E8` → deep rose `#C93655`; midnight plum `#241728`; cards `#34263A`; text `#FFF7FB`; secondary text `#C8B9D5`.

**Screenshots** — *placeholders; to be added after your first local run:*

- `docs/screenshots/home.png` — *(not yet captured)*
- `docs/screenshots/generate.png` — *(not yet captured)*
- `docs/screenshots/train.png` — *(not yet captured)*
- `docs/screenshots/dataset-explorer.png` — *(not yet captured)*

## Project structure

```
ECHO/
├─ app.py                     Streamlit interface (5 pages): sidebar status panel, KPIs, checklist
├─ app_classic.py             The simpler original interface (run it the same way)
├─ config.py                  Branding, paths, event representation, limits, ModelConfig
├─ requirements.txt
├─ .streamlit/config.toml     Dark base theme and brand colours
├─ src/
│  ├─ dataset.py              Scan folders, duplicates, uploads, dataset sources
│  ├─ preprocessing.py        MIDI → events, vocabulary, split, windows, reports
│  ├─ model.py                The Keras LSTM
│  ├─ train.py                Training workflow, checkpoints, atomic save, metadata
│  ├─ generate.py             Load model, sampling, validation, MIDI result
│  ├─ midi_export.py          Events → verified MIDI file
│  ├─ audio_renderer.py       FluidSynth → WAV, optional ffmpeg → MP3
│  ├─ ui_theme.py             ECHO CSS, HTML components, themed charts
│  └─ utils.py                Error types, logging, dependency checks, safe files
├─ scripts/
│  ├─ setup_dataset.py        sources / status / import / maestro
│  ├─ train_model.py          Command-line training
│  └─ smoke_test.py           One-minute end-to-end check
├─ tests/                     pytest suite (temporary folders only)
├─ data/raw  data/processed   Your MIDI files · cached processed data
├─ models/                    Trained models (one folder per genre)
├─ outputs/midi  outputs/audio  Generated files
├─ assets/soundfonts/         Put a .sf2 / .sf3 here
└─ logs/echo.log              Detailed diagnostics
```

## Testing

```powershell
pytest                       # whole suite
python scripts/smoke_test.py # end-to-end on synthetic data, about a minute
```

All tests run in temporary folders and never touch your real data, models or outputs. Tests that need TensorFlow, pretty_midi, FluidSynth or ffmpeg **skip themselves** when those are absent and say why.

**What has actually been run.** In the environment where this code was written, `tensorflow`, `pretty_midi`, `streamlit` and FluidSynth could not be installed (no package-index access). Results there, under real `pytest` 9.1:

> **66 passed, 13 skipped, 0 failed**

The 13 skipped tests are exactly those that need TensorFlow (LSTM construction, training, generation round-trip) or pretty_midi (reading and writing real MIDI files, real audio rendering). Therefore **not executed by the author:** LSTM training and generation, MIDI file reading/writing, real WAV rendering, and the live Streamlit app. Static analysis (`ruff`) reported only style notes (`zip()` without `strict=`) and no errors. The theme was checked by rendering its real CSS and HTML in headless Chromium at 1366 px and 390 px wide (no horizontal scrolling), which also caught and fixed layout bugs and dark-on-dark chart titles. Every page of `app.py` was also executed against a stand-in `streamlit` module (empty and populated studio, with and without a generated result) with no exceptions; that checks the page logic and the HTML, not Streamlit's own widgets. **Please run `pytest` and `python scripts/smoke_test.py` on your machine** — together they cover everything that was skipped — and open an issue if anything fails.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `Activate.ps1 cannot be loaded` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `No matching distribution found for tensorflow` | Wrong Python. Use 3.11 or 3.12, 64-bit; recreate the venv (`py -3.12 -m venv .venv`). |
| `DLL load failed` / TensorFlow import error on Windows | Install the *Microsoft Visual C++ Redistributable 2015–2022 (x64)*, then restart the terminal. |
| `No module named 'pkg_resources'` | `pip install "setuptools<81"` (needed by `pretty_midi` 0.2.10; already in `requirements.txt`). |
| pip dependency conflict (numpy / protobuf / keras) | Use a fresh venv and install only from `requirements.txt`; do not mix with other packages. |
| "No MIDI files found" | Put `.mid` / `.midi` files in `data/raw/<genre>/`; run `python scripts/setup_dataset.py status`. |
| "Not enough music to train" | Add more files, or lower *Sequence length* / *Window stride* on the Train page. |
| "Model … could not be loaded / does not match" | It was saved by another TensorFlow version or encoding. Retrain it. |
| Audio badge says *Audio setup* | Install FluidSynth and a SoundFont — see [Audio setup](#audio-setup-optional-but-recommended). |
| "Cannot write to the folder" | Folder is read-only, or OneDrive/antivirus is locking it. Move the project out of OneDrive. |
| Fonts look plain | You are offline; the layout still works with fallback fonts. |
| Anything else | Details are in `logs/echo.log`. |

## Known limitations

- **Melody only.** Chords and multiple voices are reduced to the top note.
- A small LSTM learns **local patterns, not full musical structure**; long pieces can wander. Output quality is not guaranteed and depends on dataset size and consistency.
- No dynamics: all notes use one velocity.
- Durations are quantised to 8 values; tuplets and ornaments are approximated.
- CPU training is slow on large datasets.
- One model per genre; there is no style-conditioned single model.
- Audio quality is set by the SoundFont you choose.
- Dependency pins are untested in the authoring environment (see [Testing](#testing)).

## Future improvements

- Multi-voice and chord-aware representations, velocity and dynamics.
- Transformer models and genre conditioning in a single model.
- Key/tempo detection with Music21 and transposition-based augmentation.
- Objective evaluation (pitch-class histograms, repetition metrics) and listening tests.
- Screenshots and a packaged Windows installer.

## Publish to GitHub

```powershell
git init
git add .
git status                    # check: no datasets, models, .venv or audio listed
git commit -m "Initial commit: ECHO music generation studio"
git branch -M main
```

Then create an empty repository at <https://github.com/new> (no README, no .gitignore) and:

```powershell
git remote add origin https://github.com/<your-username>/echo-music-studio.git
git push -u origin main
```

With the GitHub CLI instead: `gh repo create echo-music-studio --public --source . --push`.

`.gitignore` keeps datasets, trained models, generated MIDI/audio, SoundFonts, logs, `.venv` and `.env` out of Git; `.gitkeep` files preserve the empty folders. Add a `LICENSE` file of your choice (for example MIT) — and remember that MAESTRO is non-commercial/share-alike, so keep its `ATTRIBUTION.md` if you ever share anything derived from it.

## Acknowledgments

- **MAESTRO dataset** — Curtis Hawthorne, Andriy Stasyuk, Adam Roberts, Ian Simon, Cheng-Zhi Anna Huang, Sander Dieleman, Erich Elsen, Jesse Engel and Douglas Eck, *Enabling Factorized Piano Music Modeling and Generation with the MAESTRO Dataset*, ICLR 2019 (CC BY-NC-SA 4.0).
- **Weimar Jazz Database** — Jazzomat Research Project, Hochschule für Musik Franz Liszt Weimar (ODbL 1.0).
- Built with TensorFlow/Keras, Streamlit, pretty_midi, NumPy, pandas, Matplotlib, FluidSynth and pytest.
