"""ECHO - Intelligent Music Generation Studio (enhanced Streamlit interface).

Run with:  streamlit run app.py

Opening the app never trains anything. Training only starts when you press
"Start training" on the Train Model page. (The simpler original interface is
still available as app_classic.py.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import config  # noqa: E402

st.set_page_config(
    page_title=f"{config.APP_NAME} - {config.APP_SUBTITLE}",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np  # noqa: F401
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment problem
    st.error(
        f"A required package is missing: {exc}. Activate your virtual environment and run "
        "`pip install -r requirements.txt`, then restart the app."
    )
    st.stop()

from src import ui_theme as ui  # noqa: E402
from src.audio_renderer import (  # noqa: E402
    INSTALL_HELP,
    check_audio_status,
    render_wav,
    wav_duration_seconds,
    wav_to_mp3,
)
from src.dataset import DATASET_SOURCES, list_genres, save_uploaded_files  # noqa: E402
from src.generate import (  # noqa: E402
    ModelInfo,
    generate_music,
    list_trained_models,
    load_bundle,
    model_signature,
)
from src.preprocessing import (  # noqa: E402
    get_or_build_processed,
    load_report,
    processed_dir,
)
from src.train import HISTORY_FILE, train_model  # noqa: E402
from src.utils import (  # noqa: E402
    EchoError,
    get_logger,
    human_size,
    module_available,
    python_version_message,
    setup_logging,
)

PATHS = config.DEFAULT_PATHS
PATHS.ensure()
setup_logging(PATHS)
log = get_logger("app")

PAGES = ["Home", "Generate Music", "Train Model", "Dataset Explorer", "About Project"]
NAV_ICONS = {"Home": "⌂", "Generate Music": "♫", "Train Model": "⚙", "Dataset Explorer": "▤", "About Project": "ⓘ"}
HISTORY_LIMIT = 6

ui.inject_css(st)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def goto(page: str) -> None:
    """Button callback: switch page (runs before the next rerun)."""
    st.session_state["nav"] = page


@st.cache_resource(show_spinner=False, max_entries=3)
def _load_bundle_cached(name: str, signature: str):
    """Load a model once. ``signature`` changes after retraining, so a retrained
    model is never served from a stale cache entry."""
    return load_bundle(name, PATHS)


def get_bundle(name: str):
    return _load_bundle_cached(name, model_signature(name, PATHS))


def show_chart(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def show_error(exc: Exception, action: str) -> None:
    """Friendly message in the UI, full traceback in logs/echo.log."""
    if isinstance(exc, EchoError):
        st.error(str(exc))
        log.warning("%s failed: %s", action, exc)
    else:
        st.error(
            f"Something unexpected went wrong while {action}: {type(exc).__name__}: {exc}. "
            "Details were written to logs/echo.log."
        )
        log.exception("%s failed", action)


def md(html_text: str) -> None:
    st.markdown(html_text, unsafe_allow_html=True)


def fmt_date(iso: str) -> str:
    return iso.replace("T", " ")[:16] if iso else "-"


def latest_model(models: list[ModelInfo]) -> ModelInfo | None:
    return max(models, key=lambda m: m.trained_at) if models else None


def recent_midi(limit: int = 5) -> list[Path]:
    try:
        files = [p for p in PATHS.midi_out.glob("*.mid") if p.is_file()]
    except OSError:
        return []
    return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:limit]


def processed_event_total() -> int:
    """Musical events across every genre folder that has been preprocessed."""
    total = 0
    for genre in list_genres(PATHS.raw):
        report = load_report(processed_dir(PATHS, genre))
        if report:
            total += report.total_events
    return total


def genre_options() -> list[str]:
    genres = [g for g, n in list_genres(PATHS.raw).items() if n > 0]
    return [config.ALL_GENRES, *genres] if genres else []


def models_dataframe(models) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": m.name,
                "Trained (UTC)": fmt_date(m.trained_at),
                "Files": m.files_processed,
                "Vocabulary": m.vocab_size,
                "Best val. loss": round(m.best_val_loss, 4),
                "Epochs": m.metadata.get("metrics", {}).get("epochs_run", ""),
            }
            for m in models
        ]
    )


def model_panel(info: ModelInfo, title: str = "Selected model") -> str:
    cfg = info.metadata.get("config", {})
    metrics = info.metadata.get("metrics", {})
    ds = info.metadata.get("dataset", {})
    pairs = [
        ("Genre", info.genre),
        ("Trained (UTC)", fmt_date(info.trained_at)),
        ("Training files", str(info.files_processed)),
        ("Vocabulary", f"{info.vocab_size} events"),
        ("Best val. loss", f"{info.best_val_loss:.3f}"),
        ("Epochs run", str(metrics.get("epochs_run", "-"))),
        ("Network", f"{cfg.get('num_layers', '?')} x LSTM {cfg.get('lstm_units', '?')}"),
        ("Context", f"{cfg.get('seq_len', '?')} events"),
        ("Windows", f"{ds.get('train_windows', '?')} train / {ds.get('val_windows', '?')} val"),
    ]
    return ui.panel(title, ui.kv_grid(pairs), eyebrow="Trained LSTM")


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> str:
    with st.sidebar:
        md(ui.sidebar_brand())
        md(ui.sidebar_label("Menu"))
        st.radio(
            "Navigate", PAGES, key="nav", label_visibility="collapsed",
            format_func=lambda page: f"{NAV_ICONS[page]}   {page}",
        )

        audio = check_audio_status(PATHS.soundfonts)
        models = list_trained_models(PATHS)
        has_tf, has_midi = module_available("tensorflow"), module_available("pretty_midi")
        md(ui.sidebar_label("System"))
        md(
            ui.sidebar_status(
                [
                    ("TensorFlow", "ok" if has_tf else "bad", "Ready" if has_tf else "Missing"),
                    ("MIDI tools", "ok" if has_midi else "bad", "Ready" if has_midi else "Missing"),
                    ("Audio", "ok" if audio.can_render else "warn", "Ready" if audio.can_render else "Setup needed"),
                    (
                        "Models",
                        "ok" if models else "warn",
                        f"{len(models)} trained" if models else "None yet",
                    ),
                ]
            )
        )

        md(ui.sidebar_label("Active model"))
        newest = latest_model(models)
        if newest:
            md(
                ui.sidebar_model(
                    newest.genre,
                    f"{newest.files_processed} files · val loss {newest.best_val_loss:.3f} · "
                    f"trained {fmt_date(newest.trained_at)}",
                )
            )
            st.button("New composition", type="primary", on_click=goto, args=("Generate Music",),
                      use_container_width=True, key="sb_generate")
        else:
            md(ui.sidebar_model("No model yet", "Add MIDI files and train one to start composing."))
            st.button("Train a model", on_click=goto, args=("Train Model",),
                      use_container_width=True, key="sb_train")

        message = python_version_message()
        if message:
            st.warning(message)
        md(
            ui.sidebar_footer(
                "Diagnostics: logs/echo.log",
            )
        )
    return st.session_state["nav"]


# --------------------------------------------------------------------------- #
# Home
# --------------------------------------------------------------------------- #
def page_home() -> None:
    genres = list_genres(PATHS.raw)
    models = list_trained_models(PATHS)
    audio = check_audio_status(PATHS.soundfonts)
    total_files = sum(genres.values())
    events_total = processed_event_total()
    midi_files = recent_midi(50)

    md(ui.hero())
    a, b, _ = st.columns([1.3, 1.3, 4])
    a.button("Generate music", type="primary", on_click=goto, args=("Generate Music",), use_container_width=True, key="home_gen")
    b.button("Train a model", on_click=goto, args=("Train Model",), use_container_width=True, key="home_train")

    md(
        ui.kpis(
            [
                ui.kpi("MIDI files", str(total_files),
                       ", ".join(f"{g}: {n}" for g, n in genres.items()) if genres else "data/raw is empty", "♪"),
                ui.kpi("Musical events", f"{events_total:,}", "after preprocessing" if events_total else "not processed yet", "◈"),
                ui.kpi("Trained models", str(len(models)),
                       ", ".join(m.name for m in models) if models else "none yet", "⚙"),
                ui.kpi("Compositions", str(len(midi_files)) if len(midi_files) < 50 else "50+",
                       "saved in outputs/midi" if midi_files else "none generated yet", "♫"),
                ui.kpi("Audio", "Ready" if audio.can_render else "Not set up",
                       "FluidSynth + SoundFont found" if audio.can_render else "MIDI download still works", "◉"),
            ]
        )
    )

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        md(ui.section("Getting started", "Your progress, read from the files on this computer."))
        md(
            ui.checklist(
                [
                    (total_files > 0, "Add MIDI files", "Upload them or copy .mid files into data/raw/<genre>/."),
                    (events_total > 0, "Preprocess the dataset", "Turns notes into (pitch, length) events and a vocabulary."),
                    (bool(models), "Train the LSTM", "An explicit action; nothing trains when the app opens."),
                    (bool(midi_files), "Generate a composition", "Creates a verified MIDI file you can download."),
                    (audio.can_render, "Set up audio playback (optional)", "FluidSynth plus a SoundFont renders WAV."),
                ]
            )
        )
    with right:
        md(ui.section("Latest model"))
        newest = latest_model(models)
        if newest:
            md(model_panel(newest, newest.name))
        else:
            md(ui.empty_state("No trained model yet",
                              "Once you train one, its settings and validation loss appear here."))

    md(ui.section("How ECHO makes music", "Five steps, all running on your own computer."))
    md(
        ui.grid(
            [
                ui.step_card(1, "Collect", "Add MIDI files you are allowed to use. Folders name the genre."),
                ui.step_card(2, "Learn", "Notes become (pitch, length) events; an LSTM studies what usually follows what."),
                ui.step_card(3, "Compose", "The model predicts one event at a time. Creativity controls how adventurous it is."),
                ui.step_card(4, "Export", "The melody is written as a verified MIDI file you can download."),
                ui.step_card(5, "Listen", "FluidSynth turns the MIDI into a WAV recording for playback."),
            ]
        )
    )

    md(ui.section("Recent compositions", "The newest files in outputs/midi."))
    if not midi_files:
        md(ui.empty_state("Nothing composed yet", "Generated MIDI files will be listed here with a download button."))
    else:
        for path in midi_files[:5]:
            c1, c2 = st.columns([4, 1])
            c1.markdown(
                ui.file_row(path.name, f"{human_size(path.stat().st_size)} · MIDI"), unsafe_allow_html=True
            )
            c2.download_button("Download", path.read_bytes(), file_name=path.name, mime="audio/midi",
                               key=f"home_dl_{path.name}", use_container_width=True)

    md(ui.section("Pipeline"))
    md(ui.flow(["MIDI files", "Preprocess", "Events + vocabulary", "LSTM", "Sampling", "MIDI export", "WAV audio", "Player"]))


# --------------------------------------------------------------------------- #
# Generate
# --------------------------------------------------------------------------- #
def run_generation(model_name: str, instrument: str, duration: int, tempo: int, creativity: float, seed: int | None) -> None:
    progress = st.progress(0, text="Loading the trained model...")
    try:
        bundle = get_bundle(model_name)
        progress.progress(25, text="Composing with the LSTM...")
        result = generate_music(
            model_name, duration, tempo, creativity, seed=seed,
            program=config.INSTRUMENTS[instrument], paths=PATHS, bundle=bundle,
        )
        progress.progress(65, text="Rendering audio...")
        record = {
            "model": model_name,
            "instrument": instrument,
            "creativity": creativity,
            "requested": duration,
            "midi_path": str(result.midi_path),
            "wav_path": None,
            "mp3_path": None,
            "audio_error": None,
            "events": result.events,
            "tempo": result.tempo_bpm,
            "seed": result.seed,
            "seconds": result.duration_seconds,
            "notes": result.note_count,
        }
        audio = check_audio_status(PATHS.soundfonts)
        if audio.can_render:
            try:
                record["wav_path"] = str(render_wav(result.midi_path, PATHS.audio_out, audio))
            except EchoError as exc:  # keep the MIDI even if audio fails
                record["audio_error"] = str(exc)
                log.warning("Audio rendering failed: %s", exc)
        else:
            record["audio_error"] = audio.summary()
        progress.progress(100, text="Done")
        st.session_state["last_generation"] = record
        history = st.session_state.setdefault("history", [])
        history.insert(0, record)
        del history[HISTORY_LIMIT:]
    except Exception as exc:
        progress.empty()
        st.session_state.pop("last_generation", None)
        show_error(exc, "generating music")
    else:
        progress.empty()


def render_generation_result(record: dict) -> None:
    md(ui.section("Your composition"))
    rests = sum(1 for p, _ in record["events"] if p == config.REST_PITCH)
    md(
        ui.kpis(
            [
                ui.kpi("Notes", str(record["notes"]), f"{rests} rests", "♪"),
                ui.kpi("Length", f"{record['seconds']:.1f} s", f"asked for {record['requested']} s", "◷"),
                ui.kpi("Tempo", f"{record['tempo']} BPM", record["instrument"], "♩"),
                ui.kpi("Creativity", f"{record['creativity']:.2f}", "sampling temperature", "✧"),
                ui.kpi("Seed", str(record["seed"]), "reuse to recreate", "#"),
            ]
        )
    )

    midi_path = Path(record["midi_path"])
    wav_path = Path(record["wav_path"]) if record["wav_path"] else None
    tab_listen, tab_roll, tab_details = st.tabs(["Listen & download", "Piano roll", "Details"])

    with tab_listen:
        if wav_path and wav_path.is_file():
            try:
                length = wav_duration_seconds(wav_path)
                st.audio(wav_path.read_bytes(), format="audio/wav")
                st.caption(f"Rendered audio, {length:.1f} s, {human_size(wav_path.stat().st_size)}.")
            except EchoError as exc:
                st.error(str(exc))
        elif record["audio_error"]:
            st.warning("Audio playback is unavailable, but your MIDI file is ready to download.")
            with st.expander("How to enable audio playback"):
                text = record["audio_error"]
                st.text(text if "Download FluidSynth" in text else text + "\n\n" + INSTALL_HELP)

        d1, d2, d3 = st.columns(3)
        if midi_path.is_file():
            d1.download_button("Download MIDI", midi_path.read_bytes(), file_name=midi_path.name,
                               mime="audio/midi", use_container_width=True, key="dl_midi")
        else:
            d1.error("The MIDI file is no longer on disk.")
        if wav_path and wav_path.is_file():
            d2.download_button("Download WAV", wav_path.read_bytes(), file_name=wav_path.name,
                               mime="audio/wav", use_container_width=True, key="dl_wav")
            mp3_path = Path(record["mp3_path"]) if record["mp3_path"] else None
            if mp3_path and mp3_path.is_file():
                d3.download_button("Download MP3", mp3_path.read_bytes(), file_name=mp3_path.name,
                                   mime="audio/mpeg", use_container_width=True, key="dl_mp3")
            elif check_audio_status(PATHS.soundfonts).can_make_mp3:
                if d3.button("Create MP3", use_container_width=True, key="make_mp3"):
                    try:
                        record["mp3_path"] = str(wav_to_mp3(wav_path))
                        st.rerun()
                    except EchoError as exc:
                        st.error(str(exc))
            else:
                d3.caption("MP3 export is optional and needs ffmpeg.")
        st.caption(f"Saved on disk: {midi_path}")

    with tab_roll:
        show_chart(ui.piano_roll_figure(record["events"], record["tempo"]))
        st.caption("Each bar is a note: higher on the chart = higher pitch. Gaps are rests.")

    with tab_details:
        md(
            ui.kv_grid(
                [
                    ("Model", record["model"]),
                    ("Instrument", record["instrument"]),
                    ("Events", str(len(record["events"]))),
                    ("Notes / rests", f"{record['notes']} / {rests}"),
                    ("Seed", str(record["seed"])),
                    ("MIDI file", midi_path.name),
                    ("WAV file", wav_path.name if wav_path else "not rendered"),
                ]
            )
        )

    earlier = st.session_state.get("history", [])[1:]
    if earlier:
        with st.expander(f"Earlier in this session ({len(earlier)})"):
            for index, old in enumerate(earlier):
                path = Path(old["midi_path"])
                c1, c2 = st.columns([4, 1])
                c1.markdown(
                    ui.file_row(path.name, f"{old['notes']} notes · {old['seconds']:.0f} s · {old['tempo']} BPM · seed {old['seed']}"),
                    unsafe_allow_html=True,
                )
                if path.is_file():
                    c2.download_button("MIDI", path.read_bytes(), file_name=path.name, mime="audio/midi",
                                       key=f"hist_{index}_{path.name}", use_container_width=True)


def page_generate() -> None:
    models = list_trained_models(PATHS)
    audio = check_audio_status(PATHS.soundfonts)
    md(
        ui.page_header(
            "Studio", "Generate music", "Compose a new melody with a model you trained.",
            chips=[
                ui.pill(f"{len(models)} model{'s' if len(models) != 1 else ''}", "ok" if models else "warn"),
                ui.pill("Audio ready" if audio.can_render else "MIDI only", "ok" if audio.can_render else "warn"),
            ],
        )
    )
    if not models:
        md(ui.empty_state("No trained model yet", "There is nothing to generate from until you train a model."))
        st.button("Go to Train Model", type="primary", on_click=goto, args=("Train Model",))
        return

    names = [m.name for m in models]
    by_name = {m.name: m for m in models}
    labels = {m.name: f"{m.genre}  ·  {m.files_processed} files  ·  val loss {m.best_val_loss:.3f}" for m in models}

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        with st.container(border=True):
            md(ui.step_heading(1, "Choose the sound", "Which model composes, and how it will be played."))
            model_name = st.selectbox(
                "Genre / model", names, format_func=lambda n: labels[n],
                help="Each genre folder you trained gets its own model. 'all' blends every file.",
            )
            instrument = st.selectbox("Instrument", list(config.INSTRUMENTS), help="Only affects playback; the notes are the same.")
            md(ui.step_heading(2, "Shape the piece", "Length, speed and how adventurous the model should be."))
            c1, c2 = st.columns(2)
            with c1:
                duration = st.slider("Approximate length (seconds)", config.MIN_DURATION_SECONDS, 120, 30, step=5)
                tempo = st.slider("Tempo (BPM)", 50, 180, 100, help="Beats per minute. Chosen at export time, so one model works at any tempo.")
            with c2:
                creativity = st.slider(
                    "Creativity", 0.4, 1.5, 1.0, step=0.05,
                    help="Sampling temperature. Lower = predictable and repetitive; higher = more surprising and more mistakes.",
                )
                fix_seed = st.checkbox("Use a fixed random seed", help="Same seed + same settings = same melody.")
                seed_value = st.number_input("Seed", 0, 2**31 - 1, 42, disabled=not fix_seed)
    with right:
        md(model_panel(by_name[model_name]))

    if st.button("Generate Music", type="primary", use_container_width=True):
        run_generation(model_name, instrument, int(duration), int(tempo), float(creativity),
                       int(seed_value) if fix_seed else None)

    record = st.session_state.get("last_generation")
    if record:
        render_generation_result(record)
    else:
        md(ui.empty_state("Your composition will appear here",
                          "Press Generate Music. You will get a piano-roll preview, MIDI download and, with audio set up, playback."))


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def model_settings_form() -> config.ModelConfig:
    preset = st.radio(
        "Preset", ["Quick first run", "Balanced (default)", "Custom"], horizontal=True,
        help="Quick = tiny model that trains fast. Balanced = the standard laptop-friendly setup.",
    )
    base = config.QUICK_PRESET if preset.startswith("Quick") else config.ModelConfig()
    k = preset[:4]  # widget keys change with the preset so sliders reset to its values
    c1, c2, c3 = st.columns(3)
    seq_len = c1.slider("Sequence length", 8, 96, base.seq_len, key=f"sl{k}", help="How many past events the model sees.")
    units = c1.select_slider("LSTM units", [32, 64, 96, 128, 192, 256], base.lstm_units, key=f"un{k}")
    layers = c1.radio("LSTM layers", [1, 2], index=base.num_layers - 1, horizontal=True, key=f"ly{k}")
    embedding = c2.select_slider("Embedding size", [16, 32, 48, 64, 96], base.embedding_dim, key=f"em{k}")
    dropout = c2.slider("Dropout", 0.0, 0.6, float(base.dropout), step=0.05, key=f"dr{k}", help="Guards against memorising the songs.")
    lr = c2.select_slider("Learning rate", [0.0005, 0.001, 0.002, 0.003, 0.005], base.learning_rate, key=f"lr{k}")
    batch = c3.select_slider("Batch size", [32, 64, 128, 256, 512], base.batch_size, key=f"bs{k}")
    epochs = c3.slider("Max epochs", 1, 100, base.epochs, key=f"ep{k}", help="Early stopping may end training sooner.")
    stride = c3.slider("Window stride", 1, 16, min(base.stride, seq_len), key=f"st{k}", help="Higher = fewer, less overlapping training windows = faster.")
    patience = c3.slider("Early-stopping patience", 1, 15, base.patience, key=f"pa{k}")
    seed = st.number_input("Random seed", 0, 2**31 - 1, base.seed, key=f"se{k}")
    return config.ModelConfig(
        seq_len=int(seq_len), embedding_dim=int(embedding), lstm_units=int(units), num_layers=int(layers),
        dropout=float(dropout), learning_rate=float(lr), batch_size=int(batch), epochs=int(epochs),
        patience=int(patience), stride=int(min(stride, seq_len)), seed=int(seed),
        val_fraction=base.val_fraction, min_event_count=base.min_event_count,
    )


def render_report(report) -> None:
    md(
        ui.kpis(
            [
                ui.kpi("Discovered", str(report.discovered), "MIDI files found"),
                ui.kpi("Processed", str(report.processed), "usable"),
                ui.kpi("Skipped", str(report.skipped_count), "see reasons below" if report.skipped_count else "none"),
                ui.kpi("Duplicates", str(report.duplicates), "identical files ignored"),
                ui.kpi("Vocabulary", str(report.vocab_size), "distinct events"),
            ]
        )
    )
    if report.skipped:
        with st.expander(f"{report.skipped_count} skipped file(s) - reasons"):
            st.dataframe(pd.DataFrame(report.skipped), use_container_width=True, hide_index=True)
    if report.unsupported:
        st.caption("Ignored non-MIDI files: " + ", ".join(report.unsupported[:20]))


def page_train() -> None:
    models = list_trained_models(PATHS)
    genres = list_genres(PATHS.raw)
    md(
        ui.page_header(
            "Studio", "Train model", "Teach an LSTM the music in your dataset. Training starts only when you press the button.",
            chips=[
                ui.pill(f"{sum(genres.values())} MIDI files", "ok" if genres else "warn"),
                ui.pill(f"{len(models)} trained", "ok" if models else "warn"),
            ],
        )
    )

    flash = st.session_state.pop("upload_flash", None)
    if flash:
        st.success(flash)

    # 1. dataset ------------------------------------------------------------ #
    md(ui.step_heading(1, "Dataset", "Each folder inside data/raw is a genre label that YOU confirmed."))
    options = genre_options()
    genre = config.ALL_GENRES  # replaced by the selectbox below when MIDI files exist
    if not options:
        md(ui.empty_state("No MIDI files yet", "Upload some below, copy .mid files into data/raw/<genre>/, or follow the README dataset steps."))
    else:
        genre = st.selectbox(
            "Train on", options,
            format_func=lambda g: f"{g}  ({sum(genres.values())} files)" if g == config.ALL_GENRES else f"{g}  ({genres[g]} files)",
            help="Each folder inside data/raw is a genre label that YOU confirmed by placing files there.",
            key="train_genre",
        )
    with st.expander("Upload MIDI files"):
        files = st.file_uploader("MIDI files", type=["mid", "midi"], accept_multiple_files=True, key="uploader")
        genre_name = st.text_input("Genre / folder name", placeholder="e.g. classical or jazz")
        confirm = st.checkbox("I confirm these files belong to this genre and I am allowed to use them")
        if st.button("Add files to dataset", disabled=not (files and genre_name and confirm)):
            try:
                saved, rejected = save_uploaded_files(PATHS.raw, genre_name, [(f.name, f.getvalue()) for f in files])
            except EchoError as exc:
                st.error(str(exc))
            else:
                for name, reason in rejected:
                    st.warning(f"{name}: {reason}")
                if saved:
                    st.session_state["upload_flash"] = f"Added {len(saved)} file(s) to data/raw/{genre_name.strip().lower()}."
                    st.rerun()

    # 2. settings ----------------------------------------------------------- #
    md(ui.step_heading(2, "Model settings", "Small defaults that train on a normal laptop CPU."))
    with st.container(border=True):
        cfg = model_settings_form()

    # 3. validate ----------------------------------------------------------- #
    md(ui.step_heading(3, "Validate and preprocess", "Reads every MIDI file, skips unusable ones, builds the vocabulary. Optional - training does this too."))
    if options:
        if st.button("Validate and preprocess dataset"):
            bar = st.progress(0, text="Reading MIDI files...")
            try:
                def on_progress(event: dict) -> None:
                    if event.get("type") == "preprocess":
                        bar.progress(min(event["done"] / max(event["total"], 1), 1.0), text=f"Reading {event['file']}")

                data, reused = get_or_build_processed(PATHS, genre, cfg, progress=on_progress)
                bar.empty()
                st.success(f"{'Loaded saved' if reused else 'Processed'} dataset: {data.report.total_events:,} events.")
                render_report(data.report)
            except Exception as exc:
                bar.empty()
                show_error(exc, "preprocessing the dataset")
    else:
        st.caption("Add MIDI files first.")

    # 4. train -------------------------------------------------------------- #
    md(ui.step_heading(4, "Train", "Keep this tab open. On a CPU this can take minutes to hours depending on your laptop and dataset."))
    if st.button("Start training", type="primary", disabled=not options):
        run_training_ui(genre, cfg)

    result = st.session_state.get("last_training")
    if result:
        st.success(
            f"Model '{result['name']}' trained: best validation loss {result['best_val_loss']:.4f} "
            f"at epoch {result['best_epoch']} of {result['epochs_run']} ({result['seconds']:.0f} s)."
        )

    # 5. saved models ------------------------------------------------------- #
    md(ui.step_heading(5, "Saved models", "Only fully completed runs are listed."))
    models = list_trained_models(PATHS)
    if not models:
        md(ui.empty_state("No completed models yet", "A model appears here only after training finishes successfully."))
        return
    st.dataframe(models_dataframe(models), use_container_width=True, hide_index=True)
    chosen = st.selectbox("Show learning curve for", [m.name for m in models], key="curve_model")
    history_path = PATHS.models / chosen / HISTORY_FILE
    try:
        show_chart(ui.loss_figure(json.loads(history_path.read_text(encoding="utf-8"))))
    except (OSError, ValueError):
        st.caption("No saved training history for this model.")


def run_training_ui(genre: str, cfg: config.ModelConfig) -> None:
    status = st.empty()
    bar = st.progress(0, text="Preparing...")
    chart_slot = st.empty()
    history: dict[str, list[float]] = {"loss": [], "val_loss": []}

    def on_event(event: dict) -> None:
        kind = event.get("type")
        if kind == "status":
            status.info(event["message"])
        elif kind == "preprocess":
            bar.progress(min(event["done"] / max(event["total"], 1), 1.0) * 0.1, text=f"Reading {event['file']}")
        elif kind == "epoch":
            history["loss"].append(event["loss"])
            history["val_loss"].append(event["val_loss"])
            bar.progress(
                event["epoch"] / event["epochs"],
                text=f"Epoch {event['epoch']}/{event['epochs']}  ·  loss {event['loss']:.3f}  ·  val loss {event['val_loss']:.3f}",
            )
            with chart_slot.container():
                show_chart(ui.loss_figure(history))

    try:
        result = train_model(genre, cfg, paths=PATHS, progress=on_event)
    except Exception as exc:
        bar.empty()
        status.empty()
        st.session_state.pop("last_training", None)
        show_error(exc, "training the model")
        return
    _load_bundle_cached.clear()  # never serve a pre-retraining model from cache
    metrics = result.metadata["metrics"]
    st.session_state["last_training"] = {
        "name": result.model_name,
        "best_val_loss": metrics["best_val_loss"],
        "best_epoch": metrics["best_epoch"],
        "epochs_run": metrics["epochs_run"],
        "seconds": result.metadata["training_seconds"],
    }
    bar.progress(1.0, text="Training complete")
    status.empty()


# --------------------------------------------------------------------------- #
# Dataset explorer
# --------------------------------------------------------------------------- #
def page_dataset() -> None:
    md(ui.page_header("Studio", "Dataset explorer", "What ECHO sees in your MIDI files."))
    options = genre_options()
    if not options:
        md(ui.empty_state("No MIDI files found", "Add some on the Train Model page, then come back to explore them."))
        render_sources()
        return
    genres = list_genres(PATHS.raw)
    genre = st.selectbox("Genre folder", options, key="explore_genre")
    st.caption("Folder contents: " + ", ".join(f"{g} ({n})" for g, n in genres.items()))

    cfg = config.ModelConfig()
    folder = processed_dir(PATHS, genre)
    report = load_report(folder)
    if st.button("Scan and process now" if report is None else "Rescan files"):
        with st.spinner("Reading MIDI files..."):
            try:
                data, _ = get_or_build_processed(PATHS, genre, cfg, force=report is not None)
                report = data.report
            except Exception as exc:
                show_error(exc, "scanning the dataset")
    if report is None:
        md(ui.empty_state("Not processed yet", "Press the button to scan this genre."))
        render_sources()
        return

    render_report(report)
    rest_share = report.rest_events / report.total_events if report.total_events else 0.0
    md(
        ui.kpis(
            [
                ui.kpi("Musical events", f"{report.total_events:,}", "notes + rests", "◈"),
                ui.kpi("Notes", f"{report.note_events:,}", "", "♪"),
                ui.kpi("Rests", f"{report.rest_events:,}", f"{rest_share:.0%} of events", "·"),
                ui.kpi("Avg. per file", f"{report.total_events / max(report.processed, 1):,.0f}", "events", "▤"),
            ]
        )
    )

    if report.pitch_counts:
        pitches = [int(p) for p in report.pitch_counts]
        weights = [report.pitch_counts[str(p)] for p in pitches]
        mean = sum(p * w for p, w in zip(pitches, weights)) / max(sum(weights), 1)
        commonest = pitches[weights.index(max(weights))]
        durations = report.duration_counts
        top_duration = max(durations, key=durations.get) if durations else "-"
        md(
            ui.panel(
                "Musical profile",
                ui.kv_grid(
                    [
                        ("Lowest note", ui.note_name(min(pitches))),
                        ("Highest note", ui.note_name(max(pitches))),
                        ("Average pitch", ui.note_name(round(mean))),
                        ("Most common note", ui.note_name(commonest)),
                        ("Most common length", f"{top_duration} beats"),
                        ("Distinct pitches", str(len(pitches))),
                    ]
                ),
                eyebrow="Computed from your files",
            )
        )
        tab_pitch, tab_rhythm, tab_files = st.tabs(["Pitch", "Rhythm", "Files"])
        with tab_pitch:
            left, right = st.columns(2)
            with left:
                show_chart(ui.pitch_figure(report.pitch_counts))
            with right:
                show_chart(ui.pitch_class_figure(report.pitch_counts))
        with tab_rhythm:
            show_chart(ui.duration_figure(report.duration_counts))
            st.caption("Lengths are quantised to 8 classes, measured in beats (1 = quarter note).")
        with tab_files:
            if report.events_per_file:
                st.dataframe(
                    pd.DataFrame(sorted(report.events_per_file.items(), key=lambda kv: -kv[1]), columns=["File", "Events"]),
                    use_container_width=True, hide_index=True,
                )
            if report.skipped:
                st.markdown("**Skipped files**")
                st.dataframe(pd.DataFrame(report.skipped), use_container_width=True, hide_index=True)
            if not report.events_per_file and not report.skipped:
                st.caption("No per-file details were recorded.")
    render_sources()


def render_sources() -> None:
    with st.expander("Where to get free MIDI data (licences)"):
        st.caption("ECHO does not bundle any dataset. Check each licence yourself before using or sharing the data.")
        st.dataframe(
            pd.DataFrame(
                [{"Source": s["name"], "Genre": s["genre"], "Licence": s["license"], "Page": s["url"] or "-"} for s in DATASET_SOURCES]
            ),
            use_container_width=True, hide_index=True,
        )


# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
def page_about() -> None:
    md(ui.page_header("Studio", "About ECHO", config.APP_TAGLINE,
                      chips=[ui.pill(f"v{config.APP_VERSION}", "ok")]))
    st.write(
        "ECHO is an internship/portfolio project that shows a complete deep-learning pipeline: "
        "music data preparation, an LSTM trained from scratch on your CPU, sequence generation, "
        "MIDI export and audio rendering. It ships with **no dataset and no pretrained model** - "
        "you supply the MIDI files and train locally."
    )
    tab_arch, tab_tech, tab_flow, tab_road = st.tabs(["Architecture", "Technologies", "Workflow", "Limits & roadmap"])

    with tab_arch:
        md(ui.flow(["MIDI files", "Preprocess (pretty_midi)", "Events + vocabulary", "LSTM (TensorFlow/Keras)", "Sampling", "MIDI export", "FluidSynth WAV", "Streamlit player"]))
        md(
            ui.grid(
                [
                    ui.card("Representation", "Each event is a (pitch, length) pair or a rest. Lengths are quantised to 8 classes measured in beats, so tempo is chosen later."),
                    ui.card("Model", "Embedding, one or two LSTM layers with dropout, and a softmax layer that predicts the next event. Adam optimiser, cross-entropy loss, early stopping."),
                    ui.card("Genres", "One model per genre folder. The genre label is whatever folder you place files in; ECHO never guesses it from file names."),
                    ui.card("Audio", "MIDI holds instructions, not sound. A local FluidSynth plus a SoundFont renders WAV. If they are missing, the MIDI download still works."),
                ]
            )
        )
    with tab_tech:
        md(
            ui.tech_grid(
                [
                    ("Python", "Language", "Everything, front end and back end."),
                    ("Streamlit", "Interface", "The studio you are looking at, with custom CSS."),
                    ("TensorFlow / Keras", "LSTM model", "Embedding, LSTM, dropout and softmax layers; CPU only."),
                    ("pretty_midi", "MIDI I/O", "Reads training files and writes the generated MIDI."),
                    ("NumPy / pandas", "Data", "Windows, sampling, tables."),
                    ("Matplotlib", "Charts", "Loss, piano roll and dataset charts in the ECHO palette."),
                    ("FluidSynth", "Audio (optional)", "Renders MIDI to WAV with a SoundFont."),
                    ("pytest", "Tests", "Runs in temporary folders only."),
                ]
            )
        )
    with tab_flow:
        md(
            ui.timeline(
                [
                    ("Scan", "The genre folder is scanned; duplicates and corrupt files are reported, not fatal."),
                    ("Extract", "One melodic line per piece (highest note at each onset), quantised to a 16th-note grid."),
                    ("Split", "Whole pieces (or the end of each piece) become validation data, so no events are shared."),
                    ("Train", "The LSTM trains with early stopping; the best epoch is kept."),
                    ("Save", "model.keras, the vocabulary, seed phrases and metadata. Metadata is written last, so a failed run never looks like a success."),
                    ("Generate", "Sample the next event from the model's probabilities, repeat, validate, export MIDI."),
                ]
            )
        )
    with tab_road:
        left, right = st.columns(2, gap="large")
        with left:
            md(ui.section("Limitations"))
            st.markdown(
                "- Melody only: chords and multiple voices are reduced to the top note.\n"
                "- A small LSTM learns local patterns, not full musical structure; long pieces can wander.\n"
                "- Quality depends heavily on dataset size and consistency. Results are not guaranteed to sound good.\n"
                "- Training on a laptop CPU is slow for large datasets.\n"
                "- Audio quality depends on the SoundFont you choose."
            )
        with right:
            md(ui.section("Future ideas"))
            st.markdown(
                "- Multi-voice and chord-aware representations, velocity and dynamics.\n"
                "- Transformer models and conditioning on genre in a single model.\n"
                "- Key/tempo detection with Music21 for transposition-based augmentation.\n"
                "- Objective evaluation (pitch-class histograms, repetition metrics) and listening tests."
            )


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
if "nav" not in st.session_state:
    st.session_state["nav"] = "Home"

current = sidebar()
{
    "Home": page_home,
    "Generate Music": page_generate,
    "Train Model": page_train,
    "Dataset Explorer": page_dataset,
    "About Project": page_about,
}[current]()
