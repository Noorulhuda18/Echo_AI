"""ECHO - Intelligent Music Generation Studio (Streamlit front end).

Run with:  streamlit run app.py

Opening the app never trains anything. Training only starts when you press
"Start training" on the Train Model page.
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


def count_raw_files() -> int:
    return sum(list_genres(PATHS.raw).values())


def genre_options() -> list[str]:
    genres = [g for g, n in list_genres(PATHS.raw).items() if n > 0]
    return [config.ALL_GENRES, *genres] if genres else []


def models_dataframe(models) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": m.name,
                "Trained (UTC)": m.trained_at.replace("T", " ")[:16],
                "Files": m.files_processed,
                "Vocabulary": m.vocab_size,
                "Best val. loss": round(m.best_val_loss, 4),
                "Epochs": m.metadata.get("metrics", {}).get("epochs_run", ""),
            }
            for m in models
        ]
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def sidebar() -> str:
    with st.sidebar:
        st.markdown(ui.brand_header(), unsafe_allow_html=True)
        st.radio("Navigate", PAGES, key="nav", label_visibility="collapsed")
        st.markdown("&nbsp;", unsafe_allow_html=True)
        audio = check_audio_status(PATHS.soundfonts)
        models = list_trained_models(PATHS)
        pills = [
            ui.pill(f"{len(models)} model{'s' if len(models) != 1 else ''}", "ok" if models else "warn"),
            ui.pill("TensorFlow", "ok" if module_available("tensorflow") else "bad"),
            ui.pill("MIDI tools", "ok" if module_available("pretty_midi") else "bad"),
            ui.pill("Audio ready" if audio.can_render else "Audio setup", "ok" if audio.can_render else "warn"),
        ]
        st.markdown("".join(pills), unsafe_allow_html=True)
        message = python_version_message()
        if message:
            st.warning(message)
    return st.session_state["nav"]


# --------------------------------------------------------------------------- #
# Home
# --------------------------------------------------------------------------- #
def page_home() -> None:
    st.markdown(ui.hero(), unsafe_allow_html=True)
    left, right, _ = st.columns([1.2, 1.2, 4])
    left.button("Generate music", type="primary", on_click=goto, args=("Generate Music",), use_container_width=True)
    right.button("Train a model", on_click=goto, args=("Train Model",), use_container_width=True)

    st.markdown(ui.section("How ECHO makes music", "Five steps, all running on your own computer."), unsafe_allow_html=True)
    st.markdown(
        ui.grid(
            [
                ui.step_card(1, "Collect", "Add MIDI files you are allowed to use. Folders name the genre."),
                ui.step_card(2, "Learn", "Notes become (pitch, length) events; an LSTM studies what usually follows what."),
                ui.step_card(3, "Compose", "The model predicts one event at a time. Creativity controls how adventurous it is."),
                ui.step_card(4, "Export", "The melody is written as a verified MIDI file you can download."),
                ui.step_card(5, "Listen", "FluidSynth turns the MIDI into a WAV recording for playback."),
            ]
        ),
        unsafe_allow_html=True,
    )

    st.markdown(ui.section("Studio status"), unsafe_allow_html=True)
    genres = list_genres(PATHS.raw)
    models = list_trained_models(PATHS)
    audio = check_audio_status(PATHS.soundfonts)
    total_files = sum(genres.values())
    st.markdown(
        ui.grid(
            [
                ui.stat_card(
                    "MIDI files",
                    str(total_files),
                    ", ".join(f"{g}: {n}" for g, n in genres.items()) if genres else "data/raw is empty",
                ),
                ui.stat_card(
                    "Trained models",
                    str(len(models)),
                    ", ".join(m.name for m in models) if models else "none yet - train one first",
                ),
                ui.stat_card(
                    "Audio rendering",
                    "Ready" if audio.can_render else "Not set up",
                    "FluidSynth + SoundFont found" if audio.can_render else "MIDI download still works",
                ),
                ui.stat_card(
                    "Python",
                    f"{sys.version_info.major}.{sys.version_info.minor}",
                    f"recommended: {config.RECOMMENDED_PYTHON}",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )
    if not models:
        st.info(
            "No trained model exists yet. "
            + ("Open **Train Model** to train one on your MIDI files."
               if total_files else "Add some MIDI files first (see **Train Model**), then train a model.")
        )


# --------------------------------------------------------------------------- #
# Generate
# --------------------------------------------------------------------------- #
def run_generation(model_name: str, program: int, duration: int, tempo: int, creativity: float, seed: int | None) -> None:
    progress = st.progress(0, text="Loading the trained model...")
    try:
        bundle = get_bundle(model_name)
        progress.progress(25, text="Composing with the LSTM...")
        result = generate_music(
            model_name, duration, tempo, creativity, seed=seed, program=program, paths=PATHS, bundle=bundle
        )
        progress.progress(65, text="Rendering audio...")
        record = {
            "model": model_name,
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
    except Exception as exc:
        progress.empty()
        st.session_state.pop("last_generation", None)
        show_error(exc, "generating music")
    else:
        progress.empty()


def render_generation_result(record: dict) -> None:
    st.markdown(ui.section("Your composition"), unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Model", record["model"])
    cols[1].metric("Notes", record["notes"])
    cols[2].metric("Length", f"{record['seconds']:.1f} s")
    cols[3].metric("Seed", record["seed"], help="Use the same seed and settings to recreate this piece.")

    show_chart(ui.piano_roll_figure(record["events"], record["tempo"]))

    midi_path = Path(record["midi_path"])
    wav_path = Path(record["wav_path"]) if record["wav_path"] else None

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
            st.text(record["audio_error"] if "Download FluidSynth" in record["audio_error"] else record["audio_error"] + "\n\n" + INSTALL_HELP)

    d1, d2, d3 = st.columns(3)
    if midi_path.is_file():
        d1.download_button(
            "Download MIDI", midi_path.read_bytes(), file_name=midi_path.name, mime="audio/midi",
            use_container_width=True, key="dl_midi",
        )
    else:
        d1.error("The MIDI file is no longer on disk.")
    if wav_path and wav_path.is_file():
        d2.download_button(
            "Download WAV", wav_path.read_bytes(), file_name=wav_path.name, mime="audio/wav",
            use_container_width=True, key="dl_wav",
        )
        mp3_path = Path(record["mp3_path"]) if record["mp3_path"] else None
        if mp3_path and mp3_path.is_file():
            d3.download_button(
                "Download MP3", mp3_path.read_bytes(), file_name=mp3_path.name, mime="audio/mpeg",
                use_container_width=True, key="dl_mp3",
            )
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


def page_generate() -> None:
    st.markdown(ui.section("Generate music", "Compose a new melody with a model you trained."), unsafe_allow_html=True)
    models = list_trained_models(PATHS)
    if not models:
        st.warning("There is no trained model yet, so nothing can be generated. Train one first.")
        st.button("Go to Train Model", type="primary", on_click=goto, args=("Train Model",))
        return

    names = [m.name for m in models]
    labels = {m.name: f"{m.genre}  ·  {m.files_processed} files  ·  val loss {m.best_val_loss:.3f}" for m in models}
    left, right = st.columns([1, 1], gap="large")
    with left:
        model_name = st.selectbox(
            "Genre / model", names, format_func=lambda n: labels[n],
            help="Each genre folder you trained gets its own model. 'all' blends every file.",
        )
        instrument = st.selectbox("Instrument", list(config.INSTRUMENTS), help="Only affects playback; the notes are the same.")
        duration = st.slider("Approximate length (seconds)", config.MIN_DURATION_SECONDS, 120, 30, step=5)
    with right:
        tempo = st.slider("Tempo (BPM)", 50, 180, 100, help="Beats per minute. Chosen at export time, so one model works at any tempo.")
        creativity = st.slider(
            "Creativity", 0.4, 1.5, 1.0, step=0.05,
            help="Sampling temperature. Lower = predictable and repetitive; higher = more surprising and more mistakes.",
        )
        fix_seed = st.checkbox("Use a fixed random seed", help="Same seed + same settings = same melody.")
        seed_value = st.number_input("Seed", 0, 2**31 - 1, 42, disabled=not fix_seed)

    if st.button("Generate Music", type="primary"):
        run_generation(
            model_name, config.INSTRUMENTS[instrument], int(duration), int(tempo),
            float(creativity), int(seed_value) if fix_seed else None,
        )

    record = st.session_state.get("last_generation")
    if record:
        render_generation_result(record)


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


def render_report(report, key: str) -> None:
    c = st.columns(5)
    c[0].metric("Discovered", report.discovered)
    c[1].metric("Processed", report.processed)
    c[2].metric("Skipped", report.skipped_count)
    c[3].metric("Duplicates", report.duplicates)
    c[4].metric("Vocabulary", report.vocab_size)
    if report.skipped:
        with st.expander(f"{report.skipped_count} skipped file(s) - reasons"):
            st.dataframe(pd.DataFrame(report.skipped), use_container_width=True, hide_index=True)
    if report.unsupported:
        st.caption("Ignored non-MIDI files: " + ", ".join(report.unsupported[:20]))


def page_train() -> None:
    st.markdown(ui.section("Train model", "Teach an LSTM the music in your dataset. Training starts only when you press the button."), unsafe_allow_html=True)

    flash = st.session_state.pop("upload_flash", None)
    if flash:
        st.success(flash)

    # 1. dataset ------------------------------------------------------------ #
    st.markdown("#### 1 · Dataset")
    genres = list_genres(PATHS.raw)
    options = genre_options()
    genre = config.ALL_GENRES  # replaced by the selectbox below when MIDI files exist
    if not options:
        st.warning(
            "`data/raw/` has no MIDI files yet. Upload some below, copy `.mid` files into "
            "`data/raw/<genre>/`, or follow the dataset steps in the README."
        )
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
    st.markdown("#### 2 · Model settings")
    cfg = model_settings_form()

    # 3. validate ----------------------------------------------------------- #
    st.markdown("#### 3 · Validate and preprocess")
    if options:
        st.caption("Reads every MIDI file, skips unusable ones, and builds the vocabulary. Optional - training does this too.")
        if st.button("Validate and preprocess dataset"):
            bar = st.progress(0, text="Reading MIDI files...")
            try:
                def on_progress(event: dict) -> None:
                    if event.get("type") == "preprocess":
                        bar.progress(min(event["done"] / max(event["total"], 1), 1.0), text=f"Reading {event['file']}")

                data, reused = get_or_build_processed(PATHS, genre, cfg, progress=on_progress)
                bar.empty()
                st.success(f"{'Loaded saved' if reused else 'Processed'} dataset: {data.report.total_events:,} events.")
                render_report(data.report, "train")
            except Exception as exc:
                bar.empty()
                show_error(exc, "preprocessing the dataset")

    # 4. train -------------------------------------------------------------- #
    st.markdown("#### 4 · Train")
    st.caption("Keep this tab open while training. On a CPU this can take from minutes to a long time depending on your laptop and dataset size.")
    if st.button("Start training", type="primary", disabled=not options):
        run_training_ui(genre, cfg)

    result = st.session_state.get("last_training")
    if result:
        st.success(
            f"Model '{result['name']}' trained: best validation loss {result['best_val_loss']:.4f} "
            f"at epoch {result['best_epoch']} of {result['epochs_run']} ({result['seconds']:.0f} s)."
        )

    # 5. saved models ------------------------------------------------------- #
    st.markdown("#### 5 · Saved models")
    models = list_trained_models(PATHS)
    if not models:
        st.info("No completed models yet.")
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
    st.markdown(ui.section("Dataset explorer", "What ECHO sees in your MIDI files."), unsafe_allow_html=True)
    options = genre_options()
    if not options:
        st.warning("No MIDI files found in `data/raw/`. Add some on the Train Model page.")
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
        st.info("This genre has not been processed yet. Press the button to scan it.")
        render_sources()
        return

    render_report(report, "explore")
    c = st.columns(3)
    c[0].metric("Musical events", f"{report.total_events:,}")
    c[1].metric("Notes", f"{report.note_events:,}")
    c[2].metric("Rests", f"{report.rest_events:,}")
    if report.pitch_counts:
        left, right = st.columns(2)
        with left:
            show_chart(ui.pitch_figure(report.pitch_counts))
        with right:
            show_chart(ui.duration_figure(report.duration_counts))
        pitches = [int(p) for p in report.pitch_counts]
        weights = [report.pitch_counts[str(p)] for p in pitches]
        mean = sum(p * w for p, w in zip(pitches, weights)) / max(sum(weights), 1)
        st.caption(f"Pitch range: {min(pitches)}-{max(pitches)} (MIDI numbers), average pitch {mean:.1f}.")
    if report.events_per_file:
        with st.expander("Events per file"):
            st.dataframe(
                pd.DataFrame(sorted(report.events_per_file.items(), key=lambda kv: -kv[1]), columns=["File", "Events"]),
                use_container_width=True, hide_index=True,
            )
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
    st.markdown(ui.section("About ECHO", config.APP_TAGLINE), unsafe_allow_html=True)
    st.write(
        "ECHO is an internship/portfolio project that shows a complete deep-learning pipeline: "
        "music data preparation, an LSTM trained from scratch on your CPU, sequence generation, "
        "MIDI export and audio rendering. It ships with **no dataset and no pretrained model** - "
        "you supply the MIDI files and train locally."
    )
    st.markdown(ui.section("Architecture"), unsafe_allow_html=True)
    st.markdown(
        ui.flow(["MIDI files", "Preprocess (pretty_midi)", "Events + vocabulary", "LSTM (TensorFlow/Keras)", "Sampling", "MIDI export", "FluidSynth WAV", "Streamlit player"]),
        unsafe_allow_html=True,
    )
    st.markdown(
        ui.grid(
            [
                ui.card("Representation", "Each event is a (pitch, length) pair or a rest. Lengths are quantised to 8 classes measured in beats, so tempo is chosen later."),
                ui.card("Model", "Embedding, one or two LSTM layers with dropout, and a softmax layer that predicts the next event. Adam optimiser, cross-entropy loss, early stopping."),
                ui.card("Genres", "One model per genre folder. The genre label is whatever folder you place files in; ECHO never guesses it from file names."),
                ui.card("Audio", "MIDI holds instructions, not sound. A local FluidSynth plus a SoundFont renders WAV. If they are missing, the MIDI download still works."),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(ui.section("Technologies"), unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            [
                ("Python", "Language"), ("Streamlit", "Interface"), ("TensorFlow / Keras", "LSTM model"),
                ("pretty_midi", "MIDI reading and writing"), ("NumPy / pandas", "Data handling"),
                ("Matplotlib", "Charts"), ("FluidSynth", "MIDI to WAV (optional)"), ("pytest", "Tests"),
            ],
            columns=["Technology", "Used for"],
        ),
        use_container_width=True, hide_index=True,
    )
    st.markdown(ui.section("Training workflow"), unsafe_allow_html=True)
    st.markdown(
        "1. **Scan** the genre folder; duplicates and corrupt files are reported, not fatal.\n"
        "2. **Extract** a single melodic line per piece (highest note at each onset) and quantise it.\n"
        "3. **Split** whole pieces (or the end of each piece) into validation data, so no events are shared.\n"
        "4. **Train** with early stopping; the best epoch is kept.\n"
        "5. **Save** `model.keras`, the vocabulary, seed phrases and metadata. Metadata is written last, so a failed run never looks like a success."
    )
    st.markdown(ui.section("Limitations"), unsafe_allow_html=True)
    st.markdown(
        "- Melody only: chords and multiple voices are reduced to the top note.\n"
        "- A small LSTM learns local patterns, not full musical structure; long pieces can wander.\n"
        "- Quality depends heavily on dataset size and consistency. Results are not guaranteed to sound good.\n"
        "- Training on a laptop CPU is slow for large datasets.\n"
        "- Audio quality depends on the SoundFont you choose."
    )
    st.markdown(ui.section("Future ideas"), unsafe_allow_html=True)
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
