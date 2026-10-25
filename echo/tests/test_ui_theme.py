"""Theme checks: palette contrast, safe HTML, and that every chart builds.

These need only matplotlib (no Streamlit), so they run anywhere.
"""

from __future__ import annotations

import pytest

import config
from src import ui_theme as ui


def _luminance(hex_color: str) -> float:
    h = hex_color.lstrip("#")
    channels = [int(h[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a: str, b: str) -> float:
    hi, lo = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


@pytest.mark.parametrize(
    "fg, bg",
    [
        (ui.TEXT, ui.MIDNIGHT), (ui.TEXT, ui.CARD),
        (ui.MUTED, ui.MIDNIGHT), (ui.MUTED, ui.CARD),
        (ui.BLUSH, ui.MIDNIGHT), (ui.BLUSH, ui.CARD),
        ("#FFFFFF", ui.CHERRY),  # label on the primary button
        (ui.MIDNIGHT, ui.BLUSH), (ui.MIDNIGHT, ui.LAVENDER),  # step numbers
    ],
)
def test_text_pairs_meet_wcag_aa(fg, bg):
    assert contrast(fg, bg) >= 4.5


@pytest.mark.parametrize("colour", [ui.BLUSH, ui.LAVENDER, ui.ROSE])
def test_chart_marks_are_visible_on_the_card(colour):
    assert contrast(colour, ui.CARD) >= 3.0


def test_brand_colours_match_the_brief():
    assert (ui.BLUSH, ui.LAVENDER, ui.CHERRY) == ("#F3A6B8", "#B7A0E8", "#C93655")
    assert (ui.MIDNIGHT, ui.CARD, ui.TEXT, ui.MUTED) == ("#241728", "#34263A", "#FFF7FB", "#C8B9D5")


def test_hero_text_keeps_its_spaces():
    # Regression: line breaks used to be stripped, producing "MIDImusic" and "downloadthe".
    markup = ui.hero()
    assert "MIDI music" in markup and "download the MIDI" in markup
    assert config.APP_TAGLINE in markup and config.APP_NAME in markup


def test_dynamic_text_is_escaped():
    evil = '<script>alert("x")</script>'
    for markup in (ui.card(evil, evil), ui.stat_card(evil, evil, evil), ui.pill(evil), ui.section(evil, evil),
                   ui.step_card(1, evil, evil), ui.flow([evil])):
        assert "<script>" not in markup


def test_css_has_reduced_motion_and_no_unfilled_placeholder():
    assert "prefers-reduced-motion" in ui.CSS
    assert "__FONT_URL__" not in ui.CSS


def test_all_charts_build():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = [
        ui.loss_figure({"loss": [3.0, 2.5, 2.2], "val_loss": [3.1, 2.7, 2.8]}),
        ui.loss_figure({"loss": [3.0], "val_loss": []}),
        ui.pitch_figure({"60": 5, "64": 9, "67": 3}),
        ui.pitch_figure({}),
        ui.duration_figure({"0.5": 10, "1.0": 4}),
        ui.piano_roll_figure([(60, 3), (config.REST_PITCH, 1), (64, 3)], 100),
        ui.piano_roll_figure([(config.REST_PITCH, 1)], 100),
    ]
    assert all(f.axes for f in figures)
    for fig in figures:
        plt.close(fig)


def test_chart_titles_are_light_on_the_dark_card():
    # Regression: titles set with loc="left" used to stay dark and vanish on the card.
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    fig = ui.loss_figure({"loss": [3.0, 2.0], "val_loss": [3.1, 2.4]})
    title = fig.axes[0].get_title(loc="left")
    colour = mcolors.to_hex(fig.axes[0]._left_title.get_color()).upper()
    plt.close(fig)
    assert title and colour == ui.TEXT.upper()


def test_pitch_class_chart_counts_octaves_together():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = ui.pitch_class_figure({"60": 3, "72": 4, "61": 1})  # C4 + C5 are both "C"
    heights = [bar.get_height() for bar in fig.axes[0].patches]
    plt.close(fig)
    assert heights[0] == 7 and heights[1] == 1 and sum(heights) == 8


def test_note_names_and_page_furniture_escape():
    assert ui.note_name(60) == "C4" and ui.note_name(21) == "A0" and ui.note_name(108) == "C8"
    evil = "<img src=x onerror=1>"
    for markup in (ui.page_header(evil, evil, evil, [ui.pill(evil)]), ui.kpi(evil, evil, evil, evil),
                   ui.checklist([(True, evil, evil), (False, evil, evil)]), ui.kv_grid([(evil, evil)]),
                   ui.timeline([(evil, evil)]), ui.tech_grid([(evil, evil, evil)]), ui.step_heading(1, evil, evil),
                   ui.empty_state(evil, evil), ui.file_row(evil, evil), ui.sidebar_status([(evil, "ok", evil)]),
                   ui.sidebar_model(evil, evil), ui.panel(evil, "", evil)):
        assert "<img" not in markup
