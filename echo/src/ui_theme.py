"""ECHO visual identity: palette, custom CSS, HTML snippets and themed charts.

Design notes
------------
* Midnight plum canvas, elevated plum cards, soft-white text.
* Blush-pink -> lavender is the signature gradient; deep rose / cherry red is
  reserved for primary actions, errors and key highlights.
* Cherry (#C93655) is 3.4:1 on the midnight background and only 2.8:1 on the card
  colour, so it is used for fills and borders (white text on top is 5.1:1) and
  never for small text. Small highlighted text uses blush pink instead. Chart
  marks use ROSE, a lifted cherry that is 3.5:1 on the card colour.
* All text pairs were measured against WCAG AA (4.5:1) and graphics against 3:1.
* Motion is limited to a slow waveform pulse and short hover transitions, and is
  switched off for visitors who ask their system for reduced motion.
"""

from __future__ import annotations

import html
from typing import Sequence

import config

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #
BLUSH = "#F3A6B8"
LAVENDER = "#B7A0E8"
CHERRY = "#C93655"
ROSE = "#DB4A6A"  # cherry lifted to 3.5:1 on CARD; used only for chart marks
MIDNIGHT = "#241728"
CARD = "#34263A"
TEXT = "#FFF7FB"
MUTED = "#C8B9D5"
GRID = "#5A4766"

FONT_URL = (
    "https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@1,600;1,700"
    "&family=Manrope:wght@400;500;600;700;800&display=swap"
)

CSS = """
@import url('__FONT_URL__');

:root {
  --blush: #F3A6B8;
  --lavender: #B7A0E8;
  --cherry: #C93655;
  --cherry-deep: #A52A48;
  --midnight: #241728;
  --card: #34263A;
  --text: #FFF7FB;
  --muted: #C8B9D5;
  --line: rgba(183, 160, 232, 0.22);
  --grad-main: linear-gradient(135deg, #F3A6B8 0%, #B7A0E8 100%);
  --grad-deep: linear-gradient(135deg, #B7A0E8 0%, #C93655 100%);
  --radius: 18px;
}

html, body, [class*="css"], .stApp, button, input, textarea, select {
  font-family: 'Manrope', 'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* ---- canvas ---------------------------------------------------------- */
[data-testid="stAppViewContainer"], .stApp {
  background:
    radial-gradient(900px 520px at 8% -8%, rgba(183, 160, 232, 0.20), transparent 60%),
    radial-gradient(760px 520px at 100% 0%, rgba(243, 166, 184, 0.13), transparent 58%),
    radial-gradient(700px 600px at 70% 110%, rgba(201, 54, 85, 0.10), transparent 60%),
    var(--midnight);
  color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }

h1, h2, h3, h4 { color: var(--text); letter-spacing: -0.01em; }
p, li, label, span { color: inherit; }
a { color: var(--blush); }
hr { border-color: var(--line); }

/* ---- sidebar --------------------------------------------------------- */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #2d1e34 0%, #241728 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.25rem; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
  padding: 0.62rem 0.9rem;
  border-radius: 12px;
  border: 1px solid transparent;
  transition: background 0.18s ease, border-color 0.18s ease;
  width: 100%;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
  background: rgba(183, 160, 232, 0.10);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
  background: linear-gradient(135deg, rgba(243,166,184,.18), rgba(183,160,232,.16));
  border-color: rgba(243, 166, 184, 0.45);
}
[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stSidebar"] div[role="radiogroup"] p { font-weight: 600; font-size: 0.95rem; }

/* ---- brand ----------------------------------------------------------- */
.echo-brand { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.2rem; }
.echo-wordmark {
  font-family: 'Cormorant Garamond', 'Playfair Display', Georgia, serif;
  font-style: italic; font-weight: 700; font-size: 2.15rem; line-height: 1;
  letter-spacing: 0.04em;
  color: var(--blush);
  background: var(--grad-main);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent;
}
.echo-tagline { color: var(--muted); font-size: 0.82rem; margin: 0.15rem 0 1rem 0; }

/* ---- waveform motif -------------------------------------------------- */
.echo-wave { display: inline-flex; align-items: center; gap: 3px; height: 34px; }
.echo-wave i {
  display: block; width: 3px; height: calc(34px * var(--h, .5)); border-radius: 3px;
  background: var(--grad-main); opacity: 0.9; transform-origin: center;
  animation: echo-pulse 2.6s ease-in-out infinite;
  animation-delay: calc(var(--i, 0) * 0.11s);
}
.echo-wave.large { height: 64px; gap: 5px; }
.echo-wave.large i { width: 5px; height: calc(64px * var(--h, .5)); }
@keyframes echo-pulse { 0%, 100% { transform: scaleY(1); } 50% { transform: scaleY(0.55); } }

/* ---- cards ----------------------------------------------------------- */
.echo-card {
  border: 1px solid transparent; border-radius: var(--radius); padding: 1.35rem 1.45rem;
  background:
    linear-gradient(rgba(52,38,58,.92), rgba(52,38,58,.92)) padding-box,
    linear-gradient(135deg, rgba(243,166,184,.55), rgba(183,160,232,.35), rgba(201,54,85,.35)) border-box;
  box-shadow: 0 10px 30px rgba(10, 4, 14, 0.35);
  backdrop-filter: blur(10px);
}
.echo-card.hoverable { transition: transform 0.2s ease, box-shadow 0.2s ease; }
.echo-card.hoverable:hover { transform: translateY(-3px); box-shadow: 0 16px 38px rgba(10, 4, 14, 0.48); }
.echo-card h4 { margin: 0 0 0.35rem 0; font-size: 1.02rem; font-weight: 700; }
.echo-card p { margin: 0; color: var(--muted); font-size: 0.92rem; line-height: 1.55; }
.echo-eyebrow {
  text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.7rem; font-weight: 700;
  color: var(--blush); margin-bottom: 0.45rem;
}
.echo-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1rem; margin: 0.6rem 0 1.2rem 0; }
.echo-step-num {
  display: inline-flex; width: 2rem; height: 2rem; border-radius: 50%; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.85rem; color: #241728; background: var(--grad-main); margin-bottom: 0.7rem;
}

.echo-hero {
  border: 1px solid transparent; border-radius: 26px; padding: 2.4rem 2.4rem 2.1rem 2.4rem;
  background:
    radial-gradient(520px 280px at 85% 20%, rgba(183,160,232,.28), transparent 70%),
    radial-gradient(420px 260px at 10% 110%, rgba(201,54,85,.22), transparent 70%),
    linear-gradient(rgba(52,38,58,.94), rgba(40,28,46,.96)) padding-box,
    linear-gradient(135deg, rgba(243,166,184,.7), rgba(183,160,232,.45), rgba(201,54,85,.5)) border-box;
  box-shadow: 0 24px 60px rgba(10, 4, 14, 0.5);
  margin-bottom: 1.4rem;
}
.echo-hero .echo-wordmark { font-size: clamp(3rem, 8vw, 5.2rem); letter-spacing: 0.06em; display: inline-block; }
.echo-hero .lead { color: var(--text); font-size: 1.28rem; margin: 0.5rem 0 0.4rem 0; font-weight: 600; }
.echo-hero .sub { color: var(--muted); max-width: 46rem; line-height: 1.65; margin: 0 0 1.2rem 0; }

.echo-section { margin: 1.8rem 0 0.6rem 0; }
.echo-section h2 { font-size: 1.45rem; margin: 0; font-weight: 800; }
.echo-section h2::after {
  content: ""; display: block; width: 44px; height: 3px; margin-top: 0.55rem;
  border-radius: 3px; background: var(--grad-deep);
}
.echo-section p { color: var(--muted); margin: 0.25rem 0 0 0; }

.echo-stat { text-align: left; }
.echo-stat .label { color: var(--muted); font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
.echo-stat .value { font-size: 1.85rem; font-weight: 800; margin-top: 0.15rem; color: var(--text); line-height: 1.15; }
.echo-stat .hint { color: var(--muted); font-size: 0.82rem; margin-top: 0.3rem; }

.echo-pill {
  display: inline-block; padding: 0.22rem 0.7rem; border-radius: 999px; font-size: 0.76rem; font-weight: 700;
  border: 1px solid var(--line); color: var(--text); margin-right: 0.35rem;
}
.echo-pill.ok { background: rgba(183,160,232,.16); border-color: rgba(183,160,232,.55); }
.echo-pill.warn { background: rgba(243,166,184,.14); border-color: rgba(243,166,184,.6); color: var(--blush); }
.echo-pill.bad { background: rgba(201,54,85,.28); border-color: rgba(201,54,85,.9); }

.echo-flow { display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem; margin: 0.5rem 0 1rem 0; }
.echo-chip {
  padding: 0.5rem 0.9rem; border-radius: 12px; font-weight: 600; font-size: 0.88rem;
  background: rgba(183,160,232,.12); border: 1px solid rgba(183,160,232,.35);
}
.echo-arrow { color: var(--blush); font-weight: 700; }

/* ---- buttons --------------------------------------------------------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
  border-radius: 999px; font-weight: 700; padding: 0.55rem 1.4rem; min-height: 2.7rem;
  border: 1px solid rgba(183,160,232,.45); background: rgba(52,38,58,.7); color: var(--text);
  transition: transform 0.15s ease, box-shadow 0.15s ease, background 0.15s ease, border-color 0.15s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-1px); border-color: var(--blush); background: rgba(183,160,232,.16); color: var(--text);
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible {
  outline: 2px solid var(--blush); outline-offset: 2px;
}
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #C93655 0%, #A52A48 100%); border: 1px solid rgba(243,166,184,.55);
  color: #FFFFFF; box-shadow: 0 8px 22px rgba(201, 54, 85, 0.38);
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover {
  background: linear-gradient(135deg, #D63E5E 0%, #B02F4E 100%); box-shadow: 0 12px 28px rgba(201, 54, 85, 0.5);
}
.stButton > button:disabled { opacity: 0.45; }

/* ---- inputs ---------------------------------------------------------- */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, .stTextInput input, .stNumberInput input {
  background: #2b1d31 !important; border-radius: 12px !important; border-color: rgba(183,160,232,.28) !important;
  color: var(--text) !important;
}
[data-baseweb="select"] > div:focus-within, [data-baseweb="input"] > div:focus-within {
  border-color: var(--lavender) !important; box-shadow: 0 0 0 3px rgba(183,160,232,.22);
}
[data-testid="stWidgetLabel"] p { color: var(--text); font-weight: 600; font-size: 0.92rem; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted); }
[data-testid="stFileUploaderDropzone"] {
  background: rgba(52,38,58,.55); border: 1.5px dashed rgba(183,160,232,.5); border-radius: var(--radius);
}

/* ---- feedback + data ------------------------------------------------- */
[data-testid="stAlert"] { border-radius: 14px; border: 1px solid var(--line); }
[data-testid="stAlert"][kind="error"], div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {
  border-left: 5px solid var(--cherry);
}
[data-testid="stMetric"] {
  background: rgba(52,38,58,.7); border: 1px solid var(--line); border-radius: 16px; padding: 0.9rem 1.1rem;
}
[data-testid="stMetricLabel"] p { color: var(--muted); }
[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 14px; background: rgba(52,38,58,.45); }
[data-testid="stDataFrame"] { border-radius: 14px; overflow: hidden; border: 1px solid var(--line); }
.stProgress > div > div > div > div { background-image: linear-gradient(90deg, #F3A6B8, #B7A0E8, #C93655); }
audio { width: 100%; height: 48px; border-radius: 999px; color-scheme: dark; }


/* ---- sidebar (pro) --------------------------------------------------- */
.sb-brand {
  display: flex; align-items: center; gap: 0.75rem; padding: 0.95rem 1rem; margin-bottom: 1.1rem;
  border-radius: 16px; border: 1px solid transparent;
  background:
    linear-gradient(rgba(52,38,58,.85), rgba(40,28,46,.9)) padding-box,
    linear-gradient(135deg, rgba(243,166,184,.6), rgba(183,160,232,.4), rgba(201,54,85,.35)) border-box;
  box-shadow: 0 10px 26px rgba(10,4,14,.35);
}
.sb-brand .echo-wordmark { font-size: 1.9rem; }
.sb-brand .sb-tag { color: var(--muted); font-size: 0.74rem; margin-top: 0.2rem; line-height: 1.3; }
.sb-label {
  text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.68rem; font-weight: 800;
  color: var(--muted); margin: 1.2rem 0 0.5rem 0.2rem; opacity: 0.9;
}
.sb-panel {
  border-radius: 14px; padding: 0.35rem 0.9rem; background: rgba(52,38,58,.55); border: 1px solid var(--line);
}
.sb-row { display: flex; align-items: center; justify-content: space-between; gap: 0.6rem; padding: 0.52rem 0; font-size: 0.84rem; }
.sb-row + .sb-row { border-top: 1px solid rgba(183,160,232,.14); }
.sb-row .name { display: flex; align-items: center; gap: 0.55rem; font-weight: 600; color: var(--text); }
.sb-row .val { color: var(--muted); font-weight: 600; font-size: 0.78rem; text-align: right; }
.dot { width: 8px; height: 8px; border-radius: 50%; flex: none; }
.dot.ok { background: var(--lavender); box-shadow: 0 0 0 3px rgba(183,160,232,.22); }
.dot.warn { background: var(--blush); box-shadow: 0 0 0 3px rgba(243,166,184,.2); }
.dot.bad { background: var(--cherry); box-shadow: 0 0 0 3px rgba(201,54,85,.3); }
.sb-model { border-radius: 14px; padding: 0.85rem 0.95rem; border: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(243,166,184,.10), rgba(183,160,232,.10)); }
.sb-model .t { font-weight: 800; font-size: 0.95rem; }
.sb-model .d { color: var(--muted); font-size: 0.78rem; margin-top: 0.25rem; line-height: 1.5; }
.sb-foot { margin-top: 1.4rem; padding-top: 0.9rem; border-top: 1px solid var(--line); color: var(--muted); font-size: 0.74rem; line-height: 1.6; }
.sb-foot b { color: var(--text); font-weight: 700; }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) { box-shadow: inset 3px 0 0 var(--blush); }
[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) p { color: var(--text); }
[data-testid="stSidebar"] div[role="radiogroup"] > label p { color: var(--muted); }

/* ---- page header ----------------------------------------------------- */
.echo-pagehead {
  display: flex; align-items: flex-end; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  padding-bottom: 1.15rem; margin-bottom: 1.3rem; border-bottom: 1px solid var(--line);
}
.echo-pagehead .crumb { text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.7rem; font-weight: 800; color: var(--blush); }
.echo-pagehead .title { font-size: 2.1rem; font-weight: 800; line-height: 1.12; margin: 0.3rem 0 0.25rem 0; letter-spacing: -0.02em; }
.echo-pagehead .sub { color: var(--muted); max-width: 46rem; line-height: 1.6; }
.echo-chips { display: flex; gap: 0.4rem; flex-wrap: wrap; }

/* ---- KPI, checklist, key/value, timeline, tech ----------------------- */
.echo-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 0.9rem; margin: 0.8rem 0 1.2rem 0; }
.echo-kpi {
  position: relative; border-radius: 16px; padding: 1.05rem 1.15rem 1rem 1.15rem; border: 1px solid var(--line);
  background: rgba(52,38,58,.72); overflow: hidden;
}
.echo-kpi::before { content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: var(--grad-main); }
.echo-kpi .label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }
.echo-kpi .value { font-size: 1.75rem; font-weight: 800; line-height: 1.2; margin-top: 0.3rem; }
.echo-kpi .hint { color: var(--muted); font-size: 0.8rem; margin-top: 0.2rem; }
.echo-kpi .ico { position: absolute; right: 0.9rem; top: 0.85rem; color: var(--blush); opacity: 0.85; font-size: 1.15rem; }

.echo-check { display: grid; gap: 0.6rem; margin: 0.4rem 0 1.2rem 0; }
.echo-check .row {
  display: flex; align-items: center; gap: 0.9rem; padding: 0.85rem 1.05rem; border-radius: 14px;
  background: rgba(52,38,58,.6); border: 1px solid var(--line);
}
.echo-check .row.done { border-color: rgba(183,160,232,.5); }
.echo-check .bubble {
  flex: none; width: 2rem; height: 2rem; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
  font-weight: 800; font-size: 0.85rem; border: 1.5px solid rgba(183,160,232,.5); color: var(--muted);
}
.echo-check .row.done .bubble { background: var(--grad-main); border-color: transparent; color: #241728; }
.echo-check .t { font-weight: 700; font-size: 0.95rem; }
.echo-check .h { color: var(--muted); font-size: 0.82rem; margin-top: 0.1rem; }
.echo-check .state { margin-left: auto; flex: none; }

.echo-kv { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 0.2rem 1.2rem; margin-top: 0.7rem; }
.echo-kv div.k { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-top: 0.6rem; }
.echo-kv div.v { font-weight: 700; font-size: 0.98rem; margin-top: 0.1rem; overflow-wrap: anywhere; }

.echo-timeline { position: relative; margin: 0.6rem 0 1.2rem 0.5rem; padding-left: 1.6rem; border-left: 2px solid rgba(183,160,232,.35); }
.echo-timeline .item { position: relative; margin-bottom: 1.15rem; }
.echo-timeline .item::before {
  content: ""; position: absolute; left: -2.18rem; top: 0.25rem; width: 0.8rem; height: 0.8rem; border-radius: 50%;
  background: var(--grad-main); box-shadow: 0 0 0 4px rgba(183,160,232,.2);
}
.echo-timeline .t { font-weight: 700; }
.echo-timeline .b { color: var(--muted); font-size: 0.9rem; line-height: 1.55; margin-top: 0.15rem; }

.echo-tech { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 0.8rem; margin: 0.5rem 0 1rem 0; }
.echo-tech .it { padding: 0.9rem 1rem; border-radius: 14px; border: 1px solid var(--line); background: rgba(52,38,58,.6); }
.echo-tech .n { font-weight: 800; }
.echo-tech .r { color: var(--blush); font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; margin-top: 0.15rem; }
.echo-tech .w { color: var(--muted); font-size: 0.84rem; margin-top: 0.35rem; line-height: 1.5; }

.echo-stephead { display: flex; align-items: center; gap: 0.8rem; margin: 1.7rem 0 0.4rem 0; }
.echo-stephead .num {
  flex: none; width: 2.1rem; height: 2.1rem; border-radius: 50%; display: inline-flex; align-items: center; justify-content: center;
  background: var(--grad-main); color: #241728; font-weight: 800; font-size: 0.9rem;
}
.echo-stephead .t { font-weight: 800; font-size: 1.15rem; }
.echo-stephead .s { color: var(--muted); font-size: 0.86rem; }

.echo-empty {
  border: 1.5px dashed rgba(183,160,232,.45); border-radius: 18px; padding: 1.6rem; text-align: center;
  background: rgba(52,38,58,.4); margin: 0.6rem 0 1rem 0;
}
.echo-empty .t { font-weight: 800; font-size: 1.05rem; }
.echo-empty .b { color: var(--muted); margin-top: 0.3rem; font-size: 0.9rem; }
.echo-file { padding: 0.15rem 0; }
.echo-file .n { font-weight: 700; font-size: 0.9rem; word-break: break-all; }
.echo-file .m { color: var(--muted); font-size: 0.78rem; }

/* ---- tabs, bordered panels, scrollbars ------------------------------- */
[data-baseweb="tab-list"] { gap: 0.3rem; border-bottom: 1px solid var(--line); }
[data-baseweb="tab"] { padding: 0.6rem 1rem; font-weight: 700; color: var(--muted); border-radius: 10px 10px 0 0; }
[data-baseweb="tab"]:hover { color: var(--text); background: rgba(183,160,232,.08); }
[data-baseweb="tab"][aria-selected="true"] { color: var(--text); }
[data-baseweb="tab-highlight"] { background: var(--grad-main) !important; height: 3px !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line) !important; border-radius: 18px !important; }
::selection { background: rgba(243,166,184,.45); color: #fff; }
* { scrollbar-width: thin; scrollbar-color: rgba(183,160,232,.45) transparent; }

/* ---- responsive + accessibility -------------------------------------- */
@media (max-width: 768px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .echo-hero { padding: 1.5rem 1.2rem; border-radius: 20px; }
  .echo-hero .lead { font-size: 1.1rem; }
  .echo-card { padding: 1.1rem; }
  .echo-stat .value { font-size: 1.5rem; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
""".replace("__FONT_URL__", FONT_URL)


# --------------------------------------------------------------------------- #
# HTML helpers (all dynamic text is escaped)
# --------------------------------------------------------------------------- #
def _compact(markup: str) -> str:
    """Remove indentation/newlines so Streamlit's Markdown never treats HTML as code."""
    return "".join(line.strip() for line in markup.splitlines())


def inject_css(st) -> None:
    st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


_WAVE_HEIGHTS = (0.35, 0.6, 0.85, 0.5, 1.0, 0.7, 0.4, 0.9, 0.55, 0.75, 0.3, 0.65, 0.45)


def waveform(large: bool = False) -> str:
    bars = "".join(f'<i style="--h:{h};--i:{i}"></i>' for i, h in enumerate(_WAVE_HEIGHTS))
    return f'<span class="echo-wave{" large" if large else ""}" aria-hidden="true">{bars}</span>'


def brand_header() -> str:
    return _compact(
        f"""
        <div class="echo-brand">{waveform()}<span class="echo-wordmark">{config.APP_NAME}</span></div>
        <div class="echo-tagline">{html.escape(config.APP_TAGLINE)}</div>
        """
    )


HERO_BLURB = (
    "A local music studio where an LSTM neural network learns the patterns in the MIDI "
    "music you provide, then composes new melodies note by note. Train it, generate, "
    "download the MIDI, and listen when a synthesizer is set up."
)


def hero() -> str:
    return _compact(
        f"""
        <div class="echo-hero">
          <div class="echo-eyebrow">{html.escape(config.APP_SUBTITLE)}</div>
          <div>{waveform(large=True)}</div>
          <div><span class="echo-wordmark">{config.APP_NAME}</span></div>
          <p class="lead">{html.escape(config.APP_TAGLINE)}</p>
          <p class="sub">{html.escape(HERO_BLURB)}</p>
        </div>
        """
    )


def section(title: str, subtitle: str = "") -> str:
    sub = f"<p>{html.escape(subtitle)}</p>" if subtitle else ""
    return _compact(f'<div class="echo-section"><h2>{html.escape(title)}</h2>{sub}</div>')


def card(title: str, body: str, eyebrow: str = "", hoverable: bool = True) -> str:
    eyebrow_html = f'<div class="echo-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    hover = " hoverable" if hoverable else ""
    return _compact(
        f'<div class="echo-card{hover}">{eyebrow_html}<h4>{html.escape(title)}</h4>'
        f"<p>{html.escape(body)}</p></div>"
    )


def step_card(number: int, title: str, body: str) -> str:
    return _compact(
        f'<div class="echo-card hoverable"><div class="echo-step-num">{number}</div>'
        f"<h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>"
    )


def stat_card(label: str, value: str, hint: str = "") -> str:
    hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
    return _compact(
        f'<div class="echo-card echo-stat"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>{hint_html}</div>'
    )


def pill(text: str, kind: str = "ok") -> str:
    return f'<span class="echo-pill {kind}">{html.escape(text)}</span>'


def grid(cards: Sequence[str]) -> str:
    return f'<div class="echo-grid">{"".join(cards)}</div>'


def flow(labels: Sequence[str]) -> str:
    parts = []
    for i, label in enumerate(labels):
        if i:
            parts.append('<span class="echo-arrow">&rarr;</span>')
        parts.append(f'<span class="echo-chip">{html.escape(label)}</span>')
    return f'<div class="echo-flow">{"".join(parts)}</div>'


# --------------------------------------------------------------------------- #
# Page furniture: header, sidebar blocks, KPIs, checklists, timelines
# --------------------------------------------------------------------------- #
_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(pitch: int) -> str:
    """MIDI number -> scientific pitch name (60 -> 'C4')."""
    return f"{_NOTE_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"


def page_header(eyebrow: str, title: str, subtitle: str = "", chips: Sequence[str] = ()) -> str:
    sub = f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else ""
    chip_html = f'<div class="echo-chips">{"".join(chips)}</div>' if chips else ""
    return _compact(
        f'<div class="echo-pagehead"><div><div class="crumb">{html.escape(eyebrow)}</div>'
        f'<div class="title">{html.escape(title)}</div>{sub}</div>{chip_html}</div>'
    )


def sidebar_brand() -> str:
    return _compact(
        f'<div class="sb-brand">{waveform()}<div><span class="echo-wordmark">{config.APP_NAME}</span>'
        f'<div class="sb-tag">{html.escape(config.APP_TAGLINE)}</div></div></div>'
    )


def sidebar_label(text: str) -> str:
    return f'<div class="sb-label">{html.escape(text)}</div>'


def sidebar_status(rows: Sequence[tuple[str, str, str]]) -> str:
    """rows: (name, state 'ok'|'warn'|'bad', value text). State is also spelled out in the value."""
    body = "".join(
        f'<div class="sb-row"><span class="name"><i class="dot {state}"></i>{html.escape(name)}</span>'
        f'<span class="val">{html.escape(value)}</span></div>'
        for name, state, value in rows
    )
    return f'<div class="sb-panel">{body}</div>'


def sidebar_model(title: str, detail: str) -> str:
    return _compact(
        f'<div class="sb-model"><div class="t">{html.escape(title)}</div>'
        f'<div class="d">{html.escape(detail)}</div></div>'
    )


def sidebar_footer(*lines: str) -> str:
    return f'<div class="sb-foot">{"<br>".join(lines)}</div>'


def kpi(label: str, value: str, hint: str = "", icon: str = "") -> str:
    ico = f'<span class="ico" aria-hidden="true">{html.escape(icon)}</span>' if icon else ""
    hint_html = f'<div class="hint">{html.escape(hint)}</div>' if hint else ""
    return _compact(
        f'<div class="echo-kpi">{ico}<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>{hint_html}</div>'
    )


def kpis(items: Sequence[str]) -> str:
    return f'<div class="echo-kpis">{"".join(items)}</div>'


def checklist(items: Sequence[tuple[bool, str, str]]) -> str:
    """items: (done, title, hint)."""
    rows = []
    for number, (done, title, hint) in enumerate(items, start=1):
        bubble = "&#10003;" if done else str(number)
        state = pill("Done", "ok") if done else pill("To do", "warn")
        rows.append(
            f'<div class="row{" done" if done else ""}"><span class="bubble">{bubble}</span>'
            f'<div><div class="t">{html.escape(title)}</div><div class="h">{html.escape(hint)}</div></div>'
            f'<span class="state">{state}</span></div>'
        )
    return f'<div class="echo-check">{"".join(rows)}</div>'


def kv_grid(pairs: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div><div class="k">{html.escape(k)}</div><div class="v">{html.escape(v)}</div></div>' for k, v in pairs
    )
    return f'<div class="echo-kv">{cells}</div>'


def panel(title: str, inner_html: str, eyebrow: str = "") -> str:
    """A glass card around ready-made (already escaped) HTML such as kv_grid()."""
    eyebrow_html = f'<div class="echo-eyebrow">{html.escape(eyebrow)}</div>' if eyebrow else ""
    return _compact(f'<div class="echo-card">{eyebrow_html}<h4>{html.escape(title)}</h4>{inner_html}</div>')


def timeline(items: Sequence[tuple[str, str]]) -> str:
    body = "".join(
        f'<div class="item"><div class="t">{html.escape(t)}</div><div class="b">{html.escape(b)}</div></div>'
        for t, b in items
    )
    return f'<div class="echo-timeline">{body}</div>'


def tech_grid(items: Sequence[tuple[str, str, str]]) -> str:
    """items: (name, role, why)."""
    body = "".join(
        f'<div class="it"><div class="n">{html.escape(n)}</div><div class="r">{html.escape(r)}</div>'
        f'<div class="w">{html.escape(w)}</div></div>'
        for n, r, w in items
    )
    return f'<div class="echo-tech">{body}</div>'


def step_heading(number: int, title: str, subtitle: str = "") -> str:
    sub = f'<div class="s">{html.escape(subtitle)}</div>' if subtitle else ""
    return _compact(
        f'<div class="echo-stephead"><span class="num">{number}</span>'
        f'<div><div class="t">{html.escape(title)}</div>{sub}</div></div>'
    )


def empty_state(title: str, body: str) -> str:
    return _compact(f'<div class="echo-empty"><div class="t">{html.escape(title)}</div><div class="b">{html.escape(body)}</div></div>')


def file_row(name: str, meta: str) -> str:
    return _compact(f'<div class="echo-file"><div class="n">{html.escape(name)}</div><div class="m">{html.escape(meta)}</div></div>')


def pitch_class_figure(pitch_counts: dict[str, int]):
    """How often each of the 12 note names (C, C#, ...) occurs, regardless of octave."""
    totals = [0] * 12
    for pitch, count in pitch_counts.items():
        totals[int(pitch) % 12] += int(count)
    fig, ax = _new_figure()
    top = max(totals) if any(totals) else 0
    colours = [BLUSH if n.endswith("#") else LAVENDER for n in _NOTE_NAMES]
    ax.bar(range(12), totals, color=colours, width=0.72, edgecolor="none")
    ax.set_xticks(range(12))
    ax.set_xticklabels(_NOTE_NAMES)
    ax.set_xlabel("Note name (sharps in blush)")
    ax.set_ylabel("Notes")
    ax.set_title("Pitch classes", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    if top:
        ax.set_ylim(0, top * 1.12)
    return fig


# --------------------------------------------------------------------------- #
# Matplotlib charts in the ECHO palette
# --------------------------------------------------------------------------- #
def _style_axes(fig, ax) -> None:
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    ax.grid(True, color=GRID, alpha=0.35, linewidth=0.7)
    ax.set_axisbelow(True)


def _new_figure(width: float = 7.2, height: float = 3.3):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Titles added with loc="left" do not follow ax.title, so every set_title call
    # below passes color=TEXT explicitly (dark text would vanish on the dark card).
    fig, ax = plt.subplots(figsize=(width, height), dpi=130)
    _style_axes(fig, ax)
    fig.tight_layout(pad=1.4)
    return fig, ax


def _legend(ax) -> None:
    legend = ax.legend(frameon=False, labelcolor=TEXT, fontsize=9)
    if legend:
        for text in legend.get_texts():
            text.set_color(TEXT)


def loss_figure(history: dict[str, list[float]]):
    """Training vs validation loss per epoch."""
    fig, ax = _new_figure()
    epochs = list(range(1, len(history.get("loss", [])) + 1))
    ax.plot(epochs, history.get("loss", []), color=BLUSH, linewidth=2.4, marker="o", markersize=4, label="Training loss")
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], color=LAVENDER, linewidth=2.4, marker="o", markersize=4, label="Validation loss")
        best = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
        ax.scatter(
            [best + 1], [history["val_loss"][best]], color=ROSE, edgecolors=TEXT, linewidths=1.2,
            s=80, zorder=5, label="Best validation",
        )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning progress", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    if epochs:
        ax.set_xticks(epochs if len(epochs) <= 15 else range(1, len(epochs) + 1, max(1, len(epochs) // 10)))
    _legend(ax)
    return fig


def bar_figure(labels: Sequence[str], values: Sequence[float], title: str, xlabel: str, ylabel: str, color: str = LAVENDER):
    fig, ax = _new_figure()
    ax.bar(range(len(values)), values, color=color, width=0.78, edgecolor="none")
    step = max(1, len(labels) // 14)
    ax.set_xticks(range(0, len(labels), step))
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), step)], rotation=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=TEXT)
    return fig


def pitch_figure(pitch_counts: dict[str, int]):
    """Histogram of note pitches across the dataset."""
    pitches = sorted(int(p) for p in pitch_counts)
    lo, hi = (min(pitches), max(pitches)) if pitches else (config.MIN_PITCH, config.MAX_PITCH)
    labels = [str(p) for p in range(lo, hi + 1)]
    values = [pitch_counts.get(str(p), 0) for p in range(lo, hi + 1)]
    return bar_figure(labels, values, "Pitch distribution", "MIDI note number (60 = middle C)", "Notes", BLUSH)


def duration_figure(duration_counts: dict[str, int]):
    """Counts of each quantised note/rest length (in beats)."""
    labels = [str(d) for d in config.DURATION_CLASSES]
    values = [duration_counts.get(label, 0) for label in labels]
    return bar_figure(labels, values, "Duration distribution", "Length in beats (1 = quarter note)", "Events", LAVENDER)


def piano_roll_figure(events, tempo_bpm: float):
    """Piano-roll preview of generated events (pitch against time)."""
    from matplotlib.colors import LinearSegmentedColormap

    fig, ax = _new_figure(7.2, 3.4)
    cmap = LinearSegmentedColormap.from_list("echo", [BLUSH, LAVENDER, ROSE])
    seconds_per_beat = 60.0 / tempo_bpm
    cursor = 0.0
    pitches = [p for p, _ in events if p != config.REST_PITCH]
    lo, hi = (min(pitches), max(pitches)) if pitches else (60, 72)
    span = max(hi - lo, 1)
    for pitch, duration_index in events:
        length = config.DURATION_CLASSES[duration_index] * seconds_per_beat
        if pitch != config.REST_PITCH:
            ax.broken_barh(
                [(cursor, length * config.NOTE_ARTICULATION)], (pitch - 0.42, 0.84),
                facecolors=cmap((pitch - lo) / span), edgecolor="none",
            )
        cursor += length
    ax.set_xlim(0, max(cursor, 1))
    ax.set_ylim(lo - 2, hi + 2)
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("MIDI pitch")
    ax.set_title("Generated melody", loc="left", fontsize=11, fontweight="bold", color=TEXT)
    return fig
