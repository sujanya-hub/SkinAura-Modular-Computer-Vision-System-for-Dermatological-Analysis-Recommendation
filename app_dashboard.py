"""
app_dashboard.py - SkinAura: AI Analysis Dashboard (v3.1)
==========================================================
Premium dark-theme Streamlit dashboard.

Fixes in v3.1:
  - build_routine() now returns consistent 3-tuples: (icon, name, description)
  - validate_routine_step() / validate_routine() defensive helpers added
  - Checkboxes for Main Concerns are now fully interactive via st.session_state
  - build_json_report() is crash-safe
  - Graceful fallbacks everywhere routine data is consumed
  - Minor UI polish: spacing, icons, layout consistency

Run with:
    streamlit run app_dashboard.py
"""
from __future__ import annotations

import io
import json
import time
from typing import Any

import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SkinAura - AI Dermatology Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PREDICT_API_URL = "https://skinaura-backend.onrender.com/api/v1/predict"
HEALTH_API_URL  = "https://skinaura-backend.onrender.com/api/v1/health"
API_TIMEOUT_SECONDS = 60
CLASS_LABELS = ["Acne", "Pigmentation", "Acne Scars", "Normal"]
CONCERN_OPTIONS = ["Acne", "Pigmentation", "Dark Spots", "Scars", "Uneven Tone"]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&family=Playfair+Display:ital,wght@0,400;0,700;1,400&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background: #0D0D14 !important;
    color: #E0D8F0 !important;
}
.main .block-container {
    padding: 0 1.5rem 3rem !important;
    max-width: 1500px !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarNav"] { display: none !important; }

h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
p, li, span, label, div, button { font-family: 'Sora', sans-serif !important; }
code, pre { font-family: 'DM Mono', monospace !important; }
[data-testid="stMarkdownContainer"] { font-family: 'Sora', sans-serif !important; }

/* ── Progress bars ── */
[data-testid="stProgress"] > div { background: rgba(255,255,255,0.06) !important; border-radius: 3px !important; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #7C3AED, #A855F7) !important; border-radius: 3px !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(124,58,237,0.06) !important;
    border: 1.5px dashed rgba(124,58,237,0.35) !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: #7A7A9A !important; font-family: 'Sora', sans-serif !important; }

/* ── Columns ── */
[data-testid="column"] { padding: 0 0.3rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #A855F7) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Sora', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 20px rgba(124,58,237,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(124,58,237,0.45) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.3); border-radius: 2px; }

/* ── Cards ── */
.sa-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    position: relative;
    height: 100%;
}
.sa-card-accent::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #7C3AED, #A855F7);
    border-radius: 12px 12px 0 0;
}

/* ── Header ── */
.sa-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 0 1rem;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-bottom: 1.2rem;
}
.sa-logo-wrap { display: flex; align-items: center; gap: 0.6rem; }
.sa-logo-icon {
    width: 34px; height: 34px;
    background: linear-gradient(135deg, #7C3AED, #A855F7);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.sa-logo-text {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #A855F7;
}
.sa-logo-sub {
    font-family: 'Sora', sans-serif;
    font-size: 0.68rem;
    color: #5A5A7A;
    margin-top: -2px;
}
.sa-nav { display: flex; gap: 2rem; align-items: center; }
.sa-nav-item {
    font-family: 'Sora', sans-serif;
    font-size: 0.82rem;
    color: #6A6A8A;
    cursor: pointer;
    padding-bottom: 2px;
    transition: color 0.2s;
}
.sa-nav-active {
    color: #A855F7 !important;
    border-bottom: 2px solid #A855F7;
    font-weight: 600;
}
.sa-dl-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Sora', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #E0D8F0;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    padding: 0.45rem 1rem;
    cursor: pointer;
}

/* ── Section labels ── */
.sa-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #4A4A6A;
    margin-bottom: 0.45rem;
}

/* ── Stat card ── */
.sa-stat-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    height: 100%;
}
.sa-stat-icon {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    margin-bottom: 0.6rem;
}
.sa-stat-label { font-family: 'DM Mono', monospace; font-size: 0.6rem; letter-spacing: 0.12em; color: #5A5A7A; text-transform: uppercase; }
.sa-stat-value { font-family: 'Playfair Display', serif; font-size: 1.8rem; font-weight: 700; line-height: 1.1; margin: 0.2rem 0 0.1rem; }
.sa-stat-sub { font-family: 'Sora', sans-serif; font-size: 0.72rem; color: #6A6A8A; }

/* ── Badge ── */
.sa-badge {
    display: inline-block;
    font-family: 'Sora', sans-serif;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    letter-spacing: 0.04em;
    margin-top: 0.3rem;
}
.sa-badge-high   { background: rgba(239,68,68,0.15); color: #EF4444; border: 1px solid rgba(239,68,68,0.25); }
.sa-badge-medium { background: rgba(234,179,8,0.15); color: #EAB308; border: 1px solid rgba(234,179,8,0.25); }
.sa-badge-low    { background: rgba(34,197,94,0.15); color: #22C55E; border: 1px solid rgba(34,197,94,0.25); }

/* ── Alert banner ── */
.sa-alert {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    background: rgba(239,68,68,0.08);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin: 0.8rem 0;
    font-family: 'Sora', sans-serif;
    font-size: 0.8rem;
    color: #EF9999;
}

/* ── Streamlit radio as tab-like ── */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    flex-wrap: wrap !important;
}
div[data-testid="stRadio"] > div > label {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: #5A5A7A !important;
    padding: 0.55rem 1.1rem !important;
    cursor: pointer !important;
    border-bottom: 2px solid transparent !important;
    margin: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    transition: all 0.2s !important;
}
div[data-testid="stRadio"] > div > label:hover { color: #A855F7 !important; }
div[data-testid="stRadio"] > div > label[data-baseweb="radio"] { display: none !important; }
div[data-testid="stRadio"] label > div:first-child { display: none !important; }

/* ── Streamlit checkbox — styled to match design ── */
[data-testid="stCheckbox"] {
    padding: 0.18rem 0 !important;
}
[data-testid="stCheckbox"] label {
    font-family: 'Sora', sans-serif !important;
    font-size: 0.78rem !important;
    color: #8A8A9A !important;
    gap: 0.5rem !important;
    align-items: center !important;
}
[data-testid="stCheckbox"] label:hover {
    color: #E0D8F0 !important;
}
[data-testid="stCheckbox"] svg {
    stroke: #7C3AED !important;
}
/* Checkbox box itself */
[data-testid="stCheckbox"] span[data-testid="stCheckboxCheckmark"] {
    background-color: #7C3AED !important;
    border-color: #7C3AED !important;
    border-radius: 3px !important;
}
[data-baseweb="checkbox"] > div:first-child {
    background-color: transparent !important;
    border: 1.5px solid rgba(168,85,247,0.4) !important;
    border-radius: 3px !important;
    width: 14px !important;
    height: 14px !important;
}
[data-baseweb="checkbox"][aria-checked="true"] > div:first-child {
    background-color: #7C3AED !important;
    border-color: #7C3AED !important;
}

/* ── Severity bar ── */
.sa-sev-row { margin-bottom: 0.8rem; }
.sa-sev-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.3rem; }
.sa-sev-name { font-family: 'Sora', sans-serif; font-size: 0.78rem; color: #8A8A9A; }
.sa-sev-level { font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 500; }
.sa-sev-bar-bg { height: 5px; background: rgba(255,255,255,0.07); border-radius: 3px; overflow: hidden; }
.sa-sev-bar-fill { height: 100%; border-radius: 3px; }

/* ── Confidence class row ── */
.sa-class-row { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.6rem; }
.sa-class-name { font-family: 'Sora', sans-serif; font-size: 0.75rem; color: #8A8A9A; min-width: 90px; }
.sa-class-bar-bg { flex: 1; height: 4px; background: rgba(255,255,255,0.06); border-radius: 2px; overflow: hidden; }
.sa-class-bar-fill { height: 100%; border-radius: 2px; }
.sa-class-pct { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #5A5A6E; min-width: 32px; text-align: right; }

/* ── Routine product row ── */
.sa-routine-product {
    display: flex;
    align-items: flex-start;
    gap: 0.8rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.sa-routine-product:last-child { border-bottom: none; }
.sa-routine-product-icon {
    width: 38px; height: 38px;
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}
.sa-routine-product-name { font-family: 'Sora', sans-serif; font-size: 0.8rem; font-weight: 600; color: #E0D8F0; }
.sa-routine-product-desc { font-family: 'Sora', sans-serif; font-size: 0.7rem; color: #5A5A7A; margin-top: 0.08rem; line-height: 1.4; }

/* ── Diet card ── */
.sa-diet-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    height: 100%;
}
.sa-diet-title {
    font-family: 'Sora', sans-serif;
    font-size: 0.82rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.sa-diet-item {
    font-family: 'Sora', sans-serif;
    font-size: 0.73rem;
    color: #7A7A9A;
    padding: 0.22rem 0;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.sa-diet-dot { width: 5px; height: 5px; border-radius: 50%; flex-shrink: 0; }

/* ── Product card ── */
.sa-product-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 0.8rem;
    text-align: center;
    height: 100%;
}
.sa-product-img {
    width: 52px; height: 68px;
    background: rgba(255,255,255,0.06);
    border-radius: 6px;
    margin: 0 auto 0.5rem;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
}
.sa-product-name { font-family: 'Sora', sans-serif; font-size: 0.72rem; font-weight: 600; color: #E0D8F0; margin-bottom: 0.2rem; }
.sa-product-type { font-family: 'Sora', sans-serif; font-size: 0.65rem; color: #5A5A7A; margin-bottom: 0.35rem; }
.sa-product-price { font-family: 'DM Mono', monospace; font-size: 0.75rem; color: #A855F7; font-weight: 500; }

/* ── Status dot ── */
.sa-status-row { display: flex; align-items: center; gap: 0.4rem; font-family: 'Sora', sans-serif; font-size: 0.75rem; color: #7A7A9A; margin-bottom: 0.3rem; }
.sa-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
.sa-dot-green { background: #22C55E; box-shadow: 0 0 5px rgba(34,197,94,0.5); }
.sa-dot-red   { background: #EF4444; box-shadow: 0 0 5px rgba(239,68,68,0.5); }

/* ── Divider ── */
.sa-divider { height: 1px; background: rgba(255,255,255,0.05); margin: 0.9rem 0; }

/* ── Consistency banner ── */
.sa-banner {
    background: linear-gradient(90deg, #4C1D95, #5B21B6, #6D28D9);
    border-radius: 10px;
    padding: 0.85rem 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.75rem;
    margin-top: 1rem;
}
.sa-banner-text { font-family: 'Sora', sans-serif; font-size: 0.82rem; font-weight: 600; color: #E9D5FF; text-align: center; }
.sa-banner-sub  { font-family: 'Sora', sans-serif; font-size: 0.72rem; color: rgba(233,213,255,0.65); text-align: center; }

/* ── Image wrapper ── */
.sa-image-wrapper {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07);
    background: rgba(255,255,255,0.02);
}

/* ── Insight box ── */
.sa-insight-box {
    background: rgba(124,58,237,0.06);
    border: 1px solid rgba(124,58,237,0.18);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
.sa-insight-title { font-family: 'Sora', sans-serif; font-size: 0.82rem; font-weight: 600; color: #A855F7; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.4rem; }
.sa-insight-text { font-family: 'Sora', sans-serif; font-size: 0.78rem; color: #8A8A9A; line-height: 1.65; }

/* ── Routine period header ── */
.sa-routine-period {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Sora', sans-serif;
    font-size: 0.85rem;
    font-weight: 700;
    color: #E0D8F0;
    margin: 1rem 0 0.6rem;
}
.sa-routine-period:first-child { margin-top: 0; }

/* ── Why routine box ── */
.sa-why-box {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-top: 0.75rem;
}
.sa-why-title { font-family: 'Sora', sans-serif; font-size: 0.78rem; font-weight: 600; color: #A855F7; margin-bottom: 0.3rem; display: flex; align-items: center; gap: 0.4rem; }
.sa-why-text { font-family: 'Sora', sans-serif; font-size: 0.73rem; color: #6A6A8A; line-height: 1.55; }

/* ── Tip / Avoid items ── */
.sa-tip-item {
    display: flex; align-items: flex-start; gap: 0.6rem;
    padding: 0.65rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-family: 'Sora', sans-serif; font-size: 0.78rem; color: #8A8A9A; line-height: 1.5;
}
.sa-tip-item:last-child { border-bottom: none; }
.sa-tip-num { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #A855F7; min-width: 18px; margin-top: 1px; }

.sa-avoid-item {
    display: flex; align-items: flex-start; gap: 0.6rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-family: 'Sora', sans-serif; font-size: 0.78rem; color: #8A8A9A;
}
.sa-avoid-item:last-child { border-bottom: none; }
.sa-avoid-x { color: #EF4444; font-weight: 700; margin-top: 1px; }

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(255,255,255,0.05) !important;
    color: #C0B8D0 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    box-shadow: none !important;
    font-size: 0.75rem !important;
    padding: 0.45rem 0.8rem !important;
}
.stDownloadButton > button:hover {
    background: rgba(168,85,247,0.1) !important;
    border-color: rgba(168,85,247,0.3) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Selectbox ── */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E0D8F0 !important;
    font-family: 'Sora', sans-serif !important;
    font-size: 0.82rem !important;
}

/* ── Analysis summary header ── */
.sa-summary-header {
    display: flex; align-items: center; gap: 0.8rem; margin-bottom: 1rem;
}
.sa-summary-icon {
    width: 38px; height: 38px;
    background: rgba(124,58,237,0.15);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.sa-summary-title { font-family: 'Playfair Display', serif; font-size: 1.3rem; font-weight: 700; color: #E0D8F0; }
.sa-summary-sub { font-family: 'Sora', sans-serif; font-size: 0.72rem; color: #5A5A7A; }

/* ── Profile section ── */
.sa-profile-section-title {
    font-family: 'Sora', sans-serif;
    font-size: 0.78rem;
    font-weight: 600;
    color: #8A8A9A;
    margin: 0.85rem 0 0.3rem;
}
.sa-concerns-hint {
    font-family: 'Sora', sans-serif;
    font-size: 0.65rem;
    color: #4A4A6A;
    margin-bottom: 0.5rem;
}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def card_open(accent: bool = False, extra: str = "") -> None:
    cls = "sa-card sa-card-accent" if accent else "sa-card"
    st.markdown(f'<div class="{cls}" style="{extra}">', unsafe_allow_html=True)

def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)

def sa_label(text: str) -> None:
    st.markdown(f'<div class="sa-label">{text}</div>', unsafe_allow_html=True)

def sa_divider() -> None:
    st.markdown('<div class="sa-divider"></div>', unsafe_allow_html=True)

def normalize(s: str) -> str:
    return s.replace("_", " ").replace("-", " ").title().strip() or "Unknown"

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))

def sev_color(prob: float) -> tuple[str, str]:
    if prob >= 0.7:   return "High",   "#EF4444"
    if prob >= 0.4:   return "Medium", "#EAB308"
    return "Low", "#22C55E"

def badge_cls(level: str) -> str:
    return {"High": "sa-badge-high", "Medium": "sa-badge-medium", "Low": "sa-badge-low"}.get(level, "sa-badge-low")


# ---------------------------------------------------------------------------
# ROUTINE VALIDATION HELPERS
# ---------------------------------------------------------------------------

# A valid routine step is a 3-tuple: (icon: str, name: str, description: str)
RoutineStep = tuple[str, str, str]

FALLBACK_STEP: RoutineStep = ("🧴", "Skincare Step", "Apply as directed.")

def validate_routine_step(step: Any) -> RoutineStep:
    """
    Coerce any value into a safe (icon, name, description) 3-tuple.
    Handles: correct 3-tuples, legacy 2-tuples (name, desc), dicts, and garbage.
    """
    if isinstance(step, (list, tuple)):
        if len(step) == 3:
            return (str(step[0]) or "🧴", str(step[1]) or "Step", str(step[2]) or "")
        if len(step) == 2:
            # Legacy 2-tuple: (name, description) — inject a default icon
            return ("🧴", str(step[0]) or "Step", str(step[1]) or "")
        if len(step) == 1:
            return ("🧴", str(step[0]) or "Step", "")
    if isinstance(step, dict):
        return (
            str(step.get("icon", "🧴")),
            str(step.get("name", "Step")),
            str(step.get("description", "")),
        )
    return FALLBACK_STEP


def validate_routine(routine: Any) -> dict[str, list[RoutineStep]]:
    """
    Ensure the full routine dict is safe: {period: [RoutineStep, ...]}
    Returns an empty-safe dict even if input is None or malformed.
    """
    if not isinstance(routine, dict):
        return {}
    result: dict[str, list[RoutineStep]] = {}
    for period, steps in routine.items():
        if not isinstance(steps, (list, tuple)):
            result[str(period)] = []
            continue
        result[str(period)] = [validate_routine_step(s) for s in steps]
    return result


# ---------------------------------------------------------------------------
# DATA / LOGIC
# ---------------------------------------------------------------------------

def call_predict_api(image_bytes: bytes, filename: str) -> dict[str, Any]:
    ext  = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    mime = "image/png" if ext == "png" else "image/jpeg"
    try:
        r = requests.post(
            PREDICT_API_URL,
            files={"image": (filename, image_bytes, mime)},
            timeout=API_TIMEOUT_SECONDS,
        )
        r.raise_for_status()
    except requests.Timeout:
        return {"success": False, "error": "Request timed out."}
    except requests.ConnectionError:
        return {"success": False, "error": "API not reachable"}
    except requests.HTTPError as exc:
        detail = None
        try:
            detail = r.json().get("detail")
        except Exception:
            pass
        return {"success": False, "error": f"HTTP error: {detail or exc}"}
    except requests.RequestException as exc:
        return {"success": False, "error": str(exc)}

    try:
        payload = r.json()
    except ValueError:
        return {"success": False, "error": "Invalid JSON from API."}

    pred = payload.get("prediction")
    conf = payload.get("confidence")
    rec  = payload.get("recommendation")
    if not isinstance(pred, str) or not isinstance(rec, str):
        return {"success": False, "error": "Missing fields in API response."}
    try:
        conf_val = clamp(float(conf), 0.0, 1.0)
    except (TypeError, ValueError):
        return {"success": False, "error": "Invalid confidence score."}

    return {
        "success": True,
        "data": {
            "prediction":     normalize(pred),
            "confidence":     conf_val,
            "recommendation": rec.strip(),
        },
    }


def fetch_system_status() -> dict[str, bool]:
    try:
        r = requests.get(HEALTH_API_URL, timeout=3)
        r.raise_for_status()
        p = r.json()
    except Exception:
        return {"model": False, "api": False}
    services = p.get("services", {}) if isinstance(p, dict) else {}
    return {
        "model": services.get("model_loader") == "loaded",
        "api":   p.get("status") in {"ok", "degraded"},
    }


def simulate_class_probs(prediction: str, confidence: float) -> dict[str, float]:
    primary = normalize(prediction)
    top = clamp(confidence, 0.55, 0.99)
    rem = 1.0 - top
    profiles = {
        "Acne":        {"Pigmentation": 0.50, "Acne Scars": 0.35, "Normal": 0.15},
        "Pigmentation":{"Normal": 0.50, "Acne Scars": 0.28, "Acne": 0.22},
        "Acne Scars":  {"Acne": 0.48, "Pigmentation": 0.27, "Normal": 0.25},
        "Normal":      {"Pigmentation": 0.38, "Acne": 0.34, "Acne Scars": 0.28},
    }
    profile = profiles.get(primary, {})
    others  = [l for l in CLASS_LABELS if l != primary]
    total_w = sum(profile.get(l, 1.0) for l in others) or 1.0
    probs: dict[str, float] = {primary: top}
    for l in others:
        probs[l] = rem * profile.get(l, 1.0) / total_w
    total = sum(probs.values()) or 1.0
    return dict(sorted({k: v / total for k, v in probs.items()}.items(), key=lambda x: x[1], reverse=True))


def build_severity(
    prediction: str,
    confidence: float,
    class_probs: dict[str, float],
) -> dict[str, tuple[str, str, float]]:
    sec_prob = list(class_probs.values())[1] if len(class_probs) > 1 else 0.1
    base = {
        "Acne":        {"Inflammation": 0.55, "Coverage": 0.48, "Scarring Risk": 0.42, "Pigmentation": 0.22},
        "Pigmentation":{"Inflammation": 0.18, "Coverage": 0.56, "Scarring Risk": 0.32, "Pigmentation": 0.68},
        "Acne Scars":  {"Inflammation": 0.22, "Coverage": 0.40, "Scarring Risk": 0.62, "Pigmentation": 0.38},
        "Normal":      {"Inflammation": 0.08, "Coverage": 0.12, "Scarring Risk": 0.10, "Pigmentation": 0.10},
    }
    profile = base.get(normalize(prediction), base["Normal"])
    result: dict[str, tuple[str, str, float]] = {}
    for metric, bv in profile.items():
        val = clamp(bv + confidence * 0.35 + sec_prob * 0.10, 0.05, 0.95)
        level, color = sev_color(val)
        result[metric] = (level, color, val)
    return result


def build_ai_insight(
    prediction: str,
    confidence: float,
    secondary_label: str,
    secondary_prob: float,
) -> str:
    primary = normalize(prediction)
    msgs = {
        "Acne":         "Detected clustered inflammation and pore congestion patterns consistent with active acne activity, especially across oil-prone facial regions. Early intervention can prevent scarring and post-inflammatory hyperpigmentation.",
        "Pigmentation": "Detected uneven melanin distribution and tonal variation suggestive of pigmentation irregularity rather than active inflammatory lesions.",
        "Acne Scars":   "Detected residual textural irregularity and post-inflammatory markings consistent with acne scar formation and healing changes.",
        "Normal":       "Detected balanced skin texture without a dominant lesion pattern, with overall features aligning to normal skin presentation.",
    }
    certainty = "strong" if confidence >= 0.8 else "moderate"
    sec_text  = ""
    if secondary_label != primary and secondary_prob >= 0.12:
        sec_text = f" Secondary overlap with {secondary_label.lower()} is present, which may reflect mixed visual features."
    return f"{msgs.get(primary, msgs['Normal'])} Model confidence is {confidence:.0%} — a {certainty} match.{sec_text}"


# ---------------------------------------------------------------------------
# BUILD ROUTINE  ← THE PRIMARY FIX
# All steps are 3-tuples: (emoji_icon, product_name, description)
# ---------------------------------------------------------------------------

def build_routine(prediction: str) -> dict[str, list[RoutineStep]]:
    routines: dict[str, dict[str, list[RoutineStep]]] = {
        "Acne": {
            "Morning": [
                ("🧴", "Salicylic Acid Cleanser",   "Unclogs pores and removes excess oil"),
                ("💧", "Niacinamide Serum",          "Controls oil production and reduces redness"),
                ("🤍", "Lightweight Moisturiser",    "Hydrates without clogging pores"),
                ("☀️", "Sunscreen SPF 50",           "Protects from UV rays and prevents marks"),
            ],
            "Night": [
                ("🌿", "Gentle Cleanser",            "Removes dirt, oil and impurities"),
                ("🎯", "Adapalene Spot Treatment",   "Reduces active acne and prevents breakouts"),
                ("🤍", "Oil-free Moisturiser",       "Repairs skin barrier and locks in hydration"),
            ],
        },
        "Pigmentation": {
            "Morning": [
                ("🧴", "Brightening Cleanser",       "Gentle cleanse that doesn't trigger irritation"),
                ("✨", "Vitamin C Serum",             "Antioxidant that targets uneven tone and dullness"),
                ("🤍", "Lightweight Moisturiser",    "Hydrates without clogging pores"),
                ("☀️", "Sunscreen SPF 50",           "Prevents pigmentation from deepening"),
            ],
            "Night": [
                ("🌿", "Hydrating Cleanser",         "Gentle foaming cleanse before actives"),
                ("💫", "Tranexamic Acid Serum",      "Targets melanin production and dark spots"),
                ("🤍", "Barrier Moisturiser",        "Ceramide-rich hydration to reduce irritation"),
            ],
        },
        "Acne Scars": {
            "Morning": [
                ("🧴", "Gentle Cleanser",            "Low-stripping cleanse to protect healing skin"),
                ("🔬", "Peptide Repair Serum",       "Supports smoother-looking skin texture"),
                ("🤍", "Lightweight Moisturiser",    "Hydrates and supports skin recovery"),
                ("☀️", "Sunscreen SPF 50",           "Prevents scar marks from appearing darker"),
            ],
            "Night": [
                ("💦", "Hydrating Toner",            "Adds comfort before reparative actives"),
                ("🌙", "Retinoid Treatment",         "Retinoid-based renewal to improve scar appearance"),
                ("🤍", "Barrier Moisturiser",        "Minimises irritation and supports recovery"),
            ],
        },
        "Normal": {
            "Morning": [
                ("🧴", "Gentle Cleanser",            "Removes overnight buildup, maintains barrier"),
                ("✨", "Antioxidant Serum",          "Vitamin C for preventive daily care"),
                ("🤍", "Lightweight Moisturiser",    "Keeps skin smooth and balanced"),
                ("☀️", "Sunscreen SPF 50",           "Protects against UV damage"),
            ],
            "Night": [
                ("🌿", "Micellar Cleanse",           "Lifts sunscreen and impurities gently"),
                ("💧", "Hyaluronic Acid Serum",      "Maintains smoothness and water balance"),
                ("🤍", "Barrier Moisturiser",        "Keeps skin calm, balanced, and resilient"),
            ],
        },
    }
    raw = routines.get(normalize(prediction), routines["Normal"])
    # Always pass through validator to guarantee safe 3-tuples
    return validate_routine(raw)


# ---------------------------------------------------------------------------
# REMAINING DATA BUILDERS (unchanged logic, safe)
# ---------------------------------------------------------------------------

def build_diet(prediction: str) -> dict[str, Any]:
    data = {
        "Acne": {
            "eat":       ["Fruits (Berries, Apple, Papaya)", "Leafy Greens", "Nuts & Seeds", "Omega-3 Rich Foods", "Green Tea"],
            "avoid":     ["Sugar & Refined Carbs", "Dairy (Milk, Cheese)", "Fried & Oily Foods", "High Glycemic Foods", "Excess Salt"],
            "lifestyle": ["Sleep 7-8 hours", "Manage Stress", "Exercise Regularly", "Keep pillowcases clean", "Don't touch your face"],
            "hydration": "Drink 2–3L\nwater daily",
        },
        "Pigmentation": {
            "eat":       ["Vitamin C rich foods", "Tomatoes & Carrots", "Dark Leafy Greens", "Flaxseeds & Walnuts", "Green Tea"],
            "avoid":     ["Processed foods", "Alcohol", "Excess caffeine", "High sugar foods", "Smoking"],
            "lifestyle": ["Sleep 7-8 hours", "Wear SPF daily", "Stay hydrated", "Manage stress", "Avoid midday sun"],
            "hydration": "Drink 2–3L\nwater daily",
        },
        "Acne Scars": {
            "eat":       ["Vitamin E foods (nuts, seeds)", "Collagen-rich broth", "Berries & antioxidants", "Zinc-rich foods", "Leafy greens"],
            "avoid":     ["High sugar foods", "Processed snacks", "Excess dairy", "Fried foods", "Alcohol"],
            "lifestyle": ["Sleep 8 hours", "Gentle exercise", "Avoid picking skin", "Manage stress", "Consistent routine"],
            "hydration": "Drink 2–3L\nwater daily",
        },
        "Normal": {
            "eat":       ["Varied whole foods", "Fruits & Vegetables", "Lean proteins", "Healthy fats", "Plenty of water"],
            "avoid":     ["Highly processed foods", "Excess sugar", "Alcohol", "Trans fats", "Excess caffeine"],
            "lifestyle": ["Sleep 7-8 hours", "Regular exercise", "Stress management", "Consistent skincare", "Annual skin check"],
            "hydration": "Drink 2–3L\nwater daily",
        },
    }
    return data.get(normalize(prediction), data["Normal"])


def build_products(prediction: str) -> list[dict]:
    products = {
        "Acne": [
            {"icon": "🧴", "name": "Minimalist Salicylic Acid Cleanser", "type": "Cleanser",    "price": "₹349"},
            {"icon": "💧", "name": "The Derma Co Niacinamide Serum",     "type": "Serum",       "price": "₹499"},
            {"icon": "☀️", "name": "La Shield SPF 50 Gel",              "type": "Sunscreen",   "price": "₹699"},
            {"icon": "🤍", "name": "Cetaphil Oil Free Moisturizer",      "type": "Moisturiser", "price": "₹549"},
        ],
        "Pigmentation": [
            {"icon": "✨", "name": "Minimalist Vitamin C 10%",           "type": "Serum",       "price": "₹399"},
            {"icon": "💫", "name": "The Ordinary Tranexamic Acid",       "type": "Treatment",   "price": "₹599"},
            {"icon": "☀️", "name": "Re'equil Mineral Sunscreen SPF50",  "type": "Sunscreen",   "price": "₹449"},
            {"icon": "🤍", "name": "Neutrogena Hydro Boost",            "type": "Moisturiser", "price": "₹749"},
        ],
        "Acne Scars": [
            {"icon": "🌙", "name": "Minimalist Retinol 0.3%",           "type": "Treatment",   "price": "₹549"},
            {"icon": "💫", "name": "The Ordinary Peptide Complex",      "type": "Serum",       "price": "₹899"},
            {"icon": "☀️", "name": "Neutrogena Ultra Sheer SPF50",     "type": "Sunscreen",   "price": "₹499"},
            {"icon": "🤍", "name": "CeraVe Moisturising Cream",        "type": "Moisturiser", "price": "₹799"},
        ],
        "Normal": [
            {"icon": "✨", "name": "Minimalist Vitamin C 10%",          "type": "Serum",       "price": "₹399"},
            {"icon": "☀️", "name": "La Shield SPF 50 Gel",             "type": "Sunscreen",   "price": "₹699"},
            {"icon": "🤍", "name": "Cetaphil Moisturising Lotion",     "type": "Moisturiser", "price": "₹449"},
            {"icon": "🧼", "name": "CeraVe Hydrating Cleanser",        "type": "Cleanser",    "price": "₹649"},
        ],
    }
    return products.get(normalize(prediction), products["Normal"])


def build_avoid(prediction: str) -> list[str]:
    items = {
        "Acne":        ["Harsh scrubs and physical exfoliants", "Heavy, pore-clogging moisturisers", "Touching or popping pimples", "Skipping sunscreen (worsens post-acne marks)", "Overwashing — more than twice daily strips barrier", "Products with fragrance or alcohol on active breakouts"],
        "Pigmentation":["Direct sun exposure without SPF", "Picking or scratching hyperpigmented areas", "Harsh bleaching agents without dermatologist guidance", "Mixing too many actives at once (irritation)", "Skipping moisturiser when using brightening actives", "Inconsistent routine — pigmentation needs consistent care"],
        "Acne Scars":  ["Over-exfoliation — damages healing skin", "Skipping SPF (scars darken without protection)", "Aggressive scrubbing over textured areas", "Starting retinoids without gradual introduction", "Combining retinoids + AHA/BHA on same night", "Picking at any residual breakouts"],
        "Normal":      ["Over-cleansing or using harsh cleansers", "Skipping sunscreen on cloudy days", "Using too many active ingredients at once", "Neglecting neck and chest in skincare", "Hot water on face — disrupts barrier", "Changing products too frequently"],
    }
    return items.get(normalize(prediction), items["Normal"])


def build_tips(prediction: str) -> list[str]:
    tips = {
        "Acne": [
            "Change your pillowcase every 2-3 days to reduce bacterial transfer.",
            "Always apply sunscreen — UV exposure worsens post-acne marks significantly.",
            "Introduce actives (BHA, BP) one at a time to identify irritants.",
            "Avoid touching your face throughout the day — hands carry bacteria.",
            "Patience is key — most treatments show results after 8-12 weeks.",
            "Track your diet and breakout patterns — food triggers vary per person.",
        ],
        "Pigmentation": [
            "SPF 50 every single morning is non-negotiable for pigmentation control.",
            "Vitamin C works best on freshly cleansed skin in the morning.",
            "Don't layer multiple brightening actives initially — introduce one at a time.",
            "Wear a physical barrier (hat, seek shade) beyond just sunscreen.",
            "Skin cycling — use actives every other night to prevent irritation.",
            "Results from pigmentation treatments typically take 12-16 weeks consistently.",
        ],
        "Acne Scars": [
            "Introduce retinoids slowly — start 2x/week and increase gradually.",
            "Never skip SPF — sun darkens post-inflammatory marks significantly.",
            "Hydration is critical when using retinoids — sandwich method helps.",
            "Professional treatments (microneedling, chemical peels) can accelerate results.",
            "Consistency is everything — skip one week and you lose two weeks of progress.",
            "Antioxidant serums (Vitamin C) in the AM pair excellently with PM retinoids.",
        ],
        "Normal": [
            "Consistent minimal routine beats a complex one you skip half the time.",
            "SPF daily protects your skin from premature aging and tone changes.",
            "A good cleanser, moisturiser, and SPF is all most skin truly needs.",
            "Introduce new products one at a time — 2-3 weeks per product minimum.",
            "Your skin barrier is your most important asset — protect it.",
            "Annual dermatologist check-in helps catch any changes early.",
        ],
    }
    return tips.get(normalize(prediction), tips["Normal"])


# ---------------------------------------------------------------------------
# VIEW MODEL BUILDER
# ---------------------------------------------------------------------------

def build_view_model(api_data: dict[str, Any]) -> dict[str, Any]:
    pred  = normalize(str(api_data["prediction"]))
    conf  = clamp(float(api_data["confidence"]), 0.0, 1.0)
    rec   = str(api_data["recommendation"]).strip()

    class_probs = simulate_class_probs(pred, conf)
    items       = list(class_probs.items())
    primary     = items[0]
    secondary   = items[1] if len(items) > 1 else items[0]
    routine     = build_routine(pred)   # already validated (safe 3-tuples)

    return {
        "prediction":     primary[0],
        "confidence":     primary[1],
        "recommendation": rec,
        "class_probs":    class_probs,
        "primary":        primary,
        "secondary":      secondary,
        "severity":       build_severity(pred, conf, class_probs),
        "ai_insight":     build_ai_insight(pred, conf, secondary[0], secondary[1]),
        "routine":        routine,
        "diet":           build_diet(pred),
        "products":       build_products(pred),
        "avoid":          build_avoid(pred),
        "tips":           build_tips(pred),
        "raw":            api_data,
    }


# ---------------------------------------------------------------------------
# REPORT BUILDERS  (crash-safe)
# ---------------------------------------------------------------------------

def build_text_report(r: dict[str, Any]) -> str:
    lines = [
        "SkinAura AI Report",
        "=" * 40,
        f"Condition:  {r['prediction']}",
        f"Confidence: {r['confidence']:.0%}",
        f"Primary:    {r['primary'][0]} ({r['primary'][1]:.0%})",
        f"Secondary:  {r['secondary'][0]} ({r['secondary'][1]:.0%})",
        "",
        "Recommendation:",
        r["recommendation"],
        "",
        "AI Insight:",
        r["ai_insight"],
        "",
        "Severity:",
    ]
    for k, (level, _, prob) in r["severity"].items():
        lines.append(f"  {k}: {level} ({prob:.0%})")
    lines += ["", "Things to Avoid:"]
    for item in r["avoid"]:
        lines.append(f"  - {item}")
    lines += ["", "Skin Tips:"]
    for i, tip in enumerate(r["tips"], 1):
        lines.append(f"  {i}. {tip}")
    return "\n".join(lines)


def build_json_report(r: dict[str, Any]) -> str:
    """Crash-safe JSON export. Validates routine before serialising."""
    safe_routine = validate_routine(r.get("routine", {}))

    routine_export: dict[str, list[dict]] = {}
    for period, steps in safe_routine.items():
        routine_export[period] = [
            {"icon": ic, "name": n, "description": d}
            for ic, n, d in steps      # guaranteed 3-tuple by validate_routine
        ]

    payload = {
        "prediction":           r["prediction"],
        "confidence":           r["confidence"],
        "recommendation":       r["recommendation"],
        "primary":   {"label": r["primary"][0],   "probability": r["primary"][1]},
        "secondary": {"label": r["secondary"][0],  "probability": r["secondary"][1]},
        "class_probabilities":  r["class_probs"],
        "severity": {
            k: {"level": lv, "probability": pb}
            for k, (lv, _, pb) in r["severity"].items()
        },
        "ai_insight":  r["ai_insight"],
        "routine":     routine_export,
        "diet":        r["diet"],
        "products":    r["products"],
        "avoid":       r["avoid"],
        "tips":        r["tips"],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# IMAGE READER
# ---------------------------------------------------------------------------

def read_image(uploaded_file: Any) -> tuple[Image.Image | None, bytes | None]:
    try:
        b   = uploaded_file.read()
        img = Image.open(io.BytesIO(b)).convert("RGB")
        return img, b
    except (UnidentifiedImageError, OSError):
        return None, None


# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

DEFAULTS: dict[str, Any] = {
    "uploaded_image":       None,
    "uploaded_image_bytes": None,
    "uploaded_filename":    "",
    "analysis_result":      None,
    "analysed":             False,
    "api_error":            "",
    "active_tab":           "Analysis",
    "show_heatmap":         False,
    "skin_type":            "Oily",
    # Concern toggles — one key per concern for proper Streamlit reactivity
    "concern_Acne":         True,
    "concern_Pigmentation": True,
    "concern_Dark Spots":   False,
    "concern_Scars":        False,
    "concern_Uneven Tone":  False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ---------------------------------------------------------------------------
# TOP BAR
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="sa-topbar">
        <div class="sa-logo-wrap">
            <div class="sa-logo-icon">✦</div>
            <div>
                <div class="sa-logo-text">SkinAura</div>
                <div class="sa-logo-sub">AI Dermatology Assistant</div>
            </div>
        </div>
        <div class="sa-nav">
            <span class="sa-nav-item sa-nav-active">Dashboard</span>
            <span class="sa-nav-item">History</span>
            <span class="sa-nav-item">Reports</span>
            <span class="sa-nav-item">About</span>
        </div>
        <div class="sa-dl-btn">⬇ Download Report</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# 3-COLUMN LAYOUT
# ---------------------------------------------------------------------------

system_status = fetch_system_status()

col_left, col_center, col_right = st.columns([1.0, 2.5, 1.2], gap="medium")

# ===========================================================================
# LEFT SIDEBAR
# ===========================================================================
with col_left:

    # ── 1. Upload Image ──────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:\'Sora\',sans-serif;font-size:0.85rem;font-weight:600;'
        'color:#E0D8F0;margin-bottom:0.5rem;">1. Upload Image'
        '&nbsp;<span style="font-size:0.7rem;color:#5A5A7A;">ⓘ</span></div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload face photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="face_upload",
    )
    if uploaded is not None:
        img, imgb = read_image(uploaded)
        if img is None:
            st.error("Not a valid image.")
        else:
            st.session_state["uploaded_image"]       = img
            st.session_state["uploaded_image_bytes"] = imgb
            st.session_state["uploaded_filename"]    = uploaded.name
            st.session_state["api_error"]            = ""

    cur_img  = st.session_state["uploaded_image"]
    cur_fn   = st.session_state["uploaded_filename"]
    cur_imgb = st.session_state["uploaded_image_bytes"]

    if cur_img is not None:
        st.markdown('<div class="sa-image-wrapper">', unsafe_allow_html=True)
        st.image(cur_img, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        w, h = cur_img.size
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;color:#3A3A4E;'
            f'margin-top:0.4rem;text-align:center;">{cur_fn} · {w}×{h}px</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("✦  Analyse", disabled=(cur_imgb is None), use_container_width=True, key="analyse_btn"):
            pb     = st.progress(0)
            st_txt = st.empty()
            try:
                st_txt.caption("Uploading…");        pb.progress(15); time.sleep(0.1)
                st_txt.caption("Running inference…"); pb.progress(45)
                api_res = call_predict_api(cur_imgb, cur_fn)
                if not api_res["success"]:
                    st.session_state["api_error"] = str(api_res["error"])
                    st.error("API not reachable")
                    system_status = {"model": False, "api": False}
                    st.session_state["analysed"] = st.session_state["analysis_result"] is not None
                else:
                    st_txt.caption("Building insights…"); pb.progress(82); time.sleep(0.1)
                    st.session_state["analysis_result"] = build_view_model(api_res["data"])
                    st.session_state["analysed"]        = True
                    st.session_state["api_error"]       = ""
                    st.session_state["active_tab"]      = "Analysis"
                    system_status = {"model": True, "api": True}
                    pb.progress(100); time.sleep(0.1)
            finally:
                pb.empty(); st_txt.empty()

    with col_b:
        if st.button("↺ Re-analyse", disabled=(cur_imgb is None), use_container_width=True, key="reanalyse_btn"):
            st.session_state["analysed"]        = False
            st.session_state["analysis_result"] = None
            st.rerun()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── 2. Profile ────────────────────────────────────────────────────
    st.markdown(
        '<div style="font-family:\'Sora\',sans-serif;font-size:0.85rem;font-weight:600;'
        'color:#E0D8F0;margin-bottom:0.5rem;">2. Your Profile'
        '&nbsp;<span style="font-size:0.7rem;color:#5A5A7A;">ⓘ</span></div>',
        unsafe_allow_html=True,
    )

    card_open()
    sa_label("Skin Type")
    skin_options = ["Oily", "Dry", "Combination", "Normal", "Sensitive"]
    st.session_state["skin_type"] = st.selectbox(
        "skin_type",
        skin_options,
        index=skin_options.index(st.session_state["skin_type"]),
        label_visibility="collapsed",
        key="skin_type_sel",
    )

    # ── Main Concerns — real interactive Streamlit checkboxes ──
    st.markdown(
        '<div class="sa-profile-section-title">Main Concerns</div>'
        '<div class="sa-concerns-hint">Select all that apply</div>',
        unsafe_allow_html=True,
    )

    for concern in CONCERN_OPTIONS:
        key = f"concern_{concern}"
        st.checkbox(
            concern,
            value=st.session_state[key],
            key=key,
        )

    sa_divider()
    if st.button("✦  Update & Personalise", use_container_width=True, key="update_profile"):
        # Collect selected concerns for use elsewhere
        selected = [c for c in CONCERN_OPTIONS if st.session_state.get(f"concern_{c}", False)]
        st.session_state["concerns"] = selected

    card_close()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── System Status ─────────────────────────────────────────────────
    card_open()
    sa_label("System Status")
    for label_name, ok in [("Model", system_status["model"]), ("API", system_status["api"])]:
        dot         = "sa-dot-green" if ok else "sa-dot-red"
        status_text = ("Loaded" if label_name == "Model" else "Connected") if ok else "Offline"
        st.markdown(
            f'<div class="sa-status-row"><div class="sa-dot {dot}"></div>'
            f'<span>{label_name}: <strong style="color:#C8C0D8;">{status_text}</strong></span></div>',
            unsafe_allow_html=True,
        )
    card_close()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Export ────────────────────────────────────────────────────────
    cur_res  = st.session_state["analysis_result"]
    exp_on   = cur_res is not None
    txt_rep  = build_text_report(cur_res) if exp_on else "No analysis yet."
    json_rep = build_json_report(cur_res) if exp_on else json.dumps({"message": "No analysis yet."})

    card_open()
    sa_label("Export Report")
    c1, c2 = st.columns(2)
    with c1:
        st.download_button("⬇ .txt",  data=txt_rep,  file_name="skinaura_report.txt",  mime="text/plain",       use_container_width=True, key="dl_txt",  disabled=not exp_on)
    with c2:
        st.download_button("⬇ .json", data=json_rep, file_name="skinaura_report.json", mime="application/json", use_container_width=True, key="dl_json", disabled=not exp_on)
    card_close()


# ===========================================================================
# CENTER PANEL
# ===========================================================================
with col_center:
    result = st.session_state["analysis_result"]

    if not st.session_state["analysed"] or result is None:
        card_open(accent=True)
        st.markdown(
            """<div style="text-align:center;padding:4rem 0;">
                 <div style="font-size:2rem;margin-bottom:0.8rem;">✦</div>
                 <div style="font-family:'Playfair Display',serif;font-size:1.5rem;color:#3A3A5A;font-weight:400;margin-bottom:0.4rem;">
                   Upload an image and click Analyse
                 </div>
                 <div style="font-family:'DM Mono',monospace;font-size:0.6rem;letter-spacing:0.12em;color:#2A2A3A;">
                   AWAITING INPUT
                 </div>
               </div>""",
            unsafe_allow_html=True,
        )
        card_close()
        st.stop()

    # Always validate routine before rendering to prevent any unpacking crash
    result["routine"] = validate_routine(result.get("routine", {}))

    prediction     = result["prediction"]
    confidence     = result["confidence"]
    recommendation = result["recommendation"]
    class_probs    = result["class_probs"]
    severity       = result["severity"]
    ai_insight     = result["ai_insight"]
    primary        = result["primary"]
    secondary      = result["secondary"]

    # ── AI Summary Header ─────────────────────────────────────────────
    st.markdown(
        """<div class="sa-summary-header">
             <div class="sa-summary-icon">✦</div>
             <div>
               <div class="sa-summary-title">AI Skin Analysis Summary</div>
               <div class="sa-summary-sub">Your skin has been analysed using advanced AI models</div>
             </div>
           </div>""",
        unsafe_allow_html=True,
    )

    # ── 3 Stat Cards ─────────────────────────────────────────────────
    top_sev_level = list(severity.values())[0][0]
    sc1, sc2, sc3 = st.columns(3, gap="medium")

    with sc1:
        sev_badge = f'<span class="sa-badge {badge_cls(top_sev_level)}">{top_sev_level} Severity</span>'
        st.markdown(
            f"""<div class="sa-stat-card">
                  <div class="sa-stat-icon" style="background:rgba(239,68,68,0.15);">🔬</div>
                  <div class="sa-stat-label">Primary Concern</div>
                  <div class="sa-stat-value" style="color:#EF4444;">{prediction}</div>
                  <div class="sa-stat-sub">Confidence: {confidence:.0%}</div>
                  {sev_badge}
                </div>""",
            unsafe_allow_html=True,
        )

    with sc2:
        st.markdown(
            f"""<div class="sa-stat-card">
                  <div class="sa-stat-icon" style="background:rgba(34,197,94,0.1);">🌿</div>
                  <div class="sa-stat-label">Skin Type</div>
                  <div class="sa-stat-value" style="color:#22C55E;">{st.session_state['skin_type']}</div>
                  <div class="sa-stat-sub">Confidence: {min(confidence + 0.08, 0.99):.0%}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    with sc3:
        cov_pct = int(list(severity.values())[1][2] * 100) if len(severity) > 1 else 45
        st.markdown(
            f"""<div class="sa-stat-card">
                  <div class="sa-stat-icon" style="background:rgba(234,179,8,0.1);">📊</div>
                  <div class="sa-stat-label">Coverage</div>
                  <div class="sa-stat-value" style="color:#EAB308;">{cov_pct}%</div>
                  <div class="sa-stat-sub">Affected Area</div>
                </div>""",
            unsafe_allow_html=True,
        )

    # ── Alert Banner ──────────────────────────────────────────────────
    if prediction != "Normal":
        st.markdown(
            f'<div class="sa-alert">⚠️&nbsp; {recommendation} Consistent care is important!</div>',
            unsafe_allow_html=True,
        )

    # ── Tab Navigation ────────────────────────────────────────────────
    tab_choice = st.radio(
        "tab",
        ["Analysis", "Routine", "Diet & Lifestyle", "Products", "Avoid", "Tips"],
        horizontal=True,
        label_visibility="collapsed",
        key="main_tab",
    )

    # ── ANALYSIS TAB ──────────────────────────────────────────────────
    if "Analysis" in tab_choice:
        ta1, ta2 = st.columns([1.3, 1], gap="medium")

        with ta1:
            card_open()
            sa_label("Skin Analysis")

            if cur_img := st.session_state["uploaded_image"]:
                show_hm = st.session_state.get("show_heatmap", False)

                if show_hm:
                    import numpy as np
                    arr  = np.array(cur_img.resize((480, 480))).astype(float)
                    gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]
                    heat = np.clip((gray - 60) / 160, 0, 1)
                    heatmap = np.zeros((*heat.shape, 3), dtype=np.uint8)
                    heatmap[:,:,0] = (heat * 255).astype(np.uint8)
                    heatmap[:,:,1] = ((1 - abs(heat - 0.5) * 2) * 255).astype(np.uint8)
                    heatmap[:,:,2] = ((1 - heat) * 255).astype(np.uint8)
                    blended     = (arr * 0.45 + heatmap * 0.55).astype(np.uint8)
                    display_img = Image.fromarray(blended)
                else:
                    display_img = cur_img

                st.markdown('<div class="sa-image-wrapper">', unsafe_allow_html=True)
                st.image(display_img, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if show_hm:
                    st.markdown(
                        """<div style="display:flex;align-items:center;gap:0.6rem;margin-top:0.4rem;justify-content:flex-end;">
                             <div style="width:10px;height:60px;border-radius:5px;
                                  background:linear-gradient(to top,#3B82F6,#22C55E,#EAB308,#EF4444);"></div>
                             <div style="font-family:'DM Mono',monospace;font-size:0.55rem;color:#5A5A7A;line-height:3.5;">
                               High<br><br><br>Low
                             </div>
                           </div>""",
                        unsafe_allow_html=True,
                    )

                col_hm, _ = st.columns([1, 2])
                with col_hm:
                    toggle_label = "🌡 Show Original" if show_hm else "🌡 Show Heatmap"
                    if st.button(toggle_label, key="heatmap_toggle", use_container_width=True):
                        st.session_state["show_heatmap"] = not show_hm
                        st.rerun()

            card_close()

        with ta2:
            # Severity breakdown
            card_open()
            sa_label("Severity Breakdown")
            for sev_name, (level, color, prob) in severity.items():
                pct = f"{prob * 100:.0f}%"
                st.markdown(
                    f"""<div class="sa-sev-row">
                          <div class="sa-sev-header">
                            <span class="sa-sev-name">{sev_name}</span>
                            <span class="sa-sev-level" style="color:{color};">{level}</span>
                          </div>
                          <div class="sa-sev-bar-bg">
                            <div class="sa-sev-bar-fill" style="width:{pct};background:{color};opacity:0.85;"></div>
                          </div>
                        </div>""",
                    unsafe_allow_html=True,
                )
            card_close()

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            st.markdown(
                f"""<div class="sa-insight-box">
                      <div class="sa-insight-title">✦ AI Insight</div>
                      <div class="sa-insight-text">{ai_insight}</div>
                    </div>""",
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            card_open()
            sa_label("Model Confidence Breakdown")
            for i, (cls_name, prob) in enumerate(class_probs.items()):
                pct  = f"{prob * 100:.0f}%"
                fill = "#A855F7" if i == 0 else "#3A3A5A"
                st.markdown(
                    f"""<div class="sa-class-row">
                          <span class="sa-class-name">{cls_name}</span>
                          <div class="sa-class-bar-bg">
                            <div class="sa-class-bar-fill" style="width:{pct};background:{fill};"></div>
                          </div>
                          <span class="sa-class-pct">{pct}</span>
                        </div>""",
                    unsafe_allow_html=True,
                )
            card_close()

    # ── ROUTINE TAB ───────────────────────────────────────────────────
    elif "Routine" in tab_choice:
        routine  = result["routine"]   # already validated
        r_col1, r_col2 = st.columns(2, gap="medium")

        period_icons = {"Morning": "🌅", "Night": "🌙"}

        for idx, (period, steps) in enumerate(routine.items()):
            col = r_col1 if idx % 2 == 0 else r_col2
            with col:
                card_open()
                icon         = period_icons.get(period, "✦")
                period_color = "#EAB308" if period == "Morning" else "#818CF8"
                st.markdown(
                    f'<div class="sa-routine-period">'
                    f'<span style="font-size:1.1rem;">{icon}</span>'
                    f'<span style="color:{period_color};">{period} Routine</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for step_icon, step_name, step_desc in steps:
                    st.markdown(
                        f"""<div class="sa-routine-product">
                              <div class="sa-routine-product-icon">{step_icon}</div>
                              <div>
                                <div class="sa-routine-product-name">{step_name}</div>
                                <div class="sa-routine-product-desc">{step_desc}</div>
                              </div>
                            </div>""",
                        unsafe_allow_html=True,
                    )
                card_close()

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""<div class="sa-why-box">
                  <div class="sa-why-title">💡 Why this routine?</div>
                  <div class="sa-why-text">{recommendation}</div>
                </div>""",
            unsafe_allow_html=True,
        )

    # ── DIET & LIFESTYLE TAB ──────────────────────────────────────────
    elif "Diet" in tab_choice:
        diet = result["diet"]
        d1, d2, d3, d4 = st.columns(4, gap="small")

        with d1:
            st.markdown(
                '<div class="sa-diet-card"><div class="sa-diet-title" style="color:#22C55E;">✅ Eat More</div>' +
                "".join(f'<div class="sa-diet-item"><div class="sa-diet-dot" style="background:#22C55E;"></div>{item}</div>' for item in diet["eat"]) +
                "</div>", unsafe_allow_html=True,
            )
        with d2:
            st.markdown(
                '<div class="sa-diet-card"><div class="sa-diet-title" style="color:#EF4444;">🚫 Avoid / Limit</div>' +
                "".join(f'<div class="sa-diet-item"><div class="sa-diet-dot" style="background:#EF4444;"></div>{item}</div>' for item in diet["avoid"]) +
                "</div>", unsafe_allow_html=True,
            )
        with d3:
            st.markdown(
                '<div class="sa-diet-card"><div class="sa-diet-title" style="color:#3B82F6;">💧 Hydration</div>'
                '<div style="text-align:center;padding:1.5rem 0;">'
                '<div style="font-size:2.2rem;margin-bottom:0.5rem;">💧</div>'
                f'<div style="font-family:\'Sora\',sans-serif;font-size:0.85rem;font-weight:600;color:#E0D8F0;">{diet["hydration"]}</div>'
                '</div></div>', unsafe_allow_html=True,
            )
        with d4:
            st.markdown(
                '<div class="sa-diet-card"><div class="sa-diet-title" style="color:#A855F7;">🏃 Lifestyle Tips</div>' +
                "".join(f'<div class="sa-diet-item"><div class="sa-diet-dot" style="background:#A855F7;"></div>{item}</div>' for item in diet["lifestyle"]) +
                "</div>", unsafe_allow_html=True,
            )

    # ── PRODUCTS TAB ──────────────────────────────────────────────────
    elif "Products" in tab_choice:
        products = result["products"]
        sa_label("Recommended Products — Curated for your skin type and concerns")
        p_cols = st.columns(4, gap="medium")
        for i, prod in enumerate(products):
            with p_cols[i % 4]:
                st.markdown(
                    f"""<div class="sa-product-card">
                          <div class="sa-product-img">{prod['icon']}</div>
                          <div class="sa-product-name">{prod['name']}</div>
                          <div class="sa-product-type">{prod['type']}</div>
                          <div class="sa-product-price">{prod['price']}</div>
                        </div>""",
                    unsafe_allow_html=True,
                )

    # ── AVOID TAB ─────────────────────────────────────────────────────
    elif "Avoid" in tab_choice:
        card_open()
        sa_label(f"Things to Avoid — {prediction}")
        st.markdown(
            '<div style="font-family:\'Sora\',sans-serif;font-size:0.78rem;color:#6A6A8A;margin-bottom:0.75rem;">'
            'Based on your skin condition, these practices can worsen your skin health.</div>',
            unsafe_allow_html=True,
        )
        for item in result["avoid"]:
            st.markdown(
                f'<div class="sa-avoid-item"><span class="sa-avoid-x">✕</span>{item}</div>',
                unsafe_allow_html=True,
            )
        card_close()

    # ── TIPS TAB ──────────────────────────────────────────────────────
    elif "Tips" in tab_choice:
        card_open()
        sa_label(f"Skin Tips — {prediction}")
        st.markdown(
            '<div style="font-family:\'Sora\',sans-serif;font-size:0.78rem;color:#6A6A8A;margin-bottom:0.75rem;">'
            'Expert tips to get the best results from your skincare routine.</div>',
            unsafe_allow_html=True,
        )
        for i, tip in enumerate(result["tips"], 1):
            st.markdown(
                f'<div class="sa-tip-item"><span class="sa-tip-num">{i:02d}.</span>{tip}</div>',
                unsafe_allow_html=True,
            )
        card_close()

    # ── Consistency Banner ────────────────────────────────────────────
    st.markdown(
        """<div class="sa-banner">
             <span style="font-size:1.2rem;">✨</span>
             <div>
               <div class="sa-banner-text">Consistency is the key!</div>
               <div class="sa-banner-sub">Visible results take time. Follow the routine, eat healthy and be patient.</div>
             </div>
             <span style="font-size:1.2rem;">✨</span>
           </div>""",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-family:\'Sora\',sans-serif;font-size:0.62rem;color:#2A2A3A;'
        'text-align:center;margin-top:0.6rem;">ⓘ This is not a substitute for professional medical advice. '
        'Consult a dermatologist for severe skin conditions.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# RIGHT PANEL — Personalised Routine + Products
# ===========================================================================
with col_right:
    if result is None:
        st.stop()

    routine      = result["routine"]   # already validated
    period_icons = {"Morning": "🌅", "Night": "🌙"}

    # ── Personalised Routine Card ─────────────────────────────────────
    card_open(accent=True)
    st.markdown(
        """<div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:0.75rem;">
             <span style="color:#A855F7;font-size:1rem;">✦</span>
             <div>
               <div style="font-family:'Sora',sans-serif;font-size:0.85rem;font-weight:700;color:#E0D8F0;">Personalised Routine</div>
               <div style="font-family:'Sora',sans-serif;font-size:0.65rem;color:#5A5A7A;">AI-generated for your skin</div>
             </div>
           </div>""",
        unsafe_allow_html=True,
    )

    for period, steps in routine.items():
        icon         = period_icons.get(period, "✦")
        period_color = "#EAB308" if period == "Morning" else "#818CF8"
        st.markdown(
            f'<div class="sa-routine-period">'
            f'<span style="font-size:1rem;">{icon}</span>'
            f'<span style="color:{period_color};">{period} Routine</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        for step_icon, step_name, step_desc in steps:
            st.markdown(
                f"""<div class="sa-routine-product">
                      <div class="sa-routine-product-icon">{step_icon}</div>
                      <div>
                        <div class="sa-routine-product-name">{step_name}</div>
                        <div class="sa-routine-product-desc">{step_desc}</div>
                      </div>
                    </div>""",
                unsafe_allow_html=True,
            )

    sa_divider()
    st.markdown(
        f"""<div class="sa-why-box">
              <div class="sa-why-title">💡 Why this routine?</div>
              <div class="sa-why-text">{recommendation}</div>
            </div>""",
        unsafe_allow_html=True,
    )
    card_close()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # ── Recommended Products Card ─────────────────────────────────────
    card_open()
    st.markdown(
        """<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:0.8rem;">
             <div>
               <div style="font-family:'Sora',sans-serif;font-size:0.85rem;font-weight:700;color:#E0D8F0;">🛒 Recommended Products</div>
               <div style="font-family:'Sora',sans-serif;font-size:0.65rem;color:#5A5A7A;">Curated for your skin type &amp; concerns</div>
             </div>
             <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#A855F7;cursor:pointer;">View All →</div>
           </div>""",
        unsafe_allow_html=True,
    )

    products = result["products"]
    pc1, pc2 = st.columns(2, gap="small")
    for i, prod in enumerate(products):
        col = pc1 if i % 2 == 0 else pc2
        with col:
            st.markdown(
                f"""<div class="sa-product-card" style="margin-bottom:0.5rem;">
                      <div class="sa-product-img">{prod['icon']}</div>
                      <div class="sa-product-name">{prod['name']}</div>
                      <div class="sa-product-type">{prod['type']}</div>
                      <div class="sa-product-price">{prod['price']}</div>
                    </div>""",
                unsafe_allow_html=True,
            )
    card_close()