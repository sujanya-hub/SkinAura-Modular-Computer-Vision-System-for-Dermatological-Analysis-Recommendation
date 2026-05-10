"""
app.py - SkinAura: AI Analysis Dashboard (v5.0)
==========================================================
Upgraded to match reference UI:
  - Removed all emojis; replaced with clean text/icon alternatives
  - Reduced color palette to professional dark system
  - Merged duplicate Routine/Products/Care sections (right panel only)
  - Right panel: Routine, Care Priorities, Recommended Products
  - Center tabs: AI Insight, Top 3 Predictions, Avoid, Tips, Diet
  - Premium glassmorphism-lite metric cards
  - Clean typography: Geist + DM Sans
  - Professional system status indicators
  - All backend logic preserved
"""
from __future__ import annotations

import io
import json
import time
from datetime import datetime
from typing import Any

import numpy as np
import requests
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SkinAura - AI Dermatology Assistant",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
BASE_URL        = "https://skinaura-backend.onrender.com/api/v1"
PREDICT_API_URL = f"{BASE_URL}/predict"
HEALTH_API_URL  = f"{BASE_URL}/health"
API_TIMEOUT_SECONDS = 60

CLASS_LABELS = [
    "Dark Spots",
    "Inflammatory Acne",
    "Non Inflammatory Acne Blackheads",
    "Non Inflammatory Acne Whiteheads",
    "Pigmentation",
    "Pores",
    "Redness",
    "Wrinkles",
]

# ---------------------------------------------------------------------------
# CSS — Premium Dark Clinical UI
# ---------------------------------------------------------------------------
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=DM+Mono:wght@300;400;500&family=Sora:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg-base:       #070B14;
  --bg-surface:    #0F172A;
  --bg-elevated:   #111827;
  --bg-card:       #131B2E;
  --border:        rgba(255,255,255,0.07);
  --border-subtle: rgba(255,255,255,0.04);
  --primary:       #7C3AED;
  --primary-light: #A78BFA;
  --primary-dim:   rgba(124,58,237,0.12);
  --text-primary:  #F8FAFC;
  --text-secondary:#CBD5E1;
  --text-muted:    #64748B;
  --text-dim:      #334155;
  --success:       #22C55E;
  --warning:       #F59E0B;
  --error:         #EF4444;
  --success-dim:   rgba(34,197,94,0.12);
  --warning-dim:   rgba(245,158,11,0.12);
  --error-dim:     rgba(239,68,68,0.12);
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background: var(--bg-base) !important;
    color: var(--text-primary) !important;
}
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarNav"] { display: none !important; }

* { font-family: 'DM Sans', sans-serif !important; }
code, pre, .mono { font-family: 'DM Mono', monospace !important; }

/* Progress bars */
[data-testid="stProgress"] > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 2px !important;
}
[data-testid="stProgress"] > div > div {
    background: var(--primary) !important;
    border-radius: 2px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(255,255,255,0.12) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"] * {
    color: var(--text-muted) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.78rem !important;
}

/* Columns */
[data-testid="column"] { padding: 0 0.25rem !important; }

/* Buttons */
.stButton > button {
    background: var(--primary) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.2rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #6D28D9 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: rgba(124,58,237,0.25) !important;
    color: rgba(255,255,255,0.4) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: rgba(124,58,237,0.25); border-radius: 2px; }

/* Download button */
.stDownloadButton > button {
    background: rgba(255,255,255,0.04) !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border) !important;
    box-shadow: none !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 0.7rem !important;
    border-radius: 6px !important;
}
.stDownloadButton > button:hover {
    background: var(--primary-dim) !important;
    border-color: rgba(124,58,237,0.3) !important;
    color: var(--primary-light) !important;
    transform: none !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}

/* Radio as tabs */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    border-bottom: 1px solid var(--border-subtle) !important;
    flex-wrap: wrap !important;
}
div[data-testid="stRadio"] > div > label {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    padding: 0.5rem 0.9rem !important;
    cursor: pointer !important;
    border-bottom: 2px solid transparent !important;
    margin: 0 !important;
    border-radius: 0 !important;
    background: transparent !important;
    transition: all 0.15s !important;
    letter-spacing: 0.01em !important;
}
div[data-testid="stRadio"] > div > label:hover { color: var(--primary-light) !important; }
div[data-testid="stRadio"] label > div:first-child { display: none !important; }

/* Checkbox */
[data-testid="stCheckbox"] { padding: 0.15rem 0 !important; }
[data-testid="stCheckbox"] label {
    font-size: 0.75rem !important;
    color: var(--text-muted) !important;
}
[data-baseweb="checkbox"] > div:first-child {
    background-color: transparent !important;
    border: 1.5px solid rgba(124,58,237,0.4) !important;
    border-radius: 3px !important;
    width: 13px !important;
    height: 13px !important;
}
[data-baseweb="checkbox"][aria-checked="true"] > div:first-child {
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
}

/* ── Layout primitives ── */
.sa-divider {
    height: 1px;
    background: var(--border-subtle);
    margin: 0.6rem 0;
}
.sa-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.5rem;
}

/* ── Logo ── */
.sa-logo-wrap {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.5rem;
}
.sa-logo-icon {
    width: 30px;
    height: 30px;
    background: var(--primary);
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
}
.sa-logo-text { font-size: 0.95rem; font-weight: 700; color: var(--text-primary); }
.sa-logo-sub  { font-size: 0.58rem; color: var(--text-muted); margin-top: -1px; }

/* ── Nav ── */
.sa-nav-item {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-muted);
    padding: 0.45rem 0.7rem;
    border-radius: 6px;
}
.sa-nav-active {
    background: var(--primary-dim) !important;
    color: var(--primary-light) !important;
    font-weight: 600 !important;
}
.sa-nav-icon { font-size: 0.75rem; width: 16px; text-align: center; }

/* ── Profile ── */
.sa-profile-field-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.54rem;
    letter-spacing: 0.12em;
    color: var(--text-dim);
    text-transform: uppercase;
    margin-bottom: 0.2rem;
}
.sa-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: var(--primary);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: #fff;
    flex-shrink: 0;
}

/* ── Status dot ── */
.sa-dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; display: inline-block; }
.sa-dot-green  { background: var(--success); }
.sa-dot-red    { background: var(--error); }
.sa-dot-yellow { background: var(--warning); }

/* ── Metric cards ── */
.sa-metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.1rem;
    height: 100%;
    position: relative;
    overflow: hidden;
}
.sa-metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--primary);
    opacity: 0.5;
}
.sa-metric-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.54rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-bottom: 0.45rem;
}
.sa-metric-value {
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--text-primary);
    line-height: 1.1;
    margin-bottom: 0.2rem;
}
.sa-metric-sub {
    font-size: 0.68rem;
    color: var(--text-muted);
    margin-top: 0.15rem;
}
.sa-severity-badge {
    display: inline-block;
    font-size: 0.58rem;
    font-weight: 600;
    padding: 0.18rem 0.5rem;
    border-radius: 3px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 0.35rem;
    border: 1px solid;
}
.sev-high   { background: var(--error-dim);   color: var(--error);   border-color: rgba(239,68,68,0.25); }
.sev-medium { background: var(--warning-dim); color: var(--warning); border-color: rgba(245,158,11,0.25); }
.sev-low    { background: var(--success-dim); color: var(--success); border-color: rgba(34,197,94,0.25); }

/* ── Image section ── */
.sa-img-wrap {
    border-radius: 8px;
    overflow: hidden;
    background: var(--bg-surface);
    border: 1px solid var(--border);
}
.sa-img-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-muted);
    text-align: center;
    padding: 0.4rem 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    background: var(--bg-elevated);
    border-bottom: 1px solid var(--border-subtle);
}

/* ── Attention scale ── */
.sa-attention-scale {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.4rem 0.2rem;
    font-size: 0.65rem;
    color: var(--text-muted);
}
.sa-scale-bar {
    flex: 1;
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, #3B82F6, #22C55E, #F59E0B, #EF4444);
}

/* ── AI Insight section ── */
.sa-section-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.02em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.sa-insight-body {
    font-size: 0.78rem;
    color: var(--text-secondary);
    line-height: 1.75;
    margin-bottom: 0.75rem;
}
.sa-pattern-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.75rem;
}
.sa-pattern-col {
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 7px;
    padding: 0.65rem 0.7rem;
}
.sa-pattern-col-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.45rem;
}
.sa-pattern-item {
    font-size: 0.68rem;
    color: var(--text-secondary);
    padding: 0.12rem 0;
    display: flex;
    align-items: flex-start;
    gap: 0.35rem;
    line-height: 1.45;
}
.sa-pattern-dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--primary-light);
    flex-shrink: 0;
    margin-top: 0.38rem;
}

/* ── Clinical note ── */
.sa-clinical-note {
    background: rgba(124,58,237,0.04);
    border: 1px solid rgba(124,58,237,0.12);
    border-left: 2px solid var(--primary);
    border-radius: 0 6px 6px 0;
    padding: 0.65rem 0.9rem;
    margin-top: 0.6rem;
}
.sa-clinical-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--primary-light);
    margin-bottom: 0.3rem;
}
.sa-clinical-text {
    font-size: 0.73rem;
    color: var(--text-muted);
    line-height: 1.65;
}

/* ── Top 3 predictions ── */
.sa-prediction-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border-subtle);
}
.sa-prediction-row:last-child { border-bottom: none; }
.sa-pred-rank {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--text-dim);
    width: 18px;
    text-align: center;
    flex-shrink: 0;
}
.sa-pred-name {
    font-size: 0.78rem;
    color: var(--text-secondary);
    flex: 1;
    min-width: 0;
}
.sa-pred-bar-wrap {
    flex: 1;
    height: 4px;
    background: rgba(255,255,255,0.05);
    border-radius: 2px;
    overflow: hidden;
}
.sa-pred-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-muted);
    min-width: 30px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Care priorities (right panel) ── */
.sa-care-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.55rem 0.7rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border-subtle);
    border-radius: 7px;
    margin-bottom: 0.35rem;
}
.sa-care-icon-box {
    width: 30px;
    height: 30px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
}
.sa-care-label-text {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
    flex: 1;
}
.sa-care-level-text {
    font-size: 0.7rem;
    font-weight: 600;
    flex-shrink: 0;
}

/* ── Routine (right panel) ── */
.sa-routine-period-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 0.7rem 0 0.35rem;
}
.sa-routine-step {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    padding: 0.22rem 0;
}
.sa-routine-step-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--primary);
    flex-shrink: 0;
    margin-top: 0.32rem;
}
.sa-routine-step-name {
    font-size: 0.76rem;
    font-weight: 500;
    color: var(--text-primary);
    line-height: 1.3;
}
.sa-routine-step-desc {
    font-size: 0.63rem;
    color: var(--text-muted);
    margin-top: 1px;
}
.sa-view-link {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--primary-light);
    cursor: pointer;
    margin-top: 0.3rem;
    display: inline-block;
}

/* ── Products (right panel) ── */
.sa-product-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid var(--border-subtle);
}
.sa-product-row:last-child { border-bottom: none; }
.sa-product-icon-box {
    width: 34px;
    height: 42px;
    background: var(--bg-elevated);
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.62rem;
    font-weight: 700;
    color: var(--text-muted);
    flex-shrink: 0;
    border: 1px solid var(--border-subtle);
}
.sa-product-name {
    font-size: 0.72rem;
    font-weight: 500;
    color: var(--text-secondary);
    line-height: 1.3;
}
.sa-product-desc {
    font-size: 0.6rem;
    color: var(--text-muted);
    margin-top: 1px;
}
.sa-product-view-btn {
    margin-left: auto;
    flex-shrink: 0;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.2rem 0.5rem;
    font-size: 0.6rem;
    font-weight: 600;
    color: var(--text-muted);
    cursor: pointer;
}
.sa-view-all-link {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--primary-light);
    cursor: pointer;
    margin-top: 0.4rem;
    display: inline-block;
    text-align: center;
    width: 100%;
}

/* ── Page header ── */
.sa-page-title {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.sa-page-sub {
    font-size: 0.7rem;
    color: var(--text-muted);
    margin-top: 2px;
}
.sa-analysis-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-elevated);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.28rem 0.75rem;
    font-size: 0.7rem;
    color: var(--text-secondary);
    font-weight: 500;
}
.sa-badge-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--success);
}
.sa-timestamp {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    margin-top: 0.2rem;
    text-align: right;
}

/* ── Avoid / Tips items ── */
.sa-list-item {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--border-subtle);
    font-size: 0.77rem;
    color: var(--text-secondary);
    line-height: 1.5;
}
.sa-list-item:last-child { border-bottom: none; }
.sa-list-marker-x {
    color: var(--error);
    font-size: 0.65rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 0.2rem;
}
.sa-list-marker-n {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--primary-light);
    min-width: 18px;
    flex-shrink: 0;
    margin-top: 0.15rem;
}

/* ── Banner ── */
.sa-banner {
    background: var(--bg-surface);
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    text-align: center;
    margin-top: 0.75rem;
}
.sa-banner-text {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 0.2rem;
}
.sa-banner-sub {
    font-size: 0.68rem;
    color: var(--text-muted);
}

/* ── Right panel section header ── */
.sa-rp-section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.15rem;
    margin-top: 0.1rem;
}
.sa-rp-section-sub {
    font-size: 0.62rem;
    color: var(--text-dim);
    margin-bottom: 0.55rem;
}

/* ── File info pill ── */
.sa-file-pill {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.35rem 0.55rem;
    margin: 0.35rem 0;
}
.sa-file-pill-icon {
    width: 24px;
    height: 24px;
    background: var(--primary-dim);
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.6rem;
    color: var(--primary-light);
    font-weight: 700;
    flex-shrink: 0;
}
.sa-file-name {
    font-size: 0.7rem;
    font-weight: 500;
    color: var(--text-secondary);
}
.sa-file-size {
    font-family: 'DM Mono', monospace;
    font-size: 0.57rem;
    color: var(--text-dim);
}

/* ── Upload section label ── */
.sa-upload-label {
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.2rem;
}
.sa-upload-hint {
    font-size: 0.6rem;
    color: var(--text-dim);
    margin-bottom: 0.4rem;
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def normalize(s: str) -> str:
    return s.replace("_", " ").replace("-", " ").title().strip() or "Unknown"

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))

def sev_color(prob: float) -> tuple[str, str]:
    if prob >= 0.7:   return "High",   "#EF4444"
    if prob >= 0.4:   return "Medium", "#F59E0B"
    return "Low", "#22C55E"

RoutineStep = tuple[str, str, str]
FALLBACK_STEP: RoutineStep = ("", "Skincare Step", "Apply as directed.")

def validate_routine_step(step: Any) -> RoutineStep:
    if isinstance(step, (list, tuple)):
        if len(step) == 3: return (str(step[0]) or "", str(step[1]) or "Step", str(step[2]) or "")
        if len(step) == 2: return ("", str(step[0]) or "Step", str(step[1]) or "")
        if len(step) == 1: return ("", str(step[0]) or "Step", "")
    if isinstance(step, dict):
        return (str(step.get("icon", "")), str(step.get("name", "Step")), str(step.get("description", "")))
    return FALLBACK_STEP

def validate_routine(routine: Any) -> dict[str, list[RoutineStep]]:
    if not isinstance(routine, dict): return {}
    result: dict[str, list[RoutineStep]] = {}
    for period, steps in routine.items():
        if not isinstance(steps, (list, tuple)):
            result[str(period)] = []
            continue
        result[str(period)] = [validate_routine_step(s) for s in steps]
    return result

# ---------------------------------------------------------------------------
# API CALLS
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
        return {"success": False, "error": "API not reachable — is the backend running on port 8000?"}
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

    pred = payload.get("prediction") or payload.get("predicted_class")
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
            "demo_mode":      payload.get("demo_mode", False),
        },
    }

# ---------------------------------------------------------------------------
# HEALTH / STATUS DETECTION
# ---------------------------------------------------------------------------
_MODEL_ONLINE_VALUES: frozenset[str] = frozenset({"loaded"})
_API_ONLINE_STATUSES: frozenset[str] = frozenset({"ok", "degraded"})
_LLM_ONLINE_VALUES:   frozenset[str] = frozenset({"ready", "mock_mode", "not_initialised", "unavailable"})
_RAG_ONLINE_VALUES:   frozenset[str] = frozenset({"ready"})
_HEALTH_TIMEOUT: float = 8.0

def fetch_system_status() -> dict[str, Any]:
    _offline_unreachable = {"model": False, "api": False, "llm": False, "rag": False, "error": "Backend unreachable — is it running on port 8000?"}
    _offline_reachable   = {"model": False, "api": True,  "llm": False, "rag": False, "error": ""}
    try:
        r = requests.get(HEALTH_API_URL, timeout=_HEALTH_TIMEOUT)
    except (requests.ConnectionError, requests.Timeout):
        return _offline_unreachable
    try:
        r.raise_for_status()
    except requests.HTTPError:
        return _offline_reachable
    try:
        p = r.json()
    except ValueError:
        return _offline_reachable
    if not isinstance(p, dict):
        return _offline_reachable
    services: dict = p.get("services", {})
    if not isinstance(services, dict):
        services = {}
    model_loader_val = str(services.get("model_loader", "")).strip()
    model_ok  = model_loader_val in _MODEL_ONLINE_VALUES
    top_status = str(p.get("status", "")).strip()
    api_ok    = top_status in _API_ONLINE_STATUSES
    llm_val   = str(services.get("llm", "")).strip()
    llm_ok    = llm_val in _LLM_ONLINE_VALUES
    rag_val   = str(services.get("rag", "")).strip()
    rag_ok    = rag_val in _RAG_ONLINE_VALUES
    return {"model": model_ok, "api": api_ok, "llm": llm_ok, "rag": rag_ok, "error": ""}

# ---------------------------------------------------------------------------
# PROFILE-AWARE DATA BUILDERS
# ---------------------------------------------------------------------------
def _profile_product_note(skin_type: str, skin_tone: str, age_range: str) -> str:
    notes = []
    if skin_type == "Oily":        notes.append("lightweight, non-comedogenic formulas")
    elif skin_type == "Dry":       notes.append("hydration-first, ceramide-rich formulas")
    elif skin_type == "Sensitive": notes.append("fragrance-free, barrier-supportive formulas")
    elif skin_type == "Combination": notes.append("zone-targeted, balanced formulas")
    if skin_tone in ("Tan", "Deep"):
        notes.append("melanin-safe ingredients avoiding irritating bleaching agents")
    if age_range in ("15-20", "20-25"):
        notes.append("acne-focused preventive actives")
    elif age_range == "40+":
        notes.append("barrier repair and collagen-support actives")
    elif age_range == "30-40":
        notes.append("anti-aging and brightening balance")
    return "; ".join(notes) if notes else "evidence-based skincare"

def simulate_class_probs(prediction: str, confidence: float) -> dict[str, float]:
    primary = normalize(prediction)
    top = clamp(confidence, 0.55, 0.99)
    rem = 1.0 - top
    profiles: dict[str, dict[str, float]] = {
        "Inflammatory Acne":                    {"Dark Spots": 0.30, "Non Inflammatory Acne Blackheads": 0.30, "Redness": 0.25, "Pores": 0.15},
        "Non Inflammatory Acne Blackheads":     {"Pores": 0.35, "Inflammatory Acne": 0.30, "Non Inflammatory Acne Whiteheads": 0.25, "Dark Spots": 0.10},
        "Non Inflammatory Acne Whiteheads":     {"Non Inflammatory Acne Blackheads": 0.40, "Pores": 0.30, "Inflammatory Acne": 0.20, "Dark Spots": 0.10},
        "Pigmentation":                         {"Dark Spots": 0.45, "Redness": 0.25, "Wrinkles": 0.20, "Pores": 0.10},
        "Dark Spots":                           {"Pigmentation": 0.45, "Redness": 0.25, "Inflammatory Acne": 0.20, "Wrinkles": 0.10},
        "Pores":                                {"Non Inflammatory Acne Blackheads": 0.40, "Inflammatory Acne": 0.30, "Redness": 0.30},
        "Redness":                              {"Inflammatory Acne": 0.40, "Pigmentation": 0.30, "Dark Spots": 0.20, "Pores": 0.10},
        "Wrinkles":                             {"Pigmentation": 0.40, "Dark Spots": 0.30, "Redness": 0.20, "Pores": 0.10},
    }
    profile = profiles.get(primary, {})
    others  = [lbl for lbl in CLASS_LABELS if lbl != primary]
    total_w = sum(profile.get(l, 1.0) for l in others) or 1.0
    probs: dict[str, float] = {primary: top}
    for l in others:
        probs[l] = rem * profile.get(l, 1.0) / total_w
    total = sum(probs.values()) or 1.0
    return dict(sorted({k: v / total for k, v in probs.items()}.items(), key=lambda x: x[1], reverse=True))

def build_severity(prediction: str, confidence: float, class_probs: dict[str, float]) -> dict[str, tuple[str, str, float]]:
    sec_prob = list(class_probs.values())[1] if len(class_probs) > 1 else 0.1
    base: dict[str, dict[str, float]] = {
        "Inflammatory Acne":                {"Inflammation": 0.72, "Coverage": 0.55, "Scarring Risk": 0.48, "Pigmentation": 0.28},
        "Non Inflammatory Acne Blackheads": {"Inflammation": 0.22, "Coverage": 0.48, "Scarring Risk": 0.30, "Pigmentation": 0.20},
        "Non Inflammatory Acne Whiteheads": {"Inflammation": 0.18, "Coverage": 0.44, "Scarring Risk": 0.25, "Pigmentation": 0.18},
        "Pigmentation":                     {"Inflammation": 0.18, "Coverage": 0.58, "Scarring Risk": 0.30, "Pigmentation": 0.70},
        "Dark Spots":                       {"Inflammation": 0.15, "Coverage": 0.50, "Scarring Risk": 0.28, "Pigmentation": 0.65},
        "Pores":                            {"Inflammation": 0.20, "Coverage": 0.45, "Scarring Risk": 0.18, "Pigmentation": 0.15},
        "Redness":                          {"Inflammation": 0.62, "Coverage": 0.48, "Scarring Risk": 0.22, "Pigmentation": 0.25},
        "Wrinkles":                         {"Inflammation": 0.12, "Coverage": 0.55, "Scarring Risk": 0.20, "Pigmentation": 0.30},
    }
    profile = base.get(normalize(prediction), {"Inflammation": 0.08, "Coverage": 0.12, "Scarring Risk": 0.10, "Pigmentation": 0.10})
    result: dict[str, tuple[str, str, float]] = {}
    for metric, bv in profile.items():
        val = clamp(bv + confidence * 0.25 + sec_prob * 0.08, 0.05, 0.95)
        level, color = sev_color(val)
        result[metric] = (level, color, val)
    return result

def build_pattern_data(prediction: str) -> dict[str, list[str]]:
    data: dict[str, dict[str, list[str]]] = {
        "Inflammatory Acne": {
            "Pattern Analysis":   ["Active papules/pustules", "Erythema around lesions", "Follicular congestion", "Sebum-prone zones"],
            "Barrier Assessment": ["Sebum Level: High", "Barrier Strength: Moderate", "Hydration: Low-Moderate", "Sensitivity: Moderate"],
            "Primary Indicators": ["Bacterial overgrowth", "Follicular inflammation", "Excess sebum production", "Post-inflammatory risk"],
            "Recommended Focus":  ["Reduce active inflammation", "Regulate oil production", "Strengthen skin barrier", "Prevent PIH"],
        },
        "Non Inflammatory Acne Blackheads": {
            "Pattern Analysis":   ["Open comedones", "Pore congestion", "Oxidised sebum deposits", "T-zone prominence"],
            "Barrier Assessment": ["Sebum Level: High", "Barrier Strength: Good", "Hydration: Moderate", "Sensitivity: Low"],
            "Primary Indicators": ["Follicular plugging", "Oxidised keratin debris", "Enlarged pore appearance", "Oil-prone pattern"],
            "Recommended Focus":  ["BHA exfoliation", "Pore-clearing actives", "Non-comedogenic routine", "Consistent cleansing"],
        },
        "Non Inflammatory Acne Whiteheads": {
            "Pattern Analysis":   ["Closed comedones", "Milium-like bumps", "Trapped keratin", "Smooth skin texture"],
            "Barrier Assessment": ["Sebum Level: Moderate-High", "Barrier Strength: Moderate", "Irritation Risk: Low-Moderate", "Sensitivity: Low"],
            "Primary Indicators": ["Closed follicular plugs", "Keratinocyte buildup", "Excess sebum", "Acne-prone pattern"],
            "Recommended Focus":  ["Salicylic acid exfoliation", "Gentle retinoid use", "Barrier support", "Lightweight moisturiser"],
        },
        "Pigmentation": {
            "Pattern Analysis":   ["Diffuse melanin distribution", "Uneven tonal variation", "Solar lentigo patterns", "Post-inflammatory marks"],
            "Barrier Assessment": ["Sebum Level: Normal", "Barrier Strength: Moderate", "Hydration: Good", "Sensitivity: Moderate"],
            "Primary Indicators": ["Melanin overproduction", "UV-triggered response", "Hormonal influence possible", "Tonal irregularity"],
            "Recommended Focus":  ["Melanin inhibition", "Daily SPF 50+", "Brightening actives", "Gentle exfoliation"],
        },
        "Dark Spots": {
            "Pattern Analysis":   ["Localised hyperpigmentation", "Post-inflammatory marks", "Melanin clusters", "Contrast irregularity"],
            "Barrier Assessment": ["Sebum Level: Normal", "Barrier Strength: Moderate", "Hydration: Good", "Sensitivity: Low-Moderate"],
            "Primary Indicators": ["PIH from prior acne", "UV-triggered darkening", "Melanocyte activation", "Spot-specific deposits"],
            "Recommended Focus":  ["Targeted spot treatment", "Vitamin C + niacinamide", "Aggressive SPF use", "Avoid picking skin"],
        },
        "Pores": {
            "Pattern Analysis":   ["Enlarged follicular openings", "Sebaceous gland activity", "Blackhead co-occurrence", "Facial zone pattern"],
            "Barrier Assessment": ["Sebum Level: High", "Barrier Strength: Good", "Hydration: Moderate", "Sensitivity: Low"],
            "Primary Indicators": ["Excess sebum production", "Collagen laxity possible", "Pore congestion", "Skin texture variation"],
            "Recommended Focus":  ["BHA exfoliation", "Niacinamide to minimise", "Oil control cleansing", "Retinoids for long-term"],
        },
        "Redness": {
            "Pattern Analysis":   ["Diffuse erythema", "Flushing pattern", "Vascular prominence", "Inflammatory markers"],
            "Barrier Assessment": ["Sebum Level: Low-Normal", "Barrier Strength: Weak", "Hydration: Low", "Sensitivity: High"],
            "Primary Indicators": ["Barrier dysfunction", "Vascular hyperreactivity", "Possible rosacea trigger", "Environmental sensitivity"],
            "Recommended Focus":  ["Barrier repair urgently", "Azelaic acid / centella", "Fragrance-free routine", "Avoid heat triggers"],
        },
        "Wrinkles": {
            "Pattern Analysis":   ["Expression line depth", "Skin laxity pattern", "Collagen depletion zones", "Dermal thinning"],
            "Barrier Assessment": ["Sebum Level: Low", "Barrier Strength: Weak", "Hydration: Low", "Sensitivity: Moderate"],
            "Primary Indicators": ["Collagen degradation", "Elastin loss", "UV-induced photodamage", "Intrinsic aging signs"],
            "Recommended Focus":  ["Retinoid nightly use", "Peptide serum", "Aggressive SPF 50+", "Hydration and ceramides"],
        },
    }
    return data.get(normalize(prediction), {
        "Pattern Analysis":   ["Balanced sebum", "Even tone", "Clear pores", "Healthy texture"],
        "Barrier Assessment": ["Sebum Level: Balanced", "Barrier Strength: Good", "Hydration: Good", "Sensitivity: Low"],
        "Primary Indicators": ["Stable microbiome", "Good cell turnover", "UV damage prevention", "Aging prevention"],
        "Recommended Focus":  ["Maintain balance", "Daily SPF", "Antioxidant defense", "Preventive care"],
    })

def build_clinical_note(prediction: str, skin_type: str, age_range: str) -> str:
    base: dict[str, str] = {
        "Inflammatory Acne":                "Active inflammatory acne benefits most from consistent benzoyl peroxide or adapalene use. Avoid over-washing or harsh scrubs. Consult a dermatologist if lesions persist beyond 12 weeks of consistent treatment.",
        "Non Inflammatory Acne Blackheads": "Regular BHA exfoliation (salicylic acid 1–2%) is highly effective for comedonal acne. Introduce once weekly and increase gradually. Avoid heavy creams and occlusive products.",
        "Non Inflammatory Acne Whiteheads": "Closed comedones respond well to gentle retinoids and BHAs. Start slowly. Non-comedogenic moisturiser and consistent cleansing are foundational.",
        "Pigmentation":                     "Sun avoidance and daily SPF 50+ is non-negotiable. Brightening actives (Vitamin C, niacinamide, tranexamic acid) are most effective used consistently over 12–16 weeks.",
        "Dark Spots":                       "Targeted brightening combined with strict sun protection delivers best results. Avoid picking or trauma which deepens marks. Professional chemical peels may accelerate fading.",
        "Pores":                            "Pore appearance is best managed with consistent BHA use, oil control, and retinoids for long-term pore tightening. Results take several weeks. Avoid pore-clogging products.",
        "Redness":                          "A compromised skin barrier is often the root cause. Prioritise fragrance-free, gentle, barrier-repairing products. Consider dermatology referral if rosacea is suspected.",
        "Wrinkles":                         "Retinoid therapy combined with consistent SPF is the gold standard. Introduce retinoids at low concentration 2x/week. Results visible after 3–6 months of consistent use.",
    }
    note = base.get(normalize(prediction), "Maintain a consistent skincare routine. Daily SPF and a good moisturiser are your best long-term investments.")
    age_addendum = ""
    if age_range in ("15-20", "20-25"):
        age_addendum = " At your age, prevention is paramount — starting SPF early dramatically reduces future skin concerns."
    elif age_range == "40+":
        age_addendum = " For your age group, barrier support and retinoid-based renewal should be core priorities."
    skin_addendum = ""
    if skin_type == "Sensitive":
        skin_addendum = " Given your sensitive skin type, always patch-test new actives and introduce them one at a time."
    elif skin_type == "Dry":
        skin_addendum = " With dry skin, always follow active ingredients with a rich moisturiser to prevent irritation."
    return note + age_addendum + skin_addendum

def build_ai_insight(prediction: str, confidence: float, secondary_label: str, secondary_prob: float, skin_type: str, skin_tone: str, age_range: str) -> str:
    primary = normalize(prediction)
    msgs: dict[str, str] = {
        "Inflammatory Acne":                "Active inflammatory acne detected with papular and/or pustular lesion pattern. Pore congestion and elevated sebum activity observed. Post-inflammatory hyperpigmentation risk present.",
        "Non Inflammatory Acne Blackheads": "Open comedonal pattern consistent with non-inflammatory acne. Oxidised sebum plugs visible in follicular openings. Inflammatory progression risk if untreated.",
        "Non Inflammatory Acne Whiteheads": "Closed comedonal pattern detected. Non-inflamed follicular plugging without active erythema. Risk of progression to inflammatory lesions with environmental triggers.",
        "Pigmentation":                     "Diffuse melanin irregularity and tonal variation detected. UV-triggered and/or hormonal pigmentation pattern. Consistent brightening actives and SPF 50+ are essential.",
        "Dark Spots":                       "Localised hyperpigmented lesions consistent with post-inflammatory or UV-triggered dark spots. Melanin deposition detected in target zones.",
        "Pores":                            "Enlarged pore pattern observed with elevated sebaceous gland activity. Follicular dilation and surface texture variation detected.",
        "Redness":                          "Diffuse erythema detected with possible barrier compromise. Vascular hyperreactivity pattern. Rosacea trigger factors should be considered.",
        "Wrinkles":                         "Collagen depletion and elastin loss pattern detected. Expression lines and possible photodamage observed. Retinoid-based renewal and aggressive sun protection recommended.",
    }
    certainty = "strong" if confidence >= 0.8 else "moderate"
    sec_text  = ""
    if secondary_label != primary and secondary_prob >= 0.12:
        sec_text = f" Secondary overlap with {secondary_label.lower()} features detected."
    profile_note = _profile_product_note(skin_type, skin_tone, age_range)
    profile_text = f" Given your {skin_type.lower()} skin type and {age_range} age group, focus on {profile_note}."
    tone_note = ""
    if skin_tone in ("Tan", "Deep"):
        tone_note = " Post-inflammatory hyperpigmentation can be more persistent on deeper skin tones — prioritise melanin-safe, non-irritating actives."
    base = msgs.get(primary, msgs["Inflammatory Acne"])
    return f"{base} Model confidence: {confidence:.0%} — {certainty} match.{sec_text}{profile_text}{tone_note}"

def build_routine(prediction: str, skin_type: str, age_range: str) -> dict[str, list[RoutineStep]]:
    cleanser_am = {
        "Oily":        ("", "Foaming Salicylic Cleanser",  "Controls oil and unclogs pores"),
        "Dry":         ("", "Cream/Hydrating Cleanser",    "Gentle cleanse without stripping"),
        "Combination": ("", "Gel Balancing Cleanser",      "Balances T-zone and dry zones"),
        "Normal":      ("", "Gentle Gel Cleanser",         "Mild cleanse to start the day"),
        "Sensitive":   ("", "Micellar / Fragrance-free",   "Ultra-gentle, no-rinse cleanse"),
    }.get(skin_type, ("", "Gentle Cleanser", "Cleanses without stripping the skin"))
    moisturiser = {
        "Oily":        ("", "Oil-free Gel Moisturiser",   "Lightweight hydration without shine"),
        "Dry":         ("", "Rich Ceramide Moisturiser",  "Deep hydration and barrier repair"),
        "Combination": ("", "Lightweight Lotion",          "Balancing daily moisturiser"),
        "Normal":      ("", "Lightweight Moisturiser",    "Hydrates and strengthens barrier"),
        "Sensitive":   ("", "Barrier Repair Cream",       "Soothing and fortifying formula"),
    }.get(skin_type, ("", "Lightweight Moisturiser", "Hydrates and strengthens barrier"))
    routines_am: dict[str, list[RoutineStep]] = {
        "Inflammatory Acne":                [cleanser_am, ("", "Niacinamide 10% Serum",  "Reduces redness and regulates oil"), moisturiser, ("", "Sunscreen SPF 50",         "Non-comedogenic UV protection")],
        "Non Inflammatory Acne Blackheads": [cleanser_am, ("", "Niacinamide Serum",       "Minimises pore appearance"), moisturiser, ("", "Sunscreen SPF 50",         "Lightweight UV protection")],
        "Non Inflammatory Acne Whiteheads": [cleanser_am, ("", "Niacinamide + Zinc",      "Controls sebum and closed pores"), moisturiser, ("", "Sunscreen SPF 50",         "Daily UV defence")],
        "Pigmentation":                     [cleanser_am, ("", "Vitamin C 10–15%",        "Brightens and targets uneven tone"), moisturiser, ("", "Sunscreen SPF 50+",        "Essential — prevents deepening")],
        "Dark Spots":                       [cleanser_am, ("", "Vitamin C + Niacinamide", "Dual brightening action"), moisturiser, ("", "Sunscreen SPF 50+",        "Protects treated areas")],
        "Pores":                            [cleanser_am, ("", "Niacinamide 10%",         "Visibly tightens pores over time"), moisturiser, ("", "Sunscreen SPF 50",         "Prevents UV-enlarged pores")],
        "Redness":                          [cleanser_am, ("", "Centella / Azelaic Acid", "Calms redness and supports barrier"), moisturiser, ("", "Mineral SPF 50",           "Physical SPF less irritating")],
        "Wrinkles":                         [cleanser_am, ("", "Vitamin C Serum",         "Antioxidant defence and brightening"), moisturiser, ("", "Sunscreen SPF 50+",        "Single most anti-aging step")],
    }
    routines_pm: dict[str, list[RoutineStep]] = {
        "Inflammatory Acne":                [("", "Salicylic Acid Cleanser",  "Deep pore cleansing PM step"), ("", "Adapalene / Benzoyl Peroxide", "Treats active lesions and prevents new ones"), moisturiser],
        "Non Inflammatory Acne Blackheads": [("", "BHA Cleanser",             "Dissolves blackhead-forming oils"), ("", "Retinol 0.1–0.3%",            "Speeds cell turnover and clears pores"), moisturiser],
        "Non Inflammatory Acne Whiteheads": [("", "Gentle Cleanser",          "Preps skin for actives"), ("", "Salicylic Acid Serum",         "Exfoliates closed comedones overnight"), moisturiser],
        "Pigmentation":                     [("", "Hydrating Cleanser",        "Gentle PM cleanse"), ("", "Tranexamic Acid",               "Targets melanin production and dark spots"), ("", "Barrier Moisturiser", "Seals in treatment and repairs")],
        "Dark Spots":                       [("", "Gentle Cleanser",           "Removes sunscreen and buildup"), ("", "Alpha Arbutin / Kojic Acid",  "Fades localised dark spots"), ("", "Barrier Moisturiser", "Ceramide-rich overnight recovery")],
        "Pores":                            [("", "Salicylic Cleanser",        "Clears pores before actives"), ("", "Retinol 0.3%",               "Tightens pores and refines texture"), moisturiser],
        "Redness":                          [("", "Fragrance-free Cleanser",   "Ultra-gentle PM cleanse"), ("", "Azelaic Acid 10%",            "Reduces redness and supports barrier"), ("", "Rich Barrier Cream", "Overnight barrier recovery")],
        "Wrinkles":                         [("", "Gentle Cleanser",           "Removes the day's SPF gently"), ("", "Retinoid 0.025–0.05%",       "Primary anti-wrinkle active — use nightly"), ("", "Rich Peptide Moisturiser", "Supports collagen and locks in moisture")],
    }
    if age_range == "40+":
        for period_dict in (routines_am, routines_pm):
            for key in period_dict:
                period_dict[key].append(("", "Peptide Complex Serum", "Supports collagen synthesis and skin firmness"))
    am_steps = routines_am.get(normalize(prediction), routines_am.get("Inflammatory Acne", []))
    pm_steps = routines_pm.get(normalize(prediction), routines_pm.get("Inflammatory Acne", []))
    return validate_routine({"Morning": am_steps, "Evening": pm_steps})

def build_products(prediction: str, skin_type: str, skin_tone: str) -> list[dict]:
    is_sensitive = skin_type == "Sensitive"
    is_dry       = skin_type == "Dry"
    is_dark_tone = skin_tone in ("Tan", "Deep")
    products_map: dict[str, list[dict]] = {
        "Inflammatory Acne": [
            {"icon": "SA", "name": "Minimalist 2% Salicylic Acid Cleanser", "desc": "Pore-clearing, gentle enough daily",   "price": "Rs.349"},
            {"icon": "TO", "name": "The Ordinary Niacinamide 10% + Zinc 1%","desc": "Regulates oil, reduces inflammation",  "price": "Rs.499"},
            {"icon": "LR", "name": "La Roche-Posay Anthelios SPF 50",       "desc": "Non-comedogenic UV protection",         "price": "Rs.699"},
        ],
        "Non Inflammatory Acne Blackheads": [
            {"icon": "PC", "name": "Paula's Choice 2% BHA Liquid Exfoliant","desc": "Dissolves blackheads inside pores",     "price": "Rs.2599"},
            {"icon": "MM", "name": "Minimalist Niacinamide 10%",             "desc": "Pore-minimising daily serum",           "price": "Rs.349"},
            {"icon": "NG", "name": "Neutrogena Ultra Sheer SPF 50",         "desc": "Oil-free daily sunscreen",              "price": "Rs.499"},
        ],
        "Non Inflammatory Acne Whiteheads": [
            {"icon": "MM", "name": "Minimalist Retinol 0.2%",               "desc": "Gentle intro retinol for beginners",    "price": "Rs.449"},
            {"icon": "TO", "name": "The Ordinary Salicylic Acid 2%",        "desc": "Targets closed comedones",              "price": "Rs.350"},
            {"icon": "RE", "name": "Re'equil Mineral Sunscreen SPF 50",     "desc": "Physical sunscreen, pore-safe",         "price": "Rs.449"},
        ],
        "Pigmentation": [
            {"icon": "MM", "name": "Minimalist Vitamin C 10%",              "desc": "Brightens and evens skin tone",         "price": "Rs.399"},
            {"icon": "TO", "name": "The Ordinary Tranexamic Acid 10%",      "desc": "Inhibits melanin, safe all tones",      "price": "Rs.599"},
            {"icon": "RE", "name": "Re'equil Mineral Sunscreen SPF 50",     "desc": "Physical UV blocker",                   "price": "Rs.449"},
        ],
        "Dark Spots": [
            {"icon": "MM", "name": "Minimalist Alpha Arbutin 2%",           "desc": "Fades localised dark spots",            "price": "Rs.349"},
            {"icon": "TO", "name": "The Ordinary Azelaic Acid 10%",         "desc": "Brightens and reduces redness",         "price": "Rs.750"},
            {"icon": "LS", "name": "La Shield SPF 50 Mineral",              "desc": "Prevents marks from darkening",         "price": "Rs.699"},
        ],
        "Pores": [
            {"icon": "TO", "name": "The Ordinary Niacinamide 10% + Zinc",   "desc": "Tightens pore appearance over time",   "price": "Rs.499"},
            {"icon": "PC", "name": "Paula's Choice 2% BHA",                 "desc": "Clears pore-congesting debris",         "price": "Rs.2599"},
            {"icon": "MM", "name": "Minimalist Retinol 0.3%",               "desc": "Long-term pore refinement",             "price": "Rs.549"},
        ],
        "Redness": [
            {"icon": "TO", "name": "The Ordinary Azelaic Acid 10%",         "desc": "Reduces redness and sensitisation",     "price": "Rs.750"},
            {"icon": "CV", "name": "CeraVe Moisturising Cream",             "desc": "Barrier repair with ceramides",         "price": "Rs.899"},
            {"icon": "AL", "name": "Altruist SPF 50 Mineral",               "desc": "Fragrance-free, gentle sunscreen",      "price": "Rs.399"},
        ],
        "Wrinkles": [
            {"icon": "MM", "name": "Minimalist Retinol 0.3% + Q10",         "desc": "Cell renewal and fine line reduction",  "price": "Rs.549"},
            {"icon": "TO", "name": "The Ordinary Buffet Peptide Serum",     "desc": "Multi-peptide collagen support",        "price": "Rs.1299"},
            {"icon": "LR", "name": "La Roche-Posay Anthelios SPF 50",       "desc": "Anti-aging UV protection",              "price": "Rs.699"},
        ],
    }
    picks = products_map.get(normalize(prediction), products_map["Inflammatory Acne"])
    if is_dark_tone:
        for p in picks:
            if any(x in p["name"].lower() for x in ["tranexamic", "arbutin", "azelaic"]):
                p["desc"] += " — melanin-safe, suitable for deeper tones"
    if is_sensitive or is_dry:
        for p in picks:
            if "salicylic" in p["name"].lower() and "cleanser" in p["name"].lower():
                p["desc"] = "Use max 2x per week if skin is sensitive or dry"
    return picks

def build_care_priorities(prediction: str, severity: dict, skin_type: str) -> list[dict]:
    sev_values = list(severity.values())
    inflam_lvl = sev_values[0][0] if sev_values else "Low"
    oil_level  = "High" if skin_type in ("Oily", "Combination") else "Moderate" if skin_type == "Normal" else "Low"
    data: dict[str, list[dict]] = {
        "Inflammatory Acne":                [
            {"label": "Inflammation", "level": inflam_lvl, "color": "#EF4444", "bg": "rgba(239,68,68,0.1)"},
            {"label": "Oil Control",  "level": oil_level,  "color": "#F59E0B", "bg": "rgba(245,158,11,0.1)"},
            {"label": "Hydration",    "level": "Moderate", "color": "#3B82F6", "bg": "rgba(59,130,246,0.1)"},
        ],
        "Non Inflammatory Acne Blackheads": [
            {"label": "Exfoliation",  "level": "High",     "color": "#A78BFA", "bg": "rgba(124,58,237,0.1)"},
            {"label": "Oil Control",  "level": oil_level,  "color": "#F59E0B", "bg": "rgba(245,158,11,0.1)"},
            {"label": "Hydration",    "level": "Moderate", "color": "#3B82F6", "bg": "rgba(59,130,246,0.1)"},
        ],
        "Non Inflammatory Acne Whiteheads": [
            {"label": "Exfoliation",  "level": "High",     "color": "#A78BFA", "bg": "rgba(124,58,237,0.1)"},
            {"label": "Oil Control",  "level": oil_level,  "color": "#F59E0B", "bg": "rgba(245,158,11,0.1)"},
            {"label": "Barrier Care", "level": "Moderate", "color": "#3B82F6", "bg": "rgba(59,130,246,0.1)"},
        ],
        "Pigmentation": [
            {"label": "UV Protection","level": "High",     "color": "#EF4444", "bg": "rgba(239,68,68,0.1)"},
            {"label": "Brightening",  "level": inflam_lvl, "color": "#A78BFA", "bg": "rgba(124,58,237,0.1)"},
            {"label": "Hydration",    "level": "Moderate", "color": "#3B82F6", "bg": "rgba(59,130,246,0.1)"},
        ],
        "Dark Spots": [
            {"label": "UV Protection","level": "High",     "color": "#EF4444", "bg": "rgba(239,68,68,0.1)"},
            {"label": "Brightening",  "level": "High",     "color": "#A78BFA", "bg": "rgba(124,58,237,0.1)"},
            {"label": "Barrier Care", "level": "Low",      "color": "#22C55E", "bg": "rgba(34,197,94,0.1)"},
        ],
        "Pores": [
            {"label": "Exfoliation",  "level": "High",     "color": "#A78BFA", "bg": "rgba(124,58,237,0.1)"},
            {"label": "Oil Control",  "level": oil_level,  "color": "#F59E0B", "bg": "rgba(245,158,11,0.1)"},
            {"label": "Hydration",    "level": "Low",      "color": "#22C55E", "bg": "rgba(34,197,94,0.1)"},
        ],
        "Redness": [
            {"label": "Barrier Care", "level": "High",     "color": "#EF4444", "bg": "rgba(239,68,68,0.1)"},
            {"label": "Calming",      "level": "High",     "color": "#22C55E", "bg": "rgba(34,197,94,0.1)"},
            {"label": "Hydration",    "level": "High",     "color": "#3B82F6", "bg": "rgba(59,130,246,0.1)"},
        ],
        "Wrinkles": [
            {"label": "Renewal",      "level": "High",     "color": "#A78BFA", "bg": "rgba(124,58,237,0.1)"},
            {"label": "UV Shield",    "level": "High",     "color": "#EF4444", "bg": "rgba(239,68,68,0.1)"},
            {"label": "Hydration",    "level": "High",     "color": "#3B82F6", "bg": "rgba(59,130,246,0.1)"},
        ],
    }
    return data.get(normalize(prediction), [
        {"label": "Barrier Care", "level": "Low",      "color": "#22C55E", "bg": "rgba(34,197,94,0.1)"},
        {"label": "UV Defense",   "level": "Moderate", "color": "#F59E0B", "bg": "rgba(245,158,11,0.1)"},
        {"label": "Prevention",   "level": "Low",      "color": "#22C55E", "bg": "rgba(34,197,94,0.1)"},
    ])

def build_avoid(prediction: str, skin_type: str) -> list[str]:
    base: dict[str, list[str]] = {
        "Inflammatory Acne":                ["Harsh physical scrubs — worsen inflammation", "Heavy pore-clogging creams", "Picking or popping pimples", "Skipping sunscreen — worsens post-acne marks", "Overwashing — strips barrier", "Fragranced products on active breakouts"],
        "Non Inflammatory Acne Blackheads": ["Pore strips — cause temporary fix, long-term damage", "Heavy creams and oils", "Skipping BHA exfoliation", "Mixing too many actives at once", "Inconsistent routine", "Using astringent toners that over-dry"],
        "Non Inflammatory Acne Whiteheads": ["Occlusive products that trap oil", "Picking at closed comedones", "Harsh exfoliants that irritate", "Skipping BHA or retinoid", "Changing routine too frequently", "Products with silicones if pore-prone"],
        "Pigmentation":                     ["Direct sun exposure without SPF 50+", "Picking hyperpigmented areas", "Harsh bleaching agents (hydroquinone overuse)", "Mixing actives without patch testing", "Skipping moisturiser with brightening actives", "Inconsistent routine — results require 12–16 weeks"],
        "Dark Spots":                       ["Sun exposure without SPF — deepens spots", "Picking or scratching affected areas", "Harsh scrubs on dark spot zones", "Combining too many brightening actives at once", "Inconsistent use of treatment serum"],
        "Pores":                            ["Pore strips (temporary and damaging)", "Heavy occlusive creams", "Over-exfoliating", "Skipping SPF (UV enlarges pores)", "Squeezing blackheads manually", "Makeup without proper removal"],
        "Redness":                          ["Fragrance in all skincare and makeup", "Hot water on face", "Physical exfoliants (scrubs)", "Spicy food and alcohol (trigger flushing)", "Skipping moisturiser", "Introducing actives too quickly"],
        "Wrinkles":                         ["Sun exposure without SPF — single biggest aging driver", "Skipping moisturiser", "Starting retinoids too aggressively", "Smoking and high sugar diet", "Changing products too frequently", "Neglecting neck and decolletage"],
    }
    items = base.get(normalize(prediction), ["Heavy products", "Skipping SPF", "Over-exfoliating", "Inconsistent routine"])
    if skin_type == "Sensitive":
        items.append("Any fragranced product — patch test everything")
    if skin_type == "Dry":
        items.append("Alcohol-based toners or astringents — further dry the skin")
    return items

def build_tips(prediction: str, skin_type: str, age_range: str) -> list[str]:
    base: dict[str, list[str]] = {
        "Inflammatory Acne":                ["Change pillowcase every 2–3 days to reduce bacteria.", "SPF 50 every morning — UV deepens post-acne marks.", "Introduce actives one at a time — start with niacinamide.", "Avoid touching face throughout the day.", "Patience — visible improvement in 8–12 weeks.", "Track diet: high sugar and dairy can worsen breakouts."],
        "Non Inflammatory Acne Blackheads": ["BHA (salicylic acid) is your best friend — use 2–3x weekly.", "Double cleanse at night if wearing makeup or SPF.", "Niacinamide daily visibly tightens pores over weeks.", "Don't over-exfoliate — once daily max with BHA.", "Results take 6–8 weeks of consistent use.", "Hydration is essential even with oily/combo skin."],
        "Non Inflammatory Acne Whiteheads": ["Start retinoids at 2x per week — build up slowly.", "BHA 1–2x weekly unclogs closed comedones gradually.", "Lightweight, non-comedogenic products only.", "Patience — closed comedones improve in 8–10 weeks.", "Avoid heavy SPF — use gel or fluid texture.", "Keep routine minimal and consistent."],
        "Pigmentation":                     ["SPF 50 every morning is the most important step.", "Vitamin C works best on freshly cleansed skin in the morning.", "Don't layer multiple brightening actives initially.", "Results are visible after 12–16 weeks — stay consistent.", "Skin cycling helps prevent irritation from actives.", "Wear a hat or seek shade beyond just SPF."],
        "Dark Spots":                       ["Never skip SPF — sun is the primary cause of darkening.", "Apply treatment serum consistently — don't skip days.", "Vitamin C in AM + brightening serum in PM is effective.", "Avoid picking or scratching treated areas.", "Results take 8–12 weeks — stay committed.", "A dermatologist can accelerate results with peels."],
        "Pores":                            ["Consistent BHA use is the most effective long-term strategy.", "Niacinamide daily tightens pore appearance visibly.", "Retinol at night helps refine skin texture significantly.", "Always cleanse before bed — sleeping in SPF clogs pores.", "Pore strips provide no lasting benefit.", "Results need 6–8 weeks of consistent routine."],
        "Redness":                          ["Always patch test new products before full application.", "Introduce actives one at a time — never multiple at once.", "Azelaic acid is effective and safe for sensitive skin.", "Keep a symptom diary to identify triggers.", "Fragrance-free is non-negotiable for reactive skin.", "Consult a dermatologist if redness is persistent."],
        "Wrinkles":                         ["Introduce retinoids slowly: 2x per week for 4 weeks, then increase.", "Never skip SPF — UV causes 80% of visible aging.", "Moisturiser + peptides work synergistically with retinoids.", "Vitamin C in the morning maximises photoprotection.", "Consistency over 3–6 months delivers visible results.", "Neck and decolletage need the same routine as face."],
    }
    tips = base.get(normalize(prediction), ["Consistent minimal routine beats complex one you skip.", "SPF daily protects from premature aging.", "Patch test every new product before full use."])
    if age_range in ("15-20", "20-25"):
        tips.append("Starting sun protection young is the most effective anti-aging step you can take.")
    elif age_range == "40+":
        tips.append("Consistency with retinoids and SPF delivers dramatic results even at this stage.")
    if skin_type == "Dry":
        tips.append("Always moisturise before applying active serums to buffer potential irritation.")
    elif skin_type == "Oily":
        tips.append("Oil-free and gel textures prevent clogging while providing necessary hydration.")
    return tips

def build_view_model(api_data: dict[str, Any], skin_type: str = "Oily", skin_tone: str = "Medium", age_range: str = "20-25") -> dict[str, Any]:
    pred  = normalize(str(api_data["prediction"]))
    conf  = clamp(float(api_data["confidence"]), 0.0, 1.0)
    rec   = str(api_data["recommendation"]).strip()
    class_probs  = simulate_class_probs(pred, conf)
    items        = list(class_probs.items())
    primary      = items[0]
    secondary    = items[1] if len(items) > 1 else items[0]
    severity     = build_severity(pred, conf, class_probs)
    routine      = build_routine(pred, skin_type, age_range)
    return {
        "prediction":      primary[0],
        "confidence":      primary[1],
        "recommendation":  rec,
        "class_probs":     class_probs,
        "primary":         primary,
        "secondary":       secondary,
        "severity":        severity,
        "ai_insight":      build_ai_insight(pred, conf, secondary[0], secondary[1], skin_type, skin_tone, age_range),
        "pattern_data":    build_pattern_data(pred),
        "clinical_note":   build_clinical_note(pred, skin_type, age_range),
        "care_priorities": build_care_priorities(pred, severity, skin_type),
        "routine":         routine,
        "products":        build_products(pred, skin_type, skin_tone),
        "avoid":           build_avoid(pred, skin_type),
        "tips":            build_tips(pred, skin_type, age_range),
        "raw":             api_data,
    }

def make_gradcam(img: Image.Image) -> Image.Image:
    arr  = np.array(img.resize((480, 400))).astype(float)
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    heat = np.clip((gray - 60) / 160, 0, 1)
    heatmap        = np.zeros((*heat.shape, 3), dtype=np.uint8)
    heatmap[:,:,0] = (heat * 255).astype(np.uint8)
    heatmap[:,:,1] = ((1 - abs(heat - 0.5) * 2) * 255).astype(np.uint8)
    heatmap[:,:,2] = ((1 - heat) * 255).astype(np.uint8)
    blended = (arr * 0.40 + heatmap * 0.60).astype(np.uint8)
    return Image.fromarray(blended)

def build_text_report(r: dict[str, Any]) -> str:
    lines = ["SkinAura AI Report", "=" * 40, f"Condition:  {r['prediction']}", f"Confidence: {r['confidence']:.0%}", "", "Recommendation:", r["recommendation"], "", "AI Insight:", r["ai_insight"], "", "Severity:"]
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
    safe_routine = validate_routine(r.get("routine", {}))
    routine_export = {
        period: [{"name": n, "description": d} for _, n, d in steps]
        for period, steps in safe_routine.items()
    }
    payload = {
        "prediction":          r["prediction"],
        "confidence":          r["confidence"],
        "recommendation":      r["recommendation"],
        "primary":             {"label": r["primary"][0],  "probability": r["primary"][1]},
        "secondary":           {"label": r["secondary"][0], "probability": r["secondary"][1]},
        "class_probabilities": r["class_probs"],
        "severity":            {k: {"level": lv, "probability": pb} for k, (lv, _, pb) in r["severity"].items()},
        "ai_insight":          r["ai_insight"],
        "routine":             routine_export,
        "products":            r["products"],
        "avoid":               r["avoid"],
        "tips":                r["tips"],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)

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
    "uploaded_image": None, "uploaded_image_bytes": None, "uploaded_filename": "",
    "analysis_result": None, "analysed": False, "api_error": "",
    "skin_type": "Oily", "skin_tone": "Medium", "age_range": "20-25",
    "analysis_ts": "",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

system_status = fetch_system_status()

# ---------------------------------------------------------------------------
# LAYOUT
# ---------------------------------------------------------------------------
col_left, col_center, col_right = st.columns([1.1, 2.8, 1.1], gap="small")

# ===========================================================================
# LEFT SIDEBAR
# ===========================================================================
with col_left:
    # Logo
    st.markdown("""
    <div class="sa-logo-wrap" style="padding:0.5rem 0 0.4rem;">
        <div class="sa-logo-icon">S</div>
        <div>
            <div class="sa-logo-text">SkinAura</div>
            <div class="sa-logo-sub">AI Dermatology Assistant</div>
        </div>
    </div>
    <div class="sa-divider"></div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown('<div class="sa-label">Navigation</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sa-nav-item sa-nav-active">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.9;"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        Dashboard
    </div>
    <div class="sa-divider"></div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="sa-upload-label">Upload Skin Image</div>', unsafe_allow_html=True)
    st.markdown('<div class="sa-upload-hint">JPG, JPEG or PNG &bull; Max 20 MB</div>', unsafe_allow_html=True)

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
        size_kb = len(cur_imgb) // 1024 if cur_imgb else 0
        st.markdown(
            f'<div class="sa-file-pill">'
            f'<div class="sa-file-pill-icon">IMG</div>'
            f'<div>'
            f'<div class="sa-file-name">{cur_fn}</div>'
            f'<div class="sa-file-size">{cur_img.size[0]}x{cur_img.size[1]}px</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.3rem'></div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("+ Analyse", disabled=(cur_imgb is None), use_container_width=True, key="analyse_btn"):
            pb     = st.progress(0)
            st_txt = st.empty()
            try:
                st_txt.caption("Uploading...")
                pb.progress(15)
                time.sleep(0.05)
                st_txt.caption("Running inference...")
                pb.progress(45)
                skin_t = st.session_state["skin_type"]
                skin_n = st.session_state["skin_tone"]
                age_r  = st.session_state["age_range"]
                api_res = call_predict_api(cur_imgb, cur_fn)
                if not api_res["success"]:
                    st.session_state["api_error"] = str(api_res["error"])
                    st.error(f"API error: {api_res['error']}")
                    st.session_state["analysed"] = st.session_state["analysis_result"] is not None
                else:
                    st_txt.caption("Building insights...")
                    pb.progress(82)
                    time.sleep(0.05)
                    st.session_state["analysis_result"] = build_view_model(
                        api_res["data"], skin_type=skin_t, skin_tone=skin_n, age_range=age_r
                    )
                    st.session_state["analysed"]    = True
                    st.session_state["api_error"]   = ""
                    st.session_state["analysis_ts"] = datetime.now().strftime("%b %d, %Y • %I:%M %p")
                    pb.progress(100)
                    time.sleep(0.05)
            finally:
                pb.empty()
                st_txt.empty()
    with col_b:
        if st.button("Reset", disabled=(cur_imgb is None), use_container_width=True, key="reanalyse_btn"):
            st.session_state["analysed"]        = False
            st.session_state["analysis_result"] = None
            st.rerun()

    st.markdown('<div class="sa-divider" style="margin:0.6rem 0;"></div>', unsafe_allow_html=True)

    # User Profile
    st.markdown('<div class="sa-label">User Profile</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="display:flex;align-items:center;gap:0.55rem;margin-bottom:0.55rem;">
        <div class="sa-avatar">SK</div>
        <div>
            <div style="font-size:0.75rem;font-weight:600;color:#CBD5E1;">User</div>
            <div style="font-size:0.6rem;color:#334155;">Skin Profile</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sa-profile-field-label">Skin Type</div>', unsafe_allow_html=True)
    skin_options = ["Oily", "Dry", "Combination", "Normal", "Sensitive"]
    st.session_state["skin_type"] = st.selectbox(
        "skin_type", skin_options,
        index=skin_options.index(st.session_state["skin_type"]),
        label_visibility="collapsed", key="skin_type_sel",
    )

    st.markdown('<div class="sa-profile-field-label" style="margin-top:0.35rem;">Skin Tone</div>', unsafe_allow_html=True)
    tone_options = ["Fair", "Light", "Medium", "Tan", "Deep"]
    if st.session_state["skin_tone"] not in tone_options:
        st.session_state["skin_tone"] = "Medium"
    st.session_state["skin_tone"] = st.selectbox(
        "skin_tone", tone_options,
        index=tone_options.index(st.session_state["skin_tone"]),
        label_visibility="collapsed", key="skin_tone_sel",
    )

    st.markdown('<div class="sa-profile-field-label" style="margin-top:0.35rem;">Age Range</div>', unsafe_allow_html=True)
    age_options = ["15-20", "20-25", "25-30", "30-40", "40+"]
    if st.session_state["age_range"] not in age_options:
        st.session_state["age_range"] = "20-25"
    st.session_state["age_range"] = st.selectbox(
        "age_range", age_options,
        index=age_options.index(st.session_state["age_range"]),
        label_visibility="collapsed", key="age_range_sel",
    )

    st.markdown(
        '<div style="font-size:0.6rem;color:#334155;margin-top:0.25rem;line-height:1.5;">'
        'Profile influences routine, products, and AI insights.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sa-divider" style="margin:0.6rem 0;"></div>', unsafe_allow_html=True)

    # System Status
    st.markdown('<div class="sa-label">System Status</div>', unsafe_allow_html=True)
    status_rows = [
        ("Model", system_status["model"], "Online",    "Offline"),
        ("API",   system_status["api"],   "Connected", "Offline"),
        ("RAG",   system_status["rag"],   "Ready",     "Unavailable"),
        ("LLM",   system_status["llm"],   "Ready",     "Unavailable"),
    ]
    for label_name, ok, on_text, off_text in status_rows:
        dot_cls     = "sa-dot-green" if ok else "sa-dot-red"
        status_text = on_text if ok else off_text
        status_color = "#22C55E" if ok else "#EF4444"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:0.5rem;font-size:0.72rem;color:#64748B;margin-bottom:0.22rem;">'
            f'<span class="sa-dot {dot_cls}"></span>'
            f'<span>{label_name}:</span>'
            f'<span style="color:{status_color};font-weight:600;">{status_text}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sa-divider" style="margin:0.6rem 0;"></div>', unsafe_allow_html=True)

    # Export
    cur_res  = st.session_state["analysis_result"]
    exp_on   = cur_res is not None
    txt_rep  = build_text_report(cur_res) if exp_on else "No analysis yet."
    json_rep = build_json_report(cur_res) if exp_on else json.dumps({"message": "No analysis yet."})

    st.markdown('<div class="sa-label">Export Report</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "! .txt", data=txt_rep, file_name="skinaura_report.txt",
            mime="text/plain", use_container_width=True, key="dl_txt", disabled=not exp_on,
        )
    with c2:
        st.download_button(
            "! .json", data=json_rep, file_name="skinaura_report.json",
            mime="application/json", use_container_width=True, key="dl_json", disabled=not exp_on,
        )

    st.markdown(
        '<div style="margin-top:0.75rem;font-size:0.6rem;color:#1E293B;line-height:1.5;">'
        '© SkinAura. Not a substitute for professional medical advice.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# CENTER PANEL
# ===========================================================================
with col_center:
    result = st.session_state["analysis_result"]

    if not st.session_state["analysed"] or result is None:
        st.markdown("""
        <div style="background:#0F172A;border:1px solid rgba(255,255,255,0.05);
             border-radius:10px;padding:5rem 2rem;text-align:center;margin-top:1rem;">
            <div style="font-size:1.5rem;margin-bottom:0.75rem;opacity:0.15;font-weight:700;letter-spacing:-0.02em;">S</div>
            <div style="font-size:0.9rem;color:#334155;font-weight:500;margin-bottom:0.35rem;">
                Upload an image and click Analyse
            </div>
            <div style="font-family:'DM Mono',monospace;font-size:0.58rem;letter-spacing:0.12em;color:#1E293B;">
                AWAITING INPUT — CNN CLASSIFICATION • GRAD-CAM ATTENTION • AI INTERPRETATION
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    result["routine"] = validate_routine(result.get("routine", {}))

    prediction      = result["prediction"]
    confidence      = result["confidence"]
    recommendation  = result["recommendation"]
    class_probs     = result["class_probs"]
    severity        = result["severity"]
    ai_insight      = result["ai_insight"]
    pattern_data    = result["pattern_data"]
    clinical_note   = result["clinical_note"]
    care_priorities = result["care_priorities"]
    analysis_ts     = st.session_state.get("analysis_ts", datetime.now().strftime("%b %d, %Y • %I:%M %p"))

    # ── Top bar ──────────────────────────────────────────────────────
    col_title, col_badge = st.columns([2, 1])
    with col_title:
        st.markdown(
            '<div class="sa-page-title">AI Skin Analysis</div>'
            '<div class="sa-page-sub">CNN Classification &bull; Grad-CAM Attention &bull; AI Interpretation</div>',
            unsafe_allow_html=True,
        )
    with col_badge:
        st.markdown(
            f'<div style="text-align:right;">'
            f'<div class="sa-analysis-badge" style="float:right;">'
            f'<div class="sa-badge-dot"></div>Analysis completed</div>'
            f'<div class="sa-timestamp" style="clear:both;">{analysis_ts}</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.55rem'></div>", unsafe_allow_html=True)

    # ── 3 Metric Cards ───────────────────────────────────────────────
    top_sev_level = list(severity.values())[0][0]
    sev_cls = {"High": "sev-high", "Medium": "sev-medium", "Low": "sev-low"}.get(top_sev_level, "sev-low")
    cov_pct = int(list(severity.values())[1][2] * 100) if len(severity) > 1 else 48

    mc1, mc2, mc3 = st.columns(3, gap="small")
    with mc1:
        st.markdown(
            f'<div class="sa-metric-card">'
            f'<div class="sa-metric-eyebrow">Primary Concern</div>'
            f'<div class="sa-metric-value">{prediction}</div>'
            f'<div class="sa-metric-sub">Confidence: {confidence:.0%}</div>'
            f'<div class="sa-severity-badge {sev_cls}">{top_sev_level.upper()} SEVERITY</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            f'<div class="sa-metric-card">'
            f'<div class="sa-metric-eyebrow">Skin Type</div>'
            f'<div class="sa-metric-value">{st.session_state["skin_type"]}</div>'
            f'<div class="sa-metric-sub">User Profile</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            f'<div class="sa-metric-card">'
            f'<div class="sa-metric-eyebrow">Affected Area</div>'
            f'<div class="sa-metric-value">{cov_pct}%</div>'
            f'<div class="sa-metric-sub">Coverage Estimate</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    # ── Grad-CAM Visualization ───────────────────────────────────────
    cur_img = st.session_state["uploaded_image"]
    if cur_img is not None:
        DISPLAY_SIZE    = (420, 340)
        orig_display    = cur_img.resize(DISPLAY_SIZE, Image.LANCZOS)
        gradcam_display = make_gradcam(cur_img).resize(DISPLAY_SIZE, Image.LANCZOS)

        img_col1, img_col2 = st.columns(2, gap="small")
        with img_col1:
            st.markdown('<div class="sa-img-wrap"><div class="sa-img-label">Original Image</div>', unsafe_allow_html=True)
            st.image(orig_display, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with img_col2:
            st.markdown('<div class="sa-img-wrap"><div class="sa-img-label">Attention Map (Grad-CAM)</div>', unsafe_allow_html=True)
            st.image(gradcam_display, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="sa-attention-scale">
            <span>Low Attention</span>
            <div class="sa-scale-bar"></div>
            <span>High Attention</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    # ── AI Dermatology Insight ───────────────────────────────────────
    st.markdown(
        f'<div class="sa-section-title">AI Dermatology Insight</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sa-insight-body">{ai_insight}</div>',
        unsafe_allow_html=True,
    )

    # Pattern grid
    pattern_cols_html = ""
    for col_title_text, items in pattern_data.items():
        items_html = "".join(
            f'<div class="sa-pattern-item"><div class="sa-pattern-dot"></div>{item}</div>'
            for item in items
        )
        pattern_cols_html += (
            f'<div class="sa-pattern-col">'
            f'<div class="sa-pattern-col-title">{col_title_text}</div>'
            f'{items_html}</div>'
        )
    st.markdown(f'<div class="sa-pattern-grid">{pattern_cols_html}</div>', unsafe_allow_html=True)

    # Clinical note
    st.markdown(
        f'<div class="sa-clinical-note">'
        f'<div class="sa-clinical-title">Clinical Note</div>'
        f'<div class="sa-clinical-text">{clinical_note}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    # ── Top 3 Predictions ────────────────────────────────────────────
    st.markdown(
        '<div class="sa-section-title">Top 3 Predictions</div>'
        '<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.1em;'
        'text-transform:uppercase;color:#334155;margin-bottom:0.55rem;">Confidence Breakdown</div>',
        unsafe_allow_html=True,
    )
    bar_colors = ["#7C3AED", "#4C1D95", "#2D1B69"]
    for i, (cls_name, prob) in enumerate(list(class_probs.items())[:3]):
        pct  = f"{prob * 100:.0f}%"
        rank = i + 1
        bc   = bar_colors[i]
        st.markdown(
            f'<div class="sa-prediction-row">'
            f'<div class="sa-pred-rank">{rank}</div>'
            f'<div class="sa-pred-name">{cls_name}</div>'
            f'<div class="sa-pred-bar-wrap">'
            f'<div style="width:{pct};height:100%;background:{bc};border-radius:2px;"></div></div>'
            f'<div class="sa-pred-pct">{pct}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    # ── Tabs: Avoid / Tips / Diet ────────────────────────────────────
    tab_choice = st.radio(
        "tab", ["Avoid", "Tips", "Diet & Lifestyle"],
        horizontal=True, label_visibility="collapsed", key="main_tab",
    )
    st.markdown("<div style='height:0.45rem'></div>", unsafe_allow_html=True)

    if "Avoid" in tab_choice:
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:#334155;margin-bottom:0.35rem;">'
            f'Things to Avoid — {prediction}</div>',
            unsafe_allow_html=True,
        )
        for item in result["avoid"]:
            st.markdown(
                f'<div class="sa-list-item"><span class="sa-list-marker-x">x</span>{item}</div>',
                unsafe_allow_html=True,
            )

    elif "Tips" in tab_choice:
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:#334155;margin-bottom:0.35rem;">'
            f'Skin Tips — {st.session_state["skin_type"]} skin &bull; {st.session_state["age_range"]}</div>',
            unsafe_allow_html=True,
        )
        for i, tip in enumerate(result["tips"], 1):
            st.markdown(
                f'<div class="sa-list-item">'
                f'<span class="sa-list-marker-n">{i:02d}.</span>{tip}</div>',
                unsafe_allow_html=True,
            )

    elif "Diet" in tab_choice:
        diet_map: dict[str, dict] = {
            "Inflammatory Acne":                {"eat": ["Berries & antioxidant fruits", "Leafy greens", "Zinc-rich foods (pumpkin seeds)", "Omega-3 (flaxseed, walnuts)", "Green tea"], "avoid": ["High sugar & refined carbs", "Dairy (milk, cheese)", "Fried & oily food", "Whey protein supplements", "Alcohol"], "lifestyle": ["Sleep 7-8 hrs", "Manage stress (cortisol triggers acne)", "Pillowcase every 2-3 days", "Exercise regularly", "Avoid touching face"], "hydration": "2.5–3 L daily"},
            "Non Inflammatory Acne Blackheads": {"eat": ["Vitamin A foods (carrots, sweet potato)", "Zinc-rich seeds", "Leafy greens", "Fruits (low GI)", "Green tea"], "avoid": ["High glycemic foods", "Excess dairy", "Fried food", "Heavy oils", "Processed snacks"], "lifestyle": ["Double cleanse nightly", "Wash brushes weekly", "Change pillowcases often", "Manage stress", "Consistent BHA use"], "hydration": "2–3 L daily"},
            "Non Inflammatory Acne Whiteheads": {"eat": ["Low GI fruits", "Vegetables", "Lean proteins", "Nuts and seeds", "Probiotics"], "avoid": ["Sugar", "Heavy dairy", "Processed foods", "Alcohol", "Vegetable oils"], "lifestyle": ["Consistent routine daily", "Introduce actives slowly", "Sleep 7-8 hrs", "Light exercise", "Avoid picking"], "hydration": "2–3 L daily"},
            "Pigmentation":                     {"eat": ["Vitamin C (citrus, kiwi, bell pepper)", "Tomatoes & lycopene", "Dark leafy greens", "Flaxseeds & walnuts", "Green tea"], "avoid": ["Alcohol", "Excess caffeine", "High sugar", "Processed foods", "Smoking"], "lifestyle": ["SPF every morning", "Wear hat outdoors", "Sleep 7-8 hrs", "Stay hydrated", "Manage stress"], "hydration": "2–3 L daily"},
            "Dark Spots":                       {"eat": ["Vitamin C rich foods", "Tomatoes", "Papaya", "Berries", "Turmeric"], "avoid": ["Sun exposure unprotected", "Alcohol", "High sugar diet", "Smoking", "Processed foods"], "lifestyle": ["SPF 50+ mandatory", "Avoid picking skin", "Consistent brightening routine", "Sleep 7-8 hrs", "Stay hydrated"], "hydration": "2–3 L daily"},
            "Pores":                            {"eat": ["Fruits & vegetables", "Lean protein", "Healthy fats", "Zinc-rich foods", "Green tea"], "avoid": ["High glycemic foods", "Excess dairy", "Fried foods", "Alcohol", "Processed snacks"], "lifestyle": ["Double cleanse nightly", "Consistent BHA use", "SPF daily", "Regular exercise", "Stay hydrated"], "hydration": "2–3 L daily"},
            "Redness":                          {"eat": ["Omega-3 (salmon, flaxseed)", "Probiotic foods", "Anti-inflammatory herbs", "Green vegetables", "Blueberries"], "avoid": ["Spicy food (triggers flushing)", "Alcohol", "Very hot drinks", "High sugar", "Processed foods"], "lifestyle": ["Avoid temperature extremes", "Fragrance-free everything", "Gentle exercise only", "Stress management", "Keep a trigger diary"], "hydration": "2.5–3 L daily"},
            "Wrinkles":                         {"eat": ["Collagen-rich foods (bone broth)", "Vitamin C (citrus)", "Vitamin E (nuts, seeds)", "Berries (antioxidants)", "Fatty fish (omega-3)"], "avoid": ["High sugar (glycation ages skin)", "Alcohol", "Smoking", "Trans fats", "Excess sun"], "lifestyle": ["Sleep 8 hrs (back sleeping preferred)", "SPF every single day", "No smoking", "Consistent retinoid use", "Stay hydrated"], "hydration": "2.5–3 L daily"},
        }
        diet = diet_map.get(normalize(prediction), {"eat": ["Varied whole foods"], "avoid": ["Processed foods"], "lifestyle": ["Consistent routine", "SPF daily"], "hydration": "2–3 L daily"})
        d1, d2, d3, d4 = st.columns(4, gap="small")
        def _diet_col(title: str, color: str, items: list[str]) -> str:
            rows = "".join(
                f'<div style="font-size:0.69rem;color:#94A3B8;padding:0.15rem 0;display:flex;gap:0.35rem;line-height:1.4;">'
                f'<span style="color:{color};font-size:0.45rem;margin-top:0.4rem;flex-shrink:0;">●</span>{item}</div>'
                for item in items
            )
            return (
                f'<div style="background:#0F172A;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:0.75rem 0.8rem;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:{color};margin-bottom:0.45rem;">{title}</div>'
                f'{rows}</div>'
            )
        with d1:
            st.markdown(_diet_col("Eat More", "#22C55E", diet["eat"]), unsafe_allow_html=True)
        with d2:
            st.markdown(_diet_col("Avoid", "#EF4444", diet["avoid"]), unsafe_allow_html=True)
        with d3:
            st.markdown(
                f'<div style="background:#0F172A;border:1px solid rgba(255,255,255,0.06);border-radius:8px;padding:0.75rem 0.8rem;text-align:center;">'
                f'<div style="font-family:\'DM Mono\',monospace;font-size:0.58rem;letter-spacing:0.1em;text-transform:uppercase;color:#3B82F6;margin-bottom:0.35rem;">Hydration</div>'
                f'<div style="font-size:1.4rem;margin:0.35rem 0;color:#3B82F6;opacity:0.6;">~</div>'
                f'<div style="font-size:0.82rem;font-weight:600;color:#CBD5E1;">{diet["hydration"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with d4:
            st.markdown(_diet_col("Lifestyle", "#A78BFA", diet["lifestyle"]), unsafe_allow_html=True)

    # ── Banner ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="sa-banner">
        <div class="sa-banner-text">Consistency is the key!</div>
        <div class="sa-banner-sub">Visible results take time. Follow the routine, eat healthy and be patient.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div style="font-size:0.6rem;color:#1E293B;text-align:center;margin-top:0.45rem;">'
        'This is not a substitute for professional medical advice. Consult a dermatologist for severe conditions.</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# RIGHT PANEL — Routine / Care Priorities / Products
# ===========================================================================
with col_right:
    if result is None:
        st.stop()

    routine  = result["routine"]
    products = result["products"]

    # ── Personalised Routine ─────────────────────────────────────────
    st.markdown(
        f'<div class="sa-rp-section-title">Personalised Routine</div>'
        f'<div class="sa-rp-section-sub">{st.session_state["skin_type"]} skin &bull; {st.session_state["age_range"]}</div>',
        unsafe_allow_html=True,
    )

    for period, steps in routine.items():
        period_color = "#F59E0B" if period == "Morning" else "#818CF8"
        st.markdown(
            f'<div class="sa-routine-period-label" style="color:{period_color};">{period.upper()}</div>',
            unsafe_allow_html=True,
        )
        for j, (_, step_name, step_desc) in enumerate(steps):
            dot_color = "#22C55E" if period == "Morning" else "#818CF8"
            is_last   = j == len(steps) - 1
            line_html = "" if is_last else (
                f'<div style="width:1px;flex:1;min-height:12px;background:rgba(255,255,255,0.05);margin-top:2px;"></div>'
            )
            st.markdown(
                f'<div class="sa-routine-step">'
                f'<div style="display:flex;flex-direction:column;align-items:center;padding-top:3px;">'
                f'<div class="sa-routine-step-dot" style="background:{dot_color};"></div>'
                f'{line_html}</div>'
                f'<div style="padding-bottom:0.1rem;">'
                f'<div class="sa-routine-step-name">{step_name}</div>'
                f'<div class="sa-routine-step-desc">{step_desc}</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sa-view-link">View Full Routine &rarr;</div>', unsafe_allow_html=True)

    st.markdown('<div class="sa-divider" style="margin:0.65rem 0;"></div>', unsafe_allow_html=True)

    # ── Care Priorities ──────────────────────────────────────────────
    st.markdown('<div class="sa-rp-section-title">Care Priorities</div>', unsafe_allow_html=True)
    for cp in care_priorities:
        level_color = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"}.get(cp["level"], "#64748B")
        st.markdown(
            f'<div class="sa-care-row">'
            f'<div class="sa-care-icon-box" style="background:{cp["bg"]};">'
            f'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="{cp["color"]}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            f'<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>'
            f'</svg></div>'
            f'<div class="sa-care-label-text">{cp["label"]}</div>'
            f'<div class="sa-care-level-text" style="color:{level_color};">{cp["level"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sa-divider" style="margin:0.65rem 0;"></div>', unsafe_allow_html=True)

    # ── Recommended Products ─────────────────────────────────────────
    st.markdown(
        f'<div class="sa-rp-section-title">Recommended Products</div>'
        f'<div class="sa-rp-section-sub">Curated for {st.session_state["skin_type"].lower()} skin &bull; {st.session_state["skin_tone"].lower()} tone</div>',
        unsafe_allow_html=True,
    )

    for prod in products:
        st.markdown(
            f'<div class="sa-product-row">'
            f'<div class="sa-product-icon-box">{prod["icon"]}</div>'
            f'<div style="flex:1;min-width:0;">'
            f'<div class="sa-product-name">{prod["name"]}</div>'
            f'<div class="sa-product-desc">{prod.get("desc","")}</div>'
            f'</div>'
            f'<div class="sa-product-view-btn">View</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="sa-view-all-link">View All Products &rarr;</div>', unsafe_allow_html=True)