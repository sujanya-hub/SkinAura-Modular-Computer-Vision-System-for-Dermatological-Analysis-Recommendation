"""
app.py — SkinAura: AI-Powered Skincare & Makeup Recommendation System
======================================================================
Production-grade Streamlit frontend.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

import requests
import streamlit as st
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

# FIX: API_URL now includes /api/v1 so _base("health") → /api/v1/health
#      and _base("predict") → /api/v1/predict  (matches backend prefix)
API_URL: str = os.getenv("SKINAURA_API_URL", "http://localhost:8000/api/v1")
REQUEST_TIMEOUT: int = int(os.getenv("SKINAURA_TIMEOUT", "60"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("skinaura.frontend")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SkinAura — AI Skincare",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────────────────────────────────────

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0A0F !important;
    color: #E8E0D5 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

/* ── Typography globals ── */
h1, h2, h3, h4, h5, h6,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    font-family: 'Cormorant Garamond', serif !important;
    letter-spacing: -0.01em;
}

p, li, span, label, div {
    font-family: 'Outfit', sans-serif !important;
}

code, pre, [data-testid="stCode"] {
    font-family: 'DM Mono', monospace !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0E0E16 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #C8C0B5 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #E8E0D5 !important;
    font-family: 'Cormorant Garamond', serif !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #C9A96E 0%, #E8C97A 50%, #C9A96E 100%) !important;
    color: #0A0A0F !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(201,169,110,0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 32px rgba(201,169,110,0.40) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px dashed rgba(201,169,110,0.35) !important;
    border-radius: 4px !important;
    padding: 0.5rem !important;
    transition: border-color 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(201,169,110,0.7) !important;
}
[data-testid="stFileUploader"] * { color: #C8C0B5 !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #7A7A8A !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Cormorant Garamond', serif !important;
    font-size: 1.8rem !important;
    color: #C9A96E !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.025) !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 4px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    color: #E8E0D5 !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #C9A96E, #E8C97A) !important;
    border-radius: 2px !important;
}

/* ── Info / warning / error ── */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    border-left-width: 3px !important;
    font-family: 'Outfit', sans-serif !important;
}

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.5rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(201,169,110,0.3); border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: rgba(201,169,110,0.6); }

/* ── Custom cards ── */
.sa-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 4px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
}
.sa-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #C9A96E, #E8C97A);
}
.sa-card-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #C9A96E;
    margin-bottom: 0.5rem;
}
.sa-card-content {
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    color: #E8E0D5;
    line-height: 1.7;
}
.sa-badge {
    display: inline-block;
    background: rgba(201,169,110,0.12);
    border: 1px solid rgba(201,169,110,0.3);
    color: #C9A96E;
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    padding: 0.2rem 0.65rem;
    border-radius: 2px;
    margin: 0.15rem;
}
.sa-badge-issue {
    background: rgba(220,80,80,0.1);
    border-color: rgba(220,80,80,0.3);
    color: #E87A7A;
}
.sa-confidence-bar {
    height: 3px;
    background: rgba(255,255,255,0.08);
    border-radius: 2px;
    margin: 0.3rem 0 0.6rem;
    overflow: hidden;
}
.sa-confidence-fill {
    height: 100%;
    background: linear-gradient(90deg, #C9A96E, #E8C97A);
    border-radius: 2px;
    transition: width 0.8s ease;
}
.sa-tone-swatch {
    width: 48px; height: 48px;
    border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.15);
    display: inline-block;
    vertical-align: middle;
    margin-right: 0.75rem;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
}
.sa-section-header {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.4rem;
    font-weight: 300;
    color: #E8E0D5;
    letter-spacing: 0.02em;
    margin: 1.8rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.sa-section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: rgba(255,255,255,0.07);
}
.sa-routine-step {
    display: flex;
    gap: 1rem;
    padding: 0.9rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.sa-routine-step:last-child { border-bottom: none; }
.sa-step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #C9A96E;
    min-width: 24px;
    padding-top: 0.15rem;
}
.sa-step-content { flex: 1; }
.sa-step-title {
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    color: #E8E0D5;
    margin-bottom: 0.25rem;
}
.sa-step-desc {
    font-family: 'Outfit', sans-serif;
    font-size: 0.85rem;
    color: #8A8A9A;
    line-height: 1.6;
}
.sa-hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 300;
    line-height: 1.1;
    color: #E8E0D5;
    letter-spacing: -0.02em;
}
.sa-hero-sub {
    font-family: 'Outfit', sans-serif;
    font-weight: 300;
    font-size: 1rem;
    color: #6A6A7A;
    letter-spacing: 0.05em;
    margin-top: 0.3rem;
}
.sa-logo-mark {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-style: italic;
    color: #C9A96E;
}
.sa-status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 0.4rem;
}
.sa-status-online  { background: #5CDB95; box-shadow: 0 0 6px #5CDB9580; }
.sa-status-offline { background: #E87A7A; box-shadow: 0 0 6px #E87A7A80; }
.sa-status-unknown { background: #7A7A8A; }
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — API
# ─────────────────────────────────────────────────────────────────────────────

def _base(path: str) -> str:
    """Construct full API URL.

    API_URL already contains the versioned prefix (e.g. http://localhost:8000/api/v1),
    so callers just pass the bare path:
        _base("health")   → http://localhost:8000/api/v1/health
        _base("predict")  → http://localhost:8000/api/v1/predict
    """
    return f"{API_URL.rstrip('/')}/{path.lstrip('/')}"


def check_health() -> dict[str, Any]:
    try:
        r = requests.get(_base("health"), timeout=5)
        r.raise_for_status()
        return {"online": True, "data": r.json()}
    except requests.exceptions.ConnectionError:
        return {"online": False, "error": "Cannot reach API server."}
    except requests.exceptions.Timeout:
        return {"online": False, "error": "Health check timed out."}
    except Exception as exc:
        return {"online": False, "error": str(exc)}


def call_predict(image_bytes: bytes, filename: str) -> dict[str, Any]:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpeg"
    mime_type = "image/png" if ext == "png" else "image/jpeg"
    try:
        files = {"image": (filename, image_bytes, mime_type)}
        r = requests.post(_base("predict"), files=files, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        payload = r.json()

        if not isinstance(payload, dict):
            return {"success": False, "error": "Invalid API response format."}

        required_keys = {"prediction", "confidence", "recommendation"}
        if not required_keys.issubset(payload):
            return {"success": False, "error": "API response is missing required fields."}

        return {"success": True, "data": payload}
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot reach API server. Is the backend running?"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": f"Request timed out after {REQUEST_TIMEOUT}s."}
    except requests.exceptions.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return {"success": False, "error": f"API error {exc.response.status_code}: {detail}"}
    except ValueError:
        return {"success": False, "error": "API returned invalid JSON."}
    except Exception as exc:
        logger.exception("Unexpected error during predict call")
        return {"success": False, "error": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — DATA PARSING (defensive — handles varied API response shapes)
# ─────────────────────────────────────────────────────────────────────────────

def normalise_label(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").title()


def parse_skin_issues(data: dict) -> list[dict]:
    """Return list of {label, confidence} dicts."""
    issues = []
    raw = (
        data.get("skin_issues")
        or data.get("issues")
        or data.get("predictions")
        or []
    )
    for item in raw:
        if isinstance(item, dict):
            issues.append({
                "label": normalise_label(
                    item.get("label") or item.get("name") or item.get("issue") or "Unknown"
                ),
                "confidence": float(item.get("confidence") or item.get("score") or 0),
            })
        elif isinstance(item, str):
            issues.append({"label": normalise_label(item), "confidence": 0.0})
    return issues


def parse_skin_tone(data: dict) -> dict:
    """Return {tone, undertone, hex, confidence}."""
    raw = data.get("skin_tone") or data.get("tone") or {}
    if isinstance(raw, str):
        raw = {"tone": raw}
    return {
        "tone":       raw.get("tone") or raw.get("label") or "N/A",
        "undertone":  raw.get("undertone") or "N/A",
        "hex":        raw.get("hex") or raw.get("color") or "#C8A882",
        "confidence": float(raw.get("confidence") or raw.get("score") or 0),
    }


def parse_routine(data: dict) -> str:
    """Return raw routine string."""
    return (
        data.get("routine")
        or data.get("skincare_routine")
        or data.get("recommendations")
        or data.get("llm_output")
        or ""
    )


def parse_products(data: dict) -> list[dict]:
    """Return list of {name, category, reason}."""
    raw = data.get("products") or data.get("recommended_products") or []
    products = []
    for p in raw:
        if isinstance(p, dict):
            products.append({
                "name":     p.get("name") or p.get("product") or "Unknown",
                "category": p.get("category") or p.get("type") or "",
                "reason":   p.get("reason") or p.get("description") or "",
            })
        elif isinstance(p, str):
            products.append({"name": p, "category": "", "reason": ""})
    return products


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — REPORT GENERATION
# ─────────────────────────────────────────────────────────────────────────────

def build_text_report(filename: str, data: dict) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    issues   = parse_skin_issues(data)
    tone     = parse_skin_tone(data)
    routine  = parse_routine(data)
    products = parse_products(data)

    lines = [
        "╔══════════════════════════════════════════════════════════════╗",
        "║            SKINAURA — AI SKINCARE ANALYSIS REPORT           ║",
        "╚══════════════════════════════════════════════════════════════╝",
        f"  Generated  : {now}",
        f"  Image      : {filename}",
        "",
        "── SKIN ISSUES DETECTED ────────────────────────────────────────",
    ]
    if issues:
        for iss in issues:
            conf = f"{iss['confidence']*100:.1f}%" if iss['confidence'] else "N/A"
            lines.append(f"  • {iss['label']:<30} confidence: {conf}")
    else:
        lines.append("  No issues detected.")

    lines += [
        "",
        "── SKIN TONE & UNDERTONE ───────────────────────────────────────",
        f"  Tone       : {tone['tone']}",
        f"  Undertone  : {tone['undertone']}",
        f"  Hex Color  : {tone['hex']}",
        f"  Confidence : {tone['confidence']*100:.1f}%" if tone['confidence'] else "  Confidence : N/A",
        "",
        "── PERSONALISED SKINCARE ROUTINE ───────────────────────────────",
    ]
    if routine:
        for line in routine.splitlines():
            lines.append(f"  {line}")
    else:
        lines.append("  No routine generated.")

    if products:
        lines += ["", "── RECOMMENDED PRODUCTS ────────────────────────────────────────"]
        for prod in products:
            lines.append(f"  • {prod['name']}")
            if prod["category"]:
                lines.append(f"    Category : {prod['category']}")
            if prod["reason"]:
                lines.append(f"    Reason   : {prod['reason']}")

    lines += [
        "",
        "────────────────────────────────────────────────────────────────",
        "  Powered by SkinAura AI Engine",
        "  Disclaimer: For informational purposes only. Consult a",
        "  dermatologist for clinical advice.",
        "────────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


def build_json_report(filename: str, data: dict) -> str:
    return json.dumps(
        {
            "report_meta": {
                "generated_at": datetime.now().isoformat(),
                "source_image": filename,
                "engine": "SkinAura AI",
            },
            "skin_issues":  parse_skin_issues(data),
            "skin_tone":    parse_skin_tone(data),
            "routine":      parse_routine(data),
            "products":     parse_products(data),
            "raw_response": data,
        },
        indent=2,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI COMPONENTS
# ─────────────────────────────────────────────────────────────────────────────

def render_hero() -> None:
    st.markdown(
        """
        <div style="padding: 2.5rem 0 1.5rem; text-align: center;">
            <div class="sa-logo-mark">✦ SkinAura</div>
            <div class="sa-hero-title" style="margin-top:0.6rem;">
                Intelligent Skincare<br><em style="color:#C9A96E;">Analysis</em>
            </div>
            <div class="sa-hero-sub" style="margin-top:0.8rem; max-width:480px; margin-left:auto; margin-right:auto;">
                Upload a clear face photograph. Our AI identifies skin conditions,<br>
                classifies your tone, and generates a bespoke skincare routine.
            </div>
        </div>
        <hr>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar(health_result: dict) -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 1.2rem 0 0.5rem; text-align:center;">
                <div class="sa-logo-mark" style="font-size:1.2rem;">✦ SkinAura</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                            letter-spacing:0.18em;color:#4A4A5A;margin-top:0.25rem;">
                    AI SKINCARE ENGINE
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")

        # API status
        if health_result.get("online"):
            dot = '<span class="sa-status-dot sa-status-online"></span>'
            status_text = "Online"
            status_color = "#5CDB95"
            ver = (health_result.get("data") or {}).get("version", "")
            sub = f"v{ver}" if ver else "Connected"
        else:
            dot = '<span class="sa-status-dot sa-status-offline"></span>'
            status_text = "Offline"
            status_color = "#E87A7A"
            sub = health_result.get("error", "Unknown error")

        st.markdown(
            f"""
            <div class="sa-card">
                <div class="sa-card-title">API Status</div>
                <div style="display:flex;align-items:center;gap:0.5rem;">
                    {dot}
                    <span style="font-family:'Outfit',sans-serif;font-size:0.95rem;
                                 font-weight:600;color:{status_color};">{status_text}</span>
                </div>
                <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                            color:#4A4A5A;margin-top:0.3rem;">{sub}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        st.markdown(
            """
            <div class="sa-card-title" style="margin-bottom:0.6rem;">How to Use</div>
            """,
            unsafe_allow_html=True,
        )
        steps = [
            ("01", "Upload a clear, well-lit face photograph (JPG or PNG)."),
            ("02", "Ensure your face is centred with minimal shadows."),
            ("03", "Click <strong>Predict Skin Issue</strong> and wait for results."),
            ("04", "Review your report and download if needed."),
        ]
        for num, text in steps:
            st.markdown(
                f"""
                <div style="display:flex;gap:0.75rem;margin-bottom:0.6rem;">
                    <span style="font-family:'DM Mono',monospace;font-size:0.65rem;
                                 color:#C9A96E;min-width:20px;padding-top:0.1rem;">{num}</span>
                    <span style="font-family:'Outfit',sans-serif;font-size:0.82rem;
                                 color:#8A8A9A;line-height:1.5;">{text}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        st.markdown(
            """
            <div class="sa-card-title" style="margin-bottom:0.5rem;">About</div>
            <div style="font-family:'Outfit',sans-serif;font-size:0.8rem;
                        color:#5A5A6A;line-height:1.6;">
                SkinAura uses state-of-the-art computer vision and large language models
                to deliver evidence-based skincare recommendations. Results are
                informational only — consult a dermatologist for clinical care.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        if st.button("↻  Refresh Status", use_container_width=True):
            st.session_state["health"] = check_health()
            st.rerun()

        st.markdown(
            """
            <div style="font-family:'DM Mono',monospace;font-size:0.6rem;
                        color:#2A2A3A;text-align:center;margin-top:1.5rem;">
                SKINAURA © 2025
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_upload_section() -> tuple[bytes | None, str]:
    """Returns (image_bytes, filename) or (None, '')."""
    st.markdown(
        '<div class="sa-section-header">01 &nbsp; Upload Image</div>',
        unsafe_allow_html=True,
    )

    col_up, col_prev = st.columns([1, 1], gap="large")

    with col_up:
        uploaded = st.file_uploader(
            "Drop a face photograph here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        st.markdown(
            """
            <div style="font-family:'DM Mono',monospace;font-size:0.65rem;
                        color:#3A3A4A;margin-top:0.5rem;text-align:center;">
                JPG / PNG · max 10 MB · face clearly visible
            </div>
            """,
            unsafe_allow_html=True,
        )

    if uploaded is not None:
        img_bytes = uploaded.read()
        try:
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        except Exception:
            st.error("Could not open the image. Please upload a valid JPG or PNG file.")
            return None, ""

        with col_prev:
            st.image(img, caption=uploaded.name, use_container_width=True)
            w, h = img.size
            st.markdown(
                f"""
                <div style="display:flex;justify-content:space-between;
                            font-family:'DM Mono',monospace;font-size:0.65rem;color:#4A4A5A;
                            margin-top:0.25rem;">
                    <span>{uploaded.name}</span>
                    <span>{w}×{h} px · {len(img_bytes)//1024} KB</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        return img_bytes, uploaded.name

    return None, ""


def render_skin_issues(issues: list[dict]) -> None:
    st.markdown(
        '<div class="sa-section-header">Skin Issues Detected</div>',
        unsafe_allow_html=True,
    )
    if not issues:
        st.markdown(
            '<div class="sa-card"><div class="sa-card-content">✦ No skin issues detected. Your skin appears healthy.</div></div>',
            unsafe_allow_html=True,
        )
        return

    cols = st.columns(min(len(issues), 3))
    for i, issue in enumerate(issues):
        with cols[i % len(cols)]:
            conf = issue["confidence"]
            conf_pct = f"{conf*100:.1f}%" if conf else "N/A"
            conf_width = f"{conf*100:.0f}%" if conf else "0%"
            st.markdown(
                f"""
                <div class="sa-card">
                    <div class="sa-card-title">Issue {i+1:02d}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:1.25rem;
                                color:#E8E0D5;margin-bottom:0.4rem;">{issue['label']}</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                                color:#C9A96E;margin-bottom:0.25rem;">
                        Confidence: {conf_pct}
                    </div>
                    <div class="sa-confidence-bar">
                        <div class="sa-confidence-fill" style="width:{conf_width};"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_skin_tone(tone: dict) -> None:
    st.markdown(
        '<div class="sa-section-header">Skin Tone Classification</div>',
        unsafe_allow_html=True,
    )
    conf = tone["confidence"]
    conf_pct = f"{conf*100:.1f}%" if conf else "N/A"
    conf_width = f"{conf*100:.0f}%" if conf else "0%"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Skin Tone", tone["tone"])
    with col2:
        st.metric("Undertone", tone["undertone"])
    with col3:
        st.metric("Confidence", conf_pct)

    hex_color = tone["hex"] if tone["hex"].startswith("#") else "#C8A882"
    st.markdown(
        f"""
        <div class="sa-card" style="margin-top:0.75rem;display:flex;align-items:center;gap:1rem;">
            <div class="sa-tone-swatch" style="background:{hex_color};"></div>
            <div>
                <div class="sa-card-title">Detected Colour</div>
                <div style="font-family:'DM Mono',monospace;font-size:0.9rem;
                            color:#E8E0D5;">{hex_color.upper()}</div>
                <div class="sa-confidence-bar" style="width:160px;margin-top:0.5rem;">
                    <div class="sa-confidence-fill" style="width:{conf_width};"></div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_routine(routine_text: str) -> None:
    st.markdown(
        '<div class="sa-section-header">Personalised Skincare Routine</div>',
        unsafe_allow_html=True,
    )
    if not routine_text.strip():
        st.info("No routine was generated for this analysis.")
        return

    with st.expander("View Full Routine", expanded=True):
        lines = [l for l in routine_text.splitlines() if l.strip()]
        step_num = 0
        html_parts = ['<div style="padding:0.5rem 0;">']
        for line in lines:
            stripped = line.strip()
            is_step = (
                (stripped[:2].rstrip(". )").isdigit())
                or stripped.lower().startswith("step ")
            )
            is_heading = stripped.startswith("#") or stripped.isupper()

            if is_step:
                step_num += 1
                text = stripped.lstrip("0123456789.)- ").lstrip()
                text = text.lstrip("Ssteep").strip() if text.lower().startswith("step") else text
                parts = text.split(":", 1)
                title = parts[0].strip()
                desc  = parts[1].strip() if len(parts) > 1 else ""
                html_parts.append(
                    f"""<div class="sa-routine-step">
                        <div class="sa-step-num">{step_num:02d}</div>
                        <div class="sa-step-content">
                            <div class="sa-step-title">{title}</div>
                            {"<div class='sa-step-desc'>" + desc + "</div>" if desc else ""}
                        </div>
                    </div>"""
                )
            elif is_heading:
                label = stripped.lstrip("#").strip()
                html_parts.append(
                    f'<div style="font-family:\'Cormorant Garamond\',serif;font-size:1.1rem;'
                    f'color:#C9A96E;margin:1rem 0 0.4rem;font-weight:400;">{label}</div>'
                )
            else:
                html_parts.append(
                    f'<div style="font-family:\'Outfit\',sans-serif;font-size:0.88rem;'
                    f'color:#8A8A9A;line-height:1.7;padding:0.2rem 0;">{stripped}</div>'
                )

        html_parts.append("</div>")
        st.markdown("\n".join(html_parts), unsafe_allow_html=True)


def render_products(products: list[dict]) -> None:
    if not products:
        return
    st.markdown(
        '<div class="sa-section-header">Recommended Products</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(min(len(products), 3))
    for i, prod in enumerate(products):
        with cols[i % 3]:
            category_badge = (
                f'<span class="sa-badge">{prod["category"]}</span>'
                if prod["category"] else ""
            )
            reason_html = (
                f'<div style="font-family:\'Outfit\',sans-serif;font-size:0.8rem;'
                f'color:#5A5A6A;margin-top:0.5rem;line-height:1.5;">{prod["reason"]}</div>'
                if prod["reason"] else ""
            )
            st.markdown(
                f"""
                <div class="sa-card">
                    <div class="sa-card-title">Product {i+1:02d}</div>
                    <div style="font-family:'Cormorant Garamond',serif;font-size:1.15rem;
                                color:#E8E0D5;margin-bottom:0.4rem;">{prod['name']}</div>
                    {category_badge}
                    {reason_html}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_download_buttons(filename: str, data: dict) -> None:
    st.markdown(
        '<div class="sa-section-header">Export Report</div>',
        unsafe_allow_html=True,
    )
    col1, col2, _ = st.columns([1, 1, 2])

    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")

    with col1:
        txt_report = build_text_report(filename, data)
        st.download_button(
            label="↓  Download .txt",
            data=txt_report.encode(),
            file_name=f"skinaura_{stem}_{ts}.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        json_report = build_json_report(filename, data)
        st.download_button(
            label="↓  Download .json",
            data=json_report.encode(),
            file_name=f"skinaura_{stem}_{ts}.json",
            mime="application/json",
            use_container_width=True,
        )


def render_results(filename: str, data: dict) -> None:
    prediction = normalise_label(data.get("prediction", "Unknown"))
    confidence = float(data.get("confidence", 0.0))
    recommendation = data.get("recommendation", "No recommendation available.")
    confidence_pct = f"{confidence * 100:.1f}%"
    confidence_width = f"{confidence * 100:.0f}%"

    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.75rem;
                    background:rgba(92,219,149,0.07);border:1px solid rgba(92,219,149,0.2);
                    border-radius:4px;padding:0.9rem 1.2rem;margin:1.5rem 0 0.5rem;">
            <span style="font-size:1.2rem;">✦</span>
            <span style="font-family:'Outfit',sans-serif;font-size:0.95rem;
                         color:#5CDB95;font-weight:500;">
                Prediction complete — result generated successfully.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sa-section-header">Prediction Result</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", prediction)
    with col2:
        st.metric("Confidence", confidence_pct)

    st.markdown(
        f"""
        <div class="sa-card">
            <div class="sa-card-title">Model Confidence</div>
            <div style="font-family:'Cormorant Garamond',serif;font-size:1.25rem;
                        color:#E8E0D5;margin-bottom:0.4rem;">{prediction}</div>
            <div style="font-family:'DM Mono',monospace;font-size:0.7rem;
                        color:#C9A96E;margin-bottom:0.25rem;">
                Confidence: {confidence_pct}
            </div>
            <div class="sa-confidence-bar">
                <div class="sa-confidence-fill" style="width:{confidence_width};"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sa-section-header">Recommendation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sa-card">
            <div class="sa-card-title">Suggested Next Step</div>
            <div class="sa-card-content">{recommendation}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render extended analysis sections if the API returns enriched data
    issues = parse_skin_issues(data)
    if issues:
        render_skin_issues(issues)

    tone = parse_skin_tone(data)
    if tone["tone"] != "N/A":
        render_skin_tone(tone)

    routine = parse_routine(data)
    if routine:
        render_routine(routine)

    products = parse_products(data)
    if products:
        render_products(products)

    render_download_buttons(filename, data)


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────

if "health" not in st.session_state:
    st.session_state["health"] = check_health()

if "results" not in st.session_state:
    st.session_state["results"] = None

if "result_filename" not in st.session_state:
    st.session_state["result_filename"] = ""


# ─────────────────────────────────────────────────────────────────────────────
# RENDER SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

render_sidebar(st.session_state["health"])


# ─────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────────────────────────

render_hero()

# ── Upload section ──────────────────────────────────────────────────────────
img_bytes, img_filename = render_upload_section()

# ── Analyse button ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="sa-section-header">02 &nbsp; Run Analysis</div>',
    unsafe_allow_html=True,
)

col_btn, col_hint = st.columns([1, 3])
with col_btn:
    analyse_clicked = st.button(
        "✦  Predict Skin Issue",
        disabled=(img_bytes is None),
        use_container_width=True,
    )

if img_bytes is None:
    with col_hint:
        st.markdown(
            """
            <div style="font-family:'Outfit',sans-serif;font-size:0.85rem;
                        color:#3A3A4A;padding-top:0.7rem;">
                Upload an image above to enable prediction.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Trigger analysis ────────────────────────────────────────────────────────
if analyse_clicked and img_bytes is not None:
    if not st.session_state["health"].get("online"):
        st.error(
            f"⚠ The SkinAura API is offline. "
            f"Please start the backend server and ensure it is reachable at **{API_URL}**.",
            icon=None,
        )
        logger.warning("Prediction attempted but API is offline.")
    else:
        st.session_state["results"] = None

        progress_bar = st.progress(0, text="Preparing image...")
        status_holder = st.empty()

        try:
            time.sleep(0.2)
            progress_bar.progress(15, text="Uploading image to prediction API...")
            time.sleep(0.3)
            progress_bar.progress(45, text="Running model inference...")

            logger.info("Calling /predict for file: %s", img_filename)
            result = call_predict(img_bytes, img_filename)

            progress_bar.progress(75, text="Validating prediction response...")
            time.sleep(0.2)
            progress_bar.progress(90, text="Preparing recommendation...")
            time.sleep(0.2)
            progress_bar.progress(100, text="Complete.")
            time.sleep(0.4)

        finally:
            progress_bar.empty()
            status_holder.empty()

        if result["success"]:
            st.session_state["results"] = result["data"]
            st.session_state["result_filename"] = img_filename
            logger.info("Prediction succeeded for %s.", img_filename)
        else:
            st.markdown(
                f"""
                <div style="background:rgba(220,80,80,0.07);border:1px solid rgba(220,80,80,0.25);
                            border-radius:4px;padding:0.9rem 1.2rem;margin:0.75rem 0;">
                    <span style="font-family:'Outfit',sans-serif;font-size:0.9rem;color:#E87A7A;">
                        ⚠ Prediction failed — {result['error']}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            logger.error("Prediction failed: %s", result["error"])

if st.session_state["results"] is not None:
    st.markdown(
        '<div class="sa-section-header">03 &nbsp; Results</div>',
        unsafe_allow_html=True,
    )
    render_results(
        st.session_state["result_filename"],
        st.session_state["results"],
    )

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin-top:3rem;">
    <div style="text-align:center;padding:1rem 0 2rem;">
        <div class="sa-logo-mark" style="font-size:1rem;margin-bottom:0.3rem;">✦ SkinAura</div>
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#2A2A3A;
                    letter-spacing:0.12em;">
            AI-POWERED SKINCARE ANALYSIS · FOR INFORMATIONAL PURPOSES ONLY
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)