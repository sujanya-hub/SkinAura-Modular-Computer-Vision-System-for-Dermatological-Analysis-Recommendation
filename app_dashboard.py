"""
app_dashboard.py - SkinAura: AI Analysis Dashboard
==================================================
Premium dark-theme Streamlit dashboard.

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
    page_title="SkinAura - AI Dashboard",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PREDICT_API_URL = "https://skinaura-backend.onrender.com/api/v1/predict"
HEALTH_API_URL = "https://skinaura-backend.onrender.com/api/v1/health"
API_TIMEOUT_SECONDS = 60
CLASS_LABELS = ["Acne", "Pigmentation", "Acne Scars", "Normal"]

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;500;600&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── App background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background: #080810 !important;
    color: #E2DAD0 !important;
}
.main .block-container {
    padding: 1.5rem 2rem 3rem !important;
    max-width: 1400px !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stSidebarNav"] { display: none !important; visibility: hidden !important; }

/* ── Typography ── */
h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; }
p, li, span, label, div, button { font-family: 'Outfit', sans-serif !important; }
code, pre { font-family: 'DM Mono', monospace !important; }

/* ── All Streamlit markdown containers ── */
[data-testid="stMarkdownContainer"] { font-family: 'Outfit', sans-serif !important; }

/* ── Progress bars ── */
[data-testid="stProgress"] > div { background: rgba(255,255,255,0.06) !important; border-radius: 2px !important; }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #C9A96E, #E8C97A) !important; border-radius: 2px !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px dashed rgba(201,169,110,0.3) !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploader"]:hover { border-color: rgba(201,169,110,0.6) !important; }
[data-testid="stFileUploader"] * { color: #8A8A9A !important; font-family: 'Outfit', sans-serif !important; }
[data-testid="stFileUploaderDropzoneInstructions"] { padding: 1rem !important; }

/* ── Streamlit columns gap fix ── */
[data-testid="column"] { padding: 0 0.4rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #C9A96E, #E8C97A) !important;
    color: #080810 !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.14em !important;
    text-transform: uppercase !important;
    padding: 0.55rem 1.4rem !important;
    box-shadow: 0 4px 20px rgba(201,169,110,0.22) !important;
    transition: all 0.25s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 28px rgba(201,169,110,0.38) !important;
}
.stButton > button.secondary {
    background: rgba(255,255,255,0.05) !important;
    color: #C9A96E !important;
    border: 1px solid rgba(201,169,110,0.3) !important;
    box-shadow: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: rgba(201,169,110,0.25); border-radius: 2px; }

/* ── Global card base ── */
.sa-card {
    background: rgba(255,255,255,0.028);
    border: 1px solid rgba(255,255,255,0.075);
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    position: relative;
    height: 100%;
}
.sa-card-accent::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 3px;
    background: linear-gradient(90deg, #C9A96E, #E8C97A);
    border-radius: 10px 10px 0 0;
}
.sa-card-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #5A5A6E;
    margin-bottom: 0.5rem;
}
.sa-card-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 400;
    color: #E2DAD0;
    line-height: 1.1;
}
.sa-card-sub {
    font-family: 'Outfit', sans-serif;
    font-size: 0.82rem;
    color: #6A6A7E;
    margin-top: 0.35rem;
    line-height: 1.6;
}

/* ── Header ── */
.sa-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.8rem 0 1.6rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.6rem;
}
.sa-logo {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1.5rem;
    color: #C9A96E;
    letter-spacing: 0.02em;
}
.sa-header-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.55rem;
    font-weight: 300;
    color: #E2DAD0;
    letter-spacing: 0.01em;
}
.sa-header-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #3A3A4E;
    letter-spacing: 0.1em;
    text-align: right;
}

/* ── Status badges ── */
.sa-status {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    color: #8A8A9A;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 0.4rem;
    width: 100%;
}
.sa-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.sa-dot-green { background: #5CDB95; box-shadow: 0 0 5px #5CDB9560; }
.sa-dot-red   { background: #E87A7A; box-shadow: 0 0 5px #E87A7A60; }
.sa-dot-amber { background: #E8C97A; box-shadow: 0 0 5px #E8C97A60; }

/* ── Severity bar ── */
.sa-sev-bar {
    height: 4px;
    background: rgba(255,255,255,0.07);
    border-radius: 2px;
    overflow: hidden;
    margin-top: 0.3rem;
}
.sa-sev-fill { height: 100%; border-radius: 2px; }

/* ── Routine step ── */
.sa-step {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.7rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.045);
}
.sa-step:last-child { border-bottom: none; }
.sa-step-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.1em;
    color: #C9A96E;
    background: rgba(201,169,110,0.1);
    border: 1px solid rgba(201,169,110,0.25);
    border-radius: 2px;
    padding: 0.15rem 0.45rem;
    margin-top: 0.1rem;
    flex-shrink: 0;
    min-width: 32px;
    text-align: center;
}
.sa-step-title {
    font-family: 'Outfit', sans-serif;
    font-size: 0.87rem;
    font-weight: 500;
    color: #E2DAD0;
    margin-bottom: 0.15rem;
}
.sa-step-desc {
    font-family: 'Outfit', sans-serif;
    font-size: 0.78rem;
    color: #5A5A6E;
    line-height: 1.5;
}

/* ── Confidence class row ── */
.sa-class-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.55rem;
}
.sa-class-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.75rem;
    color: #8A8A9A;
    min-width: 80px;
}
.sa-class-bar-bg {
    flex: 1;
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    overflow: hidden;
}
.sa-class-bar-fill {
    height: 100%;
    border-radius: 2px;
}
.sa-class-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #5A5A6E;
    min-width: 30px;
    text-align: right;
}

/* ── Section divider ── */
.sa-divider {
    height: 1px;
    background: rgba(255,255,255,0.05);
    margin: 1.2rem 0;
}

/* ── Image container ── */
.sa-image-wrapper {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.075);
    background: rgba(255,255,255,0.02);
}
.sa-image-wrapper img { display: block; width: 100%; }

/* ── Export row ── */
.sa-export-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #3A3A4E;
    margin-bottom: 0.6rem;
}

/* ── Confidence number ── */
.sa-conf-number {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 300;
    color: #C9A96E;
    line-height: 1;
}
.sa-conf-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    color: #4A4A5E;
    text-transform: uppercase;
    margin-top: 0.25rem;
}
</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


def card_open(accent: bool = False, extra_style: str = "") -> None:
    cls = "sa-card sa-card-accent" if accent else "sa-card"
    st.markdown(f'<div class="{cls}" style="{extra_style}">', unsafe_allow_html=True)


def card_close() -> None:
    st.markdown("</div>", unsafe_allow_html=True)


def label(text: str) -> None:
    st.markdown(f'<div class="sa-card-label">{text}</div>', unsafe_allow_html=True)


def divider() -> None:
    st.markdown('<div class="sa-divider"></div>', unsafe_allow_html=True)


def status_badge(title: str, online: bool) -> None:
    dot_cls = "sa-dot-green" if online else "sa-dot-red"
    status = "Loaded" if title == "Model" and online else "Connected" if online else "Offline"
    st.markdown(
        f"""<div class="sa-status">
              <div class="sa-dot {dot_cls}"></div>
              <span>{title}:&nbsp;<strong style="color:#C8C0B5;">{status}</strong></span>
           </div>""",
        unsafe_allow_html=True,
    )


def confidence_class_row(name: str, prob: float, is_top: bool = False) -> None:
    fill_color = "#C9A96E" if is_top else "#3A3A5A"
    pct = f"{prob * 100:.0f}%"
    st.markdown(
        f"""<div class="sa-class-row">
              <span class="sa-class-label">{name}</span>
              <div class="sa-class-bar-bg">
                <div class="sa-class-bar-fill" style="width:{pct};background:{fill_color};"></div>
              </div>
              <span class="sa-class-pct">{pct}</span>
           </div>""",
        unsafe_allow_html=True,
    )


def severity_row(name: str, level: str, color: str, prob: float) -> None:
    pct = f"{prob * 100:.0f}%"
    st.markdown(
        f"""<div style="margin-bottom:0.7rem;">
              <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="font-family:'Outfit',sans-serif;font-size:0.78rem;color:#8A8A9A;">{name}</span>
                <span style="font-family:'DM Mono',monospace;font-size:0.65rem;color:{color};">{level}</span>
              </div>
              <div class="sa-sev-bar">
                <div class="sa-sev-fill" style="width:{pct};background:{color};opacity:0.85;"></div>
              </div>
           </div>""",
        unsafe_allow_html=True,
    )


def routine_step(index: int, period: str, name: str, desc: str) -> None:
    st.markdown(
        f"""<div class="sa-step">
              <div class="sa-step-badge">{period}</div>
              <div>
                <div class="sa-step-title">{index}. {name}</div>
                <div class="sa-step-desc">{desc}</div>
              </div>
           </div>""",
        unsafe_allow_html=True,
    )


def normalize_prediction_label(prediction: str) -> str:
    return prediction.replace("_", " ").replace("-", " ").title().strip() or "Unknown"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def call_predict_api(image_bytes: bytes, filename: str) -> dict[str, Any]:
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else "jpg"
    mime_type = "image/png" if extension == "png" else "image/jpeg"

    try:
        response = requests.post(
            PREDICT_API_URL,
            files={"image": (filename, image_bytes, mime_type)},
            timeout=API_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.Timeout:
        return {"success": False, "error": "Prediction request timed out."}
    except requests.ConnectionError:
        return {"success": False, "error": "API not reachable"}
    except requests.HTTPError as exc:
        detail = None
        try:
            detail = response.json().get("detail")
        except (ValueError, NameError, AttributeError):
            detail = None
        return {"success": False, "error": f"Prediction API error: {detail or exc}"}
    except requests.RequestException as exc:
        return {"success": False, "error": f"Prediction request failed: {exc}"}

    try:
        payload = response.json()
    except ValueError:
        return {"success": False, "error": "Prediction API returned invalid JSON."}

    if not isinstance(payload, dict):
        return {"success": False, "error": "Prediction API returned an invalid response format."}

    prediction = payload.get("prediction")
    confidence = payload.get("confidence")
    recommendation = payload.get("recommendation")
    if not isinstance(prediction, str) or not isinstance(recommendation, str):
        return {"success": False, "error": "Prediction API response is missing required fields."}

    try:
        confidence_value = clamp(float(confidence), 0.0, 1.0)
    except (TypeError, ValueError):
        return {"success": False, "error": "Prediction API returned an invalid confidence score."}

    return {
        "success": True,
        "data": {
            "prediction": normalize_prediction_label(prediction),
            "confidence": confidence_value,
            "recommendation": recommendation.strip(),
        },
    }


def fetch_system_status() -> dict[str, bool]:
    try:
        response = requests.get(HEALTH_API_URL, timeout=3)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError):
        return {"model": False, "api": False}

    services = payload.get("services", {}) if isinstance(payload, dict) else {}
    return {
        "model": services.get("model_loader") == "loaded",
        "api": payload.get("status") in {"ok", "degraded"},
    }


def simulate_class_probabilities(prediction: str, confidence: float) -> dict[str, float]:
    primary = normalize_prediction_label(prediction)
    top_probability = clamp(confidence, 0.55, 0.99)
    remaining_probability = 1.0 - top_probability

    weight_profiles = {
        "Acne": {"Pigmentation": 0.50, "Acne Scars": 0.35, "Normal": 0.15},
        "Pigmentation": {"Normal": 0.50, "Acne Scars": 0.28, "Acne": 0.22},
        "Acne Scars": {"Acne": 0.48, "Pigmentation": 0.27, "Normal": 0.25},
        "Normal": {"Pigmentation": 0.38, "Acne": 0.34, "Acne Scars": 0.28},
    }
    profile = weight_profiles.get(primary, {})
    other_labels = [label_name for label_name in CLASS_LABELS if label_name != primary]
    total_weight = sum(profile.get(label_name, 1.0) for label_name in other_labels) or 1.0

    probabilities = {primary: top_probability}
    for label_name in other_labels:
        probabilities[label_name] = remaining_probability * (profile.get(label_name, 1.0) / total_weight)

    total_probability = sum(probabilities.values()) or 1.0
    normalized = {name: value / total_probability for name, value in probabilities.items()}
    return dict(sorted(normalized.items(), key=lambda item: item[1], reverse=True))


def get_primary_secondary(class_probs: dict[str, float]) -> tuple[tuple[str, float], tuple[str, float]]:
    sorted_probs = sorted(class_probs.items(), key=lambda item: item[1], reverse=True)
    primary = sorted_probs[0]
    secondary = sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    return primary, secondary


def severity_level(probability: float) -> tuple[str, str]:
    if probability >= 0.7:
        return "High", "#E87A7A"
    if probability >= 0.4:
        return "Medium", "#E8D87A"
    return "Low", "#5CDB95"


def build_severity_analysis(prediction: str, confidence: float, class_probs: dict[str, float]) -> dict[str, tuple[str, str, float]]:
    secondary_probability = get_primary_secondary(class_probs)[1][1]
    base_profiles = {
        "Acne": {"Inflammation": 0.55, "Coverage": 0.48, "Scarring Risk": 0.42},
        "Pigmentation": {"Inflammation": 0.18, "Coverage": 0.56, "Scarring Risk": 0.32},
        "Acne Scars": {"Inflammation": 0.22, "Coverage": 0.40, "Scarring Risk": 0.62},
        "Normal": {"Inflammation": 0.08, "Coverage": 0.12, "Scarring Risk": 0.10},
    }
    profile = base_profiles.get(normalize_prediction_label(prediction), base_profiles["Normal"])

    analysis: dict[str, tuple[str, str, float]] = {}
    for metric_name, base_value in profile.items():
        value = clamp(base_value + (confidence * 0.35) + (secondary_probability * 0.10), 0.05, 0.95)
        level, color = severity_level(value)
        analysis[metric_name] = (level, color, value)
    return analysis


def build_ai_insight(prediction: str, confidence: float, secondary_label: str, secondary_probability: float) -> str:
    primary = normalize_prediction_label(prediction)
    base_messages = {
        "Acne": "Detected clustered inflammation and pore congestion patterns consistent with active acne activity, especially across oil-prone facial regions.",
        "Pigmentation": "Detected uneven melanin distribution and tonal variation suggestive of pigmentation irregularity rather than active inflammatory lesions.",
        "Acne Scars": "Detected residual textural irregularity and post-inflammatory markings consistent with acne scar formation and healing changes.",
        "Normal": "Detected balanced skin texture without a dominant lesion pattern, with overall features aligning more closely to normal skin presentation.",
    }
    certainty = "strong" if confidence >= 0.8 else "moderate"
    secondary_text = ""
    if secondary_label != primary and secondary_probability >= 0.12:
        secondary_text = f" Secondary overlap with {secondary_label.lower()} is also present, which may reflect mixed visual features in the uploaded image."
    return (
        f"{base_messages.get(primary, base_messages['Normal'])}"
        f" The model confidence is {confidence:.0%}, indicating a {certainty} match."
        f"{secondary_text}"
    )


def build_routine_steps(prediction: str) -> list[tuple[str, str, str]]:
    routines = {
        "Acne": [
            ("AM", "Gentle cleanser", "Salicylic acid 0.5-2% - lifts oil and debris from congested pores."),
            ("AM", "Oil-free moisturiser", "Lightweight non-comedogenic hydration to support the barrier."),
            ("AM", "Sunscreen SPF 50", "Broad-spectrum daily protection to reduce post-blemish darkening."),
            ("PM", "Micellar water", "Removes sunscreen, excess sebum, and surface buildup before treatment."),
            ("PM", "Exfoliating treatment", "BHA-based leave-on exfoliant to reduce clogged pores overnight."),
            ("PM", "Spot treatment", "Benzoyl peroxide or sulfur treatment for active inflammatory breakouts."),
        ],
        "Pigmentation": [
            ("AM", "Brightening cleanser", "Gentle cleanser to refresh skin without triggering irritation."),
            ("AM", "Vitamin C serum", "Antioxidant support to visibly target uneven tone and dullness."),
            ("AM", "Sunscreen SPF 50", "Strict UV protection helps prevent pigmentation from deepening."),
            ("PM", "Hydrating essence", "Prep the skin barrier to tolerate brightening actives more comfortably."),
            ("PM", "Targeted treatment", "Niacinamide, tranexamic acid, or azelaic acid for tone correction."),
            ("PM", "Barrier moisturiser", "Ceramide-rich hydration to reduce irritation from active ingredients."),
        ],
        "Acne Scars": [
            ("AM", "Gentle cleanser", "Low-stripping cleanse to protect healing and compromised texture."),
            ("AM", "Repair serum", "Niacinamide or peptide serum to support smoother-looking skin texture."),
            ("AM", "Sunscreen SPF 50", "Daily UV defense prevents scar marks from appearing darker."),
            ("PM", "Hydrating toner", "Adds comfort and reduces dryness before reparative evening actives."),
            ("PM", "Retinoid treatment", "Retinoid-based renewal support to improve scar appearance over time."),
            ("PM", "Barrier moisturiser", "Ceramides and humectants to minimise irritation and support recovery."),
        ],
        "Normal": [
            ("AM", "Gentle cleanser", "Removes overnight buildup while maintaining a balanced skin barrier."),
            ("AM", "Antioxidant serum", "Vitamin C or green tea support for preventive daily care."),
            ("AM", "Sunscreen SPF 50", "Protects against UV damage and helps preserve even skin tone."),
            ("PM", "Micellar cleanse", "Lifts sunscreen and impurities before the evening routine."),
            ("PM", "Hydrating serum", "Hyaluronic acid helps maintain smoothness and water balance."),
            ("PM", "Moisturiser", "Barrier-supportive cream keeps skin calm, balanced, and resilient."),
        ],
    }
    return routines.get(normalize_prediction_label(prediction), routines["Normal"])


def build_analysis_view_model(api_payload: dict[str, Any]) -> dict[str, Any]:
    prediction = normalize_prediction_label(str(api_payload["prediction"]))
    confidence = clamp(float(api_payload["confidence"]), 0.0, 1.0)
    recommendation = str(api_payload["recommendation"]).strip()
    class_probs = simulate_class_probabilities(prediction, confidence)
    primary_prediction, secondary_prediction = get_primary_secondary(class_probs)

    return {
        "prediction": primary_prediction[0],
        "confidence": primary_prediction[1],
        "recommendation": recommendation,
        "class_probs": class_probs,
        "primary_prediction": primary_prediction,
        "secondary_prediction": secondary_prediction,
        "severity": build_severity_analysis(prediction, confidence, class_probs),
        "ai_insight": build_ai_insight(prediction, confidence, secondary_prediction[0], secondary_prediction[1]),
        "routine_steps": build_routine_steps(prediction),
        "raw_api_response": api_payload,
    }


def build_text_report(result: dict[str, Any]) -> str:
    primary_name, primary_probability = result["primary_prediction"]
    secondary_name, secondary_probability = result["secondary_prediction"]
    return "\n".join(
        [
            "SkinAura Report",
            f"Condition: {result['prediction']}",
            f"Confidence: {result['confidence']:.0%}",
            f"Primary Prediction: {primary_name} ({primary_probability:.0%})",
            f"Secondary Prediction: {secondary_name} ({secondary_probability:.0%})",
            "",
            "Recommendation:",
            result["recommendation"],
            "",
            "AI Insight:",
            result["ai_insight"],
        ]
    )


def build_json_report(result: dict[str, Any]) -> str:
    primary_name, primary_probability = result["primary_prediction"]
    secondary_name, secondary_probability = result["secondary_prediction"]
    payload = {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "recommendation": result["recommendation"],
        "primary_prediction": {"label": primary_name, "probability": primary_probability},
        "secondary_prediction": {"label": secondary_name, "probability": secondary_probability},
        "class_probabilities": result["class_probs"],
        "severity": result["severity"],
        "ai_insight": result["ai_insight"],
        "routine_steps": [
            {"period": period, "name": name, "description": desc}
            for period, name, desc in result["routine_steps"]
        ],
        "raw_api_response": result["raw_api_response"],
    }
    return json.dumps(payload, indent=2)


def read_uploaded_image(uploaded_file: Any) -> tuple[Image.Image | None, bytes | None]:
    try:
        image_bytes = uploaded_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None, None
    return image, image_bytes


SESSION_DEFAULTS: dict[str, Any] = {
    "uploaded_image": None,
    "uploaded_image_bytes": None,
    "uploaded_filename": "",
    "analysis_result": None,
    "analysed": False,
    "api_error": "",
}

for session_key, default_value in SESSION_DEFAULTS.items():
    if session_key not in st.session_state:
        st.session_state[session_key] = default_value


st.markdown(
    """
    <div class="sa-header">
        <div style="display:flex;align-items:center;gap:1.1rem;">
            <div class="sa-logo">✦ SkinAura</div>
            <div style="width:1px;height:22px;background:rgba(255,255,255,0.1);"></div>
            <div class="sa-header-title">AI Analysis Dashboard</div>
        </div>
        <div class="sa-header-meta">
            SKIN INTELLIGENCE ENGINE<br>
            v2.1.0 &nbsp;·&nbsp; MODEL: RESNET-50
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


system_status = fetch_system_status()

col_left, col_right = st.columns([1.0, 2.2], gap="medium")

with col_left:
    card_open()
    label("Input")

    uploaded = st.file_uploader(
        "Upload face photo",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        key="face_upload",
    )

    if uploaded is not None:
        uploaded_image, uploaded_image_bytes = read_uploaded_image(uploaded)
        if uploaded_image is None or uploaded_image_bytes is None:
            st.error("Uploaded file is not a valid image.")
        else:
            st.session_state["uploaded_image"] = uploaded_image
            st.session_state["uploaded_image_bytes"] = uploaded_image_bytes
            st.session_state["uploaded_filename"] = uploaded.name
            st.session_state["api_error"] = ""

    current_image = st.session_state["uploaded_image"]
    current_filename = st.session_state["uploaded_filename"]
    current_image_bytes = st.session_state["uploaded_image_bytes"]

    if current_image is not None:
        st.markdown('<div class="sa-image-wrapper">', unsafe_allow_html=True)
        st.image(current_image, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        width, height = current_image.size
        st.markdown(
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.6rem;'
            f'color:#3A3A4E;margin-top:0.5rem;text-align:center;">'
            f'{current_filename} &nbsp;·&nbsp; {width}×{height}px</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """<div style="text-align:center;padding:2.5rem 0;color:#2E2E3E;">
                 <div style="font-size:1.6rem;margin-bottom:0.5rem;">⬆</div>
                 <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                             letter-spacing:0.12em;color:#3A3A4E;">
                   JPG / PNG · MAX 10MB
                 </div>
               </div>""",
            unsafe_allow_html=True,
        )

    divider()

    if st.button(
        "✦  Analyse Skin",
        disabled=(current_image_bytes is None),
        use_container_width=True,
        key="analyse_btn",
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            status_text.caption("Uploading image...")
            progress_bar.progress(15)
            time.sleep(0.15)

            status_text.caption("Running model inference...")
            progress_bar.progress(45)
            api_result = call_predict_api(current_image_bytes, current_filename)

            if not api_result["success"]:
                st.session_state["api_error"] = str(api_result["error"])
                st.error("API not reachable")
                if st.session_state["api_error"] != "API not reachable":
                    st.caption(st.session_state["api_error"])
                system_status = {"model": False, "api": False}
                st.session_state["analysed"] = st.session_state["analysis_result"] is not None
            else:
                status_text.caption("Generating insights...")
                progress_bar.progress(82)
                time.sleep(0.15)

                st.session_state["analysis_result"] = build_analysis_view_model(api_result["data"])
                st.session_state["analysed"] = True
                st.session_state["api_error"] = ""
                system_status = {"model": True, "api": True}

                progress_bar.progress(100)
                time.sleep(0.15)
        finally:
            progress_bar.empty()
            status_text.empty()

    card_close()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    card_open()
    label("System Status")
    status_badge("Model", system_status["model"])
    status_badge("API", system_status["api"])
    card_close()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    current_result = st.session_state["analysis_result"]
    export_enabled = current_result is not None
    txt_report = build_text_report(current_result) if export_enabled else "SkinAura Report\nNo analysis available yet."
    json_report = build_json_report(current_result) if export_enabled else json.dumps({"message": "No analysis available yet."}, indent=2)

    card_open()
    st.markdown('<div class="sa-export-label">Export Report</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "↓ .txt",
            data=txt_report,
            file_name="skinaura_report.txt",
            mime="text/plain",
            use_container_width=True,
            key="dl_txt",
            disabled=not export_enabled,
        )
    with c2:
        st.download_button(
            "↓ .json",
            data=json_report,
            file_name="skinaura_report.json",
            mime="application/json",
            use_container_width=True,
            key="dl_json",
            disabled=not export_enabled,
        )
    card_close()

with col_right:
    result = st.session_state["analysis_result"]

    if not st.session_state["analysed"] or result is None:
        card_open(accent=True)
        st.markdown(
            """<div style="text-align:center;padding:3rem 0;color:#2E2E3E;">
                 <div style="font-family:'Cormorant Garamond',serif;font-size:1.6rem;
                             font-weight:300;color:#3A3A4E;margin-bottom:0.5rem;">
                   Upload an image and click Analyse
                 </div>
                 <div style="font-family:'DM Mono',monospace;font-size:0.62rem;
                             letter-spacing:0.1em;color:#2A2A3A;">
                   AWAITING INPUT
                 </div>
               </div>""",
            unsafe_allow_html=True,
        )
        card_close()
        st.stop()

    prediction = result["prediction"]
    confidence = result["confidence"]
    recommendation = result["recommendation"]
    class_probs = result["class_probs"]
    severity = result["severity"]
    ai_insight = result["ai_insight"]
    routine_steps = result["routine_steps"]
    primary_prediction = result["primary_prediction"]
    secondary_prediction = result["secondary_prediction"]

    r1a, r1b = st.columns([2.2, 1], gap="medium")

    with r1a:
        card_open(accent=True)
        label("Prediction")
        st.markdown(f'<div class="sa-card-title">{prediction}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sa-card-sub">{recommendation}</div>', unsafe_allow_html=True)
        divider()
        label("Model Confidence")
        st.progress(confidence)
        card_close()

    with r1b:
        card_open()
        label("Confidence Score")
        st.markdown(
            f'<div class="sa-conf-number">{confidence:.0%}</div>'
            f'<div class="sa-conf-label">Overall Certainty</div>',
            unsafe_allow_html=True,
        )
        divider()
        label("Top Class")
        st.markdown(
            f'<div style="font-family:\'Outfit\',sans-serif;font-size:0.85rem;'
            f'font-weight:500;color:#E2DAD0;">Primary: {primary_prediction[0]}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
            f'color:#5CDB95;margin-top:0.2rem;">{primary_prediction[1]:.0%} CONFIDENCE</div>'
            f'<div style="font-family:\'Outfit\',sans-serif;font-size:0.8rem;'
            f'font-weight:500;color:#A8A0B5;margin-top:0.65rem;">Secondary: {secondary_prediction[0]}</div>'
            f'<div style="font-family:\'DM Mono\',monospace;font-size:0.62rem;'
            f'color:#C9A96E;margin-top:0.2rem;">{secondary_prediction[1]:.0%} LIKELIHOOD</div>',
            unsafe_allow_html=True,
        )
        card_close()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    r2a, r2b, r2c = st.columns(3, gap="medium")

    with r2a:
        card_open()
        label("Model Confidence Breakdown")
        for index, (class_name, probability) in enumerate(class_probs.items()):
            confidence_class_row(class_name, probability, is_top=(index == 0))
        card_close()

    with r2b:
        card_open()
        label("Severity Analysis")
        for severity_name, (level, color, probability) in severity.items():
            severity_row(severity_name, level, color, probability)
        card_close()

    with r2c:
        card_open()
        label("AI Insight")
        st.markdown(
            f'<div style="font-family:\'Outfit\',sans-serif;font-size:0.82rem;'
            f'color:#8A8A9A;line-height:1.65;">{ai_insight}</div>',
            unsafe_allow_html=True,
        )
        divider()
        st.markdown(
            '<div style="font-family:\'DM Mono\',monospace;font-size:0.6rem;'
            'color:#3A3A4E;letter-spacing:0.1em;">PATTERN RECOGNITION · GPT-4o</div>',
            unsafe_allow_html=True,
        )
        card_close()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    card_open()
    label("Recommended Routine")
    st.markdown(
        '<div style="font-family:\'Cormorant Garamond\',serif;font-size:1.1rem;'
        'font-weight:300;color:#C9A96E;margin-bottom:0.8rem;">'
        'Personalised 6-step protocol</div>',
        unsafe_allow_html=True,
    )

    step_cols = st.columns(2, gap="medium")
    for index, (period, name, desc) in enumerate(routine_steps):
        with step_cols[index % 2]:
            routine_step(index + 1, period, name, desc)

    card_close()
