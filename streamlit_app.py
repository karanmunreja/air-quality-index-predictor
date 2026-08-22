"""
AQI Forecast — Lahore
A Streamlit dashboard for the daily-trained AQI forecasting model,
served via a FastAPI + Hopsworks Model Serving backend.

Run:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd
import os
import base64
from pathlib import Path
from datetime import datetime

# ==========================================================
# PAGE CONFIG
# ==========================================================
from PIL import Image
try:
    _page_icon = Image.open("assets/icon.png")
except Exception:
    _page_icon = "🌫️"

st.set_page_config(
    page_title="AQI Forecast — Lahore",
    page_icon=_page_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==========================================================
# DESIGN TOKENS
# ==========================================================
# "Living haze" — the palette moves from clear-sky navy at rest to the
# actual EPA severity colors, which are the only saturated accents in
# the UI. The worse the air, the more the interface visually thickens.

BG = "#090D14"
PANEL = "#131A26"
PANEL_RAISED = "#1B2433"
INK = "#EDF1F7"
INK_MUTED = "#8A99B3"
HAIRLINE = "#26314A"

# AQI severity scale (US EPA convention)
AQI_BANDS = [
    (0, 50, "Good", "#35D97A",
     "Air quality is satisfactory. Enjoy normal outdoor activities."),
    (51, 100, "Moderate", "#F4C430",
     "Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion."),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF9142",
     "Children, the elderly, and people with respiratory conditions should limit prolonged outdoor exertion."),
    (151, 200, "Unhealthy", "#FF4D5E",
     "Everyone may begin to experience health effects. Limit prolonged outdoor exertion."),
    (201, 300, "Very Unhealthy", "#A855F7",
     "Health alert. Avoid outdoor exertion — everyone may experience more serious effects."),
    (301, 500, "Hazardous", "#7A1030",
     "Health emergency. Avoid all outdoor physical activity."),
]
AQI_SCALE_MAX = 500


def aqi_band(value: float):
    """Returns (label, color, advice, severity_index 0-5)."""
    for idx, (lo, hi, label, color, advice) in enumerate(AQI_BANDS):
        if lo <= value <= hi:
            return label, color, advice, idx
    if value > AQI_SCALE_MAX:
        lo, hi, label, color, advice = AQI_BANDS[-1]
        return label, color, advice, len(AQI_BANDS) - 1
    lo, hi, label, color, advice = AQI_BANDS[0]
    return label, color, advice, 0


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def chart_theme(title: str | None = None) -> dict:
    """Shared Plotly styling so every legend, axis label, title, and
    tooltip stays clearly readable against the dark background — this
    matters for a live demo where judges are reading from a distance."""
    layout = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Mono", color=INK, size=13),
        legend=dict(
            font=dict(family="IBM Plex Mono", color=INK, size=12),
            bgcolor=hex_to_rgba(PANEL_RAISED, 0.92),
            bordercolor=HAIRLINE,
            borderwidth=1,
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
        ),
        hoverlabel=dict(
            bgcolor=PANEL_RAISED,
            bordercolor=HAIRLINE,
            font=dict(color=INK, family="IBM Plex Mono", size=12),
        ),
    )
    if title:
        layout["title"] = dict(text=title, font=dict(family="Space Grotesk", color=INK, size=17), x=0, xanchor="left")
    return layout


def axis_style(text: str) -> dict:
    """Bright, legible axis title + tick styling for dark backgrounds."""
    return dict(
        title=dict(text=text, font=dict(family="Inter", color=INK, size=13)),
        tickfont=dict(family="IBM Plex Mono", color=INK_MUTED, size=11),
        gridcolor=HAIRLINE,
        zerolinecolor=HAIRLINE,
    )


def delta_badge(current: float, previous: float) -> str:
    diff = current - previous
    if abs(diff) < 0.5:
        return f'<span style="color:{INK_MUTED};">steady</span>'
    arrow = "▲" if diff > 0 else "▼"
    color = "#FF4D5E" if diff > 0 else "#35D97A"
    return f'<span style="color:{color};">{arrow} {abs(diff):.0f} vs prior</span>'


def render_spectrum(current_value: float | None = None) -> str:
    """A persistent Good→Hazardous ruler; doubles as legend and page signature."""
    segs = ""
    for lo, hi, label, color, _ in AQI_BANDS:
        width_pct = (hi - lo + 1) / AQI_SCALE_MAX * 100
        segs += f'<div class="aqi-spectrum-seg" style="background:{color};flex-basis:{width_pct}%;" title="{label} ({lo}-{hi})"></div>'
    marker = ""
    if current_value is not None:
        pct = max(0, min(current_value, AQI_SCALE_MAX)) / AQI_SCALE_MAX * 100
        marker = f'<div class="aqi-spectrum-marker" style="left:{pct}%;" title="Current reading: {current_value:.0f}"></div>'
    labels = "".join(f'<span>{lo}</span>' for lo, *_ in AQI_BANDS[1:])
    return f'''<div class="aqi-spectrum-wrap"><div class="aqi-spectrum-row">{segs}{marker}</div>
    <div class="aqi-spectrum-labels"><span>0</span>{labels}<span>500</span></div></div>'''


def asset_data_uri(path: str) -> str:
    """Embed a small local illustration so Streamlit serves it reliably."""
    try:
        encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
        return f"data:image/png;base64,{encoded}"
    except OSError:
        return ""


STICKER_URI = asset_data_uri("assets/aqi-guardian-sticker.png")


# ==========================================================
# GLOBAL CSS
# ==========================================================

st.markdown(f"""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">

<style>
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
        font-size: 17px;
    }}

    .stApp {{
        background:
            radial-gradient(circle at 12% -8%, rgba(53,217,122,0.06), transparent 38%),
            radial-gradient(circle at 88% 2%, rgba(168,85,247,0.055), transparent 42%),
            {BG};
        color: {INK};
    }}

    .stApp p, .stApp label {{
        color: {INK} !important;
    }}

    .stCaption, [data-testid="stCaptionContainer"] {{
        color: #B9C6D9 !important;
        font-size: 0.95rem !important;
    }}

    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    [data-testid="stMetricLabel"] p, [data-testid="stMetricValue"] {{
        color: {INK} !important;
    }}

    [data-testid="stMetric"] {{
        background: {PANEL};
        border: 1px solid {HAIRLINE};
        border-radius: 12px;
        padding: 0.85rem 1rem;
    }}
    [data-testid="stMetricLabel"] p {{ font-size: 1rem !important; font-weight: 600; }}
    [data-testid="stMetricValue"] {{ font-size: 2.35rem !important; font-weight: 700 !important; }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.45rem;
        border-bottom: 1px solid {HAIRLINE};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: {INK_MUTED} !important;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        padding: 0.65rem 1rem;
    }}
    .stTabs [aria-selected="true"] {{
        color: {INK} !important;
        border-bottom-color: #35D97A !important;
    }}

    [data-testid="stAlert"] {{
        color: {INK} !important;
        background: {PANEL_RAISED};
        border-color: {HAIRLINE};
    }}
    [data-testid="stAlert"] * {{ color: {INK} !important; }}

    [data-baseweb="select"] > div {{
        background: {PANEL_RAISED} !important;
        border-color: {HAIRLINE} !important;
        color: {INK} !important;
        font-family: 'IBM Plex Mono', monospace;
    }}

    #MainMenu, footer, header {{ visibility: hidden; }}

    section[data-testid="stSidebar"] {{
        display: none;
    }}

    .aqi-display, .aqi-mono {{
        font-family: 'IBM Plex Mono', monospace;
    }}

    .aqi-heading {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        letter-spacing: -0.01em;
    }}

    /* ---- Spectrum ruler (signature element) ---- */
    .aqi-spectrum-wrap {{
        padding: 0.9rem 0 0.6rem 0;
    }}
    .aqi-spectrum-row {{
        position: relative;
        display: flex;
        gap: 3px;
        height: 7px;
    }}
    .aqi-spectrum-seg {{
        border-radius: 3px;
        opacity: 0.9;
    }}
    .aqi-spectrum-marker {{
        position: absolute;
        top: -7px;
        width: 0; height: 0;
        border-left: 6px solid transparent;
        border-right: 6px solid transparent;
        border-top: 9px solid #ffffff;
        transform: translateX(-50%);
        filter: drop-shadow(0 0 5px rgba(255,255,255,0.65));
    }}
    .aqi-spectrum-labels {{
        display: flex;
        justify-content: space-between;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.65rem;
        color: {INK_MUTED};
        margin-top: 0.4rem;
        opacity: 0.75;
    }}

    /* ---- Top status strip ---- */
    .aqi-topbar {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.4rem 0 1.2rem 0;
        border-bottom: 1px solid {HAIRLINE};
        margin-bottom: 1.6rem;
    }}
    .aqi-app-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        padding: 0.85rem 0 1.1rem;
        border-bottom: 1px solid {HAIRLINE};
        margin-bottom: 0.55rem;
    }}
    .aqi-brand {{
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        letter-spacing: -0.04em;
        color: {INK};
    }}
    .aqi-brand span {{ color: #35D97A; }}
    .aqi-brand .aqi-brand-sub {{
        display: block;
        margin-top: 0.08rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.63rem;
        color: {INK_MUTED};
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }}
    .aqi-location {{
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.46rem 0.7rem;
        border: 1px solid {HAIRLINE};
        border-radius: 999px;
        color: {INK};
        background: rgba(27,36,51,0.72);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
    }}
    .aqi-location-pin {{ color: #FF4D5E; font-size: 0.9rem; }}
    .aqi-location-pin {{
        display: inline-flex;
        width: 18px;
        height: 18px;
        color: #FF6473;
        filter: drop-shadow(0 0 5px rgba(255,77,94,0.35));
    }}
    .aqi-location-pin svg {{ width: 18px; height: 18px; fill: currentColor; }}
    .aqi-topbar-left {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        color: {INK_MUTED};
        letter-spacing: 0.06em;
        text-transform: uppercase;
    }}
    .aqi-live-dot {{
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        margin-right: 6px;
        background: #35D97A;
        box-shadow: 0 0 6px #35D97A;
        animation: aqi-pulse 2.4s ease-in-out infinite;
    }}
    @keyframes aqi-pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.4; }}
    }}

    /* ---- Hero / living-haze band ---- */
    .aqi-hero {{
        position: relative;
        border-radius: 18px;
        padding: 2.6rem 2.4rem;
        overflow: hidden;
        border: 1px solid {HAIRLINE};
        margin-bottom: 1.8rem;
    }}
    .aqi-hero::before {{
        content: "";
        position: absolute;
        inset: -25%;
        background-image:
            radial-gradient(circle, rgba(255,255,255,0.55) 1px, transparent 1.7px),
            radial-gradient(circle, rgba(255,255,255,0.32) 1px, transparent 1.7px);
        background-size: 54px 54px, 84px 84px;
        background-position: 0 0, 26px 38px;
        opacity: var(--haze-opacity, 0.08);
        animation: haze-drift var(--haze-speed, 60s) linear infinite;
        pointer-events: none;
    }}
    @keyframes haze-drift {{
        from {{ transform: translate(0, 0); }}
        to   {{ transform: translate(-64px, 46px); }}
    }}
    @media (prefers-reduced-motion: reduce) {{
        .aqi-hero::before, .aqi-live-dot {{ animation: none !important; }}
    }}
    .aqi-hero-city {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.75);
        margin-bottom: 0.6rem;
        position: relative;
        z-index: 1;
    }}
    .aqi-hero-label {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1.35rem;
        margin-top: 0.2rem;
        position: relative;
        z-index: 1;
    }}
    .aqi-hero-sub {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        color: rgba(255,255,255,0.8);
        margin-top: 0.9rem;
        position: relative;
        z-index: 1;
    }}
    .aqi-hero-advice {{
        font-size: 0.95rem;
        color: rgba(255,255,255,0.92);
        max-width: 640px;
        margin-top: 0.9rem;
        line-height: 1.5;
        position: relative;
        z-index: 1;
    }}

    /* ---- Ring gauge (Google-style) ---- */
    .aqi-hero-flex {{
        display: flex;
        align-items: center;
        gap: 2.4rem;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }}
    .aqi-hero-text {{
        flex: 1;
        min-width: 260px;
    }}
    .aqi-guardian {{
        width: min(172px, 24vw);
        height: auto;
        object-fit: contain;
        flex-shrink: 0;
        filter: drop-shadow(0 16px 22px rgba(0,0,0,0.28));
        animation: guardian-float 4.5s ease-in-out infinite;
    }}
    @keyframes guardian-float {{
        0%, 100% {{ transform: translateY(0); }}
        50% {{ transform: translateY(-7px); }}
    }}
    @media (max-width: 680px) {{
        .aqi-app-header {{ align-items: flex-start; flex-direction: column; }}
        .aqi-guardian {{ width: 122px; }}
    }}
    .aqi-ring {{
        position: relative;
        width: 196px;
        height: 196px;
        flex-shrink: 0;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .aqi-ring-track {{
        position: absolute;
        inset: 0;
        border-radius: 50%;
        background: conic-gradient(var(--ring-color) calc(var(--ring-pct) * 1%), rgba(255,255,255,0.10) 0);
        filter: drop-shadow(0 0 20px var(--ring-glow));
        transition: background 0.4s ease;
    }}
    .aqi-ring-hole {{
        position: absolute;
        inset: 16px;
        border-radius: 50%;
        background: {BG};
        box-shadow: inset 0 0 0 1px rgba(255,255,255,0.07);
    }}
    .aqi-ring-center {{
        position: relative;
        z-index: 2;
        text-align: center;
    }}
    .aqi-ring-number {{
        font-family: 'Fraunces', serif;
        font-optical-sizing: auto;
        font-weight: 600;
        font-size: 3.4rem;
        line-height: 1;
        color: #ffffff;
    }}
    .aqi-ring-scale {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        color: rgba(255,255,255,0.55);
        margin-top: 0.3rem;
    }}

    /* ---- Forecast cards ---- */
    .aqi-card {{
        background: {PANEL};
        border: 1px solid {HAIRLINE};
        border-top: 3px solid var(--band-color, {HAIRLINE});
        border-radius: 14px;
        padding: 1.2rem 1.3rem 0.6rem 1.3rem;
        height: 100%;
        box-shadow: 0 0 0 rgba(0,0,0,0);
        transition: box-shadow 0.25s ease;
    }}
    .aqi-card:hover {{
        box-shadow: 0 0 28px -8px var(--band-color, transparent);
    }}
    .aqi-card-eyebrow {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {INK_MUTED};
        margin-bottom: 0.2rem;
    }}
    .aqi-card-cat {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        text-align: center;
        margin-top: -0.6rem;
    }}
    .aqi-card-delta {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        text-align: center;
        margin-top: 0.35rem;
        padding-bottom: 0.9rem;
    }}

    /* ---- Section headings ---- */
    .aqi-section-label {{
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.92rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #D8E2F1;
        margin: 4rem 0 1.2rem 0;
        border-bottom: 1px solid {HAIRLINE};
        padding: 0 0 0.75rem 0;
    }}
    .aqi-section-spacer {{
        height: 3.5rem;
        border-bottom: 1px solid rgba(38,49,74,0.55);
        margin: 3.5rem 0 0;
    }}

    /* ---- Advisory panel ---- */
    .aqi-advisory {{
        background: {PANEL};
        border: 1px solid {HAIRLINE};
        border-left: 4px solid var(--advisory-color, {INK_MUTED});
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
    }}
    .aqi-advisory-title {{
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.35rem;
    }}
    .aqi-advisory-text {{
        color: #D4DEEB;
        font-size: 1.05rem;
        line-height: 1.55;
    }}

    .stButton > button {{
        background: #1B2E24;
        color: #FFFFFF;
        border: 1px solid #2F6B4A;
        border-radius: 8px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }}
    .stButton > button:hover {{
        background: #21402D;
        border-color: #35D97A;
        color: #ffffff;
    }}

    .stTextInput > div > div > input {{
        font-family: 'IBM Plex Mono', monospace;
        background: {PANEL_RAISED};
        color: {INK};
        border-color: {HAIRLINE};
    }}
</style>
""", unsafe_allow_html=True)


# DATA FETCH
api_base_url = os.getenv("AQI_API_BASE_URL", "http://localhost:8000")

def fetch_prediction(base_url: str):
    """Calls the FastAPI /predict endpoint and normalizes the response
    into a dict with keys: current, h24, h48, h72 (all AQI floats)."""
    resp = requests.post(f"{base_url.rstrip('/')}/predict", timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    # Be flexible about response shape — Hopsworks python predictors
    # commonly return {"predictions": [[v24, v48, v72]]}
    if isinstance(payload, dict) and "predictions" in payload:
        preds = payload["predictions"]
        row = preds[0] if isinstance(preds[0], list) else preds
    elif isinstance(payload, list):
        row = payload[0] if isinstance(payload[0], list) else payload
    else:
        raise ValueError(f"Unrecognized response shape: {payload}")

    if len(row) < 3:
        raise ValueError(f"Expected 3 forecast values, got: {row}")

    return {"h24": float(row[0]), "h48": float(row[1]), "h72": float(row[2])}


def load_forecast():
    try:
        st.session_state["prediction"] = fetch_prediction(api_base_url)
        st.session_state["prediction_time"] = datetime.now()
        st.session_state["prediction_error"] = None
    except Exception as exc:
        st.session_state["prediction_error"] = str(exc)


def load_analysis():
    """Runs after the forecast has rendered, so insight work never hides the AQI result."""
    try:
        response = requests.get(f"{api_base_url.rstrip('/')}/history", params={"days": 30}, timeout=30)
        response.raise_for_status()
        st.session_state["hist_df"] = pd.DataFrame(response.json())
        st.session_state.pop("history_error", None)
    except requests.RequestException as exc:
        st.session_state["history_error"] = str(exc)

    try:
        response = requests.post(f"{api_base_url.rstrip('/')}/explain", timeout=120)
        response.raise_for_status()
        st.session_state["explain"] = response.json()
        st.session_state.pop("explain_error", None)
    except requests.RequestException as exc:
        st.session_state["explain_error"] = str(exc)


def refresh_dashboard():
    load_forecast()
    load_analysis()


loading_screen = st.empty()
if "dashboard_loaded" not in st.session_state:
    # Small animated spinner buffer (rotating SVG) while initial data loads
    loading_screen.markdown(f'''
    <div class="aqi-hero" style="background:linear-gradient(135deg,#16213A,{BG});">
            <div style="margin-top:1rem;display:flex;align-items:center;gap:12px;">
            <style>
                .aqi-spinner {{ width:40px; height:40px; display:inline-block; }}
                @keyframes aqi-spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}
                .aqi-spinner svg {{ animation: aqi-spin 1s linear infinite; width:40px; height:40px; transform-origin: center; }}
            </style>
            <div class="aqi-spinner">
                <svg viewBox="0 0 50 50" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <circle cx="25" cy="25" r="20" fill="none" stroke="{INK_MUTED}" stroke-width="5" stroke-opacity="0.2"/>
                    <path d="M45 25a20 20 0 0 1-20 20" stroke="{INK}" stroke-width="5" stroke-linecap="round" fill="none"/>
                </svg>
            </div>
            <div style="color:{INK};font-family:'Fraunces',serif;font-size:1.05rem;font-weight:600;">AQI forecastor - Lahore</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    # Use the custom SVG spinner above; call loader without Streamlit's spinner overlay
    load_forecast()
    st.session_state["dashboard_loaded"] = True
loading_screen.empty()


current_value_for_spectrum = None
if "prediction" in st.session_state and not st.session_state.get("prediction_error"):
    current_value_for_spectrum = st.session_state["prediction"]["h24"]
st.markdown(render_spectrum(current_value_for_spectrum), unsafe_allow_html=True)

now_str = datetime.now().strftime("%a %d %b · %H:%M")
st.markdown(f'''
<div class="aqi-app-header">
    <div class="aqi-brand">AQI <span>Forecaster</span><span class="aqi-brand-sub">Lahore air-quality intelligence</span></div>
    <div class="aqi-location"><span class="aqi-location-pin" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 2a7 7 0 0 0-7 7c0 5.25 7 13 7 13s7-7.75 7-13a7 7 0 0 0-7-7Zm0 10.1A3.1 3.1 0 1 1 12 5.9a3.1 3.1 0 0 1 0 6.2Z"/></svg></span> Lahore, Pakistan &nbsp;·&nbsp; {now_str}</div>
</div>
''', unsafe_allow_html=True)

# Render sections in one scrollable page: forecast first, then EDA and SHAP.
tab_forecast = st.container()
tab_eda = st.container()
tab_shap = st.container()

with tab_forecast:
    if st.session_state.get("prediction_error"):
        st.error(f"Couldn't reach the forecast service: {st.session_state['prediction_error']}")
        if st.button("↻ Retry dashboard", key="retry_dashboard"):
            for state_key in ("dashboard_loaded", "analysis_loaded", "prediction", "hist_df", "explain"):
                st.session_state.pop(state_key, None)
            st.rerun()
    elif "prediction" not in st.session_state:
        st.info("The forecast service is unavailable. EDA and explainability may still be shown below.")
    else:
        pred, pred_time = st.session_state["prediction"], st.session_state["prediction_time"]
        value = pred["h24"]
        label, color, advice, sev_idx = aqi_band(value)
        haze_opacity = 0.06 + sev_idx * 0.045
        haze_speed = 70 - sev_idx * 10
        ring_pct = max(0, min(value, AQI_SCALE_MAX)) / AQI_SCALE_MAX * 100
        ring_glow = hex_to_rgba(color, 0.55)
        st.markdown(f'''<div class="aqi-hero" style="background:linear-gradient(135deg,{color}45,{BG} 72%);--haze-opacity:{haze_opacity};--haze-speed:{haze_speed}s;">
            <div class="aqi-hero-flex">
                <div class="aqi-ring" style="--ring-color:{color};--ring-pct:{ring_pct:.1f};--ring-glow:{ring_glow};">
                    <div class="aqi-ring-track"></div>
                    <div class="aqi-ring-hole"></div>
                    <div class="aqi-ring-center">
                        <div class="aqi-ring-number">{value:.0f}</div>
                        <div class="aqi-ring-scale">AQI · 0&ndash;500</div>
                    </div>
                </div>
                <div class="aqi-hero-text">
                    <div class="aqi-hero-city">Lahore · Next 24h Outlook</div>
                    <div class="aqi-hero-label" style="color:{color};">{label}</div>
                    <div class="aqi-hero-sub">Forecast generated {pred_time.strftime('%H:%M:%S')} · model: aqi_forecast_multi</div>
                    <div class="aqi-hero-advice">{advice}</div>
                </div>
                {f'<img class="aqi-guardian" src="{STICKER_URI}" alt="AQI air-quality guardian sticker">' if STICKER_URI else ''}
            </div>
        </div>''', unsafe_allow_html=True)

        horizons = [
            ("24 HOURS", pred["h24"], None),
            ("48 HOURS", pred["h48"], pred["h24"]),
            ("72 HOURS", pred["h72"], pred["h48"]),
        ]
        peak_title, peak_forecast, _ = max(horizons, key=lambda item: item[1])
        peak_label, _, peak_advice, peak_severity = aqi_band(peak_forecast)
        alert_message = (
            f"{peak_label} AQI alert — peak forecast is {peak_forecast:.0f} for "
            f"{peak_title}. {peak_advice}"
        )
        if peak_severity == 0:
            st.success(f"✅ {alert_message}")
        elif peak_severity == 1:
            st.info(f"ℹ️ {alert_message}")
        elif peak_severity < 4:
            st.warning(f"⚠️ {alert_message}")
        else:
            st.error(f"🚨 {alert_message}")
        for col, (title, forecast, prev) in zip(st.columns(3), horizons):
            with col:
                band, band_color, _, _ = aqi_band(forecast)
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=forecast,
                    number={"font": {"family": "IBM Plex Mono", "size": 34, "color": INK}},
                    gauge={
                        "axis": {"range": [0, AQI_SCALE_MAX], "tickcolor": INK_MUTED},
                        "bar": {"color": band_color, "thickness": .28},
                        "bgcolor": "rgba(0,0,0,0)",
                        "borderwidth": 0,
                        "threshold": {"line": {"color": "#ffffff", "width": 2}, "thickness": 0.75, "value": forecast},
                        "steps": [
                            {"range": [lo, hi], "color": hex_to_rgba(c, 0.14)}
                            for lo, hi, _, c, _ in AQI_BANDS
                        ],
                    },
                ))
                fig.update_layout(**chart_theme(), height=190, margin=dict(l=20, r=20, t=10, b=0), showlegend=False)
                st.markdown(f'<div class="aqi-card" style="--band-color:{band_color};"><div class="aqi-card-eyebrow">{title}</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                delta_line = delta_badge(forecast, prev) if prev is not None else f'<span style="color:{INK_MUTED};">Nearest forecast</span>'
                st.markdown(f'<div class="aqi-card-cat" style="color:{band_color};">{band}</div><div class="aqi-card-delta">{delta_line}</div></div>', unsafe_allow_html=True)

        refresh_col, _ = st.columns([1, 4])
        with refresh_col:
            if st.button("↻ Refresh forecast", key="refresh_forecast"):
                with st.spinner("Refreshing AQI forecast..."):
                    load_forecast()
                st.rerun()

        values = [pred["h24"], pred["h48"], pred["h72"]]
        trend = go.Figure()
        for lo, hi, band_label, band_color, _ in AQI_BANDS:
            trend.add_hrect(y0=lo, y1=min(hi, AQI_SCALE_MAX), fillcolor=hex_to_rgba(band_color, 0.05), line_width=0)
        trend.add_trace(go.Scatter(
            x=["+24h", "+48h", "+72h"], y=values, mode="lines+markers",
            line=dict(color="#8FA0BD", width=2, shape="spline"),
            marker=dict(size=12, color=[aqi_band(item)[1] for item in values], line=dict(width=2, color=BG)),
            fill="tozeroy", fillcolor="rgba(143,160,189,.05)",
        ))
        trend.update_layout(
            **chart_theme(),
            height=260, margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(tickfont=dict(family="IBM Plex Mono", color=INK_MUTED, size=12)),
            yaxis={**axis_style("AQI"), "range": [0, max(values) * 1.25 + 10]},
            showlegend=False,
        )
        st.plotly_chart(trend, use_container_width=True, config={"displayModeBar": False})
        worst_value = max(values)
        worst_label, worst_color, worst_advice, _ = aqi_band(worst_value)
        st.markdown(f'''<div class="aqi-advisory" style="--advisory-color:{worst_color};"><div class="aqi-advisory-title" style="color:{worst_color};">Peak forecast: {worst_label} ({worst_value:.0f})</div><div class="aqi-advisory-text">{worst_advice}</div></div>''', unsafe_allow_html=True)

        if "hist_df" in st.session_state and not st.session_state["hist_df"].empty:
            latest_features = st.session_state["hist_df"].copy()
            latest_features["time"] = pd.to_datetime(latest_features["time"], errors="coerce")
            latest_row = latest_features.sort_values("time").iloc[-1]
            pollutant_cards = [
                ("pm2_5", "Particulate 2.5", "µg/m³", "#35D97A"),
                ("pm10", "Particulate 10", "µg/m³", "#F4C430"),
                ("ozone", "Ozone (O₃)", "µg/m³", "#60C7FF"),
            ]
            available_cards = [card for card in pollutant_cards if card[0] in latest_row.index and pd.notna(latest_row[card[0]])]
            if available_cards:
                # Add a small vertical spacer for improved readability on narrow screens
                st.markdown('<div style="height:1.0rem"></div>', unsafe_allow_html=True)
                st.markdown('<p class="aqi-section-label">Latest pollutant snapshot</p>', unsafe_allow_html=True)
                for col, (column, title, unit, accent) in zip(st.columns(len(available_cards)), available_cards):
                    with col:
                        reading = float(latest_row[column])
                        st.markdown(f'''<div class="aqi-card" style="--band-color:{accent};"><div class="aqi-card-eyebrow">{title}</div><div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:{INK};">{reading:.1f}<span style="font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:{INK_MUTED};"> {unit}</span></div><div style="height:4px;background:{HAIRLINE};border-radius:99px;margin-top:1rem;overflow:hidden;"><div style="height:100%;width:{min(reading / 150 * 100, 100):.0f}%;background:{accent};border-radius:99px;"></div></div><div class="aqi-card-delta">latest measured feature</div></div>''', unsafe_allow_html=True)

st.markdown('<div class="aqi-section-spacer"></div>', unsafe_allow_html=True)

with tab_eda:
    # Extra spacing to keep the section label visually separated on small screens
    st.markdown('<div style="height:1.0rem"></div>', unsafe_allow_html=True)
    st.markdown('<p class="aqi-section-label">Historical feature analysis</p>', unsafe_allow_html=True)
    st.caption("Latest 30 days of engineered training features.")
    if "hist_df" not in st.session_state:
        # Show a loading info while the background loader runs; show a retry button only
        # when an actual error has been recorded in session state.
        if "history_error" in st.session_state:
            st.warning(f"Historical data could not be loaded: {st.session_state.get('history_error')}")
            if st.button("↻ Retry historical data", key="retry_history"):
                with st.spinner("Retrying historical data..."):
                    try:
                        response = requests.get(f"{api_base_url.rstrip('/')}/history", params={"days": 30}, timeout=30)
                        response.raise_for_status()
                        st.session_state["hist_df"] = pd.DataFrame(response.json())
                        st.session_state.pop("history_error", None)
                    except Exception as exc:
                        st.session_state["history_error"] = str(exc)
                st.experimental_rerun()
        else:
            st.info("Historical data is loading...")
    else:
        hist_df = st.session_state["hist_df"].copy()
        hist_df["time"] = pd.to_datetime(hist_df["time"], errors="coerce")
        numeric = hist_df.select_dtypes(include="number").columns.tolist()
        target = "aqi" if "aqi" in hist_df else "target_aqi_24"
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{len(hist_df):,}")
        c2.metric("Feature completeness", f"{(1 - hist_df[numeric].isna().mean().mean()) * 100:.1f}%")
        c3.metric("Latest observation", hist_df["time"].max().strftime("%d %b %H:%M") if hist_df["time"].notna().any() else "Unknown")
        trend = go.Figure()
        if target in hist_df:
            for lo, hi, band_label, band_color, _ in AQI_BANDS:
                trend.add_hrect(y0=lo, y1=min(hi, AQI_SCALE_MAX), fillcolor=hex_to_rgba(band_color, 0.05), line_width=0)
        trend.add_trace(go.Scatter(x=hist_df["time"], y=hist_df[target], mode="lines", name="Observed AQI", line=dict(color="#8FA0BD")))
        for feature, color in (("pm2_5", "#FF9142"), ("pm10", "#F4C430")):
            if feature in hist_df:
                trend.add_trace(go.Scatter(x=hist_df["time"], y=hist_df[feature], mode="lines", name=feature, line=dict(color=color), visible="legendonly"))
        trend.update_layout(
            **chart_theme("AQI and pollutant history"),
            height=380,
            margin=dict(t=90, b=40),
            xaxis=dict(tickfont=dict(family="IBM Plex Mono", color=INK_MUTED, size=11), gridcolor=HAIRLINE),
            yaxis=axis_style("Value"),
            showlegend=True,
        )
        st.plotly_chart(trend, use_container_width=True)
        st.caption("Tip: pm2_5 and pm10 start hidden — click their name in the legend above to show them.")

        usable = [col for col in numeric if hist_df[col].notna().sum() > 2]
        corr = hist_df[usable].corr().loc[usable, usable]
        heatmap = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns, colorscale="RdBu", zmid=0,
            colorbar=dict(
                title=dict(text="r", font=dict(color=INK, family="IBM Plex Mono")),
                tickfont=dict(color=INK_MUTED, family="IBM Plex Mono"),
                outlinecolor=HAIRLINE,
            ),
        ))
        heatmap.update_layout(
            **chart_theme("Feature correlation"),
            height=620,
            xaxis=dict(tickfont=dict(family="IBM Plex Mono", color=INK_MUTED, size=10)),
            yaxis=dict(tickfont=dict(family="IBM Plex Mono", color=INK_MUTED, size=10)),
            showlegend=False,
        )
        st.plotly_chart(heatmap, use_container_width=True)

        candidates = [col for col in usable if col != target and "target" not in col]
        if candidates:
            feature = st.selectbox("Feature vs observed AQI", candidates)
            scatter = go.Figure(go.Scatter(
                x=hist_df[feature], y=hist_df[target], mode="markers",
                marker=dict(
                    color=hist_df[target], colorscale="Viridis", size=8, opacity=.75,
                    line=dict(width=0.5, color=BG),
                    colorbar=dict(
                        title=dict(text=target, font=dict(color=INK, family="IBM Plex Mono")),
                        tickfont=dict(color=INK_MUTED, family="IBM Plex Mono"),
                        outlinecolor=HAIRLINE,
                    ),
                ),
                text=hist_df["time"].astype(str),
            ))
            scatter.update_layout(
                **chart_theme(f"{feature} vs {target}"),
                xaxis=axis_style(feature),
                yaxis=axis_style(target),
                showlegend=False,
            )
            st.plotly_chart(scatter, use_container_width=True)

st.markdown('<div class="aqi-section-spacer"></div>', unsafe_allow_html=True)

with tab_shap:
    st.markdown('<p class="aqi-section-label">Why the latest forecast changed</p>', unsafe_allow_html=True)
    st.caption("SHAP compares the latest online feature vector with a sample of historical training data.")
    if "explain" in st.session_state:
        exp = st.session_state["explain"]
        horizon = st.selectbox("Forecast horizon", list(exp["horizons"]), key="shap_horizon")
        detail = exp["horizons"][horizon]
        shap_df = pd.DataFrame({"feature": exp["features"], "shap_value": detail["shap_values"]})
        shap_df["feature_value"] = shap_df["feature"].map(exp["feature_values"])
        shap_df = shap_df.reindex(shap_df["shap_value"].abs().sort_values().tail(15).index)
        colors = ["#FF4D5E" if value > 0 else "#35D97A" for value in shap_df["shap_value"]]
        chart = go.Figure(go.Bar(x=shap_df["shap_value"], y=shap_df["feature"], orientation="h", marker_color=colors, customdata=shap_df[["feature_value"]], hovertemplate="%{y}<br>SHAP: %{x:.2f}<br>Feature value: %{customdata[0]:.2f}<extra></extra>"))
        chart.update_layout(
            **chart_theme(f"Top feature contributions — {horizon} forecast"),
            height=480,
            xaxis=axis_style("Impact on predicted AQI"),
            yaxis=dict(tickfont=dict(family="IBM Plex Mono", color=INK, size=12)),
            showlegend=False,
        )
        st.plotly_chart(chart, use_container_width=True)
        st.caption(f"Prediction: {detail['prediction']:.1f} AQI · Baseline: {detail['base_value']:.1f}. Red raises the predicted AQI; green lowers it.")
    else:
        if "explain_error" in st.session_state:
            st.warning(f"Explanation unavailable: {st.session_state.get('explain_error')}")
            if st.button("↻ Retry explanation", key="retry_explain"):
                with st.spinner("Retrying explanation..."):
                    try:
                        response = requests.post(f"{api_base_url.rstrip('/')}/explain", timeout=120)
                        response.raise_for_status()
                        st.session_state["explain"] = response.json()
                        st.session_state.pop("explain_error", None)
                    except Exception as exc:
                        st.session_state["explain_error"] = str(exc)
                st.experimental_rerun()
        else:
            st.info("Model explanation is loading...")


if "analysis_loaded" not in st.session_state:
    load_analysis()
    st.session_state["analysis_loaded"] = True
    st.rerun()
