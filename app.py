"""
app.py  —  Subsea VFM Pro  |  Streamlit Dashboard
===================================================
Run:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import io
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Subsea VFM Pro",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Imports ──────────────────────────────────────────────────────
try:
    from engine import FieldController, VFMError
    from data_loader import (
        load_and_prepare, load_client_data,
        detect_columns, get_well_configs_from_data,
        prepare_ml_dataset, prepare_well_tests,
        VOLVE_SHORT_NAMES
    )
    from services.vfm_service      import VFMService, VFMInputs
    from services.training_service import TrainingService
    from storage.model_store       import ModelStore
    ENGINE_OK = True
except Exception as e:
    st.error(f"Engine import failed: {e}")
    ENGINE_OK = False
    st.stop()

# ════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Industrial dark / petroleum aesthetic
# Palette: near-black bg, molten amber accent, electric teal signal
# ════════════════════════════════════════════════════════════════
THEME = dict(
    bg0       = "#08090b",   # void black — page background
    bg1       = "#0f1117",   # surface — cards, panels
    bg2       = "#161921",   # raised surface — expanders, inputs
    bg3       = "#1e2330",   # border / divider
    amber     = "#f5a623",   # primary accent — oil / energy
    amber_dim = "#b87d1a",   # amber muted
    teal      = "#00e5c3",   # signal green — safe / active
    red       = "#ff4545",   # alert / critical
    yellow    = "#ffd166",   # warning
    text0     = "#f0f4ff",   # primary text
    text1     = "#8b95b0",   # secondary text
    text2     = "#4a5270",   # tertiary / disabled
)

st.markdown(f"""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@600;700;800&family=Inter:wght@300;400;500&display=swap');

  /* ── Root tokens ── */
  :root {{
    --bg0:   {THEME['bg0']};
    --bg1:   {THEME['bg1']};
    --bg2:   {THEME['bg2']};
    --bg3:   {THEME['bg3']};
    --amber: {THEME['amber']};
    --teal:  {THEME['teal']};
    --red:   {THEME['red']};
    --yellow:{THEME['yellow']};
    --t0:    {THEME['text0']};
    --t1:    {THEME['text1']};
    --t2:    {THEME['text2']};
  }}

  /* ── Global reset ── */
  html, body, [data-testid="stAppViewContainer"],
  [data-testid="stApp"] {{
    background: var(--bg0) !important;
    font-family: 'Inter', sans-serif;
    color: var(--t0);
  }}

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {{
    background: var(--bg1) !important;
    border-right: 1px solid var(--bg3);
  }}
  [data-testid="stSidebar"] * {{ color: var(--t0) !important; }}

  /* ── Header strip ── */
  header[data-testid="stHeader"] {{
    background: var(--bg0) !important;
    border-bottom: 1px solid var(--bg3);
  }}

  /* ── Main container padding ── */
  .main .block-container {{
    padding: 2rem 2.5rem 3rem;
    max-width: 1400px;
  }}

  /* ── Typography ── */
  h1, h2, h3 {{
    font-family: 'Syne', sans-serif !important;
    letter-spacing: -0.02em;
    color: var(--t0) !important;
  }}
  code, pre, .mono {{
    font-family: 'Space Mono', monospace !important;
  }}

  /* ── Metric cards ── */
  .vfm-card {{
    background: var(--bg1);
    border: 1px solid var(--bg3);
    border-top: 2px solid var(--amber);
    border-radius: 6px;
    padding: 18px 20px 14px;
    margin-bottom: 12px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.15s;
  }}
  .vfm-card:hover {{ transform: translateY(-1px); border-color: var(--amber); }}
  .vfm-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--amber), transparent);
    opacity: 0.5;
  }}
  .vfm-card-label {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--t1);
    margin-bottom: 8px;
  }}
  .vfm-card-value {{
    font-family: 'Syne', sans-serif;
    font-size: 30px;
    font-weight: 800;
    color: var(--t0);
    line-height: 1;
  }}
  .vfm-card-unit {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--t1);
    margin-top: 4px;
  }}
  .vfm-card-accent {{ border-top-color: var(--teal); }}
  .vfm-card-warn   {{ border-top-color: var(--yellow); }}
  .vfm-card-crit   {{ border-top-color: var(--red); }}
  .vfm-card-gas    {{ border-top-color: #c792ea; }}

  /* ── Section headers ── */
  .sec-head {{
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--amber);
    border-bottom: 1px solid var(--bg3);
    padding-bottom: 8px;
    margin: 24px 0 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  .sec-head::before {{
    content: '';
    display: inline-block;
    width: 3px;
    height: 12px;
    background: var(--amber);
    border-radius: 2px;
  }}

  /* ── Hero banner ── */
  .hero {{
    background: linear-gradient(135deg, #0c0e14 0%, #111520 50%, #0a0d13 100%);
    border: 1px solid var(--bg3);
    border-left: 3px solid var(--amber);
    border-radius: 8px;
    padding: 28px 32px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
  }}
  .hero::after {{
    content: '⬡';
    position: absolute;
    right: -20px;
    top: -30px;
    font-size: 180px;
    color: rgba(245,166,35,0.03);
    line-height: 1;
    pointer-events: none;
  }}
  .hero-title {{
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: var(--amber);
    margin: 0 0 6px;
    letter-spacing: -0.02em;
  }}
  .hero-sub {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--t1);
    letter-spacing: 1px;
    margin: 0;
  }}
  .hero-badge {{
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    letter-spacing: 2px;
    text-transform: uppercase;
    background: rgba(0, 229, 195, 0.08);
    color: var(--teal);
    border: 1px solid rgba(0, 229, 195, 0.2);
    padding: 3px 10px;
    border-radius: 2px;
    margin-top: 10px;
    margin-right: 6px;
  }}

  /* ── Well status pills ── */
  .well-pill {{
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: 4px;
    margin-bottom: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
  }}
  .well-pill .dot {{
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
  }}
  .dot-live  {{ background: var(--teal); box-shadow: 0 0 6px var(--teal); }}
  .dot-dead  {{ background: var(--t2); }}
  .well-pill .name {{ color: var(--t0); font-weight: 700; flex: 1; }}
  .well-pill .cal  {{ color: var(--amber); font-size: 9px; letter-spacing: 1px; }}

  /* ── Logo / wordmark ── */
  .sidebar-logo {{
    font-family: 'Syne', sans-serif;
    font-size: 18px;
    font-weight: 800;
    color: var(--amber);
    letter-spacing: -0.02em;
    margin-bottom: 2px;
  }}
  .sidebar-tagline {{
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: var(--t2);
    letter-spacing: 2px;
    text-transform: uppercase;
  }}

  /* ── Nav radio override ── */
  [data-testid="stRadio"] label {{
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.5px !important;
    color: var(--t1) !important;
    padding: 6px 0 !important;
  }}
  [data-testid="stRadio"] label:hover {{ color: var(--amber) !important; }}
  [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p {{
    font-family: 'Space Mono', monospace !important;
  }}

  /* ── Inputs ── */
  [data-baseweb="input"], [data-baseweb="select"],
  [data-testid="stNumberInput"] input,
  [data-testid="stTextInput"] input {{
    background: var(--bg2) !important;
    border-color: var(--bg3) !important;
    color: var(--t0) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 13px !important;
    border-radius: 4px !important;
  }}
  [data-baseweb="input"]:focus-within,
  [data-testid="stNumberInput"] input:focus {{
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 1px var(--amber) !important;
  }}

  /* ── Select ── */
  [data-baseweb="select"] > div {{
    background: var(--bg2) !important;
    border-color: var(--bg3) !important;
  }}

  /* ── Sliders ── */
  [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background: var(--amber) !important;
  }}

  /* ── Buttons ── */
  [data-testid="stButton"] button[kind="primary"] {{
    background: var(--amber) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 10px 24px !important;
    transition: opacity 0.15s, transform 0.1s !important;
  }}
  [data-testid="stButton"] button[kind="primary"]:hover {{
    opacity: 0.88 !important;
    transform: translateY(-1px) !important;
  }}
  [data-testid="stButton"] button:not([kind="primary"]) {{
    background: transparent !important;
    border: 1px solid var(--bg3) !important;
    color: var(--t1) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 1px !important;
    border-radius: 4px !important;
  }}
  [data-testid="stButton"] button:not([kind="primary"]):hover {{
    border-color: var(--amber) !important;
    color: var(--amber) !important;
  }}

  /* ── Download button ── */
  [data-testid="stDownloadButton"] button {{
    background: transparent !important;
    border: 1px solid var(--bg3) !important;
    color: var(--teal) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 1px !important;
    border-radius: 4px !important;
  }}
  [data-testid="stDownloadButton"] button:hover {{
    border-color: var(--teal) !important;
    background: rgba(0,229,195,0.05) !important;
  }}

  /* ── Progress bar ── */
  [data-testid="stProgress"] > div > div {{
    background: var(--amber) !important;
  }}
  [data-testid="stProgress"] > div {{
    background: var(--bg3) !important;
  }}

  /* ── Expanders ── */
  [data-testid="stExpander"] {{
    background: var(--bg2) !important;
    border: 1px solid var(--bg3) !important;
    border-radius: 6px !important;
  }}
  [data-testid="stExpander"] summary {{
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    color: var(--t1) !important;
  }}

  /* ── Dataframe ── */
  [data-testid="stDataFrame"] {{
    border: 1px solid var(--bg3) !important;
    border-radius: 6px !important;
    overflow: hidden;
  }}

  /* ── Alerts ── */
  [data-testid="stAlert"] {{
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
  }}

  /* ── Divider ── */
  hr {{ border-color: var(--bg3) !important; }}

  /* ── Metric native ── */
  [data-testid="stMetric"] {{
    background: var(--bg2) !important;
    padding: 12px 16px !important;
    border-radius: 6px !important;
    border: 1px solid var(--bg3) !important;
  }}
  [data-testid="stMetricLabel"] {{ color: var(--t1) !important; font-size: 11px !important; }}
  [data-testid="stMetricValue"] {{
    font-family: 'Syne', sans-serif !important;
    color: var(--t0) !important;
  }}

  /* ── Checkboxes & file uploaders ── */
  [data-testid="stFileUploader"] {{
    background: var(--bg2) !important;
    border: 1px dashed var(--bg3) !important;
    border-radius: 6px !important;
  }}
  [data-testid="stFileUploader"]:hover {{
    border-color: var(--amber) !important;
  }}

  /* ── Tabs (if used) ── */
  [data-baseweb="tab"] {{
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
  }}

  /* ── Caption ── */
  [data-testid="stCaptionContainer"] p {{
    font-family: 'Space Mono', monospace !important;
    font-size: 10px !important;
    color: var(--t2) !important;
    letter-spacing: 0.5px;
  }}

  /* ── Scrollbar ── */
  ::-webkit-scrollbar {{ width: 4px; height: 4px; }}
  ::-webkit-scrollbar-track {{ background: var(--bg0); }}
  ::-webkit-scrollbar-thumb {{ background: var(--bg3); border-radius: 2px; }}
  ::-webkit-scrollbar-thumb:hover {{ background: var(--amber); }}

  /* ── Status badge ── */
  .status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
  }}
  .badge-ok   {{ background: rgba(0,229,195,0.08); color: var(--teal); border: 1px solid rgba(0,229,195,0.2); }}
  .badge-warn {{ background: rgba(255,209,102,0.08); color: var(--yellow); border: 1px solid rgba(255,209,102,0.2); }}
  .badge-crit {{ background: rgba(255,69,69,0.08); color: var(--red); border: 1px solid rgba(255,69,69,0.2); }}

  /* ── Uncertainty range bar ── */
  .unc-bar-wrap {{
    background: var(--bg3);
    height: 4px;
    border-radius: 2px;
    margin: 10px 0 6px;
    position: relative;
  }}
  .unc-bar-fill {{
    height: 4px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--amber), var(--teal));
    position: absolute;
  }}

  /* ── About section ── */
  .about-table th {{
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--t1);
  }}

  /* ── Sidebar nav ── */
  .nav-item {{
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--t1);
    padding: 6px 0;
    cursor: pointer;
    letter-spacing: 0.5px;
  }}
  .nav-item.active {{ color: var(--amber); }}

  /* Hide streamlit branding */
  #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
# WELL CONFIGS & MAPS
# ════════════════════════════════════════════════════════════════
WELL_CONFIGS = {
    "F-14 H": {"depth":2900,"diameter":0.152,"t_res":373.0,"t_sea":278.0,
               "api":34.0,"sg_gas":0.71,"productivity_index":30.0,
               "wc":0.30,"gor":846.0,"rho_brine":1028.0,"choke_size_64":12.0},
    "F-12 H": {"depth":2800,"diameter":0.127,"t_res":371.0,"t_sea":278.0,
               "api":33.0,"sg_gas":0.72,"productivity_index":18.0,
               "wc":0.55,"gor":850.0,"rho_brine":1028.0,"choke_size_64":7.0},
    "F-11 H": {"depth":2800,"diameter":0.127,"t_res":371.0,"t_sea":278.0,
               "api":33.5,"sg_gas":0.72,"productivity_index":12.0,
               "wc":0.40,"gor":857.0,"rho_brine":1028.0,"choke_size_64":21.0},
    "F-15 D": {"depth":2700,"diameter":0.127,"t_res":370.0,"t_sea":278.0,
               "api":33.0,"sg_gas":0.72,"productivity_index":8.0,
               "wc":0.20,"gor":846.0,"rho_brine":1028.0,"choke_size_64":9.0},
    "F-1 C":  {"depth":2750,"diameter":0.127,"t_res":370.0,"t_sea":278.0,
               "api":33.0,"sg_gas":0.72,"productivity_index":6.0,
               "wc":0.50,"gor":842.0,"rho_brine":1028.0,"choke_size_64":12.0},
}
VOLVE_CODE_MAP = {
    "NO 15/9-F-14 H": "F-14 H", "NO 15/9-F-12 H": "F-12 H",
    "NO 15/9-F-11 H": "F-11 H", "NO 15/9-F-15 D": "F-15 D",
    "NO 15/9-F-1 C":  "F-1 C",
}

# ════════════════════════════════════════════════════════════════
# MODEL STORE & SESSION STATE
# ════════════════════════════════════════════════════════════════
_store = ModelStore(store_dir="pretrained_models")

def _get_pretrained_meta() -> dict:
    return _store.meta()

if "ctrl"              not in st.session_state: st.session_state.ctrl              = None
if "trained"           not in st.session_state: st.session_state.trained           = {}
if "calibrated"        not in st.session_state: st.session_state.calibrated        = {}
if "clean_data"        not in st.session_state: st.session_state.clean_data        = None
if "last_results"      not in st.session_state: st.session_state.last_results      = []
if "batch_results"     not in st.session_state: st.session_state.batch_results     = None
if "client_mode"       not in st.session_state: st.session_state.client_mode       = False
if "detected_col_map"  not in st.session_state: st.session_state.detected_col_map  = None
if "pretrained_loaded" not in st.session_state: st.session_state.pretrained_loaded = False
if "active_well_configs" not in st.session_state:
    st.session_state.active_well_configs = WELL_CONFIGS.copy()

if not st.session_state.pretrained_loaded:
    _payload = _store.load()
    if _payload is not None:
        st.session_state.ctrl                = _payload["ctrl"]
        st.session_state.trained             = _payload.get("trained", {})
        st.session_state.calibrated          = _payload.get("calibrated", {})
        st.session_state.active_well_configs = _payload.get("active_well_configs", WELL_CONFIGS.copy())
    else:
        st.session_state.active_well_configs = WELL_CONFIGS.copy()
    st.session_state.pretrained_loaded = True

# ════════════════════════════════════════════════════════════════
# SERVICE HELPERS
# ════════════════════════════════════════════════════════════════
def get_ctrl(well_configs: dict = None) -> FieldController:
    configs = well_configs or st.session_state.active_well_configs
    if st.session_state.ctrl is None:
        ctrl = FieldController()
        for name, cfg in configs.items():
            ctrl.add_well(name, cfg)
        st.session_state.ctrl = ctrl
    return st.session_state.ctrl

def get_vfm_service() -> VFMService:
    return VFMService(ctrl=get_ctrl(), trained=st.session_state.trained,
                      calibrated=st.session_state.calibrated)

def get_training_service(n_boot=20, n_est=80, on_progress=None) -> TrainingService:
    return TrainingService(ctrl=get_ctrl(), trained=st.session_state.trained,
                           calibrated=st.session_state.calibrated,
                           n_boot=n_boot, n_est=n_est,
                           on_progress=on_progress or (lambda *a: None))

# ════════════════════════════════════════════════════════════════
# UI HELPERS
# ════════════════════════════════════════════════════════════════
def metric_card(label, value, unit="", variant="default"):
    cls_map = {"default":"", "teal":"vfm-card-accent",
               "warn":"vfm-card-warn", "crit":"vfm-card-crit", "gas":"vfm-card-gas"}
    extra = cls_map.get(variant, "")
    st.markdown(f"""
    <div class="vfm-card {extra}">
      <div class="vfm-card-label">{label}</div>
      <div class="vfm-card-value">{value}</div>
      <div class="vfm-card-unit">{unit}</div>
    </div>""", unsafe_allow_html=True)

def sec_head(title):
    st.markdown(f'<div class="sec-head">{title}</div>', unsafe_allow_html=True)

def badge(text, kind="ok"):
    st.markdown(f'<span class="status-badge badge-{kind}">{text}</span>',
                unsafe_allow_html=True)

def hydrate_variant(status):
    return {"SAFE": "teal", "WARNING": "warn", "CRITICAL": "crit"}.get(status, "warn")

# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 20px;">
      <div class="sidebar-logo">⬡ VFM PRO</div>
      <div class="sidebar-tagline">Subsea Virtual Flow Meter</div>
    </div>""", unsafe_allow_html=True)

    st.divider()

    tab_choice = st.radio(
        "nav", ["🏠  Setup", "⚡  Live VFM", "📊  Field Overview",
                "📁  Batch Run", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.divider()

    # Well status
    st.markdown('<div style="font-family:Space Mono,monospace;font-size:9px;'
                'letter-spacing:2px;text-transform:uppercase;color:#4a5270;'
                'margin-bottom:10px;">Well Status</div>', unsafe_allow_html=True)

    for name in st.session_state.active_well_configs:
        trained    = st.session_state.trained.get(name, False)
        cal        = st.session_state.calibrated.get(name, False)
        dot_cls    = "dot-live" if trained else "dot-dead"
        cal_label  = "CAL ✓" if cal else ""
        st.markdown(f"""
        <div class="well-pill">
          <div class="dot {dot_cls}"></div>
          <div class="name">{name}</div>
          <div class="cal">{cal_label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # Pretrained meta
    _meta = _get_pretrained_meta()
    if _meta:
        _date = (_meta.get('generated_at') or _meta.get('saved_at',''))[:10]
        volve  = _meta.get("volve_summary", {})
        n_rec  = volve.get('total_records', 0)
        st.markdown(f"""
        <div style="font-family:Space Mono,monospace;font-size:9px;color:#4a5270;
                    letter-spacing:1px;text-transform:uppercase;margin-bottom:6px;">
          Model Source
        </div>
        <div style="font-family:Space Mono,monospace;font-size:10px;color:#00e5c3;">
          ● PRE-TRAINED
        </div>
        <div style="font-family:Space Mono,monospace;font-size:9px;color:#4a5270;
                    margin-top:3px;">{_date} · {n_rec:,} records</div>
        """, unsafe_allow_html=True)
        st.divider()

    st.markdown("""
    <div style="font-family:Space Mono,monospace;font-size:9px;color:#2a3050;
                letter-spacing:1px;line-height:1.8;">
      v4.1 · MAPE 6.1%<br>
      Volve 2008–2016
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB: SETUP
# ════════════════════════════════════════════════════════════════
if tab_choice == "🏠  Setup":
    st.markdown("""
    <div class="hero">
      <div class="hero-title">🛢️ Subsea VFM Pro</div>
      <p class="hero-sub">Hybrid Physics-ML Virtual Flow Meter · Equinor Volve Validated</p>
      <div style="margin-top:12px;">
        <span class="hero-badge">MAPE 6.1%</span>
        <span class="hero-badge">5,545 Records</span>
        <span class="hero-badge">Beggs & Brill</span>
        <span class="hero-badge">Bootstrap GBR</span>
      </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns([1.5, 1])

    with col1:
        sec_head("📂 Upload Field Data")

        data_mode = st.radio(
            "Data source",
            ["📊 Equinor Volve dataset", "🏭 Client / custom field data"],
            horizontal=True,
        )
        is_client_mode = (data_mode == "🏭 Client / custom field data")
        st.session_state.client_mode = is_client_mode

        if not is_client_mode:
            uploaded = st.file_uploader(
                "Upload Volve_production_data.xlsx", type=["xlsx"],
                help="Equinor Volve daily production Excel. Sheet: 'Daily Production Data'",
            )
            if uploaded:
                with st.spinner("Loading Volve data…"):
                    try:
                        clean, ml_data, well_tests, summary = load_and_prepare(uploaded)
                        st.session_state.clean_data = clean
                        st.session_state.active_well_configs = WELL_CONFIGS.copy()
                        st.session_state.ctrl = None
                        st.success(f"✅ {summary['total_records']:,} records · {len(summary['wells'])} wells · {summary['date_range']}")
                        with st.expander("Dataset statistics"):
                            c1,c2,c3 = st.columns(3)
                            c1.metric("BHP", f"{summary['bhp_range_psi'][0]:.0f}–{summary['bhp_range_psi'][1]:.0f}", "psi")
                            c2.metric("WHP", f"{summary['whp_range_psi'][0]:.0f}–{summary['whp_range_psi'][1]:.0f}", "psi")
                            c3.metric("Wells", len(summary["wells"]))
                            for code, n in summary["per_well_counts"].items():
                                short = VOLVE_CODE_MAP.get(code, code)
                                st.caption(f"  {short}: {n:,} records")
                    except Exception as e:
                        st.error(f"Data load failed: {e}")
                        ml_data = well_tests = None
        else:
            st.info("Upload your field production data (Excel or CSV).")
            uploaded = st.file_uploader("Upload production data", type=["xlsx","csv"])
            sheet_name = 0
            if uploaded and uploaded.name.endswith(".xlsx"):
                import openpyxl
                wb = openpyxl.load_workbook(uploaded, read_only=True)
                sheet_options = wb.sheetnames
                uploaded.seek(0)
                sheet_name = st.selectbox("Select sheet", sheet_options)

            if uploaded:
                try:
                    uploaded.seek(0)
                    preview_df = (pd.read_csv(uploaded, nrows=5)
                                  if uploaded.name.endswith(".csv")
                                  else pd.read_excel(uploaded, sheet_name=sheet_name, engine="openpyxl", nrows=5))
                    uploaded.seek(0)
                    detected = detect_columns(preview_df)
                    st.session_state.detected_col_map = detected
                    st.markdown("**Preview (first 5 rows):**")
                    st.dataframe(preview_df, use_container_width=True)

                    sec_head("🗂️ Column Mapping")
                    st.caption("Auto-detected suggestions shown. Correct any mismatches.")
                    all_cols = ["None"] + list(preview_df.columns)
                    col_map  = {}

                    def col_picker(label, key, required=True):
                        default = detected.get(key)
                        idx = all_cols.index(default) if default in all_cols else 0
                        chosen = st.selectbox(label+(" *" if required else ""),
                                              all_cols, index=idx, key=f"cm_{key}")
                        return None if chosen == "None" else chosen

                    mc1, mc2 = st.columns(2)
                    with mc1:
                        col_map["well"]       = col_picker("Well name column", "well")
                        col_map["bhp_psi"]    = col_picker("BHP column", "bhp_psi")
                        col_map["whp_psi"]    = col_picker("WHP column", "whp_psi")
                        col_map["wht_k"]      = col_picker("Wellhead temp", "wht_k")
                        col_map["q_oil_stbd"] = col_picker("Oil rate/volume", "q_oil_stbd")
                    with mc2:
                        col_map["q_gas_scfd"]    = col_picker("Gas rate/volume", "q_gas_scfd", False)
                        col_map["q_wat_stbd"]    = col_picker("Water rate/volume", "q_wat_stbd", False)
                        col_map["choke_size_64"] = col_picker("Choke size", "choke_size_64", False)
                        col_map["depth_m"]       = col_picker("Depth [m]", "depth_m", False)
                        col_map["date"]          = col_picker("Date column", "date", False)

                    sec_head("⚙️ Unit Settings")
                    uc1,uc2,uc3 = st.columns(3)
                    pressure_unit = uc1.selectbox("Pressure units", ["auto","bar","psi"])
                    temp_unit     = uc2.selectbox("Temperature units", ["auto","C","K"])
                    volume_unit   = uc3.selectbox("Volume units", ["auto","sm3","field (STB/SCF)"])
                    if volume_unit == "field (STB/SCF)": volume_unit = "field"
                    default_depth = st.number_input("Default depth [m]", 500, 6000, 2800)
                    default_t_res = st.number_input("Default reservoir temp [K]", 300, 450, 373)

                    if st.button("📥 Load client data", type="primary"):
                        if not col_map.get("well") or not col_map.get("bhp_psi") or not col_map.get("q_oil_stbd"):
                            st.error("Well name, BHP, and oil rate columns are required.")
                        else:
                            with st.spinner("Processing client data…"):
                                try:
                                    uploaded.seek(0)
                                    clean, ml_data, well_tests, summary = load_client_data(
                                        filepath=uploaded, col_map=col_map,
                                        sheet_name=sheet_name,
                                        pressure_unit=pressure_unit,
                                        temp_unit=temp_unit, volume_unit=volume_unit,
                                        default_depth_m=float(default_depth),
                                        default_t_res_k=float(default_t_res),
                                    )
                                    st.session_state.clean_data = clean
                                    new_configs = get_well_configs_from_data(clean)
                                    st.session_state.active_well_configs = new_configs
                                    st.session_state.ctrl = None
                                    st.session_state.trained = {}
                                    st.session_state.calibrated = {}
                                    wells_found = sorted(clean["well"].unique().tolist())
                                    st.success(f"✅ {summary['total_records']:,} records · {len(wells_found)} wells: {', '.join(wells_found)}")
                                except Exception as e:
                                    st.error(f"Client data load failed: {e}")
                                    import traceback; st.code(traceback.format_exc())
                except Exception as e:
                    st.error(f"Could not preview file: {e}")

        # ── Train & Calibrate ─────────────────────────────────
        ml_data = well_tests = None
        if st.session_state.clean_data is not None:
            ml_data    = prepare_ml_dataset(st.session_state.clean_data)
            well_tests = prepare_well_tests(st.session_state.clean_data)

        if ml_data:
            sec_head("🤖 Train ML Models")
            n_boot = st.slider("Bootstrap iterations (N_BOOT)", 10, 50, 20,
                               help="More = better uncertainty, slower training")
            if st.button("🚀 Train all wells", type="primary"):
                prog = st.progress(0); prog_text = st.empty()
                def on_progress(well, step, frac):
                    prog.progress(frac)
                    if step != "complete": prog_text.caption(f"Training {well}…")
                svc    = get_training_service(n_boot=n_boot, n_est=80, on_progress=on_progress)
                report = svc.train_all(ml_data, source="field_data")
                prog_text.empty()
                for well, r in report["results"].items():
                    if r["status"] == "trained":
                        st.caption(f"  ✓ {well}: {r['n_records']:,} records | {r['elapsed_s']}s")
                    else: st.warning(f"  ✗ {well}: {r.get('error')}")
                st.success(f"✅ {report['n_trained']}/{len(ml_data)} wells trained.")

            sec_head("⚙️ Calibrate Physics")
            st.caption("Tunes friction & PI multipliers via Nelder-Mead. Run after ML training.")
            if st.button("⚙️ Calibrate all wells"):
                prog2 = st.progress(0); prog_text2 = st.empty()
                def on_progress2(well, step, frac):
                    prog2.progress(frac)
                    if step != "complete": prog_text2.caption(f"Calibrating {well}…")
                svc2    = get_training_service(on_progress=on_progress2)
                report2 = svc2.calibrate_all(well_tests)
                prog_text2.empty()
                for well, r in report2["results"].items():
                    if r["status"] == "calibrated":
                        st.caption(f"  ✓ {well}: MAPE={r['mape_pct']}%  fm={r['friction_mult']:.3f}  pm={r['pi_mult']:.3f}")
                    elif r["status"] == "skipped": st.warning(f"  ~ {well}: {r.get('error','train ML first')}")
                    else: st.warning(f"  ✗ {well}: {r.get('error')}")
                st.success(f"✅ {report2['n_calibrated']}/{len(well_tests)} wells calibrated.")

    with col2:
        sec_head("ℹ️ Quick Start")
        _meta = _get_pretrained_meta()
        if _meta and any(st.session_state.trained.values()):
            st.markdown("""
            <div style="background:rgba(0,229,195,0.06);border:1px solid rgba(0,229,195,0.2);
                        border-radius:6px;padding:14px 16px;margin-bottom:16px;">
              <div style="font-family:Space Mono,monospace;font-size:10px;color:#00e5c3;
                          letter-spacing:1px;margin-bottom:4px;">● MODELS READY</div>
              <div style="font-family:Space Mono,monospace;font-size:9px;color:#4a5270;">
                Pre-trained models loaded. Go straight to Live VFM.
              </div>
            </div>""", unsafe_allow_html=True)
            with st.expander("Pre-trained model details"):
                volve = _meta.get("volve_summary", {})
                st.caption(f"Generated: {_meta.get('generated_at','N/A')}")
                st.caption(f"Records  : {volve.get('total_records',0):,}")
                st.caption(f"Period   : {volve.get('date_range','N/A')}")
                for w, v in _meta.get("training", {}).items():
                    cal = _meta.get("calibration", {}).get(w, {})
                    cal_str = f"  MAPE={cal['mape_pct']}%" if cal.get("status")=="calibrated" else ""
                    st.caption(f"  {w}: {v.get('n_records',0):,} records{cal_str}")
            st.divider()

        st.markdown("""
**Retrain on your own data:**

1. Select **"Client / custom field data"**
2. Upload your Excel or CSV
3. Map column names → required fields
4. Click **Load client data** → **Train all wells**

---

**Re-train on Volve:**

1. Select **"Equinor Volve dataset"**
2. Upload `Volve_production_data.xlsx`
3. Click **Train all wells**

---
Without uploading data, pre-trained Volve models run immediately.
Go to **⚡ Live VFM** and enter readings.
        """)


# ════════════════════════════════════════════════════════════════
# TAB: LIVE VFM
# ════════════════════════════════════════════════════════════════
elif tab_choice == "⚡  Live VFM":
    st.markdown("""
    <div style="margin-bottom:24px;">
      <h2 style="font-family:Syne,sans-serif;font-weight:800;font-size:24px;
                 color:#f5a623;margin:0 0 4px;">⚡ Live VFM</h2>
      <div style="font-family:Space Mono,monospace;font-size:10px;color:#4a5270;
                  letter-spacing:1px;">REAL-TIME FLOW ESTIMATION · ENTER SENSOR READINGS BELOW</div>
    </div>""", unsafe_allow_html=True)

    col_inputs, col_results = st.columns([1, 1.5])

    with col_inputs:
        sec_head("📡 Sensor Inputs")
        well_name = st.selectbox("Well", list(st.session_state.active_well_configs.keys()))

        c1, c2 = st.columns(2)
        bhp_psi = c1.number_input("BHP [psi]", 500, 6000, 3500, 10, help="Avg downhole pressure")
        whp_psi = c2.number_input("WHP [psi]", 14, 3000, 700, 10, help="Avg wellhead pressure")
        wht_c   = c1.number_input("WHT [°C]", 0, 120, 75, 1, help="Wellhead temperature")
        p_res   = c2.number_input("P_res [psi]", 500, 8000, 4500, 10, help="Reservoir pressure")

        c3, c4 = st.columns(2)
        wc  = c3.slider("Water cut", 0.0, 0.99, 0.30, 0.01, format="%.2f")
        gor = c4.number_input("GOR [SCF/STB]", 50, 5000, 846, 10)

        with st.expander("Optional: choke & downstream pressure"):
            choke_64 = st.number_input("Choke size [1/64 in]", 0, 128, 12, 1)
            p_dn     = st.number_input("Downstream pressure [psi]", 0, 3000, 0, 10,
                                        help="Leave 0 if not available")
        p_dn_val = p_dn if p_dn > 0 else None
        run = st.button("▶  RUN VFM", type="primary", use_container_width=True)

    with col_results:
        sec_head("📈 VFM Output")

        if run:
            wht_k = wht_c + 273.15
            vfm   = get_vfm_service()
            try:
                r = vfm.predict(well_name, VFMInputs(
                    p_res=float(p_res), whp_psi=float(whp_psi), wht_k=float(wht_k),
                    wc=float(wc), gor=float(gor),
                    bhp_psi=float(bhp_psi) if bhp_psi > whp_psi + 50 else None,
                    p_choke_dn=p_dn_val,
                    choke_size_64=float(choke_64) if choke_64 > 0 else None,
                ))
                elapsed = r.get("_compute_ms", 0)

                # ── Rate cards ────────────────────────────────
                c1, c2, c3 = st.columns(3)
                with c1: metric_card("Total Liquid", f"{r['q_total_stbd']:,.0f}", "STB/D")
                with c2: metric_card("Oil Rate", f"{r['q_oil_stbd']:,.0f}", "STB/D", "teal")
                with c3: metric_card("Water Rate", f"{r['q_water_stbd']:,.0f}", "STB/D", "default")

                c4, c5, c6 = st.columns(3)
                with c4: metric_card("Gas Rate", f"{r['q_gas_mscfd']:,.1f}", "MSCF/D", "gas")
                with c5: metric_card("GVF In-Situ", f"{r['gvf_insitu']*100:.1f}", "%")
                with c6:
                    hv = hydrate_variant(r["hydrate_status"])
                    metric_card("Hydrate Risk", r["hydrate_status"],
                                f"ΔT = {r['hydrate_delta_t_k']:.1f} K", hv)

                # ── ML Uncertainty ────────────────────────────
                if r.get("ml_uncertainty"):
                    ml = r["ml_uncertainty"]
                    sec_head("🎯 ML Uncertainty  (Bootstrap GBR)")
                    uc1, uc2, uc3, uc4 = st.columns(4)
                    uc1.metric("P10", f"{ml['p10']:,.0f}", "STB/D")
                    uc2.metric("P50", f"{ml['p50']:,.0f}", "STB/D")
                    uc3.metric("P90", f"{ml['p90']:,.0f}", "STB/D")
                    uc4.metric("CV",  f"{ml['cv']:.1f}", "%")

                    p10, p50, p90 = ml["p10"], ml["p50"], ml["p90"]
                    rng  = max(p90 - p10, 1)
                    pmin = max(p10 - rng*0.1, 0)
                    pmax = p90 + rng*0.1
                    left_pct  = max((p10 - pmin) / (pmax - pmin) * 100, 0)
                    width_pct = max((p90 - p10) / (pmax - pmin) * 100, 2)
                    st.markdown(f"""
                    <div class="unc-bar-wrap">
                      <div class="unc-bar-fill"
                           style="left:{left_pct:.1f}%;width:{width_pct:.1f}%;"></div>
                    </div>
                    <div style="font-family:Space Mono,monospace;font-size:9px;
                                color:#4a5270;letter-spacing:1px;">
                      PHYSICS WEIGHT {r['physics_weight_pct']:.0f}% ·
                      ML WEIGHT {100-r['physics_weight_pct']:.0f}% ·
                      P80 RANGE {p10:,.0f}–{p90:,.0f} STB/D
                    </div>""", unsafe_allow_html=True)
                else:
                    st.info("Upload data and train ML for uncertainty estimates.")

                # ── Physics uncertainty ───────────────────────
                phys_u = r.get("physics_uncertainty")
                if phys_u:
                    sec_head("📐 Physics Uncertainty")
                    pu1, pu2, pu3, pu4 = st.columns(4)
                    pu1.metric("P10", f"{phys_u['p10']:,.0f}", "STB/D")
                    pu2.metric("P50", f"{phys_u['p50']:,.0f}", "STB/D")
                    pu3.metric("P90", f"{phys_u['p90']:,.0f}", "STB/D")
                    pu4.metric("±Range", f"{phys_u['half_range_pct']:.0f}", "%")

                # ── Diagnostics ───────────────────────────────
                with st.expander("Diagnostics"):
                    dc1, dc2 = st.columns(2)
                    with dc1:
                        st.markdown("**Sensor Status**")
                        wf   = r["whp_sensor_flag"]
                        bf   = r["bhp_sensor_flag"]
                        wico = "🟢" if wf in ("OK","INITIALIZING","INIT") else "🟡"
                        bico = "🟢" if bf in ("OK","INITIALIZING","INIT","NOT_PROVIDED") else "🟡"
                        st.write(f"{wico} WHP: **{wf}**")
                        st.write(f"{bico} BHP: **{bf}**")
                        st.write(f"Nodal q: **{r['q_nodal_stbd']:,.0f}** STB/D")
                        if r["q_choke_stbd"]:
                            st.write(f"Choke q: **{r['q_choke_stbd']:,.0f}** STB/D")
                            st.write(f"Agreement: **{r['choke_agreement']}**")
                    with dc2:
                        st.markdown("**Hydrate & Wax**")
                        st.write(f"T_hyd: **{r['hydrate_delta_t_k']:+.1f} K** margin")
                        st.write(f"MEG dose: **{r['meg_dose_vol_pct']:.1f}** vol%")
                        st.write(f"MeOH dose: **{r['meoh_dose_vol_pct']:.1f}** vol%")
                        wax_s  = r.get("wax_status", "SAFE")
                        wax_ic = {"SAFE":"🟢","MONITOR":"🟡","RISK":"🔴"}.get(wax_s,"🟡")
                        st.write(f"{wax_ic} Wax: **{wax_s}** | ΔT_WAT={r.get('wax_delta_t_k',0):+.1f} K")
                        if r.get("deviation_penalty_pct", 0) > 0:
                            st.write(f"⚠️ Deviation penalty: **{r['deviation_penalty_pct']:.1f}%**")
                        if r.get("calibrated"):
                            st.write(f"Cal MAPE: **{r['calibration_mape']}%**")
                    st.caption(f"Compute time: {elapsed:.0f} ms")

                r["_well"] = well_name
                st.session_state.last_results = [r]

            except VFMError as e:
                st.error(f"VFM Error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        else:
            st.markdown("""
            <div style="background:var(--bg2);border:1px dashed var(--bg3);
                        border-radius:8px;padding:40px 24px;text-align:center;
                        margin-top:12px;">
              <div style="font-family:Space Mono,monospace;font-size:24px;
                          color:#2a3050;margin-bottom:12px;">⬡</div>
              <div style="font-family:Space Mono,monospace;font-size:11px;
                          color:#4a5270;letter-spacing:1px;">
                Enter sensor readings and click RUN VFM
              </div>
            </div>""", unsafe_allow_html=True)
            if not any(st.session_state.trained.values()):
                st.warning("No ML models trained — physics-only mode active. "
                           "Go to **🏠 Setup** to upload training data.")


# ════════════════════════════════════════════════════════════════
# TAB: FIELD OVERVIEW
# ════════════════════════════════════════════════════════════════
elif tab_choice == "📊  Field Overview":
    st.markdown("""
    <div style="margin-bottom:24px;">
      <h2 style="font-family:Syne,sans-serif;font-weight:800;font-size:24px;
                 color:#f5a623;margin:0 0 4px;">📊 Field Overview</h2>
      <div style="font-family:Space Mono,monospace;font-size:10px;color:#4a5270;
                  letter-spacing:1px;">MULTI-WELL PRODUCTION DASHBOARD</div>
    </div>""", unsafe_allow_html=True)

    sec_head("📡 Field Telemetry Input")
    st.caption("Enter current readings for each active well.")

    telem = {}
    active_wells = st.multiselect(
        "Active wells", list(st.session_state.active_well_configs.keys()),
        default=list(st.session_state.active_well_configs.keys())[:2]
    )

    well_inputs = {}
    for wn in active_wells:
        with st.expander(f"🔧 {wn}", expanded=(len(active_wells) == 1)):
            c1,c2,c3,c4,c5 = st.columns(5)
            well_inputs[wn] = {
                "p_res":   c1.number_input(f"P_res [psi]##{wn}", 1000, 8000, 4500, key=f"pr_{wn}"),
                "whp_psi": c2.number_input(f"WHP [psi]##{wn}", 14, 3000, 700, key=f"wp_{wn}"),
                "wht_k":   c3.number_input(f"WHT [°C]##{wn}", 0, 120, 75, key=f"wt_{wn}") + 273.15,
                "wc":      c4.number_input(f"WC##{wn}", 0.0, 0.99, 0.30, key=f"wc_{wn}", format="%.2f"),
                "gor":     c5.number_input(f"GOR [SCF/STB]##{wn}", 50, 5000, 846, key=f"go_{wn}"),
                "bhp_psi": c1.number_input(f"BHP [psi]##{wn}", 500, 6000, 3500, key=f"bp_{wn}"),
            }

    fiscal = st.number_input("Fiscal/export oil meter [STB/D]", 0, 200000, 0,
                              help="Leave 0 to skip reconciliation")

    if st.button("🔄 Run field VFM", type="primary"):
        for wn, inputs in well_inputs.items():
            telem[wn] = inputs
        vfm = get_vfm_service()
        with st.spinner("Running field VFM…"):
            field_out = vfm.predict_field(telem, fiscal_oil_stbd=fiscal if fiscal>0 else None)
        results = field_out["results"]
        summary = field_out["summary"]
        st.session_state.last_results = results

        # ── Field totals ──────────────────────────────────────
        st.divider()
        sec_head("🏭 Field Totals")
        fc1,fc2,fc3,fc4,fc5 = st.columns(5)
        with fc1: metric_card("Total Oil",   f"{summary['total_oil_stbd']:,.0f}",   "STB/D", "teal")
        with fc2: metric_card("Total Water", f"{summary['total_water_stbd']:,.0f}", "STB/D")
        with fc3: metric_card("Total Gas",   f"{summary['total_gas_mscfd']:,.0f}",  "MSCF/D", "gas")
        with fc4: metric_card("Field WC",    f"{summary['field_wc']:.3f}", "FRACTION")
        with fc5: metric_card("Recon Factor",f"{summary['reconciliation_factor']:.4f}")

        if summary["hydrate_alerts"]:
            st.error(f"🧊 Hydrate alerts: {', '.join(summary['hydrate_alerts'])}")
        if summary.get("wax_alerts"):
            st.warning(f"🕯️ Wax risk: {', '.join(summary['wax_alerts'])}")

        sec_head("📋 Well Allocation")
        rows = []
        for r in results:
            if "error" in r:
                rows.append({"Well": r["well"], "Error": r["error"]}); continue
            alloc = next((a for a in summary["well_allocation"] if a["well"]==r["well"]), {})
            ml = r.get("ml_uncertainty") or {}
            rows.append({
                "Well":          r["well"],
                "Oil [STB/D]":   f"{r['q_oil_stbd']:,.0f}",
                "Water [STB/D]": f"{r['q_water_stbd']:,.0f}",
                "Gas [MSCF/D]":  f"{r['q_gas_mscfd']:.1f}",
                "WC":            f"{r['wc']:.3f}",
                "GVF %":         f"{r['gvf_insitu']*100:.1f}",
                "Alloc Oil":     f"{alloc.get('q_oil_alloc',0):,.0f}",
                "Field %":       f"{alloc.get('contribution_pct',0):.1f}",
                "Hydrate":       r.get("hydrate_status","–"),
                "ML P50":        f"{ml.get('p50',0):,.0f}" if ml else "–",
                "Phys wt%":      f"{r.get('physics_weight_pct',100):.0f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB: BATCH RUN
# ════════════════════════════════════════════════════════════════
elif tab_choice == "📁  Batch Run":
    st.markdown("""
    <div style="margin-bottom:24px;">
      <h2 style="font-family:Syne,sans-serif;font-weight:800;font-size:24px;
                 color:#f5a623;margin:0 0 4px;">📁 Batch Run</h2>
      <div style="font-family:Space Mono,monospace;font-size:10px;color:#4a5270;
                  letter-spacing:1px;">BULK CSV PROCESSING · HISTORIAN EXPORT READY</div>
    </div>""", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1.6])

    with col_left:
        sec_head("Upload CSV")
        active_well_names = list(st.session_state.active_well_configs.keys())
        well_name_list    = ", ".join(active_well_names)

        st.markdown(f"""
<div style="background:var(--bg2);border:1px solid var(--bg3);border-radius:6px;
            padding:14px 16px;margin-bottom:16px;font-family:Space Mono,monospace;
            font-size:10px;color:#4a5270;line-height:2;">
  <div style="color:#8b95b0;margin-bottom:6px;letter-spacing:1px;">REQUIRED COLUMNS</div>
  <code style="color:#f5a623;">well · p_res · whp_psi · wht_k · wc · gor</code><br>
  <div style="margin-top:8px;color:#8b95b0;letter-spacing:1px;">OPTIONAL</div>
  <code style="color:#8b95b0;">bhp_psi · choke_size_64 · p_choke_dn</code><br>
  <div style="margin-top:8px;color:#8b95b0;letter-spacing:1px;">ACTIVE WELLS</div>
  <span style="color:#00e5c3;">{well_name_list}</span>
</div>""", unsafe_allow_html=True)

        first_well = active_well_names[0] if active_well_names else "Well-1"
        template = pd.DataFrame({
            "well": [first_well, first_well],
            "p_res": [4500, 4200], "whp_psi": [750, 600],
            "wht_k": [348, 345],   "wc": [0.30, 0.55],
            "gor": [846, 850],     "bhp_psi": [3600, 3400],
            "choke_size_64": [12, 10],
        })
        buf = io.BytesIO()
        template.to_csv(buf, index=False)
        st.download_button("📥 Download template CSV", buf.getvalue(),
                           "vfm_template.csv", "text/csv")

        csv_file = st.file_uploader("Upload readings CSV", type=["csv"])
        well_filter = st.multiselect("Filter wells (blank = all)", active_well_names)
        run_batch = st.button("▶  RUN BATCH", type="primary", disabled=(csv_file is None))

    with col_right:
        sec_head("Results")

        if run_batch and csv_file:
            df_in   = pd.read_csv(csv_file)
            required = {"well","p_res","whp_psi","wht_k","wc","gor"}
            missing  = required - set(df_in.columns)
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                if well_filter:
                    df_in = df_in[df_in["well"].isin(well_filter)]
                vfm       = get_vfm_service()
                rows      = df_in.to_dict("records")
                prog      = st.progress(0)
                processed = []
                for i, row in enumerate(rows):
                    result = vfm.predict_batch([row])[0]
                    if "error" not in result:
                        ml = result.get("ml_uncertainty") or {}
                        result.update({
                            "q_total_stbd":   round(result.get("q_total_stbd",0), 1),
                            "q_oil_stbd":     round(result.get("q_oil_stbd",0), 1),
                            "q_water_stbd":   round(result.get("q_water_stbd",0), 1),
                            "q_gas_mscfd":    round(result.get("q_gas_mscfd",0), 2),
                            "gvf_pct":        round(result.get("gvf_insitu",0)*100, 1),
                            "hydrate_status": result.get("hydrate_status",""),
                            "hydrate_dt_k":   result.get("hydrate_delta_t_k",""),
                            "meg_dose_pct":   result.get("meg_dose_vol_pct",""),
                            "ml_p10": ml.get("p10",""), "ml_p50": ml.get("p50",""),
                            "ml_p90": ml.get("p90",""),
                            "physics_wt_pct": result.get("physics_weight_pct",100),
                        })
                    processed.append(result)
                    prog.progress((i+1)/len(rows))

                good   = [r for r in processed if "error" not in r]
                errors = len(processed) - len(good)
                if good:
                    df_out = pd.DataFrame(processed)
                    st.session_state.batch_results = df_out
                    st.success(f"✅ {len(good)} records processed ({errors} errors/skipped)")
                    st.dataframe(df_out, use_container_width=True)
                    st.download_button("📥 Download results CSV",
                                       df_out.to_csv(index=False).encode(),
                                       "vfm_results.csv", "text/csv")
                else:
                    st.error("No records processed. Check well names and column format.")

        elif st.session_state.batch_results is not None:
            df_out = st.session_state.batch_results
            st.dataframe(df_out, use_container_width=True)
            st.download_button("📥 Download results CSV",
                               df_out.to_csv(index=False).encode(),
                               "vfm_results.csv", "text/csv")
        else:
            st.markdown("""
            <div style="background:var(--bg2);border:1px dashed var(--bg3);
                        border-radius:8px;padding:40px 24px;text-align:center;">
              <div style="font-family:Space Mono,monospace;font-size:24px;
                          color:#2a3050;margin-bottom:12px;">📁</div>
              <div style="font-family:Space Mono,monospace;font-size:11px;
                          color:#4a5270;letter-spacing:1px;">
                Upload a CSV and click RUN BATCH
              </div>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# TAB: ABOUT
# ════════════════════════════════════════════════════════════════
elif tab_choice == "ℹ️  About":
    st.markdown("""
    <div style="margin-bottom:24px;">
      <h2 style="font-family:Syne,sans-serif;font-weight:800;font-size:24px;
                 color:#f5a623;margin:0 0 4px;">ℹ️ About VFM Pro</h2>
      <div style="font-family:Space Mono,monospace;font-size:10px;color:#4a5270;
                  letter-spacing:1px;">METHODOLOGY · BENCHMARKS · LIMITATIONS</div>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])

    with c1:
        sec_head("What this tool does")
        st.markdown("""
Subsea VFM Pro estimates **oil, water, and gas flow rates** from subsea wells
without a physical multiphase flow meter (MPFM). It combines:

- **Beggs & Brill wellbore traverse** — BHP→WHP pressure drop
- **Vogel IPR model** — reservoir inflow performance
- **Gilbert + Sachdeva choke model** — independent second estimator
- **Bootstrap GBR ensemble (N=50)** — ML with P10/P50/P90 uncertainty
- **Sage-Husa Adaptive Kalman Filter** — sensor drift detection
- **Makogon hydrate model** — flowline risk + inhibitor dose
        """)

        sec_head("Validated Accuracy  —  Volve 2008–2016")
        acc_data = {
            "Well":    ["F-14 H", "F-11 H", "F-12 H", "F-15 D", "F-1 C", "Overall"],
            "Records": [2512, 1054, 863, 717, 399, 5545],
            "MAPE":    ["5.9%", "4.6%", "6.2%", "7.4%", "9.3%", "6.1%"],
            "P50":     ["3.3%", "2.8%", "3.8%", "4.1%", "5.6%", "3.5%"],
            "P90":     ["13.5%","9.3%","12.7%","15.2%","19.5%","—"],
        }
        st.dataframe(pd.DataFrame(acc_data), use_container_width=True, hide_index=True)
        st.markdown("""
        <div style="font-family:Space Mono,monospace;font-size:10px;color:#00e5c3;
                    margin-top:8px;letter-spacing:1px;">
          ✓ COMMERCIAL TARGET: &lt;10% MAPE
        </div>""", unsafe_allow_html=True)

    with c2:
        sec_head("Hybrid Weighting")
        st.markdown("""
**With real field data trained:**
- 75% ML / 25% physics — ML dominates
- Physics provides a sanity bound

**Physics-only mode:**
- 100% Beggs & Brill nodal analysis + Vogel IPR
- Accuracy degrades at WC > 50%
        """)

        sec_head("Known Limitations")
        st.markdown("""
- **No transient flow** — steady-state only
- **No manifold constraints** — single-well
- **Deviation supported** ✅ — TVD via min-curvature; penalty shown in diagnostics
- **Per-well training required** — cross-well MAPE 15–90%
- **Wax onset modelled** ✅ — Lindeloff WAT from API gravity
- **Physics uncertainty propagated** ✅ — WC-adaptive ±% bounds
- **Not for fiscal metering** — production monitoring only
        """)

        sec_head("Built by")
        st.markdown("""
        <div style="background:var(--bg2);border:1px solid var(--bg3);
                    border-radius:6px;padding:16px 18px;">
          <div style="font-family:Syne,sans-serif;font-size:16px;font-weight:800;
                      color:#f5a623;">Edward Aryee</div>
          <div style="font-family:Space Mono,monospace;font-size:10px;color:#4a5270;
                      letter-spacing:1px;margin-top:4px;line-height:1.8;">
            PROCESS ENGINEER · ZOIL SERVICES LIMITED<br>
            KNUST PETROLEUM ENGINEERING · FIRST CLASS · 2025
          </div>
        </div>""", unsafe_allow_html=True)
