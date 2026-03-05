"""
Semantix — Enterprise Video Intelligence Platform
app.py  — Main Streamlit application

Pages:
  1. 🎬 Process Video     Upload SRT/VTT or YouTube URL
  2. 🔍 Search            Hybrid semantic + BM25 search
  3. 📺 Scene Explorer    Per-scene analysis browser
  4. 📢 Ad Engine         Ad matching and placement planning
  5. 📊 Analytics         Performance dashboard
  6. 🎯 Franchise Intel   Cross-video theme tracking
"""

import io
import json
import re
import time
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Core imports ──────────────────────────────────────────────────────────────
from core.video_processor import VideoProcessor, VideoMetadata, fetch_youtube_transcript, fetch_youtube_metadata
from core.scene_detector import Scene
from core.ad_engine import AdMatchingEngine, create_default_inventory
from core.search_engine import HybridSearchEngine
from core.embeddings import _IAB_NAMES

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Semantix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Professional dark UI with amber accents
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ─── Reset & Base ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #0C0D10 !important;
    color: #D4D8E1 !important;
}

/* Hide streamlit default chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ─── Main Layout ───────────────────────────────────────────────────────── */
[data-testid="stAppViewContainer"] {
    background: #0C0D10 !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: #0C0D10 !important;
}
[data-testid="block-container"] {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1380px !important;
}

/* ─── Sidebar ───────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #0F1014 !important;
    border-right: 1px solid #1E2028 !important;
}
[data-testid="stSidebar"] > div:first-child {
    background: #0F1014 !important;
    padding: 1.5rem 1rem !important;
}
[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown p {
    color: #6B7280 !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* Sidebar selectbox */
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: #16181D !important;
    border: 1px solid #252830 !important;
    border-radius: 8px !important;
    color: #D4D8E1 !important;
    font-size: 0.85rem !important;
    font-weight: 400 !important;
}

/* Sidebar divider */
[data-testid="stSidebar"] hr {
    border-color: #1E2028 !important;
    margin: 1rem 0 !important;
}

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    background: transparent !important;
    color: #8B909E !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.55rem 0.9rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    transition: all 0.15s ease !important;
    margin-bottom: 2px !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #16181D !important;
    color: #F0F2F5 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ─── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background: #F59E0B !important;
    color: #0C0D10 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    letter-spacing: 0.01em !important;
    padding: 0.55rem 1.25rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
.stButton > button:hover {
    background: #FBBF24 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(245,158,11,0.25) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ─── Inputs ────────────────────────────────────────────────────────────── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
.stTextInput input,
.stTextArea textarea {
    background: #16181D !important;
    border: 1px solid #252830 !important;
    border-radius: 8px !important;
    color: #D4D8E1 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.15s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #F59E0B !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.12) !important;
    outline: none !important;
}
[data-testid="stTextInput"] input::placeholder,
[data-testid="stTextArea"] textarea::placeholder {
    color: #4B5260 !important;
}

/* Input labels */
.stTextInput label,
.stTextArea label,
.stSelectbox label,
.stMultiSelect label,
.stSlider label,
.stCheckbox label,
.stFileUploader label,
label {
    color: #8B909E !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    margin-bottom: 0.35rem !important;
}

/* ─── File Uploader ─────────────────────────────────────────────────────── */
[data-testid="stFileUploader"] {
    background: #16181D !important;
    border: 1.5px dashed #2A2D38 !important;
    border-radius: 10px !important;
    transition: border-color 0.15s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #F59E0B !important;
}
[data-testid="stFileUploader"] * {
    color: #8B909E !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] p {
    font-size: 0.85rem !important;
}

/* ─── Selectbox ─────────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background: #16181D !important;
    border: 1px solid #252830 !important;
    border-radius: 8px !important;
    color: #D4D8E1 !important;
    font-size: 0.875rem !important;
}

/* ─── Multiselect ───────────────────────────────────────────────────────── */
[data-testid="stMultiSelect"] > div > div {
    background: #16181D !important;
    border: 1px solid #252830 !important;
    border-radius: 8px !important;
    min-height: 42px !important;
}
[data-testid="stMultiSelect"] span[data-baseweb="tag"] {
    background: rgba(245,158,11,0.15) !important;
    color: #F59E0B !important;
    border: 1px solid rgba(245,158,11,0.25) !important;
    border-radius: 5px !important;
    font-size: 0.78rem !important;
}

/* ─── Sliders ───────────────────────────────────────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background: #F59E0B !important;
}
[data-testid="stSlider"] [data-baseweb="slider"] > div:first-child {
    background: #252830 !important;
}

/* ─── Tabs ──────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #16181D !important;
    border: 1px solid #1E2028 !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #6B7280 !important;
    border-radius: 7px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 6px 16px !important;
    border: none !important;
    transition: all 0.15s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
    color: #D4D8E1 !important;
    background: rgba(255,255,255,0.04) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #F59E0B !important;
    color: #0C0D10 !important;
    font-weight: 600 !important;
}

/* ─── Metrics ───────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
    background: #13151A !important;
    border: 1px solid #1E2028 !important;
    border-radius: 10px !important;
    padding: 1rem 1.25rem !important;
}
[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #F0F2F5 !important;
    font-size: 1.5rem !important;
    font-weight: 500 !important;
}
[data-testid="stMetricDelta"] {
    font-size: 0.75rem !important;
    font-family: 'DM Mono', monospace !important;
}

/* ─── DataFrames ────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid #1E2028 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}
[data-testid="stDataFrame"] thead th {
    background: #16181D !important;
    color: #6B7280 !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    border-bottom: 1px solid #1E2028 !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(even) {
    background: rgba(255,255,255,0.015) !important;
}
[data-testid="stDataFrame"] tbody td {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.82rem !important;
    color: #C0C5D0 !important;
    border-color: #1A1C22 !important;
}

/* ─── Expanders ─────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: #13151A !important;
    border: 1px solid #1E2028 !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #D4D8E1 !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
}
[data-testid="stExpander"] summary:hover {
    color: #F0F2F5 !important;
}

/* ─── Alerts ────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: none !important;
    font-size: 0.875rem !important;
}

/* ─── Progress ──────────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #F59E0B, #FBBF24) !important;
}

/* ─── Spinner ───────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] > div {
    border-top-color: #F59E0B !important;
}

/* ─── Divider ───────────────────────────────────────────────────────────── */
hr {
    border-color: #1E2028 !important;
    margin: 1.25rem 0 !important;
}

/* ─── Plotly ────────────────────────────────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    border: 1px solid #1E2028 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    background: #13151A !important;
}

/* ─── Checkbox ──────────────────────────────────────────────────────────── */
[data-testid="stCheckbox"] > label {
    color: #8B909E !important;
    font-size: 0.85rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 400 !important;
}

/* ═══════════════════════════════════════════════════════════════════════════
   CUSTOM COMPONENT CLASSES
   ═══════════════════════════════════════════════════════════════════════════ */

/* Page header */
.page-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 2rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #1E2028;
}
.page-header-icon {
    width: 44px;
    height: 44px;
    background: rgba(245,158,11,0.12);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    flex-shrink: 0;
}
.page-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F0F2F5;
    letter-spacing: -0.02em;
    line-height: 1.2;
    margin: 0;
}
.page-subtitle {
    font-size: 0.82rem;
    color: #6B7280;
    margin-top: 2px;
    font-weight: 400;
}

/* Logo */
.sidebar-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.5rem;
    padding: 0.5rem 0;
}
.sidebar-logo-mark {
    font-family: 'DM Sans', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: #F59E0B;
    letter-spacing: -0.01em;
}
.sidebar-logo-sub {
    font-size: 0.65rem;
    color: #4B5260;
    font-weight: 400;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

/* Nav item active state */
.nav-active > button {
    background: rgba(245,158,11,0.1) !important;
    color: #F59E0B !important;
}

/* Cards */
.card {
    background: #13151A;
    border: 1px solid #1E2028;
    border-radius: 10px;
    padding: 1.125rem 1.25rem;
    margin-bottom: 0.625rem;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.card:hover {
    border-color: #2A2D38;
    box-shadow: 0 2px 12px rgba(0,0,0,0.2);
}
.card.card-highlight {
    border-color: rgba(245,158,11,0.3);
    background: rgba(245,158,11,0.04);
}

/* Tags / badges */
.badge {
    display: inline-flex;
    align-items: center;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin: 2px 2px 2px 0;
    white-space: nowrap;
}
.badge-amber  { background: rgba(245,158,11,0.12); color: #F59E0B; border: 1px solid rgba(245,158,11,0.2); }
.badge-green  { background: rgba(52,211,153,0.12);  color: #34D399; border: 1px solid rgba(52,211,153,0.2); }
.badge-red    { background: rgba(239,68,68,0.12);   color: #F87171; border: 1px solid rgba(239,68,68,0.2); }
.badge-blue   { background: rgba(96,165,250,0.12);  color: #60A5FA; border: 1px solid rgba(96,165,250,0.2); }
.badge-purple { background: rgba(167,139,250,0.12); color: #A78BFA; border: 1px solid rgba(167,139,250,0.2); }
.badge-gray   { background: rgba(255,255,255,0.06); color: #8B909E; border: 1px solid rgba(255,255,255,0.08); }

/* Stat pill */
.stat-row {
    display: flex;
    align-items: center;
    gap: 6px;
    flex-wrap: wrap;
}
.stat-item {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #6B7280;
}
.stat-item strong { color: #D4D8E1; font-weight: 500; }

/* Score bar */
.score-bar-wrap {
    height: 3px;
    background: rgba(255,255,255,0.06);
    border-radius: 2px;
    margin-top: 8px;
    overflow: hidden;
}
.score-bar-fill {
    height: 3px;
    border-radius: 2px;
    background: linear-gradient(90deg, #F59E0B, #FBBF24);
}

/* Section heading */
.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: #4B5260;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    margin-top: 0.25rem;
}

/* Empty state */
.empty-state {
    text-align: center;
    padding: 3.5rem 2rem;
    color: #4B5260;
    border: 1.5px dashed #1E2028;
    border-radius: 12px;
    background: #13151A;
}
.empty-state-icon { font-size: 2.5rem; margin-bottom: 1rem; opacity: 0.6; }
.empty-state-text { font-size: 0.9rem; line-height: 1.6; }

/* Mono value */
.mono { font-family: 'DM Mono', monospace; }

/* Timeline dot */
.timeline-time {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #F59E0B;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
def _init():
    defaults = {
        "videos":        {},       # video_id → VideoMetadata
        "search_engine": HybridSearchEngine(),
        "ad_engine":     AdMatchingEngine(),
        "page":          "process",
        "selected_video": None,
        "yt_api_key":    "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY THEME
# ══════════════════════════════════════════════════════════════════════════════
PLOTLY_THEME = dict(
    plot_bgcolor  = "#13151A",
    paper_bgcolor = "#13151A",
    font          = dict(family="DM Mono, monospace", color="#8B909E", size=11),
    margin        = dict(l=16, r=16, t=44, b=16),
    colorway      = ["#F59E0B", "#34D399", "#60A5FA", "#A78BFA",
                     "#FB923C", "#F472B6", "#38BDF8", "#4ADE80"],
    xaxis         = dict(gridcolor="#1E2028", linecolor="#252830",
                         tickfont=dict(color="#6B7280", size=10)),
    yaxis         = dict(gridcolor="#1E2028", linecolor="#252830",
                         tickfont=dict(color="#6B7280", size=10)),
    hoverlabel    = dict(bgcolor="#1E2028", bordercolor="#2A2D38",
                         font=dict(family="DM Mono", size=11, color="#D4D8E1")),
)


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
NAV_PAGES = [
    ("process",   "🎬", "Process Video"),
    ("search",    "🔍", "Semantic Search"),
    ("scenes",    "📺", "Scene Explorer"),
    ("ads",       "📢", "Ad Engine"),
    ("analytics", "📊", "Analytics"),
    ("franchise", "🎯", "Franchise Intel"),
]

with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div>
            <div class="sidebar-logo-mark">⚡ SEMANTIX</div>
            <div class="sidebar-logo-sub">Video Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Navigation</div>', unsafe_allow_html=True)

    for page_id, icon, label in NAV_PAGES:
        active = st.session_state.page == page_id
        container = st.container()
        if active:
            container.markdown('<div class="nav-active">', unsafe_allow_html=True)
        if container.button(f"{icon}  {label}", key=f"nav_{page_id}", use_container_width=True):
            st.session_state.page = page_id
            st.rerun()
        if active:
            container.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Video selector
    if st.session_state.videos:
        st.markdown('<div class="section-title">Active Video</div>', unsafe_allow_html=True)
        video_options = ["— All Videos —"] + [
            f"{v.title[:28]}…" if len(v.title) > 28 else v.title
            for v in st.session_state.videos.values()
        ]
        video_ids = [None] + list(st.session_state.videos.keys())
        sel = st.selectbox("Active video", video_options, key="video_selector",
                           label_visibility="collapsed")
        sel_idx = video_options.index(sel)
        st.session_state.selected_video = video_ids[sel_idx]

    # Stats
    se_stats = st.session_state.search_engine.stats
    if se_stats["total_scenes"] > 0:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Index Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style='display:flex;flex-direction:column;gap:6px'>
            <div class="stat-item">🎬 <strong>{se_stats['total_videos']}</strong> videos</div>
            <div class="stat-item">🎞 <strong>{se_stats['total_scenes']}</strong> scenes</div>
            <div class="stat-item">📚 <strong>{se_stats['vocab_size']:,}</strong> vocab terms</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">YouTube API Key</div>', unsafe_allow_html=True)
    yt_key = st.text_input("YouTube API Key", type="password",
                            value=st.session_state.yt_api_key,
                            key="yt_key_input", label_visibility="collapsed",
                            placeholder="Optional — for metadata fetch")
    if yt_key != st.session_state.yt_api_key:
        st.session_state.yt_api_key = yt_key


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _register_video(vm: VideoMetadata):
    """Add video to session state, index scenes, sync ad engine vocab."""
    st.session_state.videos[vm.video_id] = vm
    st.session_state.search_engine.add_scenes(vm.scenes)
    if st.session_state.search_engine.vectorizer is not None:
        st.session_state.ad_engine.sync_vectorizer(
            st.session_state.search_engine.vectorizer
        )
    st.session_state.selected_video = vm.video_id


def _sentiment_color(label: str) -> str:
    return {"positive": "#34D399", "negative": "#F87171", "neutral": "#6B7280"}.get(label, "#6B7280")


def _safety_color(score: float) -> str:
    if score >= 0.8: return "#34D399"
    if score >= 0.5: return "#F59E0B"
    return "#F87171"


def _score_bar(score: float, color: str = "#F59E0B") -> str:
    pct = int(score * 100)
    return f"""<div class="score-bar-wrap"><div class="score-bar-fill" style="width:{pct}%;background:{color}"></div></div>"""


def _iab_tags(iab_list: list[dict], max_n: int = 3) -> str:
    colors = ["badge-amber", "badge-blue", "badge-purple"]
    tags = ""
    for i, cat in enumerate(iab_list[:max_n]):
        cls = colors[i % len(colors)]
        tags += f'<span class="badge {cls}">{cat["name"]}</span>'
    return tags


def _page_header(icon: str, title: str, subtitle: str):
    st.markdown(f"""
    <div class="page-header">
        <div class="page-header-icon">{icon}</div>
        <div>
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def _render_scene_card(scene: Scene, is_key: bool = False):
    safety = scene.brand_safety.get("safety_score", 1.0)
    sentiment = scene.sentiment
    highlight = "card-highlight" if is_key else ""
    key_badge = '<span class="badge badge-amber">★ Key Scene</span>' if is_key else ""

    iab_html = _iab_tags(scene.iab_categories)
    safety_color = _safety_color(safety)
    sent_color = _sentiment_color(sentiment.get("label", "neutral"))
    sent_label = sentiment.get("label", "neutral").title()

    st.markdown(f"""
    <div class="card {highlight}">
        <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
            <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                <span class="timeline-time">{scene.start_fmt} → {scene.end_fmt}</span>
                <span class="stat-item">{scene.duration_sec:.0f}s · {len(scene.text.split())} words</span>
                {key_badge}
            </div>
            <div style="display:flex;align-items:center;gap:10px">
                <span style="font-family:'DM Mono',monospace;font-size:0.72rem;color:{safety_color}">🛡 {safety:.0%}</span>
                <span style="font-size:0.72rem;color:{sent_color}">● {sent_label}</span>
            </div>
        </div>
        <p style="margin:0 0 10px;font-size:0.875rem;color:#C0C5D0;line-height:1.6">
            {scene.text[:220]}{"…" if len(scene.text) > 220 else ""}
        </p>
        <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
            {iab_html}
            <span class="stat-item" style="margin-left:4px">eng <strong>{scene.engagement_score:.2f}</strong></span>
            <span class="stat-item">ad <strong>{scene.ad_suitability:.2f}</strong></span>
        </div>
        {_score_bar(scene.ad_suitability)}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PROCESS VIDEO
# ══════════════════════════════════════════════════════════════════════════════
def page_process():
    _page_header("🎬", "Process Video", "Upload subtitles or fetch from YouTube to extract semantic scenes")

    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "▶ YouTube URL", "📋 Paste Text"])

    # ── Tab 1: File Upload ────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            uploaded = st.file_uploader(
                "Subtitle File",
                type=["srt", "vtt"],
                help="Supports SRT and WebVTT formats",
            )
            video_title = st.text_input("Video Title", placeholder="Leave blank to use filename")

        with col2:
            st.markdown('<div class="section-title">Detection Settings</div>', unsafe_allow_html=True)
            min_scene = st.slider("Min scene length (s)", 10, 60, 20)
            max_scene = st.slider("Max scene length (s)", 60, 300, 120)
            threshold = st.slider("Semantic sensitivity", 0.2, 0.7, 0.35, 0.05,
                                  help="Higher = fewer but more distinct scenes")

        if uploaded:
            if st.button("Process Subtitles →", use_container_width=False):
                content = uploaded.read().decode("utf-8", errors="replace")
                title = video_title or uploaded.name
                fmt = "vtt" if uploaded.name.lower().endswith(".vtt") else "srt"

                with st.spinner("Parsing subtitles and detecting scenes…"):
                    t0 = time.time()
                    processor = VideoProcessor(min_scene, max_scene, threshold)
                    vm = processor.process_file(content, title, fmt)
                    elapsed = time.time() - t0

                if not vm.scenes:
                    st.error("No scenes detected. Check your file format.")
                    return

                _register_video(vm)
                st.success(f"✓ Processed {len(vm.scenes)} scenes in {elapsed:.2f}s")
                _show_processing_summary(vm)
        else:
            st.markdown("""
            <div class="empty-state">
                <div class="empty-state-icon">📁</div>
                <div class="empty-state-text">Drop an SRT or VTT subtitle file above to get started</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tab 2: YouTube URL ────────────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            yt_url = st.text_input("YouTube URL", placeholder="https://youtube.com/watch?v=... or video ID")
        with col2:
            st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
            min_s2 = st.slider("Min scene (s)", 10, 60, 20, key="yt_min")
            max_s2 = st.slider("Max scene (s)", 60, 300, 120, key="yt_max")
            thr2   = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="yt_thr")

        if yt_url:
            if st.button("Fetch & Process →", use_container_width=False):
                vid_id = _extract_yt_id(yt_url)
                if not vid_id:
                    st.error("Could not parse YouTube video ID from URL.")
                    return

                with st.spinner("Fetching YouTube transcript…"):
                    transcript = fetch_youtube_transcript(vid_id)

                if transcript is None:
                    st.error("Could not fetch transcript. The video may not have captions enabled.")
                    return

                meta = None
                if st.session_state.yt_api_key:
                    with st.spinner("Fetching metadata…"):
                        meta = fetch_youtube_metadata(vid_id, st.session_state.yt_api_key)

                title = meta.get("title", f"YouTube: {vid_id}") if meta else f"YouTube: {vid_id}"

                with st.spinner("Detecting scenes…"):
                    t0 = time.time()
                    processor = VideoProcessor(min_s2, max_s2, thr2)
                    vm = processor.process_youtube_transcript(transcript, vid_id, title, meta)
                    elapsed = time.time() - t0

                _register_video(vm)
                st.success(f"✓ Processed {len(vm.scenes)} scenes in {elapsed:.2f}s")
                _show_processing_summary(vm)

    # ── Tab 3: Paste Text ─────────────────────────────────────────────────────
    with tab3:
        col1, col2 = st.columns([3, 2], gap="large")
        with col1:
            paste_title = st.text_input("Video Title", placeholder="My Video", key="paste_title")
            pasted = st.text_area(
                "Paste SRT or VTT content",
                height=220,
                placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello, welcome to our show...",
            )
        with col2:
            st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
            min_s3 = st.slider("Min scene (s)", 10, 60, 20, key="p_min")
            max_s3 = st.slider("Max scene (s)", 60, 300, 120, key="p_max")

        if pasted:
            if st.button("Process Text →", use_container_width=False):
                title = paste_title or "Pasted Subtitles"
                with st.spinner("Processing…"):
                    t0 = time.time()
                    processor = VideoProcessor(min_s3, max_s3)
                    vm = processor.process_file(pasted, title)
                    elapsed = time.time() - t0

                if not vm.scenes:
                    st.error("No scenes detected. Check your subtitle format.")
                    return

                _register_video(vm)
                st.success(f"✓ {len(vm.scenes)} scenes in {elapsed:.2f}s")
                _show_processing_summary(vm)


def _extract_yt_id(url: str) -> Optional[str]:
    patterns = [
        r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def _show_processing_summary(vm: VideoMetadata):
    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenes Detected", vm.scene_count)
    c2.metric("Duration", vm.fmt_duration())
    c3.metric("Total Cues", vm.total_cues)
    avg_dur = round(sum(s.duration_sec for s in vm.scenes) / max(vm.scene_count, 1), 1)
    c4.metric("Avg Scene", f"{avg_dur}s")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-title" style="margin-top:1rem">Narrative Structure</div>', unsafe_allow_html=True)
        st.markdown(f'<span class="badge badge-amber">{vm.narrative_structure}</span>', unsafe_allow_html=True)
        st.markdown('<div class="section-title" style="margin-top:1rem">Dominant Topics</div>', unsafe_allow_html=True)
        tags_html = "".join(f'<span class="badge badge-blue" style="margin-right:4px">{cat["name"]}</span>' for cat in vm.dominant_iab[:4])
        st.markdown(tags_html, unsafe_allow_html=True)

    with col2:
        if vm.emotional_arc:
            arc_df = pd.DataFrame(vm.emotional_arc)
            fig = px.area(arc_df, x="start_sec", y="sentiment_score",
                          color_discrete_sequence=["#F59E0B"])
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
            fig.update_layout(**PLOTLY_THEME, height=180, showlegend=False,
                              title=dict(text="Emotional Arc", font=dict(size=12, color="#8B909E")),
                              xaxis_title="Time (s)", yaxis_title="Sentiment")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════
def page_search():
    _page_header("🔍", "Semantic Search", "Natural language discovery across all indexed scenes")

    se = st.session_state.search_engine
    if se.stats["total_scenes"] == 0:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🔍</div>
            <div class="empty-state-text">No videos indexed yet. Process a video first to start searching.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    # Search bar row
    col1, col2, col3 = st.columns([4, 1, 1], gap="small")
    with col1:
        query = st.text_input("Search query", placeholder="e.g. emotional confrontation between characters…",
                              label_visibility="collapsed")
    with col2:
        top_k = st.selectbox("Results", [5, 10, 20], index=1, label_visibility="collapsed")
    with col3:
        min_safety_label = st.selectbox("Safety", ["Any", "Safe 0.5+", "Brand Safe 0.8+"],
                                        label_visibility="collapsed")
        safety_map = {"Any": 0.0, "Safe 0.5+": 0.5, "Brand Safe 0.8+": 0.8}
        min_safety_val = safety_map[min_safety_label]

    # Filters row
    col4, col5 = st.columns([3, 1], gap="small")
    with col4:
        iab_choices = ["— All —"] + [f"{k}: {v}" for k, v in _IAB_NAMES.items()]
        iab_sel = st.multiselect("IAB filter", iab_choices, default=[],
                                  label_visibility="collapsed",
                                  placeholder="Filter by content category…")
        iab_filter = [s.split(":")[0] for s in iab_sel if s != "— All —"] or None
    with col5:
        diversify = st.checkbox("Diversify results", value=True)

    if st.session_state.videos and len(st.session_state.videos) > 1:
        vid_choices = ["All Videos"] + [vm.title for vm in st.session_state.videos.values()]
        vid_ids = [None] + list(st.session_state.videos.keys())
        vid_sel = st.selectbox("Filter by video", vid_choices, label_visibility="collapsed")
        vid_filter = vid_ids[vid_choices.index(vid_sel)]
    else:
        vid_filter = None

    if query:
        with st.spinner("Searching…"):
            results = se.search(
                query, top_k=top_k, diversify=diversify,
                video_id=vid_filter, min_safety=min_safety_val,
                iab_filter=iab_filter, expand=True,
            )

        if not results:
            st.warning("No results found. Try a different query or adjust filters.")
            return

        st.markdown(f'<div class="section-title" style="margin-bottom:12px">Found <span style="color:#F59E0B">{len(results)}</span> scenes</div>', unsafe_allow_html=True)

        for r in results:
            scene = r.scene
            vm = st.session_state.videos.get(scene.video_id)
            video_title = vm.title if vm else scene.video_id
            is_key = vm and scene.scene_id in vm.key_scenes if vm else False

            safety = scene.brand_safety.get("safety_score", 1.0)
            iab_html = _iab_tags(scene.iab_categories)
            sentiment = scene.sentiment
            sent_color = _sentiment_color(sentiment.get("label", "neutral"))
            safety_color = _safety_color(safety)
            key_badge = '<span class="badge badge-amber">★ Key</span>' if is_key else ""

            st.markdown(f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                    <div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap">
                        <span style="font-family:'DM Mono',monospace;color:#F59E0B;font-weight:600;font-size:0.78rem">#{r.rank}</span>
                        <span style="color:#6B7280;font-size:0.78rem">{video_title}</span>
                        {key_badge}
                    </div>
                    <span style="font-family:'DM Mono',monospace;color:#F59E0B;font-size:0.78rem;font-weight:600">{r.score:.3f}</span>
                </div>
                <div style="margin-bottom:6px">
                    <span class="timeline-time">⏱ {scene.start_fmt} → {scene.end_fmt}</span>
                    <span class="stat-item" style="margin-left:10px">{scene.duration_sec:.0f}s</span>
                </div>
                <p style="color:#C0C5D0;font-size:0.875rem;line-height:1.6;margin:0 0 10px">
                    {scene.text[:300]}{"…" if len(scene.text) > 300 else ""}
                </p>
                <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">
                    {iab_html}
                    <span style="color:{sent_color};font-size:0.7rem">● {sentiment.get("label","neutral").title()}</span>
                    <span style="color:{safety_color};font-size:0.7rem">🛡 {safety:.0%}</span>
                    <span class="stat-item" style="margin-left:4px">vec <strong>{r.vector_score:.2f}</strong></span>
                    <span class="stat-item">bm25 <strong>{r.bm25_score:.2f}</strong></span>
                </div>
                {_score_bar(r.score)}
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown('<div class="section-title">Suggested Queries</div>', unsafe_allow_html=True)
        suggestions = [
            "exciting action sequence", "emotional dialogue",
            "product demonstration", "travel destination",
            "health and wellness advice", "financial discussion",
            "comedy moment", "suspenseful climax",
        ]
        cols = st.columns(4)
        for i, sug in enumerate(suggestions):
            if cols[i % 4].button(sug, key=f"sug_{i}"):
                st.session_state["_search_q"] = sug
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SCENE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def page_scenes():
    _page_header("📺", "Scene Explorer", "Browse and analyse every detected scene")

    if not st.session_state.videos:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📺</div>
            <div class="empty-state-text">No videos processed yet. Go to Process Video to get started.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    vm = _get_active_video()
    if not vm:
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scenes", vm.scene_count)
    c2.metric("Duration", vm.fmt_duration())
    avg_eng = round(sum(s.engagement_score for s in vm.scenes) / max(vm.scene_count, 1), 2)
    c3.metric("Avg Engagement", avg_eng)
    avg_safe = round(sum(s.brand_safety.get("safety_score", 1.0) for s in vm.scenes) / max(vm.scene_count, 1), 2)
    c4.metric("Avg Safety", f"{avg_safe:.0%}")
    c5.metric("Narrative", vm.narrative_structure.split("(")[0].strip())

    st.markdown("<hr>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🎭 Emotional Arc", "📊 Analysis", "🗂 Scene List"])

    with tab1:
        if vm.emotional_arc:
            df = pd.DataFrame(vm.emotional_arc)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["start_sec"], y=df["sentiment_score"],
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                line=dict(color="#F59E0B", width=2),
                mode="lines+markers", marker=dict(size=5, color="#F59E0B"),
                name="Sentiment",
                hovertemplate="<b>%{x:.0f}s</b><br>Sentiment: %{y:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=df["start_sec"], y=df["engagement"],
                line=dict(color="#34D399", width=2, dash="dot"),
                mode="lines", name="Engagement",
                hovertemplate="<b>%{x:.0f}s</b><br>Engagement: %{y:.3f}<extra></extra>",
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.1)")

            for kid in vm.key_scenes:
                ks = next((s for s in vm.scenes if s.scene_id == kid), None)
                if ks:
                    fig.add_vline(x=ks.start_sec, line_dash="dash",
                                  line_color="rgba(245,158,11,0.3)")

            fig.update_layout(**PLOTLY_THEME, height=320, showlegend=True,
                              title=dict(text="Sentiment & Engagement Over Time",
                                         font=dict(size=12, color="#8B909E")),
                              xaxis_title="Time (seconds)",
                              yaxis_title="Score",
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6B7280")))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emotional arc data available.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            iab_counts: dict[str, int] = {}
            for s in vm.scenes:
                for cat in s.iab_categories[:1]:
                    iab_counts[cat["name"]] = iab_counts.get(cat["name"], 0) + 1
            if iab_counts:
                df_iab = pd.DataFrame(
                    sorted(iab_counts.items(), key=lambda x: x[1], reverse=True)[:8],
                    columns=["Category", "Scenes"]
                )
                fig = px.bar(df_iab, x="Scenes", y="Category", orientation="h",
                             color_discrete_sequence=["#F59E0B"])
                fig.update_layout(**PLOTLY_THEME, height=300,
                                  title=dict(text="Top IAB Categories", font=dict(size=12, color="#8B909E")),
                                  yaxis=dict(autorange="reversed", **PLOTLY_THEME["yaxis"]))
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            safety_scores = [s.brand_safety.get("safety_score", 1.0) for s in vm.scenes]
            fig = px.histogram(safety_scores, nbins=10,
                               color_discrete_sequence=["#34D399"],
                               labels={"value": "Safety Score", "count": "Scenes"})
            fig.update_layout(**PLOTLY_THEME, height=300,
                              title=dict(text="Brand Safety Distribution", font=dict(size=12, color="#8B909E")),
                              showlegend=False, xaxis_title="Safety Score", yaxis_title="Scene Count")
            st.plotly_chart(fig, use_container_width=True)

        eng_scores = [s.engagement_score for s in vm.scenes]
        suit_scores = [s.ad_suitability for s in vm.scenes]
        times = [s.start_sec for s in vm.scenes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=eng_scores,
            mode="markers", marker=dict(
                size=[s * 20 + 4 for s in suit_scores],
                color=eng_scores, colorscale=[[0, "#1E2028"], [1, "#F59E0B"]],
                showscale=True, colorbar=dict(title="Engagement", tickfont=dict(color="#6B7280")),
            ),
            hovertemplate="<b>%{x:.0f}s</b><br>Engagement: %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_THEME, height=280,
                          title=dict(text="Scene Engagement Map — bubble size = ad suitability",
                                     font=dict(size=12, color="#8B909E")),
                          xaxis_title="Time (s)", yaxis_title="Engagement")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sent_f = st.multiselect("Sentiment", ["positive", "neutral", "negative"],
                                    default=["positive", "neutral", "negative"])
        with fc2:
            min_eng = st.slider("Min engagement", 0.0, 1.0, 0.0, 0.05)
        with fc3:
            min_safe = st.slider("Min safety", 0.0, 1.0, 0.0, 0.05)

        filtered = [
            s for s in vm.scenes
            if s.sentiment.get("label", "neutral") in sent_f
            and s.engagement_score >= min_eng
            and s.brand_safety.get("safety_score", 1.0) >= min_safe
        ]

        st.markdown(f'<div class="section-title">{len(filtered)} scenes</div>', unsafe_allow_html=True)

        for scene in filtered:
            is_key = scene.scene_id in vm.key_scenes
            _render_scene_card(scene, is_key)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: AD ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def page_ads():
    _page_header("📢", "Ad Engine", "Contextual ad matching and placement optimisation")

    if not st.session_state.videos:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📢</div>
            <div class="empty-state-text">Process a video first to run the Ad Engine.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    vm = _get_active_video()
    if not vm:
        return

    ae = st.session_state.ad_engine

    tab1, tab2, tab3 = st.tabs(["🎯 Placement Plan", "🔍 Scene Matching", "📦 Inventory"])

    with tab1:
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:
            st.markdown('<div class="section-title">Configuration</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                placement_types = st.multiselect(
                    "Placement types",
                    ["pre-roll", "mid-roll", "post-roll"],
                    default=["pre-roll", "mid-roll", "post-roll"],
                )
                min_safety_ads = st.slider("Min brand safety", 0.0, 1.0, 0.5, 0.05)
            with c2:
                st.markdown(f"""
                <div class="card" style="margin-top:0">
                    <div class="stat-item" style="margin-bottom:6px">⏱ Min ad gap: <strong>3 min</strong></div>
                    <div class="stat-item" style="margin-bottom:6px">📊 Max density: <strong>1 / 5 min</strong></div>
                    <div class="stat-item" style="margin-bottom:6px">🎬 Scenes: <strong style="color:#F59E0B">{vm.scene_count}</strong></div>
                    <div class="stat-item">⏱ Duration: <strong>{vm.fmt_duration()}</strong></div>
                </div>
                """, unsafe_allow_html=True)

        if st.button("Generate Placement Plan →"):
            with st.spinner("Optimising placements…"):
                placements = ae.plan_placements(vm.scenes, vm.duration_ms, placement_types)
                perf = ae.simulate_performance(placements)
                st.session_state["_placements"] = placements
                st.session_state["_perf"] = perf

        if "_placements" in st.session_state and st.session_state["_placements"]:
            placements = st.session_state["_placements"]
            perf = st.session_state["_perf"]

            st.markdown("<hr>", unsafe_allow_html=True)
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Placements", perf["total_placements"])
            c2.metric("Est. Revenue", f"${perf['total_revenue_usd']:.2f}")
            c3.metric("Est. Impressions", f"{perf['total_impressions']:,}")
            c4.metric("Est. Clicks", f"{perf['estimated_clicks']:,}")
            c5.metric("Avg CPM", f"${perf['avg_cpm']:.2f}")

            p_df = pd.DataFrame([p.to_dict() for p in placements])
            type_colors = {"pre-roll": "#F59E0B", "mid-roll": "#34D399", "post-roll": "#60A5FA"}

            fig = go.Figure()
            for p_type in p_df["placement_type"].unique():
                sub = p_df[p_df["placement_type"] == p_type]
                fig.add_trace(go.Scatter(
                    x=sub["timestamp_ms"] / 1000,
                    y=sub["relevance_score"],
                    mode="markers+text",
                    name=p_type,
                    text=sub["brand"].str[:12],
                    textposition="top center",
                    textfont=dict(size=9, color="#8B909E"),
                    marker=dict(
                        size=sub["estimated_cpm"] * 2 + 10,
                        color=type_colors.get(p_type, "#F59E0B"),
                        opacity=0.8,
                    ),
                    hovertemplate=(
                        f"<b>%{{text}}</b><br>"
                        f"Time: %{{x:.0f}}s<br>"
                        f"Relevance: %{{y:.3f}}<br>"
                        f"Type: {p_type}<extra></extra>"
                    ),
                ))
            fig.update_layout(**PLOTLY_THEME, height=320,
                              title=dict(text="Ad Placement Timeline — bubble size = CPM",
                                         font=dict(size=12, color="#8B909E")),
                              xaxis_title="Time (seconds)", yaxis_title="Relevance Score",
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6B7280")))
            st.plotly_chart(fig, use_container_width=True)

            display_cols = ["timestamp_fmt", "placement_type", "ad_title", "brand",
                            "relevance_score", "safety_score", "estimated_cpm"]
            st.dataframe(
                p_df[display_cols].rename(columns={
                    "timestamp_fmt": "Time", "placement_type": "Type",
                    "ad_title": "Ad", "brand": "Brand",
                    "relevance_score": "Relevance", "safety_score": "Safety",
                    "estimated_cpm": "CPM ($)",
                }),
                use_container_width=True, hide_index=True,
            )

    with tab2:
        st.markdown('<div class="section-title">Match ads to a specific scene</div>', unsafe_allow_html=True)
        scene_options = [f"[{s.start_fmt}] {s.text[:55]}…" for s in vm.scenes]
        sel_scene_idx = st.selectbox("Select scene", range(len(vm.scenes)),
                                     format_func=lambda i: scene_options[i])
        scene = vm.scenes[sel_scene_idx]
        matches = ae.match_ads(scene, top_k=5)

        if not matches:
            st.warning("No eligible ads for this scene.")
        else:
            for ad, score_info in matches:
                total = score_info["total"]
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
                        <div>
                            <div style="font-weight:600;color:#F0F2F5;font-size:0.9rem">{ad.title}</div>
                            <div style="color:#6B7280;font-size:0.78rem;margin-top:2px">{ad.brand} · ${ad.cpm_base:.0f} base CPM</div>
                        </div>
                        <div style="text-align:right">
                            <div style="font-family:'DM Mono',monospace;color:#F59E0B;font-size:1.15rem;font-weight:600">{total:.3f}</div>
                            <div style="font-size:0.65rem;color:#4B5260;letter-spacing:0.04em">MATCH SCORE</div>
                        </div>
                    </div>
                    <p style="color:#8B909E;font-size:0.82rem;margin:0 0 10px;line-height:1.5">{ad.description}</p>
                    <div style="display:flex;gap:4px;height:4px;border-radius:3px;overflow:hidden;margin-bottom:6px">
                        <div style="flex:{score_info['content_sim']};background:#F59E0B"></div>
                        <div style="flex:{score_info['iab_match']};background:#34D399"></div>
                        <div style="flex:{score_info['safety']};background:#60A5FA"></div>
                        <div style="flex:{score_info['demographic']};background:#A78BFA"></div>
                        <div style="flex:{score_info['performance']};background:#FB923C"></div>
                    </div>
                    <div class="stat-row" style="margin-top:4px">
                        <span style="font-size:0.65rem;color:#F59E0B">■ content {score_info['content_sim']:.2f}</span>
                        <span style="font-size:0.65rem;color:#34D399">■ IAB {score_info['iab_match']:.2f}</span>
                        <span style="font-size:0.65rem;color:#60A5FA">■ safety {score_info['safety']:.2f}</span>
                        <span style="font-size:0.65rem;color:#A78BFA">■ demo {score_info['demographic']:.2f}</span>
                        <span style="font-size:0.65rem;color:#FB923C">■ perf {score_info['performance']:.2f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        inv_data = [ad.to_dict() for ad in ae.inventory]
        inv_df = pd.DataFrame(inv_data)
        display = ["title", "brand", "cpm_base", "historical_ctr", "performance_score",
                   "brand_safety_min", "budget_remaining"]
        st.dataframe(
            inv_df[display].rename(columns={
                "title": "Ad", "brand": "Brand", "cpm_base": "Base CPM",
                "historical_ctr": "CTR", "performance_score": "Perf",
                "brand_safety_min": "Min Safety", "budget_remaining": "Budget Left",
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    _page_header("📊", "Analytics Dashboard", "Performance metrics and content intelligence insights")

    if not st.session_state.videos:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <div class="empty-state-text">Process a video to see analytics.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    all_scenes = [s for vm in st.session_state.videos.values() for s in vm.scenes]

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Videos", len(st.session_state.videos))
    c2.metric("Total Scenes", len(all_scenes))
    avg_eng = round(sum(s.engagement_score for s in all_scenes) / max(len(all_scenes), 1), 3)
    c3.metric("Avg Engagement", avg_eng)
    safe_pct = round(sum(1 for s in all_scenes if s.brand_safety.get("safety_score", 1.0) >= 0.8) / max(len(all_scenes), 1) * 100, 1)
    c4.metric("Brand Safe", f"{safe_pct}%")
    positive = round(sum(1 for s in all_scenes if s.sentiment.get("label") == "positive") / max(len(all_scenes), 1) * 100, 1)
    c5.metric("Positive Sentiment", f"{positive}%")

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        labels = [s.sentiment.get("label", "neutral") for s in all_scenes]
        label_counts = {l: labels.count(l) for l in set(labels)}
        fig = px.pie(
            values=list(label_counts.values()),
            names=list(label_counts.keys()),
            color_discrete_map={"positive": "#34D399", "neutral": "#4B5260", "negative": "#F87171"},
            hole=0.5,
        )
        fig.update_layout(**PLOTLY_THEME, height=280,
                          title=dict(text="Sentiment Distribution", font=dict(size=12, color="#8B909E")),
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6B7280")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        iab_all: dict[str, int] = {}
        for s in all_scenes:
            for cat in s.iab_categories[:1]:
                iab_all[cat["name"]] = iab_all.get(cat["name"], 0) + 1
        if iab_all:
            top_iab = sorted(iab_all.items(), key=lambda x: x[1], reverse=True)[:10]
            df_iab = pd.DataFrame(top_iab, columns=["Category", "Scenes"])
            fig = px.bar(df_iab, x="Scenes", y="Category", orientation="h",
                         color="Scenes", color_continuous_scale=[[0, "#1E2028"], [1, "#F59E0B"]])
            fig.update_layout(**PLOTLY_THEME, height=280,
                              title=dict(text="Top Content Categories", font=dict(size=12, color="#8B909E")),
                              yaxis=dict(autorange="reversed", **PLOTLY_THEME["yaxis"]),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        eng_vals = [s.engagement_score for s in all_scenes]
        safe_vals = [s.brand_safety.get("safety_score", 1.0) for s in all_scenes]
        suit_vals = [s.ad_suitability for s in all_scenes]
        fig = go.Figure(go.Scatter(
            x=eng_vals, y=safe_vals, mode="markers",
            marker=dict(
                size=[v * 15 + 4 for v in suit_vals],
                color=suit_vals, colorscale=[[0, "#1E2028"], [1, "#F59E0B"]],
                showscale=True, opacity=0.7,
                colorbar=dict(title="Ad Suit.", tickfont=dict(color="#6B7280")),
            ),
            hovertemplate="eng: %{x:.2f}<br>safety: %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_THEME, height=300,
                          title=dict(text="Engagement vs Safety", font=dict(size=12, color="#8B909E")),
                          xaxis_title="Engagement Score", yaxis_title="Brand Safety Score")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.histogram([s.ad_suitability for s in all_scenes], nbins=15,
                           color_discrete_sequence=["#FB923C"])
        fig.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                          title=dict(text="Ad Suitability Distribution", font=dict(size=12, color="#8B909E")),
                          xaxis_title="Ad Suitability Score", yaxis_title="Scene Count")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Video Breakdown</div>', unsafe_allow_html=True)
    rows = []
    for vm in st.session_state.videos.values():
        if not vm.scenes:
            continue
        avg_e = round(sum(s.engagement_score for s in vm.scenes) / len(vm.scenes), 3)
        avg_s = round(sum(s.brand_safety.get("safety_score", 1.0) for s in vm.scenes) / len(vm.scenes), 3)
        rows.append({
            "Title": vm.title[:40],
            "Duration": vm.fmt_duration(),
            "Scenes": vm.scene_count,
            "Narrative": vm.narrative_structure.split("(")[0].strip(),
            "Avg Engagement": avg_e,
            "Avg Safety": f"{avg_s:.0%}",
            "Top Category": vm.dominant_iab[0]["name"] if vm.dominant_iab else "—",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: FRANCHISE INTEL
# ══════════════════════════════════════════════════════════════════════════════
def page_franchise():
    _page_header("🎯", "Franchise Intelligence", "Cross-video theme tracking and recurring ad opportunities")

    if len(st.session_state.videos) < 1:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-state-icon">🎯</div>
            <div class="empty-state-text">Process at least one video to see franchise analysis.</div>
        </div>
        """, unsafe_allow_html=True)
        return

    all_scenes = [s for vm in st.session_state.videos.values() for s in vm.scenes]
    se = st.session_state.search_engine

    theme_counts: dict[str, dict[str, int]] = {}
    for vm in st.session_state.videos.values():
        for theme in vm.franchise_themes:
            if theme not in theme_counts:
                theme_counts[theme] = {}
            theme_counts[theme][vm.title[:25]] = theme_counts[theme].get(vm.title[:25], 0) + 1

    if theme_counts:
        st.markdown('<div class="section-title">Recurring Themes</div>', unsafe_allow_html=True)
        theme_df_rows = []
        for theme, vids in theme_counts.items():
            for vid, cnt in vids.items():
                theme_df_rows.append({"Theme": theme, "Video": vid, "Occurrences": cnt})
        if theme_df_rows:
            tdf = pd.DataFrame(theme_df_rows)
            fig = px.bar(tdf, x="Theme", y="Occurrences", color="Video",
                         color_discrete_sequence=PLOTLY_THEME["colorway"])
            fig.update_layout(**PLOTLY_THEME, height=300,
                              title=dict(text="Theme Frequency by Video", font=dict(size=12, color="#8B909E")),
                              xaxis_tickangle=-30, showlegend=True,
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6B7280")))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">Top Recurring Ad Opportunities</div>', unsafe_allow_html=True)

    iab_opportunity: dict[str, list[dict]] = {}
    for vm in st.session_state.videos.values():
        for scene in vm.scenes:
            if scene.ad_suitability > 0.6:
                for cat in scene.iab_categories[:1]:
                    cat_name = cat["name"]
                    if cat_name not in iab_opportunity:
                        iab_opportunity[cat_name] = []
                    iab_opportunity[cat_name].append({
                        "video": vm.title[:30],
                        "scene_id": scene.scene_id,
                        "ad_suitability": scene.ad_suitability,
                        "engagement": scene.engagement_score,
                    })

    if iab_opportunity:
        sorted_opps = sorted(iab_opportunity.items(),
                             key=lambda x: sum(o["ad_suitability"] for o in x[1]),
                             reverse=True)[:6]

        for cat_name, scenes_list in sorted_opps[:4]:
            avg_suit = round(sum(o["ad_suitability"] for o in scenes_list) / len(scenes_list), 3)
            avg_eng  = round(sum(o["engagement"]     for o in scenes_list) / len(scenes_list), 3)
            unique_videos = len(set(o["video"] for o in scenes_list))
            st.markdown(f"""
            <div class="card">
                <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <div style="display:flex;align-items:center;gap:8px">
                        <span class="badge badge-amber">{cat_name}</span>
                        <span style="color:#6B7280;font-size:0.78rem">{len(scenes_list)} scenes · {unique_videos} video(s)</span>
                    </div>
                    <div style="font-family:'DM Mono',monospace;text-align:right">
                        <span style="color:#F59E0B;font-size:0.9rem">{avg_suit:.3f}</span>
                        <span style="color:#4B5260;font-size:0.68rem"> suit</span>
                        <span style="color:#34D399;font-size:0.9rem;margin-left:10px">{avg_eng:.3f}</span>
                        <span style="color:#4B5260;font-size:0.68rem"> eng</span>
                    </div>
                </div>
                {_score_bar(avg_suit)}
            </div>
            """, unsafe_allow_html=True)

    if len(all_scenes) > 1 and se.stats["total_scenes"] > 0:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Find Similar Scenes Across Videos</div>', unsafe_allow_html=True)
        scene_opt = [f"[{s.video_id[:8]}] {s.start_fmt} — {s.text[:50]}…" for s in all_scenes[:50]]
        sel = st.selectbox("Reference scene", range(len(all_scenes[:50])),
                           format_func=lambda i: scene_opt[i])
        ref_scene = all_scenes[sel]
        similar = se.find_similar_scenes(ref_scene, top_k=5, exclude_same_video=True)
        if similar:
            for r in similar:
                vm2 = st.session_state.videos.get(r.scene.video_id)
                st.markdown(f"""
                <div class="card">
                    <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                        <span style="color:#8B909E;font-size:0.78rem">{vm2.title[:35] if vm2 else r.scene.video_id}</span>
                        <span style="font-family:'DM Mono',monospace;color:#F59E0B;font-size:0.78rem">sim {r.score:.3f}</span>
                    </div>
                    <span class="timeline-time">{r.scene.start_fmt}</span>
                    <p style="color:#C0C5D0;font-size:0.85rem;margin:6px 0 0;line-height:1.6">{r.scene.text[:200]}…</p>
                    {_score_bar(r.score)}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No similar scenes found in other videos.")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _get_active_video() -> Optional[VideoMetadata]:
    if st.session_state.selected_video:
        vm = st.session_state.videos.get(st.session_state.selected_video)
        if vm:
            return vm
    if st.session_state.videos:
        return next(iter(st.session_state.videos.values()))
    return None


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
PAGE_MAP = {
    "process":   page_process,
    "search":    page_search,
    "scenes":    page_scenes,
    "ads":       page_ads,
    "analytics": page_analytics,
    "franchise": page_franchise,
}

PAGE_MAP[st.session_state.page]()
