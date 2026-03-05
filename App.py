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
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Dark industrial UI with amber accent
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
    background: #0A0A0F !important;
    color: #E2E4EA !important;
}

/* ── Main container ── */
[data-testid="stAppViewContainer"] > .main {
    background: #0A0A0F;
}
[data-testid="block-container"] {
    padding: 1.5rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] > div:first-child {
    background: #0F0F18 !important;
    border-right: 1px solid rgba(255,170,0,0.12) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stSelectbox label {
    color: #9AA0B0 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.5px;
}
[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E2E4EA !important;
    font-size: 0.85rem !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(255,170,0,0.1) !important; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #13131F !important;
    border: 1px solid rgba(255,170,0,0.15) !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    color: #6B7280 !important;
    font-size: 0.7rem !important;
    letter-spacing: 1px !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #FFAA00 !important;
    font-size: 1.6rem !important;
}
[data-testid="stMetricDelta"] { font-size: 0.75rem !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #FFAA00, #FF7700) !important;
    color: #0A0A0F !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.5px !important;
    padding: 8px 20px !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(255,170,0,0.35) !important;
}

/* ── Sidebar nav buttons ── */
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    color: #9AA0B0 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    text-align: left !important;
    width: 100% !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 9px 14px !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,170,0,0.08) !important;
    border-color: rgba(255,170,0,0.3) !important;
    color: #FFAA00 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Text inputs ── */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #13131F !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E2E4EA !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #FFAA00 !important;
    box-shadow: 0 0 0 2px rgba(255,170,0,0.15) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #13131F !important;
    border: 2px dashed rgba(255,170,0,0.25) !important;
    border-radius: 12px !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #13131F !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] summary {
    color: #E2E4EA !important;
    font-weight: 600 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #13131F !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    color: #6B7280 !important;
    background: transparent !important;
    border-radius: 7px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    padding: 8px 18px !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(255,170,0,0.15) !important;
    color: #FFAA00 !important;
}

/* ── Progress bar ── */
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #FFAA00, #FF7700) !important;
}

/* ── Info / warning / error ── */
[data-testid="stAlert"] {
    border-radius: 10px !important;
    border: none !important;
}

/* ── Plotly charts ── */
[data-testid="stPlotlyChart"] {
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    overflow: hidden;
    background: #13131F;
}

/* ── Custom components ── */
.page-header {
    display: flex; align-items: center; gap: 14px;
    margin-bottom: 28px; padding-bottom: 18px;
    border-bottom: 1px solid rgba(255,170,0,0.15);
}
.page-title {
    font-size: 1.7rem; font-weight: 800; color: #FFFFFF;
    letter-spacing: -0.5px; line-height: 1;
}
.page-subtitle { font-size: 0.82rem; color: #6B7280; margin-top: 4px; }
.page-icon { font-size: 2rem; }

.scene-card {
    background: #13131F;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 10px;
    cursor: pointer;
    transition: border-color 0.15s;
}
.scene-card:hover { border-color: rgba(255,170,0,0.35); }
.scene-card.key { border-color: rgba(255,170,0,0.45); }

.tag {
    display: inline-block; padding: 2px 10px; border-radius: 20px;
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.5px;
    margin: 2px; text-transform: uppercase;
}
.tag-amber  { background: rgba(255,170,0,0.15); color: #FFAA00; }
.tag-green  { background: rgba(54,211,153,0.15); color: #36D399; }
.tag-red    { background: rgba(255,82,82,0.15);  color: #FF5252; }
.tag-blue   { background: rgba(59,130,246,0.15); color: #60A5FA; }
.tag-purple { background: rgba(167,139,250,0.15);color: #A78BFA; }

.stat-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px; padding: 4px 12px;
    font-size: 0.75rem; color: #9AA0B0;
    margin: 2px;
}
.stat-pill strong { color: #E2E4EA; }

.ad-card {
    background: #16161F;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 16px;
    margin-bottom: 8px;
}
.ad-score-bar {
    height: 4px; border-radius: 2px;
    background: linear-gradient(90deg, #FFAA00, #FF7700);
    margin-top: 8px;
}

.search-result-card {
    background: #13131F;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 18px;
    margin-bottom: 8px;
    transition: border-color 0.15s;
}
.search-result-card:hover { border-color: rgba(255,170,0,0.3); }

.logo-mark {
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 1.15rem;
    background: linear-gradient(135deg, #FFAA00, #FF7700);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
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
    plot_bgcolor  = "#13131F",
    paper_bgcolor = "#13131F",
    font          = dict(family="JetBrains Mono, monospace", color="#9AA0B0", size=11),
    margin        = dict(l=16, r=16, t=40, b=16),
    colorway      = ["#FFAA00", "#FF7700", "#36D399", "#60A5FA", "#A78BFA",
                     "#FB923C", "#F472B6", "#34D399"],
    xaxis         = dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    yaxis         = dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.1)"),
    hoverlabel    = dict(bgcolor="#1E1E2E", bordercolor="#FFAA00",
                         font=dict(family="JetBrains Mono", size=11)),
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
    st.markdown('<div class="logo-mark">⚡ SEMANTIX</div>', unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.65rem;color:#4B5563;margin-top:2px;margin-bottom:16px'>Enterprise Video Intelligence</div>", unsafe_allow_html=True)
    st.divider()

    for page_id, icon, label in NAV_PAGES:
        active = st.session_state.page == page_id
        style = "color:#FFAA00 !important;border-color:rgba(255,170,0,0.4) !important;background:rgba(255,170,0,0.08) !important;" if active else ""
        if st.button(f"{icon}  {label}", key=f"nav_{page_id}",
                     use_container_width=True):
            st.session_state.page = page_id
            st.rerun()

    st.divider()

    # Video selector
    if st.session_state.videos:
        st.markdown("**Loaded Videos**")
        video_options = ["— All Videos —"] + [
            f"{v.title[:28]}…" if len(v.title) > 28 else v.title
            for v in st.session_state.videos.values()
        ]
        video_ids = [None] + list(st.session_state.videos.keys())
        sel = st.selectbox("Select video", video_options, key="video_selector",
                           label_visibility="collapsed")
        sel_idx = video_options.index(sel)
        st.session_state.selected_video = video_ids[sel_idx]

    # Stats
    se_stats = st.session_state.search_engine.stats
    if se_stats["total_scenes"] > 0:
        st.divider()
        st.markdown(f"""
        <div style='font-size:0.7rem;color:#6B7280'>
            <div>🎬 <strong style='color:#9AA0B0'>{se_stats['total_videos']}</strong> videos indexed</div>
            <div style='margin-top:4px'>🎞️ <strong style='color:#9AA0B0'>{se_stats['total_scenes']}</strong> scenes indexed</div>
            <div style='margin-top:4px'>📚 <strong style='color:#9AA0B0'>{se_stats['vocab_size']:,}</strong> vocab terms</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    yt_key = st.text_input("YouTube API Key (optional)", type="password",
                            value=st.session_state.yt_api_key,
                            key="yt_key_input", label_visibility="visible")
    if yt_key != st.session_state.yt_api_key:
        st.session_state.yt_api_key = yt_key


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _register_video(vm: VideoMetadata):
    """Add video to session state, index scenes, sync ad engine vocab."""
    st.session_state.videos[vm.video_id] = vm
    st.session_state.search_engine.add_scenes(vm.scenes)
    # Sync ad embeddings to the same vocabulary space so cosine similarity
    # between scene and ad vectors is valid (shared vocab = aligned space).
    if st.session_state.search_engine.vectorizer is not None:
        st.session_state.ad_engine.sync_vectorizer(
            st.session_state.search_engine.vectorizer
        )
    st.session_state.selected_video = vm.video_id


def _sentiment_color(label: str) -> str:
    return {"positive": "#36D399", "negative": "#FF5252", "neutral": "#9AA0B0"}.get(label, "#9AA0B0")


def _safety_color(score: float) -> str:
    if score >= 0.8: return "#36D399"
    if score >= 0.5: return "#FFAA00"
    return "#FF5252"


def _score_bar(score: float, color: str = "#FFAA00", width: int = 100) -> str:
    pct = int(score * 100)
    return f"""<div style='width:{width}%;background:rgba(255,255,255,0.06);border-radius:3px;height:5px;margin-top:5px'>
        <div style='width:{pct}%;background:{color};border-radius:3px;height:5px'></div></div>"""


def _iab_tags(iab_list: list[dict], max_n: int = 3) -> str:
    colors = ["tag-amber", "tag-blue", "tag-purple"]
    tags = ""
    for i, cat in enumerate(iab_list[:max_n]):
        cls = colors[i % len(colors)]
        tags += f'<span class="tag {cls}">{cat["name"]}</span>'
    return tags


def _render_scene_card(scene: Scene, is_key: bool = False, show_ads: bool = False):
    safety = scene.brand_safety.get("safety_score", 1.0)
    sentiment = scene.sentiment
    card_class = "scene-card key" if is_key else "scene-card"
    key_badge = '<span class="tag tag-amber">★ KEY SCENE</span>' if is_key else ""

    iab_html = _iab_tags(scene.iab_categories)
    safety_html = f'<span style="color:{_safety_color(safety)};font-size:0.72rem;font-family:\'JetBrains Mono\',monospace">🛡 {safety:.0%}</span>'
    sentiment_html = f'<span style="color:{_sentiment_color(sentiment.get("label","neutral"))};font-size:0.72rem">● {sentiment.get("label","neutral").title()}</span>'

    st.markdown(f"""
    <div class="{card_class}">
        <div style='display:flex;justify-content:space-between;align-items:flex-start'>
            <div>
                <span style='font-family:"JetBrains Mono",monospace;color:#FFAA00;font-size:0.75rem'>
                    {scene.start_fmt} → {scene.end_fmt}
                </span>
                <span style='color:#4B5563;font-size:0.72rem;margin-left:10px'>
                    {scene.duration_sec:.0f}s · {len(scene.text.split())} words
                </span>
            </div>
            <div>{key_badge} {safety_html} {sentiment_html}</div>
        </div>
        <p style='margin:8px 0 6px;font-size:0.85rem;color:#C4C9D8;line-height:1.5'>
            {scene.text[:200]}{"…" if len(scene.text) > 200 else ""}
        </p>
        <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
            {iab_html}
            <span style='color:#4B5563;font-size:0.7rem'>eng: {scene.engagement_score:.2f}</span>
            <span style='color:#4B5563;font-size:0.7rem'>ad: {scene.ad_suitability:.2f}</span>
        </div>
        {_score_bar(scene.ad_suitability, "#FFAA00")}
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PROCESS VIDEO
# ══════════════════════════════════════════════════════════════════════════════
def page_process():
    st.markdown("""
    <div class="page-header">
        <span class="page-icon">🎬</span>
        <div>
            <div class="page-title">Process Video</div>
            <div class="page-subtitle">Upload subtitles or fetch from YouTube to extract semantic scenes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📁 Upload File", "▶ YouTube URL", "📝 Paste Text"])

    # ── Tab 1: File Upload ────────────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded = st.file_uploader(
                "Drop your SRT or VTT subtitle file here",
                type=["srt", "vtt"],
                help="Supports SRT and WebVTT formats",
            )
            video_title = st.text_input("Video Title (optional)", placeholder="My Awesome Video")

        with col2:
            st.markdown("**Detection Settings**")
            min_scene = st.slider("Min scene length (s)", 10, 60, 20)
            max_scene = st.slider("Max scene length (s)", 60, 300, 120)
            threshold = st.slider("Semantic sensitivity", 0.2, 0.7, 0.35, 0.05,
                                  help="Higher = fewer but more distinct scenes")

        if uploaded and st.button("🚀 Process Subtitles", use_container_width=True):
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
            st.success(f"✓ Processed in {elapsed:.2f}s")
            _show_processing_summary(vm)

    # ── Tab 2: YouTube URL ────────────────────────────────────────────────────
    with tab2:
        yt_url = st.text_input("YouTube URL or Video ID",
                               placeholder="https://youtube.com/watch?v=... or video ID")
        col1, col2 = st.columns(2)
        with col1:
            min_s2 = st.slider("Min scene (s)", 10, 60, 20, key="yt_min")
            max_s2 = st.slider("Max scene (s)", 60, 300, 120, key="yt_max")
        with col2:
            thr2 = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="yt_thr")

        if st.button("▶ Fetch & Process", use_container_width=True):
            vid_id = _extract_yt_id(yt_url)
            if not vid_id:
                st.error("Could not parse YouTube video ID from URL.")
                return

            with st.spinner("Fetching YouTube transcript…"):
                transcript = fetch_youtube_transcript(vid_id)

            if transcript is None:
                st.error("Could not fetch transcript. The video may not have captions, "
                         "or youtube-transcript-api may not be installed.\n\n"
                         "Install: `pip install youtube-transcript-api`")
                return

            meta = None
            if st.session_state.yt_api_key:
                with st.spinner("Fetching YouTube metadata…"):
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
        paste_title = st.text_input("Video Title", placeholder="My Video", key="paste_title")
        pasted = st.text_area(
            "Paste SRT or VTT content here",
            height=250,
            placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello, welcome to our show...",
        )
        min_s3 = st.slider("Min scene (s)", 10, 60, 20, key="p_min")
        max_s3 = st.slider("Max scene (s)", 60, 300, 120, key="p_max")

        if pasted and st.button("🚀 Process Text", use_container_width=True):
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
    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenes Detected", vm.scene_count)
    c2.metric("Duration", vm.fmt_duration())
    c3.metric("Total Cues", vm.total_cues)
    avg_dur = round(sum(s.duration_sec for s in vm.scenes) / max(vm.scene_count, 1), 1)
    c4.metric("Avg Scene", f"{avg_dur}s")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Narrative Structure**")
        st.markdown(f"<span class='tag tag-amber'>{vm.narrative_structure}</span>", unsafe_allow_html=True)
        st.markdown("**Dominant Topics**")
        for cat in vm.dominant_iab[:4]:
            st.markdown(f"<span class='tag tag-blue'>{cat['name']}</span>", unsafe_allow_html=True)

    with col2:
        st.markdown("**Emotional Arc**")
        if vm.emotional_arc:
            arc_df = pd.DataFrame(vm.emotional_arc)
            fig = px.area(arc_df, x="start_sec", y="sentiment_score",
                          color_discrete_sequence=["#FFAA00"])
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.2)")
            fig.update_layout(**PLOTLY_THEME,
                              height=200, showlegend=False,
                              title=dict(text="", x=0),
                              xaxis_title="Time (s)", yaxis_title="Sentiment")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SEMANTIC SEARCH
# ══════════════════════════════════════════════════════════════════════════════
def page_search():
    st.markdown("""
    <div class="page-header">
        <span class="page-icon">🔍</span>
        <div>
            <div class="page-title">Semantic Search</div>
            <div class="page-subtitle">Natural language discovery across all indexed scenes</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    se = st.session_state.search_engine
    if se.stats["total_scenes"] == 0:
        st.info("No videos indexed yet. Process a video first.")
        return

    col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
    with col1:
        query = st.text_input("", placeholder="🔍  Search: 'emotional confrontation between characters'…",
                              label_visibility="collapsed")
    with col2:
        top_k = st.selectbox("Results", [5, 10, 20], index=1, label_visibility="collapsed")
    with col3:
        min_safety = st.selectbox("Safety", ["Any", "Safe (0.5+)", "Brand Safe (0.8+)"],
                                  label_visibility="collapsed")
        safety_map = {"Any": 0.0, "Safe (0.5+)": 0.5, "Brand Safe (0.8+)": 0.8}
        min_safety_val = safety_map[min_safety]
    with col4:
        diversify = st.checkbox("Diversify", value=True)

    # IAB filter
    iab_choices = ["— All —"] + [f"{k}: {v}" for k, v in _IAB_NAMES.items()]
    iab_sel = st.multiselect("Filter by IAB Category", iab_choices,
                              default=[], label_visibility="collapsed",
                              placeholder="Filter by content category…")
    iab_filter = [s.split(":")[0] for s in iab_sel if s != "— All —"] or None

    # Video filter
    if len(st.session_state.videos) > 1:
        vid_choices = ["All Videos"] + [
            f"{vm.title}" for vm in st.session_state.videos.values()
        ]
        vid_ids = [None] + list(st.session_state.videos.keys())
        vid_sel = st.selectbox("Filter by Video", vid_choices, label_visibility="collapsed")
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

        st.markdown(f"<div style='color:#6B7280;font-size:0.78rem;margin-bottom:12px'>Found <strong style='color:#FFAA00'>{len(results)}</strong> scenes</div>", unsafe_allow_html=True)

        for r in results:
            scene = r.scene
            vm = st.session_state.videos.get(scene.video_id)
            video_title = vm.title if vm else scene.video_id
            is_key = vm and scene.scene_id in vm.key_scenes if vm else False

            safety = scene.brand_safety.get("safety_score", 1.0)
            iab_html = _iab_tags(scene.iab_categories)
            sentiment = scene.sentiment

            st.markdown(f"""
            <div class="search-result-card">
                <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px'>
                    <div>
                        <span style='color:#FFAA00;font-weight:700;font-size:0.9rem'>#{r.rank}</span>
                        <span style='color:#4B5563;font-size:0.75rem;margin-left:8px'>{video_title}</span>
                        {"<span class='tag tag-amber' style='margin-left:8px'>★ KEY</span>" if is_key else ""}
                    </div>
                    <div style='font-family:"JetBrains Mono",monospace;color:#FFAA00;font-size:0.75rem'>
                        score: {r.score:.3f}
                    </div>
                </div>
                <div style='font-family:"JetBrains Mono",monospace;color:#6B7280;font-size:0.72rem;margin-bottom:6px'>
                    ⏱ {scene.start_fmt} → {scene.end_fmt}  ·  {scene.duration_sec:.0f}s
                </div>
                <p style='color:#C4C9D8;font-size:0.88rem;line-height:1.55;margin:0 0 10px'>
                    {scene.text[:280]}{"…" if len(scene.text) > 280 else ""}
                </p>
                <div style='display:flex;align-items:center;gap:8px;flex-wrap:wrap'>
                    {iab_html}
                    <span style='color:{_sentiment_color(sentiment.get("label","neutral"))};font-size:0.7rem'>
                        ● {sentiment.get("label","").title()}
                    </span>
                    <span style='color:{_safety_color(safety)};font-size:0.7rem'>
                        🛡 {safety:.0%}
                    </span>
                    <span style='color:#4B5563;font-size:0.7rem'>
                        vec:{r.vector_score:.2f} bm25:{r.bm25_score:.2f}
                    </span>
                </div>
                {_score_bar(r.score)}
            </div>
            """, unsafe_allow_html=True)

    else:
        # Show search suggestions
        st.markdown("**Suggested Queries**")
        suggestions = [
            "exciting action sequence", "emotional dialogue",
            "product demonstration", "travel destination",
            "health and wellness advice", "financial discussion",
            "comedy moment", "suspenseful scene",
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
    st.markdown("""
    <div class="page-header">
        <span class="page-icon">📺</span>
        <div>
            <div class="page-title">Scene Explorer</div>
            <div class="page-subtitle">Browse and analyse every detected scene</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.videos:
        st.info("No videos processed yet.")
        return

    vm = _get_active_video()
    if not vm:
        return

    # Video summary header
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Scenes", vm.scene_count)
    c2.metric("Duration", vm.fmt_duration())
    avg_eng = round(sum(s.engagement_score for s in vm.scenes) / max(vm.scene_count, 1), 2)
    c3.metric("Avg Engagement", avg_eng)
    avg_safe = round(sum(s.brand_safety.get("safety_score", 1.0) for s in vm.scenes) / max(vm.scene_count, 1), 2)
    c4.metric("Avg Safety", f"{avg_safe:.0%}")
    c5.metric("Narrative", vm.narrative_structure.split("(")[0].strip())

    st.divider()

    # Visualisations
    tab1, tab2, tab3 = st.tabs(["🎭 Emotional Arc", "📊 Scene Analysis", "🗂 Scene List"])

    with tab1:
        if vm.emotional_arc:
            df = pd.DataFrame(vm.emotional_arc)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["start_sec"], y=df["sentiment_score"],
                fill="tozeroy", fillcolor="rgba(255,170,0,0.12)",
                line=dict(color="#FFAA00", width=2),
                mode="lines+markers", marker=dict(size=6, color="#FFAA00"),
                name="Sentiment",
                hovertemplate="<b>%{x:.0f}s</b><br>Sentiment: %{y:.3f}<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=df["start_sec"], y=df["engagement"],
                line=dict(color="#36D399", width=2, dash="dot"),
                mode="lines", name="Engagement",
                hovertemplate="<b>%{x:.0f}s</b><br>Engagement: %{y:.3f}<extra></extra>",
            ))
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")

            # Mark key scenes
            for kid in vm.key_scenes:
                ks = next((s for s in vm.scenes if s.scene_id == kid), None)
                if ks:
                    fig.add_vline(x=ks.start_sec, line_dash="dash",
                                  line_color="rgba(255,170,0,0.4)")

            fig.update_layout(**PLOTLY_THEME, height=320, showlegend=True,
                              title=dict(text="Emotional Arc + Engagement Over Time",
                                         font=dict(size=13, color="#E2E4EA")),
                              xaxis_title="Time (seconds)",
                              yaxis_title="Score",
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9AA0B0")))
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            # IAB distribution
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
                             color_discrete_sequence=["#FFAA00"])
                fig.update_layout(**PLOTLY_THEME, height=300,
                                  title=dict(text="Top IAB Categories", font=dict(size=12, color="#E2E4EA")),
                                  yaxis=dict(autorange="reversed", **PLOTLY_THEME["yaxis"]))
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            # Safety distribution
            safety_scores = [s.brand_safety.get("safety_score", 1.0) for s in vm.scenes]
            fig = px.histogram(safety_scores, nbins=10,
                               color_discrete_sequence=["#36D399"],
                               labels={"value": "Safety Score", "count": "Scenes"})
            fig.update_layout(**PLOTLY_THEME, height=300,
                              title=dict(text="Brand Safety Distribution", font=dict(size=12, color="#E2E4EA")),
                              showlegend=False, xaxis_title="Safety Score", yaxis_title="Scene Count")
            st.plotly_chart(fig, use_container_width=True)

        # Engagement heatmap
        eng_scores = [s.engagement_score for s in vm.scenes]
        suit_scores = [s.ad_suitability for s in vm.scenes]
        times = [s.start_sec for s in vm.scenes]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=eng_scores,
            mode="markers", marker=dict(
                size=[s * 20 + 4 for s in suit_scores],
                color=eng_scores, colorscale="YlOrRd",
                showscale=True, colorbar=dict(title="Engagement", tickfont=dict(color="#9AA0B0")),
            ),
            hovertemplate="<b>%{x:.0f}s</b><br>Engagement: %{y:.3f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_THEME, height=280,
                          title=dict(text="Scene Engagement Map (bubble size = ad suitability)",
                                     font=dict(size=12, color="#E2E4EA")),
                          xaxis_title="Time (s)", yaxis_title="Engagement")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        # Filters
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

        st.markdown(f"<div style='color:#6B7280;font-size:0.78rem;margin-bottom:10px'>{len(filtered)} scenes shown</div>", unsafe_allow_html=True)

        for scene in filtered:
            is_key = scene.scene_id in vm.key_scenes
            _render_scene_card(scene, is_key)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: AD ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def page_ads():
    st.markdown("""
    <div class="page-header">
        <span class="page-icon">📢</span>
        <div>
            <div class="page-title">Ad Engine</div>
            <div class="page-subtitle">Contextual ad matching and placement optimisation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.videos:
        st.info("Process a video first.")
        return

    vm = _get_active_video()
    if not vm:
        return

    ae = st.session_state.ad_engine

    tab1, tab2, tab3 = st.tabs(["🎯 Placement Plan", "🔍 Scene Matching", "📦 Inventory"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown("**Placement Configuration**")
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
                <div style='font-size:0.78rem;color:#6B7280;line-height:1.8'>
                    <div>⏱ Min gap between ads: <strong style='color:#E2E4EA'>3 min</strong></div>
                    <div>📊 Max density: <strong style='color:#E2E4EA'>1 per 5 min</strong></div>
                    <div>🎬 Available scenes: <strong style='color:#FFAA00'>{vm.scene_count}</strong></div>
                    <div>⏱ Video duration: <strong style='color:#E2E4EA'>{vm.fmt_duration()}</strong></div>
                </div>
                """, unsafe_allow_html=True)

        if st.button("🚀 Generate Placement Plan", use_container_width=True):
            with st.spinner("Optimising placements…"):
                placements = ae.plan_placements(vm.scenes, vm.duration_ms, placement_types)
                perf = ae.simulate_performance(placements)
                st.session_state["_placements"] = placements
                st.session_state["_perf"] = perf

        if "_placements" in st.session_state and st.session_state["_placements"]:
            placements = st.session_state["_placements"]
            perf = st.session_state["_perf"]

            # KPIs
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Placements", perf["total_placements"])
            c2.metric("Est. Revenue", f"${perf['total_revenue_usd']:.2f}")
            c3.metric("Est. Impressions", f"{perf['total_impressions']:,}")
            c4.metric("Est. Clicks", f"{perf['estimated_clicks']:,}")
            c5.metric("Avg CPM", f"${perf['avg_cpm']:.2f}")

            # Timeline chart
            p_df = pd.DataFrame([p.to_dict() for p in placements])
            type_colors = {"pre-roll": "#FFAA00", "mid-roll": "#36D399", "post-roll": "#60A5FA"}

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
                    marker=dict(
                        size=sub["estimated_cpm"] * 2 + 10,
                        color=type_colors.get(p_type, "#FFAA00"),
                        opacity=0.85,
                        symbol="circle",
                    ),
                    hovertemplate=(
                        f"<b>%{{text}}</b><br>"
                        f"Time: %{{x:.0f}}s<br>"
                        f"Relevance: %{{y:.3f}}<br>"
                        f"Type: {p_type}<extra></extra>"
                    ),
                ))
            fig.update_layout(**PLOTLY_THEME, height=320,
                              title=dict(text="Ad Placement Timeline (bubble size = CPM)",
                                         font=dict(size=12, color="#E2E4EA")),
                              xaxis_title="Time (seconds)", yaxis_title="Relevance Score",
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9AA0B0")))
            st.plotly_chart(fig, use_container_width=True)

            # Placements table
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
        st.markdown("**Match ads to a specific scene**")
        scene_options = [f"[{s.start_fmt}] {s.text[:50]}…" for s in vm.scenes]
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
                <div class="ad-card">
                    <div style='display:flex;justify-content:space-between'>
                        <div>
                            <div style='font-weight:700;color:#E2E4EA'>{ad.title}</div>
                            <div style='color:#6B7280;font-size:0.78rem'>{ad.brand} · ${ad.cpm_base:.0f} CPM base</div>
                        </div>
                        <div style='text-align:right'>
                            <div style='font-family:"JetBrains Mono",monospace;color:#FFAA00;font-size:1.1rem;font-weight:700'>
                                {total:.3f}
                            </div>
                            <div style='font-size:0.65rem;color:#6B7280'>match score</div>
                        </div>
                    </div>
                    <p style='color:#9AA0B0;font-size:0.8rem;margin:8px 0 6px'>{ad.description}</p>
                    <div style='display:flex;gap:16px;font-size:0.72rem;font-family:"JetBrains Mono",monospace;color:#6B7280'>
                        <span>content <strong style='color:#FFAA00'>{score_info['content_sim']:.3f}</strong></span>
                        <span>IAB <strong style='color:#36D399'>{score_info['iab_match']:.3f}</strong></span>
                        <span>safety <strong style='color:#60A5FA'>{score_info['safety']:.3f}</strong></span>
                        <span>demo <strong style='color:#A78BFA'>{score_info['demographic']:.3f}</strong></span>
                        <span>perf <strong style='color:#FB923C'>{score_info['performance']:.3f}</strong></span>
                    </div>
                    <div style='display:flex;gap:3px;margin-top:10px'>
                        <div style='flex:{score_info["content_sim"]};background:#FFAA00;height:3px;border-radius:2px'></div>
                        <div style='flex:{score_info["iab_match"]};background:#36D399;height:3px;border-radius:2px'></div>
                        <div style='flex:{score_info["safety"]};background:#60A5FA;height:3px;border-radius:2px'></div>
                        <div style='flex:{score_info["demographic"]};background:#A78BFA;height:3px;border-radius:2px'></div>
                        <div style='flex:{score_info["performance"]};background:#FB923C;height:3px;border-radius:2px'></div>
                    </div>
                    <div style='font-size:0.6rem;color:#4B5563;margin-top:2px;display:flex;gap:8px'>
                        <span style='color:#FFAA00'>■ content</span>
                        <span style='color:#36D399'>■ IAB</span>
                        <span style='color:#60A5FA'>■ safety</span>
                        <span style='color:#A78BFA'>■ demo</span>
                        <span style='color:#FB923C'>■ perf</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with tab3:
        st.markdown("**Ad Inventory**")
        inv_data = [ad.to_dict() for ad in ae.inventory]
        inv_df = pd.DataFrame(inv_data)
        display = ["title", "brand", "cpm_base", "historical_ctr", "performance_score",
                   "brand_safety_min", "budget_remaining"]
        st.dataframe(
            inv_df[display].rename(columns={
                "title": "Ad", "brand": "Brand", "cpm_base": "Base CPM",
                "historical_ctr": "CTR", "performance_score": "Perf Score",
                "brand_safety_min": "Min Safety", "budget_remaining": "Budget Left",
            }),
            use_container_width=True, hide_index=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("""
    <div class="page-header">
        <span class="page-icon">📊</span>
        <div>
            <div class="page-title">Analytics Dashboard</div>
            <div class="page-subtitle">Performance metrics and content intelligence insights</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.videos:
        st.info("Process a video to see analytics.")
        return

    all_scenes = [s for vm in st.session_state.videos.values() for s in vm.scenes]

    # Global KPIs
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Videos", len(st.session_state.videos))
    c2.metric("Total Scenes", len(all_scenes))
    avg_eng = round(sum(s.engagement_score for s in all_scenes) / max(len(all_scenes), 1), 3)
    c3.metric("Avg Engagement", avg_eng)
    safe_pct = round(sum(1 for s in all_scenes if s.brand_safety.get("safety_score", 1.0) >= 0.8) / max(len(all_scenes), 1) * 100, 1)
    c4.metric("Brand Safe Scenes", f"{safe_pct}%")
    positive = round(sum(1 for s in all_scenes if s.sentiment.get("label") == "positive") / max(len(all_scenes), 1) * 100, 1)
    c5.metric("Positive Sentiment", f"{positive}%")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        # Sentiment distribution
        labels = [s.sentiment.get("label", "neutral") for s in all_scenes]
        label_counts = {l: labels.count(l) for l in set(labels)}
        fig = px.pie(
            values=list(label_counts.values()),
            names=list(label_counts.keys()),
            color_discrete_map={"positive": "#36D399", "neutral": "#6B7280", "negative": "#FF5252"},
            hole=0.45,
        )
        fig.update_layout(**PLOTLY_THEME, height=280,
                          title=dict(text="Sentiment Distribution", font=dict(size=12, color="#E2E4EA")),
                          showlegend=True,
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9AA0B0")))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # IAB category heatmap
        iab_all: dict[str, int] = {}
        for s in all_scenes:
            for cat in s.iab_categories[:1]:
                iab_all[cat["name"]] = iab_all.get(cat["name"], 0) + 1
        if iab_all:
            top_iab = sorted(iab_all.items(), key=lambda x: x[1], reverse=True)[:10]
            df_iab = pd.DataFrame(top_iab, columns=["Category", "Scenes"])
            fig = px.bar(df_iab, x="Scenes", y="Category", orientation="h",
                         color="Scenes", color_continuous_scale=["#1E1E2E", "#FFAA00"])
            fig.update_layout(**PLOTLY_THEME, height=280,
                              title=dict(text="Top Content Categories", font=dict(size=12, color="#E2E4EA")),
                              yaxis=dict(autorange="reversed", **PLOTLY_THEME["yaxis"]),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        # Engagement vs Safety scatter
        eng_vals = [s.engagement_score for s in all_scenes]
        safe_vals = [s.brand_safety.get("safety_score", 1.0) for s in all_scenes]
        suit_vals = [s.ad_suitability for s in all_scenes]
        vid_names = [
            st.session_state.videos[s.video_id].title[:20]
            if s.video_id in st.session_state.videos else s.video_id
            for s in all_scenes
        ]
        fig = go.Figure(go.Scatter(
            x=eng_vals, y=safe_vals,
            mode="markers",
            marker=dict(
                size=[v * 15 + 5 for v in suit_vals],
                color=suit_vals, colorscale="YlOrRd",
                showscale=True, opacity=0.75,
                colorbar=dict(title="Ad Suit.", tickfont=dict(color="#9AA0B0")),
            ),
            text=[f"{n}<br>ad suit: {v:.2f}" for n, v in zip(vid_names, suit_vals)],
            hovertemplate="%{text}<br>eng: %{x:.2f}<br>safety: %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(**PLOTLY_THEME, height=300,
                          title=dict(text="Engagement vs Safety (bubble = ad suitability)",
                                     font=dict(size=12, color="#E2E4EA")),
                          xaxis_title="Engagement Score",
                          yaxis_title="Brand Safety Score")
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        # Ad suitability distribution
        fig = px.histogram(suit_vals, nbins=15,
                           color_discrete_sequence=["#FF7700"])
        fig.update_layout(**PLOTLY_THEME, height=300, showlegend=False,
                          title=dict(text="Ad Suitability Distribution",
                                     font=dict(size=12, color="#E2E4EA")),
                          xaxis_title="Ad Suitability Score",
                          yaxis_title="Scene Count")
        st.plotly_chart(fig, use_container_width=True)

    # Per-video breakdown table
    st.divider()
    st.markdown("**Video Breakdown**")
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
    st.markdown("""
    <div class="page-header">
        <span class="page-icon">🎯</span>
        <div>
            <div class="page-title">Franchise Intelligence</div>
            <div class="page-subtitle">Cross-video theme tracking and recurring ad opportunities</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.videos) < 1:
        st.info("Process at least one video to see franchise analysis.")
        return

    all_scenes = [s for vm in st.session_state.videos.values() for s in vm.scenes]
    se = st.session_state.search_engine

    # Theme frequency across all videos
    theme_counts: dict[str, dict[str, int]] = {}
    for vm in st.session_state.videos.values():
        for theme in vm.franchise_themes:
            if theme not in theme_counts:
                theme_counts[theme] = {}
            theme_counts[theme][vm.title[:25]] = theme_counts[theme].get(vm.title[:25], 0) + 1

    if theme_counts:
        st.markdown("**Recurring Themes Across Videos**")
        theme_df_rows = []
        for theme, vids in theme_counts.items():
            for vid, cnt in vids.items():
                theme_df_rows.append({"Theme": theme, "Video": vid, "Occurrences": cnt})
        if theme_df_rows:
            tdf = pd.DataFrame(theme_df_rows)
            fig = px.bar(
                tdf, x="Theme", y="Occurrences", color="Video",
                color_discrete_sequence=PLOTLY_THEME["colorway"],
            )
            fig.update_layout(**PLOTLY_THEME, height=300,
                              title=dict(text="Theme Frequency by Video",
                                         font=dict(size=12, color="#E2E4EA")),
                              xaxis_tickangle=-30, showlegend=True,
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9AA0B0")))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Cross-video ad opportunities
    st.markdown("**Top Recurring Ad Opportunities**")
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
                             reverse=True)[:8]

        for cat_name, scenes_list in sorted_opps[:4]:
            avg_suit = round(sum(o["ad_suitability"] for o in scenes_list) / len(scenes_list), 3)
            avg_eng  = round(sum(o["engagement"]     for o in scenes_list) / len(scenes_list), 3)
            st.markdown(f"""
            <div class="ad-card">
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <div>
                        <span class='tag tag-amber'>{cat_name}</span>
                        <span style='color:#6B7280;font-size:0.75rem;margin-left:10px'>{len(scenes_list)} scenes across {len(set(o["video"] for o in scenes_list))} video(s)</span>
                    </div>
                    <div style='font-family:"JetBrains Mono",monospace;text-align:right'>
                        <span style='color:#FFAA00;font-size:0.9rem'>{avg_suit:.3f}</span>
                        <span style='color:#4B5563;font-size:0.7rem'> avg suit</span>
                        <span style='color:#36D399;font-size:0.9rem;margin-left:8px'>{avg_eng:.3f}</span>
                        <span style='color:#4B5563;font-size:0.7rem'> avg eng</span>
                    </div>
                </div>
                {_score_bar(avg_suit)}
            </div>
            """, unsafe_allow_html=True)

    # Similar scenes across videos
    if len(all_scenes) > 1 and se.stats["total_scenes"] > 0:
        st.divider()
        st.markdown("**Find Similar Scenes Across Videos**")
        scene_opt = [f"[{s.video_id[:8]}] {s.start_fmt} — {s.text[:50]}…" for s in all_scenes[:50]]
        sel = st.selectbox("Pick a scene", range(len(all_scenes[:50])),
                           format_func=lambda i: scene_opt[i])
        ref_scene = all_scenes[sel]
        similar = se.find_similar_scenes(ref_scene, top_k=5, exclude_same_video=True)
        if similar:
            for r in similar:
                vm2 = st.session_state.videos.get(r.scene.video_id)
                st.markdown(f"""
                <div class="search-result-card">
                    <div style='display:flex;justify-content:space-between'>
                        <span style='color:#6B7280;font-size:0.75rem'>{vm2.title[:30] if vm2 else r.scene.video_id}</span>
                        <span style='font-family:"JetBrains Mono",monospace;color:#FFAA00;font-size:0.78rem'>sim: {r.score:.3f}</span>
                    </div>
                    <div style='color:#6B7280;font-size:0.7rem;margin:3px 0'>{r.scene.start_fmt}</div>
                    <p style='color:#C4C9D8;font-size:0.83rem;margin:5px 0'>{r.scene.text[:200]}…</p>
                    {_score_bar(r.score)}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No similar scenes found in other videos. Process more videos for cross-video analysis.")


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def _get_active_video() -> Optional[VideoMetadata]:
    if st.session_state.selected_video:
        vm = st.session_state.videos.get(st.session_state.selected_video)
        if vm:
            return vm
    # Fall back to first video
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
