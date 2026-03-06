"""
Semantix — Video Intelligence Platform
Clean, simple UX. Light-ish dark theme. YouTube player built in.
"""

import re
import time
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core.video_processor import VideoProcessor, VideoMetadata, fetch_youtube_transcript, fetch_youtube_metadata
from core.scene_detector import Scene
from core.ad_engine import AdMatchingEngine, create_default_inventory
from core.search_engine import HybridSearchEngine
from core.embeddings import _IAB_NAMES

st.set_page_config(
    page_title="Semantix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design system ──────────────────────────────────────────────────────────────
# Softer dark — charcoal not pitch black, more breathing room
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ══ LIGHT THEME ══ */
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
#MainMenu, footer { visibility: hidden; }

/* Main background: clean white */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="block-container"] {
    background: #ffffff !important;
    color: #111827 !important;
}
[data-testid="block-container"] {
    padding: 1.5rem 2.5rem 3rem !important;
    max-width: 1300px !important;
}

/* All text dark */
p, span, label, div, h1, h2, h3, h4, li { color: #111827 !important; }
.stMarkdown p, .stMarkdown span { color: #374151 !important; }

/* Sidebar: very light grey */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div:first-child {
    background: #f8f9fa !important;
    border-right: 1px solid #e5e7eb !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #374151 !important; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    border: none !important;
    border-radius: 8px !important;
    color: #6b7280 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    padding: 0.55rem 1rem !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #f0f1f3 !important;
    color: #111827 !important;
    transform: none !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #fff7ed !important;
    color: #d97706 !important;
    border-left: 3px solid #f59e0b !important;
    border-radius: 0 8px 8px 0 !important;
    font-weight: 600 !important;
}

/* Main action buttons */
section.main .stButton > button,
[data-testid="stMain"] .stButton > button {
    background: #f59e0b !important;
    color: #1a1a1a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.5rem !important;
    box-shadow: 0 1px 4px rgba(245,158,11,0.3) !important;
}
section.main .stButton > button:hover,
[data-testid="stMain"] .stButton > button:hover {
    background: #d97706 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(245,158,11,0.3) !important;
}

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #f3f4f6 !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 7px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border: none !important;
    padding: 6px 16px !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #f59e0b !important;
    color: #1a1a1a !important;
    font-weight: 600 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #f9fafb !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] > div {
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    color: #9ca3af !important;
}
[data-testid="stMetricValue"] > div {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: #111827 !important;
}

/* Scene / content cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
}

/* Inputs */
input, textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    background: #ffffff !important;
    border: 1.5px solid #d1d5db !important;
    border-radius: 8px !important;
    color: #111827 !important;
}
input:focus, textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.15) !important;
}
input::placeholder, textarea::placeholder { color: #9ca3af !important; }

/* Select */
[data-baseweb="select"] > div {
    background: #ffffff !important;
    border-color: #d1d5db !important;
    color: #111827 !important;
}
[data-baseweb="popover"], [data-baseweb="menu"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1) !important;
}
[role="option"] { background: #ffffff !important; color: #111827 !important; }
[role="option"]:hover { background: #f9fafb !important; }

/* File uploader */
[data-testid="stFileUploaderDropzone"] {
    background: #fafafa !important;
    border: 2px dashed #d1d5db !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploaderDropzone"] * { color: #6b7280 !important; }
[data-testid="stFileUploaderDropzone"] button {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    color: #374151 !important;
    border-radius: 6px !important;
}

/* Progress bar */
[data-testid="stProgress"] > div > div { background: #f59e0b !important; }
[data-testid="stProgress"] > div { background: #f3f4f6 !important; }

/* Alerts */
[data-testid="stAlert"] { border-radius: 8px !important; }

/* Divider */
hr { border-color: #e5e7eb !important; }

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 8px !important;
}

/* Plotly */
[data-testid="stPlotlyChart"] {
    border: 1px solid #e5e7eb !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

/* Radio buttons */
[data-testid="stRadio"] label { color: #374151 !important; }

/* Caption text */
.stCaption, [data-testid="stCaptionContainer"] { color: #6b7280 !important; }
small { color: #6b7280 !important; }

/* YouTube embed */
.yt-container {
    position: relative;
    padding-bottom: 56.25%;
    height: 0;
    overflow: hidden;
    border-radius: 10px;
    border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.yt-container iframe {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
}

/* Sliders */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #f59e0b !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
_DEFAULTS = {
    "videos": {},
    "page": "process",
    "selected_video": None,
    "yt_api_key": "",
    "search_engine": None,
    "ad_engine": None,
    "last_yt_id": None,       # for the embedded player
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.search_engine is None:
    st.session_state.search_engine = HybridSearchEngine()
if st.session_state.ad_engine is None:
    st.session_state.ad_engine = AdMatchingEngine()

# ── Plotly theme ───────────────────────────────────────────────────────────────
_BG = "#20232d"
_GRID = "#2e3140"
_TEXT = "#6b7280"
_AMBER = "#f59e0b"

PT = dict(
    plot_bgcolor=_BG,
    paper_bgcolor=_BG,
    font=dict(family="Inter, sans-serif", color=_TEXT, size=11),
    margin=dict(l=20, r=20, t=44, b=20),
    colorway=[_AMBER, "#34d399", "#60a5fa", "#a78bfa", "#fb923c", "#f472b6"],
)
_XAXIS = dict(gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=_TEXT, size=10))
_YAXIS = dict(gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=_TEXT, size=10))

# ── Sidebar ────────────────────────────────────────────────────────────────────
NAV = [
    ("process",   "🎬", "Load Video"),
    ("watch",     "▶️",  "Watch & Explore"),
    ("search",    "🔍", "Search Moments"),
    ("ads",       "📢", "Ad Matching"),
    ("analytics", "📊", "Analytics"),
]

with st.sidebar:
    st.markdown("### ⚡ Semantix")
    st.caption("Video Intelligence")
    st.divider()

    for pid, icon, label in NAV:
        active = st.session_state.page == pid
        # Use primary kind trick for active highlight via CSS
        if active:
            st.button(f"{icon}  {label}", key=f"nav_{pid}",
                      use_container_width=True, type="primary")
        else:
            if st.button(f"{icon}  {label}", key=f"nav_{pid}",
                         use_container_width=True):
                st.session_state.page = pid
                st.rerun()

    st.divider()

    if st.session_state.videos:
        vms = list(st.session_state.videos.values())
        labels = [vm.title[:28] + ("…" if len(vm.title) > 28 else "") for vm in vms]
        sel_idx = st.selectbox("Active video", range(len(vms)),
                               format_func=lambda i: labels[i],
                               key="vm_sel", label_visibility="collapsed")
        st.session_state.selected_video = vms[sel_idx].video_id

        se = st.session_state.search_engine.stats
        c1, c2 = st.columns(2)
        c1.metric("Videos", se["total_videos"])
        c2.metric("Scenes", se["total_scenes"])

    st.divider()
    yt = st.text_input("YouTube API Key", type="password",
                        value=st.session_state.yt_api_key,
                        placeholder="Optional",
                        key="yt_key_in", label_visibility="visible")
    if yt != st.session_state.yt_api_key:
        st.session_state.yt_api_key = yt


# ── Helpers ────────────────────────────────────────────────────────────────────
def _register(vm: VideoMetadata):
    st.session_state.videos[vm.video_id] = vm
    st.session_state.search_engine.add_scenes(vm.scenes)
    if st.session_state.search_engine.vectorizer is not None:
        st.session_state.ad_engine.sync_vectorizer(
            st.session_state.search_engine.vectorizer)
    st.session_state.selected_video = vm.video_id

def _active_vm() -> Optional[VideoMetadata]:
    if st.session_state.selected_video:
        vm = st.session_state.videos.get(st.session_state.selected_video)
        if vm:
            return vm
    return next(iter(st.session_state.videos.values()), None)

def _iab_str(cats, n=2):
    return "  ·  ".join(c["name"] for c in cats[:n]) if cats else "—"

def _sent_badge(label):
    colors = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}
    return colors.get(label, "🔵")

def _yt_id(url: str) -> Optional[str]:
    for p in [r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})",
               r"^([A-Za-z0-9_-]{11})$"]:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def _parse_ts(t: str) -> Optional[int]:
    t = t.strip()
    if not t:
        return None
    if ":" in t:
        parts = t.split(":")
        try:
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except ValueError:
            return None
    try:
        return int(float(t))
    except ValueError:
        return None

def _yt_embed(vid_id: str, start_sec: int = 0):
    """Render a YouTube iframe player."""
    url = f"https://www.youtube.com/embed/{vid_id}?start={start_sec}&autoplay=0&rel=0&modestbranding=1"
    st.markdown(f"""
    <div class="yt-container">
        <iframe src="{url}" frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen></iframe>
    </div>
    """, unsafe_allow_html=True)

def _scene_row(scene: Scene, vm: VideoMetadata = None, score: float = None,
               show_jump: bool = False, yt_id: str = None):
    """Compact scene card with optional score and YouTube jump button."""
    safety = scene.brand_safety.get("safety_score", 1.0)
    sent = scene.sentiment.get("label", "neutral")
    is_key = vm and scene.scene_id in vm.key_scenes if vm else False

    with st.container(border=True):
        top_l, top_r = st.columns([6, 2])
        with top_l:
            key_star = " ⭐" if is_key else ""
            score_str = f"  —  match **{score:.0%}**" if score is not None else ""
            st.markdown(
                f"`{scene.start_fmt}→{scene.end_fmt}`  **{scene.duration_sec:.0f}s**"
                f"{key_star}{score_str}"
            )
        with top_r:
            cols = top_r.columns(3) if show_jump and yt_id else top_r.columns(2)
            cols[0].caption(f"{_sent_badge(sent)} {sent[:3]}")
            cols[1].caption(f"🛡 {safety:.0%}")

        # Main text — truncated nicely
        preview = scene.text[:200].strip()
        if len(scene.text) > 200:
            preview += "…"
        st.write(preview)

        # Tags row
        tags = _iab_str(scene.iab_categories)
        st.caption(f"**{tags}**  ·  engagement {scene.engagement_score:.2f}  ·  ad fit {scene.ad_suitability:.2f}")

        # Jump to timestamp in player
        if show_jump and yt_id:
            if st.button(f"▶ Play at {scene.start_fmt}", key=f"jump_{scene.scene_id}_{yt_id}"):
                st.session_state["player_seek"] = scene.start_sec
                st.session_state["player_yt_id"] = yt_id
                st.session_state.page = "watch"
                st.rerun()

        if score is not None:
            st.progress(min(score, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LOAD VIDEO
# ══════════════════════════════════════════════════════════════════════════════
def page_process():
    st.markdown("## 🎬 Load a Video")
    st.caption("Upload a subtitle file or paste a YouTube link to get started")
    st.divider()

    tab_yt, tab_file, tab_text = st.tabs([
        "▶️  YouTube URL", "📁  Upload SRT/VTT", "📋  Paste Text"
    ])

    # ── YouTube tab (primary, shown first) ────────────────────────────────
    with tab_yt:
        st.markdown("#### Paste a YouTube URL")
        yt_url = st.text_input("", placeholder="https://youtube.com/watch?v=...",
                               key="yt_url_in", label_visibility="collapsed")

        ca, cb = st.columns([2, 3])
        with ca:
            st.caption("SCENE DETECTION")
            min_s = st.slider("Min scene length", 10, 90, 20, 5, key="yt_min",
                              help="Minimum seconds per scene")
            max_s = st.slider("Max scene length", 60, 300, 120, 10, key="yt_max")
            sens = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="yt_sens",
                             help="Higher = fewer, larger scenes")

        with cb:
            if yt_url:
                vid_id = _yt_id(yt_url)
                if vid_id:
                    st.caption("PREVIEW")
                    _yt_embed(vid_id)
                else:
                    st.warning("Couldn't parse a video ID from that URL.")

        if yt_url and st.button("⚡ Analyse This Video", key="proc_yt"):
            vid_id = _yt_id(yt_url)
            if not vid_id:
                st.error("Invalid URL.")
                return
            with st.spinner("Fetching transcript from YouTube…"):
                transcript = fetch_youtube_transcript(vid_id)
            if transcript is None:
                st.error("No captions found. Try a video with auto-generated subtitles enabled.")
                return
            meta = None
            if st.session_state.yt_api_key:
                with st.spinner("Fetching video metadata…"):
                    meta = fetch_youtube_metadata(vid_id, st.session_state.yt_api_key)
            title = meta.get("title", f"YouTube · {vid_id}") if meta else f"YouTube · {vid_id}"
            with st.spinner(f'Detecting scenes in "{title}"…'):
                t0 = time.time()
                vm = VideoProcessor(min_s, max_s, sens).process_youtube_transcript(
                    transcript, vid_id, title, meta)
                elapsed = time.time() - t0
            _register(vm)
            st.session_state.last_yt_id = vid_id
            st.success(f"✅  Found **{len(vm.scenes)} scenes** in {elapsed:.1f}s")
            _summary_strip(vm)
            st.info("👉  Go to **Watch & Explore** to play the video alongside every scene.")

    # ── Upload tab ─────────────────────────────────────────────────────────
    with tab_file:
        st.markdown("#### Upload a subtitle file")
        st.caption("Supported: .srt (SubRip) and .vtt (WebVTT)")
        ca, cb = st.columns([3, 2], gap="large")
        with ca:
            uploaded = st.file_uploader("Drop your subtitle file here", type=["srt", "vtt"],
                                         key="up_file")
            title_f = st.text_input("Video title", placeholder="Leave blank to use filename",
                                    key="up_title")
            yt_link = st.text_input("YouTube URL (optional — to enable the player)",
                                    placeholder="https://youtube.com/watch?v=...",
                                    key="up_yt")
        with cb:
            st.caption("DETECTION SETTINGS")
            min_f = st.slider("Min scene (s)", 10, 90, 20, 5, key="f_min")
            max_f = st.slider("Max scene (s)", 60, 300, 120, 10, key="f_max")
            sens_f = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="f_sens")

        if uploaded and st.button("⚡ Process File", key="proc_file"):
            content = uploaded.read().decode("utf-8", errors="replace")
            fmt = "vtt" if uploaded.name.lower().endswith(".vtt") else "srt"
            title = title_f or uploaded.name
            with st.spinner("Analysing scenes…"):
                t0 = time.time()
                vm = VideoProcessor(min_f, max_f, sens_f).process_file(content, title, fmt)
                elapsed = time.time() - t0
            if not vm.scenes:
                st.error("No scenes detected — check file format.")
                return
            _register(vm)
            if yt_link:
                vid_id = _yt_id(yt_link)
                if vid_id:
                    st.session_state.last_yt_id = vid_id
                    # Store yt_id against this video
                    vm.yt_id = vid_id
            st.success(f"✅  **{len(vm.scenes)} scenes** detected in {elapsed:.1f}s")
            _summary_strip(vm)

    # ── Paste tab ──────────────────────────────────────────────────────────
    with tab_text:
        st.markdown("#### Paste subtitle content")
        ca, cb = st.columns([3, 2], gap="large")
        with ca:
            title_p = st.text_input("Title", placeholder="My Video", key="p_title")
            pasted = st.text_area("", height=220,
                                  placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello world...",
                                  key="p_text", label_visibility="collapsed")
        with cb:
            st.caption("DETECTION SETTINGS")
            min_p = st.slider("Min scene (s)", 10, 90, 20, 5, key="p_min")
            max_p = st.slider("Max scene (s)", 60, 300, 120, 10, key="p_max")

        if pasted and st.button("⚡ Process Text", key="proc_paste"):
            with st.spinner("Processing…"):
                t0 = time.time()
                vm = VideoProcessor(min_p, max_p).process_file(pasted, title_p or "Pasted Video")
                elapsed = time.time() - t0
            if not vm.scenes:
                st.error("No scenes found — check format.")
                return
            _register(vm)
            st.success(f"✅  **{len(vm.scenes)} scenes** in {elapsed:.1f}s")
            _summary_strip(vm)


def _summary_strip(vm: VideoMetadata):
    """4-metric strip shown after processing."""
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenes", vm.scene_count)
    c2.metric("Duration", vm.fmt_duration())
    c3.metric("Total Cues", vm.total_cues)
    avg = round(sum(s.duration_sec for s in vm.scenes) / max(vm.scene_count, 1), 0)
    c4.metric("Avg Scene", f"{avg:.0f}s")
    if vm.dominant_iab:
        st.caption("**Top topics:** " + "  ·  ".join(c["name"] for c in vm.dominant_iab[:4]))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — WATCH & EXPLORE  (YouTube player + scene list side-by-side)
# ══════════════════════════════════════════════════════════════════════════════
def page_watch():
    st.markdown("## ▶️ Watch & Explore")
    st.caption("Play the video and click any scene to jump directly to that moment")
    st.divider()

    if not st.session_state.videos:
        st.info("No video loaded yet — go to **Load Video** first.")
        return

    vm = _active_vm()
    if not vm:
        return

    # Try to find a YouTube ID for this video
    yt_id = getattr(vm, "yt_id", None) or st.session_state.get("last_yt_id")

    start_sec = st.session_state.pop("player_seek", 0)

    if yt_id:
        # Side-by-side: player left, scene list right
        col_player, col_scenes = st.columns([5, 3], gap="large")

        with col_player:
            st.caption(f"**{vm.title}**  ·  {vm.fmt_duration()}  ·  {vm.scene_count} scenes")
            _yt_embed(yt_id, start_sec)
            st.caption("💡 Click **▶ Play at** buttons on any scene to jump to that moment")

            # Mini stats under player
            st.divider()
            c1, c2, c3 = st.columns(3)
            c1.metric("Scenes", vm.scene_count)
            c2.metric("Duration", vm.fmt_duration())
            avg_eng = round(sum(s.engagement_score for s in vm.scenes) / max(vm.scene_count, 1), 2)
            c3.metric("Avg Engagement", avg_eng)

            # Emotional arc
            if vm.emotional_arc:
                df = pd.DataFrame(vm.emotional_arc)
                fig = px.area(df, x="start_sec", y="sentiment_score",
                              color_discrete_sequence=[_AMBER])
                fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.08)")
                fig.update_layout(
                    **PT, height=150, showlegend=False,
                    xaxis=dict(**_XAXIS, title=""),
                    yaxis=dict(**_YAXIS, title=""),
                    title=dict(text="Emotional Arc", font=dict(size=11, color=_TEXT)),
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_scenes:
            st.caption("SCENES — click to jump in player")
            # Filter controls
            fc1, fc2 = st.columns(2)
            with fc1:
                sent_filter = st.multiselect("Sentiment",
                    ["positive", "neutral", "negative"],
                    default=["positive", "neutral", "negative"],
                    key="w_sent", label_visibility="collapsed",
                    placeholder="Sentiment…")
            with fc2:
                min_suit = st.slider("Min ad fit", 0.0, 1.0, 0.0, 0.1,
                                     key="w_suit", label_visibility="collapsed")

            filtered = [s for s in vm.scenes
                        if s.sentiment.get("label", "neutral") in sent_filter
                        and s.ad_suitability >= min_suit]
            st.caption(f"**{len(filtered)}** of {vm.scene_count} scenes")
            for scene in filtered:
                _scene_row(scene, vm, show_jump=True, yt_id=yt_id)
    else:
        # No YouTube ID — show scene list with text only
        st.info("**No YouTube player available** — this video was loaded from a file. "
                "To enable the player, re-process via the **YouTube URL** tab or add the URL when uploading.")
        st.divider()
        st.markdown(f"### {vm.title}  ·  {vm.fmt_duration()}")
        for scene in vm.scenes:
            _scene_row(scene, vm)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SEARCH MOMENTS
# ══════════════════════════════════════════════════════════════════════════════
def page_search():
    st.markdown("## 🔍 Search Moments")
    st.caption("Find scenes by topic, emotion, keyword — or jump to any timestamp")
    st.divider()

    se = st.session_state.search_engine
    if se.stats["total_scenes"] == 0:
        st.info("No videos indexed yet — load a video first.")
        return

    mode = st.radio("", ["🧠 Describe what you're looking for",
                         "🏷️ Browse by tags", "⏱️ Jump to timestamp"],
                    horizontal=True, key="search_mode",
                    label_visibility="collapsed")
    st.divider()

    vm = _active_vm()
    yt_id = getattr(vm, "yt_id", None) or st.session_state.get("last_yt_id") if vm else None

    if "Describe" in mode:
        _semantic_search(se, yt_id)
    elif "tags" in mode:
        _tag_search(yt_id)
    else:
        _timestamp_jump(vm, yt_id)


def _semantic_search(se, yt_id=None):
    # Use staging key so chips can populate the input before it renders
    if "sem_q_staged" not in st.session_state:
        st.session_state["sem_q_staged"] = ""

    default_val = st.session_state.pop("sem_q_staged", "")
    query = st.text_input("",
        placeholder="Try: 'tense confrontation'  /  'product demo'  /  'emotional speech'  /  'cricket six'",
        key="sem_q", value=default_val,
        label_visibility="collapsed")

    ca, cb, cc = st.columns([2, 1, 1])
    with ca:
        top_k = st.select_slider("Results", [3, 5, 10, 20], value=5, key="sem_k")
    with cb:
        safety_th = st.selectbox("Brand safety",
            ["Any", "Moderate (50%+)", "Strict (80%+)"],
            key="sem_safe", label_visibility="collapsed")
        smap = {"Any": 0.0, "Moderate (50%+)": 0.5, "Strict (80%+)": 0.8}
    with cc:
        diversify = st.checkbox("Diversify", value=True, key="sem_div")

    if not query:
        # Show suggested queries as chips
        st.caption("**Try one of these:**")
        examples = [
            "cricket six or four", "emotional celebration", "expert interview",
            "product review", "dramatic moment", "audience reaction",
            "landscape wide shot", "breaking news update", "comedy sketch",
            "behind the scenes", "tutorial walkthrough", "conflict argument"
        ]
        cols = st.columns(4)
        for i, ex in enumerate(examples):
            if cols[i % 4].button(ex, key=f"ex_{i}"):
                st.session_state["sem_q_staged"] = ex
                st.rerun()
        return

    with st.spinner("Searching…"):
        results = se.search(query, top_k=top_k, diversify=diversify,
                            min_safety=smap[safety_th], expand=True)

    if not results:
        st.warning("No matching scenes. Try different words or remove the safety filter.")
        return

    st.success(f"**{len(results)} scenes** matched for *{query}*")
    st.divider()

    for r in results:
        scene = r.scene
        _vm = st.session_state.videos.get(scene.video_id)
        _yt = getattr(_vm, "yt_id", None) or yt_id
        if _vm and len(st.session_state.videos) > 1:
            st.caption(f"📹 {_vm.title[:50]}")
        _scene_row(scene, _vm, score=r.score, show_jump=True, yt_id=_yt)


def _tag_search(yt_id=None):
    st.markdown("#### Browse scenes by category or sentiment")

    ca, cb, cc = st.columns(3)
    with ca:
        iab_sel = st.multiselect("Content category",
            [f"{k}: {v}" for k, v in list(_IAB_NAMES.items())[:30]],
            placeholder="Any category…", key="tag_iab",
            label_visibility="visible")
    with cb:
        sent_f = st.multiselect("Sentiment",
            ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"],
            key="tag_sent", label_visibility="visible")
    with cc:
        min_fit = st.slider("Min ad fit", 0.0, 1.0, 0.0, 0.1,
                            key="tag_fit", label_visibility="visible")

    kw = st.text_input("Keyword in text", placeholder="e.g. cricket  /  revenue  /  love",
                       key="tag_kw")

    iab_codes = [x.split(":")[0].strip() for x in iab_sel]
    filtered = []
    for _vm in st.session_state.videos.values():
        for s in _vm.scenes:
            if s.sentiment.get("label", "neutral") not in sent_f:
                continue
            if s.ad_suitability < min_fit:
                continue
            if iab_codes:
                s_codes = [c.get("iab_code", c.get("id", "")) for c in s.iab_categories]
                if not any(code in s_codes for code in iab_codes):
                    continue
            if kw and kw.lower() not in s.text.lower():
                continue
            filtered.append((_vm, s))

    st.caption(f"**{len(filtered)}** scenes match")
    for _vm, scene in filtered[:40]:
        _scene_row(scene, _vm, show_jump=True, yt_id=getattr(_vm, "yt_id", None) or yt_id)


def _timestamp_jump(vm, yt_id=None):
    if not vm:
        st.info("No active video.")
        return

    st.markdown(f"#### {vm.title}  ·  {vm.fmt_duration()}")

    ca, cb = st.columns(2)
    with ca:
        ts_from = st.text_input("From", placeholder="00:01:30  or  90",
                                key="ts_from", label_visibility="visible")
    with cb:
        ts_to = st.text_input("To (optional)", placeholder="00:05:00  or  300",
                              key="ts_to", label_visibility="visible")

    start_s = _parse_ts(ts_from) if ts_from else None
    end_s = _parse_ts(ts_to) if ts_to else None

    shown = 0
    for scene in vm.scenes:
        if start_s is not None and scene.end_sec < start_s:
            continue
        if end_s is not None and scene.start_sec > end_s:
            continue
        _scene_row(scene, vm, show_jump=True, yt_id=yt_id)
        shown += 1
        if shown >= 25:
            st.caption(f"Showing first 25 matching scenes")
            break

    if shown == 0:
        st.info("No scenes in that time range.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — AD MATCHING  (simplified — scene picker + top ads)
# ══════════════════════════════════════════════════════════════════════════════
def page_ads():
    st.markdown("## 📢 Ad Matching")
    st.caption("Pick any scene and instantly see which ads fit best")
    st.divider()

    if not st.session_state.videos:
        st.info("Load a video first.")
        return

    vm = _active_vm()
    if not vm:
        return
    ae = st.session_state.ad_engine

    tab1, tab2, tab3 = st.tabs(["🎯  Match a Scene", "📅  Placement Plan", "📦  Ad Inventory"])

    # ── Match a scene ──────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Select a scene to find matching ads")
        scene_opts = [f"{s.start_fmt} — {s.text[:65]}…" for s in vm.scenes]
        sel_idx = st.selectbox("Scene", range(len(vm.scenes)),
                               format_func=lambda i: scene_opts[i],
                               key="ad_scene_sel", label_visibility="collapsed")
        scene = vm.scenes[sel_idx]

        # Show the selected scene
        with st.container(border=True):
            st.markdown(f"**Selected:** `{scene.start_fmt} → {scene.end_fmt}`  ·  {scene.duration_sec:.0f}s")
            st.write(scene.text[:250] + "…")
            ca, cb, cc = st.columns(3)
            safety = scene.brand_safety.get("safety_score", 1.0)
            sent = scene.sentiment.get("label", "neutral")
            ca.caption(f"{_sent_badge(sent)} {sent.title()}")
            cb.caption(f"🛡 Brand safety: {safety:.0%}")
            cc.caption(f"Ad fit: {scene.ad_suitability:.2f}")

        st.divider()
        st.caption("TOP MATCHING ADS")

        matches = ae.match_ads(scene, top_k=5)
        if not matches:
            st.warning("No eligible ads for this scene.")
        else:
            for i, (ad, si) in enumerate(matches, 1):
                with st.container(border=True):
                    h1, h2 = st.columns([5, 1])
                    with h1:
                        st.markdown(f"**{i}. {ad.title}** — *{ad.brand}*")
                        st.caption(ad.description)
                    with h2:
                        st.metric("Match", f"{si['total']:.0%}")

                    ca, cb, cc, cd = st.columns(4)
                    ca.caption(f"Content `{si['content_sim']:.2f}`")
                    cb.caption(f"Category `{si['iab_match']:.2f}`")
                    cc.caption(f"Safety `{si['safety']:.2f}`")
                    cd.caption(f"Perf `{si['performance']:.2f}`")
                    st.progress(min(si["total"], 1.0))

    # ── Placement plan ─────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Auto-generate a full placement plan")
        ca, cb = st.columns([2, 1])
        with ca:
            p_types = st.multiselect("Include placement types",
                ["pre-roll", "mid-roll", "post-roll"],
                default=["pre-roll", "mid-roll", "post-roll"],
                key="pl_types")
        with cb:
            min_safe = st.slider("Min brand safety", 0.0, 1.0, 0.5, 0.1, key="pl_safe")

        if st.button("⚡ Generate Plan", key="gen_plan"):
            with st.spinner("Optimising placements…"):
                placements = ae.plan_placements(vm.scenes, vm.duration_ms, p_types)
                perf = ae.simulate_performance(placements)
                st.session_state["_pl"] = placements
                st.session_state["_perf"] = perf

        if st.session_state.get("_pl"):
            pl = st.session_state["_pl"]
            perf = st.session_state["_perf"]
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Placements", perf["total_placements"])
            c2.metric("Est. Revenue", f"${perf['total_revenue_usd']:.2f}")
            c3.metric("Impressions", f"{perf['total_impressions']:,}")
            c4.metric("Clicks", f"{perf['estimated_clicks']:,}")
            c5.metric("Avg CPM", f"${perf['avg_cpm']:.2f}")

            p_df = pd.DataFrame([p.to_dict() for p in pl])
            show = ["timestamp_fmt", "placement_type", "ad_title", "brand",
                    "relevance_score", "estimated_cpm"]
            st.dataframe(p_df[show].rename(columns={
                "timestamp_fmt": "Time", "placement_type": "Type",
                "ad_title": "Ad", "brand": "Brand",
                "relevance_score": "Relevance", "estimated_cpm": "CPM ($)"}),
                use_container_width=True, hide_index=True)

    # ── Inventory ─────────────────────────────────────────────────────────
    with tab3:
        inv_df = pd.DataFrame([ad.to_dict() for ad in ae.inventory])
        keep = ["title", "brand", "cpm_base", "historical_ctr",
                "performance_score", "brand_safety_min"]
        st.dataframe(inv_df[keep].rename(columns={
            "title": "Ad", "brand": "Brand", "cpm_base": "CPM",
            "historical_ctr": "CTR", "performance_score": "Score",
            "brand_safety_min": "Min Safety"}),
            use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ANALYTICS  (clean, no PT["yaxis"] conflict)
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("## 📊 Analytics")
    st.caption("Content intelligence overview across all your videos")
    st.divider()

    if not st.session_state.videos:
        st.info("Load a video first.")
        return

    all_s = [s for vm in st.session_state.videos.values() for s in vm.scenes]
    n = max(len(all_s), 1)

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Videos", len(st.session_state.videos))
    c2.metric("Total Scenes", n)
    c3.metric("Avg Engagement",
              round(sum(s.engagement_score for s in all_s) / n, 3))
    safe_n = sum(1 for s in all_s if s.brand_safety.get("safety_score", 1.0) >= 0.8)
    c4.metric("Brand Safe", f"{safe_n / n:.0%}")
    pos_n = sum(1 for s in all_s if s.sentiment.get("label") == "positive")
    c5.metric("Positive", f"{pos_n / n:.0%}")

    st.divider()
    ca, cb = st.columns(2)

    # Sentiment donut
    with ca:
        labels = [s.sentiment.get("label", "neutral") for s in all_s]
        lc = {l: labels.count(l) for l in set(labels)}
        fig = go.Figure(go.Pie(
            values=list(lc.values()),
            labels=list(lc.keys()),
            hole=0.55,
            marker_colors={"positive": "#34d399", "neutral": "#4b5563",
                           "negative": "#f87171"}.values()
        ))
        fig.update_layout(
            **PT,
            height=260,
            title=dict(text="Sentiment", font=dict(size=12, color=_TEXT)),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#9ca3af")),
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # IAB categories bar
    with cb:
        iab_all: dict[str, int] = {}
        for s in all_s:
            for cat in s.iab_categories[:1]:
                iab_all[cat["name"]] = iab_all.get(cat["name"], 0) + 1
        if iab_all:
            top = sorted(iab_all.items(), key=lambda x: x[1], reverse=True)[:8]
            df2 = pd.DataFrame(top, columns=["Category", "Scenes"])
            fig = go.Figure(go.Bar(
                x=df2["Scenes"],
                y=df2["Category"],
                orientation="h",
                marker_color=_AMBER,
            ))
            fig.update_layout(
                **PT,
                height=260,
                title=dict(text="Top Categories", font=dict(size=12, color=_TEXT)),
                xaxis=dict(**_XAXIS, title=""),
                yaxis=dict(**_YAXIS, autorange="reversed", title=""),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Engagement vs Safety scatter
    ca, cb = st.columns(2)
    with ca:
        eng_v = [s.engagement_score for s in all_s]
        safe_v = [s.brand_safety.get("safety_score", 1.0) for s in all_s]
        suit_v = [s.ad_suitability for s in all_s]
        fig = go.Figure(go.Scatter(
            x=eng_v, y=safe_v, mode="markers",
            marker=dict(
                size=[v * 14 + 5 for v in suit_v],
                color=suit_v,
                colorscale=[[0, "#252830"], [1, _AMBER]],
                showscale=True, opacity=0.75,
                colorbar=dict(title="Ad Fit", tickfont=dict(color=_TEXT)),
            ),
            hovertemplate="eng: %{x:.2f}<br>safety: %{y:.2f}<extra></extra>",
        ))
        fig.update_layout(
            **PT, height=280,
            title=dict(text="Engagement vs Safety  (bubble = ad fit)",
                       font=dict(size=12, color=_TEXT)),
            xaxis=dict(**_XAXIS, title="Engagement"),
            yaxis=dict(**_YAXIS, title="Safety"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        fig = go.Figure(go.Histogram(
            x=[s.ad_suitability for s in all_s],
            nbinsx=12,
            marker_color="#fb923c",
        ))
        fig.update_layout(
            **PT, height=280, showlegend=False,
            title=dict(text="Ad Suitability Distribution",
                       font=dict(size=12, color=_TEXT)),
            xaxis=dict(**_XAXIS, title="Ad Suitability"),
            yaxis=dict(**_YAXIS, title="Scenes"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Per-video table
    st.divider()
    st.caption("PER-VIDEO BREAKDOWN")
    rows = []
    for vm in st.session_state.videos.values():
        if not vm.scenes:
            continue
        sc = vm.scenes
        rows.append({
            "Title": vm.title[:40],
            "Duration": vm.fmt_duration(),
            "Scenes": vm.scene_count,
            "Narrative": vm.narrative_structure.split("(")[0].strip(),
            "Avg Engagement": round(sum(s.engagement_score for s in sc) / len(sc), 3),
            "Brand Safe": f"{sum(s.brand_safety.get('safety_score', 1) >= 0.8 for s in sc) / len(sc):.0%}",
            "Top Topic": vm.dominant_iab[0]["name"] if vm.dominant_iab else "—",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Router ─────────────────────────────────────────────────────────────────────
pages = {
    "process":   page_process,
    "watch":     page_watch,
    "search":    page_search,
    "ads":       page_ads,
    "analytics": page_analytics,
}
pages.get(st.session_state.page, page_process)()
