"""
Semantix — Enterprise Video Intelligence Platform
app.py  — Main Streamlit application
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

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Semantix",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Force dark base everywhere ── */
html, body { background-color: #0c0d10 !important; color: #d4d8e1 !important; }
[class*="css"] { font-family: 'Inter', sans-serif !important; }

/* Main app background */
.stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background-color: #0c0d10 !important;
}

/* Block container */
[data-testid="block-container"] {
    background-color: #0c0d10 !important;
    padding: 1.75rem 2.25rem 3rem !important;
    max-width: 1380px !important;
}

/* All text defaults */
p, span, label, div, h1, h2, h3, h4, h5, h6 {
    color: #d4d8e1 !important;
}

/* Hide default chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div:first-child,
section[data-testid="stSidebar"] {
    min-width: 240px !important;
    max-width: 260px !important;
    display: block !important;
    visibility: visible !important;
    background-color: #111318 !important;
    border-right: 1px solid #22252e !important;
}
[data-testid="stSidebar"] * { color: #d4d8e1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f0f2f5 !important; }
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small { color: #6b7280 !important; }

/* Sidebar nav buttons */
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    background: transparent !important;
    color: #8b909e !important;
    border: none !important;
    border-radius: 6px !important;
    padding: 0.45rem 0.75rem !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #1e2029 !important;
    color: #e8eaf0 !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Active nav item */
.nav-active button {
    background: rgba(245,158,11,0.12) !important;
    color: #f59e0b !important;
    border-left: 2px solid #f59e0b !important;
    padding-left: calc(0.75rem - 2px) !important;
}

/* Main buttons */
.stButton > button {
    background: #f59e0b !important;
    color: #0c0d10 !important;
    border: none !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.25) !important;
}
.stButton > button:hover {
    background: #fbbf24 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(245,158,11,0.3) !important;
}

/* ── Inputs ── */
input, textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea,
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea {
    background-color: #16181d !important;
    border: 1px solid #272a35 !important;
    border-radius: 7px !important;
    color: #d4d8e1 !important;
}
input:focus, textarea:focus {
    border-color: #f59e0b !important;
    box-shadow: 0 0 0 3px rgba(245,158,11,0.1) !important;
}
input::placeholder, textarea::placeholder { color: #4a5060 !important; }

/* ── Selectbox / Multiselect ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div,
[data-baseweb="select"] > div {
    background-color: #16181d !important;
    border: 1px solid #272a35 !important;
    border-radius: 7px !important;
    color: #d4d8e1 !important;
}
[data-baseweb="popover"] { background-color: #1e2028 !important; }
[data-baseweb="menu"] { background-color: #1e2028 !important; }
[role="option"] { background-color: #1e2028 !important; color: #d4d8e1 !important; }
[role="option"]:hover { background-color: #272a35 !important; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #16181d !important;
    border: 1px solid #22252e !important;
    border-radius: 8px !important;
    padding: 3px !important;
    gap: 2px !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b7280 !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 5px 14px !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #f59e0b !important;
    color: #0c0d10 !important;
    font-weight: 600 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #13151a !important;
    border: 1px solid #22252e !important;
    border-radius: 8px !important;
    padding: 0.9rem 1.1rem !important;
}
[data-testid="stMetricLabel"] {
    color: #6b7280 !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.07em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #f0f2f5 !important;
    font-size: 1.45rem !important;
    font-weight: 500 !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border: 1px solid #22252e !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"],
[data-testid="stFileUploaderDropzone"] {
    background-color: #16181d !important;
    border: 1.5px dashed #2a2d38 !important;
    border-radius: 8px !important;
    color: #8b909e !important;
}
[data-testid="stFileUploader"] * { color: #8b909e !important; }
[data-testid="stFileUploader"] button {
    background-color: #272a35 !important;
    color: #d4d8e1 !important;
    border-radius: 6px !important;
    border: 1px solid #363a47 !important;
}

/* Plotly charts */
[data-testid="stPlotlyChart"] {
    border: 1px solid #22252e !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* Divider */
hr { border-color: #22252e !important; }

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: #f59e0b !important;
}

/* Success / error / info */
[data-testid="stAlert"] { border-radius: 7px !important; }

/* Progress */
[data-testid="stProgress"] > div > div {
    background: #f59e0b !important;
}

/* Main container */
[data-testid="block-container"] {
    padding: 1.75rem 2.25rem 3rem !important;
    max-width: 1380px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "videos": {},
        "search_engine": HybridSearchEngine(),
        "ad_engine": AdMatchingEngine(),
        "page": "process",
        "selected_video": None,
        "yt_api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
_init()

# ── Plotly theme ───────────────────────────────────────────────────────────────
PT = dict(
    plot_bgcolor="#13151a",
    paper_bgcolor="#13151a",
    font=dict(family="JetBrains Mono, monospace", color="#6b7280", size=11),
    margin=dict(l=16, r=16, t=44, b=16),
    colorway=["#f59e0b", "#34d399", "#60a5fa", "#a78bfa", "#fb923c", "#f472b6"],
    xaxis=dict(gridcolor="#1e2028", linecolor="#272a35", tickfont=dict(color="#6b7280", size=10)),
    yaxis=dict(gridcolor="#1e2028", linecolor="#272a35", tickfont=dict(color="#6b7280", size=10)),
    hoverlabel=dict(bgcolor="#1e2028", bordercolor="#2a2d38", font=dict(family="JetBrains Mono", size=11, color="#d4d8e1")),
)

# ── Sidebar ────────────────────────────────────────────────────────────────────
NAV = [
    ("process",   "🎬", "Process Video"),
    ("search",    "🔍", "Semantic Search"),
    ("scenes",    "📺", "Scene Explorer"),
    ("ads",       "📢", "Ad Engine"),
    ("analytics", "📊", "Analytics"),
    ("franchise", "🎯", "Franchise Intel"),
]

with st.sidebar:
    st.markdown("### ⚡ SEMANTIX")
    st.caption("Enterprise Video Intelligence")
    st.divider()

    for pid, icon, label in NAV:
        active = st.session_state.page == pid
        btn_label = f"**{icon}  {label}**" if active else f"{icon}  {label}"
        if active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        if st.button(btn_label, key=f"nav_{pid}", use_container_width=True):
            st.session_state.page = pid
            st.rerun()
        if active:
            st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    if st.session_state.videos:
        video_options = ["— All Videos —"] + [
            (v.title[:30] + "…" if len(v.title) > 30 else v.title)
            for v in st.session_state.videos.values()
        ]
        video_ids = [None] + list(st.session_state.videos.keys())
        sel = st.selectbox("Active Video", video_options, key="video_selector", label_visibility="visible")
        st.session_state.selected_video = video_ids[video_options.index(sel)]

    se = st.session_state.search_engine.stats
    if se["total_scenes"] > 0:
        st.divider()
        st.caption("INDEX STATS")
        col1, col2 = st.columns(2)
        col1.metric("Videos", se["total_videos"])
        col2.metric("Scenes", se["total_scenes"])

    st.divider()
    yt = st.text_input("YouTube API Key", type="password",
                        value=st.session_state.yt_api_key,
                        placeholder="Optional", key="yt_key_input")
    if yt != st.session_state.yt_api_key:
        st.session_state.yt_api_key = yt

# ── Helpers ────────────────────────────────────────────────────────────────────
def _register_video(vm: VideoMetadata):
    st.session_state.videos[vm.video_id] = vm
    st.session_state.search_engine.add_scenes(vm.scenes)
    if st.session_state.search_engine.vectorizer is not None:
        st.session_state.ad_engine.sync_vectorizer(st.session_state.search_engine.vectorizer)
    st.session_state.selected_video = vm.video_id

def _get_active_video() -> Optional[VideoMetadata]:
    if st.session_state.selected_video:
        vm = st.session_state.videos.get(st.session_state.selected_video)
        if vm:
            return vm
    return next(iter(st.session_state.videos.values()), None)

def _safety_color(s: float) -> str:
    return "normal" if s >= 0.8 else ("off" if s >= 0.5 else "inverse")

def _sent_icon(label: str) -> str:
    return {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}.get(label, "⚪")

def _iab_list(iab: list[dict], n: int = 3) -> str:
    return "  ·  ".join(cat["name"] for cat in iab[:n]) if iab else "—"

def _fmt_score(v: float) -> str:
    return f"{v:.3f}"

def _extract_yt_id(url: str) -> Optional[str]:
    for p in [r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", r"^([A-Za-z0-9_-]{11})$"]:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None

def _scene_card(scene: Scene, is_key: bool = False, rank: int | None = None, score: float | None = None):
    """Render a scene using only native Streamlit elements."""
    safety = scene.brand_safety.get("safety_score", 1.0)
    sent = scene.sentiment.get("label", "neutral")
    
    with st.container(border=True):
        # Header row
        h1, h2, h3 = st.columns([3, 1, 1])
        with h1:
            prefix = f"**#{rank}**  " if rank else ""
            key_tag = "  ⭐ Key" if is_key else ""
            st.markdown(f"{prefix}`{scene.start_fmt} → {scene.end_fmt}`  ·  {scene.duration_sec:.0f}s{key_tag}")
        with h2:
            st.caption(f"🛡 Safety: **{safety:.0%}**")
        with h3:
            score_str = f"  ·  score **{score:.3f}**" if score is not None else ""
            st.caption(f"{_sent_icon(sent)} {sent.title()}{score_str}")

        # Text
        st.write(scene.text[:280] + ("…" if len(scene.text) > 280 else ""))

        # Tags row
        tags = _iab_list(scene.iab_categories)
        ad_suit = scene.ad_suitability
        eng = scene.engagement_score
        st.caption(f"**{tags}**  ·  engagement `{eng:.2f}`  ·  ad suitability `{ad_suit:.2f}`")
        st.progress(ad_suit)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: PROCESS VIDEO
# ══════════════════════════════════════════════════════════════════════════════
def page_process():
    st.markdown("## 🎬 Process Video")
    st.caption("Upload subtitles or fetch from YouTube to extract semantic scenes")
    st.divider()

    tab1, tab2, tab3 = st.tabs(["📁  Upload File", "▶  YouTube URL", "📋  Paste Text"])

    with tab1:
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            uploaded = st.file_uploader("Subtitle File (SRT or VTT)", type=["srt", "vtt"])
            title1 = st.text_input("Video Title", placeholder="Leave blank to use filename", key="t1")
        with c2:
            st.caption("DETECTION SETTINGS")
            min1 = st.slider("Min scene (s)", 10, 60, 20, key="m1")
            max1 = st.slider("Max scene (s)", 60, 300, 120, key="mx1")
            thr1 = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="th1",
                             help="Higher = fewer, more distinct scenes")

        if uploaded:
            if st.button("Process Subtitles →", key="proc1"):
                content = uploaded.read().decode("utf-8", errors="replace")
                fmt = "vtt" if uploaded.name.lower().endswith(".vtt") else "srt"
                title = title1 or uploaded.name
                with st.spinner("Detecting scenes…"):
                    t0 = time.time()
                    vm = VideoProcessor(min1, max1, thr1).process_file(content, title, fmt)
                    elapsed = time.time() - t0
                if not vm.scenes:
                    st.error("No scenes detected. Check file format.")
                    return
                _register_video(vm)
                st.success(f"✓ {len(vm.scenes)} scenes in {elapsed:.2f}s")
                _show_summary(vm)

    with tab2:
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            yt_url = st.text_input("YouTube URL or Video ID", placeholder="https://youtube.com/watch?v=...")
        with c2:
            st.caption("SETTINGS")
            min2 = st.slider("Min scene (s)", 10, 60, 20, key="m2")
            max2 = st.slider("Max scene (s)", 60, 300, 120, key="mx2")
            thr2 = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="th2")

        if yt_url and st.button("Fetch & Process →", key="proc2"):
            vid_id = _extract_yt_id(yt_url)
            if not vid_id:
                st.error("Could not parse YouTube video ID.")
                return
            with st.spinner("Fetching transcript…"):
                transcript = fetch_youtube_transcript(vid_id)
            if transcript is None:
                st.error("No transcript available for this video.")
                return
            meta = None
            if st.session_state.yt_api_key:
                with st.spinner("Fetching metadata…"):
                    meta = fetch_youtube_metadata(vid_id, st.session_state.yt_api_key)
            title = meta.get("title", f"YouTube: {vid_id}") if meta else f"YouTube: {vid_id}"
            with st.spinner("Detecting scenes…"):
                t0 = time.time()
                vm = VideoProcessor(min2, max2, thr2).process_youtube_transcript(transcript, vid_id, title, meta)
                elapsed = time.time() - t0
            _register_video(vm)
            st.success(f"✓ {len(vm.scenes)} scenes in {elapsed:.2f}s")
            _show_summary(vm)

    with tab3:
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            title3 = st.text_input("Video Title", placeholder="My Video", key="t3")
            pasted = st.text_area("Paste SRT or VTT content", height=200,
                                  placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello...")
        with c2:
            st.caption("SETTINGS")
            min3 = st.slider("Min scene (s)", 10, 60, 20, key="m3")
            max3 = st.slider("Max scene (s)", 60, 300, 120, key="mx3")

        if pasted and st.button("Process Text →", key="proc3"):
            with st.spinner("Processing…"):
                t0 = time.time()
                vm = VideoProcessor(min3, max3).process_file(pasted, title3 or "Pasted Subtitles")
                elapsed = time.time() - t0
            if not vm.scenes:
                st.error("No scenes detected. Check format.")
                return
            _register_video(vm)
            st.success(f"✓ {len(vm.scenes)} scenes in {elapsed:.2f}s")
            _show_summary(vm)


def _show_summary(vm: VideoMetadata):
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Scenes", vm.scene_count)
    c2.metric("Duration", vm.fmt_duration())
    c3.metric("Total Cues", vm.total_cues)
    avg_dur = round(sum(s.duration_sec for s in vm.scenes) / max(vm.scene_count, 1), 1)
    c4.metric("Avg Scene", f"{avg_dur}s")

    c5, c6 = st.columns(2)
    with c5:
        st.markdown(f"**Narrative:** `{vm.narrative_structure}`")
        if vm.dominant_iab:
            cats = "  ·  ".join(c["name"] for c in vm.dominant_iab[:4])
            st.caption(f"Topics: {cats}")
    with c6:
        if vm.emotional_arc:
            df = pd.DataFrame(vm.emotional_arc)
            fig = px.area(df, x="start_sec", y="sentiment_score",
                          color_discrete_sequence=["#f59e0b"])
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.15)")
            fig.update_layout(**PT, height=160, showlegend=False,
                              title=dict(text="Emotional Arc", font=dict(size=11, color="#6b7280")),
                              xaxis_title="", yaxis_title="")
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: SEARCH
# ══════════════════════════════════════════════════════════════════════════════
def page_search():
    st.markdown("## 🔍 Semantic Search")
    st.caption("Natural language discovery across all indexed scenes")
    st.divider()

    se = st.session_state.search_engine
    if se.stats["total_scenes"] == 0:
        st.info("No videos indexed yet — process a video first.")
        return

    c1, c2, c3 = st.columns([4, 1, 1])
    with c1:
        query = st.text_input("Search", placeholder="e.g. emotional confrontation between characters…",
                              label_visibility="collapsed")
    with c2:
        top_k = st.selectbox("Results", [5, 10, 20], index=1, label_visibility="collapsed")
    with c3:
        safety_label = st.selectbox("Safety", ["Any", "Safe 0.5+", "Brand Safe 0.8+"],
                                    label_visibility="collapsed")
        smap = {"Any": 0.0, "Safe 0.5+": 0.5, "Brand Safe 0.8+": 0.8}

    c4, c5 = st.columns([3, 1])
    with c4:
        iab_sel = st.multiselect("IAB filter", ["— All —"] + [f"{k}: {v}" for k, v in _IAB_NAMES.items()],
                                  label_visibility="collapsed", placeholder="Filter by content category…")
        iab_filter = [s.split(":")[0] for s in iab_sel if s != "— All —"] or None
    with c5:
        diversify = st.checkbox("Diversify", value=True)

    vid_filter = None
    if len(st.session_state.videos) > 1:
        choices = ["All Videos"] + [vm.title for vm in st.session_state.videos.values()]
        ids = [None] + list(st.session_state.videos.keys())
        vsel = st.selectbox("Video filter", choices, label_visibility="collapsed")
        vid_filter = ids[choices.index(vsel)]

    if not query:
        st.caption("SUGGESTED QUERIES")
        suggestions = ["exciting action sequence", "emotional dialogue", "product demonstration",
                       "travel destination", "health and wellness", "financial discussion",
                       "comedy moment", "suspenseful scene"]
        cols = st.columns(4)
        for i, s in enumerate(suggestions):
            if cols[i % 4].button(s, key=f"sug_{i}"):
                st.session_state["_sq"] = s
                st.rerun()
        return

    with st.spinner("Searching…"):
        results = se.search(query, top_k=top_k, diversify=diversify,
                            video_id=vid_filter, min_safety=smap[safety_label],
                            iab_filter=iab_filter, expand=True)

    if not results:
        st.warning("No results. Try a different query or relax your filters.")
        return

    st.caption(f"**{len(results)}** scenes found")

    for r in results:
        scene = r.scene
        vm = st.session_state.videos.get(scene.video_id)
        is_key = vm and scene.scene_id in vm.key_scenes if vm else False
        video_title = vm.title if vm else scene.video_id
        
        with st.container(border=True):
            h1, h2 = st.columns([5, 1])
            with h1:
                key_tag = "  ⭐" if is_key else ""
                st.markdown(f"**#{r.rank}** — {video_title}{key_tag}")
                st.caption(f"`{scene.start_fmt} → {scene.end_fmt}`  ·  {scene.duration_sec:.0f}s")
            with h2:
                st.metric("Score", _fmt_score(r.score))

            st.write(scene.text[:300] + ("…" if len(scene.text) > 300 else ""))

            c1, c2, c3, c4 = st.columns(4)
            safety = scene.brand_safety.get("safety_score", 1.0)
            sent = scene.sentiment.get("label", "neutral")
            c1.caption(f"**Topics:** {_iab_list(scene.iab_categories)}")
            c2.caption(f"**Sentiment:** {_sent_icon(sent)} {sent.title()}")
            c3.caption(f"**Safety:** {safety:.0%}")
            c4.caption(f"**vec** `{r.vector_score:.2f}`  **bm25** `{r.bm25_score:.2f}`")
            st.progress(r.score)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: SCENE EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def page_scenes():
    st.markdown("## 📺 Scene Explorer")
    st.caption("Browse and analyse every detected scene")
    st.divider()

    if not st.session_state.videos:
        st.info("No videos processed yet.")
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
    c5.metric("Narrative", vm.narrative_structure.split("(")[0].strip()[:12])

    st.divider()
    tab1, tab2, tab3 = st.tabs(["🎭  Emotional Arc", "📊  Analysis", "🗂  Scene List"])

    with tab1:
        if vm.emotional_arc:
            df = pd.DataFrame(vm.emotional_arc)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["start_sec"], y=df["sentiment_score"],
                fill="tozeroy", fillcolor="rgba(245,158,11,0.08)",
                line=dict(color="#f59e0b", width=2), mode="lines+markers",
                marker=dict(size=5), name="Sentiment"))
            fig.add_trace(go.Scatter(x=df["start_sec"], y=df["engagement"],
                line=dict(color="#34d399", width=2, dash="dot"),
                mode="lines", name="Engagement"))
            fig.add_hline(y=0, line_dash="dot", line_color="rgba(255,255,255,0.1)")
            for kid in vm.key_scenes:
                ks = next((s for s in vm.scenes if s.scene_id == kid), None)
                if ks:
                    fig.add_vline(x=ks.start_sec, line_dash="dash",
                                  line_color="rgba(245,158,11,0.35)")
            fig.update_layout(**PT, height=320, showlegend=True,
                              title=dict(text="Sentiment & Engagement Over Time",
                                         font=dict(size=12, color="#6b7280")),
                              xaxis_title="Time (s)", yaxis_title="Score",
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7280")))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No emotional arc data.")

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            iab_c: dict[str, int] = {}
            for s in vm.scenes:
                for cat in s.iab_categories[:1]:
                    iab_c[cat["name"]] = iab_c.get(cat["name"], 0) + 1
            if iab_c:
                df_iab = pd.DataFrame(sorted(iab_c.items(), key=lambda x: x[1], reverse=True)[:8],
                                      columns=["Category", "Scenes"])
                fig = px.bar(df_iab, x="Scenes", y="Category", orientation="h",
                             color_discrete_sequence=["#f59e0b"])
                fig.update_layout(**PT, height=300,
                                  title=dict(text="Top IAB Categories", font=dict(size=12, color="#6b7280")),
                                  yaxis=dict(autorange="reversed", **PT["yaxis"]))
                st.plotly_chart(fig, use_container_width=True)

        with c2:
            safe_scores = [s.brand_safety.get("safety_score", 1.0) for s in vm.scenes]
            fig = px.histogram(safe_scores, nbins=10, color_discrete_sequence=["#34d399"])
            fig.update_layout(**PT, height=300, showlegend=False,
                              title=dict(text="Brand Safety Distribution", font=dict(size=12, color="#6b7280")),
                              xaxis_title="Safety Score", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)

        eng_s = [s.engagement_score for s in vm.scenes]
        suit_s = [s.ad_suitability for s in vm.scenes]
        times = [s.start_sec for s in vm.scenes]
        fig = go.Figure(go.Scatter(
            x=times, y=eng_s, mode="markers",
            marker=dict(size=[s * 18 + 4 for s in suit_s],
                        color=eng_s, colorscale=[[0, "#1e2028"], [1, "#f59e0b"]],
                        showscale=True, opacity=0.75,
                        colorbar=dict(title="Engagement", tickfont=dict(color="#6b7280"))),
            hovertemplate="%{x:.0f}s — engagement: %{y:.3f}<extra></extra>"))
        fig.update_layout(**PT, height=280,
                          title=dict(text="Scene Engagement Map — bubble = ad suitability",
                                     font=dict(size=12, color="#6b7280")),
                          xaxis_title="Time (s)", yaxis_title="Engagement")
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            sent_f = st.multiselect("Sentiment", ["positive", "neutral", "negative"],
                                    default=["positive", "neutral", "negative"])
        with fc2:
            min_eng = st.slider("Min engagement", 0.0, 1.0, 0.0, 0.05, key="se_eng")
        with fc3:
            min_safe = st.slider("Min safety", 0.0, 1.0, 0.0, 0.05, key="se_safe")

        filtered = [s for s in vm.scenes
                    if s.sentiment.get("label", "neutral") in sent_f
                    and s.engagement_score >= min_eng
                    and s.brand_safety.get("safety_score", 1.0) >= min_safe]

        st.caption(f"{len(filtered)} scenes")
        for scene in filtered:
            _scene_card(scene, is_key=scene.scene_id in vm.key_scenes)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4: AD ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def page_ads():
    st.markdown("## 📢 Ad Engine")
    st.caption("Contextual ad matching and placement optimisation")
    st.divider()

    if not st.session_state.videos:
        st.info("Process a video first.")
        return

    vm = _get_active_video()
    if not vm:
        return
    ae = st.session_state.ad_engine

    tab1, tab2, tab3 = st.tabs(["🎯  Placement Plan", "🔍  Scene Matching", "📦  Inventory"])

    with tab1:
        c1, c2 = st.columns([2, 1], gap="large")
        with c1:
            placement_types = st.multiselect("Placement types",
                ["pre-roll", "mid-roll", "post-roll"],
                default=["pre-roll", "mid-roll", "post-roll"])
            min_safety_ads = st.slider("Min brand safety", 0.0, 1.0, 0.5, 0.05, key="ad_safe")
        with c2:
            st.caption("VIDEO INFO")
            st.metric("Scenes", vm.scene_count)
            st.metric("Duration", vm.fmt_duration())

        if st.button("Generate Placement Plan →", key="gen_plan"):
            with st.spinner("Optimising…"):
                placements = ae.plan_placements(vm.scenes, vm.duration_ms, placement_types)
                perf = ae.simulate_performance(placements)
                st.session_state["_placements"] = placements
                st.session_state["_perf"] = perf

        if "_placements" in st.session_state and st.session_state["_placements"]:
            placements = st.session_state["_placements"]
            perf = st.session_state["_perf"]
            st.divider()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Placements", perf["total_placements"])
            c2.metric("Est. Revenue", f"${perf['total_revenue_usd']:.2f}")
            c3.metric("Impressions", f"{perf['total_impressions']:,}")
            c4.metric("Clicks", f"{perf['estimated_clicks']:,}")
            c5.metric("Avg CPM", f"${perf['avg_cpm']:.2f}")

            p_df = pd.DataFrame([p.to_dict() for p in placements])
            type_colors = {"pre-roll": "#f59e0b", "mid-roll": "#34d399", "post-roll": "#60a5fa"}
            fig = go.Figure()
            for p_type in p_df["placement_type"].unique():
                sub = p_df[p_df["placement_type"] == p_type]
                fig.add_trace(go.Scatter(
                    x=sub["timestamp_ms"] / 1000, y=sub["relevance_score"],
                    mode="markers+text", name=p_type,
                    text=sub["brand"].str[:10], textposition="top center",
                    textfont=dict(size=9, color="#6b7280"),
                    marker=dict(size=sub["estimated_cpm"] * 2 + 10,
                                color=type_colors.get(p_type, "#f59e0b"), opacity=0.8)))
            fig.update_layout(**PT, height=300,
                              title=dict(text="Placement Timeline", font=dict(size=12, color="#6b7280")),
                              xaxis_title="Time (s)", yaxis_title="Relevance",
                              legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7280")))
            st.plotly_chart(fig, use_container_width=True)

            show_cols = ["timestamp_fmt", "placement_type", "ad_title", "brand",
                         "relevance_score", "safety_score", "estimated_cpm"]
            st.dataframe(p_df[show_cols].rename(columns={
                "timestamp_fmt": "Time", "placement_type": "Type", "ad_title": "Ad",
                "brand": "Brand", "relevance_score": "Relevance",
                "safety_score": "Safety", "estimated_cpm": "CPM ($)"}),
                use_container_width=True, hide_index=True)

    with tab2:
        scene_opts = [f"[{s.start_fmt}] {s.text[:60]}…" for s in vm.scenes]
        idx = st.selectbox("Select a scene", range(len(vm.scenes)),
                           format_func=lambda i: scene_opts[i])
        scene = vm.scenes[idx]
        matches = ae.match_ads(scene, top_k=5)

        if not matches:
            st.warning("No eligible ads for this scene.")
        else:
            for ad, si in matches:
                with st.container(border=True):
                    h1, h2 = st.columns([4, 1])
                    with h1:
                        st.markdown(f"**{ad.title}**")
                        st.caption(f"{ad.brand}  ·  ${ad.cpm_base:.0f} base CPM")
                        st.write(ad.description)
                    with h2:
                        st.metric("Match", _fmt_score(si["total"]))

                    c1, c2, c3, c4, c5 = st.columns(5)
                    c1.caption(f"content `{si['content_sim']:.2f}`")
                    c2.caption(f"IAB `{si['iab_match']:.2f}`")
                    c3.caption(f"safety `{si['safety']:.2f}`")
                    c4.caption(f"demo `{si['demographic']:.2f}`")
                    c5.caption(f"perf `{si['performance']:.2f}`")
                    st.progress(si["total"])

    with tab3:
        inv_df = pd.DataFrame([ad.to_dict() for ad in ae.inventory])
        cols = ["title", "brand", "cpm_base", "historical_ctr", "performance_score",
                "brand_safety_min", "budget_remaining"]
        st.dataframe(inv_df[cols].rename(columns={
            "title": "Ad", "brand": "Brand", "cpm_base": "Base CPM",
            "historical_ctr": "CTR", "performance_score": "Perf",
            "brand_safety_min": "Min Safety", "budget_remaining": "Budget Left"}),
            use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5: ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
def page_analytics():
    st.markdown("## 📊 Analytics Dashboard")
    st.caption("Performance metrics and content intelligence insights")
    st.divider()

    if not st.session_state.videos:
        st.info("Process a video to see analytics.")
        return

    all_scenes = [s for vm in st.session_state.videos.values() for s in vm.scenes]
    n = max(len(all_scenes), 1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Videos", len(st.session_state.videos))
    c2.metric("Total Scenes", len(all_scenes))
    c3.metric("Avg Engagement", round(sum(s.engagement_score for s in all_scenes) / n, 3))
    safe_pct = round(sum(1 for s in all_scenes if s.brand_safety.get("safety_score", 1.0) >= 0.8) / n * 100, 1)
    c4.metric("Brand Safe", f"{safe_pct}%")
    pos = round(sum(1 for s in all_scenes if s.sentiment.get("label") == "positive") / n * 100, 1)
    c5.metric("Positive Sent.", f"{pos}%")

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        labels = [s.sentiment.get("label", "neutral") for s in all_scenes]
        lc = {l: labels.count(l) for l in set(labels)}
        fig = px.pie(values=list(lc.values()), names=list(lc.keys()),
                     color_discrete_map={"positive": "#34d399", "neutral": "#4b5260", "negative": "#f87171"},
                     hole=0.5)
        fig.update_layout(**PT, height=280,
                          title=dict(text="Sentiment Distribution", font=dict(size=12, color="#6b7280")),
                          legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7280")))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        iab_all: dict[str, int] = {}
        for s in all_scenes:
            for cat in s.iab_categories[:1]:
                iab_all[cat["name"]] = iab_all.get(cat["name"], 0) + 1
        if iab_all:
            top = sorted(iab_all.items(), key=lambda x: x[1], reverse=True)[:10]
            df_iab = pd.DataFrame(top, columns=["Category", "Scenes"])
            fig = px.bar(df_iab, x="Scenes", y="Category", orientation="h",
                         color="Scenes", color_continuous_scale=[[0, "#1e2028"], [1, "#f59e0b"]])
            fig.update_layout(**PT, height=280,
                              title=dict(text="Top Content Categories", font=dict(size=12, color="#6b7280")),
                              yaxis=dict(autorange="reversed", **PT["yaxis"]),
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        eng_v = [s.engagement_score for s in all_scenes]
        safe_v = [s.brand_safety.get("safety_score", 1.0) for s in all_scenes]
        suit_v = [s.ad_suitability for s in all_scenes]
        fig = go.Figure(go.Scatter(x=eng_v, y=safe_v, mode="markers",
            marker=dict(size=[v * 14 + 4 for v in suit_v],
                        color=suit_v, colorscale=[[0, "#1e2028"], [1, "#f59e0b"]],
                        showscale=True, opacity=0.7,
                        colorbar=dict(title="Ad Suit.", tickfont=dict(color="#6b7280"))),
            hovertemplate="eng: %{x:.2f}<br>safety: %{y:.2f}<extra></extra>"))
        fig.update_layout(**PT, height=300,
                          title=dict(text="Engagement vs Safety", font=dict(size=12, color="#6b7280")),
                          xaxis_title="Engagement", yaxis_title="Safety")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram([s.ad_suitability for s in all_scenes], nbins=15,
                           color_discrete_sequence=["#fb923c"])
        fig.update_layout(**PT, height=300, showlegend=False,
                          title=dict(text="Ad Suitability Distribution", font=dict(size=12, color="#6b7280")),
                          xaxis_title="Ad Suitability", yaxis_title="Scenes")
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("VIDEO BREAKDOWN")
    rows = []
    for vm in st.session_state.videos.values():
        if not vm.scenes:
            continue
        rows.append({
            "Title": vm.title[:40],
            "Duration": vm.fmt_duration(),
            "Scenes": vm.scene_count,
            "Narrative": vm.narrative_structure.split("(")[0].strip(),
            "Avg Engagement": round(sum(s.engagement_score for s in vm.scenes) / len(vm.scenes), 3),
            "Avg Safety": f"{round(sum(s.brand_safety.get('safety_score',1) for s in vm.scenes)/len(vm.scenes),2):.0%}",
            "Top Category": vm.dominant_iab[0]["name"] if vm.dominant_iab else "—",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6: FRANCHISE INTEL
# ══════════════════════════════════════════════════════════════════════════════
def page_franchise():
    st.markdown("## 🎯 Franchise Intelligence")
    st.caption("Cross-video theme tracking and recurring ad opportunities")
    st.divider()

    if not st.session_state.videos:
        st.info("Process at least one video.")
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
        rows = [{"Theme": t, "Video": v, "Occurrences": c}
                for t, vids in theme_counts.items() for v, c in vids.items()]
        if rows:
            fig = px.bar(pd.DataFrame(rows), x="Theme", y="Occurrences", color="Video",
                         color_discrete_sequence=PT["colorway"])
            fig.update_layout(**PT, height=280,
                              title=dict(text="Theme Frequency by Video", font=dict(size=12, color="#6b7280")),
                              xaxis_tickangle=-30, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#6b7280")))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.caption("TOP AD OPPORTUNITIES")

    iab_opps: dict[str, list] = {}
    for vm in st.session_state.videos.values():
        for scene in vm.scenes:
            if scene.ad_suitability > 0.6:
                for cat in scene.iab_categories[:1]:
                    cn = cat["name"]
                    iab_opps.setdefault(cn, []).append({
                        "video": vm.title[:30],
                        "ad_suitability": scene.ad_suitability,
                        "engagement": scene.engagement_score,
                    })

    if iab_opps:
        sorted_opps = sorted(iab_opps.items(),
                             key=lambda x: sum(o["ad_suitability"] for o in x[1]),
                             reverse=True)[:6]
        for cat_name, scenes_list in sorted_opps[:4]:
            avg_suit = sum(o["ad_suitability"] for o in scenes_list) / len(scenes_list)
            avg_eng = sum(o["engagement"] for o in scenes_list) / len(scenes_list)
            unique_vids = len(set(o["video"] for o in scenes_list))
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
                c1.markdown(f"**{cat_name}**")
                c1.caption(f"{len(scenes_list)} scenes · {unique_vids} video(s)")
                c2.metric("Avg Suit.", f"{avg_suit:.3f}")
                c3.metric("Avg Eng.", f"{avg_eng:.3f}")
                c4.metric("Videos", unique_vids)
                st.progress(avg_suit)

    if len(all_scenes) > 1 and se.stats["total_scenes"] > 0:
        st.divider()
        st.caption("SIMILAR SCENES ACROSS VIDEOS")
        opts = [f"[{s.video_id[:8]}] {s.start_fmt} — {s.text[:50]}…" for s in all_scenes[:50]]
        sel = st.selectbox("Reference scene", range(len(all_scenes[:50])),
                           format_func=lambda i: opts[i])
        similar = se.find_similar_scenes(all_scenes[sel], top_k=5, exclude_same_video=True)
        if similar:
            for r in similar:
                vm2 = st.session_state.videos.get(r.scene.video_id)
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1])
                    c1.markdown(f"**{vm2.title[:35] if vm2 else r.scene.video_id}**")
                    c1.caption(f"`{r.scene.start_fmt}`")
                    c1.write(r.scene.text[:200] + "…")
                    c2.metric("Similarity", _fmt_score(r.score))
                    st.progress(r.score)
        else:
            st.info("No similar scenes found in other videos.")


# ── Router ─────────────────────────────────────────────────────────────────────
{
    "process":   page_process,
    "search":    page_search,
    "scenes":    page_scenes,
    "ads":       page_ads,
    "analytics": page_analytics,
    "franchise": page_franchise,
}[st.session_state.page]()
