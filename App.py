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
    "last_yt_id": None,
    "demo_video_b64": None,
    "demo_video_type": None,
    "demo_markers": [],
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
    ("demo",      "🎥", "Video Ad Demo"),
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
                               key="vm_sel")
        st.session_state.selected_video = vms[sel_idx].video_id

        se = st.session_state.search_engine.stats
        c1, c2 = st.columns(2)
        c1.metric("Videos", se["total_videos"])
        c2.metric("Scenes", se["total_scenes"])

    st.divider()
    yt = st.text_input("YouTube API Key", type="password",
                        value=st.session_state.yt_api_key,
                        placeholder="Optional",
                        key="yt_key_in")
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

def _yt_player_with_scenes(vid_id: str, scenes: list, key_scene_ids: set):
    """
    Self-contained YouTube player + clickable scene list in a single HTML component.
    All seeking happens inside the component via YouTube IFrame API — no Streamlit rerun needed.
    """
    import json

    # Build scene data for JS
    scene_data = []
    for i, s in enumerate(scenes):
        safety = s.brand_safety.get("safety_score", 1.0)
        sent = s.sentiment.get("label", "neutral")
        sent_icon = {"positive": "🟢", "negative": "🔴", "neutral": "🔵"}.get(sent, "🔵")
        iab = "  ·  ".join(c["name"] for c in s.iab_categories[:2]) if s.iab_categories else ""
        is_key = s.scene_id in key_scene_ids
        scene_data.append({
            "idx": i,
            "start": s.start_sec,
            "end": s.end_sec,
            "start_fmt": s.start_fmt,
            "end_fmt": s.end_fmt,
            "dur": int(s.duration_sec),
            "text": s.text[:220].replace('"', '\"').replace("\n", " "),
            "sent_icon": sent_icon,
            "sent": sent,
            "safety": f"{safety:.0%}",
            "iab": iab,
            "eng": f"{s.engagement_score:.2f}",
            "ad_fit": f"{s.ad_suitability:.2f}",
            "is_key": is_key,
        })

    scenes_json = json.dumps(scene_data)

    html = f"""
<!DOCTYPE html>
<html>
<head>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: Inter, sans-serif; }}
  body {{ background: #fff; }}
  #wrapper {{ display: flex; gap: 16px; width: 100%; }}
  #player-col {{ flex: 0 0 58%; }}
  #player-wrap {{
    position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;
    border-radius: 10px; border: 1px solid #e5e7eb;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
  }}
  #player-wrap iframe {{
    position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  }}
  #now-playing {{
    margin-top: 10px; padding: 8px 12px;
    background: #fff7ed; border: 1px solid #fed7aa; border-radius: 8px;
    font-size: 13px; color: #92400e; display: none;
  }}
  #scenes-col {{
    flex: 1; height: 520px; overflow-y: auto;
    border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px;
    background: #fafafa;
  }}
  .scene-card {{
    background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
    padding: 10px 12px; margin-bottom: 8px; cursor: pointer;
    transition: border-color 0.15s, box-shadow 0.15s;
  }}
  .scene-card:hover {{ border-color: #f59e0b; box-shadow: 0 2px 8px rgba(245,158,11,0.15); }}
  .scene-card.active {{ border-color: #f59e0b; background: #fffbeb; box-shadow: 0 2px 8px rgba(245,158,11,0.2); }}
  .scene-card.key {{ border-left: 3px solid #f59e0b; }}
  .ts {{ font-family: monospace; font-size: 12px; color: #6b7280;
          background: #f3f4f6; padding: 2px 7px; border-radius: 4px; }}
  .dur {{ font-size: 12px; font-weight: 600; color: #374151; margin-left: 6px; }}
  .badges {{ display: flex; gap: 6px; align-items: center; margin: 4px 0; font-size: 12px; color: #6b7280; }}
  .scene-text {{ font-size: 13px; color: #374151; line-height: 1.5; margin: 6px 0 4px; }}
  .iab {{ font-size: 11px; color: #9ca3af; }}
  .play-btn {{
    display: inline-flex; align-items: center; gap: 5px;
    margin-top: 7px; padding: 5px 12px;
    background: #f59e0b; color: #111; border: none; border-radius: 6px;
    font-size: 12px; font-weight: 600; cursor: pointer;
    transition: background 0.15s;
  }}
  .play-btn:hover {{ background: #d97706; }}
  #filter-bar {{ display: flex; gap: 8px; margin-bottom: 10px; align-items: center; flex-wrap: wrap; }}
  #filter-bar select, #filter-bar input {{
    border: 1px solid #d1d5db; border-radius: 6px; padding: 4px 8px;
    font-size: 12px; background: #fff; color: #374151;
  }}
  #filter-bar label {{ font-size: 12px; color: #6b7280; font-weight: 500; }}
  #scene-count {{ font-size: 12px; color: #9ca3af; margin-left: auto; }}
</style>
</head>
<body>
<div id="wrapper">
  <div id="player-col">
    <div id="player-wrap">
      <iframe id="yt-iframe"
        src="https://www.youtube.com/embed/{vid_id}?enablejsapi=1&rel=0&modestbranding=1&playsinline=1"
        frameborder="0"
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowfullscreen></iframe>
    </div>
    <div id="now-playing">▶ Now playing from <span id="np-time"></span></div>
  </div>

  <div id="scenes-col">
    <div id="filter-bar">
      <label>Sentiment:</label>
      <select id="sent-filter" onchange="renderScenes()">
        <option value="all">All</option>
        <option value="positive">Positive</option>
        <option value="neutral">Neutral</option>
        <option value="negative">Negative</option>
      </select>
      <label>Min ad fit:</label>
      <input type="range" id="fit-filter" min="0" max="1" step="0.1" value="0"
             oninput="document.getElementById('fit-val').textContent=this.value; renderScenes()">
      <span id="fit-val" style="font-size:12px;color:#6b7280">0</span>
      <span id="scene-count"></span>
    </div>
    <div id="scenes-list"></div>
  </div>
</div>

<script>
var scenes = {scenes_json};
var player = null;
var activeIdx = -1;

// Load YouTube IFrame API
var tag = document.createElement('script');
tag.src = "https://www.youtube.com/iframe_api";
document.head.appendChild(tag);

function onYouTubeIframeAPIReady() {{
  player = new YT.Player('yt-iframe', {{
    events: {{ 'onReady': function(e) {{ console.log('YT player ready'); }} }}
  }});
}}

function seekTo(startSec, idx, startFmt) {{
  activeIdx = idx;
  // Mark active card
  document.querySelectorAll('.scene-card').forEach(function(c) {{ c.classList.remove('active'); }});
  var card = document.getElementById('card-'+idx);
  if (card) {{
    card.classList.add('active');
    card.scrollIntoView({{behavior:'smooth', block:'nearest'}});
  }}
  // Show now-playing banner
  var np = document.getElementById('now-playing');
  np.style.display = 'block';
  document.getElementById('np-time').textContent = startFmt;
  // Seek using YT API if available, else reload src with start param
  if (player && player.seekTo) {{
    player.seekTo(startSec, true);
    player.playVideo();
  }} else {{
    // Fallback: reload iframe with start time
    var iframe = document.getElementById('yt-iframe');
    iframe.src = "https://www.youtube.com/embed/{vid_id}?enablejsapi=1&autoplay=1&start="+startSec+"&rel=0&modestbranding=1&playsinline=1";
  }}
}}

function renderScenes() {{
  var sentFilter = document.getElementById('sent-filter').value;
  var fitFilter = parseFloat(document.getElementById('fit-filter').value);
  var list = document.getElementById('scenes-list');
  list.innerHTML = '';
  var shown = 0;
  scenes.forEach(function(s) {{
    if (sentFilter !== 'all' && s.sent !== sentFilter) return;
    if (parseFloat(s.ad_fit) < fitFilter) return;
    shown++;
    var keyClass = s.is_key ? ' key' : '';
    var activeClass = s.idx === activeIdx ? ' active' : '';
    var keyBadge = s.is_key ? ' ⭐' : '';
    list.innerHTML += '<div class="scene-card'+keyClass+activeClass+'" id="card-'+s.idx+'">' +
      '<div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap">' +
        '<span class="ts">'+s.start_fmt+' → '+s.end_fmt+'</span>' +
        '<span class="dur">'+s.dur+'s'+keyBadge+'</span>' +
      '</div>' +
      '<div class="badges">'+s.sent_icon+' '+s.sent+' &nbsp;·&nbsp; 🛡 '+s.safety+'</div>' +
      '<div class="scene-text">'+s.text+(s.text.length>=220?'…':'')+'</div>' +
      '<div class="iab">'+s.iab+' &nbsp;·&nbsp; eng '+s.eng+' · ad fit '+s.ad_fit+'</div>' +
      '<button class="play-btn" onclick="seekTo('+s.start+','+s.idx+',''+s.start_fmt+'')">▶ Play at '+s.start_fmt+'</button>' +
    '</div>';
  }});
  document.getElementById('scene-count').textContent = shown + ' scenes';
}}

renderScenes();
</script>
</body>
</html>
"""
    st.components.v1.html(html, height=580, scrolling=False)


def _yt_embed(vid_id: str):
    """Simple YouTube embed (no scene list)."""
    st.markdown(f"""
    <div style="position:relative;padding-bottom:56.25%;height:0;overflow:hidden;
                border-radius:10px;border:1px solid #e5e7eb;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
        <iframe src="https://www.youtube.com/embed/{vid_id}?rel=0&modestbranding=1"
            style="position:absolute;top:0;left:0;width:100%;height:100%;"
            frameborder="0"
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

        # Jump to timestamp — open YouTube at exact second in new tab
        if show_jump and yt_id:
            seek_s = scene.start_sec
            yt_url = f"https://www.youtube.com/watch?v={yt_id}&t={seek_s}s"
            st.link_button(f"▶ Play at {scene.start_fmt}", yt_url)

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

    # ── YouTube tab ────────────────────────────────────────────────────────
    with tab_yt:
        st.markdown("#### Paste a YouTube URL")
        yt_url = st.text_input("YouTube URL or Video ID",
                               placeholder="https://youtube.com/watch?v=...",
                               key="yt_url_in")

        ca, cb = st.columns([2, 3], gap="large")
        with ca:
            st.markdown("**Scene Detection Settings**")
            min_s = st.slider("Min scene length (seconds)", 10, 90, 20, 5, key="yt_min")
            max_s = st.slider("Max scene length (seconds)", 60, 300, 120, 10, key="yt_max")
            sens  = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="yt_sens",
                              help="Higher = fewer but bigger scenes")
        with cb:
            if yt_url:
                vid_id = _yt_id(yt_url)
                if vid_id:
                    st.markdown("**Video Preview**")
                    _yt_embed(vid_id)
                else:
                    st.warning("Could not parse a video ID from that URL.")

        if yt_url:
            if st.button("⚡ Analyse This Video", key="proc_yt", type="primary"):
                vid_id = _yt_id(yt_url)
                if not vid_id:
                    st.error("Invalid URL — please paste a full YouTube link.")
                    st.stop()
                with st.spinner("Fetching transcript from YouTube…"):
                    transcript = fetch_youtube_transcript(vid_id)
                if transcript is None:
                    st.error("No captions found. Try a video with auto-generated subtitles enabled.")
                    st.stop()
                meta = None
                if st.session_state.yt_api_key:
                    with st.spinner("Fetching video metadata…"):
                        meta = fetch_youtube_metadata(vid_id, st.session_state.yt_api_key)
                title = meta.get("title", f"YouTube · {vid_id}") if meta else f"YouTube · {vid_id}"
                with st.spinner(f"Detecting scenes…"):
                    t0 = time.time()
                    vm = VideoProcessor(min_s, max_s, sens).process_youtube_transcript(
                        transcript, vid_id, title, meta)
                    elapsed = time.time() - t0
                _register(vm)
                st.session_state.last_yt_id = vid_id
                vm.yt_id = vid_id
                st.success(f"✅  Found **{len(vm.scenes)} scenes** in {elapsed:.1f}s")
                _summary_strip(vm)
                st.info("👉  Click **Watch & Explore** in the sidebar to play the video and explore scenes.")

    # ── Upload SRT/VTT tab ─────────────────────────────────────────────────
    with tab_file:
        st.markdown("#### Upload a subtitle file (.srt or .vtt)")

        ca, cb = st.columns([3, 2], gap="large")
        with ca:
            uploaded = st.file_uploader(
                "Choose your subtitle file",
                type=["srt", "vtt"],
                key="up_file",
                help="SubRip (.srt) or WebVTT (.vtt) format"
            )
            title_f = st.text_input("Video title (optional)",
                                    placeholder="Leave blank to use filename",
                                    key="up_title")
            yt_link = st.text_input(
                "YouTube URL (optional — enables the video player)",
                placeholder="https://youtube.com/watch?v=...",
                key="up_yt"
            )
        with cb:
            st.markdown("**Scene Detection Settings**")
            min_f = st.slider("Min scene length (seconds)", 10, 90, 20, 5, key="f_min")
            max_f = st.slider("Max scene length (seconds)", 60, 300, 120, 10, key="f_max")
            sens_f = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="f_sens")

        if uploaded is not None:
            st.success(f"File ready: **{uploaded.name}** ({uploaded.size:,} bytes)")
            if st.button("⚡ Process File", key="proc_file", type="primary"):
                content_text = uploaded.read().decode("utf-8", errors="replace")
                fmt = "vtt" if uploaded.name.lower().endswith(".vtt") else "srt"
                title = title_f.strip() or uploaded.name
                with st.spinner("Analysing scenes…"):
                    t0 = time.time()
                    vm = VideoProcessor(min_f, max_f, sens_f).process_file(content_text, title, fmt)
                    elapsed = time.time() - t0
                if not vm.scenes:
                    st.error("No scenes detected — check that the file is a valid SRT/VTT.")
                    st.stop()
                _register(vm)
                if yt_link.strip():
                    vid_id = _yt_id(yt_link.strip())
                    if vid_id:
                        st.session_state.last_yt_id = vid_id
                        vm.yt_id = vid_id
                st.success(f"✅  **{len(vm.scenes)} scenes** detected in {elapsed:.1f}s")
                _summary_strip(vm)
        else:
            st.info("👆 Choose a .srt or .vtt file to get started")

    # ── Paste Text tab ─────────────────────────────────────────────────────
    with tab_text:
        st.markdown("#### Paste subtitle content directly")

        ca, cb = st.columns([3, 2], gap="large")
        with ca:
            title_p = st.text_input("Video title", placeholder="My Video", key="p_title")
            pasted = st.text_area(
                "Paste SRT or VTT content here",
                height=220,
                placeholder="1\n00:00:01,000 --> 00:00:05,000\nHello world...\n\n2\n00:00:06,000 --> 00:00:10,000\nNext line here...",
                key="p_text"
            )
        with cb:
            st.markdown("**Scene Detection Settings**")
            min_p = st.slider("Min scene length (seconds)", 10, 90, 20, 5, key="p_min")
            max_p = st.slider("Max scene length (seconds)", 60, 300, 120, 10, key="p_max")

        if pasted.strip():
            if st.button("⚡ Process Text", key="proc_paste", type="primary"):
                with st.spinner("Processing…"):
                    t0 = time.time()
                    vm = VideoProcessor(min_p, max_p).process_file(
                        pasted, title_p.strip() or "Pasted Video")
                    elapsed = time.time() - t0
                if not vm.scenes:
                    st.error("No scenes found — check that the content is valid SRT/VTT format.")
                    st.stop()
                _register(vm)
                st.success(f"✅  **{len(vm.scenes)} scenes** in {elapsed:.1f}s")
                _summary_strip(vm)
        else:
            st.info("👆 Paste subtitle content above to get started")


def _summary_strip(vm: VideoMetadata):
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
# PAGE 2 — WATCH & EXPLORE
# ══════════════════════════════════════════════════════════════════════════════
def page_watch():
    st.markdown("## ▶️ Watch & Explore")
    st.caption("Player on the left — click any scene card to jump to that moment instantly")
    st.divider()

    if not st.session_state.videos:
        st.info("No video loaded yet — go to **Load Video** first.")
        return

    vm = _active_vm()
    if not vm:
        return

    yt_id = getattr(vm, "yt_id", None) or st.session_state.get("last_yt_id")

    if not yt_id:
        st.warning("No YouTube link for this video. Re-process via the **YouTube URL** tab "
                   "or paste a YouTube URL when uploading the SRT file.")
        st.divider()
        for scene in vm.scenes:
            _scene_row(scene, vm)
        return

    st.caption(f"**{vm.title}**  ·  {vm.fmt_duration()}  ·  {vm.scene_count} scenes")

    # ── All-in-one HTML component: player + scene list + seek ─────────────
    _yt_player_with_scenes(yt_id, vm.scenes, set(vm.key_scenes))

    # ── Emotional arc below player ─────────────────────────────────────────
    if vm.emotional_arc:
        st.divider()
        df = pd.DataFrame(vm.emotional_arc)
        fig = px.area(df, x="start_sec", y="sentiment_score",
                      color_discrete_sequence=[_AMBER])
        fig.add_hline(y=0, line_dash="dot", line_color="rgba(0,0,0,0.1)")
        fig.update_layout(
            **PT, height=150, showlegend=False,
            xaxis=dict(**_XAXIS, title="Time (s)"),
            yaxis=dict(**_YAXIS, title=""),
            title=dict(text="Emotional Arc", font=dict(size=11, color=_TEXT)),
        )
        st.plotly_chart(fig, use_container_width=True)


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

    mode = st.radio("Search mode", ["🧠 Describe what you're looking for",
                         "🏷️ Browse by tags", "⏱️ Jump to timestamp"],
                    horizontal=True, key="search_mode")
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
    query = st.text_input("What moment are you looking for?",
        placeholder="Try: 'tense confrontation'  /  'product demo'  /  'emotional speech'  /  'cricket six'",
        key="sem_q", value=default_val)

    ca, cb, cc = st.columns([2, 1, 1])
    with ca:
        top_k = st.select_slider("Results", [3, 5, 10, 20], value=5, key="sem_k")
    with cb:
        safety_th = st.selectbox("Brand safety",
            ["Any", "Moderate (50%+)", "Strict (80%+)"],
            key="sem_safe")
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
)
    with cb:
        sent_f = st.multiselect("Sentiment",
            ["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"],
            key="tag_sent")
    with cc:
        min_fit = st.slider("Min ad fit", 0.0, 1.0, 0.0, 0.1,
                            key="tag_fit")

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
                                key="ts_from")
    with cb:
        ts_to = st.text_input("To (optional)", placeholder="00:05:00  or  300",
                              key="ts_to")

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
                               key="ad_scene_sel")
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



# ══════════════════════════════════════════════════════════════════════════════
# PAGE — VIDEO AD DEMO  (upload MP4, auto-detect scenes, inject sample ads)
# ══════════════════════════════════════════════════════════════════════════════
def page_demo():
    import base64, json

    st.markdown("## 🎥 Video Ad Demo")
    st.caption("Upload any MP4 · the engine detects key moments · sample ads play at those moments")
    st.divider()

    tab_upload, tab_player = st.tabs(["📁  Upload & Configure", "▶️  Watch with Ads"])

    # ── Sample ads library ─────────────────────────────────────────────────
    SAMPLE_ADS = [
        {"id": "ad1", "brand": "Nike",        "title": "Just Do It",
         "color": "#f59e0b", "emoji": "👟",
         "cta": "Shop Now",  "url": "https://nike.com",
         "bg": "linear-gradient(135deg,#f59e0b,#d97706)",
         "headline": "Push Your Limits",
         "body": "New season collection — built for champions.",
         "categories": ["Sports", "Fashion", "Health"]},
        {"id": "ad2", "brand": "Spotify",     "title": "Music for Every Mood",
         "color": "#1db954", "emoji": "🎵",
         "cta": "Listen Free", "url": "https://spotify.com",
         "bg": "linear-gradient(135deg,#1db954,#158a3e)",
         "headline": "Soundtrack Your Life",
         "body": "3 months Premium free — no ads, offline play.",
         "categories": ["Entertainment", "Music", "Arts"]},
        {"id": "ad3", "brand": "Amazon",      "title": "Deals of the Day",
         "color": "#ff9900", "emoji": "📦",
         "cta": "Shop Deals", "url": "https://amazon.com",
         "bg": "linear-gradient(135deg,#ff9900,#e47911)",
         "headline": "Today Only — Up to 60% Off",
         "body": "Lightning deals on electronics, home & more.",
         "categories": ["Shopping", "Technology", "Home"]},
        {"id": "ad4", "brand": "Netflix",     "title": "Stories Worth Watching",
         "color": "#e50914", "emoji": "🎬",
         "cta": "Watch Now", "url": "https://netflix.com",
         "bg": "linear-gradient(135deg,#e50914,#a30610)",
         "headline": "New Episodes Every Week",
         "body": "Award-winning series and films — start streaming today.",
         "categories": ["Entertainment", "Arts", "Drama"]},
        {"id": "ad5", "brand": "Duolingo",    "title": "Learn a Language",
         "color": "#58cc02", "emoji": "🦜",
         "cta": "Start Free",  "url": "https://duolingo.com",
         "bg": "linear-gradient(135deg,#58cc02,#3d9900)",
         "headline": "5 Minutes a Day Changes Everything",
         "body": "40+ languages. Free forever. Used by 500M people.",
         "categories": ["Education", "Language", "Personal Development"]},
        {"id": "ad6", "brand": "Uber Eats",   "title": "Food at Your Door",
         "color": "#06c167", "emoji": "🍔",
         "cta": "Order Now",   "url": "https://ubereats.com",
         "bg": "linear-gradient(135deg,#06c167,#038a47)",
         "headline": "Craving Something?",
         "body": "Your favourite restaurants delivered in 30 minutes.",
         "categories": ["Food", "Delivery", "Lifestyle"]},
    ]

    # ── UPLOAD TAB ─────────────────────────────────────────────────────────
    with tab_upload:
        ca, cb = st.columns([3, 2], gap="large")
        with ca:
            st.markdown("#### 1. Upload your video")
            video_file = st.file_uploader(
                "Choose an MP4, MOV or WebM file (max 200MB)",
                type=["mp4", "mov", "webm", "avi"],
                key="demo_video_up"
            )
            if video_file:
                st.success(f"✅ **{video_file.name}** ({video_file.size/1024/1024:.1f} MB)")

            st.markdown("#### 2. Add ad markers")
            st.caption("Specify timestamps where ads should appear (mid-roll), "
                        "plus auto pre-roll and post-roll will be added.")

            if "demo_markers" not in st.session_state or not isinstance(st.session_state.demo_markers, list):
                st.session_state.demo_markers = []

            # Add marker UI
            mc1, mc2, mc3 = st.columns([2, 3, 1])
            with mc1:
                new_ts = st.text_input("Timestamp", placeholder="00:01:30 or 90",
                                        key="new_marker_ts")
            with mc2:
                new_ad_name = st.selectbox("Ad to show", [a["brand"] + " — " + a["title"] for a in SAMPLE_ADS],
                                            key="new_marker_ad")
            with mc3:
                st.write("")
                st.write("")
                if st.button("➕ Add", key="add_marker"):
                    sec = _parse_ts(new_ts)
                    if sec is not None:
                        ad_idx = [a["brand"] + " — " + a["title"] for a in SAMPLE_ADS].index(new_ad_name)
                        marker = {"sec": sec, "ad": SAMPLE_ADS[ad_idx],
                                  "fmt": f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"}
                        st.session_state.demo_markers.append(marker)
                        st.rerun()
                    else:
                        st.error("Invalid timestamp")

            # Show current markers
            if st.session_state.demo_markers:
                st.markdown("**Current markers:**")
                for i, m in enumerate(st.session_state.demo_markers):
                    mc1, mc2, mc3 = st.columns([1, 4, 1])
                    mc1.markdown(f"`{m['fmt']}`")
                    mc2.markdown(f"{m['ad']['emoji']} **{m['ad']['brand']}** — {m['ad']['title']}")
                    if mc3.button("🗑", key=f"del_m_{i}"):
                        st.session_state.demo_markers.pop(i)
                        st.rerun()

        with cb:
            st.markdown("#### Ad library")
            for ad in SAMPLE_ADS:
                with st.container(border=True):
                    c1, c2 = st.columns([1, 4])
                    c1.markdown(f"<div style='font-size:2rem;text-align:center'>{ad['emoji']}</div>",
                                unsafe_allow_html=True)
                    with c2:
                        st.markdown(f"**{ad['brand']}** — {ad['title']}")
                        st.caption(" · ".join(ad["categories"][:2]))

        st.divider()
        st.markdown("#### 3. Generate ad plan from AI scene analysis")
        st.caption("If you've already processed this video via Load Video, "
                   "the AI will suggest additional markers at key engagement moments.")

        vm = _active_vm()
        if vm and vm.scenes:
            if st.button("⚡ Auto-suggest markers from scene analysis", key="auto_markers"):
                ae = st.session_state.ad_engine
                # Pick top 3 scenes by ad suitability for mid-roll
                top_scenes = sorted(vm.scenes, key=lambda s: s.ad_suitability, reverse=True)[:4]
                added = 0
                for scene in top_scenes:
                    matches = ae.match_ads(scene, top_k=1)
                    if matches:
                        best_ad_obj, score = matches[0]
                        # Find matching sample ad by category overlap
                        iab_names = [c["name"] for c in scene.iab_categories[:2]]
                        chosen_ad = SAMPLE_ADS[0]  # default
                        for samp in SAMPLE_ADS:
                            if any(cat in " ".join(iab_names) for cat in samp["categories"]):
                                chosen_ad = samp
                                break
                        sec = scene.start_sec
                        fmt = f"{sec//3600:02d}:{(sec%3600)//60:02d}:{sec%60:02d}"
                        # Avoid duplicates
                        existing = [m["sec"] for m in st.session_state.demo_markers]
                        if sec not in existing:
                            st.session_state.demo_markers.append(
                                {"sec": sec, "ad": chosen_ad, "fmt": fmt,
                                 "scene_text": scene.text[:80], "score": score.get("total", 0)})
                            added += 1
                st.success(f"Added {added} markers from top ad-suitable scenes")
                st.rerun()
        else:
            st.info("Process a video on the **Load Video** page first to enable AI-suggested markers.")

        # Save video to session state
        if video_file and st.button("✅ Save & Go to Player", key="save_demo", type="primary"):
            raw = video_file.read()
            b64 = base64.b64encode(raw).decode()
            st.session_state.demo_video_b64 = b64
            ext = video_file.name.split(".")[-1].lower()
            mime_map = {"mp4": "video/mp4", "mov": "video/mp4",
                        "webm": "video/webm", "avi": "video/x-msvideo"}
            st.session_state.demo_video_type = mime_map.get(ext, "video/mp4")
            st.session_state.page = "demo"
            # Switch to player tab by setting a flag
            st.session_state["demo_show_player"] = True
            st.rerun()

    # ── PLAYER TAB ─────────────────────────────────────────────────────────
    with tab_player:
        if not st.session_state.get("demo_video_b64"):
            st.info("Upload a video in the **Upload & Configure** tab first.")
            return

        markers = st.session_state.demo_markers
        video_b64 = st.session_state.demo_video_b64
        video_type = st.session_state.demo_video_type

        # Add pre-roll and post-roll to the markers list for the player
        all_markers = []
        # Pre-roll at t=0
        all_markers.append({"sec": 0, "ad": SAMPLE_ADS[0], "type": "pre-roll"})
        # Mid-roll markers from user
        for m in sorted(markers, key=lambda x: x["sec"]):
            all_markers.append({**m, "type": "mid-roll"})
        # Post-roll handled by JS at video end

        markers_json = json.dumps([{
            "sec": m["sec"],
            "type": m.get("type", "mid-roll"),
            "ad_id": m["ad"]["id"],
            "ad_brand": m["ad"]["brand"],
            "ad_title": m["ad"]["title"],
            "ad_headline": m["ad"]["headline"],
            "ad_body": m["ad"]["body"],
            "ad_cta": m["ad"]["cta"],
            "ad_emoji": m["ad"]["emoji"],
            "ad_bg": m["ad"]["bg"],
            "ad_color": m["ad"]["color"],
        } for m in all_markers])

        post_roll_ad = SAMPLE_ADS[3]  # Netflix for post-roll

        html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; font-family: Inter, sans-serif; }}
body {{ background: #f9fafb; padding: 0; }}

#player-wrap {{
  position: relative; background: #000;
  border-radius: 12px; overflow: hidden;
  border: 1px solid #e5e7eb;
  box-shadow: 0 4px 20px rgba(0,0,0,0.12);
}}
#main-video {{
  width: 100%; display: block; max-height: 460px; background: #000;
}}

/* ── Ad overlay ── */
#ad-overlay {{
  display: none;
  position: absolute; top: 0; left: 0; width: 100%; height: 100%;
  z-index: 20;
  align-items: center; justify-content: center;
  flex-direction: column;
}}
#ad-card {{
  width: 90%; max-width: 500px; border-radius: 16px;
  padding: 28px 32px; color: #fff; text-align: center;
  box-shadow: 0 8px 32px rgba(0,0,0,0.3);
  position: relative;
}}
#ad-emoji {{ font-size: 3rem; margin-bottom: 8px; }}
#ad-brand {{ font-size: 12px; font-weight: 700; text-transform: uppercase;
              letter-spacing: 0.12em; opacity: 0.8; margin-bottom: 4px; }}
#ad-headline {{ font-size: 22px; font-weight: 700; margin-bottom: 8px; }}
#ad-body {{ font-size: 14px; opacity: 0.9; margin-bottom: 20px; line-height: 1.5; }}
#ad-cta {{
  display: inline-block; background: rgba(255,255,255,0.2);
  border: 2px solid rgba(255,255,255,0.6);
  color: #fff; padding: 10px 28px; border-radius: 30px;
  font-weight: 700; font-size: 14px; cursor: pointer;
  transition: background 0.2s;
}}
#ad-cta:hover {{ background: rgba(255,255,255,0.35); }}
#ad-skip {{
  position: absolute; bottom: 14px; right: 16px;
  font-size: 12px; opacity: 0.7; cursor: pointer;
  background: rgba(0,0,0,0.3); padding: 4px 10px;
  border-radius: 12px; color: #fff;
}}
#ad-skip:hover {{ opacity: 1; }}
#ad-countdown {{
  position: absolute; top: 14px; right: 16px;
  font-size: 12px; background: rgba(0,0,0,0.4);
  color: #fff; padding: 4px 10px; border-radius: 12px;
}}
#ad-type-badge {{
  position: absolute; top: 14px; left: 16px;
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: 0.1em; background: rgba(0,0,0,0.4);
  color: #fff; padding: 4px 10px; border-radius: 12px;
}}

/* ── Timeline ── */
#timeline-wrap {{
  background: #fff; padding: 12px 16px; border-top: 1px solid #e5e7eb;
}}
#tl-bar-wrap {{
  position: relative; height: 6px; background: #e5e7eb;
  border-radius: 4px; margin-bottom: 8px; cursor: pointer;
}}
#tl-progress {{
  height: 100%; background: #f59e0b; border-radius: 4px;
  width: 0%; transition: width 0.3s linear;
}}
.tl-marker {{
  position: absolute; top: -4px; width: 14px; height: 14px;
  background: #ef4444; border: 2px solid #fff;
  border-radius: 50%; transform: translateX(-50%);
  cursor: pointer; z-index: 5; transition: transform 0.15s;
}}
.tl-marker:hover {{ transform: translateX(-50%) scale(1.4); }}
.tl-marker-label {{
  position: absolute; top: 18px; transform: translateX(-50%);
  font-size: 9px; color: #6b7280; white-space: nowrap; font-weight: 600;
}}

/* ── Controls ── */
#controls {{
  display: flex; align-items: center; gap: 12px;
  padding: 8px 0 4px;
}}
#play-btn {{
  width: 36px; height: 36px; border-radius: 50%;
  background: #f59e0b; border: none; cursor: pointer;
  font-size: 14px; color: #fff; display: flex;
  align-items: center; justify-content: center;
  box-shadow: 0 2px 6px rgba(245,158,11,0.3);
}}
#time-display {{ font-size: 13px; color: #374151; font-variant-numeric: tabular-nums; }}
#vol-wrap {{ display: flex; align-items: center; gap: 6px; margin-left: auto; }}
#vol-wrap input {{ width: 70px; accent-color: #f59e0b; }}
#vol-icon {{ font-size: 14px; cursor: pointer; }}

/* ── Scene chips ── */
#scene-chips {{
  display: flex; flex-wrap: wrap; gap: 6px;
  padding: 10px 16px 12px; background: #fff;
  border-top: 1px solid #e5e7eb;
}}
.chip {{
  padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: 500;
  cursor: pointer; border: 1px solid #e5e7eb; background: #f9fafb; color: #374151;
  transition: all 0.15s;
}}
.chip:hover, .chip.active {{
  background: #fff7ed; border-color: #f59e0b; color: #92400e;
}}
.chip.ad-chip {{
  background: #fef3c7; border-color: #fcd34d; color: #92400e;
}}

#status-bar {{
  padding: 6px 16px; font-size: 12px; color: #6b7280;
  background: #f9fafb; border-top: 1px solid #f3f4f6;
  border-radius: 0 0 12px 12px;
}}
</style>
</head>
<body>
<div id="player-wrap">
  <video id="main-video" preload="metadata">
    <source src="data:{video_type};base64,{video_b64}" type="{video_type}">
  </video>

  <!-- Ad overlay -->
  <div id="ad-overlay">
    <div id="ad-card">
      <div id="ad-type-badge">mid-roll</div>
      <div id="ad-countdown">Skip in 5s</div>
      <div id="ad-emoji">🎬</div>
      <div id="ad-brand">Brand</div>
      <div id="ad-headline">Headline</div>
      <div id="ad-body">Body text</div>
      <div id="ad-cta" onclick="skipAd()">Learn More</div>
      <div id="ad-skip" onclick="skipAd()">Skip Ad ›</div>
    </div>
  </div>

  <!-- Timeline -->
  <div id="timeline-wrap">
    <div id="tl-bar-wrap" id="tl-bar" onclick="seekFromBar(event)">
      <div id="tl-progress"></div>
      <!-- markers injected by JS -->
    </div>
    <div id="controls">
      <button id="play-btn" onclick="togglePlay()">▶</button>
      <span id="time-display">0:00 / 0:00</span>
      <div id="vol-wrap">
        <span id="vol-icon" onclick="toggleMute()">🔊</span>
        <input type="range" id="vol-slider" min="0" max="1" step="0.05" value="1"
               oninput="setVol(this.value)">
      </div>
    </div>
  </div>

  <!-- Scene chips -->
  <div id="scene-chips">
    <span style="font-size:12px;color:#9ca3af;margin-right:4px;line-height:2">Jump to:</span>
    <!-- injected by JS -->
  </div>

  <div id="status-bar">Ready — press play</div>
</div>

<script>
var VIDEO = document.getElementById('main-video');
var MARKERS = {markers_json};
var POST_ROLL = {{
  sec: -1, type: "post-roll",
  ad_id: "post", ad_brand: "{post_roll_ad["brand"]}",
  ad_title: "{post_roll_ad["title"]}",
  ad_headline: "{post_roll_ad["headline"]}",
  ad_body: "{post_roll_ad["body"]}",
  ad_cta: "{post_roll_ad["cta"]}",
  ad_emoji: "{post_roll_ad["emoji"]}",
  ad_bg: "{post_roll_ad["bg"]}",
  ad_color: "{post_roll_ad["color"]}",
}};

var shownMarkers = {{}};
var adCountdown = null;
var skipAfter = 5;
var currentAd = null;
var videoEnded = false;

// ── Build timeline markers ──
VIDEO.addEventListener('loadedmetadata', function() {{
  var dur = VIDEO.duration;
  var barWrap = document.getElementById('tl-bar-wrap');
  MARKERS.forEach(function(m) {{
    if (m.sec <= 0) return;  // skip pre-roll marker
    var pct = (m.sec / dur) * 100;
    var dot = document.createElement('div');
    dot.className = 'tl-marker';
    dot.style.left = pct + '%';
    dot.title = m.ad_brand + ' at ' + fmtTime(m.sec);
    dot.onclick = function(e) {{ e.stopPropagation(); VIDEO.currentTime = m.sec; }};
    var lbl = document.createElement('div');
    lbl.className = 'tl-marker-label';
    lbl.style.left = pct + '%';
    lbl.textContent = '📢 ' + fmtTime(m.sec);
    barWrap.appendChild(dot);
    barWrap.appendChild(lbl);
  }});
  // Build scene chips from markers
  var chips = document.getElementById('scene-chips');
  MARKERS.forEach(function(m) {{
    var chip = document.createElement('span');
    chip.className = 'chip ad-chip';
    chip.textContent = m.ad_emoji + ' ' + m.ad_brand + ' @ ' + (m.sec > 0 ? fmtTime(m.sec) : 'start');
    chip.onclick = function() {{
      if (m.sec > 0) VIDEO.currentTime = m.sec - 1;
      VIDEO.play();
    }};
    chips.appendChild(chip);
  }});
  updateStatus('Video ready · ' + MARKERS.length + ' ad markers · Press play');
}});

// ── Timeupdate: check for markers ──
VIDEO.addEventListener('timeupdate', function() {{
  var t = VIDEO.currentTime;
  var dur = VIDEO.duration || 1;
  document.getElementById('tl-progress').style.width = (t/dur*100) + '%';
  document.getElementById('time-display').textContent = fmtTime(t) + ' / ' + fmtTime(dur);
  document.getElementById('play-btn').textContent = VIDEO.paused ? '▶' : '⏸';

  // Check markers
  MARKERS.forEach(function(m) {{
    if (!shownMarkers[m.ad_id + '_' + m.sec] && t >= m.sec && m.sec >= 0) {{
      shownMarkers[m.ad_id + '_' + m.sec] = true;
      showAd(m);
    }}
  }});
}});

// ── Post-roll ──
VIDEO.addEventListener('ended', function() {{
  if (!videoEnded) {{
    videoEnded = true;
    showAd(POST_ROLL);
  }}
}});

// ── Show ad overlay ──
function showAd(m) {{
  VIDEO.pause();
  currentAd = m;
  var ov = document.getElementById('ad-overlay');
  document.getElementById('ad-card').style.background = m.ad_bg;
  document.getElementById('ad-type-badge').textContent = m.type || 'mid-roll';
  document.getElementById('ad-emoji').textContent = m.ad_emoji;
  document.getElementById('ad-brand').textContent = m.ad_brand;
  document.getElementById('ad-headline').textContent = m.ad_headline;
  document.getElementById('ad-body').textContent = m.ad_body;
  document.getElementById('ad-cta').textContent = m.ad_cta;
  ov.style.display = 'flex';
  updateStatus('📢 Ad playing: ' + m.ad_brand + ' — ' + m.ad_title);

  // Countdown
  var secs = skipAfter;
  document.getElementById('ad-countdown').textContent = 'Skip in ' + secs + 's';
  adCountdown = setInterval(function() {{
    secs--;
    if (secs <= 0) {{
      document.getElementById('ad-countdown').textContent = 'Skip Ad ›';
      document.getElementById('ad-skip').style.display = 'block';
      clearInterval(adCountdown);
    }} else {{
      document.getElementById('ad-countdown').textContent = 'Skip in ' + secs + 's';
    }}
  }}, 1000);
}}

function skipAd() {{
  clearInterval(adCountdown);
  document.getElementById('ad-overlay').style.display = 'none';
  currentAd = null;
  if (!videoEnded) VIDEO.play();
  updateStatus('Playing');
}}

function togglePlay() {{
  if (VIDEO.paused) {{ VIDEO.play(); updateStatus('Playing'); }}
  else {{ VIDEO.pause(); updateStatus('Paused'); }}
}}

function seekFromBar(e) {{
  var bar = document.getElementById('tl-bar-wrap');
  var rect = bar.getBoundingClientRect();
  var pct = (e.clientX - rect.left) / rect.width;
  VIDEO.currentTime = pct * (VIDEO.duration || 0);
}}

function setVol(v) {{ VIDEO.volume = v; }}
function toggleMute() {{
  VIDEO.muted = !VIDEO.muted;
  document.getElementById('vol-icon').textContent = VIDEO.muted ? '🔇' : '🔊';
}}
function fmtTime(s) {{
  s = Math.floor(s || 0);
  var m = Math.floor(s/60); var ss = s%60;
  return m + ':' + (ss < 10 ? '0' : '') + ss;
}}
function updateStatus(msg) {{
  document.getElementById('status-bar').textContent = msg;
}}

// Trigger pre-roll immediately on first play
VIDEO.addEventListener('play', function onFirstPlay() {{
  if (!shownMarkers['pre-roll']) {{
    shownMarkers['pre-roll'] = true;
    var preRoll = MARKERS.find(function(m) {{ return m.sec === 0; }});
    if (preRoll) {{
      VIDEO.pause();
      showAd(preRoll);
    }}
  }}
  VIDEO.removeEventListener('play', onFirstPlay);
}}, {{once: true}});
</script>
</body>
</html>"""

        # Show marker summary above player
        if all_markers:
            c1, c2, c3 = st.columns(3)
            c1.metric("Ad Markers", len(all_markers))
            c2.metric("Pre-roll", "1")
            c3.metric("Mid-roll", len([m for m in markers]))

        st.components.v1.html(html, height=640, scrolling=False)

        st.divider()
        if all_markers:
            st.caption("**AD SCHEDULE**")
            rows = []
            for m in all_markers:
                rows.append({
                    "Type": m.get("type", "mid-roll"),
                    "Timestamp": "Pre-roll (0s)" if m["sec"] == 0 else f"{m['fmt'] if 'fmt' in m else fmtTime_py(m['sec'])}",
                    "Brand": m["ad"]["brand"],
                    "Ad": m["ad"]["title"],
                    "CTA": m["ad"]["cta"],
                })
            rows.append({
                "Type": "post-roll",
                "Timestamp": "End of video",
                "Brand": post_roll_ad["brand"],
                "Ad": post_roll_ad["title"],
                "CTA": post_roll_ad["cta"],
            })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def fmtTime_py(s):
    s = int(s or 0)
    return f"{s//60}:{s%60:02d}"


# ── Router ─────────────────────────────────────────────────────────────────────
pages = {
    "process":   page_process,
    "watch":     page_watch,
    "search":    page_search,
    "ads":       page_ads,
    "analytics": page_analytics,
    "demo":      page_demo,
}
pages.get(st.session_state.page, page_process)()
