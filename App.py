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
_BG    = "#ffffff"
_GRID  = "#e5e7eb"
_TEXT  = "#6b7280"
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
        color_map = {"positive": "#34d399", "neutral": "#6b7280", "negative": "#f87171"}
        pie_labels = list(lc.keys())
        pie_values = list(lc.values())
        pie_colors = [color_map.get(l, "#9ca3af") for l in pie_labels]
        fig = go.Figure(go.Pie(
            values=pie_values,
            labels=pie_labels,
            hole=0.55,
            marker=dict(colors=pie_colors),
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
# PAGE — VIDEO AD DEMO


# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — VIDEO AD DEMO  v3
# ══════════════════════════════════════════════════════════════════════════════
def page_demo():
    import base64 as _b64, json as _json, math, re as _re

    # ── Built-in ads ───────────────────────────────────────────────────────
    BUILTIN_ADS = [
        {"id":"ad1","brand":"Nike","title":"Just Do It","emoji":"👟",
         "cta":"Shop Now","bg":"linear-gradient(135deg,#f59e0b,#d97706)",
         "headline":"Push Your Limits","body":"New season collection — built for champions.",
         "tags":"sports fitness action competition energy lifestyle motivation exercise athletic running"},
        {"id":"ad2","brand":"Spotify","title":"Music for Every Mood","emoji":"🎵",
         "cta":"Listen Free","bg":"linear-gradient(135deg,#1db954,#158a3e)",
         "headline":"Soundtrack Your Life","body":"3 months Premium free — no ads, offline play.",
         "tags":"music entertainment emotion arts drama movies relaxation streaming audio"},
        {"id":"ad3","brand":"Amazon","title":"Deals of the Day","emoji":"📦",
         "cta":"Shop Deals","bg":"linear-gradient(135deg,#ff9900,#e47911)",
         "headline":"Today Only — Up to 60% Off","body":"Lightning deals on electronics, home & more.",
         "tags":"shopping technology gadgets home lifestyle deals consumer ecommerce retail"},
        {"id":"ad4","brand":"Netflix","title":"Stories Worth Watching","emoji":"🎬",
         "cta":"Watch Now","bg":"linear-gradient(135deg,#e50914,#a30610)",
         "headline":"New Episodes Every Week","body":"Award-winning series — start streaming today.",
         "tags":"entertainment drama story adventure fiction film celebrity arts television streaming"},
        {"id":"ad5","brand":"Duolingo","title":"Learn a Language","emoji":"🦜",
         "cta":"Start Free","bg":"linear-gradient(135deg,#58cc02,#3d9900)",
         "headline":"5 Minutes a Day Changes Everything","body":"40+ languages. Free forever.",
         "tags":"education learning language travel culture knowledge students school skills"},
        {"id":"ad6","brand":"Uber Eats","title":"Food at Your Door","emoji":"🍔",
         "cta":"Order Now","bg":"linear-gradient(135deg,#06c167,#038a47)",
         "headline":"Craving Something?","body":"Your favourite restaurants in 30 minutes.",
         "tags":"food cooking restaurant lifestyle family celebration delivery dining parenting"},
        {"id":"ad7","brand":"Mastercard","title":"Priceless Moments","emoji":"💳",
         "cta":"Learn More","bg":"linear-gradient(135deg,#eb5757,#b91c1c)",
         "headline":"There Are Things Money Can't Buy","body":"For everything else, there's Mastercard.",
         "tags":"finance business economy success achievement luxury banking news career"},
        {"id":"ad8","brand":"BMW","title":"The Ultimate Drive","emoji":"🚗",
         "cta":"Book Test Drive","bg":"linear-gradient(135deg,#1e40af,#1e3a8a)",
         "headline":"Sheer Driving Pleasure","body":"New BMW 5 Series. Redefining performance.",
         "tags":"automotive cars speed luxury engineering technology premium travel"},
    ]

    # ── Session state ──────────────────────────────────────────────────────
    for k, v in {
        "demo_vm": None, "demo_video_b64": None, "demo_video_type": "video/mp4",
        "demo_markers": [], "demo_analysed": False,
        "custom_ads": [], "demo_video_meta": {},
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    def all_ads():
        return BUILTIN_ADS + st.session_state.custom_ads

    # ── Tokenise a string into a normalised word set ───────────────────────
    def _tok(text: str) -> set:
        """Lowercase, strip punctuation, split on spaces/commas/&."""
        text = text.lower()
        text = _re.sub(r"[&,\-/|]+", " ", text)
        return {w for w in text.split() if len(w) > 2}

    # ── Strong similarity: multi-signal ───────────────────────────────────
    def _similarity(ad: dict, scene) -> dict:
        """
        Returns dict with overall score + breakdown per signal:
          tag_overlap, iab_overlap, text_overlap, sentiment_boost, engagement_boost
        """
        # Ad token set (tags field is a space-separated string or list)
        tags = ad.get("tags", "")
        if isinstance(tags, list):
            tags = " ".join(tags)
        ad_tokens = _tok(tags)

        # Scene signal 1: IAB category tokens
        iab_full = " ".join(c["name"] for c in scene.iab_categories[:4])
        iab_tokens = _tok(iab_full)

        # Scene signal 2: scene text tokens (top 60 words)
        text_tokens = _tok(" ".join(scene.text.split()[:60]))

        # Scene signal 3: combined
        scene_all = iab_tokens | text_tokens

        # Overlap counts
        tag_iab  = len(ad_tokens & iab_tokens)
        tag_text = len(ad_tokens & text_tokens)
        tag_all  = len(ad_tokens & scene_all)
        union    = len(ad_tokens | scene_all) or 1

        # Jaccard on combined
        jaccard = tag_all / union

        # Weighted sub-scores
        iab_score  = min(tag_iab  / max(len(ad_tokens), 1), 1.0) * 0.45
        text_score = min(tag_text / max(len(ad_tokens), 1), 1.0) * 0.30
        jac_score  = jaccard * 0.15

        # Boosts
        eng_boost  = scene.engagement_score * 0.06
        safe_boost = scene.brand_safety.get("safety_score", 1.0) * 0.04
        sent = scene.sentiment.get("label", "neutral")
        sent_boost = 0.03 if sent == "positive" else 0.0

        total = min(iab_score + text_score + jac_score + eng_boost + safe_boost + sent_boost, 1.0)
        return {
            "total":      round(total, 3),
            "iab":        round(iab_score,  3),
            "text":       round(text_score, 3),
            "jaccard":    round(jac_score,  3),
            "engagement": round(eng_boost,  3),
            "safety":     round(safe_boost, 3),
            "matched_iab":  sorted(ad_tokens & iab_tokens),
            "matched_text": sorted(ad_tokens & text_tokens)[:6],
        }

    def _best_ad(scene):
        scored = [(ad, _similarity(ad, scene)) for ad in all_ads()]
        scored.sort(key=lambda x: x[1]["total"], reverse=True)
        return scored[0]

    def _top_ads(scene, n=3):
        scored = [(ad, _similarity(ad, scene)) for ad in all_ads()]
        scored.sort(key=lambda x: x[1]["total"], reverse=True)
        return scored[:n]

    def _fmt(s):
        s = int(s or 0)
        return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

    def _score_bar(score, label=""):
        pct = int(score * 100)
        color = "#16a34a" if score > 0.45 else "#f59e0b" if score > 0.25 else "#9ca3af"
        return (
            f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
            f'<div style="flex:1;height:6px;background:#f3f4f6;border-radius:3px">'
            f'<div style="width:{pct}%;height:6px;background:{color};border-radius:3px;'
            f'transition:width .4s"></div></div>'
            f'<span style="font-size:12px;font-weight:700;color:{color};min-width:32px">{pct}%</span>'
            + (f'<span style="font-size:11px;color:#9ca3af">{label}</span>' if label else '')
            + '</div>'
        )

    # ══════════════════════════════════════════════════════════════════════
    st.markdown("## 🎥 Video Ad Demo")
    st.caption("Upload video + subtitles → AI generates metadata & tags → similarity-ranked ad matching → live playback")
    st.divider()

    t1, t2, t3 = st.tabs(["📁  Upload & Analyse", "📢  Ad Library", "▶️  Watch with Ads"])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — UPLOAD & ANALYSE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with t1:
        # Upload section
        with st.expander("**Step 1 — Upload files**",
                         expanded=not st.session_state.demo_analysed):
            ca, cb = st.columns([3, 2], gap="large")
            with ca:
                video_file = st.file_uploader("Video (MP4 / MOV / WebM)",
                    type=["mp4","mov","webm","avi"], key="dv_file")
                srt_file   = st.file_uploader("Subtitle (.srt or .vtt) — required for analysis",
                    type=["srt","vtt"], key="ds_file")
                vid_title  = st.text_input("Title", placeholder="e.g. Spider-Man Episode 3", key="dv_title")
            with cb:
                st.markdown("**Detection settings**")
                min_s = st.slider("Min scene (s)", 10, 90, 20, 5,  key="dm_min")
                max_s = st.slider("Max scene (s)", 60, 300, 120, 10, key="dm_max")
                sens  = st.slider("Sensitivity",  0.2, 0.7, 0.35, 0.05, key="dm_sens")
            if not srt_file:
                st.info("📌 Upload a subtitle file to enable analysis")
            if srt_file and st.button("⚡ Analyse & Generate Metadata", type="primary", key="dm_go"):
                srt_content = srt_file.read().decode("utf-8", errors="replace")
                fmt_s = "vtt" if srt_file.name.lower().endswith(".vtt") else "srt"
                title = vid_title.strip() or (video_file.name if video_file else "Demo Video")
                with st.spinner("Detecting scenes · classifying IAB categories · analysing sentiment…"):
                    t0 = time.time()
                    vm = VideoProcessor(min_s, max_s, sens).process_file(srt_content, title, fmt_s)
                    elapsed = time.time() - t0
                if not vm.scenes:
                    st.error("No scenes detected."); st.stop()
                _register(vm)
                st.session_state.demo_vm = vm
                st.session_state.demo_analysed = True
                st.session_state.demo_markers = []
                if video_file:
                    video_file.seek(0)
                    raw = video_file.read()
                    st.session_state.demo_video_b64 = _b64.b64encode(raw).decode()
                    ext = video_file.name.split(".")[-1].lower()
                    st.session_state.demo_video_type = {
                        "mp4":"video/mp4","mov":"video/mp4",
                        "webm":"video/webm","avi":"video/x-msvideo"}.get(ext,"video/mp4")
                st.session_state.demo_video_meta = {
                    "title": title, "duration": vm.fmt_duration(),
                    "scenes": vm.scene_count, "key_moments": len(vm.key_scenes),
                    "narrative": vm.narrative_structure,
                    "top_iab": [c["name"] for c in vm.dominant_iab[:6]],
                    "avg_eng": round(sum(s.engagement_score for s in vm.scenes)/max(vm.scene_count,1),3),
                    "brand_safe": f"{sum(1 for s in vm.scenes if s.brand_safety.get('safety_score',1)>=0.7)/max(vm.scene_count,1):.0%}",
                }
                st.success(f"✅ {vm.scene_count} scenes analysed in {elapsed:.1f}s")
                st.rerun()

        if not st.session_state.demo_analysed or not st.session_state.demo_vm:
            return
        vm   = st.session_state.demo_vm
        meta = st.session_state.demo_video_meta

        # ── Metadata card ──
        st.markdown("### 🏷️ Video Metadata & Content Tags")
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Scenes",       meta["scenes"])
        c2.metric("Duration",     meta["duration"])
        c3.metric("Key Moments",  meta["key_moments"])
        c4.metric("Avg Engagement", meta["avg_eng"])
        c5.metric("Brand Safe",   meta["brand_safe"])

        m_col, t_col = st.columns(2)
        with m_col:
            with st.container(border=True):
                st.markdown("**Video metadata**")
                st.markdown(f"**Title:** {meta['title']}")
                st.markdown(f"**Narrative:** `{meta['narrative']}`")
        with t_col:
            with st.container(border=True):
                st.markdown("**IAB Content Tags** *(matched against ad tags)*")
                tag_html = "".join(
                    f'<span style="display:inline-block;background:#fef3c7;border:1px solid #fcd34d;'
                    f'color:#92400e;padding:3px 10px;border-radius:14px;font-size:12px;'
                    f'font-weight:500;margin:3px">{t}</span>' for t in meta["top_iab"])
                st.markdown(tag_html + '<div style="margin-top:6px;font-size:11px;color:#9ca3af">'
                    'These tags + scene text are matched against ad tags using multi-signal similarity</div>',
                    unsafe_allow_html=True)

        st.divider()

        # ── Scene intelligence ──
        st.markdown("### 🎬 Scene-by-Scene Intelligence & Ad Matching")
        st.caption("Similarity = IAB overlap (45%) + text overlap (30%) + Jaccard (15%) + engagement (6%) + safety (4%)")

        for i, scene in enumerate(vm.scenes):
            is_key = scene.scene_id in vm.key_scenes
            safety = scene.brand_safety.get("safety_score", 1.0)
            sent   = scene.sentiment.get("label","neutral")
            sent_icon = {"positive":"🟢","negative":"🔴","neutral":"🔵"}.get(sent,"🔵")
            top_matches = _top_ads(scene, 3)
            best_sim = top_matches[0][1]["total"] if top_matches else 0

            with st.container(border=True):
                h1, h2, h3, h4, h5 = st.columns([3,1,1,1,1])
                h1.markdown(f"**Scene {i+1}** {'⭐' if is_key else ''} — `{scene.start_fmt}→{scene.end_fmt}`")
                h2.caption(f"{sent_icon} {sent}")
                h3.caption(f"🛡 {safety:.0%}")
                h4.caption(f"⚡ {scene.engagement_score:.2f}")
                sc = "#16a34a" if best_sim>0.45 else "#f59e0b" if best_sim>0.25 else "#9ca3af"
                h5.markdown(f'<div style="padding-top:4px"><span style="background:#f9fafb;'
                    f'border:1px solid {sc};color:{sc};font-weight:700;padding:3px 8px;'
                    f'border-radius:8px;font-size:12px">🎯 {best_sim:.0%}</span></div>',
                    unsafe_allow_html=True)

                st.caption(scene.text[:160] + ("…" if len(scene.text)>160 else ""))

                # IAB tokens for this scene (shown so user can debug matching)
                scene_iab_toks = _tok(" ".join(c["name"] for c in scene.iab_categories[:4]))
                chip_html = "".join(
                    f'<span style="background:#f0fdf4;border:1px solid #bbf7d0;color:#166534;'
                    f'padding:2px 7px;border-radius:8px;font-size:11px;margin:2px">{c["name"]}</span>'
                    for c in scene.iab_categories[:4])
                if chip_html:
                    st.markdown(chip_html, unsafe_allow_html=True)
                st.markdown("")

                # Top 3 ad matches with full breakdown
                cols = st.columns(3)
                for j, (ad, sim) in enumerate(top_matches):
                    with cols[j]:
                        is_custom = ad.get("id","").startswith("custom")
                        badge = ' <span style="background:#dbeafe;color:#1d4ed8;font-size:9px;padding:1px 5px;border-radius:4px">CUSTOM</span>' if is_custom else ""
                        st.markdown(
                            f'<div style="border:1px solid {"#bfdbfe" if is_custom else "#e5e7eb"};'
                            f'border-radius:10px;padding:10px;background:{"#eff6ff" if is_custom else "#fafafa"}">'
                            f'<div style="font-size:13px;font-weight:700">{ad["emoji"]} {ad["brand"]}{badge}</div>'
                            f'<div style="font-size:11px;color:#6b7280;margin:2px 0 6px">{ad["title"]}</div>'
                            + _score_bar(sim["total"])
                            + f'<div style="font-size:10px;color:#9ca3af;margin-top:5px;line-height:1.6">'
                            f'IAB: {sim["iab"]:.0%} · text: {sim["text"]:.0%}<br>'
                            + (f'✅ <b>{", ".join(sim["matched_iab"][:4])}</b>' if sim["matched_iab"] else '⚠️ no IAB overlap')
                            + '</div></div>',
                            unsafe_allow_html=True)

        st.divider()

        # ── Ad plan ──
        st.markdown("### 📢 Ad Placement Plan")
        left, right = st.columns([3, 2], gap="large")

        with right:
            with st.container(border=True):
                st.markdown("**Auto-generate**")
                n_mid = st.slider("Mid-roll count", 1, min(8,vm.scene_count), min(3,vm.scene_count), key="dm_n")
                strat = st.radio("Strategy", ["🏆 Top similarity match","🔀 Evenly spaced","⭐ Key moments"], key="dm_str")
                if st.button("🤖 Generate Plan", type="primary", key="dm_gen"):
                    scenes = vm.scenes
                    if "similarity" in strat:
                        cands = sorted(scenes, key=lambda s: _best_ad(s)[1]["total"]*s.ad_suitability, reverse=True)[:n_mid]
                    elif "spaced" in strat:
                        step = max(1, len(scenes)//(n_mid+1))
                        cands = [scenes[i*step] for i in range(1,n_mid+1) if i*step<len(scenes)]
                    else:
                        cands = [s for s in scenes if s.scene_id in vm.key_scenes][:n_mid] \
                                or sorted(scenes, key=lambda s: s.engagement_score, reverse=True)[:n_mid]
                    new_markers = []
                    ad0, sim0 = _best_ad(scenes[0])
                    new_markers.append({"sec":0,"fmt":"00:00:00","ad":ad0,"mode":"auto",
                        "duration":15,"sim":sim0["total"],
                        "reason":f"Pre-roll · IAB: {', '.join(sim0['matched_iab'][:3]) or 'general'} · {sim0['total']:.0%}"})
                    for sc in cands:
                        ad, sim = _best_ad(sc)
                        iab_str = " · ".join(c["name"] for c in sc.iab_categories[:2])
                        reason = (f"{iab_str} · {sim['total']:.0%} match"
                                  + (" · ⭐ key" if sc.scene_id in vm.key_scenes else "")
                                  + (f" · matched: {', '.join(sim['matched_iab'][:3])}" if sim['matched_iab'] else ""))
                        new_markers.append({"sec":sc.start_sec,"fmt":_fmt(sc.start_sec),
                            "ad":ad,"mode":"auto","duration":15,"sim":sim["total"],"reason":reason})
                    st.session_state.demo_markers = new_markers
                    st.success(f"✅ {len(new_markers)} markers — edit below then Save")
                    st.rerun()

            with st.container(border=True):
                st.markdown("**Add manually**")
                new_ts  = st.text_input("Timestamp", placeholder="00:01:30", key="dm_mts")
                ad_opts = [f"{a['emoji']} {a['brand']} — {a['title']}" for a in all_ads()]
                ad_pick = st.selectbox("Ad", ad_opts, key="dm_mad")
                dur_p   = st.number_input("Duration (s)", 5, 60, 15, key="dm_mdur")
                if st.button("➕ Add", key="dm_madd"):
                    sec = _parse_ts(new_ts)
                    if sec is not None:
                        if sec not in [m["sec"] for m in st.session_state.demo_markers]:
                            ad_obj = all_ads()[ad_opts.index(ad_pick)]
                            closest = min(vm.scenes, key=lambda s: abs(s.start_sec-sec))
                            sim = _similarity(ad_obj, closest)
                            st.session_state.demo_markers.append({
                                "sec":sec,"fmt":_fmt(sec),"ad":ad_obj,"mode":"manual",
                                "duration":int(dur_p),"sim":sim["total"],"reason":"Manual"})
                        st.rerun()
                    else:
                        st.error("Invalid timestamp")

        with left:
            markers = st.session_state.demo_markers
            if not markers:
                st.info("Generate a plan or add markers →")
            else:
                sorted_m = sorted(markers, key=lambda x: x["sec"])
                ad_opts2 = [f"{a['emoji']} {a['brand']} — {a['title']}" for a in all_ads()]

                for i, m in enumerate(sorted_m):
                    type_lbl = "🟡 Pre-roll" if m["sec"]==0 else "🔴 Mid-roll"
                    sim_c = "#16a34a" if m["sim"]>0.45 else "#f59e0b" if m["sim"]>0.25 else "#9ca3af"
                    with st.container(border=True):
                        t_col, s_col = st.columns([3,2])
                        with t_col:
                            st.markdown(f"**{type_lbl}** · {'🤖' if m['mode']=='auto' else '✋'}")
                            st.text_input("Time", value=m["fmt"], key=f"m_ts_{i}")
                        with s_col:
                            st.markdown(
                                f'<div style="padding-top:30px;text-align:right">'
                                f'<span style="font-size:18px;font-weight:800;color:{sim_c}">🎯 {m["sim"]:.0%}</span><br>'
                                f'<span style="font-size:10px;color:#9ca3af">similarity</span></div>',
                                unsafe_allow_html=True)
                        cur_lbl = f"{m['ad']['emoji']} {m['ad']['brand']} — {m['ad']['title']}"
                        cur_idx = ad_opts2.index(cur_lbl) if cur_lbl in ad_opts2 else 0
                        st.selectbox("Ad", ad_opts2, index=cur_idx, key=f"m_ad_{i}")
                        d_col, r_col = st.columns([4,1])
                        d_col.slider("Duration (s)", 5, 60, int(m.get("duration",15)), key=f"m_dur_{i}")
                        if r_col.button("🗑", key=f"m_del_{i}"):
                            st.session_state.demo_markers = [x for x in markers if x["sec"]!=m["sec"]]
                            st.rerun()
                        st.caption(m.get("reason",""))

                if st.button("💾 Save & Go to Player →", type="primary", key="dm_save"):
                    saved = []
                    for i, m in enumerate(sorted_m):
                        raw_ts   = st.session_state.get(f"m_ts_{i}", m["fmt"])
                        new_sec  = _parse_ts(raw_ts) or m["sec"]
                        ad_lbl   = st.session_state.get(f"m_ad_{i}",
                                       f"{m['ad']['emoji']} {m['ad']['brand']} — {m['ad']['title']}")
                        new_ad   = all_ads()[ad_opts2.index(ad_lbl)] if ad_lbl in ad_opts2 else m["ad"]
                        new_dur  = int(st.session_state.get(f"m_dur_{i}", m.get("duration",15)))
                        closest  = min(vm.scenes, key=lambda s: abs(s.start_sec-new_sec))
                        new_sim  = _similarity(new_ad, closest)["total"]
                        saved.append({**m, "sec":new_sec, "fmt":_fmt(new_sec),
                                      "ad":new_ad, "duration":new_dur, "sim":new_sim})
                    st.session_state.demo_markers = saved
                    st.success(f"✅ Saved {len(saved)} markers — open the **Watch with Ads** tab ▶️")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — AD LIBRARY
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with t2:
        st.markdown("#### Built-in Ads")
        st.caption("These ads have keyword tags that are matched against scene IAB categories and subtitle text.")
        cols = st.columns(4)
        for i, ad in enumerate(BUILTIN_ADS):
            tags_list = ad["tags"].split() if isinstance(ad["tags"], str) else ad["tags"]
            with cols[i%4]:
                with st.container(border=True):
                    st.markdown(f"{ad['emoji']} **{ad['brand']}**")
                    st.caption(ad["title"])
                    tag_preview = " · ".join(tags_list[:5])
                    st.caption(f"🏷️ {tag_preview}")

        st.divider()
        st.markdown("#### ➕ Add Custom Ad")
        st.caption("Use the **same words** as IAB categories shown in scene analysis above — e.g. `arts entertainment`, `family parenting`, `news` — for best matching. Separate with commas or spaces.")

        with st.container(border=True):
            # Show current video tags as hint
            if st.session_state.demo_analysed and st.session_state.demo_video_meta.get("top_iab"):
                iab_hint = " · ".join(st.session_state.demo_video_meta["top_iab"])
                st.info(f"💡 **Video tags to target:** {iab_hint}")

            ua, ub = st.columns(2)
            with ua:
                c_brand    = st.text_input("Brand name", key="ca_brand")
                c_title    = st.text_input("Ad title / campaign", key="ca_title")
                c_headline = st.text_input("Headline", key="ca_hl")
                c_body     = st.text_area("Body copy", key="ca_body", height=75)
                c_cta      = st.text_input("CTA button", value="Learn More", key="ca_cta")
                c_emoji    = st.text_input("Emoji", value="📣", key="ca_emoji")
            with ub:
                st.markdown("**🏷️ Content tags for matching**")
                st.caption("These are tokenised and matched against scene IAB tags and subtitle text. "
                           "More relevant keywords = higher similarity score.")
                c_tags = st.text_area("Tags (comma or space separated)", key="ca_tags", height=120,
                    placeholder="arts entertainment, family parenting, news, sports")
                # Live preview: show which video scenes this would match
                if c_tags.strip() and st.session_state.demo_vm:
                    dummy_ad = {"tags": c_tags, "id":"preview"}
                    preview_scores = [(s, _similarity(dummy_ad, s)["total"])
                                      for s in st.session_state.demo_vm.scenes]
                    preview_scores.sort(key=lambda x: x[1], reverse=True)
                    top3 = preview_scores[:3]
                    if top3[0][1] > 0:
                        st.markdown("**Live preview** — top matching scenes:")
                        for sc, sim in top3:
                            bar_w = int(sim*100)
                            c = "#16a34a" if sim>0.45 else "#f59e0b" if sim>0.25 else "#9ca3af"
                            st.markdown(
                                f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0;'
                                f'font-size:12px">'
                                f'<code style="color:#374151">{sc.start_fmt}</code>'
                                f'<div style="flex:1;height:5px;background:#e5e7eb;border-radius:3px">'
                                f'<div style="width:{bar_w}%;height:5px;background:{c};border-radius:3px"></div></div>'
                                f'<b style="color:{c}">{sim:.0%}</b></div>',
                                unsafe_allow_html=True)
                    else:
                        st.warning("No strong matches yet — try adding more relevant keywords")
                c_c1 = st.color_picker("Gradient start", "#6366f1", key="ca_c1")
                c_c2 = st.color_picker("Gradient end",   "#8b5cf6", key="ca_c2")

            if st.button("➕ Add to Library", key="ca_add", type="primary"):
                if c_brand and c_title and c_tags.strip():
                    new_ad = {
                        "id": f"custom_{len(st.session_state.custom_ads)+1}",
                        "brand": c_brand, "title": c_title,
                        "headline": c_headline or c_title,
                        "body": c_body or f"{c_brand} — {c_title}",
                        "cta": c_cta, "emoji": c_emoji,
                        "bg": f"linear-gradient(135deg,{c_c1},{c_c2})",
                        "tags": c_tags,  # keep raw — _tok() handles parsing
                    }
                    st.session_state.custom_ads.append(new_ad)
                    st.success(f"✅ Added **{c_brand}** — regenerate the plan to see it in matches")
                    st.rerun()
                else:
                    st.error("Brand, title and tags are all required")

        if st.session_state.custom_ads:
            st.divider()
            st.markdown("#### Your Custom Ads")
            for i, ad in enumerate(st.session_state.custom_ads):
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2, 5, 1])
                    c1.markdown(f"{ad['emoji']} **{ad['brand']}** — {ad['title']}")
                    # Show top match for this ad if video analysed
                    if st.session_state.demo_vm:
                        top_sc = max(st.session_state.demo_vm.scenes,
                                     key=lambda s: _similarity(ad, s)["total"])
                        top_sim = _similarity(ad, top_sc)
                        sc = "#16a34a" if top_sim["total"]>0.45 else "#f59e0b" if top_sim["total"]>0.25 else "#9ca3af"
                        c2.markdown(
                            f'Best match: `{top_sc.start_fmt}` — '
                            f'<span style="color:{sc};font-weight:700">{top_sim["total"]:.0%}</span> · '
                            f'IAB: {", ".join(top_sim["matched_iab"][:4]) or "none"} · '
                            f'text: {", ".join(top_sim["matched_text"][:3]) or "none"}',
                            unsafe_allow_html=True)
                    else:
                        c2.caption("🏷️ " + (ad["tags"] if isinstance(ad["tags"],str) else " ".join(ad["tags"]))[:60])
                    if c3.button("🗑", key=f"del_ca_{i}"):
                        st.session_state.custom_ads.pop(i); st.rerun()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3 — WATCH WITH ADS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with t3:
        if not st.session_state.demo_analysed:
            st.info("Complete Upload & Analyse first."); return
        if not st.session_state.demo_video_b64:
            st.warning("No video uploaded — add an MP4 in Tab 1."); return
        if not st.session_state.demo_markers:
            st.info("Add ad markers in Tab 1 first."); return

        vm      = st.session_state.demo_vm
        markers = sorted(st.session_state.demo_markers, key=lambda x: x["sec"])
        post_ad, post_sim_d = _best_ad(vm.scenes[-1])
        post_sim = post_sim_d["total"]

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Pre-roll",    sum(1 for m in markers if m["sec"]==0))
        c2.metric("Mid-roll",    sum(1 for m in markers if m["sec"]>0))
        c3.metric("Post-roll",   1)
        c4.metric("Auto-matched",sum(1 for m in markers if m.get("mode")=="auto"))
        avg_sim = sum(m["sim"] for m in markers)/max(len(markers),1)
        c5.metric("Avg match",   f"{avg_sim:.0%}")

        video_b64  = st.session_state.demo_video_b64
        video_type = st.session_state.demo_video_type

        markers_js = _json.dumps([{
            "sec":m["sec"],"fmt":m["fmt"],
            "type":"pre-roll" if m["sec"]==0 else "mid-roll",
            "mode":m.get("mode","manual"),"sim":round(m.get("sim",0),3),
            "duration":m.get("duration",15),"reason":m.get("reason",""),
            "ad_brand":m["ad"]["brand"],"ad_title":m["ad"]["title"],
            "ad_headline":m["ad"]["headline"],"ad_body":m["ad"]["body"],
            "ad_cta":m["ad"]["cta"],"ad_emoji":m["ad"]["emoji"],"ad_bg":m["ad"]["bg"],
        } for m in markers])

        post_js = _json.dumps({
            "sec":-1,"fmt":"end","type":"post-roll","mode":"auto","sim":round(post_sim,3),
            "duration":15,"reason":"Best match for final scene",
            "ad_brand":post_ad["brand"],"ad_title":post_ad["title"],
            "ad_headline":post_ad["headline"],"ad_body":post_ad["body"],
            "ad_cta":post_ad["cta"],"ad_emoji":post_ad["emoji"],"ad_bg":post_ad["bg"],
        })

        scenes_js = _json.dumps([{
            "sec":s.start_sec,"fmt":s.start_fmt,"key":s.scene_id in vm.key_scenes,
            "label":s.text[:35].replace('"','').replace("\\",""),
            "eng":round(s.engagement_score,2),
        } for s in vm.scenes])

        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:Inter,sans-serif}}
body{{background:#f9fafb;padding:10px}}
#wrap{{background:#fff;border-radius:14px;border:1px solid #e5e7eb;
       box-shadow:0 4px 24px rgba(0,0,0,.08);overflow:hidden}}
#vw{{position:relative;background:#000;aspect-ratio:16/9;max-height:440px}}
video{{width:100%;height:100%;display:block;object-fit:contain;background:#000}}

#adov{{display:none;position:absolute;inset:0;z-index:30;
       background:rgba(0,0,0,.82);align-items:center;justify-content:center}}
#adcard{{width:min(92%,520px);border-radius:20px;padding:30px 36px;color:#fff;
          text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.5);position:relative}}
.badge{{position:absolute;padding:4px 11px;border-radius:10px;font-size:10px;
         font-weight:700;text-transform:uppercase;letter-spacing:.1em;background:rgba(0,0,0,.4)}}
#b-type{{top:14px;left:14px}}
#b-mode{{top:14px;left:100px}}
#b-sim{{top:14px;right:50px;background:rgba(0,0,0,.45)}}
#b-skip{{top:14px;right:12px;cursor:pointer}}
#ad-cd{{font-size:13px;margin-bottom:10px;opacity:.7;margin-top:12px}}
#ad-emoji{{font-size:3.5rem;margin-bottom:6px}}
#ad-brand{{font-size:11px;font-weight:700;text-transform:uppercase;
            letter-spacing:.14em;opacity:.75;margin-bottom:3px}}
#ad-hl{{font-size:23px;font-weight:800;line-height:1.2;margin-bottom:9px}}
#ad-body{{font-size:14px;opacity:.88;line-height:1.55;margin-bottom:22px}}
#ad-cta{{display:inline-block;background:rgba(255,255,255,.18);
          border:2px solid rgba(255,255,255,.5);color:#fff;
          padding:11px 30px;border-radius:30px;font-weight:700;cursor:pointer;font-size:14px}}
#ad-cta:hover{{background:rgba(255,255,255,.32)}}
#ad-reason{{font-size:11px;opacity:.6;margin-top:14px}}

/* Progress */
#cb{{padding:10px 16px 5px;background:#fff;border-top:1px solid #f3f4f6}}
#pw{{position:relative;height:8px;background:#e5e7eb;border-radius:4px;cursor:pointer;margin-bottom:10px}}
#pf{{height:100%;background:#f59e0b;border-radius:4px;width:0%;pointer-events:none}}
.pm{{position:absolute;top:-5px;width:18px;height:18px;border-radius:50%;
      border:2px solid #fff;transform:translateX(-50%);cursor:pointer;z-index:5;
      box-shadow:0 2px 5px rgba(0,0,0,.25);transition:transform .12s}}
.pm:hover{{transform:translateX(-50%) scale(1.5)}}
.pm-l{{position:absolute;top:16px;transform:translateX(-50%);font-size:9px;
         white-space:nowrap;font-weight:600;pointer-events:none}}

#ctrl{{display:flex;align-items:center;gap:10px}}
#pbtn{{width:36px;height:36px;border-radius:50%;background:#f59e0b;border:none;
        cursor:pointer;font-size:14px;color:#fff;display:flex;align-items:center;
        justify-content:center;box-shadow:0 2px 8px rgba(245,158,11,.35);flex-shrink:0}}
#td{{font-size:12px;color:#374151;font-variant-numeric:tabular-nums}}
#vol{{display:flex;align-items:center;gap:6px;margin-left:auto}}
#vol input{{width:72px;accent-color:#f59e0b}}
#vico{{cursor:pointer;font-size:14px}}
#sb{{padding:4px 16px 7px;font-size:11px;color:#9ca3af;background:#fff}}

#sp{{border-top:1px solid #f3f4f6;padding:10px 16px 12px;background:#fafafa}}
#sp-hd{{font-size:12px;font-weight:600;color:#374151;margin-bottom:7px}}
#sp-list{{display:flex;flex-wrap:wrap;gap:5px}}
.chip{{padding:4px 10px;border-radius:14px;font-size:11px;font-weight:500;cursor:pointer;
        border:1px solid #e5e7eb;background:#fff;color:#374151;white-space:nowrap;
        max-width:190px;overflow:hidden;text-overflow:ellipsis;transition:all .12s}}
.chip:hover,.chip.act{{background:#fff7ed;border-color:#f59e0b;color:#92400e}}
.chip.key{{border-left:3px solid #f59e0b}}
.chip.adchip{{background:#fef3c7;border-color:#fcd34d;color:#92400e;font-weight:600}}
</style></head><body>
<div id="wrap">
  <div id="vw">
    <video id="vid" preload="auto" playsinline>
      <source src="data:{video_type};base64,{video_b64}" type="{video_type}">
    </video>
    <div id="adov">
      <div id="adcard">
        <div class="badge" id="b-type">mid-roll</div>
        <div class="badge" id="b-mode">🤖 auto</div>
        <div class="badge" id="b-sim">🎯 —%</div>
        <div class="badge" id="b-skip" onclick="skipAd()">✕ Skip</div>
        <div id="ad-cd"></div>
        <div id="ad-emoji">🎬</div>
        <div id="ad-brand">Brand</div>
        <div id="ad-hl">Headline</div>
        <div id="ad-body">Body</div>
        <div id="ad-cta" onclick="skipAd()">CTA</div>
        <div id="ad-reason"></div>
      </div>
    </div>
  </div>
  <div id="cb">
    <div id="pw" onclick="seekBar(event)"><div id="pf"></div></div>
    <div id="ctrl">
      <button id="pbtn" onclick="togglePlay()">▶</button>
      <span id="td">0:00 / 0:00</span>
      <div id="vol">
        <span id="vico" onclick="toggleMute()">🔊</span>
        <input type="range" min="0" max="1" step=".05" value="1" oninput="VID.volume=this.value">
      </div>
    </div>
  </div>
  <div id="sb">Ready · {len(markers)} ad markers · press ▶</div>
  <div id="sp">
    <div id="sp-hd">📍 Ad markers (click to preview) · ⭐ key scenes · scenes timeline</div>
    <div id="sp-list"></div>
  </div>
</div>
<script>
var VID=document.getElementById('vid');
var M={markers_js};
var POST={post_js};
var SC={scenes_js};
var shown={{}};var cdt=null;var adOn=false;

VID.addEventListener('loadedmetadata',function(){{
  var dur=VID.duration,pw=document.getElementById('pw');
  M.forEach(function(m){{
    if(m.sec<=0)return;
    var pct=(m.sec/dur)*100;
    var dot=document.createElement('div');dot.className='pm';dot.style.left=pct+'%';
    dot.style.background=m.mode==='auto'?'#f59e0b':'#ef4444';
    dot.title=m.ad_brand+' @ '+m.fmt+' · '+Math.round(m.sim*100)+'% sim · '+m.mode;
    dot.onclick=function(e){{e.stopPropagation();VID.currentTime=Math.max(0,m.sec-0.5);VID.play();}};
    var lbl=document.createElement('div');lbl.className='pm-l';lbl.style.left=pct+'%';
    lbl.style.color=m.mode==='auto'?'#d97706':'#dc2626';
    lbl.textContent='📢'+m.fmt.slice(3);
    pw.appendChild(dot);pw.appendChild(lbl);
  }});
  var sl=document.getElementById('sp-list');
  M.forEach(function(m){{
    var c=document.createElement('span');c.className='chip adchip';
    c.title=m.ad_brand+' — '+m.ad_title+' | '+Math.round(m.sim*100)+'% | '+m.mode;
    c.textContent=m.ad_emoji+' '+(m.sec===0?'Pre':m.fmt.slice(3))+' '+m.ad_brand+' '+Math.round(m.sim*100)+'%';
    c.onclick=function(){{VID.currentTime=Math.max(0,m.sec-0.5);VID.play();}};
    sl.appendChild(c);
  }});
  SC.forEach(function(s){{
    var c=document.createElement('span');
    c.className='chip'+(s.key?' key':'');c.id='sc'+s.sec;
    c.title=s.label+' | eng '+s.eng;
    c.textContent=(s.key?'⭐':'')+s.fmt+' '+s.label;
    c.onclick=function(){{VID.currentTime=s.sec;VID.play();}};
    sl.appendChild(c);
  }});
  document.getElementById('sb').textContent=
    'Ready · '+M.length+' ads · '+SC.length+' scenes · press ▶ to start';
}});

VID.addEventListener('timeupdate',function(){{
  var t=VID.currentTime,dur=VID.duration||1;
  document.getElementById('pf').style.width=(t/dur*100)+'%';
  document.getElementById('td').textContent=fmt(t)+' / '+fmt(dur);
  document.getElementById('pbtn').textContent=VID.paused?'▶':'⏸';
  SC.forEach(function(s){{
    var el=document.getElementById('sc'+s.sec);
    if(el)el.classList.toggle('act',t>=s.sec&&t<s.sec+20);
  }});
  if(adOn)return;
  M.forEach(function(m){{
    var k=m.ad_brand+'_'+m.sec;
    if(!shown[k]&&t>=m.sec&&m.sec>=0){{shown[k]=true;showAd(m);}}
  }});
}});
VID.addEventListener('ended',function(){{if(!shown.post){{shown.post=true;showAd(POST);}}}});
VID.addEventListener('play',function onFirst(){{
  var pre=M.find(function(m){{return m.sec===0;}});
  if(pre&&!shown[pre.ad_brand+'_0']){{shown[pre.ad_brand+'_0']=true;VID.pause();showAd(pre);}}
  VID.removeEventListener('play',onFirst);
}},{{once:true}});

function showAd(m){{
  adOn=true;VID.pause();
  document.getElementById('adov').style.display='flex';
  document.getElementById('adcard').style.background=m.ad_bg;
  document.getElementById('b-type').textContent=m.type;
  document.getElementById('b-mode').textContent=m.mode==='auto'?'🤖 AI matched':'✋ Manual';
  document.getElementById('b-sim').textContent='🎯 '+Math.round((m.sim||0)*100)+'% match';
  document.getElementById('b-skip').style.visibility='hidden';
  document.getElementById('ad-emoji').textContent=m.ad_emoji;
  document.getElementById('ad-brand').textContent=m.ad_brand;
  document.getElementById('ad-hl').textContent=m.ad_headline;
  document.getElementById('ad-body').textContent=m.ad_body;
  document.getElementById('ad-cta').textContent=m.ad_cta;
  document.getElementById('ad-reason').textContent=m.reason||'';
  var s=Math.min(5,m.duration||15);
  document.getElementById('ad-cd').textContent='Skippable in '+s+'s';
  document.getElementById('sb').textContent=
    '📢 '+m.type+' · '+m.ad_brand+' — '+m.ad_title+' · '+Math.round((m.sim||0)*100)+'% similarity match';
  cdt=setInterval(function(){{s--;
    if(s<=0){{clearInterval(cdt);document.getElementById('ad-cd').textContent='';
      document.getElementById('b-skip').style.visibility='visible';}}
    else document.getElementById('ad-cd').textContent='Skippable in '+s+'s';
  }},1000);
}}
function skipAd(){{
  clearInterval(cdt);adOn=false;
  document.getElementById('adov').style.display='none';
  if(!VID.ended)VID.play();
  document.getElementById('sb').textContent='▶ Playing';
}}
function togglePlay(){{VID.paused?VID.play():VID.pause();}}
function seekBar(e){{
  var r=document.getElementById('pw').getBoundingClientRect();
  VID.currentTime=((e.clientX-r.left)/r.width)*(VID.duration||0);
}}
function toggleMute(){{VID.muted=!VID.muted;document.getElementById('vico').textContent=VID.muted?'🔇':'🔊';}}
function fmt(s){{var m=Math.floor(s/60),ss=Math.floor(s%60);return m+':'+(ss<10?'0':'')+ss;}}
</script></body></html>"""

        st.components.v1.html(html, height=720, scrolling=False)

        st.divider()
        st.caption("**AD SCHEDULE — with similarity breakdown**")
        rows = []
        for m in sorted(markers, key=lambda x: x["sec"]):
            sc = min(vm.scenes, key=lambda s: abs(s.start_sec-m["sec"]))
            sim_d = _similarity(m["ad"], sc)
            rows.append({
                "Time":       "Pre-roll" if m["sec"]==0 else m["fmt"],
                "Type":       "pre-roll" if m["sec"]==0 else "mid-roll",
                "By":         "🤖 AI" if m.get("mode")=="auto" else "✋ Manual",
                "Brand":      m["ad"]["brand"],
                "Ad":         m["ad"]["title"],
                "🎯 Overall": f"{m['sim']:.0%}",
                "IAB match":  f"{sim_d['iab']:.0%}",
                "Text match": f"{sim_d['text']:.0%}",
                "Dur":        f"{m.get('duration',15)}s",
                "Matched on": ", ".join(sim_d["matched_iab"][:4]) or "—",
            })
        rows.append({
            "Time":"Post-roll","Type":"post-roll","By":"🤖 AI",
            "Brand":post_ad["brand"],"Ad":post_ad["title"],
            "🎯 Overall":f"{post_sim:.0%}",
            "IAB match":f"{post_sim_d['iab']:.0%}",
            "Text match":f"{post_sim_d['text']:.0%}",
            "Dur":"15s","Matched on":", ".join(post_sim_d["matched_iab"][:4]) or "—",
        })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)



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
