"""
Semantix — Video Intelligence Platform  v5
Improvements applied:
  1. Fixed double AI metadata call bug (_generate_ai_meta now single-path)
  2. Semantic ad matching using shared TF-IDF vectorizer (cosine similarity)
  3. Ad insertion rules engine (spacing, min-duration, sentiment guard)
  4. Publisher-facing HTML Ad Report export
  5. Monetisation Score computed at ingest, shown on library cards
  6. CPM disclaimers added everywhere
  7. Multi-video semantic search as hero feature on Insights
  8. Ad Library active/paused toggle
  9. Timeline truncation warning (>200 scenes)
 10. Persona-aware default page routing
"""

import re
import time
import base64 as _b64
import json as _json
import numpy as np
from typing import List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pydantic import BaseModel, Field
from groq import Groq
import instructor

from core.video_processor import VideoProcessor, VideoMetadata, fetch_youtube_transcript, fetch_youtube_metadata
from core.scene_detector import Scene
from core.ad_engine import AdMatchingEngine, create_default_inventory
from core.search_engine import HybridSearchEngine
from core.embeddings import _IAB_NAMES

# ── Pydantic schema for AI metadata ────────────────────────────────────────
class VideoMetadataSchema(BaseModel):
    summary: str
    short_description: str = Field(..., max_length=15)
    content_rating: str
    rating_reason: str
    primary_genre: str
    target_audience: str
    key_themes: List[str]
    mood: str
    keywords: List[str]
    content_warnings: List[str]
    advertiser_suitability: str
    advertiser_reason: str
    seo_title: str = Field(..., max_length=60)
    seo_description: str = Field(..., max_length=155)

st.set_page_config(
    page_title="Semantix · Video Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────
def load_css():
    try:
        with open("style.css", "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass

load_css()

# ── Session state ──────────────────────────────────────────────────────────
_DEFAULTS = {
    "page": "library",
    "videos": {},
    "selected_video": None,
    "yt_api_key": "",
    "search_engine": None,
    "ad_engine": None,
    "custom_ads": [],
    "ad_markers": {},
    "video_b64": {},
    "video_mime": {},
    "ai_meta": {},
    "_persona_routed": False,   # NEW: tracks first-load persona routing
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.search_engine is None:
    st.session_state.search_engine = HybridSearchEngine()
if st.session_state.ad_engine is None:
    st.session_state.ad_engine = AdMatchingEngine()

_qp = st.query_params
if "page" in _qp and _qp["page"] in ("library","analyse","monetise","insights"):
    if st.session_state.page == "library":
        st.session_state.page = _qp["page"]
if "vid" in _qp and _qp["vid"] in st.session_state.videos:
    if st.session_state.selected_video is None:
        st.session_state.selected_video = _qp["vid"]

def _sync_qp():
    st.query_params["page"] = st.session_state.page
    if st.session_state.selected_video:
        st.query_params["vid"] = st.session_state.selected_video

# ── Persona-aware routing (NEW) ────────────────────────────────────────────
PERSONA_DEFAULT_PAGE = {
    "Publisher":    "library",
    "Advertiser":   "insights",
    "Editor":       "analyse",
    "Analyst":      "insights",
    "Monetisation": "monetise",
}

# ── Plotly theme ───────────────────────────────────────────────────────────
_BG = "#ffffff"; _GRID = "#e5e7eb"; _TEXT = "#6b7280"; _AMBER = "#f59e0b"
PT = dict(plot_bgcolor=_BG, paper_bgcolor=_BG,
          font=dict(family="Inter,sans-serif", color=_TEXT, size=11),
          margin=dict(l=20,r=20,t=44,b=20),
          colorway=[_AMBER,"#34d399","#60a5fa","#a78bfa","#fb923c","#f472b6"])
_XA = dict(gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=_TEXT,size=10))
_YA = dict(gridcolor=_GRID, linecolor=_GRID, tickfont=dict(color=_TEXT,size=10))

# ── Navigation ─────────────────────────────────────────────────────────────
NAV = [
    ("library",  "🗂️",  "Library"),
    ("analyse",  "🔬",  "Analyse"),
    ("monetise", "📢",  "Monetise"),
    ("insights", "📊",  "Insights"),
]

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:20px 16px 14px;border-bottom:1px solid #f1f5f9;margin-bottom:6px">
      <div style="display:flex;align-items:center;gap:10px">
        <div style="width:34px;height:34px;background:#f59e0b;border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:16px;flex-shrink:0;box-shadow:0 2px 6px rgba(245,158,11,.3)">⚡</div>
        <div>
          <div style="font-size:1.05rem;font-weight:800;color:#0f172a;letter-spacing:-.02em;line-height:1.1">Semantix</div>
          <div style="font-size:.62rem;color:#94a3b8;letter-spacing:.1em;text-transform:uppercase;margin-top:2px">Video Intelligence · v5</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Persona selector (NEW — actually changes routing)
    persona_opts = list(PERSONA_DEFAULT_PAGE.keys())
    if "persona" not in st.session_state:
        st.session_state.persona = "Publisher"
    sel_persona = st.selectbox("Role / Persona", persona_opts,
                               index=persona_opts.index(st.session_state.persona),
                               key="_persona_sel")
    if sel_persona != st.session_state.persona:
        st.session_state.persona = sel_persona
        st.session_state.page = PERSONA_DEFAULT_PAGE[sel_persona]
        st.session_state._persona_routed = True
        st.rerun()

    st.markdown('<div style="padding:6px 8px 8px">', unsafe_allow_html=True)
    for pid, icon, label in NAV:
        active = st.session_state.page == pid
        if active:
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;'
                f'background:#fef3c7;border:1px solid #fcd34d;'
                f'border-radius:8px;padding:9px 14px;margin:2px 0">'
                f'<span style="font-size:1rem">{icon}</span>'
                f'<span style="font-size:.875rem;font-weight:700;color:#92400e">{label}</span>'
                f'</div>', unsafe_allow_html=True)
            st.button(f"{icon} {label}", key=f"nav_{pid}", disabled=True, use_container_width=True)
        else:
            if st.button(f"{icon} {label}", key=f"nav_{pid}", use_container_width=True):
                st.session_state.page = pid
                _sync_qp(); st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #f1f5f9;margin:4px 16px 10px">', unsafe_allow_html=True)

    n_vids   = len(st.session_state.videos)
    n_scenes = sum(v.scene_count for v in st.session_state.videos.values())

    if n_vids:
        st.markdown(
            f'<div style="display:flex;gap:6px;margin:0 8px 10px">'
            f'<div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="font-size:1.3rem;font-weight:800;color:#0f172a;line-height:1">{n_vids}</div>'
            f'<div style="font-size:.6rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-top:3px">Videos</div>'
            f'</div>'
            f'<div style="flex:1;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px;text-align:center">'
            f'<div style="font-size:1.3rem;font-weight:800;color:#0f172a;line-height:1">{n_scenes}</div>'
            f'<div style="font-size:.6rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;margin-top:3px">Scenes</div>'
            f'</div></div>', unsafe_allow_html=True)

        vms     = list(st.session_state.videos.values())
        vid_ids = [vm.video_id for vm in vms]
        labels  = [vm.title[:26] + ("…" if len(vm.title) > 26 else "") for vm in vms]
        cur_sel = st.session_state.get("selected_video")
        correct_idx = vid_ids.index(cur_sel) if cur_sel in vid_ids else 0

        idx = st.selectbox("Active video", range(len(vms)),
                           format_func=lambda i: labels[i],
                           index=correct_idx, key="vm_sel",
                           label_visibility="collapsed")

        chosen_vid = vid_ids[idx]
        if chosen_vid != st.session_state.get("selected_video"):
            st.session_state.selected_video = chosen_vid
            _sync_qp()
        active_vm_obj = vms[idx]

        ai_meta = st.session_state.ai_meta.get(active_vm_obj.video_id, {})
        meta_bits = [active_vm_obj.fmt_duration(), f"{active_vm_obj.scene_count} scenes"]
        if ai_meta.get("content_rating"): meta_bits.append(ai_meta["content_rating"])
        if ai_meta.get("primary_genre"):  meta_bits.append(ai_meta["primary_genre"])

        # NEW: show monetisation score in sidebar
        mscore = ai_meta.get("monetisation_score", None)
        if mscore is not None:
            score_color = "#16a34a" if mscore >= 70 else "#d97706" if mscore >= 40 else "#dc2626"
            st.markdown(
                f'<div style="margin:0 8px 8px;background:#f8fafc;border:1px solid #e2e8f0;'
                f'border-radius:8px;padding:8px 12px;display:flex;justify-content:space-between;align-items:center">'
                f'<span style="font-size:.7rem;color:#6b7280;font-weight:600">💰 Monetisation Score</span>'
                f'<span style="font-size:1.1rem;font-weight:800;color:{score_color}">{mscore}/100</span>'
                f'</div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="font-size:.71rem;color:#94a3b8;padding:4px 10px 8px;line-height:1.6">'
            f'{"  ·  ".join(meta_bits)}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-size:.8rem;color:#94a3b8;padding:8px 16px 12px;line-height:1.5">'
            '📂 No videos yet.<br>Add one in Library.</div>', unsafe_allow_html=True)

    st.markdown('<hr style="border:none;border-top:1px solid #f1f5f9;margin:4px 16px 10px">', unsafe_allow_html=True)

    st.markdown('<div style="font-size:.7rem;color:#94a3b8;padding:0 16px 4px;font-weight:600;letter-spacing:.04em">YOUTUBE API KEY</div>', unsafe_allow_html=True)
    yt = st.text_input("YouTube API Key", type="password", value=st.session_state.yt_api_key,
                       placeholder="Optional — enables richer metadata", key="yt_key_in",
                       label_visibility="collapsed")
    if yt != st.session_state.yt_api_key:
        st.session_state.yt_api_key = yt

    if n_vids:
        st.markdown(
            '<div style="margin:12px 8px 8px;background:#fffbeb;border:1px solid #fde68a;'
            'border-radius:8px;padding:8px 12px">'
            '<div style="font-size:.65rem;font-weight:700;color:#92400e;margin-bottom:2px">⚠️ Session only</div>'
            '<div style="font-size:.62rem;color:#78350f;line-height:1.5">Data is lost on refresh. Export reports from Insights → Export.</div>'
            '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _compute_monetisation_score(vm: VideoMetadata) -> int:
    """
    NEW: Compute a 0–100 monetisation score at ingest time.
    Factors: avg ad fit, brand safety, scene count density, content category, key moment ratio.
    """
    if not vm.scenes:
        return 0
    avg_fit    = sum(s.ad_suitability for s in vm.scenes) / len(vm.scenes)
    avg_safe   = sum(s.brand_safety.get("safety_score", 1) for s in vm.scenes) / len(vm.scenes)
    avg_eng    = sum(s.engagement_score for s in vm.scenes) / len(vm.scenes)
    key_ratio  = len(vm.key_scenes) / max(vm.scene_count, 1)
    # Scene density: more scenes per minute = more ad opportunities
    scene_density = min(vm.scene_count / max(vm.duration_sec / 60, 1), 10) / 10
    score = (avg_fit * 35) + (avg_safe * 30) + (avg_eng * 20) + (key_ratio * 10) + (scene_density * 5)
    return min(100, int(score * 100))


# ── FIX 1: Single-path AI metadata generation ─────────────────────────────
def _generate_ai_meta(vm: VideoMetadata, transcript_sample: str):
    """
    FIXED: Single execution path — try Groq/instructor first, fall back to
    Anthropic API, fall back to heuristics. Previous version ran BOTH and
    the second call silently overwrote the first.
    """
    scene_summary = "\n".join(
        f"- [{s.start_fmt}] {s.sentiment.get('label','?').upper()} | "
        f"eng:{s.engagement_score:.2f} | {s.text[:80]}"
        for s in vm.scenes[:20])
    iab_tags = ", ".join({c["name"] for s in vm.scenes for c in s.iab_categories[:2]})
    iab_list = list({c["name"] for s in vm.scenes for c in s.iab_categories[:2]})

    prompt = f"""You are a video metadata expert for a content monetisation platform.
Analyse this video and return ONLY a JSON object (no markdown, no explanation):

Video title: {vm.title}
Duration: {vm.fmt_duration()}
Scene count: {vm.scene_count}
Narrative arc: {vm.narrative_structure}
IAB categories detected: {iab_tags}

Scene breakdown (first 20):
{scene_summary}

Transcript sample:
{transcript_sample[:2000]}

Return this exact JSON structure:
{{
  "summary": "2-3 sentence engaging description",
  "short_description": "One punchy sentence (max 15 words)",
  "content_rating": "G | PG | PG-13 | R",
  "rating_reason": "Brief reason for the rating",
  "primary_genre": "e.g. Action, Documentary, Comedy, Tutorial",
  "target_audience": "e.g. Adults 25-44, Teens, Families",
  "key_themes": ["theme1", "theme2", "theme3", "theme4"],
  "mood": "e.g. Exciting, Informative, Emotional, Humorous",
  "keywords": ["kw1","kw2","kw3","kw4","kw5","kw6","kw7","kw8"],
  "content_warnings": [],
  "advertiser_suitability": "High | Medium | Low",
  "advertiser_reason": "Why advertisers would or wouldn't want this content",
  "seo_title": "SEO-optimised title (max 60 chars)",
  "seo_description": "SEO meta description (max 155 chars)"
}}"""

    meta = None

    # Path 1: Groq via instructor
    try:
        client = instructor.from_openai(
            Groq(api_key=st.secrets["GROQ_API_KEY"]),
            mode=instructor.Mode.JSON_SCHEMA
        )
        result = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_model=VideoMetadataSchema,
            messages=[{"role": "user", "content": prompt}]
        )
        meta = result.model_dump()
    except Exception as e:
        st.warning(f"⚠️ Groq metadata unavailable ({type(e).__name__}), trying fallback…")

    # Path 2: Anthropic API (only if Groq failed)
    if meta is None:
        try:
            import urllib.request
            body = _json.dumps({
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1000,
                "messages": [{"role": "user", "content": prompt}]
            }).encode()
            req = urllib.request.Request(
                "https://api.anthropic.com/v1/messages",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST")
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read())
            raw = data["content"][0]["text"].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            meta = _json.loads(raw)
        except Exception as e:
            st.warning(f"⚠️ Anthropic fallback failed ({type(e).__name__}), using heuristics.")

    # Path 3: Heuristic fallback (only if both AI paths failed)
    if meta is None:
        pos = sum(1 for s in vm.scenes if s.sentiment.get("label") == "positive")
        neg = sum(1 for s in vm.scenes if s.sentiment.get("label") == "negative")
        mood = "Exciting" if pos > neg * 2 else "Tense" if neg > pos else "Balanced"
        avg_safety = sum(s.brand_safety.get("safety_score", 1) for s in vm.scenes) / max(vm.scene_count, 1)
        meta = {
            "summary": f"{vm.title} — a {vm.fmt_duration()} video with {vm.scene_count} scenes across a {vm.narrative_structure} arc.",
            "short_description": vm.title[:60],
            "content_rating": "PG", "rating_reason": "General content",
            "primary_genre": iab_list[0] if iab_list else "General",
            "target_audience": "General audience",
            "key_themes": iab_list[:4],
            "mood": mood, "keywords": iab_list[:8],
            "content_warnings": [],
            "advertiser_suitability": "High" if avg_safety > 0.7 else "Medium",
            "advertiser_reason": "Based on content analysis",
            "seo_title": vm.title[:60],
            "seo_description": f"{vm.title} — {vm.scene_count} scenes, {vm.fmt_duration()}"
        }

    # Attach monetisation score (NEW)
    meta["monetisation_score"] = _compute_monetisation_score(vm)
    st.session_state.ai_meta[vm.video_id] = meta


def _show_ai_meta(vm: VideoMetadata):
    meta = st.session_state.ai_meta.get(vm.video_id)
    if not meta:
        return

    rating_color = {"G":"#16a34a","PG":"#2563eb","PG-13":"#d97706","R":"#dc2626"}.get(meta.get("content_rating","PG"),"#6b7280")
    mscore = meta.get("monetisation_score", None)
    mscore_color = "#16a34a" if mscore and mscore >= 70 else "#d97706" if mscore and mscore >= 40 else "#dc2626"

    st.markdown(f'''
        <div class="ai-intelligence-card" style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;padding:16px 20px;margin-bottom:12px">
            <div style="display:flex; align-items:flex-start; gap:16px; flex-wrap:wrap">
                <div style="flex:1; min-width:220px">
                    <div style="font-size:11px; font-weight:700; text-transform:uppercase; color:#92400e; margin-bottom:4px">Summary</div>
                    <div style="font-size:14px; color:#111827; line-height:1.6">{meta.get("summary","")}</div>
                </div>
                <div style="display:flex; flex-direction:column; gap:8px; min-width:180px">
                    <div>
                        <span style="background:{rating_color}; color:#fff; font-size:11px; font-weight:700; padding:3px 10px; border-radius:20px">
                            {meta.get("content_rating","PG")}
                        </span>
                    </div>
                    <div style="font-size:12px"><b>Genre:</b> {meta.get("primary_genre","—")}</div>
                    <div style="font-size:12px"><b>Mood:</b> {meta.get("mood","—")}</div>
                    <div style="font-size:12px"><b>Audience:</b> {meta.get("target_audience","—")}</div>
                    {"" if mscore is None else f'<div style="font-size:13px;font-weight:700;color:{mscore_color}">💰 Monetisation Score: {mscore}/100</div>'}
                </div>
            </div>
        </div>
    ''', unsafe_allow_html=True)

    with st.expander("🔍 SEO Metadata"):
        st.text_input("SEO Title", value=meta.get("seo_title",""), key=f"seo_t_{vm.video_id}")
        st.text_area("SEO Description", value=meta.get("seo_description",""), key=f"seo_d_{vm.video_id}")

    if st.button("🔄 Regenerate Metadata", key=f"regen_{vm.video_id}"):
        srt_sample = " ".join(s.text for s in vm.scenes[:15])
        with st.spinner("Regenerating…"):
            _generate_ai_meta(vm, srt_sample)
        st.rerun()


def _register(vm: VideoMetadata):
    st.session_state.videos[vm.video_id] = vm
    st.session_state.search_engine.add_scenes(vm.scenes)
    if st.session_state.search_engine.vectorizer is not None:
        st.session_state.ad_engine.sync_vectorizer(st.session_state.search_engine.vectorizer)
    st.session_state.selected_video = vm.video_id


def _active_vm() -> Optional[VideoMetadata]:
    sel = st.session_state.get("selected_video")
    if sel:
        vm = st.session_state.videos.get(sel)
        if vm: return vm
    vm = next(iter(st.session_state.videos.values()), None)
    if vm: st.session_state.selected_video = vm.video_id
    return vm


def _yt_id(url: str) -> Optional[str]:
    for p in [r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})", r"^([A-Za-z0-9_-]{11})$"]:
        m = re.search(p, url)
        if m: return m.group(1)
    return None

def _parse_ts(t: str) -> Optional[int]:
    t = t.strip()
    if not t: return None
    if ":" in t:
        parts = t.split(":")
        try:
            if len(parts)==2: return int(parts[0])*60+int(parts[1])
            if len(parts)==3: return int(parts[0])*3600+int(parts[1])*60+int(parts[2])
        except ValueError: return None
    try: return int(float(t))
    except ValueError: return None

def _fmt_sec(s: int) -> str:
    s = int(s or 0)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def _sent_icon(label): return {"positive":"🟢","negative":"🔴","neutral":"🔵"}.get(label,"🔵")
def _iab_str(cats,n=2): return "  ·  ".join(c["name"] for c in cats[:n]) if cats else "—"

def _tag_chip(text, bg="#fef3c7", border="#fcd34d", color="#92400e", size="12px"):
    return (f'<span style="display:inline-block;background:{bg};border:1px solid {border};'
            f'color:{color};padding:3px 10px;border-radius:14px;font-size:{size};'
            f'font-weight:500;margin:2px">{text}</span>')

def _tok(text: str) -> set:
    text = text.lower()
    text = re.sub(r"[&,\-/|]+", " ", text)
    return {w for w in text.split() if len(w) > 2}


# ── FIX 2: Semantic ad similarity using shared vectorizer ─────────────────
def _ad_similarity(ad: dict, scene: Scene) -> dict:
    """
    IMPROVED: Uses vector similarity when the shared TF-IDF vectorizer is
    available (after first video is indexed), falls back to token intersection.
    Weights: vector 50%, IAB token 30%, text token 20% — then boosted by
    engagement and brand safety.
    """
    tags = ad.get("tags", "")
    if isinstance(tags, list): tags = " ".join(tags)
    ad_text = f"{tags} {ad.get('headline','')} {ad.get('body','')}"

    # Semantic vector similarity (NEW)
    vec_score = 0.0
    vectorizer = st.session_state.search_engine.vectorizer if st.session_state.search_engine else None
    if vectorizer is not None and scene.embedding is not None:
        try:
            from core.embeddings import cosine_matrix
            ad_vec = vectorizer.embed(ad_text)
            sim = cosine_matrix(ad_vec.reshape(1, -1), scene.embedding.reshape(1, -1))[0][0]
            vec_score = float(max(0, sim))
        except Exception:
            vec_score = 0.0

    # Token overlap (kept as supplement)
    ad_t   = _tok(ad_text)
    iab_t  = _tok(" ".join(c["name"] for c in scene.iab_categories[:4]))
    txt_t  = _tok(" ".join(scene.text.split()[:60]))
    scene_all = iab_t | txt_t

    iab_score  = min(len(ad_t & iab_t) / max(len(ad_t),1), 1.0)
    text_score = min(len(ad_t & txt_t) / max(len(ad_t),1), 1.0)

    # Blend: vector 50% + IAB tokens 30% + text tokens 20%
    content_score = vec_score * 0.50 + iab_score * 0.30 + text_score * 0.20
    # Boost by engagement + brand safety
    eng_b  = scene.engagement_score * 0.06
    safe_b = scene.brand_safety.get("safety_score",1.0) * 0.04
    total  = min(content_score + eng_b + safe_b, 1.0)

    return {
        "total": round(total,3), "vector": round(vec_score,3),
        "iab": round(iab_score,3), "text": round(text_score,3),
        "matched_iab": sorted(ad_t & iab_t),
        "matched_text": sorted(ad_t & txt_t)[:5],
    }


# ── FIX 3: Ad insertion rules engine (NEW) ────────────────────────────────
AD_RULES = {
    "min_gap_sec": 60,          # No two ads within 60s of each other
    "no_ad_first_sec": 45,      # No ad in the first 45s
    "min_scene_duration": 20,   # Scene must be at least 20s for a mid-roll
    "block_negative_for_safe_ads": True,  # Don't place in negative scenes if advertiser_suitability="High"
}

def _apply_ad_rules(markers: list, scenes: list, vm_duration: int) -> tuple[list, list]:
    """
    Returns (valid_markers, rejected_reasons).
    Validates and filters a marker list against AD_RULES.
    """
    valid, rejected = [], []
    sorted_m = sorted(markers, key=lambda x: x["sec"])
    last_ad_sec = -999

    for m in sorted_m:
        reasons = []
        sec = m["sec"]
        ad  = m.get("ad", {})

        if sec < AD_RULES["no_ad_first_sec"] and sec != 0:
            reasons.append(f"Too early — first {AD_RULES['no_ad_first_sec']}s reserved (pre-roll excepted)")
        if sec != 0 and (sec - last_ad_sec) < AD_RULES["min_gap_sec"]:
            reasons.append(f"Too close to previous ad ({sec - last_ad_sec}s gap, min {AD_RULES['min_gap_sec']}s)")

        # Find nearest scene
        if scenes:
            nearest = min(scenes, key=lambda s: abs(s.start_sec - sec))
            if nearest.duration_sec < AD_RULES["min_scene_duration"] and sec != 0:
                reasons.append(f"Scene too short ({nearest.duration_sec:.0f}s, min {AD_RULES['min_scene_duration']}s)")
            if (AD_RULES["block_negative_for_safe_ads"]
                    and nearest.sentiment.get("label") == "negative"
                    and ad.get("advertiser_suitability","") == "High"):
                reasons.append("Negative sentiment scene — blocked for brand-safe ad")

        if reasons:
            rejected.append({"marker": m, "reasons": reasons})
        else:
            valid.append(m)
            last_ad_sec = sec

    return valid, rejected


# ── CPM helpers (with disclaimer) ─────────────────────────────────────────
_IAB_CPM = {
    "Automotive":(8,18),"Finance":(12,28),"Technology":(7,16),
    "Sports":(6,14),"Entertainment":(4,10),"Travel":(5,13),
    "Health & Fitness":(5,12),"Food & Drink":(4,9),"Arts":(3,8),
    "News":(5,14),"Education":(4,10),"Business":(9,22),
    "Shopping":(5,12),"Science":(4,9),"Family":(4,9),
    "Music":(3,7),"default":(3,8),
}
CPM_DISCLAIMER = "ℹ️ *CPM ranges are indicative benchmarks based on IAB category averages. Actual rates vary by geography, ad format, and buyer.*"

def _est_cpm(scene: "Scene") -> tuple:
    iab = scene.iab_categories[0]["name"] if scene.iab_categories else ""
    lo, hi = _IAB_CPM.get(iab, _IAB_CPM["default"])
    boost = 0.8 + (0.4 * getattr(scene, "engagement_score", 0.5))
    return (round(lo * boost, 2), round(hi * boost, 2))


def _best_ad_for_scene(scene: Scene):
    ads = [a for a in BUILTIN_ADS + st.session_state.custom_ads if a.get("active", True)]
    if not ads: ads = BUILTIN_ADS
    scored = [(ad, _ad_similarity(ad, scene)) for ad in ads]
    scored.sort(key=lambda x: x[1]["total"], reverse=True)
    return scored[0] if scored else (BUILTIN_ADS[0], {"total":0,"iab":0,"text":0,"vector":0,"matched_iab":[],"matched_text":[]})

def _top_ads_for_scene(scene: Scene, n=4):
    ads = [a for a in BUILTIN_ADS + st.session_state.custom_ads if a.get("active", True)]
    if not ads: ads = BUILTIN_ADS
    scored = [(ad, _ad_similarity(ad, scene)) for ad in ads]
    scored.sort(key=lambda x: x[1]["total"], reverse=True)
    return scored[:n]


# ── Built-in ad inventory (with active flag) ──────────────────────────────
BUILTIN_ADS = [
    {"id":"ad1","brand":"Nike","title":"Just Do It","emoji":"👟","cta":"Shop Now",
     "bg":"linear-gradient(135deg,#f59e0b,#d97706)","headline":"Push Your Limits",
     "body":"New season collection — built for champions.","advertiser_suitability":"High",
     "tags":"sports fitness action competition energy lifestyle motivation exercise athletic running",
     "active": True},
    {"id":"ad2","brand":"Spotify","title":"Music for Every Mood","emoji":"🎵","cta":"Listen Free",
     "bg":"linear-gradient(135deg,#1db954,#158a3e)","headline":"Soundtrack Your Life",
     "body":"3 months Premium free — no ads, offline play.","advertiser_suitability":"High",
     "tags":"music entertainment emotion arts drama movies relaxation streaming audio",
     "active": True},
    {"id":"ad3","brand":"Amazon","title":"Deals of the Day","emoji":"📦","cta":"Shop Deals",
     "bg":"linear-gradient(135deg,#ff9900,#e47911)","headline":"Today Only — Up to 60% Off",
     "body":"Lightning deals on electronics, home & more.","advertiser_suitability":"Medium",
     "tags":"shopping technology gadgets home lifestyle deals consumer ecommerce retail",
     "active": True},
    {"id":"ad4","brand":"Netflix","title":"Stories Worth Watching","emoji":"🎬","cta":"Watch Now",
     "bg":"linear-gradient(135deg,#e50914,#a30610)","headline":"New Episodes Every Week",
     "body":"Award-winning series — start streaming today.","advertiser_suitability":"High",
     "tags":"entertainment drama story adventure fiction film celebrity arts television streaming",
     "active": True},
    {"id":"ad5","brand":"Duolingo","title":"Learn a Language","emoji":"🦜","cta":"Start Free",
     "bg":"linear-gradient(135deg,#58cc02,#3d9900)","headline":"5 Minutes a Day Changes Everything",
     "body":"40+ languages. Free forever.","advertiser_suitability":"High",
     "tags":"education learning language travel culture knowledge students school skills",
     "active": True},
    {"id":"ad6","brand":"Uber Eats","title":"Food at Your Door","emoji":"🍔","cta":"Order Now",
     "bg":"linear-gradient(135deg,#06c167,#038a47)","headline":"Craving Something?",
     "body":"Your favourite restaurants in 30 minutes.","advertiser_suitability":"Medium",
     "tags":"food cooking restaurant lifestyle family celebration delivery dining parenting",
     "active": True},
    {"id":"ad7","brand":"Mastercard","title":"Priceless Moments","emoji":"💳","cta":"Learn More",
     "bg":"linear-gradient(135deg,#eb5757,#b91c1c)","headline":"There Are Things Money Can't Buy",
     "body":"For everything else, there's Mastercard.","advertiser_suitability":"High",
     "tags":"finance business economy success achievement luxury banking news career",
     "active": True},
    {"id":"ad8","brand":"BMW","title":"The Ultimate Drive","emoji":"🚗","cta":"Book Test Drive",
     "bg":"linear-gradient(135deg,#1e40af,#1e3a8a)","headline":"Sheer Driving Pleasure",
     "body":"New BMW 5 Series. Redefining performance.","advertiser_suitability":"High",
     "tags":"automotive cars speed luxury engineering technology premium travel",
     "active": True},
]


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIBRARY
# ══════════════════════════════════════════════════════════════════════════════
def page_library():
    st.markdown('<div style="font-size:2rem;font-weight:800;color:#111827;padding:4px 0 6px">🗂️ Video Library</div>', unsafe_allow_html=True)
    st.caption("Your content hub — ingest any video, explore and monetise your catalogue")
    st.divider()

    if "lib_open" not in st.session_state:
        st.session_state.lib_open = not bool(st.session_state.videos)

    btn_label = "➖  Close" if st.session_state.lib_open else "➕  Add a Video"
    if st.button(btn_label, key="lib_toggle"):
        st.session_state.lib_open = not st.session_state.lib_open
        st.rerun()

    if st.session_state.lib_open:
        with st.container(border=True):
            st.markdown("&nbsp;", unsafe_allow_html=True)
            lib_src = st.radio("Source", ["YouTube URL", "Upload MP4 + Subtitle"], horizontal=True, key="lib_src")
            st.divider()

            if lib_src == "YouTube URL":
                yt_url = st.text_input("YouTube URL or video ID",
                    placeholder="https://youtube.com/watch?v=… or paste the 11-char video ID", key="lib_yt")
                st.markdown("**Subtitle file** *(optional)*")
                manual_srt = st.file_uploader("Upload .srt or .vtt", type=["srt","vtt"], key="lib_yt_srt")
                if manual_srt: st.success(f"✅ Using uploaded subtitle: {manual_srt.name}")
                else: st.caption("ℹ️ Auto-fetch works for most public videos.")

                a1, a2, a3 = st.columns(3)
                yt_min_s = a1.slider("Min scene (s)", 10, 90, 20, 5, key="lib_min")
                yt_max_s = a2.slider("Max scene (s)", 60, 300, 120, 10, key="lib_max")
                yt_sens  = a3.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="lib_sens")
                do_ai_yt = st.checkbox("Generate AI metadata after processing", value=True, key="lib_do_ai_yt")

                if st.button("⚡ Process YouTube Video", type="primary", key="lib_yt_go"):
                    vid_id = _yt_id(yt_url.strip()) if yt_url else None
                    if not vid_id: st.error("Invalid YouTube URL or video ID"); st.stop()
                    if vid_id in st.session_state.videos: st.warning("Already in library."); st.stop()
                    status = st.status("Processing video…", expanded=True)
                    try:
                        with status:
                            if manual_srt:
                                transcript = manual_srt.read().decode("utf-8", errors="replace")
                                fmt = "vtt" if manual_srt.name.lower().endswith(".vtt") else "srt"
                            else:
                                st.write("🌐 Fetching transcript…")
                                try: transcript = fetch_youtube_transcript(vid_id)
                                except Exception as e: transcript = None; st.warning(str(e))
                                if not transcript:
                                    status.update(label="❌ No transcript", state="error")
                                    st.error("Could not fetch transcript. Upload an SRT file."); st.stop()
                                fmt = "srt"
                            st.write("📋 Fetching metadata…")
                            try:
                                meta  = fetch_youtube_metadata(vid_id, st.session_state.yt_api_key or None)
                                title = meta.get("title", f"YouTube · {vid_id}") if meta else f"YouTube · {vid_id}"
                            except Exception: title = f"YouTube · {vid_id}"
                            st.write("🔬 Detecting scenes…")
                            vm = VideoProcessor(yt_min_s, yt_max_s, yt_sens).process_file(transcript, title, fmt)
                            vm.yt_id = vid_id
                            st.write(f"🗂️ Indexing {vm.scene_count} scenes…")
                            _register(vm); _sync_qp()
                            if do_ai_yt:
                                st.write("🤖 Generating AI metadata…")
                                _generate_ai_meta(vm, transcript[:4000])
                            status.update(label=f"✅ {vm.scene_count} scenes", state="complete")
                    except Exception as e:
                        status.update(label="❌ Failed", state="error")
                        st.error(str(e)); st.stop()
                    st.rerun()

            else:  # Upload branch
                c1, c2 = st.columns([3, 2], gap="large")
                with c1:
                    mp4_file = st.file_uploader("Video file (MP4, MOV, WebM) — optional, enables player",
                                                type=["mp4","mov","webm","avi"], key="lib_mp4")
                    srt_file = st.file_uploader("Subtitle file (.srt or .vtt) — required",
                                                type=["srt","vtt"], key="lib_srt")
                    title_in = st.text_input("Video title (optional)", key="lib_title")
                    if mp4_file:
                        size_mb = mp4_file.size / (1024*1024)
                        if size_mb > 50:
                            st.warning(f"⚠️ File is {size_mb:.0f}MB. Files over 50MB may not play correctly in the browser. Consider using YouTube URL instead.")
                with c2:
                    st.markdown("**Detection settings**")
                    up_min_s = st.slider("Min scene (s)", 10, 90, 20, 5, key="lib_u_min")
                    up_max_s = st.slider("Max scene (s)", 60, 300, 120, 10, key="lib_u_max")
                    up_sens  = st.slider("Sensitivity", 0.2, 0.7, 0.35, 0.05, key="lib_u_sens")
                    do_ai = st.checkbox("Generate AI metadata", value=True, key="lib_do_ai")

                if srt_file and st.button("⚡ Process Video", type="primary", key="lib_up_go"):
                    srt_file.seek(0)
                    srt_content = srt_file.read().decode("utf-8", errors="replace")
                    fmt   = "vtt" if srt_file.name.lower().endswith(".vtt") else "srt"
                    title = title_in.strip() or (mp4_file.name if mp4_file else "Uploaded Video")
                    status = st.status("Processing video…", expanded=True)
                    try:
                        with status:
                            st.write("🔬 Detecting scenes…")
                            vm = VideoProcessor(up_min_s, up_max_s, up_sens).process_file(srt_content, title, fmt)
                            if not vm.scenes:
                                status.update(label="❌ No scenes", state="error")
                                st.error("No scenes detected — check subtitle format."); st.stop()
                            vm.yt_id = None
                            if mp4_file:
                                st.write("🎞️ Storing video file…")
                                mp4_file.seek(0)
                                raw = mp4_file.read()
                                st.session_state.video_b64[vm.video_id] = _b64.b64encode(raw).decode()
                                ext = mp4_file.name.rsplit(".", 1)[-1].lower()
                                st.session_state.video_mime[vm.video_id] = {
                                    "mp4":"video/mp4","mov":"video/mp4",
                                    "webm":"video/webm","avi":"video/x-msvideo"
                                }.get(ext, "video/mp4")
                            st.write(f"🗂️ Indexing {vm.scene_count} scenes…")
                            _register(vm); _sync_qp()
                            if do_ai:
                                st.write("🤖 Generating AI metadata…")
                                _generate_ai_meta(vm, srt_content[:4000])
                            status.update(label=f"✅ {vm.scene_count} scenes", state="complete")
                    except Exception as e:
                        status.update(label="❌ Failed", state="error")
                        st.error(str(e)); st.stop()
                    st.rerun()

    if not st.session_state.videos:
        st.markdown("""<div style="text-align:center;padding:40px 20px 16px">
          <div style="font-size:3rem">🎬</div>
          <div style="font-size:1.2rem;font-weight:700;margin-top:12px;color:#111827">No videos yet</div>
          <div style="color:#6b7280;margin-top:6px">Add a video above — or try a demo</div>
        </div>""", unsafe_allow_html=True)
        demo_col, _ = st.columns([2, 3])
        with demo_col:
            with st.container(border=True):
                st.markdown("**🎬 Load a demo video**")
                demos = {
                    "TED Talk — Do schools kill creativity?": "iG9CE55wbtY",
                    "NASA Artemis Launch Highlights":         "KHFozTSMzqA",
                    "Veritasium — How Electricity Works":     "oI_X2cMHNe0",
                }
                demo_choice = st.selectbox("Choose demo", list(demos.keys()), key="demo_sel")
                if st.button("⚡ Load Demo", type="primary", key="demo_load"):
                    demo_id = demos[demo_choice]
                    if demo_id not in st.session_state.videos:
                        status = st.status("Loading demo...", expanded=True)
                        try:
                            with status:
                                st.write("🌐 Fetching transcript...")
                                try: transcript = fetch_youtube_transcript(demo_id)
                                except: transcript = None
                                if not transcript:
                                    status.update(label="❌ Demo unavailable", state="error")
                                    st.error("Could not fetch demo. Try adding a video manually."); st.stop()
                                meta  = fetch_youtube_metadata(demo_id, st.session_state.yt_api_key or None)
                                title = meta.get("title", demo_choice) if meta else demo_choice
                                vm = VideoProcessor(20, 120, 0.35).process_file(transcript, title, "srt")
                                vm.yt_id = demo_id
                                _register(vm); _sync_qp()
                                st.write("🤖 Generating AI metadata…")
                                _generate_ai_meta(vm, transcript[:4000])
                                status.update(label=f"✅ {vm.scene_count} scenes", state="complete")
                        except Exception as e:
                            st.error(f"Error: {e}"); st.stop()
                    st.rerun()
        return

    # Library KPIs
    all_dur = sum(v.duration_sec for v in st.session_state.videos.values())
    dur_str = f"{int(all_dur//3600)}h {int((all_dur%3600)//60)}m" if all_dur>3600 else f"{int(all_dur//60)}m"

    # NEW: avg monetisation score across library
    all_scores = [st.session_state.ai_meta.get(v.video_id,{}).get("monetisation_score") for v in st.session_state.videos.values()]
    all_scores = [s for s in all_scores if s is not None]
    avg_mscore = int(sum(all_scores)/len(all_scores)) if all_scores else None

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Videos",    len(st.session_state.videos))
    c2.metric("Total Scenes",    sum(v.scene_count for v in st.session_state.videos.values()))
    c3.metric("Total Duration",  dur_str)
    c4.metric("Key Moments",     sum(len(v.key_scenes) for v in st.session_state.videos.values()))
    if avg_mscore is not None:
        c5.metric("Avg Monetisation Score", f"{avg_mscore}/100")

    # Sort library by monetisation score descending (NEW)
    st.markdown("### Your Videos")
    st.caption("💡 Sorted by Monetisation Score — highest ad opportunity first")
    sorted_vms = sorted(
        st.session_state.videos.values(),
        key=lambda v: st.session_state.ai_meta.get(v.video_id,{}).get("monetisation_score", 0),
        reverse=True
    )
    for vm in sorted_vms:
        _library_card(vm)


def _library_card(vm: VideoMetadata):
    is_active = vm.video_id == st.session_state.selected_video
    safe_pct  = sum(1 for s in vm.scenes if s.brand_safety.get("safety_score",1)>=0.7) / max(vm.scene_count,1)
    avg_eng   = sum(s.engagement_score for s in vm.scenes) / max(vm.scene_count,1)
    has_mp4   = vm.video_id in st.session_state.video_b64
    has_yt    = bool(getattr(vm,"yt_id",None))
    meta      = st.session_state.ai_meta.get(vm.video_id, {})
    mscore    = meta.get("monetisation_score", None)

    with st.container(border=True):
        active_badge = ' <span style="background:#fef3c7;color:#d97706;font-size:11px;padding:2px 8px;border-radius:6px;font-weight:600">ACTIVE</span>' if is_active else ""
        src_badge    = (' <span style="background:#fee2e2;color:#dc2626;font-size:11px;padding:2px 8px;border-radius:6px">▶ YouTube</span>' if has_yt
                        else ' <span style="background:#dbeafe;color:#2563eb;font-size:11px;padding:2px 8px;border-radius:6px">🎞 MP4</span>' if has_mp4
                        else ' <span style="background:#f3f4f6;color:#6b7280;font-size:11px;padding:2px 8px;border-radius:6px">📄 SRT only</span>')

        # NEW: Monetisation score badge on card
        mscore_badge = ""
        if mscore is not None:
            sc = "#16a34a" if mscore >= 70 else "#d97706" if mscore >= 40 else "#dc2626"
            bg = "#dcfce7" if mscore >= 70 else "#fef3c7" if mscore >= 40 else "#fee2e2"
            mscore_badge = f' <span style="background:{bg};color:{sc};font-size:11px;padding:2px 8px;border-radius:6px;font-weight:700">💰 {mscore}/100</span>'

        st.markdown(f'<div style="font-size:1.1rem;font-weight:700;color:#111827">{vm.title}{active_badge}{src_badge}{mscore_badge}</div>', unsafe_allow_html=True)
        st.caption(f"{vm.fmt_duration()}  ·  {vm.scene_count} scenes  ·  {len(vm.key_scenes)} key moments  ·  arc: **{vm.narrative_structure}**")

        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Avg Engagement", f"{avg_eng:.2f}")
        m2.metric("Brand Safe",     f"{safe_pct:.0%}")
        m3.metric("Key Scenes",     len(vm.key_scenes))
        best_scene = max(vm.scenes, key=lambda s: s.ad_suitability * s.engagement_score) if vm.scenes else None
        m4.metric("Top Ad Moment",  best_scene.start_fmt if best_scene else "—")

        tags_html = "".join(_tag_chip(c["name"]) for c in {c["name"]: c for c in vm.dominant_iab[:8]}.values())
        if meta:
            rating_color = {"G":"#16a34a","PG":"#2563eb","PG-13":"#d97706","R":"#dc2626"}.get(meta.get("content_rating","PG"),"#6b7280")
            tags_html += f'<span style="background:{rating_color};color:#fff;font-size:11px;font-weight:700;padding:3px 10px;border-radius:20px;margin:2px">{meta.get("content_rating","?")} · {meta.get("primary_genre","")}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)
        if meta and meta.get("short_description"):
            st.caption(f'💬 {meta["short_description"]}')
        st.markdown("")

        a1,a2,a3,a4,a5 = st.columns(5)
        def _nav_to(vid_id, page):
            st.session_state.selected_video = vid_id
            st.session_state.page = page
            _sync_qp()

        if a1.button("🔬 Analyse",  key=f"lib_an_{vm.video_id}"): _nav_to(vm.video_id,"analyse");  st.rerun()
        if a2.button("📢 Monetise", key=f"lib_mo_{vm.video_id}"): _nav_to(vm.video_id,"monetise"); st.rerun()
        if a3.button("📊 Insights", key=f"lib_in_{vm.video_id}"): _nav_to(vm.video_id,"insights"); st.rerun()
        if a4.button("📄 Report",   key=f"lib_rp_{vm.video_id}"):
            _nav_to(vm.video_id,"insights")
            st.session_state._insights_tab = "report"
            st.rerun()
        if a5.button("🗑 Remove",   key=f"lib_dl_{vm.video_id}"):
            del st.session_state.videos[vm.video_id]
            for d in [st.session_state.video_b64, st.session_state.video_mime,
                      st.session_state.ad_markers, st.session_state.ai_meta]:
                d.pop(vm.video_id, None)
            if st.session_state.selected_video == vm.video_id:
                st.session_state.selected_video = next(iter(st.session_state.videos), None)
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE  (unchanged from v4 except CPM disclaimer added)
# ══════════════════════════════════════════════════════════════════════════════
def page_analyse():
    vm = _active_vm()
    if not vm:
        st.info("No video selected — go to Library first.")
        if st.button("← Library"): st.session_state.page="library"; st.rerun()
        return

    yt_id   = getattr(vm,"yt_id",None)
    has_mp4 = vm.video_id in st.session_state.video_b64
    avg_eng = round(sum(s.engagement_score for s in vm.scenes)/max(vm.scene_count,1),2)
    safe_pct= f"{sum(1 for s in vm.scenes if s.brand_safety.get('safety_score',1)>=0.7)/max(vm.scene_count,1):.0%}"

    st.markdown(f'<div style="font-size:1.8rem;font-weight:800;color:#111827;padding:4px 0 4px">🔬 {vm.title}</div>', unsafe_allow_html=True)
    st.caption(f"{vm.fmt_duration()} · {vm.scene_count} scenes · arc: **{vm.narrative_structure}**")

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Scenes",         vm.scene_count)
    c2.metric("Duration",       vm.fmt_duration())
    c3.metric("Key Moments",    len(vm.key_scenes))
    c4.metric("Avg Engagement", avg_eng)
    c5.metric("Brand Safe",     safe_pct)

    tags_html = "".join(_tag_chip(c["name"]) for c in {c["name"]: c for c in vm.dominant_iab[:8]}.values())
    if getattr(vm,"franchise_themes",None):
        tags_html += "".join(_tag_chip(t,"#eff6ff","#bfdbfe","#1d4ed8") for t in vm.franchise_themes[:4])
    st.markdown(tags_html, unsafe_allow_html=True)
    st.divider()

    tab_watch, tab_timeline, tab_scenes, tab_search, tab_opps, tab_aimeta = st.tabs([
        "▶️  Watch", "📈  Timeline", "🎬  Scenes", "🔍  Search", "🎯  Ad Opportunities", "✨  AI Metadata"
    ])

    with tab_watch:
        if yt_id:
            _yt_player_with_scenes(yt_id, vm.scenes, vm.key_scenes)
        elif has_mp4:
            import io, base64 as _b64v
            raw_bytes = _b64v.b64decode(st.session_state.video_b64[vm.video_id])
            mime = st.session_state.video_mime.get(vm.video_id, "video/mp4")
            st.video(io.BytesIO(raw_bytes), format=mime)
        else:
            st.info("No playable video — upload an MP4 in Library, or explore scenes below.")

    with tab_timeline:
        _render_timeline(vm)

    with tab_scenes:
        f1,f2,f3,f4 = st.columns(4)
        sent_f   = f1.multiselect("Sentiment", ["positive","neutral","negative"], default=["positive","neutral","negative"], key="sc_sent")
        min_eng  = f2.slider("Min engagement", 0.0,1.0,0.0,0.05, key="sc_eng")
        min_fit  = f3.slider("Min ad fit",     0.0,1.0,0.0,0.05, key="sc_fit")
        key_only = f4.checkbox("Key moments only", key="sc_key")
        shown = [s for s in vm.scenes
                 if s.sentiment.get("label","neutral") in sent_f
                 and s.engagement_score >= min_eng
                 and s.ad_suitability   >= min_fit
                 and (not key_only or s.scene_id in vm.key_scenes)]
        st.caption(f"**{len(shown)}** of {vm.scene_count} scenes")
        for scene in shown:
            _scene_card(scene, vm, yt_id=yt_id)

    with tab_search:
        _search_panel(vm)

    with tab_opps:
        _ad_opportunity_panel(vm, yt_id)
        st.caption(CPM_DISCLAIMER)

    with tab_aimeta:
        _show_ai_meta(vm)


def _search_panel(vm: VideoMetadata):
    se = st.session_state.search_engine
    srch_mode = st.radio("Mode", ["🔍 Semantic Search", "📢 Ad Targeting", "🎬 Similar Scenes"],
                         horizontal=True, key="srch_mode")
    st.markdown("")

    if srch_mode == "🔍 Semantic Search":
        if "aq_staged" not in st.session_state: st.session_state.aq_staged = ""
        default_q = st.session_state.pop("aq_staged", "")
        query = st.text_input("Search by meaning",
            placeholder="e.g. 'tense confrontation' · 'product demo' · 'emotional speech'",
            key="aq", value=default_q)
        if not query:
            st.caption("**Quick searches:**")
            examples = ["action sequence","emotional moment","expert interview","product showcase",
                        "dramatic reveal","comedy scene","tutorial step","key decision","outdoor scene","crowd reaction"]
            cols = st.columns(5)
            for i, ex in enumerate(examples):
                if cols[i%5].button(ex, key=f"aq_{i}"):
                    st.session_state.aq_staged = ex; st.rerun()
        else:
            k1,k2,k3 = st.columns(3)
            top_k  = k1.select_slider("Results", [3,5,10,20], value=5, key="aq_k")
            safety = k2.selectbox("Safety", ["Any","Moderate (50%+)","Strict (80%+)"], key="aq_safe")
            scope  = k3.radio("Scope", ["This video","All videos"], horizontal=True, key="aq_scope")
            smap   = {"Any":0.0,"Moderate (50%+)":0.5,"Strict (80%+)":0.8}
            with st.spinner("Searching…"):
                results = se.search(query, top_k=top_k, diversify=True, min_safety=smap[safety], expand=True)
            if scope == "This video":
                results = [r for r in results if r.scene.video_id == vm.video_id]
            if not results:
                st.warning("No matches — try different words.")
            else:
                st.success(f"**{len(results)} scenes** matched · sorted by relevance")
                for r in results:
                    _vm2 = st.session_state.videos.get(r.scene.video_id, vm)
                    if scope == "All videos" and len(st.session_state.videos) > 1:
                        st.caption(f"📹 {_vm2.title[:50]}")
                    _scene_card(r.scene, _vm2, score=r.score,
                                yt_id=getattr(_vm2,"yt_id",None),
                                ad_match=_best_ad_for_scene(r.scene))

    elif srch_mode == "📢 Ad Targeting":
        at1, at2 = st.columns(2)
        ad_brand  = at1.text_input("Brand / Product", placeholder="e.g. Nike, Tesla, Duolingo", key="at_brand")
        ad_tags   = at2.text_input("Targeting keywords", placeholder="e.g. sports fitness motivation energy", key="at_tags")
        ad_safety = st.select_slider("Min brand safety", [0.5,0.6,0.7,0.8,0.9,1.0], value=0.7, key="at_safe")
        scope_at  = st.radio("Search scope", ["This video","All videos"], horizontal=True, key="at_scope")
        if st.button("🎯 Find Best Placements", type="primary", key="at_go", disabled=not(ad_brand or ad_tags)):
            dummy_ad = {"id":"preview","brand":ad_brand,"tags":ad_tags,"title":ad_brand,
                        "headline":ad_brand,"body":ad_tags}
            pool = (vm.scenes if scope_at=="This video"
                    else [s for v in st.session_state.videos.values() for s in v.scenes])
            scored = []
            for scene in pool:
                if scene.brand_safety.get("safety_score",1) < ad_safety: continue
                sim = _ad_similarity(dummy_ad, scene)
                opp = sim["total"] * scene.engagement_score * scene.brand_safety.get("safety_score",1)
                scored.append((scene, sim, opp))
            scored.sort(key=lambda x: x[2], reverse=True)
            top = scored[:10]
            if not top:
                st.warning("No suitable scenes found.")
            else:
                st.success(f"**{len(top)} best placements** for **{ad_brand or ad_tags}**")
                for scene, sim, opp in top:
                    _vm2 = st.session_state.videos.get(scene.video_id, vm)
                    c1, c2 = st.columns([4,1])
                    with c1: _scene_card(scene, _vm2, score=opp, yt_id=getattr(_vm2,"yt_id",None))
                    with c2:
                        if sim["matched_iab"]:
                            st.caption("**Matched:** " + ", ".join(sim["matched_iab"][:3]))
                        # NEW: show vector score
                        st.caption(f"🧠 Vector: {sim.get('vector',0):.0%}")
                        if st.button("➕ Add to Plan", key=f"at_add_{scene.scene_id}"):
                            vid = scene.video_id
                            if vid not in st.session_state.ad_markers:
                                st.session_state.ad_markers[vid] = []
                            if scene.start_sec not in [m["sec"] for m in st.session_state.ad_markers[vid]]:
                                ad_obj = {"id":"custom_at","brand":ad_brand,"title":ad_brand,
                                          "emoji":"📢","headline":ad_brand,"body":ad_tags,
                                          "cta":"Learn More","bg":"linear-gradient(135deg,#6366f1,#8b5cf6)",
                                          "tags":ad_tags,"active":True}
                                st.session_state.ad_markers[vid].append({
                                    "sec":scene.start_sec,"fmt":_fmt_sec(scene.start_sec),
                                    "ad":ad_obj,"mode":"manual","duration":15,
                                    "sim":sim["total"],"reason":f"Ad targeting: {ad_brand}"})
                                st.success("Added!")
                            else:
                                st.info("Already in plan")

    else:  # Similar Scenes
        scene_labels = [f"{s.start_fmt} · {s.text[:50]}…" for s in vm.scenes]
        scene_idx = st.selectbox("Reference scene", range(len(vm.scenes)),
                                 format_func=lambda i: scene_labels[i], key="sim_sc")
        ref_scene = vm.scenes[scene_idx]
        sim_scope = st.radio("Scope", ["This video","All videos"], horizontal=True, key="sim_scope")
        if st.button("🔍 Find Similar", type="primary", key="sim_go"):
            ref_iab  = {c["name"] for c in ref_scene.iab_categories}
            ref_sent = ref_scene.sentiment.get("label","neutral")
            ref_tok  = _tok(ref_scene.text)
            pool = (vm.scenes if sim_scope=="This video"
                    else [s for v in st.session_state.videos.values() for s in v.scenes])
            scored2 = []
            for scene in pool:
                if scene.scene_id == ref_scene.scene_id: continue
                sc_iab = {c["name"] for c in scene.iab_categories}
                iab_sim = len(ref_iab & sc_iab) / max(len(ref_iab | sc_iab), 1)
                sent_sim = 1.0 if scene.sentiment.get("label") == ref_sent else 0.3
                sc_tok  = _tok(scene.text)
                tok_sim = len(ref_tok & sc_tok) / max(len(ref_tok | sc_tok), 1)
                eng_sim = 1 - abs(scene.engagement_score - ref_scene.engagement_score)
                total = iab_sim*0.4 + sent_sim*0.2 + tok_sim*0.3 + eng_sim*0.1
                scored2.append((scene, total))
            scored2.sort(key=lambda x: x[1], reverse=True)
            st.markdown(f"**Reference:** `{ref_scene.start_fmt}` — {ref_scene.text[:80]}…")
            st.divider()
            for scene, sim_score in scored2[:8]:
                _vm2 = st.session_state.videos.get(scene.video_id, vm)
                _scene_card(scene, _vm2, score=sim_score, yt_id=getattr(_vm2,"yt_id",None))


def _ad_opportunity_panel(vm: VideoMetadata, yt_id=None):
    st.markdown("### 🎯 Top Ad Opportunities")
    st.caption("AI-scored moments ranked by engagement × ad fit × brand safety × context.")

    def _opp_score(s):
        return s.engagement_score * s.ad_suitability * s.brand_safety.get("safety_score",1.0)

    ranked = sorted(vm.scenes, key=_opp_score, reverse=True)[:5]

    for rank, scene in enumerate(ranked):
        score  = _opp_score(scene)
        ad, sim = _best_ad_for_scene(scene)
        sent   = scene.sentiment.get("label","neutral")
        safety = scene.brand_safety.get("safety_score",1.0)
        is_key = scene.scene_id in vm.key_scenes

        reasons = []
        if scene.engagement_score > 0.6:  reasons.append(f"⚡ High engagement ({scene.engagement_score:.0%})")
        if scene.ad_suitability   > 0.6:  reasons.append(f"🎯 Strong ad fit ({scene.ad_suitability:.0%})")
        if safety >= 0.8:                  reasons.append("🛡 Fully brand-safe")
        if sent == "positive":             reasons.append("🟢 Positive sentiment")
        if is_key:                         reasons.append("⭐ Key narrative moment")
        if sim["total"] > 0.4:             reasons.append(f"📢 Strong ad match ({sim['total']:.0%})")
        if sim.get("vector",0) > 0.3:      reasons.append(f"🧠 Semantic match ({sim['vector']:.0%})")
        if not reasons:                    reasons.append("📍 Contextually suitable placement")

        medal = ["🥇","🥈","🥉","4️⃣","5️⃣"][rank]
        pct   = int(score * 100)
        bar_c = "#16a34a" if score>0.4 else "#f59e0b" if score>0.2 else "#9ca3af"

        with st.container(border=True):
            h1,h2 = st.columns([7,3])
            with h1:
                st.markdown(f"**{medal} Rank #{rank+1}** — `{scene.start_fmt} → {scene.end_fmt}`" + (" ⭐" if is_key else ""))
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:10px;margin:4px 0">'
                    f'<div style="flex:1;height:8px;background:#f3f4f6;border-radius:4px">'
                    f'<div style="width:{pct}%;height:8px;background:{bar_c};border-radius:4px"></div></div>'
                    f'<span style="font-size:14px;font-weight:800;color:{bar_c}">{pct}%</span>'
                    f'<span style="font-size:11px;color:#9ca3af">opportunity score</span></div>',
                    unsafe_allow_html=True)
                st.caption(" · ".join(reasons))
            with h2:
                st.markdown(
                    f'<div style="background:{"#f0fdf4" if sim["total"]>0.4 else "#fefce8"};'
                    f'border:1px solid {"#bbf7d0" if sim["total"]>0.4 else "#fef08a"};'
                    f'border-radius:10px;padding:10px;text-align:center">'
                    f'<div style="font-size:1.4rem">{ad["emoji"]}</div>'
                    f'<div style="font-weight:700;font-size:13px">{ad["brand"]}</div>'
                    f'<div style="font-size:11px;color:#6b7280">{ad["title"]}</div>'
                    f'<div style="font-weight:700;color:{"#16a34a" if sim["total"]>0.4 else "#ca8a04"};font-size:13px">'
                    f'🎯 {sim["total"]:.0%} match</div>'
                    f'<div style="font-size:10px;color:#9ca3af">🧠 {sim.get("vector",0):.0%} semantic</div>'
                    f'</div>', unsafe_allow_html=True)

            st.write(scene.text[:150] + ("…" if len(scene.text)>150 else ""))
            tag_html = "".join(_tag_chip(c["name"],"#f0fdf4","#bbf7d0","#166534","11px") for c in scene.iab_categories[:3])
            if tag_html: st.markdown(tag_html, unsafe_allow_html=True)

            cpm_lo, cpm_hi = _est_cpm(scene)
            sig_cols = st.columns(5)
            sig_cols[0].metric("Engagement",  f"{scene.engagement_score:.0%}")
            sig_cols[1].metric("Ad Fit",       f"{scene.ad_suitability:.0%}")
            sig_cols[2].metric("Brand Safety", f"{safety:.0%}")
            sig_cols[3].metric("Ad Match",     f"{sim['total']:.0%}")
            sig_cols[4].metric("Est. CPM",     f"${cpm_lo:.0f}–${cpm_hi:.0f}", help="Indicative CPM range. Actual rates vary.")

            ac1, ac2, ac3 = st.columns(3)
            if yt_id:
                ac1.link_button(f"▶ Jump to {scene.start_fmt}", f"https://youtube.com/watch?v={yt_id}&t={scene.start_sec}s")
            if ac2.button(f"➕ Add to Ad Plan", key=f"opp_add_{scene.scene_id}"):
                vid_id = vm.video_id
                if vid_id not in st.session_state.ad_markers:
                    st.session_state.ad_markers[vid_id] = []
                existing = [m["sec"] for m in st.session_state.ad_markers[vid_id]]
                if scene.start_sec not in existing:
                    st.session_state.ad_markers[vid_id].append({
                        "sec":scene.start_sec,"fmt":_fmt_sec(scene.start_sec),
                        "ad":ad,"mode":"auto","duration":15,
                        "sim":sim["total"],"reason":" · ".join(reasons[:3])})
                    st.success(f"Added {scene.start_fmt} → Monetise tab")
                else:
                    st.info("Already in plan")


def _scene_card(scene: Scene, vm: VideoMetadata, score=None, yt_id=None, ad_match=None):
    is_key = scene.scene_id in vm.key_scenes
    sent   = scene.sentiment.get("label","neutral")
    safety = scene.brand_safety.get("safety_score",1.0)
    with st.container(border=True):
        r1,r2 = st.columns([7,3])
        with r1:
            st.markdown(f'`{scene.start_fmt}→{scene.end_fmt}` **{scene.duration_sec:.0f}s**'
                        +(" ⭐" if is_key else "")+(f" — 🎯 **{score:.0%}**" if score else ""))
        with r2:
            st.caption(f'{_sent_icon(sent)} {sent}  ·  🛡 {safety:.0%}  ·  ⚡ {scene.engagement_score:.2f}')
        st.write(scene.text[:200]+("…" if len(scene.text)>200 else ""))
        tag_html = "".join(_tag_chip(c["name"],"#f0fdf4","#bbf7d0","#166534","11px") for c in scene.iab_categories[:3])
        if tag_html: st.markdown(tag_html, unsafe_allow_html=True)
        if ad_match:
            ad,sim = ad_match
            sc = "#16a34a" if sim["total"]>0.45 else "#f59e0b" if sim["total"]>0.25 else "#9ca3af"
            st.markdown(
                f'<span style="font-size:12px;color:{sc};font-weight:600">'
                f'💡 {ad["emoji"]} {ad["brand"]} — {sim["total"]:.0%} match'
                f' (🧠 {sim.get("vector",0):.0%} semantic)</span>', unsafe_allow_html=True)
        if score: st.progress(min(score,1.0))
        if yt_id: st.link_button(f"▶ Open at {scene.start_fmt}", f"https://youtube.com/watch?v={yt_id}&t={scene.start_sec}s")


def _render_timeline(vm: VideoMetadata):
    import json as _j

    st.markdown("#### 📈 Scene Timeline")

    # NEW: warn if timeline is truncated
    MAX_SEG = 200
    if vm.scene_count > MAX_SEG:
        st.warning(f"⚠️ This video has {vm.scene_count} scenes. The interactive timeline shows the first {MAX_SEG} for performance. All scenes are available in the Scenes tab.")

    scene_data = []
    for s in vm.scenes:
        sent  = s.sentiment.get("label","neutral")
        color = {"positive":"#34d399","negative":"#f87171","neutral":"#60a5fa"}.get(sent,"#60a5fa")
        scene_data.append({
            "start":s.start_sec,"end":s.end_sec,"dur":s.duration_sec,
            "eng":s.engagement_score,"sent":sent,"color":color,
            "fit":s.ad_suitability,"key":s.scene_id in vm.key_scenes,
            "label":s.start_fmt,"text":s.text[:50].replace("'","").replace('"',""),
            "iab":_iab_str(s.iab_categories),"safe":s.brand_safety.get("safety_score",1.0),
        })

    markers = st.session_state.ad_markers.get(vm.video_id, [])
    markers_j = _j.dumps([{"sec":m["sec"],"fmt":m["fmt"],"brand":m["ad"]["brand"],
                            "emoji":m["ad"]["emoji"],"sim":m.get("sim",0)} for m in markers])

    df = pd.DataFrame(scene_data)
    fig = go.Figure()
    for sent, col in [("positive","#34d399"),("neutral","#60a5fa"),("negative","#f87171")]:
        sub = df[df["sent"]==sent]
        if sub.empty: continue
        fig.add_trace(go.Bar(
            x=sub["start"], y=sub["eng"], name=sent.capitalize(),
            marker_color=col, width=sub["dur"]*0.8,
            customdata=sub[["label","iab","text","fit","safe"]].values,
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[2]}<br>IAB: %{customdata[1]}<br>Eng: %{y:.0%} · Fit: %{customdata[3]:.0%} · Safe: %{customdata[4]:.0%}<extra></extra>"
        ))
    ks = [s for s in scene_data if s["key"]]
    if ks:
        fig.add_trace(go.Scatter(
            x=[s["start"] for s in ks], y=[s["eng"]+0.06 for s in ks],
            mode="markers", name="Key moment",
            marker=dict(symbol="star",size=14,color="#f59e0b"),
            hovertemplate="Key moment @ %{x}s<extra></extra>"))
    for m in markers:
        fig.add_vline(x=m["sec"], line_color="#f59e0b", line_width=2, line_dash="dot",
                      annotation_text=f'📢{m["fmt"][3:]}',
                      annotation_font_size=9, annotation_font_color="#d97706")
    fig.update_layout(**PT, height=260, barmode="overlay",
        xaxis=dict(**_XA,title="Time (seconds)"),
        yaxis=dict(**_YA,title="Engagement",range=[0,1.15]),
        legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color=_TEXT)))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🎯 Ad Opportunity Heat Map")
    strip_html = '<div style="display:flex;flex-wrap:wrap;gap:3px;margin:8px 0">'
    for s in vm.scenes:
        score = s.ad_suitability * s.engagement_score * s.brand_safety.get("safety_score",1.0)
        alpha = 0.2 + score * 0.8
        is_key = s.scene_id in vm.key_scenes
        border = "2px solid #f59e0b" if is_key else "1px solid #e5e7eb"
        has_marker = any(abs(m["sec"]-s.start_sec)<5 for m in markers)
        bg = f"rgba(22,163,74,{alpha:.2f})"
        title_attr = f"{s.start_fmt}: opp {score:.0%}, eng {s.engagement_score:.0%}, fit {s.ad_suitability:.0%}"
        strip_html += (f'<div title="{title_attr}" style="width:32px;height:32px;border-radius:5px;'
                       f'background:{bg};border:{border};display:flex;align-items:center;'
                       f'justify-content:center;font-size:11px">'
                       + ("⭐" if is_key else ("📢" if has_marker else "")) + "</div>")
    strip_html += "</div>"
    st.markdown(strip_html, unsafe_allow_html=True)


def _yt_player_with_scenes(vid_id, scenes, key_scene_ids):
    import json
    sd = []
    for i,s in enumerate(scenes):
        sd.append({"idx":i,"start":s.start_sec,"end":s.end_sec,
            "start_fmt":s.start_fmt,"end_fmt":s.end_fmt,"dur":int(s.duration_sec),
            "text":s.text[:200].replace('"','').replace("\n"," "),
            "sent":s.sentiment.get("label","neutral"),
            "sent_icon":_sent_icon(s.sentiment.get("label","neutral")),
            "safety":f"{s.brand_safety.get('safety_score',1):.0%}",
            "iab":_iab_str(s.iab_categories),"eng":f"{s.engagement_score:.2f}",
            "fit":f"{s.ad_suitability:.2f}","is_key":s.scene_id in key_scene_ids})
    sj = json.dumps(sd)
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:Inter,sans-serif}}
body{{background:#fff}}
#wrap{{display:flex;gap:14px;width:100%}}
#pc{{flex:0 0 58%}}
#pw{{position:relative;padding-bottom:56.25%;height:0;overflow:hidden;border-radius:10px;border:1px solid #e5e7eb}}
#pw iframe{{position:absolute;top:0;left:0;width:100%;height:100%}}
#np{{margin-top:8px;padding:7px 12px;background:#fff7ed;border:1px solid #fed7aa;border-radius:8px;font-size:12px;color:#92400e;display:none}}
#sc{{flex:1;height:520px;overflow-y:auto;border:1px solid #e5e7eb;border-radius:10px;padding:8px;background:#fafafa}}
#fb{{display:flex;gap:8px;margin-bottom:9px;align-items:center;flex-wrap:wrap}}
#fb select,#fb input{{border:1px solid #d1d5db;border-radius:6px;padding:4px 8px;font-size:12px;background:#fff;color:#374151}}
#fb label{{font-size:12px;color:#6b7280;font-weight:500}}#fc{{font-size:12px;color:#9ca3af;margin-left:auto}}
.card{{background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px 12px;margin-bottom:7px;cursor:pointer;transition:all .15s}}
.card:hover{{border-color:#f59e0b}}.card.active{{border-color:#f59e0b;background:#fffbeb}}.card.key{{border-left:3px solid #f59e0b}}
.ts{{font-family:monospace;font-size:11px;color:#6b7280;background:#f3f4f6;padding:2px 6px;border-radius:4px}}
.txt{{font-size:12px;color:#374151;line-height:1.5;margin:5px 0 3px}}.iab{{font-size:10px;color:#9ca3af}}
.pb{{display:inline-flex;align-items:center;gap:4px;margin-top:6px;padding:4px 11px;background:#f59e0b;color:#111;border:none;border-radius:6px;font-size:11px;font-weight:600;cursor:pointer}}
</style></head><body>
<div id="wrap">
  <div id="pc">
    <div id="pw"><iframe id="yti" src="https://www.youtube.com/embed/{vid_id}?enablejsapi=1&rel=0&modestbranding=1&playsinline=1" frameborder="0" allowfullscreen></iframe></div>
    <div id="np">▶ Playing from <span id="npt"></span></div>
  </div>
  <div id="sc">
    <div id="fb">
      <label>Sentiment:</label>
      <select id="sf" onchange="render()"><option value="all">All</option><option value="positive">Positive</option><option value="neutral">Neutral</option><option value="negative">Negative</option></select>
      <label>Min fit:</label>
      <input type="range" id="ff" min="0" max="1" step="0.1" value="0" oninput="document.getElementById('fv').textContent=this.value;render()">
      <span id="fv" style="font-size:11px;color:#6b7280">0</span><span id="fc"></span>
    </div>
    <div id="sl"></div>
  </div>
</div>
<script>
var scenes={sj},pl=null,ai=-1;
var tag=document.createElement('script');tag.src="https://www.youtube.com/iframe_api";document.head.appendChild(tag);
function onYouTubeIframeAPIReady(){{pl=new YT.Player('yti',{{events:{{'onReady':function(){{}}}}}}); }}
function seek(t,i,f){{
  ai=i;document.querySelectorAll('.card').forEach(function(c){{c.classList.remove('active')}});
  var c=document.getElementById('c'+i);if(c){{c.classList.add('active');c.scrollIntoView({{behavior:'smooth',block:'nearest'}});}}
  document.getElementById('np').style.display='block';document.getElementById('npt').textContent=f;
  if(pl&&pl.seekTo){{pl.seekTo(t,true);pl.playVideo();}}
  else{{document.getElementById('yti').src="https://www.youtube.com/embed/{vid_id}?enablejsapi=1&autoplay=1&start="+t+"&rel=0&modestbranding=1";}}
}}
function render(){{
  var sf=document.getElementById('sf').value,ff=parseFloat(document.getElementById('ff').value);
  var sl=document.getElementById('sl');sl.innerHTML='';var n=0;
  scenes.forEach(function(s){{
    if(sf!=='all'&&s.sent!==sf)return;if(parseFloat(s.fit)<ff)return;n++;
    var kc=s.is_key?' key':'',ac=s.idx===ai?' active':'',kb=s.is_key?' ⭐':'';
    sl.innerHTML+='<div class="card'+kc+ac+'" id="c'+s.idx+'">'
      +'<div><span class="ts">'+s.start_fmt+' → '+s.end_fmt+'</span> <b style="font-size:12px">'+s.dur+'s'+kb+'</b></div>'
      +'<div class="meta" style="font-size:11px;color:#6b7280">'+s.sent_icon+' '+s.sent+' · 🛡 '+s.safety+' · ⚡ '+s.eng+'</div>'
      +'<div class="txt">'+s.text+(s.text.length>=200?'…':'')+'</div>'
      +'<div class="iab">'+s.iab+' · ad fit '+s.fit+'</div>'
      +'<button class="pb" onclick="seek('+s.start+','+s.idx+',\''+s.start_fmt+'\')">▶ Play</button>'
      +'</div>';
  }});
  document.getElementById('fc').textContent=n+' scenes';
}}
render();
</script></body></html>"""
    st.components.v1.html(html, height=580, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MONETISE  (+ rules engine integration)
# ══════════════════════════════════════════════════════════════════════════════
def page_monetise():
    vm = _active_vm()
    if not vm:
        st.info("No video selected.")
        if st.button("← Library"): st.session_state.page="library"; st.rerun()
        return

    vid_id = vm.video_id
    if vid_id not in st.session_state.ad_markers:
        st.session_state.ad_markers[vid_id] = []

    all_ads_list = [a for a in BUILTIN_ADS + st.session_state.custom_ads if a.get("active", True)]
    ad_opts = [f"{a['emoji']} {a['brand']} — {a['title']}" for a in all_ads_list]

    st.markdown('<div style="font-size:1.8rem;font-weight:800;color:#111827;padding:4px 0 4px">📢 Monetise</div>', unsafe_allow_html=True)
    st.caption(f"**{vm.title}** · {vm.fmt_duration()} · {vm.scene_count} scenes")
    st.divider()

    tab_plan, tab_rules, tab_preview, tab_adlib = st.tabs([
        "🗓️  Ad Plan", "⚙️  Rules Check", "▶️  Live Preview", "📦  Ad Library"
    ])

    with tab_plan:
        left, right = st.columns([3,2], gap="large")

        with right:
            with st.container(border=True):
                st.markdown("**🤖 Scene Match Summary**")
                for scene in vm.scenes[:7]:
                    ad, sim = _best_ad_for_scene(scene)
                    is_key = scene.scene_id in vm.key_scenes
                    sc = "#16a34a" if sim["total"]>0.45 else "#f59e0b" if sim["total"]>0.25 else "#9ca3af"
                    bar = int(sim["total"]*100)
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;padding:4px 0;border-bottom:1px solid #f3f4f6">'
                        f'<code style="font-size:10px;color:#374151;min-width:46px">{scene.start_fmt}</code>'
                        f'{"⭐" if is_key else "  "}'
                        f'<div style="flex:1"><div style="font-size:12px;font-weight:600">{ad["emoji"]} {ad["brand"]}</div>'
                        f'<div style="display:flex;align-items:center;gap:4px;margin-top:2px">'
                        f'<div style="flex:1;height:4px;background:#e5e7eb;border-radius:2px">'
                        f'<div style="width:{bar}%;height:4px;background:{sc};border-radius:2px"></div></div>'
                        f'<span style="font-size:10px;font-weight:700;color:{sc}">{bar}%</span>'
                        f'</div></div></div>', unsafe_allow_html=True)

            st.markdown("")
            with st.container(border=True):
                st.markdown("**⚙️ Auto-generate**")
                n_mid = st.slider("Mid-roll count", 1, min(8,vm.scene_count), min(3,vm.scene_count), key="m_n")
                strat = st.radio("Strategy",["🏆 Best similarity","🔀 Even spacing","⭐ Key moments","🎯 Top opportunities"], key="m_str")
                if st.button("Generate Plan →", type="primary", key="m_gen"):
                    scenes = vm.scenes
                    def opp(s): return s.engagement_score*s.ad_suitability*s.brand_safety.get("safety_score",1)
                    if "similarity" in strat:
                        cands = sorted(scenes,key=lambda s:_best_ad_for_scene(s)[1]["total"]*s.ad_suitability,reverse=True)[:n_mid]
                    elif "Even" in strat:
                        step=max(1,len(scenes)//(n_mid+1))
                        cands=[scenes[i*step] for i in range(1,n_mid+1) if i*step<len(scenes)]
                    elif "Key" in strat:
                        cands=[s for s in scenes if s.scene_id in vm.key_scenes][:n_mid] or sorted(scenes,key=lambda s:s.engagement_score,reverse=True)[:n_mid]
                    else:
                        cands = sorted(scenes,key=opp,reverse=True)[:n_mid]

                    new_markers = []
                    ad0,sim0 = _best_ad_for_scene(scenes[0])
                    new_markers.append({"sec":0,"fmt":"00:00:00","ad":ad0,"mode":"auto","duration":15,"sim":sim0["total"],"reason":f"Pre-roll · {sim0['total']:.0%}"})
                    for sc2 in cands:
                        ad,sim = _best_ad_for_scene(sc2)
                        reason = _iab_str(sc2.iab_categories)+f" · {sim['total']:.0%}"+(" · ⭐" if sc2.scene_id in vm.key_scenes else "")
                        new_markers.append({"sec":sc2.start_sec,"fmt":_fmt_sec(sc2.start_sec),"ad":ad,"mode":"auto","duration":15,"sim":sim["total"],"reason":reason})

                    # Apply rules engine automatically
                    valid, rejected = _apply_ad_rules(new_markers, vm.scenes, vm.duration_sec)
                    st.session_state.ad_markers[vid_id] = valid
                    if rejected:
                        st.warning(f"⚙️ Rules engine removed {len(rejected)} marker(s). See Rules Check tab.")
                    st.success(f"✅ {len(valid)} valid markers"); st.rerun()

            with st.container(border=True):
                st.markdown("**✋ Manual add**")
                new_ts  = st.text_input("Timestamp", placeholder="00:01:30", key="m_mts")
                ad_pick = st.selectbox("Ad", ad_opts, key="m_mad")
                dur_p   = st.number_input("Duration (s)", 5, 60, 15, key="m_dur_new")
                if st.button("➕ Add", key="m_madd"):
                    sec = _parse_ts(new_ts)
                    if sec is not None:
                        if sec not in [m["sec"] for m in st.session_state.ad_markers[vid_id]]:
                            ad_obj = all_ads_list[ad_opts.index(ad_pick)]
                            closest = min(vm.scenes, key=lambda s: abs(s.start_sec-sec))
                            sim = _ad_similarity(ad_obj, closest)
                            st.session_state.ad_markers[vid_id].append({
                                "sec":sec,"fmt":_fmt_sec(sec),"ad":ad_obj,
                                "mode":"manual","duration":int(dur_p),
                                "sim":sim["total"],"reason":"Manual"})
                        st.rerun()
                    else: st.error("Invalid timestamp")

        with left:
            markers = st.session_state.ad_markers[vid_id]
            if not markers:
                st.markdown('<div style="text-align:center;padding:48px 24px;color:#9ca3af;border:2px dashed #e5e7eb;border-radius:12px"><div style="font-size:2rem">📋</div><div style="font-weight:600;margin-top:8px">No markers yet</div></div>', unsafe_allow_html=True)
            else:
                sorted_m = sorted(markers, key=lambda x: x["sec"])
                avg_sim = sum(m["sim"] for m in markers)/len(markers)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Pre-roll",  sum(1 for m in markers if m["sec"]==0))
                c2.metric("Mid-roll",  sum(1 for m in markers if m["sec"]>0))
                c3.metric("Post-roll", 1)
                c4.metric("Avg match", f"{avg_sim:.0%}")

                for i,m in enumerate(sorted_m):
                    tl  = "🟡 Pre-roll" if m["sec"]==0 else "🔴 Mid-roll"
                    sc  = "#16a34a" if m["sim"]>0.45 else "#f59e0b" if m["sim"]>0.25 else "#9ca3af"
                    with st.container(border=True):
                        h1,h2 = st.columns([4,2])
                        with h1:
                            st.markdown(f"**{tl}** · {'🤖' if m['mode']=='auto' else '✋'}")
                            st.text_input("Time", value=m["fmt"], key=f"mt_{i}")
                        with h2:
                            st.markdown(f'<div style="padding-top:28px;text-align:right"><span style="font-size:20px;font-weight:800;color:{sc}">🎯 {m["sim"]:.0%}</span></div>', unsafe_allow_html=True)
                        cur_lbl = f"{m['ad']['emoji']} {m['ad']['brand']} — {m['ad']['title']}"
                        cur_idx = ad_opts.index(cur_lbl) if cur_lbl in ad_opts else 0
                        st.selectbox("Ad", ad_opts, index=cur_idx, key=f"ma_{i}")
                        dc,rc = st.columns([4,1])
                        dc.slider("Duration (s)", 5, 60, int(m.get("duration",15)), key=f"md_{i}")
                        if rc.button("🗑", key=f"mD_{i}"):
                            st.session_state.ad_markers[vid_id] = [x for x in markers if x["sec"]!=m["sec"]]; st.rerun()
                        st.caption(m.get("reason",""))

                if st.button("💾 Save & Preview →", type="primary", key="m_save"):
                    saved = []
                    for i,m in enumerate(sorted_m):
                        raw_ts  = st.session_state.get(f"mt_{i}", m["fmt"])
                        new_sec = _parse_ts(raw_ts) or m["sec"]
                        ad_lbl  = st.session_state.get(f"ma_{i}", f"{m['ad']['emoji']} {m['ad']['brand']} — {m['ad']['title']}")
                        new_ad  = all_ads_list[ad_opts.index(ad_lbl)] if ad_lbl in ad_opts else m["ad"]
                        new_dur = int(st.session_state.get(f"md_{i}", m.get("duration",15)))
                        closest = min(vm.scenes, key=lambda s: abs(s.start_sec-new_sec))
                        new_sim = _ad_similarity(new_ad, closest)["total"]
                        saved.append({**m,"sec":new_sec,"fmt":_fmt_sec(new_sec),"ad":new_ad,"duration":new_dur,"sim":new_sim})
                    st.session_state.ad_markers[vid_id] = saved
                    st.success("✅ Saved")

    # ── NEW: Rules Check tab ───────────────────────────────────────────────
    with tab_rules:
        st.markdown("#### ⚙️ Ad Insertion Rules Engine")
        st.caption("Validates your ad plan against broadcast best-practice rules.")

        with st.expander("📋 Current Rules"):
            c1,c2 = st.columns(2)
            c1.markdown(f"**Min gap between ads:** {AD_RULES['min_gap_sec']}s")
            c1.markdown(f"**No ads in first:** {AD_RULES['no_ad_first_sec']}s (pre-roll excepted)")
            c2.markdown(f"**Min scene duration:** {AD_RULES['min_scene_duration']}s")
            c2.markdown(f"**Block negative scenes for brand-safe ads:** {AD_RULES['block_negative_for_safe_ads']}")

        markers = st.session_state.ad_markers.get(vid_id, [])
        if not markers:
            st.info("Build an ad plan first.")
        else:
            valid, rejected = _apply_ad_rules(markers, vm.scenes, vm.duration_sec)
            col1, col2 = st.columns(2)
            col1.metric("✅ Valid Markers",   len(valid))
            col2.metric("❌ Rule Violations", len(rejected))

            if rejected:
                st.markdown("#### ❌ Violations")
                for item in rejected:
                    m = item["marker"]
                    with st.container(border=True):
                        st.markdown(f"**{m['fmt']}** — {m['ad']['brand']}")
                        for r in item["reasons"]:
                            st.markdown(f"  • {r}")
                if st.button("🔧 Auto-fix — remove violations", type="primary", key="rules_fix"):
                    st.session_state.ad_markers[vid_id] = valid
                    st.success(f"Removed {len(rejected)} violation(s)."); st.rerun()
            else:
                st.success("✅ All markers pass rules validation.")

    with tab_preview:
        markers = st.session_state.ad_markers[vid_id]
        has_mp4 = vid_id in st.session_state.video_b64
        yt_id_m = getattr(vm, "yt_id", None)
        if not markers:
            st.info("Build an ad plan first.")
        elif not has_mp4 and not yt_id_m:
            st.warning("No playable video.")
        else:
            sorted_m = sorted(markers, key=lambda x: x["sec"])
            post_ad, post_sim_d = _best_ad_for_scene(vm.scenes[-1])
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Pre-roll",  sum(1 for m in markers if m["sec"]==0))
            c2.metric("Mid-roll",  sum(1 for m in markers if m["sec"]>0))
            c3.metric("Post-roll", 1)
            c4.metric("Avg match", f"{sum(m['sim'] for m in markers)/max(len(markers),1):.0%}")
            c5.metric("Total ads", len(sorted_m)+1)
            if has_mp4:
                _monetise_player(vm, sorted_m)
            else:
                st.info("ℹ️ Live ad injection works with uploaded MP4.")
        st.divider()
        sorted_m2 = sorted(st.session_state.ad_markers.get(vid_id,[]), key=lambda x: x["sec"])
        post_ad2, post_sim_d2 = _best_ad_for_scene(vm.scenes[-1])
        rows = []
        for m in sorted_m2:
            cl = min(vm.scenes, key=lambda s: abs(s.start_sec-m["sec"]))
            sd = _ad_similarity(m["ad"], cl)
            rows.append({"Time":"Pre-roll" if m["sec"]==0 else m["fmt"],"Type":"pre-roll" if m["sec"]==0 else "mid-roll",
                         "By":"🤖 AI" if m.get("mode")=="auto" else "✋ Manual","Brand":m["ad"]["brand"],"Ad":m["ad"]["title"],
                         "🎯 Match":f"{m['sim']:.0%}","🧠 Semantic":f"{sd.get('vector',0):.0%}","Dur":f"{m.get('duration',15)}s",
                         "Matched":  ", ".join(sd["matched_iab"][:3]) or "—"})
        rows.append({"Time":"Post-roll","Type":"post-roll","By":"🤖 AI","Brand":post_ad2["brand"],"Ad":post_ad2["title"],
                     "🎯 Match":f"{post_sim_d2['total']:.0%}","🧠 Semantic":f"{post_sim_d2.get('vector',0):.0%}",
                     "Dur":"15s","Matched":", ".join(post_sim_d2["matched_iab"][:3]) or "—"})
        if rows: st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_adlib:
        st.markdown("#### Built-in Ads")
        # NEW: active/paused toggle
        for i, ad in enumerate(BUILTIN_ADS):
            c1, c2, c3 = st.columns([1, 5, 1])
            with c1:
                is_active = ad.get("active", True)
                if st.toggle("", value=is_active, key=f"ad_active_{ad['id']}",
                             help="Toggle active/paused"):
                    BUILTIN_ADS[i]["active"] = True
                else:
                    BUILTIN_ADS[i]["active"] = False
            with c2:
                st.markdown(f"{'✅' if ad.get('active',True) else '⏸️'} {ad['emoji']} **{ad['brand']}** — {ad['title']}")
                st.caption("🏷️ " + " · ".join(ad["tags"].split()[:5]))
            with c3:
                st.caption("High" if ad.get("advertiser_suitability")=="High" else "Med")

        st.caption(f"*{sum(1 for a in BUILTIN_ADS if a.get('active',True))}/{len(BUILTIN_ADS)} ads active*")
        st.divider()
        st.markdown("#### ➕ Add Custom Ad")
        if vm.dominant_iab:
            st.info(f"💡 **Video tags to target:** {' · '.join(c['name'] for c in vm.dominant_iab[:5])}")
        with st.container(border=True):
            ua, ub = st.columns(2)
            with ua:
                c_brand = st.text_input("Brand", key="ca_brand")
                c_title = st.text_input("Title", key="ca_title")
                c_hl    = st.text_input("Headline", key="ca_hl")
                c_body  = st.text_area("Body", key="ca_body", height=70)
                c_cta   = st.text_input("CTA", "Learn More", key="ca_cta")
                c_em    = st.text_input("Emoji", "📣", key="ca_emoji")
                c_suit  = st.selectbox("Advertiser Suitability", ["High","Medium","Low"], key="ca_suit")
            with ub:
                st.markdown("**🏷️ Tags**")
                c_tags = st.text_area("Tags", key="ca_tags", height=90, placeholder="arts entertainment, family parenting, news")
                if c_tags.strip():
                    dummy = {"tags":c_tags,"id":"preview","headline":"","body":""}
                    scores = sorted([(s,_ad_similarity(dummy,s)["total"]) for s in vm.scenes],key=lambda x:x[1],reverse=True)[:3]
                    if scores[0][1] > 0:
                        st.markdown("**Live preview:**")
                        for sc2,sim in scores:
                            bar_w = int(sim*100); c = "#16a34a" if sim>0.45 else "#f59e0b" if sim>0.25 else "#9ca3af"
                            st.markdown(f'<div style="display:flex;align-items:center;gap:8px;font-size:12px;margin:3px 0"><code>{sc2.start_fmt}</code><div style="flex:1;height:5px;background:#e5e7eb;border-radius:3px"><div style="width:{bar_w}%;height:5px;background:{c};border-radius:3px"></div></div><b style="color:{c}">{sim:.0%}</b></div>', unsafe_allow_html=True)
                c1_ = st.color_picker("Gradient start","#6366f1",key="ca_c1")
                c2_ = st.color_picker("Gradient end","#8b5cf6",key="ca_c2")
            if st.button("➕ Add to Library", type="primary", key="ca_add"):
                if c_brand and c_title and c_tags.strip():
                    st.session_state.custom_ads.append({
                        "id":f"custom_{len(st.session_state.custom_ads)+1}",
                        "brand":c_brand,"title":c_title,"headline":c_hl or c_title,
                        "body":c_body or f"{c_brand} — {c_title}","cta":c_cta,"emoji":c_em,
                        "bg":f"linear-gradient(135deg,{c1_},{c2_})","tags":c_tags,
                        "advertiser_suitability":c_suit,"active":True})
                    st.success(f"✅ Added {c_brand}"); st.rerun()
                else: st.error("Brand, title and tags required")

        if st.session_state.custom_ads:
            st.divider(); st.markdown("#### Your Custom Ads")
            for i,ad in enumerate(st.session_state.custom_ads):
                top_sc  = max(vm.scenes, key=lambda s: _ad_similarity(ad,s)["total"])
                top_sim = _ad_similarity(ad, top_sc)
                sc = "#16a34a" if top_sim["total"]>0.45 else "#f59e0b" if top_sim["total"]>0.25 else "#9ca3af"
                with st.container(border=True):
                    c1_,c2_,c3_,c4_ = st.columns([1,2,4,1])
                    is_active = ad.get("active",True)
                    if c1_.toggle("", value=is_active, key=f"cad_active_{i}"):
                        st.session_state.custom_ads[i]["active"] = True
                    else:
                        st.session_state.custom_ads[i]["active"] = False
                    c2_.markdown(f"{ad['emoji']} **{ad['brand']}**")
                    c3_.markdown(f'Best: `{top_sc.start_fmt}` — <span style="color:{sc};font-weight:700">{top_sim["total"]:.0%}</span> (🧠 {top_sim.get("vector",0):.0%} semantic)', unsafe_allow_html=True)
                    if c4_.button("🗑", key=f"dca_{i}"): st.session_state.custom_ads.pop(i); st.rerun()


def _monetise_player(vm, markers):
    vid_id = vm.video_id
    b64    = st.session_state.video_b64.get(vid_id, "")
    mime   = st.session_state.video_mime.get(vid_id, "video/mp4")
    post_ad, post_sim_d = _best_ad_for_scene(vm.scenes[-1])
    mjs = _json.dumps([{"sec":m["sec"],"fmt":m["fmt"],"type":"pre-roll" if m["sec"]==0 else "mid-roll",
        "mode":m.get("mode","manual"),"sim":round(m.get("sim",0),3),"duration":m.get("duration",15),
        "reason":m.get("reason",""),"ad_brand":m["ad"]["brand"],"ad_title":m["ad"]["title"],
        "ad_headline":m["ad"]["headline"],"ad_body":m["ad"]["body"],"ad_cta":m["ad"]["cta"],
        "ad_emoji":m["ad"]["emoji"],"ad_bg":m["ad"]["bg"]} for m in markers])
    pjs = _json.dumps({"sec":-1,"fmt":"end","type":"post-roll","mode":"auto",
        "sim":round(post_sim_d["total"],3),"duration":15,"reason":"Best match for final scene",
        "ad_brand":post_ad["brand"],"ad_title":post_ad["title"],"ad_headline":post_ad["headline"],
        "ad_body":post_ad["body"],"ad_cta":post_ad["cta"],"ad_emoji":post_ad["emoji"],"ad_bg":post_ad["bg"]})
    sjs = _json.dumps([{"sec":s.start_sec,"fmt":s.start_fmt,"key":s.scene_id in vm.key_scenes,
                        "label":s.text[:28].replace('"','')} for s in vm.scenes])
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<style>
*{{box-sizing:border-box;margin:0;padding:0;font-family:Inter,sans-serif}}
body{{background:#f9fafb;padding:8px}}
#wrap{{background:#fff;border-radius:14px;border:1px solid #e5e7eb;box-shadow:0 4px 20px rgba(0,0,0,.08);overflow:hidden}}
#vw{{position:relative;background:#000;aspect-ratio:16/9;max-height:440px}}
video{{width:100%;height:100%;display:block;object-fit:contain}}
#ov{{display:none;position:absolute;inset:0;z-index:30;background:rgba(0,0,0,.82);align-items:center;justify-content:center}}
#ac{{width:min(92%,520px);border-radius:20px;padding:28px 34px;color:#fff;text-align:center;box-shadow:0 20px 60px rgba(0,0,0,.5);position:relative}}
.b{{position:absolute;padding:4px 11px;border-radius:10px;font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.1em;background:rgba(0,0,0,.4)}}
#bt{{top:13px;left:13px}}#bm{{top:13px;left:105px}}#bs{{top:13px;right:13px;cursor:pointer}}#bsim{{bottom:13px;right:13px}}
#acd{{font-size:13px;opacity:.7;margin-top:12px;margin-bottom:8px}}
#aem{{font-size:3.2rem;margin-bottom:5px}}#abr{{font-size:11px;font-weight:700;text-transform:uppercase;opacity:.75;margin-bottom:3px}}
#ahl{{font-size:22px;font-weight:800;line-height:1.2;margin-bottom:8px}}#abo{{font-size:14px;opacity:.88;line-height:1.55;margin-bottom:20px}}
#act{{display:inline-block;background:rgba(255,255,255,.18);border:2px solid rgba(255,255,255,.5);color:#fff;padding:10px 28px;border-radius:30px;font-weight:700;cursor:pointer;font-size:14px}}
#act:hover{{background:rgba(255,255,255,.32)}}#are{{font-size:11px;opacity:.6;margin-top:12px}}
#cb{{padding:10px 14px 5px;background:#fff;border-top:1px solid #f3f4f6}}
#pw{{position:relative;height:8px;background:#e5e7eb;border-radius:4px;cursor:pointer;margin-bottom:9px}}
#pf{{height:100%;background:#f59e0b;border-radius:4px;width:0%;pointer-events:none}}
.pm{{position:absolute;top:-5px;width:18px;height:18px;border-radius:50%;border:2px solid #fff;transform:translateX(-50%);cursor:pointer;z-index:5;box-shadow:0 2px 5px rgba(0,0,0,.2)}}
.pml{{position:absolute;top:16px;transform:translateX(-50%);font-size:9px;white-space:nowrap;font-weight:600;pointer-events:none}}
#ctrl{{display:flex;align-items:center;gap:10px}}
#pb{{width:36px;height:36px;border-radius:50%;background:#f59e0b;border:none;cursor:pointer;font-size:14px;color:#111;display:flex;align-items:center;justify-content:center;flex-shrink:0}}
#td{{font-size:12px;color:#374151;font-variant-numeric:tabular-nums}}
#vol{{display:flex;align-items:center;gap:6px;margin-left:auto}}
#vol input{{width:70px;accent-color:#f59e0b}}#vi{{cursor:pointer;font-size:14px}}
#sb{{padding:4px 14px 7px;font-size:11px;color:#9ca3af;background:#fff}}
#sp{{border-top:1px solid #f3f4f6;padding:8px 14px 10px;background:#fafafa}}
#sp-list{{display:flex;flex-wrap:wrap;gap:5px;margin-top:4px}}
.chip{{padding:3px 9px;border-radius:12px;font-size:11px;font-weight:500;cursor:pointer;border:1px solid #e5e7eb;background:#fff;color:#374151;white-space:nowrap}}
.chip:hover,.chip.act{{background:#fff7ed;border-color:#f59e0b;color:#92400e}}.chip.key{{border-left:3px solid #f59e0b}}.chip.adchip{{background:#fef3c7;border-color:#fcd34d;color:#92400e;font-weight:600}}
</style></head><body>
<div id="wrap">
  <div id="vw">
    <video id="vid" preload="auto" playsinline><source src="data:{mime};base64,{b64}" type="{mime}"></video>
    <div id="ov"><div id="ac">
      <div class="b" id="bt">mid-roll</div><div class="b" id="bm">🤖 auto</div>
      <div class="b" id="bs" onclick="skip()">✕ Skip</div><div class="b" id="bsim">🎯 —%</div>
      <div id="acd"></div><div id="aem">🎬</div><div id="abr">Brand</div>
      <div id="ahl">Headline</div><div id="abo">Body</div>
      <div id="act" onclick="skip()">CTA</div><div id="are"></div>
    </div></div>
  </div>
  <div id="cb">
    <div id="pw" onclick="seekBar(event)"><div id="pf"></div></div>
    <div id="ctrl">
      <button id="pb" onclick="toggle()">▶</button><span id="td">0:00 / 0:00</span>
      <div id="vol"><span id="vi" onclick="mute()">🔊</span><input type="range" min="0" max="1" step=".05" value="1" oninput="VID.volume=this.value"></div>
    </div>
  </div>
  <div id="sb">Ready · {len(markers)} ad markers · press ▶</div>
  <div id="sp"><div style="font-size:11px;font-weight:600;color:#374151;margin-bottom:4px">📢 Ad markers · ⭐ key scenes</div><div id="sp-list"></div></div>
</div>
<script>
var VID=document.getElementById('vid'),M={mjs},P={pjs},SC={sjs};
var shown={{}},cdt=null,adOn=false;
VID.addEventListener('loadedmetadata',function(){{
  var dur=VID.duration,pw=document.getElementById('pw');
  M.forEach(function(m){{
    if(m.sec<=0)return;
    var pct=(m.sec/dur)*100;
    var d=document.createElement('div');d.className='pm';d.style.left=pct+'%';
    d.style.background=m.mode==='auto'?'#f59e0b':'#ef4444';d.title=m.ad_brand+' · '+Math.round(m.sim*100)+'%';
    d.onclick=function(e){{e.stopPropagation();VID.currentTime=Math.max(0,m.sec-.5);VID.play();}};
    var l=document.createElement('div');l.className='pml';l.style.left=pct+'%';l.style.color='#d97706';l.textContent='📢'+m.fmt.slice(3);
    pw.appendChild(d);pw.appendChild(l);
  }});
  var sl=document.getElementById('sp-list');
  M.forEach(function(m){{var c=document.createElement('span');c.className='chip adchip';c.textContent=m.ad_emoji+' '+(m.sec===0?'Pre':m.fmt.slice(3))+' '+m.ad_brand;c.onclick=function(){{VID.currentTime=Math.max(0,m.sec-.5);VID.play();}};sl.appendChild(c);}});
  SC.forEach(function(s){{var c=document.createElement('span');c.className='chip'+(s.key?' key':'');c.id='sc'+s.sec;c.textContent=(s.key?'⭐':'')+s.fmt+' '+s.label;c.onclick=function(){{VID.currentTime=s.sec;VID.play();}};sl.appendChild(c);}});
}});
VID.addEventListener('timeupdate',function(){{
  var t=VID.currentTime,d=VID.duration||1;
  document.getElementById('pf').style.width=(t/d*100)+'%';
  var m=Math.floor(t/60),s=Math.floor(t%60),dm=Math.floor(d/60),ds=Math.floor(d%60);
  document.getElementById('td').textContent=m+':'+(s<10?'0':'')+s+' / '+dm+':'+(ds<10?'0':'')+ds;
  document.getElementById('pb').textContent=VID.paused?'▶':'⏸';
  SC.forEach(function(sc){{var el=document.getElementById('sc'+sc.sec);if(el)el.classList.toggle('act',t>=sc.sec&&t<sc.sec+20);}});
  if(adOn)return;
  M.forEach(function(m){{var k=m.ad_brand+'_'+m.sec;if(!shown[k]&&t>=m.sec&&m.sec>=0){{shown[k]=true;showAd(m);}}}});
}});
VID.addEventListener('ended',function(){{if(!shown.post){{shown.post=true;showAd(P);}}}});
VID.addEventListener('play',function onFirst(){{var pre=M.find(function(m){{return m.sec===0;}});if(pre&&!shown[pre.ad_brand+'_0']){{shown[pre.ad_brand+'_0']=true;VID.pause();showAd(pre);}}VID.removeEventListener('play',onFirst);}},{{once:true}});
function showAd(m){{
  adOn=true;VID.pause();document.getElementById('ov').style.display='flex';
  document.getElementById('ac').style.background=m.ad_bg;
  document.getElementById('bt').textContent=m.type;document.getElementById('bm').textContent=m.mode==='auto'?'🤖 AI':'✋ Manual';
  document.getElementById('bs').style.visibility='hidden';document.getElementById('bsim').textContent='🎯 '+Math.round((m.sim||0)*100)+'%';
  document.getElementById('aem').textContent=m.ad_emoji;document.getElementById('abr').textContent=m.ad_brand;
  document.getElementById('ahl').textContent=m.ad_headline;document.getElementById('abo').textContent=m.ad_body;
  document.getElementById('act').textContent=m.ad_cta;document.getElementById('are').textContent=m.reason||'';
  var s=Math.min(5,m.duration||15);document.getElementById('acd').textContent='Skippable in '+s+'s';
  cdt=setInterval(function(){{s--;if(s<=0){{clearInterval(cdt);document.getElementById('acd').textContent='';document.getElementById('bs').style.visibility='visible';}}else document.getElementById('acd').textContent='Skippable in '+s+'s';}},1000);
}}
function skip(){{clearInterval(cdt);adOn=false;document.getElementById('ov').style.display='none';if(!VID.ended)VID.play();}}
function toggle(){{VID.paused?VID.play():VID.pause();}}
function seekBar(e){{var r=document.getElementById('pw').getBoundingClientRect();VID.currentTime=((e.clientX-r.left)/r.width)*(VID.duration||0);}}
function mute(){{VID.muted=!VID.muted;document.getElementById('vi').textContent=VID.muted?'🔇':'🔊';}}
document.addEventListener('keydown',function(e){{
  if(e.target.tagName==='INPUT'||e.target.tagName==='TEXTAREA')return;
  if(e.code==='Space'){{e.preventDefault();if(!adOn)toggle();}}
  else if(e.code==='ArrowLeft'){{e.preventDefault();VID.currentTime=Math.max(0,VID.currentTime-5);}}
  else if(e.code==='ArrowRight'){{e.preventDefault();VID.currentTime=Math.min(VID.duration||0,VID.currentTime+5);}}
  else if(e.code==='KeyM'){{mute();}}
}});
</script></body></html>"""
    st.components.v1.html(html, height=680, scrolling=False)


# ══════════════════════════════════════════════════════════════════════════════
# NEW: Publisher Ad Report Generator
# ══════════════════════════════════════════════════════════════════════════════
def _generate_publisher_report(vm: VideoMetadata) -> str:
    """Generate a self-contained HTML report publishers can send to advertisers."""
    meta    = st.session_state.ai_meta.get(vm.video_id, {})
    markers = st.session_state.ad_markers.get(vm.video_id, [])
    mscore  = meta.get("monetisation_score","—")
    yt_id   = getattr(vm,"yt_id",None)
    thumb   = f"https://img.youtube.com/vi/{yt_id}/hqdefault.jpg" if yt_id else ""

    def _opp_score(s):
        return s.engagement_score * s.ad_suitability * s.brand_safety.get("safety_score",1.0)

    top_moments = sorted(vm.scenes, key=_opp_score, reverse=True)[:5]
    avg_safe = sum(s.brand_safety.get("safety_score",1) for s in vm.scenes) / max(vm.scene_count,1)
    avg_eng  = sum(s.engagement_score for s in vm.scenes) / max(vm.scene_count,1)

    moments_html = ""
    for rank, scene in enumerate(top_moments):
        ad, sim = _best_ad_for_scene(scene)
        cpm_lo, cpm_hi = _est_cpm(scene)
        opp = _opp_score(scene)
        moments_html += f"""
        <tr>
          <td style="padding:10px 8px;font-weight:700;color:#f59e0b">#{rank+1}</td>
          <td style="padding:10px 8px;font-family:monospace;color:#374151">{scene.start_fmt}</td>
          <td style="padding:10px 8px;color:#374151">{scene.text[:80]}{'…' if len(scene.text)>80 else ''}</td>
          <td style="padding:10px 8px;text-align:center">
            <span style="background:#dcfce7;color:#166534;padding:2px 8px;border-radius:10px;font-size:12px;font-weight:700">{opp:.0%}</span>
          </td>
          <td style="padding:10px 8px;text-align:center;color:#6b7280">{ad['emoji']} {ad['brand']}</td>
          <td style="padding:10px 8px;text-align:center;font-weight:700;color:#0f172a">${cpm_lo:.0f}–${cpm_hi:.0f}</td>
        </tr>"""

    ad_schedule_html = ""
    for m in sorted(markers, key=lambda x: x["sec"]):
        ad_schedule_html += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:8px 0;border-bottom:1px solid #f3f4f6">
          <span style="font-family:monospace;font-size:12px;color:#6b7280;min-width:60px">{'Pre-roll' if m['sec']==0 else m['fmt']}</span>
          <span style="font-size:18px">{m['ad']['emoji']}</span>
          <div>
            <div style="font-weight:600;font-size:13px">{m['ad']['brand']} — {m['ad']['title']}</div>
            <div style="font-size:11px;color:#9ca3af">{m.get('duration',15)}s · {m['sim']:.0%} contextual match</div>
          </div>
        </div>"""

    report_html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Ad Opportunity Report — {vm.title}</title>
<style>
  body{{font-family:Inter,Arial,sans-serif;margin:0;padding:0;background:#f8fafc;color:#111827}}
  .wrap{{max-width:900px;margin:0 auto;padding:32px 24px}}
  .header{{background:linear-gradient(135deg,#1e293b,#0f172a);color:#fff;border-radius:16px;padding:32px;margin-bottom:24px}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
  .kpi{{background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px;text-align:center}}
  .kpi .val{{font-size:1.6rem;font-weight:800;color:#0f172a}}
  .kpi .lbl{{font-size:.65rem;text-transform:uppercase;letter-spacing:.06em;color:#94a3b8;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;background:#fff;border-radius:12px;overflow:hidden;border:1px solid #e2e8f0}}
  th{{background:#f8fafc;padding:10px 8px;text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:#6b7280;border-bottom:1px solid #e2e8f0}}
  tr:nth-child(even){{background:#fafafa}}
  .section{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px;margin-bottom:20px}}
  .section h2{{font-size:1rem;font-weight:700;margin:0 0 12px;color:#0f172a}}
  .disclaimer{{font-size:11px;color:#9ca3af;margin-top:24px;padding:12px;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0}}
  .mscore{{font-size:2.2rem;font-weight:800;color:{'#16a34a' if isinstance(mscore,int) and mscore>=70 else '#d97706' if isinstance(mscore,int) and mscore>=40 else '#dc2626'}}}
</style>
</head><body><div class="wrap">
  <div class="header">
    {'<img src="'+thumb+'" style="width:100%;border-radius:8px;margin-bottom:16px" />' if thumb else ''}
    <div style="font-size:.75rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:#94a3b8;margin-bottom:6px">Ad Opportunity Report</div>
    <div style="font-size:1.8rem;font-weight:800;line-height:1.2;margin-bottom:8px">{vm.title}</div>
    <div style="font-size:.9rem;color:#94a3b8">{vm.fmt_duration()} · {vm.scene_count} scenes · {meta.get('primary_genre','—')} · {meta.get('content_rating','—')}</div>
    <div style="margin-top:12px;font-size:.9rem;color:#cbd5e1">{meta.get('summary','')}</div>
  </div>

  <div class="kpi-grid">
    <div class="kpi"><div class="val mscore">{mscore}/100</div><div class="lbl">Monetisation Score</div></div>
    <div class="kpi"><div class="val">{avg_safe:.0%}</div><div class="lbl">Brand Safe Scenes</div></div>
    <div class="kpi"><div class="val">{avg_eng:.2f}</div><div class="lbl">Avg Engagement</div></div>
    <div class="kpi"><div class="val">{len(markers)}</div><div class="lbl">Ad Slots Available</div></div>
  </div>

  <div class="section">
    <h2>🎯 Top 5 Ad Moments</h2>
    <table>
      <thead><tr>
        <th>Rank</th><th>Timestamp</th><th>Scene Context</th>
        <th style="text-align:center">Opp Score</th>
        <th style="text-align:center">Best Ad Match</th>
        <th style="text-align:center">Est. CPM (USD)</th>
      </tr></thead>
      <tbody>{moments_html}</tbody>
    </table>
  </div>

  {'<div class="section"><h2>📋 Proposed Ad Schedule</h2>' + ad_schedule_html + '</div>' if markers else ''}

  <div class="section">
    <h2>🏷️ Content Profile</h2>
    <div style="display:flex;flex-wrap:wrap;gap:8px">
      {''.join(f'<span style="background:#fef3c7;border:1px solid #fcd34d;color:#92400e;padding:4px 12px;border-radius:14px;font-size:12px">{c["name"]}</span>' for c in vm.dominant_iab[:8])}
    </div>
    <div style="margin-top:12px;font-size:13px;color:#374151">
      <b>Target Audience:</b> {meta.get('target_audience','—')} &nbsp;·&nbsp;
      <b>Mood:</b> {meta.get('mood','—')} &nbsp;·&nbsp;
      <b>Advertiser Suitability:</b> {meta.get('advertiser_suitability','—')}
    </div>
  </div>

  <div class="disclaimer">
    ⚠️ <b>Disclaimer:</b> CPM estimates are indicative benchmarks based on IAB category industry averages.
    Actual rates vary by geography, ad format, audience demographics, and buyer. Opportunity scores are
    computed algorithmically from engagement, brand safety, and contextual relevance signals.
    Generated by Semantix Video Intelligence Platform.
  </div>
</div></body></html>"""
    return report_html


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — INSIGHTS  (+ multi-video search hero + report export)
# ══════════════════════════════════════════════════════════════════════════════
def page_insights():
    st.markdown('<div style="font-size:1.8rem;font-weight:800;color:#111827;padding:4px 0 4px">📊 Insights</div>', unsafe_allow_html=True)
    st.caption("Cross-video analytics · multi-video search · publisher reports")
    st.divider()

    if not st.session_state.videos:
        st.info("No videos yet."); return

    all_s = [s for vm in st.session_state.videos.values() for s in vm.scenes]
    vms   = list(st.session_state.videos.values())
    n_s   = len(all_s)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Videos",         len(vms))
    c2.metric("Total Scenes",   n_s)
    c3.metric("Avg Engagement", f"{sum(s.engagement_score for s in all_s)/max(n_s,1):.2f}")
    c4.metric("Avg Brand Safe", f"{sum(s.brand_safety.get('safety_score',1) for s in all_s)/max(n_s,1):.0%}")
    c5.metric("Avg Ad Fit",     f"{sum(s.ad_suitability for s in all_s)/max(n_s,1):.0%}")
    st.divider()

    # Tab order: Library Search first for Advertiser persona
    t_search, t1, t2, t3, t4, t5 = st.tabs([
        "🔍  Library Search", "🏷️  Content", "⚡  Engagement",
        "📢  Ad Intelligence", "📹  Compare", "📥  Export & Reports"
    ])

    # ── NEW HERO FEATURE: Multi-video semantic library search ──────────────
    with t_search:
        st.markdown("#### 🔍 Search Across Your Entire Video Library")
        st.caption("Find relevant scenes, moments, and ad placements across all videos simultaneously using semantic AI search.")
        se = st.session_state.search_engine
        lib_q = st.text_input("Search your library",
            placeholder="e.g. 'outdoor adventure', 'expert explaining technology', 'emotional family moment'",
            key="lib_search_q")

        col1, col2, col3 = st.columns(3)
        lib_k      = col1.select_slider("Max results", [5,10,20,50], value=10, key="lib_k")
        lib_safety = col2.selectbox("Brand safety", ["Any","Moderate (50%+)","Strict (80%+)"], key="lib_safe_f")
        lib_sent   = col3.multiselect("Sentiment", ["positive","neutral","negative"],
                                      default=["positive","neutral","negative"], key="lib_sent_f")

        if lib_q and len(st.session_state.videos) > 0:
            smap = {"Any":0.0,"Moderate (50%+)":0.5,"Strict (80%+)":0.8}
            with st.spinner(f"Searching {n_s} scenes across {len(vms)} video(s)…"):
                results = se.search(lib_q, top_k=lib_k, diversify=True,
                                    min_safety=smap[lib_safety], expand=True)
            results = [r for r in results if r.scene.sentiment.get("label","neutral") in lib_sent]
            if not results:
                st.warning("No matches — try broader search terms.")
            else:
                st.success(f"**{len(results)} scenes** found across **{len({r.scene.video_id for r in results})} video(s)**")
                # Group by video
                from collections import defaultdict
                by_video = defaultdict(list)
                for r in results:
                    by_video[r.scene.video_id].append(r)

                for vid_id_r, vid_results in by_video.items():
                    vm_r = st.session_state.videos.get(vid_id_r)
                    if not vm_r: continue
                    mscore = st.session_state.ai_meta.get(vid_id_r,{}).get("monetisation_score","—")
                    st.markdown(f"**📹 {vm_r.title}** · {len(vid_results)} match(es) · 💰 {mscore}/100")
                    for r in vid_results:
                        _scene_card(r.scene, vm_r, score=r.score,
                                    yt_id=getattr(vm_r,"yt_id",None),
                                    ad_match=_best_ad_for_scene(r.scene))
        elif not lib_q:
            st.markdown("##### 💡 Search ideas:")
            example_searches = [
                "🏃 action and sport", "💡 educational explainer", "😂 comedy and humor",
                "🌍 travel and outdoors", "💼 business and professional", "❤️ emotional moments",
                "🎵 music and performance", "🍕 food and cooking", "🚗 automotive"
            ]
            cols = st.columns(3)
            for i, ex in enumerate(example_searches):
                if cols[i%3].button(ex, key=f"lib_ex_{i}"):
                    st.session_state["lib_search_q"] = ex.split(" ",1)[1]
                    st.rerun()

    with t1:
        c1,c2 = st.columns(2)
        with c1:
            labels = [s.sentiment.get("label","neutral") for s in all_s]
            lc     = {l:labels.count(l) for l in set(labels)}
            cmap   = {"positive":"#34d399","neutral":"#60a5fa","negative":"#f87171"}
            pl     = list(lc.keys())
            fig = go.Figure(go.Pie(values=list(lc.values()),labels=pl,hole=0.55,
                marker=dict(colors=[cmap.get(l,"#9ca3af") for l in pl])))
            fig.update_layout(**PT,height=280,title=dict(text="Sentiment Mix",font=dict(size=13,color=_TEXT)),showlegend=True,legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            iab_c = {}
            for s in all_s:
                for cat in s.iab_categories[:1]: iab_c[cat["name"]] = iab_c.get(cat["name"],0)+1
            top_iab = sorted(iab_c.items(),key=lambda x:x[1],reverse=True)[:12]
            if top_iab:
                names = [k for k,_ in top_iab]; vals = [v for _,v in top_iab]
                fig2 = go.Figure(go.Bar(x=vals,y=names,orientation="h",marker_color=_AMBER))
                fig2.update_layout(**PT,height=280,title=dict(text="Top IAB Categories",font=dict(size=13,color=_TEXT)),xaxis=dict(**_XA,title="Scenes"),yaxis=dict(**_YA))
                st.plotly_chart(fig2, use_container_width=True)

        # NEW: Monetisation score leaderboard
        st.markdown("#### 💰 Monetisation Score Leaderboard")
        mscore_rows = []
        for vm in vms:
            meta = st.session_state.ai_meta.get(vm.video_id,{})
            mscore = meta.get("monetisation_score","—")
            mscore_rows.append({"Video":vm.title[:40],"Score":mscore,"Genre":meta.get("primary_genre","—"),
                                 "Rating":meta.get("content_rating","—"),"Scenes":vm.scene_count,
                                 "Duration":vm.fmt_duration(),"Arc":vm.narrative_structure})
        mscore_rows.sort(key=lambda r: r["Score"] if isinstance(r["Score"],int) else 0, reverse=True)
        st.dataframe(pd.DataFrame(mscore_rows), use_container_width=True, hide_index=True)

    with t2:
        eng_vals = [s.engagement_score for s in all_s]
        fig3 = go.Figure(go.Histogram(x=eng_vals,nbinsx=20,marker_color=_AMBER,opacity=0.85))
        fig3.update_layout(**PT,height=260,title=dict(text="Engagement Distribution",font=dict(size=13,color=_TEXT)),xaxis=dict(**_XA,title="Engagement Score"),yaxis=dict(**_YA,title="Scenes"))
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("#### 🔥 Top 10 Most Engaging Scenes")
        for s in sorted(all_s,key=lambda s:s.engagement_score,reverse=True)[:10]:
            _vm2 = st.session_state.videos.get(s.video_id)
            if _vm2 and len(vms)>1: st.caption(f"📹 {_vm2.title[:50]}")
            _scene_card(s, _vm2 or vms[0], yt_id=getattr(_vm2,"yt_id",None) if _vm2 else None)

    with t3:
        st.markdown("#### Best Ad Opportunities Across All Videos")
        def _opp(s): return s.ad_suitability*s.engagement_score*s.brand_safety.get("safety_score",1)
        top_ad = sorted(all_s,key=_opp,reverse=True)[:12]
        rows = []
        for s in top_ad:
            _vm2 = st.session_state.videos.get(s.video_id, vms[0])
            ad,sim = _best_ad_for_scene(s)
            rows.append({"Video":_vm2.title[:28],"Time":s.start_fmt,"Key":"⭐" if s.scene_id in _vm2.key_scenes else "",
                         "Opp Score":f"{_opp(s):.0%}","Ad Fit":f"{s.ad_suitability:.0%}",
                         "Engagement":f"{s.engagement_score:.2f}","Safety":f"{s.brand_safety.get('safety_score',1):.0%}",
                         "Best Ad":f"{ad['emoji']} {ad['brand']}","Match":f"{sim['total']:.0%}",
                         "🧠 Semantic":f"{sim.get('vector',0):.0%}","IAB":_iab_str(s.iab_categories)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(CPM_DISCLAIMER)

    with t4:
        if len(vms) < 2:
            st.info("Add at least 2 videos to compare.")
        else:
            rows2 = [{"Title":vm.title[:35],"Score":st.session_state.ai_meta.get(vm.video_id,{}).get("monetisation_score","—"),
                      "Duration":vm.fmt_duration(),"Scenes":vm.scene_count,"Key":len(vm.key_scenes),
                      "Avg Eng":f"{sum(s.engagement_score for s in vm.scenes)/max(vm.scene_count,1):.2f}",
                      "Brand Safe":f"{sum(1 for s in vm.scenes if s.brand_safety.get('safety_score',1)>=0.7)/max(vm.scene_count,1):.0%}",
                      "Ad Fit":f"{sum(s.ad_suitability for s in vm.scenes)/max(vm.scene_count,1):.0%}",
                      "Arc":vm.narrative_structure} for vm in vms]
            st.dataframe(pd.DataFrame(rows2), use_container_width=True, hide_index=True)
            fig5 = go.Figure()
            metrics = ["Engagement","Ad Fit","Brand Safety","Key Density"]
            for vm in vms[:5]:
                s_all = vm.scenes
                vals = [sum(s.engagement_score for s in s_all)/max(len(s_all),1),
                        sum(s.ad_suitability for s in s_all)/max(len(s_all),1),
                        sum(s.brand_safety.get("safety_score",1) for s in s_all)/max(len(s_all),1),
                        len(vm.key_scenes)/max(vm.scene_count,1)]
                fig5.add_trace(go.Scatterpolar(r=vals+[vals[0]],theta=metrics+[metrics[0]],
                    name=vm.title[:20],fill="toself",opacity=0.6))
            fig5.update_layout(**PT,height=340,polar=dict(radialaxis=dict(visible=True,range=[0,1],gridcolor=_GRID)),showlegend=True)
            st.plotly_chart(fig5, use_container_width=True)

    with t5:
        st.markdown("#### 📥 Export Data & Publisher Reports")

        # NEW: Publisher report section
        st.markdown("##### 📄 Publisher Ad Report")
        st.caption("Generate a polished HTML report to send to advertisers — includes thumbnail, top ad moments, CPM estimates, and content profile.")
        report_vm = st.selectbox("Select video for report", vms, format_func=lambda v: v.title, key="report_vm_sel")
        col1_r, col2_r = st.columns(2)
        if col1_r.button("👁 Preview Report", key="preview_report"):
            st.session_state["_report_preview"] = _generate_publisher_report(report_vm)
            st.rerun()
        if "_report_preview" in st.session_state:
            st.components.v1.html(st.session_state["_report_preview"], height=600, scrolling=True)
        report_html = _generate_publisher_report(report_vm)
        col2_r.download_button(
            "📥 Download HTML Report",
            report_html.encode("utf-8"),
            f"semantix_report_{report_vm.video_id[:8]}.html",
            "text/html",
            use_container_width=True,
            key="dl_report")

        st.divider()
        st.markdown("##### CSV Exports")
        exp1, exp2, exp3 = st.columns(3)

        all_scene_rows = []
        for vm in vms:
            for s in vm.scenes:
                cpm_lo, cpm_hi = _est_cpm(s)
                ad, sim = _best_ad_for_scene(s)
                all_scene_rows.append({
                    "Video":vm.title,"Scene ID":s.scene_id,"Start":s.start_fmt,"End":s.end_fmt,
                    "Duration (s)":int(s.duration_sec),"Sentiment":s.sentiment.get("label","neutral"),
                    "Engagement":round(s.engagement_score,3),"Ad Fit":round(s.ad_suitability,3),
                    "Brand Safety":round(s.brand_safety.get("safety_score",1),3),
                    "IAB Category":s.iab_categories[0]["name"] if s.iab_categories else "",
                    "Key Moment":s.scene_id in vm.key_scenes,
                    "Best Ad Brand":ad["brand"],"Ad Match":round(sim["total"],3),
                    "Semantic Score":round(sim.get("vector",0),3),
                    "Est CPM Low ($)":cpm_lo,"Est CPM High ($)":cpm_hi,
                    "Text Preview":s.text[:100],
                })
        df_scenes = pd.DataFrame(all_scene_rows)
        exp1.download_button("📄 All Scenes CSV", df_scenes.to_csv(index=False).encode(),
            "semantix_scenes.csv","text/csv",use_container_width=True,key="dl_scenes")
        exp1.caption(f"{len(all_scene_rows)} rows · includes semantic match scores")

        opp_rows = []
        for vm in vms:
            def _opp_e(s): return s.ad_suitability*s.engagement_score*s.brand_safety.get("safety_score",1)
            for s in sorted(vm.scenes,key=_opp_e,reverse=True)[:10]:
                cpm_lo,cpm_hi = _est_cpm(s)
                ad,sim = _best_ad_for_scene(s)
                opp_rows.append({"Video":vm.title,"Timestamp":s.start_fmt,
                    "Opportunity Score":round(_opp_e(s),3),"Key Moment":s.scene_id in vm.key_scenes,
                    "Best Ad":ad["brand"],"Ad Match":round(sim["total"],3),
                    "Semantic":round(sim.get("vector",0),3),
                    "Matched IAB":", ".join(sim["matched_iab"][:3]),
                    "Est CPM Low ($)":cpm_lo,"Est CPM High ($)":cpm_hi,
                    "IAB":s.iab_categories[0]["name"] if s.iab_categories else ""})
        df_opps = pd.DataFrame(opp_rows)
        exp2.download_button("🎯 Ad Opportunities CSV", df_opps.to_csv(index=False).encode(),
            "semantix_opportunities.csv","text/csv",use_container_width=True,key="dl_opps")
        exp2.caption(f"{len(opp_rows)} rows · top 10 per video")

        sched_rows = []
        for vid_id, mrks in st.session_state.ad_markers.items():
            vm_s = st.session_state.videos.get(vid_id)
            if not vm_s or not mrks: continue
            for m in sorted(mrks, key=lambda x: x["sec"]):
                sched_rows.append({"Video":vm_s.title,"Type":"pre-roll" if m["sec"]==0 else "mid-roll",
                    "Timestamp":m["fmt"],"Brand":m["ad"]["brand"],"Ad Title":m["ad"]["title"],
                    "Duration (s)":m.get("duration",15),"Match Score":round(m.get("sim",0),3),
                    "Added By":m.get("mode","manual")})
        if sched_rows:
            df_sched = pd.DataFrame(sched_rows)
            exp3.download_button("📋 Ad Schedule CSV", df_sched.to_csv(index=False).encode(),
                "semantix_schedule.csv","text/csv",use_container_width=True,key="dl_sched")
            exp3.caption(f"{len(sched_rows)} markers")
        else:
            exp3.info("Build an ad plan in Monetise first")

        st.caption(CPM_DISCLAIMER)


# ══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════════
pages = {
    "library":  page_library,
    "analyse":  page_analyse,
    "monetise": page_monetise,
    "insights": page_insights,
}
pages.get(st.session_state.page, page_library)()
