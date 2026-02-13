import streamlit as st
import pysrt
import webvtt
import uuid
import os
import tempfile
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from keybert import KeyBERT
import chromadb
from chromadb.config import Settings
from mistralai import Mistral
from typing import List, Dict, Tuple, Optional
import json
import time
from collections import OrderedDict
from datetime import datetime

# ------------------------------------------------------------------
# PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Project Cairo – Semantic Video Search",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# SESSION STATE INITIALISATION
# ------------------------------------------------------------------
_DEFAULT_STATE = {
    "theme": "dark",
    "last_query": "",
    "selected_result": None,
    "show_advanced": False,
    "collection": None,
    "enriched_metadata": {},
    "full_texts": {},
    "processed_hashes": set(),
    # 🚀 ENHANCEMENT: Search history + cache
    "search_history": [],          # list of (query, timestamp)
    "query_cache": {},            # {rewritten_query: original_query}
    "result_cache": {},          # {query: (results, timestamp)}
    "filter_filename": None,     # active filename filter
    "result_offset": 0,         # for load more
}
for key, val in _DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ------------------------------------------------------------------
# SECRETS HANDLING (Mistral AI – optional, app works without it)
# ------------------------------------------------------------------
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")

# ------------------------------------------------------------------
# 🚀 ENHANCEMENT: Glassmorphism CSS + smoother animations
# ------------------------------------------------------------------
def get_theme_css(theme: str) -> str:
    bg_main = "#0a0c10" if theme == "dark" else "#ffffff"
    bg_sidebar = "rgba(20, 22, 27, 0.95)" if theme == "dark" else "rgba(248, 250, 252, 0.95)"
    text_primary = "#fafafa" if theme == "dark" else "#1e293b"
    text_secondary = "#a0aec0" if theme == "dark" else "#64748b"
    border_color = "rgba(45, 50, 61, 0.5)" if theme == "dark" else "rgba(226, 232, 240, 0.8)"
    accent = "#4a6fa5" if theme == "dark" else "#3b82f6"
    card_bg = "rgba(26, 28, 36, 0.7)" if theme == "dark" else "rgba(255, 255, 255, 0.7)"
    tag_bg = "rgba(74, 111, 165, 0.2)" if theme == "dark" else "rgba(59, 130, 246, 0.1)"
    tag_text = text_primary

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {{
            font-family: 'Inter', sans-serif;
        }}
        .stApp, .stApp header, .stApp footer {{
            background: linear-gradient(145deg, {bg_main}, {'#0a0c10' if theme == 'dark' else '#fafafa'}) !important;
            color: var(--text-primary) !important;
        }}
        section[data-testid="stSidebar"] {{
            background: {bg_sidebar} !important;
            backdrop-filter: blur(12px) !important;
            border-right: 1px solid {border_color} !important;
        }}
        .stTextInput input, .stNumberInput input, .stSelectbox, .stTextArea textarea {{
            background-color: {'rgba(38, 39, 48, 0.8)' if theme == 'dark' else 'rgba(255,255,255,0.8)'} !important;
            color: var(--text-primary) !important;
            border-color: {border_color} !important;
            border-radius: 12px !important;
            backdrop-filter: blur(4px) !important;
            transition: all 0.2s ease;
        }}
        .stTextInput input:focus, .stNumberInput input:focus {{
            border-color: var(--accent) !important;
            box-shadow: 0 0 0 2px rgba(74, 111, 165, 0.2) !important;
        }}
        .stButton button {{
            background-color: {'rgba(45, 47, 54, 0.8)' if theme == 'dark' else 'rgba(255,255,255,0.8)'} !important;
            color: var(--text-primary) !important;
            border: 1px solid {border_color} !important;
            border-radius: 12px !important;
            backdrop-filter: blur(4px) !important;
            transition: all 0.2s ease;
            font-weight: 500;
        }}
        .stButton button:hover {{
            border-color: var(--accent) !important;
            background-color: {'rgba(58, 60, 68, 0.9)' if theme == 'dark' else 'rgba(241, 245, 249, 0.9)'} !important;
            transform: translateY(-1px);
        }}
        div[data-testid="stExpander"] {{
            background-color: {'rgba(30, 32, 40, 0.5)' if theme == 'dark' else 'rgba(248, 249, 250, 0.7)'} !important;
            border: 1px solid {border_color} !important;
            border-radius: 16px !important;
            backdrop-filter: blur(8px) !important;
        }}
        .result-card {{
            background: {card_bg};
            backdrop-filter: blur(12px) !important;
            border-radius: 20px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid {border_color};
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: fadeIn 0.5s ease;
        }}
        .result-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 35px rgba(0,0,0,0.3);
            border-color: var(--accent);
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .confidence-badge {{
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 0.3em 0.8em;
            border-radius: 30px;
            font-weight: 600;
            font-size: 0.8em;
            display: inline-block;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
        }}
        .tag-pill {{
            background: {tag_bg};
            color: {tag_text};
            padding: 0.25em 0.9em;
            border-radius: 30px;
            font-size: 0.8em;
            margin-right: 0.4em;
            margin-bottom: 0.3em;
            display: inline-block;
            border: 1px solid {border_color};
            backdrop-filter: blur(4px);
            transition: all 0.2s ease;
        }}
        .tag-pill:hover {{
            background: {accent};
            color: white;
            cursor: pointer;
        }}
        .video-preview-placeholder {{
            background: linear-gradient(145deg, #1e293b, #0f172a);
            border-radius: 16px;
            height: 140px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #94a3b8;
            font-weight: 500;
            margin-bottom: 1rem;
            border: 1px solid {border_color};
            box-shadow: 0 8px 20px rgba(0,0,0,0.2);
        }}
        .stProgress > div > div {{
            background: linear-gradient(90deg, {accent}, #60a5fa) !important;
            border-radius: 10px !important;
        }}
        .st-bb, .st-at {{
            background-color: transparent !important;
        }}
        /* 🚀 ENHANCEMENT: Toast notifications */
        .toast {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: {accent};
            color: white;
            padding: 12px 24px;
            border-radius: 50px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease;
            z-index: 9999;
        }}
        @keyframes slideIn {{
            from {{ transform: translateX(100%); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
    </style>
    """

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOAD ML MODELS (cached, lightweight bi-encoder + cross-encoder)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    return bi_encoder, cross_encoder, kw_model

bi_encoder, cross_encoder, kw_model = load_models()

# ------------------------------------------------------------------
# PERSISTENT CHROMADB (with versioning, dimension 384)
# ------------------------------------------------------------------
@st.cache_resource
def init_chromadb():
    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    collection_name = "cairo_subtitles"
    try:
        collection = client.get_collection(collection_name)
        if collection.metadata and collection.metadata.get("embedding_model") != "all-MiniLM-L6-v2":
            st.warning("⚠️ Database created with a different model. Please reset database in sidebar.")
            return None
        return collection
    except:
        collection = client.create_collection(
            name=collection_name,
            metadata={
                "hnsw:space": "ip",
                "embedding_model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "created_at": time.time()
            }
        )
        return collection

st.session_state.collection = init_chromadb()
collection = st.session_state.collection
COLLECTION_VALID = collection is not None

# ------------------------------------------------------------------
# 🚀 ENHANCEMENT: LLM Query Rewriting + Semantic Cache
# ------------------------------------------------------------------
def rewrite_query_with_llm(query: str) -> str:
    """Use Mistral to rewrite the query for better entity understanding."""
    if not MISTRAL_API_KEY:
        return query
    
    # Simple in-memory cache to avoid repeated API calls
    if query in st.session_state.query_cache:
        return st.session_state.query_cache[query]
    
    prompt = f"""Rewrite this search query to be more specific and disambiguate entities.
    - Expand names (e.g., 'Ronaldo' → 'Cristiano Ronaldo')
    - Clarify temporal terms (e.g., 'first goal' → 'debut goal, first goal')
    - Remove ambiguity.
    Output ONLY the rewritten query, no extra text.

    Original: {query}
    Rewritten:"""
    
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=50
        )
        rewritten = response.choices[0].message.content.strip()
        if rewritten and len(rewritten) > 3:
            st.session_state.query_cache[query] = rewritten
            return rewritten
    except Exception as e:
        st.warning(f"⚠️ Query rewriting failed: {str(e)[:50]}")
    
    return query

# 🚀 Simple LRU cache for search results
def get_cached_results(query: str, n_results: int):
    cache_key = f"{query}_{n_results}"
    if cache_key in st.session_state.result_cache:
        results, timestamp = st.session_state.result_cache[cache_key]
        # Cache TTL: 1 hour
        if time.time() - timestamp < 3600:
            return results
    return None

def cache_results(query: str, n_results: int, results):
    cache_key = f"{query}_{n_results}"
    st.session_state.result_cache[cache_key] = (results, time.time())
    # Limit cache size to 50 entries (simple LRU)
    if len(st.session_state.result_cache) > 50:
        # Remove oldest
        oldest = min(st.session_state.result_cache.items(), key=lambda x: x[1][1])
        del st.session_state.result_cache[oldest[0]]

# ------------------------------------------------------------------
# HELPER FUNCTIONS (unchanged)
# ------------------------------------------------------------------
def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

def expand_query(query: str) -> str:
    keywords = kw_model.extract_keywords(
        query,
        keyphrase_ngram_range=(1, 2),
        stop_words='english',
        top_n=3
    )
    if keywords:
        keyword_terms = [kw[0] for kw in keywords]
        return query + " " + " ".join(keyword_terms)
    return query

def get_embeddings_batch(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    if not texts:
        return []
    texts = [t if t.strip() else " " for t in texts]
    return bi_encoder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True
    ).tolist()

# ------------------------------------------------------------------
# SLIDING WINDOW CHUNKING (unchanged)
# ------------------------------------------------------------------
def sliding_window_chunks_srt(filepath: str, window: int, overlap: int) -> List[Dict]:
    # ... (keep your existing implementation)
    subs = pysrt.open(filepath)
    if not subs:
        return []
    entries = []
    for sub in subs:
        text = sub.text.replace("\n", " ").strip()
        if text:
            entries.append({
                "start": sub.start.ordinal / 1000.0,
                "end": sub.end.ordinal / 1000.0,
                "text": text
            })
    if not entries:
        return []
    chunks = []
    n = len(entries)
    start_idx = 0
    current_start = entries[0]["start"]
    last_end = entries[-1]["end"]
    while current_start < last_end:
        window_end = current_start + window
        while start_idx < n and entries[start_idx]["start"] < current_start:
            start_idx += 1
        window_entries = []
        idx = start_idx
        while idx < n and entries[idx]["start"] < window_end:
            window_entries.append(entries[idx])
            idx += 1
        if window_entries:
            merged_text = " ".join([e["text"] for e in window_entries])
            chunk_start = window_entries[0]["start"]
            chunk_end = window_entries[-1]["end"]
            chunks.append({
                "text": merged_text,
                "start": chunk_start,
                "end": chunk_end,
                "subtitle_count": len(window_entries)
            })
        current_start += overlap
    return chunks

def sliding_window_chunks_vtt(filepath: str, window: int, overlap: int) -> List[Dict]:
    subs = webvtt.read(filepath)
    entries = []
    for sub in subs:
        text = sub.text.replace("\n", " ").strip()
        if text:
            entries.append({
                "start": sub.start_in_seconds,
                "end": sub.end_in_seconds,
                "text": text
            })
    if not entries:
        return []
    chunks = []
    n = len(entries)
    start_idx = 0
    current_start = entries[0]["start"]
    last_end = entries[-1]["end"]
    while current_start < last_end:
        window_end = current_start + window
        while start_idx < n and entries[start_idx]["start"] < current_start:
            start_idx += 1
        window_entries = []
        idx = start_idx
        while idx < n and entries[idx]["start"] < window_end:
            window_entries.append(entries[idx])
            idx += 1
        if window_entries:
            merged_text = " ".join([e["text"] for e in window_entries])
            chunk_start = window_entries[0]["start"]
            chunk_end = window_entries[-1]["end"]
            chunks.append({
                "text": merged_text,
                "start": chunk_start,
                "end": chunk_end,
                "subtitle_count": len(window_entries)
            })
        current_start += overlap
    return chunks

# ------------------------------------------------------------------
# ENHANCED METADATA PIPELINE (unchanged)
# ------------------------------------------------------------------
def _parse_llm_output(text: str) -> Dict:
    result = {"summary": "", "themes": [], "entities": [], "tags": []}
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if line.lower().startswith("summary:"):
            result["summary"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("themes:"):
            result["themes"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif line.lower().startswith("entities:"):
            result["entities"] = [e.strip() for e in line.split(":", 1)[1].split(",") if e.strip()]
        elif line.lower().startswith("tags:"):
            result["tags"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
    if not result["summary"] and lines:
        result["summary"] = lines[0][:150] + "..."
    return result

def _enrich_with_llm(full_text: str, domain_hint: str) -> Dict:
    if not MISTRAL_API_KEY:
        return {}
    full_text = full_text[:8000]
    prompt = f"""You are an expert content analyst specializing in {domain_hint}. Analyze this video transcript:
Transcript:
{full_text}
Extract structured metadata following EXACTLY this format:
Summary: <one concise sentence capturing core content>
Themes: <theme1>, <theme2>, <theme3>, <theme4>
Entities: <person1>, <person2>, <organization1>, <location1>
Tags: <tag1>, <tag2>, <tag3>, <tag4>, <tag5>, <tag6>, <tag7>, <tag8>, <tag9>, <tag10>
Rules:
- Be specific and concrete
- Prioritize named entities and domain-specific concepts
- Output ONLY the four sections above
"""
    try:
        client = Mistral(api_key=MISTRAL_API_KEY)
        chat_response = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        return _parse_llm_output(chat_response.choices[0].message.content)
    except Exception as e:
        st.warning(f"⚠️ LLM enrichment failed: {str(e)[:100]}")
        return {}

def generate_enhanced_metadata(full_text: str, filename: str, domain_hint: str = "general content") -> Dict:
    keywords = kw_model.extract_keywords(
        full_text[:5000],
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=15,
        diversity=0.7
    )
    semantic_tags = [kw[0] for kw in keywords]
    llm_meta = _enrich_with_llm(full_text, domain_hint)
    merged = {
        "summary": llm_meta.get("summary", " ".join(semantic_tags[:3])),
        "themes": llm_meta.get("themes", semantic_tags[:5]),
        "entities": llm_meta.get("entities", []),
        "tags": {
            "primary": llm_meta.get("tags", [])[:5],
            "secondary": semantic_tags[5:10],
            "keywords": semantic_tags[:8]
        },
        "confidence": {
            "summary": 0.9 if llm_meta.get("summary") else 0.6,
            "themes": 0.85 if llm_meta.get("themes") else 0.7,
            "tags": 0.9 if llm_meta.get("tags") else 0.75
        },
        "domain_hint": domain_hint,
        "filename": filename
    }
    return merged

# ------------------------------------------------------------------
# PROCESS FILE (unchanged)
# ------------------------------------------------------------------
def process_file_optimized(uploaded_file, window: int, overlap: int, domain_hint: str = "general content") -> Tuple[Optional[int], Optional[str]]:
    if not COLLECTION_VALID:
        return None, "Database is invalid. Please reset."
    file_hash = get_file_hash(uploaded_file.getvalue())
    existing = collection.get(where={"file_hash": file_hash})
    if existing['ids']:
        filename = existing['metadatas'][0].get('filename')
        st.info(f"✅ {filename} already indexed (deduplicated)")
        if filename not in st.session_state.full_texts:
            all_texts = existing['documents']
            st.session_state.full_texts[filename] = " ".join(all_texts)
        return len(existing['ids']), None
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    filename = uploaded_file.name
    if filename.endswith('.srt'):
        chunks = sliding_window_chunks_srt(tmp_path, window, overlap)
    elif filename.endswith('.vtt'):
        chunks = sliding_window_chunks_vtt(tmp_path, window, overlap)
    else:
        os.unlink(tmp_path)
        return None, "Unsupported file type."
    valid_chunks = [c for c in chunks if c["text"].strip() and len(c["text"]) > 10]
    if not valid_chunks:
        os.unlink(tmp_path)
        return 0, "No valid subtitle content found."
    texts = [c["text"] for c in valid_chunks]
    metadatas = []
    for c in valid_chunks:
        mins = int(c["start"] // 60)
        secs = int(c["start"] % 60)
        metadatas.append({
            "filename": filename,
            "file_hash": file_hash,
            "start": c["start"],
            "end": c["end"],
            "duration": round(c["end"] - c["start"], 1),
            "timecode": f"{mins}:{secs:02d}",
            "subtitle_count": c.get("subtitle_count", 0),
            "domain_hint": domain_hint,
            "speaker": "unknown"
        })
    embeddings = get_embeddings_batch(texts)
    if len(embeddings) != len(texts):
        os.unlink(tmp_path)
        return None, f"Embedding mismatch: {len(embeddings)} vs {len(texts)}"
    tags_list = []
    for text in texts:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words='english',
            top_n=3
        )
        tag_string = ", ".join([kw[0] for kw in keywords]) if keywords else "general"
        tags_list.append(tag_string)
    for i, tags in enumerate(tags_list):
        metadatas[i]["tags"] = tags
    ids = [f"{filename}_{i}_{uuid.uuid4()}" for i in range(len(texts))]
    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )
    full_text = " ".join(texts)
    st.session_state.full_texts[filename] = full_text
    with st.spinner("🧠 Generating AI metadata..."):
        enriched = generate_enhanced_metadata(full_text, filename, domain_hint)
        st.session_state.enriched_metadata[filename] = enriched
    os.unlink(tmp_path)
    return len(valid_chunks), None

# ------------------------------------------------------------------
# 🚀 ENHANCEMENT: Search with adaptive diversity + min score filter
# ------------------------------------------------------------------
def diversify_results(results: List[Dict], lambda_param: float = 0.6, top_k: int = 5) -> List[Dict]:
    if len(results) <= top_k:
        return results
    texts = [r["text"] for r in results]
    embeddings = get_embeddings_batch(texts)
    embeddings = np.array(embeddings)
    scores = np.array([r["score"] for r in results])
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    selected_indices = []
    remaining_indices = list(range(len(results)))
    first_idx = np.argmax(scores)
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)
    for _ in range(min(top_k, len(results)) - 1):
        if not remaining_indices:
            break
        mmr_scores = []
        for idx in remaining_indices:
            rel = scores[idx]
            sim_max = max(cosine_similarity(embeddings[idx].reshape(1, -1), embeddings[sel].reshape(1, -1))[0][0]
                          for sel in selected_indices)
            mmr = lambda_param * rel - (1 - lambda_param) * sim_max
            mmr_scores.append(mmr)
        best_idx = remaining_indices[np.argmax(mmr_scores)]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
    return [results[i] for i in selected_indices]

def search_subtitles(
    query: str,
    n_final: int = 5,
    n_candidates: int = 100,
    diversify: bool = True,
    lambda_param: float = 0.6,
    min_score: float = -10.0
) -> List[Dict]:
    if not COLLECTION_VALID:
        return []
    try:
        expanded = expand_query(query)
        query_vec = get_embeddings_batch([expanded])
        results = collection.query(
            query_embeddings=query_vec,
            n_results=n_candidates
        )
        if not results['ids'][0]:
            return []
        pairs = [[query, doc] for doc in results['documents'][0]]
        cross_scores = cross_encoder.predict(pairs)
        reranked = []
        for i in range(len(results['ids'][0])):
            score = float(cross_scores[i])
            if score < min_score:
                continue
            reranked.append({
                "score": score,
                "text": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "id": results['ids'][0][i]
            })
        reranked.sort(key=lambda x: x["score"], reverse=True)
        if diversify:
            reranked = diversify_results(reranked, lambda_param=lambda_param, top_k=n_final)
        else:
            reranked = reranked[:n_final]
        return reranked
    except Exception as e:
        st.error(f"❌ Search failed: {e}")
        return []

def delete_file(filename: str):
    if not COLLECTION_VALID:
        return
    collection.delete(where={"filename": filename})
    if filename in st.session_state.full_texts:
        del st.session_state.full_texts[filename]
    if filename in st.session_state.enriched_metadata:
        del st.session_state.enriched_metadata[filename]

# ------------------------------------------------------------------
# SIDEBAR UI (modern, collapsible, theme toggle)
# ------------------------------------------------------------------
with st.sidebar:
    col_logo, col_theme = st.columns([4, 1])
    with col_logo:
        st.title("🎥 Cairo")
    with col_theme:
        icon = "🌙" if st.session_state.theme == "dark" else "☀️"
        if st.button(icon, key="theme_toggle", help="Toggle light/dark mode"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    st.divider()

    with st.expander("🎛️ Scene Chunking", expanded=False):
        st.caption("Controls how videos are split into scenes. Changes apply to **new uploads**.")
        window_size = st.slider("Window (seconds)", 10, 60, 30, 5)
        overlap = st.slider("Overlap (seconds)", 5, 30, 15, 5)

    st.divider()

    st.subheader("📤 Upload Content")
    domain_hint = st.selectbox(
        "Content domain (improves AI metadata)",
        ["general content", "sports", "news", "entertainment", "education", "technology"],
        index=0,
        help="Helps the LLM generate more relevant tags"
    )

    uploaded_files = st.file_uploader(
        "Subtitles (.srt/.vtt)",
        type=['srt', 'vtt'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files and COLLECTION_VALID:
        for uploaded_file in uploaded_files:
            with st.status(f"Processing {uploaded_file.name}...", expanded=False) as status:
                chunk_count, error = process_file_optimized(
                    uploaded_file,
                    window_size,
                    overlap,
                    domain_hint
                )
                if error:
                    status.update(label=f"❌ {error}", state="error")
                else:
                    status.update(label=f"✅ {chunk_count} scenes indexed", state="complete")

    st.divider()

    # 🚀 ENHANCEMENT: Library stats with more metrics
    if COLLECTION_VALID:
        try:
            all_items = collection.get()
            total_chunks = len(all_items['ids'])
            files_set = set()
            for m in all_items['metadatas']:
                fname = m.get('filename')
                if fname:
                    files_set.add(fname)
            files_indexed = files_set
        except:
            total_chunks = 0
            files_indexed = set()
    else:
        total_chunks = 0
        files_indexed = set()

    st.subheader("📊 Library Stats")
    col1, col2 = st.columns(2)
    col1.metric("Videos", len(files_indexed))
    col2.metric("Scenes", f"{total_chunks:,}")

    if files_indexed:
        with st.expander("📁 Manage Videos", expanded=False):
            for filename in sorted(files_indexed):
                col1, col2 = st.columns([5, 1])
                col1.caption(filename[:25] + "..." if len(filename) > 25 else filename)
                if col2.button("🗑️", key=f"del_{filename}", help="Remove from library"):
                    delete_file(filename)
                    st.rerun()
            if st.button("🗑️ Clear All"):
                for filename in files_indexed:
                    delete_file(filename)
                st.rerun()

    st.divider()

    with st.expander("⚙️ System", expanded=False):
        if st.button("🔄 Reset Database"):
            try:
                persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
                client = chromadb.PersistentClient(path=persist_dir)
                try:
                    client.delete_collection("cairo_subtitles")
                except:
                    pass
                st.session_state.collection = None
                st.session_state.enriched_metadata = {}
                st.session_state.full_texts = {}
                st.rerun()
            except Exception as e:
                st.error(f"Reset failed: {e}")

    st.divider()

    with st.expander("🧠 AI Video Summary", expanded=False):
        if COLLECTION_VALID and files_indexed:
            selected_file = st.selectbox(
                "Choose a video",
                sorted(list(files_indexed)),
                key="llm_selector"
            )
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button("✨ Generate Summary"):
                    full_text = st.session_state.full_texts.get(selected_file, "")
                    if not full_text:
                        file_chunks = collection.get(where={"filename": selected_file})
                        if file_chunks['documents']:
                            full_text = " ".join(file_chunks['documents'])
                    if full_text:
                        with st.status(f"🧠 Mistral analyzing..."):
                            enriched = generate_enhanced_metadata(full_text, selected_file, domain_hint)
                            st.session_state.enriched_metadata[selected_file] = enriched
                            if "error" not in enriched:
                                st.success("✅ Metadata generated!")
                            else:
                                st.error(f"LLM error: {enriched['error']}")
                    else:
                        st.error("No subtitle text found.")
            with col2:
                if st.button("🗑️", key=f"clear_{selected_file}"):
                    if selected_file in st.session_state.enriched_metadata:
                        del st.session_state.enriched_metadata[selected_file]
                        st.rerun()

            if selected_file in st.session_state.enriched_metadata:
                meta = st.session_state.enriched_metadata[selected_file]
                if "error" in meta:
                    st.error(f"LLM error: {meta['error']}")
                else:
                    st.markdown(f"**📝 Summary**  \n{meta.get('summary', 'N/A')[:150]}...")
                    st.markdown(f"**🎯 Themes**  \n{', '.join(meta.get('themes', ['N/A']))}")
                    st.markdown("**🏷️ Tags**")
                    tags = meta.get('tags', {}).get('primary', [])
                    if tags:
                        tag_html = " ".join([f'<span class="tag-pill">{tag}</span>' for tag in tags[:6]])
                        st.markdown(tag_html, unsafe_allow_html=True)
        else:
            if COLLECTION_VALID:
                st.info("Upload videos to enable AI summaries.")

# ------------------------------------------------------------------
# MAIN AREA – SEARCH & RESULTS
# ------------------------------------------------------------------
st.title("🔍 Semantic Video Search")
st.markdown("Find exact moments using natural language. *Powered by subtitle semantics + AI enrichment.*")

# 🚀 ENHANCEMENT: Search history dropdown (quick re-runs)
if st.session_state.search_history:
    with st.expander("📜 Search History", expanded=False):
        cols = st.columns(5)
        for idx, (past_query, timestamp) in enumerate(list(reversed(st.session_state.search_history))[:5]):
            if cols[idx % 5].button(f"\"{past_query[:15]}...\"", key=f"history_{idx}"):
                st.session_state.last_query = past_query
                st.rerun()

search_container = st.container()
with search_container:
    col_search, col_opts = st.columns([4, 1])
    with col_search:
        query = st.text_input(
            "Search your video library",
            placeholder="e.g., 'when did Messi score his first goal?'",
            value=st.session_state.last_query,
            label_visibility="collapsed"
        )
    with col_opts:
        if st.button("⚙️", help="Advanced options"):
            st.session_state.show_advanced = not st.session_state.show_advanced

    # 🚀 ENHANCEMENT: Advanced options with adaptive diversity & score filter
    if st.session_state.show_advanced:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            n_results = st.number_input("Results", 1, 20, 5)
        with col2:
            diversify = st.checkbox("Diversify", value=True, help="Show varied moments vs. similar repeats")
        with col3:
            lambda_param = st.slider("Diversity λ", 0.0, 1.0, 0.6, 0.1,
                                    help="Higher = more relevance, Lower = more diversity")
        with col4:
            min_score = st.slider("Min score", -5.0, 5.0, -3.0, 0.5,
                                 help="Filter out low-confidence results")
        n_candidates = st.number_input("Candidates", 20, 200, 100, step=10,
                                       help="Number of raw scenes retrieved before reranking.")
    else:
        n_results = 5
        diversify = True
        lambda_param = 0.6
        min_score = -3.0
        n_candidates = 100

st.divider()

# 🚀 ENHANCEMENT: Filter by filename (if multiple files exist)
if files_indexed and len(files_indexed) > 1:
    with st.expander("🎯 Filter by Video", expanded=False):
        filename_filter = st.selectbox(
            "Narrow search to a specific video",
            ["All files"] + sorted(files_indexed),
            index=0
        )
        st.session_state.filter_filename = filename_filter if filename_filter != "All files" else None
else:
    st.session_state.filter_filename = None

if query and COLLECTION_VALID:
    # Add to search history
    if query != st.session_state.last_query:
        st.session_state.search_history.append((query, datetime.now().strftime("%H:%M")))
        # Keep only last 20 searches
        st.session_state.search_history = st.session_state.search_history[-20:]
    st.session_state.last_query = query

    # 🚀 ENHANCEMENT: Query rewriting with Mistral
    rewritten_query = rewrite_query_with_llm(query)
    if rewritten_query != query and MISTRAL_API_KEY:
        st.caption(f"✨ Expanded query: *{rewritten_query}*")

    # 🚀 ENHANCEMENT: Check cache
    cached = get_cached_results(rewritten_query, n_candidates)
    if cached is not None:
        results = cached
        st.toast("⚡ Results loaded from cache", icon="⚡")
    else:
        with st.status("🔍 Searching library...", expanded=False) as status:
            results = search_subtitles(
                rewritten_query,
                n_final=n_results,
                n_candidates=n_candidates,
                diversify=diversify,
                lambda_param=lambda_param,
                min_score=min_score
            )
            # Apply filename filter if active
            if st.session_state.filter_filename:
                results = [r for r in results if r['metadata']['filename'] == st.session_state.filter_filename]
            cache_results(rewritten_query, n_candidates, results)
            status.update(label=f"✅ Found {len(results)} relevant moments", state="complete")

    if results:
        # 🚀 ENHANCEMENT: Export all results as CSV
        if len(results) > 0:
            export_df = pd.DataFrame([
                {
                    "Scene": i+1,
                    "File": r['metadata']['filename'],
                    "Timecode": r['metadata']['timecode'],
                    "Duration": r['metadata']['duration'],
                    "Confidence": f"{max(0, (r['score']+5)/20*100):.0f}%",
                    "Text": r['text'][:200]
                } for i, r in enumerate(results)
            ])
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export All Results (CSV)",
                data=csv,
                file_name=f"cairo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        # Display results with pagination
        results_per_page = 5
        if 'result_offset' not in st.session_state:
            st.session_state.result_offset = 0

        start_idx = st.session_state.result_offset
        end_idx = min(start_idx + results_per_page, len(results))
        displayed_results = results[start_idx:end_idx]

        for i, r in enumerate(displayed_results, start=start_idx):
            meta = r["metadata"]
            score = r["score"]
            norm_score = max(0.0, min(1.0, (score + 5) / 20))

            with st.container():
                st.markdown('<div class="result-card">', unsafe_allow_html=True)

                col1, col2 = st.columns([3, 1])
                with col1:
                    display_name = meta['filename'][:30] + "..." if len(meta['filename']) > 30 else meta['filename']
                    st.markdown(f"**{display_name}**")
                    st.caption(f"⏱️ {meta['timecode']} | 📏 {meta['duration']}s scene")
                with col2:
                    st.markdown(f"<div class='confidence-badge'>{norm_score*100:.0f}% match</div>", unsafe_allow_html=True)

                st.progress(norm_score)

                if meta.get("tags"):
                    tags = meta["tags"].split(", ")[:8]
                    tag_html = " ".join([f'<span class="tag-pill">{tag}</span>' for tag in tags if tag != "general"])
                    st.markdown(f"🏷️ {tag_html}", unsafe_allow_html=True)

                with st.expander("💬 Scene transcript"):
                    st.write(r["text"])

                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("▶️ Jump to", key=f"jump_{i}", use_container_width=True):
                        st.session_state.selected_result = r
                        st.rerun()
                with col_btn2:
                    st.download_button(
                        "📥 Export",
                        data=json.dumps(r, indent=2),
                        file_name=f"scene_{i}_{int(meta['start'])}.json",
                        mime="application/json",
                        use_container_width=True
                    )

                st.markdown('</div>', unsafe_allow_html=True)
                st.divider()

        # 🚀 ENHANCEMENT: Load more button (pagination)
        if end_idx < len(results):
            if st.button("⬇️ Load more results", use_container_width=True):
                st.session_state.result_offset = end_idx
                st.rerun()
        elif st.session_state.result_offset > 0:
            if st.button("⬆️ Show fewer", use_container_width=True):
                st.session_state.result_offset = 0
                st.rerun()
    else:
        st.warning("No relevant moments found. Try rephrasing your query, lowering the min score, or uploading more content.")

    # 🚀 ENHANCEMENT: Selected moment detail (unchanged)
    if st.session_state.selected_result:
        st.divider()
        st.subheader("🎬 Selected Moment")
        sel = st.session_state.selected_result
        meta_sel = sel["metadata"]

        col_vid, col_meta = st.columns([2, 1])
        with col_vid:
            st.markdown('<div class="video-preview-placeholder">🎬 Video Preview (Integration Point)</div>', unsafe_allow_html=True)
            st.caption(f"⏱️ {meta_sel['timecode']} in {meta_sel['filename']}")
            col_play, col_time = st.columns([1, 3])
            with col_play:
                st.button("⏯️ Play", use_container_width=True)
            with col_time:
                st.slider("Timeline", 0.0, float(meta_sel['end']),
                         float(meta_sel['start']),
                         label_visibility="collapsed")
        with col_meta:
            st.markdown("**📝 Context**")
            st.write(sel["text"][:200] + "...")
            st.markdown("**🏷️ Semantic Tags**")
            meta_enriched = st.session_state.enriched_metadata.get(meta_sel['filename'], {})
            if meta_enriched:
                for category, tags in meta_enriched.get("tags", {}).items():
                    if tags:
                        st.markdown(f"*{category.title()}*: " + ", ".join(tags[:4]))
            else:
                st.caption("Generate metadata in sidebar for richer context")

        if st.button("← Back to results"):
            st.session_state.selected_result = None
            st.rerun()

elif COLLECTION_VALID and total_chunks == 0:
    st.info("👋 Start by uploading subtitle files in the sidebar.")
else:
    st.markdown("""
    ### Welcome to Project Cairo

    Transform your video library into a **searchable, ad‑ready asset**:

    - ✅ **Low‑hanging fruit**: Start with subtitles (fast, accurate)
    - ✅ **Premium tier**: Later integrate vision models for non‑subtitled content
    - ✅ **Ad‑ready**: Hierarchical tags enable contextual ad targeting
    - ✅ **Curated discovery**: Semantic search finds exact moments, not just videos

    **Next step**: Upload your first subtitle file in the sidebar →
    """)
