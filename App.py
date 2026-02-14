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
from rank_bm25 import BM25Okapi
from typing import List, Dict, Tuple, Optional
import json
import time
from datetime import datetime
import re
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs

# ------------------------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Project Cairo – Enterprise Semantic Search",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------------------------
_DEFAULT_STATE = {
    "theme": "dark",
    "last_query": "",
    "selected_result": None,
    "show_advanced": False,
    "collection": None,               # scene‑level ChromaDB collection
    "video_meta_collection": None,    # video‑level metadata collection
    "enriched_metadata": {},          # cache for video metadata (loaded from DB)
    "full_texts": {},
    "processed_hashes": set(),
    "search_history": [],
    "query_cache": {},
    "result_cache": {},
    "filter_filename": None,
    "result_offset": 0,
    "bm25_index": None,
    "bm25_docs": [],
    "bm25_doc_ids": [],
    "iab_taxonomy": None,
}
for key, val in _DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ------------------------------------------------------------------
# SECRETS
# ------------------------------------------------------------------
MISTRAL_API_KEY = st.secrets.get("MISTRAL_API_KEY")
HF_TOKEN = st.secrets.get("HF_TOKEN")  # Hugging Face token for Gemma 2

# ------------------------------------------------------------------
# IAB TAXONOMY LOADER (simplified mapping)
# ------------------------------------------------------------------
def load_iab_taxonomy() -> Dict[str, List[str]]:
    """Return a simple mapping from keywords to IAB categories."""
    return {
        "sports": ["IAB17", "IAB17-6", "IAB17-8"],
        "football": ["IAB17-6"],
        "soccer": ["IAB17-6"],
        "cricket": ["IAB17-8"],
        "news": ["IAB12"],
        "politics": ["IAB12-2"],
        "technology": ["IAB19"],
        "entertainment": ["IAB1"],
        "movies": ["IAB1-2"],
        "music": ["IAB1-5"],
    }

st.session_state.iab_taxonomy = load_iab_taxonomy()

# ------------------------------------------------------------------
# PROFESSIONAL THEME – Clean, modern, enterprise
# ------------------------------------------------------------------
def get_theme_css(theme: str) -> str:
    if theme == "dark":
        bg_main = "#0b0e14"
        bg_sidebar = "#1a1e26"
        text_primary = "#e2e8f0"
        text_secondary = "#9aa4b5"
        border_color = "#2d3748"
        accent = "#4f7eb3"
        card_bg = "#1f2630"
        tag_bg = "#2d3748"
    else:
        bg_main = "#f8fafc"
        bg_sidebar = "#ffffff"
        text_primary = "#1e293b"
        text_secondary = "#64748b"
        border_color = "#e2e8f0"
        accent = "#3b82f6"
        card_bg = "#ffffff"
        tag_bg = "#f1f5f9"

    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
        * {{ font-family: 'Inter', sans-serif; }}
        .stApp, .stApp header {{ background-color: {bg_main}; color: {text_primary}; }}
        section[data-testid="stSidebar"] {{
            background-color: {bg_sidebar};
            border-right: 1px solid {border_color};
        }}
        .stTextInput input, .stNumberInput input, .stSelectbox, .stTextArea textarea {{
            background-color: {bg_sidebar};
            color: {text_primary};
            border: 1px solid {border_color};
            border-radius: 8px;
        }}
        .stButton button {{
            background-color: {bg_sidebar};
            color: {text_primary};
            border: 1px solid {border_color};
            border-radius: 8px;
            transition: all 0.2s;
        }}
        .stButton button:hover {{
            border-color: {accent};
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        div[data-testid="stExpander"] {{
            background-color: {bg_sidebar};
            border: 1px solid {border_color};
            border-radius: 12px;
            padding: 0.5rem;
        }}
        .result-card {{
            background-color: {card_bg};
            border: 1px solid {border_color};
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}
        .result-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1);
        }}
        .confidence-badge {{
            background: {accent};
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 30px;
            font-weight: 600;
            font-size: 0.75rem;
        }}
        .tag-pill {{
            background-color: {tag_bg};
            color: {text_primary};
            padding: 0.25rem 0.75rem;
            border-radius: 30px;
            font-size: 0.75rem;
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            display: inline-block;
            border: 1px solid {border_color};
            transition: background-color 0.2s;
        }}
        .tag-pill.editable:hover {{
            background-color: {accent};
            color: white;
            cursor: pointer;
        }}
        .iab-badge {{
            background-color: {accent}20;
            color: {accent};
            padding: 0.2rem 0.6rem;
            border-radius: 4px;
            font-size: 0.7rem;
            font-weight: 500;
            margin-right: 0.3rem;
        }}
        .video-preview-placeholder {{
            background: linear-gradient(145deg, {bg_sidebar}, {card_bg});
            border: 1px solid {border_color};
            border-radius: 12px;
            height: 140px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: {text_secondary};
        }}
        .stProgress > div > div {{
            background-color: {accent};
            border-radius: 10px;
        }}
        .sentiment-tag {{
            background-color: {accent}30;
            color: {accent};
            padding: 0.2rem 0.6rem;
            border-radius: 12px;
            font-size: 0.7rem;
            margin-right: 0.3rem;
            display: inline-block;
        }}
    </style>
    """

st.markdown(get_theme_css(st.session_state.theme), unsafe_allow_html=True)

# ------------------------------------------------------------------
# LOAD MODELS (cached)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    bi_encoder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu')
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    return bi_encoder, cross_encoder, kw_model

bi_encoder, cross_encoder, kw_model = load_models()

# ------------------------------------------------------------------
# PERSISTENT CHROMADB – Scene collection and Video metadata collection
# ------------------------------------------------------------------
@st.cache_resource
def init_chromadb():
    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)

    # Scene-level collection
    scene_collection_name = "cairo_subtitles"
    try:
        scene_collection = client.get_collection(scene_collection_name)
        if scene_collection.metadata and scene_collection.metadata.get("embedding_model") != "all-MiniLM-L6-v2":
            st.warning("⚠️ Scene database created with different model. Reset DB in sidebar.")
            scene_collection = None
    except:
        scene_collection = client.create_collection(
            name=scene_collection_name,
            metadata={
                "hnsw:space": "ip",
                "embedding_model": "all-MiniLM-L6-v2",
                "dimension": 384,
                "created_at": time.time()
            }
        )

    # Video‑level metadata collection (stores one document per video)
    video_meta_collection_name = "video_metadata"
    try:
        video_meta_collection = client.get_collection(video_meta_collection_name)
    except:
        video_meta_collection = client.create_collection(
            name=video_meta_collection_name,
            metadata={"description": "video‑level metadata"}
        )

    return scene_collection, video_meta_collection

scene_collection, video_meta_collection = init_chromadb()
st.session_state.collection = scene_collection
st.session_state.video_meta_collection = video_meta_collection
COLLECTION_VALID = scene_collection is not None

# ------------------------------------------------------------------
# LOAD VIDEO METADATA FROM DB INTO SESSION (with error handling)
# ------------------------------------------------------------------
def load_video_metadata_into_session():
    """Safely load video metadata from ChromaDB into session state and parse JSON fields."""
    if st.session_state.video_meta_collection is None:
        return
    try:
        all_videos = st.session_state.video_meta_collection.get()
        for i, vid_id in enumerate(all_videos['ids']):
            meta = all_videos['metadatas'][i]
            filename = meta.get('filename')
            if filename:
                # Parse JSON fields back to Python objects
                parsed_meta = {}
                for key, value in meta.items():
                    if isinstance(value, str) and value.startswith(('{', '[')):
                        try:
                            parsed_meta[key] = json.loads(value)
                        except:
                            parsed_meta[key] = value
                    else:
                        parsed_meta[key] = value
                st.session_state.enriched_metadata[filename] = parsed_meta
    except Exception as e:
        st.warning(f"⚠️ Could not load video metadata: {e}")
        st.session_state.video_meta_collection = None

load_video_metadata_into_session()

# ------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------
def get_file_hash(file_content: bytes) -> str:
    return hashlib.md5(file_content).hexdigest()

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

def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL."""
    parsed = urlparse(url)
    if parsed.hostname in ('youtu.be', 'www.youtu.be'):
        return parsed.path[1:]
    if parsed.hostname in ('youtube.com', 'www.youtube.com', 'm.youtube.com'):
        if parsed.path == '/watch':
            return parse_qs(parsed.query).get('v', [None])[0]
        elif parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
    return None

def fetch_youtube_subtitles(video_id: str, languages=['en']) -> Optional[List[Dict]]:
    """Fetch YouTube subtitles and return as list of dicts with start, text, duration."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Try to find manual or auto-generated English transcript
        transcript = None
        try:
            transcript = transcript_list.find_manually_created_transcript(languages)
        except:
            try:
                transcript = transcript_list.find_generated_transcript(languages)
            except:
                pass
        if not transcript:
            return None
        # Fetch actual transcript
        transcript_data = transcript.fetch()
        # Convert to our format (start, end, text)
        subtitles = []
        for entry in transcript_data:
            subtitles.append({
                "start": entry['start'],
                "end": entry['start'] + entry['duration'],
                "text": entry['text']
            })
        return subtitles
    except Exception as e:
        st.warning(f"Could not fetch YouTube subtitles: {e}")
        return None

# ------------------------------------------------------------------
# BUILD BM25 INDEX (with error handling)
# ------------------------------------------------------------------
def build_bm25_index():
    """Build BM25 index from all documents in ChromaDB, if collection is valid."""
    if not COLLECTION_VALID or scene_collection is None:
        st.session_state.bm25_index = None
        st.session_state.bm25_docs = []
        st.session_state.bm25_doc_ids = []
        return
    try:
        all_items = scene_collection.get()
        if not all_items['documents']:
            st.session_state.bm25_index = None
            st.session_state.bm25_docs = []
            st.session_state.bm25_doc_ids = []
            return
        tokenized_docs = [re.findall(r'\w+', doc.lower()) for doc in all_items['documents']]
        st.session_state.bm25_index = BM25Okapi(tokenized_docs)
        st.session_state.bm25_docs = all_items['documents']
        st.session_state.bm25_doc_ids = all_items['ids']
    except Exception as e:
        st.warning(f"⚠️ Could not build BM25 index: {e}")
        st.session_state.bm25_index = None
        st.session_state.bm25_docs = []
        st.session_state.bm25_doc_ids = []

# Build BM25 index on startup (safe call)
build_bm25_index()

# ------------------------------------------------------------------
# DOMAIN DETECTION & QUERY REWRITING
# ------------------------------------------------------------------
def detect_domain(query: str) -> str:
    query_lower = query.lower()
    if any(word in query_lower for word in ["cricket", "test", "ashes", "bcci", "icc", "bowler", "wicket", "six", "four", "out"]):
        return "cricket"
    elif any(word in query_lower for word in ["football", "soccer", "goal", "fifa", "world cup", "penalty", "messi", "ronaldo"]):
        return "football"
    elif any(word in query_lower for word in ["basketball", "nba", "lebron", "curry"]):
        return "basketball"
    elif any(word in query_lower for word in ["tennis", "grand slam", "federer", "nadal", "djokovic"]):
        return "tennis"
    return "general"

def rewrite_query_with_llm(query: str, domain: str) -> str:
    if not MISTRAL_API_KEY:
        return query
    if query in st.session_state.query_cache:
        return st.session_state.query_cache[query]

    domain_prompts = {
        "cricket": "This is a cricket search query. Expand player names (e.g., 'Kohli' → 'Virat Kohli'), clarify match context (e.g., 'Ind vs Aus' → 'India vs Australia Test match'), and add cricket-specific terms.",
        "football": "This is a football/soccer search query. Expand player names, add team names, and use football terminology.",
        "general": "Expand the query with common synonyms and entity disambiguation."
    }
    prompt = domain_prompts.get(domain, domain_prompts["general"])

    prompt = f"""{prompt}
    Rewrite this search query to be more specific and disambiguate entities.
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
    except:
        pass
    return query

# ------------------------------------------------------------------
# HYBRID SEARCH (Vector + BM25) with RRF
# ------------------------------------------------------------------
def hybrid_search(query_vec, query_text: str, n_candidates: int = 100) -> Dict:
    # Vector search
    vector_results = scene_collection.query(
        query_embeddings=query_vec,
        n_results=n_candidates * 2
    )

    # BM25 search
    bm25_ids, bm25_docs, bm25_metadatas = [], [], []
    if st.session_state.bm25_index and st.session_state.bm25_docs:
        tokenized_query = re.findall(r'\w+', query_text.lower())
        bm25_scores = st.session_state.bm25_index.get_scores(tokenized_query)
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:n_candidates]
        bm25_ids = [st.session_state.bm25_doc_ids[i] for i in top_bm25_indices]
        bm25_docs = [st.session_state.bm25_docs[i] for i in top_bm25_indices]
        bm25_metadatas = [scene_collection.get(ids=[id])['metadatas'][0] for id in bm25_ids]

    # Combine with RRF
    all_ids = list(vector_results['ids'][0]) + bm25_ids
    all_docs = list(vector_results['documents'][0]) + bm25_docs
    all_metadatas = list(vector_results['metadatas'][0]) + bm25_metadatas

    seen = set()
    unique_ids, unique_docs, unique_metadatas = [], [], []
    for i, id_ in enumerate(all_ids):
        if id_ not in seen:
            seen.add(id_)
            unique_ids.append(id_)
            unique_docs.append(all_docs[i])
            unique_metadatas.append(all_metadatas[i])

    rrf_scores = {}
    for rank, id_ in enumerate(vector_results['ids'][0]):
        rrf_scores[id_] = rrf_scores.get(id_, 0) + 1 / (60 + rank + 1)
    for rank, id_ in enumerate(bm25_ids):
        rrf_scores[id_] = rrf_scores.get(id_, 0) + 1 / (60 + rank + 1)

    sorted_ids = sorted(unique_ids, key=lambda x: rrf_scores.get(x, 0), reverse=True)

    result_dict = {
        'ids': [sorted_ids[:n_candidates]],
        'documents': [[unique_docs[unique_ids.index(id_)] for id_ in sorted_ids[:n_candidates]]],
        'metadatas': [[unique_metadatas[unique_ids.index(id_)] for id_ in sorted_ids[:n_candidates]]],
        'distances': []
    }
    return result_dict

# ------------------------------------------------------------------
# DIVERSITY + MIN SCORE
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
        query_vec = get_embeddings_batch([query])
        results = hybrid_search(query_vec, query, n_candidates * 2)
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

# ------------------------------------------------------------------
# ENHANCED METADATA GENERATION (Gemma 2 / Mistral + KeyBERT + IAB)
# ------------------------------------------------------------------
def _parse_llm_output(text: str) -> Dict:
    """Parse the structured LLM output (now includes sentiment and tone)."""
    result = {
        "summary": "",
        "themes": [],
        "entities": [],
        "iab_categories": [],
        "sentiment": "",
        "tone": ""
    }
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines:
        if line.lower().startswith("summary:"):
            result["summary"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("themes:"):
            result["themes"] = [t.strip() for t in line.split(":", 1)[1].split(",") if t.strip()]
        elif line.lower().startswith("entities:"):
            result["entities"] = [e.strip() for e in line.split(":", 1)[1].split(",") if e.strip()]
        elif line.lower().startswith("iab:"):
            result["iab_categories"] = [c.strip() for c in line.split(":", 1)[1].split(",") if c.strip()]
        elif line.lower().startswith("sentiment:"):
            result["sentiment"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("tone:"):
            result["tone"] = line.split(":", 1)[1].strip()
    return result

def call_gemma_llm(prompt: str) -> Optional[str]:
    """Call Gemma 2 via Hugging Face Inference API."""
    if not HF_TOKEN:
        return None
    API_URL = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 400,
            "temperature": 0.2,
            "return_full_text": False
        }
    }
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            output = response.json()
            if isinstance(output, list) and len(output) > 0:
                return output[0].get("generated_text", "")
            return output.get("generated_text", "")
        else:
            st.warning(f"Gemma API error {response.status_code}")
    except Exception as e:
        st.warning(f"Gemma call failed: {e}")
    return None

def enrich_with_llm(full_text: str, domain_hint: str) -> Dict:
    """Use either Gemma 2 (if HF token) or Mistral (if Mistral key) to generate rich metadata."""
    full_text = full_text[:8000]  # token limit approx

    prompt = f"""You are an expert content analyst specializing in {domain_hint}. Analyze this video transcript and extract structured metadata including sentiment and tone.

Transcript:
{full_text}

Output in EXACTLY this format (no extra text):
Summary: <one concise sentence summarizing the video>
Themes: <theme1>, <theme2>, <theme3>, <theme4>, <theme5> (up to 10)
Entities: <person1>, <person2>, <organization1>, <location1>, <event1> (list important named entities)
IAB: <iab1>, <iab2> (list relevant IAB content categories, e.g., IAB17-6 for sports/football)
Sentiment: <positive/negative/neutral>
Tone: <one or two words describing the overall tone, e.g., excited, serious, humorous, dramatic>

Rules:
- Be specific, avoid generic terms
- Use IAB categories if applicable
- Output only the six lines above
"""

    # Try Gemma 2 first if HF token exists
    if HF_TOKEN:
        response_text = call_gemma_llm(prompt)
        if response_text:
            return _parse_llm_output(response_text)

    # Fallback to Mistral if available
    if MISTRAL_API_KEY:
        try:
            client = Mistral(api_key=MISTRAL_API_KEY)
            response = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=400
            )
            return _parse_llm_output(response.choices[0].message.content)
        except Exception as e:
            st.warning(f"⚠️ Mistral enrichment failed: {str(e)[:100]}")

    return {}

def generate_video_metadata(full_text: str, filename: str, domain_hint: str, video_url: Optional[str] = None) -> Dict:
    """Generate video-level metadata by combining LLM and KeyBERT.
       Returns a dictionary where all list/dict values are JSON-encoded strings
       for ChromaDB compatibility."""
    # LLM enrichment
    llm_meta = enrich_with_llm(full_text, domain_hint)

    # KeyBERT keywords for fallback/extra tags
    keywords = kw_model.extract_keywords(
        full_text[:5000],
        keyphrase_ngram_range=(1, 3),
        stop_words='english',
        top_n=20,
        diversity=0.7
    )
    keyword_tags = [kw[0] for kw in keywords]

    # Map keywords to IAB categories (simple keyword matching)
    iab_categories = set()
    for kw in keyword_tags + llm_meta.get("themes", []):
        kw_lower = kw.lower()
        for key, cats in st.session_state.iab_taxonomy.items():
            if key in kw_lower:
                iab_categories.update(cats)

    # Build the raw metadata (with lists/dicts)
    raw_meta = {
        "filename": filename,
        "domain_hint": domain_hint,
        "summary": llm_meta.get("summary", " ".join(keyword_tags[:3])),
        "themes": llm_meta.get("themes", keyword_tags[:8]),
        "entities": llm_meta.get("entities", []),
        "iab_categories": list(iab_categories) if iab_categories else ["IAB1"],
        "sentiment": llm_meta.get("sentiment", "neutral"),
        "tone": llm_meta.get("tone", "general"),
        "keywords": keyword_tags[:15],
        "confidence": {
            "summary": 0.9 if llm_meta.get("summary") else 0.6,
            "themes": 0.85 if llm_meta.get("themes") else 0.7,
            "entities": 0.8 if llm_meta.get("entities") else 0.6,
            "iab": 0.7 if iab_categories else 0.5,
            "sentiment": 0.8 if llm_meta.get("sentiment") else 0.5,
            "tone": 0.8 if llm_meta.get("tone") else 0.5,
        },
        "user_tags": [],
        "created_at": time.time(),
        "video_url": video_url or "",  # store YouTube URL if available
    }

    # Convert all list/dict values to JSON strings for ChromaDB storage
    encoded_meta = {}
    for key, value in raw_meta.items():
        if isinstance(value, (list, dict)):
            encoded_meta[key] = json.dumps(value)
        else:
            encoded_meta[key] = value

    return encoded_meta

# ------------------------------------------------------------------
# PROCESS FILE (with video metadata storage) – now also handles YouTube
# ------------------------------------------------------------------
def sliding_window_chunks_from_entries(entries: List[Dict], window: int, overlap: int) -> List[Dict]:
    """Convert subtitle entries (with start, end, text) into sliding window chunks."""
    if not entries:
        return []
    entries.sort(key=lambda x: x["start"])
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

def sliding_window_chunks_srt(filepath: str, window: int, overlap: int) -> List[Dict]:
    subs = pysrt.open(filepath)
    entries = []
    for sub in subs:
        text = sub.text.replace("\n", " ").strip()
        if text:
            entries.append({
                "start": sub.start.ordinal / 1000.0,
                "end": sub.end.ordinal / 1000.0,
                "text": text
            })
    return sliding_window_chunks_from_entries(entries, window, overlap)

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
    return sliding_window_chunks_from_entries(entries, window, overlap)

def process_video_source(source_type: str, source_identifier: str, window: int, overlap: int, domain_hint: str, video_url: Optional[str] = None) -> Tuple[Optional[int], Optional[str]]:
    """Common function to process a video from either file or YouTube URL."""
    if not COLLECTION_VALID:
        return None, "Database is invalid. Please reset."

    # For YouTube, source_identifier is video_id, for file it's file content (handled separately)
    if source_type == "youtube":
        video_id = source_identifier
        # Create a filename for display
        filename = f"YouTube_{video_id}"
        # Check if already indexed by hash (using video_id as hash)
        file_hash = hashlib.md5(video_id.encode()).hexdigest()
        existing = scene_collection.get(where={"file_hash": file_hash})
        if existing['ids']:
            st.info(f"✅ YouTube video {video_id} already indexed")
            if filename not in st.session_state.full_texts:
                all_texts = existing['documents']
                st.session_state.full_texts[filename] = " ".join(all_texts)
            return len(existing['ids']), None

        # Fetch subtitles
        entries = fetch_youtube_subtitles(video_id)
        if not entries:
            return None, "No subtitles found for this YouTube video."
        chunks = sliding_window_chunks_from_entries(entries, window, overlap)
    else:
        return None, "Invalid source type"

    if not chunks:
        return 0, "No valid subtitle content found."

    texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
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
        return None, f"Embedding mismatch: {len(embeddings)} vs {len(texts)}"

    # Per-chunk keyword tags (KeyBERT)
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

    scene_collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    full_text = " ".join(texts)
    st.session_state.full_texts[filename] = full_text

    # Generate and store video-level metadata
    with st.spinner("🧠 Generating AI video metadata..."):
        video_meta = generate_video_metadata(full_text, filename, domain_hint, video_url)
        # Store in video metadata collection
        video_meta_id = f"video_{filename}_{uuid.uuid4()}"
        video_meta_collection.upsert(
            ids=[video_meta_id],
            metadatas=[video_meta],
            documents=[full_text[:1000]]
        )
        # Also update session cache (decode JSON for internal use)
        decoded_meta = {}
        for key, value in video_meta.items():
            if isinstance(value, str) and value.startswith(('{', '[')):
                try:
                    decoded_meta[key] = json.loads(value)
                except:
                    decoded_meta[key] = value
            else:
                decoded_meta[key] = value
        st.session_state.enriched_metadata[filename] = decoded_meta

    # Rebuild BM25 index
    build_bm25_index()
    return len(valid_chunks), None

def process_file_optimized(uploaded_file, window: int, overlap: int, domain_hint: str = "general") -> Tuple[Optional[int], Optional[str]]:
    """Process uploaded file (SRT/VTT)."""
    if not COLLECTION_VALID:
        return None, "Database is invalid. Please reset."

    file_hash = get_file_hash(uploaded_file.getvalue())

    # Deduplication
    existing = scene_collection.get(where={"file_hash": file_hash})
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

    # Per-chunk keyword tags (KeyBERT)
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

    scene_collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    full_text = " ".join(texts)
    st.session_state.full_texts[filename] = full_text

    # Generate and store video-level metadata (no video URL)
    with st.spinner("🧠 Generating AI video metadata..."):
        video_meta = generate_video_metadata(full_text, filename, domain_hint)
        # Store in video metadata collection
        video_meta_id = f"video_{filename}_{uuid.uuid4()}"
        video_meta_collection.upsert(
            ids=[video_meta_id],
            metadatas=[video_meta],
            documents=[full_text[:1000]]
        )
        # Also update session cache (decode JSON for internal use)
        decoded_meta = {}
        for key, value in video_meta.items():
            if isinstance(value, str) and value.startswith(('{', '[')):
                try:
                    decoded_meta[key] = json.loads(value)
                except:
                    decoded_meta[key] = value
            else:
                decoded_meta[key] = value
        st.session_state.enriched_metadata[filename] = decoded_meta

    os.unlink(tmp_path)
    # Rebuild BM25 index
    build_bm25_index()
    return len(valid_chunks), None

def delete_file(filename: str):
    if not COLLECTION_VALID:
        return
    scene_collection.delete(where={"filename": filename})
    # Delete from video metadata collection
    existing_video = video_meta_collection.get(where={"filename": filename})
    if existing_video['ids']:
        video_meta_collection.delete(ids=existing_video['ids'])
    if filename in st.session_state.full_texts:
        del st.session_state.full_texts[filename]
    if filename in st.session_state.enriched_metadata:
        del st.session_state.enriched_metadata[filename]
    build_bm25_index()

# ------------------------------------------------------------------
# EDITABLE TAGS (per video)
# ------------------------------------------------------------------
def render_editable_tags(video_filename: str, current_tags: List[str], key_prefix: str):
    """Display tags as editable pills with add/remove."""
    tags = current_tags if current_tags else []
    edit_key = f"edit_tags_{video_filename}_{key_prefix}"
    if edit_key not in st.session_state:
        st.session_state[edit_key] = False

    col1, col2 = st.columns([10, 1])
    with col1:
        if tags:
            tag_html = " ".join([f'<span class="tag-pill editable" onclick="alert(\'Click edit button to modify\')">{tag}</span>' for tag in tags[:8]])
            st.markdown(f"✏️ User tags: {tag_html}", unsafe_allow_html=True)
        else:
            st.caption("No user tags")
    with col2:
        if st.button("✏️", key=f"edit_btn_{video_filename}_{key_prefix}", help="Edit tags"):
            st.session_state[edit_key] = True
            st.rerun()

    if st.session_state[edit_key]:
        with st.container():
            new_tag = st.text_input("Add tag", key=f"new_tag_{video_filename}_{key_prefix}")
            col_add, col_done = st.columns([1, 1])
            with col_add:
                if st.button("➕ Add", key=f"add_{video_filename}_{key_prefix}"):
                    if new_tag and new_tag not in tags:
                        tags.append(new_tag)
                        # Update in DB and session
                        video_meta = st.session_state.enriched_metadata.get(video_filename, {})
                        video_meta["user_tags"] = tags
                        # Encode user_tags as JSON for ChromaDB
                        encoded_meta = {
                            "user_tags": json.dumps(tags)
                        }
                        existing_video = video_meta_collection.get(where={"filename": video_filename})
                        if existing_video['ids']:
                            video_meta_collection.update(
                                ids=[existing_video['ids'][0]],
                                metadatas=[encoded_meta]
                            )
                        st.rerun()
            with col_done:
                if st.button("✓ Done", key=f"done_{video_filename}_{key_prefix}"):
                    st.session_state[edit_key] = False
                    st.rerun()
            if tags:
                st.write("**Current tags (click to remove):**")
                for tag in tags[:10]:
                    col_tag, col_rem = st.columns([5, 1])
                    col_tag.write(f"- {tag}")
                    if col_rem.button("✕", key=f"remove_{video_filename}_{tag}"):
                        tags.remove(tag)
                        video_meta = st.session_state.enriched_metadata.get(video_filename, {})
                        video_meta["user_tags"] = tags
                        encoded_meta = {
                            "user_tags": json.dumps(tags)
                        }
                        existing_video = video_meta_collection.get(where={"filename": video_filename})
                        if existing_video['ids']:
                            video_meta_collection.update(
                                ids=[existing_video['ids'][0]],
                                metadatas=[encoded_meta]
                            )
                        st.rerun()

# ------------------------------------------------------------------
# SIDEBAR UI – Clean, collapsible, minimal
# ------------------------------------------------------------------
with st.sidebar:
    col_logo, col_theme = st.columns([4, 1])
    with col_logo:
        st.title("🎥 Cairo")
    with col_theme:
        icon = "🌙" if st.session_state.theme == "dark" else "☀️"
        if st.button(icon, key="theme_toggle", help="Toggle theme"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()
    st.divider()

    with st.expander("🎛️ Scene Chunking", expanded=False):
        window_size = st.slider("Window (s)", 10, 60, 30, 5)
        overlap = st.slider("Overlap (s)", 5, 30, 15, 5)
    st.divider()

    st.subheader("📤 Upload Content")
    domain_hint = st.selectbox(
        "Domain",
        ["general", "sports", "news", "entertainment", "education", "technology"],
        index=0,
        help="Improves AI metadata"
    )

    # File upload section
    st.write("**Upload SRT/VTT files**")
    uploaded_files = st.file_uploader(
        "SRT/VTT",
        type=['srt', 'vtt'],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key="file_uploader"
    )
    if uploaded_files and COLLECTION_VALID:
        for f in uploaded_files:
            with st.status(f"Processing {f.name}...", expanded=False):
                cnt, err = process_file_optimized(f, window_size, overlap, domain_hint)
                if err:
                    st.error(err)
                else:
                    st.success(f"✅ {cnt} scenes")

    # YouTube section
    st.write("**Or ingest from YouTube**")
    youtube_url = st.text_input("YouTube URL", placeholder="https://youtu.be/...", key="youtube_url")
    if st.button("📥 Fetch YouTube Subtitles", use_container_width=True):
        if youtube_url:
            video_id = extract_youtube_video_id(youtube_url)
            if video_id:
                with st.status(f"Fetching subtitles for {video_id}...", expanded=False):
                    cnt, err = process_video_source("youtube", video_id, window_size, overlap, domain_hint, youtube_url)
                    if err:
                        st.error(err)
                    else:
                        st.success(f"✅ {cnt} scenes indexed")
            else:
                st.error("Invalid YouTube URL")
        else:
            st.warning("Please enter a YouTube URL")

    st.divider()

    # Library stats
    if COLLECTION_VALID:
        try:
            all_items = scene_collection.get()
            total_chunks = len(all_items['ids'])
            files_set = {m.get('filename') for m in all_items['metadatas'] if m.get('filename')}
        except:
            total_chunks, files_set = 0, set()
    else:
        total_chunks, files_set = 0, set()
    st.metric("Videos", len(files_set))
    st.metric("Scenes", f"{total_chunks:,}")

    if files_set:
        with st.expander("📁 Manage", expanded=False):
            for fname in sorted(files_set):
                c1, c2 = st.columns([5, 1])
                c1.caption(fname[:20] + "…" if len(fname) > 20 else fname)
                if c2.button("🗑️", key=f"del_{fname}"):
                    delete_file(fname)
                    st.rerun()
            if st.button("🗑️ Clear All"):
                for fname in files_set:
                    delete_file(fname)
                st.rerun()
    st.divider()

    with st.expander("⚙️ System", expanded=False):
        if st.button("🔄 Reset DB"):
            # Clear all cached resources (models and ChromaDB connections)
            st.cache_resource.clear()
            # Delete the physical database files
            persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
            client = chromadb.PersistentClient(path=persist_dir)
            try:
                client.delete_collection("cairo_subtitles")
                client.delete_collection("video_metadata")
            except:
                pass
            # Reset session state
            st.session_state.collection = None
            st.session_state.video_meta_collection = None
            st.session_state.enriched_metadata = {}
            st.session_state.full_texts = {}
            st.rerun()

# ------------------------------------------------------------------
# MAIN SEARCH AREA
# ------------------------------------------------------------------
st.title("🔍 Semantic Video Search")
st.markdown("Natural language search for your video library.")

# Search history
if st.session_state.search_history:
    with st.expander("📜 Recent Searches", expanded=False):
        cols = st.columns(min(5, len(st.session_state.search_history)))
        for i, (q, _) in enumerate(reversed(st.session_state.search_history[-5:])):
            if cols[i % 5].button(f"\"{q[:15]}…\"", key=f"hist_{i}"):
                st.session_state.last_query = q
                st.rerun()

# Main query input
col_q, col_adv = st.columns([5, 1])
with col_q:
    query = st.text_input(
        "Query",
        placeholder="e.g., 'who won India vs Australia match'",
        value=st.session_state.last_query,
        label_visibility="collapsed"
    )
with col_adv:
    if st.button("⚙️", help="Advanced options"):
        st.session_state.show_advanced = not st.session_state.show_advanced

# Advanced options
if st.session_state.show_advanced:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_results = st.number_input("Results", 1, 20, 5)
    with col2:
        diversify = st.checkbox("Diversify", value=True)
    with col3:
        lambda_param = st.slider("Diversity λ", 0.0, 1.0, 0.6, 0.1)
    with col4:
        min_score = st.slider("Min score", -5.0, 5.0, -3.0, 0.5)
    n_candidates = st.number_input("Candidates", 20, 200, 100, step=10)
else:
    n_results, diversify, lambda_param, min_score, n_candidates = 5, True, 0.6, -3.0, 100

# Filter by video
if files_set and len(files_set) > 1:
    with st.expander("🎯 Filter by Video", expanded=False):
        filter_choice = st.selectbox(
            "Narrow to",
            ["All files"] + sorted(files_set),
            index=0
        )
        st.session_state.filter_filename = filter_choice if filter_choice != "All files" else None
else:
    st.session_state.filter_filename = None

# Search execution
if query and COLLECTION_VALID:
    if query != st.session_state.last_query:
        st.session_state.search_history.append((query, datetime.now().strftime("%H:%M")))
        st.session_state.search_history = st.session_state.search_history[-20:]
    st.session_state.last_query = query

    domain = detect_domain(query)
    rewritten = rewrite_query_with_llm(query, domain)
    if rewritten != query and MISTRAL_API_KEY:
        st.caption(f"✨ Expanded: *{rewritten}*")

    cache_key = f"{rewritten}_{n_candidates}_{st.session_state.filter_filename}"
    if cache_key in st.session_state.result_cache:
        results = st.session_state.result_cache[cache_key]
        st.toast("⚡ Results from cache", icon="⚡")
    else:
        with st.status("🔍 Searching...", expanded=False):
            results = search_subtitles(
                rewritten,
                n_final=n_candidates,
                n_candidates=n_candidates * 2,
                diversify=False,
                lambda_param=lambda_param,
                min_score=min_score
            )
            if st.session_state.filter_filename:
                results = [r for r in results if r['metadata']['filename'] == st.session_state.filter_filename]
            if diversify:
                results = diversify_results(results, lambda_param=lambda_param, top_k=n_results)
            else:
                results = results[:n_results]
            st.session_state.result_cache[cache_key] = results
            if len(st.session_state.result_cache) > 50:
                oldest = min(st.session_state.result_cache.keys(), key=lambda k: st.session_state.result_cache[k][1] if isinstance(st.session_state.result_cache[k], tuple) else 0)
                del st.session_state.result_cache[oldest]

    if results:
        # Export all
        if len(results) > 0:
            export_df = pd.DataFrame([
                {
                    "File": r['metadata']['filename'],
                    "Timecode": r['metadata']['timecode'],
                    "Duration": r['metadata']['duration'],
                    "Confidence": f"{(r['score']+5)/20*100:.0f}%",
                    "Text": r['text'][:100]
                } for r in results
            ])
            csv = export_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export CSV",
                data=csv,
                file_name=f"cairo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

        # Paginate results
        if 'result_offset' not in st.session_state:
            st.session_state.result_offset = 0
        start = st.session_state.result_offset
        end = min(start + 5, len(results))
        for i, r in enumerate(results[start:end], start=start):
            meta = r['metadata']
            score = r['score']
            norm_score = max(0, min(1, (score + 5) / 20))

            with st.container():
                st.markdown(f'<div class="result-card">', unsafe_allow_html=True)
                col1, col2 = st.columns([4, 1])
                with col1:
                    fname = meta['filename'][:30] + "…" if len(meta['filename']) > 30 else meta['filename']
                    st.markdown(f"**{fname}**")
                    st.caption(f"⏱️ {meta['timecode']} | 📏 {meta['duration']}s")
                with col2:
                    st.markdown(f"<span class='confidence-badge'>{norm_score*100:.0f}%</span>", unsafe_allow_html=True)
                st.progress(norm_score)

                # Scene-level tags
                if meta.get("tags") and meta["tags"] != "general":
                    scene_tags = [t.strip() for t in meta["tags"].split(", ") if t.strip() and t != "general"][:6]
                    scene_tag_html = " ".join([f'<span class="tag-pill">{tag}</span>' for tag in scene_tags])
                    st.markdown(f"🎬 Scene tags: {scene_tag_html}", unsafe_allow_html=True)

                # Video-level metadata
                video_meta = st.session_state.enriched_metadata.get(meta['filename'], {})
                video_keywords = video_meta.get('keywords', [])
                if video_keywords:
                    kw_html = " ".join([f'<span class="tag-pill">{kw}</span>' for kw in video_keywords[:6]])
                    st.markdown(f"📌 Video keywords: {kw_html}", unsafe_allow_html=True)

                # IAB categories
                iab_cats = video_meta.get('iab_categories', [])
                if iab_cats:
                    iab_html = " ".join([f'<span class="iab-badge">{cat}</span>' for cat in iab_cats[:3]])
                    st.markdown(f"📺 {iab_html}", unsafe_allow_html=True)

                # Sentiment & Tone
                sentiment = video_meta.get('sentiment')
                tone = video_meta.get('tone')
                if sentiment or tone:
                    sentiment_html = ""
                    if sentiment:
                        sentiment_html += f'<span class="sentiment-tag">Sentiment: {sentiment}</span> '
                    if tone:
                        sentiment_html += f'<span class="sentiment-tag">Tone: {tone}</span>'
                    st.markdown(f"🎭 {sentiment_html}", unsafe_allow_html=True)

                # Editable user tags
                user_tags = video_meta.get('user_tags', [])
                render_editable_tags(meta['filename'], user_tags, f"scene_{i}")

                with st.expander("💬 Transcript"):
                    st.write(r['text'][:500] + ("…" if len(r['text']) > 500 else ""))

                col_b1, col_b2 = st.columns(2)
                with col_b1:
                    if st.button("▶️ Jump to", key=f"jump_{i}", use_container_width=True):
                        # If video has YouTube URL, open with timestamp
                        video_url = video_meta.get('video_url', '')
                        if video_url and 'youtu' in video_url:
                            # Build timestamp URL
                            t = int(meta['start'])  # ±3 seconds? We'll use exact start
                            # Ensure it's within bounds
                            # Create URL with timestamp
                            if 'watch?v=' in video_url:
                                base = video_url.split('&')[0]  # remove any existing query
                                jump_url = f"{base}&t={t}s"
                            elif 'youtu.be/' in video_url:
                                base = video_url.split('?')[0]
                                jump_url = f"{base}?t={t}s"
                            else:
                                jump_url = video_url
                            # Open in new tab
                            st.markdown(f'<meta http-equiv="refresh" content="0; url={jump_url}">', unsafe_allow_html=True)
                        else:
                            st.session_state.selected_result = r
                            st.rerun()
                with col_b2:
                    st.download_button(
                        "📄 JSON",
                        data=json.dumps(r, indent=2),
                        file_name=f"scene_{i}_{int(meta['start'])}.json",
                        mime="application/json",
                        use_container_width=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)

        # Load more
        if end < len(results):
            if st.button("⬇️ Load more", use_container_width=True):
                st.session_state.result_offset = end
                st.rerun()
        elif st.session_state.result_offset > 0:
            if st.button("⬆️ Show less", use_container_width=True):
                st.session_state.result_offset = 0
                st.rerun()
    else:
        st.warning("No results. Try a different query or adjust filters.")

elif COLLECTION_VALID and total_chunks == 0:
    st.info("👋 Upload subtitle files or a YouTube URL to start searching.")
else:
    st.info("✨ Enter a query above to search your library.")

# Selected moment detail (for non-YouTube videos, or if Jump to didn't redirect)
if st.session_state.selected_result:
    st.divider()
    st.subheader("🎬 Selected Moment")
    sel = st.session_state.selected_result
    meta_sel = sel['metadata']
    video_meta = st.session_state.enriched_metadata.get(meta_sel['filename'], {})
    video_url = video_meta.get('video_url', '')

    col_vid, col_info = st.columns([2, 1])
    with col_vid:
        if video_url and 'youtu' in video_url:
            # Embed YouTube player with start time
            t = int(meta_sel['start'])
            # Convert to embed URL
            if 'watch?v=' in video_url:
                video_id = video_url.split('v=')[1].split('&')[0]
                embed_url = f"https://www.youtube.com/embed/{video_id}?start={t}"
            elif 'youtu.be/' in video_url:
                video_id = video_url.split('/')[-1].split('?')[0]
                embed_url = f"https://www.youtube.com/embed/{video_id}?start={t}"
            else:
                embed_url = None
            if embed_url:
                st.video(embed_url)
            else:
                st.markdown('<div class="video-preview-placeholder">🎬 YouTube video (embed failed)</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="video-preview-placeholder">🎬 Video Preview (integration point)</div>', unsafe_allow_html=True)
        st.caption(f"⏱️ {meta_sel['timecode']} in {meta_sel['filename']}")
        if video_url:
            t = int(meta_sel['start'])
            if 'watch?v=' in video_url:
                base = video_url.split('&')[0]
                jump_url = f"{base}&t={t}s"
            elif 'youtu.be/' in video_url:
                base = video_url.split('?')[0]
                jump_url = f"{base}?t={t}s"
            else:
                jump_url = video_url
            st.link_button("▶️ Open in YouTube", jump_url, use_container_width=True)
    with col_info:
        st.markdown("**📝 Context**")
        st.write(sel['text'][:200] + "…")
        st.markdown("**🏷️ Video Metadata**")
        if video_meta:
            st.markdown(f"*Summary:* {video_meta.get('summary', 'N/A')}")
            st.markdown(f"*Themes:* {', '.join(video_meta.get('themes', [])[:5])}")
            st.markdown(f"*IAB:* {', '.join(video_meta.get('iab_categories', []))}")
            st.markdown(f"*Sentiment:* {video_meta.get('sentiment', 'N/A')}")
            st.markdown(f"*Tone:* {video_meta.get('tone', 'N/A')}")
        else:
            st.caption("No video metadata yet.")
        render_editable_tags(meta_sel['filename'], video_meta.get('user_tags', []), "detail")

    if st.button("← Back"):
        st.session_state.selected_result = None
        st.rerun()
