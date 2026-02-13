import streamlit as st
import pysrt
import webvtt
import uuid
import os
import tempfile
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from keybert import KeyBERT

# ------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Project Cairo – Subtitle Search",
    page_icon="🎥",
    layout="wide"
)

# ------------------------------------------------------------------
# CACHED MODELS (lightweight)
# ------------------------------------------------------------------
@st.cache_resource
def load_models():
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT()
    return embed_model, kw_model

embedding_model, kw_model = load_models()

# ------------------------------------------------------------------
# EPHEMERAL CHROMADB (no disk I/O, no permissions issues)
# ------------------------------------------------------------------
@st.cache_resource
def init_db():
    client = chromadb.Client(Settings(anonymized_telemetry=False))
    # Start fresh each session (no persistence – simpler)
    try:
        client.delete_collection("cairo_subtitles")
    except:
        pass
    return client.create_collection(
        name="cairo_subtitles",
        metadata={"hnsw:space": "cosine"}
    )

collection = init_db()

# ------------------------------------------------------------------
# CHUNKING – One subtitle per chunk (simple, fast)
# ------------------------------------------------------------------
def chunk_subtitles_srt(filepath):
    subs = pysrt.open(filepath)
    chunks = []
    for sub in subs:
        text = sub.text.replace("\n", " ").strip()
        if text:
            chunks.append({
                "text": text,
                "start": sub.start.ordinal / 1000.0,
                "end": sub.end.ordinal / 1000.0
            })
    return chunks

def chunk_subtitles_vtt(filepath):
    subs = webvtt.read(filepath)
    chunks = []
    for sub in subs:
        text = sub.text.replace("\n", " ").strip()
        if text:
            chunks.append({
                "text": text,
                "start": sub.start_in_seconds,
                "end": sub.end_in_seconds
            })
    return chunks

# ------------------------------------------------------------------
# PROCESS FILE
# ------------------------------------------------------------------
def process_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    filename = uploaded_file.name

    if filename.endswith('.srt'):
        chunks = chunk_subtitles_srt(tmp_path)
    elif filename.endswith('.vtt'):
        chunks = chunk_subtitles_vtt(tmp_path)
    else:
        os.unlink(tmp_path)
        return None, "Unsupported file type"

    texts = [c["text"] for c in chunks]
    metadatas = [{
        "filename": filename,
        "start": c["start"],
        "end": c["end"]
    } for c in chunks]

    # Generate embeddings
    embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()

    # Generate tags
    tags_list = []
    for text in texts:
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1,2),
            stop_words='english',
            top_n=3
        )
        tag_string = ", ".join([kw[0] for kw in keywords]) if keywords else "general"
        tags_list.append(tag_string)

    for i, tags in enumerate(tags_list):
        metadatas[i]["tags"] = tags

    ids = [f"{filename}_{i}_{uuid.uuid4()}" for i in range(len(chunks))]

    collection.add(
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

    os.unlink(tmp_path)
    return len(chunks), None

# ------------------------------------------------------------------
# SEARCH
# ------------------------------------------------------------------
def search_subtitles(query, n_results=10):
    query_vec = embedding_model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_vec,
        n_results=n_results
    )
    return results

# ------------------------------------------------------------------
# UI – SIDEBAR
# ------------------------------------------------------------------
st.sidebar.title("📁 Subtitle Library")
uploaded_files = st.sidebar.file_uploader(
    "Upload .srt or .vtt files",
    type=['srt', 'vtt'],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        existing = collection.get(where={"filename": uploaded_file.name})
        if existing['ids']:
            st.sidebar.info(f"✅ {uploaded_file.name} already indexed.")
        else:
            with st.sidebar.status(f"Processing {uploaded_file.name}..."):
                chunk_count, error = process_file(uploaded_file)
                if error:
                    st.sidebar.error(error)
                else:
                    st.sidebar.success(f"✅ {chunk_count} chunks indexed.")

# Collection stats
all_items = collection.get()
total_chunks = len(all_items['ids'])
files_indexed = set([m.get('filename') for m in all_items['metadatas'] if m.get('filename')])

st.sidebar.markdown("---")
st.sidebar.metric("Total Chunks", total_chunks)
st.sidebar.metric("Files", len(files_indexed))

# ------------------------------------------------------------------
# MAIN – SEARCH
# ------------------------------------------------------------------
st.title("🎥 Project Cairo – Subtitle Semantic Search")
st.markdown("Ask a question in plain English – search across **all uploaded subtitle files**.")

query = st.text_input("🔍 Your question", placeholder="e.g., when did Messi score his first goal?")
n_results = st.slider("Number of results", 1, 20, 5)

if query:
    with st.spinner("Searching..."):
        results = search_subtitles(query, n_results=n_results)

    if not results['ids'][0]:
        st.warning("No relevant scenes found. Try a different question.")
    else:
        st.success(f"Found {len(results['ids'][0])} matching scenes")
        for i in range(len(results['ids'][0])):
            score = results['distances'][0][i]
            text = results['documents'][0][i][:250] + "..."
            meta = results['metadatas'][0][i]
            with st.container():
                cols = st.columns([1, 5])
                with cols[0]:
                    st.markdown(f"**#{i+1}**")
                    st.caption(f"🎯 {score:.3f}")
                with cols[1]:
                    st.markdown(f"**{meta.get('filename', 'Unknown')}**")
                    st.caption(f"⏱️ {meta.get('start',0):.1f}s → {meta.get('end',0):.1f}s")
                    st.markdown(f"🏷️ **Tags:** `{meta.get('tags', 'N/A')}`")
                    st.markdown(f"💬 {text}")
                st.divider()
else:
    if total_chunks == 0:
        st.info("👋 Start by uploading subtitle files in the sidebar.")
    else:
        st.info("✨ Enter a question above to search your library.")
