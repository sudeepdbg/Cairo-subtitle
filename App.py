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

# ------------------------------
#  Page configuration
# ------------------------------
st.set_page_config(
    page_title="Project Cairo - Subtitle Semantic Search",
    page_icon="🎥",
    layout="wide"
)

# Add debugging header (remove in production)
st.write("🔧 **Debug:** App initialized")

# ------------------------------
#  Cache heavy models (load once)
# ------------------------------
@st.cache_resource
def load_models():
    try:
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        kw_model = KeyBERT()
        return embed_model, kw_model
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

embedding_model, kw_model = load_models()

# ------------------------------
#  Initialize persistent ChromaDB
# ------------------------------
@st.cache_resource
def init_db():
    try:
        # Use persistent storage with better path handling for Streamlit Cloud
        db_path = "./chroma_db"  # Use relative path for Streamlit Cloud
        os.makedirs(db_path, exist_ok=True)
        
        client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection
        collection = client.get_or_create_collection(
            name="cairo_subtitles",
            metadata={"hnsw:space": "cosine"}
        )
        return collection
    except Exception as e:
        st.error(f"❌ Database initialization failed: {e}")
        st.stop()

collection = init_db()
st.write("✅ Database initialized")

# ------------------------------
#  Chunking functions
# ------------------------------
def chunk_subtitles_srt(filepath):
    try:
        subs = pysrt.open(filepath)
        chunks = []
        current = {"text": "", "start": None, "end": None}
        
        for sub in subs:
            if current["start"] is None:
                current["start"] = sub.start.ordinal / 1000.0
            current["text"] += " " + sub.text.replace("\n", " ")
            current["end"] = sub.end.ordinal / 1000.0
            
            if len(current["text"]) > 300 or sub.text.strip() == "":
                chunks.append(current)
                current = {"text": "", "start": None, "end": None}
        
        if current["text"]:
            chunks.append(current)
        return chunks
    except Exception as e:
        st.error(f"❌ SRT parsing failed: {e}")
        return []

def chunk_subtitles_vtt(filepath):
    try:
        subs = webvtt.read(filepath)
        chunks = []
        for sub in subs:
            chunks.append({
                "text": sub.text.replace("\n", " "),
                "start": sub.start_in_seconds,
                "end": sub.end_in_seconds
            })
        return chunks
    except Exception as e:
        st.error(f"❌ VTT parsing failed: {e}")
        return []

# ------------------------------
#  Process and store a file
# ------------------------------
def process_file(uploaded_file):
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        filename = uploaded_file.name
        
        # Chunk based on extension
        if filename.endswith('.srt'):
            chunks = chunk_subtitles_srt(tmp_path)
        elif filename.endswith('.vtt'):
            chunks = chunk_subtitles_vtt(tmp_path)
        else:
            os.unlink(tmp_path)
            return None, "Unsupported file type. Please upload .srt or .vtt"
        
        if not chunks:
            os.unlink(tmp_path)
            return None, "No subtitle chunks extracted from file"
        
        # Prepare texts and metadata
        texts = [c["text"].strip() for c in chunks if c["text"].strip()]
        
        if not texts:
            os.unlink(tmp_path)
            return None, "No valid text content found in file"
        
        metadatas = [{
            "filename": filename,
            "start": chunks[i].get("start", 0),
            "end": chunks[i].get("end", 0)
        } for i in range(len(texts))]
        
        # Generate embeddings
        embeddings = embedding_model.encode(texts, show_progress_bar=False).tolist()
        
        # Generate tags
        tags_list = []
        for text in texts:
            try:
                keywords = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_n=3
                )
                tag_string = ", ".join([kw[0] for kw in keywords]) if keywords else "general"
                tags_list.append(tag_string)
            except:
                tags_list.append("general")
        
        for i, tags in enumerate(tags_list):
            metadatas[i]["tags"] = tags
        
        # Generate IDs
        ids = [f"{filename}_{i}_{uuid.uuid4()}" for i in range(len(texts))]
        
        # Add to ChromaDB
        collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return len(texts), None
    except Exception as e:
        return None, f"Processing error: {str(e)}"

# ------------------------------
#  Search function
# ------------------------------
def search_subtitles(query, n_results=10):
    try:
        query_vec = embedding_model.encode([query]).tolist()
        results = collection.query(
            query_embeddings=query_vec,
            n_results=n_results
        )
        return results
    except Exception as e:
        st.error(f"❌ Search error: {e}")
        return None

# ------------------------------
#  UI – Sidebar: File Upload
# ------------------------------
st.sidebar.title("📁 Subtitle Library")
st.sidebar.markdown("Upload `.srt` or `.vtt` files to build your search index.")

uploaded_files = st.sidebar.file_uploader(
    "Choose files",
    type=['srt', 'vtt'],
    accept_multiple_files=True,
    key="file_uploader"
)

if uploaded_files:
    st.sidebar.markdown("**Processing Files...**")
    for uploaded_file in uploaded_files:
        try:
            # Check if already processed (simple dedup by filename)
            existing = collection.get(where={"filename": {"$eq": uploaded_file.name}})
            
            if existing and existing.get('ids') and len(existing['ids']) > 0:
                st.sidebar.info(f"✅ {uploaded_file.name} already indexed.")
            else:
                with st.sidebar.status(f"Processing {uploaded_file.name}..."):
                    chunk_count, error = process_file(uploaded_file)
                    if error:
                        st.sidebar.error(error)
                    else:
                        st.sidebar.success(f"✅ {uploaded_file.name} – {chunk_count} chunks indexed.")
        except Exception as e:
            st.sidebar.error(f"Failed to process {uploaded_file.name}: {str(e)}")

# Show collection stats
try:
    all_items = collection.get()
    total_chunks = len(all_items['ids']) if all_items.get('ids') else 0
    files_indexed = set([m.get('filename') for m in all_items.get('metadatas', []) if m.get('filename')])
except Exception as e:
    st.sidebar.error(f"Failed to load collection stats: {e}")
    total_chunks = 0
    files_indexed = set()

st.sidebar.markdown("---")
st.sidebar.metric("Total Chunks Indexed", total_chunks)
st.sidebar.metric("Files in Library", len(files_indexed))

# ------------------------------
#  Main area: Search
# ------------------------------
st.title("🎥 Project Cairo – Subtitle Semantic Search")
st.markdown("Ask a question in plain English – search across **all uploaded subtitle files**.")

col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input("🔍 Your question", placeholder="e.g., scenes where they talk about betrayal", key="search_query")
with col2:
    n_results = st.number_input("Results", min_value=1, max_value=20, value=5)

if query and len(query.strip()) > 0:
    with st.spinner("Searching..."):
        results = search_subtitles(query, n_results=n_results)
    
    if results is None:
        st.error("Search failed. Please try again.")
    elif not results.get('ids') or not results['ids'][0]:
        st.warning("No relevant scenes found. Try a different question or upload more files.")
    else:
        st.success(f"Found {len(results['ids'][0])} matching scenes")
        
        for i in range(len(results['ids'][0])):
            score = results['distances'][0][i]
            text = results['documents'][0][i]
            text_short = text[:250] + "..." if len(text) > 250 else text
            meta = results['metadatas'][0][i]
            
            with st.container():
                cols = st.columns([1, 5])
                with cols[0]:
                    st.markdown(f"**#{i+1}**")
                    st.caption(f"🎯 {score:.3f}")
                with cols[1]:
                    st.markdown(f"**{meta.get('filename', 'Unknown')}**")
                    st.caption(f"⏱️ {meta.get('start', 0):.1f}s → {meta.get('end', 0):.1f}s")
                    st.markdown(f"🏷️ **Tags:** `{meta.get('tags', 'N/A')}`")
                    st.markdown(f"💬 {text_short}")
            st.markdown("---")
else:
    if total_chunks == 0:
        st.info("👋 Start by uploading subtitle files in the sidebar.")
    else:
        st.info("Enter a search query above to find relevant scenes.")

# ------------------------------
#  Footer: Show indexed files
# ------------------------------
if total_chunks > 0:
    with st.expander("📚 Currently indexed files"):
        for f in sorted(files_indexed):
            st.write(f"- {f}")
