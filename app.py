import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="IR App", page_icon="🔎", layout="centered")

# ----------------------------
# Load model (cached)
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ----------------------------
# Load documents + embeddings (cached)
# ----------------------------
@st.cache_data
def load_data():
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]

    embeddings = np.load("embeddings.npy")  # shape: (num_docs, dim)
    return embeddings.astype(np.float32), documents

embeddings, documents = load_data()

# Safety check
if embeddings.shape[0] != len(documents):
    st.error(
        f"Mismatch! embeddings rows = {embeddings.shape[0]} but documents = {len(documents)}.\n"
        "Fix: documents.txt must have the same number of lines as embeddings.npy rows."
    )
    st.stop()

# ----------------------------
# Cosine similarity (NumPy only)
# ----------------------------
def cosine_sims(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    q = query_vec.astype(np.float32)
    D = doc_matrix.astype(np.float32)

    q_norm = np.linalg.norm(q) + 1e-10
    D_norm = np.linalg.norm(D, axis=1) + 1e-10
    sims = (D @ q) / (D_norm * q_norm)
    return sims

def retrieve_top_k(query: str, k: int):
    # real embedding for query
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

    sims = cosine_sims(q_emb, embeddings)
    top_idx = np.argsort(sims)[::-1][:k]
    return [(documents[i], float(sims[i])) for i in top_idx]

# ----------------------------
# UI
# ----------------------------
st.title("🔎 IR App (Streamlit) — Fine-Tuned Version")
st.caption("Uses real sentence embeddings (all-MiniLM-L6-v2) + cosine similarity ranking.")

st.write(f"📄 Loaded **{len(documents)}** documents.")

query = st.text_input("Enter your query:", placeholder="e.g., transformers, embeddings, streamlit...")
top_k = st.slider("Top K results", min_value=1, max_value=min(10, len(documents)), value=5)

show_scores = st.checkbox("Show similarity scores", value=True)

if st.button("Search"):
    if not query.strip():
        st.warning("Please type a query first.")
    else:
        results = retrieve_top_k(query.strip(), top_k)

        st.subheader("Results")
        for rank, (doc, score) in enumerate(results, start=1):
            st.markdown(f"**{rank}. {doc}**")
            if show_scores:
                st.caption(f"Similarity: {score:.4f}")
            st.divider()
