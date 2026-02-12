
import numpy as np
from sentence_transformers import SentenceTransformer

with open("documents.txt", "r", encoding="utf-8") as f:
    docs = [line.strip() for line in f if line.strip()]

model = SentenceTransformer("all-MiniLM-L6-v2")
emb = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)

np.save("embeddings.npy", emb)
print("Saved embeddings.npy:", emb.shape)
