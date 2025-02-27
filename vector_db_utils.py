from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model globally
try:
    MODEL = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"Failed to load the embedding model: {e}")

def validate_text_input(text):
    if isinstance(text, str):
        text = [text]
    if not isinstance(text, list) or not all(isinstance(t, str) for t in text):
        raise ValueError("Input must be a string or a list of strings.")
    return [t.strip() for t in text if t.strip()]

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    chunks = validate_text_input(chunks)
    embeddings = MODEL.encode(chunks, convert_to_tensor=False)
    embeddings = np.array(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-10)  # Add small value for numerical stability
    return normalized_embeddings

def store_in_faiss(embeddings, db_file="vector_db_cosine.index"):
    try:
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Use inner product (for cosine similarity with normalized vectors)
        index.add(embeddings)
        faiss.write_index(index, str(db_file)) #cast to string for cross-platform
        print(f"FAISS index saved to {db_file}")
        return index
    except Exception as e:
        raise RuntimeError(f"Failed to store FAISS index: {e}")

def load_faiss_index(db_file):
    try:
        db_file = Path(db_file).resolve()
        return faiss.read_index(str(db_file))
    except Exception as e:
        raise RuntimeError(f"Failed to load FAISS index: {e}")

def query_faiss_index(query_text, vector_index, chunks, model_name="all-MiniLM-L6-v2", top_k=2):
    try:
        query_text = validate_text_input(query_text)
        if not query_text:
            raise ValueError("Query text cannot be empty.")
        query_embedding = MODEL.encode(query_text, convert_to_tensor=False)
        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10)
        distances, indices = vector_index.search(query_embedding, top_k)
        results = [(chunks[idx], distances[0][i], idx) for i, idx in enumerate(indices[0])]
        return results
    except Exception as e:
        raise RuntimeError(f"Failed to query FAISS index: {e}")