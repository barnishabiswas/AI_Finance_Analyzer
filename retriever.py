"""
retriever.py
-------------
Member 2 — Vector Database
Fixed version — all bugs resolved

BUG FIX 8: Error handling for missing files
BUG FIX 9: fetch_k capped at reasonable number
"""

import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
MODEL_NAME    = "BAAI/bge-small-en-v1.5"

# ─────────────────────────────────────────────
# BUG FIX 8: Proper error handling for missing files
# ─────────────────────────────────────────────
print("[retriever] Loading FAISS index and metadata...")

_index_path    = os.path.join(VECTOR_DB_DIR, "faiss_index.bin")
_metadata_path = os.path.join(VECTOR_DB_DIR, "metadata.json")

# Check files exist
if not os.path.exists(_index_path):
    print(f"[retriever] ERROR: faiss_index.bin not found at {_index_path}")
    print("[retriever] Please run embed_chunks.py then build_faiss_index.py first.")
    _index = None
    _metadata = []
else:
    try:
        _index = faiss.read_index(_index_path)
        print(f"[retriever] Loaded index with {_index.ntotal} vectors")
    except Exception as e:
        print(f"[retriever] ERROR loading FAISS index: {e}")
        _index = None
        _metadata = []

if not os.path.exists(_metadata_path):
    print(f"[retriever] ERROR: metadata.json not found at {_metadata_path}")
    _metadata = []
else:
    try:
        with open(_metadata_path, "r", encoding="utf-8") as f:
            _metadata = json.load(f)
        print(f"[retriever] Loaded {len(_metadata)} metadata entries")
    except Exception as e:
        print(f"[retriever] ERROR loading metadata: {e}")
        _metadata = []

# Load embedding model
try:
    print("[retriever] Loading embedding model...")
    _model = SentenceTransformer(MODEL_NAME)
    print("[retriever] Ready.")
except Exception as e:
    print(f"[retriever] ERROR loading model: {e}")
    _model = None


def search(query: str, top_k: int = 5, company: str = None, year: str = None):
    """
    Search the vector DB for chunks relevant to query.

    Args:
        query:   the user's question
        top_k:   how many chunks to return (default 5)
        company: optional filter e.g. "apple"
        year:    optional filter e.g. "2023"

    Returns:
        List of dicts: {text, company, year, source, score}
    """
    # Return empty if not initialized
    if _index is None or _model is None or not _metadata:
        print("[retriever] WARNING: retriever not fully initialized")
        return []

    try:
        # Prefix query for bge model
        instructed_query = f"Represent this sentence for searching relevant passages: {query}"

        query_vector = _model.encode(
            [instructed_query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")

        # BUG FIX 9: Cap fetch_k at reasonable number (max 200)
        if company or year:
            fetch_k = min(top_k * 10, 200, _index.ntotal)
        else:
            fetch_k = min(top_k, _index.ntotal)

        scores, indices = _index.search(query_vector, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx >= len(_metadata):
                continue

            record = _metadata[idx]

            # Apply filters
            if company and record.get("company", "").lower() != company.lower():
                continue
            if year and str(record.get("year", "")) != str(year):
                continue

            results.append({
                "text":    record.get("text", ""),
                "company": record.get("company", "unknown"),
                "year":    record.get("year", "unknown"),
                "source":  record.get("source", "unknown"),
                "score":   float(score)
            })

            if len(results) >= top_k:
                break

        return results

    except Exception as e:
        print(f"[retriever] Search error: {e}")
        return []