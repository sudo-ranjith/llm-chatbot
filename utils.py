\
import io
import time
import math
import uuid
from typing import List, Dict, Any, Tuple, Optional

import fitz  # PyMuPDF
import numpy as np
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels


# -----------------------------
# PDF TEXT EXTRACTION (PyMuPDF)
# -----------------------------
def extract_text_pymupdf(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Returns a list of pages: [{"page": i, "text": "..."}]
    Uses PyMuPDF for robust extraction (multi-column-friendly).
    """
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text") or ""
        pages.append({"page": i + 1, "text": text})
    return pages


# -----------------------------
# CHUNKING
# -----------------------------
def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    """
    Simple, reliable character-based chunking with overlap.
    ~1800 chars ≈ 500-700 tokens (varies).
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap if end - overlap > start else end
    return chunks


# -----------------------------
# EMBEDDINGS (Hugging Face bge-m3)
# -----------------------------
class BGEEmbeddings:
    """
    Uses HuggingFace Inference API to get embeddings for BAAI/bge-m3.
    Normalizes vectors to unit length (recommended for cosine).
    """
    def __init__(self, hf_token: str, model: str = "BAAI/bge-m3", timeout: float = 30.0):
        self.client = InferenceClient(model=model, token=hf_token, timeout=timeout)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        # HF InferenceClient embeddings endpoint
        # InferenceClient has a .feature_extraction() for some models; for consistency,
        # we'll call the /embeddings task using .post (fallback) if needed.
        vecs = []
        for t in texts:
            t = t.strip()
            if not t:
                vecs.append([0.0] * 1024)  # bge-m3 is 1024-dim
                continue
            # Newer huggingface_hub clients provide .embeddings() helper; fallback to .post if not.
            try:
                out = self.client.post(json={"inputs": t, "truncate": True, "parameters": {"task": "embeddings"}})
                # Expected format: {'embeddings': {'dtype': '...', 'shape': [1, 1024], 'data': [...] } }
                if isinstance(out, dict) and "embeddings" in out:
                    vec = out["embeddings"]["data"]
                else:
                    # Some deployments return list[float] directly
                    vec = out if isinstance(out, list) else out[0]
            except Exception:
                # Fallback to feature_extraction API
                vec = self.client.feature_extraction(t)
                if isinstance(vec, list) and isinstance(vec[0], list):
                    vec = vec[0]

            # Normalize
            arr = np.array(vec, dtype=np.float32)
            norm = np.linalg.norm(arr) + 1e-12
            arr = arr / norm
            vecs.append(arr.tolist())
        return vecs

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]


# -----------------------------
# QDRANT HELPERS
# -----------------------------
def ensure_collection(
    client: QdrantClient,
    collection: str,
    vector_size: int = 384,

    distance: qmodels.Distance = qmodels.Distance.COSINE,
) -> None:
    try:
        existing = client.get_collection(collection)
        # If exists, do nothing
        _ = existing.status
    except Exception:
        client.recreate_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=distance),
        )

def upsert_chunks(client: QdrantClient, collection: str, points: List[Dict[str, Any]]) -> None:
    """
    points: [{"id": str, "vector": [...], "payload": {...}}, ...]
    """
    client.upsert(
        collection_name=collection,
        points=[
            qmodels.PointStruct(
                id=p["id"],
                vector=p["vector"],
                payload=p["payload"]
            )
            for p in points
        ],
    )


def search(
    client: QdrantClient,
    collection: str,
    query_vector: List[float],
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    res = client.search(
        collection_name=collection,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )
    out = []
    for r in res:
        out.append({
            "id": r.id,
            "score": float(r.score),
            "payload": r.payload,
        })
    return out


# -----------------------------
# PROMPT ASSEMBLY
# -----------------------------
def build_prompt(question: str, passages: List[Dict[str, Any]]) -> str:
    ctx_blocks = []
    for i, p in enumerate(passages, 1):
        meta = p["payload"]
        title = meta.get("doc_name", "document")
        page = meta.get("page", "?")
        ctx_blocks.append(f"[{i}] Source: {title} page {page}\n{meta.get('text','')}\n")
    context = "\n\n".join(ctx_blocks)
    system_rules = (
        "You are a helpful RAG assistant. Answer ONLY from the provided context.\n"
        "If the answer is not in the context, say 'I don't know based on the provided documents.'\n"
        "Return citations as [number] where the number refers to the source block.\n"
    )
    user_block = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    return system_rules + "\n" + user_block


# -----------------------------
# SIMPLE CONFIDENCE (AVG SCORE)
# -----------------------------
def confidence_from_scores(results: List[Dict[str, Any]]) -> float:
    if not results:
        return 0.0
    scores = [r["score"] for r in results]
    return float(sum(scores) / max(1, len(scores)))


class MiniLMEmbeddings:
    def __init__(self, hf_token: str, model: str = "sentence-transformers/all-MiniLM-L6-v2", timeout: float = 30.0):
        from huggingface_hub import InferenceClient
        self.client = InferenceClient(model=model, token=hf_token, timeout=timeout)
        self.model = model

    def embed(self, texts: List[str]) -> List[List[float]]:
        vecs = []
        for t in texts:
            if not t.strip():
                vecs.append([0.0] * 384)
                continue
            out = self.client.feature_extraction(t)
            if isinstance(out[0], list):
                vecs.append(out[0])  # 2D → 1D
            else:
                vecs.append(out)
        return vecs

    def embed_one(self, text: str) -> List[float]:
        return self.embed([text])[0]
