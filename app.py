import os
import time
import json
from typing import List, Dict, Any
import uuid

import streamlit as st
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from langchain_groq import ChatGroq
from huggingface_hub import InferenceClient

from utils import (
    extract_text_pymupdf,
    chunk_text,
    MiniLMEmbeddings,
    ensure_collection,
    upsert_chunks,
    search,
    build_prompt,
    confidence_from_scores,
)
# from loadenv
from dotenv import load_dotenv
load_dotenv()
# -----------------------------
# CONFIG / SECRETS
# -----------------------------
st.set_page_config(page_title="RAG Assistant (Qdrant + Groq + HF)", page_icon="📚", layout="wide")

# QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL", ""))
# QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY", ""))

# HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
# GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

# LANGCHAIN_TRACING_V2 = st.secrets.get("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
# LANGCHAIN_API_KEY = st.secrets.get("LANGCHAIN_API_KEY", os.getenv("LANGCHAIN_API_KEY", ""))

# COLLECTION = st.secrets.get("QDRANT_COLLECTION", os.getenv("QDRANT_COLLECTION", "rag_chunks"))
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "false")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
COLLECTION = os.getenv("QDRANT_COLLECTION", "rag_chunks")
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_SIZE = 384  # bge-m3 dim


# -----------------------------
# HELPERS
# -----------------------------
@st.cache_resource
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)


@st.cache_resource
def get_embedder():
    return MiniLMEmbeddings(hf_token=HF_TOKEN, model=EMBED_MODEL)


@st.cache_resource
def get_llm():
    if GROQ_API_KEY:
        return ChatGroq(api_key=GROQ_API_KEY, model_name="llama-3.3-70b-versatile", temperature=0.2, max_tokens=8000)
    else:
        # Fallback to HF Inference text generation (CPU free tier)
        return InferenceClient(model="mistralai/Mixtral-8x7B-Instruct-v0.1", token=HF_TOKEN, timeout=60)


def call_llm(llm, prompt: str) -> str:
    # If using ChatGroq (LangChain), call .invoke
    if isinstance(llm, ChatGroq):
        resp = llm.invoke(prompt)
        return resp.content
    # If using HF InferenceClient text generation
    if isinstance(llm, InferenceClient):
        out = llm.text_generation(prompt, max_new_tokens=400, temperature=0.2, do_sample=False, return_full_text=False)
        if isinstance(out, str):
            return out
        try:
            return out[0]["generated_text"]
        except Exception:
            return str(out)
    return "LLM not configured."


def ingest_pdf(uploaded_file, qdrant: QdrantClient, embedder, collection: str):
    file_bytes = uploaded_file.read()
    pages = extract_text_pymupdf(file_bytes)
    # qdrant.delete_collection(COLLECTION)

    ensure_collection(qdrant, collection, vector_size=VECTOR_SIZE, distance=qmodels.Distance.COSINE)

    doc_id = os.path.splitext(uploaded_file.name)[0] + "_" + str(int(time.time()))
    total_chunks = 0
    points = []

    for page in pages:
        page_no = page["page"]
        text = page["text"]
        chunks = chunk_text(text, max_chars=1800, overlap=200)
        if not chunks:
            continue
        vecs = embedder.embed(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
            pid = str(uuid.uuid4())   # ✅ Use UUID for valid Qdrant IDs
            payload = {
                "doc_id": doc_id,
                "doc_name": uploaded_file.name,
                "page": page_no,
                "chunk_id": i,
                "text": chunk,
                "upload_ts": int(time.time()),
            }
            points.append({"id": pid, "vector": vec, "payload": payload})
        total_chunks += len(chunks)

        # Upsert in batches to avoid huge payloads
        if len(points) >= 64:
            upsert_chunks(qdrant, collection, points)
            points = []

    if points:
        upsert_chunks(qdrant, collection, points)

    return {"doc_id": doc_id, "chunks": total_chunks, "pages": len(pages)}


def retrieve(qdrant: QdrantClient, embedder: MiniLMEmbeddings, collection: str, question: str, top_k: int = 5):
    qvec = embedder.embed_one(question)
    results = search(qdrant, collection, qvec, top_k=top_k)
    return results


# -----------------------------
# UI
# -----------------------------
st.title("🔎 RAG Assistant  \n+ Qdrant (vectorDB) \n+ HuggingFace Embedding (sentence-transformers/all-MiniLM-L6-v2) \n+ Groq (gemma2-9b-it) \n+ HuggingFace Inference (mistralai/Mixtral-8x7B-Instruct-v0.1) ")

with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("These should be set in Streamlit Secrets in production.")
    st.text_input("QDRANT_URL", value=QDRANT_URL, type="default", disabled=True)
    st.text_input("QDRANT_API_KEY", value="••••••••", type="password", disabled=True)
    st.text_input("HF_TOKEN", value="••••••••", type="password", disabled=True)
    st.text_input("GROQ_API_KEY", value="••••••••", type="password", disabled=True)
    st.markdown("---")
    st.caption(f"Collection: `{COLLECTION}` | Embeddings: `{EMBED_MODEL}` | Vector size: {VECTOR_SIZE}")

tab1, tab2 = st.tabs(["📤 Ingest PDFs", "💬 Ask Questions"])

with tab1:
    st.subheader("Upload and index PDFs")
    up = st.file_uploader("Select one or more PDF files", type=["pdf"], accept_multiple_files=True)
    if up and st.button("Ingest to Qdrant"):
        qdrant = get_qdrant()
        embedder = get_embedder()
        reports = []
        for f in up:
            with st.spinner(f"Ingesting {f.name} ..."):
                rep = ingest_pdf(f, qdrant, embedder, COLLECTION)
                reports.append(rep)
        st.success("Ingestion complete.")
        st.json(reports)

with tab2:
    st.subheader("Ask a question")
    question = st.text_area("Your question", placeholder="Ask about your uploaded documents...")
    k = st.slider("Top-k passages", min_value=3, max_value=8, value=5, step=1)
    if st.button("Search and Answer", type="primary") and question.strip():
        qdrant = get_qdrant()
        embedder = get_embedder()
        llm = get_llm()

        with st.spinner("Retrieving..."):
            results = retrieve(qdrant, embedder, COLLECTION, question, top_k=k)
        st.write(f"Retrieved {len(results)} passages.")
        for i, r in enumerate(results, 1):
            meta = r["payload"]
            with st.expander(f"Passage {i} • score={r['score']:.3f} • {meta.get('doc_name')} p.{meta.get('page')}"):
                st.write(meta.get("text", "")[:1200] + ("..." if len(meta.get("text",""))>1200 else ""))

        if not results:
            st.warning("No relevant passages found in Qdrant.")
        else:
            with st.spinner("Asking LLM..."):
                prompt = build_prompt(question, results)
                answer = call_llm(llm, prompt)
                conf = confidence_from_scores(results)
            st.markdown("### ✅ Answer")
            st.write(answer)
            st.markdown(f"**Confidence (avg similarity)**: `{conf:.3f}`")

            # Optional raw prompt view
            with st.expander("Show prompt (debug)"):
                st.code(prompt)

st.markdown("---")
st.caption("Powered by PyMuPDF • BAAI/bge-m3 • Qdrant Cloud • Groq • LangSmith-ready")

