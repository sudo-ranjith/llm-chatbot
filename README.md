# RAG Streamlit App (Qdrant + bge-m3 + Groq)

## Quick Start
1) Create a Qdrant Cloud cluster and get `QDRANT_URL`, `QDRANT_API_KEY`.
2) Get a Hugging Face token (`HF_TOKEN`) with access to Inference API.
3) (Optional but recommended) Get a Groq API key (`GROQ_API_KEY`) for fast LLM.
4) Deploy on Streamlit Community Cloud (recommended).

### Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Streamlit Secrets
Add the following in your Streamlit Cloud secrets:

```toml
QDRANT_URL = "https://YOUR-CLUSTER-URL.qdrant.tech"
QDRANT_API_KEY = "YOUR_QDRANT_KEY"

HF_TOKEN = "YOUR_HF_TOKEN"
GROQ_API_KEY = "YOUR_GROQ_KEY"

# Optional LangSmith
LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "YOUR_LANGSMITH_KEY"
```

## Notes
- PDF extraction uses **PyMuPDF** for robust text.
- Embeddings use **BAAI/bge-m3** via Hugging Face Inference API (1024-dim).
- Vector search only (no hybrid) in Qdrant.
- LLM defaults to **Groq** (Llama3 70B). Falls back to **HF Inference** if GROQ not set.
- Citations are included via source block numbers in the prompt.
