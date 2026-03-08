"""Shared configuration and singleton instances."""

import os
from pathlib import Path

import chromadb
import redis
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from sentence_transformers import CrossEncoder

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
MODELS_CACHE_DIR = Path("models_cache")

# ---------------------------------------------------------------------------
# External service URLs
# ---------------------------------------------------------------------------
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8100"))

# ---------------------------------------------------------------------------
# Singletons
# ---------------------------------------------------------------------------
redis_client = redis.from_url(REDIS_URL, decode_responses=True)

embeddings = HuggingFaceEmbeddings(
    model_name="paraphrase-multilingual-MiniLM-L12-v2",
    cache_folder=str(MODELS_CACHE_DIR),
    encode_kwargs={"normalize_embeddings": True},
)

# Lazy ChromaDB client — created on first use so docker compose can start first
_chroma_client = None


def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return _chroma_client


def get_vectorstore(embedding_function=None) -> Chroma:
    """Create a fresh Chroma vectorstore instance via the shared HttpClient.

    Accepts an optional embedding_function override for Celery workers
    that lazily load their own model.
    """
    return Chroma(
        client=_get_chroma_client(),
        collection_name="bankdoc",
        embedding_function=embedding_function or embeddings,
    )


llm = OllamaLLM(model="llama3.2:3b", temperature=0.3, num_predict=350, top_p=0.9)
llm_json = OllamaLLM(model="llama3.2:3b", temperature=0.1, num_predict=500, top_p=0.9, format="json")

# ---------------------------------------------------------------------------
# Cross-encoder re-ranker (loaded once, works offline after first download)
# ---------------------------------------------------------------------------
reranker = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    max_length=512,
)

# Re-ranker confidence threshold: if the best chunk scores below this,
# the system will decline to answer (citation enforcement).
RERANKER_CONFIDENCE_THRESHOLD = float(os.getenv("RERANKER_THRESHOLD", "-4.0"))
