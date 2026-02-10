# 🏦 BankDoc AI

**Local RAG system** for confidential document analysis. Upload PDFs, ask questions in natural language, get answers with page references. **100% offline** — no external APIs.

## Features

- **RAG Pipeline** - Retrieval-Augmented Generation with hybrid search
- **Smart Q&A** - Ask questions, get cited answers from PDFs
- **Hybrid Search** - Semantic embeddings + keyword matching (BM25)
- **Local LLM** - Ollama/Llama 3.2 (no cloud calls)
- **PDF Viewer** - Built-in viewer with text selection & highlighting
- **Async Processing** - Background tasks for extraction/embedding

## Quick Start


### Setup
```bash
# Clone & install
git clone https://github.com/lissovoyd/bankdoc-ai.git


# Configure (create .env file)
DATABASE_URL=postgresql://user:password@localhost/bankdoc
REDIS_URL=redis://localhost:6379/0

# Download LLM model
ollama pull llama3.2:3b
```

### Run
```bash
# Terminal 1: API
python main.py

# Terminal 2: Worker
celery -A celery_app worker --loglevel=info --pool=solo -I tasks

# Terminal 3: LLM (if not auto-running)
ollama serve
```

Open **http://localhost:8000**

## Tech Stack

**Backend:** FastAPI, PostgreSQL, Redis, Celery, ChromaDB  
**RAG:** sentence-transformers (embeddings), Ollama (LLM), BM25 (keyword search)  
**Frontend:** Vanilla JS, PDF.js  

## Architecture
```
Question → Retrieve (ChromaDB + BM25) → Augment (context) → Generate (Ollama)
                          ↓
        FastAPI → PostgreSQL + Redis + Celery
                → 100% Local Processing
```


