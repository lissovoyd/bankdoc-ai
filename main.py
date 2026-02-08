# celery -A celery_app worker --loglevel=info --pool=solo
# bankdoc_env\Scripts\activate

import shutil
import os
import fitz  # PyMuPDF
import redis
import json
import time
import chromadb
import requests
import numpy as np
import re


from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.responses import FileResponse
from rank_bm25 import BM25Okapi
from enum import Enum as PyEnum
from fastapi import Form
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from typing import Optional
from celery_app import celery_app
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException # type: ignore
from sqlalchemy.orm import Session
from typing import List
from models import DocContent
from pathlib import Path
from database import engine, get_db, Base
from models import Document, DocumentStatus
from pydantic import BaseModel
from celery.result import AsyncResult
from tasks import extract_document as extract_document_task
from database import Base, engine
from models import Document, DocContent, DocumentStatus
Base.metadata.create_all(bind=engine)




# Pydantic schemas for responses
class DocumentResponse(BaseModel):
    id: int
    title: str
    filename: str
    uploaded_at: str
    status: str
    
    class Config:
        from_attributes = True

class PageContent(BaseModel):
    page_num: int
    text: str
    char_count: int

class DocumentContentResponse(BaseModel):
    doc_id: int
    title: str
    status: str
    pages: List[PageContent]

class QueryRequest(BaseModel):
    query: str
    top_k: int = 20
    doc_id: Optional[int] = None
    department: Optional[str] = None
    corpus_id: Optional[str] = None

class QueryHit(BaseModel):
    distance: float
    document: str
    metadata: dict

class QueryResponse(BaseModel):
    query: str
    top_k: int
    doc_id: Optional[int]
    hits: List[QueryHit]

class AskRequest(BaseModel):
    question: str

class SourceReference(BaseModel):
    page_num: int
    chunk_index: int
    text_excerpt: str
    relevance_score: float
    start_sentence: int | None = None
    end_sentence: int | None = None
    sentence_count: int | None = None

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceReference]
    doc_id: int
    doc_title: str

# class Document(Base):
#     __tablename__ = "documents"

#     id = Column(Integer, primary_key=True, index=True)
#     title = Column(String, nullable=False)
#     filename = Column(String, nullable=False)
#     status = Column(...)  # your enum
#     uploaded_at = Column(...)  # your existing

#     # ✅ new fields
#     department = Column(String, nullable=True)   # e.g. "finance", "legal", "service"
#     corpus_id = Column(String, nullable=True)  

class Department(str, PyEnum):
    unknown = "unknown"
    finance = "finance"
    legal = "legal"
    service = "service"
    hr = "hr"
    risk = "risk"
    it = "it"


def tokenize(text: str) -> list[str]:
    text = text.lower()
    # words + numbers + section-like tokens 8.3, 13.5, etc
    return re.findall(r"[а-яёa-z]+|\d+(?:\.\d+)+|\d+", text)


def call_ollama(prompt: str, model: str = "llama3.2:3b") -> str:
    """
    Call local Ollama LLM.
    """
    try:
        response = requests.post(
            'http://localhost:11434/api/generate',
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower = more focused answers
                    "top_p": 0.9,
                    "num_predict": 350   # Max tokens in response
                }
            },
            timeout=200
        )
        
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error: {response.status_code}"
    
    except requests.exceptions.ConnectionError:
        return "Error: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"



REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

CHROMA_DB_DIR = Path("chroma_db")
MODELS_CACHE_DIR = Path("models_cache")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="BankDoc AI", version="1.0.0")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Serve the main UI
@app.get("/")
def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Serve uploaded PDFs for viewing
@app.get("/uploads/{filename}")
def serve_pdf(filename: str):
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="application/pdf")

redis_client = redis.from_url(REDIS_URL, decode_responses=True)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))
collection = chroma_client.get_or_create_collection(name="bankdoc")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", cache_folder=str(MODELS_CACHE_DIR))





@app.get("/health")
def health_check():
    return {"status": "ok", "service": "BankDoc AI"}

@app.post("/docs", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: Department = Form(Department.unknown),
    corpus_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    """Upload a PDF document"""
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Save file to disk
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create database record
    doc = Document(
        title=file.filename.replace('.pdf', ''),
        filename=file.filename,
        status=DocumentStatus.PENDING,
        department=department.value,
        corpus_id=corpus_id
    )

    db.add(doc)
    db.commit()
    db.refresh(doc)
    
    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        filename=doc.filename,
        uploaded_at=str(doc.uploaded_at),
        status=doc.status.value
    )

@app.get("/api/docs", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    """List all documents"""
    # Add .order_by() to sort by newest first
    docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    
    return [
        DocumentResponse(
            id=doc.id,
            title=doc.title,
            filename=doc.filename,
            uploaded_at=str(doc.uploaded_at),
            status=doc.status.value
        )
        for doc in docs
    ]

@app.get("/docs/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    """Get a specific document"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        filename=doc.filename,
        uploaded_at=str(doc.uploaded_at),
        status=doc.status.value
    )


@app.get("/docs/{doc_id}/content", response_model=DocumentContentResponse)
def get_document_content(doc_id: int, db: Session = Depends(get_db)):
    """Get extracted content from document (with Redis caching)"""
    cache_key = f"doc:{doc_id}:content"
    
    # Try cache first
    start_time = time.time()
    cached = redis_client.get(cache_key)
    
    if cached:
        cache_time = time.time() - start_time
        print(f"Cache HIT for doc {doc_id} - took {cache_time*1000:.2f}ms")
        data = json.loads(cached)
        return DocumentContentResponse(**data)
    
    # Cache miss - query database
    print(f"Cache MISS for doc {doc_id} - querying database")
    
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc.status == DocumentStatus.PENDING:
        raise HTTPException(status_code=400, detail="Document not extracted yet")
    
    # Get all pages
    pages = db.query(DocContent).filter(DocContent.doc_id == doc_id).order_by(DocContent.page_num).all()
    
    response_data = {
        "doc_id": doc.id,
        "title": doc.title,
        "status": doc.status.value,
        "pages": [
            {"page_num": p.page_num, "text": p.text, "char_count": p.char_count}
            for p in pages
        ]
    }
    
    # Store in cache with 1 hour TTL
    redis_client.setex(cache_key, 3600, json.dumps(response_data))
    
    db_time = time.time() - start_time
    print(f"DB query completed - took {db_time*1000:.2f}ms")
    
    return DocumentContentResponse(**response_data)

@app.get("/docs/{doc_id}/status")
def get_document_status(doc_id: int, db: Session = Depends(get_db)):
    """Get document processing status"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "doc_id": doc.id,
        "status": doc.status.value,
        "title": doc.title,
        "filename": doc.filename
    }

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """Get Celery task status"""
    task = AsyncResult(task_id, app=celery_app)
    
    if task.state == 'PENDING':
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': 'Task is waiting in queue...'
        }
    elif task.state == 'STARTED':
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': 'Task is currently running...'
        }
    elif task.state == 'SUCCESS':
        response = {
            'task_id': task_id,
            'state': task.state,
            'result': task.result
        }
    elif task.state == 'FAILURE':
        response = {
            'task_id': task_id,
            'state': task.state,
            'error': str(task.info)
        }
    else:
        response = {
            'task_id': task_id,
            'state': task.state,
            'status': str(task.info)
        }
    
    return response




# ---- API aliases for frontend ----

@app.post("/api/docs", response_model=DocumentResponse)
async def api_upload_document(
    file: UploadFile = File(...),
    department: Department = Form(Department.unknown),
    corpus_id: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    return await upload_document(file=file, department=department, corpus_id=corpus_id, db=db)


@app.get("/api/docs/{doc_id}", response_model=DocumentResponse)
def api_get_document(doc_id: int, db: Session = Depends(get_db)):
    return get_document(doc_id=doc_id, db=db)


@app.post("/api/docs/{doc_id}/extract")
def api_extract_document(doc_id: int, db: Session = Depends(get_db)):
    return extract_document_async(doc_id=doc_id, db=db)


@app.post("/api/docs/{doc_id}/ask", response_model=AskResponse)
def api_ask_document(doc_id: int, payload: AskRequest, db: Session = Depends(get_db)):
    return ask_document(doc_id=doc_id, payload=payload, db=db)






@app.post("/docs/{doc_id}/extract")
def extract_document_async(doc_id: int, db: Session = Depends(get_db)):
    """Queue document extraction as background task"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc.status != DocumentStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Document status is {doc.status.value}, expected PENDING")
    
    # Enqueue the task
    task = extract_document_task.delay(doc_id)
    
    return {
        "doc_id": doc_id,
        "task_id": task.id,
        "status": "queued",
        "message": "Extraction started in background. Use /tasks/{task_id} to check progress."
    }


@app.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest):
    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    top_k = max(1, min(payload.top_k, 20))  # clamp

    query_emb = embedding_model.encode(
        [q],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()

    filters = []

    if payload.doc_id is not None:
        filters.append({"doc_id": payload.doc_id})

    if payload.department:
        filters.append({"department": payload.department})

    if payload.corpus_id:
        filters.append({"corpus_id": payload.corpus_id})

    where = None
    if len(filters) == 1:
        where = filters[0]
    elif len(filters) > 1:
        where = {"$and": filters}



    results = collection.query(
        query_embeddings=query_emb,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]

    for doc_text, meta, dist in zip(docs, metas, dists):
        hits.append(QueryHit(
            distance=float(dist),
            document=doc_text,
            metadata=meta
        ))

    return QueryResponse(
        query=q,
        top_k=top_k,
        doc_id=payload.doc_id,
        hits=hits
    )


@app.post("/docs/{doc_id}/ask", response_model=AskResponse)
def ask_document(doc_id: int, payload: AskRequest, db: Session = Depends(get_db)):
    """
    Ask a question about a document using RAG with hybrid search.
    
    Pipeline:
    1. Retrieve: Find relevant chunks from ChromaDB (semantic search)
    2. Rerank: Apply BM25 keyword scoring (hybrid)
    3. Augment: Build prompt with top-ranked context
    4. Generate: Use Ollama LLM to answer
    """
    # Validate document exists and is embedded
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if doc.status != DocumentStatus.EMBEDDED:
        raise HTTPException(
            status_code=400, 
            detail=f"Document must be embedded first. Current status: {doc.status.value}"
        )
    
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    # Step 1: RETRIEVE - Get relevant chunks from ChromaDB (semantic search)
    print(f"[RAG] Question: {question}")
    
    # Embed the question
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).tolist()
    
    # Query ChromaDB for relevant chunks (filter by doc_id)
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=20,  # Get more candidates for reranking
        where={"doc_id": doc_id},
        include=["documents", "metadatas", "distances"]
    )
    
    # Extract results
    chunks = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    
    if not chunks:
        raise HTTPException(
            status_code=404,
            detail="No relevant content found in document"
        )
    
    print(f"[RAG] Retrieved {len(chunks)} chunks from semantic search")
    
    # Step 1.5: RERANK with BM25 (hybrid search)
    # Step 1.5: HYBRID candidate pool (semantic + BM25) + rerank

    # --- A) Get ALL chunks for this doc for BM25 ---
    doc_chunks = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"]   # <-- remove "ids"
    )

    docs_all = doc_chunks["documents"]
    metas_all = doc_chunks["metadatas"]
    ids_all  = doc_chunks["ids"]            # <-- ids exist here automatically


    if not docs_all:
        raise HTTPException(status_code=404, detail="No chunks found in Chroma for this doc")

    # Build BM25 over ALL doc chunks
    tokenized_all = [tokenize(t) for t in docs_all]
    bm25 = BM25Okapi(tokenized_all)

    q_tokens = tokenize(question)
    bm25_raw = bm25.get_scores(q_tokens)  # numpy array

    # Normalize BM25 to 0..1
    bm25_max = float(bm25_raw.max()) if len(bm25_raw) else 0.0
    bm25_norm = (bm25_raw / bm25_max) if bm25_max > 0 else np.zeros(len(bm25_raw))

    # Take top BM25 candidates
    BM25_K = 30
    top_bm25_idx = np.argsort(bm25_raw)[::-1][:BM25_K]

    # --- B) Build a dict for semantic candidates (by chunk_id) ---
    # NOTE: your semantic query does not include ids. We’ll rebuild ids from metadata.
    # If you already store chunk_id in metadata, use that.
    semantic_by_id = {}

    for chunk_text, meta, dist in zip(chunks, metadatas, distances):
        chunk_id = meta.get("chunk_id")
        if not chunk_id:
            # fallback if chunk_id not stored: reconstruct (must match how you created ids)
            chunk_id = f"doc_{meta['doc_id']}_page_{meta['page_num']}_chunk_{meta['chunk_index']}"

        semantic_by_id[chunk_id] = {
            "chunk": chunk_text,
            "meta": meta,
            "dist": float(dist),
        }

    # --- C) Union candidates: semantic set + bm25 top set ---
    candidates = {}

    # add semantic
    for cid, item in semantic_by_id.items():
        candidates[cid] = {
            "chunk": item["chunk"],
            "meta": item["meta"],
            "dist": item["dist"],
            "bm25": 0.0,  # will fill if present
        }

    # add BM25 top
    for idx in top_bm25_idx:
        cid = ids_all[idx]
        if cid not in candidates:
            candidates[cid] = {
                "chunk": docs_all[idx],
                "meta": metas_all[idx],
                "dist": None,     # no semantic distance for BM25-only
                "bm25": float(bm25_norm[idx]),
            }
        else:
            candidates[cid]["bm25"] = float(bm25_norm[idx])

    # also fill BM25 for semantic candidates if they exist in ids_all
    # (fast map from id -> index)
    id_to_index = {cid: i for i, cid in enumerate(ids_all)}
    for cid in list(candidates.keys()):
        if cid in id_to_index:
            candidates[cid]["bm25"] = float(bm25_norm[id_to_index[cid]])

    # --- D) Score candidates ---
    final_scores = []
    SEM_W = 0.85
    BM25_W = 0.15

    for cid, item in candidates.items():
        dist = item["dist"]
        semantic_score = 0.0
        if dist is not None:
            semantic_score = 1.0 / (1.0 + dist)

        bm25_score = item["bm25"]

        hybrid_score = SEM_W * semantic_score + BM25_W * bm25_score

        final_scores.append({
            "chunk_id": cid,
            "chunk": item["chunk"],
            "metadata": item["meta"],
            "distance": dist,
            "semantic_score": semantic_score,
            "bm25_score": bm25_score,
            "hybrid_score": hybrid_score,
        })

    final_scores.sort(key=lambda x: x["hybrid_score"], reverse=True)

    # Take top 10
    top_chunks = final_scores[:7]

    print(f"[RAG] Hybrid reranking applied (union candidates = {len(final_scores)}):")
    for i, sc in enumerate(top_chunks, start=1):
        print(f"  [{i}] Hybrid: {sc['hybrid_score']:.3f} "
            f"(semantic: {sc['semantic_score']:.3f}, bm25: {sc['bm25_score']:.3f}) "
            f"| Page {sc['metadata'].get('page_num')} | id={sc['chunk_id']}")

    
    # Extract reranked chunks for context
    chunks = [sc['chunk'] for sc in top_chunks]
    metadatas = [sc['metadata'] for sc in top_chunks]
    distances = [sc['distance'] for sc in top_chunks]
    
    # Step 2: AUGMENT - Build prompt with context
    context = ""
    for i, (chunk, meta) in enumerate(zip(chunks, metadatas)):
        context += f"\n[Страница {meta['page_num']}, Фрагмент {meta['chunk_index']}]\n{chunk}\n"
    
    # Prompt engineering: instruction + context + question
    prompt = f"""Ты — юридический ассистент. Отвечай на вопросы о договоре на основе предоставленного контекста.

                ИНСТРУКЦИИ:
                - Используй ТОЛЬКО информацию из контекста
                - Если ответ есть в контексте, процитируй точно (с номером раздела, если есть)
                - Если в контексте нет полного ответа, скажи "Неполная информация" и укажи, что найдено
                - ВСЕГДА указывай номера страниц и разделов документа
                - Для юридических терминов используй формулировки из документа

                КОНТЕКСТ ИЗ ДОКУМЕНТА:
                {context}

                ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

                ОТВЕТ (с указанием раздела и страницы):"""
    
    print(f"[RAG] Prompt length: {len(prompt)} chars")
    
    # Step 3: GENERATE - Call Ollama LLM
    print(f"[RAG] Calling Ollama...")
    answer = call_ollama(prompt)
    
    if answer.startswith("Error"):
        raise HTTPException(status_code=500, detail=answer)
    
    print(f"[RAG] Answer generated ({len(answer)} chars)")
    
    # Build source references (using hybrid scores)
    sources = []
    for sc in top_chunks:
        meta = sc["metadata"]
        sources.append(SourceReference(
            page_num=meta["page_num"],
            chunk_index=meta["chunk_index"],
            text_excerpt=sc["chunk"][:750] + "..." if len(sc["chunk"]) > 750 else sc["chunk"],
            relevance_score=round(sc["hybrid_score"], 4),
            start_sentence=meta.get("start_sentence"),
            end_sentence=meta.get("end_sentence"),
            sentence_count=meta.get("sentence_count"),
        ))
    
    return AskResponse(
        question=question,
        answer=answer.strip(),
        sources=sources,
        doc_id=doc.id,
        doc_title=doc.title
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

