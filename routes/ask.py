"""Ask (RAG) and raw query endpoints."""

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from config import get_vectorstore
from database import get_db
from models import Document, DocumentStatus
from retrieval import build_rag_response
from schemas import (
    AskRequest,
    AskResponse,
    QueryHit,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()


# ---------------------------------------------------------------------------
# Raw semantic search (debugging / exploration)
# ---------------------------------------------------------------------------
@router.post("/query", response_model=QueryResponse)
def query_docs(payload: QueryRequest):
    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Query must not be empty")

    top_k = max(1, min(payload.top_k, 20))

    where: dict = {}
    if payload.doc_id is not None:
        where["doc_id"] = payload.doc_id
    if payload.department:
        where["department"] = payload.department
    if payload.corpus_id:
        where["corpus_id"] = payload.corpus_id

    vs = get_vectorstore()
    results = vs.similarity_search_with_score(q, k=top_k, filter=where or None)

    hits = [
        QueryHit(distance=float(score), document=doc.page_content, metadata=doc.metadata)
        for doc, score in results
    ]
    return QueryResponse(query=q, top_k=top_k, doc_id=payload.doc_id, hits=hits)


# ---------------------------------------------------------------------------
# RAG Q&A
# ---------------------------------------------------------------------------
@router.post("/docs/{doc_id}/ask", response_model=AskResponse)
def ask_document(doc_id: int, payload: AskRequest, db: Session = Depends(get_db)):
    """RAG Q&A: hybrid retrieval (semantic + BM25 via EnsembleRetriever) → Ollama LLM."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != DocumentStatus.EMBEDDED:
        raise HTTPException(status_code=400, detail=f"Document must be embedded. Status: {doc.status.value}")
    return build_rag_response(doc_id, payload, doc)


@router.post("/api/docs/{doc_id}/ask", response_model=AskResponse)
def api_ask_document(doc_id: int, payload: AskRequest, db: Session = Depends(get_db)):
    return ask_document(doc_id=doc_id, payload=payload, db=db)
