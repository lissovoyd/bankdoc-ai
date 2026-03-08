"""Document CRUD, extraction, and content endpoints."""

import json
import shutil
import time
import logging
from typing import List, Optional

from fastapi import APIRouter, File, Form, UploadFile, Depends, HTTPException
from sqlalchemy.orm import Session
from celery.result import AsyncResult

from celery_app import celery_app
from config import UPLOAD_DIR, redis_client, get_vectorstore
from database import get_db
from models import Document, DocContent, DocumentStatus
from schemas import (
    Department,
    DocumentContentResponse,
    DocumentResponse,
)
from tasks import extract_document as extract_document_task

logger = logging.getLogger(__name__)

router = APIRouter()


def _doc_response(doc: Document) -> DocumentResponse:
    return DocumentResponse(
        id=doc.id,
        title=doc.title,
        filename=doc.filename,
        uploaded_at=str(doc.uploaded_at),
        status=doc.status.value,
    )


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
async def _upload(file: UploadFile, department: Department, corpus_id: Optional[str], db: Session):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    file_path = UPLOAD_DIR / file.filename
    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)
    doc = Document(
        title=file.filename.replace(".pdf", ""),
        filename=file.filename,
        status=DocumentStatus.PENDING,
        department=department.value,
        corpus_id=corpus_id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return _doc_response(doc)


@router.post("/docs", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
    department: Department = Form(Department.unknown),
    corpus_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    return await _upload(file, department, corpus_id, db)


@router.post("/api/docs", response_model=DocumentResponse)
async def api_upload_document(
    file: UploadFile = File(...),
    department: Department = Form(Department.unknown),
    corpus_id: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    return await _upload(file, department, corpus_id, db)


# ---------------------------------------------------------------------------
# List / Get
# ---------------------------------------------------------------------------
@router.get("/api/docs", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    return [_doc_response(d) for d in docs]


@router.get("/docs/{doc_id}", response_model=DocumentResponse)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_response(doc)


@router.get("/api/docs/{doc_id}", response_model=DocumentResponse)
def api_get_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _doc_response(doc)


@router.get("/docs/{doc_id}/status")
def get_document_status(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"doc_id": doc.id, "status": doc.status.value, "title": doc.title, "filename": doc.filename}


@router.get("/docs/{doc_id}/content", response_model=DocumentContentResponse)
def get_document_content(doc_id: int, db: Session = Depends(get_db)):
    """Return per-page text with Redis caching."""
    cache_key = f"doc:{doc_id}:content"
    t0 = time.time()
    cached = redis_client.get(cache_key)
    if cached:
        logger.info("Cache HIT  doc=%d (%.1fms)", doc_id, (time.time() - t0) * 1000)
        return DocumentContentResponse(**json.loads(cached))

    logger.info("Cache MISS doc=%d", doc_id)
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status == DocumentStatus.PENDING:
        raise HTTPException(status_code=400, detail="Document not extracted yet")

    pages = db.query(DocContent).filter(DocContent.doc_id == doc_id).order_by(DocContent.page_num).all()
    data = {
        "doc_id": doc.id,
        "title": doc.title,
        "status": doc.status.value,
        "pages": [{"page_num": p.page_num, "text": p.text, "char_count": p.char_count} for p in pages],
    }
    redis_client.setex(cache_key, 3600, json.dumps(data))
    logger.info("DB query doc=%d (%.1fms)", doc_id, (time.time() - t0) * 1000)
    return DocumentContentResponse(**data)


# ---------------------------------------------------------------------------
# Extraction / task endpoints
# ---------------------------------------------------------------------------
def _enqueue_extract(doc_id: int, db: Session):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc.status != DocumentStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Status is {doc.status.value}, expected PENDING")
    task = extract_document_task.delay(doc_id)
    return {
        "doc_id": doc_id,
        "task_id": task.id,
        "status": "queued",
        "message": "Extraction started. Poll /tasks/{task_id} for progress.",
    }


@router.post("/docs/{doc_id}/extract")
def extract_document_async(doc_id: int, db: Session = Depends(get_db)):
    return _enqueue_extract(doc_id, db)


@router.post("/api/docs/{doc_id}/extract")
def api_extract_document(doc_id: int, db: Session = Depends(get_db)):
    return _enqueue_extract(doc_id, db)


@router.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    task = AsyncResult(task_id, app=celery_app)
    base = {"task_id": task_id, "state": task.state}
    if task.state == "SUCCESS":
        return {**base, "result": task.result}
    if task.state == "FAILURE":
        return {**base, "error": str(task.info)}
    return {**base, "status": str(task.info) if task.info else task.state}


# ---------------------------------------------------------------------------
# Delete document (+ content, embeddings, file, cache)
# ---------------------------------------------------------------------------
def _delete_doc(doc_id: int, db: Session):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # 1. Delete ChromaDB embeddings for this doc
    try:
        vs = get_vectorstore()
        result = vs.get(where={"doc_id": doc_id})
        ids_to_delete = result.get("ids", [])
        if ids_to_delete:
            vs.delete(ids=ids_to_delete)
            logger.info("Deleted %d embeddings for doc_id=%d", len(ids_to_delete), doc_id)
    except Exception as e:
        logger.warning("ChromaDB cleanup failed for doc_id=%d: %s", doc_id, e)

    # 2. Delete Redis cache
    try:
        redis_client.delete(f"doc:{doc_id}:content")
    except Exception:
        pass

    # 3. Delete PDF file from disk
    try:
        file_path = UPLOAD_DIR / doc.filename
        if file_path.exists():
            file_path.unlink()
    except Exception as e:
        logger.warning("File cleanup failed for doc_id=%d: %s", doc_id, e)

    # 4. Delete DB records (content rows first, then document)
    db.query(DocContent).filter(DocContent.doc_id == doc_id).delete()
    db.query(Document).filter(Document.id == doc_id).delete()
    db.commit()

    logger.info("Deleted document doc_id=%d title=%s", doc_id, doc.title)
    return {"doc_id": doc_id, "deleted": True}


@router.delete("/api/docs/{doc_id}")
def api_delete_document(doc_id: int, db: Session = Depends(get_db)):
    return _delete_doc(doc_id, db)
