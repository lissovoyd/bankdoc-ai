"""Pydantic schemas for API request/response models."""

from enum import Enum as PyEnum
from typing import List, Optional

from pydantic import BaseModel


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
    prompt_version: Optional[str] = None


class SourceReference(BaseModel):
    page_num: int
    chunk_index: int
    section_title: str | None = None
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
    confidence: float = 0.0
    grounded: bool = True
    declined: bool = False


class RAGAnswer(BaseModel):
    """Structured LLM output enforced via Ollama JSON mode + Pydantic validation."""
    answer: str
    cited_pages: List[int] = []
    confidence_self: str = "medium"  # "high" / "medium" / "low"


class Department(str, PyEnum):
    unknown = "unknown"
    finance = "finance"
    legal = "legal"
    service = "service"
    hr = "hr"
    risk = "risk"
    it = "it"
