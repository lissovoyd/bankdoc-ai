from sqlalchemy import Column, Integer, String, DateTime, Enum, Text, ForeignKey
from sqlalchemy.sql import func
from database import Base
import enum

class DocumentStatus(str, enum.Enum):
    PENDING = "PENDING"
    EXTRACTED = "EXTRACTED"
    EMBEDDED = "EMBEDDED"

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING)

    # ✅ add these
    department = Column(String, nullable=True)   # e.g. "finance", "legal", ...
    corpus_id = Column(String, nullable=True)    # e.g. "little_prince", "policy_2024"

class DocContent(Base):
    __tablename__ = "doc_contents"
    
    id = Column(Integer, primary_key=True, index=True)
    doc_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    page_num = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    char_count = Column(Integer, nullable=False)
