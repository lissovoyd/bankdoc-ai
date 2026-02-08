import chromadb
import os
import re
import redis
import fitz  # PyMuPDF
import traceback

from celery_app import celery_app
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Document, DocContent, DocumentStatus
from pathlib import Path
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path
from typing import List, Tuple





def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences. Handles Russian and English punctuation.
    """
    # Pattern for sentence endings: . ! ? followed by space or end of string
    # Also handles abbreviations like "т.д.", "и т.п."
    pattern = r'(?<!\w\.\w.)(?<![А-ЯA-Z][а-яa-z]\.)(?<=\.|\!|\?)\s+'
    
    sentences = re.split(pattern, text)
    
    # Clean up: remove empty, strip whitespace
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def create_chunks(text: str, max_chars: int = 800, overlap_sentences: int = 2) -> List[dict]:
    """
    Smart chunking: paragraphs first, then sentences if paragraph is too long.
    Respects numbered sections (8.1, 8.2, etc.).
    """
    section_pattern = r'^\s*\[?\d+\.\d+\.'
    
    # Try splitting by paragraphs first
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = []
    current_length = 0
    start_idx = 0
    
    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        para_len = len(para)
        
        # If paragraph itself is > max_chars, split it into sentences
        if para_len > max_chars:
            # Split this long paragraph into sentences
            sentences = split_into_sentences(para)
            
            for sentence in sentences:
                sent_len = len(sentence)
                
                # If adding this sentence exceeds limit AND we have content
                if current_length + sent_len > max_chars and current_chunk:
                    # Save current chunk
                    chunk_text = " ".join(current_chunk)
                    chunks.append({
                        "text": chunk_text,
                        "start_sentence": start_idx,
                        "end_sentence": i,
                        "sentence_count": len(current_chunk)
                    })
                    
                    # Start new chunk with overlap
                    if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                        current_chunk = current_chunk[-overlap_sentences:]
                        current_length = sum(len(s) for s in current_chunk) + len(current_chunk)
                    else:
                        current_chunk = []
                        current_length = 0
                    
                    start_idx = i
                
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_length += sent_len + 1
        
        # If paragraph is small enough, treat it as one unit
        else:
            # Check if this starts a new section
            if re.match(section_pattern, para) and current_chunk and current_length > 300:
                # Save current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_sentence": start_idx,
                    "end_sentence": i - 1,
                    "sentence_count": len(current_chunk)
                })
                current_chunk = [para]
                current_length = para_len
                start_idx = i
            
            # If adding this para exceeds limit
            elif current_length + para_len > max_chars and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append({
                    "text": chunk_text,
                    "start_sentence": start_idx,
                    "end_sentence": i - 1,
                    "sentence_count": len(current_chunk)
                })
                
                # Overlap
                if overlap_sentences > 0 and len(current_chunk) > overlap_sentences:
                    current_chunk = current_chunk[-overlap_sentences:]
                    current_length = sum(len(p) for p in current_chunk)
                else:
                    current_chunk = [para]
                    current_length = para_len
                
                start_idx = i
            
            else:
                current_chunk.append(para)
                current_length += para_len + 1
    
    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk)
        chunks.append({
            "text": chunk_text,
            "start_sentence": start_idx,
            "end_sentence": len(paragraphs) - 1,
            "sentence_count": len(current_chunk)
        })
    
    return chunks




UPLOAD_DIR = Path("uploads")
MODELS_CACHE_DIR = Path("models_cache")
CHROMA_DB_DIR = Path("chroma_db")

# Make sure directory exists
CHROMA_DB_DIR.mkdir(exist_ok=True)

# Initialize ChromaDB with PersistentClient (ensures disk persistence)
chroma_client = chromadb.PersistentClient(path=str(CHROMA_DB_DIR))

# Get or create collection
collection = chroma_client.get_or_create_collection(
    name="bankdoc",
    metadata={"description": "Document embeddings for BankDoc AI"}
)

# Initialize LOCAL embedding model (multilingual - great for Russian!)
print("Loading local embedding model...")
embedding_model = SentenceTransformer(
    'paraphrase-multilingual-MiniLM-L12-v2',
    cache_folder=str(MODELS_CACHE_DIR)
)
print("✓ Embedding model loaded (100% local, no API calls)")

# Redis client for cache invalidation
redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)






def extract_text_from_pdf(filepath: str):
    """Extract text from PDF page by page.
    Returns list of (page_num, text, char_count) tuples"""
    doc = fitz.open(filepath)
    pages = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        pages.append((page_num + 1, text, len(text)))
    
    doc.close()
    return pages







@celery_app.task(name='tasks.extract_document', bind=True)
def extract_document(self, doc_id: int):
    """Celery task: Extract text from PDF"""
    db = SessionLocal()
    
    try:
        # Get document
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return {"error": "Document not found", "doc_id": doc_id}
        
        # Extract text
        filepath = UPLOAD_DIR / doc.filename
        if not filepath.exists():
            return {"error": "File not found on disk", "doc_id": doc_id}
        
        print(f"[Task {self.request.id}] Extracting text from: {doc.filename}")
        pages = extract_text_from_pdf(str(filepath))
        
        # Store in database
        for page_num, text, char_count in pages:
            content = DocContent(
                doc_id=doc.id,
                page_num=page_num,
                text=text,
                char_count=char_count
            )
            db.add(content)
        
        # Update status
        doc.status = DocumentStatus.EXTRACTED
        db.commit()
        
        # Invalidate Redis cache
        cache_key = f"doc:{doc.id}:content"
        redis_client.delete(cache_key)
        print(f"[Task {self.request.id}] ✓ Extracted {len(pages)} pages")
        
        # Chain: Trigger embedding task
        print(f"[Task {self.request.id}] → Queuing embedding task...")
        embed_document.delay(doc_id)
        
        return {
            "doc_id": doc.id,
            "status": "EXTRACTED",
            "pages_extracted": len(pages),
            "total_chars": sum(c for _, _, c in pages),
            "next": "Embedding task queued"
        }
        
    except Exception as e:
        print(f"[Task {self.request.id}] ✗ Error: {repr(e)}")
        print(traceback.format_exc())
        return {"error": repr(e), "doc_id": doc_id}
    finally:
        db.close()





@celery_app.task(name='tasks.embed_document', bind=True)
def embed_document(self, doc_id: int):
    """Celery task: Embed document pages using LOCAL model with smart sentence-based chunking"""
    db = SessionLocal()
    
    try:
        # Get document
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return {"error": "Document not found", "doc_id": doc_id}
        
        if doc.status != DocumentStatus.EXTRACTED:
            return {"error": "Document not extracted yet", "doc_id": doc_id}
        
        # Get all pages
        pages = db.query(DocContent).filter(DocContent.doc_id == doc_id).order_by(DocContent.page_num).all()
        
        if not pages:
            return {"error": "No content found", "doc_id": doc_id}
        
        print(f"[Task {self.request.id}] Embedding {len(pages)} pages using LOCAL model with smart chunking...")
        
        # Prepare chunks for all pages
        texts = []
        metadatas = []
        ids = []
        
        for page in pages:
            page_text = page.text.strip()
            
            # Normalize newlines - BUT keep the text!
            page_text = re.sub(r"\n{3,}", "\n\n", page_text)  # 3+ newlines → 2
            page_text = re.sub(r"(?<!\n)\n(?!\n)", " ", page_text)  # single newlines → space
            
            # Skip empty pages AFTER normalization
            if not page_text or len(page_text) < 10:  # At least 10 chars
                continue
            
            # Create sentence-aware chunks
            chunks = create_chunks(page_text, max_chars=800, overlap_sentences=2)
            
            # Skip if no chunks created
            if not chunks:
                continue
            
            for chunk_idx, chunk_data in enumerate(chunks):
                chunk_id = f"doc_{doc_id}_page_{page.page_num}_chunk_{chunk_idx}"
                
                # Skip empty chunks
                if not chunk_data["text"] or len(chunk_data["text"]) < 10:
                    continue
                
                texts.append(chunk_data["text"])
                metadatas.append({
                    "doc_id": doc_id,
                    "page_num": page.page_num,
                    "chunk_index": chunk_idx,
                    "char_count": len(chunk_data["text"]),
                    "sentence_count": chunk_data["sentence_count"],
                    "start_sentence": chunk_data["start_sentence"],
                    "end_sentence": chunk_data["end_sentence"],
                    "title": doc.title,
                    "department": doc.department or "unknown",
                    "corpus_id": doc.corpus_id or f"doc_{doc.id}",
                    "chunk_id": chunk_id  # ← Make sure this is here
                })
                ids.append(chunk_id)
        
        if not texts:
            return {"error": "No text to embed after chunking", "doc_id": doc_id}
        
        print(f"[Task {self.request.id}] Created {len(texts)} sentence-aware chunks")
        
        # Log chunking stats
        avg_sentences = sum(m["sentence_count"] for m in metadatas) / len(metadatas)
        avg_chars = sum(m["char_count"] for m in metadatas) / len(metadatas)
        print(f"[Task {self.request.id}] Avg chunk size: {avg_chars:.0f} chars, {avg_sentences:.1f} sentences")
        
        # Embed in batches
        BATCH_SIZE = 8
        
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i + BATCH_SIZE]
            batch_ids = ids[i:i + BATCH_SIZE]
            batch_metas = metadatas[i:i + BATCH_SIZE]
            
            # Generate embeddings for this batch
            batch_embeddings = embedding_model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=BATCH_SIZE
            ).tolist()
            
            # Store in ChromaDB
            collection.add(
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metas,
                ids=batch_ids
            )
            
            processed = min(i + BATCH_SIZE, len(texts))
            print(f"[Task {self.request.id}] ✓ Embedded {processed}/{len(texts)} chunks")
        
        # Update document status
        doc.status = DocumentStatus.EMBEDDED
        db.commit()
        
        print(f"[Task {self.request.id}] ✓ Completed: {len(pages)} pages → {len(texts)} sentence-aware chunks")
        
        return {
            "doc_id": doc_id,
            "status": "EMBEDDED",
            "pages_embedded": len(pages),
            "chunks_created": len(texts),
            "avg_chunk_size": int(sum(m["char_count"] for m in metadatas) / len(metadatas)),
            "avg_sentences_per_chunk": round(sum(m["sentence_count"] for m in metadatas) / len(metadatas), 1),
            "collection": "bankdoc",
            "model": "paraphrase-multilingual-MiniLM-L12-v2 (LOCAL)",
            "privacy": "100% on-premise, no external API calls"
        }
        
    except Exception as e:
        print(f"[Task {self.request.id}] ✗ Error: {repr(e)}")
        print(traceback.format_exc())
        return {"error": repr(e), "doc_id": doc_id}
    finally:
        db.close()
