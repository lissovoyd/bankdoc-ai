import re
import bisect
import traceback

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document as LCDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

from celery_app import celery_app
from config import UPLOAD_DIR, MODELS_CACHE_DIR, get_vectorstore
from database import SessionLocal
from models import Document, DocContent, DocumentStatus


# Lazy-loaded singleton so the embedding model is only loaded when the worker starts
_embeddings = None


def _get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    global _embeddings
    if _embeddings is None:
        print("Loading embedding model...")
        _embeddings = HuggingFaceEmbeddings(
            model_name="paraphrase-multilingual-MiniLM-L12-v2",
            cache_folder=str(MODELS_CACHE_DIR),
            encode_kwargs={"normalize_embeddings": True},
        )
        print("✓ Embedding model loaded")
    return _embeddings


# --- Section header detector ---
_SECTION_RE = re.compile(
    r"^\s*(\d+(?:\.\d+)*[-\d]*\.?)\s+([^\n]{3,80})", re.MULTILINE
)


def _detect_section(text: str) -> str | None:
    """Return the first section header found in the chunk text, e.g. '3.2 Права клиента'."""
    m = _SECTION_RE.search(text)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    return None


@celery_app.task(name="tasks.extract_document", bind=True)
def extract_document(self, doc_id: int):
    """Celery task: extract text from PDF page-by-page and store in DB."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return {"error": "Document not found", "doc_id": doc_id}

        filepath = UPLOAD_DIR / doc.filename
        if not filepath.exists():
            return {"error": "File not found on disk", "doc_id": doc_id}

        print(f"[Task {self.request.id}] Extracting: {doc.filename}")

        loader = PyPDFLoader(str(filepath))
        pages = loader.load()  # one LangChain Document per page

        for page_doc in pages:
            page_num = page_doc.metadata.get("page", 0) + 1  # 0-indexed → 1-indexed
            db.add(DocContent(
                doc_id=doc.id,
                page_num=page_num,
                text=page_doc.page_content,
                char_count=len(page_doc.page_content),
            ))

        doc.status = DocumentStatus.EXTRACTED
        db.commit()
        print(f"[Task {self.request.id}] ✓ Extracted {len(pages)} pages")

        # Chain: trigger embedding
        embed_document.delay(doc_id)

        return {
            "doc_id": doc_id,
            "status": "EXTRACTED",
            "pages_extracted": len(pages),
            "next": "Embedding task queued",
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": repr(e), "doc_id": doc_id}
    finally:
        db.close()


@celery_app.task(name="tasks.embed_document", bind=True)
def embed_document(self, doc_id: int):
    """Celery task: chunk and embed document into ChromaDB using LangChain."""
    db = SessionLocal()
    try:
        doc = db.query(Document).filter(Document.id == doc_id).first()
        if not doc:
            return {"error": "Document not found", "doc_id": doc_id}
        if doc.status != DocumentStatus.EXTRACTED:
            return {"error": f"Expected EXTRACTED, got {doc.status.value}", "doc_id": doc_id}

        pages = (
            db.query(DocContent)
            .filter(DocContent.doc_id == doc_id)
            .order_by(DocContent.page_num)
            .all()
        )
        if not pages:
            return {"error": "No content found", "doc_id": doc_id}

        print(f"[Task {self.request.id}] Chunking {len(pages)} pages with LangChain...")

        # --- Delete any existing chunks for this doc (safe re-embed) ---
        vs = get_vectorstore(embedding_function=_get_embeddings())
        existing = vs.get(where={"doc_id": doc_id})
        if existing["ids"]:
            vs.delete(ids=existing["ids"])
            print(f"[Task {self.request.id}] Deleted {len(existing['ids'])} old chunks for doc_id={doc_id}")

        # --- Merge all pages into one continuous text ---
        # This allows chunks to span page boundaries (e.g. a section that
        # starts at the bottom of page 3 and continues on page 4).
        merged_text = ""
        page_boundaries = []  # (char_offset, page_num) — start of each page
        for p in pages:
            if not p.text or len(p.text.strip()) < 10:
                continue
            page_boundaries.append((len(merged_text), p.page_num))
            merged_text += p.text + "\n\n"

        if not merged_text.strip():
            return {"error": "No usable text after merge", "doc_id": doc_id}

        boundary_offsets = [b[0] for b in page_boundaries]
        boundary_pages = [b[1] for b in page_boundaries]

        def offset_to_page(offset: int) -> int:
            """Map a character offset in merged_text to a page number."""
            idx = bisect.bisect_right(boundary_offsets, offset) - 1
            return boundary_pages[max(0, idx)]

        # --- Split the merged text into chunks ---
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            is_separator_regex=True,
            separators=[
                r"\n\d+\.\d+[\d.]*[-\d]*\.?\s",  # subsection: 2.1.2. / 13.5-1.
                r"\n\d+\.\s",                      # top section: 1. / 13.
                r"\n\n",
                r"\n",
                r"\.\s",
                r" ",
                r"",
            ],
        )
        raw_chunks = splitter.split_text(merged_text)

        # Track the last known section title as we walk through chunks in order
        last_section: str | None = None

        # --- Build LangChain Documents with page metadata ---
        chunks = []
        search_from = 0
        for text in raw_chunks:
            pos = merged_text.find(text, search_from)
            if pos == -1:
                pos = merged_text.find(text)
            page_num = offset_to_page(pos if pos != -1 else 0)
            if pos != -1:
                search_from = pos + 1
            chunks.append(LCDocument(
                page_content=text,
                metadata={"page_num": page_num, "doc_id": doc_id},
            ))

        # Enrich chunk metadata and assign stable IDs
        chunk_counter: dict[int, int] = {}
        ids = []
        for chunk in chunks:
            page_num = chunk.metadata["page_num"]
            idx = chunk_counter.get(page_num, 0)
            chunk_counter[page_num] = idx + 1
            chunk_id = f"doc_{doc_id}_page_{page_num}_chunk_{idx}"
            section = _detect_section(chunk.page_content)
            if section:
                last_section = section
            chunk.metadata.update({
                "chunk_index": idx,
                "char_count": len(chunk.page_content),
                "title": doc.title,
                "department": doc.department or "unknown",
                "corpus_id": doc.corpus_id or f"doc_{doc.id}",
                "chunk_id": chunk_id,
                "section_title": last_section or "",
            })
            ids.append(chunk_id)

        print(f"[Task {self.request.id}] Created {len(chunks)} chunks, embedding...")

        BATCH = 8
        for i in range(0, len(chunks), BATCH):
            vs.add_documents(documents=chunks[i:i + BATCH], ids=ids[i:i + BATCH])
            print(f"[Task {self.request.id}] ✓ {min(i + BATCH, len(chunks))}/{len(chunks)} chunks embedded")

        doc.status = DocumentStatus.EMBEDDED
        db.commit()

        avg_chars = sum(len(c.page_content) for c in chunks) // len(chunks)
        print(f"[Task {self.request.id}] ✓ Done: {len(chunks)} chunks, avg {avg_chars} chars")

        return {
            "doc_id": doc_id,
            "status": "EMBEDDED",
            "pages_embedded": len(pages),
            "chunks_created": len(chunks),
            "avg_chunk_size": avg_chars,
            "model": "paraphrase-multilingual-MiniLM-L12-v2 (local)",
        }

    except Exception as e:
        print(traceback.format_exc())
        return {"error": repr(e), "doc_id": doc_id}
    finally:
        db.close()
