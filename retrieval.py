"""RAG retrieval pipeline: hybrid search → cross-encoder re-ranking → structured LLM output."""

import json
import math
import re
import logging

from fastapi import HTTPException
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

import pymorphy3

from langchain_core.prompts import PromptTemplate
from config import get_vectorstore, llm, llm_json, reranker, RERANKER_CONFIDENCE_THRESHOLD
from models import Document
from prompts import get_prompt, STRUCTURED_PROMPT
from schemas import AskRequest, AskResponse, RAGAnswer, SourceReference

logger = logging.getLogger(__name__)

DECLINE_ANSWER_RU = (
    "На основании предоставленных фрагментов документа ответить на этот вопрос не удалось. "
    "Попробуйте переформулировать вопрос или уточнить, какой раздел документа вас интересует."
)

MAX_JSON_RETRIES = 2

# Russian morphological analyzer — normalizes word forms to dictionary lemmas
_morph = pymorphy3.MorphAnalyzer()


def tokenize(text: str) -> list[str]:
    """Tokeniser with Russian lemmatization + English + numeric section tokens."""
    tokens = re.findall(r"[а-яёa-z]+|\d+(?:\.\d+)+|\d+", text.lower())
    return [_morph.parse(t)[0].normal_form if re.match(r"[а-яё]", t) else t for t in tokens]


CONFIDENCE_BOOST = 1.5  # Display multiplier — raw logit threshold comparison is unaffected

_SECTION_HEADER_RE = re.compile(r"(?m)^\s*(\d+(?:\.\d+)*[-\d]*\.?)\s+[^\n]{3,80}")


def _sigmoid_pct(score: float) -> float:
    """Convert raw cross-encoder logit to a boosted 0–100 relevance percentage."""
    raw = 1.0 / (1.0 + math.exp(-score)) * 100
    return round(min(raw * CONFIDENCE_BOOST, 100.0), 1)


EXPANSION_PROMPT = PromptTemplate.from_template(
    """Перефразируй вопрос пользователя 3 способами, используя юридическую/банковскую терминологию.
Каждый вариант на отдельной строке. Без нумерации, только текст вопроса.

Вопрос: {question}
Варианты:"""
)


def _expand_query(question: str) -> list[str]:
    """Generate 2-3 reformulations of the question using legal terminology."""
    try:
        raw = (EXPANSION_PROMPT | llm | StrOutputParser()).invoke({"question": question})
        variants = [line.strip() for line in raw.strip().split("\n") if line.strip() and len(line.strip()) > 10]
        logger.info("[RAG] Query expansion: %d variants", len(variants))
        return variants[:3]
    except Exception as exc:
        logger.warning("[RAG] Query expansion failed: %s", exc)
        return []


def _rerank(question: str, docs: list[LCDocument], top_n: int = 5) -> list[tuple[LCDocument, float]]:
    """Re-rank documents using the cross-encoder and return (doc, score) pairs."""
    if not docs:
        return []
    pairs = [(question, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def _generate_structured(context: str, question: str) -> RAGAnswer | None:
    """Try to get a structured JSON answer from the LLM with retry on validation failure."""
    for attempt in range(1, MAX_JSON_RETRIES + 1):
        try:
            raw_output = (STRUCTURED_PROMPT | llm_json | StrOutputParser()).invoke(
                {"context": context, "question": question}
            )
            parsed = json.loads(raw_output)
            return RAGAnswer(**parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("[RAG] Structured output attempt %d failed: %s", attempt, exc)
    return None


def _section_excerpt(chunk: LCDocument, max_chars: int = 1200) -> str:
    """Return chunk text, trimming any preamble that appears before the section header."""
    text = chunk.page_content
    m = _SECTION_HEADER_RE.search(text)
    if m and m.start() > 30:
        text = text[m.start():]
    return text[:max_chars] + ("..." if len(text) > max_chars else "")


def build_rag_response(doc_id: int, payload: AskRequest, doc: Document) -> AskResponse:
    """Run the full RAG pipeline: hybrid retrieval → re-ranking → LLM generation → cited answer."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    logger.info("[RAG] Question: %s", question)

    # --- Query expansion: generate legal reformulations ---
    expansions = _expand_query(question)
    all_queries = [question] + expansions
    logger.info("[RAG] Searching with %d queries: %s", len(all_queries), all_queries)

    # --- Get vectorstore via shared HttpClient ---
    vs = get_vectorstore()

    # --- Fetch all chunks for this doc (used by BM25 + dedup) ---
    raw = vs.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])

    if not raw["documents"]:
        raise HTTPException(status_code=404, detail="No chunks found — embed the document first")

    lc_docs = [
        LCDocument(page_content=text, metadata=meta)
        for text, meta in zip(raw["documents"], raw["metadatas"])
    ]

    # --- Semantic retriever (ChromaDB, filtered by doc_id) ---
    semantic_retriever = vs.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20, "filter": {"doc_id": doc_id}},
    )

    # --- BM25 retriever ---
    bm25_retriever = BM25Retriever.from_documents(lc_docs, k=10, preprocess_func=tokenize)

    # --- Ensemble: RRF fusion of semantic + BM25 ---
    ensemble = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.6, 0.4],
    )

    # --- Run retrieval for all query variants, deduplicate by content ---
    seen_content = set()
    candidates = []
    for q in all_queries:
        for doc_candidate in ensemble.invoke(q)[:15]:
            content_key = doc_candidate.page_content[:200]
            if content_key not in seen_content:
                seen_content.add(content_key)
                candidates.append(doc_candidate)

    logger.info("[RAG] %d unique candidates from hybrid retrieval (%d queries)", len(candidates), len(all_queries))

    # --- Cross-encoder re-ranking ---
    ranked = _rerank(question, candidates, top_n=5)
    top_docs = [d for d, _score in ranked]
    top_scores = [score for _d, score in ranked]

    max_score = top_scores[0] if top_scores else 0.0
    logger.info("[RAG] Re-ranked: top score=%.3f (%.1f%%), threshold=%.3f", max_score, _sigmoid_pct(max_score), RERANKER_CONFIDENCE_THRESHOLD)
    for i, (d, s) in enumerate(ranked, 1):
        logger.debug("  [%d] score=%.3f (%.1f%%) page=%s chunk=%s", i, s, _sigmoid_pct(s), d.metadata.get("page_num"), d.metadata.get("chunk_index"))

    # --- Citation enforcement: decline if no relevant chunks ---
    if max_score < RERANKER_CONFIDENCE_THRESHOLD:
        logger.warning("[RAG] Declining — max relevance %.3f < threshold %.3f", max_score, RERANKER_CONFIDENCE_THRESHOLD)
        return AskResponse(
            question=question,
            answer=DECLINE_ANSWER_RU,
            sources=[],
            doc_id=doc.id,
            doc_title=doc.title,
            confidence=_sigmoid_pct(max_score),
            grounded=False,
            declined=True,
        )

    # --- Build context ---
    context = "\n\n".join(
        f"[Страница {d.metadata.get('page_num', '?')}, "
        f"Фрагмент {d.metadata.get('chunk_index', '?')}]\n{d.page_content}"
        for d in top_docs
    )

    # --- Generate answer: try structured JSON first, fall back to plain text ---
    logger.info("[RAG] Calling Ollama (structured JSON mode)...")
    structured = _generate_structured(context, question)

    if structured:
        logger.info("[RAG] Structured output OK (confidence_self=%s)", structured.confidence_self)
        answer_text = structured.answer
    else:
        logger.info("[RAG] Structured output failed, falling back to plain text...")
        prompt = get_prompt(payload.prompt_version)
        answer_text = (prompt | llm | StrOutputParser()).invoke(
            {"context": context, "question": question}
        )

    logger.info("[RAG] Answer: %d chars", len(answer_text))

    sources = [
        SourceReference(
            page_num=d.metadata.get("page_num", 0),
            chunk_index=d.metadata.get("chunk_index", 0),
            section_title=d.metadata.get("section_title"),
            text_excerpt=_section_excerpt(d),
            relevance_score=_sigmoid_pct(float(score)),
            start_sentence=d.metadata.get("start_sentence"),
            end_sentence=d.metadata.get("end_sentence"),
            sentence_count=d.metadata.get("sentence_count"),
        )
        for d, score in ranked
    ]

    return AskResponse(
        question=question,
        answer=answer_text.strip(),
        sources=sources,
        doc_id=doc.id,
        doc_title=doc.title,
        confidence=_sigmoid_pct(float(max_score)),
        grounded=True,
        declined=False,
    )
