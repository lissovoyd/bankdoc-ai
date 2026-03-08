"""Tests for Pydantic schema validation."""

import pytest
from schemas import (
    AskRequest,
    AskResponse,
    DocumentResponse,
    RAGAnswer,
    SourceReference,
)


class TestRAGAnswer:
    def test_valid_structured_output(self):
        answer = RAGAnswer(
            answer="Договор действует до 31.12.2025 (раздел 10.1, стр. 8)",
            cited_pages=[8],
            confidence_self="high",
        )
        assert answer.answer.startswith("Договор")
        assert answer.cited_pages == [8]
        assert answer.confidence_self == "high"

    def test_defaults(self):
        answer = RAGAnswer(answer="Test")
        assert answer.cited_pages == []
        assert answer.confidence_self == "medium"

    def test_invalid_missing_answer(self):
        with pytest.raises(Exception):
            RAGAnswer()


class TestAskResponse:
    def test_declined_response(self):
        resp = AskResponse(
            question="test",
            answer="declined",
            sources=[],
            doc_id=1,
            doc_title="test",
            confidence=0.1,
            grounded=False,
            declined=True,
        )
        assert resp.declined is True
        assert resp.grounded is False

    def test_normal_response_with_sources(self):
        resp = AskResponse(
            question="test",
            answer="answer",
            sources=[
                SourceReference(
                    page_num=1,
                    chunk_index=0,
                    text_excerpt="some text",
                    relevance_score=0.85,
                )
            ],
            doc_id=1,
            doc_title="doc",
            confidence=2.5,
            grounded=True,
            declined=False,
        )
        assert len(resp.sources) == 1
        assert resp.sources[0].relevance_score == 0.85


class TestAskRequest:
    def test_with_prompt_version(self):
        req = AskRequest(question="test?", prompt_version="v1")
        assert req.prompt_version == "v1"

    def test_default_prompt_version(self):
        req = AskRequest(question="test?")
        assert req.prompt_version is None
