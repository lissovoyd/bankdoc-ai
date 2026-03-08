"""Tests for retrieval module (tokenizer, utilities)."""

import pytest


# Import only the pure functions that don't require service connections
def test_tokenize_russian():
    from retrieval import tokenize
    tokens = tokenize("Договор аренды помещения")
    assert "договор" in tokens
    assert "аренды" in tokens
    assert "помещения" in tokens


def test_tokenize_english():
    from retrieval import tokenize
    tokens = tokenize("Contract for rental agreement")
    assert "contract" in tokens
    assert "rental" in tokens


def test_tokenize_section_numbers():
    from retrieval import tokenize
    tokens = tokenize("Раздел 8.3 и пункт 13.5.2")
    assert "8.3" in tokens
    assert "13.5.2" in tokens


def test_tokenize_mixed():
    from retrieval import tokenize
    tokens = tokenize("Статья 5 (Article 5) — стоимость 1000 руб")
    assert "статья" in tokens
    assert "article" in tokens
    assert "5" in tokens
    assert "1000" in tokens
    assert "руб" in tokens


def test_tokenize_empty():
    from retrieval import tokenize
    assert tokenize("") == []
    assert tokenize("   ") == []
