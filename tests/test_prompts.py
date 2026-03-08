"""Tests for prompt versioning."""

from prompts import get_prompt, PROMPTS, STRUCTURED_PROMPT


def test_get_prompt_default():
    prompt = get_prompt()
    assert prompt is not None
    assert "context" in prompt.input_variables
    assert "question" in prompt.input_variables


def test_get_prompt_v1():
    prompt = get_prompt("v1")
    assert prompt == PROMPTS["v1"]


def test_get_prompt_v2():
    prompt = get_prompt("v2")
    assert prompt == PROMPTS["v2"]


def test_get_prompt_invalid_falls_back():
    prompt = get_prompt("nonexistent")
    # Falls back to active version
    assert prompt is not None


def test_structured_prompt_has_json_format():
    template = STRUCTURED_PROMPT.template
    assert "JSON" in template
    assert "answer" in template
    assert "cited_pages" in template
    assert "confidence_self" in template


def test_all_prompts_have_required_vars():
    for version, prompt in PROMPTS.items():
        assert "context" in prompt.input_variables, f"{version} missing 'context'"
        assert "question" in prompt.input_variables, f"{version} missing 'question'"
