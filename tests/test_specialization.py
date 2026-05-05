import pytest
from discovery.llm_introspection import DomainProfile, ToolSpec
from specialization.prompt_rewriter import PromptRewriter

PROFILE = DomainProfile(
    domain="Deep Research",
    workflow=["query decomposition", "multi-source search", "cross-referencing", "synthesis", "citation"],
    required_tools=["web_search", "url_reader", "summarizer"],
    tool_specs={},
    quality_criteria=["factual accuracy", "source coverage", "synthesis quality"]
)

def test_rewritten_prompt_contains_domain():
    rewriter = PromptRewriter()
    prompt = rewriter.rewrite(PROFILE)
    assert "Deep Research" in prompt

def test_rewritten_prompt_contains_all_workflow_steps():
    rewriter = PromptRewriter()
    prompt = rewriter.rewrite(PROFILE)
    for step in PROFILE.workflow:
        assert step in prompt

def test_rewritten_prompt_contains_quality_criteria():
    rewriter = PromptRewriter()
    prompt = rewriter.rewrite(PROFILE)
    for criterion in PROFILE.quality_criteria:
        assert criterion in prompt

def test_rewritten_prompt_is_longer_than_base():
    rewriter = PromptRewriter()
    from config import BASE_SYSTEM_PROMPT
    prompt = rewriter.rewrite(PROFILE)
    assert len(prompt) > len(BASE_SYSTEM_PROMPT)

def test_rewritten_prompt_replaces_not_appends():
    rewriter = PromptRewriter()
    from config import BASE_SYSTEM_PROMPT
    prompt = rewriter.rewrite(PROFILE)
    assert BASE_SYSTEM_PROMPT not in prompt
