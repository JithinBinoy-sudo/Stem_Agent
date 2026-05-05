import pytest
from unittest.mock import patch, MagicMock
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


# ---------------------------------------------------------------------------
# Task 10 – ToolGenerator tests
# ---------------------------------------------------------------------------

from specialization.tool_generator import ToolGenerator

SUMMARIZER_SPEC = ToolSpec(
    name="summarizer",
    description="Summarizes long text into key bullet points",
    parameters={"text": {"type": "string", "description": "Text to summarize"}},
    returns={"type": "object", "fields": ["summary", "bullets"]}
)

SUMMARIZER_CODE = '''
def summarizer(text: str) -> dict:
    words = text.split()
    summary = " ".join(words[:30])
    return {"summary": summary, "bullets": [summary]}
'''


@pytest.fixture(autouse=True)
def patch_openai():
    """Patch OpenAI constructor so tests run without real API keys."""
    with patch("specialization.tool_generator.OpenAI"):
        yield


def test_generate_tool_returns_schema_and_code():
    generator = ToolGenerator()
    with patch.object(generator._client.chat.completions, "create") as mock_create:
        choice = MagicMock()
        choice.message.content = SUMMARIZER_CODE
        mock_create.return_value.choices = [choice]
        result = generator.generate("summarizer", SUMMARIZER_SPEC)
    assert result is not None
    assert "schema" in result
    assert "source_code" in result
    assert result["schema"]["function"]["name"] == "summarizer"


def test_generate_skips_prebuilt_tools():
    generator = ToolGenerator()
    result = generator.generate("web_search", None)
    assert result is None  # pre-built, skip generation


def test_spec_to_openai_schema():
    generator = ToolGenerator()
    schema = generator.spec_to_schema(SUMMARIZER_SPEC)
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "summarizer"
    assert "text" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["required"] == ["text"]
