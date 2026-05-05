import pytest
from unittest.mock import patch, MagicMock
from discovery.llm_introspection import DomainProfile, ToolSpec, LLMIntrospector

MOCK_LLM_RESPONSE = {
    "domain": "Deep Research",
    "workflow": ["query decomposition", "multi-source search", "synthesis"],
    "required_tools": ["web_search", "url_reader", "summarizer"],
    "tool_specs": {
        "summarizer": {
            "name": "summarizer",
            "description": "Summarizes long text into key points",
            "parameters": {
                "text": {"type": "string", "description": "Text to summarize"}
            },
            "returns": {"type": "object", "fields": ["summary"]}
        }
    },
    "quality_criteria": ["factual accuracy", "source coverage"]
}

@pytest.fixture
def mock_openai():
    with patch("discovery.llm_introspection.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        choice = MagicMock()
        choice.message.content = str(MOCK_LLM_RESPONSE).replace("'", '"')
        instance.chat.completions.create.return_value.choices = [choice]
        yield instance

def test_introspect_returns_domain_profile(mock_openai):
    import json
    mock_openai.chat.completions.create.return_value.choices[0].message.content = json.dumps(MOCK_LLM_RESPONSE)
    introspector = LLMIntrospector()
    profile = introspector.introspect("Deep Research")
    assert isinstance(profile, DomainProfile)
    assert profile.domain == "Deep Research"

def test_profile_has_workflow(mock_openai):
    import json
    mock_openai.chat.completions.create.return_value.choices[0].message.content = json.dumps(MOCK_LLM_RESPONSE)
    introspector = LLMIntrospector()
    profile = introspector.introspect("Deep Research")
    assert len(profile.workflow) >= 1
    assert isinstance(profile.workflow[0], str)

def test_profile_has_tool_specs(mock_openai):
    import json
    mock_openai.chat.completions.create.return_value.choices[0].message.content = json.dumps(MOCK_LLM_RESPONSE)
    introspector = LLMIntrospector()
    profile = introspector.introspect("Deep Research")
    assert "summarizer" in profile.tool_specs
    assert isinstance(profile.tool_specs["summarizer"], ToolSpec)

def test_profile_has_quality_criteria(mock_openai):
    import json
    mock_openai.chat.completions.create.return_value.choices[0].message.content = json.dumps(MOCK_LLM_RESPONSE)
    introspector = LLMIntrospector()
    profile = introspector.introspect("Deep Research")
    assert len(profile.quality_criteria) >= 1
