import pytest
import os
from unittest.mock import patch
from discovery import DiscoveryEngine
from discovery.llm_introspection import DomainProfile, ToolSpec

MOCK_PROFILE = DomainProfile(
    domain="Deep Research",
    workflow=["decompose", "search", "synthesize"],
    required_tools=["web_search", "url_reader", "summarizer"],
    tool_specs={
        "summarizer": ToolSpec("summarizer", "Summarizes text", {"text": {"type": "string", "description": "text"}}, {})
    },
    quality_criteria=["accuracy", "coverage"]
)

@pytest.fixture
def tmp_cache(tmp_path):
    return str(tmp_path / "cache")

@pytest.fixture(autouse=True)
def patch_api_clients():
    """Patch external API clients so constructors don't require real credentials."""
    with patch("discovery.llm_introspection.OpenAI"):
        with patch("discovery.web_research.web_search", return_value={"results": []}):
            yield

def test_run_returns_domain_profile(tmp_cache):
    engine = DiscoveryEngine(cache_dir=tmp_cache)
    with patch.object(engine._introspector, "introspect", return_value=MOCK_PROFILE):
        with patch.object(engine._researcher, "research", return_value=[]):
            profile = engine.run("Deep Research")
    assert isinstance(profile, DomainProfile)
    assert profile.domain == "Deep Research"

def test_result_is_cached_to_disk(tmp_cache):
    engine = DiscoveryEngine(cache_dir=tmp_cache)
    with patch.object(engine._introspector, "introspect", return_value=MOCK_PROFILE):
        with patch.object(engine._researcher, "research", return_value=[]):
            engine.run("Deep Research")
    cache_file = os.path.join(tmp_cache, "deep_research.json")
    assert os.path.exists(cache_file)

def test_second_run_uses_cache(tmp_cache):
    engine = DiscoveryEngine(cache_dir=tmp_cache)
    with patch.object(engine._introspector, "introspect", return_value=MOCK_PROFILE) as mock_introspect:
        with patch.object(engine._researcher, "research", return_value=[]):
            engine.run("Deep Research")
            engine.run("Deep Research")
    assert mock_introspect.call_count == 1

def test_force_rediscover_bypasses_cache(tmp_cache):
    engine = DiscoveryEngine(cache_dir=tmp_cache)
    with patch.object(engine._introspector, "introspect", return_value=MOCK_PROFILE) as mock_introspect:
        with patch.object(engine._researcher, "research", return_value=[]):
            engine.run("Deep Research")
            engine.run("Deep Research", force=True)
    assert mock_introspect.call_count == 2
