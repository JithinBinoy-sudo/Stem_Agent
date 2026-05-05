import pytest
from unittest.mock import patch, MagicMock
from discovery.web_research import WebResearcher

@pytest.fixture
def mock_tavily():
    with patch("discovery.web_research.TavilyClient") as MockClient:
        instance = MockClient.return_value
        instance.search.return_value = MagicMock(results=[
            MagicMock(title="AgentBench", url="https://arxiv.org/agent", content="Agents need planning and tool use skills"),
            MagicMock(title="ReAct Paper", url="https://arxiv.org/react", content="ReAct combines reasoning and acting"),
        ])
        yield instance

def test_research_returns_list_of_findings(mock_tavily):
    researcher = WebResearcher()
    findings = researcher.research("Deep Research")
    assert isinstance(findings, list)
    assert len(findings) > 0

def test_each_finding_has_source_and_insight(mock_tavily):
    researcher = WebResearcher()
    findings = researcher.research("Deep Research")
    for finding in findings:
        assert "source" in finding
        assert "insight" in finding
        assert isinstance(finding["source"], str)
        assert isinstance(finding["insight"], str)

def test_research_queries_multiple_topics(mock_tavily):
    researcher = WebResearcher()
    researcher.research("Deep Research")
    assert mock_tavily.search.call_count >= 2

def test_empty_results_handled_gracefully(mock_tavily):
    mock_tavily.search.return_value = MagicMock(results=[])
    researcher = WebResearcher()
    findings = researcher.research("Deep Research")
    assert isinstance(findings, list)
