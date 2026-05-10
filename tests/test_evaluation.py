import pytest
import json
from unittest.mock import patch, MagicMock
from evaluation.benchmark import BenchmarkRunner, TaskResult

SAMPLE_TASKS = [
    {"id": 1, "query": "Who created Python?", "reference_answer": "Guido van Rossum", "key_facts": ["Guido van Rossum"]}
]

@pytest.fixture(autouse=True)
def mock_openai():
    """Mock OpenAI at module level so BenchmarkRunner() never hits the real client."""
    with patch("evaluation.benchmark.OpenAI") as MockOpenAI:
        MockOpenAI.return_value = MagicMock()
        yield MockOpenAI

def make_mock_response(content=None, tool_call=None):
    choice = MagicMock()
    if tool_call:
        choice.finish_reason = "tool_calls"
        choice.message.content = None
        tc = MagicMock()
        tc.id = "call_123"
        tc.function.name = tool_call["name"]
        tc.function.arguments = json.dumps(tool_call["args"])
        choice.message.tool_calls = [tc]
    else:
        choice.finish_reason = "stop"
        choice.message.content = content
        choice.message.tool_calls = None
    resp = MagicMock()
    resp.choices = [choice]
    return resp

def test_run_task_returns_task_result():
    runner = BenchmarkRunner()
    with patch.object(runner._client.chat.completions, "create",
                      return_value=make_mock_response("Guido van Rossum created Python.")):
        result = runner.run_task(
            task=SAMPLE_TASKS[0],
            system_prompt="You are helpful.",
            tool_schemas=[],
        )
    assert isinstance(result, TaskResult)
    assert result.task_id == 1
    assert "Guido" in result.answer

def test_react_loop_calls_tool_then_stops():
    runner = BenchmarkRunner()
    responses = [
        make_mock_response(tool_call={"name": "web_search", "args": {"query": "Python creator"}}),
        make_mock_response("Guido van Rossum.")
    ]
    with patch("evaluation.benchmark.get_tool") as mock_get_tool:
        mock_get_tool.return_value = lambda query: {"results": [{"content": "Guido"}]}
        with patch.object(runner._client.chat.completions, "create", side_effect=responses):
            result = runner.run_task(
                task=SAMPLE_TASKS[0],
                system_prompt="You are helpful.",
                tool_schemas=[{"type": "function", "function": {"name": "web_search"}}],
            )
    assert result.tool_calls_made == 1
    assert "Guido" in result.answer

def test_react_loop_halts_at_max_tool_calls():
    import config
    runner = BenchmarkRunner()
    tool_response = make_mock_response(tool_call={"name": "web_search", "args": {"query": "test"}})
    responses = [tool_response] * (config.MAX_TOOL_CALLS_PER_TASK + 5)
    with patch("evaluation.benchmark.get_tool") as mock_get_tool:
        mock_get_tool.return_value = lambda query: {"results": []}
        with patch.object(runner._client.chat.completions, "create", side_effect=responses):
            result = runner.run_task(
                task=SAMPLE_TASKS[0],
                system_prompt="You are helpful.",
                tool_schemas=[{"type": "function", "function": {"name": "web_search"}}],
            )
    assert result.tool_calls_made == config.MAX_TOOL_CALLS_PER_TASK
    assert result.failed is False

def test_failed_task_returns_zero_answer():
    runner = BenchmarkRunner()
    with patch.object(runner._client.chat.completions, "create", side_effect=Exception("API error")):
        result = runner.run_task(
            task=SAMPLE_TASKS[0],
            system_prompt="You are helpful.",
            tool_schemas=[],
        )
    assert result.failed is True
    assert result.answer == ""

from evaluation.judge import Judge, CriterionScores, compute_coverage_score, compute_citation_score, composite_score

def test_compute_coverage_score_zero_sources():
    assert compute_coverage_score(0) == 0.0

def test_compute_coverage_score_five_sources():
    assert compute_coverage_score(5) == 10.0

def test_compute_coverage_score_ten_sources_capped():
    assert compute_coverage_score(10) == 10.0

def test_compute_coverage_score_two_sources():
    assert abs(compute_coverage_score(2) - 4.0) < 0.01

def test_compute_citation_score_all_cited():
    text = "Fact A [source1]. Fact B [source2]. Fact C [source3]."
    assert compute_citation_score(text) == 10.0

def test_compute_citation_score_none_cited():
    text = "Fact A. Fact B. Fact C."
    assert compute_citation_score(text) == 0.0

def test_composite_score_formula():
    scores = CriterionScores(accuracy=8.0, coverage=6.0, synthesis=7.0, citation=5.0)
    expected = (8.0*0.35 + 6.0*0.25 + 7.0*0.25 + 5.0*0.15) * 10.0
    assert abs(composite_score(scores) - expected) < 0.001

def test_composite_score_perfect():
    scores = CriterionScores(accuracy=10.0, coverage=10.0, synthesis=10.0, citation=10.0)
    assert composite_score(scores) == 100.0

def test_composite_score_zero():
    scores = CriterionScores(accuracy=0.0, coverage=0.0, synthesis=0.0, citation=0.0)
    assert composite_score(scores) == 0.0


# Phase 3: dynamic rubric tests

def test_rubric_drives_scoring_with_custom_criteria():
    """A custom rubric with only deterministic criteria should bypass the LLM judge."""
    from evaluation.judge import Judge
    judge = Judge.__new__(Judge)
    judge._client = MagicMock()
    rubric = [
        {"name": "domains", "weight": 0.6, "method": "deterministic", "scorer": "unique_domains"},
        {"name": "cited", "weight": 0.4, "method": "deterministic", "scorer": "sentence_citations"},
    ]
    results = [
        TaskResult(task_id=1, query="q", answer="A [https://a.com]. B [https://b.com]. C [https://c.com].",
                   tool_calls_made=0, failed=False),
    ]
    tasks = [{"id": 1, "reference_answer": "ref"}]
    out = judge.score_all(results, tasks, rubric=rubric)
    judge._client.chat.completions.create.assert_not_called()
    assert "domains" in out["average"]
    assert "cited" in out["average"]
    assert out["average"]["cited"] == 10.0
    assert out["average"]["domains"] == pytest.approx(3 / 5 * 10)
    expected_composite = (out["average"]["domains"] * 0.6 + out["average"]["cited"] * 0.4) * 10.0
    assert out["average"]["composite"] == pytest.approx(expected_composite)


def test_rubric_with_only_llm_criteria_skips_deterministic():
    """LLM-only rubric: no deterministic scorers run."""
    from evaluation.judge import Judge
    judge = Judge.__new__(Judge)
    judge._client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = json.dumps({"scores": [{"correctness": 8, "style": 7}]})
    judge._client.chat.completions.create.return_value = fake_response
    rubric = [
        {"name": "correctness", "weight": 0.7, "method": "llm", "description": "is the answer correct"},
        {"name": "style", "weight": 0.3, "method": "llm", "description": "is it well-styled"},
    ]
    results = [
        TaskResult(task_id=1, query="q", answer="some answer", tool_calls_made=0, failed=False),
    ]
    tasks = [{"id": 1, "reference_answer": "ref"}]
    out = judge.score_all(results, tasks, rubric=rubric)
    assert out["average"]["correctness"] == 8.0
    assert out["average"]["style"] == 7.0
    expected = (8.0 * 0.7 + 7.0 * 0.3) * 10.0
    assert out["average"]["composite"] == pytest.approx(expected)
