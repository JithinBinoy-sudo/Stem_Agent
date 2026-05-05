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
    runner = BenchmarkRunner()
    tool_response = make_mock_response(tool_call={"name": "web_search", "args": {"query": "test"}})
    responses = [tool_response] * 15
    with patch("evaluation.benchmark.get_tool") as mock_get_tool:
        mock_get_tool.return_value = lambda query: {"results": []}
        with patch.object(runner._client.chat.completions, "create", side_effect=responses):
            result = runner.run_task(
                task=SAMPLE_TASKS[0],
                system_prompt="You are helpful.",
                tool_schemas=[{"type": "function", "function": {"name": "web_search"}}],
            )
    assert result.tool_calls_made == 10
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
