import pytest
import json
import os
from unittest.mock import patch, MagicMock, PropertyMock
from stem_agent import StemAgent
from discovery.llm_introspection import DomainProfile, ToolSpec
from safeguards.versioning import VersionedState

MOCK_PROFILE = DomainProfile(
    domain="Deep Research",
    workflow=["search", "synthesize"],
    required_tools=["web_search", "url_reader"],
    tool_specs={},
    quality_criteria=["accuracy"]
)

MOCK_SCORES_LOW = {"average": {"composite": 40.0, "accuracy": 4.0, "coverage": 4.0, "synthesis": 4.0, "citation": 4.0}, "per_task": []}
MOCK_SCORES_HIGH = {"average": {"composite": 80.0, "accuracy": 8.0, "coverage": 8.0, "synthesis": 8.0, "citation": 8.0}, "per_task": []}

def test_run_returns_final_state(tmp_path):
    agent = StemAgent(results_dir=str(tmp_path))
    with patch("stem_agent.DiscoveryEngine") as MockDisc, \
         patch("stem_agent.SpecializationEngine") as MockSpec, \
         patch("stem_agent.EvaluationEngine") as MockEval, \
         patch("stem_agent.VersionManager") as MockVM:

        MockDisc.return_value.run.return_value = MOCK_PROFILE
        MockSpec.return_value.specialize.return_value = {
            "system_prompt": "You are specialist.", "tool_schemas": [],
            "tool_source_files": {}, "active_tools": ["web_search"]
        }
        MockEval.return_value.evaluate.return_value = MOCK_SCORES_HIGH
        MockVM.return_value.list_versions.return_value = []
        MockVM.return_value.get_best.return_value = VersionedState(
            version=1, system_prompt="You are specialist.", active_tools=["web_search"],
            tool_schemas=[], tool_source_files={}, composite_score=80.0, timestamp=""
        )

        result = agent.run("Deep Research")
    assert result is not None
    assert result.composite_score >= 75.0

def test_run_stops_when_threshold_met(tmp_path):
    agent = StemAgent(results_dir=str(tmp_path))
    with patch("stem_agent.DiscoveryEngine") as MockDisc, \
         patch("stem_agent.SpecializationEngine") as MockSpec, \
         patch("stem_agent.EvaluationEngine") as MockEval, \
         patch("stem_agent.VersionManager") as MockVM:

        MockDisc.return_value.run.return_value = MOCK_PROFILE
        MockSpec.return_value.specialize.return_value = {
            "system_prompt": "You are specialist.", "tool_schemas": [],
            "tool_source_files": {}, "active_tools": ["web_search"]
        }
        MockEval.return_value.evaluate.return_value = MOCK_SCORES_HIGH
        MockVM.return_value.list_versions.return_value = []
        MockVM.return_value.get_best.return_value = VersionedState(
            version=1, system_prompt="", active_tools=[], tool_schemas=[],
            tool_source_files={}, composite_score=80.0, timestamp=""
        )

        agent.run("Deep Research")
    assert MockEval.return_value.evaluate.call_count == 1

def test_run_triggers_rollback_on_regression(tmp_path):
    agent = StemAgent(results_dir=str(tmp_path))
    scores = [MOCK_SCORES_LOW, {"average": {"composite": 30.0, "accuracy": 3.0, "coverage": 3.0, "synthesis": 3.0, "citation": 3.0}, "per_task": []}]
    with patch("stem_agent.DiscoveryEngine") as MockDisc, \
         patch("stem_agent.SpecializationEngine") as MockSpec, \
         patch("stem_agent.EvaluationEngine") as MockEval, \
         patch("stem_agent.VersionManager") as MockVM:

        MockDisc.return_value.run.return_value = MOCK_PROFILE
        MockSpec.return_value.specialize.return_value = {
            "system_prompt": "You are specialist.", "tool_schemas": [],
            "tool_source_files": {}, "active_tools": ["web_search"]
        }
        MockEval.return_value.evaluate.side_effect = scores * 5
        prev_state = VersionedState(version=1, system_prompt="v1", active_tools=[],
                                     tool_schemas=[], tool_source_files={}, composite_score=40.0, timestamp="")
        MockVM.return_value.rollback.return_value = prev_state
        MockVM.return_value.list_versions.return_value = [1]
        MockVM.return_value.get_best.return_value = prev_state

        agent.run("Deep Research")
    MockVM.return_value.rollback.assert_called()


# Phase 4: dynamic diagnostic patch tests

def test_diagnose_patch_uses_known_template_for_citation():
    from stem_agent import _diagnose_patch
    prev = {"accuracy": 9.0, "coverage": 8.0, "synthesis": 7.0, "citation": 1.0, "composite": 50.0}
    rubric = [
        {"name": "accuracy", "weight": 0.35, "method": "llm"},
        {"name": "synthesis", "weight": 0.25, "method": "llm"},
        {"name": "coverage", "weight": 0.25, "method": "deterministic"},
        {"name": "citation", "weight": 0.15, "method": "deterministic"},
    ]
    patch, weakest = _diagnose_patch(prev, rubric)
    assert weakest == "citation"
    assert "EVERY single sentence" in patch
    assert "https://exact-source-url" in patch


def test_diagnose_patch_generates_dynamically_for_unknown_criterion():
    from stem_agent import _diagnose_patch
    prev = {"correctness": 9.0, "security": 2.0, "style": 7.0, "composite": 50.0}
    rubric = [
        {"name": "correctness", "weight": 0.4, "method": "llm", "description": "code is functionally correct"},
        {"name": "security", "weight": 0.4, "method": "llm", "description": "code has no security holes"},
        {"name": "style", "weight": 0.2, "method": "llm", "description": "code follows style conventions"},
    ]
    with patch("stem_agent.OpenAI") as MockOpenAI:
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = "Audit input boundaries and avoid eval/exec calls."
        MockOpenAI.return_value.chat.completions.create.return_value = fake_response
        patch_text, weakest = _diagnose_patch(prev, rubric)
    assert weakest == "security"
    assert "security was lowest" in patch_text
    assert "Audit input boundaries" in patch_text
