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
