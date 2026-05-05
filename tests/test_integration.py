import pytest
import json
import os
from unittest.mock import patch, MagicMock
from stem_agent import StemAgent
from discovery.llm_introspection import DomainProfile, ToolSpec
from safeguards.versioning import VersionedState

PROFILE = DomainProfile(
    domain="Deep Research",
    workflow=["search", "synthesize", "cite"],
    required_tools=["web_search", "url_reader"],
    tool_specs={},
    quality_criteria=["accuracy", "coverage"]
)

HIGH_SCORE = {
    "average": {"composite": 78.0, "accuracy": 8.0, "coverage": 7.0, "synthesis": 8.0, "citation": 7.0},
    "per_task": []
}

def make_final_state():
    return VersionedState(
        version=1, system_prompt="You are a Deep Research specialist.",
        active_tools=["web_search", "url_reader"],
        tool_schemas=[], tool_source_files={},
        composite_score=78.0, timestamp="2026-05-05T00:00:00Z"
    )

def test_full_run_produces_final_agent(tmp_path):
    agent = StemAgent(results_dir=str(tmp_path))
    with patch("stem_agent.DiscoveryEngine") as MockDisc, \
         patch("stem_agent.SpecializationEngine") as MockSpec, \
         patch("stem_agent.EvaluationEngine") as MockEval, \
         patch("stem_agent.VersionManager") as MockVM:

        MockDisc.return_value.run.return_value = PROFILE
        MockSpec.return_value.specialize.return_value = {
            "system_prompt": "You are a Deep Research specialist.",
            "tool_schemas": [], "tool_source_files": {}, "active_tools": ["web_search"]
        }
        MockEval.return_value.evaluate.return_value = HIGH_SCORE
        MockVM.return_value.list_versions.return_value = [1]
        MockVM.return_value.get_best.return_value = make_final_state()
        MockVM.return_value.rollback.return_value = make_final_state()

        final = agent.run("Deep Research")

    assert final.composite_score >= 75.0
    assert os.path.exists(os.path.join(str(tmp_path), "final_agent", "agent_config.json"))
    assert os.path.exists(os.path.join(str(tmp_path), "final_agent", "run.py"))
    assert os.path.exists(os.path.join(str(tmp_path), "scores.json"))

def test_scores_json_records_iteration(tmp_path):
    agent = StemAgent(results_dir=str(tmp_path))
    with patch("stem_agent.DiscoveryEngine") as MockDisc, \
         patch("stem_agent.SpecializationEngine") as MockSpec, \
         patch("stem_agent.EvaluationEngine") as MockEval, \
         patch("stem_agent.VersionManager") as MockVM:

        MockDisc.return_value.run.return_value = PROFILE
        MockSpec.return_value.specialize.return_value = {
            "system_prompt": "You are specialist.", "tool_schemas": [],
            "tool_source_files": {}, "active_tools": ["web_search"]
        }
        MockEval.return_value.evaluate.return_value = HIGH_SCORE
        MockVM.return_value.list_versions.return_value = []
        MockVM.return_value.get_best.return_value = make_final_state()

        agent.run("Deep Research")

    with open(os.path.join(str(tmp_path), "scores.json")) as f:
        history = json.load(f)
    assert len(history) >= 1
    assert "composite" in history[0]
    assert "iteration" in history[0]
