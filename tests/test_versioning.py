import pytest
import json
import os
import tempfile
from safeguards.versioning import VersionedState, VersionManager

SAMPLE_STATE = VersionedState(
    version=1,
    system_prompt="You are a Deep Research specialist.",
    active_tools=["web_search", "url_reader"],
    tool_schemas=[{"type": "function", "function": {"name": "web_search"}}],
    tool_source_files={"url_reader": "def url_reader(url): return {'text': ''}"},
    composite_score=52.5,
    timestamp="2026-05-05T10:00:00Z"
)

@pytest.fixture
def tmp_versions_dir(tmp_path):
    return str(tmp_path / "versions")

def test_save_creates_json_file(tmp_versions_dir):
    mgr = VersionManager(versions_dir=tmp_versions_dir)
    mgr.save(SAMPLE_STATE)
    expected_path = os.path.join(tmp_versions_dir, "v1.json")
    assert os.path.exists(expected_path)

def test_load_restores_all_fields(tmp_versions_dir):
    mgr = VersionManager(versions_dir=tmp_versions_dir)
    mgr.save(SAMPLE_STATE)
    loaded = mgr.load(version=1)
    assert loaded.version == 1
    assert loaded.system_prompt == SAMPLE_STATE.system_prompt
    assert loaded.active_tools == SAMPLE_STATE.active_tools
    assert loaded.composite_score == SAMPLE_STATE.composite_score

def test_get_best_returns_highest_score(tmp_versions_dir):
    mgr = VersionManager(versions_dir=tmp_versions_dir)
    state_a = VersionedState(version=1, system_prompt="a", active_tools=[],
                              tool_schemas=[], tool_source_files={}, composite_score=40.0, timestamp="")
    state_b = VersionedState(version=2, system_prompt="b", active_tools=[],
                              tool_schemas=[], tool_source_files={}, composite_score=68.0, timestamp="")
    mgr.save(state_a)
    mgr.save(state_b)
    best = mgr.get_best()
    assert best.version == 2
    assert best.composite_score == 68.0

def test_rollback_returns_previous_version(tmp_versions_dir):
    mgr = VersionManager(versions_dir=tmp_versions_dir)
    state_v1 = VersionedState(version=1, system_prompt="v1", active_tools=[],
                               tool_schemas=[], tool_source_files={}, composite_score=50.0, timestamp="")
    state_v2 = VersionedState(version=2, system_prompt="v2", active_tools=[],
                               tool_schemas=[], tool_source_files={}, composite_score=45.0, timestamp="")
    mgr.save(state_v1)
    mgr.save(state_v2)
    restored = mgr.rollback(from_version=2)
    assert restored.version == 1
    assert restored.system_prompt == "v1"

def test_rollback_with_no_previous_raises(tmp_versions_dir):
    mgr = VersionManager(versions_dir=tmp_versions_dir)
    mgr.save(SAMPLE_STATE)
    with pytest.raises(ValueError, match="No previous version"):
        mgr.rollback(from_version=1)

def test_list_versions_returns_sorted(tmp_versions_dir):
    mgr = VersionManager(versions_dir=tmp_versions_dir)
    for v in [1, 2, 3]:
        s = VersionedState(version=v, system_prompt="", active_tools=[],
                            tool_schemas=[], tool_source_files={}, composite_score=float(v*10), timestamp="")
        mgr.save(s)
    versions = mgr.list_versions()
    assert versions == [1, 2, 3]
