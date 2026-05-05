import json
import pytest

def load_tasks():
    with open("benchmarks/deep_research_tasks.json") as f:
        return json.load(f)

def test_benchmark_has_five_tasks():
    data = load_tasks()
    assert len(data["tasks"]) == 5

def test_each_task_has_required_fields():
    data = load_tasks()
    for task in data["tasks"]:
        assert "id" in task
        assert "query" in task
        assert "reference_answer" in task
        assert "key_facts" in task
        assert isinstance(task["key_facts"], list)
        assert len(task["key_facts"]) >= 2

def test_task_ids_are_unique():
    data = load_tasks()
    ids = [t["id"] for t in data["tasks"]]
    assert len(ids) == len(set(ids))
