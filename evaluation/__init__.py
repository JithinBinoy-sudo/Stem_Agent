from evaluation.benchmark import BenchmarkRunner, TaskResult
from evaluation.judge import Judge, CriterionScores

class EvaluationEngine:
    def __init__(self):
        self._runner = BenchmarkRunner()
        self._judge = Judge()

    def evaluate(self, tasks: list, system_prompt: str, tool_schemas: list, is_final: bool = False, pipeline: dict | None = None, rubric: list | None = None) -> dict:
        results = self._runner.run_all(tasks, system_prompt, tool_schemas, pipeline)
        return self._judge.score_all(results, tasks, is_final=is_final, rubric=rubric)
