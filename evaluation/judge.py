import json
import re
from dataclasses import dataclass
from openai import OpenAI
from evaluation.benchmark import TaskResult
import config

@dataclass
class CriterionScores:
    accuracy: float
    coverage: float
    synthesis: float
    citation: float

def compute_coverage_score(unique_domain_count: int) -> float:
    return min(unique_domain_count / 5, 1.0) * 10.0

def compute_citation_score(text: str) -> float:
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r'\[.+?\]|\(https?://\S+\)|https?://\S+', s))
    return (cited / len(sentences)) * 10.0

def composite_score(scores: CriterionScores) -> float:
    return (scores.accuracy * 0.35 + scores.coverage * 0.25 +
            scores.synthesis * 0.25 + scores.citation * 0.15) * 10.0

def _count_unique_domains(text: str) -> int:
    urls = re.findall(r'https?://([^/\s\)]+)', text)
    domains = {u.split("/")[0] for u in urls}
    return len(domains)

JUDGE_PROMPT = """You are an expert evaluator. Score the following research outputs.

For each output, return one JSON object with two integer fields, each 0-10:
- "accuracy": how factually correct the output is compared to the reference answer
- "synthesis": coherence, depth, and quality of the synthesis

Reference answers and outputs:
{pairs}

Return ONLY a JSON object with a single key "scores" whose value is an array of {n} score objects in the same order as the tasks above. Example for 2 tasks:
{{"scores": [{{"accuracy": 7, "synthesis": 8}}, {{"accuracy": 4, "synthesis": 6}}]}}
"""


def _extract_scores_list(raw, expected_len: int) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        if "scores" in raw and isinstance(raw["scores"], list):
            return raw["scores"]
        for v in raw.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                return v
        if all(isinstance(v, dict) for v in raw.values()) and len(raw) == expected_len:
            return list(raw.values())
    return []

class Judge:
    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def score_all(self, results, tasks, is_final: bool = False) -> dict:
        import sys
        model = config.MODEL_STRONG if is_final else config.MODEL_WEAK
        pairs = "\n\n".join(
            f"Task {r.task_id}:\nReference: {t['reference_answer']}\nOutput: {r.answer}"
            for r, t in zip(results, tasks)
        )
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": JUDGE_PROMPT.format(pairs=pairs, n=len(results))}],
                response_format={"type": "json_object"},
            )
            raw = json.loads(response.choices[0].message.content)
            llm_scores = _extract_scores_list(raw, expected_len=len(results))
            if len(llm_scores) != len(results):
                print(f"[judge] expected {len(results)} score objects, got {len(llm_scores)}; raw shape={type(raw).__name__} keys={list(raw.keys()) if isinstance(raw, dict) else 'N/A'}", file=sys.stderr)
                llm_scores = (llm_scores + [{}] * len(results))[:len(results)]
        except Exception as e:
            print(f"[judge] scoring failed: {type(e).__name__}: {e}", file=sys.stderr)
            llm_scores = [{"accuracy": 0, "synthesis": 0}] * len(results)

        all_scores = []
        for r, llm in zip(results, llm_scores):
            if r.failed:
                all_scores.append(CriterionScores(accuracy=0.0, coverage=0.0, synthesis=0.0, citation=0.0))
                continue
            accuracy = float(llm.get("accuracy", 0))
            synthesis = float(llm.get("synthesis", 0))
            coverage = compute_coverage_score(_count_unique_domains(r.answer))
            citation = compute_citation_score(r.answer)
            all_scores.append(CriterionScores(accuracy=accuracy, coverage=coverage,
                                               synthesis=synthesis, citation=citation))

        avg = CriterionScores(
            accuracy=sum(s.accuracy for s in all_scores) / len(all_scores),
            coverage=sum(s.coverage for s in all_scores) / len(all_scores),
            synthesis=sum(s.synthesis for s in all_scores) / len(all_scores),
            citation=sum(s.citation for s in all_scores) / len(all_scores),
        )
        return {
            "per_task": [{"task_id": r.task_id, "accuracy": s.accuracy, "coverage": s.coverage,
                           "synthesis": s.synthesis, "citation": s.citation,
                           "composite": composite_score(s)}
                          for r, s in zip(results, all_scores)],
            "average": {
                "accuracy": avg.accuracy,
                "coverage": avg.coverage,
                "synthesis": avg.synthesis,
                "citation": avg.citation,
                "composite": composite_score(avg),
            }
        }
