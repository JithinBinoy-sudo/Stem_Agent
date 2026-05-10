import json
import re
import sys
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
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
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


DETERMINISTIC_SCORERS = {
    "unique_domains": lambda r: compute_coverage_score(_count_unique_domains(r.answer)),
    "sentence_citations": lambda r: compute_citation_score(r.answer),
}


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


def _build_judge_prompt(pairs: str, llm_criteria: list, n_tasks: int) -> str:
    criteria_lines = "\n".join(
        f"- \"{c['name']}\": {c.get('description', c['name'])} (integer 0-10)"
        for c in llm_criteria
    )
    example_obj = ", ".join(f'"{c["name"]}": 7' for c in llm_criteria)
    return (
        "You are an expert evaluator. Score each output below on each criterion.\n\n"
        "For each output, return a JSON object with these integer fields, each 0-10:\n"
        f"{criteria_lines}\n\n"
        "Reference answers and outputs:\n"
        f"{pairs}\n\n"
        f"Return ONLY a JSON object with a single key \"scores\" whose value is an array of {n_tasks} score objects in the same order as the tasks above. Example for 2 tasks:\n"
        f"{{\"scores\": [{{{example_obj}}}, {{{example_obj}}}]}}\n"
    )


class Judge:
    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def score_all(self, results, tasks, is_final: bool = False, rubric: list | None = None) -> dict:
        if rubric is None:
            from discovery.llm_introspection import DEFAULT_RUBRIC
            rubric = DEFAULT_RUBRIC

        llm_criteria = [c for c in rubric if c.get("method") == "llm"]

        if llm_criteria:
            llm_scores_per_task = self._call_llm_judge(results, tasks, llm_criteria, is_final)
        else:
            llm_scores_per_task = [{} for _ in results]

        per_task = []
        sums = {c["name"]: 0.0 for c in rubric}
        for r, llm_scores in zip(results, llm_scores_per_task):
            task_scores = {}
            for crit in rubric:
                name = crit["name"]
                if r.failed:
                    score = 0.0
                elif crit.get("method") == "llm":
                    try:
                        score = float(llm_scores.get(name, 0))
                    except (TypeError, ValueError):
                        score = 0.0
                else:
                    fn = DETERMINISTIC_SCORERS.get(crit.get("scorer", ""))
                    score = float(fn(r)) if fn else 0.0
                task_scores[name] = score
                sums[name] += score
            composite = sum(task_scores[c["name"]] * c["weight"] for c in rubric) * 10.0
            per_task.append({"task_id": r.task_id, "composite": composite, **task_scores})

        n = len(results) or 1
        averages = {name: total / n for name, total in sums.items()}
        avg_composite = sum(averages[c["name"]] * c["weight"] for c in rubric) * 10.0
        averages["composite"] = avg_composite

        return {"per_task": per_task, "average": averages}

    def _call_llm_judge(self, results, tasks, llm_criteria, is_final):
        model = config.MODEL_STRONG if is_final else config.MODEL_WEAK
        pairs = "\n\n".join(
            f"Task {r.task_id}:\nReference: {t['reference_answer']}\nOutput: {r.answer}"
            for r, t in zip(results, tasks)
        )
        prompt = _build_judge_prompt(pairs, llm_criteria, len(results))
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            raw = json.loads(response.choices[0].message.content)
            llm_scores = _extract_scores_list(raw, expected_len=len(results))
            if len(llm_scores) != len(results):
                print(
                    f"[judge] expected {len(results)} score objects, got {len(llm_scores)}; "
                    f"raw shape={type(raw).__name__} keys={list(raw.keys()) if isinstance(raw, dict) else 'N/A'}",
                    file=sys.stderr,
                )
                llm_scores = (llm_scores + [{}] * len(results))[: len(results)]
            return llm_scores
        except Exception as e:
            print(f"[judge] scoring failed: {type(e).__name__}: {e}", file=sys.stderr)
            return [{c["name"]: 0 for c in llm_criteria} for _ in results]
