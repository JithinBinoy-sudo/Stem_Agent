import json
import os
from datetime import datetime, timezone
from openai import OpenAI
from discovery import DiscoveryEngine
from specialization import SpecializationEngine
from evaluation import EvaluationEngine
from safeguards.versioning import VersionedState, VersionManager
import config


_KNOWN_DIAGNOSTIC_PATCHES = {
    "citation": (
        "EVERY single sentence that states a fact MUST end with `[https://exact-source-url]` "
        "immediately before the period. Do NOT add a 'References' section. Prose attribution "
        "like 'according to Wikipedia' does not count — only inline `[https://...]` brackets count."
    ),
    "coverage": (
        "Cite at least 5 DIFFERENT domain URLs across your answer. Do not cite the same URL "
        "more than twice. Spread citations across multiple sentences using different sources "
        "for different facts."
    ),
    "synthesis": (
        "Write coherent multi-sentence prose that explicitly connects facts (use transitions "
        "like 'because', 'as a result', 'furthermore'). Do not produce a list of disconnected "
        "bullet points. Show why facts matter together, not just that they are true."
    ),
    "accuracy": (
        "State only facts that appear verbatim in the provided sources. If a fact is not in "
        "the sources, do not include it. Quote source phrasings closely rather than paraphrasing."
    ),
}


def _generate_patch_for_unknown_criterion(criterion_name: str, score: float, criterion_description: str = "") -> str:
    """LLM-generate a targeted improvement instruction for a criterion not in the known set."""
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        prompt = (
            f"An AI agent was scored on the criterion '{criterion_name}' "
            f"({criterion_description or 'no description provided'}) and got {score:.1f}/10. "
            f"Write a single concise instruction (40-70 words) that the agent should follow "
            f"to improve specifically on '{criterion_name}'. Output ONLY the instruction, "
            f"no preamble, no quotes."
        )
        response = client.chat.completions.create(
            model=config.MODEL_WEAK,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        import sys
        print(f"[stem_agent] dynamic patch generation failed: {type(e).__name__}: {e}", file=sys.stderr)
        return f"Improve the '{criterion_name}' aspect of your answer; previous score was {score:.1f}/10."


def _diagnose_patch(prev_average: dict, rubric: list | None = None) -> tuple[str, str]:
    """Return (patch_text, weakest_criterion_name). patch_text is empty string if no diagnosis possible."""
    if not prev_average:
        return "", ""
    if rubric:
        criteria_names = [c["name"] for c in rubric]
    else:
        criteria_names = ["accuracy", "coverage", "synthesis", "citation"]
    scored = {n: prev_average.get(n, 10.0) for n in criteria_names if n in prev_average}
    if not scored:
        return "", ""
    weakest = min(scored, key=scored.get)
    score = scored[weakest]
    if weakest in _KNOWN_DIAGNOSTIC_PATCHES:
        body = _KNOWN_DIAGNOSTIC_PATCHES[weakest]
    else:
        description = ""
        if rubric:
            for c in rubric:
                if c["name"] == weakest:
                    description = c.get("description", "")
                    break
        body = _generate_patch_for_unknown_criterion(weakest, score, description)
    return f"\n\nDIAGNOSTIC PATCH ({weakest} was lowest last iteration, score {score:.1f}/10): {body}", weakest


class StemAgent:
    def __init__(self, results_dir: str = config.RESULTS_DIR):
        self._results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "versions"), exist_ok=True)

    def run(self, domain: str, force_rediscover: bool = False) -> VersionedState:
        discovery = DiscoveryEngine()
        specialization = SpecializationEngine()
        evaluation = EvaluationEngine()
        version_mgr = VersionManager(versions_dir=os.path.join(self._results_dir, "versions"))

        tasks = self._load_tasks(domain)
        profile = discovery.run(domain, force=force_rediscover)

        scores_history = []
        last_average = None
        current_source_files = {}
        iteration = 0

        while iteration < config.MAX_ITERATIONS:
            print(f"\n[Iteration {iteration}] Specializing...")
            spec_result = specialization.specialize(profile, current_source_files)
            is_final_attempt = (iteration == config.MAX_ITERATIONS - 1)

            patched_prompt = spec_result["system_prompt"]
            if getattr(config, "DIAGNOSE_AND_REFINE", True) and last_average is not None:
                patch, weakest = _diagnose_patch(last_average, profile.rubric)
                if patch:
                    patched_prompt = patched_prompt + patch
                    print(f"[Iteration {iteration}] Diagnostic patch applied (weakest: {weakest}, score {last_average.get(weakest, 0):.1f})")

            print(f"[Iteration {iteration}] Evaluating...")
            eval_schemas = spec_result["tool_schemas"]
            if getattr(config, "BENCHMARK_BASE_TOOLS_ONLY", False):
                eval_schemas = [
                    s for s in eval_schemas
                    if s["function"]["name"] in {"web_search", "url_reader"}
                ]
            pipeline = {
                "prefetch_sources": profile.requires_web_research,
                "polish_citations": profile.requires_citations,
            }
            scores = evaluation.evaluate(
                tasks,
                patched_prompt,
                eval_schemas,
                is_final=is_final_attempt,
                pipeline=pipeline,
                rubric=profile.rubric,
            )
            composite = scores["average"]["composite"]
            print(f"[Iteration {iteration}] Composite score: {composite:.1f}")

            state = VersionedState(
                version=iteration,
                system_prompt=patched_prompt,
                active_tools=spec_result["active_tools"],
                tool_schemas=spec_result["tool_schemas"],
                tool_source_files=spec_result["tool_source_files"],
                composite_score=composite,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
            version_mgr.save(state)
            self._append_score(iteration, scores)

            # Check regression
            if scores_history and composite < scores_history[-1]:
                print(f"[Iteration {iteration}] Regression detected. Rolling back.")
                version_mgr.rollback(from_version=iteration)
                break

            scores_history.append(composite)
            last_average = scores["average"]
            current_source_files = spec_result["tool_source_files"]

            # Check stopping criteria
            if composite >= config.SCORE_THRESHOLD:
                print(f"[Iteration {iteration}] Threshold reached. Stopping.")
                break

            if len(scores_history) >= 2:
                improvement = scores_history[-1] - scores_history[-2]
                if improvement < config.MIN_IMPROVEMENT:
                    print(f"[Iteration {iteration}] Improvement {improvement:.1f} < {config.MIN_IMPROVEMENT}. Stopping.")
                    break

            iteration += 1

        best = version_mgr.get_best()
        self._export_final_agent(best)
        return best

    def _load_tasks(self, domain: str) -> list:
        from discovery.llm_introspection import sanitize_tool_name
        slug = sanitize_tool_name(domain)
        path = os.path.join("benchmarks", f"{slug}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No benchmark file for domain '{domain}'. "
                f"Expected: {path}. "
                f"Create the file with a 'tasks' list (see benchmarks/deep_research.json for the schema)."
            )
        with open(path) as f:
            return json.load(f)["tasks"]

    def _append_score(self, iteration: int, scores: dict):
        path = os.path.join(self._results_dir, "scores.json")
        history = []
        if os.path.exists(path):
            with open(path) as f:
                history = json.load(f)
        history.append({"iteration": iteration, **scores["average"]})
        with open(path, "w") as f:
            json.dump(history, f, indent=2)

    def _export_final_agent(self, state: VersionedState):
        final_dir = os.path.join(self._results_dir, "final_agent")
        tools_dir = os.path.join(final_dir, "tools")
        os.makedirs(tools_dir, exist_ok=True)

        config_data = {
            "system_prompt": state.system_prompt,
            "active_tools": state.active_tools,
            "tool_schemas": state.tool_schemas,
            "composite_score": state.composite_score,
            "version": state.version,
        }
        with open(os.path.join(final_dir, "agent_config.json"), "w") as f:
            json.dump(config_data, f, indent=2)

        for name, code in state.tool_source_files.items():
            with open(os.path.join(tools_dir, f"{name}.py"), "w") as f:
                f.write(code)

        run_py = '''#!/usr/bin/env python3
import json, importlib.util, os, sys
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
AGENT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(AGENT_DIR, "agent_config.json")) as f:
    cfg = json.load(f)

tool_registry = {}
for fname in os.listdir(os.path.join(AGENT_DIR, "tools")):
    if fname.endswith(".py"):
        name = fname[:-3]
        spec = importlib.util.spec_from_file_location(name, os.path.join(AGENT_DIR, "tools", fname))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        tool_registry[name] = getattr(mod, name)

# Add base tools
from tools.base_tools import web_search, url_reader
tool_registry["web_search"] = web_search
tool_registry["url_reader"] = url_reader

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
query = sys.stdin.read().strip() if not sys.stdin.isatty() else input("Query: ")
messages = [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": query}]

for _ in range(10):
    resp = client.chat.completions.create(model="gpt-4o", messages=messages, tools=cfg["tool_schemas"] or None)
    choice = resp.choices[0]
    if choice.finish_reason == "stop" or not choice.message.tool_calls:
        print(choice.message.content)
        break
    messages.append({"role": "assistant", "content": None,
                      "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls]})
    for tc in choice.message.tool_calls:
        result = tool_registry[tc.function.name](**json.loads(tc.function.arguments))
        messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
'''
        with open(os.path.join(final_dir, "run.py"), "w") as f:
            f.write(run_py)
