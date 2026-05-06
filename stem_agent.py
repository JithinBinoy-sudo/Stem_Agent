import json
import os
from datetime import datetime, timezone
from discovery import DiscoveryEngine
from specialization import SpecializationEngine
from evaluation import EvaluationEngine
from safeguards.versioning import VersionedState, VersionManager
import config

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

        tasks = self._load_tasks()
        profile = discovery.run(domain, force=force_rediscover)

        scores_history = []
        current_source_files = {}
        iteration = 0

        while iteration < config.MAX_ITERATIONS:
            print(f"\n[Iteration {iteration}] Specializing...")
            spec_result = specialization.specialize(profile, current_source_files)
            is_final_attempt = (iteration == config.MAX_ITERATIONS - 1)

            print(f"[Iteration {iteration}] Evaluating...")
            eval_schemas = spec_result["tool_schemas"]
            if getattr(config, "BENCHMARK_BASE_TOOLS_ONLY", False):
                eval_schemas = [
                    s for s in eval_schemas
                    if s["function"]["name"] in {"web_search", "url_reader"}
                ]
            scores = evaluation.evaluate(
                tasks,
                spec_result["system_prompt"],
                eval_schemas,
                is_final=is_final_attempt,
            )
            composite = scores["average"]["composite"]
            print(f"[Iteration {iteration}] Composite score: {composite:.1f}")

            state = VersionedState(
                version=iteration,
                system_prompt=spec_result["system_prompt"],
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

    def _load_tasks(self) -> list:
        with open("benchmarks/deep_research_tasks.json") as f:
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
