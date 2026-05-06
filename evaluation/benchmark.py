import json
from dataclasses import dataclass
from openai import OpenAI
from tools import get_tool
import config

@dataclass
class TaskResult:
    task_id: int
    query: str
    answer: str
    tool_calls_made: int
    failed: bool

class BenchmarkRunner:
    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def run_task(self, task: dict, system_prompt: str, tool_schemas: list) -> TaskResult:
        wrapped_query = (
            f"{task['query']}\n\n"
            "Before answering, you MUST: (1) call web_search at least twice with different queries, "
            "(2) call url_reader on at least 5 results from DIFFERENT domains, "
            "(3) write your answer with [https://source-url] after every factual sentence. "
            "Do not answer from memory."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": wrapped_query},
        ]
        tool_calls_made = 0
        try:
            for step in range(config.MAX_TOOL_CALLS_PER_TASK):
                kwargs = {"model": config.MODEL_STRONG, "messages": messages}
                if tool_schemas:
                    kwargs["tools"] = tool_schemas
                    if step == 0:
                        kwargs["tool_choice"] = "required"
                response = self._client.chat.completions.create(**kwargs)
                choice = response.choices[0]

                if choice.finish_reason == "stop" or not choice.message.tool_calls:
                    return TaskResult(
                        task_id=task["id"],
                        query=task["query"],
                        answer=choice.message.content or "",
                        tool_calls_made=tool_calls_made,
                        failed=False,
                    )

                messages.append({"role": "assistant", "content": None,
                                  "tool_calls": [tc.model_dump() for tc in choice.message.tool_calls]})
                for tc in choice.message.tool_calls:
                    tool_calls_made += 1
                    tool_fn = get_tool(tc.function.name)
                    args = json.loads(tc.function.arguments)
                    tool_result = tool_fn(**args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_result),
                    })

            last_content = next(
                (m.get("content", "") for m in reversed(messages) if m["role"] == "assistant" and m.get("content")),
                ""
            )
            return TaskResult(task_id=task["id"], query=task["query"],
                               answer=last_content, tool_calls_made=tool_calls_made, failed=False)
        except Exception as e:
            import sys, traceback
            print(f"[benchmark] task {task['id']} failed: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return TaskResult(task_id=task["id"], query=task["query"],
                               answer="", tool_calls_made=tool_calls_made, failed=True)

    def run_all(self, tasks: list, system_prompt: str, tool_schemas: list) -> list:
        return [self.run_task(task, system_prompt, tool_schemas) for task in tasks]
