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

    def _prefetch_sources(self, query: str) -> str:
        if not getattr(config, "PREFETCH_SOURCES", True):
            return ""
        try:
            results = get_tool("web_search")(query=query).get("results", [])
        except Exception as e:
            import sys
            print(f"[benchmark] prefetch failed: {type(e).__name__}: {e}", file=sys.stderr)
            return ""
        if not results:
            return ""
        lines = ["You have been provided the following sources. You MUST use them to answer."]
        for i, r in enumerate(results[:6], 1):
            url = r.get("url", "")
            title = r.get("title", "")
            content = (r.get("content") or "")[:400]
            lines.append(f"SOURCE {i}: {url}\nTitle: {title}\nExcerpt: {content}")
        lines.append(
            "\nOUTPUT FORMAT — NON-NEGOTIABLE:\n"
            "- Answer as 5 to 10 bullet points. Each bullet states one fact.\n"
            "- Every bullet MUST end with the EXACT source URL in square brackets, e.g. `[https://en.wikipedia.org/wiki/Tim_Berners-Lee]`.\n"
            "- Use the URLs above verbatim — do not invent, shorten, or paraphrase them.\n"
            "- Cite at least 4 DIFFERENT source URLs across your bullets (use different sources for different facts).\n"
            "- Do NOT write paragraphs. Do NOT add a 'References' or 'Sources' section. Cite inline only.\n"
            "- Do NOT use prose attributions like 'according to Wikipedia' — only `[https://...]` brackets count."
        )
        return "\n".join(lines)

    def run_task(self, task: dict, system_prompt: str, tool_schemas: list) -> TaskResult:
        prefetched = ""
        if any(s.get("function", {}).get("name") == "web_search" for s in tool_schemas):
            prefetched = self._prefetch_sources(task["query"])
        user_content = task["query"]
        if prefetched:
            user_content = f"{task['query']}\n\n{prefetched}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        tool_calls_made = 0
        try:
            for _ in range(config.MAX_TOOL_CALLS_PER_TASK):
                kwargs = {"model": config.MODEL_STRONG, "messages": messages}
                if tool_schemas:
                    kwargs["tools"] = tool_schemas
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
