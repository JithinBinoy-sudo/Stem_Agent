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
        lines = ["You have been provided the following sources. Use them to answer."]
        for r in results[:6]:
            url = r.get("url", "")
            title = r.get("title", "")
            content = (r.get("content") or "")[:400]
            lines.append(f"- {url} — {title}: {content}")
        lines.append(
            "\nCITATION REQUIREMENTS (your score depends on this):\n"
            "- Write a coherent paragraph or short multi-paragraph answer (NOT bullet points).\n"
            "- After EVERY sentence that states a fact, append the exact source URL in square brackets, e.g. `[https://en.wikipedia.org/wiki/Python_(programming_language)]`.\n"
            "- Use at least 4 different source URLs across the answer (vary the citations between sentences).\n"
            "- Use the URLs from the list above verbatim — do not invent or shorten them.\n"
            "- Do NOT add a 'References' or 'Sources' section. All citations must be inline `[https://...]` at sentence end.\n"
            "- Keep your usual depth and synthesis quality — citations are added to good prose, not in place of it."
        )
        return "\n".join(lines)

    def _polish_citations(self, answer: str, prefetched: str) -> str:
        if not getattr(config, "POLISH_CITATIONS", True):
            return answer
        if not answer.strip() or not prefetched:
            return answer
        polish_prompt = (
            "Rewrite the DRAFT ANSWER below so that every sentence that states a fact "
            "ends with the most relevant source URL in square brackets, e.g. "
            "`[https://exact-url-from-sources]`. Use ONLY the URLs from the SOURCES list "
            "below — do not invent, shorten, or paraphrase URLs. Keep all factual content "
            "and prose style intact; only adjust where citations appear. Use at least 4 "
            "different source URLs across the rewritten answer. Do not add a References "
            "section. Output ONLY the rewritten answer, no preamble.\n\n"
            f"SOURCES:\n{prefetched}\n\n"
            f"DRAFT ANSWER:\n{answer}\n\n"
            "REWRITTEN ANSWER:"
        )
        try:
            response = self._client.chat.completions.create(
                model=config.MODEL_WEAK,
                messages=[{"role": "user", "content": polish_prompt}],
            )
            polished = response.choices[0].message.content or ""
            return polished.strip() or answer
        except Exception as e:
            import sys
            print(f"[benchmark] polish failed: {type(e).__name__}: {e}", file=sys.stderr)
            return answer

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
                    raw_answer = choice.message.content or ""
                    polished = self._polish_citations(raw_answer, prefetched)
                    return TaskResult(
                        task_id=task["id"],
                        query=task["query"],
                        answer=polished,
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
            polished = self._polish_citations(last_content, prefetched)
            return TaskResult(task_id=task["id"], query=task["query"],
                               answer=polished, tool_calls_made=tool_calls_made, failed=False)
        except Exception as e:
            import sys, traceback
            print(f"[benchmark] task {task['id']} failed: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return TaskResult(task_id=task["id"], query=task["query"],
                               answer="", tool_calls_made=tool_calls_made, failed=True)

    def run_all(self, tasks: list, system_prompt: str, tool_schemas: list) -> list:
        return [self.run_task(task, system_prompt, tool_schemas) for task in tasks]
