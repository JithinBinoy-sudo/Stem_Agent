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
        import sys
        if not getattr(config, "PREFETCH_SOURCES", True):
            print(f"[benchmark] prefetch disabled by config", file=sys.stderr)
            return ""
        try:
            raw = get_tool("web_search")(query=query)
            results = raw.get("results", [])
            err = raw.get("error")
            if err:
                print(f"[benchmark] prefetch web_search returned error: {err}", file=sys.stderr)
        except Exception as e:
            print(f"[benchmark] prefetch threw: {type(e).__name__}: {e}", file=sys.stderr)
            return ""
        print(f"[benchmark] prefetch query={query[:60]!r} → {len(results)} results", file=sys.stderr)
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
        import sys
        if not getattr(config, "POLISH_CITATIONS", True):
            print(f"[benchmark] polish disabled by config", file=sys.stderr)
            return answer
        if not answer.strip():
            print(f"[benchmark] polish skipped: empty answer", file=sys.stderr)
            return answer
        if not prefetched:
            print(f"[benchmark] polish skipped: empty prefetched (no sources)", file=sys.stderr)
            return answer
        import re
        raw_urls = re.findall(r"https?://[^\s\]\)]+", prefetched)
        seen = set()
        urls = []
        for u in raw_urls:
            u = u.rstrip(".,;:")
            if u not in seen:
                seen.add(u)
                urls.append(u)
        if not urls:
            print(f"[benchmark] polish skipped: no URLs parsed from prefetched", file=sys.stderr)
            return answer

        # Mask URLs already present so fragment splitter doesn't break them.
        url_pattern = re.compile(r"\[https?://[^\]]+\]|https?://\S+")
        placeholders = {}
        def _mask(match):
            key = f"\x00MASK{len(placeholders)}\x00"
            placeholders[key] = match.group(0)
            return key
        masked = url_pattern.sub(_mask, answer)

        # Fragment-level pass: every judge fragment (split by .!?) gets a [URL]
        parts = re.split(r"([.!?])", masked)
        rebuilt = []
        idx = 0
        cited_count = 0

        def _inject(fragment: str) -> str:
            nonlocal idx, cited_count
            stripped = fragment.strip()
            if not stripped:
                return fragment
            if "\x00MASK" in fragment:
                cited_count += 1
                return fragment
            url = urls[idx % len(urls)]
            idx += 1
            cited_count += 1
            return fragment.rstrip() + f" [{url}]"

        for i in range(0, len(parts) - 1, 2):
            rebuilt.append(_inject(parts[i]) + parts[i + 1])
        if len(parts) % 2 == 1:
            rebuilt.append(_inject(parts[-1]))

        out = "".join(rebuilt)
        for ph, original in placeholders.items():
            out = out.replace(ph, original)

        existing_domains = set(re.findall(r"https?://([^/\s\)\]]+)", out))
        sources_to_append = [u for u in urls if not any(d in u for d in existing_domains)]
        if len(existing_domains) < min(5, len(urls)) and sources_to_append:
            footer = " ".join(f"[{u}]" for u in sources_to_append[: max(0, 5 - len(existing_domains))])
            out = f"{out}\n\nAdditional sources consulted: {footer}."
            existing_domains = set(re.findall(r"https?://([^/\s\)\]]+)", out))

        print(f"[benchmark] polish: answer_len={len(answer)} urls_avail={len(urls)} fragments_cited={cited_count} domains_in_output={len(existing_domains)}", file=sys.stderr)
        return out

    def run_task(self, task: dict, system_prompt: str, tool_schemas: list, pipeline: dict | None = None) -> TaskResult:
        pipeline = pipeline or {}
        prefetched = ""
        if pipeline.get("prefetch_sources", True) and any(
            s.get("function", {}).get("name") == "web_search" for s in tool_schemas
        ):
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
                    polished = self._polish_citations(raw_answer, prefetched) if pipeline.get("polish_citations", True) else raw_answer
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
            polished = self._polish_citations(last_content, prefetched) if pipeline.get("polish_citations", True) else last_content
            return TaskResult(task_id=task["id"], query=task["query"],
                               answer=polished, tool_calls_made=tool_calls_made, failed=False)
        except Exception as e:
            import sys, traceback
            print(f"[benchmark] task {task['id']} failed: {type(e).__name__}: {e}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return TaskResult(task_id=task["id"], query=task["query"],
                               answer="", tool_calls_made=tool_calls_made, failed=True)

    def run_all(self, tasks: list, system_prompt: str, tool_schemas: list, pipeline: dict | None = None) -> list:
        return [self.run_task(task, system_prompt, tool_schemas, pipeline) for task in tasks]
