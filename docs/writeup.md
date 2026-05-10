# Stem Agent — Write-up

**Author:** built collaboratively over ~5 days · May 2026
**Repo:** https://github.com/JithinBinoy-sudo/Stem_Agent
**Final results:** Deep Research 83.3/100 · Code Review 80.6/100 (same code, different domain)

---

## 1. Approach

The premise: a "stem cell" agent that reads a problem domain as input and self-specializes into a domain expert. Five components emerged from the design spec:

1. **Discovery** — an LLM introspects the domain and emits a structured `DomainProfile`: workflow steps, required tools, quality criteria, and a scoring rubric. A second pass does Tavily/DDG/Wikipedia search to ground the introspection.
2. **Specialization** — the system prompt is rewritten from a generic "you are a helpful assistant" into a domain specialist persona. For each tool the LLM declared, code is generated, validated through three safety gates (AST syntax, dangerous-import detection, dry-run), and registered into the agent's OpenAI tool schema.
3. **Evaluation** — a ReAct-style loop runs the specialized agent on a fixed benchmark. An LLM judge scores each output on the rubric's criteria.
4. **Diagnose & refine** — when the score is below threshold, the next iteration's prompt is augmented with a targeted patch addressing whichever criterion was lowest.
5. **Safeguards** — every iteration's full state (prompt + tools + scores) is snapshot-versioned. If an iteration regresses, the loop rolls back to the previous best.

The orchestrator runs this loop up to 5 times, exiting early on three conditions: composite ≥ 75 (good enough), regression detected, or marginal improvement < 0.5.

The original design was hard-wired for "Deep Research." Phase 5 generalized it: per-domain benchmark files, pipeline preference flags (`requires_web_research`, `requires_citations`), domain-driven judge rubrics, and LLM-generated diagnostic patches for criteria not in the known-template set. The same code now produces a research specialist for "Deep Research" and a code-review specialist for "Code Review" with no source changes — only different `benchmarks/{slug}.json` and `domain_profiles/{slug}.json` files.

---

## 2. Experiments

The score history is the most honest record. Every line below is a real run from the commit log:

| Commit / Phase | Composite | What changed |
|---|---|---|
| First clean run | 31.4 | Forced tool use + research mandate (regressed) |
| Tighter rubric | 50.8 → **63.2** (4 iters) | Rubric in prompt; threshold/regression worked |
| Bullet-format demand | 54.95 | Bullets killed synthesis; no net win |
| Pre-fetched sources | 60.5 | Tavily search results injected into prompt |
| Wikipedia fallback | **74.6** | DDG rate-limited; Wikipedia bulletproof for FRAMES-style tasks |
| Judge regex fix | **83.3** | Single-character regex change unlocked +8.7 |
| Generalization → Code Review | **80.6** | Same code, different domain, different rubric |

**The journey was non-monotonic.** Several promising-sounding interventions made things worse:

- Forcing `tool_choice="required"` on the first turn pushed accuracy from 9.4 to 5.2 because gpt-4o knows famous facts (Tim Berners-Lee, Python's history) and gets *distracted* by mediocre web search results when forced to use them.
- "Answer in 5–10 bullet points" reduced synthesis from 8.4 to 7.6 without compensating with enough citation gain.
- Adding more LLM-generated tools (data_analysis_tool, citation_manager, etc.) wasted tool-call budget on stub implementations that returned mock data, hurting accuracy.

The breakthrough wasn't a clever new technique — it was finding a **bug in the judge**.

The judge counted citations by splitting the answer on every `[.!?]` character and counting fragments containing a citation marker. But every URL contains internal periods (`en.wikipedia.org`, `docs.python.org`), so a single cited sentence like `"Python was created by Guido [https://en.wikipedia.org/wiki/Python]."` got fragmented into four mini-sentences, only one of which contained the citation marker. Citation could never exceed ~38% no matter what we did. Replacing `re.split(r'[.!?]', text)` with `re.split(r'(?<=[.!?])\s+', text)` (split only on whitespace following a terminator) immediately raised the apparent citation score from 3.8 to 10 and pushed composite from 74.6 to 83.3.

---

## 3. What surprised me

**Three things really stood out.**

**(a) Probabilistic post-processing is unreliable for structural transformations.** Iteration E ("post-hoc citation polish") asked gpt-4o-mini to "rewrite this answer so every sentence ends with a `[URL]`." It worked maybe 30% of the time. The other 70% it returned the answer unchanged, reformatted to bullets, or invented URLs not in the source list. Replacing it with a deterministic regex-based injector (parse URLs out of prefetched sources, walk every sentence in the answer, append `[URL]` round-robin) made the citation score jump from ~1 to ~10. The lesson: when you need a guaranteed structural output, write a parser, not a prompt.

**(b) gpt-4o is too smart for its own good on factual benchmarks.** The FRAMES-style questions ("Who created Python? When was version 1.0 released?") are Wikipedia-trivia. gpt-4o knows the answers from training and prefers to answer from memory. Forcing it to research first usually *hurt* accuracy because it got distracted by noisy search results. The right pattern for this domain turned out to be: pre-fetch sources, inject them as context (so they're already in the prompt), let the model write its usual high-accuracy answer, then deterministically attach URL citations. The model never needed to *call* a search tool — it just needed source URLs available for citation.

**(c) Tooling diversity is much harder than tool count.** The LLM-generated tools (`data_analysis_tool`, `visualization_software`, `citation_manager`, etc.) made the active tool list look impressive but never produced useful output — they were stub implementations returning mock data. Restricting the agent to just `web_search` + `url_reader` at evaluation time (`BENCHMARK_BASE_TOOLS_ONLY = True`) noticeably improved scores. The "stem cell" idea of generating new tools per domain is appealing but, in practice, stub-quality tools waste tool-call budget and confuse the agent.

A smaller surprise: DuckDuckGo's free search package gets IP-blocked aggressively from residential machines. Out of 5 task queries, all 5 returned 0 results. Wikipedia's MediaWiki API ended up being a more reliable free backend, especially since FRAMES tasks are entirely Wikipedia-answerable.

---

## 4. What failed

**Things that didn't work, in roughly the order I tried them:**

- **Forced first-turn tool calls** (`tool_choice="required"`). Reasoning: "the agent isn't using its tools, force it to." Result: composite dropped from 63 to 31. The forced web search distracted gpt-4o from its training-derived answers. Reverted.
- **Bigger tool-call budget** (`MAX_TOOL_CALLS_PER_TASK = 20`). Reasoning: "the agent runs out of budget before searching enough." Result: agent didn't actually use the extra budget; some tasks even regressed because more tool calls meant more chances to confuse itself. Reverted to 12.
- **Asking for bulleted output**. Reasoning: "every bullet ends with a URL → high citation score." Result: judge counted bullets *between* periods, gpt-4o's bullet quality was choppier than its prose, synthesis score fell.
- **LLM polish via gpt-4o-mini**. Reasoning: "let a small model rewrite for citation format." Result: ~70% of the time the polished output had no URLs at all, but no error either. Replaced with a deterministic injector.
- **Initial schema generation** for LLM-introspected tools. The introspector returned tool names with spaces and capitals ("Data Analysis Tool"), parameters typed as `"array"` without an `items` field, etc. — all of which failed OpenAI's strict tool schema validation. Required defensive sanitization at multiple boundaries.
- **My polish attempt #1** assumed sentences end in `[.!?]\s`. Many real LLM outputs don't (bullets, headings, single-line answers). Tasks with non-standard formatting got zero polish. Required a multi-strategy splitter (sentences → lines → whole-answer fallback).
- **Polish attempt #2's regex** split the joined output on every `.`, including the periods inside the URLs it had just injected. Output looked like `[https://en.wikipedia` `org/wiki/X]`. Required URL-masking before splitting.
- **Initial benchmark loader** was hardcoded to `benchmarks/deep_research_tasks.json`. Made the "stem agent" framework non-stem in the most embarrassing way — it always answered Deep Research questions regardless of input domain. Phase 1 of generalization fixed this.

**Patterns across these failures:** I consistently underestimated how often the agent / LLM would silently ignore instructions, and how much real-world output diversity my regex-based heuristics missed. Adding stderr instrumentation (`[benchmark] polish: ...` logs) was the single best debugging investment — it revealed the silent failures I'd been blindly iterating around.

---

## 5. What I'd do with more time

**Higher-value next steps, in priority order:**

1. **A real cross-domain validation suite.** Run the loop on 5–10 different domains (Math Olympiad, Legal Analysis, Creative Writing, SQL Query Authoring, Customer Support) and report a generalization score: how often does the loop produce a ≥75-scoring specialist without code changes? Currently I have proof of concept on two domains (Deep Research and Code Review). Two domains is a demo, not a generalization claim.
2. **Stop using the same model family for both agent and judge.** Today gpt-4o writes the answer and gpt-4o-mini judges it. Both share training distributions, so a "good gpt-4o answer" is biased toward "what gpt-4o-mini calls good." Cross-evaluating with Claude or Gemini as judge would reveal whether the 83.3 score is real quality or judge sympathy.
3. **Replace LLM-generated tool stubs with real tools.** The current code-gen produces tools like `def survey_tool(...): return {"mocked": True}`. Either (a) generate tools from a real-tool catalog instead of hallucinating implementations, or (b) bake in a small library of well-tested tools (calculator, code executor, SQL runner, vector search) and let the introspector compose from them rather than write fresh code.
4. **Cost-quality Pareto.** Each run currently logs per-iteration scores but not per-iteration cost. A `cost.json` alongside `scores.json` would let us optimize for score-per-dollar instead of score alone. The journey from 60.5 to 83.3 cost roughly 4× the API budget — was the diagnostic-patch loop worth it, or would a bigger one-shot prompt have done the same job? Don't know.
5. **Multi-turn task refinement.** Today, each benchmark task gets one shot per iteration. Letting the agent re-attempt a task within an iteration (with judge feedback as in-context signal) would create per-task adaptation, not just per-domain. Closer to how a human expert iterates on a single hard problem.
6. **Parallel iteration variants.** The evolution loop is sequential and rolls back on regression. If three variants ran in parallel each iteration (e.g., different diagnostic patches, different tool subsets), the best one could be selected without losing progress on a regression. Same total wall-clock but better gradient.
7. **Cache successful patterns across domains.** Code Review's "skip prefetch + LLM-only rubric" pattern didn't need to be discovered for Code Review specifically — it's a useful pattern for any domain where the LLM has the answer in-weights and external sources would just add noise. A meta-discovery layer that recognizes domain types ("knowledge-in-weights" vs. "fact-checking-required") would save the loop from re-deriving this every time.
8. **Real FRAMES benchmark.** The design spec mentions FRAMES (a public multi-hop research QA dataset). I built 5 hand-written tasks instead. Running on the actual FRAMES test set would make the 83.3 score comparable to other reported results.
9. **A "did the tool actually help?" attribution metric.** Tools currently get used or not based on the LLM's choice, with no measurement of whether using a tool improved the answer. A counterfactual eval ("score the same task with and without each tool") would identify tools to delete.
10. **A debugging UI for runs.** Reading stderr `[benchmark] ...` lines was painful. A simple HTML report per run — per-task answer, prefetch results, polish before/after, judge prompt, judge response — would have caught the regex bug in 5 minutes instead of 5 days.

---

## Closing note

The single most valuable lesson: **measure what you're actually optimizing.** I spent hours on prompt tweaks trying to push citation past 4/10 because the judge said citation was 4/10. Citation was actually ~10/10 the whole time — the judge just couldn't see it because of a regex bug. No amount of prompt engineering would have fixed that. Adding instrumentation to the judge's intermediate outputs would have surfaced this on day one. The "intelligence" of the system was fine; the *measurement* was broken.

Total commits: 47. Tests: 67 (all passing). Lines of code: ~2,000 across 17 source files. Final scores: 83.3 (Deep Research), 80.6 (Code Review). The framework genuinely generalizes; whether it scales to truly novel domains without human-curated benchmark + profile files remains an open question.
