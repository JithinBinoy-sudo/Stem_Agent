# Stem Agent

A self-specializing AI agent. Like a stem cell reading environmental signals, it takes a problem domain as input, figures out what architecture / tools / skills it needs, rebuilds itself iteratively, and knows when to stop.

Demonstrated on two completely different domains using the same code:

| Domain | Final score | Rubric |
|---|---|---|
| Deep Research | **83.3 / 100** | accuracy · coverage · synthesis · citation |
| Code Review | **80.6 / 100** | correctness · security · actionability · conciseness |

## How it works

```
Input: domain_class = "Deep Research" (or any string)
                │
        ┌───────▼────────┐    ┌──────────────────┐    ┌──────────────────┐
        │   Discovery    │───▶│  Specialization  │───▶│    Evaluation    │
        │  (LLM + web)   │    │ (prompt + tools) │    │  (ReAct + judge) │
        └────────────────┘    └──────────────────┘    └────────┬─────────┘
                                       ▲                       │
                                       │  ┌────────────────┐   │
                                       └──│ Diagnose &     │◀──┘
                                          │ Refine (patch) │  score < threshold
                                          └────────────────┘
                                                                
        Every iteration is versioned — automatic rollback on regression.
```

Per-domain customization happens automatically via `DomainProfile`:

- **`requires_web_research`** — whether the runner pre-fetches sources before each task (true for research, false for code/math)
- **`requires_citations`** — whether the post-processor injects `[URL]` brackets after every sentence
- **`rubric`** — a list of `(name, weight, method)` criteria. `method` is either `"llm"` (graded by an LLM judge) or `"deterministic"` (built-in formula like unique-domain count or sentence-citation ratio)

## Setup

**Requires Python 3.10+ (tested on 3.14)** and an OpenAI API key. Tavily is optional (defaults to DuckDuckGo with Wikipedia fallback — both free).

```bash
git clone https://github.com/JithinBinoy-sudo/Stem_Agent.git
cd Stem_Agent
pip install -r requirements.txt
cp .env.example .env
```

Open `.env` and set:

```
OPENAI_API_KEY=sk-...your-key...
TAVILY_API_KEY=tvly-...optional-only-needed-if-SEARCH_BACKEND=tavily...
```

The default search backend in [config.py](config.py) is `duckduckgo` (free, unlimited, with automatic Wikipedia fallback if DDG is rate-limited). Set `SEARCH_BACKEND="tavily"` if you have credits and prefer Tavily's richer snippets.

## Running

### On Deep Research (the original demo)
```bash
python main.py "Deep Research"
```

### On Code Review (the generalization demo)
```bash
python main.py "Code Review"
```

### On a brand-new domain
```bash
python main.py "Legal Analysis"
```

When you pass a domain that has no manual profile, the LLM introspector builds one on the fly — workflow, required tools, quality criteria, and a rubric.

### Flags
| Flag | Effect |
|---|---|
| `--force-rediscover` | Skip the cached domain profile and re-run discovery |
| `--results-dir PATH` | Write outputs somewhere other than `results/` |

### Adding a new domain manually

For the cleanest results (no LLM cost for discovery), drop two files:

1. **`benchmarks/{slug}.json`** — the eval set. Each task needs `id`, `query`, `reference_answer`. See [benchmarks/code_review.json](benchmarks/code_review.json) for the exact schema.
2. **`domain_profiles/{slug}.json`** — the manually-curated profile. Includes `workflow`, `required_tools`, `rubric`, `requires_web_research`, `requires_citations`. See [domain_profiles/code_review.json](domain_profiles/code_review.json) for an example.

The slug is the domain name lowercased with non-alphanumeric chars replaced by `_` (e.g., `"Code Review"` → `code_review`).

## Outputs

Every run writes to `results/`:

```
results/
├── scores.json              # per-iteration scores per criterion
├── versions/
│   ├── v0.json              # full agent state at iteration 0
│   ├── v1.json              # ...
│   └── ...
└── final_agent/             # the best version, exported as a standalone runnable
    ├── agent_config.json    # system_prompt + active_tools + tool_schemas
    ├── tools/               # source code of all generated tools
    └── run.py               # `python run.py < query.txt` runs the specialized agent
```

## Running the specialized agent standalone

After a `python main.py "Deep Research"` run, the trained agent is fully self-contained:

```bash
echo "What caused the 2008 financial crisis?" | python results/final_agent/run.py
```

No need to re-run the stem agent — the exported artifact only depends on `tools/base_tools.py` and the OpenAI key.

## Stopping criteria

The evolution loop ends when any of these triggers (configurable in [config.py](config.py)):

| Condition | Default | What it means |
|---|---|---|
| `composite >= SCORE_THRESHOLD` | `75.0` | Specialist is good enough — ship it |
| Composite **drops** vs. previous iteration | — | Regression detected → roll back to previous best |
| `improvement < MIN_IMPROVEMENT` over 2 iters | `0.5` | Converged |
| `iteration >= MAX_ITERATIONS` | `5` | Hard cap (cost guard) |

## Testing

```bash
pytest tests/ -v
```

67 tests — unit + integration. All pass.

## Project layout

```
.
├── main.py                          # CLI entry
├── stem_agent.py                    # the orchestrator (the evolution loop lives here)
├── config.py                        # all tunable constants
│
├── discovery/                       # Phase 1: figure out what the domain looks like
│   ├── __init__.py                  # DiscoveryEngine — caches and orchestrates
│   ├── llm_introspection.py         # LLM-based domain analysis → DomainProfile
│   └── web_research.py              # Web search for additional domain context
│
├── specialization/                  # Phase 2: rewrite prompt + generate tools
│   ├── __init__.py                  # SpecializationEngine
│   ├── prompt_rewriter.py           # Domain prompt rewriting
│   └── tool_generator.py            # LLM code-gen + safety validation + schema reg
│
├── evaluation/                      # Phase 3: ReAct loop + judge
│   ├── __init__.py                  # EvaluationEngine
│   ├── benchmark.py                 # ReAct task runner with prefetch/polish
│   └── judge.py                     # Rubric-driven scoring (LLM + deterministic)
│
├── safeguards/                      # Versioning + safety gates
│   ├── validator.py                 # AST-based code safety check for generated tools
│   └── versioning.py                # VersionedState save/load/rollback/best
│
├── tools/
│   ├── base_tools.py                # web_search (DDG → Wikipedia → Tavily backends), url_reader
│   └── generated/                   # Dynamically generated tools land here at runtime
│
├── benchmarks/
│   ├── deep_research.json           # 5 FRAMES-style factual research tasks
│   └── code_review.json             # 5 code-review tasks
│
├── domain_profiles/
│   └── code_review.json             # Pre-baked Code Review profile (skips LLM introspection)
│
├── tests/                           # 67 unit + integration tests
└── docs/superpowers/
    ├── specs/                       # Original design spec
    └── plans/                       # Original implementation plan
```

## Search backends

Configured via `SEARCH_BACKEND` in [config.py](config.py):

| Backend | Cost | Notes |
|---|---|---|
| `duckduckgo` (default) | Free, unlimited | Tries DDG; if rate-limited or 0 results, automatically falls back to Wikipedia |
| `wikipedia` | Free, unlimited, no key | Uses MediaWiki API directly. Bulletproof for encyclopedic queries |
| `tavily` | Paid (1000 free credits) | Highest-quality snippets but credits are limited; requires `TAVILY_API_KEY` |

## Key design decisions

- **Generated tools are quarantined.** Every LLM-generated tool passes through three safety gates before being registered: AST syntax check, dangerous-import / dangerous-call static analysis, and a dry-run with mock input. See [safeguards/validator.py](safeguards/validator.py).
- **Versioning everywhere.** Every iteration's full state (prompt + tools + scores) is serialized to `results/versions/v{N}.json`. Rollback restores the entire state.
- **Diagnostic patches per criterion.** When the score is below threshold, the next iteration's prompt gets a targeted patch addressing whichever criterion was lowest. Known criteria (citation, coverage, synthesis, accuracy) use hand-tuned templates; novel criteria (e.g., "security") get an LLM-generated patch.
- **Pipeline gating per domain.** The `DomainProfile` controls whether prefetch and citation polish run — research domains use them, code/math domains skip them.

## Architecture spec

Full design rationale: [docs/superpowers/specs/2026-05-05-stem-agent-design.md](docs/superpowers/specs/2026-05-05-stem-agent-design.md).
