# Stem Agent

A minimal AI agent that self-specializes into a domain-specific agent through an iterative evolution loop.

## Setup

1. Clone the repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy `.env.example` to `.env` and fill in your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your OPENAI_API_KEY and TAVILY_API_KEY
   ```

## Run

```bash
python main.py "Deep Research"
```

Flags:
- `--force-rediscover` — bypass discovery cache and re-run domain analysis
- `--results-dir PATH` — save results to a custom directory (default: `results/`)

## Output

After the run completes:
- `results/scores.json` — per-iteration composite scores
- `results/versions/` — versioned state snapshots
- `results/final_agent/` — standalone specialized agent

## Run the Specialized Agent

```bash
echo "What caused the 2008 financial crisis?" | python results/final_agent/run.py
```

## Run Tests

```bash
pytest tests/ -v
```

## Architecture

See `docs/superpowers/specs/2026-05-05-stem-agent-design.md` for the full design spec.
