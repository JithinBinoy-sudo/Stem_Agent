import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

SCORE_THRESHOLD = 75.0
MIN_IMPROVEMENT = 3.0
MAX_ITERATIONS = 5
MAX_TOOL_CALLS_PER_TASK = 12
TOOL_GEN_MAX_RETRIES = 2
BENCHMARK_BASE_TOOLS_ONLY = True

DISCOVERY_CACHE_DIR = "cache"
RESULTS_DIR = "results"
VERSIONS_DIR = "results/versions"
FINAL_AGENT_DIR = "results/final_agent"
GENERATED_TOOLS_DIR = "tools/generated"

MODEL_STRONG = "gpt-4o"
MODEL_WEAK = "gpt-4o-mini"
BASE_SYSTEM_PROMPT = "You are a helpful AI assistant."

DANGEROUS_IMPORTS = {"os.system", "subprocess", "eval", "exec", "shutil", "__import__"}
