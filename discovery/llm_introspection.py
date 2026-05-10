import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any
from openai import OpenAI
import config


def sanitize_tool_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name.strip()).strip("_")
    return cleaned.lower() or "unnamed_tool"

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, Any]

@dataclass
class DomainProfile:
    domain: str
    workflow: List[str]
    required_tools: List[str]
    tool_specs: Dict[str, ToolSpec]
    quality_criteria: List[str]
    requires_web_research: bool = True
    requires_citations: bool = True
    rubric: List[Dict[str, Any]] = None

INTROSPECTION_PROMPT = """You are an expert AI systems architect.

Given the problem domain "{domain}", return a JSON object describing how an expert agent would approach this domain.

Return ONLY valid JSON with this exact structure:
{{
  "domain": "{domain}",
  "workflow": ["step1", "step2", ...],
  "required_tools": ["tool_name1", "tool_name2", ...],
  "tool_specs": {{
    "tool_name": {{
      "name": "tool_name",
      "description": "what this tool does",
      "parameters": {{
        "param_name": {{"type": "string", "description": "what this param is"}}
      }},
      "returns": {{"type": "object", "fields": ["field1", "field2"]}}
    }}
  }},
  "quality_criteria": ["criterion1", "criterion2", ...],
  "requires_web_research": true,
  "requires_citations": true,
  "rubric": [
    {{"name": "accuracy", "weight": 0.35, "method": "llm"}},
    {{"name": "synthesis", "weight": 0.25, "method": "llm"}},
    {{"name": "coverage", "weight": 0.25, "method": "deterministic", "scorer": "unique_domains"}},
    {{"name": "citation", "weight": 0.15, "method": "deterministic", "scorer": "sentence_citations"}}
  ]
}}

Rules:
- workflow: ordered list of steps an expert takes to solve tasks in this domain
- required_tools: include "web_search" and "url_reader" always, plus 3-4 domain-specific tools
- tool_specs: provide a spec for every tool in required_tools EXCEPT web_search and url_reader (those are pre-built)
- quality_criteria: how to judge output quality in this domain
- requires_web_research: true if the agent should pre-fetch web sources before answering (research, news, fact-checking, summarization). False for code generation, math, creative writing, planning.
- requires_citations: true if every factual claim must end with a [URL] (research, journalism). False if URLs would be inappropriate (code, math).
- rubric: 3-5 criteria. weights MUST sum to 1.0. method=\"llm\" means an LLM judge scores 0-10. method=\"deterministic\" means a built-in formula computes the score; valid scorers: \"unique_domains\" (counts distinct domain hostnames), \"sentence_citations\" (fraction of sentences ending in [URL]).
"""

class LLMIntrospector:
    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)

    def introspect(self, domain: str) -> DomainProfile:
        prompt = INTROSPECTION_PROMPT.format(domain=domain)
        response = self._client.chat.completions.create(
            model=config.MODEL_WEAK,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        data = json.loads(response.choices[0].message.content)
        tool_specs = {}
        for raw_name, spec in data.get("tool_specs", {}).items():
            canonical = sanitize_tool_name(spec.get("name", raw_name))
            tool_specs[canonical] = ToolSpec(
                name=canonical,
                description=spec["description"],
                parameters=spec.get("parameters", {}),
                returns=spec.get("returns", {})
            )
        required_tools = [sanitize_tool_name(t) for t in data["required_tools"]]
        rubric = data.get("rubric") or DEFAULT_RUBRIC
        return DomainProfile(
            domain=data["domain"],
            workflow=data["workflow"],
            required_tools=required_tools,
            tool_specs=tool_specs,
            quality_criteria=data["quality_criteria"],
            requires_web_research=bool(data.get("requires_web_research", True)),
            requires_citations=bool(data.get("requires_citations", True)),
            rubric=rubric,
        )


DEFAULT_RUBRIC = [
    {"name": "accuracy", "weight": 0.35, "method": "llm"},
    {"name": "synthesis", "weight": 0.25, "method": "llm"},
    {"name": "coverage", "weight": 0.25, "method": "deterministic", "scorer": "unique_domains"},
    {"name": "citation", "weight": 0.15, "method": "deterministic", "scorer": "sentence_citations"},
]
