import json
from dataclasses import dataclass
from typing import List, Dict, Any
from openai import OpenAI
import config

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
  "quality_criteria": ["criterion1", "criterion2", ...]
}}

Rules:
- workflow: ordered list of steps an expert takes to solve tasks in this domain
- required_tools: include "web_search" and "url_reader" always, plus 3-4 domain-specific tools
- tool_specs: provide a spec for every tool in required_tools EXCEPT web_search and url_reader (those are pre-built)
- quality_criteria: how to judge output quality in this domain
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
        tool_specs = {
            name: ToolSpec(
                name=spec["name"],
                description=spec["description"],
                parameters=spec.get("parameters", {}),
                returns=spec.get("returns", {})
            )
            for name, spec in data.get("tool_specs", {}).items()
        }
        return DomainProfile(
            domain=data["domain"],
            workflow=data["workflow"],
            required_tools=data["required_tools"],
            tool_specs=tool_specs,
            quality_criteria=data["quality_criteria"],
        )
