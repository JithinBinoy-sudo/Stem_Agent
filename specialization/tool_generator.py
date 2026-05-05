import os
import importlib.util
from openai import OpenAI
from safeguards.validator import Validator
from tools import register_tool
import config
from discovery.llm_introspection import ToolSpec

PREBUILT_TOOLS = {"web_search", "url_reader"}

CODE_GEN_PROMPT = """Write a Python function implementing the following tool.

Tool name: {name}
Description: {description}
Parameters: {parameters}
Returns: {returns}

Requirements:
- Function signature must match the parameter names exactly
- Return a dict with the fields specified in Returns
- Use only stdlib + requests + beautifulsoup4 — no other third-party imports
- Keep it concise (under 30 lines)
- No imports of os, subprocess, eval, exec

Return ONLY the Python function code, no markdown, no explanation.
"""


class ToolGenerator:
    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._validator = Validator()
        os.makedirs(config.GENERATED_TOOLS_DIR, exist_ok=True)

    def spec_to_schema(self, spec: ToolSpec) -> dict:
        properties = {
            name: {"type": meta["type"], "description": meta.get("description", "")}
            for name, meta in spec.parameters.items()
        }
        return {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(spec.parameters.keys()),
                }
            }
        }

    def generate(self, name: str, spec: ToolSpec | None):
        if name in PREBUILT_TOOLS:
            return None

        prompt = CODE_GEN_PROMPT.format(
            name=spec.name,
            description=spec.description,
            parameters=spec.parameters,
            returns=spec.returns,
        )
        mock_input = {k: "test" for k in spec.parameters.keys()}

        for attempt in range(config.TOOL_GEN_MAX_RETRIES + 1):
            response = self._client.chat.completions.create(
                model=config.MODEL_WEAK,
                messages=[{"role": "user", "content": prompt}],
            )
            code = response.choices[0].message.content.strip()
            if code.startswith("```"):
                code = "\n".join(code.split("\n")[1:-1])

            result = self._validator.validate(code, mock_input=mock_input)
            if result.passed:
                schema = self.spec_to_schema(spec)
                return {"source_code": code, "schema": schema, "name": name}

            prompt += f"\n\nPrevious attempt failed at stage '{result.stage}': {result.error}\nPlease fix and try again."

        return None

    def load_and_register(self, name: str, source_code: str, schema: dict):
        path = os.path.join(config.GENERATED_TOOLS_DIR, f"{name}.py")
        with open(path, "w") as f:
            f.write(source_code)
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        func = getattr(module, name)
        register_tool(name, func)
