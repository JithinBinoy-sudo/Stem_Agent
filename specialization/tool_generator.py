import os
import importlib.util
from openai import OpenAI
from safeguards.validator import Validator
from tools import register_tool
import config
from discovery.llm_introspection import ToolSpec

PREBUILT_TOOLS = {"web_search", "url_reader"}

_VALID_JSON_TYPES = {"string", "number", "integer", "boolean", "array", "object", "null"}


def _coerce_json_schema(meta: dict) -> dict:
    raw_type = meta.get("type", "string")
    t = raw_type if raw_type in _VALID_JSON_TYPES else "string"
    out = {"type": t, "description": meta.get("description", "")}
    if t == "array":
        items = meta.get("items")
        if not isinstance(items, dict) or "type" not in items:
            items = {"type": "string"}
        out["items"] = items
    elif t == "object":
        out["properties"] = meta.get("properties", {})
    return out

CODE_GEN_PROMPT = """Write a Python function implementing the following tool.

Tool name: {name}
Description: {description}
Parameters: {parameters}
Returns: {returns}

Requirements:
- Function name must be EXACTLY `{name}` — lowercase, no capitalization changes
- Function signature must match the parameter names exactly
- Return a dict with the fields specified in Returns
- Use only stdlib + requests + beautifulsoup4 — no other third-party imports
- Keep it concise (under 30 lines)
- No imports of os, subprocess, eval, exec
- Do not call input(), print(), or any interactive I/O

Return ONLY the Python function code, no markdown, no explanation.
"""


class ToolGenerator:
    def __init__(self):
        self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        self._validator = Validator()
        os.makedirs(config.GENERATED_TOOLS_DIR, exist_ok=True)

    def spec_to_schema(self, spec: ToolSpec) -> dict:
        from discovery.llm_introspection import sanitize_tool_name
        properties = {
            pname: _coerce_json_schema(meta)
            for pname, meta in spec.parameters.items()
        }
        return {
            "type": "function",
            "function": {
                "name": sanitize_tool_name(spec.name),
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
        func = getattr(module, name, None)
        if func is None or not callable(func):
            for attr in dir(module):
                if attr.startswith("_"):
                    continue
                if attr.lower() == name.lower():
                    func = getattr(module, attr)
                    break
            else:
                for attr in dir(module):
                    if attr.startswith("_"):
                        continue
                    candidate = getattr(module, attr)
                    if callable(candidate) and getattr(candidate, "__module__", None) == module.__name__:
                        func = candidate
                        break
        if func is None or not callable(func):
            raise AttributeError(f"no callable found in generated module for tool '{name}'")
        register_tool(name, func)
