from specialization.prompt_rewriter import PromptRewriter
from specialization.tool_generator import ToolGenerator
from discovery.llm_introspection import DomainProfile
from tools.base_tools import get_openai_schema

PREBUILT_TOOLS = {"web_search", "url_reader"}


class SpecializationEngine:
    def __init__(self):
        self._rewriter = PromptRewriter()
        self._generator = ToolGenerator()

    def specialize(self, profile: DomainProfile, existing_source_files: dict) -> dict:
        new_prompt = self._rewriter.rewrite(profile)
        tool_schemas = []
        tool_source_files = dict(existing_source_files)
        active_tools = list(PREBUILT_TOOLS)

        for tool_name in PREBUILT_TOOLS:
            tool_schemas.append(get_openai_schema(tool_name))

        for tool_name in profile.required_tools:
            if tool_name in PREBUILT_TOOLS:
                continue
            if tool_name in existing_source_files:
                spec = profile.tool_specs.get(tool_name)
                if spec:
                    schema = self._generator.spec_to_schema(spec)
                    tool_schemas.append(schema)
                    active_tools.append(tool_name)
                continue
            spec = profile.tool_specs.get(tool_name)
            if not spec:
                continue
            result = self._generator.generate(tool_name, spec)
            if result:
                self._generator.load_and_register(tool_name, result["source_code"], result["schema"])
                tool_schemas.append(result["schema"])
                tool_source_files[tool_name] = result["source_code"]
                active_tools.append(tool_name)

        return {
            "system_prompt": new_prompt,
            "tool_schemas": tool_schemas,
            "tool_source_files": tool_source_files,
            "active_tools": active_tools,
        }
