import json
import os
from dataclasses import asdict
from discovery.llm_introspection import LLMIntrospector, DomainProfile, ToolSpec, DEFAULT_RUBRIC
from discovery.web_research import WebResearcher

class DiscoveryEngine:
    def __init__(self, cache_dir: str = "cache"):
        self._introspector = LLMIntrospector()
        self._researcher = WebResearcher()
        self._cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, domain: str) -> str:
        safe_name = domain.lower().replace(" ", "_")
        return os.path.join(self._cache_dir, f"{safe_name}.json")

    def _load_cache(self, domain: str) -> DomainProfile | None:
        path = self._cache_path(domain)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        tool_specs = {
            name: ToolSpec(**spec)
            for name, spec in data.get("tool_specs", {}).items()
        }
        return DomainProfile(
            domain=data["domain"],
            workflow=data["workflow"],
            required_tools=data["required_tools"],
            tool_specs=tool_specs,
            quality_criteria=data["quality_criteria"],
            requires_web_research=bool(data.get("requires_web_research", True)),
            requires_citations=bool(data.get("requires_citations", True)),
            rubric=data.get("rubric") or DEFAULT_RUBRIC,
        )

    def _save_cache(self, profile: DomainProfile):
        path = self._cache_path(profile.domain)
        data = {
            "domain": profile.domain,
            "workflow": profile.workflow,
            "required_tools": profile.required_tools,
            "tool_specs": {
                name: asdict(spec) for name, spec in profile.tool_specs.items()
            },
            "quality_criteria": profile.quality_criteria,
            "requires_web_research": profile.requires_web_research,
            "requires_citations": profile.requires_citations,
            "rubric": profile.rubric,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def run(self, domain: str, force: bool = False) -> DomainProfile:
        if not force:
            cached = self._load_cache(domain)
            if cached:
                return cached

        profile = self._introspector.introspect(domain)
        self._researcher.research(domain)  # informational; results merged into future iterations

        self._save_cache(profile)
        return profile
