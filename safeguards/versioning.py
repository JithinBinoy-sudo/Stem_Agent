import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

@dataclass
class VersionedState:
    version: int
    system_prompt: str
    active_tools: List[str]
    tool_schemas: List[Dict]
    tool_source_files: Dict[str, str]
    composite_score: float
    timestamp: str

class VersionManager:
    def __init__(self, versions_dir: str = "results/versions"):
        self.versions_dir = versions_dir
        os.makedirs(versions_dir, exist_ok=True)

    def _path(self, version: int) -> str:
        return os.path.join(self.versions_dir, f"v{version}.json")

    def save(self, state: VersionedState):
        with open(self._path(state.version), "w") as f:
            json.dump(asdict(state), f, indent=2)

    def load(self, version: int) -> VersionedState:
        with open(self._path(version)) as f:
            data = json.load(f)
        return VersionedState(**data)

    def list_versions(self) -> List[int]:
        files = [f for f in os.listdir(self.versions_dir) if f.startswith("v") and f.endswith(".json")]
        versions = sorted([int(f[1:-5]) for f in files])
        return versions

    def get_best(self) -> Optional[VersionedState]:
        versions = self.list_versions()
        if not versions:
            return None
        states = [self.load(v) for v in versions]
        return max(states, key=lambda s: s.composite_score)

    def rollback(self, from_version: int) -> VersionedState:
        versions = self.list_versions()
        previous = [v for v in versions if v < from_version]
        if not previous:
            raise ValueError(f"No previous version to roll back to from v{from_version}")
        return self.load(max(previous))
