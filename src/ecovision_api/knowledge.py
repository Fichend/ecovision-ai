from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Guidance:
    region_name: str
    material: str
    statements: list[str]
    sources: list[dict[str, str]]
    knowledge_version: str


class KnowledgeBase:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = json.loads(path.read_text(encoding="utf-8"))

    @property
    def version(self) -> str:
        return str(self.data["version"])

    def retrieve(self, region: str, material: str, city: str | None = None) -> Guidance:
        if region not in self.data["regions"]:
            raise ValueError(f"Unsupported region: {region}")
        region_data = self.data["regions"][region]
        material_key = material if material in region_data["materials"] else "trash"
        statements = list(region_data["general_guidance"])
        statements.extend(region_data["materials"][material_key])
        if region == "mainland_china" and city:
            statements.insert(0, f"Location supplied by the user: {city}.")
        return Guidance(
            region_name=region_data["display_name"],
            material=material_key,
            statements=statements,
            sources=list(region_data["sources"]),
            knowledge_version=self.version,
        )
