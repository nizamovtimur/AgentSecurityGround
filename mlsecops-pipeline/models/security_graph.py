"""Normalized security graph structures for static analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SecurityNode:
    """Normalized node representation for security analysis."""

    node_id: str
    display_name: str
    node_type: str
    role: str
    template_fields: dict[str, Any] = field(default_factory=dict)
    risk_flags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SecurityEdge:
    """Normalized edge representation with semantic handles."""

    source: str
    target: str
    source_data_type: str | None = None
    source_handle_name: str | None = None
    target_field_name: str | None = None
    target_input_types: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SecurityGraph:
    """Top-level normalized graph for downstream services."""

    nodes: list[SecurityNode]
    edges: list[SecurityEdge]
    entrypoints: list[str] = field(default_factory=list)
    controls: list[str] = field(default_factory=list)
    assets: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "entrypoints": self.entrypoints,
            "controls": self.controls,
            "assets": self.assets,
        }

