"""
Network file loader for supply chain modules.

Loads supply chain network YAML files and converts them to
structured data for simulation use.
"""

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class NetworkLoadError(Exception):
    """Error loading a network file."""
    pass


@dataclass
class SupplyChainActor:
    """An actor (nation/bloc) in the supply chain."""
    id: str
    name: str
    type: str  # consumer, producer, consumer_producer
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SupplyDependency:
    """A dependency relationship between actors for a commodity."""
    actor_id: str
    commodity: str
    sources: Dict[str, float]  # source_id -> percentage

    def validate(self) -> List[str]:
        """Validate dependency data. Returns list of errors."""
        errors = []
        total = sum(self.sources.values())
        if abs(total - 100) > 0.01:
            errors.append(
                f"{self.actor_id}'s {self.commodity} dependencies sum to {total}%, not 100%"
            )
        return errors


@dataclass
class LeverageRelationship:
    """A leverage relationship where one actor has power over others."""
    source: str
    targets: List[str]
    commodity: str
    leverage_type: str
    notes: str = ""


@dataclass
class SupplyChainNetwork:
    """
    Complete supply chain network representation.

    Contains actors, dependencies, export positions, stockpiles,
    and leverage relationships.
    """
    name: str
    description: str
    granularity: str  # strategic, operational, tactical
    primary_commodity: Optional[str] = None

    actors: Dict[str, SupplyChainActor] = field(default_factory=dict)
    dependencies: Dict[str, Dict[str, SupplyDependency]] = field(default_factory=dict)
    # commodity -> actor_id -> SupplyDependency
    export_positions: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # commodity -> actor_id -> market_share %
    stockpiles: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # actor_id -> commodity -> stockpile_level
    leverage_relationships: List[LeverageRelationship] = field(default_factory=list)

    # Raw metadata for extensions
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_actor(self, actor_id: str) -> Optional[SupplyChainActor]:
        """Get actor by ID."""
        return self.actors.get(actor_id)

    def get_dependency(self, actor_id: str, commodity: str) -> Optional[SupplyDependency]:
        """Get an actor's dependency for a commodity."""
        commodity_deps = self.dependencies.get(commodity, {})
        return commodity_deps.get(actor_id)

    def get_export_position(self, actor_id: str, commodity: str) -> float:
        """Get an actor's export market share for a commodity."""
        return self.export_positions.get(commodity, {}).get(actor_id, 0.0)

    def get_stockpile(self, actor_id: str, commodity: str) -> float:
        """Get an actor's stockpile level for a commodity."""
        return self.stockpiles.get(actor_id, {}).get(commodity, 0.0)

    def get_leverage_over(self, target_id: str) -> List[LeverageRelationship]:
        """Get all leverage relationships where target is subject to leverage."""
        return [
            rel for rel in self.leverage_relationships
            if target_id in rel.targets
        ]

    def get_leverage_by(self, source_id: str) -> List[LeverageRelationship]:
        """Get all leverage relationships where source has leverage."""
        return [
            rel for rel in self.leverage_relationships
            if rel.source == source_id
        ]

    def get_actors_dependent_on(self, supplier_id: str, commodity: str) -> List[Tuple[str, float]]:
        """
        Get all actors dependent on a supplier for a commodity.

        Returns list of (actor_id, dependency_percentage) tuples.
        """
        result = []
        commodity_deps = self.dependencies.get(commodity, {})
        for actor_id, dep in commodity_deps.items():
            if supplier_id in dep.sources:
                result.append((actor_id, dep.sources[supplier_id]))
        return sorted(result, key=lambda x: -x[1])  # Sort by dependency %

    def validate(self) -> List[str]:
        """Validate the network. Returns list of error messages."""
        errors = []

        # Validate all dependencies sum to 100%
        for commodity, actor_deps in self.dependencies.items():
            for actor_id, dep in actor_deps.items():
                errors.extend(dep.validate())

        # Validate actor references in dependencies
        for commodity, actor_deps in self.dependencies.items():
            for actor_id, dep in actor_deps.items():
                if actor_id not in self.actors and actor_id != "other":
                    errors.append(f"Unknown actor '{actor_id}' in {commodity} dependencies")
                for source_id in dep.sources:
                    if source_id not in self.actors and source_id not in ("domestic", "other"):
                        errors.append(f"Unknown source '{source_id}' in {actor_id}'s {commodity} dependency")

        return errors


def load_network_file(
    network_path: str | Path,
    base_dir: Path | None = None
) -> SupplyChainNetwork:
    """
    Load a supply chain network file.

    Args:
        network_path: Path to the network YAML file
        base_dir: Base directory for relative paths (defaults to src/)

    Returns:
        SupplyChainNetwork object

    Raises:
        NetworkLoadError: If file not found or invalid
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent  # src/

    full_path = base_dir / network_path

    if not full_path.exists():
        raise NetworkLoadError(f"Network file not found: {full_path}")

    try:
        with open(full_path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise NetworkLoadError(f"Invalid YAML in network file: {e}")

    if not raw:
        raise NetworkLoadError(f"Empty network file: {full_path}")

    return parse_network_data(raw, str(full_path))


def parse_network_data(data: Dict[str, Any], source: str = "unknown") -> SupplyChainNetwork:
    """
    Parse network data into a SupplyChainNetwork.

    Args:
        data: Parsed YAML data
        source: Source file path for error messages

    Returns:
        SupplyChainNetwork object
    """
    network_meta = data.get("network", {})

    network = SupplyChainNetwork(
        name=network_meta.get("name", "Unnamed Network"),
        description=network_meta.get("description", ""),
        granularity=network_meta.get("granularity", "strategic"),
        primary_commodity=network_meta.get("primary_commodity"),
    )

    # Parse actors
    for actor_data in data.get("actors", []):
        if not isinstance(actor_data, dict) or "id" not in actor_data:
            raise NetworkLoadError(f"Invalid actor in {source}: {actor_data}")

        actor = SupplyChainActor(
            id=actor_data["id"],
            name=actor_data.get("name", actor_data["id"]),
            type=actor_data.get("type", "consumer_producer"),
            properties=actor_data.get("properties", {}),
        )
        network.actors[actor.id] = actor

    # Parse dependencies
    deps_data = data.get("dependencies", {})
    for commodity, actor_deps in deps_data.items():
        network.dependencies[commodity] = {}
        for actor_id, sources in actor_deps.items():
            network.dependencies[commodity][actor_id] = SupplyDependency(
                actor_id=actor_id,
                commodity=commodity,
                sources=sources,
            )

    # Parse export positions
    network.export_positions = data.get("export_positions", {})

    # Parse stockpiles
    network.stockpiles = data.get("stockpiles", {})

    # Parse leverage relationships
    for rel_data in data.get("leverage_relationships", []):
        network.leverage_relationships.append(LeverageRelationship(
            source=rel_data["source"],
            targets=rel_data.get("targets", []),
            commodity=rel_data.get("commodity", ""),
            leverage_type=rel_data.get("leverage_type", ""),
            notes=rel_data.get("notes", ""),
        ))

    # Store remaining data as metadata
    known_keys = {"network", "actors", "dependencies", "export_positions",
                  "stockpiles", "leverage_relationships"}
    for key, value in data.items():
        if key not in known_keys:
            network.metadata[key] = value

    return network
