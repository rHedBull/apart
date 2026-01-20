"""
Persona file loader with inheritance support.

Personas are reusable agent definitions that can be extended and customized.
They support inheritance via the 'extends' key, allowing base personas to be
specialized for different scenarios.
"""

from pathlib import Path
from typing import Any
import yaml


class PersonaLoader:
    """
    Loads and resolves persona files with inheritance support.

    Persona files are YAML documents that define agent characteristics:
    - id: Unique identifier
    - name: Display name
    - extends: Optional parent persona path (relative to personas dir)
    - description: Agent description
    - system_prompt: The agent's system prompt
    - goals: List of agent goals
    - default_variables: Default variable values

    When a persona extends another, all fields are merged with the child
    taking precedence over the parent.
    """

    def __init__(self, personas_dir: Path | str | None = None):
        """
        Initialize the PersonaLoader.

        Args:
            personas_dir: Directory containing persona files.
                         Defaults to 'personas/' relative to project root.
        """
        if personas_dir is None:
            # Default to project root / personas
            self.personas_dir = Path(__file__).parent.parent.parent / "personas"
        else:
            self.personas_dir = Path(personas_dir)

        self._cache: dict[str, dict[str, Any]] = {}

    def load(self, persona_path: str) -> dict[str, Any]:
        """
        Load a persona file and resolve inheritance.

        Args:
            persona_path: Path to persona relative to personas dir (e.g., 'geopolitical/usa_state_dept')
                         Can include or omit .yaml extension.

        Returns:
            Fully resolved persona dictionary with all inherited fields merged.

        Raises:
            FileNotFoundError: If persona file doesn't exist
            ValueError: If persona has circular inheritance or invalid format
        """
        # Normalize path
        if not persona_path.endswith('.yaml'):
            persona_path = f"{persona_path}.yaml"

        # Check cache
        if persona_path in self._cache:
            return self._cache[persona_path].copy()

        # Load and resolve
        resolved = self._load_and_resolve(persona_path, set())
        self._cache[persona_path] = resolved
        return resolved.copy()

    def _load_and_resolve(
        self,
        persona_path: str,
        visited: set[str]
    ) -> dict[str, Any]:
        """
        Recursively load and resolve persona inheritance.

        Args:
            persona_path: Path to persona file
            visited: Set of already visited paths (for cycle detection)

        Returns:
            Resolved persona dictionary
        """
        # Cycle detection
        if persona_path in visited:
            cycle = " -> ".join(visited) + f" -> {persona_path}"
            raise ValueError(f"Circular inheritance detected: {cycle}")

        visited = visited | {persona_path}

        # Load the persona file
        full_path = self.personas_dir / persona_path
        if not full_path.exists():
            raise FileNotFoundError(
                f"Persona file not found: {full_path}"
            )

        with open(full_path) as f:
            persona = yaml.safe_load(f)

        if persona is None:
            raise ValueError(f"Empty persona file: {full_path}")

        if not isinstance(persona, dict):
            raise ValueError(
                f"Persona file must be a YAML dictionary: {full_path}"
            )

        # Validate required fields
        self._validate_persona(persona, full_path)

        # Check for inheritance
        if 'extends' in persona:
            parent_path = persona['extends']
            if not parent_path.endswith('.yaml'):
                parent_path = f"{parent_path}.yaml"

            # Load parent recursively
            parent = self._load_and_resolve(parent_path, visited)

            # Merge parent and child (child takes precedence)
            resolved = self._merge_personas(parent, persona)
        else:
            resolved = persona.copy()

        # Remove 'extends' from final output
        resolved.pop('extends', None)

        return resolved

    def _validate_persona(self, persona: dict[str, Any], path: Path) -> None:
        """
        Validate persona has required fields.

        Args:
            persona: Persona dictionary
            path: Path to file (for error messages)

        Raises:
            ValueError: If required fields are missing
        """
        required_fields = ['id', 'name']

        for field in required_fields:
            if field not in persona:
                raise ValueError(
                    f"Persona '{path}' missing required field: {field}"
                )

    def _merge_personas(
        self,
        parent: dict[str, Any],
        child: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Deep merge two persona dictionaries.

        Child values take precedence. Lists are concatenated (child first).
        Nested dicts are recursively merged.

        Args:
            parent: Parent persona dictionary
            child: Child persona dictionary

        Returns:
            Merged persona dictionary
        """
        result = parent.copy()

        for key, child_value in child.items():
            if key == 'extends':
                continue  # Skip extends key

            if key not in result:
                result[key] = child_value
            elif isinstance(child_value, dict) and isinstance(result[key], dict):
                # Deep merge dictionaries
                result[key] = self._merge_dicts(result[key], child_value)
            elif isinstance(child_value, list) and isinstance(result[key], list):
                # Concatenate lists (child first, for goals/behaviors)
                result[key] = child_value + result[key]
            else:
                # Child value takes precedence
                result[key] = child_value

        return result

    def _merge_dicts(
        self,
        parent: dict[str, Any],
        child: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Deep merge two dictionaries.

        Args:
            parent: Parent dictionary
            child: Child dictionary (takes precedence)

        Returns:
            Merged dictionary
        """
        result = parent.copy()

        for key, child_value in child.items():
            if key not in result:
                result[key] = child_value
            elif isinstance(child_value, dict) and isinstance(result[key], dict):
                result[key] = self._merge_dicts(result[key], child_value)
            else:
                result[key] = child_value

        return result

    def list_personas(self, category: str | None = None) -> list[str]:
        """
        List available personas.

        Args:
            category: Optional category to filter (e.g., 'base', 'geopolitical')

        Returns:
            List of persona paths relative to personas directory
        """
        if category:
            search_dir = self.personas_dir / category
        else:
            search_dir = self.personas_dir

        if not search_dir.exists():
            return []

        personas = []
        for yaml_file in search_dir.rglob("*.yaml"):
            rel_path = yaml_file.relative_to(self.personas_dir)
            personas.append(str(rel_path))

        return sorted(personas)

    def clear_cache(self) -> None:
        """Clear the persona cache."""
        self._cache.clear()


def resolve_persona_in_agent(
    agent_config: dict[str, Any],
    persona_loader: PersonaLoader
) -> dict[str, Any]:
    """
    Resolve a persona reference in an agent configuration.

    If the agent config has a 'persona' key, load the persona and merge
    its fields into the agent config. Agent-level values take precedence
    over persona values.

    Args:
        agent_config: Agent configuration dictionary
        persona_loader: PersonaLoader instance

    Returns:
        Agent configuration with persona fields merged in

    Example:
        Input agent_config:
            persona: geopolitical/usa_state_dept
            variables:
                diplomatic_leverage: 90

        Output:
            name: "US State Department"
            system_prompt: "..."
            goals: [...]
            variables:
                diplomatic_leverage: 90  # Agent override
                military_readiness: 70   # From persona default_variables
    """
    if 'persona' not in agent_config:
        return agent_config

    persona_path = agent_config['persona']
    persona = persona_loader.load(persona_path)

    # Start with persona as base
    result = {}

    # Map persona fields to agent config fields
    if 'name' in persona:
        result['name'] = persona['name']
    if 'description' in persona:
        result['description'] = persona['description']
    if 'system_prompt' in persona:
        result['system_prompt'] = persona['system_prompt']
    if 'llm' in persona:
        result['llm'] = persona['llm']

    # Handle default_variables -> variables
    if 'default_variables' in persona:
        result['variables'] = persona['default_variables'].copy()

    # Merge agent config on top (agent takes precedence)
    for key, value in agent_config.items():
        if key == 'persona':
            continue  # Skip persona key
        if key == 'variables' and 'variables' in result:
            # Merge variables (agent overrides persona defaults)
            result['variables'].update(value)
        else:
            result[key] = value

    return result
