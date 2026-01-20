"""Unit tests for the PersonaLoader."""

import sys
from pathlib import Path
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
import yaml
from utils.persona_loader import PersonaLoader, resolve_persona_in_agent


class TestPersonaLoader:
    """Tests for PersonaLoader class."""

    @pytest.fixture
    def temp_personas_dir(self):
        """Create a temporary personas directory with test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            personas_dir = Path(tmpdir)

            # Create base directory
            (personas_dir / "base").mkdir()
            (personas_dir / "specific").mkdir()

            # Create base persona
            base_persona = {
                "id": "base_agent",
                "name": "Base Agent",
                "description": "A base agent",
                "system_prompt": "You are a base agent.",
                "goals": ["Goal 1", "Goal 2"],
                "default_variables": {
                    "skill_level": 50,
                    "cooperation": 60
                }
            }
            with open(personas_dir / "base" / "agent.yaml", "w") as f:
                yaml.dump(base_persona, f)

            # Create child persona that extends base
            child_persona = {
                "id": "child_agent",
                "name": "Child Agent",
                "extends": "base/agent",
                "description": "A specialized agent",
                "goals": ["Goal 3"],  # Should be prepended
                "default_variables": {
                    "skill_level": 80  # Override
                }
            }
            with open(personas_dir / "specific" / "child.yaml", "w") as f:
                yaml.dump(child_persona, f)

            # Create grandchild persona (multi-level inheritance)
            grandchild_persona = {
                "id": "grandchild_agent",
                "name": "Grandchild Agent",
                "extends": "specific/child",
                "system_prompt": "You are a grandchild agent."
            }
            with open(personas_dir / "specific" / "grandchild.yaml", "w") as f:
                yaml.dump(grandchild_persona, f)

            # Create circular reference personas
            circular_a = {
                "id": "circular_a",
                "name": "Circular A",
                "extends": "base/circular_b"
            }
            circular_b = {
                "id": "circular_b",
                "name": "Circular B",
                "extends": "base/circular_a"
            }
            with open(personas_dir / "base" / "circular_a.yaml", "w") as f:
                yaml.dump(circular_a, f)
            with open(personas_dir / "base" / "circular_b.yaml", "w") as f:
                yaml.dump(circular_b, f)

            # Create standalone persona (no inheritance)
            standalone = {
                "id": "standalone",
                "name": "Standalone Agent",
                "system_prompt": "I am standalone.",
                "default_variables": {"power": 100}
            }
            with open(personas_dir / "standalone.yaml", "w") as f:
                yaml.dump(standalone, f)

            yield personas_dir

    def test_load_standalone_persona(self, temp_personas_dir):
        """Test loading a persona without inheritance."""
        loader = PersonaLoader(temp_personas_dir)
        persona = loader.load("standalone")

        assert persona["id"] == "standalone"
        assert persona["name"] == "Standalone Agent"
        assert persona["system_prompt"] == "I am standalone."
        assert persona["default_variables"]["power"] == 100

    def test_load_persona_with_yaml_extension(self, temp_personas_dir):
        """Test that .yaml extension is optional."""
        loader = PersonaLoader(temp_personas_dir)

        persona1 = loader.load("standalone")
        persona2 = loader.load("standalone.yaml")

        assert persona1 == persona2

    def test_load_persona_with_inheritance(self, temp_personas_dir):
        """Test loading a persona that extends another."""
        loader = PersonaLoader(temp_personas_dir)
        persona = loader.load("specific/child")

        # Child values should take precedence
        assert persona["id"] == "child_agent"
        assert persona["name"] == "Child Agent"
        assert persona["description"] == "A specialized agent"

        # System prompt inherited from parent
        assert persona["system_prompt"] == "You are a base agent."

        # Goals concatenated (child first)
        assert persona["goals"] == ["Goal 3", "Goal 1", "Goal 2"]

        # Variables merged (child override)
        assert persona["default_variables"]["skill_level"] == 80
        assert persona["default_variables"]["cooperation"] == 60

        # extends key should be removed
        assert "extends" not in persona

    def test_load_persona_with_multi_level_inheritance(self, temp_personas_dir):
        """Test loading a persona with grandparent inheritance."""
        loader = PersonaLoader(temp_personas_dir)
        persona = loader.load("specific/grandchild")

        # Grandchild values
        assert persona["id"] == "grandchild_agent"
        assert persona["name"] == "Grandchild Agent"
        assert persona["system_prompt"] == "You are a grandchild agent."

        # Inherited from child
        assert persona["description"] == "A specialized agent"

        # Inherited from base (through child)
        assert persona["default_variables"]["cooperation"] == 60

        # Child's override should still apply
        assert persona["default_variables"]["skill_level"] == 80

    def test_circular_inheritance_detection(self, temp_personas_dir):
        """Test that circular inheritance raises error."""
        loader = PersonaLoader(temp_personas_dir)

        with pytest.raises(ValueError, match="Circular inheritance detected"):
            loader.load("base/circular_a")

    def test_missing_persona_file(self, temp_personas_dir):
        """Test that missing persona file raises error."""
        loader = PersonaLoader(temp_personas_dir)

        with pytest.raises(FileNotFoundError):
            loader.load("nonexistent/persona")

    def test_persona_missing_required_fields(self, temp_personas_dir):
        """Test that persona missing required fields raises error."""
        # Create invalid persona
        invalid = {"description": "Missing id and name"}
        with open(temp_personas_dir / "invalid.yaml", "w") as f:
            yaml.dump(invalid, f)

        loader = PersonaLoader(temp_personas_dir)

        with pytest.raises(ValueError, match="missing required field"):
            loader.load("invalid")

    def test_caching(self, temp_personas_dir):
        """Test that personas are cached."""
        loader = PersonaLoader(temp_personas_dir)

        # Load twice
        persona1 = loader.load("standalone")
        persona2 = loader.load("standalone")

        # Should be equal but not same object (copy returned)
        assert persona1 == persona2
        assert persona1 is not persona2

        # Modify one shouldn't affect the other
        persona1["id"] = "modified"
        persona3 = loader.load("standalone")
        assert persona3["id"] == "standalone"

    def test_clear_cache(self, temp_personas_dir):
        """Test clearing the cache."""
        loader = PersonaLoader(temp_personas_dir)

        loader.load("standalone")
        assert len(loader._cache) > 0

        loader.clear_cache()
        assert len(loader._cache) == 0

    def test_list_personas(self, temp_personas_dir):
        """Test listing available personas."""
        loader = PersonaLoader(temp_personas_dir)

        all_personas = loader.list_personas()
        assert "standalone.yaml" in all_personas
        assert "base/agent.yaml" in all_personas
        assert "specific/child.yaml" in all_personas

    def test_list_personas_by_category(self, temp_personas_dir):
        """Test listing personas filtered by category."""
        loader = PersonaLoader(temp_personas_dir)

        base_personas = loader.list_personas("base")
        assert "base/agent.yaml" in base_personas
        assert all("specific" not in p for p in base_personas)


class TestResolvePersonaInAgent:
    """Tests for resolve_persona_in_agent function."""

    @pytest.fixture
    def temp_personas_dir(self):
        """Create a temporary personas directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            personas_dir = Path(tmpdir)

            persona = {
                "id": "test_agent",
                "name": "Test Agent",
                "description": "A test agent",
                "system_prompt": "You are a test agent.",
                "goals": ["Do testing"],
                "default_variables": {
                    "accuracy": 70,
                    "speed": 50
                }
            }
            with open(personas_dir / "test_agent.yaml", "w") as f:
                yaml.dump(persona, f)

            yield personas_dir

    def test_resolve_agent_without_persona(self, temp_personas_dir):
        """Test that agent without persona key is unchanged."""
        loader = PersonaLoader(temp_personas_dir)

        agent_config = {
            "name": "Direct Agent",
            "system_prompt": "I am configured directly."
        }

        result = resolve_persona_in_agent(agent_config, loader)

        assert result == agent_config

    def test_resolve_agent_with_persona(self, temp_personas_dir):
        """Test resolving agent with persona reference."""
        loader = PersonaLoader(temp_personas_dir)

        agent_config = {
            "persona": "test_agent"
        }

        result = resolve_persona_in_agent(agent_config, loader)

        assert result["name"] == "Test Agent"
        assert result["system_prompt"] == "You are a test agent."
        assert result["variables"]["accuracy"] == 70
        assert result["variables"]["speed"] == 50
        assert "persona" not in result

    def test_agent_overrides_persona_values(self, temp_personas_dir):
        """Test that agent values override persona defaults."""
        loader = PersonaLoader(temp_personas_dir)

        agent_config = {
            "persona": "test_agent",
            "name": "Custom Name",  # Override name
            "variables": {
                "accuracy": 95  # Override variable
            }
        }

        result = resolve_persona_in_agent(agent_config, loader)

        assert result["name"] == "Custom Name"
        assert result["variables"]["accuracy"] == 95
        assert result["variables"]["speed"] == 50  # Not overridden

    def test_agent_adds_llm_config(self, temp_personas_dir):
        """Test that agent can add LLM config to persona."""
        loader = PersonaLoader(temp_personas_dir)

        agent_config = {
            "persona": "test_agent",
            "llm": {
                "provider": "openai",
                "model": "gpt-4"
            }
        }

        result = resolve_persona_in_agent(agent_config, loader)

        assert result["llm"]["provider"] == "openai"
        assert result["llm"]["model"] == "gpt-4"


class TestPersonaLoaderWithRealFiles:
    """Tests using the actual persona files in the project."""

    def test_load_base_diplomat(self):
        """Test loading the base diplomat persona."""
        loader = PersonaLoader()

        persona = loader.load("base/diplomat")

        assert persona["id"] == "diplomat"
        assert persona["name"] == "Diplomat"
        assert "diplomatic" in persona["description"].lower()
        assert len(persona["goals"]) > 0

    def test_load_usa_state_dept(self):
        """Test loading USA state dept persona with inheritance."""
        loader = PersonaLoader()

        persona = loader.load("geopolitical/usa_state_dept")

        assert persona["id"] == "usa_state_dept"
        assert "US" in persona["name"] or "State" in persona["name"]

        # Should have inherited and extended goals
        assert len(persona["goals"]) >= 4  # Base has 4, child adds more

        # Should have variables from both
        assert "diplomatic_leverage" in persona["default_variables"]

    def test_load_safety_officer(self):
        """Test loading AI safety officer persona."""
        loader = PersonaLoader()

        persona = loader.load("ai_safety/safety_officer")

        assert persona["id"] == "safety_officer"
        assert "safety" in persona["name"].lower()
        assert "skepticism_level" in persona["default_variables"]
