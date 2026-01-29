"""Unit tests for Orchestrator configuration loading and initialization."""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest


class TestOrchestratorConfigLoading:
    """Tests for Orchestrator configuration loading."""

    def test_load_config_valid_yaml(self, tmp_path):
        """Test loading a valid YAML configuration."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Test Scenario
max_steps: 5
time_step_duration: "1 hour"
agents:
  - name: TestAgent
    response_template: "I respond with {input}"
engine:
  provider: mock
  model: test
initial_state:
  global_vars:
    test_var: 100
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm
            mock_persist.return_value.run_id = "test-run"
            mock_persist.return_value.run_dir = tmp_path

            orch = Orchestrator(str(config_file), "test", save_frequency=1)

            assert orch.config["name"] == "Test Scenario"
            assert orch.max_steps == 5
            assert orch.time_step_duration == "1 hour"

    def test_load_config_file_not_found(self, tmp_path):
        """Test loading a non-existent configuration file."""
        from core.orchestrator import Orchestrator

        with pytest.raises(FileNotFoundError):
            Orchestrator("/nonexistent/path.yaml", "test", save_frequency=1)

    def test_load_config_invalid_yaml(self, tmp_path):
        """Test loading an invalid YAML file."""
        from core.orchestrator import Orchestrator

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            Orchestrator(str(config_file), "test", save_frequency=1)

    def test_default_max_steps(self, tmp_path):
        """Test that max_steps defaults to 5 if not specified."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Test
agents: []
"""
        config_file = tmp_path / "minimal.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm
            mock_persist.return_value.run_id = "test-run"
            mock_persist.return_value.run_dir = tmp_path

            orch = Orchestrator(str(config_file), "test", save_frequency=1)

            assert orch.max_steps == 5

    def test_default_time_step_duration(self, tmp_path):
        """Test that time_step_duration defaults to '1 turn'."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Test
agents: []
"""
        config_file = tmp_path / "no_duration.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm
            mock_persist.return_value.run_id = "test-run"
            mock_persist.return_value.run_dir = tmp_path

            orch = Orchestrator(str(config_file), "test", save_frequency=1)

            assert orch.time_step_duration == "1 turn"

    def test_default_simulator_awareness(self, tmp_path):
        """Test that simulator_awareness defaults to True."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Test
agents: []
"""
        config_file = tmp_path / "no_awareness.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm
            mock_persist.return_value.run_id = "test-run"
            mock_persist.return_value.run_dir = tmp_path

            orch = Orchestrator(str(config_file), "test", save_frequency=1)

            assert orch.simulator_awareness is True


class TestOrchestratorAgentInitialization:
    """Tests for agent initialization in Orchestrator."""

    def test_initialize_template_agent(self, tmp_path):
        """Test initializing a template-based agent (no LLM)."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Template Test
max_steps: 1
agents:
  - name: TemplateAgent
    response_template: "Fixed response"
"""
        config_file = tmp_path / "template_agent.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm
            mock_persist.return_value.run_id = "test-run"
            mock_persist.return_value.run_dir = tmp_path
            mock_persist.return_value.logger = MagicMock()

            orch = Orchestrator(str(config_file), "test", save_frequency=1)

            assert len(orch.agents) == 1
            assert orch.agents[0].name == "TemplateAgent"
            assert orch.agents[0].llm_provider is None

    def test_multiple_agents_initialization(self, tmp_path):
        """Test initializing multiple agents."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Multi Agent
max_steps: 1
agents:
  - name: Agent1
    response_template: "Response 1"
  - name: Agent2
    response_template: "Response 2"
  - name: Agent3
    response_template: "Response 3"
"""
        config_file = tmp_path / "multi_agent.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm
            mock_persist.return_value.run_id = "test-run"
            mock_persist.return_value.run_dir = tmp_path
            mock_persist.return_value.logger = MagicMock()

            orch = Orchestrator(str(config_file), "test", save_frequency=1)

            assert len(orch.agents) == 3
            names = [a.name for a in orch.agents]
            assert names == ["Agent1", "Agent2", "Agent3"]

    def test_unknown_llm_provider_raises_error(self, tmp_path):
        """Test that unknown LLM provider raises ValueError."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Unknown Provider
max_steps: 1
agents:
  - name: BadAgent
    llm:
      provider: unknown_provider
      model: test
"""
        config_file = tmp_path / "unknown_provider.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm
            mock_persist.return_value.run_id = "test-run"
            mock_persist.return_value.run_dir = tmp_path
            mock_persist.return_value.logger = MagicMock()

            with pytest.raises(ValueError, match="Unknown LLM provider"):
                Orchestrator(str(config_file), "test", save_frequency=1)


class TestOrchestratorRunId:
    """Tests for run ID handling."""

    def test_custom_run_id_passed_to_persistence(self, tmp_path):
        """Test that custom run_id is passed to RunPersistence."""
        from core.orchestrator import Orchestrator

        config_content = """
name: RunID Test
max_steps: 1
agents: []
"""
        config_file = tmp_path / "runid_test.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as MockPersist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm

            mock_persist_instance = MagicMock()
            mock_persist_instance.run_id = "custom-id-123"
            mock_persist_instance.run_dir = tmp_path
            mock_persist_instance.logger = MagicMock()
            MockPersist.return_value = mock_persist_instance

            Orchestrator(
                str(config_file),
                "test",
                save_frequency=1,
                run_id="custom-id-123"
            )

        MockPersist.assert_called_once()
        call_args = MockPersist.call_args
        assert call_args[1]["run_id"] == "custom-id-123"

    def test_none_run_id_allows_auto_generation(self, tmp_path):
        """Test that None run_id allows auto-generation."""
        from core.orchestrator import Orchestrator

        config_content = """
name: AutoID Test
max_steps: 1
agents: []
"""
        config_file = tmp_path / "autoid_test.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as MockPersist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm

            mock_persist_instance = MagicMock()
            mock_persist_instance.run_id = "auto-generated-id"
            mock_persist_instance.run_dir = tmp_path
            mock_persist_instance.logger = MagicMock()
            MockPersist.return_value = mock_persist_instance

            Orchestrator(str(config_file), "test", save_frequency=1)

        MockPersist.assert_called_once()
        call_args = MockPersist.call_args
        assert call_args[1]["run_id"] is None


class TestEngineLLMProviderCreation:
    """Tests for engine LLM provider creation."""

    def test_engine_llm_unavailable_raises_error(self, tmp_path):
        """Test that unavailable engine LLM raises error."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Engine LLM Test
max_steps: 1
agents: []
engine:
  provider: openai
  model: gpt-4
"""
        config_file = tmp_path / "engine_llm.yaml"
        config_file.write_text(config_content)

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch("core.orchestrator.UnifiedLLMProvider") as MockProvider, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = False
            MockProvider.return_value = mock_llm

            mock_persist.return_value.run_id = "test"
            mock_persist.return_value.run_dir = tmp_path
            mock_persist.return_value.logger = MagicMock()

            with pytest.raises(ValueError, match="Engine LLM provider not available"):
                Orchestrator(str(config_file), "test", save_frequency=1)

    def test_injected_engine_provider_used(self, tmp_path):
        """Test that injected engine LLM provider is used."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Injected Provider
max_steps: 1
agents: []
"""
        config_file = tmp_path / "injected.yaml"
        config_file.write_text(config_content)

        mock_injected_llm = MagicMock()
        mock_injected_llm.is_available.return_value = True

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_create, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent") as MockSimAgent, \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_persist.return_value.run_id = "test"
            mock_persist.return_value.run_dir = tmp_path
            mock_persist.return_value.logger = MagicMock()

            # Pass injected provider
            Orchestrator(
                str(config_file),
                "test",
                save_frequency=1,
                engine_llm_provider=mock_injected_llm
            )

        # Should NOT call _create_llm_provider_for_engine
        mock_create.assert_not_called()

        # SimulatorAgent should receive the injected provider
        MockSimAgent.assert_called_once()
        call_kwargs = MockSimAgent.call_args[1]
        assert call_kwargs["llm_provider"] is mock_injected_llm


class TestOrchestratorModuleComposition:
    """Tests for module composition in Orchestrator."""

    def test_modules_merged_into_config(self, tmp_path):
        """Test that module variables are merged into config."""
        from core.orchestrator import Orchestrator

        config_content = """
name: Module Test
max_steps: 1
agents: []
modules:
  - test_module
"""
        config_file = tmp_path / "module_test.yaml"
        config_file.write_text(config_content)

        mock_modules = MagicMock()
        mock_modules.spatial_graph = None
        mock_modules.movement_config = None
        mock_modules.geojson = None

        with patch.object(Orchestrator, "_initialize_agents", return_value=[]), \
             patch.object(Orchestrator, "_create_llm_provider_for_engine") as mock_provider, \
             patch("core.orchestrator.parse_modules") as mock_parse, \
             patch("core.orchestrator.merge_module_variables") as mock_merge, \
             patch("core.orchestrator.GameEngine"), \
             patch("core.orchestrator.SimulatorAgent"), \
             patch("core.orchestrator.RunPersistence") as mock_persist:

            mock_llm = MagicMock()
            mock_llm.is_available.return_value = True
            mock_provider.return_value = mock_llm

            mock_parse.return_value = mock_modules
            mock_merge.return_value = {"name": "Module Test", "max_steps": 1, "agents": []}

            mock_persist.return_value.run_id = "test"
            mock_persist.return_value.run_dir = tmp_path
            mock_persist.return_value.logger = MagicMock()

            Orchestrator(str(config_file), "test", save_frequency=1)

        mock_parse.assert_called_once()
        mock_merge.assert_called_once()
