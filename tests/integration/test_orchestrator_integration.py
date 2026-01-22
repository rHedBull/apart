import pytest
import sys
import json
from pathlib import Path
from tempfile import TemporaryDirectory

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.orchestrator import Orchestrator
from utils.logging_config import MessageCode


@pytest.fixture
def test_config_file(tmp_path):
    """Create a minimal test configuration file."""
    config = """
max_steps: 3

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "Test"
  simulation_plan: "Test"

game_state:
  initial_resources: 100
  difficulty: "easy"

global_vars:
  test_var:
    type: float
    default: 1.0
    min: 0.0
    max: 10.0

agent_vars:
  agent_score:
    type: int
    default: 0
    min: 0

agents:
  - name: "Agent A"
    response_template: "Agent A ready"
  - name: "Agent B"
    response_template: "Agent B ready"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config)
    return str(config_file)


class TestOrchestratorIntegration:
    """Integration tests for Orchestrator."""

    def test_orchestrator_initialization(self, test_config_file, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that orchestrator initializes all components."""
        monkeypatch.chdir(tmp_path)

        orchestrator = Orchestrator(test_config_file, "test", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)

        # Verify components initialized
        assert orchestrator.config is not None
        assert orchestrator.max_steps == 3
        assert len(orchestrator.agents) == 2
        assert orchestrator.agents[0].name == "Agent A"
        assert orchestrator.agents[1].name == "Agent B"
        assert orchestrator.game_engine is not None
        assert orchestrator.persistence is not None
        assert orchestrator.logger is not None

        orchestrator.persistence.close()

    def test_orchestrator_full_simulation(self, test_config_file, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test complete simulation run."""
        monkeypatch.chdir(tmp_path)

        orchestrator = Orchestrator(test_config_file, "test", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Verify state file exists and has correct structure
        state_file = orchestrator.persistence.state_file
        assert state_file.exists()

        with open(state_file) as f:
            state = json.load(f)

        assert "run_id" in state
        assert "snapshots" in state
        # Should have 3 steps + final (4 total, final duplicates last step)
        assert len(state["snapshots"]) == 4

        # Verify snapshots have required fields
        for snapshot in state["snapshots"]:
            assert "step" in snapshot
            assert "game_state" in snapshot
            assert "global_vars" in snapshot
            assert "agent_vars" in snapshot
            assert "messages" in snapshot

        # Verify steps (1, 2, 3, 3 with final duplicate)
        steps = [s["step"] for s in state["snapshots"]]
        assert steps == [1, 2, 3, 3]

        # Verify log file exists
        log_file = orchestrator.persistence.log_file
        assert log_file.exists()

    def test_orchestrator_logs_simulation_lifecycle(self, test_config_file, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that orchestrator logs all lifecycle events."""
        monkeypatch.chdir(tmp_path)

        orchestrator = Orchestrator(test_config_file, "test", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read log file
        with open(orchestrator.persistence.log_file) as f:
            logs = [json.loads(line) for line in f]

        # Extract message codes
        codes = [log["code"] for log in logs]

        # Verify key lifecycle events
        assert "AGT001" in codes  # Agent initialized
        assert "PER001" in codes  # Run directory created
        assert "SIM001" in codes  # Simulation started
        assert "SIM003" in codes  # Step started
        assert "AGT002" in codes  # Message sent to agent
        assert "AGT003" in codes  # Response received
        assert "SIM004" in codes  # Step completed
        assert "PER002" in codes  # Snapshot saved
        assert "PER003" in codes  # Final state saved
        assert "SIM002" in codes  # Simulation completed

    def test_orchestrator_agent_interaction(self, test_config_file, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that agents interact properly with orchestrator."""
        monkeypatch.chdir(tmp_path)

        orchestrator = Orchestrator(test_config_file, "test", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Check first snapshot messages
        first_snapshot = state["snapshots"][0]
        messages = first_snapshot["messages"]

        # Should have 4 messages (2 agents × 2 messages each)
        assert len(messages) == 4

        # Verify message structure (v2.0: SimulatorAgent manages agent communication)
        # With parallel agent execution, all outgoing messages are sent first,
        # then all responses are collected
        assert messages[0]["from"] == "SimulatorAgent"
        assert messages[0]["to"] == "Agent A"
        assert messages[1]["from"] == "SimulatorAgent"
        assert messages[1]["to"] == "Agent B"
        assert messages[2]["from"] == "Agent A"
        assert messages[2]["to"] == "SimulatorAgent"
        assert messages[3]["from"] == "Agent B"
        assert messages[3]["to"] == "SimulatorAgent"

    def test_orchestrator_state_evolution(self, test_config_file, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that game state evolves across steps."""
        monkeypatch.chdir(tmp_path)

        orchestrator = Orchestrator(test_config_file, "test", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Verify round advances (with final duplicate at end)
        rounds = [snapshot["game_state"]["round"] for snapshot in state["snapshots"]]
        # Rounds: 0 (start), 1 (after step 1), 2 (after step 2), 3 (final after step 3)
        assert rounds == [0, 1, 2, 3]

    def test_orchestrator_with_save_frequency_zero(self, test_config_file, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test orchestrator with save_frequency=0 (final only)."""
        monkeypatch.chdir(tmp_path)

        orchestrator = Orchestrator(test_config_file, "test", save_frequency=0, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Should only have final snapshot
        assert len(state["snapshots"]) == 1
        assert state["snapshots"][0]["step"] == 3

    def test_orchestrator_with_save_frequency_two(self, test_config_file, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test orchestrator with save_frequency=2."""
        monkeypatch.chdir(tmp_path)

        orchestrator = Orchestrator(test_config_file, "test", save_frequency=2, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # With save_frequency=2: save at step 2, then final at step 3
        assert len(state["snapshots"]) == 2
        steps = [s["step"] for s in state["snapshots"]]
        assert steps == [2, 3]

    def test_orchestrator_logger_cleanup(self, test_config_file, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that logger is properly closed after simulation."""
        monkeypatch.chdir(tmp_path)

        orchestrator = Orchestrator(test_config_file, "test", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Logger handle should be closed
        assert orchestrator.logger._log_handle is None


class TestOrchestratorErrorHandling:
    """Integration tests for orchestrator error handling."""

    def test_orchestrator_handles_missing_config(self, tmp_path, monkeypatch):
        """Test error handling for missing configuration file."""
        monkeypatch.chdir(tmp_path)

        with pytest.raises(FileNotFoundError) as exc_info:
            Orchestrator("nonexistent.yaml", "test", save_frequency=1)

        assert "Configuration file not found" in str(exc_info.value)

    def test_orchestrator_handles_invalid_yaml(self, tmp_path, monkeypatch):
        """Test error handling for invalid YAML."""
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("invalid: yaml: [")

        with pytest.raises(ValueError) as exc_info:
            Orchestrator(str(config_file), "test", save_frequency=1)

        assert "Invalid YAML" in str(exc_info.value)

    def test_orchestrator_handles_invalid_config_structure(self, tmp_path, monkeypatch):
        """Test error handling for invalid configuration values."""
        monkeypatch.chdir(tmp_path)

        # Config with invalid variable constraint
        config_file = tmp_path / "bad_config.yaml"
        config_file.write_text("""
max_steps: 2
engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "Test"
  simulation_plan: "Test"
global_vars:
  bad_var:
    type: float
    default: 10.0
    max: 1.0
agents:
  - name: "Agent A"
    response_template: "Test"
""")

        with pytest.raises(ValueError) as exc_info:
            Orchestrator(str(config_file), "test", save_frequency=1)

        assert "cannot be greater than max" in str(exc_info.value).lower()


class TestOrchestratorPerformance:
    """Performance and stress tests for orchestrator."""

    def test_orchestrator_with_many_agents(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test orchestrator with multiple agents."""
        monkeypatch.chdir(tmp_path)

        # Create config with 5 agents
        config_file = tmp_path / "many_agents.yaml"
        config_file.write_text("""
max_steps: 2
engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "Test"
  simulation_plan: "Test"
agents:
  - name: "Agent 1"
    response_template: "Agent 1"
  - name: "Agent 2"
    response_template: "Agent 2"
  - name: "Agent 3"
    response_template: "Agent 3"
  - name: "Agent 4"
    response_template: "Agent 4"
  - name: "Agent 5"
    response_template: "Agent 5"
""")

        orchestrator = Orchestrator(str(config_file), "test", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Each snapshot should have 10 messages (5 agents × 2 messages)
        for snapshot in state["snapshots"]:
            assert len(snapshot["messages"]) == 10

    def test_orchestrator_with_many_steps(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test orchestrator with many simulation steps."""
        monkeypatch.chdir(tmp_path)

        config_file = tmp_path / "many_steps.yaml"
        config_file.write_text("""
max_steps: 20
engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "Test"
  simulation_plan: "Test"
agents:
  - name: "Agent A"
    response_template: "Ready"
""")

        orchestrator = Orchestrator(str(config_file), "test", save_frequency=5, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Should have snapshots at steps 5, 10, 15, 20 (final duplicates 20)
        assert len(state["snapshots"]) == 5
        assert [s["step"] for s in state["snapshots"]] == [5, 10, 15, 20, 20]
