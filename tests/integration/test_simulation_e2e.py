import pytest
import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from core.orchestrator import Orchestrator


class TestEndToEndSimulation:
    """End-to-end simulation tests using real scenario files."""

    def test_default_scenario_runs_successfully(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that default scenario runs without errors."""
        monkeypatch.chdir(tmp_path)

        # Use the actual project scenario
        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"

        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Verify results
        assert orchestrator.persistence.run_dir.exists()
        assert orchestrator.persistence.state_file.exists()
        assert orchestrator.persistence.log_file.exists()

        # Verify state file structure
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        assert len(state["snapshots"]) > 0
        assert state["scenario"] == "config"

    def test_simulation_produces_complete_audit_trail(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that simulation produces complete audit trail."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"
        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Verify state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Check all snapshots have required fields
        for snapshot in state["snapshots"]:
            assert "step" in snapshot
            assert "game_state" in snapshot
            assert "global_vars" in snapshot
            assert "agent_vars" in snapshot
            assert "messages" in snapshot

            # Verify game_state structure
            assert "resources" in snapshot["game_state"]
            assert "difficulty" in snapshot["game_state"]
            assert "round" in snapshot["game_state"]

        # Verify log file
        with open(orchestrator.persistence.log_file) as f:
            logs = [json.loads(line) for line in f]

        assert len(logs) > 0

        # Verify log structure
        for log in logs:
            assert "timestamp" in log
            assert "level" in log
            assert "code" in log
            assert "message" in log
            assert "context" in log

    def test_simulation_with_variable_tracking(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that variables are properly tracked throughout simulation."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"
        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Verify global variables are present
        first_snapshot = state["snapshots"][0]
        assert "interest_rate" in first_snapshot["global_vars"]
        assert "market_volatility" in first_snapshot["global_vars"]
        assert "simulation_active" in first_snapshot["global_vars"]

        # Verify agent variables
        assert "Agent Alpha" in first_snapshot["agent_vars"]
        assert "Agent Beta" in first_snapshot["agent_vars"]

        agent_alpha_vars = first_snapshot["agent_vars"]["Agent Alpha"]
        assert "economic_strength" in agent_alpha_vars
        assert "risk_tolerance" in agent_alpha_vars
        assert "action_count" in agent_alpha_vars

        # Verify Agent Alpha has custom values
        assert agent_alpha_vars["economic_strength"] == 2500.0
        assert agent_alpha_vars["risk_tolerance"] == 0.8

        # Verify Agent Beta has default values
        agent_beta_vars = first_snapshot["agent_vars"]["Agent Beta"]
        assert agent_beta_vars["economic_strength"] == 1000.0
        assert agent_beta_vars["risk_tolerance"] == 0.5

    def test_simulation_message_flow(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test complete message flow between orchestrator and agents."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"
        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Read state file
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Analyze message flow
        first_step = state["snapshots"][0]
        messages = first_step["messages"]

        # Should have messages for both agents (4 total: 2 to agents, 2 from agents)
        assert len(messages) >= 4

        # Verify message structure (v2.0: SimulatorAgent manages agent communication)
        simulator_messages = [m for m in messages if m["from"] == "SimulatorAgent"]
        agent_messages = [m for m in messages if m["from"] != "SimulatorAgent"]

        assert len(simulator_messages) >= 2  # One per agent
        assert len(agent_messages) >= 2  # One from each agent

        # Verify agents respond correctly
        for agent_msg in agent_messages:
            assert "to" in agent_msg
            assert agent_msg["to"] == "SimulatorAgent"
            assert "content" in agent_msg
            assert len(agent_msg["content"]) > 0


class TestSimulationDataIntegrity:
    """Test data integrity across simulation run."""

    def test_round_numbers_increment_correctly(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that round numbers increment as expected."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"
        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Extract rounds from snapshots
        rounds = [snapshot["game_state"]["round"] for snapshot in state["snapshots"]]

        # Rounds should increment
        for i in range(1, len(rounds)):
            assert rounds[i] >= rounds[i-1]

    def test_step_numbers_are_sequential(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that step numbers are sequential."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"
        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Extract steps
        steps = [snapshot["step"] for snapshot in state["snapshots"]]

        # Steps should be sequential (and increasing or equal for final duplicate)
        for i in range(1, len(steps)):
            assert steps[i] >= steps[i-1], f"Steps should not decrease: {steps}"

        # First step should be 1
        assert steps[0] == 1

        # Last two should be the same (final duplicate)
        if len(steps) > 1:
            assert steps[-1] == steps[-2], f"Final should duplicate last step: {steps}"

    def test_timestamps_are_chronological(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that log timestamps are in chronological order."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"
        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        with open(orchestrator.persistence.log_file) as f:
            logs = [json.loads(line) for line in f]

        # Extract timestamps
        timestamps = [log["timestamp"] for log in logs]

        # Timestamps should be in order
        assert timestamps == sorted(timestamps)


class TestSimulationReproducibility:
    """Test that simulations are reproducible."""

    def test_same_config_produces_consistent_structure(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that running same config produces consistent structure."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"

        # Run simulation twice
        orch1 = Orchestrator(str(scenario_path), "config1", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orch1.run()

        orch2 = Orchestrator(str(scenario_path), "config2", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orch2.run()

        # Read both state files
        with open(orch1.persistence.state_file) as f:
            state1 = json.load(f)

        with open(orch2.persistence.state_file) as f:
            state2 = json.load(f)

        # Should have same number of snapshots
        assert len(state1["snapshots"]) == len(state2["snapshots"])

        # Should have same structure (keys)
        for snap1, snap2 in zip(state1["snapshots"], state2["snapshots"]):
            assert snap1.keys() == snap2.keys()
            assert snap1["game_state"].keys() == snap2["game_state"].keys()


class TestSimulationOutputFormats:
    """Test simulation output file formats."""

    def test_state_file_is_valid_json(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that state file is valid JSON."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"
        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Should be able to parse without errors
        with open(orchestrator.persistence.state_file) as f:
            state = json.load(f)

        # Verify it's a dictionary
        assert isinstance(state, dict)

    def test_log_file_is_valid_jsonl(self, tmp_path, monkeypatch, mock_engine_llm_provider):
        """Test that log file is valid JSONL."""
        monkeypatch.chdir(tmp_path)

        scenario_path = Path(__file__).parent.parent.parent / "scenarios" / "config.yaml"
        orchestrator = Orchestrator(str(scenario_path), "config", save_frequency=1, engine_llm_provider=mock_engine_llm_provider)
        orchestrator.run()

        # Each line should be valid JSON
        with open(orchestrator.persistence.log_file) as f:
            for line_num, line in enumerate(f, 1):
                try:
                    log = json.loads(line)
                    assert isinstance(log, dict)
                except json.JSONDecodeError as e:
                    pytest.fail(f"Invalid JSON on line {line_num}: {e}")
