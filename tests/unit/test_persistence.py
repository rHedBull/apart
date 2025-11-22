import pytest
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.persistence import RunPersistence
from utils.logging_config import LogLevel, MessageCode


class TestRunPersistence:
    """Test RunPersistence class."""

    def test_initialization(self, tmp_path, monkeypatch):
        """Test persistence initialization."""
        # Change to tmp_path for results directory
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test_scenario", save_frequency=1)

        assert persistence.scenario_name == "test_scenario"
        assert persistence.save_frequency == 1
        assert persistence.run_id.startswith("run_test_scenario_")
        assert persistence.run_dir.exists()
        assert persistence.state_file.exists()
        assert persistence.log_file.parent.exists()

        persistence.close()

    def test_scenario_name_sanitization(self, tmp_path, monkeypatch):
        """Test that scenario names are sanitized."""
        monkeypatch.chdir(tmp_path)

        # Test various invalid characters
        persistence = RunPersistence("test/scenario<>:name", save_frequency=1)
        assert "/" not in persistence.scenario_name
        assert "<" not in persistence.scenario_name
        assert ">" not in persistence.scenario_name
        assert ":" not in persistence.scenario_name
        # All invalid chars replaced with underscores
        assert persistence.scenario_name == "test_scenario___name"

        persistence.close()

    def test_scenario_name_truncation(self, tmp_path, monkeypatch):
        """Test that long scenario names are truncated."""
        monkeypatch.chdir(tmp_path)

        long_name = "a" * 100
        persistence = RunPersistence(long_name, save_frequency=1)
        assert len(persistence.scenario_name) == 50

        persistence.close()

    def test_run_directory_collision(self, tmp_path, monkeypatch):
        """Test handling of directory name collisions."""
        monkeypatch.chdir(tmp_path)

        # Create first persistence
        p1 = RunPersistence("test", save_frequency=1)
        run_id_1 = p1.run_id

        # Create directory with same timestamp (simulate collision)
        results_dir = Path("results")
        collision_dir = results_dir / run_id_1
        collision_dir.mkdir(parents=True, exist_ok=True)

        # Create second persistence with same scenario name
        # Should append _1
        p2 = RunPersistence("test", save_frequency=1)

        # Verify it created a different directory
        assert p2.run_dir != p1.run_dir
        assert p2.run_id != p1.run_id

        p1.close()
        p2.close()

    def test_state_file_initialization(self, tmp_path, monkeypatch):
        """Test that state file is properly initialized."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=2)

        # Read state file
        with open(persistence.state_file) as f:
            state = json.load(f)

        assert "run_id" in state
        assert "scenario" in state
        assert state["scenario"] == "test"
        assert "started_at" in state
        assert "snapshots" in state
        assert state["snapshots"] == []

        persistence.close()

    def test_should_save_logic(self, tmp_path, monkeypatch):
        """Test save_frequency logic."""
        monkeypatch.chdir(tmp_path)

        # Test save_frequency = 0 (never save intermediate)
        p0 = RunPersistence("test", save_frequency=0)
        assert p0.should_save(1) is False
        assert p0.should_save(5) is False
        p0.close()

        # Test save_frequency = 1 (save every step)
        p1 = RunPersistence("test", save_frequency=1)
        assert p1.should_save(1) is True
        assert p1.should_save(2) is True
        assert p1.should_save(10) is True
        p1.close()

        # Test save_frequency = 3 (save every 3 steps)
        p3 = RunPersistence("test", save_frequency=3)
        assert p3.should_save(1) is False
        assert p3.should_save(2) is False
        assert p3.should_save(3) is True
        assert p3.should_save(4) is False
        assert p3.should_save(6) is True
        p3.close()

    def test_save_snapshot(self, tmp_path, monkeypatch):
        """Test saving a snapshot."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=1)

        game_state = {"resources": 100, "difficulty": "normal", "round": 1}
        global_vars = {"interest_rate": 0.05}
        agent_vars = {"Agent1": {"economic_strength": 1000.0}}
        messages = [{"from": "Orchestrator", "to": "Agent1", "content": "Test"}]

        persistence.save_snapshot(1, game_state, global_vars, agent_vars, messages)

        # Read state file
        with open(persistence.state_file) as f:
            state = json.load(f)

        assert len(state["snapshots"]) == 1
        snapshot = state["snapshots"][0]

        assert snapshot["step"] == 1
        assert snapshot["game_state"] == game_state
        assert snapshot["global_vars"] == global_vars
        assert snapshot["agent_vars"] == agent_vars
        assert snapshot["messages"] == messages

        persistence.close()

    def test_save_multiple_snapshots(self, tmp_path, monkeypatch):
        """Test saving multiple snapshots."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=1)

        for step in range(1, 4):
            persistence.save_snapshot(
                step,
                {"round": step},
                {"var": step},
                {"Agent1": {"value": step}},
                [{"step": step}]
            )

        # Read state file
        with open(persistence.state_file) as f:
            state = json.load(f)

        assert len(state["snapshots"]) == 3
        assert state["snapshots"][0]["step"] == 1
        assert state["snapshots"][1]["step"] == 2
        assert state["snapshots"][2]["step"] == 3

        persistence.close()

    def test_save_final(self, tmp_path, monkeypatch):
        """Test saving final state."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=2)

        # Save intermediate snapshot at step 2
        persistence.save_snapshot(2, {}, {}, {}, [])

        # Save final at step 5
        persistence.save_final(5, {"final": True}, {}, {}, [])

        with open(persistence.state_file) as f:
            state = json.load(f)

        assert len(state["snapshots"]) == 2
        assert state["snapshots"][0]["step"] == 2
        assert state["snapshots"][1]["step"] == 5
        assert state["snapshots"][1]["game_state"]["final"] is True

        persistence.close()

    def test_logger_integration(self, tmp_path, monkeypatch):
        """Test that logger is properly initialized."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=1, min_log_level=LogLevel.DEBUG)

        assert persistence.logger is not None
        assert persistence.logger.log_file == persistence.log_file
        assert persistence.logger.min_level == LogLevel.DEBUG

        # Log something
        persistence.logger.info(MessageCode.PER001, "Test message")

        persistence.close()

        # Verify log file exists and has content
        assert persistence.log_file.exists()
        with open(persistence.log_file) as f:
            log = json.loads(f.read())
        assert log["message"] == "Test message"

    def test_close_closes_logger(self, tmp_path, monkeypatch):
        """Test that close() closes the logger."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=1)
        persistence.logger.info(MessageCode.PER001, "Before close")

        persistence.close()

        # Logger handle should be closed
        assert persistence.logger._log_handle is None

    def test_atomicity_of_writes(self, tmp_path, monkeypatch):
        """Test that state writes are atomic (temp file used)."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=1)

        # Save snapshot
        persistence.save_snapshot(1, {"test": True}, {}, {}, [])

        # Temp file should not exist after successful write
        temp_file = persistence.state_file.with_suffix(".json.tmp")
        assert not temp_file.exists()

        # State file should exist
        assert persistence.state_file.exists()

        persistence.close()

    def test_read_state_handles_missing_file(self, tmp_path, monkeypatch):
        """Test that _read_state handles missing files gracefully."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=1)

        # Delete state file
        persistence.state_file.unlink()

        # Should return empty structure
        state = persistence._read_state()
        assert "snapshots" in state
        assert state["snapshots"] == []

        persistence.close()


class TestPerformanceLogging:
    """Test that persistence operations are logged."""

    def test_snapshot_save_logs_performance(self, tmp_path, monkeypatch):
        """Test that save_snapshot logs performance metrics."""
        monkeypatch.chdir(tmp_path)

        persistence = RunPersistence("test", save_frequency=1, min_log_level=LogLevel.DEBUG)

        persistence.save_snapshot(1, {}, {}, {}, [])
        persistence.close()

        # Read log file
        with open(persistence.log_file) as f:
            logs = [json.loads(line) for line in f]

        # Should have performance log (DEBUG level PRF001)
        perf_logs = [log for log in logs if log["code"] == "PRF001"]
        assert len(perf_logs) > 0

        perf_log = perf_logs[0]
        assert "Save snapshot" in perf_log["message"]
        assert perf_log["duration_ms"] is not None
        assert perf_log["context"]["step"] == 1
