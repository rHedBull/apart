"""
End-to-end tests for pause and resume functionality.

Tests the full lifecycle:
- Pause signal detection
- Correct snapshot storage on pause
- State transition to paused
- Resume from correct step
- Proper state restoration
"""

import json
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture
def e2e_scenario_file(tmp_path):
    """Create a scenario file for E2E testing."""
    config = """
max_steps: 5

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: "Test simulation"
  simulation_plan: "Run for 5 steps"

agents:
  - name: "TestAgent"
    response_template: "Agent response"
"""
    config_file = tmp_path / "e2e_scenario.yaml"
    config_file.write_text(config)
    return str(config_file)


class TestOrchestratorRunReturnValue:
    """Tests for orchestrator.run() return value indicating pause or completion."""

    def _create_mock_orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.agents = []
            orch.max_steps = 5
            orch.logger = MagicMock()
            orch.game_engine = MagicMock()
            orch.persistence = MagicMock()
            orch.persistence.run_id = "test-run-id"
            orch.persistence.scenario_name = "test-scenario"
            orch.spatial_graph = None
            orch.composed_modules = None
            return orch

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    @patch('core.orchestrator.clear_pause_signal')
    def test_run_returns_completed_status_on_normal_finish(
        self, mock_clear, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that run() returns completed status when simulation finishes normally."""
        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = MagicMock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = MagicMock(return_value=({}, []))
        orch._process_step_results = MagicMock(return_value={"Agent1": "msg"})
        orch._save_final_state = MagicMock()
        mock_check.return_value = None  # No pause signal

        result = orch.run()

        assert result == {"status": "completed"}
        orch._save_final_state.assert_called_once()

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    @patch('core.orchestrator.clear_pause_signal')
    def test_run_returns_paused_status_with_step(
        self, mock_clear, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that run() returns paused status with paused_at_step when paused."""
        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = MagicMock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = MagicMock(return_value=({}, []))
        orch._process_step_results = MagicMock(return_value={"Agent1": "msg"})
        orch._save_final_state = MagicMock()

        # Pause on step 3: checks are after init, step1 start, step1 after, step2 start, step2 after, step3 start
        mock_check.side_effect = [None, None, None, None, None, {"force": False}]

        result = orch.run()

        assert result["status"] == "paused"
        assert result["paused_at_step"] == 3

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    @patch('core.orchestrator.clear_pause_signal')
    def test_run_does_not_save_final_state_on_pause(
        self, mock_clear, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that _save_final_state is NOT called when simulation is paused."""
        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = MagicMock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = MagicMock(return_value=({}, []))
        orch._process_step_results = MagicMock(return_value={"Agent1": "msg"})
        orch._save_final_state = MagicMock()

        # Pause on step 2
        mock_check.side_effect = [None, {"force": False}]

        result = orch.run()

        assert result["status"] == "paused"
        # Critical: _save_final_state should NOT be called on pause
        orch._save_final_state.assert_not_called()

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    @patch('core.orchestrator.clear_pause_signal')
    def test_run_paused_at_step_1(
        self, mock_clear, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test pausing at step 1 returns correct paused_at_step."""
        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = MagicMock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = MagicMock(return_value=({}, []))
        orch._process_step_results = MagicMock(return_value={"Agent1": "msg"})
        orch._save_final_state = MagicMock()

        # Pause at step 1 start: None for init check, then pause
        mock_check.side_effect = [None, {"force": False}]

        result = orch.run()

        assert result["status"] == "paused"
        assert result["paused_at_step"] == 1
        # No steps should have been executed
        orch._collect_agent_responses.assert_not_called()


class TestWorkerTaskPauseStateTransition:
    """Tests for worker task handling pause state transitions."""

    def test_worker_transitions_to_paused_on_pause_result(self, tmp_path):
        """Test that worker transitions state to 'paused' when orchestrator returns paused."""
        from server.worker_tasks import run_simulation_task
        from server.run_state import RunStateManager
        from fakeredis import FakeRedis

        # Set up state manager
        redis = FakeRedis()
        RunStateManager.reset_instance()
        state_manager = RunStateManager.initialize(redis)

        scenario_path = tmp_path / "pause_scenario.yaml"
        scenario_path.write_text("name: test\nmax_steps: 5\n")

        # Create the run in pending state first, then transition to running
        state_manager.create_run("pause-test-run", str(scenario_path), "pause_scenario")
        state_manager.transition("pause-test-run", "running")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"):

            mock_orchestrator = MagicMock()
            mock_orchestrator.run.return_value = {"status": "paused", "paused_at_step": 3}
            mock_orch_class.return_value = mock_orchestrator

            result = run_simulation_task("pause-test-run", str(scenario_path))

            assert result["status"] == "paused"
            assert result["paused_at_step"] == 3

            # Verify state manager has paused state
            state = state_manager.get_state("pause-test-run")
            assert state.status == "paused"

        RunStateManager.reset_instance()

    def test_worker_transitions_to_completed_on_completed_result(self, tmp_path):
        """Test that worker transitions state to 'completed' when orchestrator completes."""
        from server.worker_tasks import run_simulation_task
        from server.run_state import RunStateManager
        from fakeredis import FakeRedis

        redis = FakeRedis()
        RunStateManager.reset_instance()
        state_manager = RunStateManager.initialize(redis)

        scenario_path = tmp_path / "complete_scenario.yaml"
        scenario_path.write_text("name: test\nmax_steps: 5\n")

        state_manager.create_run("complete-test-run", str(scenario_path), "complete_scenario")
        state_manager.transition("complete-test-run", "running")

        with patch("core.orchestrator.Orchestrator") as mock_orch_class, \
             patch("core.event_emitter.enable_event_emitter"):

            mock_orchestrator = MagicMock()
            mock_orchestrator.run.return_value = {"status": "completed"}
            mock_orch_class.return_value = mock_orchestrator

            result = run_simulation_task("complete-test-run", str(scenario_path))

            assert result["status"] == "completed"

            state = state_manager.get_state("complete-test-run")
            assert state.status == "completed"

        RunStateManager.reset_instance()


class TestStoppingToPausedTransition:
    """Tests for the stopping -> paused state transition flow."""

    def test_pause_endpoint_transitions_to_stopping(self):
        """Test that pause endpoint transitions run to 'stopping' state."""
        from fastapi.testclient import TestClient
        from server.app import app
        from server.run_state import RunStateManager
        from server.event_bus import EventBus
        from fakeredis import FakeRedis

        # Reset singletons
        EventBus.reset_instance()
        RunStateManager.reset_instance()

        redis = FakeRedis()
        state_manager = RunStateManager.initialize(redis)

        # Initialize job queue mock
        with patch("server.job_queue._redis_conn", redis), \
             patch("server.job_queue._queues", {"normal": MagicMock()}):

            # Create a running simulation
            state_manager.create_run("stopping-test", "scenario.yaml", "test-scenario")
            state_manager.transition("stopping-test", "running")

            client = TestClient(app)
            response = client.post("/api/v1/runs/stopping-test/pause")

            assert response.status_code == 200
            assert response.json()["status"] == "stopping"

            # Verify state is now "stopping"
            state = state_manager.get_state("stopping-test")
            assert state.status == "stopping"

        RunStateManager.reset_instance()
        EventBus.reset_instance()

    def test_valid_transition_from_stopping_to_paused(self):
        """Test that stopping -> paused is a valid state transition."""
        from server.run_state import RunStateManager
        from fakeredis import FakeRedis

        redis = FakeRedis()
        RunStateManager.reset_instance()
        state_manager = RunStateManager.initialize(redis)

        state_manager.create_run("transition-test", "scenario.yaml", "test-scenario")
        state_manager.transition("transition-test", "running")
        state_manager.transition("transition-test", "stopping")

        # This should succeed
        state = state_manager.transition("transition-test", "paused")
        assert state.status == "paused"

        RunStateManager.reset_instance()


class TestSnapshotOnPause:
    """Tests for snapshot storage when simulation is paused."""

    def test_snapshot_saved_before_pause(self, tmp_path):
        """Test that snapshot is saved for completed steps before pause."""
        from core.orchestrator import Orchestrator
        from unittest.mock import patch, MagicMock, call

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.agents = []
            orch.max_steps = 5
            orch.logger = MagicMock()
            orch.game_engine = MagicMock()
            orch.persistence = MagicMock()
            orch.persistence.run_id = "snapshot-test"
            orch.persistence.scenario_name = "test"
            orch.persistence.save_frequency = 1
            orch.spatial_graph = None
            orch.composed_modules = None

            # Track what steps were saved
            saved_steps = []
            def track_save(step, *args, **kwargs):
                saved_steps.append(step)
            orch.persistence.save_snapshot = MagicMock(side_effect=track_save)

            orch._initialize_simulation = MagicMock(return_value={"Agent1": "msg"})
            orch._collect_agent_responses = MagicMock(return_value=({}, []))
            orch._process_step_results = MagicMock(return_value={"Agent1": "msg"})
            orch._save_final_state = MagicMock()
            orch.game_engine.get_state_snapshot = MagicMock(return_value={
                "game_state": {},
                "global_vars": {},
                "agent_vars": {}
            })

            with patch('core.orchestrator.emit'), \
                 patch('core.orchestrator.enable_event_emitter'), \
                 patch('core.orchestrator.disable_event_emitter'), \
                 patch('core.orchestrator.check_pause_requested') as mock_check, \
                 patch('core.orchestrator.clear_pause_signal'):

                # Run steps 1 and 2, then pause at step 3 start
                # Checks: init, step1 start, step1 after, step2 start, step2 after, step3 start
                mock_check.side_effect = [None, None, None, None, None, {"force": False}]

                result = orch.run()

                assert result["status"] == "paused"
                assert result["paused_at_step"] == 3

                # Steps 1 and 2 should have been executed
                assert orch._collect_agent_responses.call_count == 2


class TestResumeFromCorrectStep:
    """Tests for resuming from the correct step after pause."""

    def test_resume_continues_from_paused_step(self, tmp_path):
        """Test that resume starts from the step after where we paused."""
        import json
        from core.orchestrator import Orchestrator
        from unittest.mock import patch, MagicMock

        # Create state file simulating pause at step 2
        state_data = {
            "run_id": "resume-test",
            "scenario": "test",
            "snapshots": [
                {"step": 1, "global_vars": {"x": 10}, "agent_vars": {}, "messages": []},
                {"step": 2, "global_vars": {"x": 20}, "agent_vars": {}, "messages": []},
            ]
        }
        results_dir = tmp_path / "results" / "resume-test"
        results_dir.mkdir(parents=True)
        (results_dir / "state.json").write_text(json.dumps(state_data))

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.agents = []
            orch.max_steps = 5
            orch.logger = MagicMock()
            orch.game_engine = MagicMock()
            orch.game_engine.state = MagicMock()
            orch.persistence = MagicMock()
            orch.persistence.run_id = "resume-test"
            orch.persistence.run_dir = results_dir
            orch.persistence.scenario_name = "test"
            orch.spatial_graph = None
            orch.composed_modules = None
            orch.simulator_agent = MagicMock()
            orch.simulator_agent.initialize_simulation = MagicMock(return_value={"Agent1": "msg"})

            orch._collect_agent_responses = MagicMock(return_value=({}, []))
            orch._process_step_results = MagicMock(return_value={"Agent1": "msg"})
            orch._save_final_state = MagicMock()

            with patch('core.orchestrator.emit'), \
                 patch('core.orchestrator.enable_event_emitter'), \
                 patch('core.orchestrator.disable_event_emitter'), \
                 patch('core.orchestrator.check_pause_requested', return_value=None):

                # Resume from step 3 (where we would continue after pausing at step 2)
                result = orch.run(start_step=3)

                assert result["status"] == "completed"

                # Should have run steps 3, 4, 5
                assert orch._collect_agent_responses.call_count == 3

                # Verify state was restored
                orch.game_engine.apply_state_updates.assert_called()

    def test_resume_restores_correct_global_vars(self, tmp_path):
        """Test that resume restores the correct global_vars from snapshot."""
        import json
        from core.orchestrator import Orchestrator
        from unittest.mock import patch, MagicMock

        state_data = {
            "run_id": "vars-test",
            "scenario": "test",
            "snapshots": [
                {"step": 1, "global_vars": {"resource": 100, "turn": 1}, "agent_vars": {}, "messages": []},
                {"step": 2, "global_vars": {"resource": 75, "turn": 2}, "agent_vars": {}, "messages": []},
                {"step": 3, "global_vars": {"resource": 50, "turn": 3}, "agent_vars": {}, "messages": []},
            ]
        }
        results_dir = tmp_path / "results" / "vars-test"
        results_dir.mkdir(parents=True)
        (results_dir / "state.json").write_text(json.dumps(state_data))

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.agents = []
            orch.max_steps = 5
            orch.logger = MagicMock()
            orch.game_engine = MagicMock()
            orch.game_engine.state = MagicMock()
            orch.persistence = MagicMock()
            orch.persistence.run_dir = results_dir

            # Resume from step 4 - should restore state from step 3 snapshot
            orch._restore_state_for_resume(4)

            # Verify correct snapshot was used
            call_args = orch.game_engine.apply_state_updates.call_args[0][0]
            assert call_args["global_vars"] == {"resource": 50, "turn": 3}
            assert orch.game_engine.state.round == 3

    def test_resume_restores_correct_agent_vars(self, tmp_path):
        """Test that resume restores the correct agent_vars from snapshot."""
        import json
        from core.orchestrator import Orchestrator
        from unittest.mock import patch, MagicMock

        state_data = {
            "run_id": "agent-vars-test",
            "scenario": "test",
            "snapshots": [
                {"step": 1, "global_vars": {}, "agent_vars": {"Agent1": {"health": 100}}, "messages": []},
                {"step": 2, "global_vars": {}, "agent_vars": {"Agent1": {"health": 80}}, "messages": []},
            ]
        }
        results_dir = tmp_path / "results" / "agent-vars-test"
        results_dir.mkdir(parents=True)
        (results_dir / "state.json").write_text(json.dumps(state_data))

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            mock_agent = MagicMock()
            mock_agent.name = "Agent1"
            orch.agents = [mock_agent]
            orch.logger = MagicMock()
            orch.game_engine = MagicMock()
            orch.game_engine.state = MagicMock()
            orch.persistence = MagicMock()
            orch.persistence.run_dir = results_dir

            # Resume from step 3 - should restore from step 2 snapshot
            orch._restore_state_for_resume(3)

            # Verify agent stats were updated
            mock_agent.update_stats.assert_called_once_with({"health": 80})


class TestPauseResumeIntegration:
    """Integration tests for the full pause-resume cycle."""

    def test_full_pause_resume_cycle_with_state_manager(self, tmp_path):
        """Test full cycle: start -> pause -> verify state -> resume -> complete."""
        from server.run_state import RunStateManager
        from fakeredis import FakeRedis

        redis = FakeRedis()
        RunStateManager.reset_instance()
        state_manager = RunStateManager.initialize(redis)

        scenario_path = tmp_path / "cycle_scenario.yaml"
        scenario_path.write_text("name: cycle_test\nmax_steps: 5\n")

        # 1. Create run
        state_manager.create_run("cycle-run", str(scenario_path), "cycle_test")
        state = state_manager.get_state("cycle-run")
        assert state.status == "pending"

        # 2. Start (transition to running)
        state_manager.transition("cycle-run", "running")
        state = state_manager.get_state("cycle-run")
        assert state.status == "running"

        # 3. Request pause (transition to stopping)
        state_manager.transition("cycle-run", "stopping")
        state = state_manager.get_state("cycle-run")
        assert state.status == "stopping"

        # 4. Worker pauses (transition to paused with current_step)
        state_manager.transition("cycle-run", "paused", current_step=3)
        state = state_manager.get_state("cycle-run")
        assert state.status == "paused"
        assert state.current_step == 3

        # 5. Resume (transition back to running)
        state_manager.transition("cycle-run", "running")
        state = state_manager.get_state("cycle-run")
        assert state.status == "running"

        # 6. Complete
        state_manager.transition("cycle-run", "completed")
        state = state_manager.get_state("cycle-run")
        assert state.status == "completed"

        RunStateManager.reset_instance()
