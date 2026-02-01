"""Unit tests for Orchestrator, focusing on parallel agent execution."""

import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from concurrent.futures import ThreadPoolExecutor


class TestProcessSingleAgent:
    """Tests for _process_single_agent method."""

    def _create_mock_orchestrator(self):
        """Create a minimal mock orchestrator for testing."""
        from core.orchestrator import Orchestrator

        # We'll patch __init__ to avoid full initialization
        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.agents = []
            orch.logger = Mock()
            orch.game_engine = Mock()
            return orch

    def test_process_single_agent_success(self):
        """Test successful agent processing."""
        orch = self._create_mock_orchestrator()

        # Create mock agent
        agent = Mock()
        agent.name = "TestAgent"
        agent.respond.return_value = "Test response"

        result = orch._process_single_agent(
            agent=agent,
            step=1,
            message="Test message",
            agent_stats={"health": 100}
        )

        assert result["agent_name"] == "TestAgent"
        assert result["response"] == "Test response"
        assert result["error"] is None
        agent.update_stats.assert_called_once_with({"health": 100})
        agent.respond.assert_called_once_with("Test message")

    def test_process_single_agent_error(self):
        """Test agent processing with error."""
        orch = self._create_mock_orchestrator()

        agent = Mock()
        agent.name = "FailingAgent"
        agent.respond.side_effect = Exception("LLM timeout")

        result = orch._process_single_agent(
            agent=agent,
            step=1,
            message="Test message",
            agent_stats={}
        )

        assert result["agent_name"] == "FailingAgent"
        assert result["response"] is None
        assert result["error"] == "LLM timeout"

    def test_process_single_agent_preserves_message(self):
        """Test that original message is preserved in result."""
        orch = self._create_mock_orchestrator()

        agent = Mock()
        agent.name = "Agent"
        agent.respond.return_value = "Response"

        result = orch._process_single_agent(
            agent=agent,
            step=1,
            message="Original message",
            agent_stats={}
        )

        assert result["message"] == "Original message"


class TestCollectAgentResponsesParallel:
    """Tests for _collect_agent_responses with parallel execution."""

    def _create_orchestrator_with_agents(self, num_agents=3):
        """Create orchestrator with mock agents."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.game_engine.get_state_snapshot.return_value = {
                "agent_vars": {f"Agent{i}": {"score": i} for i in range(num_agents)}
            }

            # Create mock agents
            orch.agents = []
            for i in range(num_agents):
                agent = Mock()
                agent.name = f"Agent{i}"
                agent.respond.return_value = f"Response from Agent{i}"
                orch.agents.append(agent)

            return orch

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "1"})
    def test_parallel_execution_with_multiple_agents(self, mock_emit):
        """Test that parallel execution is used when enabled and multiple agents."""
        orch = self._create_orchestrator_with_agents(3)

        agent_messages = {
            "Agent0": "Message 0",
            "Agent1": "Message 1",
            "Agent2": "Message 2",
        }

        responses, step_messages = orch._collect_agent_responses(1, agent_messages)

        # All agents should have responded
        assert len(responses) == 3
        assert "Agent0" in responses
        assert "Agent1" in responses
        assert "Agent2" in responses

        # Check responses
        assert responses["Agent0"] == "Response from Agent0"
        assert responses["Agent1"] == "Response from Agent1"
        assert responses["Agent2"] == "Response from Agent2"

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "0"})
    def test_sequential_execution_when_disabled(self, mock_emit):
        """Test sequential execution when parallel is disabled."""
        orch = self._create_orchestrator_with_agents(3)

        agent_messages = {
            "Agent0": "Message 0",
            "Agent1": "Message 1",
            "Agent2": "Message 2",
        }

        responses, step_messages = orch._collect_agent_responses(1, agent_messages)

        assert len(responses) == 3
        # All agents should have been called
        for agent in orch.agents:
            agent.respond.assert_called_once()

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "1"})
    def test_single_agent_uses_sequential(self, mock_emit):
        """Test that single agent uses sequential execution even if parallel enabled."""
        orch = self._create_orchestrator_with_agents(1)

        agent_messages = {"Agent0": "Message 0"}

        # With only 1 agent, should use sequential path
        responses, step_messages = orch._collect_agent_responses(1, agent_messages)

        assert len(responses) == 1
        assert responses["Agent0"] == "Response from Agent0"

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "1"})
    def test_step_messages_order(self, mock_emit):
        """Test that step_messages maintains correct order."""
        orch = self._create_orchestrator_with_agents(2)

        agent_messages = {
            "Agent0": "Message 0",
            "Agent1": "Message 1",
        }

        responses, step_messages = orch._collect_agent_responses(1, agent_messages)

        # Should have 4 messages: 2 outgoing + 2 responses
        assert len(step_messages) == 4

        # First two should be from SimulatorAgent (outgoing)
        assert step_messages[0]["from"] == "SimulatorAgent"
        assert step_messages[0]["to"] == "Agent0"
        assert step_messages[1]["from"] == "SimulatorAgent"
        assert step_messages[1]["to"] == "Agent1"

        # Last two should be responses (in agent order)
        assert step_messages[2]["from"] == "Agent0"
        assert step_messages[2]["to"] == "SimulatorAgent"
        assert step_messages[3]["from"] == "Agent1"
        assert step_messages[3]["to"] == "SimulatorAgent"


class TestCollectAgentResponsesErrorHandling:
    """Tests for error handling in _collect_agent_responses."""

    def _create_orchestrator_with_failing_agent(self):
        """Create orchestrator with one failing agent."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.game_engine.get_state_snapshot.return_value = {
                "agent_vars": {"GoodAgent": {}, "BadAgent": {}}
            }

            good_agent = Mock()
            good_agent.name = "GoodAgent"
            good_agent.respond.return_value = "Good response"

            bad_agent = Mock()
            bad_agent.name = "BadAgent"
            bad_agent.respond.side_effect = Exception("Connection failed")

            orch.agents = [good_agent, bad_agent]
            return orch

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "1"})
    def test_partial_failure_parallel(self, mock_emit):
        """Test that one agent failure doesn't crash others in parallel mode."""
        orch = self._create_orchestrator_with_failing_agent()

        agent_messages = {
            "GoodAgent": "Hello",
            "BadAgent": "Hello",
        }

        responses, step_messages = orch._collect_agent_responses(1, agent_messages)

        # Both should have entries
        assert len(responses) == 2

        # Good agent should succeed
        assert responses["GoodAgent"] == "Good response"

        # Bad agent should have error message
        assert "ERROR:" in responses["BadAgent"]
        assert "Connection failed" in responses["BadAgent"]

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "0"})
    def test_partial_failure_sequential(self, mock_emit):
        """Test that one agent failure doesn't crash others in sequential mode."""
        orch = self._create_orchestrator_with_failing_agent()

        agent_messages = {
            "GoodAgent": "Hello",
            "BadAgent": "Hello",
        }

        responses, step_messages = orch._collect_agent_responses(1, agent_messages)

        assert len(responses) == 2
        assert responses["GoodAgent"] == "Good response"
        assert "ERROR:" in responses["BadAgent"]


class TestParallelAgentEnvironmentVariable:
    """Tests for APART_PARALLEL_AGENTS environment variable handling."""

    def _create_orchestrator_with_agents(self, num_agents=2):
        """Create orchestrator with mock agents."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.game_engine.get_state_snapshot.return_value = {
                "agent_vars": {f"Agent{i}": {} for i in range(num_agents)}
            }

            orch.agents = []
            for i in range(num_agents):
                agent = Mock()
                agent.name = f"Agent{i}"
                agent.respond.return_value = f"Response{i}"
                orch.agents.append(agent)

            return orch

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "true"})
    def test_env_var_true(self, mock_emit):
        """Test APART_PARALLEL_AGENTS=true enables parallel."""
        orch = self._create_orchestrator_with_agents(2)
        agent_messages = {"Agent0": "msg", "Agent1": "msg"}

        responses, _ = orch._collect_agent_responses(1, agent_messages)

        # Should work (parallel path)
        assert len(responses) == 2

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "yes"})
    def test_env_var_yes(self, mock_emit):
        """Test APART_PARALLEL_AGENTS=yes enables parallel."""
        orch = self._create_orchestrator_with_agents(2)
        agent_messages = {"Agent0": "msg", "Agent1": "msg"}

        responses, _ = orch._collect_agent_responses(1, agent_messages)

        assert len(responses) == 2

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "false"})
    def test_env_var_false(self, mock_emit):
        """Test APART_PARALLEL_AGENTS=false disables parallel."""
        orch = self._create_orchestrator_with_agents(2)
        agent_messages = {"Agent0": "msg", "Agent1": "msg"}

        responses, _ = orch._collect_agent_responses(1, agent_messages)

        assert len(responses) == 2

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "no"})
    def test_env_var_no(self, mock_emit):
        """Test APART_PARALLEL_AGENTS=no disables parallel."""
        orch = self._create_orchestrator_with_agents(2)
        agent_messages = {"Agent0": "msg", "Agent1": "msg"}

        responses, _ = orch._collect_agent_responses(1, agent_messages)

        assert len(responses) == 2

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {}, clear=True)
    def test_env_var_default(self, mock_emit):
        """Test default behavior when env var not set (parallel enabled)."""
        # Remove the env var if it exists
        os.environ.pop("APART_PARALLEL_AGENTS", None)

        orch = self._create_orchestrator_with_agents(2)
        agent_messages = {"Agent0": "msg", "Agent1": "msg"}

        responses, _ = orch._collect_agent_responses(1, agent_messages)

        # Default is parallel enabled
        assert len(responses) == 2


class TestThreadSafety:
    """Tests for thread safety in parallel execution."""

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "1"})
    def test_agents_called_with_correct_messages(self, mock_emit):
        """Test that each agent receives its correct message in parallel."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.game_engine.get_state_snapshot.return_value = {
                "agent_vars": {"A": {}, "B": {}, "C": {}}
            }

            received_messages = {}

            def make_respond(name):
                def respond(msg):
                    received_messages[name] = msg
                    return f"Response from {name}"
                return respond

            orch.agents = []
            for name in ["A", "B", "C"]:
                agent = Mock()
                agent.name = name
                agent.respond.side_effect = make_respond(name)
                orch.agents.append(agent)

            agent_messages = {
                "A": "Message for A",
                "B": "Message for B",
                "C": "Message for C",
            }

            responses, _ = orch._collect_agent_responses(1, agent_messages)

            # Verify each agent got the correct message
            assert received_messages["A"] == "Message for A"
            assert received_messages["B"] == "Message for B"
            assert received_messages["C"] == "Message for C"

    @patch('core.orchestrator.emit')
    @patch.dict(os.environ, {"APART_PARALLEL_AGENTS": "1"})
    def test_agents_receive_correct_stats(self, mock_emit):
        """Test that each agent receives its correct stats in parallel."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.game_engine.get_state_snapshot.return_value = {
                "agent_vars": {
                    "Agent1": {"health": 100, "gold": 50},
                    "Agent2": {"health": 80, "gold": 30},
                }
            }

            received_stats = {}

            orch.agents = []
            for name in ["Agent1", "Agent2"]:
                agent = Mock()
                agent.name = name
                agent.respond.return_value = f"Response"

                def capture_stats(stats, n=name):
                    received_stats[n] = stats.copy()
                agent.update_stats.side_effect = capture_stats

                orch.agents.append(agent)

            agent_messages = {"Agent1": "msg", "Agent2": "msg"}

            responses, _ = orch._collect_agent_responses(1, agent_messages)

            # Each agent should receive its own stats
            assert received_stats["Agent1"] == {"health": 100, "gold": 50}
            assert received_stats["Agent2"] == {"health": 80, "gold": 30}


class TestSimulationFailedEventEmission:
    """Tests for SIMULATION_FAILED event emission on errors."""

    def _create_mock_orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.agents = []
            orch.max_steps = 3
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.persistence = Mock()
            orch.persistence.run_id = "test-run-id"
            orch.spatial_graph = None
            orch.composed_modules = None
            return orch

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    def test_simulation_failed_emitted_on_exception(self, mock_disable, mock_enable, mock_emit):
        """Test that SIMULATION_FAILED is emitted when run() catches an exception."""
        from core.event_emitter import EventTypes

        orch = self._create_mock_orchestrator()

        # Make _initialize_simulation raise an exception
        orch._initialize_simulation = Mock(side_effect=RuntimeError("Test error"))

        with pytest.raises(RuntimeError, match="Test error"):
            orch.run()

        # Verify SIMULATION_FAILED was emitted
        fail_calls = [c for c in mock_emit.call_args_list
                      if c[0][0] == EventTypes.SIMULATION_FAILED]
        assert len(fail_calls) == 1
        assert fail_calls[0][1]["error"] == "Test error"

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    def test_simulation_failed_emitted_on_keyboard_interrupt(self, mock_disable, mock_enable, mock_emit):
        """Test that SIMULATION_FAILED is emitted on KeyboardInterrupt."""
        from core.event_emitter import EventTypes

        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = Mock(side_effect=KeyboardInterrupt())

        with pytest.raises(KeyboardInterrupt):
            orch.run()

        fail_calls = [c for c in mock_emit.call_args_list
                      if c[0][0] == EventTypes.SIMULATION_FAILED]
        assert len(fail_calls) == 1
        assert "interrupted" in fail_calls[0][1]["error"].lower()

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    @patch('core.orchestrator.clear_pause_signal')
    def test_simulation_failed_not_emitted_on_success(
        self, mock_clear, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that SIMULATION_FAILED is NOT emitted when simulation succeeds."""
        from core.event_emitter import EventTypes

        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = Mock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = Mock(return_value=({}, []))
        orch._process_step_results = Mock(return_value={"Agent1": "msg"})
        orch._save_final_state = Mock()

        # Mock: no pause signal
        mock_check.return_value = None

        orch.run()

        # Verify SIMULATION_FAILED was NOT emitted
        fail_calls = [c for c in mock_emit.call_args_list
                      if c[0][0] == EventTypes.SIMULATION_FAILED]
        assert len(fail_calls) == 0

        # But SIMULATION_STARTED and SIMULATION_COMPLETED should be emitted
        event_types = [c[0][0] for c in mock_emit.call_args_list]
        assert EventTypes.SIMULATION_STARTED in event_types


class TestPauseSignalHandling:
    """Tests for pause signal handling in orchestrator."""

    def _create_mock_orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.agents = []
            orch.max_steps = 3
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.persistence = Mock()
            orch.persistence.run_id = "test-run-id"
            orch.spatial_graph = None
            orch.composed_modules = None
            return orch

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    @patch('core.orchestrator.clear_pause_signal')
    def test_orchestrator_checks_pause_signal(
        self, mock_clear, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that orchestrator checks for pause signal between steps."""
        from core.event_emitter import EventTypes

        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = Mock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = Mock(return_value=({}, []))
        orch._process_step_results = Mock(return_value={"Agent1": "msg"})
        orch._save_final_state = Mock()

        # Mock pause check: None for init, None for step 1 start, None for step 1 after responses,
        # then pause at step 2 start
        mock_check.side_effect = [None, None, None, {"force": False}]

        # Run simulation - should pause at step 2
        orch.run()

        # Verify pause was checked
        assert mock_check.call_count >= 1
        mock_check.assert_called_with("test-run-id")

        # Verify pause signal was cleared
        mock_clear.assert_called_once_with("test-run-id")

        # Verify SIMULATION_PAUSED was emitted
        pause_calls = [c for c in mock_emit.call_args_list
                       if c[0][0] == EventTypes.SIMULATION_PAUSED]
        assert len(pause_calls) == 1
        assert pause_calls[0][1]["step"] == 2

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    @patch('core.orchestrator.clear_pause_signal')
    def test_orchestrator_completes_without_pause_signal(
        self, mock_clear, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that orchestrator completes normally when no pause signal."""
        from core.event_emitter import EventTypes

        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = Mock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = Mock(return_value=({}, []))
        orch._process_step_results = Mock(return_value={"Agent1": "msg"})
        orch._save_final_state = Mock()

        # Mock: no pause signal ever
        mock_check.return_value = None

        orch.run()

        # Pause checks: 1 after init + 2 per step (at start + after agent responses) = 1 + 3*2 = 7
        assert mock_check.call_count == 7  # 1 after init + 3 steps * 2 checks per step

        # Clear should not be called (no pause)
        mock_clear.assert_not_called()

        # SIMULATION_PAUSED should NOT be emitted
        pause_calls = [c for c in mock_emit.call_args_list
                       if c[0][0] == EventTypes.SIMULATION_PAUSED]
        assert len(pause_calls) == 0

        # Verify _save_final_state was called (which emits SIMULATION_COMPLETED)
        orch._save_final_state.assert_called_once()

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    @patch('core.orchestrator.clear_pause_signal')
    def test_orchestrator_handles_force_pause(
        self, mock_clear, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that orchestrator handles force pause correctly."""
        from core.event_emitter import EventTypes

        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = Mock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = Mock(return_value=({}, []))
        orch._process_step_results = Mock(return_value={"Agent1": "msg"})
        orch._save_final_state = Mock()

        # Force pause: None for init check, then force pause at step 1 start
        mock_check.side_effect = [None, {"force": True}]

        orch.run()

        # Should pause at step 1
        pause_calls = [c for c in mock_emit.call_args_list
                       if c[0][0] == EventTypes.SIMULATION_PAUSED]
        assert len(pause_calls) == 1
        assert pause_calls[0][1]["step"] == 1


class TestOrchestratorResumeSupport:
    """Tests for orchestrator resume functionality with start_step parameter."""

    def _create_mock_orchestrator(self):
        """Create orchestrator with mocked dependencies."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.agents = []
            orch.max_steps = 5
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.persistence = Mock()
            orch.persistence.run_id = "test-run-id"
            orch.persistence.run_dir = Mock()
            orch.spatial_graph = None
            orch.composed_modules = None
            orch.simulator_agent = Mock()
            return orch

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    def test_orchestrator_run_with_start_step(
        self, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that orchestrator can start from a specific step."""
        orch = self._create_mock_orchestrator()
        orch._restore_state_for_resume = Mock()
        orch.simulator_agent.initialize_simulation = Mock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = Mock(return_value=({}, []))
        orch._process_step_results = Mock(return_value={"Agent1": "msg"})
        orch._save_final_state = Mock()
        mock_check.return_value = None

        # Run starting from step 3
        orch.run(start_step=3)

        # Verify _restore_state_for_resume was called
        orch._restore_state_for_resume.assert_called_once_with(3)

        # Verify _initialize_simulation was NOT called (resume path)
        # Instead, simulator_agent.initialize_simulation should be called
        orch.simulator_agent.initialize_simulation.assert_called_once()

        # Verify we started from step 3 (should run steps 3, 4, 5)
        assert orch._collect_agent_responses.call_count == 3

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    def test_orchestrator_run_default_start_step(
        self, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that orchestrator defaults to start_step=1."""
        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = Mock(return_value={"Agent1": "msg"})
        orch._collect_agent_responses = Mock(return_value=({}, []))
        orch._process_step_results = Mock(return_value={"Agent1": "msg"})
        orch._save_final_state = Mock()
        mock_check.return_value = None

        # Run without start_step (default)
        orch.run()

        # Verify _initialize_simulation was called (not resume)
        orch._initialize_simulation.assert_called_once()

        # Verify all 5 steps were run
        assert orch._collect_agent_responses.call_count == 5

    @patch('core.orchestrator.emit')
    @patch('core.orchestrator.enable_event_emitter')
    @patch('core.orchestrator.disable_event_emitter')
    @patch('core.orchestrator.check_pause_requested')
    def test_orchestrator_run_start_step_1_is_not_resume(
        self, mock_check, mock_disable, mock_enable, mock_emit
    ):
        """Test that start_step=1 explicitly is treated as normal start, not resume."""
        orch = self._create_mock_orchestrator()
        orch._initialize_simulation = Mock(return_value={"Agent1": "msg"})
        orch._restore_state_for_resume = Mock()
        orch._collect_agent_responses = Mock(return_value=({}, []))
        orch._process_step_results = Mock(return_value={"Agent1": "msg"})
        orch._save_final_state = Mock()
        mock_check.return_value = None

        # Run with explicit start_step=1
        orch.run(start_step=1)

        # Verify _initialize_simulation was called (normal start)
        orch._initialize_simulation.assert_called_once()

        # Verify _restore_state_for_resume was NOT called
        orch._restore_state_for_resume.assert_not_called()

    def test_restore_state_for_resume_with_valid_snapshot(self, tmp_path):
        """Test that _restore_state_for_resume restores state correctly."""
        import json
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.agents = []

            # Create mock agent
            mock_agent = Mock()
            mock_agent.name = "TestAgent"
            orch.agents = [mock_agent]

            # Set up persistence with tmp_path
            orch.persistence = Mock()
            orch.persistence.run_dir = tmp_path

            # Create state file with snapshots
            state_data = {
                "run_id": "test-run",
                "scenario": "test",
                "snapshots": [
                    {
                        "step": 1,
                        "global_vars": {"resource": 100},
                        "agent_vars": {"TestAgent": {"health": 90}},
                        "messages": []
                    },
                    {
                        "step": 2,
                        "global_vars": {"resource": 80},
                        "agent_vars": {"TestAgent": {"health": 75}},
                        "messages": []
                    }
                ]
            }
            state_file = tmp_path / "state.json"
            with open(state_file, "w") as f:
                json.dump(state_data, f)

            # Call restore for step 3 (should use snapshot from step 2)
            result = orch._restore_state_for_resume(3)

            # Verify game engine state was updated
            orch.game_engine.apply_state_updates.assert_called_once()
            call_args = orch.game_engine.apply_state_updates.call_args[0][0]
            assert call_args["global_vars"] == {"resource": 80}
            assert call_args["agent_vars"] == {"TestAgent": {"health": 75}}

            # Verify round was set
            assert orch.game_engine.state.round == 2

            # Verify agent stats were updated
            mock_agent.update_stats.assert_called_once_with({"health": 75})

    def test_restore_state_for_resume_no_state_file(self, tmp_path):
        """Test _restore_state_for_resume handles missing state file."""
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.agents = []

            orch.persistence = Mock()
            orch.persistence.run_dir = tmp_path

            # No state file exists
            result = orch._restore_state_for_resume(3)

            # Should return empty dict and log warning
            assert result == {}
            orch.logger.warning.assert_called()

    def test_restore_state_for_resume_empty_snapshots(self, tmp_path):
        """Test _restore_state_for_resume handles empty snapshots."""
        import json
        from core.orchestrator import Orchestrator

        with patch.object(Orchestrator, '__init__', lambda self: None):
            orch = Orchestrator()
            orch.logger = Mock()
            orch.game_engine = Mock()
            orch.agents = []

            orch.persistence = Mock()
            orch.persistence.run_dir = tmp_path

            # Create state file with no snapshots
            state_data = {
                "run_id": "test-run",
                "scenario": "test",
                "snapshots": []
            }
            state_file = tmp_path / "state.json"
            with open(state_file, "w") as f:
                json.dump(state_data, f)

            result = orch._restore_state_for_resume(3)

            assert result == {}
            orch.logger.warning.assert_called()
