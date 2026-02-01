"""Unit tests for API models (Pydantic schemas)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from pydantic import ValidationError


class TestSimulationStatus:
    """Tests for SimulationStatus enum."""

    def test_all_status_values(self):
        """Test all expected status values exist."""
        from server.models import SimulationStatus

        assert SimulationStatus.PENDING == "pending"
        assert SimulationStatus.RUNNING == "running"
        assert SimulationStatus.PAUSED == "paused"
        assert SimulationStatus.COMPLETED == "completed"
        assert SimulationStatus.FAILED == "failed"
        assert SimulationStatus.STOPPED == "stopped"

    def test_status_is_string_enum(self):
        """Test that status values are strings."""
        from server.models import SimulationStatus

        assert isinstance(SimulationStatus.PENDING.value, str)
        # Can be used as string via .value
        assert SimulationStatus.RUNNING.value == "running"

    def test_simulation_status_includes_paused(self):
        """Test that SimulationStatus enum includes PAUSED."""
        from server.models import SimulationStatus

        assert hasattr(SimulationStatus, "PAUSED")
        assert SimulationStatus.PAUSED.value == "paused"


class TestAgentInfo:
    """Tests for AgentInfo model."""

    def test_minimal_agent_info(self):
        """Test creating AgentInfo with only required fields."""
        from server.models import AgentInfo

        agent = AgentInfo(name="TestAgent")
        assert agent.name == "TestAgent"
        assert agent.llm_provider is None
        assert agent.llm_model is None

    def test_full_agent_info(self):
        """Test creating AgentInfo with all fields."""
        from server.models import AgentInfo

        agent = AgentInfo(
            name="FullAgent",
            llm_provider="openai",
            llm_model="gpt-4"
        )
        assert agent.name == "FullAgent"
        assert agent.llm_provider == "openai"
        assert agent.llm_model == "gpt-4"

    def test_agent_info_serialization(self):
        """Test AgentInfo JSON serialization."""
        from server.models import AgentInfo

        agent = AgentInfo(name="Test", llm_provider="anthropic")
        data = agent.model_dump()

        assert data["name"] == "Test"
        assert data["llm_provider"] == "anthropic"
        assert data["llm_model"] is None


class TestSimulationSummary:
    """Tests for SimulationSummary model."""

    def test_minimal_summary(self):
        """Test creating SimulationSummary with minimal fields."""
        from server.models import SimulationSummary, SimulationStatus

        summary = SimulationSummary(
            run_id="test-run",
            status=SimulationStatus.PENDING
        )
        assert summary.run_id == "test-run"
        assert summary.status == SimulationStatus.PENDING
        assert summary.current_step == 0
        assert summary.max_steps is None
        assert summary.agent_count == 0

    def test_full_summary(self):
        """Test creating SimulationSummary with all fields."""
        from server.models import SimulationSummary, SimulationStatus

        summary = SimulationSummary(
            run_id="full-run",
            status=SimulationStatus.RUNNING,
            scenario_name="Test Scenario",
            started_at="2024-01-01T00:00:00",
            completed_at=None,
            current_step=5,
            max_steps=10,
            agent_count=3
        )
        assert summary.run_id == "full-run"
        assert summary.scenario_name == "Test Scenario"
        assert summary.current_step == 5
        assert summary.max_steps == 10

    def test_summary_from_string_status(self):
        """Test creating summary with string status (enum coercion)."""
        from server.models import SimulationSummary, SimulationStatus

        summary = SimulationSummary(
            run_id="test",
            status="running"
        )
        assert summary.status == SimulationStatus.RUNNING


class TestSimulationDetails:
    """Tests for SimulationDetails model."""

    def test_details_inherits_summary(self):
        """Test that SimulationDetails has all SimulationSummary fields."""
        from server.models import SimulationDetails, SimulationStatus

        details = SimulationDetails(
            run_id="test",
            status=SimulationStatus.COMPLETED,
            current_step=10,
            max_steps=10
        )
        # Inherited fields
        assert details.run_id == "test"
        assert details.status == SimulationStatus.COMPLETED
        assert details.current_step == 10

        # Extended fields
        assert details.agents == []
        assert details.error_message is None

    def test_details_with_agents(self):
        """Test SimulationDetails with agent list."""
        from server.models import SimulationDetails, SimulationStatus, AgentInfo

        details = SimulationDetails(
            run_id="agent-test",
            status=SimulationStatus.RUNNING,
            agents=[
                AgentInfo(name="Agent1", llm_provider="openai"),
                AgentInfo(name="Agent2", llm_provider="anthropic"),
            ]
        )
        assert len(details.agents) == 2
        assert details.agents[0].name == "Agent1"

    def test_details_with_error(self):
        """Test SimulationDetails with error message."""
        from server.models import SimulationDetails, SimulationStatus

        details = SimulationDetails(
            run_id="error-test",
            status=SimulationStatus.FAILED,
            error_message="Simulation crashed due to timeout"
        )
        assert details.status == SimulationStatus.FAILED
        assert "timeout" in details.error_message


class TestJobPriority:
    """Tests for JobPriority enum."""

    def test_priority_values(self):
        """Test all priority values."""
        from server.models import JobPriority

        assert JobPriority.HIGH == "high"
        assert JobPriority.NORMAL == "normal"
        assert JobPriority.LOW == "low"


class TestStartSimulationRequest:
    """Tests for StartSimulationRequest model."""

    def test_minimal_request(self):
        """Test creating request with only required fields."""
        from server.models import StartSimulationRequest, JobPriority

        request = StartSimulationRequest(scenario_path="/path/to/scenario.yaml")
        assert request.scenario_path == "/path/to/scenario.yaml"
        assert request.run_id is None
        assert request.priority == JobPriority.NORMAL

    def test_full_request(self):
        """Test creating request with all fields."""
        from server.models import StartSimulationRequest, JobPriority

        request = StartSimulationRequest(
            scenario_path="/path/to/scenario.yaml",
            run_id="custom-id",
            priority=JobPriority.HIGH
        )
        assert request.run_id == "custom-id"
        assert request.priority == JobPriority.HIGH

    def test_request_validation_missing_path(self):
        """Test that scenario_path is required."""
        from server.models import StartSimulationRequest

        with pytest.raises(ValidationError):
            StartSimulationRequest()


class TestStartSimulationResponse:
    """Tests for StartSimulationResponse model."""

    def test_response_creation(self):
        """Test creating response."""
        from server.models import StartSimulationResponse, SimulationStatus

        response = StartSimulationResponse(
            run_id="new-run",
            status=SimulationStatus.PENDING,
            message="Simulation queued"
        )
        assert response.run_id == "new-run"
        assert response.status == SimulationStatus.PENDING
        assert "queued" in response.message


class TestDangerSignal:
    """Tests for DangerSignal model."""

    def test_danger_signal_creation(self):
        """Test creating DangerSignal."""
        from server.models import DangerSignal

        signal = DangerSignal(
            category="deception",
            description="Agent provided misleading information",
            confidence=0.85,
            step=7,
            agent_name="SuspectAgent",
            timestamp="2024-01-01T00:05:00"
        )
        assert signal.category == "deception"
        assert signal.confidence == 0.85
        assert signal.agent_name == "SuspectAgent"

    def test_danger_signal_without_agent(self):
        """Test DangerSignal without agent_name."""
        from server.models import DangerSignal

        signal = DangerSignal(
            category="power-seeking",
            description="System-wide anomaly detected",
            confidence=0.6,
            step=10,
            timestamp="2024-01-01T00:10:00"
        )
        assert signal.agent_name is None


class TestDangerSummary:
    """Tests for DangerSummary model."""

    def test_danger_summary_creation(self):
        """Test creating DangerSummary."""
        from server.models import DangerSummary, DangerSignal

        summary = DangerSummary(
            run_id="danger-test",
            total_signals=3,
            by_category={"deception": 2, "power-seeking": 1},
            signals=[
                DangerSignal(
                    category="deception",
                    description="Test signal",
                    confidence=0.9,
                    step=1,
                    timestamp="2024-01-01T00:00:00"
                )
            ]
        )
        assert summary.total_signals == 3
        assert summary.by_category["deception"] == 2
        assert len(summary.signals) == 1

    def test_danger_summary_defaults(self):
        """Test DangerSummary default values."""
        from server.models import DangerSummary

        summary = DangerSummary(run_id="test", total_signals=0)
        assert summary.by_category == {}
        assert summary.signals == []
