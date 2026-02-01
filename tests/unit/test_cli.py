"""Unit tests for the APART CLI."""

import re
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from typer.testing import CliRunner

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from cli import app, get_api_url, DEFAULT_API_URL

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class TestCLIHelp:
    """Tests for CLI help commands."""

    def test_cli_help(self):
        """Test that main help displays correctly."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "APART" in result.output
        assert "Multi-agent simulation framework" in result.output

    def test_cli_run_help(self):
        """Test run command help."""
        result = runner.invoke(app, ["run", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "scenario" in output.lower()
        assert "--name" in output
        assert "--priority" in output

    def test_cli_pause_help(self):
        """Test pause command help."""
        result = runner.invoke(app, ["pause", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "run_id" in output.lower()
        assert "--force" in output

    def test_cli_resume_help(self):
        """Test resume command help."""
        result = runner.invoke(app, ["resume", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "run_id" in output.lower()
        assert "paused" in output.lower()

    def test_cli_list_help(self):
        """Test list command help."""
        result = runner.invoke(app, ["list", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--status" in output

    def test_cli_show_help(self):
        """Test show command help."""
        result = runner.invoke(app, ["show", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "run_id" in output.lower()

    def test_cli_delete_help(self):
        """Test delete command help."""
        result = runner.invoke(app, ["delete", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "--force" in output
        assert "--yes" in output

    def test_cli_status_help(self):
        """Test status command help."""
        result = runner.invoke(app, ["status", "--help"])
        output = strip_ansi(result.output)
        assert result.exit_code == 0
        assert "server" in output.lower()


class TestAPIURL:
    """Tests for API URL configuration."""

    def test_default_api_url(self):
        """Test that default API URL is used when env var is not set."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove APART_API_URL if present
            import os
            if "APART_API_URL" in os.environ:
                del os.environ["APART_API_URL"]
            url = get_api_url()
            assert url == DEFAULT_API_URL

    def test_custom_api_url(self):
        """Test that custom API URL from env var is used."""
        with patch.dict("os.environ", {"APART_API_URL": "http://custom:9000"}):
            url = get_api_url()
            assert url == "http://custom:9000"


class TestRunCommand:
    """Tests for the run command."""

    def test_run_missing_scenario(self):
        """Test run command fails without scenario argument."""
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage" in result.output

    def test_run_invalid_priority(self, tmp_path):
        """Test run command fails with invalid priority."""
        # Create a temp scenario file
        scenario_file = tmp_path / "test.yaml"
        scenario_file.write_text("name: test")

        with patch("requests.post"):
            result = runner.invoke(app, ["run", str(scenario_file), "--priority", "invalid"])
            assert result.exit_code == 1
            assert "Invalid priority" in result.output

    def test_run_success(self, tmp_path):
        """Test successful run command."""
        # Create a temp scenario file
        scenario_file = tmp_path / "test.yaml"
        scenario_file.write_text("name: test")

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "test123",
            "status": "pending",
            "message": "Simulation queued",
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = runner.invoke(app, ["run", str(scenario_file), "--name", "test123"])
            assert result.exit_code == 0
            assert "started successfully" in result.output
            assert "test123" in result.output

    def test_run_server_error(self, tmp_path):
        """Test run command handles server errors."""
        scenario_file = tmp_path / "test.yaml"
        scenario_file.write_text("name: test")

        from requests.exceptions import HTTPError

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"detail": "Scenario not found"}

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = HTTPError(response=mock_response)
            mock_post.return_value.json.return_value = {"detail": "Scenario not found"}
            result = runner.invoke(app, ["run", str(scenario_file)])
            assert result.exit_code == 1


class TestPauseCommand:
    """Tests for the pause command."""

    def test_pause_missing_run_id(self):
        """Test pause command fails without run_id argument."""
        result = runner.invoke(app, ["pause"])
        assert result.exit_code != 0

    def test_pause_success(self):
        """Test successful pause command."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "test123",
            "status": "stopping",
            "message": "Pause signal sent",
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            result = runner.invoke(app, ["pause", "test123"])
            assert result.exit_code == 0
            assert "Pause requested" in result.output
            assert "test123" in result.output

    def test_pause_with_force(self):
        """Test pause command with force flag."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "test123",
            "status": "stopping (force)",
            "message": "Force pause signal sent",
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response) as mock_post:
            result = runner.invoke(app, ["pause", "test123", "--force"])
            assert result.exit_code == 0
            # Verify force parameter was sent
            call_args = mock_post.call_args
            assert call_args[1]["params"] == {"force": "true"}

    def test_pause_not_found(self):
        """Test pause command handles 404 error."""
        from requests.exceptions import HTTPError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Run test123 not found"}

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = HTTPError(response=mock_response)
            mock_post.return_value.json.return_value = {"detail": "Run test123 not found"}
            result = runner.invoke(app, ["pause", "test123"])
            assert result.exit_code == 1
            assert "not found" in result.output.lower()


class TestResumeCommand:
    """Tests for the resume command."""

    def test_resume_missing_run_id(self):
        """Test resume command fails without run_id argument."""
        result = runner.invoke(app, ["resume"])
        assert result.exit_code != 0

    def test_resume_success(self):
        """Test successful resume command."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "run_id": "test123",
            "status": "resumed",
            "resuming_from_step": 5,
            "message": "Simulation resumed",
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.post", return_value=mock_response):
            result = runner.invoke(app, ["resume", "test123"])
            assert result.exit_code == 0
            assert "resumed" in result.output.lower()
            assert "step" in result.output.lower()

    def test_resume_not_paused(self):
        """Test resume command handles simulation not paused."""
        from requests.exceptions import HTTPError

        mock_response = Mock()
        mock_response.status_code = 409
        mock_response.json.return_value = {"detail": "Cannot resume simulation with status 'running'"}

        with patch("requests.post") as mock_post:
            mock_post.return_value.raise_for_status.side_effect = HTTPError(response=mock_response)
            mock_post.return_value.json.return_value = mock_response.json.return_value
            result = runner.invoke(app, ["resume", "test123"])
            assert result.exit_code == 1


class TestListCommand:
    """Tests for the list command."""

    def test_list_empty(self):
        """Test list command with no runs."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"runs": []}
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "No simulation runs found" in result.output

    def test_list_with_runs(self):
        """Test list command with runs."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "runs": [
                {
                    "runId": "run1",
                    "scenario": "test_scenario",
                    "status": "running",
                    "currentStep": 5,
                    "totalSteps": 10,
                    "dangerCount": 2,
                    "startedAt": "2024-01-01T00:00:00",
                },
                {
                    "runId": "run2",
                    "scenario": "another_scenario",
                    "status": "completed",
                    "currentStep": 10,
                    "totalSteps": 10,
                    "dangerCount": 0,
                    "startedAt": "2024-01-02T00:00:00",
                },
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            result = runner.invoke(app, ["list"])
            assert result.exit_code == 0
            assert "run1" in result.output
            assert "run2" in result.output
            assert "running" in result.output.lower()
            assert "completed" in result.output.lower()

    def test_list_filter_by_status(self):
        """Test list command with status filter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "runs": [
                {
                    "runId": "run1",
                    "scenario": "test",
                    "status": "running",
                    "currentStep": 5,
                    "totalSteps": 10,
                    "dangerCount": 0,
                    "startedAt": "2024-01-01T00:00:00",
                },
                {
                    "runId": "run2",
                    "scenario": "test2",
                    "status": "paused",
                    "currentStep": 3,
                    "totalSteps": 10,
                    "dangerCount": 0,
                    "startedAt": "2024-01-02T00:00:00",
                },
            ]
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            result = runner.invoke(app, ["list", "--status", "paused"])
            assert result.exit_code == 0
            assert "run2" in result.output
            # run1 should be filtered out (it's running, not paused)
            # Note: the CLI filters client-side after fetching all runs


class TestShowCommand:
    """Tests for the show command."""

    def test_show_missing_run_id(self):
        """Test show command fails without run_id argument."""
        result = runner.invoke(app, ["show"])
        assert result.exit_code != 0

    def test_show_success(self):
        """Test successful show command."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "runId": "test123",
            "scenario": "test_scenario",
            "status": "running",
            "currentStep": 5,
            "maxSteps": 10,
            "startedAt": "2024-01-01T00:00:00",
            "agentNames": ["Agent1", "Agent2"],
            "dangerSignals": [
                {
                    "step": 3,
                    "category": "power-seeking",
                    "agentName": "Agent1",
                    "metric": "resource_accumulation",
                }
            ],
            "messages": [
                {
                    "step": 1,
                    "agentName": "Agent1",
                    "direction": "received",
                    "content": "Hello",
                }
            ],
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.get", return_value=mock_response):
            result = runner.invoke(app, ["show", "test123"])
            assert result.exit_code == 0
            assert "test123" in result.output
            assert "running" in result.output.lower()
            assert "Agent1" in result.output
            assert "Danger" in result.output

    def test_show_not_found(self):
        """Test show command handles 404 error."""
        from requests.exceptions import HTTPError

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.json.return_value = {"detail": "Run not found"}

        with patch("requests.get") as mock_get:
            mock_get.return_value.raise_for_status.side_effect = HTTPError(response=mock_response)
            mock_get.return_value.status_code = 404
            result = runner.invoke(app, ["show", "nonexistent"])
            assert result.exit_code == 1
            assert "not found" in result.output.lower()


class TestDeleteCommand:
    """Tests for the delete command."""

    def test_delete_missing_run_id(self):
        """Test delete command fails without run_id argument."""
        result = runner.invoke(app, ["delete"])
        assert result.exit_code != 0

    def test_delete_cancelled(self):
        """Test delete command can be cancelled."""
        result = runner.invoke(app, ["delete", "test123"], input="n\n")
        assert result.exit_code == 0
        assert "Cancelled" in result.output

    def test_delete_success(self):
        """Test successful delete command."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "deleted",
            "run_id": "test123",
            "deleted_results": True,
            "deleted_events": True,
            "deleted_database": False,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.delete", return_value=mock_response):
            result = runner.invoke(app, ["delete", "test123", "--yes"])
            assert result.exit_code == 0
            assert "Deleted" in result.output
            assert "test123" in result.output

    def test_delete_with_force(self):
        """Test delete command with force flag."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "status": "deleted",
            "run_id": "test123",
            "deleted_results": True,
            "deleted_events": True,
            "deleted_database": False,
        }
        mock_response.raise_for_status = Mock()

        with patch("requests.delete", return_value=mock_response) as mock_delete:
            result = runner.invoke(app, ["delete", "test123", "--force", "--yes"])
            assert result.exit_code == 0
            # Verify force parameter was sent
            call_args = mock_delete.call_args
            assert call_args[1]["params"] == {"force": "true"}


class TestStatusCommand:
    """Tests for the status command."""

    def test_status_server_healthy(self):
        """Test status command with healthy server."""
        mock_health = Mock()
        mock_health.status_code = 200
        mock_health.json.return_value = {"status": "healthy"}
        mock_health.raise_for_status = Mock()

        mock_detailed = Mock()
        mock_detailed.status_code = 200
        mock_detailed.json.return_value = {
            "status": "healthy",
            "total_run_ids": 5,
            "event_bus_subscribers": 2,
            "persistence_mode": "jsonl",
            "queue_stats": {
                "queued": 1,
                "started": 2,
                "finished": 10,
                "failed": 0,
            },
        }
        mock_detailed.raise_for_status = Mock()

        with patch("requests.get", side_effect=[mock_health, mock_detailed]):
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 0
            assert "healthy" in result.output.lower()

    def test_status_server_unavailable(self):
        """Test status command when server is unavailable."""
        from requests.exceptions import ConnectionError

        with patch("requests.get", side_effect=ConnectionError("Connection refused")):
            result = runner.invoke(app, ["status"])
            assert result.exit_code == 1
            assert "Cannot connect" in result.output


class TestConnectionErrors:
    """Tests for connection error handling."""

    def test_connection_error_message(self, tmp_path):
        """Test that connection errors show helpful message."""
        from requests.exceptions import ConnectionError

        scenario_file = tmp_path / "test.yaml"
        scenario_file.write_text("name: test")

        with patch("requests.post", side_effect=ConnectionError("Connection refused")):
            result = runner.invoke(app, ["run", str(scenario_file)])
            assert result.exit_code == 1
            assert "server is running" in result.output.lower()
