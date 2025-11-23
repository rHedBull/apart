"""Unit tests for HTML report generator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import pytest
import tempfile
import shutil
import json
from html_report_generator import HTMLReportGenerator, generate_detailed_run_page, generate_html_report


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_run_result():
    """Create sample benchmark run result."""
    return {
        "model_name": "test_model",
        "provider": "test_provider",
        "start_time": "2025-11-23T10:00:00",
        "end_time": "2025-11-23T10:05:00",
        "total_time": 300.0,
        "step_times": [100.0, 100.0, 100.0],
        "avg_step_time": 100.0,
        "total_tokens": None,
        "avg_tokens_per_step": None,
        "variable_changes": [
            {
                "step": 1,
                "global_vars": {"round": 1},
                "agent_vars": {
                    "Agent A": {"score": 110},
                    "Agent B": {"score": 105}
                }
            },
            {
                "step": 2,
                "global_vars": {"round": 2},
                "agent_vars": {
                    "Agent A": {"score": 120},
                    "Agent B": {"score": 110}
                }
            }
        ],
        "final_state": {
            "game_state": {"round": 2},
            "global_vars": {"round": 2},
            "agent_vars": {
                "Agent A": {"score": 120},
                "Agent B": {"score": 110}
            }
        },
        "decision_count": 4,
        "constraint_violations": 0,
        "completed": True,
        "error_count": 0,
        "step_failures": [],
        "error_messages": [],
        "custom_metrics": {},
        "conversation": [
            {
                "step": 1,
                "exchanges": [
                    {
                        "agent": "Agent A",
                        "message_to_agent": "What is your move?",
                        "response_from_agent": "I choose action X"
                    },
                    {
                        "agent": "Agent B",
                        "message_to_agent": "What is your move?",
                        "response_from_agent": "I choose action Y"
                    }
                ]
            }
        ]
    }


@pytest.fixture
def sample_benchmark_config():
    """Create sample benchmark configuration."""
    return {
        "name": "test_benchmark",
        "description": "Test benchmark description",
        "model_pool": {
            "test_model": {
                "provider": "test",
                "model": "test-1"
            }
        },
        "benchmark_runs": [
            {
                "name": "test_run",
                "description": "Test run description",
                "agent_model_mapping": {
                    "Agent A": "test_model",
                    "Agent B": "test_model"
                }
            }
        ]
    }


@pytest.fixture
def sample_scenario_config():
    """Create sample scenario configuration."""
    return {
        "max_steps": 3,
        "orchestrator_message": "What is your decision?",
        "global_vars": {
            "round": {"type": "int", "default": 0, "description": "Round number"}
        },
        "agents": [
            {
                "name": "Agent A",
                "system_prompt": "You are agent A",
                "variables": {"score": 100}
            },
            {
                "name": "Agent B",
                "system_prompt": "You are agent B",
                "variables": {"score": 100}
            }
        ]
    }


class TestHTMLReportGenerator:
    """Tests for HTMLReportGenerator class."""

    def test_initialization(self, sample_run_result):
        """Test report generator initialization."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")

        assert generator.benchmark_name == "test_benchmark"
        assert len(generator.results) == 1
        assert generator.results[0]["model_name"] == "test_model"

    def test_generate_creates_file(self, temp_output_dir, sample_run_result):
        """Test that generate creates HTML file."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")
        output_path = temp_output_dir / "report.html"

        generator.generate(output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_html_structure(self, temp_output_dir, sample_run_result):
        """Test basic HTML structure."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")
        output_path = temp_output_dir / "report.html"
        generator.generate(output_path)

        with open(output_path) as f:
            content = f.read()

        # Check basic HTML structure
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "<head>" in content
        assert "<body>" in content
        assert "</html>" in content

    def test_includes_summary_section(self, temp_output_dir, sample_run_result):
        """Test that summary section is included."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")
        output_path = temp_output_dir / "report.html"
        generator.generate(output_path)

        with open(output_path) as f:
            content = f.read()

        assert "Summary" in content or "SUMMARY" in content
        assert "Total Runs" in content
        assert "Completed" in content

    def test_includes_performance_charts(self, temp_output_dir, sample_run_result):
        """Test that performance charts are included."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")
        output_path = temp_output_dir / "report.html"
        generator.generate(output_path)

        with open(output_path) as f:
            content = f.read()

        assert "performanceChart" in content
        assert "errorChart" in content
        assert "Chart.js" in content or "chart.js" in content

    def test_includes_details_table(self, temp_output_dir, sample_run_result):
        """Test that details table is included."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")
        output_path = temp_output_dir / "report.html"
        generator.generate(output_path)

        with open(output_path) as f:
            content = f.read()

        assert "test_model" in content
        assert "test_provider" in content
        assert "Completed" in content or "completed" in content

    def test_includes_scenario_configuration(self, temp_output_dir, sample_run_result,
                                            sample_benchmark_config, sample_scenario_config):
        """Test that scenario configuration is included."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark",
                                       sample_benchmark_config, sample_scenario_config)
        output_path = temp_output_dir / "report.html"
        generator.generate(output_path)

        with open(output_path) as f:
            content = f.read()

        assert "Scenario Configuration" in content
        assert "Agent A" in content
        assert "Agent B" in content

    def test_completion_status_colors(self, temp_output_dir, sample_run_result):
        """Test that status colors are applied."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")
        output_path = temp_output_dir / "report.html"
        generator.generate(output_path)

        with open(output_path) as f:
            content = f.read()

        # Check for green completed color
        assert "#22c55e" in content
        # Check for red failed color
        assert "#ef4444" in content

    def test_detail_page_links(self, temp_output_dir, sample_run_result):
        """Test that links to detail pages are included."""
        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")
        output_path = temp_output_dir / "report.html"
        generator.generate(output_path)

        with open(output_path) as f:
            content = f.read()

        assert "test_model_detail.html" in content
        assert "detail-link" in content


class TestDetailedRunPage:
    """Tests for detailed run page generation."""

    def test_generate_detailed_page(self, temp_output_dir, sample_run_result):
        """Test detailed page generation."""
        output_path = temp_output_dir / "detail.html"

        generate_detailed_run_page(sample_run_result, output_path, "test_benchmark")

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_includes_conversation_transcript(self, temp_output_dir, sample_run_result):
        """Test that conversation transcript is included."""
        output_path = temp_output_dir / "detail.html"
        generate_detailed_run_page(sample_run_result, output_path, "test_benchmark")

        with open(output_path) as f:
            content = f.read()

        assert "Conversation Transcript" in content
        assert "Orchestrator" in content
        assert "Agent A" in content
        assert "What is your move?" in content
        assert "I choose action X" in content

    def test_includes_variable_evolution_chart(self, temp_output_dir, sample_run_result):
        """Test that variable evolution chart is included."""
        output_path = temp_output_dir / "detail.html"
        generate_detailed_run_page(sample_run_result, output_path, "test_benchmark")

        with open(output_path) as f:
            content = f.read()

        assert "Variable Evolution" in content
        assert "variableChart" in content

    def test_includes_back_link(self, temp_output_dir, sample_run_result):
        """Test that back link is included."""
        output_path = temp_output_dir / "detail.html"
        generate_detailed_run_page(sample_run_result, output_path, "test_benchmark")

        with open(output_path) as f:
            content = f.read()

        assert "Back to Overview" in content
        assert "test_benchmark.html" in content

    def test_conversation_message_styling(self, temp_output_dir, sample_run_result):
        """Test that conversation messages have proper styling."""
        output_path = temp_output_dir / "detail.html"
        generate_detailed_run_page(sample_run_result, output_path, "test_benchmark")

        with open(output_path) as f:
            content = f.read()

        # Check for message styling classes
        assert "message-block" in content
        assert "message orchestrator" in content
        assert "message agent" in content
        assert "message-header" in content
        assert "message-content" in content


class TestFullReportGeneration:
    """Tests for full report generation with multiple pages."""

    def test_generates_all_files(self, temp_output_dir, sample_run_result):
        """Test that all report files are generated."""
        json_path = temp_output_dir / "results.json"
        html_path = temp_output_dir / "report.html"

        # Write JSON data
        with open(json_path, "w") as f:
            json.dump([sample_run_result], f)

        # Generate reports
        generate_html_report(json_path, html_path, "test_benchmark")

        # Check main report exists
        assert html_path.exists()

        # Check detailed page exists
        detail_path = temp_output_dir / "test_model_detail.html"
        assert detail_path.exists()

    def test_multiple_runs(self, temp_output_dir, sample_run_result):
        """Test report generation with multiple runs."""
        # Create second run result
        run2 = sample_run_result.copy()
        run2["model_name"] = "test_model_2"

        json_path = temp_output_dir / "results.json"
        html_path = temp_output_dir / "report.html"

        with open(json_path, "w") as f:
            json.dump([sample_run_result, run2], f)

        generate_html_report(json_path, html_path, "test_benchmark")

        # Check both detail pages exist
        assert (temp_output_dir / "test_model_detail.html").exists()
        assert (temp_output_dir / "test_model_2_detail.html").exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_conversation(self, temp_output_dir):
        """Test handling of empty conversation data."""
        result = {
            "model_name": "test",
            "provider": "test",
            "start_time": "2025-01-01T00:00:00",
            "end_time": "2025-01-01T00:05:00",
            "total_time": 300.0,
            "step_times": [100.0],
            "avg_step_time": 100.0,
            "variable_changes": [],
            "completed": True,
            "error_count": 0,
            "conversation": [],  # Empty conversation
            "final_state": {},
            "decision_count": 0,
            "constraint_violations": 0,
            "step_failures": [],
            "error_messages": [],
            "custom_metrics": {}
        }

        output_path = temp_output_dir / "detail.html"
        generate_detailed_run_page(result, output_path, "test_benchmark")

        with open(output_path) as f:
            content = f.read()

        # Should show message about no data
        assert "No conversation data available" in content or "conversation" in content.lower()

    def test_failed_run(self, temp_output_dir, sample_run_result):
        """Test handling of failed run."""
        sample_run_result["completed"] = False
        sample_run_result["error_count"] = 5

        generator = HTMLReportGenerator([sample_run_result], "test_benchmark")
        output_path = temp_output_dir / "report.html"
        generator.generate(output_path)

        with open(output_path) as f:
            content = f.read()

        # Check for failed status
        assert "status-failed" in content
        assert "Failed" in content or "failed" in content
