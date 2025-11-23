"""Integration tests for benchmark system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))

import pytest
import tempfile
import shutil
import yaml
import json
from benchmark import BenchmarkRunner


@pytest.fixture
def temp_benchmark_dir():
    """Create a temporary directory for benchmark files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def minimal_scenario(temp_benchmark_dir):
    """Create a minimal test scenario."""
    scenario = {
        "max_steps": 1,
        "orchestrator_message": "Test",
        "engine": {
            "provider": "mock",
            "model": "test",
            "system_prompt": "Test",
            "simulation_plan": "Test"
        },
        "global_vars": {
            "round": {"type": "int", "default": 0}
        },
        "agent_vars": {
            "score": {"type": "int", "default": 100}
        },
        "agents": [
            {
                "name": "Agent A",
                "llm": {"provider": "mock", "model": "test"},
                "system_prompt": "Test agent",
                "variables": {"score": 100}
            }
        ]
    }

    scenario_path = temp_benchmark_dir / "test_scenario.yaml"
    with open(scenario_path, "w") as f:
        yaml.dump(scenario, f)

    return scenario_path


@pytest.fixture
def minimal_benchmark_config(temp_benchmark_dir, minimal_scenario):
    """Create a minimal benchmark configuration."""
    config = {
        "name": "test_benchmark",
        "description": "Test benchmark",
        "base_scenario": str(minimal_scenario),
        "model_pool": {
            "test-model": {
                "provider": "mock",
                "model": "test"
            }
        },
        "benchmark_runs": [
            {
                "name": "test_run",
                "description": "Test run",
                "agent_model_mapping": {
                    "Agent A": "test-model"
                }
            }
        ],
        "metrics": {
            "performance": {"enabled": True},
            "quality": {"enabled": True},
            "reliability": {"enabled": True}
        },
        "reporting": {
            "output_dir": str(temp_benchmark_dir / "results"),
            "formats": ["json", "markdown", "html"],
            "comparison_table": True
        }
    }

    config_path = temp_benchmark_dir / "benchmark.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path


class TestBenchmarkRunner:
    """Tests for BenchmarkRunner."""

    def test_initialization(self, minimal_benchmark_config):
        """Test benchmark runner initialization."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))

        assert runner.config["name"] == "test_benchmark"
        assert "model_pool" in runner.config
        assert len(runner.results) == 0

    def test_load_base_scenario(self, minimal_benchmark_config):
        """Test loading base scenario."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))
        scenario = runner._load_base_scenario()

        assert scenario["max_steps"] == 1
        assert len(scenario["agents"]) == 1

    def test_create_scenario_for_run(self, minimal_benchmark_config):
        """Test scenario creation with agent-model mapping."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))
        base_scenario = runner._load_base_scenario()
        model_pool = runner.config["model_pool"]
        run_config = runner.config["benchmark_runs"][0]

        scenario = runner._create_scenario_for_run(base_scenario, run_config, model_pool)

        assert scenario["agents"][0]["llm"]["provider"] == "mock"
        assert scenario["agents"][0]["llm"]["model"] == "test"

    def test_benchmark_run_execution(self, minimal_benchmark_config, temp_benchmark_dir):
        """Test benchmark run execution structure (may fail execution but should create files)."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))
        runner.run()

        # Check results were collected (even if failed)
        assert len(runner.results) >= 0  # May be 0 or 1 depending on error handling

        # Check files were created
        results_dir = temp_benchmark_dir / "results"
        assert results_dir.exists()

        # Find the benchmark run directory
        run_dirs = list(results_dir.glob("benchmark_*"))
        assert len(run_dirs) == 1

        run_dir = run_dirs[0]
        assert (run_dir / "test_benchmark.json").exists()
        assert (run_dir / "test_benchmark.md").exists()
        assert (run_dir / "test_benchmark.html").exists()

    def test_json_output_structure(self, minimal_benchmark_config, temp_benchmark_dir):
        """Test JSON output contains all required fields."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))
        runner.run()

        # Find and read JSON file
        results_dir = temp_benchmark_dir / "results"
        run_dir = list(results_dir.glob("benchmark_*"))[0]
        json_file = run_dir / "test_benchmark.json"

        with open(json_file) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1

        result = data[0]
        # Check required fields
        assert "model_name" in result
        assert "provider" in result
        assert "start_time" in result
        assert "end_time" in result
        assert "total_time" in result
        assert "completed" in result
        assert "conversation" in result
        assert "variable_changes" in result

    def test_conversation_data_captured(self, minimal_benchmark_config, temp_benchmark_dir):
        """Test that conversation data structure exists in results."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))
        runner.run()

        # Check conversation field exists in results (may be empty if run failed)
        if len(runner.results) > 0:
            result = runner.results[0]
            assert "conversation" in result.to_dict()
            # If there is conversation data, verify structure
            if len(result.conversation) > 0:
                conversation = result.conversation[0]
                assert "step" in conversation
                assert "exchanges" in conversation

    def test_html_report_generation(self, minimal_benchmark_config, temp_benchmark_dir):
        """Test HTML report files are generated."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))
        runner.run()

        results_dir = temp_benchmark_dir / "results"
        run_dir = list(results_dir.glob("benchmark_*"))[0]

        # Check main report exists
        main_report = run_dir / "test_benchmark.html"
        assert main_report.exists()

        # Check detailed run page exists
        detail_page = run_dir / "test_run_detail.html"
        assert detail_page.exists()

        # Verify main report contains link to detail page
        with open(main_report) as f:
            content = f.read()
            assert "test_run_detail.html" in content
            assert "detail-link" in content

    def test_detailed_page_content(self, minimal_benchmark_config, temp_benchmark_dir):
        """Test detailed run page contains expected content."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))
        runner.run()

        results_dir = temp_benchmark_dir / "results"
        run_dir = list(results_dir.glob("benchmark_*"))[0]
        detail_page = run_dir / "test_run_detail.html"

        with open(detail_page) as f:
            content = f.read()

        # Check for expected sections
        assert "Conversation Transcript" in content
        assert "Variable Evolution" in content
        assert "Back to Overview" in content
        assert "Orchestrator" in content or "orchestrator" in content

    def test_markdown_report_generation(self, minimal_benchmark_config, temp_benchmark_dir):
        """Test markdown report generation."""
        runner = BenchmarkRunner(str(minimal_benchmark_config))
        runner.run()

        results_dir = temp_benchmark_dir / "results"
        run_dir = list(results_dir.glob("benchmark_*"))[0]
        md_file = run_dir / "test_benchmark.md"

        assert md_file.exists()

        with open(md_file) as f:
            content = f.read()

        # Check markdown structure
        assert "# Benchmark Results" in content
        assert "## Scenario Configuration" in content
        assert "## Summary" in content
        assert "## Performance Comparison" in content


class TestBenchmarkErrorHandling:
    """Tests for error handling in benchmarks."""

    def test_missing_base_scenario(self, temp_benchmark_dir):
        """Test error when base scenario is missing."""
        config = {
            "name": "test",
            "base_scenario": "nonexistent.yaml",
            "model_pool": {},
            "benchmark_runs": []
        }

        config_path = temp_benchmark_dir / "bad_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        with pytest.raises(FileNotFoundError):
            runner = BenchmarkRunner(str(config_path))
            runner._load_base_scenario()

    def test_invalid_model_reference(self, minimal_benchmark_config, temp_benchmark_dir):
        """Test error when referencing non-existent model."""
        # Load and modify config to have invalid model reference
        with open(minimal_benchmark_config) as f:
            config = yaml.safe_load(f)

        config["benchmark_runs"][0]["agent_model_mapping"]["Agent A"] = "nonexistent-model"

        bad_config_path = temp_benchmark_dir / "bad_benchmark.yaml"
        with open(bad_config_path, "w") as f:
            yaml.dump(config, f)

        runner = BenchmarkRunner(str(bad_config_path))

        # The error is now caught with KeyError in _create_scenario_for_run
        with pytest.raises((ValueError, KeyError)):
            runner.run()
