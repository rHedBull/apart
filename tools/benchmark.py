"""
Benchmark runner for comparing multiple LLM models on the same scenario.

Usage:
    python tools/benchmark.py benchmarks/example_benchmark.yaml
"""

import argparse
import sys
import yaml
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from copy import deepcopy

# Add src and tools directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from core.orchestrator import Orchestrator
from core.metrics_collector import MetricsCollector, RunMetrics
from core.danger_detector import SignalCollector, DangerAnalyzer
from html_report_generator import generate_html_report


class BenchmarkRunner:
    """Runs benchmarks across multiple models and generates comparison reports."""

    def __init__(self, benchmark_config_path: str):
        """
        Initialize benchmark runner.

        Args:
            benchmark_config_path: Path to benchmark configuration YAML
        """
        self.config_path = benchmark_config_path
        self.config = self._load_config(benchmark_config_path)
        self.results: List[RunMetrics] = []
        self.base_scenario_config = None

    def _load_config(self, config_path: str) -> dict:
        """Load benchmark configuration."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Benchmark config not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in benchmark config: {e}")

    def _load_base_scenario(self) -> dict:
        """Load the base scenario configuration."""
        base_scenario_path = self.config.get("base_scenario")
        if not base_scenario_path:
            raise ValueError("Benchmark config missing 'base_scenario' field")

        try:
            with open(base_scenario_path, "r") as f:
                scenario = yaml.safe_load(f)
            return scenario
        except FileNotFoundError:
            raise FileNotFoundError(f"Base scenario not found: {base_scenario_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in base scenario: {e}")

    def _create_scenario_for_run(self, base_scenario: dict, run_config: dict, model_pool: dict) -> dict:
        """
        Create a scenario configuration for a benchmark run with agent-model mappings.

        Args:
            base_scenario: Base scenario config
            run_config: Benchmark run configuration with agent_model_mapping
            model_pool: Pool of available models

        Returns:
            Modified scenario config with specific agent-model assignments
        """
        scenario = deepcopy(base_scenario)

        # Get agent-model mapping
        agent_model_mapping = run_config.get("agent_model_mapping", {})
        engine_model_id = run_config.get("engine_model")

        # Update engine configuration if specified
        if engine_model_id:
            if engine_model_id not in model_pool:
                raise ValueError(f"Engine model '{engine_model_id}' not found in model_pool")

            engine_model = model_pool[engine_model_id]
            if "engine" not in scenario:
                scenario["engine"] = {}

            scenario["engine"]["provider"] = engine_model["provider"]
            scenario["engine"]["model"] = engine_model["model"]
            if "base_url" in engine_model:
                scenario["engine"]["base_url"] = engine_model["base_url"]

        # Update each agent's LLM configuration based on mapping
        for agent in scenario.get("agents", []):
            agent_name = agent["name"]

            if agent_name in agent_model_mapping:
                model_id = agent_model_mapping[agent_name]

                if model_id not in model_pool:
                    raise ValueError(f"Model '{model_id}' not found in model_pool")

                model_config = model_pool[model_id]

                # Update agent's LLM config
                if "llm" not in agent:
                    agent["llm"] = {}

                agent["llm"]["provider"] = model_config["provider"]
                agent["llm"]["model"] = model_config["model"]
                if "base_url" in model_config:
                    agent["llm"]["base_url"] = model_config["base_url"]

        return scenario

    def _save_temp_scenario(self, scenario: dict, model_name: str) -> str:
        """
        Save temporary scenario file for this model.

        Args:
            scenario: Scenario configuration
            model_name: Name of the model

        Returns:
            Path to temporary scenario file
        """
        temp_dir = Path("benchmarks/.temp")
        temp_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = temp_dir / f"scenario_{model_name}_{timestamp}.yaml"

        with open(temp_path, "w") as f:
            yaml.dump(scenario, f, default_flow_style=False)

        return str(temp_path)

    def run_single_benchmark_config(self, run_config: dict, model_pool: dict, base_scenario: dict, signal_collector=None) -> Optional[RunMetrics]:
        """
        Run benchmark for a single configuration (new format with agent-model mapping).

        Args:
            run_config: Benchmark run configuration
            model_pool: Pool of available models
            base_scenario: Base scenario config
            signal_collector: Optional SignalCollector for danger detection

        Returns:
            RunMetrics object or None if run failed
        """
        run_name = run_config["name"]
        run_description = run_config.get("description", "No description")
        agent_model_mapping = run_config.get("agent_model_mapping", {})

        print(f"\n{'='*70}")
        print(f"Running benchmark: {run_name}")
        print(f"Description: {run_description}")
        print(f"Agent-Model Mapping:")
        for agent_name, model_id in agent_model_mapping.items():
            model_info = model_pool[model_id]
            print(f"  {agent_name} -> {model_id} ({model_info['provider']}/{model_info['model']})")
        if "engine_model" in run_config:
            engine_model_id = run_config["engine_model"]
            engine_info = model_pool[engine_model_id]
            print(f"  [Engine] -> {engine_model_id} ({engine_info['provider']}/{engine_info['model']})")
        print(f"{'='*70}\n")

        # Create scenario for this run configuration
        scenario = self._create_scenario_for_run(base_scenario, run_config, model_pool)
        temp_scenario_path = self._save_temp_scenario(scenario, run_name)

        # Initialize metrics collector
        metrics_config = self.config.get("metrics", {})

        # Create a description of the model combination for metrics
        model_combo_desc = ", ".join([f"{agent}:{model_id}" for agent, model_id in agent_model_mapping.items()])

        collector = MetricsCollector(
            model_name=run_name,
            provider=model_combo_desc,  # Store the combination info
            metrics_config=metrics_config
        )

        try:
            # Create orchestrator with custom scenario
            orchestrator = OrchestratorWithMetrics(
                config_path=temp_scenario_path,
                scenario_name=f"benchmark_{run_name}",
                save_frequency=0,  # Only save final state for benchmarks
                metrics_collector=collector,
                signal_collector=signal_collector  # Pass signal collector
            )

            # Run simulation
            orchestrator.run()

            # Mark as completed
            collector.record_completion(completed=True)

        except KeyboardInterrupt:
            print(f"\n\nBenchmark interrupted for {run_name}")
            collector.record_completion(completed=False)
            raise

        except Exception as e:
            print(f"\nERROR: Benchmark failed for {run_name}: {e}")
            if not self.config.get("run_config", {}).get("continue_on_error", True):
                raise
            collector.record_completion(completed=False)
            # Add error to metrics
            collector.metrics.error_messages.append(str(e))

        finally:
            # Clean up temp file
            try:
                Path(temp_scenario_path).unlink()
            except Exception:
                pass

        return collector.get_metrics()

    def run(self):
        """Run the full benchmark suite."""
        print(f"\n{'='*70}")
        print(f"Benchmark: {self.config.get('name', 'Unnamed')}")
        print(f"Description: {self.config.get('description', 'No description')}")
        print(f"{'='*70}\n")

        # Load base scenario
        base_scenario = self._load_base_scenario()
        self.base_scenario_config = base_scenario  # Store for reporting

        # Validate required fields
        model_pool = self.config.get("model_pool", {})
        if not model_pool:
            print("ERROR: No model_pool defined in benchmark config", file=sys.stderr)
            print("Benchmark config must have 'model_pool' with available models.", file=sys.stderr)
            sys.exit(1)

        benchmark_runs = self.config.get("benchmark_runs", [])
        if not benchmark_runs:
            print("ERROR: No benchmark_runs defined in benchmark config", file=sys.stderr)
            print("Benchmark config must have 'benchmark_runs' with agent-model mappings.", file=sys.stderr)
            sys.exit(1)

        print(f"Model pool: {list(model_pool.keys())}")
        print(f"Benchmark runs: {[r['name'] for r in benchmark_runs]}\n")

        # Initialize danger detection if enabled
        danger_config = self.config.get("danger_detection", {})
        danger_enabled = danger_config.get("enabled", False)
        signal_collector = SignalCollector() if danger_enabled else None

        if danger_enabled:
            print("Danger detection: ENABLED")
            judge_model = danger_config.get("judge_model", {})
            print(f"Judge model: {judge_model.get('provider')}/{judge_model.get('model')}\n")

        # Run each benchmark configuration
        for run_config in benchmark_runs:
            metrics = self.run_single_benchmark_config(run_config, model_pool, base_scenario, signal_collector)
            if metrics:
                self.results.append(metrics)

        # Run danger analysis if enabled
        if danger_enabled and signal_collector:
            print("\n" + "="*70)
            print("Running danger analysis...")
            print("="*70 + "\n")

            self._run_danger_analysis(signal_collector, danger_config)

        # Generate reports
        self.generate_reports()

    def generate_reports(self):
        """Generate comparison reports."""
        if not self.results:
            print("\nNo results to report.")
            return

        print(f"\n\n{'='*70}")
        print("BENCHMARK RESULTS")
        print(f"{'='*70}\n")

        # Create per-run directory similar to normal results
        base_output_dir = Path(self.config.get("reporting", {}).get("output_dir", "benchmarks/results"))
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        benchmark_name = self.config.get("name", "benchmark")

        # Create directory with format: benchmark_<name>_<timestamp>
        run_dir = base_output_dir / f"benchmark_{benchmark_name}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save raw JSON
        json_path = None
        if "json" in self.config.get("reporting", {}).get("formats", ["json"]):
            json_path = run_dir / f"{benchmark_name}.json"
            with open(json_path, "w") as f:
                json.dump([r.to_dict() for r in self.results], f, indent=2)
            print(f"Raw results saved to: {json_path}")

        # Generate markdown report
        if "markdown" in self.config.get("reporting", {}).get("formats", ["markdown"]):
            md_path = run_dir / f"{benchmark_name}.md"
            self._generate_markdown_report(md_path)
            print(f"Markdown report saved to: {md_path}")

        # Generate HTML report
        if "html" in self.config.get("reporting", {}).get("formats", ["json", "markdown"]):
            if json_path:
                html_path = run_dir / f"{benchmark_name}.html"
                try:
                    generate_html_report(json_path, html_path, benchmark_name,
                                       self.config, self.base_scenario_config)
                except Exception as e:
                    print(f"Warning: Failed to generate HTML report: {e}", file=sys.stderr)

        print(f"\nAll benchmark results saved to: {run_dir}")

        # Print comparison table
        if self.config.get("reporting", {}).get("comparison_table", True):
            self._print_comparison_table()

    def _run_danger_analysis(self, signal_collector: SignalCollector, danger_config: dict):
        """
        Run danger analysis on all collected signals.

        Args:
            signal_collector: SignalCollector with signals from all runs
            danger_config: Danger detection configuration
        """
        judge_model_config = danger_config.get("judge_model", {})
        analyzer = DangerAnalyzer(judge_model_config)

        # Analyze each agent in each run
        for metrics in self.results:
            run_name = metrics.model_name

            # Get agent names from conversation
            agent_names = set()
            for turn in metrics.conversation:
                for exchange in turn["exchanges"]:
                    agent_names.add(exchange["agent"])

            # Analyze each agent
            danger_scores = {}
            for agent_name in agent_names:
                print(f"Analyzing {agent_name} in {run_name}...")

                # Collect agent's response history
                response_history = []
                for turn in metrics.conversation:
                    for exchange in turn["exchanges"]:
                        if exchange["agent"] == agent_name:
                            response_history.append(exchange["response_from_agent"])

                # Get agent's signals
                agent_signals = signal_collector.get_signals_for_agent(agent_name)

                # Get agent role from base scenario
                agent_role = self._get_agent_role(agent_name)

                # Analyze
                try:
                    scores = analyzer.analyze_agent(
                        agent_name=agent_name,
                        run_name=run_name,
                        agent_role=agent_role,
                        response_history=response_history,
                        signals=agent_signals
                    )
                    danger_scores[agent_name] = scores

                    print(f"  Power-Seeking: {scores.power_seeking}/10")
                    print(f"  Deception: {scores.deception}/10")
                    print(f"  Rule Exploitation: {scores.rule_exploitation}/10")
                    print(f"  Average: {scores.average_score:.1f}/10\n")

                except Exception as e:
                    print(f"  ERROR: Failed to analyze {agent_name}: {e}\n")

            # Add danger scores to metrics
            metrics.danger_scores = danger_scores

    def _get_agent_role(self, agent_name: str) -> str:
        """Get agent's system prompt/role from base scenario."""
        if not self.base_scenario_config:
            return "Unknown role"

        agents = self.base_scenario_config.get("agents", [])
        for agent in agents:
            if agent.get("name") == agent_name:
                return agent.get("system_prompt", "No role specified")

        return "Unknown role"

    def _print_comparison_table(self):
        """Print comparison table to console."""
        print("\n## Performance Comparison\n")

        # Table header
        print(f"{'Model':<25} {'Completed':<12} {'Total Time':<12} {'Avg Step Time':<15} {'Errors':<8}")
        print("-" * 80)

        for result in self.results:
            completed = "✓" if result.completed else "✗"
            total_time = f"{result.total_time:.2f}s" if result.total_time else "N/A"
            avg_step = f"{result.avg_step_time:.3f}s" if result.avg_step_time else "N/A"
            errors = str(result.error_count)

            print(f"{result.model_name:<25} {completed:<12} {total_time:<12} {avg_step:<15} {errors:<8}")

        print()

    def _write_scenario_section_md(self, f):
        """Write scenario configuration section to markdown report."""
        # Benchmark runs configuration
        f.write("### Benchmark Runs\n\n")
        model_pool = self.config.get("model_pool", {})
        benchmark_runs = self.config.get("benchmark_runs", [])

        for run in benchmark_runs:
            f.write(f"**{run['name']}**\n\n")
            f.write(f"- Description: {run.get('description', 'N/A')}\n")
            f.write(f"- Agent-Model Mapping:\n")
            for agent, model_id in run.get("agent_model_mapping", {}).items():
                model = model_pool.get(model_id, {})
                f.write(f"  - `{agent}` → {model_id} ({model.get('provider', 'N/A')}/{model.get('model', 'N/A')})\n")
            if "engine_model" in run:
                engine_id = run["engine_model"]
                engine = model_pool.get(engine_id, {})
                f.write(f"  - `[Engine]` → {engine_id} ({engine.get('provider', 'N/A')}/{engine.get('model', 'N/A')})\n")
            f.write("\n")

        # Base scenario info
        if self.base_scenario_config:
            f.write("### Base Scenario\n\n")
            f.write(f"- **Max Steps:** {self.base_scenario_config.get('max_steps', 'N/A')}\n")
            f.write(f"- **Orchestrator Message:** {self.base_scenario_config.get('orchestrator_message', 'N/A')}\n\n")

            # Agents
            agents = self.base_scenario_config.get("agents", [])
            if agents:
                f.write("**Agents:**\n\n")
                for agent in agents:
                    f.write(f"- **{agent.get('name', 'Unnamed')}**\n")
                    system_prompt = agent.get('system_prompt', '').strip()
                    if system_prompt:
                        # Take first line or first 80 chars
                        prompt_preview = system_prompt.split('\n')[0][:80]
                        f.write(f"  - Prompt: {prompt_preview}...\n")
                    vars = agent.get('variables', {})
                    if vars:
                        f.write(f"  - Initial Variables: {', '.join(f'{k}={v}' for k, v in vars.items())}\n")
                    f.write("\n")

            # Global variables
            global_vars = self.base_scenario_config.get("global_vars", {})
            if global_vars:
                f.write("**Global Variables:**\n\n")
                for var_name, var_config in global_vars.items():
                    f.write(f"- `{var_name}` ({var_config.get('type', 'any')}): {var_config.get('description', 'N/A')} (default: {var_config.get('default', 'N/A')})\n")
                f.write("\n")

    def _generate_markdown_report(self, output_path: Path):
        """Generate detailed markdown report."""
        with open(output_path, "w") as f:
            f.write(f"# Benchmark Results: {self.config.get('name', 'Unnamed')}\n\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Description:** {self.config.get('description', 'No description')}\n\n")

            # Scenario Configuration section
            f.write("## Scenario Configuration\n\n")
            self._write_scenario_section_md(f)

            f.write("## Summary\n\n")
            f.write(f"- Total models tested: {len(set(r.model_name for r in self.results))}\n")
            f.write(f"- Total runs: {len(self.results)}\n")
            f.write(f"- Successful completions: {sum(1 for r in self.results if r.completed)}\n\n")

            f.write("## Performance Comparison\n\n")
            f.write("| Model | Provider | Completed | Total Time | Avg Step Time | Errors |\n")
            f.write("|-------|----------|-----------|------------|---------------|--------|\n")

            for result in self.results:
                completed = "✓" if result.completed else "✗"
                total_time = f"{result.total_time:.2f}s" if result.total_time else "N/A"
                avg_step = f"{result.avg_step_time:.3f}s" if result.avg_step_time else "N/A"

                f.write(f"| {result.model_name} | {result.provider} | {completed} | {total_time} | {avg_step} | {result.error_count} |\n")

            # Quality metrics
            if any(r.constraint_violations for r in self.results):
                f.write("\n## Quality Metrics\n\n")
                f.write("| Model | Constraint Violations | Decisions Made |\n")
                f.write("|-------|-----------------------|----------------|\n")
                for result in self.results:
                    f.write(f"| {result.model_name} | {result.constraint_violations} | {result.decision_count} |\n")

            # Custom metrics
            if any(r.custom_metrics for r in self.results):
                f.write("\n## Custom Metrics (Final State)\n\n")
                for result in self.results:
                    if result.custom_metrics:
                        f.write(f"\n### {result.model_name}\n\n")
                        f.write("```json\n")
                        f.write(json.dumps(result.custom_metrics, indent=2))
                        f.write("\n```\n")

            # Errors
            if any(r.error_messages for r in self.results):
                f.write("\n## Errors\n\n")
                for result in self.results:
                    if result.error_messages:
                        f.write(f"\n### {result.model_name}\n\n")
                        for error in result.error_messages:
                            f.write(f"- {error}\n")


class OrchestratorWithMetrics(Orchestrator):
    """Extended orchestrator that collects metrics during execution."""

    def __init__(self, config_path: str, scenario_name: str, save_frequency: int,
                 metrics_collector: MetricsCollector, engine_llm_provider=None,
                 signal_collector=None):
        self.metrics_collector = metrics_collector
        self.signal_collector = signal_collector  # Optional signal collector
        super().__init__(config_path, scenario_name, save_frequency, engine_llm_provider)

    def run(self):
        """Run simulation with metrics collection."""
        try:
            # Run initialization
            print("=== Step 0: Initialization ===")
            agent_names = [agent.name for agent in self.agents]
            agent_messages = self.simulator_agent.initialize_simulation(agent_names)
            print("SimulatorAgent initialized simulation")

            # Main simulation loop
            for step in range(1, self.max_steps + 1):
                print(f"\n=== Step {step}/{self.max_steps} ===")

                # Start step timing
                self.metrics_collector.start_step(step)
                step_errors = []

                # Capture state before step
                if self.signal_collector:
                    state_before = self.game_engine.get_state()

                # Collect agent responses
                agent_responses = {}

                for agent in self.agents:
                    try:
                        message = agent_messages[agent.name]
                        print(f"SimulatorAgent -> {agent.name}: {message}")

                        response = agent.respond(message)
                        print(f"{agent.name} -> SimulatorAgent: {response}")

                        agent_responses[agent.name] = response

                    except Exception as e:
                        error_msg = f"Agent {agent.name} failed: {e}"
                        print(f"ERROR: {error_msg}", file=sys.stderr)
                        agent_responses[agent.name] = f"ERROR: {str(e)}"
                        step_errors.append(error_msg)

                # SimulatorAgent processes step
                try:
                    next_agent_messages = self.simulator_agent.process_step(step, agent_responses)
                    print(f"[SimulatorAgent processed step {step}]")
                except Exception as e:
                    error_msg = f"Simulation failed at step {step}: {e}"
                    print(f"\nERROR: {error_msg}", file=sys.stderr)
                    step_errors.append(error_msg)
                    raise

                # Collect danger signals if enabled
                if self.signal_collector:
                    state_after = self.game_engine.get_state()
                    self.signal_collector.collect_step_signals(
                        step, agent_responses, state_before, state_after
                    )

                # Record conversation turn
                self.metrics_collector.record_conversation_turn(step, agent_messages, agent_responses)

                # End step and record metrics
                snapshot = self.game_engine.get_state_snapshot()
                self.metrics_collector.end_step(step, snapshot, step_errors if step_errors else None)

                # Update messages for next step
                agent_messages = next_agent_messages

                # Advance round
                self.game_engine.advance_round()

            # Record final state
            final_snapshot = self.game_engine.get_state_snapshot()
            self.metrics_collector.record_final_state(final_snapshot)

            print("\nSimulation completed.")
            print(f"Final game state: {self.game_engine.get_state()}")

        finally:
            if hasattr(self, 'persistence'):
                self.persistence.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run LLM benchmarks on simulation scenarios"
    )
    parser.add_argument(
        "benchmark_config",
        help="Path to benchmark configuration YAML"
    )

    args = parser.parse_args()

    try:
        runner = BenchmarkRunner(args.benchmark_config)
        runner.run()

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\nFatal Error: {e}", file=sys.stderr)
        print("\nFull traceback:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
