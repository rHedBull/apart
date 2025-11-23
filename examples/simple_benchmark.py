#!/usr/bin/env python3
"""
Simple Benchmark Runner - Compare models on the same scenario

Usage:
    python examples/simple_benchmark.py

Requirements:
    - For Gemini: Set GEMINI_API_KEY in .env
    - For Ollama: Run `ollama serve` and pull models
"""

import sys
import yaml
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.orchestrator import Orchestrator


class SimpleBenchmarkRunner:
    """Run the same scenario with different models and compare results."""

    def __init__(self, scenario_path: str):
        self.scenario_path = scenario_path
        self.results = []

    def run_benchmark(self, models: list[dict]) -> list[dict]:
        """
        Run scenario with each model.

        Args:
            models: List of dicts with keys: name, provider, model
                Example: {"name": "gemini-flash", "provider": "gemini", "model": "gemini-1.5-flash"}

        Returns:
            List of results for each model
        """
        print("=" * 80)
        print(f"BENCHMARK: {self.scenario_path}")
        print(f"Models: {len(models)}")
        print("=" * 80)

        for i, model_config in enumerate(models, 1):
            print(f"\n[{i}/{len(models)}] Running: {model_config['name']}")
            print("-" * 80)

            try:
                result = self._run_single_model(model_config)
                self.results.append(result)
                print(f"✅ {model_config['name']}: {result['status']}")

            except Exception as e:
                error_result = {
                    "model": model_config['name'],
                    "status": "failed",
                    "error": str(e)
                }
                self.results.append(error_result)
                print(f"❌ {model_config['name']}: {e}")

        return self.results

    def _run_single_model(self, model_config: dict) -> dict:
        """Run scenario with a single model."""

        # Load base scenario
        with open(self.scenario_path) as f:
            scenario = yaml.safe_load(f)

        # Update LLM config for all agents
        for agent in scenario.get("agents", []):
            if "llm" in agent:
                agent["llm"]["provider"] = model_config["provider"]
                agent["llm"]["model"] = model_config["model"]

        # Create temp scenario file
        temp_scenario = Path(f"scenarios/_temp_benchmark_{model_config['name']}.yaml")
        with open(temp_scenario, "w") as f:
            yaml.dump(scenario, f)

        try:
            # Run orchestrator
            orchestrator = Orchestrator(str(temp_scenario))
            orchestrator.run()

            # Collect metrics
            result = {
                "model": model_config['name'],
                "provider": model_config['provider'],
                "status": "completed",
                "run_dir": str(orchestrator.run_dir),
                "metrics": self._extract_metrics(orchestrator.run_dir)
            }

            return result

        finally:
            # Cleanup temp file
            temp_scenario.unlink(missing_ok=True)

    def _extract_metrics(self, run_dir: Path) -> dict:
        """Extract metrics from run directory."""

        # Read final state
        state_file = run_dir / "state.json"
        with open(state_file) as f:
            states = json.load(f)
            final_state = states[-1] if isinstance(states, list) else states

        # Read logs
        log_file = run_dir / "simulation.jsonl"
        logs = []
        with open(log_file) as f:
            for line in f:
                logs.append(json.loads(line))

        # Count events
        agent_responses = [log for log in logs if log.get("code") == "AGT003"]
        errors = [log for log in logs if log.get("level") == "ERROR"]

        # Calculate total time
        sim_start = next((log for log in logs if log.get("code") == "SIM001"), None)
        sim_end = next((log for log in logs if log.get("code") == "SIM002"), None)

        total_time_ms = None
        if sim_start and sim_end:
            start = datetime.fromisoformat(sim_start["timestamp"])
            end = datetime.fromisoformat(sim_end["timestamp"])
            total_time_ms = (end - start).total_seconds() * 1000

        return {
            "rounds_completed": final_state.get("game_state", {}).get("round", 0),
            "total_events": len(final_state.get("game_state", {}).get("events", [])),
            "agent_responses": len(agent_responses),
            "error_count": len(errors),
            "total_time_ms": round(total_time_ms, 2) if total_time_ms else None,
            "global_vars": final_state.get("global_vars", {}),
        }

    def print_comparison(self):
        """Print comparison table."""
        print("\n" + "=" * 100)
        print("BENCHMARK RESULTS")
        print("=" * 100)

        # Header
        print(f"{'Model':<25} {'Provider':<12} {'Status':<12} {'Rounds':<8} {'Responses':<10} {'Time (ms)':<12}")
        print("-" * 100)

        # Rows
        for result in self.results:
            if result["status"] == "completed":
                m = result["metrics"]
                print(f"{result['model']:<25} "
                      f"{result['provider']:<12} "
                      f"{result['status']:<12} "
                      f"{m['rounds_completed']:<8} "
                      f"{m['agent_responses']:<10} "
                      f"{m['total_time_ms'] or 'N/A':<12}")
            else:
                print(f"{result['model']:<25} "
                      f"{result.get('provider', 'N/A'):<12} "
                      f"{result['status']:<12} "
                      f"{'N/A':<8} "
                      f"{'N/A':<10} "
                      f"{'N/A':<12}")

        print("=" * 100)

    def print_detailed_results(self):
        """Print detailed results for each model."""
        print("\n" + "=" * 100)
        print("DETAILED RESULTS")
        print("=" * 100)

        for result in self.results:
            print(f"\n📊 {result['model']} ({result['provider']})")
            print("-" * 100)

            if result["status"] == "completed":
                m = result["metrics"]
                print(f"  Status: ✅ Completed")
                print(f"  Rounds: {m['rounds_completed']}")
                print(f"  Responses: {m['agent_responses']}")
                print(f"  Errors: {m['error_count']}")
                print(f"  Time: {m['total_time_ms']:.2f} ms")
                print(f"  Run directory: {result['run_dir']}")

                if m['global_vars']:
                    print(f"  Global vars:")
                    for key, value in m['global_vars'].items():
                        print(f"    {key}: {value}")
            else:
                print(f"  Status: ❌ Failed")
                print(f"  Error: {result.get('error', 'Unknown error')}")

        print("=" * 100)

    def save_results(self, output_file: str = "benchmark_results.json"):
        """Save results to JSON file."""
        output_data = {
            "scenario": self.scenario_path,
            "timestamp": datetime.now().isoformat(),
            "models_tested": len(self.results),
            "results": self.results
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Run example benchmark."""

    # Configuration
    scenario = "scenarios/config.yaml"  # Use your scenario

    models = [
        # Gemini (requires GEMINI_API_KEY)
        {
            "name": "gemini-flash",
            "provider": "gemini",
            "model": "gemini-1.5-flash"
        },

        # Ollama models (requires ollama serve)
        # Uncomment if you have Ollama running:
        # {
        #     "name": "mistral-7b",
        #     "provider": "ollama",
        #     "model": "mistral:7b"
        # },
        # {
        #     "name": "llama3-8b",
        #     "provider": "ollama",
        #     "model": "llama3.1:8b"
        # },
        # {
        #     "name": "gemma-2b",
        #     "provider": "ollama",
        #     "model": "gemma3:2b"
        # },
    ]

    # Run benchmark
    runner = SimpleBenchmarkRunner(scenario)
    runner.run_benchmark(models)

    # Print results
    runner.print_comparison()
    runner.print_detailed_results()

    # Save results
    runner.save_results("benchmark_results.json")

    print("\n✅ Benchmark complete!")
    print("\nTo analyze results:")
    print("  cat benchmark_results.json | jq .")
    print("  cat benchmark_results.json | jq '.results[] | {model, status, rounds: .metrics.rounds_completed}'")


if __name__ == "__main__":
    main()
