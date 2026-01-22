#!/usr/bin/env python
"""
Benchmark script for measuring parallel vs sequential agent execution.

Usage:
    uv run python scripts/benchmark_parallel.py

This runs the benchmark scenario twice:
1. With parallel agent execution (APART_PARALLEL_AGENTS=1)
2. With sequential agent execution (APART_PARALLEL_AGENTS=0)

Uses a mock engine LLM (instant) to isolate agent LLM timing.
"""

import json
import os
import sys
import time
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.orchestrator import Orchestrator
from llm.llm_provider import LLMProvider


class MockEngineLLMProvider(LLMProvider):
    """
    Mock LLM provider for the engine that returns valid JSON instantly.
    This isolates the benchmark to measure only agent LLM call parallelization.
    """

    def __init__(self, agent_names: list[str]):
        self.agent_names = agent_names
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        self.call_count += 1

        # Generate valid engine response JSON
        agent_messages = {name: f"Report your status for step {self.call_count}" for name in self.agent_names}
        agent_vars = {name: {"counter": self.call_count} for name in self.agent_names}

        response = {
            "reasoning": f"Processing step {self.call_count}",
            "events": [],
            "agent_messages": agent_messages,
            "state_updates": {
                "global_vars": {"total_count": self.call_count * len(self.agent_names)},
                "agent_vars": agent_vars
            }
        }
        return json.dumps(response)


def run_benchmark(parallel: bool) -> tuple[float, int]:
    """Run benchmark and return (elapsed_time, num_agents)."""
    mode = "parallel" if parallel else "sequential"
    os.environ["APART_PARALLEL_AGENTS"] = "1" if parallel else "0"

    scenario_path = "scenarios/benchmark_parallel.yaml"
    run_id = f"benchmark_{mode}_{int(time.time())}"

    print(f"\n{'='*60}")
    print(f"Running {mode.upper()} benchmark...")
    print(f"{'='*60}\n")

    # Create mock engine provider
    agent_names = ["Agent_Alpha", "Agent_Beta", "Agent_Gamma", "Agent_Delta"]
    mock_engine = MockEngineLLMProvider(agent_names)

    start = time.time()

    try:
        orchestrator = Orchestrator(
            scenario_path,
            f"benchmark_{mode}",
            save_frequency=0,  # Don't save snapshots
            run_id=run_id,
            engine_llm_provider=mock_engine  # Use mock engine
        )
        num_agents = len(orchestrator.agents)
        orchestrator.run()
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return -1, 0

    elapsed = time.time() - start

    # Clean up results directory
    results_dir = Path("results") / f"run_{run_id}"
    if results_dir.exists():
        shutil.rmtree(results_dir)

    return elapsed, num_agents


def main():
    print("\n" + "="*60)
    print("PARALLEL AGENT EXECUTION BENCHMARK")
    print("="*60)
    print("\nScenario: benchmark_parallel.yaml")
    print("  - 4 agents (real Ollama LLM)")
    print("  - 2 steps")
    print("  - Mock engine (instant response)")
    print("\nThis benchmark measures the speedup from parallel agent calls.")
    print("Engine uses mock LLM to isolate agent timing.")

    # Check if Ollama is available
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("\nERROR: Ollama server not responding. Start it with: ollama serve")
            sys.exit(1)
        print("\n✓ Ollama server is running")
    except Exception:
        print("\nERROR: Cannot connect to Ollama. Start it with: ollama serve")
        sys.exit(1)

    # Run sequential first (baseline)
    seq_time, num_agents = run_benchmark(parallel=False)
    if seq_time < 0:
        print("Sequential benchmark failed!")
        sys.exit(1)

    # Run parallel
    par_time, _ = run_benchmark(parallel=True)
    if par_time < 0:
        print("Parallel benchmark failed!")
        sys.exit(1)

    # Calculate speedup
    speedup = seq_time / par_time if par_time > 0 else 0
    time_saved = seq_time - par_time
    pct_saved = (1 - par_time / seq_time) * 100 if seq_time > 0 else 0

    # Report results
    print("\n" + "="*60)
    print("BENCHMARK RESULTS")
    print("="*60)
    print(f"\n  Agents:               {num_agents}")
    print(f"  Steps:                2")
    print(f"  LLM calls per step:   {num_agents} (agents) + 1 (engine mock)")
    print(f"\n  Sequential execution: {seq_time:>8.1f}s")
    print(f"  Parallel execution:   {par_time:>8.1f}s")
    print(f"  {'─'*30}")
    print(f"  Speedup:              {speedup:>8.2f}x")
    print(f"  Time saved:           {time_saved:>8.1f}s ({pct_saved:.0f}%)")
    print("\n" + "="*60 + "\n")

    # Interpretation
    if speedup >= 2.5:
        print("✓ Excellent parallelization! GPU handles multiple requests well.")
    elif speedup >= 1.5:
        print("✓ Good parallelization. Some GPU queuing but still beneficial.")
    elif speedup >= 1.1:
        print("~ Modest improvement. GPU may be serializing requests.")
    else:
        print("✗ No significant speedup. Check GPU utilization.")


if __name__ == "__main__":
    main()
