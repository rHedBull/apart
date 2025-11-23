# Benchmarking Framework

This directory contains the benchmarking framework for comparing different agent-model combinations on simulation scenarios.

## Quick Start

```bash
# Run the example benchmark
uv run python tools/benchmark.py benchmarks/example_benchmark.yaml

# Create your own benchmark
cp benchmarks/example_benchmark.yaml benchmarks/my_benchmark.yaml
# Edit my_benchmark.yaml to define your model pool and benchmark runs
```

## How It Works

The benchmark runner:
1. Loads a base scenario configuration (defines the agents)
2. Loads a model pool (all available models for the benchmark)
3. For each benchmark run, creates a scenario with specific agent-model mappings
4. Runs simulations and collects metrics
5. Generates comparison reports in JSON and Markdown formats

## Benchmark Configuration

### Structure

```yaml
name: "my_benchmark"
description: "What I'm testing"
base_scenario: "scenarios/simple_agents.yaml"

# Define all available models for this benchmark
model_pool:
  model-id-1:
    provider: "gemini"
    model: "gemini-2.5-flash"

  model-id-2:
    provider: "ollama"
    model: "mistral:7b"
    base_url: "http://localhost:11434"

# Define specific agent-model combinations to test
benchmark_runs:
  - name: "run_1"
    description: "What this run tests"
    agent_model_mapping:
      "Agent A": "model-id-1"    # Agent name from scenario
      "Agent B": "model-id-2"
    engine_model: "model-id-1"   # Optional: which model runs the simulation engine

# Metrics, reporting, and run config (see below)
```

### Required Fields

**Top-level:**
- `name`: Benchmark name
- `description`: What you're testing
- `base_scenario`: Path to scenario YAML file (defines the agents)
- `model_pool`: Dictionary of available models
- `benchmark_runs`: List of specific combinations to test

**model_pool entry:**
- `provider`: "gemini", "ollama", etc.
- `model`: Model identifier (e.g., "gemini-2.5-flash", "mistral:7b")
- `base_url`: (Optional) For Ollama or self-hosted models

**benchmark_runs entry:**
- `name`: Unique identifier for this run
- `agent_model_mapping`: Maps each agent name to a model ID from model_pool
- `engine_model`: (Optional) Which model runs the simulation engine

### Agent-Model Mapping

The `agent_model_mapping` maps agent names (from your scenario) to model IDs (from your model_pool):

```yaml
# In your scenario file (e.g., scenarios/simple_agents.yaml)
agents:
  - name: "Agent A"  # <-- This name
    ...
  - name: "Agent B"
    ...

# In your benchmark config
benchmark_runs:
  - name: "test_run"
    agent_model_mapping:
      "Agent A": "gemini-flash"   # Maps "Agent A" to gemini-flash from model_pool
      "Agent B": "mistral-7b"     # Maps "Agent B" to mistral-7b from model_pool
```

**Important:** Agent names in `agent_model_mapping` must exactly match agent names in your scenario file.

### Metrics Configuration

```yaml
metrics:
  performance:
    enabled: true
    collect:
      - total_time          # Total execution time
      - step_times          # Time per step (list)
      - avg_step_time       # Average time per step

  quality:
    enabled: true
    collect:
      - variable_changes    # Track all variable changes
      - final_state         # Final state of all variables
      - decision_count      # Number of decisions made
      - constraint_violations  # Count of min/max violations

  reliability:
    enabled: true
    collect:
      - completion_rate     # Did simulation complete?
      - error_count         # Number of errors
      - step_failures       # Which steps failed

  custom:
    enabled: true
    track_variables:
      - interest_rate       # Global variable from scenario
      - "*.score"           # All agent scores (pattern matching)
```

### Reporting Options

```yaml
reporting:
  output_dir: "benchmarks/results"  # Where to save results
  formats:
    - json       # Raw data
    - markdown   # Human-readable report
  comparison_table: true      # Print comparison table to console
  save_run_logs: true         # Save individual run logs
```

### Run Configuration

```yaml
run_config:
  execution_mode: "sequential"  # Run benchmark configs one at a time
  continue_on_error: true       # Continue if a run fails
```

## Use Cases

### 1. Compare Models Head-to-Head

Test the same model against itself vs. against a different model:

```yaml
benchmark_runs:
  - name: "gemini_vs_gemini"
    agent_model_mapping:
      "Agent A": "gemini-flash"
      "Agent B": "gemini-flash"

  - name: "gemini_vs_mistral"
    agent_model_mapping:
      "Agent A": "gemini-flash"
      "Agent B": "mistral-7b"
```

**Question:** Does Gemini perform differently when paired with itself vs. paired with Mistral?

### 2. Role-Specific Model Testing

Test if certain models are better for specific roles:

```yaml
benchmark_runs:
  # Large model for aggressive, small for conservative
  - name: "gpt4_aggressive_llama_conservative"
    agent_model_mapping:
      "Aggressive Strategist": "gpt-4"
      "Conservative Analyst": "llama-7b"

  # Swap them
  - name: "llama_aggressive_gpt4_conservative"
    agent_model_mapping:
      "Aggressive Strategist": "llama-7b"
      "Conservative Analyst": "gpt-4"
```

**Question:** Are expensive models only needed for certain agent types?

### 3. Cost Optimization

Find the cheapest combination that maintains quality:

```yaml
model_pool:
  expensive: {provider: "openai", model: "gpt-4"}
  medium: {provider: "gemini", model: "gemini-2.5-flash"}
  cheap: {provider: "ollama", model: "llama2:7b"}

benchmark_runs:
  - name: "all_expensive"
    agent_model_mapping: {...all use expensive...}

  - name: "mixed_expensive_medium"
    agent_model_mapping: {...some expensive, some medium...}

  - name: "all_cheap"
    agent_model_mapping: {...all use cheap...}
```

**Question:** What's the minimum cost to maintain acceptable simulation quality?

### 4. Asymmetric Scenarios

Test what happens when agents have different capabilities:

```yaml
benchmark_runs:
  - name: "advantage_agent_a"
    description: "Agent A has GPT-4, Agent B has Llama2"
    agent_model_mapping:
      "Agent A": "gpt-4"
      "Agent B": "llama2"

  - name: "advantage_agent_b"
    description: "Agent B has GPT-4, Agent A has Llama2"
    agent_model_mapping:
      "Agent A": "llama2"
      "Agent B": "gpt-4"
```

**Question:** Does having a more powerful model give an agent a competitive advantage?

## Output Files

After running a benchmark:

### Console Output
- Real-time progress for each run
- Agent-model mapping for each run
- Comparison table with key metrics

### JSON Report
`benchmarks/results/{benchmark_name}_{timestamp}.json`
- Raw metrics data
- All collected metrics for each run
- Machine-readable for further analysis

### Markdown Report
`benchmarks/results/{benchmark_name}_{timestamp}.md`
- Human-readable summary
- Performance comparison table
- Quality metrics
- Custom metrics (final state)
- Error details (if any)

## Example Workflow

1. **Create your scenario** (or use existing):
   ```yaml
   # scenarios/my_scenario.yaml
   agents:
     - name: "Agent A"
       ...
     - name: "Agent B"
       ...
   ```

2. **Create benchmark config**:
   ```bash
   cp benchmarks/example_benchmark.yaml benchmarks/my_test.yaml
   ```

3. **Edit benchmark config**:
   - Set `base_scenario` to your scenario file
   - Define `model_pool` with models you want to test
   - Define `benchmark_runs` with specific agent-model combinations

4. **Run benchmark**:
   ```bash
   uv run python tools/benchmark.py benchmarks/my_test.yaml
   ```

5. **Analyze results**:
   - Check console output for quick comparison
   - Open markdown report for detailed analysis
   - Use JSON for custom analysis or plotting

## Tips

- **Start small**: Test 1-2 runs first, then expand
- **Watch API limits**: Cloud providers have rate limits (Gemini free tier: 10 req/min)
- **Use local models**: Ollama models have no API limits
- **Reduce steps**: Edit `max_steps` in base scenario for faster testing
- **Match agent names**: Ensure `agent_model_mapping` uses exact agent names from scenario
- **Engine model**: Set `engine_model` if you want to control which model runs the simulation logic

## Troubleshooting

**"No model_pool defined"**
- Your benchmark config needs a `model_pool` section

**"No benchmark_runs defined"**
- Your benchmark config needs a `benchmark_runs` section

**"Model 'xyz' not found in model_pool"**
- Check that the model ID in `agent_model_mapping` exists in `model_pool`

**"Unknown agent 'xyz' in agent_model_mapping"**
- Check that agent names in `agent_model_mapping` exactly match agent names in your scenario file

**"Provider not available"**
- For Ollama: Check `ollama serve` is running
- For Gemini: Verify `GEMINI_API_KEY` environment variable

**"Quota exceeded"**
- Gemini free tier: 10 requests/minute - wait or use Ollama
- Consider using fewer benchmark runs or reducing `max_steps` in scenario

## Architecture

```
tools/benchmark.py              # Main benchmark runner
src/core/metrics_collector.py  # Metrics collection during runs
scenarios/
  └── simple_agents.yaml        # Example 2-agent scenario
benchmarks/
  ├── example_benchmark.yaml    # Example with agent-model mappings
  └── results/                  # Generated reports
```
