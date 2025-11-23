# apart

Multi-agent orchestration framework for running configurable simulation scenarios.

**Note:** As of v2.0, all scenarios require an LLM-powered simulation engine. See [Design Documentation](docs/plans/2025-11-22-llm-simulation-engine-design.md) for details.

**Recommended LLM Providers:**
- **Google Gemini** (gemini-1.5-flash): Fast, reliable, excellent JSON compliance (requires API key)
- **Ollama** (local, tested 2025-11-23): Privacy-friendly, no API key needed
  - ✅ **Excellent**: mistral:7b, llama3.1:8b, codellama:latest, deepseek-coder-v2:latest
  - ⚠️  **Basic**: gemma3:1b (simple prompts only)
  - ❌ **Avoid**: deepseek-r1:* models (reasoning models, not JSON-focused)

## Quick Start

### Run AI Safety Scenarios

We provide 4 sophisticated AI safety scenarios for testing ethical reasoning, governance, and strategic decision-making:

```bash
# 1. Autonomous Systems Cascade (10-second trolley problem)
uv run src/main.py scenarios/autonomous_systems_cascade.yaml

# 2. Prometheus Board Decision (10-turn governance simulation)
uv run src/main.py scenarios/prometheus_board_decision.yaml

# 3. AI Safety 2027 Landscape (24-month ecosystem simulation)
uv run src/main.py scenarios/ai_safety_2027_landscape.yaml

# 4. Taiwan Strait Crisis (14-day geopolitical conflict)
uv run src/main.py scenarios/taiwan_strait_crisis.yaml
```

See [AI Safety Scenarios Overview](scenarios/AI_SAFETY_SCENARIOS_OVERVIEW.md) for detailed descriptions.

### Run Benchmarks

Compare different AI models on the same scenario:

```bash
# Compare Gemini Flash, Gemini Pro, and Grok as AI Safety Monitor
./benchmarks/run_cascade_comparison.sh

# Or run directly:
uv run tools/benchmark.py benchmarks/cascade_monitor_comparison.yaml
```

**Setup:** Add `XAI_API_KEY` to `.env` for Grok testing (see [benchmark README](benchmarks/cascade_monitor_comparison_README.md))

### Custom Scenarios

Run with custom save frequency:
```bash
uv run src/main.py --save-frequency 2  # Save every 2 steps
uv run src/main.py --save-frequency 0  # Save only final state
```

Run your own scenario:
```bash
uv run src/main.py scenarios/my_scenario.yaml
```

## Documentation

### Core Documentation
- **[Architecture](docs/architecture.md)** - System design and component overview
- **[Scenario Creation Guide](docs/scenario-creation.md)** - How to create custom scenarios

### AI Safety Scenarios

**[AI Safety Scenarios Overview](scenarios/AI_SAFETY_SCENARIOS_OVERVIEW.md)** - Comprehensive guide to all scenarios

**Individual Scenario Documentation:**
1. **[Autonomous Systems Cascade](scenarios/autonomous_systems_cascade_README.md)** - 10-second trolley problem variant
   - Tests ethical reasoning under 70% uncertainty
   - 6 agents (5 systems + AI Safety Monitor)
   - Moral weight calculations: 65-75 immediate deaths vs 5-10 + 500K affected

2. **[Prometheus Board Decision](scenarios/prometheus_board_decision_README.md)** - 10-turn governance simulation
   - Tests group decision-making under pressure
   - 8 board members (CEO, CTO, CSO, Investor, Ethics Director, COO, Legal, Gov Liaison)
   - Deploy near-AGI with 30% deception rate and 67% kill switch reliability?

3. **[AI Safety 2027 Landscape](scenarios/ai_safety_2027_landscape_README.md)** - 24-month ecosystem simulation
   - Tests multi-actor strategic forecasting
   - 13 actors (6 labs, 5 governments/regulators, researchers, public)
   - Tracks cooperative development, racing dynamics, incidents, breakthroughs

4. **[Taiwan Strait Crisis](scenarios/taiwan_strait_crisis_README.md)** - 14-day geopolitical conflict
   - Tests robustness under fog of war (60-80% intelligence accuracy)
   - 10 actors (China, Taiwan, US, Japan, Australia, South Korea, ASEAN, EU, Russia, UN)
   - Cascading escalation from blockade to potential nuclear signaling

### Benchmarks

- **[Cascade Monitor Comparison](benchmarks/cascade_monitor_comparison_README.md)** - Compare models in AI Safety Monitor role
  - Tests: Gemini Flash, Gemini Pro, Grok
  - Evaluates: Ethical reasoning, uncertainty handling, stakeholder consideration

## Configuration

Edit `scenarios/config.yaml` to customize:
- Number of simulation steps
- Orchestrator messages
- Agent behaviors and responses
- Global and per-agent variables with type validation

## Persistence

Each simulation run creates a unique directory in `results/` with the format `run_<scenario>_<timestamp>/`.

### Output Files

**`state.json`** - Simulation snapshots:
- Game state (resources, difficulty, round)
- Global and per-agent variable values
- All orchestrator-agent messages

**`simulation.jsonl`** - Structured logs:
- Human-readable console output with colors
- Machine-readable JSONL for analysis
- Predefined message codes (SIM001, AGT002, etc.)
- Performance metrics and timing data

Control save frequency with `--save-frequency N`:
- `N=0`: Save only final state
- `N=1`: Save after every step (default)
- `N=k`: Save every k steps

### Analyzing Logs

Filter by message code:
```bash
# All agent responses
grep "AGT003" results/run_*/simulation.jsonl | jq .

# Persistence operations
grep "PER00" results/run_*/simulation.jsonl | jq .

# Performance metrics
grep "PRF001" results/run_*/simulation.jsonl | jq .
```
