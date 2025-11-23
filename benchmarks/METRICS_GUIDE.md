# Metrics Guide

Complete guide to metrics tracked during benchmarks and how to add custom metrics.

## Currently Tracked Metrics

### 1. Performance Metrics

**Automatically tracked:**
- `total_time` (float): Total simulation duration in seconds
- `step_times` (list[float]): Duration of each step in seconds
- `avg_step_time` (float): Average step duration
- `total_tokens` (int, optional): Total tokens used (if provider supports it)
- `avg_tokens_per_step` (float, optional): Average tokens per step

**Collection:** Happens automatically via timing in `OrchestratorWithMetrics.run()`

### 2. Quality Metrics

**Automatically tracked:**
- `variable_changes` (list): Full state snapshot at each step
  - Includes `global_vars` and `agent_vars` at each step
- `final_state` (dict): Complete final state snapshot
- `decision_count` (int): Number of decisions made (currently counts `decisions_made` variable)
- `constraint_violations` (int): Number of constraint violations detected

**Constraint checks currently implemented:**
- `interest_rate`: 0.0 to 1.0
- `market_volatility`: 0.0 to 1.0
- `capital`: >= 0
- `risk_tolerance`: 0.0 to 1.0

### 3. Reliability Metrics

**Automatically tracked:**
- `completed` (bool): Did simulation complete successfully?
- `error_count` (int): Number of errors encountered
- `step_failures` (list[int]): Which steps had errors
- `error_messages` (list[str]): All error messages

### 4. Custom Metrics

**Configured in benchmark config:**
```yaml
metrics:
  custom:
    enabled: true
    track_variables:
      - interest_rate        # Global variable
      - market_volatility    # Global variable
      - "*.capital"          # All agent capital values
      - "*.score"            # All agent scores
```

**Pattern matching:**
- `variable_name`: Extract global variable
- `*.variable_name`: Extract variable from all agents

## How to Add New Metrics

### Method 1: Add to Custom Metrics (No Code Changes)

Best for: Scenario-specific metrics using existing variables

**Steps:**
1. Ensure your scenario tracks the variable
2. Add to benchmark config:

```yaml
metrics:
  custom:
    enabled: true
    track_variables:
      - my_global_variable
      - "*.my_agent_variable"
```

**Example - Track economic indicators:**
```yaml
# In scenario
global_vars:
  gdp_growth:
    type: float
    default: 0.02
  inflation_rate:
    type: float
    default: 0.03

# In benchmark config
metrics:
  custom:
    enabled: true
    track_variables:
      - gdp_growth
      - inflation_rate
      - "*.wealth"
      - "*.debt"
```

### Method 2: Add to MetricsCollector (Code Changes)

Best for: Calculated metrics, cross-step analysis, or complex aggregations

**Steps:**

1. **Add field to `RunMetrics` dataclass:**

```python
# In src/core/metrics_collector.py

@dataclass
class RunMetrics:
    # ... existing fields ...

    # Add your new metric
    my_new_metric: float = 0.0
    score_volatility: List[float] = field(default_factory=list)
```

2. **Add calculation logic:**

```python
class MetricsCollector:

    def end_step(self, step: int, state_snapshot: Optional[Dict] = None, errors: Optional[List[str]] = None):
        # ... existing code ...

        # Add your calculation
        if self._is_enabled('quality') and state_snapshot:
            # Example: Track score volatility
            agent_vars = state_snapshot.get('agent_vars', {})
            scores = [data.get('score', 0) for data in agent_vars.values()]
            if len(scores) > 1:
                import statistics
                volatility = statistics.stdev(scores)
                self.metrics.score_volatility.append(volatility)
```

3. **Optional: Add to `get_metrics()` for final aggregation:**

```python
def get_metrics(self) -> RunMetrics:
    # ... existing code ...

    # Calculate final aggregate
    if self.metrics.score_volatility:
        self.metrics.my_new_metric = sum(self.metrics.score_volatility) / len(self.metrics.score_volatility)

    return self.metrics
```

### Method 3: Add Per-Step Metrics

Best for: Metrics that need to be tracked at each step

**Add to `StepMetrics` dataclass:**

```python
@dataclass
class StepMetrics:
    step: int
    duration: float
    tokens_used: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    variable_changes: Dict[str, Any] = field(default_factory=dict)

    # Add your per-step metric
    agent_interaction_count: int = 0
    market_events: List[str] = field(default_factory=list)
```

Then track it in `MetricsCollector`:

```python
def __init__(...):
    self.step_metrics: List[StepMetrics] = []

def end_step(self, step: int, ...):
    # Create step metric
    step_metric = StepMetrics(
        step=step,
        duration=duration,
        agent_interaction_count=len(state_snapshot.get('agent_vars', {}))
    )
    self.step_metrics.append(step_metric)
```

## Examples of Custom Metrics to Add

### Economic Simulation Metrics

```python
# In RunMetrics
wealth_inequality_gini: float = 0.0  # Gini coefficient
market_efficiency: float = 0.0
total_trades: int = 0
```

```python
# In end_step or get_metrics
def calculate_gini(self, wealth_values: List[float]) -> float:
    """Calculate Gini coefficient for wealth inequality."""
    n = len(wealth_values)
    if n == 0:
        return 0.0

    sorted_wealth = sorted(wealth_values)
    cumsum = 0
    for i, val in enumerate(sorted_wealth):
        cumsum += (2 * (i + 1) - n - 1) * val

    return cumsum / (n * sum(sorted_wealth))
```

### Agent Behavior Metrics

```python
# In RunMetrics
strategy_consistency: Dict[str, float] = field(default_factory=dict)
decision_diversity: float = 0.0
cooperation_score: float = 0.0
```

### Performance Optimization Metrics

```python
# In RunMetrics
llm_call_count: int = 0
cache_hit_rate: float = 0.0
retry_count: int = 0
avg_response_length: float = 0.0
```

## Accessing Metrics in Reports

Metrics are automatically included in:

1. **JSON report** - All metrics in `RunMetrics` are serialized
2. **Markdown report** - Key metrics shown in tables
3. **Console output** - Summary table with core metrics

### Customize Report Output

Edit `BenchmarkRunner.generate_reports()` in `tools/benchmark.py`:

```python
def _generate_markdown_report(self, output_path: Path):
    # ... existing code ...

    # Add custom metrics section
    f.write("\n## Economic Metrics\n\n")
    f.write("| Model | Wealth Inequality (Gini) | Market Efficiency |\n")
    f.write("|-------|--------------------------|-------------------|\n")
    for result in self.results:
        gini = result.wealth_inequality_gini
        efficiency = result.market_efficiency
        f.write(f"| {result.model_name} | {gini:.3f} | {efficiency:.3f} |\n")
```

## Advanced: Real-Time Metrics

To track metrics in real-time (during simulation):

```python
# In OrchestratorWithMetrics.run()

for step in range(1, self.max_steps + 1):
    # ... existing code ...

    # Add real-time metric tracking
    snapshot = self.game_engine.get_state_snapshot()

    # Example: Track score changes
    if step > 1:
        prev_snapshot = self.metrics_collector.metrics.variable_changes[-1]
        current_scores = snapshot['agent_vars']
        prev_scores = prev_snapshot['agent_vars']

        for agent_name in current_scores:
            score_change = (current_scores[agent_name]['score'] -
                          prev_scores[agent_name]['score'])
            # Log or track score_change
```

## Metric Categories Best Practices

### Performance Metrics
- Should be automatic (no scenario changes needed)
- Focus on speed, resource usage, cost
- Examples: time, tokens, memory, API calls

### Quality Metrics
- Measure simulation realism and coherence
- Check variable constraints
- Track logical consistency
- Examples: constraint violations, state validity

### Reliability Metrics
- Track failures and errors
- Completion rates
- Recovery attempts
- Examples: errors, retries, crashes

### Custom Metrics
- Domain-specific measurements
- Experiment-specific tracking
- Business/research metrics
- Examples: Gini coefficient, agent performance, strategy effectiveness

## Quick Reference

| Metric Type | Added Where | Code Changes | Complexity |
|-------------|-------------|--------------|------------|
| Track existing variable | Benchmark config | None | ⭐ Easy |
| Simple calculation | `end_step()` | Minimal | ⭐⭐ Medium |
| Cross-step analysis | `get_metrics()` | Medium | ⭐⭐⭐ Advanced |
| Per-step tracking | `StepMetrics` + `end_step()` | Moderate | ⭐⭐⭐ Advanced |
| Real-time metrics | `OrchestratorWithMetrics.run()` | Complex | ⭐⭐⭐⭐ Expert |

## Need Help?

1. **For simple metrics**: Use custom metrics in benchmark config
2. **For calculations**: Add to `get_metrics()` in `MetricsCollector`
3. **For complex analysis**: Extend `RunMetrics` and modify collection logic
4. **For visualization**: Export JSON and use external tools (pandas, matplotlib)
