"""
Metrics collection for benchmark runs.
Tracks performance, quality, and reliability metrics during simulation.
"""

import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime


@dataclass
class StepMetrics:
    """Metrics for a single simulation step."""
    step: int
    duration: float  # seconds
    tokens_used: Optional[int] = None
    errors: List[str] = field(default_factory=list)
    variable_changes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunMetrics:
    """Complete metrics for a simulation run."""
    model_name: str
    provider: str
    start_time: str
    end_time: Optional[str] = None

    # Performance metrics
    total_time: float = 0.0
    step_times: List[float] = field(default_factory=list)
    avg_step_time: float = 0.0
    total_tokens: Optional[int] = None
    avg_tokens_per_step: Optional[float] = None

    # Quality metrics
    variable_changes: List[Dict[str, Any]] = field(default_factory=list)
    final_state: Dict[str, Any] = field(default_factory=dict)
    decision_count: int = 0
    constraint_violations: int = 0

    # Reliability metrics
    completed: bool = False
    error_count: int = 0
    step_failures: List[int] = field(default_factory=list)
    error_messages: List[str] = field(default_factory=list)

    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    # Conversation transcript
    conversation: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)


class MetricsCollector:
    """Collects metrics during simulation run."""

    def __init__(self, model_name: str, provider: str, metrics_config: Optional[Dict] = None):
        """
        Initialize metrics collector.

        Args:
            model_name: Name of the model being benchmarked
            provider: Provider name (gemini, ollama, etc.)
            metrics_config: Configuration for which metrics to collect
        """
        self.metrics = RunMetrics(
            model_name=model_name,
            provider=provider,
            start_time=datetime.now().isoformat()
        )
        self.metrics_config = metrics_config or {}
        self.step_start_times: Dict[int, float] = {}

    def start_step(self, step: int):
        """Mark the start of a step."""
        self.step_start_times[step] = time.time()

    def end_step(self, step: int, state_snapshot: Optional[Dict] = None, errors: Optional[List[str]] = None):
        """
        Mark the end of a step and record metrics.

        Args:
            step: Step number
            state_snapshot: Current game state snapshot
            errors: Any errors that occurred during the step
        """
        if step not in self.step_start_times:
            return

        duration = time.time() - self.step_start_times[step]
        self.metrics.step_times.append(duration)

        # Track errors
        if errors:
            self.metrics.error_count += len(errors)
            self.metrics.error_messages.extend(errors)
            self.metrics.step_failures.append(step)

        # Track variable changes if quality metrics enabled
        if self._is_enabled('quality') and state_snapshot:
            changes = {
                'step': step,
                'global_vars': state_snapshot.get('global_vars', {}),
                'agent_vars': state_snapshot.get('agent_vars', {})
            }
            self.metrics.variable_changes.append(changes)

            # Check constraint violations
            violations = self._check_constraints(state_snapshot)
            self.metrics.constraint_violations += violations

    def record_final_state(self, final_snapshot: Dict):
        """Record final simulation state."""
        self.metrics.final_state = final_snapshot

        # Extract custom metrics if configured
        if self._is_enabled('custom'):
            self.metrics.custom_metrics = self._extract_custom_metrics(final_snapshot)

    def record_completion(self, completed: bool):
        """Record whether simulation completed successfully."""
        self.metrics.completed = completed
        self.metrics.end_time = datetime.now().isoformat()

        # Calculate aggregate metrics
        if self.metrics.step_times:
            self.metrics.total_time = sum(self.metrics.step_times)
            self.metrics.avg_step_time = self.metrics.total_time / len(self.metrics.step_times)

    def record_tokens(self, step: int, tokens: int):
        """Record token usage for a step (if available from provider)."""
        if not self._is_enabled('performance'):
            return

        if self.metrics.total_tokens is None:
            self.metrics.total_tokens = 0
        self.metrics.total_tokens += tokens

    def record_conversation_turn(self, step: int, agent_messages: Dict[str, str], agent_responses: Dict[str, str]):
        """
        Record a conversation turn in the simulation.

        Args:
            step: The step number
            agent_messages: Messages sent to each agent (from orchestrator/simulator)
            agent_responses: Responses from each agent
        """
        turn = {
            'step': step,
            'exchanges': []
        }

        for agent_name in agent_messages.keys():
            exchange = {
                'agent': agent_name,
                'message_to_agent': agent_messages.get(agent_name, ''),
                'response_from_agent': agent_responses.get(agent_name, '')
            }
            turn['exchanges'].append(exchange)

        self.metrics.conversation.append(turn)

    def get_metrics(self) -> RunMetrics:
        """Get the collected metrics."""
        # Calculate final aggregates
        if self.metrics.total_tokens is not None and self.metrics.step_times:
            self.metrics.avg_tokens_per_step = self.metrics.total_tokens / len(self.metrics.step_times)

        # Count total decisions (from variable changes)
        for change in self.metrics.variable_changes:
            agent_vars = change.get('agent_vars', {})
            for agent_data in agent_vars.values():
                if 'decisions_made' in agent_data:
                    self.metrics.decision_count += 1

        return self.metrics

    def _is_enabled(self, metric_category: str) -> bool:
        """Check if a metric category is enabled."""
        if not self.metrics_config:
            return True  # Default: all enabled

        category_config = self.metrics_config.get(metric_category, {})
        return category_config.get('enabled', True)

    def _check_constraints(self, state_snapshot: Dict) -> int:
        """Check for constraint violations in state snapshot."""
        violations = 0

        # This is a simplified check - would need schema info for full validation
        # For now, just check if values are within reasonable bounds
        global_vars = state_snapshot.get('global_vars', {})
        agent_vars = state_snapshot.get('agent_vars', {})

        # Check global vars (interest_rate, market_volatility should be 0-1)
        if 'interest_rate' in global_vars:
            val = global_vars['interest_rate']
            if val < 0 or val > 1:
                violations += 1

        if 'market_volatility' in global_vars:
            val = global_vars['market_volatility']
            if val < 0 or val > 1:
                violations += 1

        # Check agent vars (capital should be non-negative)
        for agent_data in agent_vars.values():
            if 'capital' in agent_data:
                if agent_data['capital'] < 0:
                    violations += 1
            if 'risk_tolerance' in agent_data:
                val = agent_data['risk_tolerance']
                if val < 0 or val > 1:
                    violations += 1

        return violations

    def _extract_custom_metrics(self, final_snapshot: Dict) -> Dict[str, Any]:
        """Extract custom metrics based on configuration."""
        custom_config = self.metrics_config.get('custom', {})
        track_vars = custom_config.get('track_variables', [])

        if not track_vars:
            return {}

        custom = {}
        global_vars = final_snapshot.get('global_vars', {})
        agent_vars = final_snapshot.get('agent_vars', {})

        for var_pattern in track_vars:
            # Simple pattern matching
            if var_pattern.startswith('*.'):
                # Agent variable pattern (e.g., "*.capital")
                var_name = var_pattern[2:]
                custom[var_pattern] = {}
                for agent_name, agent_data in agent_vars.items():
                    if var_name in agent_data:
                        custom[var_pattern][agent_name] = agent_data[var_name]
            else:
                # Global variable
                if var_pattern in global_vars:
                    custom[var_pattern] = global_vars[var_pattern]

        return custom
