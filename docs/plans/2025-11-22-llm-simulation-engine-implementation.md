# LLM-Powered Simulation Engine Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement SimulatorAgent - an LLM-powered game master that manages world state evolution, interprets agent responses into realistic stat changes, generates personalized narratives, and creates emergent/scripted events.

**Architecture:** Create `SimulatorAgent` class that uses `GameEngine` for state storage, calls LLM once per step to process agent responses and generate updates, validates responses with retry logic, applies constraint clamping with feedback, and maintains sliding history window. Simplify `GameEngine` to pure state management. Add `EngineValidator` for validation logic.

**Tech Stack:** Python 3.12, Pydantic for validation, existing LLM providers (Gemini/Ollama), pytest for testing

---

## Task 1: Add Engine Logging Codes

**Files:**
- Modify: `src/utils/logging_config.py:19-53`

**Step 1: Add engine-specific message codes**

Add after line 52 (after `PRF001`):

```python
    # Engine operations (ENG)
    ENG001 = "ENG001"  # SimulatorAgent initialized
    ENG002 = "ENG002"  # Processing step N
    ENG003 = "ENG003"  # Calling LLM
    ENG004 = "ENG004"  # LLM response received
    ENG005 = "ENG005"  # Validation successful
    ENG006 = "ENG006"  # Validation failed, retrying
    ENG007 = "ENG007"  # Retry attempt N
    ENG008 = "ENG008"  # Fatal error, stopping simulation
    ENG009 = "ENG009"  # Constraint hit (variable clamped)
    ENG010 = "ENG010"  # State updates applied
    ENG011 = "ENG011"  # Event generated
    ENG012 = "ENG012"  # Scripted event triggered
```

**Step 2: Commit**

```bash
git add src/utils/logging_config.py
git commit -m "feat: add engine logging codes (ENG001-ENG012)"
```

---

## Task 2: Create Data Models for SimulatorAgent

**Files:**
- Create: `src/core/engine_models.py`

**Step 1: Write test for data models**

Create `tests/unit/test_engine_models.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from core.engine_models import ScriptedEvent, EngineOutput, ConstraintHit, StepRecord


class TestScriptedEvent:
    def test_creation(self):
        event = ScriptedEvent(step=20, type="major_war", description="War begins")
        assert event.step == 20
        assert event.type == "major_war"
        assert event.description == "War begins"


class TestEngineOutput:
    def test_creation_minimal(self):
        output = EngineOutput(
            state_updates={"global_vars": {}, "agent_vars": {}},
            events=[],
            agent_messages={"Agent A": "Message"},
            reasoning="Test reasoning"
        )
        assert output.state_updates == {"global_vars": {}, "agent_vars": {}}
        assert output.events == []
        assert output.agent_messages == {"Agent A": "Message"}
        assert output.reasoning == "Test reasoning"

    def test_creation_with_updates(self):
        output = EngineOutput(
            state_updates={
                "global_vars": {"tension": 0.8},
                "agent_vars": {"Agent A": {"health": 95}}
            },
            events=[{"type": "crisis", "description": "Border dispute"}],
            agent_messages={"Agent A": "Tensions rise"},
            reasoning="Conflict escalates"
        )
        assert output.state_updates["global_vars"]["tension"] == 0.8
        assert output.state_updates["agent_vars"]["Agent A"]["health"] == 95
        assert len(output.events) == 1


class TestConstraintHit:
    def test_global_var_constraint(self):
        hit = ConstraintHit(
            agent_name=None,
            var_name="interest_rate",
            attempted_value=1.5,
            clamped_value=1.0,
            constraint_type="max"
        )
        assert hit.agent_name is None
        assert hit.var_name == "interest_rate"
        assert hit.attempted_value == 1.5
        assert hit.clamped_value == 1.0
        assert hit.constraint_type == "max"

    def test_agent_var_constraint(self):
        hit = ConstraintHit(
            agent_name="Agent A",
            var_name="economic_strength",
            attempted_value=-50.0,
            clamped_value=0.0,
            constraint_type="min"
        )
        assert hit.agent_name == "Agent A"
        assert hit.var_name == "economic_strength"


class TestStepRecord:
    def test_creation(self):
        record = StepRecord(
            step_number=5,
            changes={"Agent A": {"health": 95}},
            events=[{"type": "combat"}],
            agent_responses={"Agent A": "I defend"},
            reasoning="Combat occurred",
            constraint_hits=[]
        )
        assert record.step_number == 5
        assert record.changes["Agent A"]["health"] == 95
        assert len(record.events) == 1
        assert record.agent_responses["Agent A"] == "I defend"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_engine_models.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'core.engine_models'"

**Step 3: Implement data models**

Create `src/core/engine_models.py`:

```python
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class ScriptedEvent:
    """A guaranteed event that must occur at a specific step."""
    step: int
    type: str
    description: str


@dataclass
class EngineOutput:
    """Output from SimulatorAgent LLM call."""
    state_updates: Dict[str, Any]  # {global_vars: {...}, agent_vars: {agent_name: {...}}}
    events: List[Dict[str, Any]]
    agent_messages: Dict[str, str]  # {agent_name: message}
    reasoning: str


@dataclass
class ConstraintHit:
    """Record of a variable hitting min/max constraint."""
    agent_name: Optional[str]  # None for global vars
    var_name: str
    attempted_value: Union[int, float]
    clamped_value: Union[int, float]
    constraint_type: str  # "min" or "max"


@dataclass
class StepRecord:
    """Record of a single simulation step for history window."""
    step_number: int
    changes: Dict[str, Any]  # Only changed variables
    events: List[Dict[str, Any]]
    agent_responses: Dict[str, str]
    reasoning: str
    constraint_hits: List[ConstraintHit]
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_engine_models.py -v
```

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/core/engine_models.py tests/unit/test_engine_models.py
git commit -m "feat: add data models for SimulatorAgent"
```

---

## Task 3: Implement EngineValidator

**Files:**
- Create: `src/core/engine_validator.py`
- Create: `tests/unit/test_engine_validator.py`

**Step 1: Write tests for validation logic**

Create `tests/unit/test_engine_validator.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import pytest
from core.engine_validator import EngineValidator, ValidationResult


class TestValidateStructure:
    def test_valid_json(self):
        valid_response = json.dumps({
            "state_updates": {"global_vars": {}, "agent_vars": {}},
            "events": [],
            "agent_messages": {},
            "reasoning": "Test"
        })
        result = EngineValidator.validate_structure(valid_response)
        assert result.success is True
        assert result.error is None

    def test_invalid_json(self):
        result = EngineValidator.validate_structure("{invalid json")
        assert result.success is False
        assert "JSON" in result.error

    def test_missing_required_key(self):
        missing_key = json.dumps({
            "state_updates": {},
            "events": []
            # Missing agent_messages and reasoning
        })
        result = EngineValidator.validate_structure(missing_key)
        assert result.success is False
        assert "missing" in result.error.lower()


class TestValidateReferences:
    def test_valid_references(self):
        output = {
            "state_updates": {
                "global_vars": {"tension": 0.8},
                "agent_vars": {"Agent A": {"health": 95}}
            },
            "events": [],
            "agent_messages": {"Agent A": "Message"},
            "reasoning": "Test"
        }
        agents = ["Agent A", "Agent B"]
        global_vars = {"tension": {}}
        agent_vars = {"health": {}}

        result = EngineValidator.validate_references(output, agents, global_vars, agent_vars)
        assert result.success is True

    def test_unknown_agent(self):
        output = {
            "state_updates": {"global_vars": {}, "agent_vars": {"Unknown": {}}},
            "events": [],
            "agent_messages": {"Agent A": "Message"},
            "reasoning": "Test"
        }
        result = EngineValidator.validate_references(output, ["Agent A"], {}, {})
        assert result.success is False
        assert "Unknown" in result.error

    def test_unknown_variable(self):
        output = {
            "state_updates": {"global_vars": {"undefined_var": 1.0}, "agent_vars": {}},
            "events": [],
            "agent_messages": {},
            "reasoning": "Test"
        }
        result = EngineValidator.validate_references(output, [], {}, {})
        assert result.success is False
        assert "undefined_var" in result.error


class TestValidateTypes:
    def test_valid_types(self):
        output = {
            "state_updates": {
                "global_vars": {"tension": 0.8},
                "agent_vars": {"Agent A": {"health": 95}}
            },
            "events": [],
            "agent_messages": {},
            "reasoning": "Test"
        }
        global_defs = {"tension": {"type": "float"}}
        agent_defs = {"health": {"type": "int"}}

        result = EngineValidator.validate_types(output, global_defs, agent_defs)
        assert result.success is True

    def test_type_mismatch(self):
        output = {
            "state_updates": {"global_vars": {"count": "not_an_int"}, "agent_vars": {}},
            "events": [],
            "agent_messages": {},
            "reasoning": "Test"
        }
        global_defs = {"count": {"type": "int"}}

        result = EngineValidator.validate_types(output, global_defs, {})
        assert result.success is False
        assert "type" in result.error.lower()


class TestApplyConstraints:
    def test_no_constraints_violated(self):
        updates = {"global_vars": {"value": 50}, "agent_vars": {}}
        global_defs = {"value": {"type": "int", "min": 0, "max": 100}}

        clamped, hits = EngineValidator.apply_constraints(updates, global_defs, {})
        assert clamped == updates
        assert len(hits) == 0

    def test_clamp_to_min(self):
        updates = {"global_vars": {"value": -10}, "agent_vars": {}}
        global_defs = {"value": {"type": "int", "min": 0, "max": 100}}

        clamped, hits = EngineValidator.apply_constraints(updates, global_defs, {})
        assert clamped["global_vars"]["value"] == 0
        assert len(hits) == 1
        assert hits[0].constraint_type == "min"
        assert hits[0].attempted_value == -10
        assert hits[0].clamped_value == 0

    def test_clamp_to_max(self):
        updates = {"global_vars": {}, "agent_vars": {"Agent A": {"health": 150}}}
        agent_defs = {"health": {"type": "int", "min": 0, "max": 100}}

        clamped, hits = EngineValidator.apply_constraints(updates, {}, agent_defs)
        assert clamped["agent_vars"]["Agent A"]["health"] == 100
        assert len(hits) == 1
        assert hits[0].agent_name == "Agent A"
        assert hits[0].constraint_type == "max"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_engine_validator.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'core.engine_validator'"

**Step 3: Implement EngineValidator**

Create `src/core/engine_validator.py`:

```python
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional
from core.engine_models import ConstraintHit


@dataclass
class ValidationResult:
    """Result of validation operation."""
    success: bool
    error: Optional[str] = None


class EngineValidator:
    """Validates SimulatorAgent LLM output against schema and constraints."""

    @staticmethod
    def validate_structure(response: str) -> ValidationResult:
        """Validate JSON structure and required keys."""
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            return ValidationResult(success=False, error=f"Invalid JSON: {e}")

        required_keys = ["state_updates", "events", "agent_messages", "reasoning"]
        missing = [key for key in required_keys if key not in data]

        if missing:
            return ValidationResult(
                success=False,
                error=f"Missing required keys: {', '.join(missing)}"
            )

        # Validate state_updates structure
        if not isinstance(data["state_updates"], dict):
            return ValidationResult(success=False, error="state_updates must be a dict")

        if "global_vars" not in data["state_updates"]:
            return ValidationResult(success=False, error="state_updates missing global_vars")

        if "agent_vars" not in data["state_updates"]:
            return ValidationResult(success=False, error="state_updates missing agent_vars")

        return ValidationResult(success=True)

    @staticmethod
    def validate_references(
        output: Dict[str, Any],
        agents: List[str],
        global_var_defs: Dict[str, Any],
        agent_var_defs: Dict[str, Any]
    ) -> ValidationResult:
        """Validate agent names and variable names exist."""
        # Check agent names in state_updates
        for agent_name in output["state_updates"]["agent_vars"].keys():
            if agent_name not in agents:
                return ValidationResult(
                    success=False,
                    error=f"Unknown agent '{agent_name}' in state_updates"
                )

        # Check agent names in agent_messages
        for agent_name in output["agent_messages"].keys():
            if agent_name not in agents:
                return ValidationResult(
                    success=False,
                    error=f"Unknown agent '{agent_name}' in agent_messages"
                )

        # Check global variable names
        for var_name in output["state_updates"]["global_vars"].keys():
            if var_name not in global_var_defs:
                return ValidationResult(
                    success=False,
                    error=f"Unknown global variable '{var_name}'"
                )

        # Check agent variable names
        for agent_name, vars_dict in output["state_updates"]["agent_vars"].items():
            for var_name in vars_dict.keys():
                if var_name not in agent_var_defs:
                    return ValidationResult(
                        success=False,
                        error=f"Unknown agent variable '{var_name}' for agent '{agent_name}'"
                    )

        return ValidationResult(success=True)

    @staticmethod
    def validate_types(
        output: Dict[str, Any],
        global_var_defs: Dict[str, Any],
        agent_var_defs: Dict[str, Any]
    ) -> ValidationResult:
        """Validate variable values match defined types."""
        # Validate global variables
        for var_name, value in output["state_updates"]["global_vars"].items():
            expected_type = global_var_defs[var_name]["type"]
            if not EngineValidator._check_type(value, expected_type):
                return ValidationResult(
                    success=False,
                    error=f"Global var '{var_name}': expected {expected_type}, got {type(value).__name__}"
                )

        # Validate agent variables
        for agent_name, vars_dict in output["state_updates"]["agent_vars"].items():
            for var_name, value in vars_dict.items():
                expected_type = agent_var_defs[var_name]["type"]
                if not EngineValidator._check_type(value, expected_type):
                    return ValidationResult(
                        success=False,
                        error=f"Agent var '{var_name}' for '{agent_name}': expected {expected_type}, got {type(value).__name__}"
                    )

        return ValidationResult(success=True)

    @staticmethod
    def _check_type(value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        if expected_type == "int":
            return isinstance(value, int) and not isinstance(value, bool)
        elif expected_type == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        elif expected_type == "bool":
            return isinstance(value, bool)
        elif expected_type == "list":
            return isinstance(value, list)
        elif expected_type == "dict":
            return isinstance(value, dict)
        return False

    @staticmethod
    def apply_constraints(
        state_updates: Dict[str, Any],
        global_var_defs: Dict[str, Any],
        agent_var_defs: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[ConstraintHit]]:
        """Apply min/max constraints, return clamped values and hits."""
        clamped_updates = {
            "global_vars": {},
            "agent_vars": {}
        }
        constraint_hits = []

        # Process global variables
        for var_name, value in state_updates["global_vars"].items():
            var_def = global_var_defs[var_name]
            clamped_value, hit = EngineValidator._clamp_value(
                value, var_def, agent_name=None, var_name=var_name
            )
            clamped_updates["global_vars"][var_name] = clamped_value
            if hit:
                constraint_hits.append(hit)

        # Process agent variables
        for agent_name, vars_dict in state_updates["agent_vars"].items():
            clamped_updates["agent_vars"][agent_name] = {}
            for var_name, value in vars_dict.items():
                var_def = agent_var_defs[var_name]
                clamped_value, hit = EngineValidator._clamp_value(
                    value, var_def, agent_name=agent_name, var_name=var_name
                )
                clamped_updates["agent_vars"][agent_name][var_name] = clamped_value
                if hit:
                    constraint_hits.append(hit)

        return clamped_updates, constraint_hits

    @staticmethod
    def _clamp_value(
        value: Any,
        var_def: Dict[str, Any],
        agent_name: Optional[str],
        var_name: str
    ) -> Tuple[Any, Optional[ConstraintHit]]:
        """Clamp a value to constraints, return clamped value and hit if any."""
        var_type = var_def["type"]

        # Only numeric types have constraints
        if var_type not in ("int", "float"):
            return value, None

        min_val = var_def.get("min")
        max_val = var_def.get("max")

        # Check min constraint
        if min_val is not None and value < min_val:
            hit = ConstraintHit(
                agent_name=agent_name,
                var_name=var_name,
                attempted_value=value,
                clamped_value=min_val,
                constraint_type="min"
            )
            return min_val, hit

        # Check max constraint
        if max_val is not None and value > max_val:
            hit = ConstraintHit(
                agent_name=agent_name,
                var_name=var_name,
                attempted_value=value,
                clamped_value=max_val,
                constraint_type="max"
            )
            return max_val, hit

        return value, None
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_engine_validator.py -v
```

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/core/engine_validator.py tests/unit/test_engine_validator.py
git commit -m "feat: add EngineValidator for LLM response validation"
```

---

## Task 4: Simplify GameEngine to State Management Only

**Files:**
- Modify: `src/core/game_engine.py`
- Modify: `tests/unit/test_game_engine.py`

**Step 1: Update tests for simplified GameEngine**

Replace content of `tests/unit/test_game_engine.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import pytest
from core.game_engine import GameEngine


class TestGameEngine:
    """Tests for simplified GameEngine (state management only)."""

    def test_initialization_default(self):
        """Test game engine initialization with minimal config."""
        config = {
            "global_vars": {},
            "agent_vars": {}
        }
        engine = GameEngine(config)
        assert engine.state.resources == 100
        assert engine.state.difficulty == "normal"

    def test_get_global_var(self):
        """Test getting global variable."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {}
        }
        engine = GameEngine(config)
        assert engine.get_global_var("tension") == 0.5

    def test_set_global_var(self):
        """Test setting global variable."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {}
        }
        engine = GameEngine(config)
        engine.set_global_var("tension", 0.8)
        assert engine.get_global_var("tension") == 0.8

    def test_get_agent_var(self):
        """Test getting agent variable."""
        config = {
            "global_vars": {},
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        engine = GameEngine(config)
        # Initialize agent
        engine.initialize_agent("Agent A", {})
        assert engine.get_agent_var("Agent A", "health") == 100

    def test_set_agent_var(self):
        """Test setting agent variable."""
        config = {
            "global_vars": {},
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        engine = GameEngine(config)
        engine.initialize_agent("Agent A", {})
        engine.set_agent_var("Agent A", "health", 95)
        assert engine.get_agent_var("Agent A", "health") == 95

    def test_apply_state_updates(self):
        """Test applying state updates from SimulatorAgent."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        engine = GameEngine(config)
        engine.initialize_agent("Agent A", {})

        updates = {
            "global_vars": {"tension": 0.8},
            "agent_vars": {"Agent A": {"health": 95}}
        }

        engine.apply_state_updates(updates)

        assert engine.get_global_var("tension") == 0.8
        assert engine.get_agent_var("Agent A", "health") == 95

    def test_get_current_state(self):
        """Test getting current state snapshot."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5}
            },
            "agent_vars": {
                "health": {"type": "int", "default": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        engine = GameEngine(config)
        engine.initialize_agent("Agent A", {})

        state = engine.get_current_state()

        assert "global_vars" in state
        assert "agent_vars" in state
        assert state["global_vars"]["tension"] == 0.5
        assert state["agent_vars"]["Agent A"]["health"] == 100
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_game_engine.py -v
```

Expected: FAIL (methods don't exist yet)

**Step 3: Simplify GameEngine implementation**

Replace content of `src/core/game_engine.py`:

```python
from typing import Any, Dict, Optional
from core.state import GameState
from utils.config_parser import create_variable_set, create_variable_set_with_overrides, validate_config


class GameEngine:
    """Manages game state - pure state management, no simulation logic."""

    def __init__(self, config: dict):
        # Validate config before using it
        validate_config(config)

        self.config = config
        self.global_var_definitions = config.get("global_vars", {})
        self.agent_var_definitions = config.get("agent_vars", {})
        self.agent_configs = {agent["name"]: agent for agent in config.get("agents", [])}
        self.state = self._initialize_state()

    def _initialize_state(self) -> GameState:
        """Initialize the game state from configuration."""
        game_config = self.config.get("game_state", {})

        # Create global variables from config
        global_vars = create_variable_set(self.config.get("global_vars"))

        return GameState(
            resources=game_config.get("initial_resources", 100),
            difficulty=game_config.get("difficulty", "normal"),
            variables=global_vars,
        )

    def initialize_agent(self, agent_name: str, variable_overrides: Optional[Dict[str, Any]] = None):
        """Initialize an agent's state with variables."""
        if self.state.get_agent(agent_name) is not None:
            return  # Already initialized

        agent_config = self.agent_configs.get(agent_name, {})
        overrides = variable_overrides or agent_config.get("variables")

        self.state.add_agent(
            agent_name,
            variables=create_variable_set_with_overrides(
                self.config.get("agent_vars"),
                overrides
            )
        )

    def get_global_var(self, var_name: str) -> Any:
        """Get global variable value."""
        return self.state.get_var(var_name)

    def set_global_var(self, var_name: str, value: Any):
        """Set global variable with type validation."""
        self.state.set_var(var_name, value)

    def get_agent_var(self, agent_name: str, var_name: str) -> Any:
        """Get agent variable value."""
        agent_state = self.state.get_agent(agent_name)
        if agent_state is None:
            raise ValueError(f"Agent '{agent_name}' not initialized")
        return agent_state.get_var(var_name)

    def set_agent_var(self, agent_name: str, var_name: str, value: Any):
        """Set agent variable with type validation."""
        agent_state = self.state.get_agent(agent_name)
        if agent_state is None:
            raise ValueError(f"Agent '{agent_name}' not initialized")
        agent_state.set_var(var_name, value)

    def apply_state_updates(self, updates: Dict[str, Any]):
        """Apply state updates from SimulatorAgent."""
        # Apply global variable updates
        for var_name, value in updates.get("global_vars", {}).items():
            self.set_global_var(var_name, value)

        # Apply agent variable updates
        for agent_name, vars_dict in updates.get("agent_vars", {}).items():
            # Ensure agent is initialized
            if self.state.get_agent(agent_name) is None:
                self.initialize_agent(agent_name)

            for var_name, value in vars_dict.items():
                self.set_agent_var(agent_name, var_name, value)

    def get_current_state(self) -> Dict[str, Any]:
        """Get current state snapshot."""
        global_vars = self.state.variables.to_dict()

        agent_vars = {}
        for agent_name, agent_state in self.state.agents.items():
            agent_vars[agent_name] = agent_state.variables.to_dict()

        return {
            "global_vars": global_vars,
            "agent_vars": agent_vars
        }

    def get_state(self) -> GameState:
        """Get the full GameState object."""
        return self.state

    def advance_round(self):
        """Advance to the next round."""
        self.state.advance_round()

    def get_state_snapshot(self) -> dict:
        """Get a snapshot for persistence (backward compatibility)."""
        game_state = {
            "resources": self.state.resources,
            "difficulty": self.state.difficulty,
            "round": self.state.round
        }

        global_vars = self.state.variables.to_dict()

        agent_vars = {}
        for agent_name, agent_state in self.state.agents.items():
            agent_vars[agent_name] = agent_state.variables.to_dict()

        return {
            "game_state": game_state,
            "global_vars": global_vars,
            "agent_vars": agent_vars
        }
```

**Step 4: Run test to verify it passes**

```bash
pytest tests/unit/test_game_engine.py -v
```

Expected: PASS (all tests)

**Step 5: Commit**

```bash
git add src/core/game_engine.py tests/unit/test_game_engine.py
git commit -m "refactor: simplify GameEngine to state management only"
```

---

## Task 5: Implement SimulatorAgent (Core Logic)

**Files:**
- Create: `src/core/simulator_agent.py`
- Create: `tests/unit/test_simulator_agent.py`

**Step 1: Write tests for SimulatorAgent**

Create `tests/unit/test_simulator_agent.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import pytest
from core.simulator_agent import SimulatorAgent
from core.game_engine import GameEngine
from llm.llm_provider import LLMProvider


class MockSimulatorLLM(LLMProvider):
    """Mock LLM for testing SimulatorAgent."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0
        self.last_prompt = None

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        self.last_prompt = prompt
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return json.dumps(response)


class TestSimulatorAgent:
    """Tests for SimulatorAgent."""

    def test_initialization(self):
        """Test SimulatorAgent initialization."""
        config = {
            "global_vars": {},
            "agent_vars": {},
            "agents": []
        }
        game_engine = GameEngine(config)
        mock_llm = MockSimulatorLLM([])

        agent = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test prompt",
            simulation_plan="Test plan",
            realism_guidelines="Test guidelines",
            scripted_events=[],
            context_window_size=5
        )

        assert agent.llm_provider == mock_llm
        assert agent.game_engine == game_engine
        assert agent.context_window_size == 5
        assert len(agent.step_history) == 0

    def test_initialize_simulation(self):
        """Test simulation initialization."""
        config = {
            "global_vars": {},
            "agent_vars": {},
            "agents": [{"name": "Agent A"}, {"name": "Agent B"}]
        }
        game_engine = GameEngine(config)

        mock_response = {
            "state_updates": {"global_vars": {}, "agent_vars": {}},
            "events": [],
            "agent_messages": {
                "Agent A": "Welcome Agent A",
                "Agent B": "Welcome Agent B"
            },
            "reasoning": "Initial setup"
        }
        mock_llm = MockSimulatorLLM([mock_response])

        agent = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[]
        )

        messages = agent.initialize_simulation(["Agent A", "Agent B"])

        assert "Agent A" in messages
        assert "Agent B" in messages
        assert messages["Agent A"] == "Welcome Agent A"
        assert messages["Agent B"] == "Welcome Agent B"

    def test_process_step_basic(self):
        """Test processing a single step."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }
        game_engine = GameEngine(config)
        game_engine.initialize_agent("Agent A", {})

        mock_response = {
            "state_updates": {
                "global_vars": {"tension": 0.6},
                "agent_vars": {"Agent A": {"health": 95}}
            },
            "events": [{"type": "combat", "description": "Minor skirmish"}],
            "agent_messages": {"Agent A": "You took damage"},
            "reasoning": "Combat occurred"
        }
        mock_llm = MockSimulatorLLM([mock_response])

        agent = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[]
        )

        agent_responses = {"Agent A": "I attack"}
        messages = agent.process_step(1, agent_responses)

        assert messages["Agent A"] == "You took damage"
        assert game_engine.get_global_var("tension") == 0.6
        assert game_engine.get_agent_var("Agent A", "health") == 95
        assert len(agent.step_history) == 1

    def test_context_window_limiting(self):
        """Test that history is limited to window size."""
        config = {
            "global_vars": {},
            "agent_vars": {},
            "agents": [{"name": "Agent A"}]
        }
        game_engine = GameEngine(config)
        game_engine.initialize_agent("Agent A", {})

        mock_response = {
            "state_updates": {"global_vars": {}, "agent_vars": {}},
            "events": [],
            "agent_messages": {"Agent A": "Message"},
            "reasoning": "Step"
        }
        mock_llm = MockSimulatorLLM([mock_response] * 10)

        agent = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[],
            context_window_size=3
        )

        # Process 10 steps
        for i in range(1, 11):
            agent.process_step(i, {"Agent A": "Response"})

        # Should only keep last 3 steps
        assert len(agent.step_history) == 3
        assert agent.step_history[0].step_number == 8
        assert agent.step_history[-1].step_number == 10
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_simulator_agent.py -v
```

Expected: FAIL with "ModuleNotFoundError: No module named 'core.simulator_agent'"

**Step 3: Implement SimulatorAgent (part 1 - basic structure)**

Create `src/core/simulator_agent.py`:

```python
import json
from typing import Any, Dict, List, Optional
from llm.llm_provider import LLMProvider
from core.game_engine import GameEngine
from core.engine_models import ScriptedEvent, EngineOutput, ConstraintHit, StepRecord
from core.engine_validator import EngineValidator, ValidationResult
from utils.logging_config import MessageCode


class SimulationError(Exception):
    """Fatal error in simulation that requires stopping."""
    pass


class SimulatorAgent:
    """LLM-powered simulation orchestrator - the game master."""

    def __init__(
        self,
        llm_provider: LLMProvider,
        game_engine: GameEngine,
        system_prompt: str,
        simulation_plan: str,
        realism_guidelines: str,
        scripted_events: List[ScriptedEvent],
        context_window_size: int = 5,
        logger=None
    ):
        self.llm_provider = llm_provider
        self.game_engine = game_engine
        self.system_prompt = system_prompt
        self.simulation_plan = simulation_plan
        self.realism_guidelines = realism_guidelines
        self.scripted_events = scripted_events
        self.context_window_size = context_window_size
        self.logger = logger

        self.step_history: List[StepRecord] = []
        self.constraint_feedback: List[ConstraintHit] = []

        if self.logger:
            self.logger.info(
                MessageCode.ENG001,
                "SimulatorAgent initialized",
                context_window_size=context_window_size,
                num_scripted_events=len(scripted_events)
            )

    def initialize_simulation(self, agent_names: List[str]) -> Dict[str, str]:
        """Generate initial agent messages for Step 0."""
        current_state = self.game_engine.get_current_state()

        prompt = self._build_initialization_prompt(agent_names, current_state)

        # Call LLM with retry logic
        output = self._call_llm_with_retry(prompt, step_number=0, agent_names=agent_names)

        # Apply state updates
        self.game_engine.apply_state_updates(output.state_updates)

        return output.agent_messages

    def process_step(
        self,
        step_number: int,
        agent_responses: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Process agent responses, update world state, generate new messages.
        Returns: agent_messages for next step
        """
        if self.logger:
            self.logger.info(MessageCode.ENG002, "Processing step", step=step_number)

        current_state = self.game_engine.get_current_state()
        agent_names = list(agent_responses.keys())

        # Build context
        prompt = self._build_step_prompt(step_number, current_state, agent_responses)

        # Call LLM with retry logic
        output = self._call_llm_with_retry(prompt, step_number, agent_names)

        # Apply constraints and get clamped updates
        clamped_updates, constraint_hits = EngineValidator.apply_constraints(
            output.state_updates,
            self.game_engine.global_var_definitions,
            self.game_engine.agent_var_definitions
        )

        # Log constraint hits
        for hit in constraint_hits:
            if self.logger:
                self.logger.warning(
                    MessageCode.ENG009,
                    "Constraint hit",
                    step=step_number,
                    agent=hit.agent_name,
                    var=hit.var_name,
                    attempted=hit.attempted_value,
                    clamped=hit.clamped_value,
                    constraint_type=hit.constraint_type
                )

        # Apply clamped updates to game engine
        self.game_engine.apply_state_updates(clamped_updates)

        if self.logger:
            self.logger.info(
                MessageCode.ENG010,
                "State updates applied",
                step=step_number,
                num_global_changes=len(clamped_updates["global_vars"]),
                num_agent_changes=sum(len(v) for v in clamped_updates["agent_vars"].values())
            )

        # Record step in history
        self._record_step(step_number, clamped_updates, output.events, agent_responses, output.reasoning, constraint_hits)

        # Store constraint hits for next step's context
        self.constraint_feedback = constraint_hits

        return output.agent_messages

    def _call_llm_with_retry(
        self,
        prompt: str,
        step_number: int,
        agent_names: List[str],
        max_attempts: int = 3
    ) -> EngineOutput:
        """Call LLM with validation and retry logic."""
        attempt = 1
        last_error = None

        while attempt <= max_attempts:
            if self.logger and attempt > 1:
                self.logger.info(
                    MessageCode.ENG007,
                    f"Retry attempt {attempt}/{max_attempts}",
                    step=step_number
                )

            try:
                if self.logger:
                    self.logger.debug(MessageCode.ENG003, "Calling LLM", step=step_number)

                response_text = self.llm_provider.generate_response(
                    prompt=prompt,
                    system_prompt=self.system_prompt
                )

                if self.logger:
                    self.logger.debug(
                        MessageCode.ENG004,
                        "LLM response received",
                        step=step_number,
                        response_length=len(response_text)
                    )

                # Validate response
                validation_result = self._validate_response(response_text, agent_names)

                if validation_result.success:
                    output = self._parse_response(response_text)

                    if self.logger:
                        self.logger.info(
                            MessageCode.ENG005,
                            "Validation successful",
                            step=step_number
                        )

                    return output
                else:
                    last_error = validation_result.error

                    if self.logger:
                        self.logger.warning(
                            MessageCode.ENG006,
                            "Validation failed",
                            step=step_number,
                            error=validation_result.error,
                            attempt=attempt
                        )

                    # Add error feedback to prompt for retry
                    prompt = self._add_error_feedback(prompt, validation_result.error)

            except Exception as e:
                last_error = str(e)

                if self.logger:
                    self.logger.error(
                        MessageCode.ENG006,
                        "LLM call failed",
                        step=step_number,
                        error=str(e),
                        attempt=attempt
                    )

            attempt += 1

        # Max attempts exceeded
        error_msg = (
            f"SimulatorAgent failed after {max_attempts} attempts at step {step_number}.\n"
            f"Last error: {last_error}"
        )

        if self.logger:
            self.logger.critical(MessageCode.ENG008, error_msg, step=step_number)

        raise SimulationError(error_msg)

    def _validate_response(self, response_text: str, agent_names: List[str]) -> ValidationResult:
        """Validate LLM response."""
        # Step 1: Structure validation
        result = EngineValidator.validate_structure(response_text)
        if not result.success:
            return result

        # Parse JSON
        output = json.loads(response_text)

        # Step 2: Reference validation
        result = EngineValidator.validate_references(
            output,
            agent_names,
            self.game_engine.global_var_definitions,
            self.game_engine.agent_var_definitions
        )
        if not result.success:
            return result

        # Step 3: Type validation
        result = EngineValidator.validate_types(
            output,
            self.game_engine.global_var_definitions,
            self.game_engine.agent_var_definitions
        )
        if not result.success:
            return result

        return ValidationResult(success=True)

    def _parse_response(self, response_text: str) -> EngineOutput:
        """Parse validated response into EngineOutput."""
        data = json.loads(response_text)
        return EngineOutput(
            state_updates=data["state_updates"],
            events=data["events"],
            agent_messages=data["agent_messages"],
            reasoning=data["reasoning"]
        )

    def _record_step(
        self,
        step_number: int,
        changes: Dict[str, Any],
        events: List[Dict[str, Any]],
        agent_responses: Dict[str, str],
        reasoning: str,
        constraint_hits: List[ConstraintHit]
    ):
        """Record step in history with sliding window."""
        record = StepRecord(
            step_number=step_number,
            changes=changes,
            events=events,
            agent_responses=agent_responses,
            reasoning=reasoning,
            constraint_hits=constraint_hits
        )

        self.step_history.append(record)

        # Maintain window size
        if len(self.step_history) > self.context_window_size:
            self.step_history.pop(0)

    def _build_initialization_prompt(self, agent_names: List[str], current_state: Dict[str, Any]) -> str:
        """Build prompt for step 0 initialization."""
        # Implementation in next step
        pass

    def _build_step_prompt(self, step_number: int, current_state: Dict[str, Any], agent_responses: Dict[str, str]) -> str:
        """Build prompt for regular step processing."""
        # Implementation in next step
        pass

    def _add_error_feedback(self, prompt: str, error: str) -> str:
        """Add error feedback to prompt for retry."""
        return f"{prompt}\n\nERROR: Your previous response was invalid: {error}\nPlease correct and try again."
```

**Step 4: Run test to verify basic tests pass**

```bash
pytest tests/unit/test_simulator_agent.py::TestSimulatorAgent::test_initialization -v
```

Expected: PASS

**Step 5: Commit (partial implementation)**

```bash
git add src/core/simulator_agent.py tests/unit/test_simulator_agent.py
git commit -m "feat: add SimulatorAgent core structure and validation"
```

---

## Task 6: Implement SimulatorAgent Prompt Building

**Files:**
- Modify: `src/core/simulator_agent.py:200-210`

**Step 1: Implement prompt building methods**

Replace the placeholder methods in `src/core/simulator_agent.py` (lines ~200-210):

```python
    def _build_initialization_prompt(self, agent_names: List[str], current_state: Dict[str, Any]) -> str:
        """Build prompt for step 0 initialization."""
        prompt = f"""=== SIMULATION SETUP ===
{self.system_prompt}

Simulation Plan:
{self.simulation_plan}

Realism Guidelines:
{self.realism_guidelines}

=== INITIAL STATE ===
Global Variables:
{self._format_variables(current_state["global_vars"])}

Agents: {', '.join(agent_names)}

=== YOUR TASK ===
Initialize the simulation for Step 0.
Generate initial personalized messages for each agent to begin the simulation.

Return JSON format:
{{
  "state_updates": {{
    "global_vars": {{}},
    "agent_vars": {{}}
  }},
  "events": [],
  "agent_messages": {{
    "Agent A": "Initial message for Agent A",
    "Agent B": "Initial message for Agent B"
  }},
  "reasoning": "Why you generated these initial messages"
}}

IMPORTANT: Include ALL agents in agent_messages.
"""
        return prompt

    def _build_step_prompt(self, step_number: int, current_state: Dict[str, Any], agent_responses: Dict[str, str]) -> str:
        """Build prompt for regular step processing."""
        sections = []

        # Setup section
        sections.append(f"""=== SIMULATION SETUP ===
{self.system_prompt}

Simulation Plan:
{self.simulation_plan}

Realism Guidelines:
{self.realism_guidelines}""")

        # Scripted events
        if self.scripted_events:
            upcoming = [e for e in self.scripted_events if e.step >= step_number]
            if upcoming:
                sections.append("=== UPCOMING SCRIPTED EVENTS ===")
                for event in upcoming[:5]:  # Show next 5 events
                    sections.append(f"Step {event.step}: {event.type} - {event.description}")

        # Current state
        sections.append(f"""=== CURRENT STATE (Step {step_number}) ===
Global Variables:
{self._format_variables(current_state["global_vars"])}

Agent Variables:""")
        for agent_name, vars_dict in current_state["agent_vars"].items():
            sections.append(f"  {agent_name}:")
            sections.append(f"{self._format_variables(vars_dict, indent='    ')}")

        # Recent history
        if self.step_history:
            sections.append(f"=== RECENT HISTORY (Last {len(self.step_history)} steps) ===")
            for record in self.step_history:
                sections.append(f"\nStep {record.step_number}:")

                # Changes (delta only)
                if record.changes["global_vars"] or record.changes["agent_vars"]:
                    sections.append("  Changes:")
                    for var_name, value in record.changes["global_vars"].items():
                        sections.append(f"    Global: {var_name} = {value}")
                    for agent_name, vars_dict in record.changes["agent_vars"].items():
                        for var_name, value in vars_dict.items():
                            sections.append(f"    {agent_name}: {var_name} = {value}")

                # Events
                if record.events:
                    sections.append("  Events:")
                    for event in record.events:
                        sections.append(f"    - {event}")

                # Agent responses
                sections.append("  Agent Responses:")
                for agent_name, response in record.agent_responses.items():
                    sections.append(f"    {agent_name}: \"{response}\"")

                # Reasoning
                sections.append(f"  Reasoning: {record.reasoning}")

                # Constraint hits
                if record.constraint_hits:
                    sections.append("  Constraint Hits:")
                    for hit in record.constraint_hits:
                        agent_str = hit.agent_name or "Global"
                        sections.append(
                            f"    {agent_str} {hit.var_name}: attempted {hit.attempted_value}, "
                            f"clamped to {hit.clamped_value} ({hit.constraint_type})"
                        )

        # Task
        sections.append(f"""=== YOUR TASK ===
Process the agent responses from Step {step_number - 1 if step_number > 1 else 0}.
Update world state realistically for Step {step_number}.
Generate personalized messages for each agent.

Agent Responses from previous step:""")
        for agent_name, response in agent_responses.items():
            sections.append(f"  {agent_name}: \"{response}\"")

        sections.append("""
Return JSON format:
{
  "state_updates": {
    "global_vars": {"var_name": new_value, ...},
    "agent_vars": {
      "Agent A": {"var_name": new_value, ...},
      "Agent B": {...}
    }
  },
  "events": [
    {
      "type": "event_type",
      "description": "What happened",
      "affects": ["Agent A"],
      "duration": 3
    }
  ],
  "agent_messages": {
    "Agent A": "Personalized narrative for Agent A",
    "Agent B": "Personalized narrative for Agent B"
  },
  "reasoning": "Why you made these updates and generated these events"
}

IMPORTANT: Only include variables that CHANGED. Omit unchanged variables from state_updates.
""")

        return "\n".join(sections)

    def _format_variables(self, vars_dict: Dict[str, Any], indent: str = "  ") -> str:
        """Format variables for prompt."""
        if not vars_dict:
            return f"{indent}(none)"
        lines = []
        for name, value in vars_dict.items():
            lines.append(f"{indent}{name}: {value}")
        return "\n".join(lines)
```

**Step 2: Run all SimulatorAgent tests**

```bash
pytest tests/unit/test_simulator_agent.py -v
```

Expected: PASS (all tests)

**Step 3: Commit**

```bash
git add src/core/simulator_agent.py
git commit -m "feat: implement SimulatorAgent prompt building logic"
```

---

## Task 7: Update Config Parser for Engine Configuration

**Files:**
- Modify: `src/utils/config_parser.py`

**Step 1: Write tests for engine config parsing**

Add to `tests/unit/test_config_parser.py`:

```python
def test_validate_engine_config():
    """Test engine configuration validation."""
    from utils.config_parser import validate_engine_config

    # Valid engine config
    config = {
        "engine": {
            "provider": "gemini",
            "model": "gemini-1.5-flash",
            "system_prompt": "Test prompt",
            "simulation_plan": "Test plan"
        }
    }
    validate_engine_config(config)  # Should not raise

    # Missing required field
    config_missing = {
        "engine": {
            "provider": "gemini"
            # Missing model, system_prompt, simulation_plan
        }
    }
    with pytest.raises(ValueError, match="missing required field"):
        validate_engine_config(config_missing)

    # Invalid scripted event
    config_bad_event = {
        "engine": {
            "provider": "gemini",
            "model": "test",
            "system_prompt": "Test",
            "simulation_plan": "Test",
            "scripted_events": [
                {"step": "not_an_int", "type": "test", "description": "test"}
            ]
        }
    }
    with pytest.raises(ValueError):
        validate_engine_config(config_bad_event)


def test_parse_scripted_events():
    """Test parsing scripted events."""
    from utils.config_parser import parse_scripted_events
    from core.engine_models import ScriptedEvent

    events_config = [
        {"step": 20, "type": "major_war", "description": "War begins"},
        {"step": 35, "type": "disaster", "description": "Earthquake"}
    ]

    events = parse_scripted_events(events_config)

    assert len(events) == 2
    assert isinstance(events[0], ScriptedEvent)
    assert events[0].step == 20
    assert events[0].type == "major_war"
    assert events[1].step == 35
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_config_parser.py::test_validate_engine_config -v
```

Expected: FAIL (functions don't exist)

**Step 3: Implement engine config parsing**

Add to end of `src/utils/config_parser.py`:

```python
from core.engine_models import ScriptedEvent


def parse_scripted_events(events_config: list[dict] | None) -> list[ScriptedEvent]:
    """
    Parse scripted events from YAML config.

    Args:
        events_config: List of event dictionaries

    Returns:
        List of ScriptedEvent objects
    """
    if not events_config:
        return []

    events = []
    for event_dict in events_config:
        if not isinstance(event_dict, dict):
            raise ValueError("Each scripted event must be a dictionary")

        required_fields = ["step", "type", "description"]
        for field in required_fields:
            if field not in event_dict:
                raise ValueError(f"Scripted event missing required field '{field}'")

        if not isinstance(event_dict["step"], int):
            raise ValueError("Scripted event 'step' must be an integer")

        events.append(ScriptedEvent(
            step=event_dict["step"],
            type=event_dict["type"],
            description=event_dict["description"]
        ))

    return events


def validate_engine_config(config: dict[str, Any]) -> None:
    """
    Validate engine configuration.

    Raises:
        ValueError: If engine configuration is invalid or missing
    """
    if "engine" not in config:
        raise ValueError("Configuration missing required 'engine' section")

    engine = config["engine"]

    if not isinstance(engine, dict):
        raise ValueError("'engine' must be a dictionary")

    required_fields = ["provider", "model", "system_prompt", "simulation_plan"]
    for field in required_fields:
        if field not in engine:
            raise ValueError(f"Engine configuration missing required field '{field}'")

    # Validate scripted events if present
    if "scripted_events" in engine:
        parse_scripted_events(engine["scripted_events"])
```

**Step 4: Update validate_config to check engine**

Add to `validate_config` function in `src/utils/config_parser.py`:

```python
def validate_config(config: dict[str, Any]) -> None:
    """
    Validate the entire configuration.

    Raises:
        ValueError: If configuration is invalid
    """
    validate_engine_config(config)  # Add this line at the top
    validate_agent_vars_config(config)
    validate_global_vars_config(config)

    # Validate agent configurations
    if "agents" in config:
        agent_vars_defs = config.get("agent_vars")
        for agent_config in config["agents"]:
            validate_agent_config(agent_config, agent_vars_defs)
```

**Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_config_parser.py -v
```

Expected: PASS (all tests)

**Step 6: Commit**

```bash
git add src/utils/config_parser.py tests/unit/test_config_parser.py
git commit -m "feat: add engine configuration validation and parsing"
```

---

## Task 8: Integrate SimulatorAgent into Orchestrator

**Files:**
- Modify: `src/core/orchestrator.py`

**Step 1: Update Orchestrator to use SimulatorAgent**

Replace the `run()` method and add SimulatorAgent initialization in `src/core/orchestrator.py`:

First, add import at top:
```python
from core.simulator_agent import SimulatorAgent, SimulationError
from utils.config_parser import parse_scripted_events
```

Then modify `__init__` method (after line 24, after `self.game_engine = GameEngine(self.config)`):

```python
        # Initialize SimulatorAgent
        engine_config = self.config.get("engine", {})
        engine_llm_config = {
            "provider": engine_config.get("provider"),
            "model": engine_config.get("model")
        }

        simulator_llm = self._create_llm_provider_for_engine(engine_llm_config)

        self.simulator_agent = SimulatorAgent(
            llm_provider=simulator_llm,
            game_engine=self.game_engine,
            system_prompt=engine_config.get("system_prompt", ""),
            simulation_plan=engine_config.get("simulation_plan", ""),
            realism_guidelines=engine_config.get("realism_guidelines", ""),
            scripted_events=parse_scripted_events(engine_config.get("scripted_events")),
            context_window_size=engine_config.get("context_window_size", 5),
            logger=self.logger
        )
```

Add new method after `_initialize_agents`:

```python
    def _create_llm_provider_for_engine(self, llm_config: dict):
        """Create LLM provider for SimulatorAgent (engine)."""
        provider_type = llm_config.get("provider", "gemini").lower()

        if provider_type == "gemini":
            model_name = llm_config.get("model", "gemini-1.5-flash")
            provider = GeminiProvider(model_name=model_name)
        elif provider_type == "ollama":
            model_name = llm_config.get("model", "llama2")
            base_url = llm_config.get("base_url")
            provider = OllamaProvider(model=model_name, base_url=base_url)
        else:
            raise ValueError(f"Unknown engine LLM provider: {provider_type}")

        # Engine LLM MUST be available
        if not provider.is_available():
            error_msg = (
                f"\n{'='*70}\n"
                f"ERROR: Engine LLM Provider Not Available\n"
                f"{'='*70}\n"
                f"Provider: {provider_type}\n"
                f"Model: {llm_config.get('model')}\n"
                f"\nThe simulation engine requires an LLM to run.\n"
                f"Please ensure the provider is configured and available.\n"
                f"{'='*70}\n"
            )
            print(error_msg, file=sys.stderr)
            raise ValueError("Engine LLM provider not available. Simulation cannot run.")

        return provider
```

Replace the `run()` method (starting at line ~123):

```python
    def run(self):
        """Run the simulation loop with SimulatorAgent."""
        self.logger.info(
            MessageCode.SIM001,
            "Simulation started",
            num_agents=len(self.agents),
            max_steps=self.max_steps
        )

        print(f"Starting simulation with {len(self.agents)} agent(s) for {self.max_steps} steps")
        print(f"Results will be saved to: {self.persistence.run_dir}\n")

        try:
            # Step 0: Initialize simulation
            print("=== Step 0: Initialization ===")
            agent_names = [agent.name for agent in self.agents]

            try:
                agent_messages = self.simulator_agent.initialize_simulation(agent_names)
                print("SimulatorAgent initialized simulation")
            except SimulationError as e:
                self.logger.critical(MessageCode.SIM002, "Initialization failed", error=str(e))
                print(f"\nERROR: Simulation initialization failed:\n{e}", file=sys.stderr)
                raise

            # Main simulation loop
            for step in range(1, self.max_steps + 1):
                with PerformanceTimer(self.logger, MessageCode.PRF001, f"Step {step}", step=step):
                    self.logger.info(MessageCode.SIM003, "Step started", step=step, max_steps=self.max_steps)
                    print(f"\n=== Step {step}/{self.max_steps} ===")

                    # Collect agent responses
                    agent_responses = {}
                    step_messages = []

                    for agent in self.agents:
                        try:
                            message = agent_messages[agent.name]
                            print(f"SimulatorAgent -> {agent.name}: {message}")

                            self.logger.info(
                                MessageCode.AGT002,
                                "Message sent to agent",
                                agent_name=agent.name,
                                step=step,
                                content=message
                            )

                            step_messages.append({
                                "from": "SimulatorAgent",
                                "to": agent.name,
                                "content": message
                            })

                            # Agent responds
                            response = agent.respond(message)
                            print(f"{agent.name} -> SimulatorAgent: {response}")

                            self.logger.info(
                                MessageCode.AGT003,
                                "Response received from agent",
                                agent_name=agent.name,
                                step=step,
                                response=response
                            )

                            step_messages.append({
                                "from": agent.name,
                                "to": "SimulatorAgent",
                                "content": response
                            })

                            agent_responses[agent.name] = response

                        except Exception as e:
                            error_msg = f"Agent {agent.name} failed: {e}"
                            self.logger.error(
                                MessageCode.AGT005,
                                error_msg,
                                agent_name=agent.name,
                                step=step,
                                error=str(e)
                            )
                            print(f"ERROR: {error_msg}", file=sys.stderr)
                            agent_responses[agent.name] = f"ERROR: {str(e)}"

                    # SimulatorAgent processes responses and generates next messages
                    try:
                        agent_messages = self.simulator_agent.process_step(step, agent_responses)
                        print(f"[SimulatorAgent processed step {step}]")
                    except SimulationError as e:
                        self.logger.critical(
                            MessageCode.SIM002,
                            "Simulation failed",
                            step=step,
                            error=str(e)
                        )
                        print(f"\nERROR: Simulation failed at step {step}:\n{e}", file=sys.stderr)
                        raise

                    # Save snapshot if needed
                    if self.persistence.should_save(step):
                        try:
                            snapshot = self.game_engine.get_state_snapshot()
                            self.persistence.save_snapshot(
                                step,
                                snapshot["game_state"],
                                snapshot["global_vars"],
                                snapshot["agent_vars"],
                                step_messages
                            )
                            print(f"[Saved snapshot at step {step}]")
                        except Exception as e:
                            error_msg = f"Failed to save snapshot: {e}"
                            self.logger.error(MessageCode.PER004, error_msg, step=step, error=str(e))
                            print(f"WARNING: {error_msg}", file=sys.stderr)

                    # Advance round
                    self.game_engine.advance_round()
                    self.logger.info(MessageCode.SIM004, "Step completed", step=step)

            # Save final state
            try:
                snapshot = self.game_engine.get_state_snapshot()
                self.persistence.save_final(
                    self.max_steps,
                    snapshot["game_state"],
                    snapshot["global_vars"],
                    snapshot["agent_vars"],
                    step_messages
                )
            except Exception as e:
                error_msg = f"Failed to save final state: {e}"
                self.logger.critical(MessageCode.PER004, error_msg, error=str(e))
                print(f"ERROR: {error_msg}", file=sys.stderr)

            self.logger.info(MessageCode.SIM002, "Simulation completed", total_steps=self.max_steps)
            print("\nSimulation completed.")
            print(f"Final game state: {self.game_engine.get_state()}")
            print(f"\nResults saved to: {self.persistence.run_dir}")

        except KeyboardInterrupt:
            self.logger.warning(MessageCode.SIM002, "Simulation interrupted by user")
            print("\n\nSimulation interrupted by user")
            raise

        except SimulationError:
            # Already logged, just re-raise
            raise

        except Exception as e:
            self.logger.critical(MessageCode.SIM002, "Simulation failed with unhandled exception", error=str(e))
            raise

        finally:
            self.persistence.close()
```

**Step 2: Remove old GameEngine methods no longer used**

The orchestrator no longer calls:
- `game_engine.get_message_for_agent()`
- `game_engine.process_agent_response()`

These can be safely removed from `src/core/game_engine.py` (lines ~31-62).

**Step 3: Commit**

```bash
git add src/core/orchestrator.py src/core/game_engine.py
git commit -m "feat: integrate SimulatorAgent into Orchestrator"
```

---

## Task 9: Create Test Scenario with Engine Configuration

**Files:**
- Create: `scenarios/engine_test.yaml`

**Step 1: Create test scenario**

Create `scenarios/engine_test.yaml`:

```yaml
max_steps: 5

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"

  system_prompt: |
    You are the game master for a simple test simulation.
    Maintain realistic cause-and-effect relationships.
    Keep responses concise for testing.

  simulation_plan: |
    This is a 5-step test simulation with 2 agents.
    Simulate basic interactions and stat changes.
    Keep it simple - this is for testing the engine.

  realism_guidelines: |
    - Small gradual changes (±5-20 per step)
    - Simple events only
    - Clear cause and effect

  context_window_size: 3

global_vars:
  tension:
    type: float
    default: 0.3
    min: 0.0
    max: 1.0
    description: "Global tension level"

agent_vars:
  health:
    type: int
    default: 100
    min: 0
    max: 100
    description: "Agent health"

  resources:
    type: int
    default: 50
    min: 0
    description: "Agent resources"

agents:
  - name: "Agent A"
    response_template: "I take action"
    variables:
      health: 100
      resources: 60

  - name: "Agent B"
    response_template: "I respond"
    variables:
      health: 100
      resources: 40
```

**Step 2: Test running the scenario**

```bash
uv run src/main.py scenarios/engine_test.yaml --save-frequency 0
```

Expected: Simulation runs successfully with SimulatorAgent

**Step 3: Commit**

```bash
git add scenarios/engine_test.yaml
git commit -m "feat: add test scenario for engine functionality"
```

---

## Task 10: Create Migration Tool

**Files:**
- Create: `tools/migrate_scenario.py`

**Step 1: Implement migration script**

Create `tools/migrate_scenario.py`:

```python
#!/usr/bin/env python3
"""
Migration tool to convert old scenario format to new engine-powered format.
"""

import sys
import yaml
from pathlib import Path


def migrate_scenario(input_path: str, output_path: str = None):
    """Migrate old scenario to new format."""
    # Load old scenario
    with open(input_path, 'r') as f:
        old_config = yaml.safe_load(f)

    # Create new config
    new_config = {
        "max_steps": old_config.get("max_steps", 10)
    }

    # Prompt for engine configuration
    print("\n=== Engine Configuration ===")
    print("The new format requires an LLM-powered simulation engine.")
    print()

    provider = input("LLM Provider (gemini/ollama) [gemini]: ").strip() or "gemini"

    if provider == "gemini":
        model = input("Model [gemini-1.5-flash]: ").strip() or "gemini-1.5-flash"
    else:
        model = input("Model [llama2]: ").strip() or "llama2"

    print("\nEngine System Prompt (multi-line, press Ctrl+D when done):")
    print("Example: You are the game master. Maintain realistic simulation.")
    system_prompt_lines = []
    try:
        while True:
            line = input()
            system_prompt_lines.append(line)
    except EOFError:
        pass
    system_prompt = "\n".join(system_prompt_lines)

    print("\nSimulation Plan (multi-line, press Ctrl+D when done):")
    print("Example: Simulate a 10-step economic scenario with gradual changes.")
    plan_lines = []
    try:
        while True:
            line = input()
            plan_lines.append(line)
    except EOFError:
        pass
    simulation_plan = "\n".join(plan_lines)

    new_config["engine"] = {
        "provider": provider,
        "model": model,
        "system_prompt": system_prompt,
        "simulation_plan": simulation_plan,
        "context_window_size": 5
    }

    # Convert game_state to global_vars if exists
    if "game_state" in old_config:
        print("\nWARNING: game_state section found. This should be converted to global_vars.")
        print("Please manually review and convert.")

    # Copy global_vars if exists
    if "global_vars" in old_config:
        new_config["global_vars"] = old_config["global_vars"]

    # Copy agent_vars if exists
    if "agent_vars" in old_config:
        new_config["agent_vars"] = old_config["agent_vars"]

    # Copy agents
    if "agents" in old_config:
        new_config["agents"] = old_config["agents"]

    # Determine output path
    if output_path is None:
        input_stem = Path(input_path).stem
        output_path = f"scenarios/{input_stem}_migrated.yaml"

    # Save migrated config
    with open(output_path, 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Migrated scenario saved to: {output_path}")
    print("\nIMPORTANT: Please review the migrated scenario:")
    print("1. Ensure engine configuration is correct")
    print("2. Convert any game_state entries to global_vars")
    print("3. Test the scenario before using in production")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tools/migrate_scenario.py <input.yaml> [output.yaml]")
        print("\nExample:")
        print("  python tools/migrate_scenario.py scenarios/old_scenario.yaml")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(input_path).exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    migrate_scenario(input_path, output_path)


if __name__ == "__main__":
    main()
```

**Step 2: Make executable**

```bash
chmod +x tools/migrate_scenario.py
```

**Step 3: Test migration tool**

```bash
# This is just a dry run - don't actually run interactively
python tools/migrate_scenario.py --help
```

**Step 4: Commit**

```bash
git add tools/migrate_scenario.py
git commit -m "feat: add scenario migration tool for engine format"
```

---

## Task 11: Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture.md`

**Step 1: Update README with engine information**

Update `README.md` to mention engine requirement and link to design doc:

After line 10, add:

```markdown
**Note:** As of v2.0, all scenarios require an LLM-powered simulation engine. See [Design Documentation](docs/plans/2025-11-22-llm-simulation-engine-design.md) for details.
```

**Step 2: Update architecture.md**

Add section to `docs/architecture.md` after "Core Components":

```markdown
### SimulatorAgent (NEW in v2.0)

**Responsibility:** LLM-powered simulation orchestration

- Interprets agent responses into realistic state changes
- Generates personalized agent messages
- Creates emergent and scripted events
- Enforces constraints through clamping with feedback
- Maintains sliding history window for context

**Key Methods:**
- `initialize_simulation(agent_names)`: Generate initial agent messages
- `process_step(step_number, agent_responses)`: Process step and return next messages
```

**Step 3: Commit**

```bash
git add README.md docs/architecture.md
git commit -m "docs: update for SimulatorAgent integration"
```

---

## Task 12: Integration Testing

**Files:**
- Create: `tests/integration/test_simulator_integration.py`

**Step 1: Write integration tests**

Create `tests/integration/test_simulator_integration.py`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

import json
import pytest
from core.simulator_agent import SimulatorAgent
from core.game_engine import GameEngine
from llm.llm_provider import LLMProvider


class MockSimulatorLLM(LLMProvider):
    """Mock LLM for integration testing."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def is_available(self) -> bool:
        return True

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        response = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return json.dumps(response)


class TestSimulatorIntegration:
    """Integration tests for SimulatorAgent + GameEngine."""

    def test_full_simulation_flow(self):
        """Test complete simulation from initialization through multiple steps."""
        config = {
            "global_vars": {
                "tension": {"type": "float", "default": 0.5, "min": 0.0, "max": 1.0}
            },
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [
                {"name": "Agent A"},
                {"name": "Agent B"}
            ]
        }

        game_engine = GameEngine(config)

        # Mock responses for initialization + 3 steps
        mock_responses = [
            # Step 0: Initialization
            {
                "state_updates": {"global_vars": {}, "agent_vars": {}},
                "events": [],
                "agent_messages": {
                    "Agent A": "Welcome Agent A",
                    "Agent B": "Welcome Agent B"
                },
                "reasoning": "Initial setup"
            },
            # Step 1
            {
                "state_updates": {
                    "global_vars": {"tension": 0.6},
                    "agent_vars": {"Agent A": {"health": 95}}
                },
                "events": [{"type": "combat", "description": "Skirmish"}],
                "agent_messages": {
                    "Agent A": "You took damage",
                    "Agent B": "Tensions rise"
                },
                "reasoning": "Combat occurred"
            },
            # Step 2
            {
                "state_updates": {
                    "global_vars": {"tension": 0.7},
                    "agent_vars": {"Agent B": {"health": 90}}
                },
                "events": [],
                "agent_messages": {
                    "Agent A": "Conflict escalates",
                    "Agent B": "You suffer damage"
                },
                "reasoning": "Escalation"
            },
            # Step 3
            {
                "state_updates": {
                    "global_vars": {"tension": 0.8},
                    "agent_vars": {
                        "Agent A": {"health": 85},
                        "Agent B": {"health": 80}
                    }
                },
                "events": [{"type": "full_battle", "description": "Major combat"}],
                "agent_messages": {
                    "Agent A": "Major battle",
                    "Agent B": "Heavy fighting"
                },
                "reasoning": "Full war"
            }
        ]

        mock_llm = MockSimulatorLLM(mock_responses)

        simulator = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[]
        )

        # Initialize
        messages = simulator.initialize_simulation(["Agent A", "Agent B"])
        assert "Agent A" in messages
        assert "Agent B" in messages

        # Run 3 steps
        for step in range(1, 4):
            responses = {
                "Agent A": f"Agent A action {step}",
                "Agent B": f"Agent B action {step}"
            }
            messages = simulator.process_step(step, responses)

            assert "Agent A" in messages
            assert "Agent B" in messages

        # Verify final state
        assert game_engine.get_global_var("tension") == 0.8
        assert game_engine.get_agent_var("Agent A", "health") == 85
        assert game_engine.get_agent_var("Agent B", "health") == 80

        # Verify history
        assert len(simulator.step_history) == 3

    def test_constraint_enforcement(self):
        """Test that constraints are enforced and fed back."""
        config = {
            "global_vars": {},
            "agent_vars": {
                "health": {"type": "int", "default": 100, "min": 0, "max": 100}
            },
            "agents": [{"name": "Agent A"}]
        }

        game_engine = GameEngine(config)

        # Response tries to set health below min
        mock_responses = [
            {
                "state_updates": {
                    "global_vars": {},
                    "agent_vars": {}
                },
                "events": [],
                "agent_messages": {"Agent A": "Start"},
                "reasoning": "Init"
            },
            {
                "state_updates": {
                    "global_vars": {},
                    "agent_vars": {"Agent A": {"health": -10}}  # Below min!
                },
                "events": [],
                "agent_messages": {"Agent A": "You died"},
                "reasoning": "Death"
            }
        ]

        mock_llm = MockSimulatorLLM(mock_responses)

        simulator = SimulatorAgent(
            llm_provider=mock_llm,
            game_engine=game_engine,
            system_prompt="Test",
            simulation_plan="Test",
            realism_guidelines="",
            scripted_events=[]
        )

        simulator.initialize_simulation(["Agent A"])
        simulator.process_step(1, {"Agent A": "I die"})

        # Should be clamped to 0
        assert game_engine.get_agent_var("Agent A", "health") == 0

        # Should have constraint hit in feedback
        assert len(simulator.constraint_feedback) == 1
        assert simulator.constraint_feedback[0].clamped_value == 0
```

**Step 2: Run integration tests**

```bash
pytest tests/integration/test_simulator_integration.py -v
```

Expected: PASS

**Step 3: Commit**

```bash
git add tests/integration/test_simulator_integration.py
git commit -m "test: add SimulatorAgent integration tests"
```

---

## Task 13: End-to-End Test with Real LLM (Manual)

**Files:**
- None (manual testing)

**Step 1: Test with Gemini (requires API key)**

```bash
# Ensure .env has GEMINI_API_KEY set
uv run src/main.py scenarios/engine_test.yaml
```

Expected: Full simulation runs with real LLM responses

**Step 2: Verify output**

Check `results/run_engine_test_*/`:
- `state.json` should show engine decisions
- `simulation.jsonl` should have ENG001-ENG012 codes
- Console output should show SimulatorAgent reasoning

**Step 3: Manual verification complete**

No commit needed - this is verification only.

---

## Task 14: Final Verification and Documentation

**Files:**
- Create: `docs/MIGRATION.md`

**Step 1: Create migration guide**

Create `docs/MIGRATION.md`:

```markdown
# Migration Guide: v1.0 to v2.0 (Engine-Powered Simulation)

## Breaking Changes

Version 2.0 introduces the **SimulatorAgent** - an LLM-powered game master that replaces the old orchestrator message system. All scenarios must be updated.

## What Changed

### Removed
- `orchestrator_message` field (replaced by engine-generated messages)
- Static message generation in GameEngine

### Required
- `engine` configuration block (LLM provider for simulation)
- All scenarios must have engine configuration

## Migration Steps

### Option 1: Use Migration Tool

```bash
python tools/migrate_scenario.py scenarios/old_scenario.yaml
```

Follow prompts to configure engine.

### Option 2: Manual Migration

1. **Remove `orchestrator_message`**
2. **Add `engine` block:**

```yaml
engine:
  provider: "gemini"  # or "ollama"
  model: "gemini-1.5-flash"
  system_prompt: |
    You are the game master.
    Maintain realistic cause-and-effect.
  simulation_plan: |
    Describe what should happen in your simulation.
  context_window_size: 5  # Optional, defaults to 5
```

3. **Convert `game_state` to `global_vars`** (if present)
4. **Test the scenario**

## Example

### Before (v1.0)
```yaml
max_steps: 10
orchestrator_message: "What do you do?"

game_state:
  initial_resources: 1000

agents:
  - name: "Agent A"
    response_template: "I act"
```

### After (v2.0)
```yaml
max_steps: 10

engine:
  provider: "gemini"
  model: "gemini-1.5-flash"
  system_prompt: |
    You are the game master.
  simulation_plan: |
    Manage a 10-step simulation.

global_vars:
  resources:
    type: int
    default: 1000
    min: 0

agents:
  - name: "Agent A"
    response_template: "I act"
```

## New Features Available

- **Personalized Messages:** Each agent gets custom narrative
- **Emergent Events:** Engine creates events based on agent actions
- **Scripted Events:** Guarantee key moments while allowing flexible execution
- **Stat Management:** Agents can't modify their own stats - engine is authoritative
- **Constraint Enforcement:** Min/max limits enforced with feedback

## Getting Help

- Design doc: `docs/plans/2025-11-22-llm-simulation-engine-design.md`
- Test scenario: `scenarios/engine_test.yaml`
- Issues: File bug reports with scenario that fails
```

**Step 2: Run all tests**

```bash
pytest tests/ -v
```

Expected: All tests pass

**Step 3: Commit**

```bash
git add docs/MIGRATION.md
git commit -m "docs: add migration guide for v2.0"
```

---

## Task 15: Final Review and Tag Release

**Files:**
- None (verification only)

**Step 1: Review implementation checklist**

Verify all components implemented:
- ✅ Engine logging codes (ENG001-ENG012)
- ✅ Data models (ScriptedEvent, EngineOutput, etc.)
- ✅ EngineValidator (validation + constraints)
- ✅ Simplified GameEngine (state only)
- ✅ SimulatorAgent (core + prompts)
- ✅ Config parser (engine validation)
- ✅ Orchestrator integration
- ✅ Test scenario
- ✅ Migration tool
- ✅ Documentation updates
- ✅ Integration tests
- ✅ Manual LLM testing

**Step 2: Run full test suite**

```bash
pytest tests/ -v --cov=src
```

Expected: High coverage, all tests pass

**Step 3: Create release commit**

```bash
git add -A
git commit -m "release: v2.0 - LLM-powered simulation engine

Complete implementation of SimulatorAgent - an intelligent game master that:
- Manages world state evolution and realism
- Interprets agent responses into realistic stat changes
- Generates personalized narratives per agent
- Creates emergent and scripted events
- Enforces constraints with feedback

Breaking changes:
- All scenarios now require 'engine' configuration
- Removed 'orchestrator_message' field
- GameEngine simplified to state management only

Migration:
- Use tools/migrate_scenario.py for automatic migration
- See docs/MIGRATION.md for manual steps

Complete design: docs/plans/2025-11-22-llm-simulation-engine-design.md"
```

**Step 4: Tag release**

```bash
git tag -a v2.0.0 -m "v2.0.0 - LLM-Powered Simulation Engine"
```

---

## Verification Steps

After completing all tasks, verify:

1. **Run test scenario:**
   ```bash
   uv run src/main.py scenarios/engine_test.yaml
   ```

2. **Check logs for ENG codes:**
   ```bash
   grep "ENG0" results/run_*/simulation.jsonl | jq .
   ```

3. **Verify all tests pass:**
   ```bash
   pytest tests/ -v
   ```

4. **Test migration tool:**
   ```bash
   python tools/migrate_scenario.py scenarios/config.yaml
   ```

## Success Criteria

- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ Test scenario runs successfully
- ✅ SimulatorAgent generates realistic updates
- ✅ Constraints enforced and logged
- ✅ History window maintained correctly
- ✅ Migration tool works
- ✅ Documentation complete

---

**Implementation complete! Ready to use SimulatorAgent-powered simulations.**
