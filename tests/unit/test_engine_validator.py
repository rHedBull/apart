"""Unit tests for EngineValidator."""

import pytest
import json
from unittest.mock import Mock
from core.engine_validator import EngineValidator, ValidationResult


class TestStripMarkdownCodeBlocks:
    """Tests for _strip_markdown_code_blocks method."""

    def test_plain_json(self):
        """Test plain JSON without markdown."""
        response = '{"key": "value"}'
        result = EngineValidator._strip_markdown_code_blocks(response)
        assert result == '{"key": "value"}'

    def test_json_with_markdown_json_block(self):
        """Test JSON wrapped in ```json block."""
        response = '```json\n{"key": "value"}\n```'
        result = EngineValidator._strip_markdown_code_blocks(response)
        assert result == '{"key": "value"}'

    def test_json_with_plain_markdown_block(self):
        """Test JSON wrapped in plain ``` block."""
        response = '```\n{"key": "value"}\n```'
        result = EngineValidator._strip_markdown_code_blocks(response)
        assert result == '{"key": "value"}'

    def test_removes_js_comments(self):
        """Test removal of JavaScript-style // comments."""
        response = '{"key": "value", // this is a comment\n"other": 1}'
        result = EngineValidator._strip_markdown_code_blocks(response)
        assert "//" not in result
        # Should be valid JSON after stripping
        data = json.loads(result)
        assert data["key"] == "value"

    def test_fixes_trailing_commas(self):
        """Test fixing trailing commas before closing braces."""
        response = '{"key": "value", }'
        result = EngineValidator._strip_markdown_code_blocks(response)
        data = json.loads(result)
        assert data["key"] == "value"

    def test_fixes_trailing_commas_in_arrays(self):
        """Test fixing trailing commas in arrays."""
        response = '{"items": [1, 2, 3, ]}'
        result = EngineValidator._strip_markdown_code_blocks(response)
        data = json.loads(result)
        assert data["items"] == [1, 2, 3]

    def test_strips_whitespace(self):
        """Test stripping leading/trailing whitespace."""
        response = '   \n  {"key": "value"}  \n   '
        result = EngineValidator._strip_markdown_code_blocks(response)
        assert result == '{"key": "value"}'

    def test_complex_markdown_response(self):
        """Test complex response with markdown and comments."""
        response = '''```json
{
    "state_updates": {
        "global_vars": {"count": 5}, // update count
        "agent_vars": {}
    },
    "events": [],
    "agent_messages": {},
    "reasoning": "test",
}
```'''
        result = EngineValidator._strip_markdown_code_blocks(response)
        data = json.loads(result)
        assert data["state_updates"]["global_vars"]["count"] == 5


class TestValidateStructure:
    """Tests for validate_structure method."""

    def test_valid_structure(self):
        """Test validation of valid JSON structure."""
        response = json.dumps({
            "state_updates": {
                "global_vars": {},
                "agent_vars": {}
            },
            "events": [],
            "agent_messages": {},
            "reasoning": "test"
        })
        result = EngineValidator.validate_structure(response)
        assert result.success is True
        assert result.error is None

    def test_invalid_json(self):
        """Test validation of invalid JSON."""
        response = "not valid json {"
        result = EngineValidator.validate_structure(response)
        assert result.success is False
        assert "Invalid JSON" in result.error

    def test_missing_state_updates(self):
        """Test validation when state_updates is missing."""
        response = json.dumps({
            "events": [],
            "agent_messages": {},
            "reasoning": "test"
        })
        result = EngineValidator.validate_structure(response)
        assert result.success is False
        assert "state_updates" in result.error

    def test_missing_events(self):
        """Test validation when events is missing."""
        response = json.dumps({
            "state_updates": {"global_vars": {}, "agent_vars": {}},
            "agent_messages": {},
            "reasoning": "test"
        })
        result = EngineValidator.validate_structure(response)
        assert result.success is False
        assert "events" in result.error

    def test_missing_agent_messages(self):
        """Test validation when agent_messages is missing."""
        response = json.dumps({
            "state_updates": {"global_vars": {}, "agent_vars": {}},
            "events": [],
            "reasoning": "test"
        })
        result = EngineValidator.validate_structure(response)
        assert result.success is False
        assert "agent_messages" in result.error

    def test_missing_reasoning(self):
        """Test validation when reasoning is missing."""
        response = json.dumps({
            "state_updates": {"global_vars": {}, "agent_vars": {}},
            "events": [],
            "agent_messages": {}
        })
        result = EngineValidator.validate_structure(response)
        assert result.success is False
        assert "reasoning" in result.error

    def test_state_updates_not_dict(self):
        """Test validation when state_updates is not a dict."""
        response = json.dumps({
            "state_updates": "not a dict",
            "events": [],
            "agent_messages": {},
            "reasoning": "test"
        })
        result = EngineValidator.validate_structure(response)
        assert result.success is False
        assert "state_updates must be a dict" in result.error

    def test_missing_global_vars(self):
        """Test validation when global_vars is missing."""
        response = json.dumps({
            "state_updates": {"agent_vars": {}},
            "events": [],
            "agent_messages": {},
            "reasoning": "test"
        })
        result = EngineValidator.validate_structure(response)
        assert result.success is False
        assert "global_vars" in result.error

    def test_missing_agent_vars(self):
        """Test validation when agent_vars is missing."""
        response = json.dumps({
            "state_updates": {"global_vars": {}},
            "events": [],
            "agent_messages": {},
            "reasoning": "test"
        })
        result = EngineValidator.validate_structure(response)
        assert result.success is False
        assert "agent_vars" in result.error

    def test_strips_markdown_before_validation(self):
        """Test that markdown is stripped before validation."""
        response = '''```json
{
    "state_updates": {"global_vars": {}, "agent_vars": {}},
    "events": [],
    "agent_messages": {},
    "reasoning": "test"
}
```'''
        result = EngineValidator.validate_structure(response)
        assert result.success is True


class TestValidateReferences:
    """Tests for validate_references method."""

    def test_valid_references(self):
        """Test validation with valid agent and variable references."""
        output = {
            "state_updates": {
                "global_vars": {"score": 10},
                "agent_vars": {"Agent1": {"health": 100}}
            },
            "agent_messages": {"Agent1": "Hello"}
        }
        result = EngineValidator.validate_references(
            output,
            agents=["Agent1", "Agent2"],
            global_var_defs={"score": {"type": "int"}},
            agent_var_defs={"health": {"type": "int"}}
        )
        assert result.success is True

    def test_unknown_agent_in_state_updates(self):
        """Test validation with unknown agent in state_updates."""
        output = {
            "state_updates": {
                "global_vars": {},
                "agent_vars": {"UnknownAgent": {"health": 100}}
            },
            "agent_messages": {}
        }
        result = EngineValidator.validate_references(
            output,
            agents=["Agent1"],
            global_var_defs={},
            agent_var_defs={"health": {"type": "int"}}
        )
        assert result.success is False
        assert "Unknown agent 'UnknownAgent'" in result.error

    def test_unknown_agent_in_messages(self):
        """Test validation with unknown agent in agent_messages."""
        output = {
            "state_updates": {
                "global_vars": {},
                "agent_vars": {}
            },
            "agent_messages": {"UnknownAgent": "Hello"}
        }
        result = EngineValidator.validate_references(
            output,
            agents=["Agent1"],
            global_var_defs={},
            agent_var_defs={}
        )
        assert result.success is False
        assert "Unknown agent 'UnknownAgent'" in result.error

    def test_unknown_global_variable(self):
        """Test validation with unknown global variable."""
        output = {
            "state_updates": {
                "global_vars": {"unknown_var": 10},
                "agent_vars": {}
            },
            "agent_messages": {}
        }
        result = EngineValidator.validate_references(
            output,
            agents=[],
            global_var_defs={"known_var": {"type": "int"}},
            agent_var_defs={}
        )
        assert result.success is False
        assert "Unknown global variable 'unknown_var'" in result.error

    def test_unknown_agent_variable(self):
        """Test validation with unknown agent variable."""
        output = {
            "state_updates": {
                "global_vars": {},
                "agent_vars": {"Agent1": {"unknown_var": 10}}
            },
            "agent_messages": {}
        }
        result = EngineValidator.validate_references(
            output,
            agents=["Agent1"],
            global_var_defs={},
            agent_var_defs={"known_var": {"type": "int"}}
        )
        assert result.success is False
        assert "Unknown agent variable 'unknown_var'" in result.error


class TestValidateTypes:
    """Tests for validate_types method."""

    def test_valid_int_type(self):
        """Test validation of valid int type."""
        output = {
            "state_updates": {
                "global_vars": {"count": 5},
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={"count": {"type": "int"}},
            agent_var_defs={}
        )
        assert result.success is True

    def test_valid_float_type(self):
        """Test validation of valid float type."""
        output = {
            "state_updates": {
                "global_vars": {"rate": 0.5},
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={"rate": {"type": "float"}},
            agent_var_defs={}
        )
        assert result.success is True

    def test_int_accepted_as_float(self):
        """Test that int is accepted for float type."""
        output = {
            "state_updates": {
                "global_vars": {"rate": 5},
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={"rate": {"type": "float"}},
            agent_var_defs={}
        )
        assert result.success is True

    def test_valid_bool_type(self):
        """Test validation of valid bool type."""
        output = {
            "state_updates": {
                "global_vars": {"active": True},
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={"active": {"type": "bool"}},
            agent_var_defs={}
        )
        assert result.success is True

    def test_valid_list_type(self):
        """Test validation of valid list type."""
        output = {
            "state_updates": {
                "global_vars": {"items": [1, 2, 3]},
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={"items": {"type": "list"}},
            agent_var_defs={}
        )
        assert result.success is True

    def test_valid_dict_type(self):
        """Test validation of valid dict type."""
        output = {
            "state_updates": {
                "global_vars": {"data": {"a": 1}},
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={"data": {"type": "dict"}},
            agent_var_defs={}
        )
        assert result.success is True

    def test_invalid_int_type(self):
        """Test validation fails when string given for int."""
        output = {
            "state_updates": {
                "global_vars": {"count": "five"},
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={"count": {"type": "int"}},
            agent_var_defs={}
        )
        assert result.success is False
        assert "expected int" in result.error

    def test_bool_not_accepted_as_int(self):
        """Test that bool is not accepted as int."""
        output = {
            "state_updates": {
                "global_vars": {"count": True},
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={"count": {"type": "int"}},
            agent_var_defs={}
        )
        assert result.success is False

    def test_agent_variable_type_validation(self):
        """Test type validation for agent variables."""
        output = {
            "state_updates": {
                "global_vars": {},
                "agent_vars": {"Agent1": {"health": "full"}}
            }
        }
        result = EngineValidator.validate_types(
            output,
            global_var_defs={},
            agent_var_defs={"health": {"type": "int"}}
        )
        assert result.success is False
        assert "Agent1" in result.error


class TestCheckType:
    """Tests for _check_type helper method."""

    def test_int_type(self):
        assert EngineValidator._check_type(5, "int") is True
        assert EngineValidator._check_type(5.0, "int") is True  # whole-number floats accepted (LLM compat)
        assert EngineValidator._check_type(5.5, "int") is False  # non-whole floats rejected
        assert EngineValidator._check_type("5", "int") is False
        assert EngineValidator._check_type(True, "int") is False  # bool is not int

    def test_float_type(self):
        assert EngineValidator._check_type(5.0, "float") is True
        assert EngineValidator._check_type(5, "float") is True  # int is acceptable
        assert EngineValidator._check_type("5.0", "float") is False
        assert EngineValidator._check_type(True, "float") is False

    def test_bool_type(self):
        assert EngineValidator._check_type(True, "bool") is True
        assert EngineValidator._check_type(False, "bool") is True
        assert EngineValidator._check_type(1, "bool") is False
        assert EngineValidator._check_type("true", "bool") is False

    def test_list_type(self):
        assert EngineValidator._check_type([1, 2, 3], "list") is True
        assert EngineValidator._check_type([], "list") is True
        assert EngineValidator._check_type((1, 2), "list") is False
        assert EngineValidator._check_type("list", "list") is False

    def test_dict_type(self):
        assert EngineValidator._check_type({"a": 1}, "dict") is True
        assert EngineValidator._check_type({}, "dict") is True
        assert EngineValidator._check_type([("a", 1)], "dict") is False

    def test_unknown_type(self):
        assert EngineValidator._check_type("value", "unknown") is False


class TestApplyConstraints:
    """Tests for apply_constraints method."""

    def test_no_constraints(self):
        """Test applying constraints when none are defined."""
        state_updates = {
            "global_vars": {"count": 50},
            "agent_vars": {}
        }
        global_defs = {"count": {"type": "int"}}

        clamped, hits = EngineValidator.apply_constraints(
            state_updates, global_defs, {}
        )

        assert clamped["global_vars"]["count"] == 50
        assert len(hits) == 0

    def test_value_below_min(self):
        """Test clamping value below minimum."""
        state_updates = {
            "global_vars": {"count": -10},
            "agent_vars": {}
        }
        global_defs = {"count": {"type": "int", "min": 0, "max": 100}}

        clamped, hits = EngineValidator.apply_constraints(
            state_updates, global_defs, {}
        )

        assert clamped["global_vars"]["count"] == 0
        assert len(hits) == 1
        assert hits[0].constraint_type == "min"
        assert hits[0].attempted_value == -10
        assert hits[0].clamped_value == 0

    def test_value_above_max(self):
        """Test clamping value above maximum."""
        state_updates = {
            "global_vars": {"count": 150},
            "agent_vars": {}
        }
        global_defs = {"count": {"type": "int", "min": 0, "max": 100}}

        clamped, hits = EngineValidator.apply_constraints(
            state_updates, global_defs, {}
        )

        assert clamped["global_vars"]["count"] == 100
        assert len(hits) == 1
        assert hits[0].constraint_type == "max"

    def test_value_within_range(self):
        """Test value within constraints is unchanged."""
        state_updates = {
            "global_vars": {"count": 50},
            "agent_vars": {}
        }
        global_defs = {"count": {"type": "int", "min": 0, "max": 100}}

        clamped, hits = EngineValidator.apply_constraints(
            state_updates, global_defs, {}
        )

        assert clamped["global_vars"]["count"] == 50
        assert len(hits) == 0

    def test_agent_variable_constraints(self):
        """Test constraints on agent variables."""
        state_updates = {
            "global_vars": {},
            "agent_vars": {"Agent1": {"health": 150}}
        }
        agent_defs = {"health": {"type": "int", "min": 0, "max": 100}}

        clamped, hits = EngineValidator.apply_constraints(
            state_updates, {}, agent_defs
        )

        assert clamped["agent_vars"]["Agent1"]["health"] == 100
        assert len(hits) == 1
        assert hits[0].agent_name == "Agent1"

    def test_non_numeric_types_unchanged(self):
        """Test that non-numeric types are not clamped."""
        state_updates = {
            "global_vars": {"active": True, "items": [1, 2, 3]},
            "agent_vars": {}
        }
        global_defs = {
            "active": {"type": "bool"},
            "items": {"type": "list"}
        }

        clamped, hits = EngineValidator.apply_constraints(
            state_updates, global_defs, {}
        )

        assert clamped["global_vars"]["active"] is True
        assert clamped["global_vars"]["items"] == [1, 2, 3]
        assert len(hits) == 0

    def test_float_constraints(self):
        """Test constraints on float variables."""
        state_updates = {
            "global_vars": {"rate": 1.5},
            "agent_vars": {}
        }
        global_defs = {"rate": {"type": "float", "min": 0.0, "max": 1.0}}

        clamped, hits = EngineValidator.apply_constraints(
            state_updates, global_defs, {}
        )

        assert clamped["global_vars"]["rate"] == 1.0
        assert len(hits) == 1


class TestClampValue:
    """Tests for _clamp_value helper method."""

    def test_clamp_to_min(self):
        """Test clamping to minimum value."""
        value, hit = EngineValidator._clamp_value(
            value=-5,
            var_def={"type": "int", "min": 0, "max": 100},
            agent_name=None,
            var_name="count"
        )
        assert value == 0
        assert hit is not None
        assert hit.var_name == "count"
        assert hit.constraint_type == "min"

    def test_clamp_to_max(self):
        """Test clamping to maximum value."""
        value, hit = EngineValidator._clamp_value(
            value=200,
            var_def={"type": "int", "min": 0, "max": 100},
            agent_name="Agent1",
            var_name="health"
        )
        assert value == 100
        assert hit is not None
        assert hit.agent_name == "Agent1"
        assert hit.constraint_type == "max"

    def test_no_clamp_needed(self):
        """Test no clamping when within range."""
        value, hit = EngineValidator._clamp_value(
            value=50,
            var_def={"type": "int", "min": 0, "max": 100},
            agent_name=None,
            var_name="count"
        )
        assert value == 50
        assert hit is None

    def test_no_min_constraint(self):
        """Test when only max constraint is defined."""
        value, hit = EngineValidator._clamp_value(
            value=-1000,
            var_def={"type": "int", "max": 100},
            agent_name=None,
            var_name="count"
        )
        assert value == -1000  # No min, so not clamped
        assert hit is None

    def test_no_max_constraint(self):
        """Test when only min constraint is defined."""
        value, hit = EngineValidator._clamp_value(
            value=1000,
            var_def={"type": "int", "min": 0},
            agent_name=None,
            var_name="count"
        )
        assert value == 1000  # No max, so not clamped
        assert hit is None


class TestValidateLocations:
    """Tests for validate_locations method."""

    def test_no_spatial_graph(self):
        """Test validation passes when no spatial graph."""
        output = {
            "state_updates": {
                "agent_vars": {"Agent1": {"location": "anywhere"}}
            }
        }
        result = EngineValidator.validate_locations(output, spatial_graph=None)
        assert result.success is True

    def test_valid_location(self):
        """Test validation with valid location."""
        mock_graph = Mock()
        mock_graph.__contains__ = Mock(return_value=True)

        output = {
            "state_updates": {
                "agent_vars": {"Agent1": {"location": "city_a"}}
            }
        }
        result = EngineValidator.validate_locations(output, mock_graph)
        assert result.success is True

    def test_invalid_location(self):
        """Test validation with invalid location."""
        mock_graph = Mock()
        mock_graph.__contains__ = Mock(return_value=False)
        mock_graph.get_node_ids = Mock(return_value=["city_a", "city_b", "city_c"])

        output = {
            "state_updates": {
                "agent_vars": {"Agent1": {"location": "unknown_place"}}
            }
        }
        result = EngineValidator.validate_locations(output, mock_graph)
        assert result.success is False
        assert "Invalid location" in result.error
        assert "unknown_place" in result.error
        assert "Agent1" in result.error

    def test_no_location_in_vars(self):
        """Test validation passes when no location in agent vars."""
        mock_graph = Mock()

        output = {
            "state_updates": {
                "agent_vars": {"Agent1": {"health": 100}}
            }
        }
        result = EngineValidator.validate_locations(output, mock_graph)
        assert result.success is True

    def test_empty_agent_vars(self):
        """Test validation with empty agent vars."""
        mock_graph = Mock()

        output = {
            "state_updates": {
                "agent_vars": {}
            }
        }
        result = EngineValidator.validate_locations(output, mock_graph)
        assert result.success is True


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_success_result(self):
        """Test creating successful result."""
        result = ValidationResult(success=True)
        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Test creating failure result."""
        result = ValidationResult(success=False, error="Something went wrong")
        assert result.success is False
        assert result.error == "Something went wrong"
