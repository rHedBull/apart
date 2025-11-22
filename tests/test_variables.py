import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from variables import VariableDefinition, VariableSet


class TestVariableDefinition:
    """Tests for VariableDefinition model."""

    def test_int_variable(self):
        """Test integer variable definition."""
        var_def = VariableDefinition(type="int", default=100, min=0, max=1000)
        assert var_def.type == "int"
        assert var_def.default == 100
        assert var_def.min == 0
        assert var_def.max == 1000

    def test_float_variable(self):
        """Test float variable definition."""
        var_def = VariableDefinition(type="float", default=0.05, min=0.0, max=1.0)
        assert var_def.type == "float"
        assert var_def.default == 0.05
        assert var_def.min == 0.0
        assert var_def.max == 1.0

    def test_bool_variable(self):
        """Test boolean variable definition."""
        var_def = VariableDefinition(type="bool", default=True)
        assert var_def.type == "bool"
        assert var_def.default is True

    def test_bool_with_min_max_fails(self):
        """Test that bool cannot have min/max constraints."""
        with pytest.raises(ValueError, match="min constraint not supported for bool"):
            VariableDefinition(type="bool", default=True, min=0)

        with pytest.raises(ValueError, match="max constraint not supported for bool"):
            VariableDefinition(type="bool", default=False, max=1)

    def test_default_below_min_fails(self):
        """Test that default cannot be below min."""
        with pytest.raises(ValueError, match="default.*cannot be less than min"):
            VariableDefinition(type="int", default=5, min=10)

    def test_default_above_max_fails(self):
        """Test that default cannot be above max."""
        with pytest.raises(ValueError, match="default.*cannot be greater than max"):
            VariableDefinition(type="int", default=100, max=50)

    def test_min_greater_than_max_fails(self):
        """Test that min cannot be greater than max."""
        with pytest.raises(ValueError, match="min.*cannot be greater than max"):
            VariableDefinition(type="int", default=50, min=100, max=10)

    def test_wrong_type_default_fails(self):
        """Test that default must match specified type."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            VariableDefinition(type="int", default="not an int")

        with pytest.raises(ValidationError):
            VariableDefinition(type="bool", default=1)

    def test_validate_value_int(self):
        """Test validating integer values."""
        var_def = VariableDefinition(type="int", default=50, min=0, max=100)

        assert var_def.validate_value(75) == 75

        with pytest.raises(ValueError, match="below minimum"):
            var_def.validate_value(-5)

        with pytest.raises(ValueError, match="above maximum"):
            var_def.validate_value(150)

        with pytest.raises(ValueError, match="Expected int"):
            var_def.validate_value(50.5)

    def test_validate_value_float(self):
        """Test validating float values."""
        var_def = VariableDefinition(type="float", default=0.5, min=0.0, max=1.0)

        assert var_def.validate_value(0.75) == 0.75
        assert var_def.validate_value(1) == 1.0  # int gets converted to float

        with pytest.raises(ValueError, match="below minimum"):
            var_def.validate_value(-0.1)

        with pytest.raises(ValueError, match="above maximum"):
            var_def.validate_value(1.5)

    def test_validate_value_bool(self):
        """Test validating boolean values."""
        var_def = VariableDefinition(type="bool", default=True)

        assert var_def.validate_value(True) is True
        assert var_def.validate_value(False) is False

        with pytest.raises(ValueError, match="Expected bool"):
            var_def.validate_value(1)


class TestVariableSet:
    """Tests for VariableSet model."""

    def test_initialization_with_defaults(self):
        """Test that variables are initialized with default values."""
        definitions = {
            "health": VariableDefinition(type="int", default=100),
            "speed": VariableDefinition(type="float", default=1.5),
            "active": VariableDefinition(type="bool", default=True),
        }
        var_set = VariableSet(definitions=definitions)

        assert var_set.get("health") == 100
        assert var_set.get("speed") == 1.5
        assert var_set.get("active") is True

    def test_get_undefined_variable_fails(self):
        """Test that getting undefined variable raises error."""
        var_set = VariableSet()

        with pytest.raises(KeyError, match="not defined"):
            var_set.get("undefined_var")

    def test_set_variable(self):
        """Test setting variable values."""
        definitions = {
            "score": VariableDefinition(type="int", default=0, min=0, max=1000),
        }
        var_set = VariableSet(definitions=definitions)

        var_set.set("score", 500)
        assert var_set.get("score") == 500

    def test_set_invalid_value_fails(self):
        """Test that setting invalid value fails validation."""
        definitions = {
            "score": VariableDefinition(type="int", default=0, min=0, max=1000),
        }
        var_set = VariableSet(definitions=definitions)

        with pytest.raises(ValueError, match="above maximum"):
            var_set.set("score", 2000)

        with pytest.raises(ValueError, match="below minimum"):
            var_set.set("score", -10)

    def test_set_undefined_variable_fails(self):
        """Test that setting undefined variable raises error."""
        var_set = VariableSet()

        with pytest.raises(KeyError, match="not defined"):
            var_set.set("undefined_var", 100)

    def test_update_multiple_variables(self):
        """Test updating multiple variables at once."""
        definitions = {
            "health": VariableDefinition(type="int", default=100, min=0, max=100),
            "mana": VariableDefinition(type="int", default=50, min=0, max=100),
        }
        var_set = VariableSet(definitions=definitions)

        var_set.update({"health": 75, "mana": 25})
        assert var_set.get("health") == 75
        assert var_set.get("mana") == 25

    def test_to_dict(self):
        """Test converting variables to dictionary."""
        definitions = {
            "x": VariableDefinition(type="int", default=10),
            "y": VariableDefinition(type="float", default=20.5),
        }
        var_set = VariableSet(definitions=definitions)

        result = var_set.to_dict()
        assert result == {"x": 10, "y": 20.5}
