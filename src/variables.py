from pydantic import BaseModel, Field, field_validator
from typing import Any, Literal


class VariableDefinition(BaseModel):
    """Definition of a state variable with type and constraints."""

    type: Literal["int", "float", "bool"]
    default: int | float | bool
    min: int | float | None = None
    max: int | float | None = None
    description: str = ""

    @field_validator("default")
    @classmethod
    def validate_default_type(cls, v: Any, info) -> Any:
        """Validate that default matches the specified type."""
        var_type = info.data.get("type")
        if var_type == "int" and not isinstance(v, int):
            raise ValueError(f"Default value must be int, got {type(v).__name__}")
        if var_type == "float" and not isinstance(v, (int, float)):
            raise ValueError(f"Default value must be float, got {type(v).__name__}")
        if var_type == "bool" and not isinstance(v, bool):
            raise ValueError(f"Default value must be bool, got {type(v).__name__}")
        return float(v) if var_type == "float" else v

    @field_validator("min")
    @classmethod
    def validate_min(cls, v: Any, info) -> Any:
        """Validate min constraint is appropriate for type."""
        var_type = info.data.get("type")
        if v is not None and var_type == "bool":
            raise ValueError("min constraint not supported for bool type")
        if v is not None and var_type == "float":
            return float(v)
        return v

    @field_validator("max")
    @classmethod
    def validate_max(cls, v: Any, info) -> Any:
        """Validate max constraint is appropriate for type."""
        var_type = info.data.get("type")
        if v is not None and var_type == "bool":
            raise ValueError("max constraint not supported for bool type")
        if v is not None and var_type == "float":
            return float(v)
        return v

    def model_post_init(self, __context: Any) -> None:
        """Validate constraints after model initialization."""
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"min ({self.min}) cannot be greater than max ({self.max})")

        if self.min is not None and self.default < self.min:
            raise ValueError(f"default ({self.default}) cannot be less than min ({self.min})")

        if self.max is not None and self.default > self.max:
            raise ValueError(f"default ({self.default}) cannot be greater than max ({self.max})")

    def validate_value(self, value: Any) -> int | float | bool:
        """Validate and convert a value according to this variable's definition."""
        # Type checking
        if self.type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"Expected int, got {type(value).__name__}")
        elif self.type == "float":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"Expected float, got {type(value).__name__}")
            value = float(value)
        elif self.type == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"Expected bool, got {type(value).__name__}")

        # Range checking
        if self.min is not None and value < self.min:
            raise ValueError(f"Value {value} is below minimum {self.min}")
        if self.max is not None and value > self.max:
            raise ValueError(f"Value {value} is above maximum {self.max}")

        return value


class VariableSet(BaseModel):
    """A set of variable definitions with values."""

    definitions: dict[str, VariableDefinition] = Field(default_factory=dict)
    values: dict[str, int | float | bool] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Initialize values with defaults if not provided."""
        for name, definition in self.definitions.items():
            if name not in self.values:
                self.values[name] = definition.default

    def get(self, name: str) -> int | float | bool:
        """Get a variable value."""
        if name not in self.definitions:
            raise KeyError(f"Variable '{name}' not defined")
        return self.values[name]

    def set(self, name: str, value: Any) -> None:
        """Set a variable value with validation."""
        if name not in self.definitions:
            raise KeyError(f"Variable '{name}' not defined")

        validated_value = self.definitions[name].validate_value(value)
        self.values[name] = validated_value

    def update(self, updates: dict[str, Any]) -> None:
        """Update multiple variables at once."""
        for name, value in updates.items():
            self.set(name, value)

    def to_dict(self) -> dict[str, int | float | bool]:
        """Get all variable values as a dict."""
        return self.values.copy()
