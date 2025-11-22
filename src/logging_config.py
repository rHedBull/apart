import json
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class MessageCode(str, Enum):
    """Predefined message codes for filtering and analysis."""
    # Simulation lifecycle (SIM)
    SIM001 = "SIM001"  # Simulation started
    SIM002 = "SIM002"  # Simulation completed
    SIM003 = "SIM003"  # Step started
    SIM004 = "SIM004"  # Step completed
    SIM005 = "SIM005"  # Round advanced

    # Agent operations (AGT)
    AGT001 = "AGT001"  # Agent initialized
    AGT002 = "AGT002"  # Agent message sent
    AGT003 = "AGT003"  # Agent response received
    AGT004 = "AGT004"  # Agent state updated
    AGT005 = "AGT005"  # Agent error

    # Persistence operations (PER)
    PER001 = "PER001"  # Run directory created
    PER002 = "PER002"  # Snapshot saved
    PER003 = "PER003"  # Final state saved
    PER004 = "PER004"  # Persistence error

    # Game engine (GME)
    GME001 = "GME001"  # Game state initialized
    GME002 = "GME002"  # Game state updated
    GME003 = "GME003"  # Variable changed
    GME004 = "GME004"  # Game engine error

    # Configuration (CFG)
    CFG001 = "CFG001"  # Configuration loaded
    CFG002 = "CFG002"  # Configuration validation error

    # Performance (PRF)
    PRF001 = "PRF001"  # Operation timing


class LogMessage(BaseModel):
    """Base structure for all log messages."""
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    level: LogLevel
    code: MessageCode
    message: str
    context: dict[str, Any] = Field(default_factory=dict)
    duration_ms: Optional[float] = None

    def to_json(self) -> str:
        """Convert to JSON string for JSONL output."""
        return self.model_dump_json()

    def to_console(self) -> str:
        """Format for human-readable console output."""
        # Color codes for different levels
        colors = {
            LogLevel.DEBUG: "\033[36m",      # Cyan
            LogLevel.INFO: "\033[32m",       # Green
            LogLevel.WARNING: "\033[33m",    # Yellow
            LogLevel.ERROR: "\033[31m",      # Red
            LogLevel.CRITICAL: "\033[35m",   # Magenta
        }
        reset = "\033[0m"

        color = colors.get(self.level, "")
        level_str = f"{color}[{self.level.value}]{reset}"
        code_str = f"[{self.code.value}]"

        # Format timestamp
        dt = datetime.fromisoformat(self.timestamp)
        time_str = dt.strftime("%H:%M:%S.%f")[:-3]  # Millisecond precision

        # Build base message
        parts = [f"{time_str} {level_str} {code_str} {self.message}"]

        # Add duration if present
        if self.duration_ms is not None:
            parts.append(f"({self.duration_ms:.2f}ms)")

        # Add important context items
        if self.context:
            context_items = []
            for key, value in self.context.items():
                if key in ["step", "agent_name", "round", "error"]:
                    context_items.append(f"{key}={value}")
            if context_items:
                parts.append(f"[{', '.join(context_items)}]")

        return " ".join(parts)


class StructuredLogger:
    """Handles dual output: console and JSONL file."""

    def __init__(self, log_file: Optional[Path] = None, min_level: LogLevel = LogLevel.INFO):
        """
        Initialize structured logger.

        Args:
            log_file: Path to JSONL log file (if None, only console output)
            min_level: Minimum log level to output
        """
        self.log_file = log_file
        self.min_level = min_level
        self._log_handle = None

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_file, "a", encoding="utf-8")

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on level."""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4,
        }
        return level_order[level] >= level_order[self.min_level]

    def log(
        self,
        level: LogLevel,
        code: MessageCode,
        message: str,
        context: Optional[dict[str, Any]] = None,
        duration_ms: Optional[float] = None
    ):
        """Log a message with structured format."""
        if not self._should_log(level):
            return

        log_msg = LogMessage(
            level=level,
            code=code,
            message=message,
            context=context or {},
            duration_ms=duration_ms
        )

        # Output to console
        print(log_msg.to_console(), file=sys.stdout)

        # Output to JSONL file
        if self._log_handle:
            self._log_handle.write(log_msg.to_json() + "\n")
            self._log_handle.flush()

    def debug(self, code: MessageCode, message: str, **context):
        """Log DEBUG level message."""
        self.log(LogLevel.DEBUG, code, message, context)

    def info(self, code: MessageCode, message: str, **context):
        """Log INFO level message."""
        self.log(LogLevel.INFO, code, message, context)

    def warning(self, code: MessageCode, message: str, **context):
        """Log WARNING level message."""
        self.log(LogLevel.WARNING, code, message, context)

    def error(self, code: MessageCode, message: str, **context):
        """Log ERROR level message."""
        self.log(LogLevel.ERROR, code, message, context)

    def critical(self, code: MessageCode, message: str, **context):
        """Log CRITICAL level message."""
        self.log(LogLevel.CRITICAL, code, message, context)

    def close(self):
        """Close log file handle."""
        if self._log_handle:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, logger: StructuredLogger, code: MessageCode, operation: str, **context):
        """
        Initialize timer.

        Args:
            logger: Logger instance
            code: Message code for the operation
            operation: Description of operation being timed
            **context: Additional context for the log message
        """
        self.logger = logger
        self.code = code
        self.operation = operation
        self.context = context
        self.start_time = None

    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timing and log."""
        duration_ms = (datetime.now() - self.start_time).total_seconds() * 1000

        if exc_type is None:
            self.logger.log(
                LogLevel.DEBUG,
                self.code,
                f"{self.operation} completed",
                self.context,
                duration_ms
            )
        else:
            self.context["error"] = str(exc_val)
            self.logger.log(
                LogLevel.ERROR,
                self.code,
                f"{self.operation} failed",
                self.context,
                duration_ms
            )
