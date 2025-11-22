import pytest
import json
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.logging_config import (
    LogLevel,
    MessageCode,
    LogMessage,
    StructuredLogger,
    PerformanceTimer
)


class TestLogMessage:
    """Test LogMessage Pydantic model."""

    def test_log_message_creation(self):
        """Test basic log message creation."""
        msg = LogMessage(
            level=LogLevel.INFO,
            code=MessageCode.SIM001,
            message="Test message",
            context={"key": "value"}
        )
        assert msg.level == LogLevel.INFO
        assert msg.code == MessageCode.SIM001
        assert msg.message == "Test message"
        assert msg.context == {"key": "value"}
        assert msg.duration_ms is None

    def test_log_message_with_duration(self):
        """Test log message with duration."""
        msg = LogMessage(
            level=LogLevel.DEBUG,
            code=MessageCode.PRF001,
            message="Operation completed",
            duration_ms=123.45
        )
        assert msg.duration_ms == 123.45

    def test_log_message_to_json(self):
        """Test JSON serialization."""
        msg = LogMessage(
            level=LogLevel.ERROR,
            code=MessageCode.AGT005,
            message="Agent error",
            context={"agent_name": "TestAgent", "error": "test error"}
        )
        json_str = msg.to_json()
        parsed = json.loads(json_str)

        assert parsed["level"] == "ERROR"
        assert parsed["code"] == "AGT005"
        assert parsed["message"] == "Agent error"
        assert parsed["context"]["agent_name"] == "TestAgent"

    def test_log_message_to_console(self):
        """Test console formatting."""
        msg = LogMessage(
            level=LogLevel.INFO,
            code=MessageCode.SIM001,
            message="Simulation started",
            context={"step": 1, "agent_name": "Agent1"}
        )
        console_str = msg.to_console()

        assert "[INFO]" in console_str
        assert "[SIM001]" in console_str
        assert "Simulation started" in console_str
        assert "step=1" in console_str
        assert "agent_name=Agent1" in console_str

    def test_log_message_console_with_duration(self):
        """Test console formatting includes duration."""
        msg = LogMessage(
            level=LogLevel.DEBUG,
            code=MessageCode.PRF001,
            message="Operation",
            duration_ms=42.5
        )
        console_str = msg.to_console()
        assert "(42.50ms)" in console_str


class TestStructuredLogger:
    """Test StructuredLogger functionality."""

    def test_logger_initialization_no_file(self):
        """Test logger without file output."""
        logger = StructuredLogger(log_file=None, min_level=LogLevel.INFO)
        assert logger.log_file is None
        assert logger.min_level == LogLevel.INFO
        assert logger._log_handle is None

    def test_logger_initialization_with_file(self, tmp_path):
        """Test logger with file output."""
        log_file = tmp_path / "test.jsonl"
        logger = StructuredLogger(log_file=log_file, min_level=LogLevel.DEBUG)

        assert logger.log_file == log_file
        assert logger.min_level == LogLevel.DEBUG
        assert logger._log_handle is not None

        logger.close()

    def test_logger_level_filtering(self, tmp_path):
        """Test that log level filtering works."""
        log_file = tmp_path / "test.jsonl"
        logger = StructuredLogger(log_file=log_file, min_level=LogLevel.WARNING)

        # Should not log DEBUG or INFO
        logger.debug(MessageCode.PRF001, "Debug message")
        logger.info(MessageCode.SIM001, "Info message")

        # Should log WARNING and ERROR
        logger.warning(MessageCode.PER004, "Warning message")
        logger.error(MessageCode.AGT005, "Error message")

        logger.close()

        # Read log file
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 2
        log1 = json.loads(lines[0])
        log2 = json.loads(lines[1])

        assert log1["level"] == "WARNING"
        assert log2["level"] == "ERROR"

    def test_logger_context_manager(self, tmp_path):
        """Test logger as context manager."""
        log_file = tmp_path / "test.jsonl"

        with StructuredLogger(log_file=log_file) as logger:
            logger.info(MessageCode.SIM001, "Test message")

        # File should exist and be closed
        assert log_file.exists()
        with open(log_file) as f:
            lines = f.readlines()
        assert len(lines) == 1

    def test_logger_writes_to_file(self, tmp_path):
        """Test that logs are written to file."""
        log_file = tmp_path / "test.jsonl"
        logger = StructuredLogger(log_file=log_file)

        logger.info(MessageCode.SIM001, "Message 1", key1="value1")
        logger.error(MessageCode.AGT005, "Message 2", key2="value2")

        logger.close()

        # Verify file contents
        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 2

        log1 = json.loads(lines[0])
        assert log1["code"] == "SIM001"
        assert log1["message"] == "Message 1"
        assert log1["context"]["key1"] == "value1"

        log2 = json.loads(lines[1])
        assert log2["code"] == "AGT005"
        assert log2["message"] == "Message 2"
        assert log2["context"]["key2"] == "value2"

    def test_logger_all_levels(self, tmp_path):
        """Test all logging levels."""
        log_file = tmp_path / "test.jsonl"
        logger = StructuredLogger(log_file=log_file, min_level=LogLevel.DEBUG)

        logger.debug(MessageCode.PRF001, "Debug")
        logger.info(MessageCode.SIM001, "Info")
        logger.warning(MessageCode.PER004, "Warning")
        logger.error(MessageCode.AGT005, "Error")
        logger.critical(MessageCode.SIM002, "Critical")

        logger.close()

        with open(log_file) as f:
            lines = f.readlines()

        assert len(lines) == 5
        levels = [json.loads(line)["level"] for line in lines]
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class TestPerformanceTimer:
    """Test PerformanceTimer context manager."""

    def test_performance_timer_success(self, tmp_path):
        """Test timer on successful operation."""
        log_file = tmp_path / "test.jsonl"
        logger = StructuredLogger(log_file=log_file, min_level=LogLevel.DEBUG)

        with PerformanceTimer(logger, MessageCode.PRF001, "Test operation", step=1):
            pass  # Simulate work

        logger.close()

        with open(log_file) as f:
            log = json.loads(f.read())

        assert log["code"] == "PRF001"
        assert log["message"] == "Test operation completed"
        assert log["context"]["step"] == 1
        assert log["duration_ms"] is not None
        assert log["duration_ms"] >= 0

    def test_performance_timer_with_exception(self, tmp_path):
        """Test timer when exception occurs."""
        log_file = tmp_path / "test.jsonl"
        logger = StructuredLogger(log_file=log_file, min_level=LogLevel.DEBUG)

        try:
            with PerformanceTimer(logger, MessageCode.PRF001, "Test operation", step=2):
                raise ValueError("Test error")
        except ValueError:
            pass

        logger.close()

        with open(log_file) as f:
            log = json.loads(f.read())

        assert log["level"] == "ERROR"
        assert log["code"] == "PRF001"
        assert log["message"] == "Test operation failed"
        assert log["context"]["step"] == 2
        assert log["context"]["error"] == "Test error"
        assert log["duration_ms"] is not None

    def test_performance_timer_measures_time(self, tmp_path):
        """Test that timer actually measures time."""
        import time

        log_file = tmp_path / "test.jsonl"
        logger = StructuredLogger(log_file=log_file, min_level=LogLevel.DEBUG)

        with PerformanceTimer(logger, MessageCode.PRF001, "Slow operation"):
            time.sleep(0.01)  # Sleep 10ms

        logger.close()

        with open(log_file) as f:
            log = json.loads(f.read())

        # Duration should be at least 10ms
        assert log["duration_ms"] >= 10.0


class TestMessageCodes:
    """Test message code enum."""

    def test_all_message_codes_exist(self):
        """Verify all documented message codes exist."""
        # Simulation codes
        assert MessageCode.SIM001 == "SIM001"
        assert MessageCode.SIM002 == "SIM002"
        assert MessageCode.SIM003 == "SIM003"
        assert MessageCode.SIM004 == "SIM004"
        assert MessageCode.SIM005 == "SIM005"

        # Agent codes
        assert MessageCode.AGT001 == "AGT001"
        assert MessageCode.AGT002 == "AGT002"
        assert MessageCode.AGT003 == "AGT003"
        assert MessageCode.AGT004 == "AGT004"
        assert MessageCode.AGT005 == "AGT005"

        # Persistence codes
        assert MessageCode.PER001 == "PER001"
        assert MessageCode.PER002 == "PER002"
        assert MessageCode.PER003 == "PER003"
        assert MessageCode.PER004 == "PER004"

        # Game engine codes
        assert MessageCode.GME001 == "GME001"
        assert MessageCode.GME002 == "GME002"
        assert MessageCode.GME003 == "GME003"
        assert MessageCode.GME004 == "GME004"

        # Config codes
        assert MessageCode.CFG001 == "CFG001"
        assert MessageCode.CFG002 == "CFG002"

        # Performance codes
        assert MessageCode.PRF001 == "PRF001"
