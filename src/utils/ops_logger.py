"""
Operational logging for infrastructure components.

This module provides logging for system operations (API requests, job queue,
workers) as opposed to simulation content logs which use StructuredLogger.

Usage:
    from utils.ops_logger import get_ops_logger
    logger = get_ops_logger("api")
    logger.info("Request received", extra={"path": "/api/simulations"})
"""

import logging
import os
import socket
from pathlib import Path
from typing import Optional


# Cache loggers to avoid duplicate handlers
_loggers: dict[str, logging.Logger] = {}

# Log directory (configurable via env)
LOG_DIR = Path(os.environ.get("APART_LOG_DIR", "logs/operations"))


class ContextFormatter(logging.Formatter):
    """Formatter that includes extra fields in the log message."""

    def format(self, record: logging.LogRecord) -> str:
        # Build base message
        base = super().format(record)

        # Append extra fields if present (excluding standard LogRecord attrs)
        standard_attrs = {
            "name", "msg", "args", "created", "filename", "funcName",
            "levelname", "levelno", "lineno", "module", "msecs",
            "pathname", "process", "processName", "relativeCreated",
            "stack_info", "exc_info", "exc_text", "thread", "threadName",
            "taskName", "message",
        }

        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in standard_attrs and not k.startswith("_")
        }

        if extras:
            extra_str = " ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base} {extra_str}"

        return base


def get_ops_logger(
    component: str,
    log_to_file: Optional[bool] = None,
) -> logging.Logger:
    """
    Get or create an operational logger for a component.

    Args:
        component: Component name (e.g., "api", "worker", "queue")
        log_to_file: Override file logging (default: check APART_LOG_TO_FILE env)

    Returns:
        Configured logger instance
    """
    logger_name = f"apart.{component}"

    # Return cached logger if exists
    if logger_name in _loggers:
        return _loggers[logger_name]

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Let handlers control level
    logger.propagate = False  # Don't duplicate to root logger

    # Determine log level from env
    log_level = os.environ.get("APART_LOG_LEVEL", "INFO").upper()

    # Console handler (always)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, log_level, logging.INFO))
    console.setFormatter(ContextFormatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(console)

    # File handler (if enabled)
    should_log_to_file = log_to_file
    if should_log_to_file is None:
        should_log_to_file = os.environ.get("APART_LOG_TO_FILE", "").lower() in ("1", "true", "yes")

    if should_log_to_file:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Include hostname for distributed workers
        hostname = socket.gethostname()
        if component == "worker":
            log_file = LOG_DIR / f"worker-{hostname}.log"
        else:
            log_file = LOG_DIR / f"{component}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # File gets everything
        file_handler.setFormatter(ContextFormatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        logger.addHandler(file_handler)

    _loggers[logger_name] = logger
    return logger


def log_with_context(
    logger: logging.Logger,
    level: int,
    message: str,
    run_id: Optional[str] = None,
    job_id: Optional[str] = None,
    **extra,
) -> None:
    """
    Log a message with optional run_id and job_id for correlation.

    Args:
        logger: Logger instance
        level: Log level (e.g., logging.INFO)
        message: Log message
        run_id: Simulation run ID for correlation
        job_id: Redis job ID for correlation
        **extra: Additional context fields
    """
    context = {}
    if run_id:
        context["run_id"] = run_id
    if job_id:
        context["job_id"] = job_id
    context.update(extra)

    logger.log(level, message, extra=context)
