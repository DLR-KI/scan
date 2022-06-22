"""Logging module of the scan package. Also check __init__.py for the default logger config."""

from __future__ import annotations

import sys

from loguru import logger


def set_logger(level: int | str = "DEBUG") -> None:
    """Set the scan package logger to the desired log level.

    Removes any already existing loggers, so that always exactly one logger is active.

    Args:
        level: The log level. Possible values are:

            - TRACE    - 5
            - DEBUG    - 10
            - INFO     - 20
            - SUCCESS  - 25
            - WARNING  - 30
            - ERROR    - 40
            - CRITICAL - 50

    """
    # Remove any already existing handlers so that we can change the log level during a script. If this is the first
    # the the function was called (e.g. during __init__.py) this removes the default loguru handler, which is good as
    # that one is not MPI save
    logger.remove()

    # Make our own logging handler MPI save (and non blocking) by setting enqueue=True.
    # Also set backtrace=True, diagnose=True for, ostensibly, better exception logging.
    #  NOTE: backtrace/diagnose "may leak sensitive data in prod" according to the loguro documentation
    logger.add(
        sys.stderr,
        level=level,
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        ),
    )

    logger.info(f"Set log level to {level}")

    # # # Test logger printing
    # logger.trace("this is a trace message")
    # logger.debug("this is a debug message")
    # logger.info("this is an info message")
    # logger.success("this is a success message")
    # logger.warning("this is a warning message")
    # logger.error("this is an error message")
    # logger.critical("this is a critical message")
