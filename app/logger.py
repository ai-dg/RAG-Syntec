"""
Application logging configuration.

This module configures structured JSON logs for the application.
"""

import logging
import sys


def configure_logging(log_level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=("%(asctime)s | %(levelname)s | " "%(name)s | %(message)s"),
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
