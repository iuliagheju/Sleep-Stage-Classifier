"""Logging utilities built on top of Rich."""
from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

CONSOLE = Console()


def configure_logging(log_dir: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("sleep_stager")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = RichHandler(console=CONSOLE, markup=True, rich_tracebacks=True)
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "run.log")
        file_handler.setLevel(level)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(file_handler)
    return logger
