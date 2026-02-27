# logger.py
# =========
# Centralized logging setup for the entire project.
#
# Why use a logger instead of print()?
# - Logs include timestamps so you know when something happened
# - You can set a level: DEBUG shows everything, INFO shows key steps,
#   WARNING/ERROR shows only problems
# - You can save logs to a file while also printing to terminal
# - In production this is how you track what the API is doing
#
# Usage in any script:
#   from utils.logger import get_logger
#   logger = get_logger(__name__)
#   logger.info("Processing started")
#   logger.warning("Something looks off")
#   logger.error("Something broke")

import logging
import os
from pathlib import Path
from datetime import datetime

# ── Log file setup ─────────────────────────────────────────────────────────────
# Save logs to logs/ directory in project root
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Each run gets its own log file with a timestamp in the name
# Example: logs/run_2024-01-12_14-35-22.log
LOG_FILE = LOG_DIR / f"run_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"


def get_logger(name: str) -> logging.Logger:
    """
    Create and return a logger with the given name.
    Logs go to both the terminal (console) and a log file.

    Args:
        name: Usually pass __name__ so the log shows which script it came from.

    Returns:
        A configured Python logger.

    Example:
        logger = get_logger(__name__)
        logger.info("Model loaded successfully")
    """

    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Format ─────────────────────────────────────────────────────────────────
    # Example output: 2024-01-12 14:35:22 | INFO     | train_layoutlm | Epoch 1 done
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ── Console handler — prints to terminal ───────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # ── File handler — saves to log file ───────────────────────────────────────
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger