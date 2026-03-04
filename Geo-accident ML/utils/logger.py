################################################################################
# FILE: utils/logger.py
################################################################################
import logging
import os
from logging.handlers import RotatingFileHandler

_FORMATTER = logging.Formatter(
    fmt="%(asctime)s | %(levelname)-8s | %(name)s -> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

_COLOURS = {
    logging.DEBUG:    "\033[94m",
    logging.INFO:     "\033[92m",
    logging.WARNING:  "\033[93m",
    logging.ERROR:    "\033[91m",
    logging.CRITICAL: "\033[95m",
}
_RESET = "\033[0m"


class _ColourHandler(logging.StreamHandler):
    """StreamHandler that adds ANSI colour codes to level names."""
    def emit(self, record):
        colour = _COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{record.levelname}{_RESET}"
        super().emit(record)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger with:
      - Colour-coded console output (INFO and above)
      - Rotating file handler in logs/app.log (DEBUG and above, max 2 MB x 3)
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    # Console
    ch = _ColourHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(_FORMATTER)

    # File (rotating)
    os.makedirs("logs", exist_ok=True)
    fh = RotatingFileHandler("logs/app.log", maxBytes=2*1024*1024, backupCount=3, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(_FORMATTER)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger