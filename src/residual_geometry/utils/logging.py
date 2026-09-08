import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger with a timestamp formatter."""
    logger = logging.getLogger(name)

    # Only add handlers if the logger doesn't already have them, to avoid
    # duplicate log entries when the function is called multiple times.
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def log_config_hash(logger: logging.Logger, config_hash: str) -> None:
    """Log the config hash for reproducibility tracking."""
    logger.info("Config hash: %s", config_hash)
