import logging
from typing import Optional


_LOGGER: Optional[logging.Logger] = None


def get_logger(name: str = "algo") -> logging.Logger:
    """Return a module-level logger configured once."""
    global _LOGGER
    if _LOGGER is not None:
        return _LOGGER
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    _LOGGER = logger
    return logger