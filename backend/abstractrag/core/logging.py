"""Single place to configure logging, so CLI, API and scripts log the same way."""

import logging
import sys

_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

    # These libraries are chatty at INFO during model loading and HTTP calls.
    for noisy in ("httpx", "httpcore", "urllib3", "docling"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
