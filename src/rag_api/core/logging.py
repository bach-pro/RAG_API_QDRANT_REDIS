from __future__ import annotations

import logging
import sys


def configure_api_logging() -> logging.Logger:
    logger = logging.getLogger("rag_api")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)

    return logger
