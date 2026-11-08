from __future__ import annotations

import sys

from loguru import logger


def configure_logging(level: str = "INFO", json: bool = False) -> None:
    logger.remove()
    if json:
        logger.add(sys.stderr, serialize=True, level=level)
    else:
        logger.add(
            sys.stderr,
            level=level,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level:<8}</level> | "
                "<cyan>{name}</cyan> - {message}"
            ),
        )
