"""IsharaConnect Desktop Client Entrypoint."""

import sys
import logging

logger = logging.getLogger("ishara_desktop")


def main() -> None:
    """Launch IsharaConnect desktop application."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Initializing IsharaConnect Desktop UI...")
    # PyQt6 QApplication launch logic will reside here
    print("IsharaConnect Desktop App launcher ready.")


if __name__ == "__main__":
    main()
