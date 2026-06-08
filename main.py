import asyncio

from loguru import logger

from src import config_store
from src.app import main as app_main
from src.logging_setup import setup_logging


def main():
    # Read the log level tolerantly straight from the config doc (no Settings):
    # the app's own composition root re-parses and validates the full document.
    doc = config_store.load()
    level = (doc.get("core") or {}).get("log_level", "INFO")
    setup_logging(level)
    logger.info("Starting zakhar-voice-assistant")
    asyncio.run(app_main())


if __name__ == "__main__":
    main()
