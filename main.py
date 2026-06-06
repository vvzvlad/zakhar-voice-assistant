import asyncio
import sys

from loguru import logger

from src.app import main as app_main
from src.settings import settings


def main():
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level)
    logger.info("Starting zakhar-voice-assistant")
    asyncio.run(app_main())


if __name__ == "__main__":
    main()
