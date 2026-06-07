import sys

from loguru import logger

from src.smarthome_mcp_server import main


def run():
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.info("Starting zakhar smart-home MCP server")
    main()


if __name__ == "__main__":
    run()
