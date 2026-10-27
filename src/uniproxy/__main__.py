"""The station bridge on its own, outside the zakhar server:

    python -m src.uniproxy --port P --server ws://host:port/device

serves the stations' /uni.ws on port P (all addresses) and links every station to the zakhar
server's device endpoint. The server serves the same /uni.ws next to its /device itself.
"""

import argparse

from aiohttp import web
from loguru import logger

from src.uniproxy.uniproxy import Bridge


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m src.uniproxy", description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, required=True, help="port of /uni.ws, the stations' uniProxyUrl")
    ap.add_argument("--server", required=True, help="the zakhar server's device endpoint, ws://host:port/device")
    args = ap.parse_args()
    bridge = Bridge(lambda: args.server)

    async def close_stations(_app: web.Application) -> None:
        await bridge.close()

    app = web.Application()
    app.add_routes([web.get("/uni.ws", bridge.handle)])
    app.on_shutdown.append(close_stations)
    logger.info(f"station bridge: /uni.ws on port {args.port}, zakhar server {args.server}")
    web.run_app(app, port=args.port, access_log=None, print=None)


if __name__ == "__main__":
    main()
