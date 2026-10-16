"""mDNS advertisement of the device server as `_zakhar._tcp`.

The device firmware carries no server address: before every connect it browses
`_zakhar._tcp` and connects to ws://<address>:<port>/device of the first answer.
The advertisement lives in app.py next to the DeviceManager, not inside it, so the
tests that start a DeviceManager never multicast on the LAN.
"""

import os
import socket
import struct

from loguru import logger
from zeroconf import ServiceInfo
from zeroconf.asyncio import AsyncZeroconf

SERVICE_TYPE = "_zakhar._tcp.local."


def _main_gateway(route_text: str) -> str | None:
    """The default gateway in the text of /proc/net/route (the main routing table)."""
    for line in route_text.splitlines()[1:]:
        fields = line.split()
        if fields[1] == "00000000" and fields[7] == "00000000":  # destination and mask 0
            return socket.inet_ntoa(struct.pack("<I", int(fields[2], 16)))
    return None


def _lan_ipv4() -> str:
    """The IPv4 the host reaches the gateway of its main routing table from.

    Not the address of the route to the internet: a full-tunnel VPN (wg-quick) puts
    its default route into a table of its own, and that address is unreachable from
    the LAN. /proc/net/route lists the main table only; without it (macOS, a dev
    run) the internet route is used. connect() on a UDP socket only picks the route;
    no packet is sent."""
    gateway = None
    if os.path.exists("/proc/net/route"):
        with open("/proc/net/route") as f:
            gateway = _main_gateway(f.read())
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.connect((gateway or "1.1.1.1", 9))
        return s.getsockname()[0]


def build_info(host: str, port: int, hostname: str) -> ServiceInfo:
    """The service record of a device server bound to host:port on `hostname`.

    A concrete bind host is advertised as is; the wildcard ("0.0.0.0" or empty)
    advertises the host's LAN IPv4 (_lan_ipv4). The A record lives under its own host
    name, zakhar-<short host name>.local: a device that lost the A record of the
    browse answer asks for that name, and it cannot clash with the host's own
    hostname record."""
    address = host if host not in ("", "0.0.0.0") else _lan_ipv4()
    name = f"zakhar-{hostname.split('.')[0]}"
    return ServiceInfo(
        SERVICE_TYPE,
        f"{name}.{SERVICE_TYPE}",
        port=port,
        parsed_addresses=[address],
        server=f"{name}.local.",
        # An empty TXT is one zero byte (RFC 6763 6.1). Without properties python-zeroconf
        # sends zero bytes, and the ESP mdns parser drops the rest of the packet at such a
        # record, the SRV record with it.
        properties=b"\x00",
    )


class MdnsAdvert:
    """Announces the device server while it runs; start() again after a rebind."""

    def __init__(self):
        self._zc: AsyncZeroconf | None = None
        self._info: ServiceInfo | None = None

    async def start(self, host: str, port: int) -> None:
        self._info = build_info(host, port, socket.gethostname())
        self._zc = AsyncZeroconf()
        announced = await self._zc.async_register_service(self._info)
        await announced
        logger.info(f"mDNS: {self._info.name} at {self._info.parsed_addresses()[0]}:{port}")

    async def stop(self) -> None:
        goodbye = await self._zc.async_unregister_service(self._info)
        await goodbye
        await self._zc.async_close()
