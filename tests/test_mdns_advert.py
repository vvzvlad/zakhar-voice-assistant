"""The device server's mDNS service record. Only the record is built: no multicast."""

from src import mdns_advert
from src.mdns_advert import _main_gateway, build_info

# /proc/net/route of a host whose wg-quick full tunnel keeps its default route in a
# table of its own: the main table still holds the LAN gateway 10.31.41.1 on eth0.
ROUTES = """\
Iface\tDestination\tGateway \tFlags\tRefCnt\tUse\tMetric\tMask\t\tMTU\tWindow\tIRTT
eth0\t00000000\t01291F0A\t0003\t0\t0\t100\t00000000\t0\t0\t0
wg-nebula\t011E140A\t00000000\t0005\t0\t0\t0\tFFFFFFFF\t0\t0\t0
eth0\t00291F0A\t00000000\t0001\t0\t0\t100\t00FFFFFF\t0\t0\t0
"""


def test_concrete_host_is_advertised():
    info = build_info("10.31.41.59", 8200, "nebula.lc")
    assert info.type == "_zakhar._tcp.local."
    assert info.server == "zakhar-nebula.local."
    assert info.port == 8200
    assert info.parsed_addresses() == ["10.31.41.59"]


def test_wildcard_host_advertises_the_lan_address(monkeypatch):
    monkeypatch.setattr(mdns_advert, "_lan_ipv4", lambda: "10.31.41.59")
    assert build_info("0.0.0.0", 8200, "nebula").parsed_addresses() == ["10.31.41.59"]


def test_main_table_gateway_is_read_from_proc_route():
    assert _main_gateway(ROUTES) == "10.31.41.1"
    assert _main_gateway(ROUTES.replace("eth0\t00000000", "eth0\t00000001")) is None
