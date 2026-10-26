// Host test of the compat mDNS packets (station-lite/compat/mdns_packet.c): the question, and the
// answers python-zeroconf gives the server's `_zakhar._tcp` record with. `make station-host-test`.
#include <arpa/inet.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "mdns_packet.h"

typedef struct {
    uint8_t b[1024];
    size_t n;
} pkt_t;

static void u8(pkt_t *p, unsigned v)
{
    p->b[p->n++] = (uint8_t)v;
}

static void u16(pkt_t *p, unsigned v)
{
    u8(p, v >> 8);
    u8(p, v & 0xFF);
}

// A dotted name as labels, ending in a pointer to `tail` when tail is not 0, else in the root label.
static size_t name(pkt_t *p, const char *dotted, size_t tail)
{
    const size_t at = p->n;
    for (const char *s = dotted; *s != '\0';) {
        const char *dot = strchr(s, '.');
        const size_t len = dot != NULL ? (size_t)(dot - s) : strlen(s);
        u8(p, (unsigned)len);
        memcpy(p->b + p->n, s, len);
        p->n += len;
        s += len + (dot != NULL);
    }
    if (tail != 0) {
        u16(p, 0xC000 | (unsigned)tail);
    } else {
        u8(p, 0);
    }
    return at;
}

static void pointer(pkt_t *p, size_t at)
{
    u16(p, 0xC000 | (unsigned)at);
}

// A record's type, class, TTL and rdata length, after its name.
static void rr(pkt_t *p, unsigned type, unsigned rdlen)
{
    u16(p, type);
    u16(p, 0x8001);  // IN, cache flush
    u16(p, 0);
    u16(p, 120);
    u16(p, rdlen);
}

static void header(pkt_t *p, unsigned an, unsigned ar)
{
    p->n = 0;
    u16(p, 0);
    u16(p, 0x8400);  // a response, authoritative
    u16(p, 0);
    u16(p, an);
    u16(p, 0);
    u16(p, ar);
}

// python-zeroconf's answer to the browse: the PTR record, then the A, TXT and SRV records as
// additionals, the A record before the SRV record that names its host.
static void browse_answer(pkt_t *p, bool with_srv)
{
    header(p, 1, with_srv ? 3 : 0);
    const size_t service = name(p, "_zakhar._tcp.local", 0);
    rr(p, 12, 14);
    const size_t instance = name(p, "zakhar-host", service);
    if (!with_srv) return;
    const size_t host = name(p, "zakhar-host.local", 0);
    rr(p, 1, 4);
    u8(p, 192), u8(p, 168), u8(p, 1), u8(p, 10);
    pointer(p, instance);
    rr(p, 16, 1);
    u8(p, 0);
    pointer(p, instance);
    rr(p, 33, 8);
    u16(p, 0), u16(p, 0), u16(p, 8765);
    pointer(p, host);
}

static void query(void)
{
    uint8_t q[64];
    static const uint8_t WANT[] = {0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 7, '_', 'z', 'a', 'k', 'h', 'a', 'r', 4, '_', 't',
                                   'c', 'p', 5, 'l', 'o', 'c', 'a', 'l', 0, 0, 12, 0, 1};
    assert(mdns_query_build(q, sizeof q, "_zakhar._tcp.local", 12) == sizeof WANT);
    assert(memcmp(q, WANT, sizeof WANT) == 0);
    assert(mdns_query_build(q, sizeof WANT - 1, "_zakhar._tcp.local", 12) == 0);
    assert(mdns_query_build(q, sizeof q, "a..local", 1) == 0);
    // Our own question looped back is not an answer.
    mdns_browse_t b;
    assert(!mdns_parse_browse(q, sizeof WANT, "_zakhar._tcp.local", &b));
}

static void browse(void)
{
    pkt_t p;
    mdns_browse_t b;
    browse_answer(&p, true);
    assert(mdns_parse_browse(p.b, p.n, "_zakhar._tcp.local", &b));
    assert(strcmp(b.instance, "zakhar-host") == 0 && strcmp(b.host, "zakhar-host") == 0 && b.port == 8765);
    char ip[16];
    assert(b.has_addr && strcmp(inet_ntop(AF_INET, &b.addr, ip, sizeof ip), "192.168.1.10") == 0);
    assert(mdns_parse_browse(p.b, p.n, "_ZAKHAR._TCP.local", &b) && b.port == 8765);
    assert(!mdns_parse_browse(p.b, p.n, "_http._tcp.local", &b));
    // Cut inside the SRV record: the PTR answer stands, without its host.
    assert(mdns_parse_browse(p.b, p.n - 3, "_zakhar._tcp.local", &b) && b.host[0] == '\0' && !b.has_addr);

    browse_answer(&p, false);
    assert(mdns_parse_browse(p.b, p.n, "_zakhar._tcp.local", &b));
    assert(strcmp(b.instance, "zakhar-host") == 0 && b.host[0] == '\0' && b.port == 0 && !b.has_addr);
}

static void a_record(void)
{
    pkt_t p;
    header(&p, 1, 0);
    name(&p, "zakhar-host.local", 0);
    rr(&p, 1, 4);
    u8(&p, 10), u8(&p, 0), u8(&p, 0), u8(&p, 7);
    uint32_t addr = 0;
    assert(mdns_parse_a(p.b, p.n, "Zakhar-Host.local", &addr));
    char ip[16];
    assert(strcmp(inet_ntop(AF_INET, &addr, ip, sizeof ip), "10.0.0.7") == 0);
    assert(!mdns_parse_a(p.b, p.n, "other.local", &addr));

    // A name that points at itself ends the parse instead of looping.
    header(&p, 1, 0);
    pointer(&p, 12);
    rr(&p, 1, 4);
    u8(&p, 10), u8(&p, 0), u8(&p, 0), u8(&p, 7);
    assert(!mdns_parse_a(p.b, p.n, "zakhar-host.local", &addr));
}

int main(void)
{
    query();
    browse();
    a_record();
    puts("mdns_packet: ok");
    return 0;
}
