#include "mdns_packet.h"

#include <string.h>
#include <strings.h>

#define TYPE_A 1
#define TYPE_PTR 12
#define TYPE_SRV 33
#define CLASS_IN 1
#define MAX_HOPS 16  // compression pointers followed in one name

size_t mdns_query_build(uint8_t *out, size_t cap, const char *name, uint16_t type)
{
    if (cap < 12) return 0;
    memset(out, 0, 12);  // id 0, a standard query
    out[5] = 1;          // one question
    size_t n = 12;
    for (const char *label = name; *label != '\0';) {
        const char *dot = strchr(label, '.');
        const size_t len = dot != NULL ? (size_t)(dot - label) : strlen(label);
        if (len == 0 || len > 63 || n + 1 + len + 5 > cap) return 0;
        out[n++] = (uint8_t)len;
        memcpy(out + n, label, len);
        n += len;
        label += len + (dot != NULL);
    }
    if (n + 5 > cap) return 0;
    out[n++] = 0;
    out[n++] = (uint8_t)(type >> 8);
    out[n++] = (uint8_t)type;
    out[n++] = 0;
    out[n++] = CLASS_IN;  // the unicast-response bit clear: QM
    return n;
}

static uint16_t u16(const uint8_t *p)
{
    return (uint16_t)(p[0] << 8 | p[1]);
}

// The name at `at`, compression pointers followed, into `out` (MDNS_NAME_MAX); returns the offset
// just past the name where it starts, 0 when it is malformed.
static size_t read_name(const uint8_t *p, size_t n, size_t at, char *out)
{
    size_t end = 0, o = 0;
    int hops = 0;
    for (;;) {
        if (at >= n) return 0;
        const uint8_t l = p[at];
        if ((l & 0xC0) == 0xC0) {
            if (at + 1 >= n || ++hops > MAX_HOPS) return 0;
            if (end == 0) end = at + 2;
            at = (size_t)(l & 0x3F) << 8 | p[at + 1];
            continue;
        }
        if (l & 0xC0) return 0;
        at++;
        if (l == 0) break;
        if (at + l > n || o + l + 2 > MDNS_NAME_MAX) return 0;
        if (o > 0) out[o++] = '.';
        memcpy(out + o, p + at, l);
        o += l;
        at += l;
    }
    out[o] = '\0';
    return end != 0 ? end : at;
}

typedef struct {
    char name[MDNS_NAME_MAX];
    uint16_t type;
    size_t rdata;
    uint16_t rdlen;
} rr_t;

// The first record of `type` owned by `name` in any section of a response; false when there is
// none, the packet is not a response or it is malformed before one.
static bool find(const uint8_t *p, size_t n, uint16_t type, const char *name, rr_t *r)
{
    if (n < 12 || !(p[2] & 0x80)) return false;
    unsigned records = (unsigned)u16(p + 6) + u16(p + 8) + u16(p + 10);
    size_t at = 12;
    for (unsigned q = u16(p + 4); q > 0; q--) {
        at = read_name(p, n, at, r->name);
        if (at == 0 || at + 4 > n) return false;
        at += 4;
    }
    for (; records > 0; records--) {
        at = read_name(p, n, at, r->name);
        if (at == 0 || at + 10 > n) return false;
        r->type = u16(p + at);
        r->rdlen = u16(p + at + 8);
        r->rdata = at + 10;
        if (r->rdata + r->rdlen > n) return false;
        at = r->rdata + r->rdlen;
        if (r->type == type && strcasecmp(r->name, name) == 0) return true;
    }
    return false;
}

// `name` without the suffix "." + `suffix`, or whole when it does not end so, into `out`.
static void strip(char *out, const char *name, const char *suffix)
{
    size_t n = strlen(name);
    const size_t s = strlen(suffix);
    if (n > s + 1 && name[n - s - 1] == '.' && strcasecmp(name + n - s, suffix) == 0) n -= s + 1;
    memcpy(out, name, n);
    out[n] = '\0';
}

bool mdns_parse_browse(const uint8_t *pkt, size_t n, const char *service, mdns_browse_t *out)
{
    rr_t r;
    char instance[MDNS_NAME_MAX], target[MDNS_NAME_MAX];
    if (!find(pkt, n, TYPE_PTR, service, &r) || read_name(pkt, n, r.rdata, instance) == 0) return false;
    memset(out, 0, sizeof *out);
    strip(out->instance, instance, service);
    if (!find(pkt, n, TYPE_SRV, instance, &r) || r.rdlen < 7 || read_name(pkt, n, r.rdata + 6, target) == 0) {
        return true;
    }
    out->port = u16(pkt + r.rdata + 4);
    strip(out->host, target, "local");
    out->has_addr = mdns_parse_a(pkt, n, target, &out->addr);
    return true;
}

bool mdns_parse_a(const uint8_t *pkt, size_t n, const char *name, uint32_t *addr)
{
    rr_t r;
    if (!find(pkt, n, TYPE_A, name, &r) || r.rdlen != 4) return false;
    memcpy(addr, pkt + r.rdata, 4);
    return true;
}
