// The mDNS packets (RFC 6762) of the compat mdns: the question, and the records of a response that a
// browse and an A query take. Names are dotted, without the trailing dot, compared ignoring case.
// Pure: host-tested (make station-host-test).
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define MDNS_NAME_MAX 256

// A query with one question for `name` of `type`, class IN, multicast answer asked for (QM);
// returns its length, 0 when it does not fit `cap`.
size_t mdns_query_build(uint8_t *out, size_t cap, const char *name, uint16_t type);

typedef struct {
    char instance[MDNS_NAME_MAX];  // the PTR target without ".<service>"
    char host[MDNS_NAME_MAX];      // the instance's SRV target without ".local"; empty: no SRV record
    uint16_t port;
    bool has_addr;
    uint32_t addr;  // network byte order: the A record of the SRV target
} mdns_browse_t;

// A response (QR set) with a PTR record for `service` ("_zakhar._tcp.local"): true, and the first
// such record with the SRV and A records the packet holds for it, in any order and section.
bool mdns_parse_browse(const uint8_t *pkt, size_t n, const char *service, mdns_browse_t *out);
// A response with an A record for `name` ("host.local"): true and its address.
bool mdns_parse_a(const uint8_t *pkt, size_t n, const char *name, uint32_t *addr);
