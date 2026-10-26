// The mdns component's queries, as zakhar_link makes them: a PTR browse for a service and an A query
// for a host, both multicast questions asked from UDP port 5353 (the answers come back multicast to
// 224.0.0.251:5353). A query ends at its timeout or as soon as max_results answers came in.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

#define MDNS_TYPE_A 0x0001
#define MDNS_TYPE_PTR 0x000C

typedef enum {
    MDNS_QUERY_MULTICAST,
} mdns_query_transmission_type_t;

#define ESP_IPADDR_TYPE_V4 0

typedef struct {
    uint32_t addr;  // network byte order
} esp_ip4_addr_t;

typedef struct {
    union {
        esp_ip4_addr_t ip4;
    } u_addr;
    uint8_t type;
} esp_ip_addr_t;

typedef struct mdns_ip_addr_s {
    esp_ip_addr_t addr;
    struct mdns_ip_addr_s *next;
} mdns_ip_addr_t;

// One answer. A browse fills instance_name (the instance label) and, from the answer's SRV record,
// hostname (its target without ".local") and port; addr holds the A record of that host when the
// answer carries it. An A query fills addr.
typedef struct mdns_result_s {
    struct mdns_result_s *next;
    char *instance_name;
    char *hostname;
    uint16_t port;
    mdns_ip_addr_t *addr;
} mdns_result_t;

#define IPSTR "%d.%d.%d.%d"
#define IP2STR(ip)                                                                         \
    ((const uint8_t *)&(ip)->addr)[0], ((const uint8_t *)&(ip)->addr)[1],                  \
        ((const uint8_t *)&(ip)->addr)[2], ((const uint8_t *)&(ip)->addr)[3]

esp_err_t mdns_init(void);
// <service>.<proto>.local PTR.
esp_err_t mdns_query_ptr(const char *service, const char *proto, uint32_t timeout_ms, size_t max_results,
                         mdns_result_t **results);
// <name>.local of `type` (MDNS_TYPE_A only); service and proto must be NULL.
esp_err_t mdns_query_generic(const char *name, const char *service, const char *proto, uint16_t type,
                             mdns_query_transmission_type_t transmission, uint32_t timeout_ms, size_t max_results,
                             mdns_result_t **results);
void mdns_query_results_free(mdns_result_t *results);

#ifdef __cplusplus
}
#endif
