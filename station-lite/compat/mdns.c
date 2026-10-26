// The compat mdns over a UDP socket per query, bound to port 5353 next to any other responder on the
// board (SO_REUSEADDR) and joined to 224.0.0.251: the answers to a multicast question come back
// multicast to that port. The question is sent again every second until the timeout, as the mdns
// component repeats its search.
#include "mdns.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netinet/in.h>
#include <poll.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "esp_log.h"
#include "mdns_packet.h"

static const char *TAG = "mdns";

#define MDNS_PORT 5353
#define MDNS_GROUP "224.0.0.251"
#define RESEND_MS 1000
#define PACKET_MAX 9000  // RFC 6762 section 17

static int64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static int open_socket(void)
{
    const int fd = socket(AF_INET, SOCK_DGRAM | SOCK_CLOEXEC, 0);
    if (fd < 0) return -1;
    const int one = 1;
    const unsigned char ttl = 255;
    struct sockaddr_in local = {.sin_family = AF_INET, .sin_port = htons(MDNS_PORT), .sin_addr.s_addr = htonl(INADDR_ANY)};
    struct ip_mreq group = {.imr_interface.s_addr = htonl(INADDR_ANY)};
    inet_pton(AF_INET, MDNS_GROUP, &group.imr_multiaddr);
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) != 0 ||
        bind(fd, (struct sockaddr *)&local, sizeof local) != 0 ||
        setsockopt(fd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &group, sizeof group) != 0 ||
        setsockopt(fd, IPPROTO_IP, IP_MULTICAST_TTL, &ttl, sizeof ttl) != 0) {
        const int err = errno;
        close(fd);
        errno = err;
        return -1;
    }
    return fd;
}

// What a query looks for in a response: a browse fills `browse`, an A query `addr`.
typedef struct {
    const char *name;
    uint16_t type;
    mdns_browse_t browse;
    uint32_t addr;
} want_t;

static bool match(const uint8_t *pkt, size_t n, want_t *w)
{
    return w->type == MDNS_TYPE_PTR ? mdns_parse_browse(pkt, n, w->name, &w->browse)
                                    : mdns_parse_a(pkt, n, w->name, &w->addr);
}

static mdns_ip_addr_t *addr_new(uint32_t addr)
{
    mdns_ip_addr_t *a = calloc(1, sizeof *a);
    if (a == NULL) return NULL;
    a->addr.type = ESP_IPADDR_TYPE_V4;
    a->addr.u_addr.ip4.addr = addr;
    return a;
}

static mdns_result_t *result_new(const want_t *w)
{
    mdns_result_t *r = calloc(1, sizeof *r);
    if (r == NULL) return NULL;
    if (w->type == MDNS_TYPE_A) {
        r->addr = addr_new(w->addr);
        return r;
    }
    r->instance_name = strdup(w->browse.instance);
    if (w->browse.host[0] != '\0') {
        r->hostname = strdup(w->browse.host);
        r->port = w->browse.port;
    }
    if (w->browse.has_addr) r->addr = addr_new(w->browse.addr);
    return r;
}

static esp_err_t query(want_t *w, uint32_t timeout_ms, size_t max_results, mdns_result_t **results)
{
    *results = NULL;
    uint8_t question[12 + MDNS_NAME_MAX + 5];
    const size_t qlen = mdns_query_build(question, sizeof question, w->name, w->type);
    if (qlen == 0) return ESP_FAIL;
    const int fd = open_socket();
    if (fd < 0) {
        ESP_LOGE(TAG, "socket on port %d: %s", MDNS_PORT, strerror(errno));
        return ESP_FAIL;
    }
    struct sockaddr_in group = {.sin_family = AF_INET, .sin_port = htons(MDNS_PORT)};
    inet_pton(AF_INET, MDNS_GROUP, &group.sin_addr);
    static uint8_t pkt[PACKET_MAX];  // one query runs at a time: zakhar_link's task asks them
    mdns_result_t **tail = results;
    size_t found = 0;
    const int64_t end = now_ms() + timeout_ms;
    int64_t send_at = now_ms();
    for (int64_t now = send_at; now < end && found < max_results; now = now_ms()) {
        if (now >= send_at) {
            if (sendto(fd, question, qlen, 0, (struct sockaddr *)&group, sizeof group) < 0) {
                ESP_LOGW(TAG, "question for %s not sent: %s", w->name, strerror(errno));
            }
            send_at = now + RESEND_MS;
        }
        struct pollfd p = {.fd = fd, .events = POLLIN};
        const int64_t until = send_at < end ? send_at : end;
        if (poll(&p, 1, (int)(until - now)) <= 0) continue;
        const ssize_t n = recv(fd, pkt, sizeof pkt, 0);
        if (n <= 0 || !match(pkt, (size_t)n, w)) continue;
        mdns_result_t *r = result_new(w);
        if (r == NULL) break;
        *tail = r;
        tail = &r->next;
        found++;
    }
    close(fd);
    return ESP_OK;
}

esp_err_t mdns_init(void)
{
    return ESP_OK;
}

esp_err_t mdns_query_ptr(const char *service, const char *proto, uint32_t timeout_ms, size_t max_results,
                         mdns_result_t **results)
{
    char name[MDNS_NAME_MAX];
    snprintf(name, sizeof name, "%s.%s.local", service, proto);
    want_t w = {.name = name, .type = MDNS_TYPE_PTR};
    return query(&w, timeout_ms, max_results, results);
}

esp_err_t mdns_query_generic(const char *name, const char *service, const char *proto, uint16_t type,
                             mdns_query_transmission_type_t transmission, uint32_t timeout_ms, size_t max_results,
                             mdns_result_t **results)
{
    (void)service;
    (void)proto;
    (void)transmission;
    char host[MDNS_NAME_MAX];
    snprintf(host, sizeof host, "%s.local", name);
    want_t w = {.name = host, .type = type};
    return query(&w, timeout_ms, max_results, results);
}

void mdns_query_results_free(mdns_result_t *results)
{
    while (results != NULL) {
        mdns_result_t *next = results->next;
        for (mdns_ip_addr_t *a = results->addr; a != NULL;) {
            mdns_ip_addr_t *an = a->next;
            free(a);
            a = an;
        }
        free(results->instance_name);
        free(results->hostname);
        free(results);
        results = next;
    }
}
