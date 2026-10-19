// A vprintf hook on esp_log and one TCP client on port 23. The hook always prints to the serial
// port first; only while a client is connected it also formats the line into a ring buffer slot,
// without waiting: a line that does not fit is dropped, so logging never blocks a task. The
// net_log task drains the ring to the client. Live lines only, no history. A new connection
// replaces the old client; a send error closes the client.
//
// The scheme is ESPHome's (task_log_buffer_esp32.cpp): the line is measured, a slot of that size
// acquired and the line formatted straight into it, so the hook's frame holds no line buffer.
//
// No recursion: the net_log task's own lines never enter the ring, and it logs nothing while a
// client is connected.
//
// Independently of the TCP client, while forwarding is on (net_log_forward) the hook formats each
// line into a second ring, the server ring, the same way; the zakhar_link task takes the lines
// from it (net_log_take) and sends them to the server.
#include "net_log.h"

#include <errno.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/ringbuf.h"
#include "freertos/task.h"
#include "lwip/sockets.h"

static const char *TAG = "net_log";

#define PORT 23
#define LOG_LINE_MAX 512      // a longer line is cut at a UTF-8 character boundary
#define RING_BYTES (8 * 1024) // a dozen of the longest lines with their item headers
#define SEND_TIMEOUT_S 1      // a client that stops reading is dropped, so a new one can come in
// The lines waiting for the server: what a few seconds without a link produce, in PSRAM.
#define SERVER_RING_BYTES (16 * 1024)

static vprintf_like_t s_prev = vprintf;  // esp_log's default until the hook returns the real one
static RingbufHandle_t s_ring;
static TaskHandle_t s_task;
static int s_listener;
static _Atomic bool s_client;  // a client is connected: lines go to the ring
static RingbufHandle_t s_server_ring;
static _Atomic bool s_forward;  // the server asks for the log: lines go to the server ring

// A line cut at `len` bytes ends with '\n' on a character boundary: the last byte kept before
// the '\n' is the last of a whole UTF-8 character.
static void cut(char *line, int len)
{
    int end = len - 1;
    while (end > 0 && ((unsigned char)line[end] & 0xC0) == 0x80) end--;
    line[end] = '\n';
    line[end + 1] = '\0';
}

// Formats a line of `len` bytes (measured, at most LOG_LINE_MAX) straight into a slot of `ring`,
// without waiting: when no slot is free the line is dropped.
static void ring_put(RingbufHandle_t ring, int len, const char *fmt, va_list args)
{
    char *line;
    if (xRingbufferSendAcquire(ring, (void **)&line, (size_t)len + 1, 0) != pdTRUE) return;
    if (vsnprintf(line, (size_t)len + 1, fmt, args) > len) cut(line, len);
    xRingbufferSendComplete(ring, line);
}

static int net_vprintf(const char *fmt, va_list args)
{
    va_list measure, tcp, server;
    va_copy(measure, args);
    va_copy(tcp, args);
    va_copy(server, args);
    const int n = s_prev(fmt, args);
    const bool to_tcp = atomic_load(&s_client) && xTaskGetCurrentTaskHandle() != s_task;
    const bool to_server = atomic_load(&s_forward);
    if (to_tcp || to_server) {
        int len = vsnprintf(NULL, 0, fmt, measure);
        if (len > LOG_LINE_MAX) len = LOG_LINE_MAX;
        if (len > 0 && to_tcp) ring_put(s_ring, len, fmt, tcp);
        if (len > 0 && to_server) ring_put(s_server_ring, len, fmt, server);
    }
    va_end(server);
    va_end(tcp);
    va_end(measure);
    return n;
}

static bool send_all(int fd, const char *p, size_t n)
{
    while (n > 0) {
        const int sent = send(fd, p, n, 0);
        if (sent <= 0) return false;
        p += sent;
        n -= (size_t)sent;
    }
    return true;
}

// The lines queued for a client that is gone are not for the next one; nor are those queued for
// the server once it no longer asks for the log.
static void drop_queued(RingbufHandle_t ring)
{
    size_t n;
    void *line;
    while ((line = xRingbufferReceive(ring, &n, 0)) != NULL) vRingbufferReturnItem(ring, line);
}

static void net_log_task(void *arg)
{
    (void)arg;
    int client = -1;
    for (;;) {
        // Blocks for the first client; with one connected, only looks for a newer one.
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(s_listener, &fds);
        struct timeval now = {0, 0};
        if (select(s_listener + 1, &fds, NULL, NULL, client < 0 ? NULL : &now) > 0) {
            const int fd = accept(s_listener, NULL, NULL);
            if (fd >= 0) {
                const struct timeval timeout = {SEND_TIMEOUT_S, 0};
                setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof timeout);
                if (client >= 0) close(client);
                client = fd;
                atomic_store(&s_client, true);
            }
        }
        if (client < 0) continue;
        size_t n;
        char *line = xRingbufferReceive(s_ring, &n, pdMS_TO_TICKS(100));
        if (line == NULL) continue;
        const bool ok = send_all(client, line, strlen(line));
        vRingbufferReturnItem(s_ring, line);
        if (!ok) {
            atomic_store(&s_client, false);
            close(client);
            client = -1;
            drop_queued(s_ring);
        }
    }
}

// The TCP ring, the listener on PORT and the net_log task; false when one of them failed (logged).
static bool tcp_start(void)
{
    s_ring = xRingbufferCreate(RING_BYTES, RINGBUF_TYPE_NOSPLIT);
    if (s_ring == NULL) {
        ESP_LOGE(TAG, "ring buffer: out of memory");
        return false;
    }
    s_listener = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);
    const int one = 1;
    const struct sockaddr_in addr = {
        .sin_family = AF_INET,
        .sin_port = htons(PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };
    if (s_listener < 0 || setsockopt(s_listener, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one) != 0 ||
        bind(s_listener, (const struct sockaddr *)&addr, sizeof addr) != 0 || listen(s_listener, 1) != 0) {
        ESP_LOGE(TAG, "listening on port %d failed: errno %d", PORT, errno);
        return false;
    }
    if (xTaskCreatePinnedToCore(net_log_task, "net_log", 3 * 1024, NULL, 1, &s_task, 0) != pdPASS) {
        ESP_LOGE(TAG, "task create failed");
        return false;
    }
    return true;
}

void net_log_start(void)
{
    s_server_ring = xRingbufferCreateWithCaps(SERVER_RING_BYTES, RINGBUF_TYPE_NOSPLIT, MALLOC_CAP_SPIRAM);
    if (s_server_ring == NULL) ESP_LOGE(TAG, "server ring buffer: out of PSRAM");
    const bool tcp = tcp_start();
    if (!tcp && s_server_ring == NULL) return;
    s_prev = esp_log_set_vprintf(net_vprintf);
    if (tcp) ESP_LOGI(TAG, "log on TCP port %d", PORT);
}

void net_log_forward(bool on)
{
    if (s_server_ring == NULL) return;
    if (atomic_exchange(&s_forward, on) && !on) drop_queued(s_server_ring);
}

size_t net_log_take(char *buf, size_t max)
{
    size_t n = 0;
    if (s_server_ring == NULL) return 0;
    // A line is at most LOG_LINE_MAX bytes, so the next one is taken only while it surely fits.
    while (max - n >= LOG_LINE_MAX) {
        size_t size;
        char *line = xRingbufferReceive(s_server_ring, &size, 0);
        if (line == NULL) break;
        const size_t len = strlen(line);
        memcpy(buf + n, line, len);
        n += len;
        vRingbufferReturnItem(s_server_ring, line);
    }
    return n;
}
