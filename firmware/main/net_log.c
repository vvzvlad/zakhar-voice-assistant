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
#include "net_log.h"

#include <errno.h>
#include <stdarg.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

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

static vprintf_like_t s_prev = vprintf;  // esp_log's default until the hook returns the real one
static RingbufHandle_t s_ring;
static TaskHandle_t s_task;
static int s_listener;
static _Atomic bool s_client;  // a client is connected: lines go to the ring

// A line cut at `len` bytes ends with '\n' on a character boundary: the last byte kept before
// the '\n' is the last of a whole UTF-8 character.
static void cut(char *line, int len)
{
    int end = len - 1;
    while (end > 0 && ((unsigned char)line[end] & 0xC0) == 0x80) end--;
    line[end] = '\n';
    line[end + 1] = '\0';
}

static int net_vprintf(const char *fmt, va_list args)
{
    va_list measure, format;
    va_copy(measure, args);
    va_copy(format, args);
    const int n = s_prev(fmt, args);
    if (atomic_load(&s_client) && xTaskGetCurrentTaskHandle() != s_task) {
        int len = vsnprintf(NULL, 0, fmt, measure);
        if (len > LOG_LINE_MAX) len = LOG_LINE_MAX;
        char *line;
        if (len > 0 && xRingbufferSendAcquire(s_ring, (void **)&line, (size_t)len + 1, 0) == pdTRUE) {
            if (vsnprintf(line, (size_t)len + 1, fmt, format) > len) cut(line, len);
            xRingbufferSendComplete(s_ring, line);
        }
    }
    va_end(format);
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

// The lines queued for a client that is gone are not for the next one.
static void drop_queued(void)
{
    size_t n;
    void *line;
    while ((line = xRingbufferReceive(s_ring, &n, 0)) != NULL) vRingbufferReturnItem(s_ring, line);
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
            drop_queued();
        }
    }
}

void net_log_start(void)
{
    s_ring = xRingbufferCreate(RING_BYTES, RINGBUF_TYPE_NOSPLIT);
    if (s_ring == NULL) {
        ESP_LOGE(TAG, "ring buffer: out of memory");
        return;
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
        return;
    }
    if (xTaskCreatePinnedToCore(net_log_task, "net_log", 3 * 1024, NULL, 1, &s_task, 0) != pdPASS) {
        ESP_LOGE(TAG, "task create failed");
        return;
    }
    s_prev = esp_log_set_vprintf(net_vprintf);
    ESP_LOGI(TAG, "log on TCP port %d", PORT);
}
