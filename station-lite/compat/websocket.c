// The compat esp_websocket_client (esp_websocket_client.h): one thread per started client, a
// blocking socket it reads frames from, and sends from any thread under the client's lock.
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/random.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#include "esp_log.h"
#include "esp_websocket_client.h"
#include "ws_frame.h"

static const char *TAG = "websocket";

// The connect, the handshake and the client's own sends (pong, close): esp_websocket_client's
// default network_timeout_ms.
#define NETWORK_TIMEOUT_MS 10000
#define RESPONSE_MAX 2048  // the handshake response's headers
#define CLOSE_NO_STATUS 1005

struct esp_websocket_client {
    struct sockaddr_in addr;
    char host[16];
    unsigned port;
    char path[128];
    esp_event_handler_t handler;
    void *handler_arg;
    // Recursive: held around each dispatch and taken by each send, so a handler may send.
    pthread_mutex_t lock;
    bool connected;  // under lock: the handshake is done and nothing failed since
    uint8_t *tx;     // under lock: the frame being sent
    size_t tx_cap;
    // The socket and the stop request, between the client's thread and destroy.
    pthread_mutex_t sock_lock;
    int fd;
    bool stop;
    pthread_t thread;
    bool started;
    uint8_t *rx;  // the client's thread only: the frame being received
    size_t rx_cap;
};

static int64_t now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

// Waits for `events` on fd until `deadline` (now_ms() time); false on timeout or a socket error.
static bool wait_fd(int fd, short events, int64_t deadline)
{
    for (;;) {
        const int64_t left = deadline - now_ms();
        if (left <= 0) return false;
        struct pollfd p = {.fd = fd, .events = events};
        const int r = poll(&p, 1, left > INT32_MAX ? INT32_MAX : (int)left);
        if (r > 0) return (p.revents & events) != 0;
        if (r < 0 && errno != EINTR) return false;
    }
}

static bool send_all(int fd, const uint8_t *p, size_t n, int64_t deadline)
{
    while (n > 0) {
        if (!wait_fd(fd, POLLOUT, deadline)) return false;
        const ssize_t k = send(fd, p, n, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (k < 0) {
            if (errno == EAGAIN || errno == EINTR) continue;
            return false;
        }
        p += k;
        n -= (size_t)k;
    }
    return true;
}

// Blocks until n bytes came in; false when the socket ended or failed (destroy shuts it down).
static bool recv_all(int fd, uint8_t *p, size_t n)
{
    while (n > 0) {
        const ssize_t k = recv(fd, p, n, 0);
        if (k == 0) return false;
        if (k < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += k;
        n -= (size_t)k;
    }
    return true;
}

static void dispatch(esp_websocket_client_handle_t c, int32_t id, uint8_t op, const uint8_t *data, size_t len,
                     int close_code)
{
    if (c->handler == NULL) return;
    esp_websocket_event_data_t d = {
        .data_ptr = (const char *)data,
        .data_len = (int)len,
        .op_code = op,
        .client = c,
        .payload_len = (int)len,
        .payload_offset = 0,
        .close_status_code = close_code,
    };
    c->handler(c->handler_arg, "WEBSOCKET_EVENTS", id, &d);
}

// Under c->lock. One frame, masked; a frame that does not go out whole by `deadline` shuts the
// socket down: what was sent of it would garble the stream.
static int send_frame(esp_websocket_client_handle_t c, uint8_t opcode, const void *data, size_t len, int64_t deadline)
{
    if (!c->connected) return -1;
    if (WS_HEADER_MAX + len > c->tx_cap) {
        uint8_t *tx = realloc(c->tx, WS_HEADER_MAX + len);
        if (tx == NULL) return -1;
        c->tx = tx;
        c->tx_cap = WS_HEADER_MAX + len;
    }
    uint8_t mask[4];
    if (getrandom(mask, sizeof mask, 0) != (ssize_t)sizeof mask) return -1;
    const size_t head = ws_header_build(c->tx, opcode, len, mask);
    memcpy(c->tx + head, data, len);
    ws_mask(c->tx + head, len, mask, 0);
    if (!send_all(c->fd, c->tx, head + len, deadline)) {
        c->connected = false;
        shutdown(c->fd, SHUT_RDWR);
        return -1;
    }
    return (int)len;
}

// The TCP connect and the HTTP upgrade.
static bool link_open(esp_websocket_client_handle_t c)
{
    const int fd = socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0);
    if (fd < 0) {
        ESP_LOGE(TAG, "socket: %s", strerror(errno));
        return false;
    }
    pthread_mutex_lock(&c->sock_lock);
    const bool stop = c->stop;
    if (!stop) c->fd = fd;
    pthread_mutex_unlock(&c->sock_lock);
    if (stop) {
        close(fd);
        return false;
    }
    const int64_t deadline = now_ms() + NETWORK_TIMEOUT_MS;
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    const int flags = fcntl(fd, F_GETFL);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int err = 0;
    socklen_t err_len = sizeof err;
    if ((connect(fd, (struct sockaddr *)&c->addr, sizeof c->addr) != 0 && errno != EINPROGRESS) ||
        !wait_fd(fd, POLLOUT, deadline) || getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &err_len) != 0 || err != 0) {
        ESP_LOGW(TAG, "connect to %s:%u failed: %s", c->host, c->port, strerror(err != 0 ? err : errno));
        return false;
    }
    fcntl(fd, F_SETFL, flags);
    uint8_t nonce[16];
    char key[25], request[512], response[RESPONSE_MAX];
    if (getrandom(nonce, sizeof nonce, 0) != (ssize_t)sizeof nonce) return false;
    ws_key(nonce, key);
    const size_t n = ws_request(request, sizeof request, c->host, c->port, c->path, key);
    if (n == 0 || !send_all(fd, (const uint8_t *)request, n, deadline)) {
        ESP_LOGW(TAG, "upgrade request to %s:%u not sent", c->host, c->port);
        return false;
    }
    // Byte by byte up to the end of the headers: a frame may follow them at once.
    size_t got = 0;
    while (got < 4 || memcmp(response + got - 4, "\r\n\r\n", 4) != 0) {
        if (got == sizeof response - 1 || !wait_fd(fd, POLLIN, deadline) || recv(fd, response + got, 1, 0) != 1) {
            ESP_LOGW(TAG, "no upgrade response from %s:%u", c->host, c->port);
            return false;
        }
        got++;
    }
    response[got] = '\0';
    if (!ws_response_ok(response, key)) {
        ESP_LOGW(TAG, "upgrade refused by %s:%u: %.*s", c->host, c->port, (int)strcspn(response, "\r"), response);
        return false;
    }
    return true;
}

// Frames until the link ends: a close from the server, a socket error, or destroy.
static void receive(esp_websocket_client_handle_t c)
{
    for (;;) {
        uint8_t head[WS_HEADER_MAX];
        ws_header_t h;
        if (!recv_all(c->fd, head, 2) || !recv_all(c->fd, head + 2, ws_header_size(head) - 2)) return;
        ws_header_parse(head, ws_header_size(head), &h);
        if (h.len > c->rx_cap) {
            uint8_t *rx = realloc(c->rx, h.len);
            if (rx == NULL) {
                ESP_LOGE(TAG, "frame of %llu B: out of memory", (unsigned long long)h.len);
                return;
            }
            c->rx = rx;
            c->rx_cap = h.len;
        }
        if (!recv_all(c->fd, c->rx, h.len)) return;
        if (h.masked) ws_mask(c->rx, h.len, h.mask, 0);
        pthread_mutex_lock(&c->lock);
        if (h.opcode == WS_OP_CLOSE) {
            const int code = h.len >= 2 ? c->rx[0] << 8 | c->rx[1] : CLOSE_NO_STATUS;
            dispatch(c, WEBSOCKET_EVENT_CLOSED, h.opcode, NULL, 0, code);
            send_frame(c, WS_OP_CLOSE, c->rx, h.len >= 2 ? 2 : 0, now_ms() + NETWORK_TIMEOUT_MS);
            pthread_mutex_unlock(&c->lock);
            return;
        }
        if (h.opcode == WS_OP_PING) send_frame(c, WS_OP_PONG, c->rx, h.len, now_ms() + NETWORK_TIMEOUT_MS);
        dispatch(c, WEBSOCKET_EVENT_DATA, h.opcode, c->rx, h.len, 0);
        pthread_mutex_unlock(&c->lock);
    }
}

static void *client_main(void *arg)
{
    esp_websocket_client_handle_t c = arg;
    const bool open = link_open(c);
    pthread_mutex_lock(&c->lock);
    c->connected = open;
    if (open) dispatch(c, WEBSOCKET_EVENT_CONNECTED, 0, NULL, 0, 0);
    pthread_mutex_unlock(&c->lock);
    if (open) receive(c);
    pthread_mutex_lock(&c->lock);
    c->connected = false;
    dispatch(c, WEBSOCKET_EVENT_FINISH, 0, NULL, 0, 0);
    pthread_mutex_unlock(&c->lock);
    return NULL;
}

esp_websocket_client_handle_t esp_websocket_client_init(const esp_websocket_client_config_t *config)
{
    esp_websocket_client_handle_t c = calloc(1, sizeof *c);
    if (c == NULL) return NULL;
    if (sscanf(config->uri, "ws://%15[0-9.]:%u%127s", c->host, &c->port, c->path) != 3 || c->port > 65535 ||
        inet_pton(AF_INET, c->host, &c->addr.sin_addr) != 1) {
        ESP_LOGE(TAG, "not a ws://<IPv4>:<port>/<path> URI: %s", config->uri);
        free(c);
        return NULL;
    }
    c->addr.sin_family = AF_INET;
    c->addr.sin_port = htons((uint16_t)c->port);
    pthread_mutexattr_t ma;
    pthread_mutexattr_init(&ma);
    pthread_mutexattr_settype(&ma, PTHREAD_MUTEX_RECURSIVE);
    pthread_mutex_init(&c->lock, &ma);
    pthread_mutexattr_destroy(&ma);
    pthread_mutex_init(&c->sock_lock, NULL);
    c->fd = -1;
    return c;
}

esp_err_t esp_websocket_register_events(esp_websocket_client_handle_t client, esp_websocket_event_id_t event,
                                        esp_event_handler_t handler, void *arg)
{
    (void)event;
    client->handler = handler;
    client->handler_arg = arg;
    return ESP_OK;
}

esp_err_t esp_websocket_client_start(esp_websocket_client_handle_t client)
{
    if (pthread_create(&client->thread, NULL, client_main, client) != 0) return ESP_FAIL;
    client->started = true;
    return ESP_OK;
}

esp_err_t esp_websocket_client_destroy(esp_websocket_client_handle_t client)
{
    pthread_mutex_lock(&client->sock_lock);
    client->stop = true;
    if (client->fd >= 0) shutdown(client->fd, SHUT_RDWR);
    pthread_mutex_unlock(&client->sock_lock);
    if (client->started) pthread_join(client->thread, NULL);
    if (client->fd >= 0) close(client->fd);
    pthread_mutex_destroy(&client->lock);
    pthread_mutex_destroy(&client->sock_lock);
    free(client->tx);
    free(client->rx);
    free(client);
    return ESP_OK;
}

static int client_send(esp_websocket_client_handle_t c, uint8_t opcode, const char *data, int len, TickType_t timeout)
{
    const int64_t deadline = now_ms() + timeout;
    struct timespec until;
    clock_gettime(CLOCK_REALTIME, &until);
    until.tv_sec += timeout / 1000;
    until.tv_nsec += (long)(timeout % 1000) * 1000000;
    if (until.tv_nsec >= 1000000000) {
        until.tv_sec++;
        until.tv_nsec -= 1000000000;
    }
    if (pthread_mutex_timedlock(&c->lock, &until) != 0) return -1;
    const int sent = send_frame(c, opcode, data, (size_t)len, deadline);
    pthread_mutex_unlock(&c->lock);
    return sent;
}

int esp_websocket_client_send_text(esp_websocket_client_handle_t client, const char *data, int len, TickType_t timeout)
{
    return client_send(client, WS_OP_TEXT, data, len, timeout);
}

int esp_websocket_client_send_bin(esp_websocket_client_handle_t client, const char *data, int len, TickType_t timeout)
{
    return client_send(client, WS_OP_BINARY, data, len, timeout);
}
