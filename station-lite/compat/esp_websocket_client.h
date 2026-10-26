// esp_websocket_client over a POSIX TCP socket: ws:// only, client frames masked. A started client
// runs a thread that connects, sends the HTTP upgrade and dispatches its events to the handler:
// CONNECTED; DATA for every frame received, whole (payload_offset 0, data_len = payload_len),
// pings included, each answered with a pong; CLOSED with the server's close code; FINISH when the
// thread ends, whether the connect failed or the link was lost. Each client has a recursive lock,
// held around every dispatch and taken by every send, so a handler may send. A send that cannot go
// out within its timeout returns -1; one cut off midway also shuts the socket down, so the link
// ends with FINISH.
#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "esp_err.h"
#include "esp_event.h"
#include "freertos/FreeRTOS.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct esp_websocket_client *esp_websocket_client_handle_t;

typedef enum {
    WEBSOCKET_EVENT_ANY = -1,
    WEBSOCKET_EVENT_CONNECTED = 1,
    WEBSOCKET_EVENT_DATA = 3,
    WEBSOCKET_EVENT_CLOSED = 4,
    WEBSOCKET_EVENT_FINISH = 7,
} esp_websocket_event_id_t;

#define WS_TRANSPORT_OPCODES_TEXT 0x01
#define WS_TRANSPORT_OPCODES_BINARY 0x02

typedef struct {
    const char *data_ptr;
    int data_len;
    uint8_t op_code;
    esp_websocket_client_handle_t client;
    int payload_len;
    int payload_offset;
    int close_status_code;
} esp_websocket_event_data_t;

// Only uri is used ("ws://<IPv4>:<port>/<path>"): the reconnect and the silence rule are the
// caller's, and a frame is received whole whatever its size.
typedef struct {
    const char *uri;
    bool disable_auto_reconnect;
    bool disable_pingpong_discon;
    int buffer_size;
    int task_stack;
    int task_prio;
    bool task_core_id_set;
    int task_core_id;
} esp_websocket_client_config_t;

esp_websocket_client_handle_t esp_websocket_client_init(const esp_websocket_client_config_t *config);
// One handler per client; `event` is not used: the handler gets every event.
esp_err_t esp_websocket_register_events(esp_websocket_client_handle_t client, esp_websocket_event_id_t event,
                                        esp_event_handler_t handler, void *arg);
esp_err_t esp_websocket_client_start(esp_websocket_client_handle_t client);
// Shuts the socket down, waits for the client's thread to end and frees the client. Not from the
// client's own thread.
esp_err_t esp_websocket_client_destroy(esp_websocket_client_handle_t client);
// The payload length when the frame went out, -1 when not connected or not sent within `timeout`.
int esp_websocket_client_send_text(esp_websocket_client_handle_t client, const char *data, int len, TickType_t timeout);
int esp_websocket_client_send_bin(esp_websocket_client_handle_t client, const char *data, int len, TickType_t timeout);

#ifdef __cplusplus
}
#endif
