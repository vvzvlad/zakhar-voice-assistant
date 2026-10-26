// The event handler type esp_websocket_client dispatches with. There is no event loop: a handler
// registered here is never called.
#pragma once

#include <stdint.h>

#include "esp_err.h"

typedef const char *esp_event_base_t;
typedef void (*esp_event_handler_t)(void *arg, esp_event_base_t base, int32_t id, void *data);

static inline esp_err_t esp_event_handler_register(esp_event_base_t base, int32_t id, esp_event_handler_t handler,
                                                   void *arg)
{
    (void)base;
    (void)id;
    (void)handler;
    (void)arg;
    return ESP_OK;
}
