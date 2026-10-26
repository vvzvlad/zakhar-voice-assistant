// The station's signal from /proc/net/wireless, and the disconnect event zakhar_link counts (never
// raised here: Wi-Fi is the board's own business).
#pragma once

#include <stdint.h>

#include "esp_err.h"
#include "esp_event.h"

#define WIFI_EVENT "WIFI_EVENT"
#define WIFI_EVENT_STA_DISCONNECTED 5

typedef struct {
    uint8_t reason;
} wifi_event_sta_disconnected_t;

typedef struct {
    int8_t rssi;
} wifi_ap_record_t;

// The signal of wlan0, dBm; ESP_FAIL when /proc/net/wireless has no wlan0 line.
esp_err_t esp_wifi_sta_get_ap_info(wifi_ap_record_t *ap);
