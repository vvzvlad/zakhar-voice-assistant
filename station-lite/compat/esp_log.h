// ESP_LOGx on stderr, one line per call in ESP-IDF's format: "I (ms since boot) tag: text". Debug
// lines are compiled out, as at ESP-IDF's default INFO level. stdio.h comes along, as with ESP-IDF's
// esp_log.h: zakhar_link.c calls snprintf without including it.
#pragma once

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void esp_log_line(char level, const char *tag, const char *fmt, ...) __attribute__((format(printf, 3, 4)));

#ifdef __cplusplus
}
#endif

#define ESP_LOGE(tag, fmt, ...) esp_log_line('E', tag, fmt, ##__VA_ARGS__)
#define ESP_LOGW(tag, fmt, ...) esp_log_line('W', tag, fmt, ##__VA_ARGS__)
#define ESP_LOGI(tag, fmt, ...) esp_log_line('I', tag, fmt, ##__VA_ARGS__)
#define ESP_LOGD(tag, fmt, ...)                                    \
    do {                                                           \
        if (0) esp_log_line('D', tag, fmt, ##__VA_ARGS__);         \
    } while (0)
