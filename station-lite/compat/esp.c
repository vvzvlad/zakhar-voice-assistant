// The small ESP-IDF services: the log, error names, heap_caps, the timer, the temperature sensor and
// the station's signal.
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "driver/temperature_sensor.h"
#include "esp_err.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "esp_wifi.h"

void esp_log_line(char level, const char *tag, const char *fmt, ...)
{
    char text[1024];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(text, sizeof text, fmt, ap);
    va_end(ap);
    // One call per line: stdio locks the stream for it, so lines of different threads never mix.
    fprintf(stderr, "%c (%lld) %s: %s\n", level, (long long)(esp_timer_get_time() / 1000), tag, text);
}

const char *esp_err_to_name(esp_err_t code)
{
    switch (code) {
    case ESP_OK:
        return "ESP_OK";
    case ESP_FAIL:
        return "ESP_FAIL";
    case ESP_ERR_NO_MEM:
        return "ESP_ERR_NO_MEM";
    default:
        return "UNKNOWN ERROR";
    }
}

int64_t esp_timer_get_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

// ---------------------------------------------------------------- heap_caps

void *heap_caps_malloc(size_t size, uint32_t caps)
{
    (void)caps;
    return malloc(size);
}

void *heap_caps_aligned_alloc(size_t alignment, size_t size, uint32_t caps)
{
    (void)caps;
    return aligned_alloc(alignment, size);
}

void heap_caps_free(void *ptr)
{
    free(ptr);
}

// MemAvailable, bytes; 0 when /proc/meminfo has none.
static size_t mem_available(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    if (f == NULL) return 0;
    char line[128];
    unsigned long kb = 0;
    while (fgets(line, sizeof line, f) != NULL && sscanf(line, "MemAvailable: %lu kB", &kb) != 1) {
    }
    fclose(f);
    return (size_t)kb * 1024;
}

static size_t s_min_free = SIZE_MAX;  // the lowest MemAvailable read: stats are taken on one task

size_t heap_caps_get_free_size(uint32_t caps)
{
    if (caps & MALLOC_CAP_SPIRAM) return 0;
    const size_t free_now = mem_available();
    if (free_now < s_min_free) s_min_free = free_now;
    return free_now;
}

size_t heap_caps_get_minimum_free_size(uint32_t caps)
{
    if (caps & MALLOC_CAP_SPIRAM) return 0;
    (void)heap_caps_get_free_size(caps);
    return s_min_free;
}

size_t heap_caps_get_largest_free_block(uint32_t caps)
{
    return heap_caps_get_free_size(caps);
}

size_t heap_caps_get_total_size(uint32_t caps)
{
    (void)caps;
    return 0;
}

// ---------------------------------------------------------------- the temperature sensor

struct temperature_sensor {
    int unused;
};

static struct temperature_sensor s_tsens;

esp_err_t temperature_sensor_install(const temperature_sensor_config_t *cfg, temperature_sensor_handle_t *out)
{
    (void)cfg;
    *out = &s_tsens;
    return ESP_OK;
}

esp_err_t temperature_sensor_uninstall(temperature_sensor_handle_t sensor)
{
    (void)sensor;
    return ESP_OK;
}

esp_err_t temperature_sensor_enable(temperature_sensor_handle_t sensor)
{
    (void)sensor;
    return ESP_OK;
}

esp_err_t temperature_sensor_get_celsius(temperature_sensor_handle_t sensor, float *celsius)
{
    (void)sensor;
    FILE *f = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (f == NULL) return ESP_FAIL;
    long milli;
    const bool ok = fscanf(f, "%ld", &milli) == 1;
    fclose(f);
    if (!ok) return ESP_FAIL;
    *celsius = (float)milli / 1000.0f;
    return ESP_OK;
}

// ---------------------------------------------------------------- Wi-Fi

// The board's rtl88x2cs driver puts dBm + 100 in /proc/net/wireless's level column (67 there while
// wpa_supplicant's signal_poll says -33).
esp_err_t esp_wifi_sta_get_ap_info(wifi_ap_record_t *ap)
{
    FILE *f = fopen("/proc/net/wireless", "r");
    if (f == NULL) return ESP_FAIL;
    char line[256];
    int level = 0;
    bool found = false;
    while (!found && fgets(line, sizeof line, f) != NULL) {
        found = sscanf(line, " wlan0: %*s %*s %d", &level) == 1;
    }
    fclose(f);
    if (!found) return ESP_FAIL;
    ap->rssi = (int8_t)(level - 100);
    return ESP_OK;
}
