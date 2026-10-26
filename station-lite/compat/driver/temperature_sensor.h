// The SoC temperature from /sys/class/thermal/thermal_zone0/temp.
#pragma once

#include "esp_err.h"

typedef struct temperature_sensor *temperature_sensor_handle_t;

typedef struct {
    int range_min;
    int range_max;
} temperature_sensor_config_t;

#define TEMPERATURE_SENSOR_CONFIG_DEFAULT(min, max) {.range_min = (min), .range_max = (max)}

esp_err_t temperature_sensor_install(const temperature_sensor_config_t *cfg, temperature_sensor_handle_t *out);
esp_err_t temperature_sensor_uninstall(temperature_sensor_handle_t sensor);
esp_err_t temperature_sensor_enable(temperature_sensor_handle_t sensor);
// ESP_FAIL when the file cannot be read.
esp_err_t temperature_sensor_get_celsius(temperature_sensor_handle_t sensor, float *celsius);
