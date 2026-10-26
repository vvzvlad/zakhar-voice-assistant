// ESP-IDF's error codes, the ones the compat layer returns.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef int esp_err_t;

#define ESP_OK 0
#define ESP_FAIL -1
#define ESP_ERR_NO_MEM 0x101

const char *esp_err_to_name(esp_err_t code);

#ifdef __cplusplus
}
#endif
