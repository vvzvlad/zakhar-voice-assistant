#include "volume.h"

#include <stdatomic.h>

#include "aic3204.h"
#include "esp_log.h"
#include "leds.h"
#include "nvs.h"
#include "zakhar_link.h"

static const char *TAG = "volume";

#define NVS_NAMESPACE "voicepe"
#define NVS_KEY "volume"
#define INITIAL 50

static nvs_handle_t s_nvs;
static _Atomic int s_volume = INITIAL;

void volume_init(void)
{
    uint8_t v = INITIAL;
    ESP_ERROR_CHECK(nvs_open(NVS_NAMESPACE, NVS_READWRITE, &s_nvs));
    if (nvs_get_u8(s_nvs, NVS_KEY, &v) == ESP_OK) atomic_store(&s_volume, v);
    leds_set_volume(atomic_load(&s_volume));
}

int volume_get(void)
{
    return atomic_load(&s_volume);
}

void volume_set(int v)
{
    atomic_store(&s_volume, v);
    aic3204_set_volume(v);
    if (nvs_set_u8(s_nvs, NVS_KEY, (uint8_t)v) != ESP_OK || nvs_commit(s_nvs) != ESP_OK) {
        ESP_LOGE(TAG, "not stored");
    }
    leds_set_volume(v);
    zl_report_volume(v);
    ESP_LOGI(TAG, "%d %%", v);
}
