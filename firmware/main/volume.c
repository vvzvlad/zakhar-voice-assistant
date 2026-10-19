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
static int s_boost;  // the controls task only: the server's volume_boost, 0 from boot

// The DAC plays the volume plus the boost, within 1..100 %; a volume of 0 stays muted. The floor
// is 1, not 0: 0 mutes the DAC, so a quiet room would silence the reply instead of lowering it.
static void dac_apply(void)
{
    const int base = atomic_load(&s_volume);
    const int v = base + s_boost;
    aic3204_set_volume(base == 0 ? 0 : v < 1 ? 1 : v > 100 ? 100 : v);
}

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
    dac_apply();
    if (nvs_set_u8(s_nvs, NVS_KEY, (uint8_t)v) != ESP_OK || nvs_commit(s_nvs) != ESP_OK) {
        ESP_LOGE(TAG, "not stored");
    }
    leds_set_volume(v);
    zl_report_volume(v);
    ESP_LOGI(TAG, "%d %%", v);
}

void volume_set_boost(int points)
{
    s_boost = points;
    dac_apply();
    ESP_LOGI(TAG, "boost %+d %%", points);
}
