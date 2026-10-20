#include "volume.h"

#include <stdatomic.h>

#include "aic3204.h"
#include "audio_core.h"
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
static _Atomic uint32_t s_music_gain = 65536;  // volume_music_gain()
static _Atomic uint32_t s_voice_gain = 65536;  // volume_voice_gain()

// The music at the volume, the replies at the volume plus the boost (audio_core volume_levels).
static void dac_apply(void)
{
    const volume_levels_t l = volume_levels(atomic_load(&s_volume), s_boost);
    aic3204_set_volume(l.dac);
    atomic_store(&s_music_gain, l.music);
    atomic_store(&s_voice_gain, l.voice);
}

uint32_t volume_music_gain(void)
{
    return atomic_load(&s_music_gain);
}

uint32_t volume_voice_gain(void)
{
    return atomic_load(&s_voice_gain);
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
