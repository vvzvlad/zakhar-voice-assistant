// The Yandex Station Lite board as a zakhar device, in place of its stock software: the wake-word
// engine, then the capture side, then the zakhar WebSocket device protocol through zakhar_link, then
// the speaker and the LED ring, as the Voice PE firmware brings them up (firmware/main/app_main.c).
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "audio.h"
#include "esp_log.h"
#include "freertos/task.h"
#include "leds.h"
#include "volume.h"
#include "wake_word.h"
#include "zakhar_link.h"

static const char *TAG = "stationlite";

#define VOLUME_DEFAULT 50

// The volume and the server's boost: written on zakhar_link's client thread, read there too.
static int s_volume;
static int s_boost;
static char s_volume_path[PATH_MAX + sizeof "/volume"];  // `volume` next to the executable

// ---------------------------------------------------------------- the volume

// The file's volume, VOLUME_DEFAULT when there is none.
static int volume_load(void)
{
    char exe[PATH_MAX];
    const ssize_t n = readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return VOLUME_DEFAULT;
    exe[n] = '\0';
    char *slash = strrchr(exe, '/');
    if (slash != NULL) *slash = '\0';
    snprintf(s_volume_path, sizeof s_volume_path, "%s/volume", exe);
    FILE *f = fopen(s_volume_path, "r");
    int v = VOLUME_DEFAULT;
    if (f != NULL) {
        if (fscanf(f, "%d", &v) != 1) v = VOLUME_DEFAULT;
        fclose(f);
    }
    return v;
}

static void volume_store(int percent)
{
    FILE *f = s_volume_path[0] != '\0' ? fopen(s_volume_path, "w") : NULL;
    if (f == NULL || fprintf(f, "%d\n", percent) < 0) ESP_LOGW(TAG, "volume %d %% not stored in %s", percent, s_volume_path);
    if (f != NULL) fclose(f);
}

static void volume_apply(void)
{
    audio_set_gain(volume_gain(s_volume, s_boost));
}

static void on_set_volume(int percent)
{
    s_volume = percent;
    volume_apply();
    volume_store(percent);
    leds_set_volume(percent);
    zl_report_volume(percent);
}

static void on_volume_boost(int points)
{
    s_boost = points;
    volume_apply();
}

// ---------------------------------------------------------------- zakhar_link callbacks

static void on_state(zl_state_t state, const char *text)
{
    (void)text;
    switch (state) {
    case ZL_STATE_LISTENING:
        leds_set_phase(LED_PHASE_LISTENING);
        break;
    case ZL_STATE_TRANSCRIBING:
    case ZL_STATE_THINKING:
        leds_set_phase(LED_PHASE_THINKING);
        break;
    case ZL_STATE_SPEAKING:
        leds_set_phase(LED_PHASE_REPLYING);
        break;
    case ZL_STATE_IDLE:
        leds_set_phase(LED_PHASE_IDLE);
        break;
    case ZL_STATE_ERROR:
        leds_set_phase(LED_PHASE_ERROR);
        break;
    }
}

static void on_detect(void)
{
    zl_wake(WAKE_WORD_NAME);
}

static bool on_model(zl_model_t kind, uint8_t *data, size_t size)
{
    wake_word_model_t model = WAKE_WORD_MODEL_WAKE;
    switch (kind) {
    case ZL_MODEL_WAKE:
        model = WAKE_WORD_MODEL_WAKE;
        break;
    case ZL_MODEL_VAD:
        model = WAKE_WORD_MODEL_VAD;
        break;
    }
    return wake_word_load(model, data, size);
}

// ---------------------------------------------------------------- start

// The hello id: wlan0's MAC as 12 lowercase hex, as the Voice PE and the wall panel send theirs.
static bool device_id(char id[13])
{
    FILE *f = fopen("/sys/class/net/wlan0/address", "r");
    unsigned m[6];
    const bool ok = f != NULL && fscanf(f, "%x:%x:%x:%x:%x:%x", &m[0], &m[1], &m[2], &m[3], &m[4], &m[5]) == 6;
    if (f != NULL) fclose(f);
    if (ok) snprintf(id, 13, "%02x%02x%02x%02x%02x%02x", m[0], m[1], m[2], m[3], m[4], m[5]);
    return ok;
}

int main(void)
{
    char id[13];
    if (!device_id(id)) {
        ESP_LOGE(TAG, "no device id: /sys/class/net/wlan0/address not read");
        return 1;
    }
    const char *fw = "stationlite " STATION_VERSION;
    s_volume = volume_load();
    volume_apply();
    leds_set_volume(s_volume);

    wake_word_set_detect_cb(on_detect);
    // The capture thread feeds the engine from its first frame; the models come later from the server.
    if (!wake_word_init() || !audio_start_capture()) return 1;

    const zl_config_t cfg = {
        .id = id,
        .fw = fw,
        .screen = false,
        .mic_channels = 2,
        .preroll_ms = 2000,
        .mic_position = audio_mic_position,
        .mic_read = audio_mic_read,
        .wake_peak_take = wake_word_take_peak,
        .play_rate = 48000,
        .play_buffer_ms = 4000,
        .volume = s_volume,
    };
    const zl_callbacks_t cb = {
        .on_connection = leds_set_link,
        .on_state = on_state,
        .on_config = wake_word_set_cutoffs,
        .on_set_volume = on_set_volume,
        .on_capture = leds_set_capture,
        .on_model = on_model,
        .on_volume_boost = on_volume_boost,
    };
    zl_start(&cfg, &cb);
    // zl_start() reports no failure; its task exists only when the start got through, and
    // zl_play_read() and zl_report_volume() must not run without it.
    if (xTaskGetHandle("zakhar_link") == NULL) {
        ESP_LOGE(TAG, "zakhar_link did not start");
        return 1;
    }
    if (!audio_start_playback()) return 1;
    leds_start();
    ESP_LOGI(TAG, "%s up as %s, volume %d %%", fw, id, s_volume);
    for (;;) pause();
}
