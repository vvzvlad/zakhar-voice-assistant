// Both I2S buses in the formats of the stock firmware. Microphones on I2S1: 16 kHz, 32-bit slots,
// stereo, Philips; the left slot is the XMOS channel 0, the right slot channel 1. Speaker on I2S0:
// 48 kHz, 16-bit samples in 32-bit slots, stereo, Philips, running all the time so the XMOS echo
// reference on the same wire never stops.
#include "audio.h"

#include <stdatomic.h>
#include <string.h>

#include "audio_core.h"
#include "board.h"
#include "driver/i2s_std.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "leds.h"
#include "wake_word.h"
#include "zakhar_link.h"

static const char *TAG = "audio";

#define MIC_RATE 16000
#define MIC_STEP 160    // frames per read: one 10 ms wake-word feature step
#define WAKE_GAIN 4     // the stock micro_wake_word gain_factor on channel 1
// The tap: 65536 frames = 4.096 s (2000 ms pre-roll plus the wait for the server), a power of two;
// readers keep 1 s away from the end being overwritten.
#define TAP_FRAMES (1u << 16)
#define TAP_MARGIN MIC_RATE

#define SPK_RATE 48000
#define SPK_BLOCK 720  // frames per write: 15 ms, the stock DMA buffer
#define LATE_CHECK_BLOCKS 67  // the speaker task reports late buffers about once a second

static i2s_chan_handle_t s_tx, s_rx;
static tap_t s_tap;
// DMA buffers the speaker played as silence because the speaker task had not refilled them in
// time: the driver's sending queue overflowed (i2s_common.c, i2s_dma_tx_callback). Each is 15 ms
// of silence that no stream reported, the speaker task having been kept off the CPU.
static _Atomic uint32_t s_late;

static bool IRAM_ATTR on_late(i2s_chan_handle_t handle, i2s_event_data_t *event, void *ctx)
{
    (void)handle;
    (void)event;
    (void)ctx;
    atomic_fetch_add_explicit(&s_late, 1, memory_order_relaxed);
    return false;
}

static void mic_task(void *arg)
{
    (void)arg;
    static int32_t raw[MIC_STEP * 2];
    static int16_t both[MIC_STEP * 2];
    static int16_t ch1[MIC_STEP];
    for (;;) {
        size_t got = 0;
        if (i2s_channel_read(s_rx, raw, sizeof raw, &got, portMAX_DELAY) != ESP_OK) continue;
        const size_t frames = got / (2 * sizeof(int32_t));
        pcm_split(raw, frames, both, ch1, WAKE_GAIN);
        tap_write(&s_tap, both, frames);
        wake_word_feed(ch1, frames);
    }
}

static void speaker_task(void *arg)
{
    (void)arg;
    static int16_t mono[SPK_BLOCK];
    static int16_t stereo[SPK_BLOCK * 2];
    // The buffers that went out before this task started are not late.
    uint32_t seen = atomic_load_explicit(&s_late, memory_order_relaxed);
    for (unsigned blocks = 1;; blocks++) {
        if (blocks % LATE_CHECK_BLOCKS == 0) {
            const uint32_t late = atomic_load_explicit(&s_late, memory_order_relaxed);
            if (late != seen) {
                ESP_LOGW(TAG, "speaker task late: %u DMA buffers (%u ms) went out as silence", (unsigned)(late - seen),
                         (unsigned)(late - seen) * SPK_BLOCK * 1000 / SPK_RATE);
                seen = late;
            }
        }
        const size_t n = zl_play_read(mono, SPK_BLOCK);
        leds_set_playing(n > 0);
        memset(mono + n, 0, (SPK_BLOCK - n) * sizeof mono[0]);
        for (size_t i = 0; i < SPK_BLOCK; i++) {
            stereo[2 * i] = mono[i];
            stereo[2 * i + 1] = mono[i];
        }
        size_t written;
        i2s_channel_write(s_tx, stereo, sizeof stereo, &written, portMAX_DELAY);
    }
}

esp_err_t audio_start(void)
{
    // auto_clear: an empty DMA buffer goes out as silence.
    i2s_chan_config_t tx_cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_SLAVE);
    tx_cc.dma_desc_num = 4;
    tx_cc.dma_frame_num = SPK_BLOCK;
    tx_cc.auto_clear = true;
    i2s_std_config_t tx_std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(SPK_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_SPK_BCLK,
            .ws = PIN_SPK_LRCLK,
            .dout = PIN_SPK_DOUT,
            .din = I2S_GPIO_UNUSED,
        },
    };
    tx_std.slot_cfg.slot_bit_width = I2S_SLOT_BIT_WIDTH_32BIT;
    tx_std.slot_cfg.ws_width = 32;
    const i2s_event_callbacks_t tx_cb = {.on_send_q_ovf = on_late};
    esp_err_t err = i2s_new_channel(&tx_cc, &s_tx, NULL);
    if (err == ESP_OK) err = i2s_channel_init_std_mode(s_tx, &tx_std);
    if (err == ESP_OK) err = i2s_channel_register_event_callback(s_tx, &tx_cb, NULL);
    if (err == ESP_OK) err = i2s_channel_enable(s_tx);
    if (err != ESP_OK) return err;

    // The stock DMA: 4 x 256 frames.
    i2s_chan_config_t rx_cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_SLAVE);
    rx_cc.dma_desc_num = 4;
    rx_cc.dma_frame_num = 256;
    const i2s_std_config_t rx_std = {
        .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(MIC_RATE),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = PIN_MIC_BCLK,
            .ws = PIN_MIC_LRCLK,
            .dout = I2S_GPIO_UNUSED,
            .din = PIN_MIC_DIN,
        },
    };
    err = i2s_new_channel(&rx_cc, NULL, &s_rx);
    if (err == ESP_OK) err = i2s_channel_init_std_mode(s_rx, &rx_std);
    if (err == ESP_OK) err = i2s_channel_enable(s_rx);
    if (err != ESP_OK) return err;

    int16_t *tap = heap_caps_calloc(TAP_FRAMES * 2, sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (tap == NULL) {
        ESP_LOGE(TAG, "microphone tap: out of PSRAM");
        return ESP_ERR_NO_MEM;
    }
    tap_init(&s_tap, tap, TAP_FRAMES, TAP_MARGIN, 2);
    // Core 1, away from Wi-Fi and zakhar_link on core 0. The stack holds the TFLite inference.
    if (xTaskCreatePinnedToCore(mic_task, "mic", 8 * 1024, NULL, 5, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "mic task create failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

esp_err_t audio_start_playback(void)
{
    // Above the mic task: a late write would starve the DAC and the echo reference.
    if (xTaskCreatePinnedToCore(speaker_task, "speaker", 3 * 1024, NULL, 6, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "speaker task create failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

uint32_t audio_mic_position(void)
{
    return tap_position(&s_tap);
}

size_t audio_mic_read(uint32_t *pos, int16_t *out, size_t max_frames)
{
    return tap_read(&s_tap, pos, out, max_frames);
}
