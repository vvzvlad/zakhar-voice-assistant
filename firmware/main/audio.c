// Both I2S buses in the formats of the stock firmware. Microphones on I2S1: 16 kHz, 32-bit slots,
// stereo, Philips; the left slot is the XMOS channel 0, the right slot channel 1. Speaker on I2S0:
// 48 kHz, 16-bit samples in 32-bit slots, stereo, Philips, running all the time so the XMOS echo
// reference on the same wire never stops.
// Channel 0 is the XMOS AGC output, channel 1 its noise-suppressed output, the wake word's input.
// Nothing here cancels the speaker's own playback; what ESP-SR's AEC does on this board is
// measured in docs/echo-cancellation.md.
// With the server's auto_volume on, every 15 s of quiet (nothing playing, no session open, no
// capture held) channel 0 is the raw microphone 0 for about a second: the room's noise is measured
// on its 32-bit slots, before the XMOS's noise suppression and AGC, and goes to the server as
// `ambient`. Playback, a session or a capture ends a measurement at once, and nothing is sent.
#include "audio.h"

#include <stdatomic.h>
#include <string.h>

#include "audio_core.h"
#include "board.h"
#include "driver/i2s_std.h"
#include "esp_attr.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "leds.h"
#include "music.h"
#include "volume.h"
#include "wake_word.h"
#include "xmos.h"
#include "zakhar_link.h"

static const char *TAG = "audio";

#define MIC_RATE 16000
#define WAKE_GAIN 4     // the stock micro_wake_word gain_factor on channel 1
// The tap: 131072 frames = 8.192 s, a power of two; readers keep 1 s away from the end being
// overwritten. What is left holds the 2000 ms pre-roll plus a send that zakhar_link waits for up to
// its 3000 ms SEND_TIMEOUT_MS: with less, a stall while the pre-roll goes out makes the reader skip
// forward and silently cuts audio out of the phrase.
#define TAP_FRAMES (1u << 17)
#define TAP_MARGIN MIC_RATE
_Static_assert(TAP_FRAMES - TAP_MARGIN >= (2000 + 3000) * MIC_RATE / 1000,
               "the tap must hold the pre-roll plus zakhar_link's send timeout");

#define SPK_RATE 48000
#define SPK_BLOCK 720  // frames per write: 15 ms, the stock DMA buffer
// The TX DMA buffers, SPK_BLOCK frames each: a block just written plays out after all of them.
#define SPK_DMA_BUFS 4
#define LATE_CHECK_BLOCKS 67  // the speaker task reports late buffers about once a second

// The mic task's unit: 512 frames = 32 ms, the protocol's microphone frame. It is two whole RX DMA
// buffers, so no read leaves one half read: IDF's i2s_channel_read skips the rest of a half-read
// buffer when its queue is nearly full.
#define CHUNK 512
#define MIC_DMA_FRAMES (CHUNK / 2)

// Auto volume: a measurement starts every AMBIENT_EVERY chunks of quiet (15 s) and takes
// AMBIENT_CHUNKS raw chunks (0.99 s).
#define AMBIENT_EVERY (15 * MIC_RATE / CHUNK)
#define AMBIENT_CHUNKS (MIC_RATE / CHUNK)

static i2s_chan_handle_t s_tx, s_rx;
static tap_t s_tap;
// The server's auto_volume, taken by the mic task per chunk; off until the first `config`.
static _Atomic bool s_auto_volume;
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

// The mic task's state. The buffers hold one chunk: the raw slots, the tap's interleaved frames
// and the wake-word input.
typedef struct {
    int32_t raw[CHUNK * 2];
    int16_t both[CHUNK * 2], wake[CHUNK];
    mic_chunks_t chunks;
    ambient_sched_t amb;
    bool wrote;  // the stage write the previous chunk asked for went through
} mic_state_t;

static mic_state_t s_mic;

// Reads until the chunk is whole.
static void read_chunk(mic_state_t *s)
{
    size_t fill = 0;
    while (fill < CHUNK) {
        size_t got = 0;
        if (i2s_channel_read(s_rx, &s->raw[fill * 2], (CHUNK - fill) * 2 * sizeof(int32_t), &got, portMAX_DELAY) ==
            ESP_OK) {
            fill += got / (2 * sizeof(int32_t));
        }
    }
}

// Channel 0's stage for the chunk: raw while a measurement wants it, else AGC.
static mic_chunk_t follow_stage(mic_state_t *s, bool measuring)
{
    const mic_chunk_t c = mic_chunks_step(&s->chunks, measuring, s->wrote);
    // Written as soon as the chunk is in, so the chunk's processing does not delay the switch.
    s->wrote = c.write && xmos_ch0_raw(c.write_raw) == ESP_OK;
    return c;
}

// Per chunk: channel 0's stage follows the auto volume measurement, which asks for the raw
// channel 0 and takes only the chunks reported raw; then the tap (channel 0 as read, at gain 1)
// and the wake-word engine (channel 1 at WAKE_GAIN).
static void mic_task(void *arg)
{
    (void)arg;
    mic_state_t *s = &s_mic;
    mic_chunks_init(&s->chunks);
    ambient_sched_init(&s->amb, AMBIENT_EVERY, AMBIENT_CHUNKS);
    for (;;) {
        read_chunk(s);
        // Quiet is auto volume on, nothing playing (zakhar's playback or a music stream), no session
        // open, no wake proposed and no capture held; its end ends a measurement.
        const bool measuring = ambient_sched_want(&s->amb, atomic_load_explicit(&s_auto_volume, memory_order_relaxed) &&
                                                               !zl_playing() && !music_active() && !zl_mic_busy());
        const mic_chunk_t c = follow_stage(s, measuring);
        pcm_split(s->raw, CHUNK, s->both, s->wake, WAKE_GAIN);
        if (c.raw && measuring && ambient_sched_take(&s->amb, true, s->raw, CHUNK)) {
            zl_report_ambient(ambient_dbfs(&s->amb.level));
        }
        tap_write(&s_tap, s->both, CHUNK);
        wake_word_feed(s->wake, CHUNK);
    }
}

// Per block: zakhar's playback (mono) on both channels plus the music stream (stereo, music.h) at
// its gain, a tenth of it while a session is open or zakhar plays. The LEDs follow zakhar's
// playback alone. The block is mixed in place, as internal RAM has no room for more buffers: the
// music comes into `stereo`, which then carries the mix.
static void speaker_task(void *arg)
{
    (void)arg;
    static int16_t mono[SPK_BLOCK];
    static int16_t stereo[SPK_BLOCK * 2];
    static mix_t mix;
    mix_init(&mix);
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
        // The DAC plays the louder of the dial and the dial plus the boost; each stream is brought
        // down to its own level here.
        gain_block(mono, n, volume_voice_gain());
        const size_t m = music_read(stereo, SPK_BLOCK);
        // Music Assistant's volume times what keeps the music at the dial.
        const uint32_t gain = (uint32_t)(((uint64_t)music_gain() * volume_music_gain()) >> 16);
        mix_block(&mix, mono, stereo, m, gain, zl_mic_busy() || zl_playing(), SPK_BLOCK, stereo);
        size_t written;
        i2s_channel_write(s_tx, stereo, sizeof stereo, &written, portMAX_DELAY);
        if (m > 0) music_played(m, esp_timer_get_time() + (int64_t)SPK_DMA_BUFS * SPK_BLOCK * 1000000 / SPK_RATE);
    }
}

esp_err_t audio_start(void)
{
    // auto_clear: an empty DMA buffer goes out as silence.
    i2s_chan_config_t tx_cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_SLAVE);
    tx_cc.dma_desc_num = SPK_DMA_BUFS;
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

    // The stock DMA: 4 x 256 frames. A filled buffer the mic task is too late for is dropped by the
    // driver.
    i2s_chan_config_t rx_cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_1, I2S_ROLE_SLAVE);
    rx_cc.dma_desc_num = 4;
    rx_cc.dma_frame_num = MIC_DMA_FRAMES;
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
    if (xTaskCreatePinnedToCore(mic_task, "mic", 12 * 1024, NULL, 5, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "mic task create failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

void audio_set_auto_volume(bool on)
{
    atomic_store_explicit(&s_auto_volume, on, memory_order_relaxed);
}

esp_err_t audio_start_playback(void)
{
    // Above the mic task: a late write would starve the DAC and the XMOS's echo reference.
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
