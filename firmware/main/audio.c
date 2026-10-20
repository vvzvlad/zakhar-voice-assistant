// Both I2S buses in the formats of the stock firmware. Microphones on I2S1: 16 kHz, 32-bit slots,
// stereo, Philips; the left slot is the XMOS channel 0, the right slot channel 1. Speaker on I2S0:
// 48 kHz, 16-bit samples in 32-bit slots, stereo, Philips, running all the time so the XMOS echo
// reference on the same wire never stops.
// The stop word hears the speaker's own playback cancelled out of a raw microphone: an AEC after
// the XMOS's processing takes about 1 dB of the echo off its channels, a raw microphone loud
// enough for it 8-28 dB (docs/echo-cancellation.md). While the speaker plays, the XMOS puts the
// raw microphone 0 (stage NONE) on channel 0, and ESP-SR's AEC runs on it against what went to the
// wire (what the speaker task wrote and the DMA buffers the driver sent as silence when it was
// late), decimated to 16 kHz and delayed by the echo's lag, which is measured on that raw
// channel while playback runs. The raw microphone comes x16 into the AEC (AEC_GAIN); its output
// is channel 0 of the tap and the stop word's input. Channel 1 and the wake word's input never
// change. While the speaker is silent, echo_cancel is 0 or no AEC could be made for the settings,
// channel 0 is the XMOS AGC output and the stop word hears channel 1 as the wake word does.
// With the server's auto_volume on, every 15 s of quiet (nothing playing, no session open, no
// capture held) channel 0 is the raw microphone 0 for about a second: the room's noise is measured
// on its 32-bit slots, before the XMOS's noise suppression and AGC, and goes to the server as
// `ambient`. Playback, a session or a capture ends a measurement at once, and nothing is sent.
#include "audio.h"

#include <math.h>
#include <stdatomic.h>
#include <string.h>

#include "audio_core.h"
#include "board.h"
#include "driver/i2s_std.h"
#include "esp_aec.h"
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
// The raw microphone's gain into the AEC, taken from its 32-bit slot. At volume 79 % the speaker's
// echo on it is about -55 dBFS at gain 1, under the AEC's own output floor of -41..-46 dBFS, and
// the AEC makes channel 0 louder; x16 puts the echo near -31 dBFS there, its peaks about -4 dBFS
// at volume 100 % (docs/echo-cancellation.md).
#define AEC_GAIN 16
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

// The mic task's unit: 512 frames = 32 ms, the AEC's frame in its SR and FD modes at 16 kHz and
// the protocol's microphone frame. It is two whole RX DMA buffers, so no read leaves one half
// read: IDF's i2s_channel_read skips the rest of a half-read buffer when its queue is nearly full,
// a loss that on_mic_overflow would not see.
#define CHUNK 512
#define MIC_DMA_FRAMES (CHUNK / 2)

// The echo canceller's settings come from the server's `config`: echo_cancel 1..3 is the residual
// echo suppression AEC_NLP_LEVEL_NORMAL, AGGR and VERYAGGR (in the full-duplex modes only),
// echo_cancel_mode indexes AEC_MODES and echo_cancel_filter is the filter length. The AEC takes
// channel 0 in and out and one reference.
static const aec_mode_t AEC_MODES[] = {
    AEC_MODE_SR_LOW_COST, AEC_MODE_SR_HIGH_PERF, AEC_MODE_FD_LOW_COST, AEC_MODE_FD_HIGH_PERF,
};
#define FD_MODES 2  // AEC_MODES from this index on are full-duplex

// The reference: the speaker's output at 16 kHz. 65536 frames = 4.1 s, a power of two, past the
// lag estimator's 2 s; readers keep 4096 frames away from the end being overwritten.
#define REF_FRAMES (1u << 16)
#define REF_MARGIN 4096
// Microphone frame m pairs with reference position m + offset - lag. The offset (mic_chunks_t) is
// taken once, at the first chunk after REF_SETTLE reference frames: past the speaker task's first
// writes, which go straight into the empty DMA buffers.
#define REF_SETTLE MIC_RATE
// The lag in use until the first clear estimate: 1344 frames = 84 ms, where the estimator settled
// on a raw XMOS microphone of the Voice PE (docs/echo-cancellation.md).
#define LAG_INIT (21 * LAG_BIN)
#define LAG_EVERY 8  // chunks between estimates while the AEC runs: 256 ms

// Auto volume: a measurement starts every AMBIENT_EVERY chunks of quiet (15 s) and takes
// AMBIENT_CHUNKS raw chunks (0.99 s).
#define AMBIENT_EVERY (15 * MIC_RATE / CHUNK)
#define AMBIENT_CHUNKS (MIC_RATE / CHUNK)

static i2s_chan_handle_t s_tx, s_rx;
static tap_t s_tap;
static tap_t s_ref;
static dec3_t s_dec;
// The server's echo_cancel, echo_cancel_mode and echo_cancel_filter, taken by the mic task per chunk.
// Until the first `config` echo_cancel is 0: no AEC is made and channel 0 stays AGC.
static _Atomic int s_echo_cancel;
static _Atomic int s_echo_mode;
static _Atomic int s_echo_filter;
// The server's auto_volume, taken by the mic task per chunk; off until the first `config`.
static _Atomic bool s_auto_volume;
// Microphone frames the RX driver dropped since the mic task last took them.
static _Atomic uint32_t s_mic_dropped;
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

// The RX driver's queue of filled DMA buffers was full when one more filled: it dropped the
// oldest, event->size bytes of 2 x 32-bit frames.
static bool IRAM_ATTR on_mic_overflow(i2s_chan_handle_t handle, i2s_event_data_t *event, void *user_ctx)
{
    (void)handle;
    (void)user_ctx;
    atomic_fetch_add_explicit(&s_mic_dropped, event->size / (2 * sizeof(int32_t)), memory_order_relaxed);
    return false;
}

static aec_nlp_level_t nlp_level(int echo_cancel)
{
    return (aec_nlp_level_t)(AEC_NLP_LEVEL_NORMAL + echo_cancel - 1);
}

// A new AEC in AEC_MODES[mode] with the filter length `filter`, its suppression at `level`
// (AEC_NLP_LEVEL_NORMAL when off: the mic task sets the level once echo_cancel turns on); NULL,
// logged, when ESP-SR cannot make it or makes it with another frame than CHUNK.
static aec_handle_t *aec_make(int mode, int filter, int level)
{
    aec_config_t ac = {
        .mic_num = 1,
        .ref_num = 1,
        .out_num = 1,
        .filter_length = filter,
        .sample_rate = MIC_RATE,
        .caps = MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT,
        .mode = AEC_MODES[mode],
        .nlp_level = level > 0 ? nlp_level(level) : AEC_NLP_LEVEL_NORMAL,
    };
    aec_handle_t *aec = aec_create_from_config(&ac);
    if (aec != NULL && aec_get_chunksize(aec) != CHUNK) {
        ESP_LOGE(TAG, "echo cancel: mode %d works in %d-frame chunks, not %d", mode, aec_get_chunksize(aec), CHUNK);
        aec_destroy(aec);
        aec = NULL;
    }
    if (aec == NULL) {
        ESP_LOGE(TAG, "echo cancel: no AEC in mode %d, filter %d: channel 0 stays AGC", mode, filter);
        return NULL;
    }
    ESP_LOGI(TAG, "echo cancel: %s, filter %d; free heap %u B internal, %u B PSRAM", aec_get_config_string(aec), filter,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    return aec;
}

static void log_playback(const aec_handle_t *aec, int filter, const aec_stats_t *st, int32_t lag)
{
    ESP_LOGI(TAG, "echo cancel: %.1f s of playback, %s, filter %d, lag %ld frames (%ld ms), correlation %.2f, "
                  "channel 0 down %.1f dB, AEC %lu us mean / %lu us max per %d ms chunk, %lu mic frames dropped",
             st->chunks * (double)CHUNK / MIC_RATE, aec_get_config_string(aec), filter, (long)lag,
             (long)(lag * 1000 / MIC_RATE), st->corr, 10 * log10((double)st->e_in / (double)st->e_out),
             (unsigned long)(st->us / st->chunks), (unsigned long)st->us_max, CHUNK * 1000 / MIC_RATE,
             (unsigned long)st->dropped);
}

// The mic task's state. The buffers hold one chunk: the raw slots, the tap's interleaved frames,
// the wake-word and stop-word inputs, and the AEC's input (channel 0), output and reference.
typedef struct {
    int32_t raw[CHUNK * 2];
    int16_t both[CHUNK * 2], wake[CHUNK], stop[CHUNK];
    _Alignas(16) int16_t in[CHUNK], out[CHUNK], ref[CHUNK];
    lag_track_t lag;
    float corr[LAG_MAX + 1];
    mic_chunks_t chunks;
    ambient_sched_t amb;
    aec_handle_t *aec;
    int made_mode, made_filter;  // the settings aec was made with, or failed to be
    int applied;                 // the level aec suppresses at
    int32_t lag_in_use;
    bool lag_set;  // the lag in use is decided, until a reboot (lag_next)
    unsigned since_estimate;
    bool wrote;  // the stage write the previous chunk asked for went through
    bool stack_logged;
    aec_stats_t st;
    // The chunk in hand: the frames dropped before it, the server's echo_cancel, echo_cancel_mode
    // and echo_cancel_filter, and whether anything plays.
    uint32_t dropped;
    int level, mode, filter;
    bool playing;
} mic_state_t;

static mic_state_t s_mic;

// Reads until the chunk is whole, then takes the frames the driver dropped before it.
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
    // Frames dropped before this chunk count toward its position, so the reference stays aligned.
    s->dropped = atomic_exchange_explicit(&s_mic_dropped, 0, memory_order_relaxed);
}

// Logs the playback the AEC ran in and starts its statistics and the lag estimate's count anew.
static void log_stats(mic_state_t *s)
{
    log_playback(s->aec, s->made_filter, &s->st, s->lag_in_use);
    s->st = (aec_stats_t){.corr = NAN};
    s->since_estimate = 0;
}

// Takes the server's echo settings and the playback for the chunk, and makes the AEC, or makes it
// anew, when they ask for it.
static void follow_echo_settings(mic_state_t *s)
{
    s->level = atomic_load_explicit(&s_echo_cancel, memory_order_relaxed);
    s->mode = atomic_load_explicit(&s_echo_mode, memory_order_relaxed);
    s->filter = atomic_load_explicit(&s_echo_filter, memory_order_relaxed);
    // Playing is zakhar's playback or a music stream.
    s->playing = zl_playing() || music_active();
    // The AEC is made at the first playback with echo_cancel on, not at boot: its internal RAM
    // then comes after the link task's stack, which at boot found no block left. Once made, a
    // new echo_cancel_mode or echo_cancel_filter re-creates it at once.
    if (!aec_remake(s->made_mode, s->made_filter, s->mode, s->filter, s->level, s->playing)) return;
    if (s->st.chunks > 0) log_stats(s);
    if (s->aec != NULL) aec_destroy(s->aec);
    s->aec = aec_make(s->mode, s->filter, s->level);
    s->made_mode = s->mode;
    s->made_filter = s->filter;
    s->applied = s->level;
}

// Channel 0's stage for the chunk: raw while the AEC is to run or a measurement wants it, else AGC.
static mic_chunk_t follow_stage(mic_state_t *s, bool measuring)
{
    const bool want_raw = (s->level > 0 && s->aec != NULL && s->playing) || measuring;
    const mic_chunk_t c = mic_chunks_step(&s->chunks, s->dropped, want_raw, s->wrote, tap_position(&s_ref));
    // Written as soon as the chunk is in, so the chunk's processing does not delay the switch.
    s->wrote = c.write && xmos_ch0_raw(c.write_raw) == ESP_OK;
    return c;
}

// A raw chunk of an auto volume measurement.
static void measure_noise(mic_state_t *s)
{
    // No AEC: channel 0 goes to the tap as read, at gain 1. The estimator still gets it, at
    // AEC_GAIN as the AEC branch gives it: its window is the raw chunks in a row
    // (mic_chunks_step), which a playback that takes the raw channel over from here continues.
    for (size_t i = 0; i < CHUNK; i++) s->in[i] = pcm_s32_to_s16(s->raw[2 * i], AEC_GAIN);
    if (ambient_sched_take(&s->amb, true, s->raw, CHUNK)) zl_report_ambient(ambient_dbfs(&s->amb.level));
}

// A raw chunk outside a measurement: channel 0 through the AEC or past it, into the tap and the
// stop word's input. True when the AEC ran on it.
static bool cancel_echo(mic_state_t *s, const mic_chunk_t *c)
{
    // While an AGC write settles, the chunk still counted raw may carry the AGC output,
    // which AEC_GAIN would clip and the AEC would adapt to: it passes the AEC by, at gain 1.
    const int gain = c->settling ? 1 : AEC_GAIN;
    for (size_t i = 0; i < CHUNK; i++) s->in[i] = pcm_s32_to_s16(s->raw[2 * i], gain);
    const uint32_t aligned = c->pos + c->offset - (uint32_t)s->lag_in_use;
    const bool cancel = !c->settling && s->aec != NULL && c->have_offset &&
                        !aec_skip(&s_ref, aligned, CHUNK, (size_t)s->filter * CHUNK);
    if (cancel) {
        if (s->level > 0 && s->level != s->applied) {
            if (s->mode >= FD_MODES) aec_set_nlp_level(s->aec, nlp_level(s->level));
            s->applied = s->level;
        }
        tap_read_at(&s_ref, aligned, s->ref, CHUNK);
        const int64_t t0 = esp_timer_get_time();
        aec_process(s->aec, s->in, s->ref, s->out);
        aec_stats_add(&s->st, s->in, s->out, CHUNK, (uint32_t)(esp_timer_get_time() - t0));
    } else {
        memcpy(s->out, s->in, sizeof s->out);
    }
    for (size_t i = 0; i < CHUNK; i++) {
        s->both[2 * i] = s->out[i];
        s->stop[i] = s->out[i];
    }
    s->st.dropped += s->dropped;
    return cancel;
}

// A raw chunk's channel 0 (s->in) into the lag estimator's window; while the AEC runs, an estimate
// every LAG_EVERY chunks once the window holds raw chunks only.
static void estimate_lag(mic_state_t *s, const mic_chunk_t *c, bool cancel)
{
    lag_track_push(&s->lag, s->in, CHUNK, 1);
    if (cancel && c->lag_ok && ++s->since_estimate == LAG_EVERY) {
        s->since_estimate = 0;
        const int est = lag_track_run(&s->lag, &s_ref, c->pos + CHUNK, c->offset, s->corr);
        s->lag_in_use = lag_next(s->lag_in_use, est, &s->lag_set);
        s->st.corr = s->corr[(s->lag_in_use + LAG_GUARD) / LAG_BIN];
    }
}

// An AGC chunk: a playback the AEC ran in is logged, the first one with the task's stack.
static void end_playback(mic_state_t *s)
{
    if (s->st.chunks > 0) {
        log_stats(s);
        if (!s->stack_logged) {
            ESP_LOGI(TAG, "mic task: %u B of its stack never used", (unsigned)uxTaskGetStackHighWaterMark(NULL));
            s->stack_logged = true;
        }
    }
}

// Per chunk: channel 0's stage follows the playback (raw while it plays, with echo_cancel on and
// an AEC made); while channel 0 is raw, the AEC runs on it while the reference it could be using
// is not all silence, else it passes through; then the tap, the wake-word engine (channel 1 at
// WAKE_GAIN) and the stop word (the AEC's output while channel 0 is raw). While the AEC runs, the
// lag is estimated every LAG_EVERY chunks once the raw chunks in a row span the estimator's
// window (mic_chunks_step). A new echo_cancel_mode or echo_cancel_filter re-creates the AEC
// between chunks. An auto volume measurement asks for the raw channel 0 through the same
// accounting and takes only the chunks it reports raw.
static void mic_task(void *arg)
{
    (void)arg;
    mic_state_t *s = &s_mic;
    s->made_mode = s->made_filter = -1;
    s->lag_in_use = LAG_INIT;
    s->st = (aec_stats_t){.corr = NAN};
    lag_track_init(&s->lag);
    mic_chunks_init(&s->chunks, CHUNK, REF_SETTLE);
    ambient_sched_init(&s->amb, AMBIENT_EVERY, AMBIENT_CHUNKS);
    for (;;) {
        read_chunk(s);
        follow_echo_settings(s);
        // Quiet is auto volume on, nothing playing, no session open, no wake proposed and no capture
        // held; its end ends a measurement.
        const bool measuring = ambient_sched_want(
            &s->amb, atomic_load_explicit(&s_auto_volume, memory_order_relaxed) && !s->playing && !zl_mic_busy());
        const mic_chunk_t c = follow_stage(s, measuring);
        pcm_split(s->raw, CHUNK, s->both, s->wake, WAKE_GAIN);
        const int16_t *stop_in = NULL;
        if (c.raw && measuring) {
            measure_noise(s);
            estimate_lag(s, &c, false);
        } else if (c.raw) {
            estimate_lag(s, &c, cancel_echo(s, &c));
            stop_in = s->stop;
        } else {
            end_playback(s);
        }
        tap_write(&s_tap, s->both, CHUNK);
        wake_word_feed(s->wake, stop_in, CHUNK);
    }
}

// Per block: zakhar's playback (mono) on both channels plus the music stream (stereo, music.h) at
// its gain, a tenth of it while a session is open or zakhar plays; the echo reference is the mono
// downmix of the block as it goes to the wire. The LEDs follow zakhar's playback alone. The block
// is mixed in place, as internal RAM has no room for more buffers: the music comes into `stereo`
// and the downmix replaces `mono`.
static void speaker_task(void *arg)
{
    (void)arg;
    static int16_t mono[SPK_BLOCK];
    static int16_t stereo[SPK_BLOCK * 2];
    static int16_t ref[SPK_BLOCK / 3];
    static const int16_t silence[SPK_BLOCK / 3];
    static mix_t mix;
    mix_init(&mix);
    // The buffers that went out before this task started are not late.
    uint32_t seen = atomic_load_explicit(&s_late, memory_order_relaxed);
    uint32_t ref_late = seen;
    for (unsigned blocks = 1;; blocks++) {
        // A buffer that went out as silence is on the wire but was never written here: its
        // silence goes into the echo reference too, or the reference falls behind the wire. It
        // lands after the block that was waiting for the buffer, which went out after it: one
        // block out of place, the count right.
        const uint32_t late_now = atomic_load_explicit(&s_late, memory_order_relaxed);
        for (; ref_late != late_now; ref_late++) tap_write(&s_ref, silence, SPK_BLOCK / 3);
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
        mix_block(&mix, mono, stereo, m, gain, zl_mic_busy() || zl_playing(), SPK_BLOCK, stereo, mono);
        // The echo reference: this block, silence included, as it goes to the wire.
        tap_write(&s_ref, ref, dec3_run(&s_dec, mono, SPK_BLOCK, ref));
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
    // driver and counted by on_mic_overflow.
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
    const i2s_event_callbacks_t rx_cb = {.on_recv_q_ovf = on_mic_overflow};
    err = i2s_new_channel(&rx_cc, NULL, &s_rx);
    if (err == ESP_OK) err = i2s_channel_init_std_mode(s_rx, &rx_std);
    if (err == ESP_OK) err = i2s_channel_register_event_callback(s_rx, &rx_cb, NULL);
    if (err == ESP_OK) err = i2s_channel_enable(s_rx);
    if (err != ESP_OK) return err;

    int16_t *tap = heap_caps_calloc(TAP_FRAMES * 2, sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (tap == NULL) {
        ESP_LOGE(TAG, "microphone tap: out of PSRAM");
        return ESP_ERR_NO_MEM;
    }
    tap_init(&s_tap, tap, TAP_FRAMES, TAP_MARGIN, 2);
    int16_t *ref = heap_caps_calloc(REF_FRAMES, sizeof(int16_t), MALLOC_CAP_SPIRAM);
    if (ref == NULL) {
        ESP_LOGE(TAG, "echo reference: out of PSRAM");
        return ESP_ERR_NO_MEM;
    }
    tap_init(&s_ref, ref, REF_FRAMES, REF_MARGIN, 1);
    dec3_init(&s_dec);
    // Core 1, away from Wi-Fi and zakhar_link on core 0. The stack holds the TFLite inference and
    // the one-channel AEC.
    if (xTaskCreatePinnedToCore(mic_task, "mic", 12 * 1024, NULL, 5, NULL, 1) != pdPASS) {
        ESP_LOGE(TAG, "mic task create failed");
        return ESP_ERR_NO_MEM;
    }
    return ESP_OK;
}

void audio_set_echo_cancel(int level, int mode, int filter)
{
    atomic_store_explicit(&s_echo_cancel, level, memory_order_relaxed);
    atomic_store_explicit(&s_echo_mode, mode, memory_order_relaxed);
    atomic_store_explicit(&s_echo_filter, filter, memory_order_relaxed);
}

void audio_set_auto_volume(bool on)
{
    atomic_store_explicit(&s_auto_volume, on, memory_order_relaxed);
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
