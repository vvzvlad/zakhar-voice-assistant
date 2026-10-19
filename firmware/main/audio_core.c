#include "audio_core.h"

#include <string.h>

#define Q25_MAX ((1 << 25) - 1)
#define Q25_MIN (-(1 << 25))

int16_t pcm_s32_to_s16(int32_t sample, int gain)
{
    int32_t q = (sample >> 6) * gain;  // Q31 -> Q25, then the gain
    if (q > Q25_MAX) q = Q25_MAX;
    if (q < Q25_MIN) q = Q25_MIN;
    return (int16_t)((q * 64) >> 16);  // Q25 -> Q31, the top 16 bits
}

void pcm_split(const int32_t *in, size_t frames, int16_t *both, int16_t *ch1, int ch1_gain)
{
    for (size_t i = 0; i < frames; i++) {
        both[2 * i] = pcm_s32_to_s16(in[2 * i], 1);
        ch1[i] = pcm_s32_to_s16(in[2 * i + 1], ch1_gain);
        // The stream carries channel 1 exactly as the wake-word engine gets it: the gain applied
        // before the cut to 16 bits keeps low bits that channel 1 at gain 1 would drop.
        both[2 * i + 1] = ch1[i];
    }
}

void tap_init(tap_t *t, int16_t *buf, uint32_t cap, uint32_t margin, int channels)
{
    t->buf = buf;
    t->cap = cap;
    t->margin = margin;
    t->channels = channels;
    atomic_init(&t->written, 0);
}

void tap_write(tap_t *t, const int16_t *frames, size_t n)
{
    const uint32_t written = atomic_load_explicit(&t->written, memory_order_relaxed);
    const size_t ch = (size_t)t->channels;
    for (size_t i = 0; i < n; i++) {
        memcpy(&t->buf[((written + i) % t->cap) * ch], &frames[i * ch], ch * sizeof(int16_t));
    }
    atomic_store_explicit(&t->written, written + (uint32_t)n, memory_order_release);
}

uint32_t tap_position(tap_t *t)
{
    return atomic_load_explicit(&t->written, memory_order_acquire);
}

size_t tap_read(tap_t *t, uint32_t *pos, int16_t *out, size_t max)
{
    const uint32_t written = atomic_load_explicit(&t->written, memory_order_acquire);
    const uint32_t window = t->cap - t->margin;
    if (written - *pos > window) *pos = written - window;
    const size_t avail = written - *pos;
    const size_t n = avail < max ? avail : max;
    const size_t ch = (size_t)t->channels;
    for (size_t i = 0; i < n; i++) {
        memcpy(&out[i * ch], &t->buf[((*pos + i) % t->cap) * ch], ch * sizeof(int16_t));
    }
    *pos += (uint32_t)n;
    return n;
}

uint8_t volume_dac_byte(int percent)
{
    // -127 + (0.40 + p / 100 * 0.45) * 175 = (7875 p - 570000) / 10000; C division truncates
    // toward zero, as the stock float -> int8 conversion does.
    const int8_t b = (int8_t)((7875 * percent - 570000) / 10000);
    return (uint8_t)b;
}

bool volume_dac_muted(int percent)
{
    return percent < 1;
}
