#include "audio_core.h"

#include <math.h>
#include <string.h>

#include "music_ring.h"

#define Q25_MAX ((1 << 25) - 1)
#define Q25_MIN (-(1 << 25))

static int16_t sat16(int32_t v)
{
    return v > INT16_MAX ? INT16_MAX : v < INT16_MIN ? INT16_MIN : (int16_t)v;
}

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

void mic_chunks_init(mic_chunks_t *c)
{
    memset(c, 0, sizeof *c);
}

mic_chunk_t mic_chunks_step(mic_chunks_t *c, bool want_raw, bool wrote)
{
    // The write the previous step asked for went out right after that chunk was read, so this chunk
    // is the first after it.
    if (c->asked && wrote) {
        c->none_written = c->asked_raw;
        c->settle = MIC_STAGE_SETTLE;
    }
    if (c->settle > 0 && --c->settle == 0) c->raw = c->none_written;
    c->asked = want_raw != c->none_written;
    c->asked_raw = want_raw;
    return (mic_chunk_t){.raw = c->raw, .write = c->asked, .write_raw = want_raw};
}

void ambient_init(ambient_t *a)
{
    memset(a, 0, sizeof *a);
}

void ambient_add(ambient_t *a, const int32_t *in, size_t frames)
{
    for (size_t i = 0; i < frames; i++) {
        const double v = in[2 * i];
        a->squares += v * v;
    }
    a->n += frames;
}

double ambient_dbfs(const ambient_t *a)
{
    const double rms = a->n > 0 ? sqrt(a->squares / (double)a->n) : 0;
    return round(200 * log10(rms / 2147483648.0)) / 10;
}

void ambient_sched_init(ambient_sched_t *s, unsigned every, unsigned chunks)
{
    memset(s, 0, sizeof *s);
    s->every = every;
    s->chunks = chunks;
}

bool ambient_sched_want(ambient_sched_t *s, bool quiet)
{
    if (!quiet) {
        s->measuring = false;
        s->quiet = 0;
    } else if (++s->quiet == s->every) {
        s->quiet = 0;
        s->measuring = true;
        s->taken = 0;
        ambient_init(&s->level);
    }
    return s->measuring;
}

bool ambient_sched_take(ambient_sched_t *s, bool raw, const int32_t *in, size_t frames)
{
    if (!s->measuring || !raw) return false;
    ambient_add(&s->level, in, frames);
    if (++s->taken < s->chunks) return false;
    s->measuring = false;
    return true;
}

void mix_init(mix_t *m)
{
    m->gain = 0;
}

void mix_block(mix_t *m, const int16_t *voice, const int16_t *music, size_t music_frames, uint32_t gain, bool duck,
               size_t frames, int16_t *out)
{
    const int64_t from = m->gain;
    const int64_t to = duck ? gain / MIX_DUCK_DIV : gain;
    for (size_t i = 0; i < frames; i++) {
        int32_t l = voice[i], r = voice[i];
        if (i < music_frames) {
            const int64_t g = from + (to - from) * (int64_t)(i + 1) / (int64_t)frames;
            l += (int32_t)((music[2 * i] * g + (1 << 15)) >> 16);
            r += (int32_t)((music[2 * i + 1] * g + (1 << 15)) >> 16);
        }
        out[2 * i] = sat16(l);
        out[2 * i + 1] = sat16(r);
    }
    m->gain = (uint32_t)to;
}

#define RING_FRAME_BYTES (2 * sizeof(int16_t))

size_t music_ring_write(music_ring_t *r, const void *in, size_t n)
{
    const uint8_t *src = in;
    const size_t space = (r->tail + r->cap - r->head - 1) % r->cap;
    if (n > space) n = space;
    const size_t first = n < r->cap - r->head ? n : r->cap - r->head;
    memcpy(&r->buf[r->head * 2], src, first * RING_FRAME_BYTES);
    memcpy(r->buf, src + first * RING_FRAME_BYTES, (n - first) * RING_FRAME_BYTES);
    r->head = (uint32_t)((r->head + n) % r->cap);
    return n;
}

size_t music_ring_read(music_ring_t *r, int16_t *out, size_t max)
{
    const size_t held = (r->head + r->cap - r->tail) % r->cap;
    const size_t n = max < held ? max : held;
    const size_t first = n < r->cap - r->tail ? n : r->cap - r->tail;
    memcpy(out, &r->buf[r->tail * 2], first * RING_FRAME_BYTES);
    memcpy(out + first * 2, r->buf, (n - first) * RING_FRAME_BYTES);
    r->tail = (uint32_t)((r->tail + n) % r->cap);
    return n;
}

void music_ring_flush(music_ring_t *r, uint32_t to)
{
    if ((to + r->cap - r->tail) % r->cap <= (r->head + r->cap - r->tail) % r->cap) r->tail = to;
}

uint8_t volume_dac_byte(int percent)
{
    // -127 + (0.40 + p / 100 * 0.45) * 175 = (7875 p - 570000) / 10000; C division truncates
    // toward zero, as the stock float -> int8 conversion does.
    const int8_t b = (int8_t)((7875 * percent - 570000) / 10000);
    return (uint8_t)b;
}

static uint32_t level_gain(int level, int dac)
{
    // 0.5 dB per register step.
    const int steps = (int8_t)volume_dac_byte(dac) - (int8_t)volume_dac_byte(level);
    if (steps <= 0) return 65536u;
    return (uint32_t)lroundf(65536.0f * powf(10.0f, -(float)steps / 40.0f));
}

volume_levels_t volume_levels(int base, int boost)
{
    const int v = base + boost;
    const int reply = base == 0 ? 0 : v < 1 ? 1 : v > 100 ? 100 : v;
    const int dac = reply > base ? reply : base;
    return (volume_levels_t){.dac = dac, .music = level_gain(base, dac), .voice = level_gain(reply, dac)};
}

void gain_block(int16_t *s, size_t n, uint32_t gain)
{
    if (gain >= 65536u) return;
    for (size_t i = 0; i < n; i++) s[i] = (int16_t)(((int32_t)s[i] * (int64_t)gain + (1 << 15)) >> 16);
}

bool volume_dac_muted(int percent)
{
    return percent < 1;
}
