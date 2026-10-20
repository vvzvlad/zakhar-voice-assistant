#include "audio_core.h"

#include <math.h>
#include <string.h>

#include "music_ring.h"

#define Q25_MAX ((1 << 25) - 1)
#define Q25_MIN (-(1 << 25))

// The lag estimator's clear peak: at least this correlation, and this much above every lag more
// than LAG_PEAK_AWAY bins (100 ms) from it, so a broad or doubled maximum keeps the lag in use.
#define LAG_CORR_MIN 0.7f
#define LAG_CORR_MARGIN 0.2f
#define LAG_PEAK_AWAY 25

// The first half of the decimator's symmetric Q15 taps; they sum to 32768 in all.
static const int16_t DEC3_H[DEC3_TAPS / 2] = {
    21, 5, -18, -31, -20, 14, 48, 50, 5, -65, -100, -53,
    62, 160, 142, -14, -209, -272, -105, 207, 424, 318, -107, -558,
    -636, -158, 613, 1072, 697, -478, -1700, -1858, -191, 3102, 6792, 9225,
};

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

// Frame p is in the readable window: 1 <= written - p <= cap - margin.
static bool tap_holds(const tap_t *t, uint32_t written, uint32_t p)
{
    return written - p - 1u < t->cap - t->margin;
}

void tap_read_at(tap_t *t, uint32_t pos, int16_t *out, size_t n)
{
    const uint32_t written = atomic_load_explicit(&t->written, memory_order_acquire);
    const size_t ch = (size_t)t->channels;
    for (size_t i = 0; i < n; i++) {
        const uint32_t p = pos + (uint32_t)i;
        if (tap_holds(t, written, p)) {
            memcpy(&out[i * ch], &t->buf[(p % t->cap) * ch], ch * sizeof(int16_t));
        } else {
            memset(&out[i * ch], 0, ch * sizeof(int16_t));
        }
    }
}

void dec3_init(dec3_t *d)
{
    memset(d, 0, sizeof *d);
}

size_t dec3_run(dec3_t *d, const int16_t *in, size_t n, int16_t *out)
{
    size_t m = 0;
    for (size_t i = 0; i < n; i++) {
        d->win[d->pos] = d->win[d->pos + DEC3_TAPS] = in[i];
        d->pos = d->pos + 1 == DEC3_TAPS ? 0 : d->pos + 1;
        if (++d->phase < 3) continue;
        d->phase = 0;
        const int16_t *x = &d->win[d->pos];  // the newest DEC3_TAPS inputs, oldest first
        int32_t acc = 1 << 14;               // rounds the Q15 sum
        for (int k = 0; k < DEC3_TAPS / 2; k++) acc += DEC3_H[k] * (x[k] + x[DEC3_TAPS - 1 - k]);
        out[m++] = sat16(acc >> 15);
    }
    return m;
}

bool aec_skip(tap_t *ref, uint32_t aligned, size_t chunk, size_t tail)
{
    const uint32_t written = atomic_load_explicit(&ref->written, memory_order_acquire);
    const size_t ch = (size_t)ref->channels;
    const uint32_t from = aligned - (uint32_t)tail;
    for (size_t i = 0; i < tail + chunk; i++) {
        const uint32_t p = from + (uint32_t)i;
        if (!tap_holds(ref, written, p)) continue;
        const int16_t *s = &ref->buf[(p % ref->cap) * ch];
        for (size_t c = 0; c < ch; c++) {
            if (s[c] != 0) return false;
        }
    }
    return true;
}

// log(1 + mean square) of LAG_BIN samples, every stride-th of x.
static float bin_env(const int16_t *x, size_t stride)
{
    int64_t sum = 0;
    for (size_t i = 0; i < LAG_BIN; i++) {
        const int32_t v = x[i * stride];
        sum += v * v;
    }
    return logf(1.0f + (float)sum / LAG_BIN);
}

void lag_track_init(lag_track_t *t)
{
    memset(t, 0, sizeof *t);
}

void lag_track_push(lag_track_t *t, const int16_t *x, size_t n, size_t stride)
{
    for (size_t i = 0; i + LAG_BIN <= n; i += LAG_BIN) t->env[t->bins++ % LAG_HIST] = bin_env(&x[i * stride], stride);
}

int lag_track_run(lag_track_t *t, tap_t *ref, uint32_t mic_end, uint32_t offset, float *corr)
{
    // ref_env[j] is the reference paired at lag 0 with microphone bin first - LAG_MAX + j, where
    // `first` is the window's first bin; microphone bin b starts at mic_end - (bins - b) * LAG_BIN.
    const uint32_t first = t->bins - LAG_DELAY - LAG_WINDOW;
    const uint32_t start = mic_end - (uint32_t)((LAG_DELAY + LAG_WINDOW + LAG_MAX) * LAG_BIN) + offset;
    int16_t buf[LAG_BIN];
    for (size_t j = 0; j < LAG_WINDOW + LAG_MAX; j++) {
        tap_read_at(ref, start + (uint32_t)(j * LAG_BIN), buf, LAG_BIN);
        t->ref_env[j] = bin_env(buf, 1);
    }
    float mean = 0;
    for (size_t i = 0; i < LAG_WINDOW; i++) mean += t->env[(first + i) % LAG_HIST];
    mean /= LAG_WINDOW;
    float sxx = 0;
    for (size_t i = 0; i < LAG_WINDOW; i++) {
        const float d = t->env[(first + i) % LAG_HIST] - mean;
        sxx += d * d;
    }
    // Lag L pairs microphone bin first + i with ref_env[LAG_MAX - L + i].
    int best = 0;
    for (int lag = 0; lag <= LAG_MAX; lag++) {
        const float *y = &t->ref_env[LAG_MAX - lag];
        float ym = 0;
        for (size_t i = 0; i < LAG_WINDOW; i++) ym += y[i];
        ym /= LAG_WINDOW;
        float sxy = 0, syy = 0;
        for (size_t i = 0; i < LAG_WINDOW; i++) {
            const float dx = t->env[(first + i) % LAG_HIST] - mean, dy = y[i] - ym;
            sxy += dx * dy;
            syy += dy * dy;
        }
        corr[lag] = sxx > 0 && syy > 0 ? sxy / sqrtf(sxx * syy) : 0;
        if (corr[lag] > corr[best]) best = lag;
    }
    float away = -1;
    for (int lag = 0; lag <= LAG_MAX; lag++) {
        if ((lag < best - LAG_PEAK_AWAY || lag > best + LAG_PEAK_AWAY) && corr[lag] > away) away = corr[lag];
    }
    return corr[best] >= LAG_CORR_MIN && corr[best] - away >= LAG_CORR_MARGIN ? best * LAG_BIN : -1;
}

void mic_chunks_init(mic_chunks_t *c, uint32_t frames, uint32_t ref_settle)
{
    memset(c, 0, sizeof *c);
    c->frames = frames;
    c->ref_settle = ref_settle;
    c->span = ((LAG_DELAY + LAG_WINDOW) * LAG_BIN + frames - 1) / frames;
}

mic_chunk_t mic_chunks_step(mic_chunks_t *c, uint32_t dropped, bool want_raw, bool wrote, uint32_t ref_written)
{
    c->pos += dropped;
    if (dropped > 0) c->raw_run = 0;
    // The write the previous step asked for went out right after that chunk was read, so this chunk
    // is the first after it.
    if (c->asked && wrote) {
        c->none_written = c->asked_raw;
        c->settle = MIC_STAGE_SETTLE;
    }
    if (c->settle > 0 && --c->settle == 0) c->raw = c->none_written;
    c->asked = want_raw != c->none_written;
    c->asked_raw = want_raw;
    if (!c->have_offset && ref_written >= c->ref_settle) {
        c->offset = ref_written - (c->pos + c->frames);
        c->have_offset = true;
    }
    c->raw_run = c->raw ? c->raw_run + 1 : 0;
    const mic_chunk_t out = {
        .pos = c->pos,
        .raw = c->raw,
        .settling = c->settle > 0,
        .write = c->asked,
        .write_raw = want_raw,
        .have_offset = c->have_offset,
        .offset = c->offset,
        .lag_ok = c->raw && c->raw_run >= c->span,
    };
    c->pos += c->frames;
    return out;
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
               size_t frames, int16_t *out, int16_t *mono)
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
        mono[i] = (int16_t)((out[2 * i] + out[2 * i + 1]) / 2);
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

uint32_t volume_music_scale(int base, int applied)
{
    // 0.5 dB per register step.
    const int steps = (int8_t)volume_dac_byte(applied) - (int8_t)volume_dac_byte(base);
    if (steps <= 0) return 65536u;
    return (uint32_t)lroundf(65536.0f * powf(10.0f, -(float)steps / 40.0f));
}

bool volume_dac_muted(int percent)
{
    return percent < 1;
}
