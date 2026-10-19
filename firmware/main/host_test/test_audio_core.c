// Host test of the Voice PE firmware's pure audio core (audio_core.c): the microphone sample
// conversion against ESPHome's MicrophoneSource, the tap ring, the echo reference and delay
// estimator, the mic task's chunk bookkeeping, and the stock volume -> DAC byte table.
// `make fw-voicepe-host-test` builds it.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio_core.h"

static void test_samples(void)
{
    // Gain 1: the top 16 bits, the low ones dropped (truncation, also for negatives).
    assert(pcm_s32_to_s16(0x12345678, 1) == 0x1234);
    assert(pcm_s32_to_s16(INT32_MAX, 1) == INT16_MAX);
    assert(pcm_s32_to_s16(INT32_MIN, 1) == INT16_MIN);
    assert(pcm_s32_to_s16(-1, 1) == -1);
    assert(pcm_s32_to_s16(-0x10000, 1) == -1);
    // Gain 4: four times the value, keeping two more bits of the slot.
    assert(pcm_s32_to_s16(0x01000000, 4) == 0x0400);
    assert(pcm_s32_to_s16(0x00004000, 4) == 1);  // below 16 bits at gain 1, one step at gain 4
    assert(pcm_s32_to_s16(0x00004000, 1) == 0);
    assert(pcm_s32_to_s16(-0x01000000, 4) == -0x0400);
    // Clamped at full scale instead of wrapping.
    assert(pcm_s32_to_s16(0x40000000, 4) == INT16_MAX);
    assert(pcm_s32_to_s16(-0x40000000, 4) == INT16_MIN);

    const int32_t in[] = {0x11110000, 0x00010000, -0x22220000, 0x10000000};
    int16_t both[4], ch1[2];
    pcm_split(in, 2, both, ch1, 4);
    assert(both[0] == 0x1111 && both[1] == 4 && both[2] == -0x2222 && both[3] == 0x4000);
    assert(ch1[0] == 4 && ch1[1] == 0x4000);
    // Below 16 bits at gain 1: the streamed channel 1 still carries it, as the wake word gets it.
    const int32_t quiet[] = {0, 0x00004000};
    pcm_split(quiet, 1, both, ch1, 4);
    assert(both[1] == 1 && ch1[0] == 1);
}

// Frame f of the test stream: channel 0 = f, channel 1 = -f (low 15 bits).
static void frames_from(uint32_t f, int16_t *out, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        out[2 * i] = (int16_t)((f + i) & 0x7FFF);
        out[2 * i + 1] = (int16_t)(-(int32_t)((f + i) & 0x7FFF));
    }
}

static void read_is(tap_t *t, uint32_t *pos, size_t max, uint32_t want_from, size_t want_n)
{
    int16_t got[2 * 64], want[2 * 64];
    assert(max <= 64);
    const size_t n = tap_read(t, pos, got, max);
    assert(n == want_n);
    frames_from(want_from, want, n);
    assert(memcmp(got, want, n * 2 * sizeof(int16_t)) == 0);
}

static void test_tap(void)
{
    int16_t buf[16 * 2];
    int16_t chunk[2 * 64];
    tap_t t;
    tap_init(&t, buf, 16, 4, 2);  // 16 frames, 12 readable
    assert(tap_position(&t) == 0);

    uint32_t pos = 0;
    read_is(&t, &pos, 8, 0, 0);  // nothing written yet
    frames_from(0, chunk, 5);
    tap_write(&t, chunk, 5);
    assert(tap_position(&t) == 5);
    read_is(&t, &pos, 3, 0, 3);
    assert(pos == 3);
    read_is(&t, &pos, 8, 3, 2);  // only what is written
    assert(pos == 5);

    // The writer laps the ring: an old position moves to the oldest readable frame.
    frames_from(5, chunk, 30);
    tap_write(&t, chunk, 30);
    assert(tap_position(&t) == 35);
    pos = 0;
    read_is(&t, &pos, 64, 35 - 12, 12);
    assert(pos == 35);

    // The uint32 counter wraps: positions stay continuous across it.
    const uint32_t start = UINT32_MAX - 6;
    atomic_store(&t.written, start);
    frames_from(start, chunk, 10);
    tap_write(&t, chunk, 10);
    assert(tap_position(&t) == start + 10 && tap_position(&t) < 10);
    pos = start + 1;
    read_is(&t, &pos, 64, start + 1, 9);
    assert(pos == (uint32_t)(start + 10));

    // A positioned read across the ring's end; frames not written yet or overwritten read as 0.
    int16_t got[2 * 8], want[2 * 8];
    tap_read_at(&t, start + 6, got, 4);  // frames start+6..start+9, ring slots 15, 0, 1, 2
    frames_from(start + 6, want, 4);
    assert(memcmp(got, want, sizeof(int16_t) * 2 * 4) == 0);
    tap_read_at(&t, start + 8, got, 4);  // the last two written, then two not yet
    frames_from(start + 8, want, 2);
    memset(&want[4], 0, sizeof(int16_t) * 4);
    assert(memcmp(got, want, sizeof(int16_t) * 2 * 4) == 0);
    tap_read_at(&t, start - 3, got, 1);  // 13 frames back from the newest: past the readable 12
    assert(got[0] == 0 && got[1] == 0);
}

static void test_decimator(void)
{
    dec3_t d;
    int16_t in[720], out[240], parts[240];
    // DC passes exactly once the 72 taps are filled.
    dec3_init(&d);
    for (size_t i = 0; i < 720; i++) in[i] = 10000;
    assert(dec3_run(&d, in, 720, out) == 240);
    for (size_t i = DEC3_TAPS / 3; i < 240; i++) assert(out[i] == 10000);
    // 1 kHz passes at full amplitude; 10 kHz, which would alias to 6 kHz, is gone.
    const double hz[] = {1000, 10000}, lo[] = {9900, 0}, hi[] = {10100, 10};
    for (int f = 0; f < 2; f++) {
        for (size_t i = 0; i < 720; i++) in[i] = (int16_t)lrint(10000 * sin(2 * M_PI * hz[f] * i / 48000));
        dec3_init(&d);
        dec3_run(&d, in, 720, out);
        int peak = 0;
        for (size_t i = DEC3_TAPS / 3; i < 240; i++) peak = abs(out[i]) > peak ? abs(out[i]) : peak;
        assert(peak >= lo[f] && peak <= hi[f]);
    }
    // The state carries across calls of any size.
    dec3_init(&d);
    size_t m = 0;
    for (size_t i = 0; i < 720; i += 7) m += dec3_run(&d, &in[i], i + 7 <= 720 ? 7 : 720 - i, &parts[m]);
    assert(m == 240 && memcmp(parts, out, sizeof out) == 0);
}

static void test_skip(void)
{
    int16_t buf[64] = {0}, one = 1;
    tap_t r;
    tap_init(&r, buf, 64, 8, 1);  // 56 readable
    int16_t zeros[40] = {0};
    tap_write(&r, zeros, 40);
    tap_write(&r, &one, 1);  // position 40
    tap_write(&r, zeros, 10);
    // The window is the chunk and the tail before it: [aligned - tail, aligned + chunk).
    assert(aec_skip(&r, 20, 10, 10));   // 10..29
    assert(!aec_skip(&r, 31, 10, 10));  // 21..40: the last frame
    assert(!aec_skip(&r, 50, 10, 10));  // 40..59: the first frame
    assert(aec_skip(&r, 51, 10, 10));   // 41..60, partly not written yet
    tap_write(&r, zeros, 40);
    tap_write(&r, zeros, 6);  // 97 written: position 40, still in its slot, is past the readable 56
    assert(aec_skip(&r, 40, 10, 10));
}

static uint32_t s_rand = 1;
static int rnd(int span)  // 0..span-1
{
    s_rand = s_rand * 1103515245u + 12345u;
    return (int)((s_rand >> 8) % (uint32_t)span);
}

// Noise under a syllable-like envelope: runs of 3..20 bins, silent or at 20..100 % amplitude.
static void speechlike(int16_t *x, size_t n)
{
    size_t i = 0;
    while (i < n) {
        const size_t run = (size_t)(3 + rnd(18)) * LAG_BIN;
        const int amp = rnd(3) == 0 ? 0 : 2000 + rnd(8000);
        for (size_t k = 0; k < run && i < n; k++, i++) x[i] = (int16_t)(amp * (rnd(2001) - 1000) / 1000);
    }
}

static void test_lag(void)
{
    enum { N = 80 * 512, DELAY = 2600 };
    static int16_t ref[N], mic[N], ring[1 << 15];
    static lag_track_t lt;
    static float corr[LAG_MAX + 1];
    speechlike(ref, N);
    // The microphone: the reference DELAY frames later at 0.3, under noise.
    for (size_t m = 0; m < N; m++) mic[m] = (int16_t)((m >= DELAY ? 3 * ref[m - DELAY] / 10 : 0) + rnd(401) - 200);
    // The reference counter runs 1000 frames behind the microphone's and wraps on the way.
    const uint32_t offset = (uint32_t)-1000;
    tap_t r;
    tap_init(&r, ring, 1 << 15, 512, 1);
    atomic_store(&r.written, offset);
    lag_track_init(&lt);
    for (size_t m = 0; m < N; m += 512) {
        tap_write(&r, &ref[m], 512);
        lag_track_push(&lt, &mic[m], 512, 1);
    }
    const int lag = lag_track_run(&lt, &r, N, offset, corr);
    assert(abs(lag - DELAY) <= LAG_BIN && corr[lag / LAG_BIN] >= 0.9f);

    // A microphone that does not hear the reference: no clear peak.
    speechlike(mic, N);
    lag_track_init(&lt);
    for (size_t m = 0; m < N; m += 512) lag_track_push(&lt, &mic[m], 512, 1);
    assert(lag_track_run(&lt, &r, N, offset, corr) == -1);
}

// One chunk of 512 frames with nothing dropped and the reference not yet settled.
static mic_chunk_t chunk(mic_chunks_t *c, bool want_raw, bool wrote)
{
    return mic_chunks_step(c, 0, want_raw, wrote, 0);
}

static void test_chunk_stages(void)
{
    mic_chunks_t c;
    mic_chunks_init(&c, 512, 16000);
    mic_chunk_t k = chunk(&c, false, false);
    assert(!k.raw && !k.write && k.pos == 0);
    // NONE is asked for at chunk 1 and goes through: raw from chunk 3, the 2nd after it.
    k = chunk(&c, true, false);
    assert(k.write && k.write_raw && !k.raw && k.pos == 512);
    k = chunk(&c, true, true);
    assert(!k.write && !k.raw);
    k = chunk(&c, true, false);
    assert(!k.write && k.raw);
    // AGC at chunk 4: still raw at chunk 5, AGC from chunk 6.
    k = chunk(&c, false, false);
    assert(k.write && !k.write_raw && k.raw);
    k = chunk(&c, false, true);
    assert(!k.write && k.raw);
    k = chunk(&c, false, false);
    assert(!k.write && !k.raw);
    // A failed write is asked for again and counts nothing down.
    k = chunk(&c, true, false);
    assert(k.write && k.write_raw);
    k = chunk(&c, true, false);
    assert(k.write && k.write_raw && !k.raw);
    k = chunk(&c, true, true);
    assert(!k.write && !k.raw);
    k = chunk(&c, true, false);
    assert(!k.write && k.raw && k.pos == 10 * 512);

    // Raw follows the stage last written, not the one asked for: NONE went through, the AGC write
    // fails as the settle ends, so the wire is still raw.
    mic_chunks_init(&c, 512, 16000);
    chunk(&c, true, false);
    chunk(&c, false, true);
    k = chunk(&c, false, false);
    assert(k.raw && k.write && !k.write_raw);
}

static void test_chunk_stage_change(void)
{
    mic_chunks_t c;
    mic_chunks_init(&c, 512, 16000);
    // NONE written, then AGC wanted before a chunk is raw: AGC is written and no chunk is raw.
    mic_chunk_t k = chunk(&c, true, false);
    assert(k.write && k.write_raw);
    k = chunk(&c, false, true);
    assert(k.write && !k.write_raw && !k.raw);
    for (int i = 0; i < 3; i++) {
        k = chunk(&c, false, i == 0);
        assert(!k.write && !k.raw);
    }
    // Raw, then AGC written and NONE wanted again before a chunk is AGC: every chunk stays raw.
    chunk(&c, true, false);
    chunk(&c, true, true);
    assert(chunk(&c, true, false).raw);
    k = chunk(&c, false, false);
    assert(k.write && !k.write_raw && k.raw);
    k = chunk(&c, true, true);
    assert(k.write && k.write_raw && k.raw);
    for (int i = 0; i < 3; i++) {
        k = chunk(&c, true, i == 0);
        assert(!k.write && k.raw);
    }
}

// The bookkeeping after three chunks, the third the first raw one.
static void chunks_raw(mic_chunks_t *c)
{
    mic_chunks_init(c, 512, 16000);
    chunk(c, true, false);
    chunk(c, true, true);
    assert(chunk(c, true, false).raw);
}

static void test_chunk_loss(void)
{
    mic_chunks_t c;
    chunks_raw(&c);
    for (int i = 0; i < 60; i++) assert(chunk(&c, true, false).lag_ok == (i >= 48));
    // 256 frames dropped before a chunk put it 256 frames later and start the raw run anew.
    const uint32_t next = chunk(&c, true, false).pos + 512;
    mic_chunk_t k = mic_chunks_step(&c, 256, true, false, 0);
    assert(k.pos == next + 256 && k.raw && !k.lag_ok);
    assert(chunk(&c, true, false).pos == next + 256 + 512);
}

static void test_chunk_lag_span(void)
{
    mic_chunks_t c;
    chunks_raw(&c);  // one raw chunk so far
    // (LAG_DELAY + LAG_WINDOW) bins of 64 frames = 50 chunks of 512: allowed from the 50th raw one.
    for (int n = 2; n <= 50; n++) assert(chunk(&c, true, false).lag_ok == (n == 50));
    // An AGC chunk ends the run.
    chunk(&c, false, false);
    chunk(&c, false, true);
    assert(!chunk(&c, false, false).raw);
    chunk(&c, true, false);
    chunk(&c, true, true);
    for (int n = 1; n <= 50; n++) {
        const mic_chunk_t k = chunk(&c, true, false);
        assert(k.raw && k.lag_ok == (n == 50));
    }
    // A loss starts it anew: the chunk after the loss is the run's first.
    assert(!mic_chunks_step(&c, 256, true, false, 0).lag_ok);
    for (int n = 2; n <= 50; n++) assert(chunk(&c, true, false).lag_ok == (n == 50));
}

static void test_chunk_offset(void)
{
    mic_chunks_t c;
    mic_chunks_init(&c, 512, 16000);
    mic_chunk_t k = mic_chunks_step(&c, 0, false, false, 15999);
    assert(!k.have_offset);
    // Taken at the first chunk with 16000 reference frames written, against the microphone frames
    // on the wire to its end, the dropped ones included; then kept.
    k = mic_chunks_step(&c, 256, false, false, 16100);
    assert(k.have_offset && k.pos == 768 && k.offset == 16100u - (768 + 512));
    k = mic_chunks_step(&c, 0, false, false, 17000);
    assert(k.have_offset && k.offset == 16100u - (768 + 512));
}

static void test_volume(void)
{
    // The stock table: UI volume -> register byte (0.5 dB steps from 0 dB = 0x00).
    const struct {
        int percent;
        uint8_t reg;
    } table[] = {
        {0, 0xC7}, {10, 0xCF}, {25, 0xDB}, {40, 0xE7}, {50, 0xEF},
        {55, 0xF3}, {70, 0xFF}, {80, 0x06}, {90, 0x0D}, {100, 0x15},
    };
    for (size_t i = 0; i < sizeof table / sizeof table[0]; i++) {
        if (volume_dac_byte(table[i].percent) != table[i].reg) {
            fprintf(stderr, "volume %d: 0x%02X, want 0x%02X\n", table[i].percent,
                    volume_dac_byte(table[i].percent), table[i].reg);
            assert(0);
        }
    }
    // Monotonic, one register step at most per percent.
    for (int p = 1; p <= 100; p++) {
        const int d = (int8_t)volume_dac_byte(p) - (int8_t)volume_dac_byte(p - 1);
        assert(d == 0 || d == 1);
    }
    assert(volume_dac_muted(0));
    assert(!volume_dac_muted(1) && !volume_dac_muted(100));
}

int main(void)
{
    test_samples();
    test_tap();
    test_decimator();
    test_skip();
    test_lag();
    test_chunk_stages();
    test_chunk_stage_change();
    test_chunk_loss();
    test_chunk_lag_span();
    test_chunk_offset();
    test_volume();
    printf("audio_core: all tests passed\n");
    return 0;
}
