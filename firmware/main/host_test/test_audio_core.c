// Host test of the Voice PE firmware's pure audio core (audio_core.c): the microphone sample
// conversion against ESPHome's MicrophoneSource, the tap ring and the stock volume -> DAC byte
// table. `make fw-voicepe-host-test` builds it.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
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
    test_volume();
    printf("audio_core: all tests passed\n");
    return 0;
}
