// The pure audio pieces of the Voice PE firmware: no ESP headers, host-tested
// (make fw-voicepe-host-test). The microphone sample conversion, the microphone tap ring and the
// speaker volume -> DAC register mapping.
#pragma once

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// ---------------------------------------------------------------- microphone samples

// One 32-bit I2S slot to 16 bits with an integer gain, as ESPHome's MicrophoneSource does it: Q31
// to Q25, times the gain, clamped to Q25, then the top 16 bits. Gain 1 is the plain top 16 bits.
int16_t pcm_s32_to_s16(int32_t sample, int gain);

// frames 32-bit stereo frames (slot 0, slot 1) to 16 bits: `ch1` gets channel 1 alone at ch1_gain,
// `both` gets channel 0 at gain 1 interleaved with that same `ch1`.
void pcm_split(const int32_t *in, size_t frames, int16_t *both, int16_t *ch1, int ch1_gain);

// ---------------------------------------------------------------- the microphone tap

// A ring of interleaved int16 frames with one writer. Positions count frames written since the
// start; cap is a power of two, so `position % cap` stays continuous when the uint32 counter
// wraps. A reader keeps `margin` frames away from the end the writer overwrites next: the writer
// overwrites that end before it publishes the new count.
typedef struct {
    int16_t *buf;  // cap * channels samples
    uint32_t cap;
    uint32_t margin;
    int channels;
    _Atomic uint32_t written;
} tap_t;

void tap_init(tap_t *t, int16_t *buf, uint32_t cap, uint32_t margin, int channels);
void tap_write(tap_t *t, const int16_t *frames, size_t n);
uint32_t tap_position(tap_t *t);
// Copies up to max frames from *pos and advances *pos past them; a *pos older than the readable
// window (cap - margin frames back from the newest) is first moved forward to its start. Returns
// the count copied.
size_t tap_read(tap_t *t, uint32_t *pos, int16_t *out, size_t max);

// ---------------------------------------------------------------- speaker volume

// The stock volume path: percent 0..100 remapped to 0.40..0.85, then the AIC3204 digital volume
// byte trunc(-127 + v * 175) for registers 65/66 (0.5 dB steps), in exact arithmetic.
uint8_t volume_dac_byte(int percent);
// Below 1 % the DAC is muted (page 0 register 64 = 0x0C).
bool volume_dac_muted(int percent);
