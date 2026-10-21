// The pure audio pieces of the Voice PE firmware: no ESP headers, host-tested
// (make fw-voicepe-host-test). The microphone sample conversion, the microphone tap ring, the mic
// task's per-chunk bookkeeping of channel 0's stage, the room's noise level, the speaker's mix of
// zakhar's playback and the music, and the speaker volume -> DAC register mapping. audio_core.c
// also implements the music ring, declared in music_ring.h.
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

// ---------------------------------------------------------------- the mic task's chunks

// Chunks read after a stage write that may still carry the stage before it: the RX DMA queue and
// the XMOS's own output buffering.
#define MIC_STAGE_SETTLE 2

// Per chunk: whether its channel 0 is the raw microphone and when the XMOS channel 0 stage is to be
// written.
typedef struct {
    bool none_written;      // channel 0's stage last written is NONE, not AGC
    bool asked, asked_raw;  // the previous step asked for a stage write, and which
    unsigned settle;        // chunks still to come before channel 0 carries none_written for sure
    bool raw;
} mic_chunks_t;

typedef struct {
    bool raw;    // its channel 0 is the raw microphone
    bool write;  // a stage write is due: channel 0 to NONE when write_raw, else to AGC
    bool write_raw;
} mic_chunk_t;

// Channel 0 starts at AGC, as the XMOS boots it.
void mic_chunks_init(mic_chunks_t *c);
// One chunk read. `want_raw` is the stage channel 0 should have; `wrote` whether the stage write
// the previous step asked for went through (a failed one is asked for again). A chunk is raw from
// the MIC_STAGE_SETTLE-th chunk after the step that asked for the NONE write that went through,
// and AGC again from the MIC_STAGE_SETTLE-th after the AGC one.
mic_chunk_t mic_chunks_step(mic_chunks_t *c, bool want_raw, bool wrote);

// ---------------------------------------------------------------- the room's noise level

// The RMS of channel 0's 32-bit slots over the chunks added, at their full resolution: a quiet
// room is a few steps of the 16-bit samples the tap keeps.
typedef struct {
    double squares;  // the sum of the squared slot values
    uint64_t n;      // the slots summed
} ambient_t;

void ambient_init(ambient_t *a);
// Adds channel 0 (slot 0) of `frames` 32-bit stereo frames (slot 0, slot 1).
void ambient_add(ambient_t *a, const int32_t *in, size_t frames);
// The RMS in dBFS of the 32-bit slot, 20 log10(rms / 2^31), rounded to one decimal; -inf when
// nothing was added or every slot was 0.
double ambient_dbfs(const ambient_t *a);

// When to measure: a measurement starts at the `every`-th chunk of quiet in a row, counted from
// the previous measurement's start, and ends after `chunks` raw chunks; the end of quiet ends it
// with nothing to report.
typedef struct {
    unsigned every, chunks;
    unsigned quiet;  // chunks of quiet in a row since the last start
    unsigned taken;  // raw chunks the running measurement measured
    bool measuring;
    ambient_t level;
} ambient_sched_t;

void ambient_sched_init(ambient_sched_t *s, unsigned every, unsigned chunks);
// Per chunk, before mic_chunks_step: whether channel 0 is to be raw for a measurement. `quiet` is
// auto volume on, nothing playing and the microphone not busy.
bool ambient_sched_want(ambient_sched_t *s, bool quiet);
// Per chunk, after mic_chunks_step, when ambient_sched_want() said true: a raw chunk (`raw`, its
// `frames` 32-bit stereo frames in `in`) is measured, any other is not. True when this chunk
// completes the measurement; ambient_dbfs(&s->level) is then its level.
bool ambient_sched_take(ambient_sched_t *s, bool raw, const int32_t *in, size_t frames);

// ---------------------------------------------------------------- the speaker's mix

// Gains are Q16: 65536 is 1.0.
#define MIX_UNITY 65536u
// The music under a voice session or zakhar's playback: -20 dB, a tenth of its gain.
#define MIX_DUCK_DIV 10

typedef struct {
    uint32_t gain;  // the music's gain at the end of the previous block
} mix_t;

// The music starts from silence.
void mix_init(mix_t *m);
// One speaker block of `frames` frames. `out` (interleaved stereo) is `voice` (mono, `frames`
// samples) on both channels plus `music` (interleaved stereo; its first music_frames frames, then
// silence) times the music's gain, saturating. The music's gain moves linearly across the block
// from where the previous block left it to `gain`, or to a tenth of it while `duck`, reaching it
// at the block's last frame. In place: `out` may be `music`, as each frame is read before it is
// written.
void mix_block(mix_t *m, const int16_t *voice, const int16_t *music, size_t music_frames, uint32_t gain, bool duck,
               size_t frames, int16_t *out);

// ---------------------------------------------------------------- speaker volume

// The stock volume path: percent 0..100 remapped to 0.40..0.85, then the AIC3204 digital volume
// byte trunc(-127 + v * 175) for registers 65/66 (0.5 dB steps), in exact arithmetic.
uint8_t volume_dac_byte(int percent);
// Below 1 % the DAC is muted (page 0 register 64 = 0x0C).
bool volume_dac_muted(int percent);
// The speaker's levels for the dial at `base` % and the server's boost in points. Zakhar's replies
// play at base + boost within 1..100 % (1, not 0: 0 mutes the DAC, so a quiet room would silence
// the reply instead of lowering it), the music at base; a base of 0 mutes both. The DAC takes the
// louder of the two, and each stream is brought down to its own level digitally by a Q16 gain
// (65536 = unity) that takes back the DAC steps it has over that level.
typedef struct {
    int dac;         // percent, for aic3204_set_volume
    uint32_t music;  // Q16
    uint32_t voice;  // Q16
} volume_levels_t;
volume_levels_t volume_levels(int base, int boost);
// Scales `n` samples in place by a Q16 gain of unity at most, rounding to nearest.
void gain_block(int16_t *s, size_t n, uint32_t gain);
