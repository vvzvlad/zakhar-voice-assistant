// The pure audio pieces of the Voice PE firmware: no ESP headers, host-tested
// (make fw-voicepe-host-test). The microphone sample conversion, the microphone tap ring, the echo
// reference (decimator, aligned reads, the skip decision), the echo delay estimator, the mic
// task's per-chunk bookkeeping, the room's noise level and the speaker volume -> DAC register
// mapping.
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
// Copies frames pos..pos+n-1 and moves no position; a frame outside the readable window (not
// written yet, or older than cap - margin frames back from the newest) reads as zeros.
void tap_read_at(tap_t *t, uint32_t pos, int16_t *out, size_t n);

// ---------------------------------------------------------------- echo reference

// 48 kHz -> 16 kHz for the echo reference: a 72-tap linear-phase low-pass (a Hamming-windowed sinc
// with its cutoff at 7 kHz: within 0.1 dB up to 6 kHz, -6 dB at 7 kHz, at least 39 dB down from
// 8 kHz and 54 dB from 8.5 kHz, DC gain exactly 1) evaluated at every third input.
#define DEC3_TAPS 72

typedef struct {
    int16_t win[2 * DEC3_TAPS];  // every input stored twice, so the newest DEC3_TAPS are contiguous
    int pos;                     // the next write index, 0..DEC3_TAPS-1
    int phase;                   // inputs since the last output, 0..2
} dec3_t;

void dec3_init(dec3_t *d);
// One output per third input, the state carried across calls; returns the outputs written.
size_t dec3_run(dec3_t *d, const int16_t *in, size_t n, int16_t *out);

// The echo canceller's skip decision for a chunk of `chunk` frames whose reference starts at ref
// position `aligned`: true when the reference the filter could still be using, the chunk's own and
// the `tail` frames before it, is all zero (frames outside the readable window count as zero).
bool aec_skip(tap_t *ref, uint32_t aligned, size_t chunk, size_t tail);

// ---------------------------------------------------------------- echo delay

// The delay of the speaker's echo in the microphone, from the cross-correlation of the short-time
// log-energy envelopes of the reference and the microphone.
#define LAG_BIN 64      // frames per envelope bin: 4 ms
#define LAG_WINDOW 384  // microphone bins correlated: 1.536 s
#define LAG_MAX 100     // lags searched, 0..LAG_MAX bins: up to 400 ms
// Bins between the newest one pushed and the window's end: 64 ms, more than the reference and
// microphone counters, each advancing in its own blocks, wander from their mean offset, so the
// window's reference is written.
#define LAG_DELAY 16
#define LAG_HIST 512  // microphone bins kept: a power of two >= LAG_DELAY + LAG_WINDOW

typedef struct {
    float env[LAG_HIST];                  // log(1 + mean square) per bin, by bin number % LAG_HIST
    uint32_t bins;                        // bins pushed
    float ref_env[LAG_WINDOW + LAG_MAX];  // lag_track_run's scratch
} lag_track_t;

void lag_track_init(lag_track_t *t);
// Pushes n microphone frames, n a multiple of LAG_BIN, from every stride-th sample of x.
void lag_track_push(lag_track_t *t, const int16_t *x, size_t n, size_t stride);
// Correlates the window of microphone bins with the reference in `ref`, microphone frame m paired
// with ref position m + offset - lag; mic_end is the microphone position just past the newest frame
// pushed. corr gets the normalized correlation at each lag 0..LAG_MAX (bins). Returns the lag in
// frames of a clear peak (correlation >= 0.7 and 0.2 above every lag more than 100 ms away from
// it), else -1.
int lag_track_run(lag_track_t *t, tap_t *ref, uint32_t mic_end, uint32_t offset, float *corr);

// ---------------------------------------------------------------- the mic task's chunks

// Chunks read after a stage write that may still carry the stage before it: the RX DMA queue and
// the XMOS's own output buffering.
#define MIC_STAGE_SETTLE 2

// Per chunk: where it sits on the wire, whether its channel 0 is the raw microphone, when the
// XMOS channel 0 stage is to be written, the reference offset and whether the lag estimator's
// window holds raw channel 0 only.
typedef struct {
    uint32_t frames;      // per chunk
    uint32_t ref_settle;  // reference frames written before the offset is taken
    uint32_t span;        // raw chunks in a row the estimator's window spans
    uint32_t pos;         // frames on the wire before the next chunk, read or dropped
    uint32_t offset;
    bool have_offset;
    bool none_written;    // channel 0's stage last written is NONE, not AGC
    bool asked, asked_raw;  // the previous step asked for a stage write, and which
    unsigned settle;      // chunks still to come before channel 0 carries none_written for sure
    bool raw;
    uint32_t raw_run;     // raw chunks in a row, none dropped among them
} mic_chunks_t;

typedef struct {
    uint32_t pos;     // the chunk's first frame on the wire
    bool raw;         // its channel 0 is the raw microphone
    bool settling;    // a stage write is settling: channel 0 may carry either stage
    bool write;       // a stage write is due: channel 0 to NONE when write_raw, else to AGC
    bool write_raw;
    bool have_offset;
    uint32_t offset;  // the reference written minus the microphone frames on the wire
    bool lag_ok;      // the raw chunks in a row, this one included, span the estimator's window
} mic_chunk_t;

// Chunks of `frames` frames; the offset is taken at the first chunk with at least ref_settle
// reference frames written. Channel 0 starts at AGC, as the XMOS boots it.
void mic_chunks_init(mic_chunks_t *c, uint32_t frames, uint32_t ref_settle);
// One chunk read. `dropped` is the frames the driver dropped since the previous chunk, which come
// before this one on the wire; `want_raw` the stage channel 0 should have; `wrote` whether the
// stage write the previous step asked for went through (a failed one is asked for again);
// `ref_written` the reference frames written so far. A chunk is raw from the MIC_STAGE_SETTLE-th
// chunk after the step that asked for the NONE write that went through, and AGC again from the
// MIC_STAGE_SETTLE-th after the AGC one.
mic_chunk_t mic_chunks_step(mic_chunks_t *c, uint32_t dropped, bool want_raw, bool wrote, uint32_t ref_written);

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

// ---------------------------------------------------------------- speaker volume

// The stock volume path: percent 0..100 remapped to 0.40..0.85, then the AIC3204 digital volume
// byte trunc(-127 + v * 175) for registers 65/66 (0.5 dB steps), in exact arithmetic.
uint8_t volume_dac_byte(int percent);
// Below 1 % the DAC is muted (page 0 register 64 = 0x0C).
bool volume_dac_muted(int percent);
