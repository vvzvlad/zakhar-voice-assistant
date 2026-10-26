// zakhar-replay: the board's microphone processing (station-lite/main/mic_chain.h, the capture
// thread's own code) over a recording instead of the sound card. It opens no sound device, so it
// runs next to a live zakhar-station.
//   zakhar-replay [--no-ns] [--wake wake.tflite cutoff vad.tflite cutoff] mics.wav [speaker.wav] out.wav
// mics.wav: hw:0,1 as arecord writes it, 8 channels, S16, 48 kHz. speaker.wav: the speaker's own
// signal (hw:0,0 capture), 2 channels, S16, 48 kHz; silence without it and past its end. out.wav: 2
// channels, S16, 16 kHz, channels 0 and 1 as the device's tap. --no-ns: noise suppression off.
// --wake: channel 1 also goes through the device's wake-word engine (wake_word.cpp) with these
// models and cutoffs, 0..255 as the engine takes them (the server's percent converted, as the
// board's "config:" log line shows them). Every 10 ms period whose raw wake probability reaches the
// wake cutoff is printed with its time in the recording, and so are the next inferences up to where
// the engine can still decide on it; the engine's own "detected" or "blocked by VAD" line comes
// right before the line of the inference it decided on (the next one, if that one's probability
// was 0 and so printed nothing, or if the check ran on the period that ended the 1 s cool-off after
// a detection, which needs no inference of its own).
// Prints the number of 10 ms periods and the microphone processing time per period; a last partial
// period of mics.wav is dropped.
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "apm.h"
#include "esp_heap_caps.h"
#include "mic_chain.h"
#include "wake_word.h"

#define RATE_IN 48000
#define RATE_OUT 16000

typedef struct {
    FILE *f;
    uint32_t left;  // bytes of the data chunk not read yet
    unsigned channels;
} wav_t;

static uint32_t le16(const uint8_t *p)
{
    return p[0] | (uint32_t)p[1] << 8;
}

static uint32_t le32(const uint8_t *p)
{
    return le16(p) | le16(p + 2) << 16;
}

// Opens `path` at its data chunk, which must be S16 at RATE_IN with `channels` channels (PCM or
// extensible format). False with the reason printed.
static bool wav_open(wav_t *w, const char *path, unsigned channels)
{
    uint8_t h[12];
    w->f = fopen(path, "rb");
    if (w->f == NULL) {
        perror(path);
        return false;
    }
    if (fread(h, 1, 12, w->f) != 12 || memcmp(h, "RIFF", 4) != 0 || memcmp(h + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "%s: not a WAV file\n", path);
        return false;
    }
    bool fmt = false;
    for (;;) {
        uint8_t c[8];
        if (fread(c, 1, 8, w->f) != 8) {
            fprintf(stderr, "%s: no data chunk\n", path);
            return false;
        }
        const uint32_t size = le32(c + 4);
        if (memcmp(c, "data", 4) == 0) {
            if (!fmt) {
                fprintf(stderr, "%s: data before fmt\n", path);
                return false;
            }
            w->left = size;
            w->channels = channels;
            return true;
        }
        if (memcmp(c, "fmt ", 4) == 0 && size >= 16) {
            uint8_t f[16];
            if (fread(f, 1, 16, w->f) != 16) break;
            const uint32_t format = le16(f);
            if ((format != 1 && format != 0xFFFE) || le16(f + 2) != channels || le32(f + 4) != RATE_IN ||
                le16(f + 14) != 16) {
                fprintf(stderr, "%s: format %u, %u channels, %u Hz, %u bits; wanted S16, %u channels, %u Hz\n", path,
                        (unsigned)format, (unsigned)le16(f + 2), (unsigned)le32(f + 4), (unsigned)le16(f + 14),
                        channels, RATE_IN);
                return false;
            }
            fmt = true;
            if (fseek(w->f, (long)(size - 16 + (size & 1)), SEEK_CUR) != 0) break;
        } else if (fseek(w->f, (long)(size + (size & 1)), SEEK_CUR) != 0) {
            break;
        }
    }
    fprintf(stderr, "%s: truncated\n", path);
    return false;
}

// Up to `frames` frames into `buf`; the number read.
static size_t wav_read(wav_t *w, int16_t *buf, size_t frames)
{
    const size_t frame = w->channels * sizeof buf[0];
    if (frames > w->left / frame) frames = w->left / frame;
    const size_t n = fread(buf, frame, frames, w->f);
    w->left -= (uint32_t)(n * frame);
    return n;
}

// The 44-byte header of a 2-channel S16 file at RATE_OUT with `bytes` of data.
static void wav_header(uint8_t *h, uint32_t bytes)
{
    const uint32_t v[] = {0x46464952 /* RIFF */, 36 + bytes, 0x45564157 /* WAVE */, 0x20746d66 /* fmt */, 16,
                          1 | 2u << 16, RATE_OUT, RATE_OUT * 4, 4 | 16u << 16, 0x61746164 /* data */, bytes};
    for (unsigned i = 0; i < sizeof v / sizeof v[0]; i++)
        for (unsigned b = 0; b < 4; b++) h[4 * i + b] = (uint8_t)(v[i] >> (8 * b));
}

// A .tflite file in a 16-byte aligned buffer, as wake_word_load() takes it; NULL with the reason
// printed.
static uint8_t *read_model(const char *path, size_t *size)
{
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        perror(path);
        return NULL;
    }
    uint8_t *data = NULL;
    long n = -1;
    if (fseek(f, 0, SEEK_END) == 0 && (n = ftell(f)) > 0 && fseek(f, 0, SEEK_SET) == 0) {
        data = heap_caps_aligned_alloc(16, (size_t)n, MALLOC_CAP_8BIT);
        if (data != NULL && fread(data, 1, (size_t)n, f) != (size_t)n) {
            heap_caps_free(data);
            data = NULL;
        }
    }
    fclose(f);
    if (data == NULL) fprintf(stderr, "%s: not read\n", path);
    *size = (size_t)n;
    return data;
}

static double now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    bool ns = true;
    const char *wake_path = NULL, *vad_path = NULL;
    unsigned long wake_cutoff = 0, vad_cutoff = 0;
    for (;;) {
        if (argc > 1 && strcmp(argv[1], "--no-ns") == 0) {
            ns = false;
            argc--;
            argv++;
        } else if (argc > 5 && strcmp(argv[1], "--wake") == 0) {
            wake_path = argv[2];
            wake_cutoff = strtoul(argv[3], NULL, 10);
            vad_path = argv[4];
            vad_cutoff = strtoul(argv[5], NULL, 10);
            argc -= 5;
            argv += 5;
        } else {
            break;
        }
    }
    if (argc != 3 && argc != 4) {
        fprintf(stderr, "usage: zakhar-replay [--no-ns] [--wake wake.tflite cutoff vad.tflite cutoff] mics.wav "
                        "[speaker.wav] out.wav\n");
        return 2;
    }
    if (wake_path != NULL) {
        size_t wake_size = 0, vad_size = 0;
        uint8_t *wake_model = read_model(wake_path, &wake_size), *vad_model = read_model(vad_path, &vad_size);
        if (wake_model == NULL || vad_model == NULL || !wake_word_init() ||
            !wake_word_load(WAKE_WORD_MODEL_WAKE, wake_model, wake_size) ||
            !wake_word_load(WAKE_WORD_MODEL_VAD, vad_model, vad_size))
            return 1;
        wake_word_set_cutoffs((uint8_t)wake_cutoff, (uint8_t)vad_cutoff);
    }
    wav_t mics, spk = {0};
    if (!wav_open(&mics, argv[1], MIC_CHANNELS) || (argc == 4 && !wav_open(&spk, argv[2], 2))) return 1;
    const char *out_path = argv[argc - 1];
    FILE *out = fopen(out_path, "wb");
    if (out == NULL) {
        perror(out_path);
        return 1;
    }
    uint8_t header[44];
    wav_header(header, 0);
    fwrite(header, 1, sizeof header, out);
    if (!apm_init(ns)) return 1;

    static int16_t m[APM_IN_FRAMES * MIC_CHANNELS], s[APM_IN_FRAMES * 2], frames[APM_OUT_FRAMES * 2],
        wake[APM_OUT_FRAMES];
    unsigned long periods = 0;
    double busy = 0;
    unsigned tail = 0;  // inferences still to print after one that reached the wake cutoff
    while (wav_read(&mics, m, APM_IN_FRAMES) == APM_IN_FRAMES) {
        const size_t n = spk.f != NULL ? wav_read(&spk, s, APM_IN_FRAMES) : 0;
        memset(s + 2 * n, 0, (APM_IN_FRAMES - n) * 2 * sizeof s[0]);
        const double t = now();
        if (!mic_chain_process(m, s, frames, wake)) return 1;
        busy += now() - t;
        if (wake_path != NULL) {
            wake_word_feed(wake, APM_OUT_FRAMES);
            // A period runs at most one inference; a non-zero peak is its probability. The engine
            // decides on the sum of the last 5 (wake_word.cpp), which tops 5 * cutoff only while
            // one of them reached the cutoff: the 4 inferences after such a one are printed too.
            const unsigned peak = wake_word_take_peak();
            if (peak >= wake_cutoff) tail = 5;
            if (peak > 0 && tail > 0) {
                tail--;
                // On stderr, as the engine's own log, so the two stay in order.
                fprintf(stderr, "%.2f s: wake %u\n", (periods + 1) / 100.0, peak);
            }
        }
        if (fwrite(frames, sizeof frames, 1, out) != 1) {
            perror(out_path);
            return 1;
        }
        periods++;
    }
    wav_header(header, (uint32_t)(periods * sizeof frames));
    if (fseek(out, 0, SEEK_SET) != 0 || fwrite(header, 1, sizeof header, out) != sizeof header || fclose(out) != 0) {
        perror(out_path);
        return 1;
    }
    printf("%lu periods of 10 ms (%.2f s), noise suppression %s; microphone processing %.3f ms per period\n", periods,
           periods / 100.0, ns ? "on" : "off", periods ? busy * 1000 / (double)periods : 0.0);
    return 0;
}
