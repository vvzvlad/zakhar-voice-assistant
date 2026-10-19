// The micro_wake_word engine: a wake model, a stop model and a VAD model, loaded at run time, over
// 16 kHz mono audio.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The word a detection stands for (the "Zakhar" manifest's wake_word).
#define WAKE_WORD_NAME "Захар"

// The three models of the engine.
typedef enum {
    WAKE_WORD_MODEL_WAKE,
    WAKE_WORD_MODEL_VAD,
    WAKE_WORD_MODEL_STOP,
} wake_word_model_t;

// Sets up the frontend and, per model, its op resolver and arenas (PSRAM); no model is loaded.
// Call once, before wake_word_load() and wake_word_feed(). False when an allocation failed
// (logged).
bool wake_word_init(void);

// Loads a .tflite model in place of the kind's current one. Takes ownership of `data` (16-byte
// aligned, released with heap_caps_free); it stays in use while the model is loaded. Waits for a
// wake_word_feed() in progress and takes a few milliseconds. False when the model did not load:
// the kind then has no model and `data` is released.
bool wake_word_load(wake_word_model_t kind, uint8_t *data, size_t size);

// Runs n samples of `wake` through the frontend and each loaded model; detection runs only while
// the wake and VAD models are loaded. A detection is logged and handed to the callback of
// wake_word_set_detect_cb, on the calling task. The stop model runs only from 1 s of playback on
// (wake_word_set_stop) until the playback ends; its detection, gated by the VAD model too, goes to
// the stop callback, and the end of the playback logs its highest window mean. It hears n samples
// of `stop` through a frontend of its own, or the wake input when `stop` is NULL.
void wake_word_feed(const int16_t *wake, const int16_t *stop, size_t n);

// The detection cutoffs of the wake and VAD models, 0..255 (probability * 255). Safe from any task.
void wake_word_set_cutoffs(uint8_t wake, uint8_t vad);

// The highest raw wake probability (0..255) since the previous call; resets it to 0.
uint8_t wake_word_take_peak(void);

void wake_word_set_detect_cb(void (*cb)(void));

// The stop word: `playing` says whether the device plays audio, asked once per wake_word_feed()
// before the engine's lock is taken; `cb` gets a stop word detection, on the feeding task. Safe
// from any task.
void wake_word_set_stop(bool (*playing)(void), void (*cb)(void));

#ifdef __cplusplus
}
#endif
