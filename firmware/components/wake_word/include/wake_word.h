// The micro_wake_word engine: a wake model and a VAD model, loaded at run time, over 16 kHz mono
// audio.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The word a detection stands for (the "Zakhar" manifest's wake_word).
#define WAKE_WORD_NAME "Захар"

// The two models of the engine.
typedef enum {
    WAKE_WORD_MODEL_WAKE,
    WAKE_WORD_MODEL_VAD,
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

// Runs n samples through the frontend and each loaded model; detection runs only while both are
// loaded. A detection is logged and handed to the callback of wake_word_set_detect_cb, on the
// calling task.
void wake_word_feed(const int16_t *samples, size_t n);

// The detection cutoffs of the wake and VAD models, 0..255 (probability * 255). Safe from any task.
void wake_word_set_cutoffs(uint8_t wake, uint8_t vad);

// The highest raw wake probability (0..255) since the previous call; resets it to 0.
uint8_t wake_word_take_peak(void);

void wake_word_set_detect_cb(void (*cb)(void));

#ifdef __cplusplus
}
#endif
