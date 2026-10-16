// The micro_wake_word engine: "Zakhar" v31 and the stock VAD model over 16 kHz mono audio.
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// The embedded wake model (microWakeWord/v31/model/zakhar.tflite) and its word (the manifest's
// wake_word).
#define WAKE_WORD_MODEL_VERSION "zakhar_v31"
#define WAKE_WORD_NAME "Захар"

// Sets up the frontend and loads both models (tensor arenas in PSRAM). Call once, on the task that
// then calls wake_word_feed(). False when an allocation or the model setup failed (logged).
bool wake_word_init(void);

// Runs n samples through the frontend and both models; a detection is logged and handed to the
// callback of wake_word_set_detect_cb, on the calling task.
void wake_word_feed(const int16_t *samples, size_t n);

// The detection cutoffs of the wake and VAD models, 0..255 (probability * 255). Safe from any task.
void wake_word_set_cutoffs(uint8_t wake, uint8_t vad);

// The highest raw wake probability (0..255) since the previous call; resets it to 0.
uint8_t wake_word_take_peak(void);

void wake_word_set_detect_cb(void (*cb)(void));

#ifdef __cplusplus
}
#endif
